/**
 * VitalCamera — core orchestrator for the VitalCamera SDK.
 *
 * Manages rPPG inference workers, real-time peak detection, HRV computation,
 * head-pose estimation, and event dispatch. DOM-free — all browser-specific
 * concerns (camera, canvas, UI) belong in the adapter layer.
 *
 * @module core/vitalcamera
 */

import RealtimePeakDetector from './peak-detect.js';
import { estimateHeadPose } from './headpose.js';
import {
    detectBvpPeaks,
    rejectAbnormalPeaks,
    refinePeaksParabolic,
    quotientFilterRR,
    madFilterRR,
    computeHrvMetrics,
} from './hrv.js';
import { createWorker } from '../workers/loader.js';

// ---------------------------------------------------------------------------
// Minimal EventEmitter mixin (browser-safe, no Node deps)
// ---------------------------------------------------------------------------

/** @mixin */
const EventEmitterMixin = {
    /** @param {string} event @param {Function} fn */
    on(event, fn) {
        if (!this._listeners) this._listeners = {};
        (this._listeners[event] ||= []).push(fn);
        return this;
    },
    /** @param {string} event @param {Function} fn */
    off(event, fn) {
        if (!this._listeners?.[event]) return this;
        this._listeners[event] = this._listeners[event].filter(f => f !== fn);
        return this;
    },
    /** @param {string} event @param {...any} args */
    emit(event, ...args) {
        if (!this._listeners?.[event]) return;
        for (const fn of this._listeners[event]) {
            try { fn(...args); } catch (e) { /* listener errors must not break the pipeline */ }
        }
    },
};

// ---------------------------------------------------------------------------
// Default configuration
// ---------------------------------------------------------------------------

const DEFAULTS = {
    workerBasePath: null,     // null = auto (fetch from SDK URL + Blob URL)
    enableEmotion: true,
    enableGaze: true,
    enableEyeState: true,
    enableHeadPose: true,
    enableHrv: true,
    hrvMinDuration: 15,       // seconds
    hrvMaxWindow: 120,        // seconds — sliding window cap (2 minutes)
    hrvUpdateInterval: 1000,  // ms
    hrvSqiThreshold: 0.6,     // only accumulate BVP samples with SQI above this
    sqiThreshold: 0.38,
    gazeConfidenceThreshold: 0.04,
    eyeStateThreshold: 0.5,         // p(open) >= threshold → "open" (used for the public 0/1 flag + display)
    gazeEyeOpenGateProb: 0.7,       // skip gaze inference unless max(L,R) eye-open prob ≥ this
};


// ---------------------------------------------------------------------------
// IndexedDB state cache — persist inference warm-start state across sessions
// ---------------------------------------------------------------------------

const IDB_NAME = 'VitalCameraSDK';
const IDB_STORE = 'states';
const IDB_KEY = 'inferenceState';

/** @returns {Promise<IDBDatabase>} */
function _openIDB() {
    return new Promise((resolve, reject) => {
        try {
            const req = indexedDB.open(IDB_NAME, 1);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(IDB_STORE)) {
                    db.createObjectStore(IDB_STORE);
                }
            };
            req.onsuccess = (e) => resolve(e.target.result);
            req.onerror = () => reject(req.error);
        } catch (_) {
            reject(new Error('IndexedDB not available'));
        }
    });
}

async function _loadCachedState() {
    try {
        const db = await _openIDB();
        return new Promise((resolve) => {
            const tx = db.transaction(IDB_STORE, 'readonly');
            const req = tx.objectStore(IDB_STORE).get(IDB_KEY);
            req.onsuccess = () => resolve(req.result || null);
            req.onerror = () => resolve(null);
        });
    } catch (_) {
        return null;
    }
}

async function _saveCachedState(stateData) {
    try {
        const db = await _openIDB();
        const tx = db.transaction(IDB_STORE, 'readwrite');
        tx.objectStore(IDB_STORE).put(stateData, IDB_KEY);
    } catch (_) {
        /* IndexedDB unavailable — silently skip */
    }
}

// ---------------------------------------------------------------------------
// VitalCamera
// ---------------------------------------------------------------------------

class VitalCamera {
    /**
     * @param {Object} config
     * @param {Object} config.models           Model ArrayBuffers
     * @param {ArrayBuffer} config.models.rppg
     * @param {ArrayBuffer} config.models.rppgProj
     * @param {ArrayBuffer} config.models.sqi
     * @param {ArrayBuffer} config.models.psd
     * @param {ArrayBuffer} [config.models.emotion]
     * @param {ArrayBuffer} [config.models.gaze]
     * @param {ArrayBuffer} [config.models.eyeState]
     * @param {string}  [config.workerBasePath='./workers/']
     * @param {boolean} [config.enableEmotion=true]
     * @param {boolean} [config.enableGaze=true]
     * @param {boolean} [config.enableEyeState=true]
     * @param {boolean} [config.enableHeadPose=true]
     * @param {boolean} [config.enableHrv=true]
     * @param {number}  [config.hrvMinDuration=15]
     * @param {number}  [config.hrvMaxWindow=120]         Max BVP window in seconds (default 2 min)
     * @param {number}  [config.hrvUpdateInterval=1000]
     * @param {number}  [config.hrvSqiThreshold=0.6]      Min SQI to accept BVP sample for HRV
     * @param {number}  [config.sqiThreshold=0.38]
     * @param {number}  [config.gazeConfidenceThreshold=0.04]  Min softmax peak to accept gaze; lower → blink/closed eyes
     * @param {number}  [config.eyeStateThreshold=0.5]    p(open) >= threshold → "open" (display flag)
     * @param {number}  [config.gazeEyeOpenGateProb=0.7]  Skip gaze inference unless max(L,R) eye-open prob ≥ this
     */
    constructor(config = {}) {
        // Mix in EventEmitter
        Object.assign(this, EventEmitterMixin);
        this._listeners = {};

        // Merge config with defaults
        this.config = { ...DEFAULTS, ...config };
        this.models = config.models || {};

        // State flags
        this.isRunning = false;
        this._workersReady = false;

        // Workers
        this._workers = {};       // keyed by name
        this._workerReady = {};   // per-worker ready flag

        // Peak detector
        this._peakDetector = new RealtimePeakDetector();

        // BVP sample buffer for HRV
        this._bvpSamples = [];    // { t: ms, v: number }
        this._lastHrvTime = 0;

        // BVP ring for PSD worker (450 samples)
        this._bvpRing = [];
        this._bvpRingSize = 450;
        this._lastPsdSendTime = 0;

        // Latest heart rate for peak validation
        this._lastHR = null;
        this._lastSqi = 0;

        // Smoothed frame dt (updated each processFrame call)
        this._dval = 1 / 30;

        // Decompressed state JSON for inference worker warm-start
        this._stateJson = null;

        // Frame counter for periodic state export
        this._frameCount = 0;
        this._stateExportInterval = 60; // export every 60 frames

        // Last accepted gaze result (used when current frame is filtered out)
        this._lastGaze = null;

        // Latest eye-state probabilities (raw sigmoid). Updated every frame and
        // consumed by the adapter to gate gaze inference when both eyes are closed.
        // null until the first eye-state result arrives.
        this._lastEyeState = null;  // { leftProb, rightProb, timestamp }
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /**
     * Create workers, load models, and wait for all `initDone` messages.
     * @returns {Promise<void>}
     */
    async init() {
        // Load inference state: prefer IndexedDB cache, fallback to gzip
        if (!this._stateJson) {
            const cached = await _loadCachedState();
            if (cached) {
                this._stateJson = cached;
            } else if (this.models.state) {
                try {
                    const ds = new DecompressionStream('gzip');
                    const reader = new Blob([this.models.state]).stream().pipeThrough(ds);
                    this._stateJson = await new Response(reader).json();
                } catch (_) {
                    this._stateJson = {};
                }
            } else {
                this._stateJson = {};
            }
        }

        const basePath = this.config.workerBasePath;  // null = auto Blob URL
        const workerNames = ['inference', 'psd'];
        if (this.config.enableEmotion && this.models.emotion) {
            workerNames.push('emotion');
        }
        if (this.config.enableGaze && this.models.gaze) {
            workerNames.push('gaze');
        }
        if (this.config.enableEyeState && this.models.eyeState) {
            workerNames.push('eye_state');
        }

        const initPromises = workerNames.map(async (name) => {
            try {
                const w = await createWorker(name, basePath);
                this._workers[name] = w;
                this._workerReady[name] = false;

                await new Promise((resolve, reject) => {
                    w.onerror = (err) => {
                        this.emit('error', { source: name, message: err?.message || String(err) });
                        reject(err);
                    };

                    w.onmessage = (e) => {
                        if (e.data?.type === 'initDone') {
                            this._workerReady[name] = true;
                            w.onmessage = (ev) => this._handleWorkerMessage(name, ev);
                            resolve();
                        } else {
                            this._handleWorkerMessage(name, e);
                        }
                    };

                    // Send init message with appropriate model buffers
                    this._sendWorkerInit(name, w);
                });
            } catch (err) {
                this.emit('error', { source: name, message: err?.message || String(err) });
                throw err;
            }
        });

        await Promise.all(initPromises);
        this._workersReady = true;
        this.emit('ready', {});
    }

    /** Start processing — resets internal state. */
    start() {
        this.isRunning = true;
        this._peakDetector.reset();
        this._bvpSamples = [];
        this._bvpRing = [];
        this._lastHrvTime = 0;
        this._lastHR = null;
        this._lastSqi = 0;
    }

    /** Pause processing (workers stay alive). */
    stop() {
        this.isRunning = false;
    }

    /** Terminate all workers and release resources. */
    destroy() {
        this.isRunning = false;
        this._workersReady = false;
        for (const w of Object.values(this._workers)) {
            try { w.terminate(); } catch (_) { /* ignore */ }
        }
        this._workers = {};
        this._workerReady = {};
    }

    // -----------------------------------------------------------------------
    // Frame input
    // -----------------------------------------------------------------------

    /**
     * Process a single video frame. Called by the adapter each frame.
     *
     * @param {Object} frame
     * @param {Float32Array}  frame.rppgInput     [1,1,36,36,3] face crop
     * @param {number}        frame.dtVal         time delta
     * @param {number}        frame.timestamp     ms timestamp
     * @param {Float32Array}  [frame.emotionInput] [1,224,224,3]
     * @param {Float32Array}  [frame.gazeInput]    [1,448,448,3]
     * @param {Float32Array}  [frame.eyeLeftInput]   [1,24,40,3]  RGB/255, no mean/std
     * @param {Float32Array}  [frame.eyeRightInput]  [1,24,40,3]  RGB/255, no mean/std
     * @param {Array}         [frame.faceKeypoints] 6 BlazeFace keypoints
     */
    processFrame(frame) {
        if (!this.isRunning) return;

        const { rppgInput, dtVal, timestamp, emotionInput, gazeInput,
                eyeLeftInput, eyeRightInput, faceKeypoints } = frame;

        // Track smoothed dt for HR formula correction
        if (dtVal > 0) this._dval = dtVal;

        // Dispatch to rPPG inference worker
        this._postIfReady('inference', {
            type: 'run',
            payload: { imgData: rppgInput, dtVal, timestamp },
        });

        // Dispatch to emotion worker
        if (this.config.enableEmotion && emotionInput && this._workers.emotion) {
            this._postIfReady('emotion', {
                type: 'run',
                payload: { imgData: emotionInput },
            });
        }

        // Dispatch to gaze worker
        if (this.config.enableGaze && gazeInput && this._workers.gaze) {
            this._postIfReady('gaze', {
                type: 'run',
                payload: { imgData: gazeInput },
            });
        }

        // Dispatch to eye-state worker (both eyes per call)
        if (this.config.enableEyeState && eyeLeftInput && eyeRightInput && this._workers.eye_state) {
            this._postIfReady('eye_state', {
                type: 'run',
                payload: { left: eyeLeftInput, right: eyeRightInput },
            });
        }

        // Head pose (synchronous, no worker needed)
        if (this.config.enableHeadPose && faceKeypoints && faceKeypoints.length >= 6) {
            try {
                const pose = estimateHeadPose(faceKeypoints);
                this.emit('headpose', {
                    yaw: pose.yaw,
                    pitch: pose.pitch,
                    roll: pose.roll,
                    normal: pose.normal,
                    timestamp,
                });
            } catch (err) {
                this.emit('error', { source: 'headpose', message: err.message });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Worker message handling
    // -----------------------------------------------------------------------

    /**
     * Route incoming worker messages to the appropriate handler.
     * @private
     */
    _handleWorkerMessage(workerName, event) {
        const data = event.data || event;
        const type = data.type;

        if (type === 'error') {
            this.emit('error', { source: workerName, message: data.msg || data.message || '' });
            return;
        }

        if (type === 'state_exported') {
            _saveCachedState(data.payload);
            this.emit('state_exported', data.payload);
            return;
        }

        if (type !== 'result') return;

        // Workers send { type: 'result', payload: { ... } } — unwrap payload
        const payload = data.payload || data;

        switch (workerName) {
            case 'inference':
                this._onInferenceResult(payload);
                break;
            case 'psd':
                this._onPsdResult(payload);
                break;
            case 'emotion':
                this._onEmotionResult(payload);
                break;
            case 'gaze':
                this._onGazeResult(payload);
                break;
            case 'eye_state':
                this._onEyeStateResult(payload);
                break;
        }
    }

    /**
     * Handle rPPG inference result — emit BVP, run peak detection, feed PSD
     * worker, and accumulate BVP samples for HRV.
     * @private
     */
    _onInferenceResult(data) {
        const { value, timestamp, time } = data;
        if (value == null) return;

        // Emit raw BVP (including inference time for latency display)
        this.emit('bvp', { value, timestamp, time });

        // Periodic state export for IndexedDB caching
        this._frameCount++;
        if (this._frameCount % this._stateExportInterval === 0) {
            this._postIfReady('inference', { type: 'export_state' });
        }

        // Peak detection → beat events
        const beat = this._peakDetector.process(timestamp, value, this._lastHR);
        if (beat) {
            this.emit('beat', { ibi: beat.ibi, timestamp: beat.peakTime });
        }

        // Accumulate BVP samples for PSD worker (ring buffer of 450)
        this._bvpRing.push(value);
        if (this._bvpRing.length > this._bvpRingSize) {
            this._bvpRing.shift();
        }

        // Throttle PSD to once every 500ms to save compute
        const now = performance.now();
        if (now - this._lastPsdSendTime >= 500) {
            this._lastPsdSendTime = now;
            const ordered = new Float32Array(this._bvpRingSize);
            const len = this._bvpRing.length;
            if (len < this._bvpRingSize) {
                ordered.set(this._bvpRing, this._bvpRingSize - len);
            } else {
                ordered.set(this._bvpRing);
            }
            this._postIfReady('psd', {
                type: 'run',
                payload: { inputData: ordered },
            });
        }

        // Accumulate BVP samples for HRV (only if SQI is good enough)
        if (this.config.enableHrv && this._lastSqi >= this.config.hrvSqiThreshold) {
            this._bvpSamples.push({ t: timestamp, v: value });

            // Trim to sliding window (default 2 min)
            const maxMs = this.config.hrvMaxWindow * 1000;
            while (this._bvpSamples.length > 1 &&
                   timestamp - this._bvpSamples[0].t > maxMs) {
                this._bvpSamples.shift();
            }

            this._maybeComputeHrv(timestamp);
        }
    }

    /**
     * Handle PSD/SQI result — emit heart rate.
     * @private
     */
    _onPsdResult(data) {
        const { sqi, hr, psd, freq, peak } = data;
        if (hr == null) return;

        // Correct HR for actual framerate (matching original FacePhys formula)
        const correctedHr = hr / 30.0 / this._dval;
        this._lastHR = correctedHr;

        const sqiVal = sqi != null ? sqi : 0;
        this._lastSqi = sqiVal;
        this.emit('heartrate', {
            hr: correctedHr,
            sqi: sqiVal,
            psd, freq, peak,
            timestamp: Date.now(),
        });
    }

    /**
     * Handle emotion result.
     * @private
     */
    _onEmotionResult(data) {
        const { emotion, probs, time } = data;
        this.emit('emotion', {
            emotion: emotion || '',
            probs: probs || [],
            time,
            timestamp: Date.now(),
        });
    }

    /**
     * Handle gaze result.  If the softmax confidence for either axis falls
     * below `gazeConfidenceThreshold` (e.g. during a blink) the measurement
     * is discarded and no event is emitted — the consumer keeps whatever
     * the previous value was.
     * @private
     */
    _onGazeResult(data) {
        const { angles, confidence, time } = data;
        if (!angles || angles.length < 2) return;

        // Filter low-confidence frames (blinks / closed eyes)
        if (confidence && confidence.length >= 2) {
            const minConf = Math.min(confidence[0], confidence[1]);
            if (minConf < this.config.gazeConfidenceThreshold) {
                // Confidence too low — skip this frame, keep last accepted gaze
                return;
            }
        }

        const gazeEvent = {
            yaw: angles[0],
            pitch: angles[1],
            confidence: confidence || null,
            time,
            timestamp: Date.now(),
        };
        this._lastGaze = gazeEvent;
        this.emit('gaze', gazeEvent);
    }

    /**
     * Handle OCEC eye-state result. Emits an `'eyestate'` event whose payload
     * shape is `{ left, right, bothClosed, time, timestamp }` where each side
     * carries `{ prob, open }` — `prob` is the raw sigmoid probability of
     * being open and `open` is the boolean decision against
     * `config.eyeStateThreshold`.
     * @private
     */
    _onEyeStateResult(data) {
        const { leftProb, rightProb, time } = data;
        if (leftProb == null || rightProb == null) return;
        // Cache for adapter-side consumers (e.g. gaze gating)
        this._lastEyeState = { leftProb, rightProb, timestamp: Date.now() };
        const t = this.config.eyeStateThreshold;
        const left  = { prob: leftProb,  open: leftProb  >= t };
        const right = { prob: rightProb, open: rightProb >= t };
        this.emit('eyestate', {
            left,
            right,
            bothClosed: !left.open && !right.open,
            time,
            timestamp: Date.now(),
        });
    }

    // -----------------------------------------------------------------------
    // HRV pipeline
    // -----------------------------------------------------------------------

    /**
     * Run the HRV computation pipeline if enough time has passed.
     * @private
     */
    _maybeComputeHrv(timestamp) {
        if (timestamp - this._lastHrvTime < this.config.hrvUpdateInterval) return;

        const duration = this._bvpSamples.length > 1
            ? (this._bvpSamples[this._bvpSamples.length - 1].t - this._bvpSamples[0].t) / 1000
            : 0;

        if (duration < this.config.hrvMinDuration) return;

        const hrv = this.computeHrvFromSamples(this._bvpSamples);
        if (hrv) {
            this._lastHrvTime = timestamp;
            this.emit('hrv', { rmssd: hrv.rmssd, timestamp });
        }
    }

    /**
     * Run the full HRV pipeline on a set of BVP samples.
     * Exposed publicly so it can be unit-tested without workers.
     *
     * @param {{ t: number, v: number }[]} samples
     * @returns {{ rmssd: number } | null}
     */
    computeHrvFromSamples(samples) {
        if (!samples || samples.length < 8) return null;

        // 1. Detect peaks directly on the irregular samples (no spline /
        //    no uniform-grid resampling — too expensive at 200 Hz × 2 min).
        let peaks = detectBvpPeaks(samples);
        if (peaks.length < 4) return null;

        // 2. Reject amplitude outliers (samples[p].v supplies amplitudes
        //    via the array-like accessor — see rejectAbnormalPeaks docs).
        const ampView = { length: samples.length };
        for (let i = 0; i < samples.length; i++) ampView[i] = samples[i].v;
        peaks = rejectAbnormalPeaks(peaks, ampView);
        if (peaks.length < 4) return null;

        // 3. Refine each peak's *timestamp* with a 3-point parabolic fit on
        //    (t_{i-1}, v_{i-1}), (t_i, v_i), (t_{i+1}, v_{i+1}).
        //    On 30 Hz data this brings ±16 ms quantisation down to ~±1 ms.
        const peakTimes = refinePeaksParabolic(peaks, samples);

        // 4. RR intervals in ms (timestamps are already in ms — no fs term).
        const rr = [];
        for (let i = 1; i < peakTimes.length; i++) {
            rr.push(peakTimes[i] - peakTimes[i - 1]);
        }

        // 5. Filter RR intervals
        const rrQ = quotientFilterRR(rr);
        const rrF = madFilterRR(rrQ);

        // 6. Compute metrics
        return computeHrvMetrics(rrF);
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /**
     * Send init message with the right model buffers to each worker.
     * @private
     */
    _sendWorkerInit(name, worker) {
        switch (name) {
            case 'inference':
                worker.postMessage({
                    type: 'init',
                    payload: {
                        modelBuffer: this.models.rppg,
                        projBuffer: this.models.rppgProj,
                        stateJson: this._stateJson || {},
                    },
                });
                break;
            case 'psd':
                worker.postMessage({
                    type: 'init',
                    payload: {
                        sqiBuffer: this.models.sqi,
                        psdBuffer: this.models.psd,
                    },
                });
                break;
            case 'emotion':
                worker.postMessage({
                    type: 'init',
                    payload: { modelBuffer: this.models.emotion },
                });
                break;
            case 'gaze':
                worker.postMessage({
                    type: 'init',
                    payload: { modelBuffer: this.models.gaze },
                });
                break;
            case 'eye_state':
                worker.postMessage({
                    type: 'init',
                    payload: { modelBuffer: this.models.eyeState },
                });
                break;
        }
    }

    /**
     * Post a message to a worker only if it is created and ready.
     * @private
     */
    _postIfReady(name, message) {
        const w = this._workers[name];
        if (!w || !this._workerReady[name]) return;
        try {
            w.postMessage(message);
        } catch (err) {
            this.emit('error', { source: name, message: err?.message || String(err) });
        }
    }
}

export default VitalCamera;
