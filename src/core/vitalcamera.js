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

    // ── Face Landmarker dependency group ─────────────────────────────────
    // The features in this group all require MediaPipe Face Landmarker
    // (~3.8 MB model, ~15–50 ms / inference). Setting `enableFaceLandmarker`
    // to false forces all three sub-features off and switches to a lightweight
    // BlazeFace-only fallback path that gives just rPPG / HRV / head-pose /
    // emotion. The fallback is meant for low-end mobile / battery-sensitive
    // contexts where the heavy mesh model is not affordable.
    enableFaceLandmarker: true,
    enableEyeState: true,           // requires enableFaceLandmarker
    enableMouth: true,              // requires enableFaceLandmarker
    enableGaze: true,               // requires enableFaceLandmarker

    // ── Face-Landmarker-independent ──────────────────────────────────────
    enableEmotion: true,
    enableHeadPose: true,
    enableHrv: true,
    hrvMinDuration: 15,       // seconds
    hrvMaxWindow: 120,        // seconds — sliding window cap (2 minutes)
    hrvUpdateInterval: 1000,  // ms
    hrvSqiThreshold: 0.6,     // only accumulate BVP samples with SQI above this
    sqiThreshold: 0.38,
    gazeConfidenceThreshold: 0.04,
    eyeStateThreshold: 0.5,         // p(open) >= threshold → "open" (used for the public 0/1 flag + display)
    gazeEyeOpenGateProb: 0.6,       // skip gaze inference unless max(L,R) eye-open prob ≥ this
};


// ---------------------------------------------------------------------------
// IndexedDB state cache — persist inference warm-start state across sessions
// ---------------------------------------------------------------------------

const IDB_NAME = 'VitalCameraSDK';
const IDB_STORE = 'states';
const IDB_KEY = 'inferenceState';
const IDB_KEY_EMOTION_BASELINE = 'emotionBaseline';

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

/**
 * Load the persisted dynamic-emotion-calibration baseline from IndexedDB.
 *
 * Mirror of `_loadCachedState`. The dynamic EMA mode mutates the worker's
 * `baselineLogits` on every inference, so without persistence each new
 * session starts from the built-in default baseline and visibly drifts
 * during the first 5–10 s of use. Caching the most recent value lets a
 * returning user pick up where they left off.
 *
 * @returns {Promise<number[]|null>}  8-logit array, or null on miss / error
 */
async function _loadCachedBaseline() {
    try {
        const db = await _openIDB();
        return new Promise((resolve) => {
            const tx = db.transaction(IDB_STORE, 'readonly');
            const req = tx.objectStore(IDB_STORE).get(IDB_KEY_EMOTION_BASELINE);
            req.onsuccess = () => {
                const v = req.result;
                if (Array.isArray(v) && v.length === 8 && v.every(x => typeof x === 'number' && Number.isFinite(x))) {
                    resolve(v);
                } else {
                    resolve(null);
                }
            };
            req.onerror = () => resolve(null);
        });
    } catch (_) {
        return null;
    }
}

async function _saveCachedBaseline(baseline) {
    if (!Array.isArray(baseline) || baseline.length !== 8) return;
    try {
        const db = await _openIDB();
        const tx = db.transaction(IDB_STORE, 'readwrite');
        tx.objectStore(IDB_STORE).put(baseline, IDB_KEY_EMOTION_BASELINE);
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
     * @param {number}  [config.gazeEyeOpenGateProb=0.6]  Skip gaze inference unless max(L,R) eye-open prob ≥ this
     */
    constructor(config = {}) {
        // Mix in EventEmitter
        Object.assign(this, EventEmitterMixin);
        this._listeners = {};

        // Merge config with defaults
        this.config = { ...DEFAULTS, ...config };
        this.models = config.models || {};

        // ── Face Landmarker dependency validation ──────────────────────
        // eyestate / mouth / gaze all consume Face Landmarker output. If
        // the caller turned the master FL switch off but left a sub-feature
        // on, force the sub-feature off and warn — silently letting it
        // through would just produce no events with no explanation.
        if (!this.config.enableFaceLandmarker) {
            for (const dep of ['enableEyeState', 'enableMouth', 'enableGaze']) {
                if (this.config[dep]) {
                    // eslint-disable-next-line no-console
                    console.warn(
                        `[VitalCamera] ${dep} requires enableFaceLandmarker=true; ` +
                        `forcing it off.`
                    );
                    this.config[dep] = false;
                }
            }
        }

        // State flags
        this.isRunning = false;
        this._workersReady = false;

        // Workers
        this._workers = {};       // keyed by name
        this._workerReady = {};   // per-worker ready flag

        // Peak detector
        this._peakDetector = new RealtimePeakDetector();

        // BVP sample buffer for HRV (sent to PSD worker on a 1 Hz cadence —
        // the worker owns the actual peak-detect / RMSSD / gate pipeline)
        this._bvpSamples = [];    // { t: ms, v: number }
        this._lastHrvSendTime = 0;

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
        this._stateExportInterval = 60; // export every 60 rPPG frames (~2 s @ 30 fps)

        // Frame counter for periodic emotion-baseline export. Same cadence
        // (~2 s) but counted in emotion-result frames, since emotion is
        // the cadence at which the baseline actually mutates.
        this._emotionFrameCount = 0;
        this._baselineExportInterval = 4; // 4 emotion frames @ 2 Hz = 2 s

        // Cached baseline loaded from IDB at init time. Used as the starting
        // point fed to the emotion worker; later overridden if the consumer
        // supplies emotionCalibration.images / .baseline.
        this._cachedEmotionBaseline = null;

        // Whether dynamic-EMA mode is currently active. Drives the IDB cache
        // (load on init, auto-save every _baselineExportInterval frames):
        // when off we never touch IDB, so a static-baseline session never
        // sees a stale dynamic baseline pop in. Initialised from
        // `config.emotionDynamicHalfLifeMs` (which BrowserAdapter resolves
        // from `emotionCalibration.dynamic`); flipped at runtime by
        // `_setEmotionDynamic`.
        this._dynamicActive = (this.config.emotionDynamicHalfLifeMs ?? 0) > 0;

        // Last accepted gaze result (used when current frame is filtered out)
        this._lastGaze = null;

        // Latest eye-state probabilities (raw sigmoid). Updated every frame and
        // consumed by the adapter to gate gaze inference when both eyes are closed.
        // null until the first eye-state result arrives.
        this._lastEyeState = null;  // { leftProb, rightProb, timestamp }

        // Eye-state grace lock. The adapter sets this every time the motion
        // gate transitions from "head moving" → "head stable" (and on the very
        // first stable frame after construction). For 1 second after that, the
        // emitted left/right probs are forced to 0.6 — same value the
        // motion-active fast path uses — so the consumer sees one continuous
        // "uncertain" span across the moving + stabilising window. 0.6 sits
        // below the default gaze gate (0.7), so gaze inference is also skipped
        // during the lock with no extra plumbing.
        this._eyeBaselineLockedUntil = 0;
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

        // Load cached emotion-calibration baseline (sibling of the rPPG
        // state cache) — but only when dynamic mode is enabled. With dynamic
        // off, the baseline never mutates, so persisting / restoring it just
        // pollutes the static path with stale data from previous sessions.
        if (this._dynamicActive) {
            this._cachedEmotionBaseline = await _loadCachedBaseline();
        }

        const basePath = this.config.workerBasePath;  // null = auto Blob URL
        const workerNames = ['inference', 'psd'];
        if (this.config.enableEmotion && this.models.emotion) {
            workerNames.push('emotion');
        }
        if (this.config.enableGaze && this.models.gaze) {
            workerNames.push('gaze');
        }
        // (eyeState / OCEC dropped in 0.6.1 — eye state is now sourced from
        //  Face Landmarker blendshapes, gated by enableFaceLandmarker.)

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
     * @param {Float32Array}  [frame.eyeBatchInput]  [16,24,40,3]  RGB/255, no mean/std
     *                                                (8 candidate crops per eye)
     * @param {Array}         [frame.faceKeypoints] 6 BlazeFace keypoints
     */
    processFrame(frame) {
        if (!this.isRunning) return;

        const { rppgInput, dtVal, timestamp, emotionInput, gazeInput,
                eyeBatchInput, faceKeypoints } = frame;

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

        // Dispatch to eye-state worker (N candidates per eye × 2 eyes,
        // looped batch=1 inferences inside the worker).
        if (this.config.enableEyeState && eyeBatchInput && this._workers.eye_state) {
            const EYE_LEN = 24 * 40 * 3;
            const n = (eyeBatchInput.length / EYE_LEN) | 0;
            this._postIfReady('eye_state', {
                type: 'run',
                payload: { batch: eyeBatchInput, n },
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

        if (type === 'baselineResult' && data.payload) {
            // Mirror of state_exported: snapshot the dynamic emotion
            // baseline to IDB so a returning user skips warm-up wobble.
            // Triggered every _baselineExportInterval emotion frames from
            // _onEmotionResult; payload.requestId is unused here (the
            // SDK doesn't await this — IDB write is fire-and-forget).
            _saveCachedBaseline(data.payload.baseline);
            return;
        }

        if (type === 'hrv_result') {
            this._onHrvResult(data.payload || data);
            return;
        }

        if (type === 'probeResult' && data.payload) {
            const r = this._probeResolvers?.get(data.payload.requestId);
            if (r) {
                this._probeResolvers.delete(data.payload.requestId);
                if (data.payload.error) r.reject(new Error(data.payload.error));
                else r.resolve(data.payload.logits);
            }
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
            // (eye_state worker dropped in 0.6.1; FL-driven eye state is fed
            //  directly through `_onEyeStateResult` from the adapter.)
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

        // HRV cadence — runs every hrvUpdateInterval ms regardless of signal
        // quality so consumers can clear the display when there's no usable
        // estimate. Accumulation of BVP samples is still gated on SQI: bad
        // samples never enter the buffer, but the cadence ticks regardless,
        // and we emit a null `'hrv'` event with a `reject` reason when we
        // can't (or shouldn't) compute one.
        if (this.config.enableHrv) {
            const goodSqi = this._lastSqi >= this.config.hrvSqiThreshold;
            if (goodSqi) {
                this._bvpSamples.push({ t: timestamp, v: value });
            }
            const maxMs = this.config.hrvMaxWindow * 1000;
            while (this._bvpSamples.length > 1 &&
                   timestamp - this._bvpSamples[0].t > maxMs) {
                this._bvpSamples.shift();
            }

            const since = timestamp - this._lastHrvSendTime;
            if (since >= this.config.hrvUpdateInterval) {
                this._lastHrvSendTime = timestamp;
                const dur = this._bvpSamples.length > 1
                    ? (this._bvpSamples[this._bvpSamples.length - 1].t -
                       this._bvpSamples[0].t) / 1000
                    : 0;

                if (!goodSqi) {
                    // Current signal is too noisy — clear the consumer.
                    this._emitHrvNull('low_sqi');
                } else if (dur < this.config.hrvMinDuration) {
                    // Not enough clean buffer yet (warm-up).
                    this._emitHrvNull('warming_up');
                } else {
                    this._postIfReady('psd', {
                        type: 'hrv_run',
                        payload: { samples: this._bvpSamples.slice() },
                    });
                }
            }
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

        // Periodic baseline snapshot for the IDB cache. Skipped entirely
        // when dynamic mode is off — same role as the 60-frame
        // `export_state` trigger in _onInferenceResult, but counted in
        // emotion frames (since the baseline only mutates on each emotion
        // inference, not each rPPG frame).
        if (this._dynamicActive) {
            this._emotionFrameCount++;
            if (this._emotionFrameCount % this._baselineExportInterval === 0) {
                this._postIfReady('emotion', { type: 'getBaseline', payload: {} });
            }
        }
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
     * Handle the HRV worker's result. Always emits an `'hrv'` event so the
     * consumer never sees a stale value: if the worker rejected the window
     * (`rmssd === null`), we emit nulls + a `reject` reason.
     *
     * Reject reasons (forwarded from the worker / synthesised here):
     *   - `low_sqi`               : current SQI < `hrvSqiThreshold`
     *   - `warming_up`            : clean BVP buffer < `hrvMinDuration`
     *   - `too_few_samples`       : fewer than 30 BVP samples
     *   - `too_few_peaks`         : peak detector found < 6 peaks
     *   - `rr_below_phys_min`     : < 5 RR intervals in 300–2000 ms range
     *   - `too_few_after_outlier_filter`
     *   - `too_few_after_compensating_pairs`
     *   - `high_rejection_rate`   : >50 % of phys-range RRs got dropped as outliers
     *
     * @private
     */
    _onHrvResult(data) {
        if (!data) return;
        if (data.rmssd == null || !Number.isFinite(data.rmssd)) {
            this._emitHrvNull(data.reject || 'invalid');
            return;
        }
        this.emit('hrv', {
            rmssd:  data.rmssd,
            sdnn:   data.sdnn,
            meanRR: data.meanRR,
            n:      data.n,
            reject: null,
            timestamp: Date.now(),
        });
    }

    /**
     * Emit an `'hrv'` event with all metric fields nulled out and a `reject`
     * string so consumers can clear their display. Used both upstream
     * (low-SQI / warm-up) and downstream (worker gate failure).
     * @param {string} reject  Why the window was rejected.
     * @private
     */
    _emitHrvNull(reject) {
        this.emit('hrv', {
            rmssd:  null,
            sdnn:   null,
            meanRR: null,
            n:      0,
            reject,
            timestamp: Date.now(),
        });
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
        if (!this.config.enableEyeState) return;
        if (data == null || data.leftProb == null || data.rightProb == null) return;
        const now = Date.now();

        // Cache for adapter-side consumers (e.g. gaze gating)
        this._lastEyeState = {
            leftProb:  data.leftProb,
            rightProb: data.rightProb,
            timestamp: now,
        };

        const th = this.config.eyeStateThreshold;
        const left  = { prob: data.leftProb,  open: data.leftProb  >= th };
        const right = { prob: data.rightProb, open: data.rightProb >= th };

        this.emit('eyestate', {
            left,
            right,
            bothClosed: !left.open && !right.open,
            time: data.time,
            timestamp: now,
        });
    }

    /**
     * Start a 1-second grace lock on the eye-state output.
     *
     * Called by the adapter on every motion-gate clear (transition from
     * "head moving" → "head stable") and once on the first stable frame
     * after construction. While the grace lock is active `_onEyeStateResult`
     * returns the neutral 0.6, so the consumer sees a single continuous
     * "uncertain" span across motion + stabilisation, and gaze inference
     * (gated on eye-prob ≥ 0.7) is automatically skipped.
     *
     * @private
     */
    _resetEyeBaseline() {
        this._eyeBaselineLockedUntil = Date.now() + 1_000;
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
                    payload: {
                        modelBuffer: this.models.emotion,
                        // Precedence (highest first):
                        //   1. config.emotionBaseline       — explicit caller-supplied vector
                        //   2. _cachedEmotionBaseline       — IDB-restored from previous session
                        //   3. (undefined → worker default) — DEFAULT_ASIAN_BASELINE
                        // BrowserAdapter still overrides via setBaseline post-init when
                        // emotionCalibration.images / .baseline is supplied.
                        baselineLogits: this.config.emotionBaseline ?? this._cachedEmotionBaseline ?? undefined,
                    },
                });
                break;
            case 'gaze':
                worker.postMessage({
                    type: 'init',
                    payload: { modelBuffer: this.models.gaze },
                });
                break;
            // (eye_state init dropped in 0.6.1)
        }
    }

    /**
     * @internal — used by BrowserAdapter to install the user-specific
     * baseline computed from `emotionCalibration.images` at init time.
     * Not part of the public surface; emotion calibration is meant to
     * be invisible to the consumer.
     */
    _setEmotionBaseline(baselineLogits) {
        this._postIfReady('emotion', { type: 'setBaseline', payload: { baselineLogits } });
    }

    /**
     * @internal — used by BrowserAdapter to enable / disable dynamic EMA
     * baseline updates inside the emotion worker. Also flips
     * `_dynamicActive`, which gates the IDB baseline cache: if you toggle
     * dynamic mode on at runtime the SDK will start auto-saving from that
     * point on; toggling it off stops further saves. (Existing IDB entries
     * are not deleted — next time the user re-enables dynamic, the most
     * recently saved baseline is loaded as the starting point on init.)
     * @param {number|null} halfLifeMs  >0 ms → enable; null/0 → disable.
     */
    _setEmotionDynamic(halfLifeMs) {
        this._dynamicActive = typeof halfLifeMs === 'number' && halfLifeMs > 0;
        this._postIfReady('emotion', { type: 'setDynamic', payload: { halfLifeMs } });
    }

    /**
     * @internal — used by BrowserAdapter to run a single emotion inference
     * during the image-based calibration flow at init time. Returns the raw
     * logits without emitting an `'emotion'` event.
     *
     * @param {Float32Array} imgData  1×224×224×3 face crop, ImageNet-normalised
     * @returns {Promise<number[]>}    raw 8-class logits
     */
    _probeEmotion(imgData) {
        const w = this._workers.emotion;
        if (!w || !this._workerReady.emotion) {
            return Promise.reject(new Error('Emotion worker not ready'));
        }
        if (!this._probeResolvers) this._probeResolvers = new Map();
        if (this._nextProbeId === undefined) this._nextProbeId = 0;
        const requestId = ++this._nextProbeId;
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                if (this._probeResolvers.has(requestId)) {
                    this._probeResolvers.delete(requestId);
                    reject(new Error('Emotion probe timed out'));
                }
            }, 5000);
            this._probeResolvers.set(requestId, { resolve: (v) => { clearTimeout(timer); resolve(v); }, reject });
            w.postMessage({ type: 'probe', payload: { imgData, requestId } });
        });
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
