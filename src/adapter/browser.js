/**
 * @file browser.js
 * @description Browser adapter for the VitalCamera SDK.
 *
 * Two usage modes:
 *
 * **Managed mode** — pass a videoElement, adapter handles camera + frame loop:
 *   const adapter = new BrowserAdapter({ videoElement, models });
 *   await adapter.init();
 *   adapter.start();
 *
 * **Manual mode** — omit videoElement, feed frames yourself:
 *   const adapter = new BrowserAdapter({ models });
 *   await adapter.init();
 *   adapter.vitalcamera.start();
 *   // In your own rAF loop:
 *   adapter.processVideoFrame(videoOrCanvas);
 *
 * In both modes, the adapter emits a `'face'` event on every detected face
 * so the user can render their own overlay (bounding box, keypoints, etc.).
 *
 * @module adapter/browser
 */

import VitalCamera from '../core/vitalcamera.js';
import { estimateHeadPose } from '../core/headpose.js';
import { createWorker } from '../workers/loader.js';
import KalmanFilter1D from '../utils/kalman.js';

/* ── ImageNet normalization constants ── */
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

/* ── Model input dimensions ── */
const RPPG_SIZE    = 36;
const EMOTION_SIZE = 224;
const GAZE_SIZE    = 448;
const EYE_W        = 40;   // OCEC input width  (matches model)
const EYE_H        = 24;   // OCEC input height (matches model)

/**
 * Eye crop width as a fraction of the inter-eye distance (right_eye -> left_eye).
 *
 * Why inter-eye distance and not face-bbox width: the BlazeFace face box can be
 * loose (especially with hair / fringe), making `face.w * fraction` unreliable.
 *
 * 0.32 was chosen empirically against OCEC's training-time crops: the upstream
 * wholebody34 detector outputs tight bounding boxes around the visible eye
 * fissure (no margin), which anatomically corresponds to ≈ 0.30·iod. Earlier
 * versions used 0.55 here, which over-included eyebrow / skin and pushed the
 * model out of distribution — small faces (iod ≤ 50 px) routinely returned
 * confident-but-wrong probabilities. At 0.32 the eye fills the 40×24 input
 * the way the model expects.
 */
const EYE_BOX_WFRAC_IOD = 0.32;

/**
 * Per-frame motion gate for eye-state. Computed as max(|dKpL|, |dKpR|) / iod —
 * the larger eye-keypoint displacement between consecutive frames as a fraction
 * of the inter-eye distance. When this exceeds the threshold the head is moving
 * too fast for the crop to be reliable.
 *
 * Behaviour when triggered: skip the OCEC inference and synthesise a "neutral"
 * eyestate with prob = EYE_MOTION_NEUTRAL_PROB. 0.55 sits between the two
 * downstream thresholds:
 *   - eyeStateThreshold      = 0.5   → 0.55 ≥ 0.5 → "open" (display + CSV)
 *                                      so no spurious blink appears on screen
 *   - gazeEyeOpenGateProb    = 0.6   → 0.55 < 0.6 → gaze gated off
 *                                      so gaze isn't updated with stale data
 *
 * 0.10 lets normal talking-head movement through (~3-5%/frame) and triggers
 * on brisk turns (≥10°/frame ≈ 10-15%/frame).
 */
const EYE_MOTION_GATE = 0.10;
const EYE_MOTION_NEUTRAL_PROB = 0.55;

/* ── Face bounding box padding factor ── */
const FACE_PAD = 0.25;

/**
 * Process-noise for the face bounding-box Kalman filter.
 *
 * `1e-2` (paired with the default `KalmanFilter1D` measurementNoise=5e-1)
 * was the original tuning for BlazeFace short-range detections. 0.6.0
 * temporarily bumped this to 5e-2 anticipating the move to Face Landmarker,
 * but the FL path bypasses this Kalman entirely (FL bbox is computed from
 * the 478-landmark min/max directly), so the bump only made the BlazeFace-
 * driven `boxRaw` jumpier without any FL-side benefit. 0.6.3 reverts to
 * the original value — `boxRaw` smoothness now matches early SDK behaviour.
 */
const KF_BOX_Q = 1e-2;

/* ── Eye-state inference: one crop per eye, raw OCEC sigmoid as the answer. ── */

/**
 * Crop a face region from a canvas context, resize it to the target
 * dimensions, and return a normalized Float32Array in NHWC layout.
 *
 * @param {CanvasRenderingContext2D|OffscreenCanvasRenderingContext2D} ctx
 * @param {{ x: number, y: number, w: number, h: number }} box
 * @param {number} targetW
 * @param {number} targetH
 * @param {'imagenet'|'simple'} [normalize='imagenet']
 * @returns {Float32Array}
 */
function cropAndResize(ctx, box, targetW, targetH, normalize = 'imagenet') {
    const tmp = new OffscreenCanvas(targetW, targetH);
    const tmpCtx = tmp.getContext('2d');

    tmpCtx.drawImage(
        ctx.canvas,
        box.x, box.y, box.w, box.h,
        0, 0, targetW, targetH
    );

    const pixels = tmpCtx.getImageData(0, 0, targetW, targetH).data;
    const out = new Float32Array(targetH * targetW * 3);

    for (let i = 0, j = 0; i < pixels.length; i += 4, j += 3) {
        const r = pixels[i]     / 255;
        const g = pixels[i + 1] / 255;
        const b = pixels[i + 2] / 255;

        if (normalize === 'imagenet') {
            out[j]     = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            out[j + 1] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            out[j + 2] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        } else {
            out[j]     = r;
            out[j + 1] = g;
            out[j + 2] = b;
        }
    }

    return out;
}

/**
 * Build an axis-aligned eye crop box centered on an eye keypoint, sized
 * relative to the inter-eye distance (`iod`). The 5:3 aspect ratio matches
 * the OCEC model (40 wide × 24 high).
 *
 * @param {{x:number,y:number}} kp     Eye keypoint in pixel coordinates
 * @param {number} iod                 Inter-eye distance in pixels
 * @param {number} canvasW
 * @param {number} canvasH
 * @returns {{x:number,y:number,w:number,h:number}}
 */
function eyeBoxFromKeypoint(kp, iod, canvasW, canvasH) {
    const w = iod * EYE_BOX_WFRAC_IOD;
    const h = w * (EYE_H / EYE_W);
    let x = kp.x - w / 2;
    let y = kp.y - h / 2;
    let bw = w, bh = h;
    if (x < 0) { bw += x; x = 0; }
    if (y < 0) { bh += y; y = 0; }
    if (x + bw > canvasW) bw = canvasW - x;
    if (y + bh > canvasH) bh = canvasH - y;
    return { x, y, w: bw, h: bh };
}

/**
 * Pad and clamp a bounding box within canvas bounds.
 */
function padBox(raw, canvasW, canvasH, pad = FACE_PAD) {
    const pw = raw.w * pad;
    const ph = raw.h * pad;
    let x = raw.x - pw;
    let y = raw.y - ph;
    let w = raw.w + pw * 2;
    let h = raw.h + ph * 2;
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > canvasW) w = canvasW - x;
    if (y + h > canvasH) h = canvasH - y;
    return { x, y, w, h };
}


/**
 * Normalize the user-facing `emotionCalibration.dynamic` config into a single
 * positive `halfLifeMs` number (or 0 when disabled). Accepts:
 *
 *   true                          → enabled, default 5000 ms half-life
 *   false / undefined / null      → disabled (returns 0)
 *   { halfLifeMs: N }             → enabled with N (must be > 0; else 5000)
 *   { enabled: false }            → disabled
 *   { enabled: true, halfLifeMs } → enabled with N
 *
 * Returning 0 lets every downstream gate just check `> 0` without needing
 * to redo the boolean / object dispatch.
 *
 * @param {boolean|{halfLifeMs?: number, enabled?: boolean}|null|undefined} dyn
 * @returns {number}  0 when disabled; positive halfLifeMs when enabled
 */
const DEFAULT_DYNAMIC_HALF_LIFE_MS = 5000;
function _resolveDynamicHalfLifeMs(dyn) {
    if (dyn === true) return DEFAULT_DYNAMIC_HALF_LIFE_MS;
    if (dyn === false || dyn == null) return 0;
    if (typeof dyn === 'object') {
        if (dyn.enabled === false) return 0;
        const hl = dyn.halfLifeMs;
        if (typeof hl === 'number' && hl > 0) return hl;
        if (dyn.enabled === true) return DEFAULT_DYNAMIC_HALF_LIFE_MS;
        // object without enabled / halfLifeMs → enabled with default,
        // matches legacy behaviour of `dynamic: {}`
        return DEFAULT_DYNAMIC_HALF_LIFE_MS;
    }
    return 0;
}

/**
 * Browser adapter for the VitalCamera SDK.
 *
 * @example <caption>Managed mode — adapter controls camera</caption>
 * const adapter = new BrowserAdapter({
 *     videoElement: document.getElementById('cam'),
 *     models: { rppg, rppgProj, sqi, psd },
 * });
 * await adapter.init();
 * adapter.vitalcamera.on('heartrate', ({ hr }) => console.log(hr));
 *
 * // Draw your own face overlay
 * adapter.vitalcamera.on('face', ({ box, keypoints, videoWidth, videoHeight }) => {
 *     drawMyOverlay(box, keypoints);
 * });
 * adapter.start();
 *
 * @example <caption>Manual mode — you control camera & preview</caption>
 * const adapter = new BrowserAdapter({ models: { rppg, rppgProj, sqi, psd } });
 * await adapter.init();
 * adapter.vitalcamera.on('heartrate', ({ hr }) => console.log(hr));
 * adapter.vitalcamera.on('face', ({ box, keypoints }) => drawOverlay(box, keypoints));
 * adapter.vitalcamera.start();
 *
 * // Your own camera + rAF loop
 * const stream = await navigator.mediaDevices.getUserMedia({ video: true });
 * myVideo.srcObject = stream;
 * function loop() {
 *     adapter.processVideoFrame(myVideo);
 *     requestAnimationFrame(loop);
 * }
 * loop();
 */
export default class BrowserAdapter {
    /**
     * @param {Object} config
     * @param {HTMLVideoElement} [config.videoElement]
     *        Video element for camera preview. If omitted, camera is NOT opened
     *        automatically — use {@link processVideoFrame} to feed frames manually.
     * @param {Object} config.models
     *        Model ArrayBuffers (rppg, rppgProj, sqi, psd, emotion?, gaze?).
     * @param {Object} [config.vitalcameraConfig]
     *        Extra config passed through to VitalCamera constructor.
     * @param {string} [config.workerBasePath='./workers/']
     * @param {string} [config.cameraFacing='user']
     *        Camera facing mode (only used when videoElement is provided).
     * @param {Object} [config.canvases]
     *        Optional plot canvases: { bvp, psd, trend }.
     * @param {Function} [config.faceDetector]
     *        Custom face detector: (sourceElement) => Promise<{box, keypoints}|null>.
     *        If omitted, MediaPipe FaceDetector (BlazeFace short-range) is
     *        loaded from CDN and reads `blaze_face_short_range.tflite` from
     *        `modelBasePath` (or from `models.faceDetector` if provided).
     * @param {boolean} [config.manageCamera=true]
     *        When true (default) and videoElement is provided, adapter opens
     *        the camera and sets video.srcObject. Set to false if you manage
     *        the MediaStream yourself but still want the auto frame loop.
     * @param {Object} [config.emotionCalibration]
     *        Per-user emotion calibration — applied invisibly to every
     *        `'emotion'` event. Pass an array of base64 face images of the
     *        end-user (≥ 2, more is better; resting / neutral expressions),
     *        e.g. captured from a one-time onboarding flow:
     *          `emotionCalibration: { images: ['data:image/jpeg;base64,…', …] }`
     *        Each image is face-detected, cropped, ImageNet-normalised and
     *        run through the emotion model; the per-class average logit
     *        becomes the user's baseline. If omitted, the SDK falls back to
     *        a built-in default baseline tuned to Asian resting faces.
     */
    constructor(config = {}) {
        this._video = config.videoElement || null;
        this._models = config.models || null;
        this._modelBasePath = config.modelBasePath || './models/';
        this._vsConfig = config.vitalcameraConfig || {};
        this._workerBasePath = config.workerBasePath || null;
        this._cameraFacing = config.cameraFacing || 'user';
        this._canvases = config.canvases || null;
        this._customDetector = config.faceDetector || null;
        this._manageCamera = config.manageCamera !== false;
        this._emotionCalibration = config.emotionCalibration || null;

        /**
         * Resolved dynamic-EMA half-life from `config.emotionCalibration.dynamic`.
         * 0 means dynamic mode is off — VitalCamera uses this both to gate the
         * IDB baseline cache (load + auto-save) and to decide whether to send
         * `setDynamic` to the emotion worker post-init.
         * @type {number}
         */
        this._emotionDynamicHalfLifeMs = _resolveDynamicHalfLifeMs(this._emotionCalibration?.dynamic);

        /** @type {VitalCamera|null} */
        this._vs = null;

        /** @type {MediaStream|null} */
        this._stream = null;

        /** @type {Worker|null} */
        this._plotWorker = null;

        /** @private scratch canvas for frame capture */
        this._scratch = null;
        this._scratchCtx = null;

        /** @private MediaPipe FaceDetector instance (BlazeFace short-range) */
        this._detector = null;

        /** @private RAF handle */
        this._rafId = null;

        /** @private running flag */
        this._running = false;

        /** @private last frame timestamp for dt */
        this._lastFrameTime = 0;

        /** @private frame counter for sub-sampling emotion/gaze */
        this._frameCount = 0;

        /** @private time-based throttle for emotion/gaze workers (eye-state runs every frame, cost is ~0.3 ms) */
        this._lastEmotionTime = 0;
        this._lastGazeTime = 0;
        this._emotionInterval = 500;  // ms — match original FacePhys
        this._gazeInterval = 200;     // ms

        /** @private Kalman filters for face bounding box (x, y, w, h) */
        this._kfBoxX = null;
        this._kfBoxY = null;
        this._kfBoxW = null;
        this._kfBoxH = null;

        /** @private Kalman filters for eye crop boxes (left, right × x,y,w,h) — same defaults as face box */
        this._kfEyeLX = null; this._kfEyeLY = null; this._kfEyeLW = null; this._kfEyeLH = null;
        this._kfEyeRX = null; this._kfEyeRY = null; this._kfEyeRW = null; this._kfEyeRH = null;

        /** @private Previous-frame eye keypoints (px) for motion-gate computation */
        this._lastEyeKpL = null;
        this._lastEyeKpR = null;
        /**
         * @private
         * Was the motion gate active on the previous frame? Initial value is
         * `true` so the very first stable frame triggers `_resetEyeBaseline()`
         * — this seeds the 1s grace lock at startup, treating boot as if
         * motion had just ended.
         */
        this._eyeMotionWasActive = true;

        /** @private 30fps frame throttle (matching original FacePhys) */
        this._frameInterval = 1000 / 30;

        /** @private smoothed dt for rPPG model (exponential moving average) */
        this._dval = 1 / 30;
        this._lastCaptureTime = 0;
        this._virtualTime = 0;

        /** @private trend update throttle (1 update/sec like original) */
        this._lastTrendUpdateTime = 0;

        /**
         * @private
         * Face Landmarker (MediaPipe Face Mesh + Blendshapes) state.
         * Runs in its own worker at `_fmIntervalMs` Hz, non-blocking — if a
         * frame is in flight when the next tick comes around we just skip it,
         * so a slow device that takes 200 ms per inference effectively runs
         * Face Landmarker at 5 fps without backing up the camera loop.
         *
         * Outputs the SDK consumes:
         *   blinkL / blinkR  → ARKit-style blendshape scores [0=open, 1=closed].
         *                      We feed `prob_open = 1 - blinkX` into the existing
         *                      `_onEyeStateResult` path. For a typical user,
         *                      open ≈ 0.9+ and fully closed ≈ 0.4 — i.e. the
         *                      blendshape rarely saturates, so the default
         *                      `eyeStateThreshold` of 0.5 works.
         */
        this._fmWorker = null;
        this._fmInFlight = false;
        this._fmLastSentT = 0;
        this._fmIntervalMs = 1000 / 15;   // 15 fps target
        this._fmReady = false;

        /**
         * @private
         * Rolling buffer of recent jawOpen blendshape values, used to
         * derive a "speaking" boolean. Variance over the window > threshold
         * means the jaw is articulating (speech / chewing); low variance
         * with sustained openness is a yawn/surprise; low variance with
         * jaw closed is silence.
         */
        this._jawOpenBuf = [];
        this._jawOpenWindowMs = 1_000;
        this._jawOpenSpeakingStdThresh = 0.04;
    }

    // ──────────────────────────── Public API ────────────────────────────

    /**
     * Initialize VitalCamera core, load face detector, and optionally start
     * the camera (only if videoElement was provided and manageCamera is true).
     */
    async init() {
        // 0. Auto-load models if not provided
        if (!this._models) {
            this._models = await BrowserAdapter.loadModels(this._modelBasePath);
        }

        // 1. Create VitalCamera core (emotion worker boots with the SDK's
        //    built-in Asian default baseline; we may override it below if
        //    the caller supplied calibration images).
        this._vs = new VitalCamera({
            ...this._vsConfig,
            models: this._models,
            workerBasePath: this._workerBasePath,
            // Tell VitalCamera up-front whether dynamic mode is on so its
            // init() can decide whether to load the persisted baseline from
            // IndexedDB and register the periodic save callback. Without
            // this hint the IDB cache would always load — even when the
            // caller didn't ask for dynamic — silently mutating their
            // emotion output across sessions.
            emotionDynamicHalfLifeMs: this._emotionDynamicHalfLifeMs,
        });
        await this._vs.init();

        // 2. Load face detector
        if (!this._customDetector) {
            await this._loadBlazeFace();
        }

        // 2a. Initialise Face Landmarker — only if the master toggle is on
        //     (vitalcameraConfig.enableFaceLandmarker, default true). When it's
        //     off the SDK runs in the lightweight BlazeFace-only fallback:
        //     no eyestate / no mouth / no gaze events, just rPPG / HRV /
        //     emotion / head-pose. Best-effort: if FL init fails (network /
        //     GPU delegate), we still emit the error event but the rest of
        //     the SDK keeps running.
        if (this._vs?.config?.enableFaceLandmarker !== false) {
            try {
                await this._initFaceLandmarker();
            } catch (err) {
                this._vs?.emit('error', { source: 'faceLandmarker', message: err.message });
            }
        }

        // 2b. Emotion calibration. Three options, can mix:
        //
        //     emotionCalibration: {
        //         images:   [b64, …],        // compute baseline from face photos
        //         baseline: [n0, n1, …, n7], // OR pass an 8-vector of logits directly
        //         dynamic:  { halfLifeMs },  // AND/OR enable runtime EMA on top
        //     }
        //
        //   - `images` precedence over `baseline` if both supplied.
        //   - `dynamic` is independent: enables continuous EMA so a sustained
        //     expression slides the baseline that way, calibration re-centres
        //     it on Neutral, and the user-visible signal becomes "deviation
        //     from your typical expression" rather than absolute emotion.
        const cal = this._emotionCalibration;
        if (cal && this._models.emotion && this._vs._workers.emotion) {
            // 2b.i — images path
            if (Array.isArray(cal.images) && cal.images.length > 0) {
                try {
                    const baseline = await this._computeEmotionBaselineFromImages(cal.images);
                    this._vs._setEmotionBaseline(baseline);
                } catch (err) {
                    this._vs?.emit('error', { source: 'emotionCalibration', message: err.message });
                }
            }
            // 2b.ii — direct baseline distribution path (8 logits). Wins over
            // images only if `images` wasn't provided; if both are supplied,
            // the images-derived baseline takes precedence (the sequence here
            // means baseline is set last only if images path wasn't taken).
            else if (Array.isArray(cal.baseline) && cal.baseline.length === 8) {
                this._vs._setEmotionBaseline(cal.baseline);
            }
            // 2b.iii — dynamic (EMA) baseline mode. The resolved half-life
            // already encodes all the legal forms of `cal.dynamic`
            // (true / false / { halfLifeMs } / { enabled }). 0 means off,
            // any positive number means on.
            if (this._emotionDynamicHalfLifeMs > 0) {
                this._vs._setEmotionDynamic(this._emotionDynamicHalfLifeMs);
            }
        }

        // 3. Start camera only if videoElement provided and manageCamera is on
        if (this._video && this._manageCamera) {
            this._ensurePlaysinline(this._video);
            await this._startCamera(this._cameraFacing);
        }

        // 4. Plot worker
        if (this._canvases) {
            await this._initPlotWorker();
        }
    }

    /**
     * Start the automatic frame loop (managed mode).
     * Requires videoElement to have been provided in the constructor.
     */
    start() {
        if (!this._video) {
            throw new Error(
                'BrowserAdapter.start() requires a videoElement. ' +
                'For manual mode, call vitalcamera.start() and use processVideoFrame() instead.'
            );
        }
        if (this._running) return;
        this._running = true;
        this._vs.start();
        this._lastFrameTime = performance.now();
        this._frameCount = 0;
        this._initScratchCanvas(this._video.videoWidth || 640, this._video.videoHeight || 480);
        this._tick();
    }

    /**
     * Stop the automatic frame loop. Camera stream stays open.
     */
    stop() {
        this._running = false;
        this._vs?.stop();
        if (this._rafId != null) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
    }

    /**
     * Fully shut down: stop processing, release camera, terminate workers.
     */
    async destroy() {
        this.stop();

        if (this._stream) {
            for (const track of this._stream.getTracks()) track.stop();
            this._stream = null;
        }
        if (this._plotWorker) {
            this._plotWorker.terminate();
            this._plotWorker = null;
        }
        if (this._fmWorker) {
            this._fmWorker.terminate();
            this._fmWorker = null;
            this._fmReady = false;
            this._fmInFlight = false;
        }
        if (this._vs) {
            this._vs.destroy();
            this._vs = null;
        }

        this._detector = null;
        this._scratch = null;
        this._scratchCtx = null;
    }

    /**
     * Process a single video frame manually (manual mode).
     *
     * Call this from your own requestAnimationFrame loop when you manage
     * the camera and preview yourself. The adapter will:
     *   1. Draw the source to an internal scratch canvas
     *   2. Run face detection
     *   3. Emit a `'face'` event with box, keypoints, and video dimensions
     *   4. Crop, normalize, and feed data to VitalCamera workers
     *
     * @param {HTMLVideoElement|HTMLCanvasElement|OffscreenCanvas|ImageBitmap} source
     *        Any drawable source. Must have valid width/height.
     * @param {number} [timestamp=performance.now()] Frame timestamp in ms.
     */
    processVideoFrame(source, timestamp) {
        const captureTime = timestamp ?? Date.now();
        this._frameCount++;

        // Smooth dt with exponential moving average (matching original FacePhys)
        if (this._lastCaptureTime > 0) {
            const rawDt = (captureTime - this._lastCaptureTime) / 1000;
            this._dval = this._dval * 0.997 + 0.003 * rawDt;
            this._virtualTime += this._dval * 1000;
            this._virtualTime = this._virtualTime * 0.997 + 0.003 * captureTime;
        } else {
            this._virtualTime = captureTime;
        }
        this._lastCaptureTime = captureTime;

        // Determine source dimensions
        const sw = source.videoWidth || source.width || 640;
        const sh = source.videoHeight || source.height || 480;

        // Ensure scratch canvas matches
        if (!this._scratch || this._scratch.width !== sw || this._scratch.height !== sh) {
            this._initScratchCanvas(sw, sh);
        }

        const ctx = this._scratchCtx;
        ctx.drawImage(source, 0, 0, sw, sh);

        // Face detection + processing (async, fire-and-forget)
        this._processDetection(ctx, sw, sh, this._dval, this._virtualTime);
    }

    /**
     * Switch to a different camera by device ID.
     * Only works in managed mode (videoElement provided).
     *
     * @param {string} deviceId
     */
    async switchCamera(deviceId) {
        if (!this._video) return;
        const wasRunning = this._running;
        this.stop();

        if (this._stream) {
            for (const track of this._stream.getTracks()) track.stop();
        }

        this._stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: deviceId } },
            audio: false,
        });
        this._video.srcObject = this._stream;
        await this._video.play();

        if (wasRunning) this.start();
    }

    /**
     * List available video input devices.
     * @returns {Promise<MediaDeviceInfo[]>}
     */
    async getCameras() {
        const devices = await navigator.mediaDevices.enumerateDevices();
        return devices.filter(d => d.kind === 'videoinput');
    }

    /**
     * Access the underlying VitalCamera instance for event subscriptions.
     *
     * Events emitted by VitalCamera:
     * - `'face'`      — { box, keypoints, videoWidth, videoHeight, timestamp }
     * - `'heartrate'` — { hr, sqi, timestamp }
     * - `'bvp'`       — { value, timestamp }
     * - `'hrv'`       — { rmssd, timestamp }
     * - `'emotion'`   — { emotion, probs, timestamp }
     * - `'gaze'`      — { yaw, pitch, timestamp }
     * - `'eyestate'`  — { left:{prob,open}, right:{prob,open}, bothClosed, timestamp }
     * - `'headpose'`  — { yaw, pitch, roll, normal, timestamp }
     * - `'beat'`      — { ibi, timestamp }
     * - `'ready'`     — {}
     * - `'error'`     — { source, message }
     *
     * @type {VitalCamera}
     */
    get vitalcamera() {
        return this._vs;
    }

    // ──────────────────────────── Private ────────────────────────────

    /**
     * Ensure the video element has iOS-required attributes for inline playback.
     * Without these, iOS Safari opens a fullscreen player which blocks canvas capture.
     * @private
     */
    _ensurePlaysinline(video) {
        video.setAttribute('playsinline', '');
        video.setAttribute('webkit-playsinline', '');
        video.setAttribute('muted', '');
        video.muted = true;
    }

    /**
     * Load MediaPipe FaceDetector (BlazeFace short-range) from CDN.
     * Tries GPU delegate first and falls back to CPU on failure.
     * The .tflite model is fetched from {@link _modelBasePath} +
     * `blaze_face_short_range.tflite` unless overridden via
     * `models.faceDetector` (ArrayBuffer / Uint8Array).
     * @private
     */
    async _loadBlazeFace() {
        const { FaceDetector, FilesetResolver } = await import(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm'
        );
        const vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm'
        );

        // Allow callers to provide the model as a buffer; otherwise fetch by URL.
        const baseOptions = {};
        const buf = this._models && this._models.faceDetector;
        if (buf) {
            baseOptions.modelAssetBuffer =
                buf instanceof Uint8Array ? buf : new Uint8Array(buf);
        } else {
            baseOptions.modelAssetPath =
                this._modelBasePath + 'blaze_face_short_range.tflite';
        }

        try {
            this._detector = await FaceDetector.createFromOptions(vision, {
                baseOptions: { ...baseOptions, delegate: 'GPU' },
                runningMode: 'VIDEO',
            });
        } catch (err) {
            this._detector = await FaceDetector.createFromOptions(vision, {
                baseOptions: { ...baseOptions, delegate: 'CPU' },
                runningMode: 'VIDEO',
            });
            // eslint-disable-next-line no-console
            console.log('FaceDetector GPU delegate not available, using CPU.');
        }
    }

    /**
     * Spawn the Face Landmarker worker, ship the WASM base + .task model URL,
     * and wire the message handler that dispatches `eyestate` events.
     *
     * Eye-state contract: `1 - blinkX` is fed into the existing
     * `_onEyeStateResult` legacy single-shot path on VitalCamera, which is
     * how the motion-gate fallback already injects neutral 0.6 — so the
     * downstream `eyestate` event payload is unchanged.
     *
     * @private
     */
    async _initFaceLandmarker() {
        const buf = this._models && this._models.faceLandmarker;
        let modelPath;
        if (buf) {
            // ArrayBuffer / Uint8Array passed in by caller — wrap as Blob URL
            // so the worker can fetch it via `modelAssetPath`. (tasks-vision
            // also accepts `modelAssetBuffer`, but our worker uses path.)
            const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
            modelPath = URL.createObjectURL(new Blob([u8]));
        } else {
            // Resolve to absolute URL so the worker's blob context can fetch it
            modelPath = new URL(
                this._modelBasePath + 'face_landmarker.task',
                location.href
            ).href;
        }

        const worker = await createWorker('face_landmarker', this._workerBasePath);
        this._fmWorker = worker;

        worker.addEventListener('message', (ev) => {
            const m = ev.data || {};
            if (m.type === 'initDone') {
                this._fmReady = true;
            } else if (m.type === 'result') {
                this._fmInFlight = false;
                const now = Date.now();

                // 1 - blink → open prob. Adapter feeds the legacy single-shot
                // path on VitalCamera; OCEC isn't involved.
                this._vs?._onEyeStateResult({
                    leftProb:  1 - m.blinkL,
                    rightProb: 1 - m.blinkR,
                    time: m.time,
                });

                // ── jawOpen → 'mouth' event with speaking heuristic ──
                // Speaking ≠ "mouth open": yawning has high jawOpen but no
                // motion. Articulation has lots of small up-down variance.
                // We compute std over a rolling 1s window of jawOpen samples;
                // if std exceeds a threshold the jaw is moving → speaking.
                this._jawOpenBuf.push({ t: now, v: m.jawOpen });
                const cutoff = now - this._jawOpenWindowMs;
                while (this._jawOpenBuf.length && this._jawOpenBuf[0].t < cutoff) {
                    this._jawOpenBuf.shift();
                }
                let speaking = false;
                let jawStd = 0;
                if (this._jawOpenBuf.length >= 5) {
                    let sum = 0;
                    for (const e of this._jawOpenBuf) sum += e.v;
                    const mean = sum / this._jawOpenBuf.length;
                    let varSum = 0;
                    for (const e of this._jawOpenBuf) varSum += (e.v - mean) ** 2;
                    jawStd = Math.sqrt(varSum / this._jawOpenBuf.length);
                    speaking = jawStd > this._jawOpenSpeakingStdThresh;
                }
                if (this._vs?.config?.enableMouth) {
                    this._vs.emit('mouth', {
                        jawOpen:  m.jawOpen,
                        jawStd,
                        speaking,
                        time:     m.time,
                        timestamp: now,
                    });
                }

                // Stash for advanced consumers (we don't use landmarks in the
                // adapter today, but exposing them is a no-op cost-wise).
                this._lastFaceLandmarks = m.landmarks;
            } else if (m.type === 'noFace') {
                this._fmInFlight = false;
                this._lastFaceLandmarks = null;
                // Face left frame — flush jaw history so we don't carry
                // stale variance into the next reappearance.
                this._jawOpenBuf.length = 0;
            } else if (m.type === 'error') {
                this._fmInFlight = false;
                this._vs?.emit('error', { source: 'faceLandmarker', message: m.msg });
                // eslint-disable-next-line no-console
                console.error('[face_landmarker]', m.msg, m.stack);
            }
        });
        worker.addEventListener('error', (e) => {
            this._fmInFlight = false;
            this._vs?.emit('error', { source: 'faceLandmarker', message: e.message });
        });

        worker.postMessage({
            type: 'init',
            payload: {
                wasmBase: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm',
                modelPath,
                delegate: 'GPU',
            },
        });
    }

    /**
     * Push the current frame to the Face Landmarker worker if it's time and
     * the worker isn't already busy. Non-blocking — when inference is slow
     * (e.g. 200 ms on a low-end device) we just skip ticks instead of queuing.
     *
     * @param {HTMLCanvasElement|OffscreenCanvas} sourceCanvas
     * @param {number} now  performance.now() timestamp
     * @private
     */
    async _pumpFaceLandmarker(sourceCanvas, now) {
        if (!this._fmReady || this._fmInFlight) return;
        if ((now - this._fmLastSentT) < this._fmIntervalMs) return;

        this._fmInFlight = true;
        this._fmLastSentT = now;

        let bitmap;
        try {
            bitmap = await createImageBitmap(sourceCanvas);
        } catch (_) {
            this._fmInFlight = false;
            return;
        }
        try {
            this._fmWorker.postMessage(
                { type: 'frame', payload: { bitmap, timestamp: now } },
                [bitmap],
            );
        } catch (err) {
            this._fmInFlight = false;
            try { bitmap.close(); } catch (_) { /* ignore */ }
        }
    }

    /**
     * Start camera and attach to video element.
     * @private
     */
    async _startCamera(facing) {
        this._stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false,
        });
        this._video.srcObject = this._stream;
        await this._video.play();
    }

    /**
     * Create or resize the scratch canvas.
     * @private
     */
    _initScratchCanvas(w, h) {
        this._scratch = new OffscreenCanvas(w, h);
        this._scratchCtx = this._scratch.getContext('2d');
    }

    /**
     * Set up plot worker with OffscreenCanvases.
     * @private
     */
    async _initPlotWorker() {
        this._plotWorker = await createWorker('plot', this._workerBasePath);

        const dpr = globalThis.devicePixelRatio || 1;
        const transferList = [];
        const payload = { dpr };

        for (const key of ['bvp', 'psd', 'trend']) {
            const canvas = this._canvases[key];
            if (!canvas) continue;

            let offscreen;
            if (typeof HTMLCanvasElement !== 'undefined' && canvas instanceof HTMLCanvasElement) {
                offscreen = canvas.transferControlToOffscreen();
            } else {
                offscreen = canvas;
            }

            payload[key + 'Canvas'] = offscreen;
            // Pass CSS dimensions (not pixel) — plot worker applies dpr via scale()
            payload[key + 'Width']  = canvas.width  / dpr;
            payload[key + 'Height'] = canvas.height / dpr;
            transferList.push(offscreen);
        }

        this._plotWorker.postMessage({ type: 'init', payload }, transferList);

        if (this._vs) {
            this._vs.on('bvp', (data) => {
                this._plotWorker?.postMessage({ type: 'bvp_data', payload: data.value });
            });
            this._vs.on('heartrate', (data) => {
                // Throttle trend updates to 1/second (matching original)
                const now = Date.now();
                if (now - this._lastTrendUpdateTime < 1000) return;
                this._lastTrendUpdateTime = now;
                this._plotWorker?.postMessage({
                    type: 'trend_data',
                    payload: { hr: data.hr, valid: data.sqi > 0.38 },
                });
            });
        }
    }

    /**
     * Managed-mode frame loop via requestAnimationFrame.
     * @private
     */
    _tick() {
        if (!this._running) return;
        this._rafId = requestAnimationFrame(() => this._tick());

        // Throttle to 30fps (matching original FacePhys)
        const now = performance.now();
        const elapsed = now - (this._lastTickTime || 0);
        if (elapsed < this._frameInterval) return;
        this._lastTickTime = now - (elapsed % this._frameInterval);

        const video = this._video;
        if (!video || video.readyState < 2) return;

        this.processVideoFrame(video);
    }

    /**
     * Detect face, emit 'face' event, crop inputs, and feed VitalCamera.
     * @private
     */
    async _processDetection(ctx, w, h, dt, timestamp) {
        let detection = null;
        try {
            detection = await this._detectFace(ctx.canvas);
        } catch (_) {
            return; // non-fatal
        }

        // Push the current frame to the Face Landmarker worker (15 fps,
        // non-blocking — see `_pumpFaceLandmarker`). Fire-and-forget; the
        // inflight flag handles backpressure on slow devices.
        if (this._fmReady) {
            this._pumpFaceLandmarker(ctx.canvas, performance.now());
        }

        if (!detection) {
            // No face — still emit so UI can hide overlay
            this._kfBoxX = null; this._kfBoxY = null;
            this._kfBoxW = null; this._kfBoxH = null;
            this._kfEyeLX = null; this._kfEyeLY = null; this._kfEyeLW = null; this._kfEyeLH = null;
            this._kfEyeRX = null; this._kfEyeRY = null; this._kfEyeRW = null; this._kfEyeRH = null;
            this._lastEyeKpL = null; this._lastEyeKpR = null;
            // Face left frame — treat the next reappearance as motion-end
            // so the baseline is reset and the 1s grace lock kicks in.
            this._eyeMotionWasActive = true;
            this._vs?.emit('face', {
                detected: false,
                box: null,
                boxRaw: null,
                keypoints: null,
                videoWidth: w,
                videoHeight: h,
                timestamp,
            });
            return;
        }

        const { box: rawBox, keypoints } = detection;

        // ── Kalman-filter the face bounding box ──
        let box;
        if (!this._kfBoxX) {
            this._kfBoxX = new KalmanFilter1D(rawBox.x, KF_BOX_Q);
            this._kfBoxY = new KalmanFilter1D(rawBox.y, KF_BOX_Q);
            this._kfBoxW = new KalmanFilter1D(rawBox.w, KF_BOX_Q);
            this._kfBoxH = new KalmanFilter1D(rawBox.h, KF_BOX_Q);
            box = { ...rawBox };
        } else {
            box = {
                x: this._kfBoxX.update(rawBox.x),
                y: this._kfBoxY.update(rawBox.y),
                w: this._kfBoxW.update(rawBox.w),
                h: this._kfBoxH.update(rawBox.h),
            };
        }

        // ── Emit face event so user can render their own overlay ──
        // `box` is the "ready-to-draw" face bbox:
        //   - When Face Landmarker is active and has produced landmarks:
        //     min/max of the 478 anatomical points with a 3 % lateral shrink.
        //   - Otherwise: the Kalman-smoothed BlazeFace bbox unchanged
        //     (same one the rPPG / emotion / gaze crops use). The SDK no
        //     longer applies any manual height-stretch / shift — emotion
        //     and gaze are trained on tight face crops, so feeding the
        //     raw detector bbox preserves the model's expected geometry.
        // `boxRaw` is always the Kalman-smoothed BlazeFace bbox, regardless
        // of FL state. With FL on, `box` is tighter (anatomical landmarks);
        // with FL off, `box === boxRaw`.
        const boxRaw = { ...box };
        let tightBox = box;
        const lms = this._lastFaceLandmarks;
        if (lms && lms.length >= 3) {
            let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
            for (let i = 0; i < lms.length; i += 3) {
                const px = lms[i] * w, py = lms[i + 1] * h;
                if (px < xMin) xMin = px;
                if (px > xMax) xMax = px;
                if (py < yMin) yMin = py;
                if (py > yMax) yMax = py;
            }
            const shrink = (xMax - xMin) * 0.03;
            xMin += shrink; xMax -= shrink;
            tightBox = { x: xMin, y: yMin, w: xMax - xMin, h: yMax - yMin };
        }

        this._vs?.emit('face', {
            detected: true,
            box: tightBox,
            boxRaw,
            keypoints: keypoints ? keypoints.map(kp => ({ ...kp })) : [],
            videoWidth: w,
            videoHeight: h,
            timestamp,
        });

        // ── Prepare model inputs ──
        // rPPG / emotion / gaze (FL-off fallback) all use the Kalman-smoothed
        // BlazeFace bbox directly. Emotion (HSEmotion ENet-B0) and gaze
        // (L2CS-Net MobileOne-S0) are trained on tight face crops, so the
        // raw detector bbox preserves the geometry the models expect.
        const rppgData = cropAndResize(ctx, box, RPPG_SIZE, RPPG_SIZE, 'simple');

        const frameInput = {
            rppgInput: rppgData,
            dtVal: dt,
            timestamp,
        };

        // Head pose
        if (keypoints && keypoints.length >= 6) {
            frameInput.faceKeypoints = keypoints;
        }

        // Emotion (time-based throttle, default 500ms)
        const now = performance.now();
        if (this._models.emotion && (now - this._lastEmotionTime > this._emotionInterval)) {
            this._lastEmotionTime = now;
            frameInput.emotionInput = cropAndResize(ctx, box, EMOTION_SIZE, EMOTION_SIZE, 'imagenet');
        }

        // (OCEC eye-state pipeline removed in 0.6.1. Eye state is now sourced
        //  from the MediaPipe Face Landmarker `eyeBlinkLeft/Right` blendshapes
        //  fed directly into `_onEyeStateResult` from `_initFaceLandmarker`.)

        // Gaze (5 Hz throttle, gated by eye-state — when both eyes have p(open) <
        // gazeEyeOpenGateProb (default 0.6) we skip the inference entirely.  No
        // 'gaze' event fires for that frame, so the demo Kalman just predicts
        // forward without a measurement update.
        if (this._models.gaze && (now - this._lastGazeTime > this._gazeInterval)) {
            const gateProb = this._vs?.config?.gazeEyeOpenGateProb ?? 0.6;
            const eyeSt    = this._vs?._lastEyeState;
            const eyesOpen = !eyeSt || (Math.max(eyeSt.leftProb, eyeSt.rightProb) >= gateProb);
            if (eyesOpen) {
                this._lastGazeTime = now;

                // Compute the gaze face bbox. L2CS-Net (the network behind
                // mobileone_s0_gaze_float16.tflite) wants a 448×448 face crop
                // with the eyes at roughly upper-third — the convention from
                // its MTCNN-bbox training pipeline. Two sources, in order of
                // preference:
                //
                //   1. Face Mesh landmark min/max  (preferred). 478 anatomical
                //      points — the bbox is tight to the visible face surface
                //      and temporally stable from Face Landmarker's internal
                //      tracking. Eyes always land at consistent in-crop
                //      coordinates regardless of how loose BlazeFace got.
                //
                //   2. BlazeFace bbox  (fallback, used during the first 1–2 s
                //      while Face Landmarker boots, or if its model fails to
                //      load). Padded by FACE_PAD as before.
                let gazeBox;
                const lms = this._lastFaceLandmarks;
                if (lms && lms.length >= 3) {
                    let xMin = Infinity, xMax = -Infinity;
                    let yMin = Infinity, yMax = -Infinity;
                    for (let i = 0; i < lms.length; i += 3) {
                        const px = lms[i]     * w;
                        const py = lms[i + 1] * h;
                        if (px < xMin) xMin = px;
                        if (px > xMax) xMax = px;
                        if (py < yMin) yMin = py;
                        if (py > yMax) yMax = py;
                    }
                    gazeBox = padBox(
                        { x: xMin, y: yMin, w: xMax - xMin, h: yMax - yMin },
                        w, h, 0.2,
                    );
                } else {
                    // FL not yet ready → use the Kalman-smoothed BlazeFace
                    // bbox directly. No extra padding (the gaze net was
                    // trained on tight face crops).
                    gazeBox = box;
                }

                frameInput.gazeInput = cropAndResize(ctx, gazeBox, GAZE_SIZE, GAZE_SIZE, 'imagenet');
            }
        }

        // Feed to core
        try {
            this._vs?.processFrame(frameInput);
        } catch (_) { /* core emits errors via events */ }
    }

    /**
     * Run face detection using MediaPipe FaceDetector or a custom detector.
     *
     * Returns box in pixel coordinates and keypoints converted from
     * MediaPipe's normalized [0, 1] space into pixel coordinates so the
     * `'face'` event payload stays drawable on the source canvas.
     *
     * MediaPipe BlazeFace short-range keypoint order:
     *   0 = right_eye, 1 = left_eye, 2 = nose_tip,
     *   3 = mouth, 4 = right_ear, 5 = left_ear
     *
     * @private
     */
    async _detectFace(source) {
        if (this._customDetector) {
            return this._customDetector(source);
        }

        if (!this._detector) return null;

        // MediaPipe FaceDetector.detectForVideo is synchronous and requires a
        // monotonically increasing timestamp in milliseconds.
        const result = this._detector.detectForVideo(source, performance.now());
        if (!result || !result.detections || result.detections.length === 0) {
            return null;
        }

        const det = result.detections[0];
        const bb = det.boundingBox;
        const box = {
            x: bb.originX,
            y: bb.originY,
            w: bb.width,
            h: bb.height,
        };

        // Keypoints from tasks-vision are normalized [0, 1]; convert to pixels.
        const sw = source.width || source.videoWidth || 1;
        const sh = source.height || source.videoHeight || 1;
        const keypoints = (det.keypoints || []).map(({ x, y }) => ({
            x: x * sw,
            y: y * sh,
        }));

        return { box, keypoints };
    }

    // ──────────────────────── Static constants ─────────────────────────

    /** Emotion class labels (8-class model output order). */
    static EMOTION_LABELS = ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise'];

    /** Emoji for each emotion class (same order as labels). */
    static EMOTION_EMOJIS = ['\u{1F620}','\u{1F612}','\u{1F922}','\u{1F628}','\u{1F604}','\u{1F610}','\u{1F622}','\u{1F632}'];

    /** Color for each emotion class bar chart (same order as labels). */
    static EMOTION_COLORS = ['#c96442','#c9964a','#8b7bb5','#5a9e6f','#c9a84a','#5a8fa8','#5a9e8f','#b5637a'];

    /** Default model filenames — maps internal keys to default filenames. */
    static MODEL_FILES = {
        rppg:         'model.tflite',
        rppgProj:     'proj.tflite',
        sqi:          'sqi_model.tflite',
        psd:          'psd_model.tflite',
        state:        'state.gz',
        emotion:        'enet_b0_8_best_vgaf_dynamic_int8.tflite',
        gaze:           'mobileone_s0_gaze_float16.tflite',
        faceDetector:   'blaze_face_short_range.tflite',
        faceLandmarker: 'face_landmarker.task',
        // (eyeState / ocec_p.tflite removed in 0.6.1; eye state is now
        //  sourced from Face Landmarker blendshapes.)
    };

    /**
     * Load all model files from a base path.
     * @param {string} basePath  Path to models directory (default './models/')
     * @param {Object} [options]
     * @param {boolean} [options.emotion=true]         Load the emotion model.
     * @param {boolean} [options.gaze=true]            Load the gaze model.
     * @param {boolean} [options.faceLandmarker=true]  Load the MediaPipe Face Landmarker
     *        bundle (~3.8 MB). Required for `eyestate`, `mouth`, and `gaze`
     *        events; pass false alongside `vitalcameraConfig.enableFaceLandmarker:false`
     *        for a lightweight rPPG / HRV / emotion / head-pose only build.
     * @returns {Promise<Object>} Model buffers ready for VitalCamera.
     */
    /**
     * Compute an emotion-calibration baseline from a set of base64 face images.
     * Averages the raw logits returned by the emotion worker — the same
     * formula that examples/calibrate.html uses, but headless.
     *
     * @param {string[]} images  Array of base64 / data-URL strings.
     * @returns {Promise<number[]>}  8-element baselineLogits.
     * @private
     */
    async _computeEmotionBaselineFromImages(images) {
        const N = 8;
        const sum = new Float64Array(N);
        let nOk = 0;
        for (const src of images) {
            let img;
            try {
                img = await new Promise((resolve, reject) => {
                    const el = new Image();
                    el.onload  = () => resolve(el);
                    el.onerror = () => reject(new Error('image load failed'));
                    el.src = src;
                });
            } catch (_) { continue; }

            const w = img.naturalWidth || img.width;
            const h = img.naturalHeight || img.height;
            if (!w || !h) continue;

            const tmp = new OffscreenCanvas(w, h);
            const tctx = tmp.getContext('2d');
            tctx.drawImage(img, 0, 0, w, h);

            // Try face detection; if it fails, fall back to using the whole image.
            let cropBox = { x: 0, y: 0, w, h };
            try {
                const det = await this._detectFace(tmp);
                if (det && det.box) cropBox = padBox(det.box, w, h, 0.15);
            } catch (_) { /* keep full-image fallback */ }

            const imgData = cropAndResize(tctx, cropBox, EMOTION_SIZE, EMOTION_SIZE, 'imagenet');
            try {
                const logits = await this._vs._probeEmotion(imgData);
                if (Array.isArray(logits) && logits.length === N) {
                    for (let i = 0; i < N; i++) sum[i] += logits[i];
                    nOk++;
                }
            } catch (_) { /* skip this image */ }
        }
        if (nOk < 2) {
            throw new Error(`emotionCalibration.images: only ${nOk} image(s) usable, need ≥ 2`);
        }
        const baseline = new Array(N);
        for (let i = 0; i < N; i++) baseline[i] = sum[i] / nOk;
        return baseline;
    }

    static async loadModels(basePath = './models/', options = {}) {
        const { emotion = true, gaze = true, faceLandmarker = true } = options;
        const base = basePath.endsWith('/') ? basePath : basePath + '/';
        const keys = ['rppg', 'rppgProj', 'sqi', 'psd', 'state'];
        if (emotion) keys.push('emotion');
        if (gaze) keys.push('gaze');
        if (faceLandmarker) keys.push('faceLandmarker');

        const buffers = {};
        await Promise.all(keys.map(async (key) => {
            const url = base + BrowserAdapter.MODEL_FILES[key];
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`[VitalCamera] Failed to load model "${key}" from ${url} (${resp.status})`);
            buffers[key] = await resp.arrayBuffer();
        }));
        return buffers;
    }
}
