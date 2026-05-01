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
 * Inter-eye distance is anatomically tight: a human eye width is roughly
 * 0.45 * inter-eye-distance, so 0.55 gives a small margin around the outer
 * canthi while keeping the eye filling most of the 40x24 OCEC input — which
 * is what the model was trained on (tight eye crops from a wholebody detector).
 */
const EYE_BOX_WFRAC_IOD = 0.55;

/* ── Face bounding box padding factor ── */
const FACE_PAD = 0.25;

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

        /** @private 30fps frame throttle (matching original FacePhys) */
        this._frameInterval = 1000 / 30;

        /** @private smoothed dt for rPPG model (exponential moving average) */
        this._dval = 1 / 30;
        this._lastCaptureTime = 0;
        this._virtualTime = 0;

        /** @private trend update throttle (1 update/sec like original) */
        this._lastTrendUpdateTime = 0;
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

        // 1. Create VitalCamera core
        this._vs = new VitalCamera({
            ...this._vsConfig,
            models: this._models,
            workerBasePath: this._workerBasePath,
        });
        await this._vs.init();

        // 2. Load face detector
        if (!this._customDetector) {
            await this._loadBlazeFace();
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

        if (!detection) {
            // No face — still emit so UI can hide overlay
            this._kfBoxX = null; this._kfBoxY = null;
            this._kfBoxW = null; this._kfBoxH = null;
            this._vs?.emit('face', {
                detected: false,
                box: null,
                keypoints: null,
                videoWidth: w,
                videoHeight: h,
                timestamp,
            });
            return;
        }

        const { box: rawBox, keypoints } = detection;

        // ── Kalman-filter the face bounding box (matches FacePhys behaviour) ──
        let box;
        if (!this._kfBoxX) {
            this._kfBoxX = new KalmanFilter1D(rawBox.x);
            this._kfBoxY = new KalmanFilter1D(rawBox.y);
            this._kfBoxW = new KalmanFilter1D(rawBox.w);
            this._kfBoxH = new KalmanFilter1D(rawBox.h);
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
        this._vs?.emit('face', {
            detected: true,
            box: { ...box },
            keypoints: keypoints ? keypoints.map(kp => ({ ...kp })) : [],
            videoWidth: w,
            videoHeight: h,
            timestamp,
        });

        // ── Prepare model inputs ──
        const rppgBox = padBox(box, w, h);
        const rppgData = cropAndResize(ctx, rppgBox, RPPG_SIZE, RPPG_SIZE, 'simple');

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
            const emotionBox = padBox(box, w, h, 0.15);
            frameInput.emotionInput = cropAndResize(ctx, emotionBox, EMOTION_SIZE, EMOTION_SIZE, 'imagenet');
        }

        // Eye-state (every frame — single forward pass per eye is ~0.3 ms).
        // BlazeFace short-range keypoint order:
        //   [0]=right_eye, [1]=left_eye  — names from the *subject's* perspective.
        // We size the crop from the inter-eye distance (anatomically tight), not
        // face.w (which can be loose around hair/fringe), so the eye fills most
        // of the 40x24 OCEC input — matching the tight crops it was trained on.
        if (this._models.eyeState && keypoints && keypoints.length >= 2) {
            const dx = keypoints[0].x - keypoints[1].x;
            const dy = keypoints[0].y - keypoints[1].y;
            const iod = Math.hypot(dx, dy);
            if (iod > 4) {
                const rightBox = eyeBoxFromKeypoint(keypoints[0], iod, w, h);
                const leftBox  = eyeBoxFromKeypoint(keypoints[1], iod, w, h);
                if (rightBox.w > 1 && rightBox.h > 1 && leftBox.w > 1 && leftBox.h > 1) {
                    frameInput.eyeRightInput = cropAndResize(ctx, rightBox, EYE_W, EYE_H, 'simple');
                    frameInput.eyeLeftInput  = cropAndResize(ctx, leftBox,  EYE_W, EYE_H, 'simple');
                    this._lastEyeBoxes = { left: leftBox, right: rightBox };
                }
            }
        }

        // Gaze (5 Hz throttle, gated by eye-state — when both eyes have p(open) <
        // gazeEyeOpenGateProb (default 0.7) we skip the inference entirely.  No
        // 'gaze' event fires for that frame, so the demo Kalman just predicts
        // forward without a measurement update.
        if (this._models.gaze && (now - this._lastGazeTime > this._gazeInterval)) {
            const gateProb = this._vs?.config?.gazeEyeOpenGateProb ?? 0.7;
            const eyeSt    = this._vs?._lastEyeState;
            const eyesOpen = !eyeSt || (Math.max(eyeSt.leftProb, eyeSt.rightProb) >= gateProb);
            if (eyesOpen) {
                this._lastGazeTime = now;
                const gazeBox = padBox(box, w, h, 0.2);
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
        emotion:      'enet_b0_8_best_vgaf_dynamic_int8.tflite',
        gaze:         'mobileone_s0_gaze_float16.tflite',
        eyeState:     'ocec_p.tflite',
        faceDetector: 'blaze_face_short_range.tflite',
    };

    /**
     * Load all model files from a base path.
     * @param {string} basePath  Path to models directory (default './models/')
     * @param {Object} [options]
     * @param {boolean} [options.emotion=true]  Load the emotion model.
     * @param {boolean} [options.gaze=true]     Load the gaze model.
     * @param {boolean} [options.eyeState=true] Load the OCEC eye open/closed model.
     * @returns {Promise<Object>} Model buffers ready for VitalCamera.
     */
    static async loadModels(basePath = './models/', options = {}) {
        const { emotion = true, gaze = true, eyeState = true } = options;
        const base = basePath.endsWith('/') ? basePath : basePath + '/';
        const keys = ['rppg', 'rppgProj', 'sqi', 'psd', 'state'];
        if (emotion) keys.push('emotion');
        if (gaze) keys.push('gaze');
        if (eyeState) keys.push('eyeState');

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
