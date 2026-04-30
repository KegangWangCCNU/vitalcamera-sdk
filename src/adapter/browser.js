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

/* ── ImageNet normalization constants ── */
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

/* ── Model input dimensions ── */
const RPPG_SIZE    = 36;
const EMOTION_SIZE = 224;
const GAZE_SIZE    = 448;

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
     *        If omitted, BlazeFace is loaded from CDN.
     * @param {boolean} [config.manageCamera=true]
     *        When true (default) and videoElement is provided, adapter opens
     *        the camera and sets video.srcObject. Set to false if you manage
     *        the MediaStream yourself but still want the auto frame loop.
     */
    constructor(config) {
        this._video = config.videoElement || null;
        this._models = config.models;
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

        /** @private BlazeFace detector instance */
        this._detector = null;

        /** @private RAF handle */
        this._rafId = null;

        /** @private running flag */
        this._running = false;

        /** @private last frame timestamp for dt */
        this._lastFrameTime = 0;

        /** @private frame counter for sub-sampling emotion/gaze */
        this._frameCount = 0;

        /** @private time-based throttle for emotion/gaze workers */
        this._lastEmotionTime = 0;
        this._lastGazeTime = 0;
        this._emotionInterval = 500;  // ms — match original FacePhys
        this._gazeInterval = 200;     // ms
    }

    // ──────────────────────────── Public API ────────────────────────────

    /**
     * Initialize VitalCamera core, load face detector, and optionally start
     * the camera (only if videoElement was provided and manageCamera is true).
     */
    async init() {
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
        const now = timestamp ?? performance.now();
        const dt = this._lastFrameTime > 0 ? (now - this._lastFrameTime) / 1000 : 1 / 30;
        this._lastFrameTime = now;
        this._frameCount++;

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
        this._processDetection(ctx, sw, sh, dt, now);
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
     * Load BlazeFace from CDN.
     * @private
     */
    async _loadBlazeFace() {
        const [tf, blazeface] = await Promise.all([
            import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core'),
            import('https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface'),
        ]);
        await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl');
        await tf.ready();
        this._detector = await blazeface.load({ maxFaces: 1 });
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
                this._plotWorker?.postMessage({
                    type: 'trend_data',
                    payload: { hr: data.hr, valid: data.sqi > 0.5 },
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

        const { box, keypoints } = detection;

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

        // Gaze (time-based throttle, default 200ms)
        if (this._models.gaze && (now - this._lastGazeTime > this._gazeInterval)) {
            this._lastGazeTime = now;
            const gazeBox = padBox(box, w, h, 0.2);
            frameInput.gazeInput = cropAndResize(ctx, gazeBox, GAZE_SIZE, GAZE_SIZE, 'imagenet');
        }

        // Feed to core
        try {
            this._vs?.processFrame(frameInput);
        } catch (_) { /* core emits errors via events */ }
    }

    /**
     * Run face detection using BlazeFace or custom detector.
     * @private
     */
    async _detectFace(source) {
        if (this._customDetector) {
            return this._customDetector(source);
        }

        if (!this._detector) return null;

        const predictions = await this._detector.estimateFaces(source, false);
        if (!predictions || predictions.length === 0) return null;

        const pred = predictions[0];
        const [x1, y1] = pred.topLeft;
        const [x2, y2] = pred.bottomRight;
        const box = { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
        const keypoints = (pred.landmarks || []).map(([x, y]) => ({ x, y }));

        return { box, keypoints };
    }
}
    