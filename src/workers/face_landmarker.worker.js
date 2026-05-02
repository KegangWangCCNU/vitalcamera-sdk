/**
 * @file face_landmarker.worker.js
 * @description Face Landmarker (MediaPipe tasks-vision) hosted in a CLASSIC
 * Web Worker. Dedicated worker because the model is heavy (~3.8 MB) and
 * inference takes ~15–50 ms; main-thread execution would block the rPPG
 * pipeline that runs at the camera frame rate.
 *
 * Why classic and not module:
 *   `@mediapipe/tasks-vision` calls `importScripts(<wasm-glue.js>)` internally
 *   to bootstrap WASM, and module workers reject `importScripts`. Classic
 *   workers permit BOTH `importScripts` AND `await import()`, so we pull the
 *   tasks-vision ESM in via dynamic import on first init.
 *
 * Outputs that the SDK consumes:
 *   - `blinkL`, `blinkR`  → ARKit-style blendshape scores [0=open, 1=closed].
 *                           SDK converts to `prob_open = 1 - blinkX` and feeds
 *                           the existing eyestate event.
 *   - `time`              → inference latency in ms (per frame).
 *   - `landmarks`         → 478 (x,y,z) packed as Float32Array(478*3),
 *                           transferred. The adapter doesn't currently use
 *                           these but they're zero-extra-cost to emit.
 *
 * Protocol:
 *   ← init   { wasmBase, modelPath, delegate }
 *   → initDone { ms }
 *   ← frame  { bitmap, timestamp }                            (bitmap is transferred)
 *   → result { time, landmarks, blinkL, blinkR, jawOpen }     (landmarks transferred)
 *   → noFace { time }
 *   → error  { msg, stack }
 */

let FilesetResolver = null;
let FaceLandmarker = null;
let landmarker = null;

self.onmessage = async (e) => {
    const { type, payload } = e.data || {};
    try {
        if (type === 'init') {
            const t0 = performance.now();

            if (!FilesetResolver) {
                const mod = await import(
                    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm'
                );
                FilesetResolver = mod.FilesetResolver;
                FaceLandmarker  = mod.FaceLandmarker;
            }

            const fileset = await FilesetResolver.forVisionTasks(payload.wasmBase);

            // Try GPU first, fall back to CPU. GPU is typically 2–3× faster
            // on desktop / mid-tier mobile. Some browsers / GPUs reject the
            // delegate (e.g. WebGL 1, headless), in which case CPU still works.
            try {
                landmarker = await FaceLandmarker.createFromOptions(fileset, {
                    baseOptions: {
                        modelAssetPath: payload.modelPath,
                        delegate: payload.delegate || 'GPU',
                    },
                    runningMode: 'VIDEO',
                    numFaces: 1,
                    outputFaceBlendshapes: true,
                    outputFacialTransformationMatrixes: false,
                });
            } catch (gpuErr) {
                landmarker = await FaceLandmarker.createFromOptions(fileset, {
                    baseOptions: {
                        modelAssetPath: payload.modelPath,
                        delegate: 'CPU',
                    },
                    runningMode: 'VIDEO',
                    numFaces: 1,
                    outputFaceBlendshapes: true,
                    outputFacialTransformationMatrixes: false,
                });
            }
            self.postMessage({ type: 'initDone', ms: performance.now() - t0 });
        } else if (type === 'frame') {
            if (!landmarker) {
                payload.bitmap?.close?.();
                self.postMessage({ type: 'error', msg: 'frame received before init' });
                return;
            }
            const { bitmap, timestamp } = payload;
            const t0 = performance.now();
            const result = landmarker.detectForVideo(bitmap, timestamp);
            const time = performance.now() - t0;
            bitmap.close();

            const hasFace = result.faceLandmarks && result.faceLandmarks.length > 0;
            if (!hasFace) {
                self.postMessage({ type: 'noFace', time });
                return;
            }

            const lms = result.faceLandmarks[0];
            const flat = new Float32Array(lms.length * 3);
            for (let i = 0; i < lms.length; i++) {
                flat[3*i]     = lms[i].x;
                flat[3*i + 1] = lms[i].y;
                flat[3*i + 2] = lms[i].z;
            }

            let blinkL = 0, blinkR = 0, jawOpen = 0;
            if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
                for (const c of result.faceBlendshapes[0].categories) {
                    if (c.categoryName === 'eyeBlinkLeft')  blinkL = c.score;
                    if (c.categoryName === 'eyeBlinkRight') blinkR = c.score;
                    if (c.categoryName === 'jawOpen')       jawOpen = c.score;
                }
            }

            self.postMessage(
                { type: 'result', time, landmarks: flat, blinkL, blinkR, jawOpen },
                [flat.buffer],
            );
        }
    } catch (err) {
        try { e.data?.payload?.bitmap?.close?.(); } catch (_) { /* ignore */ }
        self.postMessage({
            type: 'error',
            msg: err && (err.message || err.toString()),
            stack: err && err.stack,
        });
    }
};
