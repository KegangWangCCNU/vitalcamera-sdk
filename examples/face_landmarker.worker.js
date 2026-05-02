/**
 * @file face_landmarker.worker.js
 * @description CLASSIC Web Worker that hosts the MediaPipe Face Landmarker.
 *
 * Why classic and not module:
 *   `@mediapipe/tasks-vision` calls `importScripts(<wasm-glue.js>)` internally
 *   to load the WASM bootstrap. Module workers reject `importScripts` (it's
 *   classic-only), so a module worker fails with
 *     "Failed to execute 'importScripts' on 'WorkerGlobalScope':
 *      Module scripts don't support importScripts()."
 *   Classic workers permit both `importScripts` AND `await import()`, so we
 *   pull MediaPipe's ESM in via dynamic import on first use.
 *
 * Frame protocol:
 *   ← init   { wasmBase, modelPath, delegate }
 *   → initDone { ms }
 *   ← frame  { bitmap, timestamp }                       (bitmap is transferred)
 *   → result { time, landmarks, blinkL, blinkR }         landmarks Float32Array x478*3 transferred
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

            // Lazy ESM import of @mediapipe/tasks-vision. This works in a
            // classic worker because dynamic `import()` is a JS feature, not a
            // module-worker feature. Once loaded, MediaPipe's internal
            // `importScripts(...)` calls succeed because we're not a module worker.
            if (!FilesetResolver) {
                const mod = await import(
                    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm'
                );
                FilesetResolver = mod.FilesetResolver;
                FaceLandmarker  = mod.FaceLandmarker;
            }

            const fileset = await FilesetResolver.forVisionTasks(payload.wasmBase);
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

            let blinkL = 0, blinkR = 0;
            if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
                for (const c of result.faceBlendshapes[0].categories) {
                    if (c.categoryName === 'eyeBlinkLeft')  blinkL = c.score;
                    if (c.categoryName === 'eyeBlinkRight') blinkR = c.score;
                }
            }

            self.postMessage(
                { type: 'result', time, landmarks: flat, blinkL, blinkR },
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
