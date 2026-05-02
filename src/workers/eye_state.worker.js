/**
 * @file eye_state.worker.js
 * @description Eye open/closed classification Web Worker for the VitalCamera SDK.
 *
 * Runs the OCEC (Open-Closed Eyes Classification) model — a tiny sigmoid
 * classifier that takes an eye crop and returns probability of being open.
 *
 * Model: PINTO0309/OCEC variant `p`, ONNX → TFLite (batch=1) via onnx2tf.
 *   Input :  [1, 24, 40, 3]  NHWC float32, RGB pixels in [0, 1]  (no mean/std)
 *   Output:  [1]              float32 sigmoid (0 = closed, 1 = open)
 *
 * The adapter packs N candidate crops per call (4 candidates × 2 eyes by default)
 * and the worker runs them as a sequential loop of batch=1 inferences, returning
 * one probability per crop. The downstream classifier in VitalCamera then
 * reduces the per-eye candidate probs against a per-user rolling KL baseline.
 *
 * Why loop instead of a fixed batched model: each `model.run()` for this 112 KB
 * net costs ~0.15 ms on modern CPUs, and a fixed-batch model only outperforms
 * looping when XNNPACK can split work across threads — which requires the host
 * page to be cross-origin-isolated (COOP/COEP). For a typical embedded SDK
 * deployment that is uncertain, so we keep the simpler single-batch model.
 *
 * Message protocol (type / payload):
 *   -> init    { modelBuffer }                    Load the OCEC model
 *   <- initDone                                   Model ready
 *   -> run     { batch, n }                       batch = Float32Array(n*EYE_LEN)
 *   <- result  { probs, time }                    Float32Array(n) sigmoids, ms
 *   <- error   { msg }                            On any thrown error
 */

/* ── LiteRT runtime (loaded dynamically from CDN) ── */
let LiteRT = null;
let Tensor = null;
let model = null;

/* ── Constants ── */
const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/wasm/';

/** Eye crop input height (matches OCEC training resolution). */
const EYE_H = 24;

/** Eye crop input width (matches OCEC training resolution). */
const EYE_W = 40;

/** Per-eye element count (H * W * 3 channels). */
const EYE_LEN = EYE_H * EYE_W * 3;

/* ── Exported for testing (Node.js only) ── */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EYE_H, EYE_W, EYE_LEN };
}

/* ── Message handler ── */
self.onmessage = async (e) => {
    const { type, payload } = e.data;
    try {
        if (type === 'init') await handleInit(payload);
        else if (type === 'run') await handleRun(payload);
    } catch (err) {
        self.postMessage({ type: 'error', msg: err.toString() });
    }
};

/**
 * Initialize the OCEC model.
 * @param {Object} params
 * @param {ArrayBuffer} params.modelBuffer  OCEC `.tflite` (batch=1)
 */
async function handleInit({ modelBuffer }) {
    const litertModule = await import('https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/+esm');
    LiteRT = litertModule;
    Tensor = litertModule.Tensor;

    const originalFetch = self.fetch;
    self.fetch = async (input, init) => {
        if (typeof input === 'string' && input.endsWith('.wasm')) {
            const fileName = input.split('/').pop();
            return originalFetch(`${WASM_BASE_URL}${fileName}`, init);
        }
        return originalFetch(input, init);
    };

    await LiteRT.loadLiteRt(WASM_BASE_URL);
    self.fetch = originalFetch;

    model = await LiteRT.loadAndCompile(
        URL.createObjectURL(new Blob([modelBuffer])),
        { accelerator: 'wasm' }
    );

    self.postMessage({ type: 'initDone' });
}

/**
 * Run OCEC sequentially on N packed crops and return a probability for each.
 *
 * @param {Object} params
 * @param {Float32Array} params.batch  Length n*EYE_LEN, RGB [0,1] NHWC interleaved
 * @param {number} params.n            Number of crops to evaluate
 */
async function handleRun({ batch, n }) {
    if (!model) return;
    if (!batch || !Number.isInteger(n) || n <= 0 || batch.length !== n * EYE_LEN) {
        self.postMessage({
            type: 'error',
            msg: `[eye_state] invalid input: n=${n}, batch.length=${batch?.length}, expected ${n * EYE_LEN}`,
        });
        return;
    }

    const start = performance.now();
    const probs = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        // Subarray view into the packed buffer — no copy.
        const view = batch.subarray(i * EYE_LEN, (i + 1) * EYE_LEN);
        const inT = new Tensor(view, [1, EYE_H, EYE_W, 3]);
        const outs = model.run([inT]);
        inT.delete();
        probs[i] = outs[0].toTypedArray()[0];
        outs[0].delete();
    }

    const end = performance.now();

    self.postMessage(
        {
            type: 'result',
            payload: { probs, time: end - start },
        },
        [probs.buffer],   // transfer to avoid copy
    );
}
