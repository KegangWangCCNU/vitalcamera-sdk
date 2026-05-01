/**
 * @file eye_state.worker.js
 * @description Eye open/closed classification Web Worker for the VitalCamera SDK.
 *
 * Runs the OCEC (Open-Closed Eyes Classification) model — a 112 KB sigmoid
 * classifier that takes an eye crop and returns probability of being open.
 *
 * Model: PINTO0309/OCEC variant `p`, ONNX → TFLite via onnx2tf.
 *   Input :  [N, 24, 40, 3]  NHWC float32, RGB pixels in [0, 1]  (no mean/std)
 *   Output:  [N]              float32 sigmoid (0 = closed, 1 = open)
 *
 * The adapter sends both eyes per frame; this worker runs two batch=1
 * inferences sequentially (≈ 0.5 ms total on CPU) and returns both probabilities.
 *
 * Message protocol (type / payload):
 *   -> init    { modelBuffer }                        Load the OCEC model
 *   <- initDone                                       Model ready
 *   -> run     { left, right }                        Two Float32Array(24*40*3)
 *   <- result  { leftProb, rightProb, time }          Sigmoid per eye, ms
 *   <- error   { msg }                                On any thrown error
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
 * @param {ArrayBuffer} params.modelBuffer  OCEC `.tflite` (≈ 112 KB)
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
 * Run OCEC on left and right eye crops sequentially.
 * @param {Object} params
 * @param {Float32Array} params.left   Left-eye crop, length 24*40*3, RGB [0,1]
 * @param {Float32Array} params.right  Right-eye crop, length 24*40*3, RGB [0,1]
 */
async function handleRun({ left, right }) {
    if (!model) return;
    if (!left || !right || left.length !== EYE_LEN || right.length !== EYE_LEN) {
        self.postMessage({ type: 'error', msg: '[eye_state] invalid input length' });
        return;
    }
    const start = performance.now();

    const inL = new Tensor(left, [1, EYE_H, EYE_W, 3]);
    const outsL = model.run([inL]);
    inL.delete();
    const leftProb = outsL[0].toTypedArray()[0];
    outsL[0].delete();

    const inR = new Tensor(right, [1, EYE_H, EYE_W, 3]);
    const outsR = model.run([inR]);
    inR.delete();
    const rightProb = outsR[0].toTypedArray()[0];
    outsR[0].delete();

    const end = performance.now();

    self.postMessage({
        type: 'result',
        payload: {
            leftProb,
            rightProb,
            time: end - start,
        }
    });
}
