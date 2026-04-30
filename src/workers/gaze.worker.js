/**
 * @file gaze.worker.js
 * @description Gaze estimation Web Worker for the VitalCamera SDK.
 *
 * Runs a gaze estimation neural network inside a dedicated Web Worker thread
 * using LiteRT (TFLite WASM backend). The model uses a bin-based angle
 * classification approach: it outputs logits over 90 bins (4 degrees each,
 * covering -180 to +180 degrees) for both yaw and pitch, then computes
 * the expected angle via soft-argmax.
 *
 * Input: a 448x448x3 face/eye crop, ImageNet-normalized (NHWC layout).
 * Output: yaw and pitch angles in radians.
 *
 * Message protocol (type / payload):
 *   -> init    { modelBuffer }           Load the gaze model
 *   <- initDone                          Model ready
 *   -> run     { imgData }               Estimate gaze for one face crop
 *   <- result  { angles: [yaw, pitch], confidence: [yawConf, pitchConf], time }
 *   <- error   { msg }                   On any thrown error
 */

/* ── LiteRT runtime (loaded dynamically from CDN) ── */
let LiteRT = null;
let Tensor = null;
let model = null;

/* ── Constants ── */
const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/wasm/';

/** Number of classification bins for angle estimation. */
const NUM_BINS = 90;

/** Width of each bin in degrees. */
const BIN_WIDTH = 4;

/** Offset to center the angle range (bins cover -180 to +180 degrees). */
const ANGLE_OFFSET = 180;

// NOTE: softmax and decodeAngle duplicated from utils/math.js for worker isolation.
// Web Workers cannot import from main-thread ES modules without a bundler,
// so we keep local copies here.

/**
 * Numerically-stable softmax over an array of logits.
 * @param {Float32Array|number[]} logits
 * @returns {Float32Array} Probabilities summing to 1
 */
function softmax(logits) {
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxVal) maxVal = logits[i];
    }
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        exps[i] = Math.exp(logits[i] - maxVal);
        sum += exps[i];
    }
    for (let i = 0; i < logits.length; i++) exps[i] /= sum;
    return exps;
}

/**
 * Decode a bin-classification logit vector into an angle (radians) and
 * a confidence score (peak softmax probability).
 *
 * Uses soft-argmax: expected bin index from softmax probabilities, then
 * converts to degrees and finally radians. The peak probability indicates
 * how confident the model is — low values (near 1/NUM_BINS ≈ 0.011)
 * suggest the model cannot determine the angle (e.g. during a blink).
 *
 * @param {Float32Array|number[]} logits - Raw logits of length NUM_BINS
 * @returns {{ angle: number, confidence: number }} Angle in radians and peak probability
 */
function decodeAngle(logits) {
    const probs = softmax(logits);
    let angle = 0;
    let maxProb = 0;
    for (let i = 0; i < NUM_BINS; i++) {
        angle += probs[i] * i;
        if (probs[i] > maxProb) maxProb = probs[i];
    }
    return {
        angle: (angle * BIN_WIDTH - ANGLE_OFFSET) * Math.PI / 180,
        confidence: maxProb,
    };
}

/* ── Exported for testing (Node.js only) ── */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { softmax, decodeAngle, NUM_BINS, BIN_WIDTH, ANGLE_OFFSET };
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
 * Initialize the gaze estimation model.
 * @param {Object} params
 * @param {ArrayBuffer} params.modelBuffer - Gaze model (.tflite)
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
 * Run gaze estimation on a single face/eye image.
 * @param {Object} params
 * @param {Float32Array} params.imgData - Preprocessed face (1x448x448x3, ImageNet-normalized)
 */
async function handleRun({ imgData }) {
    if (!model) return;
    const start = performance.now();

    const inputTensor = new Tensor(imgData, [1, 448, 448, 3]);
    const results = model.run([inputTensor]);
    inputTensor.delete();

    // Two outputs: yaw and pitch logits, each [1, 90]
    const out0 = results[0].toTypedArray();
    const out1 = results[1].toTypedArray();
    results[0].delete();
    results[1].delete();

    const r0 = decodeAngle(out0);
    const r1 = decodeAngle(out1);
    const end = performance.now();

    self.postMessage({
        type: 'result',
        payload: {
            angles: [r0.angle, r1.angle],
            confidence: [r0.confidence, r1.confidence],
            time: end - start,
        }
    });
}
