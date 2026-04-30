/**
 * @file emotion.worker.js
 * @description Emotion classification Web Worker for the VitalCamera SDK.
 *
 * Runs a facial emotion recognition model (8-class) inside a dedicated
 * Web Worker thread using LiteRT (TFLite WASM backend).
 *
 * Input: a 224x224x3 face crop, ImageNet-normalized (NHWC layout).
 * Output: top emotion label, index, and probability distribution over
 * 8 emotions: Anger, Contempt, Disgust, Fear, Happiness, Neutral,
 * Sadness, Surprise.
 *
 * Message protocol (type / payload):
 *   -> init    { modelBuffer }           Load the emotion model
 *   <- initDone                          Model ready
 *   -> run     { imgData }               Classify one face crop
 *   <- result  { emotion, emotionIdx, probs, time }
 *   <- error   { msg }                   On any thrown error
 */

/* ── LiteRT runtime (loaded dynamically from CDN) ── */
let LiteRT = null;
let Tensor = null;
let model = null;

/* ── Constants ── */
const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/wasm/';

/** The 8 emotion classes in model output order. */
const EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'];

// NOTE: softmax duplicated from utils/math.js for worker isolation.
// Web Workers cannot import from main-thread ES modules without a bundler,
// so we keep a local copy here.

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
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
        exps[i] = Math.exp(logits[i] - maxVal);
        sumExp += exps[i];
    }
    for (let i = 0; i < logits.length; i++) exps[i] /= sumExp;
    return exps;
}

/* ── Exported for testing (Node.js only) ── */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { softmax, EMOTIONS };
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
 * Initialize the emotion classification model.
 * @param {Object} params
 * @param {ArrayBuffer} params.modelBuffer - Emotion model (.tflite)
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
 * Run emotion classification on a single face image.
 * @param {Object} params
 * @param {Float32Array} params.imgData - Preprocessed face (1x224x224x3, ImageNet-normalized)
 */
async function handleRun({ imgData }) {
    if (!model) return;
    const start = performance.now();

    const inputTensor = new Tensor(imgData, [1, 224, 224, 3]);
    const results = model.run([inputTensor]);
    inputTensor.delete();

    const logits = results[0].toTypedArray();
    results[0].delete();

    const probs = softmax(logits);

    // Find the top-scoring emotion
    let topIdx = 0;
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[topIdx]) topIdx = i;
    }

    const end = performance.now();

    self.postMessage({
        type: 'result',
        payload: {
            emotion: EMOTIONS[topIdx],
            emotionIdx: topIdx,
            probs: Array.from(probs),
            time: end - start,
        }
    });
}
