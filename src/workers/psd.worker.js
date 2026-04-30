/**
 * @file psd.worker.js
 * @description SQI (Signal Quality Index) and PSD (Power Spectral Density)
 * heart rate estimation Web Worker for the VitalCamera SDK.
 *
 * Runs two models inside a dedicated Web Worker thread using LiteRT:
 *   1. SQI model  - evaluates signal quality of a 450-sample BVP window
 *   2. PSD model  - estimates heart rate via spectral analysis
 *
 * Input: 450 BVP samples (15 seconds at 30 fps).
 * Output: SQI value, estimated HR (bpm), frequency array, PSD array,
 * and peak frequency index.
 *
 * Message protocol (type / payload):
 *   -> init    { sqiBuffer, psdBuffer }  Load both models
 *   <- initDone                          Models ready
 *   -> run     { inputData }             Analyze a 450-sample BVP window
 *   <- result  { sqi, hr, freq, psd, peak, time }
 *   -> setMode { isLowPower }            Toggle low-power mode (throttles runs)
 *   <- error   { msg }                   On any thrown error
 */

/* ── LiteRT runtime (loaded dynamically from CDN) ── */
let LiteRT = null;
let Tensor = null;
let sqiModel = null;
let psdModel = null;
let lastRunTime = 0;
let lowPowerMode = false;

/* ── Constants ── */
const INPUT_SHAPE = [1, 450];
const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/wasm/';

/* ── Message handler ── */
self.onmessage = async (e) => {
    const { type, payload } = e.data;
    try {
        if (type === 'init') await handleInit(payload);
        else if (type === 'run') await handleRun(payload);
        else if (type === 'setMode') { lowPowerMode = payload.isLowPower; }
    } catch (err) {
        self.postMessage({ type: 'error', msg: err.toString() });
    }
};

/**
 * Initialize the SQI and PSD models.
 * @param {Object} params
 * @param {ArrayBuffer} params.sqiBuffer - SQI model (.tflite)
 * @param {ArrayBuffer} params.psdBuffer - PSD model (.tflite)
 */
async function handleInit({ sqiBuffer, psdBuffer }) {
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

    sqiModel = await LiteRT.loadAndCompile(URL.createObjectURL(new Blob([sqiBuffer])), { accelerator: 'wasm' });
    psdModel = await LiteRT.loadAndCompile(URL.createObjectURL(new Blob([psdBuffer])), { accelerator: 'wasm' });

    self.postMessage({ type: 'initDone' });
}

/**
 * Run SQI + PSD analysis on a 450-sample BVP window.
 * In low-power mode, runs are throttled to at most once every 500ms.
 *
 * @param {Object} params
 * @param {Float32Array|number[]} params.inputData - 450 BVP samples
 */
async function handleRun({ inputData }) {
    if (!sqiModel || !psdModel) return;

    // Throttle in low-power mode
    const now = performance.now();
    if (lowPowerMode && (now - lastRunTime < 500)) {
        return;
    }
    lastRunTime = now;
    const start = performance.now();

    const data = inputData instanceof Float32Array ? inputData : new Float32Array(inputData);
    const inputTensor = new Tensor(data, INPUT_SHAPE);

    // Run SQI model
    const sqiResults = sqiModel.run([inputTensor]);
    const sqiVal = sqiResults[0] ? sqiResults[0].toTypedArray()[0] : 0;
    if (sqiResults[0]) sqiResults[0].delete();

    // Run PSD model
    const psdResults = psdModel.run([inputTensor]);

    const tHr = psdResults[0];
    const tFreq = psdResults[1];
    const tPsd = psdResults[2];
    const tPeak = psdResults[3];

    const resultPayload = {
        sqi: sqiVal,
        hr: tHr ? tHr.toTypedArray()[0] : 0,
        freq: tFreq ? Array.from(tFreq.toTypedArray()) : [],
        psd: tPsd ? Array.from(tPsd.toTypedArray()) : [],
        peak: tPeak ? tPeak.toTypedArray()[0] : 0,
        time: performance.now() - start
    };

    psdResults.forEach(t => t && t.delete());
    inputTensor.delete();

    self.postMessage({
        type: 'result',
        payload: resultPayload
    });
}
