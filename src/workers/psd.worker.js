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
 *   -> init     { sqiBuffer, psdBuffer }  Load both models
 *   <- initDone                           Models ready
 *   -> run      { inputData }             Analyze a 450-sample BVP window (HR/SQI)
 *   <- result   { sqi, hr, freq, psd, peak, time }
 *   -> hrv_run  { samples: [{t,v},…] }    Run HRV pipeline on a long window
 *   <- hrv_result { rmssd, cv, confidence, time }   rmssd === null when gates fail
 *   -> setMode  { isLowPower }            Toggle low-power mode (throttles runs)
 *   <- error    { msg }                   On any thrown error
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


/* ── HRV pipeline (inlined from hrv.js so the worker is self-contained) ── */

function _detectBvpPeaks(samples, minIbiMs = 333) {
    const n = samples.length;
    if (n < 6) return [];
    let sum = 0;
    for (const s of samples) sum += s.v;
    const mean = sum / n;
    let sumSq = 0;
    for (const s of samples) { const d = s.v - mean; sumSq += d * d; }
    const std = Math.sqrt(sumSq / n);
    const thresh = mean + 0.3 * std;
    const peaks = [];
    for (let i = 2; i < n - 2; i++) {
        const v = samples[i].v;
        if (v > thresh
            && v >= samples[i - 1].v && v >= samples[i + 1].v
            && v >= samples[i - 2].v && v >= samples[i + 2].v) {
            if (peaks.length === 0 || samples[i].t - samples[peaks[peaks.length - 1]].t >= minIbiMs) {
                peaks.push(i);
            } else if (v > samples[peaks[peaks.length - 1]].v) {
                peaks[peaks.length - 1] = i;
            }
        }
    }
    return peaks;
}

function _detectBvpValleys(samples, minIbiMs = 333) {
    const n = samples.length;
    if (n < 6) return [];
    let sum = 0;
    for (const s of samples) sum += s.v;
    const mean = sum / n;
    let sumSq = 0;
    for (const s of samples) { const d = s.v - mean; sumSq += d * d; }
    const std = Math.sqrt(sumSq / n);
    const thresh = mean - 0.3 * std;
    const valleys = [];
    for (let i = 2; i < n - 2; i++) {
        const v = samples[i].v;
        if (v < thresh
            && v <= samples[i - 1].v && v <= samples[i + 1].v
            && v <= samples[i - 2].v && v <= samples[i + 2].v) {
            if (valleys.length === 0 || samples[i].t - samples[valleys[valleys.length - 1]].t >= minIbiMs) {
                valleys.push(i);
            } else if (v < samples[valleys[valleys.length - 1]].v) {
                valleys[valleys.length - 1] = i;
            }
        }
    }
    return valleys;
}

function _rejectAbnormalPeaks(peaks, signalLike) {
    if (peaks.length < 4) return peaks;
    const amps = peaks.map(p => signalLike[p]);
    const sorted = amps.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const deviations = amps.map(a => Math.abs(a - median));
    const devSorted = deviations.slice().sort((a, b) => a - b);
    const mad = devSorted[Math.floor(devSorted.length / 2)] || 1e-6;
    return peaks.filter((_, i) => Math.abs(amps[i] - median) <= 3 * mad);
}

function _refineExtrema(idxs, samples, expectMax) {
    const n = samples.length;
    return idxs.map(i => {
        if (i <= 0 || i >= n - 1) return samples[i].t;
        const x0 = samples[i - 1].t, y0 = samples[i - 1].v;
        const x1 = samples[i].t,     y1 = samples[i].v;
        const x2 = samples[i + 1].t, y2 = samples[i + 1].v;
        const denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
        if (Math.abs(denom) < 1e-12) return x1;
        const a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
        const b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
        if (Math.abs(a) < 1e-12) return x1;
        if (expectMax && a > 0) return x1;
        if (!expectMax && a < 0) return x1;
        const xv = -b / (2 * a);
        if (xv < x0 || xv > x2) return x1;
        return xv;
    });
}

function _quotientFilter(rr) {
    if (rr.length < 3) return rr;
    const out = [];
    for (let i = 0; i < rr.length; i++) {
        let ok = true;
        if (i > 0) { const r = rr[i] / rr[i - 1]; if (r < 0.75 || r > 1.33) ok = false; }
        if (i < rr.length - 1) { const r = rr[i] / rr[i + 1]; if (r < 0.75 || r > 1.33) ok = false; }
        if (rr[i] < 300 || rr[i] > 2000) ok = false;
        if (ok) out.push(rr[i]);
    }
    return out;
}

function _madFilter(rr) {
    if (rr.length < 4) return rr;
    const sorted = rr.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const deviations = rr.map(v => Math.abs(v - median));
    const devSorted = deviations.slice().sort((a, b) => a - b);
    const mad = devSorted[Math.floor(devSorted.length / 2)] || 1e-6;
    return rr.filter((v) => (0.6745 * Math.abs(v - median) / mad) < 5);
}

/**
 * Run the full HRV pipeline on the given timestamped samples.
 * Mirror of computeHrv() in hrv.js.
 */
function _runHrv(samples, dbg) {
    if (!samples || samples.length < 8) { dbg.reject = 'too_few_samples'; return null; }
    const n = samples.length;
    const ampView = { length: n }, negAmpView = { length: n };
    for (let i = 0; i < n; i++) { ampView[i] = samples[i].v; negAmpView[i] = -samples[i].v; }

    let peaks = _detectBvpPeaks(samples);
    dbg.peaksRaw = peaks.length;
    if (peaks.length < 4) { dbg.reject = 'too_few_peaks_raw'; return null; }
    peaks = _rejectAbnormalPeaks(peaks, ampView);
    dbg.peaksAfterReject = peaks.length;
    if (peaks.length < 4) { dbg.reject = 'too_few_peaks_after_reject'; return null; }
    const peakTimes = _refineExtrema(peaks, samples, true);
    const rrUpper = [];
    for (let i = 1; i < peakTimes.length; i++) rrUpper.push(peakTimes[i] - peakTimes[i - 1]);
    dbg.rrUpper = rrUpper.length;

    let valleys = _detectBvpValleys(samples);
    dbg.valleysRaw = valleys.length;
    if (valleys.length >= 4) valleys = _rejectAbnormalPeaks(valleys, negAmpView);
    dbg.valleysAfterReject = valleys.length;
    if (valleys.length >= 4) {
        const valleyTimes = _refineExtrema(valleys, samples, false);
        const rrLower = [];
        for (let i = 1; i < valleyTimes.length; i++) rrLower.push(valleyTimes[i] - valleyTimes[i - 1]);
        const cd = Math.abs(rrUpper.length - rrLower.length);
        dbg.countDiff = cd;
        if (cd > 2) { dbg.reject = 'count_diff_too_large'; return null; }
        const mU = rrUpper.reduce((a, b) => a + b, 0) / rrUpper.length;
        const mL = rrLower.reduce((a, b) => a + b, 0) / rrLower.length;
        const meanRatio = Math.abs(mU - mL) / mU;
        dbg.meanRatio = +meanRatio.toFixed(4);
        if (meanRatio > 0.05) { dbg.reject = 'mean_diff_too_large'; return null; }
    }

    const rrQ = _quotientFilter(rrUpper);
    const rrF = _madFilter(rrQ);
    dbg.rrQ = rrQ.length;
    dbg.rrF = rrF.length;
    const survival = rrF.length / rrUpper.length;
    dbg.survival = +survival.toFixed(3);
    if (survival < 0.6) { dbg.reject = 'low_survival'; return null; }
    if (rrF.length < 6) { dbg.reject = 'too_few_rrf'; return null; }

    const meanF = rrF.reduce((a, b) => a + b, 0) / rrF.length;
    let varF = 0;
    for (const r of rrF) varF += (r - meanF) ** 2;
    const stdF = Math.sqrt(varF / rrF.length);
    const cv = stdF / meanF;
    dbg.cv = +cv.toFixed(4);
    if (cv < 0.005) { dbg.reject = 'cv_too_low'; return null; }
    if (cv > 0.25)  { dbg.reject = 'cv_too_high'; return null; }

    let sumSq = 0;
    for (let i = 1; i < rrF.length; i++) { const d = rrF[i] - rrF[i - 1]; sumSq += d * d; }
    const rmssd = Math.sqrt(sumSq / (rrF.length - 1));
    const cvHealthy = (cv >= 0.01 && cv <= 0.15) ? 1 : (cv < 0.01 ? cv / 0.01 : 0.15 / cv);
    const confidence = Math.max(0, Math.min(1, survival * cvHealthy));
    return { rmssd, cv, confidence };
}

/* ── Message handler ── */
self.onmessage = async (e) => {
    const { type, payload } = e.data;
    try {
        if (type === 'init') await handleInit(payload);
        else if (type === 'run') await handleRun(payload);
        else if (type === 'hrv_run') {
            const start = performance.now();
            const dbg = {};
            const result = _runHrv(payload.samples, dbg);
            const elapsed = performance.now() - start;
            self.postMessage({
                type: 'hrv_result',
                payload: result === null
                    ? { rmssd: null, reject: dbg.reject, time: elapsed }
                    : { ...result, time: elapsed }
            });
        }
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
                                                                                                                               