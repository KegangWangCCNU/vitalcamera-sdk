/**
 * @file emotion.worker.js
 * @description Emotion classification Web Worker for the VitalCamera SDK.
 *
 * Runs an 8-class facial emotion recognition model inside a dedicated Web Worker
 * thread using LiteRT (TFLite WASM backend).
 *
 * Input:  a 224x224x3 face crop, ImageNet-normalized (NHWC layout).
 * Output: top emotion label, index, and probabilities. Calibration is
 * always applied (against the built-in Asian baseline by default, or the
 * baseline that the BrowserAdapter computed from `emotionCalibration.images`
 * at init time) — the consumer doesn't see whether or how it happened.
 *
 * Message protocol (all probe / setBaseline messages are SDK-internal):
 *   -> init       { modelBuffer, baselineLogits? }
 *   <- initDone
 *   -> run        { imgData }                 normal inference (always calibrated)
 *   <- result     { emotion, emotionIdx, probs, time }
 *   -> probe      { imgData, requestId }      raw logits only — only the BrowserAdapter
 *                                             uses this, during emotionCalibration.images
 *   <- probeResult { logits, requestId }
 *   -> setBaseline { baselineLogits | null }  install / clear the baseline
 *   <- baselineSet { hasBaseline }
 *   <- error      { msg }
 */

/* ── LiteRT runtime (loaded dynamically from CDN) ── */
let LiteRT = null;
let Tensor = null;
let model = null;

/* ── Constants ── */
const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/wasm/';

/** The 8 emotion classes in model output order. */
const EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'];
const NEUTRAL_IDX = 5;

/* ── Calibration constants — locked, baked into the SDK ── */
const TAU = 0.6;             // KL temperature
const W_BASELINE = 0.2;      // weight on the natural baseline distribution
const W_CALIBRATED = 0.8;    // weight on the calibrated distribution

/**
 * Default baseline logits — captured from an Asian male resting face
 * (n=11 samples, 2026-05). The 8-class HSEmotion model strongly over-fires
 * Anger / Sadness on this demographic; this default removes that bias for
 * users who haven't run their own calibration.
 *
 * Override by passing `emotionCalibration.baselineLogits` (or images) to
 * the BrowserAdapter constructor. Pass `emotionCalibration: { disabled: true }`
 * to disable calibration entirely and read raw model probabilities.
 */
const DEFAULT_ASIAN_BASELINE = [
    3.2577777125618677,    // Anger
    0.3067513170906089,    // Contempt
   -3.9749914299358022,    // Disgust
   -3.2980240258303555,    // Fear
   -3.2181908000599253,    // Happiness
    4.531865293329412 ,    // Neutral
    4.372382185675881 ,    // Sadness
   -0.5291195159608667,    // Surprise
];

/* ── Mutable state ── */
let baselineLogits = Float64Array.from(DEFAULT_ASIAN_BASELINE);   // Float64Array(8) | null

/**
 * Optional dynamic baseline mode.
 *
 * When enabled, after each successful run() we exponential-moving-average
 * the just-computed raw logits into the baseline. The EMA coefficient is
 * derived from a half-life so the user can think in seconds:
 *   alpha = 1 - 0.5^(dt / halfLifeMs)
 * where dt is the wall-clock interval between this inference and the
 * previous one. With halfLifeMs = 5_000:
 *   - holding a single expression for 5 s drifts the baseline 50 % of the
 *     way toward those logits → output starts pulling toward Neutral
 *     (the calibrated dist always centres on Neutral when current matches
 *     baseline).
 *   - blink / brief expressions don't move the baseline much.
 *
 * `dynamicLastT` is the timestamp (ms via performance.now()) of the last
 * EMA update; null means "this is the first frame, just record t".
 */
let dynamicHalfLifeMs = null;   // null → static baseline, no auto-update
let dynamicLastT = null;

// NOTE: softmax duplicated from utils/math.js for worker isolation.
function softmax(logits) {
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > maxVal) maxVal = logits[i];
    const exps = new Float32Array(logits.length);
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) { exps[i] = Math.exp(logits[i] - maxVal); sumExp += exps[i]; }
    for (let i = 0; i < logits.length; i++) exps[i] /= sumExp;
    return exps;
}

/**
 * KL-blend calibration.
 *   pCur  = softmax(logits)
 *   pBase = softmax(baseline)
 *   klContrib[i] = pCur[i] · log(pCur[i]/pBase[i])
 *   KL    = Σ klContrib                  (≥ 0; 0 iff pCur ≡ pBase)
 *   conf  = exp(−KL / τ)                  ← "still at rest" confidence
 *   pCal[Neutral] = conf
 *   pCal[i ≠ N]   = (1 − conf) · max(0, klContrib[i]) / Σ pos klContrib
 *   output        = 0.3 · pBase + 0.7 · pCal
 *
 * τ = 0.6, weights 0.3/0.7 are locked SDK-side; they aren't user knobs.
 */
function applyCalibration(logits, baseline) {
    const N = logits.length;
    const pCur = softmax(logits);
    const pBase = softmax(baseline);
    const eps = 1e-9;
    const klContrib = new Float64Array(N);
    let kl = 0;
    for (let i = 0; i < N; i++) {
        const p = pCur[i];
        if (p > 0) { klContrib[i] = p * Math.log((p + eps) / (pBase[i] + eps)); kl += klContrib[i]; }
    }
    if (kl < 0) kl = 0;
    const conf = Math.exp(-kl / TAU);

    let klPosSum = 0;
    for (let i = 0; i < N; i++) if (i !== NEUTRAL_IDX && klContrib[i] > 0) klPosSum += klContrib[i];
    const denom = klPosSum + eps;
    const remaining = 1 - conf;

    const pCal = new Array(N).fill(0);
    pCal[NEUTRAL_IDX] = conf;
    for (let i = 0; i < N; i++) {
        if (i === NEUTRAL_IDX) continue;
        const c = klContrib[i] > 0 ? klContrib[i] : 0;
        pCal[i] = remaining * c / denom;
    }

    const out = new Array(N);
    for (let i = 0; i < N; i++) out[i] = W_BASELINE * pBase[i] + W_CALIBRATED * pCal[i];
    return out;
}

/* ── Exported for testing (Node.js only) ── */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { softmax, applyCalibration, EMOTIONS, NEUTRAL_IDX, TAU, DEFAULT_ASIAN_BASELINE };
}

/* ── Message handler ── */
self.onmessage = async (e) => {
    const { type, payload } = e.data;
    try {
        if (type === 'init')          await handleInit(payload);
        else if (type === 'run')      await handleRun(payload);
        else if (type === 'probe')    await handleProbe(payload);
        else if (type === 'setBaseline') {
            // payload.baselineLogits === null     → disable calibration (raw probs)
            // array of 8 floats                   → install as the baseline
            // otherwise (missing / wrong shape)   → restore the default Asian baseline
            const incoming = payload?.baselineLogits;
            if (incoming === null) {
                baselineLogits = null;
            } else if (Array.isArray(incoming) && incoming.length === EMOTIONS.length) {
                baselineLogits = Float64Array.from(incoming);
            } else {
                baselineLogits = Float64Array.from(DEFAULT_ASIAN_BASELINE);
            }
            dynamicLastT = null;   // reset EMA timer on any baseline replacement
            self.postMessage({ type: 'baselineSet', payload: { hasBaseline: !!baselineLogits } });
        }
        else if (type === 'setDynamic') {
            // payload.halfLifeMs:  number > 0  → enable EMA mode
            //                      null/0/missing → disable
            const hl = payload?.halfLifeMs;
            dynamicHalfLifeMs = (typeof hl === 'number' && hl > 0) ? hl : null;
            dynamicLastT = null;   // restart timer
        }
    } catch (err) {
        self.postMessage({ type: 'error', msg: err.toString() });
    }
};

async function handleInit({ modelBuffer, baselineLogits: initialBaseline, dynamicHalfLifeMs: initialHl }) {
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

    // Calibration baseline:
    //   - undefined / not provided  → keep the built-in DEFAULT_ASIAN_BASELINE
    //   - array of 8 floats         → install as user-provided baseline
    //   - null                      → DISABLE calibration; emit raw softmax probs
    if (initialBaseline === null) {
        baselineLogits = null;
    } else if (Array.isArray(initialBaseline) && initialBaseline.length === EMOTIONS.length) {
        baselineLogits = Float64Array.from(initialBaseline);
    }
    // (otherwise: leave the default in place)

    if (typeof initialHl === 'number' && initialHl > 0) {
        dynamicHalfLifeMs = initialHl;
    }

    self.postMessage({ type: 'initDone' });
}

async function handleRun({ imgData }) {
    if (!model) return;
    const start = performance.now();

    const inputTensor = new Tensor(imgData, [1, 224, 224, 3]);
    const results = model.run([inputTensor]);
    inputTensor.delete();

    const logitsArr = Array.from(results[0].toTypedArray());
    results[0].delete();

    const probs = baselineLogits
        ? applyCalibration(logitsArr, baselineLogits)
        : Array.from(softmax(logitsArr));

    let topIdx = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[topIdx]) topIdx = i;

    // ── Dynamic baseline EMA ──
    // After each successful inference, optionally fold the just-observed
    // raw logits into the calibration baseline using a half-life-derived
    // EMA. The semantics are: a sustained expression slides the baseline
    // toward those logits, which calibration then re-centres on Neutral —
    // so any expression a user holds will fade to neutral over ~halfLife,
    // and only DEVIATIONS from their resting expression light up.
    if (dynamicHalfLifeMs && baselineLogits) {
        const now = performance.now();
        if (dynamicLastT !== null) {
            const dt = now - dynamicLastT;
            // alpha = 1 - 0.5^(dt / halfLife)  — fraction of "new" sample
            const alpha = 1 - Math.pow(0.5, dt / dynamicHalfLifeMs);
            for (let i = 0; i < baselineLogits.length; i++) {
                baselineLogits[i] = (1 - alpha) * baselineLogits[i] + alpha * logitsArr[i];
            }
        }
        dynamicLastT = now;
    }

    self.postMessage({
        type: 'result',
        payload: {
            emotion: EMOTIONS[topIdx],
            emotionIdx: topIdx,
            probs,
            time: performance.now() - start,
        }
    });
}

async function handleProbe({ imgData, requestId }) {
    if (!model) {
        self.postMessage({ type: 'probeResult', payload: { logits: null, requestId, error: 'model not ready' } });
        return;
    }
    const inputTensor = new Tensor(imgData, [1, 224, 224, 3]);
    const results = model.run([inputTensor]);
    inputTensor.delete();
    const logits = Array.from(results[0].toTypedArray());
    results[0].delete();
    self.postMessage({ type: 'probeResult', payload: { logits, requestId } });
}
