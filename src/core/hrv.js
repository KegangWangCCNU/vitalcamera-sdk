/**
 * HRV (Heart Rate Variability) computation engine.
 * Operates directly on irregularly-sampled BVP timestamped samples — no
 * pre-resampling, no cubic spline. Peak times are recovered to sub-sample
 * precision via 3-point parabolic refinement around each detected peak.
 *
 * @module core/hrv
 */

/**
 * Detect BVP peaks directly on irregularly-sampled timestamped data.
 *
 * A sample at index i is a peak if
 *   1. it is a 5-point local maximum  (>= its 4 nearest neighbours)
 *   2. it sits above mean + 0.3 * std  of the whole window (rough thresholding)
 *   3. it is at least `minIbiMs` after the previous accepted peak
 *      (otherwise the higher of the two replaces the lower).
 *
 * @param {{t:number, v:number}[]} samples  Timestamp (ms) + amplitude pairs
 * @param {number} [minIbiMs=333]            Minimum inter-beat interval (180 BPM)
 * @returns {number[]}                       Peak indices into `samples`
 */
export function detectBvpPeaks(samples, minIbiMs = 333) {
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
            if (peaks.length === 0
                || samples[i].t - samples[peaks[peaks.length - 1]].t >= minIbiMs) {
                peaks.push(i);
            } else if (v > samples[peaks[peaks.length - 1]].v) {
                peaks[peaks.length - 1] = i;
            }
        }
    }
    return peaks;
}

/**
 * Detect BVP valleys (lower-half local minima) on irregularly-sampled data —
 * mirror of {@link detectBvpPeaks}. Used as an independent estimate of the
 * inter-beat interval for cross-validation.
 *
 * @param {{t:number, v:number}[]} samples
 * @param {number} [minIbiMs=333]
 * @returns {number[]} Valley indices into `samples`
 */
export function detectBvpValleys(samples, minIbiMs = 333) {
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
            if (valleys.length === 0
                || samples[i].t - samples[valleys[valleys.length - 1]].t >= minIbiMs) {
                valleys.push(i);
            } else if (v < samples[valleys[valleys.length - 1]].v) {
                valleys[valleys.length - 1] = i;
            }
        }
    }
    return valleys;
}

/**
 * MAD-based peak amplitude rejection (3x MAD threshold).
 * Generic helper — `signalLike[p]` must yield the amplitude for peak index p.
 *
 * @param {number[]} peaks
 * @param {ArrayLike<number>} signalLike  e.g. amplitudes array, or a
 *                                        plain array of `{v}` values
 * @returns {number[]}
 */
export function rejectAbnormalPeaks(peaks, signalLike) {
    if (peaks.length < 4) return peaks;
    const amps = peaks.map(p => signalLike[p]);
    const sorted = amps.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const deviations = amps.map(a => Math.abs(a - median));
    const devSorted = deviations.slice().sort((a, b) => a - b);
    const mad = devSorted[Math.floor(devSorted.length / 2)] || 1e-6;
    const threshold = 3 * mad;
    return peaks.filter((p, i) => Math.abs(amps[i] - median) <= threshold);
}

/**
 * Refine each peak's *time* using a 3-point parabolic fit through the
 * peak sample and its two neighbours. Works on irregularly-spaced
 * timestamps — the parabola is fit through (t_{i-1}, v_{i-1}),
 * (t_i, v_i), (t_{i+1}, v_{i+1}) and the vertex's t-coordinate is
 * returned (in ms — the same units as the input timestamps).
 *
 * For 30 Hz BVP this typically pushes peak-time uncertainty from
 * ±16 ms (half the inter-sample interval) down to roughly ±1 ms,
 * without needing a global resampling step.
 *
 * @param {number[]} peaks                  Indices from detectBvpPeaks
 * @param {{t:number, v:number}[]} samples
 * @returns {number[]}                      Refined peak timestamps (ms)
 */
function _refineExtremaParabolic(idxs, samples, expectMax) {
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
        if (expectMax && a > 0) return x1;     // not a downward-opening parabola
        if (!expectMax && a < 0) return x1;    // not an upward-opening parabola
        const xv = -b / (2 * a);
        if (xv < x0 || xv > x2) return x1;
        return xv;
    });
}

export function refinePeaksParabolic(peaks, samples) {
    return _refineExtremaParabolic(peaks, samples, /*expectMax=*/true);
}

/**
 * Same fit as {@link refinePeaksParabolic} but for local minima — the
 * parabola must open upward (a > 0). Used to cross-validate beat timing
 * against the upper peaks.
 */
export function refineValleysParabolic(valleys, samples) {
    return _refineExtremaParabolic(valleys, samples, /*expectMax=*/false);
}

/**
 * Adjacent RR ratio filter (0.75-1.33) + physiological bounds (300-2000ms).
 * @param {number[]} rr - RR intervals in ms
 * @returns {number[]}
 */
export function quotientFilterRR(rr) {
    if (rr.length < 3) return rr;
    const filtered = [];
    for (let i = 0; i < rr.length; i++) {
        let ok = true;
        if (i > 0) { const r = rr[i] / rr[i - 1]; if (r < 0.75 || r > 1.33) ok = false; }
        if (i < rr.length - 1) { const r = rr[i] / rr[i + 1]; if (r < 0.75 || r > 1.33) ok = false; }
        if (rr[i] < 300 || rr[i] > 2000) ok = false;
        if (ok) filtered.push(rr[i]);
    }
    return filtered;
}

/**
 * Modified z-score MAD filter with 3.5 threshold.
 * @param {number[]} rr - RR intervals in ms
 * @returns {number[]}
 */
export function madFilterRR(rr) {
    if (rr.length < 4) return rr;
    const sorted = rr.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const deviations = rr.map(v => Math.abs(v - median));
    const devSorted = deviations.slice().sort((a, b) => a - b);
    const mad = devSorted[Math.floor(devSorted.length / 2)] || 1e-6;
    // Use 5 instead of the textbook 3.5 — at 30 Hz BVP the parabolic
    // refinement is precise enough that a 3.5 cutoff over-prunes
    // genuine short-term variability and collapses RMSSD to <1 ms.
    return rr.filter((v) => (0.6745 * Math.abs(v - median) / mad) < 5);
}

/**
 * Compute time-domain HRV metrics (RMSSD) from filtered RR intervals.
 * @param {number[]} rrFiltered - filtered RR intervals in ms
 * @returns {{ rmssd: number } | null}
 */
export function computeHrvMetrics(rrFiltered) {
    if (rrFiltered.length < 6) return null;          // need ≥ 5 diffs for a stable RMSSD
    const diffs = [];
    for (let i = 1; i < rrFiltered.length; i++) diffs.push(rrFiltered[i] - rrFiltered[i - 1]);
    if (diffs.length < 5) return null;
    let sumSq = 0;
    for (const d of diffs) sumSq += d * d;
    const rmssd = Math.sqrt(sumSq / diffs.length);
    return { rmssd };
}

/**
 * One-shot HRV pipeline: detect peaks (and valleys for cross-validation),
 * refine timing with parabolic fits on irregular samples, run the three
 * confidence gates, and emit { rmssd, cv, confidence } or null when the
 * window is too noisy / smooth / short.
 *
 * Pure function — same code lives inline inside psd.worker.js so the heavy
 * compute doesn't run on the main thread. Exported here for direct callers
 * (tests, custom pipelines).
 *
 * @param {{t:number, v:number}[]} samples
 * @returns {{ rmssd:number, cv:number, confidence:number } | null}
 */
export function computeHrv(samples) {
    if (!samples || samples.length < 8) return null;

    const n = samples.length;
    const ampView = { length: n }, negAmpView = { length: n };
    for (let i = 0; i < n; i++) { ampView[i] = samples[i].v; negAmpView[i] = -samples[i].v; }

    let peaks = detectBvpPeaks(samples);
    if (peaks.length < 4) return null;
    peaks = rejectAbnormalPeaks(peaks, ampView);
    if (peaks.length < 4) return null;
    const peakTimes = refinePeaksParabolic(peaks, samples);
    const rrUpper = [];
    for (let i = 1; i < peakTimes.length; i++) rrUpper.push(peakTimes[i] - peakTimes[i - 1]);

    // Gate 1: valley sanity (loose: count + mean only)
    let valleys = detectBvpValleys(samples);
    if (valleys.length >= 4) valleys = rejectAbnormalPeaks(valleys, negAmpView);
    if (valleys.length >= 4) {
        const valleyTimes = refineValleysParabolic(valleys, samples);
        const rrLower = [];
        for (let i = 1; i < valleyTimes.length; i++) rrLower.push(valleyTimes[i] - valleyTimes[i - 1]);
        if (Math.abs(rrUpper.length - rrLower.length) > 2) return null;
        const meanU = rrUpper.reduce((a, b) => a + b, 0) / rrUpper.length;
        const meanL = rrLower.reduce((a, b) => a + b, 0) / rrLower.length;
        if (Math.abs(meanU - meanL) / meanU > 0.05) return null;
    }

    // Gate 2: filter survival rate
    const rrQ = quotientFilterRR(rrUpper);
    const rrF = madFilterRR(rrQ);
    if (rrF.length / rrUpper.length < 0.6) return null;

    // Gate 3: CV bounds
    const meanF = rrF.reduce((a, b) => a + b, 0) / rrF.length;
    let varF = 0;
    for (const r of rrF) varF += (r - meanF) ** 2;
    const stdF = Math.sqrt(varF / rrF.length);
    const cv = stdF / meanF;
    if (cv < 0.005 || cv > 0.25) return null;

    const m = computeHrvMetrics(rrF);
    if (!m) return null;
    const survival = rrF.length / rrUpper.length;
    const cvHealthy = (cv >= 0.01 && cv <= 0.15) ? 1.0 : (cv < 0.01 ? cv / 0.01 : 0.15 / cv);
    const confidence = Math.max(0, Math.min(1, survival * cvHealthy));
    return { rmssd: m.rmssd, cv, confidence };
}

