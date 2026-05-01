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
 * Drop RR intervals that sit too far from the in-window median. Catches
 * gross outliers that the per-pair compensating filter doesn't see —
 * e.g. a long stretch where peak detection consistently misses a beat
 * (giving doubled RRs that drift in alone, with no neighbouring spike).
 *
 * Default tolerance is ±25 % of the median, matching the heartpy
 * quotient filter's traditional 0.75–1.33 acceptance range.
 *
 * @param {number[]} rr   RR intervals, ms
 * @param {number} [k=0.25]  Tolerance — keep RRs within (1±k)·median
 * @returns {number[]}    Filtered RR intervals
 */
export function filterRROutliers(rr, k = 0.25) {
    if (rr.length < 4) return rr.slice();
    const sorted = rr.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    return rr.filter(v => Math.abs(v - median) / median <= k);
}

/**
 * Filter compensating-pair RR artefacts. When peak detection mis-locates
 * a single peak by 1-2 frames, RR[i-1] and RR[i] swap mass — one too long,
 * one too short — so |RR[i] - RR[i-1]| spikes far above the typical
 * inter-RR variation. We use the median |ΔRR| in this window as the
 * scale and drop both members of any pair whose |ΔRR| exceeds k × that
 * (with a floor so very-still subjects don't have a near-zero scale and
 * get over-pruned).
 *
 * @param {number[]} rr   RR intervals, ms
 * @returns {number[]}    Filtered RR intervals
 */
export function filterCompensatingPairs(rr) {
    if (rr.length < 4) return rr.slice();
    const diffs = new Array(rr.length - 1);
    for (let i = 1; i < rr.length; i++) diffs[i - 1] = Math.abs(rr[i] - rr[i - 1]);

    // Robust scale: 25th-percentile of |ΔRR|; floor scales with mean RR (12 %)
    const sorted = diffs.slice().sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length / 4)];
    let sumRR = 0; for (const v of rr) sumRR += v;
    const meanRR = sumRR / rr.length;
    const threshold = Math.max(4 * q1, 0.12 * meanRR);

    // Median RR — used to decide which member of a flagged pair to drop.
    // Without this, a clean→bad transition (e.g. 800→650) drops both 800
    // and 650 even though only the 650 is wrong, cascading the rejection
    // outward from a single artefact.
    const sortedRR = rr.slice().sort((a, b) => a - b);
    const medianRR = sortedRR[Math.floor(sortedRR.length / 2)];
    const NEAR_MED = 0.05;     // within 5 % of median = "fine"
    const offMed = v => Math.abs(v - medianRR) / medianRR;

    const keep = new Array(rr.length).fill(true);
    for (let i = 1; i < rr.length; i++) {
        if (diffs[i - 1] > threshold) {
            const offPrev = offMed(rr[i - 1]);
            const offCur  = offMed(rr[i]);
            if (offPrev > NEAR_MED && offCur > NEAR_MED) {
                // Both off-median → genuine compensating pair → drop both
                keep[i - 1] = false;
                keep[i] = false;
            } else if (offPrev > offCur) {
                // Only the previous one is off → that's the artefact
                keep[i - 1] = false;
            } else {
                keep[i] = false;
            }
        }
    }
    const out = [];
    for (let i = 0; i < rr.length; i++) if (keep[i]) out.push(rr[i]);
    return out;
}

/**
 * Minimal HRV pipeline. SQI gating happens upstream (samples are admitted
 * to the buffer only when SQI ≥ config.hrvSqiThreshold). Here we just:
 *   detect peaks → refine timing → physiological bounds → drop
 *   compensating-pair artefacts → RMSSD.
 *
 * @param {{t:number, v:number}[]} samples
 * @returns {{ rmssd:number, sdnn:number, meanRR:number, n:number } | null}
 */
export function computeHrv(samples) {
    if (!samples || samples.length < 30) return null;

    const peaks = detectBvpPeaks(samples);
    if (peaks.length < 6) return null;

    const peakTimes = refinePeaksParabolic(peaks, samples);
    const rrRaw = [];
    for (let i = 1; i < peakTimes.length; i++) rrRaw.push(peakTimes[i] - peakTimes[i - 1]);

    // Physiological bounds — 300 ms (200 BPM) to 2000 ms (30 BPM)
    const rrPhys = rrRaw.filter(v => v >= 300 && v <= 2000);
    if (rrPhys.length < 5) return null;

    // Two-stage outlier rejection:
    //   1. drop RRs >25 % from the in-window median (gross drift / missed beats)
    //   2. drop compensating pairs where |ΔRR| > scale-aware threshold
    const rrInRange = filterRROutliers(rrPhys);
    if (rrInRange.length < 5) return null;
    const rrClean = filterCompensatingPairs(rrInRange);
    if (rrClean.length < 5) return null;

    // RMSSD — root-mean-square of successive RR differences (short-term HRV)
    let sumSqDiff = 0;
    for (let i = 1; i < rrClean.length; i++) {
        const d = rrClean[i] - rrClean[i - 1];
        sumSqDiff += d * d;
    }
    const rmssd = Math.sqrt(sumSqDiff / (rrClean.length - 1));

    // SDNN — standard deviation of NN (RR) intervals (overall HRV)
    let sum = 0;
    for (const r of rrClean) sum += r;
    const meanRR = sum / rrClean.length;
    let sumSqDev = 0;
    for (const r of rrClean) sumSqDev += (r - meanRR) * (r - meanRR);
    const sdnn = Math.sqrt(sumSqDev / rrClean.length);

    return { rmssd, sdnn, meanRR, n: rrClean.length };
}
