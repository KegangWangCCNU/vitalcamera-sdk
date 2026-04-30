/**
 * HRV (Heart Rate Variability) computation engine.
 * Heartpy-style signal processing: spline interpolation, peak detection,
 * RR interval filtering, and time-domain HRV metrics.
 *
 * @module core/hrv
 */

/**
 * Natural cubic spline interpolation.
 * @param {number[]} xs - x coordinates (knots), must be sorted ascending
 * @param {number[]} ys - y coordinates
 * @returns {{ a: number[], b: number[], c: number[], d: number[], x: number[] } | null}
 */
export function buildCubicSpline(xs, ys) {
    const n = xs.length - 1;
    if (n < 1) return null;
    const h = new Array(n);
    for (let i = 0; i < n; i++) h[i] = xs[i + 1] - xs[i];
    const alpha = new Array(n + 1).fill(0);
    for (let i = 1; i < n; i++) {
        alpha[i] = 3 / h[i] * (ys[i + 1] - ys[i]) - 3 / h[i - 1] * (ys[i] - ys[i - 1]);
    }
    const l = new Array(n + 1), mu = new Array(n + 1), z = new Array(n + 1);
    l[0] = 1; mu[0] = 0; z[0] = 0;
    for (let i = 1; i < n; i++) {
        l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[n] = 1; z[n] = 0;
    const c = new Array(n + 1), b = new Array(n), d = new Array(n);
    c[n] = 0;
    for (let j = n - 1; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }
    return { a: ys.slice(0, n), b, c: c.slice(0, n), d, x: xs };
}

/**
 * Evaluate a cubic spline at an array of x values.
 * @param {{ a, b, c, d, x }} sp - spline coefficients from buildCubicSpline
 * @param {number[]} xArr - x values to evaluate at
 * @returns {Float64Array}
 */
export function evalSpline(sp, xArr) {
    const { a, b, c, d, x } = sp;
    const n = a.length;
    const out = new Float64Array(xArr.length);
    let j = 0;
    for (let i = 0; i < xArr.length; i++) {
        while (j < n - 1 && xArr[i] > x[j + 1]) j++;
        if (j >= n) j = n - 1;
        const dx = xArr[i] - x[j];
        out[i] = a[j] + b[j] * dx + c[j] * dx * dx + d[j] * dx * dx * dx;
    }
    return out;
}

/**
 * Interpolate BVP (blood volume pulse) samples onto a uniform grid.
 * @param {{ t: number, v: number }[]} samples - timestamped samples
 * @param {number} targetFs - target sampling frequency in Hz
 * @returns {{ values: Float64Array, fs: number, tStart: number, tEnd: number } | null}
 */
export function interpolateBvp(samples, targetFs) {
    if (samples.length < 4) return null;
    const xs = samples.map(s => s.t);
    const ys = samples.map(s => s.v);
    const tStart = xs[0], tEnd = xs[xs.length - 1];
    const duration = (tEnd - tStart) / 1000;
    if (duration < 2) return null;
    const numSamples = Math.floor(duration * targetFs);
    const step = (tEnd - tStart) / (numSamples - 1);
    const xNew = new Array(numSamples);
    for (let i = 0; i < numSamples; i++) xNew[i] = tStart + i * step;
    const sp = buildCubicSpline(xs, ys);
    if (!sp) return null;
    return { values: evalSpline(sp, xNew), fs: targetFs, tStart, tEnd };
}

/**
 * Moving-average peak detection for BVP signals.
 * @param {Float64Array|number[]} signal
 * @param {number} fs - sampling frequency in Hz
 * @returns {number[]} array of peak indices
 */
export function detectBvpPeaks(signal, fs) {
    const n = signal.length;
    if (n < fs * 2) return [];
    const winHalf = Math.floor(0.75 * fs / 2);
    const movAvg = new Float64Array(n);
    let runSum = 0, runCount = 0;
    for (let i = 0; i < Math.min(winHalf + 1, n); i++) { runSum += signal[i]; runCount++; }
    movAvg[0] = runSum / runCount;
    for (let i = 1; i < n; i++) {
        const addIdx = i + winHalf;
        const remIdx = i - winHalf - 1;
        if (addIdx < n) { runSum += signal[addIdx]; runCount++; }
        if (remIdx >= 0) { runSum -= signal[remIdx]; runCount--; }
        movAvg[i] = runSum / runCount;
    }
    const minDist = Math.round(0.33 * fs);
    const peaks = [];
    for (let i = 2; i < n - 2; i++) {
        if (signal[i] > movAvg[i] &&
            signal[i] >= signal[i - 1] && signal[i] >= signal[i + 1] &&
            signal[i] >= signal[i - 2] && signal[i] >= signal[i + 2]) {
            if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDist) {
                peaks.push(i);
            } else if (signal[i] > signal[peaks[peaks.length - 1]]) {
                peaks[peaks.length - 1] = i;
            }
        }
    }
    return peaks;
}

/**
 * MAD-based peak amplitude rejection (3x MAD threshold).
 * @param {number[]} peaks - peak indices
 * @param {Float64Array|number[]} signal
 * @returns {number[]} filtered peak indices
 */
export function rejectAbnormalPeaks(peaks, signal) {
    if (peaks.length < 4) return peaks;
    const amps = peaks.map(p => signal[p]);
    const sorted = amps.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const deviations = amps.map(a => Math.abs(a - median));
    const devSorted = deviations.slice().sort((a, b) => a - b);
    const mad = devSorted[Math.floor(devSorted.length / 2)] || 1e-6;
    const threshold = 3 * mad;
    return peaks.filter((p, i) => Math.abs(amps[i] - median) <= threshold);
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
    return rr.filter((v, i) => (0.6745 * Math.abs(v - median) / mad) < 3.5);
}

/**
 * Compute time-domain HRV metrics (RMSSD) from filtered RR intervals.
 * @param {number[]} rrFiltered - filtered RR intervals in ms
 * @returns {{ rmssd: number } | null}
 */
export function computeHrvMetrics(rrFiltered) {
    if (rrFiltered.length < 3) return null;
    const diffs = [];
    for (let i = 1; i < rrFiltered.length; i++) diffs.push(rrFiltered[i] - rrFiltered[i - 1]);
    if (diffs.length < 2) return null;
    let sumSq = 0;
    for (const d of diffs) sumSq += d * d;
    const rmssd = Math.sqrt(sumSq / diffs.length);
    return { rmssd };
}
