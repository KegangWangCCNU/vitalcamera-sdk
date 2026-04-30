import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
    buildCubicSpline,
    evalSpline,
    interpolateBvp,
    detectBvpPeaks,
    rejectAbnormalPeaks,
    quotientFilterRR,
    madFilterRR,
    computeHrvMetrics,
} from '../src/core/hrv.js';

import RealtimePeakDetector from '../src/core/peak-detect.js';

// ---------------------------------------------------------------------------
// buildCubicSpline
// ---------------------------------------------------------------------------
describe('buildCubicSpline', () => {
    it('returns null for fewer than 2 points', () => {
        assert.equal(buildCubicSpline([1], [2]), null);
        assert.equal(buildCubicSpline([], []), null);
    });

    it('returns a valid spline object for linear data', () => {
        const sp = buildCubicSpline([0, 1, 2, 3], [0, 1, 2, 3]);
        assert.ok(sp);
        assert.ok(Array.isArray(sp.a));
        assert.ok(Array.isArray(sp.b));
        assert.ok(Array.isArray(sp.c));
        assert.ok(Array.isArray(sp.d));
        assert.ok(Array.isArray(sp.x));
        assert.equal(sp.a.length, 3);
    });

    it('spline passes through original points', () => {
        const xs = [0, 1, 2, 3, 4];
        const ys = [0, 1, 4, 9, 16]; // x^2
        const sp = buildCubicSpline(xs, ys);
        const evaluated = evalSpline(sp, xs);
        for (let i = 0; i < xs.length; i++) {
            assert.ok(Math.abs(evaluated[i] - ys[i]) < 1e-10,
                `Point ${i}: expected ${ys[i]}, got ${evaluated[i]}`);
        }
    });
});

// ---------------------------------------------------------------------------
// evalSpline
// ---------------------------------------------------------------------------
describe('evalSpline', () => {
    it('evaluates correctly at knot points', () => {
        const xs = [0, 1, 2, 3];
        const ys = [0, 2, 1, 3];
        const sp = buildCubicSpline(xs, ys);
        const vals = evalSpline(sp, xs);
        for (let i = 0; i < xs.length; i++) {
            assert.ok(Math.abs(vals[i] - ys[i]) < 1e-10);
        }
    });

    it('interpolates between knots (monotone linear case)', () => {
        const sp = buildCubicSpline([0, 10], [0, 10]);
        const vals = evalSpline(sp, [5]);
        assert.ok(Math.abs(vals[0] - 5) < 1e-10);
    });
});

// ---------------------------------------------------------------------------
// interpolateBvp
// ---------------------------------------------------------------------------
describe('interpolateBvp', () => {
    it('returns null for fewer than 4 samples', () => {
        const samples = [{ t: 0, v: 1 }, { t: 100, v: 2 }, { t: 200, v: 3 }];
        assert.equal(interpolateBvp(samples, 30), null);
    });

    it('returns null for duration < 2 seconds', () => {
        const samples = [
            { t: 0, v: 1 }, { t: 100, v: 2 },
            { t: 200, v: 3 }, { t: 300, v: 4 },
        ];
        assert.equal(interpolateBvp(samples, 30), null);
    });

    it('returns valid interpolation for good data', () => {
        // 3 seconds of data at ~10 Hz-ish irregular
        const samples = [];
        for (let i = 0; i <= 30; i++) {
            samples.push({ t: i * 100, v: Math.sin(i * 0.2) });
        }
        const result = interpolateBvp(samples, 30);
        assert.ok(result);
        assert.equal(result.fs, 30);
        assert.equal(result.tStart, 0);
        assert.equal(result.tEnd, 3000);
        assert.ok(result.values instanceof Float64Array);
        assert.ok(result.values.length > 0);
    });
});

// ---------------------------------------------------------------------------
// detectBvpPeaks
// ---------------------------------------------------------------------------
describe('detectBvpPeaks', () => {
    it('returns empty for too-short signal', () => {
        const fs = 30;
        const signal = new Float64Array(30); // less than fs*2
        assert.deepEqual(detectBvpPeaks(signal, fs), []);
    });

    it('finds peaks in a sinusoidal signal', () => {
        const fs = 30;
        const duration = 5; // 5 seconds
        const n = fs * duration;
        const signal = new Float64Array(n);
        const freqHz = 1.2; // ~72 BPM
        for (let i = 0; i < n; i++) {
            signal[i] = Math.sin(2 * Math.PI * freqHz * i / fs);
        }
        const peaks = detectBvpPeaks(signal, fs);
        assert.ok(peaks.length >= 4, `Expected >= 4 peaks, got ${peaks.length}`);
        // Peaks should be roughly fs/freqHz apart
        const expectedSpacing = fs / freqHz;
        for (let i = 1; i < peaks.length; i++) {
            const spacing = peaks[i] - peaks[i - 1];
            assert.ok(Math.abs(spacing - expectedSpacing) < expectedSpacing * 0.2,
                `Spacing ${spacing} too far from expected ${expectedSpacing}`);
        }
    });
});

// ---------------------------------------------------------------------------
// rejectAbnormalPeaks
// ---------------------------------------------------------------------------
describe('rejectAbnormalPeaks', () => {
    it('keeps normal peaks unchanged', () => {
        const signal = [0, 10, 0, 11, 0, 10.5, 0, 9.8, 0];
        const peaks = [1, 3, 5, 7];
        const result = rejectAbnormalPeaks(peaks, signal);
        assert.deepEqual(result, peaks);
    });

    it('removes outlier peaks', () => {
        // One peak has a wildly different amplitude
        const signal = [0, 10, 0, 10, 0, 100, 0, 10, 0, 10, 0];
        const peaks = [1, 3, 5, 7, 9];
        const result = rejectAbnormalPeaks(peaks, signal);
        assert.ok(!result.includes(5), 'Outlier peak at index 5 should be removed');
        assert.ok(result.includes(1));
        assert.ok(result.includes(3));
    });
});

// ---------------------------------------------------------------------------
// quotientFilterRR
// ---------------------------------------------------------------------------
describe('quotientFilterRR', () => {
    it('keeps consistent intervals', () => {
        const rr = [800, 810, 790, 805, 815];
        const result = quotientFilterRR(rr);
        assert.deepEqual(result, rr);
    });

    it('rejects out-of-range ratio intervals', () => {
        // 400 is way off from 800 (ratio 0.5)
        const rr = [800, 810, 400, 790, 805];
        const result = quotientFilterRR(rr);
        assert.ok(!result.includes(400), '400ms should be rejected (ratio too far)');
    });

    it('rejects physiologically impossible values', () => {
        const rr = [800, 810, 200, 805, 815]; // 200 < 300
        const result = quotientFilterRR(rr);
        assert.ok(!result.includes(200));
    });

    it('returns input unchanged for < 3 intervals', () => {
        const rr = [800, 810];
        assert.deepEqual(quotientFilterRR(rr), rr);
    });
});

// ---------------------------------------------------------------------------
// madFilterRR
// ---------------------------------------------------------------------------
describe('madFilterRR', () => {
    it('returns input for fewer than 4 values', () => {
        const rr = [800, 810, 790];
        assert.deepEqual(madFilterRR(rr), rr);
    });

    it('filters outliers from a mostly-normal distribution', () => {
        const rr = [800, 810, 790, 805, 795, 2500, 808, 802];
        const result = madFilterRR(rr);
        assert.ok(!result.includes(2500), '2500 should be filtered as outlier');
        assert.ok(result.includes(800));
        assert.ok(result.includes(810));
    });
});

// ---------------------------------------------------------------------------
// computeHrvMetrics
// ---------------------------------------------------------------------------
describe('computeHrvMetrics', () => {
    it('returns null for fewer than 3 intervals', () => {
        assert.equal(computeHrvMetrics([800, 810]), null);
    });

    it('computes correct RMSSD for known input', () => {
        // RR = [800, 810, 790, 805]
        // diffs = [10, -20, 15]
        // sumSq = 100 + 400 + 225 = 725
        // rmssd = sqrt(725 / 3) = sqrt(241.666...) ≈ 15.5456
        const result = computeHrvMetrics([800, 810, 790, 805]);
        assert.ok(result);
        const expected = Math.sqrt(725 / 3);
        assert.ok(Math.abs(result.rmssd - expected) < 1e-6,
            `Expected RMSSD ~${expected}, got ${result.rmssd}`);
    });

    it('returns RMSSD=0 for constant intervals', () => {
        const result = computeHrvMetrics([800, 800, 800, 800]);
        assert.ok(result);
        assert.ok(Math.abs(result.rmssd) < 1e-10);
    });
});

// ---------------------------------------------------------------------------
// RealtimePeakDetector
// ---------------------------------------------------------------------------
describe('RealtimePeakDetector', () => {
    it('returns null before enough samples accumulated', () => {
        const det = new RealtimePeakDetector({ windowSize: 90 });
        for (let i = 0; i < 20; i++) {
            const result = det.process(i * 33, Math.sin(i * 0.2));
            assert.equal(result, null);
        }
    });

    it('detects peaks in a synthetic sinusoid', () => {
        const det = new RealtimePeakDetector({ windowSize: 60, minIbiMs: 300, maxIbiMs: 2000 });
        const fs = 30; // 30 Hz
        const freqHz = 1.0; // 1 beat/sec => IBI=1000ms
        const duration = 10; // seconds
        const n = fs * duration;
        const beats = [];

        for (let i = 0; i < n; i++) {
            const t = i * (1000 / fs);
            const v = Math.sin(2 * Math.PI * freqHz * i / fs) + 1; // shift up so baseline ~1
            const result = det.process(t, v);
            if (result) beats.push(result);
        }

        // With 1 Hz sine over 10 seconds, expect several detected beats
        assert.ok(beats.length >= 3,
            `Expected >= 3 beats, got ${beats.length}`);

        // IBIs should be roughly 1000ms
        for (const b of beats) {
            assert.ok(b.ibi > 500 && b.ibi < 1500,
                `IBI ${b.ibi} out of expected range`);
        }
    });

    it('reset clears all state', () => {
        const det = new RealtimePeakDetector();
        for (let i = 0; i < 100; i++) det.process(i * 33, Math.sin(i * 0.2));
        det.reset();
        assert.equal(det.ring.length, 0);
        assert.equal(det.movSum, 0);
        assert.equal(det.state, 0);
        assert.equal(det.lastConfirmedPeak, 0);
        assert.deepEqual(det.recentIBIs, []);
    });
});
