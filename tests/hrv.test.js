import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import {
    detectBvpPeaks,
    rejectAbnormalPeaks,
    refinePeaksParabolic,
    quotientFilterRR,
    madFilterRR,
    computeHrvMetrics,
} from '../src/core/hrv.js';

import RealtimePeakDetector from '../src/core/peak-detect.js';

// Helper: build a regularly-sampled timestamped sinusoid as {t, v} pairs.
function makeSinusoid(fs, durationSec, freqHz, phase = 0) {
    const n = Math.floor(fs * durationSec);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
        out[i] = {
            t: i * (1000 / fs),
            v: Math.sin(2 * Math.PI * freqHz * i / fs + phase),
        };
    }
    return out;
}

// ---------------------------------------------------------------------------
// detectBvpPeaks  (now operates on {t, v}[] directly)
// ---------------------------------------------------------------------------
describe('detectBvpPeaks', () => {
    it('returns empty for too-short input', () => {
        assert.deepEqual(detectBvpPeaks([]), []);
        assert.deepEqual(detectBvpPeaks([{ t: 0, v: 0 }]), []);
    });

    it('finds peaks in a sinusoidal signal at 30 Hz', () => {
        const samples = makeSinusoid(30, 5, 1.2);   // ~6 peaks
        const peaks = detectBvpPeaks(samples);
        assert.ok(peaks.length >= 4, `Expected >= 4 peaks, got ${peaks.length}`);
        // Inter-peak time should be near 1/1.2 s = 833 ms
        for (let i = 1; i < peaks.length; i++) {
            const ibi = samples[peaks[i]].t - samples[peaks[i - 1]].t;
            assert.ok(ibi > 700 && ibi < 950, `IBI ${ibi} out of expected range`);
        }
    });

    it('respects the min-IBI gap', () => {
        // Two close peaks; only the higher should survive
        const samples = [
            { t: 0,  v: 0 }, { t: 33, v: 0 }, { t: 66, v: 0 },
            { t: 100, v: 1.0 },   // peak A
            { t: 133, v: 0.5 },
            { t: 166, v: 0.9 },   // close to A but lower → suppressed
            { t: 200, v: 0.4 },
            { t: 600, v: 1.1 },   // peak B (>333 ms after A)
            { t: 633, v: 0.5 }, { t: 666, v: 0.4 },
        ];
        const peaks = detectBvpPeaks(samples, 333);
        // peak A at idx 3 (t=100), peak B at idx 7 (t=600)
        assert.ok(peaks.length <= 2);
    });
});

// ---------------------------------------------------------------------------
// refinePeaksParabolic  (now returns refined timestamps in ms)
// ---------------------------------------------------------------------------
describe('refinePeaksParabolic', () => {
    it('returns the raw timestamp at array boundaries', () => {
        const samples = [{t:0,v:1},{t:33,v:2},{t:66,v:3}];
        const refined = refinePeaksParabolic([0, 2], samples);
        assert.equal(refined[0], 0);
        assert.equal(refined[1], 66);
    });

    it('hits sub-grid precision on a synthetic sinusoid', () => {
        // 1.2 Hz sine sampled at 30 Hz; integer peaks miss true peak by up to 16 ms.
        const fs = 30, freq = 1.2, phase = 0.123;
        const samples = makeSinusoid(fs, 6, freq, phase);
        const peaks = detectBvpPeaks(samples);
        const refined = refinePeaksParabolic(peaks, samples);

        // Ground-truth peak times: where 2π·f·i/fs + φ = π/2 + 2πk
        const trueT0 = 1000 * (0.25 - phase / (2 * Math.PI)) / freq;
        const truePeriod = 1000 / freq;

        let intErr = 0, refErr = 0;
        for (let k = 0; k < peaks.length; k++) {
            const tTrue = trueT0 + k * truePeriod;
            intErr += Math.abs(samples[peaks[k]].t - tTrue);
            refErr += Math.abs(refined[k] - tTrue);
        }
        const meanIntErr = intErr / peaks.length;
        const meanRefErr = refErr / peaks.length;
        // Refined error should be a *lot* smaller than integer error.
        assert.ok(meanRefErr < meanIntErr * 0.3,
            `Refinement should improve > 3x: integer ${meanIntErr.toFixed(2)} ms, refined ${meanRefErr.toFixed(2)} ms`);
        // And in absolute terms, well under 5 ms.
        assert.ok(meanRefErr < 5,
            `Refined error ${meanRefErr.toFixed(2)} ms should be sub-5 ms`);
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
        const rr = [800, 810, 400, 790, 805];
        const result = quotientFilterRR(rr);
        assert.ok(!result.includes(400));
    });

    it('rejects physiologically impossible values', () => {
        const rr = [800, 810, 200, 805, 815];
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
        assert.ok(!result.includes(2500));
        assert.ok(result.includes(800));
        assert.ok(result.includes(810));
    });
});

// ---------------------------------------------------------------------------
// computeHrvMetrics
// ---------------------------------------------------------------------------
describe('computeHrvMetrics', () => {
    it('returns null for fewer than 6 intervals', () => {
        assert.equal(computeHrvMetrics([800, 810]), null);
        assert.equal(computeHrvMetrics([800, 810, 790, 805, 815]), null);
    });

    it('computes correct RMSSD for known input', () => {
        // 6 RRs → 5 diffs: [10, -20, 15, 0, -5]
        // sumSq = 100 + 400 + 225 + 0 + 25 = 750
        const result = computeHrvMetrics([800, 810, 790, 805, 805, 800]);
        assert.ok(result);
        const expected = Math.sqrt(750 / 5);
        assert.ok(Math.abs(result.rmssd - expected) < 1e-6);
    });

    it('returns RMSSD=0 for constant intervals', () => {
        const result = computeHrvMetrics([800, 800, 800, 800, 800, 800]);
        assert.ok(result);
        assert.ok(Math.abs(result.rmssd) < 1e-10);
    });
});

// ---------------------------------------------------------------------------
// RealtimePeakDetector  (unchanged)
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
        const fs = 30, freqHz = 1.0, duration = 10;
        const n = fs * duration;
        const beats = [];
        for (let i = 0; i < n; i++) {
            const t = i * (1000 / fs);
            const v = Math.sin(2 * Math.PI * freqHz * i / fs) + 1;
            const result = det.process(t, v);
            if (result) beats.push(result);
        }
        assert.ok(beats.length >= 3);
        for (const b of beats) {
            assert.ok(b.ibi > 500 && b.ibi < 1500);
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

// ---------------------------------------------------------------------------
// computeHrv  (one-shot pipeline used by both worker and direct callers)
// ---------------------------------------------------------------------------
import { computeHrv } from '../src/core/hrv.js';

import { filterCompensatingPairs } from '../src/core/hrv.js';

describe('computeHrv', () => {
    it('returns null for too-short input', () => {
        assert.equal(computeHrv([]), null);
        assert.equal(computeHrv([{t:0,v:1},{t:33,v:0.5}]), null);
    });

    it('produces RMSSD + SDNN for a realistic 30 s window', () => {
        const fs = 30, dur = 30, baseFreq = 1.2;
        const samples = [];
        let phase = 0;
        for (let i = 0; i < fs * dur; i++) {
            const t = i * (1000 / fs);
            const f = baseFreq * (1 + 0.05 * Math.sin(t / 4000));
            phase += 2 * Math.PI * f / fs;
            samples.push({ t, v: Math.sin(phase) });
        }
        const r = computeHrv(samples);
        assert.ok(r, 'should produce a result');
        assert.ok(Number.isFinite(r.rmssd) && r.rmssd >= 0);
        assert.ok(Number.isFinite(r.sdnn) && r.sdnn >= 0);
        assert.ok(Number.isFinite(r.meanRR) && r.meanRR > 0);
        assert.ok(typeof r.n === 'number' && r.n > 4);
    });
});

describe('filterCompensatingPairs', () => {
    it('passes a clean RR series unchanged', () => {
        const rr = [800, 810, 795, 802, 798, 808, 805, 800];
        const out = filterCompensatingPairs(rr);
        assert.deepEqual(out, rr);
    });

    it('drops both members of an obvious compensating pair', () => {
        // peak detected ~150 ms early between idx 3 & 4 → bad pair
        const rr = [800, 800, 800, 650, 950, 800, 800];
        const out = filterCompensatingPairs(rr);
        assert.equal(out.length, 5);
        for (const v of out) assert.ok(Math.abs(v - 800) < 50);
    });

    it('keeps natural HRV variation', () => {
        const rr = [800, 830, 770, 815, 825, 790, 805, 810, 795];
        const out = filterCompensatingPairs(rr);
        assert.ok(out.length >= rr.length - 2,
            `expected ≥ ${rr.length - 2}, got ${out.length}`);
    });
});
