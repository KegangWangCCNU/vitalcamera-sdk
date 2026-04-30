import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import KalmanFilter1D from '../src/utils/kalman.js';
import { softmax, clamp } from '../src/utils/math.js';

// --------------- KalmanFilter1D ---------------

describe('KalmanFilter1D', () => {
    it('uses default noise parameters', () => {
        const kf = new KalmanFilter1D(0);
        assert.equal(kf.x, 0);
        assert.equal(kf.p, 1.0);
        assert.equal(kf.q, 1e-2);
        assert.equal(kf.r, 5e-1);
    });

    it('accepts custom noise parameters', () => {
        const kf = new KalmanFilter1D(5, 0.1, 0.2);
        assert.equal(kf.q, 0.1);
        assert.equal(kf.r, 0.2);
    });

    it('converges toward a constant measurement', () => {
        const kf = new KalmanFilter1D(0);
        const target = 10;
        let estimate;
        for (let i = 0; i < 100; i++) {
            estimate = kf.update(target);
        }
        assert.ok(
            Math.abs(estimate - target) < 0.01,
            `Expected convergence to ${target}, got ${estimate}`
        );
    });

    it('smooths a noisy signal', () => {
        const kf = new KalmanFilter1D(50);
        const trueValue = 50;
        const measurements = Array.from({ length: 200 }, () =>
            trueValue + (Math.random() - 0.5) * 20
        );
        let lastEstimate;
        for (const m of measurements) {
            lastEstimate = kf.update(m);
        }
        // filtered estimate should be closer to the true mean than a
        // typical single noisy sample would be
        assert.ok(
            Math.abs(lastEstimate - trueValue) < 5,
            `Filtered estimate ${lastEstimate} too far from ${trueValue}`
        );
    });

    it('returns the updated state from update()', () => {
        const kf = new KalmanFilter1D(0);
        const result = kf.update(10);
        assert.equal(result, kf.x);
    });

    it('predict() grows uncertainty without changing state', () => {
        const kf = new KalmanFilter1D(5, 0.02, 0.3);
        const stateBefore = kf.x;
        const pBefore = kf.p;
        const result = kf.predict();
        assert.equal(result, stateBefore, 'state should not change');
        assert.equal(kf.x, stateBefore, 'x should not change');
        assert.ok(kf.p > pBefore, 'uncertainty should grow');
        assert.ok(
            Math.abs(kf.p - (pBefore + kf.q)) < 1e-10,
            'p should increase by exactly q'
        );
    });

    it('multiple predicts make next update respond faster', () => {
        // After many predicts, uncertainty is high → Kalman gain is large
        // → update pulls harder toward measurement
        const kf1 = new KalmanFilter1D(0, 0.02, 0.3);
        kf1.update(10); // one normal update

        const kf2 = new KalmanFilter1D(0, 0.02, 0.3);
        for (let i = 0; i < 50; i++) kf2.predict(); // grow uncertainty
        kf2.update(10);

        // kf2 should be closer to 10 since high uncertainty → high gain
        assert.ok(
            kf2.x > kf1.x,
            `After predicts, update should pull harder: kf2=${kf2.x} vs kf1=${kf1.x}`
        );
    });

    it('predict-then-update pattern works for sub-sampled signals', () => {
        const kf = new KalmanFilter1D(70, 0.02, 0.3);
        // Simulate: 5 frames of predict, then 1 measurement, repeat
        for (let cycle = 0; cycle < 10; cycle++) {
            for (let f = 0; f < 5; f++) kf.predict();
            kf.update(80);
        }
        assert.ok(
            Math.abs(kf.x - 80) < 1,
            `Should converge to 80, got ${kf.x}`
        );
    });
});

// --------------- softmax ---------------

describe('softmax', () => {
    it('output sums to 1', () => {
        const out = softmax([1, 2, 3, 4]);
        const sum = Array.from(out).reduce((a, b) => a + b, 0);
        assert.ok(Math.abs(sum - 1) < 1e-5, `Sum was ${sum}`);
    });

    it('returns uniform distribution for equal logits', () => {
        const out = softmax([0, 0, 0, 0]);
        for (let i = 0; i < out.length; i++) {
            assert.ok(
                Math.abs(out[i] - 0.25) < 1e-5,
                `Expected 0.25, got ${out[i]}`
            );
        }
    });

    it('is numerically stable with large values', () => {
        const out = softmax([1000, 1001, 1002]);
        const sum = Array.from(out).reduce((a, b) => a + b, 0);
        assert.ok(Math.abs(sum - 1) < 1e-5, `Sum was ${sum}`);
        // largest logit should have largest probability
        assert.ok(out[2] > out[1]);
        assert.ok(out[1] > out[0]);
    });

    it('assigns dominant probability to a single large logit', () => {
        const out = softmax([0, 0, 100]);
        assert.ok(out[2] > 0.99, `Dominant value was ${out[2]}`);
    });

    it('returns a Float32Array', () => {
        const out = softmax([1, 2]);
        assert.ok(out instanceof Float32Array);
    });
});

// --------------- clamp ---------------

describe('clamp', () => {
    it('clamps value below min', () => {
        assert.equal(clamp(-5, 0, 10), 0);
    });

    it('clamps value above max', () => {
        assert.equal(clamp(15, 0, 10), 10);
    });

    it('returns value when within range', () => {
        assert.equal(clamp(5, 0, 10), 5);
    });

    it('returns min when value equals min', () => {
        assert.equal(clamp(0, 0, 10), 0);
    });

    it('returns max when value equals max', () => {
        assert.equal(clamp(10, 0, 10), 10);
    });
});
