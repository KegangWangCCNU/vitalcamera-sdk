import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { softmax } from '../src/utils/math.js';

const NUM_BINS = 90;
const BIN_WIDTH = 4;
const ANGLE_OFFSET = 180;

function decodeAngle(logits) {
    const probs = softmax(logits);
    let angle = 0;
    for (let i = 0; i < NUM_BINS; i++) {
        angle += probs[i] * i;
    }
    return (angle * BIN_WIDTH - ANGLE_OFFSET) * Math.PI / 180;
}

const IMG_IDX = 1;
const DT_IDX = 0;
const INPUT_COUNT = 48;

const STATE_MAP = [
    { inIdx: 2,  outIdx: 1  }, { inIdx: 3,  outIdx: 12 }, { inIdx: 14, outIdx: 23 },
    { inIdx: 25, outIdx: 34 }, { inIdx: 36, outIdx: 42 }, { inIdx: 43, outIdx: 43 },
    { inIdx: 44, outIdx: 44 }, { inIdx: 45, outIdx: 45 }, { inIdx: 46, outIdx: 46 },
    { inIdx: 47, outIdx: 2  }, { inIdx: 4,  outIdx: 3  }, { inIdx: 5,  outIdx: 4  },
    { inIdx: 6,  outIdx: 5  }, { inIdx: 7,  outIdx: 6  }, { inIdx: 8,  outIdx: 7  },
    { inIdx: 9,  outIdx: 8  }, { inIdx: 10, outIdx: 9  }, { inIdx: 11, outIdx: 10 },
    { inIdx: 12, outIdx: 11 }, { inIdx: 13, outIdx: 13 }, { inIdx: 15, outIdx: 14 },
    { inIdx: 16, outIdx: 15 }, { inIdx: 17, outIdx: 16 }, { inIdx: 18, outIdx: 17 },
    { inIdx: 19, outIdx: 18 }, { inIdx: 20, outIdx: 19 }, { inIdx: 21, outIdx: 20 },
    { inIdx: 22, outIdx: 21 }, { inIdx: 23, outIdx: 22 }, { inIdx: 24, outIdx: 24 },
    { inIdx: 26, outIdx: 25 }, { inIdx: 27, outIdx: 26 }, { inIdx: 28, outIdx: 27 },
    { inIdx: 29, outIdx: 28 }, { inIdx: 30, outIdx: 29 }, { inIdx: 31, outIdx: 30 },
    { inIdx: 32, outIdx: 31 }, { inIdx: 33, outIdx: 32 }, { inIdx: 34, outIdx: 33 },
    { inIdx: 35, outIdx: 35 }, { inIdx: 37, outIdx: 36 }, { inIdx: 38, outIdx: 37 },
    { inIdx: 39, outIdx: 38 }, { inIdx: 40, outIdx: 39 }, { inIdx: 41, outIdx: 40 },
    { inIdx: 42, outIdx: 41 }
];

const PROJ_INPUT_SOURCE_INDICES = [4, 5, 12, 15, 16, 23, 26, 27, 34, 37, 38, 40, 41, 46];

describe('softmax (worker shared)', () => {
    it('probabilities sum to 1', () => {
        const r = softmax(new Float32Array([1, 2, 3, 4]));
        const s = r.reduce((a, b) => a + b, 0);
        assert.ok(Math.abs(s - 1) < 1e-6);
    });
    it('equal logits give equal probs', () => {
        const r = softmax(new Float32Array([5, 5, 5]));
        for (const v of r) assert.ok(Math.abs(v - 1/3) < 1e-6);
    });
    it('concentrates on max logit', () => {
        const r = softmax(new Float32Array([0, 0, 100]));
        assert.ok(r[2] > 0.99);
    });
    it('handles single element', () => {
        const r = softmax(new Float32Array([42]));
        assert.ok(Math.abs(r[0] - 1) < 1e-6);
    });
    it('handles large negative logits', () => {
        const r = softmax(new Float32Array([-1000, -999, -998]));
        assert.ok(!r.some(v => isNaN(v)));
    });
});

describe('decodeAngle', () => {
    it('bin=0 -> -pi', () => {
        const l = new Float32Array(NUM_BINS).fill(-100);
        l[0] = 100;
        assert.ok(Math.abs(decodeAngle(l) - (-Math.PI)) < 0.01);
    });
    it('bin=45 -> 0', () => {
        const l = new Float32Array(NUM_BINS).fill(-100);
        l[45] = 100;
        assert.ok(Math.abs(decodeAngle(l)) < 0.01);
    });
    it('bin=89 -> near +pi', () => {
        const l = new Float32Array(NUM_BINS).fill(-100);
        l[89] = 100;
        const expected = (89 * BIN_WIDTH - ANGLE_OFFSET) * Math.PI / 180;
        assert.ok(Math.abs(decodeAngle(l) - expected) < 0.01);
    });
    it('uniform logits -> weighted average', () => {
        const l = new Float32Array(NUM_BINS).fill(0);
        const expectedBin = (NUM_BINS - 1) / 2;
        const expected = (expectedBin * BIN_WIDTH - ANGLE_OFFSET) * Math.PI / 180;
        assert.ok(Math.abs(decodeAngle(l) - expected) < 0.01);
    });
});

describe('STATE_MAP', () => {
    it('all inIdx in [0, INPUT_COUNT)', () => {
        for (const e of STATE_MAP) assert.ok(e.inIdx >= 0 && e.inIdx < INPUT_COUNT);
    });
    it('all outIdx in [0, INPUT_COUNT)', () => {
        for (const e of STATE_MAP) assert.ok(e.outIdx >= 0 && e.outIdx < INPUT_COUNT);
    });
    it('no IMG_IDX or DT_IDX as inIdx', () => {
        for (const e of STATE_MAP) {
            assert.notEqual(e.inIdx, IMG_IDX);
            assert.notEqual(e.inIdx, DT_IDX);
        }
    });
    it('unique inIdx values', () => {
        const ids = STATE_MAP.map(e => e.inIdx);
        assert.equal(new Set(ids).size, ids.length);
    });
    it('PROJ indices in range', () => {
        for (const idx of PROJ_INPUT_SOURCE_INDICES) assert.ok(idx >= 0 && idx < INPUT_COUNT);
    });
});
