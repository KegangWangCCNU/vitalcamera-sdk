import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { estimateHeadPose } from '../src/core/headpose.js';

// --- helpers ---

/** Frontal face: eyes at y=50, mouth at y=100, nose at 0.4 of eye-mouth span */
function makeFrontalKps() {
    return [
        { x: 30, y: 50 },   // right_eye
        { x: 70, y: 50 },   // left_eye
        { x: 50, y: 66 },   // nose_tip  (midY=50, mouth.y=100 → 0.4*(100-50)=20 → 50+20 = 70? let's compute: noseFrac = (66-50)/(100-50) = 16/50 = 0.32)
        { x: 50, y: 100 },  // mouth
        { x: 10, y: 50 },   // right_ear
        { x: 90, y: 50 },   // left_ear
    ];
}

/**
 * Make keypoints with nose at a specific position to control yaw/pitch.
 * noseX controls yaw, noseY controls pitch.
 */
function makeKps({ noseX = 50, noseY = 70, eyeAngle = 0 } = {}) {
    const midX = 50, midY = 50;
    const halfSpan = 20;
    const dx = Math.cos(eyeAngle * Math.PI / 180) * halfSpan;
    const dy = Math.sin(eyeAngle * Math.PI / 180) * halfSpan;
    return [
        { x: midX - dx, y: midY - dy },   // right_eye
        { x: midX + dx, y: midY + dy },   // left_eye
        { x: noseX, y: noseY },           // nose_tip
        { x: 50, y: 100 },                // mouth
        { x: 10, y: 50 },                 // right_ear
        { x: 90, y: 50 },                 // left_ear
    ];
}

function approx(actual, expected, tolerance = 2, msg = '') {
    assert.ok(
        Math.abs(actual - expected) <= tolerance,
        `${msg} expected ~${expected}, got ${actual} (tol=${tolerance})`
    );
}

// --- tests ---

describe('estimateHeadPose', () => {

    it('frontal face: yaw ≈ 0, pitch ≈ 0, roll ≈ 0', () => {
        // For true frontal: noseRatio = (50-30)/(70-30) = 0.5 → yaw = 0
        // noseFrac = (70-50)/(100-50) = 0.4 → pitch = 0
        const kps = makeKps({ noseX: 50, noseY: 70 });
        const r = estimateHeadPose(kps);
        approx(r.yaw, 0, 1, 'yaw');
        approx(r.pitch, 0, 1, 'pitch');
        approx(r.roll, 0, 1, 'roll');
    });

    it('frontal face: normal ≈ (0, 0, 1)', () => {
        const kps = makeKps({ noseX: 50, noseY: 70 });
        const r = estimateHeadPose(kps);
        approx(r.normal.x, 0, 0.05, 'nx');
        approx(r.normal.y, 0, 0.05, 'ny');
        approx(r.normal.z, 1, 0.05, 'nz');
    });

    it('head turned right: positive yaw', () => {
        // Nose shifted toward left_eye (higher x)
        const kps = makeKps({ noseX: 60, noseY: 70 });
        const r = estimateHeadPose(kps);
        assert.ok(r.yaw > 5, `expected positive yaw, got ${r.yaw}`);
    });

    it('head turned left: negative yaw', () => {
        // Nose shifted toward right_eye (lower x)
        const kps = makeKps({ noseX: 40, noseY: 70 });
        const r = estimateHeadPose(kps);
        assert.ok(r.yaw < -5, `expected negative yaw, got ${r.yaw}`);
    });

    it('head tilted (roll): non-zero roll', () => {
        const kps = makeKps({ eyeAngle: 15 });
        const r = estimateHeadPose(kps);
        approx(r.roll, 15, 2, 'roll');
    });

    it('looking up: positive pitch', () => {
        // noseFrac < 0.4 → pitch > 0
        // noseFrac = (noseY - midY) / (mouth.y - midY)
        // midY = 50, mouth.y = 100 → (noseY - 50) / 50 < 0.4 → noseY < 70
        const kps = makeKps({ noseX: 50, noseY: 60 });
        const r = estimateHeadPose(kps);
        assert.ok(r.pitch > 0, `expected positive pitch (looking up), got ${r.pitch}`);
    });

    it('looking down: negative pitch', () => {
        // noseFrac > 0.4 → pitch < 0
        // noseY > 70
        const kps = makeKps({ noseX: 50, noseY: 80 });
        const r = estimateHeadPose(kps);
        assert.ok(r.pitch < 0, `expected negative pitch (looking down), got ${r.pitch}`);
    });

    it('normal vector for ~45° yaw', () => {
        // noseRatio such that yaw ≈ 45 → (noseRatio - 0.5)*90 = 45 → noseRatio = 1.0
        // noseRatio = (noseX - reX) / (leX - reX)
        // reX=30, leX=70 → noseX = 30 + 1.0*40 = 70
        const kps = makeKps({ noseX: 70, noseY: 70 });
        const r = estimateHeadPose(kps);
        approx(r.yaw, 45, 3, 'yaw');
        approx(r.normal.x, Math.sin(45 * Math.PI / 180), 0.1, 'nx ~sin(45°)');
    });

    it('edge case: zero eye span → graceful handling', () => {
        const kps = [
            { x: 50, y: 50 },  // right_eye
            { x: 50, y: 50 },  // left_eye (same as right → zero span)
            { x: 50, y: 70 },  // nose
            { x: 50, y: 100 }, // mouth
            { x: 10, y: 50 },  // right_ear
            { x: 90, y: 50 },  // left_ear
        ];
        const r = estimateHeadPose(kps);
        // Should not throw, noseRatio defaults to 0.5
        assert.ok(Number.isFinite(r.yaw), 'yaw should be finite');
        assert.ok(Number.isFinite(r.pitch), 'pitch should be finite');
        assert.ok(Number.isFinite(r.roll), 'roll should be finite');
        approx(r.yaw, 0, 1, 'yaw with zero span');
    });

    it('makeFrontalKps helper produces near-frontal results', () => {
        const kps = makeFrontalKps();
        const r = estimateHeadPose(kps);
        // noseRatio = (50-30)/(70-30) = 0.5 → yaw = 0
        // noseFrac = (66-50)/(100-50) = 0.32 → pitch = (0.32-0.4)*-90 = 7.2
        approx(r.yaw, 0, 1, 'yaw');
        approx(r.roll, 0, 1, 'roll');
        // pitch won't be exactly 0 with this helper since noseFrac ≈ 0.32
        assert.ok(Math.abs(r.pitch) < 15, `pitch should be small-ish, got ${r.pitch}`);
    });
});
