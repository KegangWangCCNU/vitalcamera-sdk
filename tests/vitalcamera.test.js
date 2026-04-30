import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import VitalCamera from '../src/core/vitalcamera.js';

// ---------------------------------------------------------------------------
// EventEmitter mixin
// ---------------------------------------------------------------------------
describe('VitalCamera EventEmitter', () => {
    it('on() registers and emit() fires listeners', () => {
        const vs = new VitalCamera();
        const received = [];
        vs.on('test', (data) => received.push(data));
        vs.emit('test', 'a');
        vs.emit('test', 'b');
        assert.deepEqual(received, ['a', 'b']);
    });

    it('off() removes a specific listener', () => {
        const vs = new VitalCamera();
        const received = [];
        const fn = (d) => received.push(d);
        vs.on('test', fn);
        vs.emit('test', 1);
        vs.off('test', fn);
        vs.emit('test', 2);
        assert.deepEqual(received, [1]);
    });

    it('emit() with no listeners does not throw', () => {
        const vs = new VitalCamera();
        assert.doesNotThrow(() => vs.emit('nonexistent', {}));
    });

    it('multiple listeners on the same event', () => {
        const vs = new VitalCamera();
        const a = [], b = [];
        vs.on('x', (v) => a.push(v));
        vs.on('x', (v) => b.push(v));
        vs.emit('x', 42);
        assert.deepEqual(a, [42]);
        assert.deepEqual(b, [42]);
    });

    it('listener errors do not break other listeners', () => {
        const vs = new VitalCamera();
        const received = [];
        vs.on('x', () => { throw new Error('boom'); });
        vs.on('x', (v) => received.push(v));
        vs.emit('x', 99);
        assert.deepEqual(received, [99]);
    });

    it('on() returns this for chaining', () => {
        const vs = new VitalCamera();
        const ret = vs.on('a', () => {});
        assert.equal(ret, vs);
    });

    it('off() returns this for chaining', () => {
        const vs = new VitalCamera();
        const ret = vs.off('a', () => {});
        assert.equal(ret, vs);
    });
});

// ---------------------------------------------------------------------------
// Constructor and defaults
// ---------------------------------------------------------------------------
describe('VitalCamera constructor', () => {
    it('applies default config values', () => {
        const vs = new VitalCamera();
        assert.equal(vs.config.workerBasePath, null);
        assert.equal(vs.config.enableEmotion, true);
        assert.equal(vs.config.enableGaze, true);
        assert.equal(vs.config.enableHeadPose, true);
        assert.equal(vs.config.enableHrv, true);
        assert.equal(vs.config.hrvTargetFs, 200);
        assert.equal(vs.config.hrvMinDuration, 15);
        assert.equal(vs.config.hrvUpdateInterval, 1000);
        assert.equal(vs.config.sqiThreshold, 0.6);
    });

    it('overrides defaults with provided config', () => {
        const vs = new VitalCamera({
            enableEmotion: false,
            hrvTargetFs: 100,
            sqiThreshold: 0.8,
        });
        assert.equal(vs.config.enableEmotion, false);
        assert.equal(vs.config.hrvTargetFs, 100);
        assert.equal(vs.config.sqiThreshold, 0.8);
        // untouched defaults
        assert.equal(vs.config.enableGaze, true);
    });

    it('initial state flags', () => {
        const vs = new VitalCamera();
        assert.equal(vs.isRunning, false);
        assert.equal(vs._workersReady, false);
    });
});

// ---------------------------------------------------------------------------
// Lifecycle (non-worker parts)
// ---------------------------------------------------------------------------
describe('VitalCamera lifecycle', () => {
    it('start() sets isRunning and resets state', () => {
        const vs = new VitalCamera();
        vs._bvpSamples.push({ t: 1, v: 2 });
        vs._lastHR = 75;
        vs.start();
        assert.equal(vs.isRunning, true);
        assert.deepEqual(vs._bvpSamples, []);
        assert.deepEqual(vs._bvpRing, []);
        assert.equal(vs._lastHR, null);
        assert.equal(vs._lastHrvTime, 0);
    });

    it('stop() clears isRunning', () => {
        const vs = new VitalCamera();
        vs.start();
        vs.stop();
        assert.equal(vs.isRunning, false);
    });

    it('destroy() clears isRunning and workersReady', () => {
        const vs = new VitalCamera();
        vs._workersReady = true;
        vs.isRunning = true;
        vs.destroy();
        assert.equal(vs.isRunning, false);
        assert.equal(vs._workersReady, false);
        assert.deepEqual(vs._workers, {});
    });
});

// ---------------------------------------------------------------------------
// processFrame gate — does nothing when not running
// ---------------------------------------------------------------------------
describe('VitalCamera processFrame gate', () => {
    it('does not emit when isRunning is false', () => {
        const vs = new VitalCamera();
        const events = [];
        vs.on('headpose', (e) => events.push(e));
        vs.processFrame({
            rppgInput: new Float32Array(1),
            dtVal: 0.033,
            timestamp: 1000,
            faceKeypoints: [
                { x: 30, y: 50 }, { x: 70, y: 50 },
                { x: 50, y: 70 }, { x: 50, y: 100 },
                { x: 10, y: 50 }, { x: 90, y: 50 },
            ],
        });
        assert.equal(events.length, 0);
    });
});

// ---------------------------------------------------------------------------
// Head pose integration
// ---------------------------------------------------------------------------
describe('VitalCamera head pose integration', () => {
    it('emits headpose event from faceKeypoints', () => {
        const vs = new VitalCamera({ enableHeadPose: true });
        vs.start();

        const events = [];
        vs.on('headpose', (e) => events.push(e));

        const kps = [
            { x: 30, y: 50 }, { x: 70, y: 50 },
            { x: 50, y: 70 }, { x: 50, y: 100 },
            { x: 10, y: 50 }, { x: 90, y: 50 },
        ];
        vs.processFrame({
            rppgInput: new Float32Array(1),
            dtVal: 0.033,
            timestamp: 5000,
            faceKeypoints: kps,
        });

        assert.equal(events.length, 1);
        const e = events[0];
        assert.equal(e.timestamp, 5000);
        assert.ok(Number.isFinite(e.yaw));
        assert.ok(Number.isFinite(e.pitch));
        assert.ok(Number.isFinite(e.roll));
        assert.ok(e.normal && Number.isFinite(e.normal.x));
    });

    it('does not emit headpose when enableHeadPose is false', () => {
        const vs = new VitalCamera({ enableHeadPose: false });
        vs.start();
        const events = [];
        vs.on('headpose', (e) => events.push(e));

        vs.processFrame({
            rppgInput: new Float32Array(1),
            dtVal: 0.033,
            timestamp: 5000,
            faceKeypoints: [
                { x: 30, y: 50 }, { x: 70, y: 50 },
                { x: 50, y: 70 }, { x: 50, y: 100 },
                { x: 10, y: 50 }, { x: 90, y: 50 },
            ],
        });

        assert.equal(events.length, 0);
    });

    it('does not emit headpose when keypoints are missing', () => {
        const vs = new VitalCamera({ enableHeadPose: true });
        vs.start();
        const events = [];
        vs.on('headpose', (e) => events.push(e));

        vs.processFrame({
            rppgInput: new Float32Array(1),
            dtVal: 0.033,
            timestamp: 5000,
        });

        assert.equal(events.length, 0);
    });
});

// ---------------------------------------------------------------------------
// Peak detector integration
// ---------------------------------------------------------------------------
describe('VitalCamera peak detector integration', () => {
    it('emits beat events when inference results contain a periodic signal', () => {
        const vs = new VitalCamera({ enableHrv: false });
        vs.start();

        const beats = [];
        vs.on('beat', (e) => beats.push(e));
        vs.on('bvp', () => {}); // just to confirm bvp is emitted too

        // Simulate inference worker results: 1 Hz sine over 10 seconds
        const fs = 30;
        const freqHz = 1.0;
        const duration = 10;
        const n = fs * duration;

        for (let i = 0; i < n; i++) {
            const t = i * (1000 / fs);
            const v = Math.sin(2 * Math.PI * freqHz * i / fs) + 1;
            // Directly call the internal handler to bypass worker
            vs._onInferenceResult({ value: v, timestamp: t });
        }

        assert.ok(beats.length >= 3, `Expected >= 3 beats, got ${beats.length}`);
        for (const b of beats) {
            assert.ok(b.ibi > 500 && b.ibi < 1500, `IBI ${b.ibi} out of range`);
            assert.ok(typeof b.timestamp === 'number');
        }
    });

    it('emits bvp event for each inference result', () => {
        const vs = new VitalCamera({ enableHrv: false });
        vs.start();

        const bvps = [];
        vs.on('bvp', (e) => bvps.push(e));

        vs._onInferenceResult({ value: 0.5, timestamp: 100 });
        vs._onInferenceResult({ value: 0.7, timestamp: 133 });

        assert.equal(bvps.length, 2);
        assert.equal(bvps[0].value, 0.5);
        assert.equal(bvps[0].timestamp, 100);
    });
});

// ---------------------------------------------------------------------------
// HRV pipeline integration
// ---------------------------------------------------------------------------
describe('VitalCamera HRV pipeline', () => {
    it('computeHrvFromSamples returns valid RMSSD for a periodic signal', () => {
        const vs = new VitalCamera({ hrvTargetFs: 200 });

        // Generate 20 seconds of 1.2 Hz sine at irregular ~30 Hz timestamps
        const samples = [];
        const freqHz = 1.2;
        for (let i = 0; i <= 600; i++) {
            const t = i * 33.3; // ~30 fps, total ~20 s
            const v = Math.sin(2 * Math.PI * freqHz * t / 1000);
            samples.push({ t, v });
        }

        const result = vs.computeHrvFromSamples(samples);
        assert.ok(result, 'Should produce HRV metrics for 20 s of data');
        assert.ok(Number.isFinite(result.rmssd), 'RMSSD should be finite');
        assert.ok(result.rmssd >= 0, 'RMSSD should be non-negative');
    });

    it('computeHrvFromSamples returns null for too-short data', () => {
        const vs = new VitalCamera();
        const samples = [
            { t: 0, v: 0 }, { t: 100, v: 1 }, { t: 200, v: 0 },
        ];
        assert.equal(vs.computeHrvFromSamples(samples), null);
    });

    it('emits hrv event after enough BVP samples accumulate', () => {
        const vs = new VitalCamera({
            enableHrv: true,
            hrvTargetFs: 200,
            hrvMinDuration: 5,       // lower threshold for test
            hrvUpdateInterval: 100,  // lower interval for test
        });
        vs.start();

        const hrvEvents = [];
        vs.on('hrv', (e) => hrvEvents.push(e));

        // Feed 8 seconds of 1 Hz sine at ~30 fps
        const freqHz = 1.0;
        const fs = 30;
        const n = fs * 8;
        for (let i = 0; i < n; i++) {
            const t = i * (1000 / fs);
            const v = Math.sin(2 * Math.PI * freqHz * i / fs);
            vs._onInferenceResult({ value: v, timestamp: t });
        }

        // With 8 s of data, minDuration 5 s, and updateInterval 100 ms,
        // we should get at least one HRV event
        assert.ok(hrvEvents.length >= 1, `Expected >= 1 HRV events, got ${hrvEvents.length}`);
        for (const e of hrvEvents) {
            assert.ok(Number.isFinite(e.rmssd));
            assert.ok(typeof e.timestamp === 'number');
        }
    });
});

// ---------------------------------------------------------------------------
// Worker message routing (unit-level, mocked)
// ---------------------------------------------------------------------------
describe('VitalCamera worker message routing', () => {
    it('_onPsdResult emits heartrate and caches last HR', () => {
        const vs = new VitalCamera();
        const hrEvents = [];
        vs.on('heartrate', (e) => hrEvents.push(e));

        vs._onPsdResult({ sqi: 0.9, hr: 72 });
        assert.equal(hrEvents.length, 1);
        assert.equal(hrEvents[0].hr, 72);
        assert.equal(hrEvents[0].sqi, 0.9);
        assert.equal(vs._lastHR, 72);
    });

    it('_onPsdResult ignores null HR', () => {
        const vs = new VitalCamera();
        const hrEvents = [];
        vs.on('heartrate', (e) => hrEvents.push(e));
        vs._onPsdResult({ sqi: 0.5, hr: null });
        assert.equal(hrEvents.length, 0);
    });

    it('_onEmotionResult emits emotion event', () => {
        const vs = new VitalCamera();
        const events = [];
        vs.on('emotion', (e) => events.push(e));

        vs._onEmotionResult({ emotion: 'Happiness', probs: [0.1, 0.9] });
        assert.equal(events.length, 1);
        assert.equal(events[0].emotion, 'Happiness');
        assert.deepEqual(events[0].probs, [0.1, 0.9]);
    });

    it('_onGazeResult emits gaze event', () => {
        const vs = new VitalCamera();
        const events = [];
        vs.on('gaze', (e) => events.push(e));

        vs._onGazeResult({ angles: [0.15, -0.3] });
        assert.equal(events.length, 1);
        assert.equal(events[0].yaw, 0.15);
        assert.equal(events[0].pitch, -0.3);
    });

    it('_onGazeResult ignores invalid angles', () => {
        const vs = new VitalCamera();
        const events = [];
        vs.on('gaze', (e) => events.push(e));

        vs._onGazeResult({ angles: null });
        vs._onGazeResult({ angles: [0.1] }); // too short
        vs._onGazeResult({}); // missing
        assert.equal(events.length, 0);
    });

    it('_handleWorkerMessage routes error type to error event', () => {
        const vs = new VitalCamera();
        const errors = [];
        vs.on('error', (e) => errors.push(e));

        vs._handleWorkerMessage('psd', { data: { type: 'error', msg: 'oops' } });
        assert.equal(errors.length, 1);
        assert.equal(errors[0].source, 'psd');
        assert.equal(errors[0].message, 'oops');
    });
});
