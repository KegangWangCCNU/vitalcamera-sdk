# Configuration

All configuration is passed through the `BrowserAdapter` constructor. Options specific to VitalCamera core go inside `vitalcameraConfig`.

## BrowserAdapter options

```javascript
const adapter = new BrowserAdapter({
  // ── Required ──
  videoElement: document.getElementById('cam'),

  // ── Optional ──
  models: null,               // Pre-loaded model buffers (skip auto-download)
  modelBasePath: './models/',  // Where to find model files
  cameraFacing: 'user',       // 'user' (front) or 'environment' (back)
  manageCamera: true,          // Auto-open camera on init()
  faceDetector: null,          // Custom face detector function
  workerBasePath: null,        // Custom worker script path (rarely needed)

  emotionCalibration: null,    // { images: string[] } — optional per-user baseline
                               // (see "Emotion calibration" section below)

  canvases: {                  // Built-in waveform rendering
    bvp: bvpCanvas,            // Canvas for BVP waveform
    trend: trendCanvas,        // Canvas for HR trend line
  },

  vitalcameraConfig: {         // Passed through to VitalCamera core
    // ... see below
  },
});
```

## VitalCamera core options

Pass these inside `vitalcameraConfig`:

```javascript
vitalcameraConfig: {
  // ── Feature toggles ──
  enableEmotion: true,       // Run emotion classification
  enableGaze: true,          // Run gaze estimation
  enableHeadPose: true,      // Run head pose estimation
  enableHrv: true,           // Run HRV computation

  // ── Thresholds ──
  sqiThreshold: 0.38,               // Signal quality threshold for HR display
  gazeConfidenceThreshold: 0.04,    // Min softmax peak to accept gaze (blink filter)

  // ── HRV settings ──
  hrvTargetFs: 200,          // Interpolation sample rate (Hz)
  hrvMinDuration: 15,        // Min seconds of data before first HRV output
  hrvUpdateInterval: 1000,   // Ms between HRV recalculations
}
```

### Parameter details

#### `sqiThreshold` (default: `0.38`)

The Signal Quality Index threshold. When SQI is below this value, heart rate readings are considered unreliable. The built-in plot worker marks these periods as invalid on the trend line.

Lower values show more readings (but more noise). Higher values are stricter. The original FacePhys uses 0.38.

#### `gazeConfidenceThreshold` (default: `0.04`)

Minimum softmax peak probability for gaze estimation. When the model's confidence drops below this (e.g., during a blink), the gaze event is suppressed — the consumer keeps the last valid reading.

The uniform baseline is 1/90 ≈ 0.011 (90 angle bins). Typical values:

| Scenario | Peak probability |
|----------|-----------------|
| Strong gaze | 0.3 – 0.6+ |
| Moderate gaze | 0.08 – 0.15 |
| Blink / closed eyes | 0.01 – 0.03 |

#### `enableEmotion` / `enableGaze` (default: `true`)

Set to `false` to skip loading the corresponding model and worker entirely. This saves bandwidth and CPU:

```javascript
vitalcameraConfig: {
  enableEmotion: false,  // No emotion worker
  enableGaze: false,     // No gaze worker
}
```

> **Note:** You also need to exclude the model files when calling `loadModels()`:
> ```javascript
> const models = await BrowserAdapter.loadModels('./models/', {
>   emotion: false,
>   gaze: false,
> });
> ```

#### `hrvMinDuration` (default: `15`)

Seconds of accumulated BVP data required before the first HRV computation. Shorter values give faster first reading but less reliable results.

#### `hrvUpdateInterval` (default: `1000`)

Milliseconds between HRV recalculations. The HRV pipeline (interpolation → peak detection → RR filtering → RMSSD) runs on every update.

---

## Emotion calibration

The 8-class emotion model has a known bias against neutral Asian male faces —
without calibration, `Anger` / `Contempt` are over-predicted at rest. The SDK
exposes four levels:

### 1. Built-in default baseline (zero config)

If you don't pass `emotionCalibration`, the SDK uses a **built-in baseline tuned
for Asian faces**. This is applied automatically inside the emotion worker;
no user code change needed. For most users this is enough.

### 2. Per-user calibration from images

Supply 2+ photos of the user at rest. The SDK runs them through the model
once during `init()` and uses the averaged logits as that user's baseline:

```javascript
emotionCalibration: {
    images: [
        'data:image/jpeg;base64,/9j/4AAQSkZJRg...',
        'data:image/jpeg;base64,/9j/4AAQSkZJRg...',
        'data:image/jpeg;base64,/9j/4AAQSkZJRg...',
    ],
}
```

**Requirements:** each entry is a data URL or any `<img>`-acceptable string;
≥ 2 must be usable; 5–10 frontal, well-lit, neutral-expression photos give
the sharpest result. If fewer than 2 are usable, init() proceeds with the
default baseline and emits an `error` event with `source: 'emotionCalibration'`.

### 3. Pre-computed baseline distribution

Skip the image step and pass an 8-vector of raw logits directly:

```javascript
emotionCalibration: {
    baseline: [3.26, 0.31, -3.97, -3.30, -3.22, 4.53, 4.37, -0.53],
    //          Anger Cont  Disg   Fear  Happ  Neut  Sad  Surp
}
```

Useful for shipping a baseline you captured offline once for a known target
population. If both `images` and `baseline` are supplied, `images` wins.

### 4. Dynamic baseline (runtime EMA drift)

```javascript
emotionCalibration: {
    dynamic: { halfLifeMs: 5000 },
}
```

After every successful inference the worker EMA-folds the just-observed raw
logits into the active baseline. The mixing coefficient is derived from a
half-life so you can think in seconds:
`alpha = 1 - 0.5^(dt / halfLifeMs)` per call. The semantic effect:

- Sustained smile / frown → baseline drifts that direction → KL-blend
  re-centres the result on Neutral. Held expressions fade to neutral over
  ~halfLife seconds; only **deviations from your typical expression** light up.
- Brief expressions (≪ halfLife) barely move the baseline.

`dynamic` stacks on top of `images` / `baseline` — the EMA starts from
whatever baseline you initialised with.

### Stacking all three

```javascript
emotionCalibration: {
    images:   [/* one-time per-user calibration at init */ ...],
    baseline: [/* fallback if images can't be processed */ ...],
    dynamic:  { halfLifeMs: 5000 },
}
```

### Under the hood (informational only)

KL-divergence-based blend internally:
- `pCal[Neutral] = exp(-KL(pCur || pBase) / 0.6)` (the "still at rest" confidence)
- positive KL contributions per class are redistributed to the non-Neutral classes
- output = **0.2 × baseline + 0.8 × calibrated** (was 0.3/0.7 before 0.6.0)

None of this surfaces in the `'emotion'` event — payload is always
`{ emotion, probs, time, timestamp }` regardless of which mode is active.

### When NOT to calibrate

- Mixed-ethnicity audience without per-user photos → just don't pass
  `emotionCalibration`. The default Asian baseline is still safer than raw
  logits for many users.
- Debugging raw model output → no public way to disable calibration; build
  against `src/workers/emotion.worker.js` directly (out of scope for
  production use).

---

## Example: minimal heart-rate only

```javascript
const models = await BrowserAdapter.loadModels('./models/', {
  emotion: false,
  gaze: false,
});

const adapter = new BrowserAdapter({
  videoElement: document.getElementById('cam'),
  models,
  vitalcameraConfig: {
    enableEmotion: false,
    enableGaze: false,
    enableHeadPose: false,
    enableHrv: false,
  },
});

await adapter.init();
adapter.start();

adapter.vitalcamera.on('heartrate', ({ hr, sqi }) => {
  if (sqi > 0.38) console.log('HR:', hr.toFixed(0), 'BPM');
});
```

This configuration loads only the 4 required models (~8 MB instead of ~15 MB) and runs only 2 workers (inference + PSD).
