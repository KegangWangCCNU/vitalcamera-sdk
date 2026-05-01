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
  enableEmotion:  true,       // Run emotion classification
  enableGaze:     true,       // Run gaze estimation
  enableEyeState: true,       // Run OCEC per-eye open/closed
  enableHeadPose: true,       // Run head pose estimation
  enableHrv:      true,       // Run HRV computation (in PSD worker)

  // ── Thresholds ──
  sqiThreshold:             0.38,    // SQI threshold for HR display
  gazeConfidenceThreshold:  0.04,    // Min softmax peak to accept gaze
  eyeStateThreshold:        0.5,     // p(open) >= threshold → "open"
  gazeEyeOpenGateProb:      0.7,     // Skip gaze unless max(L,R) eye-prob ≥ this
  hrvSqiThreshold:          0.6,     // Min SQI to admit a sample to the HRV buffer

  // ── HRV settings ──
  hrvMinDuration:    15,      // Seconds of data before first HRV output
  hrvMaxWindow:     120,      // Sliding window cap (seconds)
  hrvUpdateInterval: 1000,    // Ms between HRV recalculations
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

Milliseconds between HRV recalculations. The HRV pipeline runs inside the
PSD worker; the main thread just dispatches the current 2-minute sample
buffer at this cadence.

#### `hrvMaxWindow` (default: `120`)

Sliding window cap in seconds — older BVP samples than this are evicted
from the buffer. SDNN especially is sensitive to window length; 2 min
matches short-term HRV literature norms.

#### `hrvSqiThreshold` (default: `0.6`)

Per-sample SQI gate — BVP samples whose latest SQI score is below this
are NOT admitted into the HRV buffer. This is the only outlier filter
at the sample level; the HRV pipeline does its own RR-interval
filtering downstream.

#### `eyeStateThreshold` (default: `0.5`)

Probability cutoff for the public `open` boolean field on the
`'eyestate'` event. Display-side decision; doesn't affect data.

#### `gazeEyeOpenGateProb` (default: `0.7`)

Stricter cutoff used to gate the gaze inference. When the latest
max(L, R) eye-open probability is below this, gaze inference is
skipped that frame — no `'gaze'` event fires, the consumer's Kalman
filter just predicts forward without a measurement update.

---

## Emotion calibration

The 8-class emotion model has a known bias against neutral Asian male faces —
without calibration, `Anger` / `Contempt` are over-predicted at rest. The SDK
addresses this in two layers:

### 1. Built-in default baseline (zero config)

If you don't pass `emotionCalibration`, the SDK uses a **built-in baseline tuned
for Asian faces**. This is applied automatically inside the emotion worker;
no user code change needed. For most users this is enough.

### 2. Per-user calibration via `emotionCalibration.images`

For best accuracy, supply 2+ photos of the user with a neutral expression. The
SDK runs them through the emotion model once during `init()` and uses the
averaged logits as that user's personal baseline:

```javascript
const adapter = new BrowserAdapter({
    videoElement: document.getElementById('cam'),
    emotionCalibration: {
        images: [
            'data:image/jpeg;base64,/9j/4AAQSkZJRg...',
            'data:image/jpeg;base64,/9j/4AAQSkZJRg...',
            'data:image/jpeg;base64,/9j/4AAQSkZJRg...',
        ],
    },
});

await adapter.init();
adapter.start();
```

**Requirements:**

- Each entry is a base64 data URL (or any string a `<img>` element accepts in `src`).
- At least 2 images must be usable. Face detection runs on each — if BlazeFace
  fails on an image, the SDK falls back to the full image. If fewer than 2 images
  yield a usable crop, init() proceeds with the **default baseline** and an
  `'error'` event is emitted with `source: 'emotionCalibration'`.
- 5–10 frontal, well-lit, neutral-expression photos give the best result.

**What happens under the hood (informational only):**

The SDK applies a KL-divergence-based blend internally, mixing 30% of the
baseline distribution with 70% of a calibration-deviation distribution. None of
this is exposed in the `'emotion'` event — payload is always
`{ emotion, probs, time, timestamp }` regardless of which baseline mode is in
use. Calibration is fully transparent to API consumers.

### When NOT to calibrate

- If you're building for a global / mixed-ethnicity audience and can't collect
  per-user photos: just don't pass `emotionCalibration`. The default Asian baseline
  is still better than no calibration for many users, and the SDK will work
  out-of-the-box.
- If you want to see raw model output for debugging: there's no public way to
  disable calibration. Build against `src/workers/emotion.worker.js` directly
  (out of scope for production use).

---

## Example: minimal heart-rate only

```javascript
const models = await BrowserAdapter.loadModels('./models/', {
  emotion: false,
  gaze: false,
  eyeState: false,
});

const adapter = new BrowserAdapter({
  videoElement: document.getElementById('cam'),
  models,
  vitalcameraConfig: {
    enableEmotion: false,
    enableGaze: false,
    enableEyeState: false,
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
