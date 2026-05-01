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
