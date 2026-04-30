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
