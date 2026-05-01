# API Reference

## Module exports

```javascript
import {
  BrowserAdapter,       // Main entry point for browser apps
  VitalCamera,          // Core orchestrator (low-level)
  KalmanFilter1D,       // 1D Kalman filter utility
  loadModels,           // Shortcut for BrowserAdapter.loadModels
  EMOTION_LABELS,       // ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
  EMOTION_EMOJIS,       // Corresponding emoji for each label
  EMOTION_COLORS,       // Corresponding color hex for each label
} from 'vitalcamera-sdk';
```

---

## BrowserAdapter

The main class for browser applications. Handles camera, face detection, frame loop, and plot rendering.

### Constructor

```javascript
const adapter = new BrowserAdapter({
  videoElement,          // HTMLVideoElement — camera preview target
  models,               // Object — preloaded model ArrayBuffers (optional)
  modelBasePath,         // string — path to model files (default './models/')
  vitalcameraConfig,     // Object — passed through to VitalCamera (see Configuration)
  workerBasePath,        // string — custom worker script path (rarely needed)
  cameraFacing,          // 'user' | 'environment' (default 'user')
  canvases,              // { bvp: Canvas, trend: Canvas } — for built-in waveform plots
  faceDetector,          // async (source) => { box, keypoints } | null — custom detector
  manageCamera,          // boolean (default true) — auto-open camera
});
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `adapter.vitalcamera` | `VitalCamera` | The underlying core instance. Use this to subscribe to events. |

### Methods

#### `adapter.init()` → `Promise<void>`

Initialize the SDK: load models (from CDN or local), create workers, load face detector, and optionally open the camera.

```javascript
await adapter.init();
```

#### `adapter.start()`

Start the automatic frame loop. Requires `videoElement` in the constructor.

```javascript
adapter.start();
```

#### `adapter.stop()`

Pause the frame loop. Camera stream stays open, workers stay alive.

```javascript
adapter.stop();
```

#### `adapter.destroy()` → `Promise<void>`

Fully shut down: stop processing, release camera, terminate all workers.

```javascript
await adapter.destroy();
```

#### `adapter.processVideoFrame(source, timestamp?)`

Process a single frame manually. Use this when you manage the camera yourself (set `manageCamera: false`).

```javascript
adapter.processVideoFrame(myVideoElement);
// or
adapter.processVideoFrame(myCanvas, performance.now());
```

**Parameters:**
- `source` — `HTMLVideoElement | HTMLCanvasElement | OffscreenCanvas | ImageBitmap`
- `timestamp` — `number` (optional, defaults to `performance.now()`)

### Static Methods

#### `BrowserAdapter.loadModels(basePath, options?)` → `Promise<Object>`

Load model files from a directory.

```javascript
// Load all models
const models = await BrowserAdapter.loadModels('./models/');

// Load only heart-rate models (skip emotion / gaze / eye-state)
const models = await BrowserAdapter.loadModels('./models/', {
  emotion: false,
  gaze: false,
  eyeState: false,
});
```

**Returns:** An object of ArrayBuffers keyed by model name (`rppg`, `rppgProj`, `sqi`, `psd`, `state`, `emotion`, `gaze`, `eyeState`).

### Static Constants

| Constant | Value |
|----------|-------|
| `EMOTION_LABELS` | `['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']` |
| `EMOTION_EMOJIS` | Corresponding emoji array |
| `EMOTION_COLORS` | Corresponding hex color array |
| `MODEL_FILES` | Default model filename mapping |

---

## VitalCamera

Low-level core class. You typically access it via `adapter.vitalcamera` rather than creating it directly.

### Events

Subscribe to events using `vitalcamera.on(event, callback)`.

#### `'heartrate'`

Emitted when a heart rate estimate is available (~1/second).

```javascript
vc.on('heartrate', ({ hr, sqi, psd, freq, peak, timestamp }) => {
  // hr       — number, BPM
  // sqi      — number, signal quality index (0–1, higher = better)
  // psd      — number[], power spectral density values
  // freq     — number[], corresponding frequency bins
  // peak     — number, peak frequency (Hz)
  // timestamp — number, Date.now()
});
```

#### `'bvp'`

Emitted every frame with the raw blood volume pulse value.

```javascript
vc.on('bvp', ({ value, timestamp, time }) => {
  // value     — number, raw BVP signal
  // timestamp — number, frame timestamp
  // time      — number, inference time in ms
});
```

#### `'beat'`

Emitted on each detected heartbeat.

```javascript
vc.on('beat', ({ ibi, timestamp }) => {
  // ibi       — number, inter-beat interval in ms
  // timestamp — number, peak time
});
```

#### `'hrv'`

Emitted when HRV metrics are computed (requires ≥15 seconds of clean signal).
Computation runs inside the PSD worker (off the main thread).

```javascript
vc.on('hrv', ({ rmssd, sdnn, meanRR, n, timestamp }) => {
  // rmssd  — number, root mean square of successive RR differences (ms)
  // sdnn   — number, standard deviation of NN intervals (ms)
  // meanRR — number, mean RR interval (ms; 60000/meanRR ≈ HR in BPM)
  // n      — number, RR intervals used after filtering
});
```

#### `'emotion'`

Emitted with emotion classification results (~2/second).

```javascript
vc.on('emotion', ({ emotion, probs, time, timestamp }) => {
  // emotion — string, top predicted emotion label
  // probs   — number[], probability for each of the 8 classes
  // time    — number, inference time in ms
});
```

#### `'gaze'`

Emitted with gaze direction estimates (~5/second). Frames with low confidence (e.g., during blinks) are automatically filtered out.

```javascript
vc.on('gaze', ({ yaw, pitch, confidence, time, timestamp }) => {
  // yaw        — number, horizontal gaze angle (radians)
  // pitch      — number, vertical gaze angle (radians)
  // confidence — [number, number], softmax peak probability for [yaw, pitch]
  // time       — number, inference time in ms
});
```

#### `'eyestate'`

Emitted every frame with per-eye open/closed probabilities. The OCEC model
(112 KB) classifies each eye independently. During fast head motion the
adapter substitutes a neutral 0.6 probability so a momentarily-blurred
crop doesn't trigger a false blink.

```javascript
vc.on('eyestate', ({ left, right, bothClosed, time, timestamp }) => {
  // left  — { prob: number, open: boolean }   (subject's left eye)
  // right — { prob: number, open: boolean }   (subject's right eye)
  // bothClosed — boolean, true when both probs < eyeStateThreshold
  // time  — number, inference time in ms
});
```

`open` is the boolean derived from `prob >= config.eyeStateThreshold`
(default 0.5). The raw `prob` is also exposed because the gaze worker
uses a stricter cutoff (`gazeEyeOpenGateProb`, default 0.7) to gate
its own updates — when both eyes' max prob is below 0.7, the gaze
inference is skipped that frame.

#### `'headpose'`

Emitted every frame with head orientation.

```javascript
vc.on('headpose', ({ yaw, pitch, roll, normal, timestamp }) => {
  // yaw    — number, degrees
  // pitch  — number, degrees
  // roll   — number, degrees
  // normal — [x, y, z], face normal vector
});
```

#### `'face'`

Emitted by BrowserAdapter every frame when a face is detected.

```javascript
adapter.vitalcamera.on('face', (data) => {
  // data.box       — { x, y, w, h }
  // data.keypoints — [{ x, y }, ...] (6 BlazeFace keypoints, pixel coords:
  //                  right_eye, left_eye, nose_tip, mouth, right_ear, left_ear)
  // data.videoW    — number, video width
  // data.videoH    — number, video height
});
```

#### `'error'`

Emitted on any worker or processing error.

```javascript
vc.on('error', ({ source, message }) => {
  console.error(`[${source}]`, message);
});
```

#### `'ready'`

Emitted once when all workers are initialized and ready.

```javascript
vc.on('ready', () => console.log('All workers ready'));
```

### Methods

| Method | Description |
|--------|-------------|
| `vc.on(event, fn)` | Subscribe to an event |
| `vc.off(event, fn)` | Unsubscribe |
| `vc.start()` | Start processing (resets internal state) |
| `vc.stop()` | Pause processing |
| `vc.destroy()` | Terminate all workers |

---

## KalmanFilter1D

A simple 1D Kalman filter for smoothing noisy signals (e.g., gaze, emotion probabilities).

```javascript
import { KalmanFilter1D } from 'vitalcamera-sdk';

const kf = new KalmanFilter1D(initialValue, processNoise, measurementNoise);
// processNoise     — how fast the signal can change (higher = more responsive)
// measurementNoise — how noisy the measurements are (higher = more smoothing)

const smoothed = kf.update(newMeasurement);  // returns filtered value
kf.predict();                                 // advance state without measurement
```

**Recommended settings:**

| Use case | processNoise | measurementNoise |
|----------|-------------|-----------------|
| Emotion smoothing | 0.02 | 0.3 |
| Gaze smoothing | 0.5 | 0.5 |

---

Next: **[Configuration →](configuration.md)** for all available options.
