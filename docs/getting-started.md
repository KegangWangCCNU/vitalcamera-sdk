# Getting Started

This guide takes you from zero to a working heart-rate monitor in about 5 minutes. No build tools, no npm, no server — just an HTML file and a browser.

## Prerequisites

- A modern browser (Chrome, Edge, Firefox, Safari)
- A webcam
- HTTPS or localhost (required for camera access)

> **Tip:** The easiest way to serve locally is `npx serve .` or Python's `python -m http.server`. Both give you localhost which satisfies the camera permission requirement.

## Step 1: Create your HTML file

Create a file called `index.html`:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>My Vital Camera App</title>
</head>
<body>
  <h1>Heart Rate Monitor</h1>
  <video id="cam" autoplay playsinline muted width="480"></video>
  <p>Heart Rate: <span id="hr">--</span> BPM</p>
  <button id="startBtn">Start</button>

  <script type="module">
    import { BrowserAdapter } from 'https://cdn.jsdelivr.net/npm/vitalcamera-sdk@0.5.0/src/index.js';

    const adapter = new BrowserAdapter({
      videoElement: document.getElementById('cam'),
    });

    document.getElementById('startBtn').addEventListener('click', async () => {
      document.getElementById('startBtn').disabled = true;
      document.getElementById('startBtn').textContent = 'Loading models...';

      await adapter.init();    // loads models, opens camera
      adapter.start();         // begins frame processing

      document.getElementById('startBtn').textContent = 'Running';

      adapter.vitalcamera.on('heartrate', ({ hr, sqi }) => {
        if (sqi > 0.38) {
          document.getElementById('hr').textContent = hr.toFixed(0);
        }
      });
    });
  </script>
</body>
</html>
```

## Step 2: Serve it

```bash
# Option A: Node.js
npx serve .

# Option B: Python
python -m http.server 8080
```

Open `http://localhost:8080` (or whatever port is shown).

## Step 3: Click Start

1. Click **Start** — the SDK downloads the models from CDN (~15 MB total, cached after first load).
2. Look at your webcam — keep your face visible and stay still.
3. After about 10 seconds, a heart rate value will appear.

That's it! You have a working heart rate monitor.

---

## Going further

### Add emotion detection

```javascript
adapter.vitalcamera.on('emotion', ({ emotion, probs }) => {
  console.log('Detected emotion:', emotion, probs);
});
```

### Add gaze tracking

```javascript
adapter.vitalcamera.on('gaze', ({ yaw, pitch, confidence }) => {
  console.log('Gaze direction:', yaw.toFixed(1), pitch.toFixed(1));
});
```

### Add head pose

```javascript
adapter.vitalcamera.on('headpose', ({ yaw, pitch, roll }) => {
  console.log('Head pose:', yaw.toFixed(1), pitch.toFixed(1), roll.toFixed(1));
});
```

### Personalize emotion classification

The default emotion baseline is tuned for Asian faces. For best per-user
accuracy, pass 2+ neutral-expression photos:

```javascript
const adapter = new BrowserAdapter({
  videoElement: document.getElementById('cam'),
  emotionCalibration: {
    images: [
      'data:image/jpeg;base64,/9j/4AAQ...',  // load from <input type="file">
      'data:image/jpeg;base64,/9j/4AAQ...',  // or anywhere else
    ],
  },
});

await adapter.init();   // calibration runs automatically here
```

Calibration is fully transparent — the `'emotion'` event payload is identical
with or without it. See [Configuration → Emotion calibration](configuration.md#emotion-calibration)
for the full details.

### Add HRV (Heart Rate Variability)

```javascript
adapter.vitalcamera.on('hrv', ({ rmssd }) => {
  console.log('RMSSD:', rmssd.toFixed(1), 'ms');
});
```

HRV requires at least 15 seconds of clean BVP signal before the first reading.

### Use with npm / bundlers

```bash
npm install vitalcamera-sdk
```

```javascript
import { BrowserAdapter } from 'vitalcamera-sdk';
// Same API as the CDN version
```

### Load only the models you need

By default, all models are loaded (rPPG, emotion, gaze). To save bandwidth:

```javascript
import { BrowserAdapter } from 'vitalcamera-sdk';

// Heart rate only — skip emotion & gaze models
const models = await BrowserAdapter.loadModels('./models/', {
  emotion: false,
  gaze: false,
});

const adapter = new BrowserAdapter({
  videoElement: document.getElementById('cam'),
  models,
});
await adapter.init();
adapter.start();

// Only rPPG + PSD workers run. Zero emotion/gaze overhead.
```

### Self-host the models

Instead of loading from CDN, you can host the model files yourself:

1. Download the models from the [GitHub releases](https://github.com/KegangWangCCNU/FacePhys-Release/releases)
2. Place them in a `models/` directory next to your HTML
3. The SDK will automatically find them there (default path is `./models/`)

Expected files in `models/`:

| File | Purpose |
|------|---------|
| `model.tflite` | rPPG inference (required) |
| `proj.tflite` | rPPG projection (required) |
| `sqi_model.tflite` | Signal quality index (required) |
| `psd_model.tflite` | PSD / heart rate (required) |
| `state.gz` | Warm-start state (required) |
| `enet_b0_8_best_vgaf_dynamic_int8.tflite` | Emotion (optional) |
| `mobileone_s0_gaze_float16.tflite` | Gaze (optional) |
| `blaze_face_short_range.tflite` | MediaPipe face detector (auto-loaded by built-in detector) |

### Bring your own face detector

The SDK uses MediaPipe BlazeFace by default. You can swap in your own:

```javascript
const adapter = new BrowserAdapter({
  videoElement: document.getElementById('cam'),
  faceDetector: async (video) => {
    // Your custom face detection logic
    // Return: { box: {x, y, w, h}, keypoints: [{x, y}, ...] } or null
    return myDetector.detect(video);
  },
});
```

### Bring your own camera

If you manage the camera stream yourself (e.g., from a canvas or WebRTC):

```javascript
const adapter = new BrowserAdapter({
  videoElement: myVideo,
  manageCamera: false,  // don't open camera automatically
});
await adapter.init();
adapter.vitalcamera.start();

// Feed frames manually in your own loop:
function loop() {
  adapter.processVideoFrame(myVideo);
  requestAnimationFrame(loop);
}
loop();
```

---

Next: **[API Reference →](api.md)** for the full class documentation.
