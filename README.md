# Vital Camera SDK

Browser-based real-time physiological sensing — extract heart rate, HRV, emotion, gaze, eye state, mouth (jawOpen + speaking), and head pose from a standard webcam. No wearables needed.

**[Live Demo](https://kegangwangccnu.github.io/vitalcamera-sdk/examples/demo.html)** · **[Documentation](https://kegangwangccnu.github.io/vitalcamera-sdk/docs/)**

## Features

- **Heart Rate (rPPG)** — remote photoplethysmography via face video using State Space Models
- **HRV** — RMSSD + SDNN from BVP peak detection, with reject-reason surfaced when the window is too noisy
- **Emotion** — 8-class facial emotion recognition (EfficientNet-B0), with three calibration modes: per-user images, pre-computed baseline distribution, or runtime EMA drift
- **Gaze** — yaw/pitch eye direction estimation (MobileOne-S0 / L2CS-Net), eye-state-gated, fed a tight Face-Landmarker-aligned face crop
- **Eye State** — per-eye open/closed from MediaPipe Face Landmarker `eyeBlink` blendshapes (478-landmark mesh)
- **Mouth** — `jawOpen` + speaking heuristic (rolling-variance) from the same Face Landmarker
- **Head Pose** — yaw/pitch/roll from MediaPipe face landmarks
- **Pure browser** — runs entirely client-side with Web Workers, TFLite/LiteRT, and MediaPipe tasks-vision

## Install

```bash
npm install vitalcamera-sdk
```

Or via CDN:

```html
<script type="module">
import { VitalCamera, BrowserAdapter } from 'https://cdn.jsdelivr.net/gh/KegangWangCCNU/vitalcamera-sdk/src/index.js';
</script>
```

## Quick Start

### Managed Mode (adapter handles camera)

```javascript
import { BrowserAdapter } from 'vitalcamera-sdk/adapter';

const adapter = new BrowserAdapter({
    videoElement: document.getElementById('cam'),
    models: { rppg, rppgProj, sqi, psd, emotion, gaze },  // ArrayBuffers
});

await adapter.init();

adapter.vitalcamera.on('heartrate', ({ hr }) => {
    console.log('Heart rate:', hr, 'bpm');
});

adapter.vitalcamera.on('emotion', ({ label, probs }) => {
    console.log('Emotion:', label, probs);
});

adapter.vitalcamera.on('gaze', ({ yaw, pitch }) => {
    console.log('Gaze:', yaw, pitch);
});

adapter.vitalcamera.on('face', ({ box, keypoints }) => {
    // Draw your own face overlay
});

adapter.start();
```

### Personalize the emotion baseline (optional)

Three independent calibration modes, all combinable. Out of the box (no
config) the SDK uses a built-in baseline tuned for Asian faces — Anger /
Contempt / Disgust biases at rest are corrected automatically.

```javascript
const adapter = new BrowserAdapter({
    videoElement: document.getElementById('cam'),
    emotionCalibration: {
        // 1) per-user calibration from 2+ neutral-expression photos
        images: ['data:image/jpeg;base64,/9j/4AAQ...', ...],

        // 2) OR pre-computed 8-vector of raw logits (e.g. captured offline)
        baseline: [3.2, 0.3, -3.9, -3.3, -3.2, 4.5, 4.4, -0.5],

        // 3) AND/OR runtime EMA: the baseline drifts toward sustained
        //    expressions, so the visible signal becomes "deviation from
        //    your typical expression".  Boolean shorthand uses a 5 s
        //    half-life; the object form lets you tune it.
        dynamic: true,                       // enable with default 5 s
        // dynamic: { halfLifeMs: 3000 },    // or specify a custom half-life
        // dynamic: false,                   // explicit off (or just omit the key)
    },
});

await adapter.init();   // images + baseline applied at init, dynamic runs continuously
```

`images` precedes `baseline` if both supplied; `dynamic` is independent of
both. When `dynamic` is enabled the SDK auto-persists the mutating baseline
to IndexedDB every ~2 s, so a returning user skips the warm-up wobble.
The `'emotion'` event payload is identical regardless of which modes are
active. See
[docs/configuration.md](docs/configuration.md#emotion-calibration) for the
KL-blend math and details.

### Heart Rate Only (minimal resource usage)

Skip every heavy module — only fetch the rPPG / PSD models, turn off everything
else, and the BlazeFace-only fallback runs at ~3 ms / frame:

```javascript
const models = await BrowserAdapter.loadModels('./models/', {
    emotion: false,
    gaze: false,
    faceLandmarker: false,   // skip the 3.8 MB FL bundle
});

const adapter = new BrowserAdapter({
    videoElement: document.getElementById('cam'),
    models,
    vitalcameraConfig: {
        enableFaceLandmarker: false,   // master switch — implies eyestate / mouth / gaze off
        enableEmotion:        false,
        enableHeadPose:       false,
        enableHrv:            false,
    },
});

await adapter.init();
adapter.vitalcamera.on('heartrate', ({ hr }) => console.log(hr));
adapter.start();
// Only rPPG + PSD workers run. Zero emotion/gaze overhead.
```

### Manual Mode (you control the camera)

```javascript
import { BrowserAdapter } from 'vitalcamera-sdk/adapter';

const adapter = new BrowserAdapter({
    models: { rppg, rppgProj, sqi, psd },
});

await adapter.init();
adapter.vitalcamera.on('heartrate', ({ hr }) => console.log(hr));
adapter.vitalcamera.start();

// Your own camera + rAF loop
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
myVideo.srcObject = stream;

function loop() {
    adapter.processVideoFrame(myVideo);
    requestAnimationFrame(loop);
}
loop();
```

### Zero-Config Workers

Workers are loaded automatically via Blob URLs — no need to copy worker files or configure paths. This works seamlessly with CDN imports and local installs alike.

## Architecture

```
VitalCamera (core, DOM-free)
  ├── inference_worker     →  rPPG SSM            →  BVP signal
  ├── psd_worker           →  PSD model           →  peak frequency → HR
  │                         └─ HRV pipeline       →  RMSSD / SDNN
  ├── emotion_worker       →  ENet-B0 + KL-blend  →  8-class probs (per-user / dynamic calibrated)
  ├── gaze_worker          →  L2CS-Net MobileOne  →  yaw/pitch
  ├── face_landmarker      →  MediaPipe Tasks     →  478 landmarks + 52 blendshapes (15 fps)
  │                         ├─ eyeBlinkL/R        →  'eyestate' event
  │                         └─ jawOpen + std      →  'mouth' event (jawOpen + speaking)
  ├── plot_worker          →  OffscreenCanvas rendering
  └── RealtimePeakDetector →  per-beat events

BrowserAdapter (optional)
  ├── Camera management (managed mode)
  ├── Face detection (MediaPipe FaceDetector — kept for face bbox)
  ├── Kalman-filtered face & eye boxes
  ├── Face-Landmarker-aligned gaze crop
  ├── Head pose estimation
  └── iOS compatibility (playsinline)
```

## Events

| Event | Payload | Rate |
|-------|---------|------|
| `heartrate` | `{ hr, sqi, psd, freq, peak, timestamp }` | ~2/s |
| `bvp` | `{ value, timestamp, time }` | 30/s |
| `beat` | `{ ibi, timestamp }` | per beat |
| `hrv` | `{ rmssd, sdnn, meanRR, n, reject, timestamp }` — `rmssd:null` + `reject:'…'` when invalid | ~1/s |
| `emotion` | `{ emotion, probs, time, timestamp }` | 2/s |
| `gaze` | `{ yaw, pitch, confidence, time, timestamp }` | 5/s |
| `eyestate` | `{ left:{prob,open}, right:{prob,open}, bothClosed, time, timestamp }` | 15/s |
| `mouth` | `{ jawOpen, jawStd, speaking, time, timestamp }` | 15/s |
| `headpose` | `{ yaw, pitch, roll, normal, timestamp }` | 30/s |
| `face` | `{ detected, box, keypoints, videoWidth, videoHeight, timestamp }` | 30/s |
| `ready` | `{}` | once after init |
| `error` | `{ source, message }` | on error |

## Performance

Each feature toggle has a different CPU cost. The full **per-model cost
table and per-feature sizing tips** live in **[Performance →](docs/performance.md)**:

| Feature | Cost (× face-detection / second) |
|---|---:|
| Face detection (always on) | 1.00× |
| Heart rate (always on) | 2.15× |
| Emotion | 1.38× |
| Face Landmarker (eye state + mouth piggyback for free) | 2.85× |
| Gaze (requires Face Landmarker) | 2.86× |
| HRV / head pose | 0× (pure JS) |

Tune via the dependency-grouped feature switches in
[Configuration](docs/configuration.md#face-landmarker-and-feature-toggles).

## Models

Models are **included** in the npm package and git repository under `models/`. The SDK loads them automatically. Included models:

| Model | File | Purpose |
|-------|------|---------|
| rppg | `model.tflite` | rPPG inference (FacePhys SSM) |
| rppgProj | `proj.tflite` | Projection matrix |
| sqi | `sqi_model.tflite` | Signal quality index |
| psd | `psd_model.tflite` | Power spectral density |
| emotion | `enet_b0_8_*.tflite` | Emotion classification (optional, ~4.5 MB) |
| gaze | `mobileone_s0_gaze_*.tflite` | Gaze estimation (optional, ~5 MB; requires Face Landmarker) |
| faceLandmarker | `face_landmarker.task` | MediaPipe Face Landmarker (~3.8 MB) — drives `eyestate`, `mouth`, gaze face crop. Disable with `loadModels({ faceLandmarker: false })` + `enableFaceLandmarker: false` for a lightweight build. |

## Citations

If you use this SDK in your research, please cite the relevant papers:

```bibtex
@article{wang2025facephys,
  title     = {FacePhys: State of the Heart Learning},
  author    = {Wang, Kegang and Tang, Jian and Wang, Yuntao and Liu, Xin
               and Fan, Yelin and Ji, Jiacheng and Shi, Yuanchun and McDuff, Daniel},
  journal   = {arXiv preprint arXiv:2512.06275},
  year      = {2025}
}

@inproceedings{bazarevsky2019blazeface,
  title     = {BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs},
  author    = {Bazarevsky, Valentin and Kartynnik, Yury and Vakunov, Andrey
               and Raveendran, Karthik and Grundmann, Matthias},
  booktitle = {CVPR Workshop on Computer Vision for AR/VR},
  year      = {2019}
}

@inproceedings{kartynnik2019facemesh,
  title     = {Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs},
  author    = {Kartynnik, Yury and Ablavatski, Artsiom and Grishchenko, Ivan
               and Grundmann, Matthias},
  booktitle = {CVPR Workshop on Computer Vision for AR/VR},
  year      = {2019}
}

@inproceedings{savchenko2022hsemotion,
  title     = {Video-Based Frame-Level Facial Analysis of Affective Behavior
               on Mobile Devices using EfficientNets},
  author    = {Savchenko, Andrey V.},
  booktitle = {CVPR Workshop on Affective Behavior Analysis in-the-Wild (ABAW)},
  year      = {2022}
}

@inproceedings{savchenko2023icml,
  title     = {Facial Expression Recognition with Adaptive Frame Rate
               based on Multiple Testing Correction},
  author    = {Savchenko, Andrey V.},
  booktitle = {Proceedings of ICML},
  pages     = {30119--30129},
  year      = {2023}
}

@inproceedings{abdelrahman2024l2cs,
  title     = {L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments},
  author    = {Abdelrahman, Ahmed A. and Hempel, Thorsten
               and Khalifa, Aly and Al-Hamadi, Ayoub},
  booktitle = {IEEE FG},
  year      = {2024}
}

@inproceedings{vasu2023mobileone,
  title     = {MobileOne: An Improved One Millisecond Mobile Backbone},
  author    = {Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff
               and Tuzel, Oncel and Ranjan, Anurag},
  booktitle = {CVPR},
  year      = {2023}
}
```

## License

MIT License with Privacy Protection Addendum — see [LICENSE](./LICENSE).

By using or distributing this software, you agree to the following additional terms:

1. **Strict Local Processing** — All biometric inference must be performed on the local device. You must NOT transmit user video feeds or physiological metrics to any external server.
2. **Consent Requirement** — You shall not use this Software to collect physiological data from any individual without their explicit consent.
3. **No Backdoors** — Redistributions must maintain these local-processing guarantees.

See [NOTICE](./NOTICE) for full third-party attribution.

---

*This SDK was built with the assistance of [Claude Code](https://claude.ai/code) by Anthropic.*
