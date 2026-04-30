# Vital Camera SDK

Browser-based real-time physiological sensing — extract heart rate, HRV, emotion, gaze direction, and head pose from a standard webcam. No wearables needed.

## Features

- **Heart Rate (rPPG)** — remote photoplethysmography via face video using State Space Models
- **HRV** — RMSSD and other heart rate variability metrics from BVP peak detection
- **Emotion** — 8-class facial emotion recognition (EfficientNet-B0)
- **Gaze** — yaw/pitch eye direction estimation (MobileOne-S0)
- **Head Pose** — yaw/pitch/roll from MediaPipe face landmarks
- **Pure browser** — runs entirely client-side with Web Workers and TFLite/LiteRT

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

### Heart Rate Only (minimal resource usage)

Only provide the models you need — workers for missing models are never created:

```javascript
const adapter = new BrowserAdapter({
    videoElement: document.getElementById('cam'),
    models: { rppg, rppgProj, sqi, psd },  // no emotion, no gaze
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
  ├── inference_worker  →  rPPG model  →  BVP signal
  ├── psd_worker        →  PSD model   →  peak frequency → HR
  ├── emotion_worker    →  ENet-B0     →  8-class probs
  ├── gaze_worker       →  MobileOne   →  yaw/pitch
  ├── plot_worker       →  OffscreenCanvas rendering
  ├── RealtimePeakDetector  →  beat events
  └── HRV pipeline      →  RMSSD metrics

BrowserAdapter (optional)
  ├── Camera management (managed mode)
  ├── Face detection (MediaPipe)
  ├── Head pose estimation
  └── iOS compatibility (playsinline)
```

## Events

| Event | Payload | Rate |
|-------|---------|------|
| `heartrate` | `{ hr, peakFreq, psd, freq }` | ~1/s |
| `bvp` | `{ value, sqi }` | 30/s |
| `beat` | `{ timestamp, interval }` | per beat |
| `hrv` | `{ rmssd, sdnn, meanRR }` | ~1/s |
| `emotion` | `{ label, probs, logits }` | 2/s |
| `gaze` | `{ yaw, pitch }` | 5/s |
| `headpose` | `{ yaw, pitch, roll, normal }` | 30/s |
| `face` | `{ box, keypoints, videoWidth, videoHeight }` | 30/s |
| `error` | `{ message }` | on error |

## Models

Models are **not included** in the npm package. You need to provide them as `ArrayBuffer` objects. Required models:

| Model | File | Purpose |
|-------|------|---------|
| rppg | `model.tflite` | rPPG inference (FacePhys SSM) |
| rppgProj | `proj.tflite` | Projection matrix |
| sqi | `sqi_model.tflite` | Signal quality index |
| psd | `psd_model.tflite` | Power spectral density |
| emotion | `enet_b0_8_*.tflite` | Emotion classification (optional) |
| gaze | `mobileone_s0_gaze_*.tflite` | Gaze estimation (optional) |

## Tests

```bash
npm test   # 94 tests across 5 files
```

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
  year      = {20