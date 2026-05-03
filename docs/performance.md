# Performance & Cost Budget

The VitalCamera SDK runs **every inference locally** via LiteRT (TFLite
WASM). Different feature toggles cost different amounts of CPU; this page
gives you the numbers so you can size the SDK for your target device.

## Methodology

All numbers were measured on **Intel Xeon Gold 6138 @ 2.0 GHz**, LiteRT
XNNPACK CPU backend, **single-thread** (matches the browser-WASM
execution model). Each model was warmed up for 20 invocations and timed
over 100 invocations; medians are reported.

> **Browser caveat.** Native CPU LiteRT is roughly **2–3× faster** than
> the browser WASM build. Treat the absolute milliseconds as a lower
> bound and use the **× face-detection ratios** instead — those are
> backend-independent and translate directly to your end users' devices.

## Per-model cost (single inference)

`face_detection` (BlazeFace short-range) is the cheapest dedicated model
in the pipeline. We use it as the unit of cost.

| Model | Median (ms, native) | × face_det |
|---|---:|---:|
| `blaze_face_short_range` (BlazeFace) | 1.14 | **1.00×** |
| `model.tflite` (rPPG main, 48-input recurrent) | 2.38 | 2.09× |
| `proj.tflite` (rPPG visualization projection) | 0.07 | 0.06× |
| `sqi_model.tflite` (signal quality on 1×450) | 0.02 | 0.02× |
| `psd_model.tflite` (PSD HR estimator on 1×450) | 1.90 | 1.66× |
| `enet_b0_8.tflite` (ENet-B0 emotion, INT8) | 23.60 | **20.67×** |
| `mobileone_s0_gaze.tflite` (L2CS gaze, fp16) | 19.60 | **17.17×** |
| Face Landmarker · `face_detector` (track-loss only) | 1.18 | 1.03× |
| Face Landmarker · `landmarks_detector` (478 pts) | 5.89 | 5.16× |
| Face Landmarker · `blendshapes` (52) | 0.64 | 0.56× |
| Face Landmarker pipeline (steady-state) | 6.53 | **5.72×** |

## Effective per-second cost

What actually matters for your CPU budget is the **rate** at which each
model runs, not just the per-invocation cost. Every feature has its own
throttle:

| Feature | Config flag | Models | Rate | **× face_det / second** |
|---|---|---|---:|---:|
| **Face detection** | _always on_ | `blaze_face` | 30 Hz | **1.00** |
| **Heart rate (rPPG)** | _always on with HR pipeline_ | `model.tflite` + `proj.tflite` | 30 Hz | **2.15** |
| **SQI + HR estimate** | _auto with HR_ | `sqi_model` + `psd_model` | 2 Hz | **0.11** |
| **HRV (RMSSD + SDNN)** | _auto with HR_ | _pure JS (no model)_ | 2 Hz | **0** |
| **Head pose** | _auto with face detection_ | _pure JS PnP from BlazeFace keypoints_ | 30 Hz | **0** |
| **Emotion** | `loadModels({ emotion: true })` | `enet_b0_8` (INT8) | 2 Hz | **1.38** |
| **Face Landmarker** | `enableFaceLandmarker: true` | landmarks + blendshapes | 15 Hz | **2.85** |
| ↳ Eye state (open/close) | `enableEyeState` (requires FL) | _reads FL blendshape_ | 15 Hz | **0** |
| ↳ Mouth / speaking | `enableMouth` (requires FL) | _reads FL blendshape_ | 15 Hz | **0** |
| ↳ Gaze | `enableGaze` (requires FL) + `loadModels({ gaze: true })` | `mobileone_s0_gaze` | 5 Hz | **2.86** |

> **Free features.** Head pose, HRV, eye state and mouth all add **zero**
> model cost — they're either pure JS (head pose / HRV) or they read a
> value the Face Landmarker already produced (eye state / mouth).

## Common configurations

Total cost in `× face_det / second`:

| Configuration | Total | vs. all-in |
|---|---:|---:|
| HR + HRV only (FL off, no emotion) | 3.3× | −68% |
| HR + HRV + Emotion (FL off) | 4.6× | −55% |
| HR + Emotion + Eye state + Mouth (FL on, no gaze) | 7.5× | −27% |
| All-in (FL on, eye + mouth + gaze + emotion + HR) | 10.3× | 0% |

To translate to milliseconds on your target: measure your device's
single-frame face-detection time once, then multiply.

## Where each model runs

The SDK keeps the main thread free for camera + UI. Only face detection
runs on the main thread (and synchronously, in ~1 ms native /
~3–5 ms browser); every other model runs in its own dedicated worker so
slow inferences never block the camera loop.

```
Main thread
  └─ MediaPipe FaceDetector.detectForVideo()       BlazeFace, 1 model

inference.worker      (LiteRT WASM)
  ├─ model.tflite                                  rPPG main, 30 Hz
  └─ proj.tflite                                   visualization, 30 Hz

psd.worker            (LiteRT WASM)
  ├─ sqi_model.tflite                              ~2 Hz
  ├─ psd_model.tflite                              ~2 Hz
  └─ HRV pipeline (pure JS — RR detection / outliers / RMSSD+SDNN)

emotion.worker        (LiteRT WASM)                only loaded when emotion is on
  └─ enet_b0_8.tflite                              2 Hz throttle

gaze.worker           (LiteRT WASM)                only loaded when gaze is on
  └─ mobileone_s0_gaze.tflite                      5 Hz, gated by eye-open prob

face_landmarker.worker (MediaPipe Tasks Vision)    only loaded when FL is on
  ├─ face_detector.tflite                          on track-loss only
  ├─ face_landmarks_detector.tflite                15 Hz target, non-blocking
  └─ face_blendshapes.tflite                       15 Hz target
```

A worker that's not needed by your config is **never spawned** — set
`enableFaceLandmarker: false` and the entire `face_landmarker.worker`
process is skipped, along with the 3.8 MB model bundle download.

## Sizing tips

- **Want max battery life on a thin device** — turn off the Face
  Landmarker (`enableFaceLandmarker: false`) and gaze. You keep the
  highest-value features (HR + HRV + emotion + head pose) at ~4.6×
  baseline, less than half the cost of the all-in configuration.

- **Want eye state / mouth without paying for gaze** — keep
  `enableFaceLandmarker: true` but `enableGaze: false` (and
  `loadModels({ gaze: false })` to skip the 12 MB gaze model download).
  Eye state and mouth come for free out of the FL blendshape output.

- **The HR pipeline is non-negotiable** — rPPG is the SDK's reason for
  existing; turning it off is roughly equivalent to not using the SDK.
  Its 2.15× cost is already lower than emotion (1.38× per second despite
  20× per-inference cost — the throttle keeps it cheap).

- **Head pose and HRV are free.** Use them.

---

Next: [Configuration →](configuration.md)
