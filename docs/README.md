# VitalCamera SDK

Browser-based real-time physiological sensing — extract **heart rate**, **HRV**, **emotion**, **gaze**, **eye state**, **mouth (jawOpen + speaking)**, and **head pose** from a standard webcam. No wearables, no server, no data upload.

## Why VitalCamera?

- **100% browser-side** — All inference runs locally via LiteRT (TFLite WASM). Zero backend dependency.
- **Privacy by design** — Video never leaves the device. See our [LICENSE](https://github.com/KegangWangCCNU/vitalcamera-sdk/blob/main/LICENSE) for details.
- **Ultra-lightweight** — Core rPPG model is just 4 MB with ~5 ms inference latency.
- **One CDN import** — No build tools, no bundler, no npm install required (though npm works too).

## What can it do?

| Feature | Output | Update Rate |
|---------|--------|-------------|
| Heart Rate | BPM value | 2/s |
| HRV | RMSSD + SDNN (ms) | 1/s |
| Emotion | 8-class probs (image / distribution / dynamic-EMA calibration) | 2/s |
| Gaze | Yaw & Pitch (degrees), Face-Landmarker-aligned crop | 5/s |
| Eye State | Per-eye P(open) from MediaPipe `eyeBlink` blendshapes | 15/s |
| Mouth | jawOpen + speaking heuristic (rolling-variance) | 15/s |
| Head Pose | Yaw, Pitch, Roll | realtime |

## Quick links

- **[Getting Started →](getting-started.md)** — From zero to heart rate in 5 minutes
- **[API Reference →](api.md)** — Full class and event documentation
- **[Configuration →](configuration.md)** — All options explained
- **[Performance →](performance.md)** — Per-model cost table + sizing guide
- **[Live Demo →](https://kegangwangccnu.github.io/vitalcamera-sdk/examples/demo.html)** — Try it now in your browser
- **[npm package →](https://www.npmjs.com/package/vitalcamera-sdk)** — `npm install vitalcamera-sdk`
- **[GitHub →](https://github.com/KegangWangCCNU/vitalcamera-sdk)** — Source code
