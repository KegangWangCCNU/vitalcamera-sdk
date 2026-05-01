# Changelog

All notable changes to `vitalcamera-sdk`.

## 0.5.0 — 2026-05-01

### Added
- **Per-user emotion calibration** via the new
  `emotionCalibration: { images: string[] }` constructor option on
  `BrowserAdapter`. Pass 2+ base64 face photos and the SDK computes a
  personalized baseline at `init()` time. The `'emotion'` event payload is
  unchanged (`{ emotion, probs, time, timestamp }`) — calibration is fully
  transparent to consumers. See
  [docs/configuration.md](docs/configuration.md#emotion-calibration).
- **Built-in default emotion baseline tuned for Asian faces** — applied
  automatically when the caller doesn't supply `emotionCalibration`. Fixes the
  long-standing Anger/Contempt/Disgust over-prediction at rest for many users.
- **`'eyestate'` event** — per-eye open/closed classification driven by the
  PINTO0309/OCEC TFLite model (~112 KB). Payload:
  `{ left: { prob, open }, right: { prob, open }, bothClosed, time, timestamp }`.
- **HRV: SDNN alongside RMSSD** — `'hrv'` payload now includes
  `{ rmssd, sdnn, meanRR, n, time, timestamp }`.
- **Examples**: `examples/calibration_test.html` end-to-end test page for the
  new calibration API.

### Changed
- **HRV pipeline refactor** — moved into the PSD worker, replaced cubic spline
  resampling with parabolic peak refinement on raw irregular samples, and
  collapsed the previous gate stack into a single compensating-pair RR filter.
  Faster, lower CPU, more accurate at the boundary of physiological RR ranges.
- **Worker source caching** — loader.js now sends `cache: 'no-cache'` so worker
  source updates take effect on hard reload during development.
- **Documentation** — refreshed `README.md`, `docs/getting-started.md`,
  `docs/api.md`, `docs/configuration.md`, and `docs/demo.html` for the new APIs;
  CDN-pinned demo bumped to `vitalcamera-sdk@0.5.0`.

### Internal
- VitalCamera core: `setEmotionBaseline` and `probeEmotion` are now
  underscore-prefixed (`_setEmotionBaseline` / `_probeEmotion`) — they are SDK
  internals consumed only by `BrowserAdapter` during init. Public API is
  unaffected.
- `examples/demo.html` migrated to local `../src/index.js` imports for dev
  iteration; published demo lives in `docs/demo.html` with CDN imports.

## 0.4.0

- HRV refactor groundwork: SDNN + tightened gating, asymmetric peak/valley
  cross-check, cascading-drop fix.

## 0.3.5

- Doc polish, demo simplification.

## Older

- See git log for history prior to 0.3.5.
