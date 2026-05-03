# Changelog

All notable changes to `vitalcamera-sdk`.

## 0.6.3 — 2026-05-02

### Fixed

- **Face bbox Kalman tuning reverted to 1e-2** (was bumped to 5e-2 in 0.6.0).
  The bump was justified by "Face Landmarker is more accurate than BlazeFace",
  but the FL bbox path doesn't go through this Kalman — it's computed from
  478-landmark min/max directly. So the bump only affected BlazeFace-driven
  `boxRaw`, making it visibly jumpier without any benefit. With this revert
  the FL-off / lite path matches the early SDK's smoother feel.

### Added

- `docs/demo_lite.html` — public CDN-pinned version of the lightweight
  (FL-off) demo, accessible without a local server.

## 0.6.2 — 2026-05-02

### Changed

- **`face` event payload now carries a ready-to-draw tight bbox.** Previously
  the `box` field was the raw BlazeFace short-range bbox — anatomically loose
  (includes forehead / hair / cheek slack), so consumers had to compute
  their own tighter geometry from Face Landmarker landmarks if they wanted
  a snug overlay. Now the SDK does it:
    - `box`     — preferred display bbox. When Face Landmarker is active and
                  has produced landmarks, it's the 478-point min/max with a
                  3 % lateral shrink (the temple / tragus points sit a bit
                  beyond the visible face surface in 2D projection, the
                  shrink snugs the box to the cheek line). When FL is off,
                  it's the BlazeFace bbox unchanged.
    - `boxRaw`  — **new** field, always the BlazeFace bbox (Kalman-smoothed).
                  Use this when you specifically want the raw detector output.

  No code change required for typical consumers — `box` becomes tighter
  automatically when FL is enabled. The `examples/demo.html` overlay code
  was simplified to just `strokeRect(box.x, box.y, box.w, box.h)`; the
  per-frame landmark-min/max math previously living in the demo is now
  centralised in the SDK.

## 0.6.1 — 2026-05-02

### Added

- **`enableFaceLandmarker` master switch** (default `true`). The Face
  Landmarker bundle is heavy (~3.8 MB, 18–50 ms / inference). Setting this
  to `false` switches the SDK to a lightweight BlazeFace-only path:
  `rPPG / HRV / emotion / head-pose` only, no `eyestate` / no `mouth` /
  no `gaze`.
- **Face-Landmarker dependency group**: `enableEyeState`, `enableMouth`,
  `enableGaze` all require `enableFaceLandmarker`. If any sub-feature is
  enabled while the master is off, the SDK forces it off and warns once.
- **`enableMouth` config knob** (default `true`) — independently mutes the
  `'mouth'` event without disabling the underlying Face Landmarker (eyestate
  / gaze can stay on).
- **`loadModels({ faceLandmarker })`** — control whether the 3.8 MB
  `face_landmarker.task` bundle is pre-fetched. Pair with
  `enableFaceLandmarker: false` for a true minimal build.

### Removed

- **OCEC eye-state pipeline**. `eye_state.worker.js` is no longer registered
  in `_initWorkers`, the OCEC eye-crop generation is gone from the per-frame
  loop, and `models/ocec_p.tflite` is dropped from `loadModels()` defaults
  (`MODEL_FILES.eyeState` removed; the file itself stays in `models/` for
  legacy callers but isn't fetched). Eye state is now Face-Landmarker-only.
- **`loadModels({ eyeState })` option** removed (was a passthrough for the
  OCEC bundle).

### Internal

- Adapter: stripped the OCEC eye-state batch generation, motion-gate
  fast-path 0.6 injection, eye-keypoint Kalman filters, and the
  `_resetEyeBaseline` 1 s grace lock — all dead code now that the FL path
  drives eyestate.
- `_onEyeStateResult` simplified: only handles the `{ leftProb, rightProb,
  time }` shape that the FL adapter feeds; legacy OCEC `data.probs` branch
  removed.

## 0.6.0 — 2026-05-02

### Added

- **MediaPipe Face Landmarker integration.** A new dedicated worker
  (`src/workers/face_landmarker.worker.js`) hosts the `face_landmarker.task`
  bundle (3.8 MB fp16, 478 landmarks + 52 blendshapes) and runs at a 15 fps
  target with non-blocking semantics — slow devices that take 200 ms per
  inference simply throttle down to 5 fps without backing up the camera loop.
  Output is dispatched on existing/new events (see below).
- **`'mouth'` event** — derived from the `jawOpen` blendshape. Payload:
  `{ jawOpen, jawStd, speaking, time, timestamp }`. `speaking` is a boolean
  inferred from the rolling 1 s standard deviation of `jawOpen`: high std =
  jaw is articulating (talking / chewing), low std = silent or sustained
  yawn / surprise.
- **Distribution-based emotion calibration.** `emotionCalibration` now
  accepts a pre-computed 8-vector via
  `emotionCalibration: { baseline: [n0, …, n7] }` alongside the existing
  `images: […]` path. Useful for shipping a calibration captured offline
  for a known target population.
- **Dynamic emotion calibration (EMA).**
  `emotionCalibration: { dynamic: { halfLifeMs: 5000 } }` enables continuous
  baseline drift: each inference EMA-folds raw logits into the baseline.
  A sustained expression slides the baseline that direction, calibration
  re-centres on Neutral, so the visible signal becomes "deviation from your
  typical expression" instead of absolute label.
  All three options can stack (`images` ∨ `baseline`, then `dynamic` on top).
- **HRV invalidation** — the `'hrv'` event now always fires on cadence with
  `rmssd === null` and a `reject` reason when the window can't produce a
  trustworthy estimate. Possible reasons: `low_sqi`, `warming_up`,
  `too_few_samples`, `too_few_peaks`, `rr_below_phys_min`,
  `too_few_after_outlier_filter`, `too_few_after_compensating_pairs`,
  **`high_rejection_rate` (new)** — when > 50 % of physiologically-plausible
  RR intervals get dropped as outliers. Demo clears the panel display on
  null `rmssd` so a stale value never lingers.
- **Examples**: `face_landmarker_test.html`,
  `face_landmarker_worker_test.html`, `bench_eye.html`. The worker-based
  test demonstrates the classic-worker + dynamic ESM-import trick that
  sidesteps the `importScripts` / module-worker incompatibility in
  `@mediapipe/tasks-vision`.

### Changed

- **Eye-state source switched from OCEC to Face Landmarker blendshapes.**
  `eyeBlinkLeft` / `eyeBlinkRight` (`1 - blink` → `prob_open`) feed the
  existing `'eyestate'` event. OCEC + BlazeFace eye-keypoint pipeline still
  compiled in as fallback for the first 1–2 s while Face Landmarker boots,
  then dormant. Resolves the BlazeFace eye-keypoint drift / per-frame jitter
  that plagued the previous tight-eye-crop pipeline.
- **Gaze face crop now uses Face Landmarker landmark min/max** (with the
  existing 0.2 padding) for a temporally-stable, anatomically-tight crop.
  Falls back to BlazeFace bbox during FL boot. Eyes land at consistent
  in-crop coordinates → L2CS-Net output is far less jittery.
- **Calibration weights**: `W_BASELINE` 0.3 → 0.2, `W_CALIBRATED` 0.7 → 0.8.
  Baseline still moderates extreme outputs but defers more to the
  KL-blended distribution.
- **`gazeEyeOpenGateProb` default** 0.7 → 0.6. Squinting / partial-blink
  frames now feed gaze instead of being skipped.
- **`EYE_MOTION_NEUTRAL_PROB`** 0.6 → 0.55 (so it stays strictly below the
  new gaze gate).
- **Face / eye-box Kalman `processNoise`** 1e-2 → 5e-2 (5× more responsive)
  — bbox tracks fast head motion instead of trailing.

### Demo

- Vertical 8-column emotion bars (was 8 horizontal rows) — far more compact,
  easier to scan instantaneous distribution.
- New mouth panel beside the eyes panel: jawOpen bar + silent/speaking pill.
- Face bbox derived from Face Landmarker 478-point min/max with 3% lateral
  shrink (was BlazeFace bbox); eye-box overlay removed.
- Dynamic emotion calibration enabled (`halfLifeMs: 5000`) so the demo's
  emotion display drifts with sustained expressions.

### Internal

- Added `_setEmotionDynamic(halfLifeMs)` to `VitalCamera` (mirrors
  `_setEmotionBaseline`). Both stay underscore-prefixed; the public surface
  is the `emotionCalibration` config object.
- `psd.worker.js` HRV pipeline emits a `dropRate` debug field and bails out
  on `dropRate > 0.5`.

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
