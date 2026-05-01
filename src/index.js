/**
 * @file index.js
 * @description VitalCamera SDK entry point.
 *
 * Re-exports all public classes, functions, and utilities from the SDK.
 *
 * @module vitalcamera-sdk
 */

export { default as VitalCamera } from './core/vitalcamera.js';
export { default as BrowserAdapter } from './adapter/browser.js';
export { default as KalmanFilter1D } from './utils/kalman.js';
export { softmax, clamp } from './utils/math.js';
export { estimateHeadPose } from './core/headpose.js';
export { default as RealtimePeakDetector } from './core/peak-detect.js';
export {
    detectBvpPeaks, detectBvpValleys,
    rejectAbnormalPeaks,
    refinePeaksParabolic, refineValleysParabolic,
    quotientFilterRR, madFilterRR, computeHrvMetrics
} from './core/hrv.js';

// Re-export convenience statics from BrowserAdapter
import _BrowserAdapter from './adapter/browser.js';

/**
 * Convenience re-export of BrowserAdapter.loadModels.
 * @see BrowserAdapter.loadModels
 */
export const loadModels = _BrowserAdapter.loadModels.bind(_BrowserAdapter);

/** Emotion class labels (8 classes). */
export const EMOTION_LABELS = _BrowserAdapter.EMOTION_LABELS;

/** Emoji for each emotion class. */
export const EMOTION_EMOJIS = _BrowserAdapter.EMOTION_EMOJIS;

/** Color for each emotion class. */
export const EMOTION_COLORS = _BrowserAdapter.EMOTION_COLORS;
