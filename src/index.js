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
    buildCubicSpline, evalSpline, interpolateBvp,
    detectBvpPeaks, rejectAbnormalPeaks,
    quotientFilterRR, madFilterRR, computeHrvMetrics
} from './core/hrv.js';
