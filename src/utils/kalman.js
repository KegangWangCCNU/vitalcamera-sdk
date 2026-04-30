/**
 * 1-D Kalman filter for smoothing scalar time-series signals.
 */
class KalmanFilter1D {
    constructor(initialValue, processNoise = 1e-2, measurementNoise = 5e-1) {
        this.x = initialValue;
        this.p = 1.0;
        this.q = processNoise;
        this.r = measurementNoise;
    }

    /**
     * Prediction step only — grows uncertainty without incorporating a measurement.
     * Call this every frame for sub-sampled signals (e.g. emotion @ 2 Hz, gaze @ 5 Hz)
     * so the filter tracks passage of time between actual measurements.
     * @returns {number} Current state estimate (unchanged).
     */
    predict() {
        this.p += this.q;
        return this.x;
    }

    /**
     * Full predict + update cycle. Call when a new measurement is available.
     * @param {number} measurement
     * @returns {number} Updated state estimate.
     */
    update(measurement) {
        const p_pred = this.p + this.q;
        const k = p_pred / (p_pred + this.r);
        this.x = this.x + k * (measurement - this.x);
        this.p = (1 - k) * p_pred;
        return this.x;
    }
}

export default KalmanFilter1D;
