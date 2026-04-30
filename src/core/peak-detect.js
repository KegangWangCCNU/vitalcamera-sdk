/**
 * Real-time peak detector for BVP/PPG signals.
 * Stateful class — feed samples one at a time via process().
 *
 * @module core/peak-detect
 */

export default class RealtimePeakDetector {
    /**
     * @param {Object} [opts]
     * @param {number} [opts.windowSize=90] - moving window size in samples
     * @param {number} [opts.minIbiMs=333]  - minimum inter-beat interval (ms)
     * @param {number} [opts.maxIbiMs=2000] - maximum inter-beat interval (ms)
     */
    constructor({ windowSize = 90, minIbiMs = 333, maxIbiMs = 2000 } = {}) {
        this.windowSize = windowSize;
        this.minIbiMs = minIbiMs;
        this.maxIbiMs = maxIbiMs;
        this.ring = [];
        this.movSum = 0;
        this.state = 0;
        this.peakVal = -Infinity;
        this.peakTime = 0;
        this.lastConfirmedPeak = 0;
        this.recentIBIs = [];
    }

    /**
     * Process a new sample.
     * @param {number} timestamp - sample timestamp in ms
     * @param {number} value - signal amplitude
     * @param {number|null} [expectedHR=null] - optional expected heart rate for validation
     * @returns {{ ibi: number, peakTime: number } | null} beat info if detected
     */
    process(timestamp, value, expectedHR = null) {
        this.ring.push(value);
        this.movSum += value;
        if (this.ring.length > this.windowSize) {
            this.movSum -= this.ring.shift();
        }
        if (this.ring.length < 30) return null;

        const movAvg = this.movSum / this.ring.length;
        let sumSq = 0;
        for (let i = 0; i < this.ring.length; i++) {
            const d = this.ring[i] - movAvg;
            sumSq += d * d;
        }
        const std = Math.sqrt(sumSq / this.ring.length);
        const threshold = movAvg + std * 0.3;

        if (value > threshold) {
            if (value > this.peakVal) {
                this.peakVal = value;
                this.peakTime = timestamp;
            }
            this.state = 1;
        } else if (this.state === 1) {
            this.state = 0;
            const peakTime = this.peakTime;
            this.peakVal = -Infinity;

            if (this.lastConfirmedPeak > 0) {
                const ibi = peakTime - this.lastConfirmedPeak;
                if (ibi >= this.minIbiMs && ibi <= this.maxIbiMs) {
                    let valid = true;
                    if (expectedHR !== null) {
                        const expectedIBI = 60000 / Math.max(40, Math.min(expectedHR, 180));
                        const ratio = ibi / expectedIBI;
                        valid = ratio > 0.6 && ratio < 1.5;
                    }
                    if (valid) {
                        this.recentIBIs.push(ibi);
                        if (this.recentIBIs.length > 5) this.recentIBIs.shift();
                        this.lastConfirmedPeak = peakTime;
                        return { ibi, peakTime };
                    }
                }
            }
            this.lastConfirmedPeak = peakTime;
        }
        return null;
    }

    /** Reset all internal state. */
    reset() {
        this.ring = [];
        this.movSum = 0;
        this.state = 0;
        this.peakVal = -Infinity;
        this.peakTime = 0;
        this.lastConfirmedPeak = 0;
        this.recentIBIs = [];
    }
}
