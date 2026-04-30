/**
 * Numerically-stable softmax over an array of logits.
 * Returns a Float32Array of probabilities that sum to 1.
 */
export function softmax(logits) {
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > maxVal) maxVal = logits[i];
    }
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        exps[i] = Math.exp(logits[i] - maxVal);
        sum += exps[i];
    }
    for (let i = 0; i < logits.length; i++) exps[i] /= sum;
    return exps;
}

/**
 * Clamp a value to the [min, max] range.
 */
export function clamp(value, min, max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}
