/**
 * @file plot.worker.js
 * @description OffscreenCanvas rendering Web Worker for the VitalCamera SDK.
 *
 * Handles all chart/waveform drawing off the main thread using OffscreenCanvas.
 * Three chart types are supported:
 *
 *   1. BVP waveform  - Scan-line style real-time BVP signal display with a
 *                      circular buffer and moving cursor gap.
 *   2. PSD spectrum  - Scatter plot of power spectral density vs frequency,
 *                      with highlighted peak frequency.
 *   3. HR trend      - Line chart of heart rate over time (up to 5 minutes),
 *                      with solid lines for valid readings and dashed for invalid.
 *
 * Message protocol (type / payload):
 *   -> init        { bvpCanvas?, bvpWidth?, bvpHeight?, psdCanvas?, psdWidth?,
 *                    psdHeight?, trendCanvas?, trendWidth?, trendHeight?, dpr }
 *                  Receive transferred OffscreenCanvases and set up contexts
 *   -> bvp_data    number (single BVP sample value)
 *   -> psd_data    { psd, freq, peakIdx }
 *   -> trend_data  { hr, valid }
 *   -> resize_trend { width, height, dpr }
 */

/* ── BVP chart state ── */
let bvpCtx = null;
let bvpWidth = 0;
let bvpHeight = 0;
const BVP_BUFFER_LEN = 225;
let bvpBuffer = new Float32Array(BVP_BUFFER_LEN);
let bvpCursor = 0;
const BVP_Y_MIN = -1.5;
const BVP_Y_MAX = 2.0;
const SCAN_GAP = 30;

/* ── PSD chart state ── */
let psdCtx = null;
let psdWidth = 0;
let psdHeight = 0;
const PSD_Y_MIN = 0;
const PSD_Y_MAX = 6000;
const PSD_X_MIN = 0.5;
const PSD_X_MAX = 3.0;

/* ── HR trend chart state ── */
let trendCtx = null;
let trendWidth = 0;
let trendHeight = 0;
const TREND_MAX_POINTS = 300;
let trendBuffer = [];
const TREND_Y_MIN = 40;
const TREND_Y_MAX = 180;

/** Device pixel ratio for HiDPI rendering. */
let dpr = 1;

/* ── Message handler ── */
self.onmessage = (e) => {
    const { type, payload } = e.data;

    if (type === 'init') {
        dpr = payload.dpr || 1;

        // BVP canvas (optional)
        if (payload.bvpCanvas) {
            const bCanvas = payload.bvpCanvas;
            bvpWidth = payload.bvpWidth;
            bvpHeight = payload.bvpHeight;
            bvpCtx = bCanvas.getContext('2d');
            bvpCtx.scale(dpr, dpr);
            bvpBuffer.fill(0);
        }

        // PSD canvas (optional)
        if (payload.psdCanvas) {
            const pCanvas = payload.psdCanvas;
            psdWidth = payload.psdWidth;
            psdHeight = payload.psdHeight;
            psdCtx = pCanvas.getContext('2d');
            psdCtx.scale(dpr, dpr);
        }

        // Trend canvas (optional)
        if (payload.trendCanvas) {
            const tCanvas = payload.trendCanvas;
            trendWidth = payload.trendWidth;
            trendHeight = payload.trendHeight;
            trendCtx = tCanvas.getContext('2d');
            trendCtx.scale(dpr, dpr);
        }

        // Initial draw
        if (bvpCtx) drawBVP();
        if (psdCtx) drawPSD([], [], 0);
        if (trendCtx) drawTrend();

    } else if (type === 'bvp_data') {
        bvpBuffer[bvpCursor] = payload;
        bvpCursor = (bvpCursor + 1) % BVP_BUFFER_LEN;
        if (bvpCtx) drawBVP();

    } else if (type === 'psd_data') {
        const { psd, freq, peakIdx } = payload;
        if (psdCtx) drawPSD(psd, freq, peakIdx);

    } else if (type === 'resize_trend') {
        const { width, height, dpr: newDpr } = payload;
        if (trendCtx) {
            dpr = newDpr;
            trendWidth = width;
            trendHeight = height;
            trendCtx.canvas.width = width * dpr;
            trendCtx.canvas.height = height * dpr;
            trendCtx.scale(dpr, dpr);
            drawTrend();
        }

    } else if (type === 'trend_data') {
        const { hr, valid } = payload;
        trendBuffer.push({ hr, valid });
        if (trendBuffer.length > TREND_MAX_POINTS) {
            trendBuffer.shift();
        }
        if (trendCtx) drawTrend();
    }
};

/* ──────────────────────────────────────────────────────
 * BVP waveform — scan-line style circular buffer display
 * ────────────────────────────────────────────────────── */

function drawBVP() {
    if (!bvpCtx) return;

    bvpCtx.fillStyle = "#eee9e0";
    bvpCtx.fillRect(0, 0, bvpWidth, bvpHeight);

    bvpCtx.beginPath();
    bvpCtx.strokeStyle = "rgba(0,0,0,0.06)";
    bvpCtx.lineWidth = 0.5;
    bvpCtx.lineCap = "butt";
    bvpCtx.lineJoin = "miter";
    const stepX = bvpWidth / 10;
    for (let x = 0; x < bvpWidth; x += stepX) {
        bvpCtx.moveTo(x, 0);
        bvpCtx.lineTo(x, bvpHeight);
    }
    bvpCtx.stroke();

    const range = BVP_Y_MAX - BVP_Y_MIN;
    const padding = 10;
    const drawH = bvpHeight - 2 * padding;
    const stepW = bvpWidth / BVP_BUFFER_LEN;

    bvpCtx.lineWidth = 2;
    bvpCtx.lineJoin = "round";
    bvpCtx.lineCap = "round";
    bvpCtx.strokeStyle = "#c96442";
    bvpCtx.beginPath();

    const cursorX = bvpCursor * stepW;
    const gapStart = cursorX;
    const gapEnd = cursorX + SCAN_GAP;

    let needMove = true;

    for (let i = 0; i < BVP_BUFFER_LEN; i++) {
        let val = bvpBuffer[i];
        if (val < BVP_Y_MIN) val = BVP_Y_MIN;
        if (val > BVP_Y_MAX) val = BVP_Y_MAX;

        const norm = (val - BVP_Y_MIN) / range;
        const x = i * stepW;
        const y = bvpHeight - padding - (norm * drawH);

        let inGap = false;
        if (x >= gapStart && x < gapEnd) {
            inGap = true;
        } else if (gapEnd > bvpWidth && x < (gapEnd - bvpWidth)) {
            inGap = true;
        }

        if (inGap) {
            needMove = true;
            continue;
        }

        if (needMove) {
            bvpCtx.moveTo(x, y);
            needMove = false;
        } else {
            bvpCtx.lineTo(x, y);
        }
    }
    bvpCtx.stroke();

    bvpCtx.fillStyle = "#eee9e0";
    if (gapEnd <= bvpWidth) {
        bvpCtx.fillRect(gapStart, 0, SCAN_GAP, bvpHeight);
    } else {
        bvpCtx.fillRect(gapStart, 0, bvpWidth - gapStart, bvpHeight);
        bvpCtx.fillRect(0, 0, gapEnd - bvpWidth, bvpHeight);
    }

    bvpCtx.beginPath();
    bvpCtx.lineWidth = 1;
    bvpCtx.lineCap = "butt";
    const sharpX = Math.floor(cursorX) + 0.5;
    bvpCtx.moveTo(sharpX, 0);
    bvpCtx.lineTo(sharpX, bvpHeight);
    bvpCtx.strokeStyle = "rgba(201, 100, 66, 0.3)";
    bvpCtx.stroke();
}

/* ──────────────────────────────────────────────────────
 * PSD spectrum — scatter plot of power vs frequency
 * ────────────────────────────────────────────────────── */

function drawPSD(psdData, freqData, peakIdx) {
    if (!psdCtx) return;

    psdCtx.fillStyle = "#eee9e0";
    psdCtx.fillRect(0, 0, psdWidth, psdHeight);

    psdCtx.strokeStyle = "rgba(0,0,0,0.08)";
    psdCtx.lineWidth = 0.5;
    psdCtx.imageSmoothingEnabled = false;

    psdCtx.font = "10px -apple-system, sans-serif";
    psdCtx.fillStyle = "#9a9a9a";
    psdCtx.textAlign = "right";

    const paddingBottom = 20;
    const paddingLeft = 10;
    const paddingRight = 10;
    const paddingTop = 10;

    const graphW = psdWidth - paddingLeft - paddingRight;
    const graphH = psdHeight - paddingBottom - paddingTop;

    const ySteps = [0, 2000, 4000, 6000];
    for (let val of ySteps) {
        const norm = (val - PSD_Y_MIN) / (PSD_Y_MAX - PSD_Y_MIN);
        const y = Math.floor(psdHeight - paddingBottom - (norm * graphH)) + 0.5;
        psdCtx.beginPath();
        psdCtx.moveTo(paddingLeft, y);
        psdCtx.lineTo(psdWidth - paddingRight, y);
        psdCtx.stroke();
    }

    psdCtx.textAlign = "center";
    for (let f = 0.5; f <= 3.0; f += 0.5) {
        const normX = (f - PSD_X_MIN) / (PSD_X_MAX - PSD_X_MIN);
        const x = Math.floor(paddingLeft + (normX * graphW)) + 0.5;
        psdCtx.beginPath();
        psdCtx.moveTo(x, psdHeight - paddingBottom);
        psdCtx.lineTo(x, psdHeight - paddingBottom + 4);
        psdCtx.stroke();
        psdCtx.fillText(f.toFixed(1), x, psdHeight - 5);
    }

    if (!psdData || psdData.length === 0) return;

    psdCtx.fillStyle = "#5a8fa8";
    for (let i = 0; i < psdData.length; i++) {
        const f = freqData[i];
        const p = psdData[i];
        if (f < PSD_X_MIN || f > PSD_X_MAX) continue;

        const normX = (f - PSD_X_MIN) / (PSD_X_MAX - PSD_X_MIN);
        let normY = (p - PSD_Y_MIN) / (PSD_Y_MAX - PSD_Y_MIN);
        if (normY > 1) normY = 1;
        if (normY < 0) normY = 0;

        const x = Math.floor(paddingLeft + normX * graphW);
        const y = Math.floor(psdHeight - paddingBottom - normY * graphH);
        psdCtx.fillRect(x, y, 2, 2);
    }

    if (peakIdx !== undefined && peakIdx >= 0 && peakIdx < freqData.length) {
        const peakFreq = freqData[peakIdx];
        const peakVal = psdData[peakIdx];

        if (peakFreq >= PSD_X_MIN && peakFreq <= PSD_X_MAX) {
            const normX = (peakFreq - PSD_X_MIN) / (PSD_X_MAX - PSD_X_MIN);
            let normY = (peakVal - PSD_Y_MIN) / (PSD_Y_MAX - PSD_Y_MIN);
            if (normY > 1) normY = 1;

            const px = Math.floor(paddingLeft + normX * graphW);
            const py = Math.floor(psdHeight - paddingBottom - normY * graphH);

            const pixelSize = 6;
            psdCtx.fillStyle = "#c96442";
            psdCtx.fillRect(px - pixelSize / 2, py - pixelSize / 2, pixelSize, pixelSize);

            const text = `${peakFreq.toFixed(2)} Hz`;
            psdCtx.font = "500 13px -apple-system, sans-serif";
            const textW = psdCtx.measureText(text).width;

            psdCtx.fillStyle = "#1a1a1a";
            psdCtx.fillRect(px - textW / 2 - 4, py - 24, textW + 8, 18);

            psdCtx.fillStyle = "#ffffff";
            psdCtx.fillText(text, px, py - 11);
        }
    }
}

/* ──────────────────────────────────────────────────────
 * HR trend — line chart over 5-minute window
 * ────────────────────────────────────────────────────── */

function drawTrend() {
    if (!trendCtx) return;

    trendCtx.imageSmoothingEnabled = false;

    trendCtx.fillStyle = "#eee9e0";
    trendCtx.fillRect(0, 0, trendWidth, trendHeight);

    const padding = 6;
    const paddingBottom = 16;
    const paddingLeft = 4;
    const paddingRight = 16;
    const graphW = trendWidth - paddingLeft - paddingRight;
    const graphH = trendHeight - padding - paddingBottom;

    // Horizontal grid lines only (subtle)
    trendCtx.lineWidth = 0.5;
    trendCtx.strokeStyle = "rgba(0,0,0,0.06)";
    const ySteps = [60, 100, 140];
    trendCtx.beginPath();
    for (let val of ySteps) {
        const norm = (val - TREND_Y_MIN) / (TREND_Y_MAX - TREND_Y_MIN);
        const y = Math.floor(trendHeight - paddingBottom - (norm * graphH)) + 0.5;
        trendCtx.moveTo(paddingLeft, y);
        trendCtx.lineTo(trendWidth - paddingRight, y);
    }
    trendCtx.stroke();

    // X-axis: fixed 0 to 5 min (TREND_MAX_POINTS = 300s)
    const stepX = graphW / (TREND_MAX_POINTS - 1);
    trendCtx.font = "10px -apple-system, sans-serif";
    trendCtx.fillStyle = "#b5b0a8";
    trendCtx.textAlign = "center";
    trendCtx.textBaseline = "top";

    // Labels every 60s: 0, 1min, 2min, 3min, 4min, 5min
    for (let sec = 0; sec <= TREND_MAX_POINTS; sec += 60) {
        const x = paddingLeft + sec * stepX;
        const label = sec === 0 ? "0" : (sec / 60) + "min";
        trendCtx.fillText(label, x, trendHeight - paddingBottom + 3);
    }

    if (trendBuffer.length < 2) return;

    // Draw the line — data fills left to right on fixed x scale
    trendCtx.lineWidth = 2;
    trendCtx.lineJoin = "round";

    for (let i = 1; i < trendBuffer.length; i++) {
        const p1 = trendBuffer[i - 1];
        const p2 = trendBuffer[i];

        const x1 = Math.floor(paddingLeft + (i - 1) * stepX) + 0.5;
        const x2 = Math.floor(paddingLeft + i * stepX) + 0.5;

        let y1 = trendHeight - paddingBottom - ((p1.hr - TREND_Y_MIN) / (TREND_Y_MAX - TREND_Y_MIN) * graphH);
        let y2 = trendHeight - paddingBottom - ((p2.hr - TREND_Y_MIN) / (TREND_Y_MAX - TREND_Y_MIN) * graphH);

        y1 = Math.floor(Math.max(padding, Math.min(trendHeight - paddingBottom, y1))) + 0.5;
        y2 = Math.floor(Math.max(padding, Math.min(trendHeight - paddingBottom, y2))) + 0.5;

        trendCtx.beginPath();

        if (p2.valid && p1.valid) {
            trendCtx.setLineDash([]);
            trendCtx.lineCap = "round";
            trendCtx.strokeStyle = "#c96442";
        } else {
            trendCtx.setLineDash([6, 4]);
            trendCtx.lineCap = "butt";
            trendCtx.strokeStyle = "#d0cbc2";
        }

        trendCtx.moveTo(x1, y1);
        trendCtx.lineTo(x2, y2);
        trendCtx.stroke();
    }

    trendCtx.setLineDash([]);
}
