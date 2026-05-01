/**
 * @file loader.js
 * @description Worker loader — creates Web Workers from Blob URLs.
 *
 * When the SDK is loaded from a CDN (cross-origin), `new Worker('path.js')`
 * fails because browsers block cross-origin worker scripts. This loader
 * fetches the worker source code via `fetch()` (which allows cross-origin),
 * then wraps it in a Blob URL so the Worker constructor sees a same-origin URL.
 *
 * Fallback: if `basePath` is explicitly provided, uses traditional file-based
 * workers (useful for local development without a server).
 *
 * @module workers/loader
 */

/** Map of worker name → filename. */
const WORKER_FILES = {
    inference: 'inference.worker.js',
    psd:       'psd.worker.js',
    emotion:   'emotion.worker.js',
    gaze:      'gaze.worker.js',
    eye_state: 'eye_state.worker.js',
    plot:      'plot.worker.js',
};

/** Derive the workers directory from this module's own URL. */
const WORKERS_DIR = new URL('./', import.meta.url).href;

/** Cache fetched worker source code to avoid redundant network requests. */
const _sourceCache = new Map();

/**
 * Create a Web Worker by name.
 *
 * @param {string} name   Worker name: 'inference' | 'psd' | 'emotion' | 'gaze' | 'eye_state' | 'plot'
 * @param {string} [basePath]  Optional explicit path to worker files.
 *        If provided, uses traditional `new Worker(basePath + file)`.
 *        If omitted, fetches the worker source from the SDK's own URL
 *        and creates a Blob URL worker (works cross-origin).
 * @returns {Promise<Worker>}
 */
export async function createWorker(name, basePath) {
    const file = WORKER_FILES[name];
    if (!file) throw new Error(`[VitalCamera] Unknown worker: "${name}"`);

    // ── Explicit basePath → traditional file-based worker ──
    if (basePath) {
        return new Worker(basePath + file);
    }

    // ── Auto mode → fetch source + Blob URL ──
    const url = WORKERS_DIR + file;

    let code = _sourceCache.get(name);
    if (!code) {
        // cache: 'no-cache' forces revalidation — without this, browsers can
        // hold on to stale worker source even across hard refreshes (the worker
        // is loaded via fetch + Blob URL, not <script>, so Ctrl+Shift+R doesn't
        // bust the cache for it).
        const resp = await fetch(url, { cache: 'no-cache' });
        if (!resp.ok) {
            throw new Error(
                `[VitalCamera] Failed to load worker "${name}" from ${url} (${resp.status})`
            );
        }
        code = await resp.text();
        _sourceCache.set(name, code);
    }

    const blob = new Blob([code], { type: 'text/javascript' });
    return new Worker(URL.createObjectURL(blob));
}
