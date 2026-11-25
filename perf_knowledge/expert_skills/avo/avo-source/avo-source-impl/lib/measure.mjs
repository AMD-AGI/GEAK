// Pure, testable measurement-rigor helpers for AVO (no I/O, no runtime globals).
//
// Over a long single lineage, benchmark noise COMPOUNDS: a noisy "win" commits and
// every later step branches from a false best. This module supplies the robust
// statistics the verifier + commit gate use to (a) aggregate repeats with outlier
// rejection, and (b) size the commit gate's significance margin to the MEASURED
// noise instead of a fixed constant. Pairs with lib/commit_gate.mjs (which consumes
// the per-candidate coefficient of variation).
//
// Unit-tested by tests/measure.test.mjs.

const finite = (xs) => (xs || []).map(Number).filter((x) => Number.isFinite(x));

export function median(xs) {
  const v = finite(xs).slice().sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

export function mean(xs) {
  const v = finite(xs);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

/** Population standard deviation. */
export function stddev(xs) {
  const v = finite(xs);
  if (v.length < 2) return v.length ? 0 : null;
  const m = mean(v);
  return Math.sqrt(v.reduce((a, x) => a + (x - m) * (x - m), 0) / v.length);
}

/** Median absolute deviation (robust spread). */
export function mad(xs) {
  const v = finite(xs);
  if (!v.length) return null;
  const med = median(v);
  return median(v.map((x) => Math.abs(x - med)));
}

/** Coefficient of variation (stddev/mean) — unitless noise level. null if undefined. */
export function cv(xs) {
  const m = mean(xs);
  const s = stddev(xs);
  if (m == null || s == null || m === 0) return null;
  return Math.abs(s / m);
}

/**
 * Drop outliers using the MAD rule: keep points within `k` MADs of the median.
 * Robust to a few spurious slow/fast runs (throttling, contention). Falls back to
 * the raw set when MAD is 0 (all-equal) or too few points.
 */
export function dropOutliers(xs, k = 3) {
  const v = finite(xs);
  if (v.length < 4) return v;
  const med = median(v);
  const m = mad(v);
  if (!m) return v;
  const kept = v.filter((x) => Math.abs(x - med) <= k * m);
  return kept.length ? kept : v;
}

/**
 * Robust summary of repeated latency samples for ONE case: outlier-filtered median
 * plus the noise level (cv) the gate needs. `n_used`/`n_raw` expose how many repeats
 * survived filtering (a low ratio is itself a noise warning).
 *
 * @param {number[]} samples  per-repeat latencies (ms)
 * @param {object}  opts      {k=3 (MAD multiplier)}
 * @returns {{median:number|null, cv:number|null, n_used:number, n_raw:number}}
 */
export function summarizeRepeats(samples, opts = {}) {
  const k = opts.k != null ? opts.k : 3;
  const raw = finite(samples);
  const kept = dropOutliers(raw, k);
  return { median: median(kept), cv: cv(kept), n_used: kept.length, n_raw: raw.length };
}

/**
 * Aggregate per-case cv into one representative noise level for the candidate
 * (max is conservative — the noisiest shape governs the gate). null when none known.
 */
export function aggregateCV(cvs) {
  const v = finite(cvs);
  return v.length ? Math.max(...v) : null;
}

export const _internals = { finite };
