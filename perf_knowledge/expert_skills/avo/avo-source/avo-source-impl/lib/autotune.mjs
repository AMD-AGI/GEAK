// Pure, testable autotune-sweep logic for AVO (no I/O, no runtime globals).
//
// The last ~20% toward the roofline is rarely one inspired edit — it is CO-TUNING a
// coupled config space (tile sizes x waves x unroll x num_stages x precision). A
// single "one focused change" step can't search that systematically. This module is
// the single source of truth for a BOUNDED best-of-N config sweep:
//   - expandConfigSpace: cartesian product of a param grid, deterministically capped;
//   - selectBestConfig:  pick the winner from measured results (verified geomean, then
//                        closeness to roofline as the tie-break);
//   - configKey:         stable identity for dedup / failed-config tracking.
//
// The variation operator (agent) runs the N benchmarks; this encodes the exact bound
// + selection rule so they are deterministic and unit-tested, not LLM-judged.
//
// Unit-tested by tests/autotune.test.mjs.

/** Stable string identity for a config (sorted keys) — dedup + failed tracking. */
export function configKey(config) {
  if (!config || typeof config !== 'object') return '';
  const keys = Object.keys(config).sort();
  return keys.map((k) => `${k}=${JSON.stringify(config[k])}`).join(',');
}

/**
 * Cartesian product of a param grid {param: [values...]}, DETERMINISTICALLY capped at
 * `max` configs. When the full product exceeds the cap, evenly subsample across the
 * space (always keeping the first combination — usually the current/default config),
 * so the sweep stays affordable without biasing to one corner.
 *
 * @param {object} space  {BLOCK_M:[64,128], num_warps:[4,8], ...}
 * @param {object} opts   {max=64}
 * @returns {object[]}     list of config dicts (length 1..max)
 */
export function expandConfigSpace(space, opts = {}) {
  const max = Math.max(1, opts.max != null ? opts.max : 64);
  const params = Object.keys(space || {}).sort();
  const axes = params.map((p) => (Array.isArray(space[p]) && space[p].length ? space[p] : [space[p]]));
  if (!params.length) return [];

  let full = [{}];
  for (let i = 0; i < params.length; i++) {
    const next = [];
    for (const partial of full) for (const v of axes[i]) next.push({ ...partial, [params[i]]: v });
    full = next;
  }
  if (full.length <= max) return full;

  // Even subsample: keep index 0, then stride across the remainder.
  const out = [];
  const stride = full.length / max;
  for (let i = 0; i < max; i++) out.push(full[Math.floor(i * stride)]);
  return out;
}

/**
 * Pick the winning config from swept results. A result: {config, verified_geomean,
 * correct?, fraction_of_roofline?}. Only correct (correct !== false) positive-geomean
 * results are eligible; best verified geomean wins, ties broken by closeness to the
 * roofline (higher fraction), then by configKey for determinism.
 *
 * @returns {object|null}  the winning result, or null if none eligible
 */
export function selectBestConfig(results, opts = {}) {
  const eps = opts.tieEps != null ? opts.tieEps : 1e-9;
  const eligible = (results || []).filter(
    (r) => r && r.correct !== false && Number(r.verified_geomean) > 0);
  if (!eligible.length) return null;
  eligible.sort((a, b) => {
    const dg = Number(b.verified_geomean) - Number(a.verified_geomean);
    if (Math.abs(dg) > eps) return dg;
    const fa = Number(a.fraction_of_roofline) || 0, fb = Number(b.fraction_of_roofline) || 0;
    if (fb !== fa) return fb - fa;
    return configKey(a.config) < configKey(b.config) ? -1 : 1;
  });
  return eligible[0];
}

/** Drop configs whose key is in `failedKeys` (don't re-sweep known-bad points). */
export function excludeFailed(configs, failedKeys) {
  const bad = new Set(failedKeys || []);
  return (configs || []).filter((c) => !bad.has(configKey(c)));
}

export const _internals = {};
