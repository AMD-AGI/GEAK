// Pure, testable start-point selection for Diverse Fan-out (no I/O).
//
// Picks which committed lineage versions to start variation workers from, so a
// step can explore from MULTIPLE points in the commit history (escape local
// optima) instead of always the single best. See
// GEAK/docs/developer/avo_parallel_start_points_design.md.
//
// committed: [{ id, ref, speedup, strategy }]  (must include the baseline v0)
// returns:   array of refs (length 1..k), the GLOBAL BEST always first (exploit),
//            the rest are exploration starts per policy.

import { selectElites } from './archive.mjs';

function bestNode(committed) {
  let b = null;
  for (const n of committed) if (n && n.ref && (!b || (n.speedup || 0) > (b.speedup || 0))) b = n;
  return b;
}

/**
 * @param {Array} committed  lineage nodes {id, ref, speedup, strategy}
 * @param {object} opts      {policy: 'best'|'topk'|'diverse', k, excludeBaseline}
 * @returns {string[]} start refs (best first), length 1..k
 */
export function selectStartPoints(committed, opts = {}) {
  const policy = opts.policy || 'best';
  const k = Math.max(1, opts.k || 1);
  const excludeBaseline = opts.excludeBaseline !== false; // default: don't start from v0
  const best = bestNode(committed || []);
  if (!best) return [];
  if (policy === 'best' || k === 1) return [best.ref];

  // 'elites' = MAP-Elites quality-diversity archive (strategy x roofline regime).
  // Delegates to lib/archive.mjs; richer than 'diverse' (2-D behavioral cells).
  if (policy === 'elites') return selectElites(committed, { k, excludeBaseline });

  // Candidate pool: real committed versions (optionally excluding the v0 baseline),
  // always including the global best.
  let pool = (committed || []).filter(n => n && n.ref);
  if (excludeBaseline) pool = pool.filter(n => n.id !== 'v0' || n.ref === best.ref);
  if (pool.length === 0) pool = [best];

  let ordered;
  if (policy === 'diverse') {
    // One representative (highest speedup) per strategy bucket, buckets ranked by
    // their best speedup; then fill by raw speedup. Maximizes route diversity.
    const buckets = new Map();
    for (const n of pool) {
      const key = (n.strategy || n.id || '').toString().toLowerCase();
      const cur = buckets.get(key);
      if (!cur || (n.speedup || 0) > (cur.speedup || 0)) buckets.set(key, n);
    }
    const reps = [...buckets.values()].sort((a, b) => (b.speedup || 0) - (a.speedup || 0));
    const rest = pool.filter(n => !reps.includes(n)).sort((a, b) => (b.speedup || 0) - (a.speedup || 0));
    ordered = [...reps, ...rest];
  } else {
    // 'topk': by speedup desc.
    ordered = [...pool].sort((a, b) => (b.speedup || 0) - (a.speedup || 0));
  }

  // Best first (exploit), then the rest (explore); de-dup by ref; cap at k.
  const refs = [best.ref];
  for (const n of ordered) if (!refs.includes(n.ref)) refs.push(n.ref);
  return refs.slice(0, k);
}
