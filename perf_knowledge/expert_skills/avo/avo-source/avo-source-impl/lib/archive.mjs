// Pure, testable quality-diversity archive for AVO (no I/O, no runtime globals).
//
// A single-lineage chain collapses to ONE local optimum: every step builds on the
// global best, so a whole class of structures (e.g. a different tiling that was
// briefly behind but had more headroom) is abandoned. MAP-Elites keeps the BEST
// version per behavioral CELL — here cell = (strategy bucket x roofline regime) —
// so exploration can resume from a diverse frontier and a later step can CROSS OVER
// two complementary elites (e.g. a memory-bound winner + a compute-bound winner).
//
// This is the deterministic substrate; the actual mutation/crossover edit is done
// by the variation operator (agent). The loop uses:
//   - buildArchive / selectElites  -> Diverse Fan-out start points (escape optima)
//   - crossoverPairs               -> propose complementary parents to combine
//
// Node shape (from lineage): {id, ref, speedup, strategy, regime?}.
// Unit-tested by tests/archive.test.mjs.

const norm = (s) => String(s == null ? '' : s).trim().toLowerCase() || '?';

/** Behavioral cell key for a committed node. Default dims: strategy x regime. */
export function cellKey(node, opts = {}) {
  const dims = opts.dims || ['strategy', 'regime'];
  return dims.map((d) => norm(node && node[d])).join('|');
}

/**
 * Best node per cell (a Map cellKey -> node). Higher speedup wins a cell; ties keep
 * the first seen (older/simpler).
 */
export function buildArchive(committed, opts = {}) {
  const archive = new Map();
  for (const n of committed || []) {
    if (!n || !n.ref) continue;
    const key = cellKey(n, opts);
    const cur = archive.get(key);
    if (!cur || (n.speedup || 0) > (cur.speedup || 0)) archive.set(key, n);
  }
  return archive;
}

function bestNode(committed) {
  let b = null;
  for (const n of committed || []) if (n && n.ref && (!b || (n.speedup || 0) > (b.speedup || 0))) b = n;
  return b;
}

/**
 * Pick up to k diverse elite start refs: the global best first (exploit), then one
 * elite per distinct cell ranked by speedup (explore). De-duped by ref, baseline
 * (v0) excluded unless it is itself the best.
 *
 * @returns {string[]} refs, best first, length 1..k
 */
export function selectElites(committed, opts = {}) {
  const k = Math.max(1, opts.k || 1);
  const best = bestNode(committed);
  if (!best) return [];
  if (k === 1) return [best.ref];

  const archive = buildArchive(committed, opts);
  const elites = [...archive.values()]
    .filter((n) => opts.excludeBaseline === false || n.id !== 'v0' || n.ref === best.ref)
    .sort((a, b) => (b.speedup || 0) - (a.speedup || 0));

  const refs = [best.ref];
  for (const n of elites) if (!refs.includes(n.ref)) refs.push(n.ref);
  return refs.slice(0, k);
}

/**
 * Propose crossover parent pairs: two elites from DIFFERENT regime cells, so the
 * combination targets complementary bottlenecks (a memory-bound win + a
 * compute-bound win). Ranked by combined speedup. Returns [] if fewer than two
 * distinct regimes are represented.
 *
 * @returns {Array<{a:node, b:node, combined:number, regimes:[string,string]}>}
 */
export function crossoverPairs(committed, opts = {}) {
  const limit = opts.limit != null ? opts.limit : 3;
  // Best elite per regime (ignoring strategy), excluding the baseline.
  const byRegime = new Map();
  for (const n of committed || []) {
    if (!n || !n.ref || n.id === 'v0') continue;
    const r = norm(n.regime);
    if (r === '?') continue;
    const cur = byRegime.get(r);
    if (!cur || (n.speedup || 0) > (cur.speedup || 0)) byRegime.set(r, n);
  }
  const reps = [...byRegime.entries()];           // [regime, node]
  const pairs = [];
  for (let i = 0; i < reps.length; i++) {
    for (let j = i + 1; j < reps.length; j++) {
      const a = reps[i][1], b = reps[j][1];
      pairs.push({ a, b, combined: (a.speedup || 0) + (b.speedup || 0),
        regimes: [reps[i][0], reps[j][0]] });
    }
  }
  pairs.sort((x, y) => y.combined - x.combined);
  return pairs.slice(0, Math.max(0, limit));
}

/**
 * Select a complementary group of a Team round's GATED passers to hand to the
 * Integrator. A merge is only well-defined when the candidates share a base ref
 * (so the patches stack onto one workspace) AND attack DIFFERENT regimes (so the
 * combination compounds). Picks, among same-base groups with >=2 regimes, the group
 * whose best-two-by-speedup have the largest combined speedup.
 *
 * @param {Array} passers  [{ ref?, startRef, regime, speedup, ...payload }]
 * @param {object} opts    {minRegimes=2}
 * @returns {{base:string, members:Array, pair:[obj,obj]}|null}
 */
export function selectMergeGroup(passers, opts = {}) {
  const minRegimes = opts.minRegimes != null ? opts.minRegimes : 2;
  // Group passers by their base (startRef); within a group keep the best per regime.
  const groups = new Map();
  for (const p of passers || []) {
    if (!p || p.startRef == null) continue;
    const r = norm(p.regime);
    if (r === '?') continue;                       // need a known regime to be "complementary"
    let g = groups.get(p.startRef);
    if (!g) { g = new Map(); groups.set(p.startRef, g); }
    const cur = g.get(r);
    if (!cur || (p.speedup || 0) > (cur.speedup || 0)) g.set(r, p);
  }
  let best = null;
  for (const [base, byRegime] of groups) {
    if (byRegime.size < minRegimes) continue;
    const members = [...byRegime.values()].sort((a, b) => (b.speedup || 0) - (a.speedup || 0));
    const combined = (members[0].speedup || 0) + (members[1].speedup || 0);
    if (!best || combined > best.combined) {
      best = { base, members, pair: [members[0], members[1]], combined };
    }
  }
  if (!best) return null;
  return { base: best.base, members: best.members, pair: best.pair };
}

/**
 * Gate for an Integrator-merged candidate: it must be worth committing OVER the best
 * individual winner it was merged from (else just commit that winner). Strictly
 * greater by `margin` (the workflow's commit gate separately checks it beats the
 * global best vs the TRUE baseline).
 *
 * @param {number} mergedSpeedup  verified geomean of the merged candidate
 * @param {Array}  members        the winners merged (with .speedup)
 * @param {object} opts           {margin=0}
 * @returns {boolean}
 */
export function mergeImproves(mergedSpeedup, members, opts = {}) {
  const margin = opts.margin != null ? opts.margin : 0;
  const m = Number(mergedSpeedup);
  if (!(m > 0)) return false;
  const bar = Math.max(0, ...(members || []).map((x) => Number(x && x.speedup) || 0));
  return m > bar * (1 + margin);
}

/** Coverage stats for logging/reporting: how much of the behavior space is filled. */
export function archiveStats(committed, opts = {}) {
  const archive = buildArchive(committed, opts);
  const regimes = new Set();
  for (const n of archive.values()) regimes.add(norm(n.regime));
  return { cells: archive.size, regimes: regimes.size, total: (committed || []).length };
}

export const _internals = { norm, bestNode };
