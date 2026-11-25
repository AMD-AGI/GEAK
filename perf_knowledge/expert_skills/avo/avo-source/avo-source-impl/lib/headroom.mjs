// Pure, testable headroom logic for AVO (no I/O, no runtime globals).
//
// Two roofline-driven control decisions that the deterministic loop owns:
//
//   1. shouldStopRoofline — stop early when the lineage is already close to the
//      hardware ceiling (geomean fraction_of_roofline >= target). A relative-only
//      AVO never knows when to quit; with a roofline it stops at the real limit
//      instead of burning budget for sub-noise gains.
//
//   2. allocateAttention — when several shapes share a kernel, spend the next step
//      on the one with the MOST remaining headroom, weighted by its share of total
//      time (Amdahl). This is what makes the run *approach* the roofline on the
//      shape that matters, instead of polishing an already-saturated case.
//
// Both consume the analyses produced by lib/roofline.mjs and degrade gracefully:
// when no case has telemetry they return "don't stop" / "no preference".
//
// Unit-tested by tests/headroom.test.mjs.

/**
 * Decide whether to stop the evolution loop because we're near the roofline.
 *
 * @param {object} summary  rooflineSummary(...) output
 * @param {object} opts     {target=0.85, commits, minCommits=1}
 * @returns {{stop:boolean, reason:string}}
 */
export function shouldStopRoofline(summary, opts = {}) {
  const target = opts.target != null ? opts.target : 0.85;
  const minCommits = opts.minCommits != null ? opts.minCommits : 1;
  const commits = opts.commits != null ? opts.commits : 0;

  if (!summary || !summary.ok || summary.geomean_fraction == null) {
    return { stop: false, reason: 'no roofline telemetry (relative-only mode)' };
  }
  if (commits < minCommits) {
    return { stop: false, reason: `roofline ${pct(summary.geomean_fraction)} but commits ${commits} < min ${minCommits}` };
  }
  if (summary.geomean_fraction >= target) {
    return { stop: true, reason: `at ${pct(summary.geomean_fraction)} of roofline >= target ${pct(target)} (diminishing returns)` };
  }
  return { stop: false, reason: `roofline ${pct(summary.geomean_fraction)} < target ${pct(target)} (headroom remains)` };
}

/**
 * Rank shapes by the optimization opportunity they represent and pick the focus.
 *
 * opportunity(case) = (potential_speedup - 1) * weight, where weight is the case's
 * share of total runtime if `weights` is given (Amdahl), else uniform. A case
 * already at the roofline (potential≈1) contributes ~0 and is skipped.
 *
 * @param {Array}  analyses  analyzeCase(...) results (with potential_speedup/regime)
 * @param {object} opts      {weights?: {name->seconds|ms|share}}
 * @returns {{focus:object|null, ranked:Array}}
 */
export function allocateAttention(analyses, opts = {}) {
  const weights = opts.weights || null;
  const scored = (analyses || [])
    .filter((a) => a && a.potential_speedup != null && a.potential_speedup > 1)
    .map((a) => {
      const w = weights && Number.isFinite(Number(weights[a.name])) ? Number(weights[a.name]) : 1;
      return { name: a.name, regime: a.regime, fraction_of_roofline: a.fraction_of_roofline,
        potential_speedup: a.potential_speedup, weight: w,
        opportunity: (a.potential_speedup - 1) * w };
    })
    .sort((x, y) => y.opportunity - x.opportunity);
  return { focus: scored[0] || null, ranked: scored };
}

const pct = (x) => (x == null ? '?' : `${(x * 100).toFixed(1)}%`);

export const _internals = { pct };
