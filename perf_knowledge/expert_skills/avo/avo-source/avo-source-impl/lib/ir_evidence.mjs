// Pure, testable IR-evidence logic for AVO (no I/O, no runtime globals).
//
// Bridges the gap between the roofline REGIME ("which ceiling binds me") and a
// SPECIFIC source lever ("which line to change, and why"). The deterministic IR
// metric vector is produced by scripts/ir_probe.py; this module turns it into
// (a) a sharpened steer hint for the operator, and (b) a cheap pre-verify negative
// screen. IR is evidence + a predictor — never the judge: the commit gate's
// independently-verified latency stays authoritative.
//
// IR vector shape (per candidate; see avo_ir_feedback_design.md §4):
//   { lang, ok, tier, metrics:{ convert_layout, num_stages, num_warps, mma_used,
//     lds_bytes, spill_bytes, vgpr, sgpr }, config, notes }
//
// Everything degrades gracefully to inert when ir.ok is false / metrics absent.
// Unit-tested by tests/ir_evidence.test.mjs.

const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : null);

/**
 * Combine the roofline regime + the candidate's IR metrics into specific, ranked
 * source levers. Returns {ok, regime, hints[]}; hints may be empty (IR usable but no
 * actionable signal). Inert (ok:false) when the IR vector is missing/failed.
 *
 * @param {string} regime  'memory-bound'|'compute-bound'|'basement'|'unknown'
 * @param {object} ir      IR metric vector (from ir_probe.py)
 */
export function irSteer(regime, ir) {
  if (!ir || !ir.ok || !ir.metrics) return { ok: false, regime, hints: [] };
  const m = ir.metrics;
  const cl = num(m.convert_layout);
  const ns = num(m.num_stages);
  const spill = num(m.spill_bytes);
  const vgpr = num(m.vgpr);
  const hints = [];

  if (regime === 'memory-bound') {
    if (cl != null && cl > 0)
      hints.push(`remove ${cl} convert_layout op(s) (LDS round-trips): choose a layout that avoids the conversion`);
    if (ns != null && ns <= 1)
      hints.push('num_stages<=1: enable software pipelining (raise num_stages) to overlap global->LDS loads');
  } else if (regime === 'compute-bound') {
    if (m.mma_used === false)
      hints.push('matrix cores NOT engaged (no mma/mfma in IR): restructure the dot to hit an MFMA path instead of FMA');
    if (spill != null && spill > 0)
      hints.push(`register spill (${spill}B scratch): cut live registers / use a smaller tile to restore occupancy`);
    if (cl != null && cl > 0)
      hints.push(`${cl} convert_layout op(s) still feeding the MMA: remove them to cut LDS traffic between stages`);
  } else if (regime === 'basement') {
    if (spill != null && spill > 0)
      hints.push(`register spill (${spill}B scratch): the kernel is occupancy-limited — reduce register pressure first`);
    if (vgpr != null && vgpr > 128)
      hints.push(`high VGPR usage (${vgpr}): occupancy-limited — shrink the tile / reuse registers to raise wave count`);
  }
  // Regime-independent red flag worth surfacing even when regime is unknown.
  if (m.mma_used === false && regime !== 'compute-bound')
    hints.push('no matrix-core op detected in IR — confirm whether this kernel should be using MFMA');

  return { ok: true, regime, hints, metrics: m };
}

/** Same config? IR for the same source can differ per config/shape (§6), so a
 *  comparison is only valid at the same config. Missing configs are treated as
 *  comparable (best-effort). */
function sameConfig(a, b) {
  const ca = a && a.config, cb = b && b.config;
  if (ca == null || cb == null) return true;
  try { return JSON.stringify(ca) === JSON.stringify(cb); } catch { return true; }
}

/**
 * Cheap pre-verify negative screen: predict a regression from a base→candidate IR
 * delta. Fires ONLY on high-confidence negatives (spill appeared, matrix cores
 * dropped, convert_layout grew) so it can *skip* an obvious loser before the
 * expensive GPU verify. It can NEVER accept — the commit gate still decides accepts.
 *
 * @returns {{regress:boolean, confidence:'high'|'low', reasons:string[]}}
 */
export function predictRegression(irBase, irCand) {
  const none = { regress: false, confidence: 'low', reasons: [] };
  if (!irBase || !irCand || !irBase.ok || !irCand.ok || !irBase.metrics || !irCand.metrics) return none;
  if (!sameConfig(irBase, irCand)) return none;     // not comparable across configs
  const b = irBase.metrics, c = irCand.metrics;
  const reasons = [];

  const bSpill = num(b.spill_bytes), cSpill = num(c.spill_bytes);
  if (bSpill != null && cSpill != null && bSpill === 0 && cSpill > 0)
    reasons.push(`register spill appeared (${cSpill}B scratch)`);

  if (b.mma_used === true && c.mma_used === false)
    reasons.push('matrix cores (mma/mfma) dropped to FMA fallback');

  const bCl = num(b.convert_layout), cCl = num(c.convert_layout);
  if (bCl != null && cCl != null && cCl > bCl)
    reasons.push(`convert_layout grew ${bCl}->${cCl} (more LDS round-trips)`);

  return { regress: reasons.length > 0, confidence: reasons.length > 0 ? 'high' : 'low', reasons };
}

export const _internals = { num, sameConfig };
