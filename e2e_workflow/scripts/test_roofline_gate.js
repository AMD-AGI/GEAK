#!/usr/bin/env node
// Regression guard for the roofline skip gate (no GPU, no model needed).
//
// The gate reverses a doctrine the orchestrator held until now ("NEVER prune a candidate on
// roofline"): a kernel whose measured `roofline_pct` is at or above its class `target_eff` has no
// recoverable time, so it is dropped from BOTH optimization tracks however large it is. That makes
// `rooflineSkip` capable of deleting the single biggest kernel in a profile, which is exactly why
// its confidence carve-outs are the load-bearing part of this file: a MODELLING failure
// (`suspect`, clamped, unclassified, low-confidence) must never read as "already optimal".
//
// `rooflineSkip` is EXTRACTED from the real source rather than reimplemented here, so the test
// cannot pass while the shipped predicate drifts. The two call sites are checked structurally.
//
// Run:  node e2e_workflow/scripts/test_roofline_gate.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WORKFLOW = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const src = fs.readFileSync(WORKFLOW, 'utf8');

// ---------------------------------------------------------------- extract the shipped predicate
const gStart = src.indexOf('const ROOFLINE_TARGET_EFF =');
const gEnd = src.indexOf('const PRE_FLAGGED_HEADS =');
ok(gStart !== -1 && gEnd !== -1 && gStart < gEnd, 'roofline gate block located in e2e_workflow.js');
if (gStart === -1 || gEnd === -1) { console.error('cannot continue'); process.exit(1); }

const rooflineSkip = new Function(
  src.slice(gStart, gEnd) + '\n return { rooflineSkip, ROOFLINE_TARGET_EFF };')().rooflineSkip;

// A full-confidence, genuinely saturated MoE head — 26% of GPU time, and still skipped.
const SATURATED = {
  short_name: 'fused_moe_kernel', pct_gpu_time: 26.45, op_kind: 'moe',
  roofline_pct: 0.88, target_eff: 0.85, headroom_class: 'saturated',
  roofline_confidence: 'high', suspect: false,
};
const withOut = (k) => { const c = { ...SATURATED }; delete c[k]; return c; };

console.log('\n-- the gate fires on a full-confidence saturated verdict');
ok(rooflineSkip(SATURATED) !== null, 'roofline 0.88 >= target 0.85 -> skipped');
ok(/0\.880 >= target_eff 0\.850/.test(rooflineSkip(SATURATED)),
  'reason states both numbers, so the report can be audited');
ok(rooflineSkip({ ...SATURATED, roofline_pct: 0.85 }) !== null, 'the bar is >=, not >');
ok(rooflineSkip({ ...SATURATED, roofline_pct: 0.8499 }) === null, 'just below the bar -> optimized');
ok(rooflineSkip({ ...SATURATED, roofline_pct: 0.84, headroom_class: 'saturated' }) === null,
  'saturated-but-below-target is still optimized: the class banding is NOT the gate');

console.log('\n-- a modelling failure never deletes a kernel');
ok(rooflineSkip({ ...SATURATED, suspect: true, roofline_pct: 1.0 }) === null,
  'suspect (clamped byte model) -> never skipped, even at a clamped 1.00');
ok(rooflineSkip({ ...SATURATED, headroom_class: 'unknown' }) === null,
  'headroom_class=unknown (dispatch-bound / infeasible) -> never skipped');
ok(rooflineSkip(withOut('headroom_class')) === null, 'missing headroom_class -> never skipped');
ok(rooflineSkip({ ...SATURATED, roofline_confidence: 'low' }) === null,
  'confidence=low -> never skipped (an unvalidated peak reads ~2x high)');
ok(rooflineSkip(withOut('roofline_confidence')) === null, 'missing confidence -> never skipped');
ok(rooflineSkip({ ...SATURATED, skip_optimization: false }) === null,
  'an explicit skip_optimization=false from the skill is honoured over the local recompute');

console.log('\n-- degradation: no analysis skill, no gate');
ok(rooflineSkip({ short_name: 'k', pct_gpu_time: 40.0 }) === null,
  'a bare candidate (analysis_skill=none) is never skipped -> pre-feature behavior preserved');
ok(rooflineSkip(null) === null && rooflineSkip(undefined) === null, 'null/undefined are safe');
ok(rooflineSkip({ ...SATURATED, roofline_pct: 'n/a' }) === null, 'non-numeric roofline_pct is safe');
ok(rooflineSkip({ ...SATURATED, roofline_pct: 0 }) === null, 'roofline_pct=0 is absence, not 0%');

console.log('\n-- target_eff resolution');
ok(rooflineSkip({ ...withOut('target_eff'), roofline_pct: 0.88 }) !== null,
  'no target_eff on the candidate -> falls back to the op_kind table (moe=0.85)');
ok(rooflineSkip({ ...withOut('target_eff'), op_kind: 'attn', roofline_pct: 0.70 }) !== null,
  'attn falls back to 0.60, so 0.70 is over the bar');
ok(rooflineSkip({ ...withOut('target_eff'), op_kind: 'moe', roofline_pct: 0.70 }) === null,
  'the same 0.70 is UNDER the moe bar -> class-specific, not one global number');
ok(rooflineSkip({ ...withOut('target_eff'), op_kind: 'reduce_scatter', roofline_pct: 0.99 }) === null,
  'an unknown op class has no bar -> no skip (a guess must not drop a kernel)');
ok(rooflineSkip({ ...withOut('target_eff'), op_kind: '', classification: 'dense gemm',
  roofline_pct: 0.90 }) !== null, 'classification is read when op_kind is absent');
ok(rooflineSkip({ ...SATURATED, target_eff: 0.95, roofline_pct: 0.90 }) === null,
  "a recalibrated per-candidate target_eff wins over the table (SKILL.md §8 rule 5)");

// ---------------------------------------------------------------- both call sites are wired
// The gate is only worth anything if it is actually consulted on both tracks. These are structural
// checks: admitHeads and the milestone planner are orchestration, not executable standalone.
console.log('\n-- the gate is consulted on both tracks');
const heads = src.slice(src.indexOf('function admitHeads('),
  src.indexOf('function admitHeads(') + 6000);
ok(/rooflineSkip\(head\)/.test(heads), 'head track: admitHeads() calls rooflineSkip');
ok(/gate: 'roofline_saturated'/.test(heads),
  'a skipped head is recorded in PRE_FLAGGED_HEADS, so it stays visible in the final report');
ok(heads.indexOf('rooflineSkip(head)') < heads.indexOf('prepareHeadSelection'),
  'the gate runs BEFORE the head is admitted, not after a budget is spent');

ok(/const planCands = planCandsRaw\.filter\(c => !belowBar\(c\) && !rooflineSkip\(c\)\)/.test(src),
  'milestone track: the plan filter applies both the %GPU bar and the roofline gate');
ok(/const skipped = planCandsRaw\.filter\(c => belowBar\(c\) \|\| rooflineSkip\(c\)\)/.test(src),
  'and both reasons land in the same skipped-kernel log line');

console.log('\n-- the %GPU bar moved 5% -> 2%');
ok(/A\.milestone_min_pct != null \? A\.milestone_min_pct : 2/.test(src), 'MILESTONE_MIN_PCT default 2');
ok(/A\.head_threshold_pct != null \? A\.head_threshold_pct : 2/.test(src), 'HEAD_THRESHOLD_PCT default 2');

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall roofline-gate checks passed');
process.exit(failures ? 1 : 0);
