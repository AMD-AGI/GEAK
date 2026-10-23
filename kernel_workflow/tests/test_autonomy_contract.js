#!/usr/bin/env node
// End-to-end contract test for the strict staged-fusion path.
//
// The existing suites test individual predicates. They missed the ordering that made those
// predicates unreachable: a slower engineer result was discarded before verify, a rung was marked
// measured before launch_shape ran, and a skew-only aggregate could still become the winner. This
// suite composes the shipped pure regions and pins those boundaries together.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');
let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};
const region = (name) => {
  const m = src.match(new RegExp(`// <<REPLAY:${name}>>([\\s\\S]*?)// <</REPLAY:${name}>>`));
  if (!m) throw new Error(`missing REPLAY region ${name}`);
  return m[1];
};

const api = new Function(`
  ${region('direction_contract')}
  ${region('autonomy_transition')}
  ${region('enabling_step')}
  ${region('overlap_gate')}
  ${region('attribution_gate')}
  ${region('launch_gate')}
  ${region('bimodal_split')}
  ${region('autonomy_acceptance')}
  return {
    enrichDirection, mandatoryArmsVerdict, shouldVerifyDirection, strictDirectionVerdict,
    functionalAcceptance, guardContract, promotionScore, autonomyAcceptanceVerdict,
    winnerCommitTransition, landedRungOutcome, finalStatusVerdict, finalMetricFailure,
    integrationEligible
  };
`)();

const LADDER = [
  {
    id: 'D1', title: 'producer substrate', gated_on: [], step_role: 'enabling', enables: 'D2',
    cost_budget_pct: 3, mandatory_arms: ['publish_only', 'null'],
  },
  {
    id: 'D2', title: 'two-launch terminal', gated_on: ['D1'], step_role: 'terminal',
    mandatory_arms: ['whole', 'overlap', 'null'],
    target_shape: {
      launches: 2, stages_fused: ['dispatch', 'gemm1', 'gemm2', 'combine'],
      require_overlap: true,
    },
  },
];

console.log('\n# planner metadata reaches the worker intact');
{
  const d = api.enrichDirection({
    id: 'r1_d0', roadmap_rung: 'D1', specialty: 'distributed', prompt: 'build producer',
  }, LADDER);
  ok(d.step_role === 'enabling' && d.enables === 'D2' && d.cost_budget_pct === 3,
     'step role, debt target and cost budget are restored from the ladder');
  ok(d.mandatory_arms.join(',') === 'publish_only,null',
     'mandatory arms are data, not a sentence the planner may drop');
}

console.log('\n# a slower staged artifact reaches independent verification');
{
  const slow = { status: 'partial', speedup_geomean: 0.97, per_case: [] };
  ok(api.shouldVerifyDirection({ step_role: 'enabling' }, slow, false, 0.97),
     'an enabling result below 1.0 is verified');
  ok(api.shouldVerifyDirection({ step_role: 'terminal' }, slow, true, 0.97),
     'a working-kernel terminal below 1.0 is verified');
  ok(api.shouldVerifyDirection({ step_role: 'terminal' }, {
    ...slow, speedup_geomean: 1.20,
  }, true, 0),
  'a verifier-scored changed_kernel candidate reaches Verify even though ENG cannot compute absolute attribution');
  ok(!api.shouldVerifyDirection({ step_role: 'terminal' }, slow, false, 0.97),
     'ordinary speedup-only work keeps the cheap self-report screen');
}

console.log('\n# prerequisites mean completed-to-spec, not merely dispatched');
{
  const d = api.enrichDirection({
    id: 'r2_d0', roadmap_rung: 'D2', specialty: 'distributed', prompt: 'close',
  }, LADDER);
  ok(!api.strictDirectionVerdict(d, LADDER, new Set()).pass,
     'dispatching D1 without completing it does not unlock D2');
  ok(api.strictDirectionVerdict(d, LADDER, new Set(['D1'])).pass,
     'D2 unlocks only after D1 completed to spec');
  ok(!api.strictDirectionVerdict({
    ...d, target_shape: { ...d.target_shape, stages_fused: ['dispatch'] },
  }, LADDER, new Set(['D1'])).pass,
  'the planner cannot shrink the terminal fused-stage set');
}

const pairs = (guard, base, cand) =>
  Array.from({ length: 5 }, (_, i) => ({ guard, base: base + i * 0.002, cand: cand + i * 0.002 }));
const GOOD = {
  status: 'verified',
  correctness: 'pass',
  activation_confirmed: 'yes',
  graph_safe: 'pass',
  artifact_distinct: 'yes',
  artifact_hash_base: 'base-isa-sha',
  artifact_hash_candidate: 'candidate-isa-sha',
  liveness: 'pass',
  replay_count: 1000,
  replay_results: [
    { guard: '8192_uniform', count: 1000, status: 'pass', graph_safe: 'pass' },
    { guard: '8192_rank-mixed-skew', count: 1000, status: 'pass', graph_safe: 'pass' },
    { guard: '512_uniform', count: 1000, status: 'pass', graph_safe: 'pass' },
    { guard: '512_rank-mixed-skew', count: 1000, status: 'pass', graph_safe: 'pass' },
  ],
  accuracy: {
    metric: 'relL2', value: 0.059, threshold: 0.10,
    guard: '8192_uniform', method: 'frozen-reference relative L2',
  },
  arms_run: ['whole', 'overlap', 'null'],
  reps: 5,
  null_arm_pct: 0.1,
  per_case: [
    { name: '8192_uniform', speedup: 1.02 },
    { name: '8192_rank-mixed-skew', speedup: 1.0 },
    { name: '512_uniform', speedup: 1.0 },
    { name: '512_rank-mixed-skew', speedup: 1.0 },
  ],
  paired_readings: [
    ...pairs('8192_uniform', 4.70, 4.65),
    ...pairs('8192_rank-mixed-skew', 5.40, 5.40),
    ...pairs('512_uniform', 0.74, 0.74),
    ...pairs('512_rank-mixed-skew', 0.86, 0.86),
  ],
  launch_shape: {
    launches_base: 4, launches_cand: 2, target: 2, per_rank: true,
    stages_fused: ['dispatch', 'gemm1', 'gemm2', 'combine'],
    how_counted: 'paired rocprofv3 dispatch trace',
  },
  overlap: {
    measured: 'yes', fraction: 0.20, cu_fraction: 0.16,
    base_reading: 0.0, forced_reading: 0.95,
  },
  attribution: {
    changed_us: 4500, replaced_sum_us: 4600,
    residual_ms_base: 0.05, residual_ms_cand: 0.02,
    guard: '8192_uniform', method: 'same-timeline same-rank trace',
    absolute_to_frozen: true,
  },
};
const OPTS = {
  functionalRequirements: {
    requireAccuracy: true, requireLiveness: true, requireGraphSafe: true,
    requireArtifactDistinct: true,
    requiredReplays: 1000, requiredArms: ['whole', 'overlap', 'null'], strict: true,
    replayGuards: ['8192_uniform', '8192_rank-mixed-skew', '512_uniform', '512_rank-mixed-skew'],
    accuracyMetric: 'relL2', accuracyThreshold: 0.10, accuracyGuards: ['8192_uniform'],
  },
  launchTarget: 2,
  targetGuards: ['8192_uniform'],
  regressionGuards: ['8192_rank-mixed-skew', '512_uniform', '512_rank-mixed-skew'],
  promotionMetric: 'operator_e2e',
  fallbackScore: 9,
  requirePaired: true,
  requiredPairs: 5,
  requireOverlap: true,
  requireAttribution: true,
  positiveControlRan: true,
};

console.log('\n# only the target guard creates credit');
{
  ok(api.promotionScore({ speedup_geomean: 9 }, 'legacy', [], [], 1.07) === 1.07,
     'callers that omit the new scoped contract retain their historical primary score');
  const gc = api.guardContract(GOOD, OPTS.targetGuards, OPTS.regressionGuards, 9);
  ok(Math.abs(gc.score - 1.02) < 1e-12,
     'the target score ignores a global aggregate and every regression-only guard');
  const skewOnly = {
    ...GOOD,
    per_case: GOOD.per_case.map((r) =>
      r.name === '8192_uniform' ? { ...r, speedup: 1.0 }
        : r.name === '8192_rank-mixed-skew' ? { ...r, speedup: 1.20 } : r),
  };
  ok(api.promotionScore(skewOnly, 'operator_e2e', OPTS.targetGuards,
     OPTS.regressionGuards, 1.20) === 1.0,
  'a skew-only aggregate win cannot become the promotion score');
}

console.log('\n# strict terminal proof');
{
  const v = api.autonomyAcceptanceVerdict(LADDER[1], GOOD, OPTS);
  ok(v.complete && v.accepted && Math.abs(v.score - 1.02) < 1e-12,
     'two launches + target win + guard safety + controlled overlap + functional evidence accepts');
}
{
  const threeLaunch = { ...GOOD, launch_shape: { ...GOOD.launch_shape, launches_cand: 3, target: 3 } };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], threeLaunch, OPTS);
  ok(!v.complete && !v.accepted && v.reasons.some((s) => /Launch count is 3/.test(s)),
     'a measured three-launch partial cannot close the two-launch terminal');
}
{
  const noArms = { ...GOOD, arms_run: ['whole'] };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], noArms, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /mandatory arm/.test(s)),
     'a result that drops its overlap/null arms stays open');
  ok(api.functionalAcceptance(noArms, {
    ...OPTS.functionalRequirements, requiredArms: [],
  }).pass,
  'the same correct running terminal remains commit-able for the next working-kernel round');
}
{
  const deadMeter = { ...GOOD, overlap: {
    ...GOOD.overlap, forced_reading: 0.0,
  } };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], deadMeter, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /controls/.test(s)),
     'an overlap meter that cannot read forced concurrency cannot prove autonomy');
}
{
  const stale = { ...GOOD, replay_count: 999,
    replay_results: GOOD.replay_results.map((r, i) => i === 0 ? { ...r, count: 999 } : r) };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], stale, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /replay evidence is missing or below 1000/.test(s)),
     '999 replays cannot satisfy a 1000-replay liveness contract');
}
{
  const sameUnknown = { ...GOOD, artifact_distinct: 'unknown' };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], sameUnknown, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /artifact distinctness/.test(s)),
     'a host marker without a distinct JIT artifact cannot prove the candidate executed');
}
{
  const emptyHashes = { ...GOOD, artifact_hash_base: '', artifact_hash_candidate: '' };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], emptyHashes, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /hashes/.test(s)),
     'artifact_distinct=yes without two non-empty different hashes is not evidence');
}
{
  const wrongAccuracy = { ...GOOD, accuracy: {
    metric: 'other', value: 0.5, threshold: 1.0, guard: '8192_uniform', method: 'self report',
  } };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], wrongAccuracy, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /missing metric relL2/.test(s)),
     'a permissive threshold on the wrong metric cannot satisfy fixed accuracy');
}
{
  const negativeAccuracy = { ...GOOD, accuracy: {
    ...GOOD.accuracy, value: -1,
  } };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], negativeAccuracy, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /invalid value/.test(s)),
     'a physically impossible negative error cannot satisfy accuracy');
}
{
  const oneRoute = { ...GOOD, replay_results: GOOD.replay_results.slice(0, 1) };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], oneRoute, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /512_uniform replay evidence/.test(s)),
     '1000 replays on one route cannot stand in for every required route');
}
{
  const emptyAttribution = { ...GOOD, attribution: {
    changed_us: 0, replaced_sum_us: 0, guard: '8192_uniform',
    residual_ms_base: 0, residual_ms_cand: 0, method: '',
  } };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], emptyAttribution, OPTS);
  ok(!v.accepted && v.reasons.some((s) => /invalid attribution/.test(s)),
     'zero/empty attribution labels cannot satisfy the launch-change evidence gate');
}
{
  const targetFlat = {
    ...GOOD,
    per_case: GOOD.per_case.map((r) =>
      r.name === '8192_uniform' ? { ...r, speedup: 1.0 } : r),
  };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], targetFlat, OPTS);
  ok(v.complete && !v.accepted && v.reasons.some((s) => /not > 1.0/.test(s)),
     'a fully fused correct artifact is preserved as complete work but is not a profitable proof');
}
{
  const slowerKernel = { ...GOOD, attribution: {
    ...GOOD.attribution, changed_us: 5000, replaced_sum_us: 4600,
  } };
  const v = api.autonomyAcceptanceVerdict(LADDER[1], slowerKernel, {
    ...OPTS, promotionMetric: 'changed_kernel',
  });
  ok(!v.accepted && Math.abs(v.score - 0.92) < 1e-12,
     'a changed_kernel campaign cannot be accepted on an operator-only target win');
  ok(api.promotionScore({ attribution: {
    changed_us: 4000, replaced_sum_us: 4600,
  } }, 'changed_kernel', [], [], 9) === 0,
  'a round-local changed-kernel ratio is not comparable to cumulative/final state');
}

console.log('\n# shipped ordering and final arbitration');
{
  ok(!api.winnerCommitTransition(true, null, { id: 'D2' }, {
    autonomy_contract: { accepted: true },
  }).acceptanceReached,
  'an accepted measurement with a missing commit result cannot declare autonomy');
  ok(!api.winnerCommitTransition(true, { committed: false }, { id: 'D2' }, {
    autonomy_contract: { accepted: true },
  }).committed,
  'committed:false keeps cumulative and canonical state unchanged');
  ok(api.winnerCommitTransition(true, { committed: true }, { id: 'D2' }, {
    autonomy_contract: { accepted: true },
  }).acceptanceReached,
  'acceptance advances only after the exact winner reports a successful canonical commit');
  ok(!api.winnerCommitTransition(true, {
    committed: true, transaction_valid: true, exact_patch: false,
    head_before: 'a', head_after: 'b',
  }, { id: 'D2' }, { autonomy_contract: { accepted: true } }, true).acceptanceReached,
  'strict transition rejects a committed label whose patch did not match exactly');
  ok(/winner_precommit_head\.txt/.test(src) && /commit transaction check r/.test(src) &&
     /commitResult\.transaction_valid === true/.test(src),
  'strict winner commit is independently matched to its pre-head and stable patch id');
  ok(/enabling_precommit_head\.txt/.test(src) && /enabling transaction check r/.test(src) &&
     /res\.transaction_valid === true/.test(src),
  'strict enabling commits are independently accounted before debt/rung state moves');
  ok(api.landedRungOutcome('measured', 'enabling', false, true, false, false, false) === 'unmeasured',
     'a skipped enabling patch cannot close its rung');
  ok(api.landedRungOutcome('measured', 'terminal', true, true, false, true, false) === 'unmeasured',
     'terminal evidence without a landed winner cannot close its rung');
  ok(api.finalStatusVerdict('flagged', false, true) === 'flagged',
     'Director correctness/apply failure outranks autonomy_incomplete');
  ok(api.finalMetricFailure('operator_e2e', ['8192_uniform'], 0.99,
    { complete: true, regression_pass: true }, {}),
  'a scoped final score at or below baseline is a final failure even if Director says accepted');
  ok(api.finalMetricFailure('changed_kernel', [], 0,
    { complete: true, regression_pass: true }, {}),
  'missing final changed-kernel attribution cannot fall back to an accepted operator geomean');
  ok(!api.integrationEligible({ conclusion: 'improved', best: {} }, 1.10, 1.02,
    { complete: true, regression_pass: false }),
  'an integrated target win with a regression-guard loss cannot enter candidates');
  const verifyAt = src.indexOf('shouldVerifyDirection(');
  const verifiedAt = src.indexOf('const verified = clean.filter');
  ok(verifyAt > 0 && verifyAt < verifiedAt,
     'the slower-artifact exception exists at the earlier verify boundary, not only at promotion');
  ok(/WORKING_KERNEL \? 'commit' : 'acceptance'/.test(src) &&
     /functionalRequirementsFor\(winnerResult && winnerResult\.d, 'commit'\)/.test(src),
  'working terminal commit uses runtime evidence while final acceptance still requires measurement arms');
  const launchAt = src.indexOf('const directionLaunchTarget');
  const gradeAt = src.indexOf('Only landed state can close a staged rung');
  ok(launchAt > 0 && gradeAt > launchAt,
     'rung grading runs after launch/overlap evidence, so a three-launch candidate cannot close early');
  ok(/finalValidationStatus = finalStatusVerdict\(directorStatus, scopedGuardFailure, strictProofFailure\)/.test(src) &&
     /validation\.workflow_unchanged === true/.test(src),
  'final status uses the tested precedence reducer and requires an unchanged committed workflow');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
