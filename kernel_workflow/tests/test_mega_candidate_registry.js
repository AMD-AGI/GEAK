#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_mega.js'), 'utf8');
const match = src.match(
  /\/\/ <<REPLAY:mega_candidate_registry>>([\s\S]*?)\/\/ <<\/REPLAY:mega_candidate_registry>>/,
);
if (!match) throw new Error('missing mega_candidate_registry replay region');
const topology = src.match(
  /\/\/ <<REPLAY:mega_topology_contract>>([\s\S]*?)\/\/ <<\/REPLAY:mega_topology_contract>>/,
);
if (!topology) throw new Error('missing mega_topology_contract replay region');

const api = new Function(`
  function guardContract(v, targets, regressions) {
    const rows = new Map((v.per_case || []).map((r) => [r.name, r]));
    const missing = regressions.filter((g) => !rows.has(g));
    return {
      complete: targets.every((g) => rows.has(g)) && missing.length === 0,
      regression_pass: missing.length === 0 &&
        regressions.every((g) => Number(rows.get(g).speedup) >= 1),
    };
  }
  function promotionScore(v, metric, targets, regressions, fallback) {
    if (!targets.length) return Number(fallback || 0);
    const vals = targets.map((g) => Number(((v.per_case || []).find((r) => r.name === g) || {}).speedup));
    if (!vals.every((x) => x > 0)) return 0;
    return vals.length === 1 ? vals[0]
      : Math.exp(vals.reduce((a, x) => a + Math.log(x), 0) / vals.length);
  }
  ${topology[1]}
  ${match[1]}
  return {
    validMegaCandidateId, megaChildCandidateId, normalizeMegaCandidate, upsertMegaCandidate,
    megaCandidateHardPass, selectMegaCandidate, selectMegaSearchParent,
    megaCalibrationClaimPass, pairedGuardReadout, megaCandidateFromVerification,
    megaRegistryForSearch, megaTargetsReadout,
  };
`)();

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const complete = (id, score) => ({
  id, source: 'search', status: 'scored', tree: `/state/${id}`, head: `${id}-head`,
  claim_complete: true, attempt_id: `${id}:1`, evidence_manifest: `${id}.json`,
  target_guard: 'target_case',
  absolute_score: score, per_case: [{ name: '8192_uniform', speedup: score }],
  paired_readings: Array.from({ length: 5 }, (_, i) => ({
    guard: 'target_case', base: 4.6 + i * 0.001,
    cand: (4.6 + i * 0.001) / score,
  })),
  null_arm_pct: 0.1, reps: 5,
  artifact_hash_base: 'base-hash', artifact_hash_candidate: `${id}-hash`,
  correctness_pass: true, activation_pass: true, head_pass: true, launch_pass: true,
  liveness_pass: true, graph_pass: true, guards_pass: true, artifact_distinct: true,
  measurement_pass: true,
});

console.log('\n# one source contract for skill-on and skill-off');
ok(api.normalizeMegaCandidate({ id: 'guided', source: 'search' }).source_valid,
  'ordinary search source is accepted');
ok(api.normalizeMegaCandidate({ id: 'integrated', source: 'integrated' }).source_valid,
  'integrated source is accepted');
ok(api.normalizeMegaCandidate({ id: 'legacy', source: 'validated_skill' }).status === 'rejected',
  'validated_skill is not a separate candidate source');
ok(api.normalizeMegaCandidate({ id: 'guided_fusion_v1', source: 'search' }).source_id_valid,
  'candidate ids are not reserved for a reproduction lane');
ok(!api.validMegaCandidateId('../escape') && !api.validMegaCandidateId('a/b'),
  'candidate ids cannot escape their state directory');
{
  const parent = 'full_persistent_pipeline_r7_r10_r15_r16_r17_r1_r2_r3_r5_r7_r8_r9';
  ok(api.megaChildCandidateId('lane_r1', 2) === 'lane_r1_r2',
    'a short lineage keeps the readable _rN child id');
  const child = api.megaChildCandidateId(parent, 10);
  ok(api.validMegaCandidateId(child) && /_r10$/.test(child) && child.length <= 64,
    'a 64-char parent (the v8 crash) yields a valid, round-tagged child id');
  ok(child === api.megaChildCandidateId(parent, 10) &&
     child !== api.megaChildCandidateId(parent.replace(/_r9$/, '_r8'), 10) &&
     child !== api.megaChildCandidateId(parent, 11),
    'compacted child ids are deterministic and distinct per parent and round');
  let id = parent;
  for (let r = 1; r <= 40; r++) id = api.megaChildCandidateId(id, r);
  ok(api.validMegaCandidateId(id), 'repeated compaction across rounds stays valid');
  ok(/megaChildCandidateId\(safeRequestedId, currentRound\)/.test(src),
    'the planner derives scored-parent child ids through the bounded helper');
}
{
  const staged = api.normalizeMegaCandidate({
    id: 'staged', source: 'search', checkpoint_complete: true,
    structural_verified: true, runtime_verified: false, score_complete: false,
    structural_report: '/eval/structure.json',
  });
  ok(staged.checkpoint_complete && staged.structural_verified &&
     !staged.runtime_verified && !staged.score_complete,
  'checkpoint/structural/runtime/score states remain independent');
}

console.log('\n# WIP cannot overwrite verified evidence');
{
  let registry = [complete('guided', 1.0448)];
  registry = api.upsertMegaCandidate(registry, {
    id: 'guided', source: 'search', status: 'authoring', claim_complete: false,
    head: 'new-head', tree: '/state/guided', attempt_id: 'guided:2',
    contract_failures: [{
      id: 'counter_layout', category: 'correctness', severity: 'required',
      messages: ['counter head exceeds capacity'],
    }],
  });
  ok(registry[0].absolute_score === 1.0448 && registry[0].head === 'guided-head',
    'an incomplete turn preserves the last verified score/head');
  ok(registry[0].working_head === 'new-head',
    'new authoring progress is retained separately');
  ok(api.megaRegistryForSearch(registry)[0].contract_failures[0].id === 'counter_layout',
    'an incomplete authoring result preserves exact structured failures for the next Planner');
}
{
  let registry = [complete('guided', 1.0448)];
  registry = api.upsertMegaCandidate(registry, {
    id: 'guided', source: 'search', status: 'authoring', claim_complete: false,
    checkpoint_complete: true, tree: '/state/guided', head: 'working-head',
    attempt_id: 'guided:2', structural_verified: true,
    structural_report: '/eval/working-structure.json',
    structural_skill_id: 'skill',
    structural_candidate_head: 'working-head',
    structural_candidate_tree_digest: 'working-tree-digest',
    structural_skill_bundle_sha256: 'working-bundle',
    structural_planner_extension_sha256: 'working-planner',
    structural_contract_revision: 'v6',
    structural_contract_sha256: 'working-contract',
    evidence_manifest: '/eval/working-evidence.json',
    contract_failures: [],
  });
  const working = registry[0].working_snapshot;
  const search = api.megaRegistryForSearch(registry)[0];
  ok(working.structural_verified &&
     working.structural_candidate_head === 'working-head' &&
     working.structural_skill_bundle_sha256 === 'working-bundle',
    'a verified WIP retains its complete structural certificate');
  ok(search.structural_verified &&
     search.structural_candidate_head === 'working-head' &&
     search.structural_contract_sha256 === 'working-contract',
    'Planner and author preflight consume the effective working certificate');
}
{
  const structural = {
    id: 'structural', source: 'search', status: 'authoring',
    tree: '/state/structural', head: 'sealed-head', claim_complete: false,
    checkpoint_complete: true, structural_verified: true,
    structural_report: '/eval/structure.json',
    structural_skill_id: 'skill',
    structural_candidate_head: 'sealed-head',
    structural_candidate_tree_digest: 'tree-digest',
    structural_skill_bundle_sha256: 'bundle-digest',
    structural_planner_extension_sha256: 'planner-digest',
    structural_contract_revision: 'v1',
    structural_contract_sha256: 'contract-digest',
  };
  let registry = [structural];
  registry = api.upsertMegaCandidate(registry, {
    id: 'structural', source: 'search', status: 'authoring',
    tree: '/state/structural', head: 'sealed-head', claim_complete: false,
    next_blocker: 'device runtime fault',
  });
  ok(registry[0].structural_verified &&
     registry[0].structural_candidate_head === 'sealed-head' &&
     registry[0].structural_contract_sha256 === 'contract-digest' &&
     registry[0].contract_failures.length === 0,
  'an incomplete runtime attempt preserves exact-HEAD structural evidence');
  const stalePersisted = {
    ...structural,
    contract_failures: [{
      id: 'stale_advisory', category: 'structure', severity: 'required',
      messages: ['old verifier prose'],
    }],
  };
  const conflicted = api.megaRegistryForSearch([stalePersisted])[0];
  ok(!conflicted.structural_verified &&
     conflicted.contract_failures[0].id === 'structural_state_conflict',
    'legacy pass plus required failures fails closed until fresh structural verification');
  registry = api.upsertMegaCandidate([structural], {
    id: 'structural', source: 'search', status: 'authoring',
    tree: '/state/structural', head: 'sealed-head', claim_complete: false,
    contract_failures: [{
      id: 'real_structural_failure', category: 'compiler', severity: 'required',
      messages: ['current verifier rejected the exact head'],
    }],
  });
  ok(!registry[0].structural_verified &&
     registry[0].contract_failures[0].id === 'real_structural_failure',
  'an exact-head structural failure revokes the old pass instead of creating contradictory state');
  registry = [structural];
  registry = api.upsertMegaCandidate(registry, {
    id: 'structural', source: 'search', status: 'authoring',
    tree: '/state/structural', head: 'changed-head', claim_complete: false,
  });
  ok(!registry[0].structural_verified,
    'a changed source HEAD still revokes structural evidence');
}

console.log('\n# selection is knowledge-blind and absolute-to-frozen');
{
  const selected = api.selectMegaCandidate([
    complete('guided', 1.0448),
    complete('autonomous', 1.06),
    complete('slow', 0.99),
  ], 1.45);
  ok(selected.selected && selected.selected.id === 'autonomous',
    'the numerically fastest passing candidate wins regardless of knowledge source');
  ok(!api.selectMegaCandidate([complete('slow', 0.99)], 0).selected,
    'a correct sub-baseline candidate cannot be final output');
  ok(api.selectMegaSearchParent([
    complete('guided', 1.0448), complete('autonomous', 1.06),
  ]).id === 'autonomous', 'continuation parent is the fastest verified candidate');
}

console.log('\n# search may score a complete partial fusion');
{
  const score = 1.02;
  const meta = {
    id: 'partial', source: 'search', status: 'authoring',
    tree: '/state/partial', head: 'partial-head', attempt_id: 'partial:1',
  };
  const ver = {
    status: 'verified', claim_complete: true, attempt_id: 'partial:1',
    evidence_manifest: '/eval/partial.json', candidate_head: 'partial-head',
    verified_geomean: score, per_case: [{ name: '8192_uniform', speedup: score }],
    paired_readings: Array.from({ length: 5 }, (_, i) => ({
      guard: '8192_uniform', base: 4.6 + i * 0.001,
      cand: (4.6 + i * 0.001) / score,
    })),
    null_arm_pct: 0.1, reps: 5,
    accuracy_results: [{ guard: '8192_uniform', metric: 'relL2', value: 0.05, threshold: 0.1 }],
    correctness: 'pass', activation_confirmed: 'yes', activation_on_hardware: 'yes',
    launch_shape: { launches_base: 4, launches_cand: 3 },
    liveness: 'pass', replay_count: 30, graph_safe: 'pass',
    artifact_distinct: 'yes', artifact_hash_base: 'base', artifact_hash_candidate: 'partial',
  };
  const opts = {
    targetGuards: ['8192_uniform'], regressionGuards: [], launchTarget: 2,
    promotionMetric: 'operator_e2e', accuracyMetric: 'relL2', accuracyThreshold: 0.1,
    requiredReplays: 30, requiredPairs: 5, allowPartialFusion: true,
  };
  const partial = api.megaCandidateFromVerification(meta, ver, opts);
  ok(partial.status === 'scored' && partial.launch_pass,
    'a faster 3-launch fusion scores against a 4-launch frozen baseline');
  ok(partial.runtime_verified && partial.score_complete,
    'hardware runtime and score completion are recorded separately from source structure');
  const fullOnly = api.megaCandidateFromVerification(meta, ver, {
    ...opts, allowPartialFusion: false,
  });
  ok(fullOnly.status !== 'scored' && !fullOnly.launch_pass,
    'the launch-target upper bound remains enforced for a full-fusion target');
  const withLaunches = (n, base = 4) => api.megaCandidateFromVerification(meta,
    { ...ver, launch_shape: { launches_base: base, launches_cand: n } },
    { ...opts, allowPartialFusion: false });
  ok(withLaunches(2).launch_pass && withLaunches(1).launch_pass,
    'launch_target 2 is an upper bound: 2 and 1 launches both pass without partial fusion');
  ok(!withLaunches(0).launch_pass && !withLaunches(2, 2).launch_pass,
    'zero launches or no reduction from the baseline never passes');
}

console.log('\n# completed hardware failures remain authoritative');
{
  const meta = {
    id: 'faulted', source: 'search', status: 'authoring',
    tree: '/state/faulted', head: 'faulted-head', attempt_id: 'faulted:1',
    next_blocker: 'if Verify still faults, use measured_partial_fallback',
    notes: 'author-only checkpoint',
  };
  const ver = {
    status: 'correctness_failed', correctness: 'fail', claim_complete: true,
    attempt_id: 'faulted:1', candidate_head: 'faulted-head',
    evidence_manifest: '/eval/faulted/verify/evidence_manifest.json',
    activation_confirmed: 'yes', activation_on_hardware: 'yes',
    notes: 'path=MEGA x8, then nil-base device fault at bs=128',
    per_case: [], accuracy_results: [], replay_count: 0,
  };
  const faulted = api.megaCandidateFromVerification(meta, ver, {
    targetGuards: ['8192_uniform'], regressionGuards: [], launchTarget: 2,
    promotionMetric: 'operator_e2e', accuracyMetric: 'relL2', accuracyThreshold: 0.1,
    requiredReplays: 30, requiredPairs: 5, allowPartialFusion: true,
  });
  const resumed = api.normalizeMegaCandidate(faulted);
  const planner = api.megaRegistryForSearch([resumed])[0];
  ok(resumed.verification_status === 'correctness_failed' &&
     resumed.correctness === 'fail' && resumed.gpu_executed &&
     resumed.activation_on_hardware === 'yes',
    'a complete failed Verify records that hardware executed and correctness failed');
  ok(resumed.next_blocker.includes('correctness_failed') &&
     resumed.next_blocker.includes('/eval/faulted/verify/evidence_manifest.json') &&
     resumed.notes.includes('nil-base'),
    'Verify failure evidence replaces the stale author-only blocker and notes');
  ok(planner.gpu_executed && planner.verification_status === 'correctness_failed' &&
     planner.notes.includes('nil-base'),
    'the next Planner receives the completed hardware failure verbatim');
}

console.log('\n# every target guard must clear 1.0 on its own');
{
  const pairs = (guard, s) => Array.from({ length: 5 }, (_, i) => ({
    guard, base: 1 + i * 0.001, cand: (1 + i * 0.001) / s,
  }));
  const single = pairs('64_uniform_mtpr64', 1.13);
  ok(api.megaTargetsReadout(single, ['64_uniform_mtpr64']).score ===
     api.pairedGuardReadout(single, '64_uniform_mtpr64').score,
    'one target guard keeps the exact single-guard readout');
  const run = (s64, s8k) => {
    const ver = {
      status: 'verified', claim_complete: true, attempt_id: 'two:1',
      evidence_manifest: '/eval/two.json', candidate_head: 'two-head',
      per_case: [{ name: '64_uniform_mtpr64', speedup: s64 }, { name: '8192_uniform', speedup: s8k }],
      paired_readings: [...pairs('64_uniform_mtpr64', s64), ...pairs('8192_uniform', s8k)],
      null_arm_pct: 0.1, reps: 5,
      accuracy_results: ['64_uniform_mtpr64', '8192_uniform'].map((guard) =>
        ({ guard, metric: 'relL2', value: 0.05, threshold: 0.1 })),
      correctness: 'pass', activation_confirmed: 'yes', activation_on_hardware: 'yes',
      launch_shape: { launches_base: 4, launches_cand: 2 },
      liveness: 'pass', replay_count: 30, graph_safe: 'pass',
      artifact_distinct: 'yes', artifact_hash_base: 'base', artifact_hash_candidate: 'two',
    };
    return api.megaCandidateFromVerification({
      id: 'two', source: 'search', status: 'authoring', tree: '/state/two',
      head: 'two-head', attempt_id: 'two:1',
    }, ver, {
      targetGuards: ['64_uniform_mtpr64', '8192_uniform'], regressionGuards: [], launchTarget: 2,
      promotionMetric: 'operator_e2e', accuracyMetric: 'relL2', accuracyThreshold: 0.1,
      requiredReplays: 30, requiredPairs: 5,
    });
  };
  const both = run(1.10, 1.05);
  ok(both.status === 'scored' && Math.abs(both.absolute_score - Math.sqrt(1.10 * 1.05)) < 1e-6,
    'two passing targets score at their geomean');
  ok(both.target_guard === '64_uniform_mtpr64,8192_uniform' && api.megaCandidateHardPass(both),
    'the record carries every target guard and stays selectable');
  const twoTargetApi = new Function('src', `const TARGET_GUARDS = ['64_uniform_mtpr64', '8192_uniform'];
    function guardContract() { return { complete: true, regression_pass: true }; }
    function promotionScore() { return 0; }
    ${topology[1]}
    ${match[1]}
    return { megaCandidateHardPass };`)();
  const oldRow = { ...both, target_guard: '64_uniform_mtpr64', absolute_score: 1.10,
    paired_readings: pairs('64_uniform_mtpr64', 1.10) };
  ok(api.megaCandidateHardPass(oldRow) && !twoTargetApi.megaCandidateHardPass(oldRow),
    'a resumed row scored on fewer targets than configured is not selectable');
  const one = run(1.13, 0.999);
  ok(one.status !== 'scored' && !one.measurement_pass,
    'a geomean above 1.0 cannot hide a target guard at or below 1.0');
}

console.log('\n# calibration remains fail-closed');
{
  const pc = {
    claim_complete: true, ran: true, switch_present: true, passed: true,
    attempt_id: 'pc:1', evidence_manifest: '/eval/pc.json', measured_pct: -5,
    control_pairs_pct: [-5, -5.1, -4.9, -5.2, -5],
    null_pairs_pct: [0.1, -0.1, 0.2, -0.2, 0.1, 0, -0.1, 0.1],
  };
  ok(api.megaCalibrationClaimPass(pc, {
    expected_pct_lo: -9, expected_pct_hi: -4,
    required_control_pairs: 5, required_null_pairs: 8,
  }), 'complete resolved control calibrates the portfolio');
  ok(!api.megaCalibrationClaimPass({ ...pc, evidence_manifest: '' }, {
    expected_pct_lo: -9, expected_pct_hi: -4,
  }), 'missing evidence manifest fails calibration');
}

console.log(failures === 0
  ? '\nPASS: one Mega candidate registry works with Expert Skills on or off.'
  : `\nFAIL: ${failures} assertion(s)`);
process.exit(failures === 0 ? 0 : 1);
