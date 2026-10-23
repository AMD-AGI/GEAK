#!/usr/bin/env node
'use strict';

// Regression tests for the Mega feedback-loop fixes: Verify time is reserved before the Author is
// sized, absolute scores stay frozen-relative, one AUTHOR_GPU_MODE decides Author GPU use, stalls are
// counted by runtime evidence, lanes can be parked or rewound, and analysis is identity-bound.

const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_mega.js'), 'utf8');
const lead = fs.readFileSync(path.resolve(__dirname, '..', 'roles', 'mega', 'mega_search_lead.md'), 'utf8');
const verifier = fs.readFileSync(path.resolve(__dirname, '..', 'roles', 'mega', 'verify_engineer.md'), 'utf8');
const analysisRole = fs.readFileSync(
  path.resolve(__dirname, '..', 'roles', 'mega', 'analysis_engineer.md'), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const region = (name) => {
  const m = src.match(new RegExp(`// <<REPLAY:${name}>>([\\s\\S]*?)// <</REPLAY:${name}>>`));
  if (!m) throw new Error(`missing ${name} replay region`);
  return m[1];
};

const api = new Function(`
  const BIMODAL_GUARDS = [];
  function guardContract(v, targets, regressions) {
    const rows = new Map((v.per_case || []).map((r) => [r.name, r]));
    return { complete: targets.every((g) => rows.has(g)), regression_pass: true };
  }
  function promotionScore(v, metric, targets, regressions, fallback) {
    if (!targets.length) return Number(fallback || 0);
    const row = (v.per_case || []).find((r) => r.name === targets[0]);
    return row ? Number(row.speedup) : 0;
  }
  ${region('mega_topology_contract')}
  ${region('mega_candidate_registry')}
  return { megaFrozenPerCase, megaCandidateFromVerification, normalizeMegaCandidate,
    normalizeContractFailures, megaRegistryForSearch, megaFinalSelectionVerdict };
`)();

const FROZEN = [{ name: '8192_uniform', latency_ms: 4.6991 }];
const verification = (baseMs, candMs) => ({
  status: 'verified', claim_complete: true, attempt_id: 'lane:r9:a1', candidate_head: 'abc1234',
  evidence_manifest: '/eval/lane/verify/evidence_manifest.json',
  per_case: [{ name: '8192_uniform', baseline_ms: baseMs, optimized_ms: candMs,
    speedup: baseMs / candMs }],
  paired_readings: Array.from({ length: 5 }, (_, i) => ({
    guard: '8192_uniform', base: baseMs + i * 0.001, cand: candMs + i * 0.001,
  })),
  null_arm_pct: 0.1, reps: 5,
  accuracy_results: [{ guard: '8192_uniform', metric: 'relL2', value: 0.03, threshold: 0.1 }],
  correctness: 'pass', activation_confirmed: 'yes', activation_on_hardware: 'yes',
  launch_shape: { launches_base: 4, launches_cand: 2 },
  liveness: 'pass', replay_count: 30, graph_safe: 'pass',
  artifact_distinct: 'yes', artifact_hash_base: 'base', artifact_hash_candidate: 'cand',
  component_breakdown: [{ component: 'dispatch_copy', busy_ms: 6.1, wait_ms: 0.2,
    method: 'ablation', evidence: '/eval/ablation.json' }],
  hot_path_counters: { stage2_atomics_per_m_block: 64512 },
});
const opts = {
  targetGuards: ['8192_uniform'], regressionGuards: [], launchTarget: 2,
  promotionMetric: 'operator_e2e', accuracyMetric: 'relL2', accuracyThreshold: 0.1,
  requiredReplays: 30, requiredPairs: 5, allowPartialFusion: false, frozenBaseline: FROZEN,
};
const meta = { id: 'lane', source: 'search', status: 'authoring', tree: '/state/lane',
  head: 'abc1234', attempt_id: 'lane:r9:a1', next_blocker: 'tune payload chunk' };

console.log('\n# absolute score is frozen-relative');
{
  const fix = api.megaFrozenPerCase(verification(11.0285, 12.8236).per_case, FROZEN, 0.05);
  ok(fix.mismatch.length === 1 && fix.mismatch[0].deviation_pct > 100 &&
     Math.abs(fix.perCase[0].speedup - 4.6991 / 12.8236) < 1e-9 &&
     Math.abs(fix.perCase[0].switch_speedup - 11.0285 / 12.8236) < 1e-9,
    'a same-tree switch-off base 134% above frozen is re-based and kept as switch_speedup');
  const rec = api.megaCandidateFromVerification(meta, verification(11.0285, 12.8236), opts);
  ok(Math.abs(rec.absolute_score - 0.3664) < 0.001,
    'the recorded absolute score is 4.70/12.82 = 0.366x, not the same-tree 0.86x');
  ok(rec.denominator_mismatch.length === 1 && rec.next_blocker.startsWith('DENOMINATOR:') &&
     rec.next_blocker.includes('tune payload chunk'),
    'the mismatch is recorded and leads the next blocker without dropping the prior handoff');
  ok(rec.measurement_pass === false && rec.status !== 'scored',
    'paired readings taken against the drifted arm cannot certify the re-based score');
  const planner = api.megaRegistryForSearch([api.normalizeMegaCandidate(rec)])[0];
  ok(planner.denominator_mismatch.length === 1 &&
     planner.component_breakdown[0].component === 'dispatch_copy' &&
     planner.hot_path_counters.stage2_atomics_per_m_block === 64512,
    'the Planner receives the mismatch, component breakdown and hot-path counters');
  const healthy = api.megaCandidateFromVerification(meta, verification(4.72, 4.50), opts);
  ok(healthy.denominator_mismatch.length === 0 &&
     Math.abs(healthy.absolute_score - 4.72 / 4.50) < 1e-9 && !/DENOMINATOR/.test(healthy.next_blocker),
    'a base arm within 5% of frozen keeps the concurrent paired score untouched');
}

console.log('\n# lane states and contract tiers survive normalization');
{
  ok(api.normalizeMegaCandidate({ id: 'old_lane', source: 'search', status: 'parked' }).status ===
     'parked', 'parked is a first-class, resumable lane status');
  const failuresN = api.normalizeContractFailures([
    { id: 'flat_stripe_completion', severity: 'required', tier: 'semantic', messages: ['x'] },
    { id: 'host_spec_names', severity: 'required', tier: 'surface', messages: ['y'] },
    { id: 'legacy_check', severity: 'required', messages: ['z'] },
  ]);
  ok(failuresN[0].tier === 'semantic' && failuresN[1].tier === 'surface' &&
     failuresN[2].tier === 'semantic',
    'contract failure tiers pass through (missing tier defaults to semantic)');
}

console.log('\n# finalists cannot ship on a drifted denominator');
{
  ok(/megaFrozenPerCase\(r\.per_case, o\.frozenBaseline, 0\.05\)\.mismatch\.length/.test(src) &&
     (src.match(/frozenBaseline: BASELINE_PER_CASE/g) || []).length >= 3,
    'finalist verdicts and round scoring both receive the frozen baseline table');
}

console.log('\n# Verify time is reserved before the Author is sized');
{
  ok(/const MEGA_VERIFY_RESERVE_S = Math\.max\(300, Number\(A\.mega_verify_reserve_s \|\| 1200\)\)/.test(src) &&
     /\(CHECK_EXPERT_SKILL_CONTRACT \? 600 : 0\) \+ MEGA_VERIFY_RESERVE_S/.test(src) &&
     /availableAfterPrepS - Math\.max\(360, 60 \+ MEGA_POST_AUTHOR_RESERVE_S\)/.test(src),
    'the Author budget subtracts the structural charge and the on-card Verify reserve');
  ok(/cannot fit a 900s Author/.test(src), 'a timeout too small for the reserve fails at launch');
  // The same arithmetic the orchestrator performs for mega_candidate_timeout_s=5400.
  const turn = 5400, prep = 90, reserve = 600 + 1200;
  const author = Math.min(Math.floor(turn * 0.85), Math.floor(turn - prep - Math.max(360, 60 + reserve)));
  const verify = Math.min(turn - prep - author - 60, turn - prep - author - 600 - 60);
  ok(author === 3450 && verify === 1200 && verify >= 300,
    'a full Author turn still leaves 1200s of on-card Verify (was 60s, below the 300s gate)');
}

console.log('\n# one AUTHOR_GPU_MODE decides Author GPU use');
{
  ok(/AUTHOR_GPU_MODE: authorGpuProhibited \? 'forbidden' : 'bounded_smoke'/.test(src) &&
     /AUTHOR_GPU_MODE=bounded_smoke: use the supplied EP8 device/.test(src) &&
     /before AND after source edits/.test(src),
    'the Author prompt grants bounded smokes before and after edits');
  ok(!/GPU authorization is revoked/.test(src),
    'no production-fix revocation rule remains in the orchestrator prompt');
  ok(/STRUCTURAL_CONTRACT_POLICY: STRUCTURAL_CONTRACT_BLOCKING \? 'blocking' : 'advisory'/.test(src) &&
     /`STRUCTURAL_CONTRACT_POLICY` is authoritative/.test(lead),
    'the controller contract policy, not TASK prose, decides whether the contract gates the card');
}

console.log('\n# stalls are counted by runtime evidence');
{
  ok(/megaStall = hasEvidence \|\| improved \|\|[\s\S]{0,120}blockerKey !== megaLastBlocker\) \? 0 : megaStall \+ 1;/.test(src),
    'progress = independent Verify, a better score, or an on-card Author result that moved the blocker');
  ok(/MEASURE_FIRST: \$\{n\} rounds produced no new runtime evidence/.test(src) &&
     /MEASURE_FIRST: String\(megaStall\)/.test(src),
    'at the limit the Author must measure the banked HEAD before editing');
  ok(/megaStall >= 2 \* MEGA_STALL_LIMIT[\s\S]{0,260}Mega stall stop/.test(src),
    'twice the limit stops the wave with a resumable lane');
  ok(/MEGA_STALL: \{ rounds: megaStall, limit: MEGA_STALL_LIMIT \}/.test(src) && /MEGA_STALL/.test(lead),
    'the Planner sees the stall count and its rule');
}

console.log('\n# lanes can be parked or rewound, never silently replaced');
{
  ok(/const parkedNow = !!parked && \['authoring', 'runnable'\]\.includes\(parked\.status\)/.test(src) &&
     /status: 'parked'/.test(src) && /const bound = parkedNow \? raw : bindMegaDirectionToWip/.test(src),
    'an explicit park freezes the lane and skips WIP rebinding for that round only');
  ok(/\/\^\[0-9a-f\]\{7,40\}\$\/\.test\(String\(d\.rewind_head \|\| ''\)\)/.test(src) &&
     /git tag -f parked_r\$\{currentRound\} HEAD && git reset --hard/.test(src) &&
     /rewind_head: planned\.rewind_head/.test(src),
    'a rewind accepts only a commit id, survives WIP binding, and keeps the old HEAD tagged');
  ok(/if \(existing && existing\.status === 'parked'\) \{[\s\S]{0,200}status: 'authoring'/.test(src),
    'naming a parked lane resumes it as authoring at the start of its turn');
  ok(/rewind_head/.test(lead) && /park_candidate_id/.test(lead) &&
     /Judge the ARCHITECTURE by its ceiling/.test(lead),
    'the Planner role documents both exits and the phase-aware architecture rule');
}

console.log('\n# WIP continuation resumes from the working checkpoint');
{
  const fn = src.match(
    /function megaWipContinuationDirection\(currentRound\) \{[\s\S]*?\n\}\n/);
  const make = new Function(`
    const megaCandidateRegistry = [{
      id: 'lane', source: 'search', status: 'authoring', base_id: 'frozen_baseline',
      tree: '/state/lane', topology: {}, working_head: 'w1', next_blocker: 'old',
      contract_failures: [],
      working_snapshot: { head: 'w1', topology: { launch_count: 2, included_regions: ['persistent'] },
        contract_failures: [{ id: 'flat_stripe_completion' }], next_blocker: 'restore stripes' },
    }];
    const analysis = { mega_plan_ir: { target: { launch_count: 2 } } };
    const MEGA_DEFAULT_SPECIALTY = 'distributed';
    ${region('mega_topology_contract')}
    ${fn[0]}
    return megaWipContinuationDirection;
  `)();
  const d = make(3);
  ok(d.target_topology.launch_count === 2 &&
     d.target_topology.included_regions[0] === 'persistent' &&
     d.contract_failures[0].id === 'flat_stripe_completion' &&
     /flat_stripe_completion/.test(d.prompt),
    'an empty top-level topology no longer blanks the resumed target_topology');
}

console.log('\n# a right-but-unfinished Planner direction survives WIP binding');
{
  const fns = src.match(
    /function megaWipContinuationDirection\(currentRound\) \{[\s\S]*?\n\}\n\nfunction bindMegaDirectionToWip\(currentRound, direction\) \{[\s\S]*?\n\}\n/);
  const bind = new Function(`
    const megaCandidateRegistry = [{
      id: 'lane', source: 'search', status: 'runnable', base_id: 'frozen_baseline',
      tree: '/state/lane', topology: { launch_count: 2 }, next_blocker: 'measured 0.93x',
      contract_failures: [
        { id: 'progressive_token_readiness', tier: 'semantic' },
        { id: 'combine_launch_bindings', tier: 'surface' },
      ],
    }];
    const analysis = { mega_plan_ir: { target: { launch_count: 2 } } };
    const MEGA_DEFAULT_SPECIALTY = 'distributed';
    ${region('mega_topology_contract')}
    ${fns[0]}
    return bindMegaDirectionToWip;
  `)();
  const kept = bind(4, { id: 'r4', candidate_id: 'lane', title: 'ablate Stage2 publication',
    prompt: 'Time the fused kernel with Stage2 publication compiled out and report busy vs wait.' });
  ok(kept.candidate_id === 'lane' && kept.prompt.startsWith('Time the fused kernel') &&
     /LANE STATE \(lane\): Resolve required semantic contract failures:.*progressive_token_readiness/.test(kept.prompt) &&
     /Surface-tier spelling hints[^.]*combine_launch_bindings/.test(kept.prompt) &&
     !/semantic contract failures:[^\]]*combine_launch_bindings/.test(kept.prompt),
    'the Planner task reaches the Author with the lane state attached; surface failures are hints only');
  const renamed = bind(4, { id: 'r4', candidate_id: 'other_lane', prompt: 'Create other_lane from lane.' });
  const restarted = bind(4, { id: 'r4', candidate_id: 'lane', prompt: 'Restart the kernel from scratch.' });
  ok(renamed.candidate_id === 'lane' && !renamed.prompt.includes('other_lane') &&
     !/from scratch/.test(restarted.prompt),
    'a prompt that names a replacement lane or asks to restart is still dropped');
  const eng = fs.readFileSync(path.resolve(__dirname, '..', 'roles', 'mega', 'engineer.md'), 'utf8');
  ok(/`title`, `focus_files`,\s*`graph_refs` and `roadmap_rung` always reach the Author/.test(lead) &&
     /the title is this turn's task and the prompt is context/.test(eng),
    'roles tell the Planner to name the task in title and the Author to follow it');
}

console.log('\n# analysis and Verify are identity-bound and decompose work vs wait');
{
  ok(/if \(ANALYSIS_SKILL_ON && ver\) profileSummary = \{ \.\.\.profileSummary,[\s\S]{0,200}candidate_evidence: \{[\s\S]{0,80}candidate_id: candidateId, head: record\.head/.test(src),
    'per-round analysis runs only on new Verify evidence and receives this candidate\'s identity');
  ok(/PROFILE_SUMMARY\.candidate_evidence` as the ONLY input/.test(analysisRole) &&
     /do NOT pair two baseline routes/.test(analysisRole),
    'the analysis role no longer scans EVAL_DIR or pairs uniform vs skew baseline routes');
  ok(/const EXPERT_SKILL_RUNTIME_FILE = PINNED_MEGA_SKILL\s*\? `\$\{EXPERT_SKILL_DIR\}\/runtime_validation\.py`/.test(src) &&
     (src.match(/^\s*EXPERT_SKILL_RUNTIME_FILE,$/gm) || []).length === 2 &&
     /--runtime-file EXPERT_SKILL_RUNTIME_FILE/.test(verifier) &&
     !/--runtime-file EXPERT_SKILL_VALIDATION/.test(verifier) &&
     /--frozen-baseline-ms/.test(verifier) && /--resource-evidence/.test(verifier),
    'Verify runs the validator file (not validation.yaml) with the frozen latency and resource evidence');
  ok(/The paired `base` arm is the FROZEN baseline/.test(verifier) &&
     /component_breakdown/.test(verifier) && /hot_path_counters/.test(verifier),
    'Verify pairs against the frozen tree and reports component busy/wait and hot-path counters');
}

console.log(failures === 0
  ? '\nPASS: Mega feedback-loop fixes hold.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
