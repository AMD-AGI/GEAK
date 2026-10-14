#!/usr/bin/env node
'use strict';
// Execute the actual dispatch branches: AgentX must reach extraction from baseline
// evidence, with zero profiler calls, including refreshes and baseline-only resumes.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
new AsyncFunction('args', src.replace('export const meta', 'const meta'));

function between(start, end, from = 0) {
  const a = src.indexOf(start, from), b = src.indexOf(end, a);
  assert(a >= 0 && b > a, `source block exists: ${start}`);
  return src.slice(a, b);
}
const schemas = between('const obj =', '// Result of the ensure_flydsl');
const selectionSchema = new Function(`${schemas}\nreturn AGENTX_SELECTION_SCHEMA;`)();
const profileSchema = new Function(`${schemas}\nreturn PROFILE_SCHEMA;`)();
const helper = between('function groupAgentXExtractionTasks(', "\nif (want('setup'))");
const initial = between("  if (searchClosed('Profile and Strategize')) {",
  '} else {\n  // Load carried state');
const resume = between('  // Load carried state', '\n}\n\n// ===');
const config = between('    let restrat;', '    // re-strategize may');
const tuning = between("    if (!searchClosed('post-tuning re-profile and re-strategize')) {", '  } else if (tuned)');
const deep = between('        let rp;', '        if (rp && Array.isArray(rp.heads))');
const head = between("  if (acceptedHeads.length && !searchClosed('post-head re-profile')) {\n    if (KERNEL_EXTRACT_ON)",
  '  log(`Head-kernel track done.');
const milestoneRefresh = between('    if (KERNEL_EXTRACT_ON) {\n      profile = await selectAgentXKernels(milestone',
  '  } else {\n    noImprove++');
const milestonePlan = between('  let plan;\n  if (KERNEL_EXTRACT_ON)', '\n  const planCandsRaw');
const milestoneSchedule = between('  const cands = planCands.slice(0, remaining)',
  '  // --- (b) PARALLEL optimize');
const carry = between('const carryState = {', '\n// What this run got from the KB');

function candidate(name) {
  return { short_name: name, pct_gpu_time: 99, entity_kind: 'gpu_kernel', device_kernel: name, op_kind: 'attn',
    kernel_extraction: { workload_json: '/eval/kernel_extraction/workload.json',
      ranking_json: '/eval/kernel_extraction/ranking.json', selection_pct_gpu_time: 7.25 } };
}
function selection() {
  return { head_candidates: [candidate('head')], kernel_candidates: [candidate('kernel')],
    profile_workload_json: '/eval/kernel_extraction/workload.json',
    kernel_ranking_json: '/eval/kernel_extraction/ranking.json', config_directions: [] };
}
async function run(body, overrides = {}, reply = selection()) {
  const calls = [], phases = [];
  const env = {
    KERNEL_EXTRACT_ON: true, EVAL_DIR: '/eval', EVAL_DIR_OVERRIDE: '/eval',
    MODEL_PATH: '/models/model', MODEL_NAME: 'model', MODEL_NAME_HINT: 'model',
    TASK: 'Review extraction coverage using /references/ut_delivery; preserve all observed families.',
    GPU_LIST: ['0'], SERVING_GPU: '0,1,2,3,4,5,6,7', WORKFLOW_DIR: '/wf',
    WORKLOAD: { kind: 'agentx_trace_replay' }, WORKLOAD_KIND: 'agentx_trace_replay', WORKLOAD_SPEC: {},
    BASELINE_TPUT: 400, NOISE_BAND: 0.5, NOISE_BAND_DEFAULT: 0.5,
    curFlags: '--tp 8', curEnv: 'MODE=deployment', curOverlay: '/accepted-overlay', curTput: 440,
    INIT_BASE_OVERLAY: '', kbSeedTput: 0, BUDGET: 2, HEAD_BUDGET: 1,
    HEAD_THRESHOLD_PCT: 5, MILESTONE_MIN_PCT: 5, CONFIG_TUNE_ENABLED: false,
    TRACELENS_INPUTS: {}, ANALYSIS_SKILL_INPUTS: {}, KB_REF_INPUTS: {},
    profile: { profile_workload_json: '/old/workload.json', kernel_ranking_json: '/old/ranking.json' },
    strategy: null, kernelQueue: [], headQueue: [], history: { ledger: [] },
    ST: {}, e2eGateCount: 2, milestone: 2, remaining: 1, belowFloor: false,
    MIN_KERNEL_TASKS: 1, dispatched: 1, acceptedHeads: [{}], acceptedKernels: [],
    flaggedHeads: [], pendingIntegrations: [], tuning: null, BACKEND: 'sglang',
    MEASUREMENT_MODE: 'isolated_server', VALIDATION_MEASUREMENT_MODE: 'isolated_server',
    VALIDATION_REPLICAS: 1, PLAN_SCHEMA: {},
    searchClosed: () => false, gateNoiseBand: x => x, remainingMin: () => 120, log: () => {}, phase: p => phases.push(p), want: () => true,
    roleAgent: (role, phase, intro, inputs) => ({ role, phase, intro, inputs }),
    safeAgent: async (prompt, opts) => { calls.push({ ...prompt, opts }); return structuredClone(reply); },
    applyOpIdentityGuard: q => q, admitHeads: q => q, ensureFlydslGate: async () => {},
    ...overrides,
  };
  const result = await new AsyncFunction('env', `let { ${Object.keys(env).join(',')} } = env;
    ${schemas}\n${helper}\n${body}
    return {profile, strategy, headQueue, kernelQueue,
      plan: typeof plan === 'undefined' ? null : plan,
      scheduled: typeof cands === 'undefined' ? null : cands,
      state: typeof carryState === 'undefined' ? null : carryState};`)(env);
  return { ...result, calls, phases };
}

(async () => {
  const { validate } = await import('../../interface/runtime/engine/schema.mjs');
  for (const round of ['baseline', 'resume', 'config', 'tuning', 'head', 'deep-1', 0, 2]) {
    const verdict = validate({ ...selection(), round }, selectionSchema);
    assert(verdict.ok, `AgentX selection accepts round ${JSON.stringify(round)}: ${verdict.errors}`);
  }
  for (const round of [null, {}, [], true]) {
    assert(!validate({ ...selection(), round }, selectionSchema).ok,
      'round must still be a named phase or numeric milestone');
  }
  assert(!validate({ round: 'resume', profile_topN_json: '/p', top_kernels: [] }, profileSchema).ok,
    'fixed-load profile schema remains unchanged');
  for (const key of ['head_candidates', 'kernel_candidates']) {
    for (const field of ['short_name', 'device_kernel', 'entity_kind', 'kernel_extraction']) {
      const bad = selection();
      delete bad[key][0][field];
      assert(!validate(bad, selectionSchema).ok, `${key} requires ${field} before routing`);
    }
    const bad = selection(); bad[key][0].entity_kind = 'op';
    assert(!validate(bad, selectionSchema).ok, 'logical ops cannot masquerade as GPU candidates');
    const missingPath = selection(); delete missingPath[key][0].kernel_extraction.workload_json;
    assert(!validate(missingPath, selectionSchema).ok, 'attributed cost must link its workload evidence');
    const badSeam = selection(); badSeam[key][0].seam_candidates = [{
      target_callable: 'pkg:call', depth: 1, role: 'dispatch_entry', device_kernels: ['kernel'],
    }];
    assert(!validate(badSeam, selectionSchema).ok, 'seams must use the existing machine-readable roles');
  }
  const noKind = selection(); delete noKind.head_candidates[0].op_kind;
  assert(!validate(noKind, selectionSchema).ok, 'head extraction needs op_kind');
  const first = await run(initial);
  assert.deepEqual(first.phases, ['KernelExtract']);
  assert.deepEqual(first.calls.map(c => c.role), ['kernel_extractor']);
  const input = first.calls[0].inputs;
  assert.equal(first.calls[0].phase, 'select');
  assert.equal(input.BASELINE_LOG, '/eval/baseline/server.log');
  assert.equal(input.BASELINE_BENCH_DIR, '/eval/baseline');
  assert.equal(input.CURRENT_FLAGS, '--tp 8');
  assert.equal(input.CURRENT_OVERLAY, '/accepted-overlay');
  assert.equal(input.PREVIOUS_RANKING_JSON, '/old/ranking.json');
  assert.equal(input.TASK,
    'Review extraction coverage using /references/ut_delivery; preserve all observed families.',
    'selection receives user scope and delivery references, not just scheduler budgets');
  assert.deepEqual(input.EXTRACTION_SELECTION_POLICY, {
    phases: ['PREFILL', 'DECODE'], top_k_per_phase: 5, overall_pct_gt: 3,
    combination: 'phase_top_k_and_overall_threshold', backfill: false,
  });
  assert.equal(first.headQueue[0].pct_gpu_time, 7.25, 'routing uses attributable share, not window pct');
  assert.equal(first.kernelQueue[0].pct_gpu_time, 7.25);
  const closed = await run(initial, { searchClosed: () => true });
  assert.deepEqual(closed.calls, [], 'a closed search must not start extraction or profiling');
  assert.deepEqual(closed.kernelQueue, []);
  assert.deepEqual(closed.headQueue, []);
  const groupTasks = new Function(`${helper}\nreturn groupAgentXExtractionTasks;`)();
  const decode = { ...candidate('decode_score'), ut_task: 'indexer', served_regimes: ['DECODE'] };
  const prefill = { ...candidate('prefill_score'), ut_task: 'indexer', served_regimes: ['PREFILL'] };
  const duplicateLeafPlan = { ...selection(), head_candidates: [decode], kernel_candidates: [decode, prefill] };
  assert.throws(() => groupTasks(duplicateLeafPlan), /one owner/, 'task name alone cannot certify a joint');
  const owner = { ...decode, selected_device_kernels: ['decode_score', 'prefill_score'],
    extraction_members: [decode, prefill], joint_ut: { phases: [
      { regime: 'DECODE', device_kernel: 'decode_score', required_device_kernels: ['decode_score', 'decode_topk'],
        seam_candidates: [{ target_callable: 'pkg:decode', depth: 0, role: 'op_seam', device_kernels: ['decode_score', 'decode_topk'] }] },
      { regime: 'PREFILL', device_kernel: 'prefill_score', required_device_kernels: ['prefill_score', 'prefill_topk'],
        seam_candidates: [{ target_callable: 'pkg:prefill', depth: 0, role: 'op_seam', device_kernels: ['prefill_score', 'prefill_topk'] }] },
    ] } };
  const groupedPlan = { ...selection(), head_candidates: [owner], kernel_candidates: [decode, prefill] };
  const grouped = groupTasks(groupedPlan);
  assert.equal(grouped.head_candidates.length, 1);
  assert.equal(grouped.kernel_candidates.length, 0, 'joint leaves cannot be dispatched again');
  assert.equal(grouped.head_candidates[0].kernel_extraction.selection_pct_gpu_time, 7.25,
    'grouping must not sum overlapping cost scopes');
  assert.equal(grouped.head_candidates[0].extraction_members.length, 2);
  assert.deepEqual(groupTasks(grouped), grouped, 'saved grouped queues remain stable');
  const dispatchOwner = { ...owner, joint_ut: undefined, op_kind: 'moe', dispatch_ut: {
    phases: owner.joint_ut.phases.map(p => ({ ...p, required_device_kernels: undefined })) } };
  const dispatchedGroup = groupTasks({ ...selection(), head_candidates: [dispatchOwner], kernel_candidates: [] });
  assert.equal(dispatchedGroup.head_candidates.length, 1);
  assert.equal(dispatchedGroup.head_candidates[0].joint_ut, undefined,
    'mutually exclusive selected kernels do not become a mandatory joint chain');
  dispatchOwner.dispatch_ut.phases.pop();
  assert.throws(() => groupTasks({ head_candidates: [dispatchOwner] }), /dispatch phases/);
  const companionPlan = structuredClone(grouped);
  const companion = { ...candidate('prefill_topk'), kernel_extraction: {
    ...candidate('prefill_topk').kernel_extraction, selected: false, selection_pct_gpu_time: 1.7 } };
  companionPlan.head_candidates[0].extraction_members.push(companion);
  const withCompanion = groupTasks(companionPlan);
  assert.equal(withCompanion.head_candidates[0].extraction_members.length, 3);
  assert.equal(withCompanion.head_candidates[0].selected_device_kernels.length, 2,
    'required companions are delivered without inflating the selected target count');
  companionPlan.head_candidates[0].selected_device_kernels.push('prefill_topk');
  assert.throws(() => groupTasks(companionPlan), /non-selected companion/);
  const missingPhase = structuredClone(groupedPlan);
  missingPhase.head_candidates[0].joint_ut.phases.pop();
  assert.throws(() => groupTasks(missingPhase), /every selected member/);
  const missingMember = structuredClone(groupedPlan);
  missingMember.head_candidates[0].selected_device_kernels.push('lost_kernel');
  assert.throws(() => groupTasks(missingMember), /original evidence/);
  let groupAttempts = 0;
  const repairedGroup = await run(initial, { safeAgent: async () => structuredClone(
    ++groupAttempts === 1 ? duplicateLeafPlan : groupedPlan) });
  assert.equal(groupAttempts, 2);
  assert.equal(repairedGroup.headQueue.length, 1);
  assert.equal(repairedGroup.kernelQueue.length, 0);
  let companionAttempts = 0;
  const repairedCompanion = await run(initial, { safeAgent: async () => structuredClone(
    ++companionAttempts === 1 ? companionPlan : withCompanion) });
  assert.equal(companionAttempts, 2);
  assert.equal(repairedCompanion.headQueue[0].selected_device_kernels.length, 2);
  assert.equal(repairedCompanion.headQueue[0].extraction_members.length, 3,
    'repairing target classification retains the companion delivery obligation');
  assert(!first.calls[0].opts.schema.required.includes('profile_topN_json'),
    'no pre-existing full profile is required');

  // Discovery may be larger than the executable batch. The production selector
  // and carried state must retain eligible work when both task budgets are one.
  const broadSelection = selection();
  broadSelection.head_candidates = ['moe', 'decode_score', 'prefill_score'].map(candidate);
  broadSelection.kernel_candidates = ['topk', 'sparse_decode', 'sparse_prefill', 'rmsnorm'].map(candidate);
  const broad = await run(initial + carry, { BUDGET: 1, HEAD_BUDGET: 1 }, broadSelection);
  assert.equal(broad.headQueue.length, 3);
  assert.equal(broad.kernelQueue.length, 4);
  const broadResume = await run(resume, { ST: broad.state, BUDGET: 1, HEAD_BUDGET: 1 });
  assert.equal(broadResume.calls.length, 0, 'saved inventory queues do not require rerunning selection');
  assert.deepEqual(broadResume.headQueue.map(c => c.short_name), broad.headQueue.map(c => c.short_name));
  assert.deepEqual(broadResume.kernelQueue.map(c => c.short_name), broad.kernelQueue.map(c => c.short_name));
  const limitedBatch = await run(milestonePlan + milestoneSchedule + carry, {
    BUDGET: 1, remaining: 1, kernelQueue: broad.kernelQueue,
    planCands: broad.kernelQueue,
  });
  assert.equal(limitedBatch.scheduled.length, 1, 'execution still honors the task budget');
  assert.deepEqual(limitedBatch.kernelQueue.map(c => c.short_name),
    ['sparse_decode', 'sparse_prefill', 'rmsnorm'], 'budget slicing does not discard unexecuted work');
  const limitedResume = await run(resume, { ST: limitedBatch.state, BUDGET: 1 });
  assert.deepEqual(limitedResume.kernelQueue.map(c => c.short_name),
    limitedBatch.kernelQueue.map(c => c.short_name));
  const legacyBatch = await run(milestoneSchedule, {
    KERNEL_EXTRACT_ON: false, remaining: 1, kernelQueue: [candidate('legacy')],
    planCands: broad.kernelQueue,
  });
  assert.equal(legacyBatch.kernelQueue[0].short_name, 'legacy', 'fixed-load queue semantics stay unchanged');

  const fixed = await run(initial, { KERNEL_EXTRACT_ON: false });
  assert.deepEqual(fixed.phases, ['Profile', 'Strategize']);
  assert.deepEqual(fixed.calls.map(c => [c.role, c.phase]),
    [['profiler', 'baseline'], ['system_architect', 'strategize']]);
  assert.equal(fixed.calls[0].inputs.OVERLAY_PYTHONPATH, '/accepted-overlay');

  for (const [name, body, expectedFixedCalls] of [
    ['config', config, 2], ['tuning', tuning, 2], ['deep', deep, 1],
    ['head', head, 1], ['milestone refresh', milestoneRefresh, 1],
    ['milestone plan', milestonePlan, 1],
  ]) {
    const actual = await run(body);
    assert.deepEqual(actual.calls.map(c => c.role), ['kernel_extractor'], name);
    assert.deepEqual(actual.phases, ['KernelExtract'], name);
    assert.equal(actual.calls[0].inputs.CURRENT_THROUGHPUT, 440, name);
    assert.equal(actual.calls[0].inputs.CURRENT_OVERLAY, '/accepted-overlay', name);
    assert.equal(actual.calls[0].inputs.TASK, input.TASK, `${name} preserves extraction scope`);
    assert.deepEqual(actual.calls[0].inputs.EXTRACTION_SELECTION_POLICY,
      input.EXTRACTION_SELECTION_POLICY, `${name} preserves the per-phase extraction rule`);
    const old = await run(body, { KERNEL_EXTRACT_ON: false });
    assert.equal(old.calls.length, expectedFixedCalls, name);
    assert(old.calls.every(c => ['profiler', 'system_architect'].includes(c.role)), name);
  }
  const queued = await run(milestonePlan, { kernelQueue: [candidate('saved')] });
  assert.equal(queued.calls.length, 0, 'consume saved candidates without rediscovery');
  assert.equal(queued.plan.kernel_candidates[0].short_name, 'saved');

  const baselineOnly = { eval_dir: '/saved', baseline_throughput_tok_s: 400 };
  const resumed = await run(resume, { ST: baselineOnly });
  assert.deepEqual(resumed.calls.map(c => c.role), ['kernel_extractor']);
  assert.equal(resumed.calls[0].inputs.BASELINE_LOG, '/saved/baseline/server.log');
  assert.equal(resumed.headQueue.length, 1, 'baseline-only resume can enter head extraction');
  const saved = await run(initial + carry);
  const restored = await run(resume, { ST: saved.state });
  assert.equal(restored.calls.length, 0, 'complete selection survives resume');
  assert.equal(restored.profile.profile_workload_json, selection().profile_workload_json);
  assert.equal(restored.profile.kernel_ranking_json, selection().kernel_ranking_json);
  const emptySaved = await run(resume, { ST: { ...baselineOnly, headQueue: [], kernelQueue: [] } });
  assert.equal(emptySaved.calls.length, 0, 'known empty queues are not a missing selection');
  const fixedResume = await run(resume, { KERNEL_EXTRACT_ON: false, ST: baselineOnly });
  assert.equal(fixedResume.calls.length, 0, 'fixed-load resume adds no calls');
  assert.deepEqual(fixedResume.profile, { profile_topN_json: '' });

  for (const bad of [null, {}, { ...selection(), kernel_ranking_json: '' },
    { ...selection(), kernel_candidates: [{ short_name: 'unmeasured', pct_gpu_time: 80 }] },
    ...[NaN, -1, 101, '7.25'].map(share => {
      const result = selection(); result.head_candidates[0].kernel_extraction.selection_pct_gpu_time = share;
      return result;
    }),
  ]) {
    await assert.rejects(run(initial, {}, bad), /AgentX kernel selection failed/);
  }
  // Exercise the real extraction function: authoring can overlap even for a shared
  // GPU (execution locks live at the command boundary), and failed UTs stay local.
  const extraction = between('async function extractWithBaseline(', '\n// abDone ==');
  const makeDispatch = new Function('safeAgent', `
    const CAPTURE_STORAGE_ENV = '', CAPTURE_BYTE_BUDGET = 1, CAPTURE_CASE_BYTE_LIMIT = 1;
    const CAPTURE_PERSIST_POLICY = '', BASELINE_EXTRACT_RETRIES = 0;
    const roleAgent = (role, phase, intro, inputs) => ({ role, phase, intro, inputs });
    const hasFrozenBaseline = ext => ext && ext.baseline_frozen === true;
    const kernelSelectionVerified = () => ({ ok: true });
    const hydrateExtractionVerdicts = async ext => ext;
    const log = () => {};
    ${extraction}
    return extractWithBaseline;
  `);
  for (const gpu of ['0', '1']) {
    const starts = [], releases = [];
    const dispatch = makeDispatch(prompt => {
      starts.push(prompt.role);
      return new Promise(resolve => releases.push(resolve));
    });
    const first = dispatch('first', 'extract_op', '', { GPU_ID: '0' });
    const second = dispatch('second', 'extract', '', { GPU_ID: gpu });
    const completion = Promise.allSettled([first, second]);
    assert.deepEqual(starts, ['first', 'second'], 'authoring does not wait for a GPU verdict');
    releases[0]({ smoke: 'fail', smoke_detail: 'GPU correctness failed', task_dir: '/task' });
    releases[1]({ unittest_smoke: 'pass', task_dir: '/task2', baseline_frozen: true });
    const results = await completion;
    assert(results.every(r => r.status === 'fulfilled'), 'failure does not reject the batch');
    assert.equal(results[0].value.smoke, 'fail');
    assert.equal(results[1].value.unittest_smoke, 'pass');
    const third = dispatch('third', 'extract', '', { GPU_ID: '0' });
    releases[2](null);
    assert.equal(await third, null, 'missing results remain local and reach caller skip guards');
  }
  const noBaseline = makeDispatch(async () => ({ unittest_smoke: 'pass', task_dir: '/task' }));
  assert.equal((await noBaseline('kernel_extractor', 'extract', '', {})).unittest_smoke, 'fail',
    'batch authoring still requires a frozen baseline before admission');
  // A live resume omitted a dispatcher marker, although its selected op seam was
  // correct. Repairing probe coverage must not force selection of a wrong seam.
  const identity = between('const CALLABLE_SPEC_RX =', 'const PRE_FLAGGED_HEADS =');
  const retryPrompts = [];
  const target = 'pkg.live:op', dispatcher = 'pkg.live:compile';
  const selectedHead = { entity_kind: 'gpu_kernel', device_kernel: 'device_kernel',
    seam_candidates: [
      { target_callable: target, role: 'op_seam', depth: 1, device_kernels: ['device_kernel'] },
      { target_callable: dispatcher, role: 'dispatcher', depth: 2, device_kernels: ['device_kernel'] },
    ] };
  const retryDispatch = new Function('safeAgent', `
    const CAPTURE_STORAGE_ENV = '', CAPTURE_BYTE_BUDGET = 1, CAPTURE_CASE_BYTE_LIMIT = 1;
    const CAPTURE_PERSIST_POLICY = '', BASELINE_EXTRACT_RETRIES = 1;
    const roleAgent = (role, phase, intro, inputs) => ({ role, phase, intro, inputs });
    const hasFrozenBaseline = ext => ext && ext.baseline_frozen === true;
    const log = () => {};
    const require = () => { throw new Error('no capture files in this CPU fixture'); };
    ${identity}
    ${extraction}
    return extractWithBaseline;
  `)(async prompt => {
    retryPrompts.push(prompt);
    return { smoke: 'pass', task_dir: '/fixture', baseline_frozen: true,
      target_callable: target, selection_validation: {
        contract: 'kernel_selection', ok: true, target_callable: target,
        device_kernel: 'device_kernel', total_calls_observed: 1,
        target_marker_calls: 1, matched_kernel_calls: 1, deepest_verified: true,
        live_candidate_targets: [target], deeper_live_candidates: [],
        candidate_targets_tested: retryPrompts.length === 1 ? [target] : [target, dispatcher],
      } };
  });
  const repaired = await retryDispatch('kernel_extractor', 'extract_op', '', { KERNEL: selectedHead });
  assert.equal(retryPrompts.length, 2, 'real selection gate requires the omitted dispatcher probe');
  assert.equal(repaired.smoke, 'pass', 'complete new evidence can certify the same target');
  assert.deepEqual(retryPrompts[1].inputs.ATTEMPTED_TARGET_CALLABLES, [target]);
  assert(retryPrompts[1].intro.includes('not a target blacklist'));
  assert(retryPrompts[1].intro.includes('new runtime evidence resolves every failed selection check'));
  const checkSelection = new Function(`${identity}\nreturn kernelSelectionVerified;`)();
  function jointPhase(regime) {
    const prefix = regime.toLowerCase(), target = `pkg.indexer:${prefix}_joint`;
    const anchor = `${prefix}_score_kernel`, companion = `${prefix}_topk_kernel`;
    const leaf = `pkg.indexer:${prefix}_score_leaf`;
    const seams = [
      { target_callable: target, role: 'op_seam', depth: 1, device_kernels: [anchor, companion] },
      { target_callable: leaf, role: 'inner_launcher', depth: 2, device_kernels: [anchor] },
    ];
    return {
      spec: { regime, device_kernel: anchor, required_device_kernels: [anchor, companion], seam_candidates: seams },
      result: { regime, target_callable: target, seam_candidates: seams, selection_validation: {
        contract: 'kernel_selection', ok: true, target_callable: target, device_kernel: anchor,
        total_calls_observed: 1, target_marker_calls: 1, matched_kernel_calls: 1,
        deepest_verified: true, live_candidate_targets: [target], deeper_live_candidates: [],
        candidate_targets_tested: [target, leaf], selection_scope: 'joint_chain',
        required_device_kernels: [anchor, companion],
        chain_members: [anchor, companion].map(device_kernel => ({ device_kernel, matched_kernel_calls: 1 })),
        joint_call_coverage: [{ call: 0, missing_members: [] }],
      } },
    };
  }
  const jointPhases = ['DECODE', 'PREFILL'].map(jointPhase);
  const jointHead = { entity_kind: 'gpu_kernel', device_kernel: 'decode_score_kernel',
    op_kind: 'attn', joint_ut: { phases: jointPhases.map(p => p.spec) } };
  const prepareSelection = new Function(`${identity}\nreturn prepareHeadSelection;`)();
  assert.equal(prepareSelection(jointHead).target_callable, 'pkg.indexer:decode_joint',
    'queue admission preserves the complete joint boundary despite a deeper score leaf');
  const jointResult = { ...jointPhases[0].result,
    phase_selection_validations: jointPhases.map(p => p.result) };
  assert(checkSelection(jointHead, jointResult).ok, 'both whole-chain phase verdicts pass');
  const makeJointExtract = reply => new Function('safeAgent', `
    const CAPTURE_STORAGE_ENV = '', CAPTURE_BYTE_BUDGET = 1, CAPTURE_CASE_BYTE_LIMIT = 1;
    const CAPTURE_PERSIST_POLICY = '', BASELINE_EXTRACT_RETRIES = 1, JOINT_BOOTSTRAP_SCHEMA = {};
    const roleAgent = (role, phase, intro, inputs) => ({ role, phase, intro, inputs });
    const hasFrozenBaseline = ext => ext && ext.baseline_frozen === true;
    const log = () => {};
    ${identity}\n${extraction}\nreturn extractWithBaseline;
  `)(reply);
  const bootstrap = { ready: true, task_dir: '/joint', report_path: '/joint/replay.json',
    phase_results: jointPhases.map(p => ({ regime: p.spec.regime,
      target_callable: p.result.target_callable, exit_code: 0, baseline_replay_passed: true })) };
  const stageCalls = [];
  const staged = await makeJointExtract(async p => {
    stageCalls.push(p);
    return stageCalls.length === 1 ? bootstrap : { ...jointResult, task_dir: '/joint',
      smoke: 'pass', baseline_frozen: true };
  })('kernel_extractor', 'extract_op', '', { KERNEL: jointHead });
  assert.equal(stageCalls.length, 2);
  assert.equal(stageCalls[0].inputs.AUTHORING_STAGE, 'joint_bootstrap');
  assert.equal(stageCalls[1].inputs.AUTHORING_STAGE, 'complete_joint_ut');
  assert.equal(stageCalls[1].inputs.PRIOR_TASK_DIR, '/joint');
  assert(stageCalls[1].intro.includes('each phase capture directory separately'));
  assert(stageCalls[1].intro.includes('Preserve one replayable oracle per phase'));
  assert.equal(staged.smoke, 'pass');
  const jointRetryCalls = [];
  const retriedJoint = await makeJointExtract(async p => {
    jointRetryCalls.push(p);
    const result = structuredClone(jointResult);
    if (jointRetryCalls.length === 1) result.phase_selection_validations.pop();
    return { ...result, task_dir: '/joint', smoke: 'pass', baseline_frozen: true };
  })('kernel_extractor', 'extract_op', '', { KERNEL: jointHead, JOINT_BOOTSTRAP: bootstrap });
  assert.equal(retriedJoint.smoke, 'pass');
  assert.equal(jointRetryCalls.length, 2);
  assert(jointRetryCalls[1].intro.includes('Preserve the passing per-phase bootstrap inputs'));
  assert(!jointRetryCalls[1].intro.includes('--cleanup-task-dir "$TASK" --no-promote'),
    'joint retries must retain all passing bootstrap oracles');
  let incompleteCalls = 0;
  const incomplete = await makeJointExtract(async () => {
    incompleteCalls++;
    return { ...bootstrap, phase_results: bootstrap.phase_results.slice(0, 1) };
  })('kernel_extractor', 'extract_op', '', { KERNEL: jointHead });
  assert.equal(incompleteCalls, 1);
  assert.equal(incomplete.smoke, 'fail', 'one phase cannot pass a joint bootstrap');
  const fullFailure = await makeJointExtract(async () => ({ ...jointResult, task_dir: '/joint',
    smoke: 'fail', baseline_frozen: true }))('kernel_extractor', 'extract_op', '',
      { KERNEL: jointHead, JOINT_BOOTSTRAP: bootstrap });
  assert.equal(fullFailure.smoke, 'fail', 'a passing bootstrap never substitutes for full UT gates');
  for (const change of [
    r => r.phase_selection_validations.pop(),
    r => r.phase_selection_validations[1].selection_validation.chain_members.pop(),
    r => r.phase_selection_validations[0].selection_validation.joint_call_coverage[0].missing_members.push('topk'),
    r => r.phase_selection_validations[1].selection_validation.selection_scope = 'single_kernel',
    r => r.phase_selection_validations[0].selection_validation.deeper_live_candidates.push('pkg:deeper_joint'),
  ]) {
    const bad = structuredClone(jointResult); change(bad);
    assert(!checkSelection(jointHead, bad).ok, 'partial or invalid joint evidence must fail');
  }
  assert(!checkSelection({ ...jointHead, joint_ut: undefined,
    seam_candidates: jointPhases[0].spec.seam_candidates }, jointResult).ok,
    'a joint scope cannot silently weaken a single-kernel selection gate');
  console.log('PASS: baseline-to-skill dispatch, all six refresh/planning branches, evidence gates and resumes; fixed-load dispatch preserved.');
})().catch(e => { console.error(e); process.exitCode = 1; });
