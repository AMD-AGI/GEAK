#!/usr/bin/env node
'use strict';
// Execute the actual dispatch branches: AgentX must reach extraction from baseline
// evidence once; downstream scheduling consumes prepared UTs without rediscovery.
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
const selectionSchema = new Function(`${schemas}\nreturn AGENTX_PREPARATION_SCHEMA;`)();
const profileSchema = new Function(`${schemas}\nreturn PROFILE_SCHEMA;`)();
const helper = between('async function prepareAgentXKernels(', "\nif (want('setup'))");
const initial = between("  if (searchClosed('Profile and Strategize')",
  '} else {\n  // Load carried state');
const resume = between('  // Load carried state', '\n}\n\n// ===');
const phaseSelector = between('const PHASES =', 'const ST = A.state');
const config = "if (want('config')) {\n" + between('      profile = await safeAgent(',
  '      // re-strategize may', src.indexOf("post-config re-profile and re-strategize")) + '\n}';
const tuning = "if (want('tune')) {\n" + between("    if (!searchClosed('post-tuning re-profile and re-strategize')) {", '  } else if (tuned)') + '\n}';
const deep = between("      if (!KERNEL_EXTRACT_ON && want('profile')", '\n    };');
const head = between("  if (!KERNEL_EXTRACT_ON && acceptedHeads.length && !searchClosed('post-head re-profile'))",
  '  log(`Head-kernel track done.');
const milestoneRefresh = between('  if (milestoneImproved) {', '\n  // --- (d) Update');
const milestonePlan = between('  let plan;\n  if (KERNEL_EXTRACT_ON)', '\n  const planCandsRaw');
const milestoneFilter = between('  const planCandsRaw =', '  if (!planCands.length)');
const milestoneSchedule = between('  const cands = planCands.slice(0, remaining)',
  '  // --- (b) PARALLEL optimize');
const carry = between('const carryState = {', '\n// What this run got from the KB');

function candidate(name) {
  // prepared_ut.result is the RETURN.json the packing adapter already validated. It travels WITH
  // the envelope because the workflow sandbox has no filesystem to re-read it from.
  return { short_name: name, pct_gpu_time: 7.25,
    prepared_ut: { result_path: '/task/run/RETURN.json', sha256: 'hash', phase: 'extract_op',
      result: { task_dir: '/task', target_callable: 'pkg:leaf', op_kind: 'attn',
        baseline_frozen: true, smoke: 'pass', unittest_smoke: 'pass' } },
    entity_kind: 'gpu_kernel', device_kernel: name, op_kind: 'attn',
    kernel_extraction: { workload_json: '/eval/kernel_extraction/workload.json',
      ranking_json: '/eval/kernel_extraction/ranking.json', selection_pct_gpu_time: 7.25 } };
}
function selection() {
  return { head_candidates: [candidate('head')], kernel_candidates: [candidate('kernel')],
    profile_workload_json: '/eval/kernel_extraction/workload.json',
    kernel_ranking_json: '/eval/kernel_extraction/ranking.json', config_directions: [], kernel_extraction_delivery_json: '/eval/delivery.json',
    failed_tasks: [{ short_name: 'failed', reason: 'GPU mismatch' }],
    deferred_tasks: [{ short_name: 'later', reason: 'budget' }], coverage_summary: { tested: 2 } };
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
    ST: {}, A: {}, FAST_MODE: false, DEEP_MODE: false,
    e2eGateCount: 2, milestone: 2, remaining: 1, belowFloor: false,
    milestoneImproved: true, noImprove: 1, lastReprofileTput: 300, DEEP_REPROFILE_GAIN: 0.1, allLanes: [],
    MIN_KERNEL_TASKS: 1, dispatched: 1, acceptedHeads: [{}], acceptedKernels: [],
    flaggedHeads: [], pendingIntegrations: [], tuning: null, BACKEND: 'sglang',
    MEASUREMENT_MODE: 'isolated_server', VALIDATION_MEASUREMENT_MODE: 'isolated_server',
    VALIDATION_REPLICAS: 1, PLAN_SCHEMA: {},
    searchClosed: () => false, searchWindowShort: () => false, gateNoiseBand: x => x, remainingMin: () => 120, log: () => {}, phase: p => phases.push(p),
    roleAgent: (role, phase, intro, inputs) => ({ role, phase, intro, inputs }),
    safeAgent: async (prompt, opts) => { calls.push({ ...prompt, opts }); return prompt.role === 'kernel_extractor' ? { delivery_path: '/eval/delivery.json', prepared: structuredClone(reply) } : structuredClone(reply); },
    readAgentXPreparation: async () => structuredClone(reply),
    applyOpIdentityGuard: q => q, admitHeads: q => q, ensureFlydslGate: async () => {},
    ...overrides,
  };
  const result = await new AsyncFunction('env', `let { ${Object.keys(env).join(',')} } = env;
    ${schemas}\n${phaseSelector}\n${helper}\n${body}
    return {profile, strategy, headQueue, kernelQueue,
      plan: typeof plan === 'undefined' ? null : plan,
      scheduled: typeof cands === 'undefined' ? null : cands,
      state: typeof carryState === 'undefined' ? null : carryState};`)(env);
  return { ...result, calls, phases };
}

(async () => {
  const { validate } = await import('../../interface/runtime/engine/schema.mjs');
  assert(validate({ delivery_path: '/eval/delivery.json', prepared: selection() }, selectionSchema).ok);
  assert(!validate({ delivery_path: '/eval/delivery.json' }, selectionSchema).ok,
    'the envelope must travel in the reply: the workflow cannot open the delivery file');
  assert(!validate({ head_candidates: [], kernel_candidates: [] }, selectionSchema).ok,
    'selection alone is no longer a workflow handoff');
  assert(!validate({ round: 'resume', profile_topN_json: '/p', top_kernels: [] }, profileSchema).ok,
    'fixed-load profile schema remains unchanged');
  const first = await run(initial);
  assert.deepEqual(first.phases, ['KernelExtract']);
  assert.deepEqual(first.calls.map(c => c.role), ['kernel_extractor']);
  const input = first.calls[0].inputs;
  assert.equal(first.calls[0].phase, 'prepare');
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
  assert.deepEqual(first.calls[0].opts.schema.required, ['delivery_path', 'prepared']);
  assert.equal(input.SOURCE_DELIVERY_REQUIRED, true);
  let release;
  const finishing = run(initial, { safeAgent: () => new Promise(resolve => { release = resolve; }) });
  let done = false;
  finishing.then(() => { done = true; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(done, false, 'downstream queues wait for the complete skill stage');
  release({ delivery_path: '/eval/delivery.json' });
  await finishing;

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
    assert.deepEqual(actual.calls, [], `${name}: no downstream extraction or profiling`);
    assert.deepEqual(actual.phases, [], name);
    const old = await run(body, { KERNEL_EXTRACT_ON: false });
    assert.equal(old.calls.length, expectedFixedCalls, name);
    assert(old.calls.every(c => ['profiler', 'system_architect'].includes(c.role)), name);
  }
  const queued = await run(milestonePlan, { kernelQueue: [candidate('saved')] });
  assert.equal(queued.calls.length, 0, 'consume saved candidates without rediscovery');
  assert.equal(queued.plan.kernel_candidates[0].short_name, 'saved');

  const lowShare = {...candidate('three_percent'), pct_gpu_time: 3.25};
  const scoped = {...candidate('explicit_scope'), pct_gpu_time: 1};
  const inventory = [lowShare, scoped, candidate('last')];
  let queue = inventory;
  for (const expected of inventory) {
    const batch = await run(milestonePlan + milestoneFilter + milestoneSchedule + milestoneRefresh + carry,
      {kernelQueue: queue, remaining: 1});
    assert.equal(batch.scheduled[0].short_name, expected.short_name);
    assert.deepEqual(batch.calls, [], 'an accepted optimization must not replenish the inventory');
    const resumedBatch = await run(resume, {ST: batch.state});
    assert.deepEqual(resumedBatch.calls, [], 'all-phase resume reuses the completed preparation');
    queue = resumedBatch.kernelQueue;
  }
  const exhausted = await run(milestonePlan, {kernelQueue: queue});
  assert.equal(exhausted.plan.stop, true);
  assert.deepEqual(exhausted.calls, [], 'queue exhaustion never triggers another extraction');
  const fixedFilter = await run(milestonePlan + milestoneFilter + milestoneSchedule,
    {KERNEL_EXTRACT_ON: false, milestone: 1, kernelQueue: inventory, remaining: 3});
  assert.deepEqual(fixedFilter.scheduled.map(k => k.short_name), ['last'],
    'fixed-load milestone still enforces its original 5% threshold');
  for (const mode of ['normal', 'fast', 'deep']) {
    const phases = new Function('A', 'KERNEL_EXTRACT_ON', 'FAST_MODE', 'DEEP_MODE',
      `${phaseSelector} return ['setup', 'profile', 'config', 'tune', 'head', 'kernel', 'final'].filter(want);`);
    assert.deepEqual(phases({}, true, mode === 'fast', mode === 'deep'),
      mode === 'normal' ? ['setup', 'head', 'kernel', 'final'] : ['setup', 'head', 'final']);
  }

  // Execute the actual optimization + integration block with disagreeing metrics.
  const optimizeAndIntegrate = between('  const optimized = await parallel(cands.map', '  // --- (c) If improved:');
  async function metricDispatch(enabled, weighted, geomean) {
    const integrated = [];
    const env = {KERNEL_EXTRACT_ON: enabled, cands: [candidate('weighted')],
      EVAL_DIR: '/eval', MODEL_PATH: '/model', WORKLOAD: {}, WORKFLOW_DIR: '/wf', profile: {},
      curOverlay: '', curFlags: '', curEnv: '', curTput: 100, NOISE_BAND: 0.5,
      KERNEL_WF_SCRIPT: '/kernel.js', KERNEL_WF_DIR: '/kernel', USE_EXPERT_SKILLS: false,
      EXPERT_SKILLS_DIR: '', KB_ARGS: {}, KERNEL_BUDGET: 1, GRAPH_REQ: '', TASK: '', EXTRACT_SCHEMA: {},
      history: {ledger: []}, acceptedKernels: [], log: () => {}, noteKernelKB: () => {},
      laneArgs: a => a, parallel: jobs => Promise.all(jobs.map(f => f())),
      extractWithBaseline: async () => ({unittest_smoke: 'pass', task_dir: '/task', target_callable: 'pkg:leaf'}),
      reserveBoundedWorkflow: async () => ({final_patch: '/patch', final_geomean: geomean, final_weighted: weighted}),
      runIntegrateBothLegs: async (_intro, inputs) => {
        integrated.push(inputs);
        return {e2e_throughput_tok_s: 110};
      },
      integAccepted: () => true, bankAccepted: () => {}, krOf: () => ({}), e2eFrom: () => ({}),
    };
    await new AsyncFunction('env', `let {${Object.keys(env).join(',')}} = env; ${optimizeAndIntegrate}`)(env);
    return integrated;
  }
  assert.equal((await metricDispatch(true, 1.4, 0.8))[0].KERNEL_RESULT.verified_isolated_speedup, 1.4);
  assert.deepEqual(await metricDispatch(true, 0.8, 1.4), [], 'unweighted wins cannot override an AgentX regression');
  assert.deepEqual(await metricDispatch(false, 1.4, 0.8), [], 'fixed-load metric choice stays unchanged');
  assert.deepEqual(await metricDispatch(true, undefined, 1.4), [], 'missing primary metrics cannot fall back to unweighted wins');

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

  const extractOnly = await run(resume, { ST: baselineOnly, A: { phases: 'extract' } });
  assert.equal(extractOnly.calls[0].phase, 'prepare', 'baseline-only extract phase completes UTs');
  const oldSelection = await run(resume, { ST: { ...baselineOnly,
    headQueue: [{ ...candidate('old'), prepared_ut: undefined }], kernelQueue: [] } });
  assert.equal(oldSelection.calls[0].phase, 'prepare', 'old selection queues are not completed UTs');
  assert.deepEqual(restored.profile.failed_tasks, selection().failed_tasks);
  assert.deepEqual(restored.profile.deferred_tasks, selection().deferred_tasks);
  assert.deepEqual(restored.profile.coverage_summary, selection().coverage_summary);
  let reads = 0;
  const repaired = await run(initial, { readAgentXPreparation: async () => {
    if (++reads === 1) throw new Error('saved manifest not found');
    return selection();
  } });
  assert.deepEqual(repaired.calls.map(c => c.phase), ['prepare', 'repair_delivery']);
  await assert.rejects(run(initial, { readAgentXPreparation: async () => {
    throw new Error('saved manifest not found');
  } }), /AgentX UT delivery unreadable/);
  // Exercise the real extraction function: authoring can overlap even for a shared
  // GPU (execution locks live at the command boundary), and failed UTs stay local.
  const extraction = between('async function extractWithBaseline(', '\n// abDone ==');
  const makeDispatch = new Function('safeAgent', `
    const CAPTURE_STORAGE_ENV = '', CAPTURE_BYTE_BUDGET = 1, CAPTURE_CASE_BYTE_LIMIT = 1;
    const CAPTURE_PERSIST_POLICY = '', BASELINE_EXTRACT_RETRIES = 0;
    const roleAgent = (role, phase, intro, inputs) => ({ role, phase, intro, inputs });
    const hasFrozenBaseline = ext => ext && ext.baseline_frozen === true;
    const kernelSelectionVerified = () => ({ ok: true });
    const guardUnhydratedVerdicts = ext => ext;
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
  // Preserve the legacy extractor's omitted-dispatcher retry behavior.
  const identity = between('const CALLABLE_SPEC_RX =', 'const PRE_FLAGGED_HEADS =');
  const target = 'pkg.live:op', dispatcher = 'pkg.live:compile';
  const selectedHead = { entity_kind: 'gpu_kernel', device_kernel: 'device_kernel',
    seam_candidates: [
      { target_callable: target, role: 'op_seam', depth: 1, device_kernels: ['device_kernel'] },
      { target_callable: dispatcher, role: 'dispatcher', depth: 2, device_kernels: ['device_kernel'] },
    ] };
  const makeRetryDispatch = new Function('safeAgent', 'KERNEL_EXTRACT_ON', `
    const CAPTURE_STORAGE_ENV = '', CAPTURE_BYTE_BUDGET = 1, CAPTURE_CASE_BYTE_LIMIT = 1;
    const CAPTURE_PERSIST_POLICY = '', BASELINE_EXTRACT_RETRIES = 1;
    const roleAgent = (role, phase, intro, inputs) => ({ role, phase, intro, inputs });
    const hasFrozenBaseline = ext => ext && ext.baseline_frozen === true;
    const log = () => {};
    const require = () => { throw new Error('no capture files in this CPU fixture'); };
    ${identity}
    ${extraction}
    return extractWithBaseline;
  `);
  {
    const enabled = false;
    const retryPrompts = [];
    const retryDispatch = makeRetryDispatch(async prompt => {
      retryPrompts.push(prompt);
      return { smoke: 'pass', task_dir: '/fixture', baseline_frozen: true,
        target_callable: target, selection_validation: {
          contract: 'kernel_selection', ok: true, target_callable: target,
          device_kernel: 'device_kernel', total_calls_observed: 1,
          target_marker_calls: 1, matched_kernel_calls: 1, deepest_verified: true,
          live_candidate_targets: [target], deeper_live_candidates: [],
          candidate_targets_tested: retryPrompts.length === 1 ? [target] : [target, dispatcher],
        } };
    }, enabled);
    const repaired = await retryDispatch('kernel_extractor', 'extract_op', '', { KERNEL: selectedHead });
    assert.equal(retryPrompts.length, 2, 'real selection gate requires the omitted dispatcher probe');
    assert.equal(repaired.smoke, 'pass', 'complete new evidence can certify the same target');
    assert.deepEqual(retryPrompts[1].inputs.ATTEMPTED_TARGET_CALLABLES, [target]);
    const retry = retryPrompts[1].intro;
    assert.equal(retry.includes('not a target blacklist'), enabled);
    assert.equal(retry.includes('new runtime evidence resolves every failed selection check'), enabled);
    assert.equal(retry.includes('selection_validation_ref'), enabled);
    assert.equal(retry.includes('PRIOR ATTEMPT DID NOT SELECT THE PROFILED GPU KERNEL'), !enabled);
    assert.equal(retry.includes('Do not return any ATTEMPTED_TARGET_CALLABLES value again.'), !enabled);
    assert.equal(retry.includes('--cleanup-task-dir "$TASK" --no-promote'), !enabled);
  }
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
  console.log('PASS: baseline-to-skill dispatch, single preparation, downstream queue consumption, evidence gates and resumes; fixed-load dispatch preserved.');
})().catch(e => { console.error(e); process.exitCode = 1; });
