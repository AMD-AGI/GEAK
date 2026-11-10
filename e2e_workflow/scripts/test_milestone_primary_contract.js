#!/usr/bin/env node
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const sourceDir = path.resolve(__dirname, '../src/e2e_workflow');
const milestoneSource = fs.readFileSync(path.join(sourceDir, '17_milestone_loop.js'), 'utf8');
const helperSource = fs.readFileSync(path.join(sourceDir, '09_runtime_helpers.js'), 'utf8');
const headSource = fs.readFileSync(path.join(sourceDir, '16_head_kernel.js'), 'utf8');
const primaryStart = helperSource.indexOf('function isolatedPrimary(');
const primaryEnd = helperSource.indexOf('const SURGEON_SCHEMA', primaryStart);
const correctiveEnd = helperSource.indexOf('// --- FAST-MODE wall-clock control', primaryEnd);
assert(primaryStart >= 0 && primaryEnd > primaryStart && correctiveEnd > primaryEnd, 'actual helper slices exist');
const primarySource = helperSource.slice(primaryStart, primaryEnd);
const HYPOTHETICAL = { final_speedup: 1.25, final_geomean: 0.9, final_patch: 'fixture.patch',
  workload_aligned: true, validation_status: 'accepted', final_validation: { status: 'pass' }, eval_dir: 'fixture.eval' };
let assertions = 0;
function check(actual, expected, label) { assert.deepStrictEqual(actual, expected, label); assertions++; }
function environment() {
  return { log() {}, HEAD_CORRECTIVE_MAX: 1, FAST_MODE: false, FAST_DEADLINE_HIT: false,
    TIME_DEADLINE_HIT: false, SURGICAL_FIX: true, KERNEL_WF_SCRIPT: 'fixture.workflow',
    KERNEL_WF_DIR: 'fixture.workflow.dir', KERNEL_KNOWLEDGE_DIR: 'fixture.knowledge',
    USE_EXPERT_SKILLS: false, EXPERT_SKILLS_DIR: '', KB_ARGS: {}, WARM_START_OFF: false,
    KERNEL_BUDGET: 1, EVAL_DIR: 'fixture.eval', GRAPH_REQ: '', TASK: '', obj: value => value };
}
async function milestone(result, workloadPath = 'HYPOTHETICAL.workload.json', integration = { gate: 'accepted',
  e2e_throughput_tok_s: 101, e2e_delta_pct: 1 }, corrective) {
  const calls = { workflows: [], integrations: [], guards: [], corrections: [] };
  const context = { ...environment(), want: () => true, dispatched: 0, BUDGET: 1, MIN_KERNEL_TASKS: 1,
    noImprove: 0, milestone: 0, phase() {}, kernelQueue: [{ short_name: 'HYPOTHETICAL', pct_gpu_time: 50 }],
    MILESTONE_MIN_PCT: 1, GPU_LIST: [0], MODEL_PATH: 'fixture.model', MODEL_NAME: 'fixture', WORKLOAD: {},
    curOverlay: '', curFlags: '', curEnv: {}, curTput: 100, BASELINE_TPUT: 100, NOISE_BAND: 0,
    WORKFLOW_DIR: 'fixture.dir', ANALYSIS_SKILL_INPUTS: {}, profile: null,
    history: { ledger: [], milestones: [] }, acceptedKernels: [], pendingIntegrations: [],
    E2E_LEARNED_KB_ENABLED: false, PROFILE_SCHEMA: {}, EXTRACT_SCHEMA: {},
    parallel: operations => Promise.all(operations.map(operation => operation())),
    extractWithBaseline: async () => ({ editable: true, unittest_smoke: 'pass', task_dir: 'fixture.task', workload_path: workloadPath }),
    laneArgs: value => value, workflow: async (reference, inputs) => { calls.workflows.push(inputs); return result; },
    noteKernelKB() {}, runIntegrateBothLegs: async (intro, inputs) => { calls.integrations.push(inputs); return integration; },
    integAccepted: (value, percentage, primary) => { calls.guards.push(primary); return value.gate === 'accepted'; },
    gateRejectReason: (value, percentage, primary) => { calls.guards.push(primary); return value?.reason || ''; },
    rejectClass: reason => reason === 'fixable' ? 'integration' : '',
    tryCorrectiveReauthor: async spec => { calls.corrections.push(spec); return corrective || { banked: false }; },
    bankAccepted: (bank, entry, kernelResult) => bank.push({ entry, kernelResult }), krOf: inputs => inputs.KERNEL_RESULT,
    e2eFrom: () => ({}), safeAgent: async () => null, roleAgent: () => '' };
  vm.runInNewContext(primarySource, context, { timeout: 1000 });
  await vm.runInNewContext(`(async () => { ${milestoneSource}\n })()`, context, { timeout: 1000 });
  return { context, calls };
}
async function correction(result, workloadPath = 'HYPOTHETICAL.workload.json', callerBasis, surgicalFix = true) {
  const calls = { native: [], surgeon: 0, integrations: [], guards: [] };
  const context = { ...environment(), SURGICAL_FIX: surgicalFix, roleAgent: () => '',
    safeAgent: async () => { calls.surgeon++; return { fixed: true, final_patch: 'legacy.patch', final_geomean: 9 }; },
    fastBoundedWorkflow: async (reference, inputs) => { calls.native.push(inputs); return result; },
    runIntegrateBothLegs: async (intro, inputs) => { calls.integrations.push(inputs); return { gate: 'accepted', e2e_throughput_tok_s: 101 }; },
    isImplausibleSpeedup: (percentage, primary) => { calls.guards.push(primary); return false; }, rejectClass: () => 'integration' };
  vm.runInNewContext(helperSource.slice(primaryStart, correctiveEnd), context, { timeout: 1000 });
  const value = await context.tryCorrectiveReauthor({ short_name: 'HYPOTHETICAL', workload_path: workloadPath,
    ...(callerBasis === undefined ? {} : { workload_aligned: callerBasis }),
    task_dir: 'fixture.task', isolated: HYPOTHETICAL.final_speedup, reason: 'fixable', pct_gpu_time: 50,
    cur: { overlay: '', flags: '', env: {}, tput: 100 }, base_inputs: { KERNEL_RESULT: { final_patch: 'old.patch', isolated_geomean: 99 } } });
  return { value, calls };
}
async function main() {
  const primaryContext = {};
  vm.runInNewContext(primarySource, primaryContext, { timeout: 1000 });
  for (const [result, workloadPath, expected] of [
    [HYPOTHETICAL, 'weighted', 1.25], [null, 'weighted', null],
    [{ final_geomean: 9 }, 'weighted', undefined], [{ final_speedup: null, final_geomean: 9 }, 'weighted', null],
    [{ final_geomean: 9, workload_aligned: true }, undefined, undefined],
    [{ final_geomean: 1.4, workload_aligned: false }, undefined, 1.4],
    [{ final_geomean: 9 }, undefined, undefined], [{ final_geomean: 9, workload_aligned: null }, undefined, undefined],
    [{ final_geomean: 9, workload_aligned: false }, 'weighted', undefined],
    [{ final_speedup: null, final_geomean: 9, workload_aligned: false }, undefined, null],
  ]) check(primaryContext.isolatedPrimary(result, workloadPath), expected,
    `shared primary: basis=${String(result?.workload_aligned)}, workload=${workloadPath || 'absent'}, declared=${String(result?.final_speedup)}`);
  const accepted = await milestone(HYPOTHETICAL);
  check(accepted.calls.workflows[0].op_spec.workload_path, 'HYPOTHETICAL.workload.json', 'extraction workload reaches optional op_spec');
  check(Object.keys(accepted.calls.workflows[0].op_spec).join(','), 'workload_path', 'milestone op_spec adds only workload_path, without head shape literals');
  check(accepted.calls.integrations[0].KERNEL_RESULT.verified_isolated_speedup, 1.25, 'weighted primary integrates despite geomean below one');
  check(accepted.calls.integrations[0].KERNEL_RESULT.isolated_geomean, 0.9, 'secondary geomean remains distinct');
  check(accepted.calls.guards[0], 1.25, 'acceptance guard receives primary');
  check(accepted.context.acceptedKernels[0].entry.isolated, 1.25, 'accepted bank carries primary');
  check(accepted.context.history.ledger[0].isolated_speedup, 1.25, 'confirmed ledger carries primary');
  check(accepted.context.history.ledger[0].verdict, 'confirmed', 'weighted winner is confirmed');
  const smallGain = await milestone({ ...HYPOTHETICAL, final_speedup: 1.01, final_geomean: Math.SQRT1_2 });
  check(smallGain.calls.integrations[0].KERNEL_RESULT.verified_isolated_speedup, 1.01,
    'qualified weighted 1.01 primary integrates even when geomean is approximately 0.707');
  check(smallGain.context.history.ledger[0].verdict, 'confirmed', 'qualified small weighted gain is not discarded by geomean');
  const missing = { ...HYPOTHETICAL, final_geomean: 9 };
  delete missing.final_speedup;
  const absentPatch = { ...HYPOTHETICAL };
  delete absentPatch.final_patch;
  for (const result of [missing, { ...missing, workload_aligned: false },
    ...[null, NaN, Infinity, 0, -1].map(primary => ({ ...missing, final_speedup: primary })),
    ...['flagged', 'rejected', undefined].map(status => ({ ...HYPOTHETICAL, validation_status: status })),
    { ...HYPOTHETICAL, final_patch: '' }, { ...HYPOTHETICAL, final_patch: ' \t\n ' }, absentPatch]) {
    const incomplete = await milestone(result);
    check(incomplete.calls.integrations.length, 0, 'unqualified result never integrates or falls back to geomean');
    check(incomplete.context.history.ledger[0].verdict, 'incomplete', 'unqualified result is incomplete rather than no-improvement');
    check(incomplete.context.history.ledger[0].isolated_speedup, Number.isFinite(result.final_speedup) ? result.final_speedup : null,
      'incomplete qualification ledger retains finite declared primary or null');
  }
  const unknownBasis = { ...missing };
  delete unknownBasis.workload_aligned;
  const unknown = await milestone(unknownBasis, '');
  check(unknown.calls.integrations.length, 0, 'missing workload path does not establish unweighted basis');
  check(unknown.context.history.ledger[0].verdict, 'incomplete', 'unknown basis missing primary is incomplete');
  const unweighted = await milestone({ ...missing, workload_aligned: false }, '');
  check(unweighted.calls.integrations[0].KERNEL_RESULT.verified_isolated_speedup, 9, 'explicitly unweighted accepted legacy missing primary uses geomean');
  check(Object.hasOwn(unweighted.calls.workflows[0], 'op_spec'), false, 'unweighted extraction omits optional op_spec');
  const explicitNull = await milestone({ ...missing, workload_aligned: false, final_speedup: null }, '');
  check(explicitNull.context.history.ledger[0].verdict, 'incomplete', 'explicit legacy null blocks geomean fallback');
  const noGain = await milestone({ ...HYPOTHETICAL, final_speedup: 1, final_geomean: 9 });
  check(noGain.context.history.ledger[0].verdict, 'dead_end', 'qualified non-improvement stays distinct from incomplete');
  const noGainNoPatch = await milestone({ ...HYPOTHETICAL, final_speedup: 1, final_patch: '' });
  check(noGainNoPatch.context.history.ledger[0].verdict, 'dead_end', 'qualified non-improvement does not require an unused patch');
  for (const integration of [null, { gate: 'incomplete' }, { gate: 'accepted', ab_complete: false }]) {
    const pending = await milestone(HYPOTHETICAL, 'HYPOTHETICAL.workload.json', integration);
    check(pending.context.pendingIntegrations[0].isolated, 1.25, 'incomplete A/B pending preserves primary');
    check(pending.context.pendingIntegrations[0].inputs.KERNEL_RESULT.verified_isolated_speedup, 1.25, 'pending replay inputs preserve primary');
    check(pending.context.history.ledger[0].isolated_speedup, 1.25, 'incomplete A/B ledger preserves primary');
  }
  const corrected = await milestone(HYPOTHETICAL, 'HYPOTHETICAL.workload.json', { gate: 'rejected', reason: 'fixable' },
    { banked: true, isolated: 1.6, isolated_geomean: 0.8, patch: 'fixed.patch', integ: { e2e_throughput_tok_s: 101 } });
  check(corrected.calls.corrections[0].workload_path, 'HYPOTHETICAL.workload.json', 'correction receives workload');
  check(corrected.calls.corrections[0].workload_aligned, true, 'milestone retains result basis and forwards it to correction');
  check(corrected.calls.corrections[0].isolated, 1.25, 'correction receives original primary');
  check(corrected.calls.guards[0], 1.25, 'rejection guard receives original primary');
  check(corrected.context.acceptedKernels[0].entry.isolated, 1.6, 'corrected bank carries corrected primary');
  check(corrected.context.history.ledger[0].isolated_speedup, 1.6, 'corrective ledger carries corrected primary');
  check(corrected.context.history.ledger[0].isolated_geomean, 0.8, 'corrective ledger carries corrected diagnostic, not the original geomean');
  const native = await correction(HYPOTHETICAL);
  check(native.calls.surgeon, 0, 'weighted corrective bypasses legacy geomean-only surgeon');
  check(native.calls.native.length, 1, 'weighted corrective invokes existing bounded native workflow once');
  check(native.calls.native[0].op_spec.workload_path, 'HYPOTHETICAL.workload.json', 'bounded corrective workload propagates');
  check(native.calls.native[0].apply_to_original, 'false', 'corrective native workflow retains non-applying contract');
  check(native.calls.integrations[0].KERNEL_RESULT.verified_isolated_speedup, 1.25, 'corrective integrator receives primary');
  check(native.calls.integrations[0].KERNEL_RESULT.isolated_geomean, 0.9, 'corrective integrator replaces stale original geomean diagnostic');
  check(native.calls.integrations[0].KERNEL_RESULT.code_patch, HYPOTHETICAL.final_patch, 'corrective code patch propagates');
  check(native.calls.integrations[0].KERNEL_RESULT.final_patch, HYPOTHETICAL.final_patch, 'corrective final patch propagates');
  check(native.calls.guards[0], 1.25, 'corrective plausibility guard receives primary');
  check(native.value.isolated, 1.25, 'corrective return carries primary');
  check(native.value.isolated_geomean, 0.9, 'corrective return carries matching diagnostic');
  for (const callerBasis of [undefined, true, null]) {
    const routed = await correction(HYPOTHETICAL, '', callerBasis);
    check(routed.calls.surgeon, 0, 'absent path with unknown or weighted caller basis cannot use legacy surgeon');
    check(routed.calls.native.length, 1, 'unknown or weighted caller basis uses bounded native workflow');
    check(routed.value.isolated, 1.25, 'native correction preserves declared primary without inventing legacy basis');
    const unqualified = await correction(missing, '', callerBasis);
    check(unqualified.value.banked, false, 'unknown or weighted caller basis cannot manufacture a geomean-only winner');
  }
  const mismatched = await correction(HYPOTHETICAL, 'HYPOTHETICAL.workload.json', false);
  check(mismatched.calls.surgeon, 0, 'weighted path bypasses surgeon even with explicitly unweighted caller basis');
  check(mismatched.calls.native.length, 1, 'weighted path mismatch routes to native qualification');
  for (const callerBasis of [undefined, false]) {
    const forwarded = await milestone({ ...HYPOTHETICAL, workload_aligned: callerBasis }, '', { gate: 'rejected', reason: 'fixable' });
    check(forwarded.calls.corrections[0].workload_aligned, callerBasis, 'milestone forwards unknown or explicit unweighted basis without inference');
  }
  for (const result of [missing, { ...missing, workload_aligned: false },
    { ...missing, final_speedup: null }, { ...HYPOTHETICAL, validation_status: 'rejected' },
    { ...HYPOTHETICAL, final_patch: ' \t\n ' }]) {
    const rejected = await correction(result);
    check(rejected.value.banked, false, 'weighted corrective cannot adopt legacy or rejected result');
    check(rejected.calls.integrations.length, 0, 'unqualified corrective never re-gates using geomean');
  }
  for (const [validationStatus, patch, banked] of [
    ['accepted', 'fixture.patch', true], ['flagged', 'fixture.patch', false],
    ['rejected', 'fixture.patch', false], [undefined, 'fixture.patch', false], ['accepted', ' \t\n ', false],
  ]) {
    const modern = await correction({ ...HYPOTHETICAL, workload_aligned: false,
      validation_status: validationStatus, final_patch: patch }, '', false, false);
    check(modern.calls.native.length, 1, 'modern unweighted corrective exercises native workflow');
    check(modern.value.banked, banked, `modern unweighted corrective status=${validationStatus}, patch=${JSON.stringify(patch)}`);
    check(modern.calls.integrations.length, banked ? 1 : 0, 'modern unweighted corrective re-gates only accepted nonblank patch');
  }
  const legacySurgeon = await correction(null, '', false);
  check(legacySurgeon.calls.surgeon, 1, 'legacy unweighted corrective retains surgeon path');
  check(legacySurgeon.calls.native.length, 0, 'successful legacy surgeon avoids native re-author');
  check(legacySurgeon.value.isolated, 9, 'legacy surgeon geomean is allowed only without weighted workload');
  check(legacySurgeon.calls.integrations[0].KERNEL_RESULT.verified_isolated_speedup, 9, 'explicitly unweighted surgeon primary reaches integrator');
  const extracted = workloadPath => ({ op_kind: 'fixture.op', shapes: { batch: 2 }, dtype: 'bf16', task_dir: 'fixture.task', workload_path: workloadPath });
  const head = { short_name: 'HYPOTHETICAL', regime: 'fixture.regime', pct_gpu_time: 50, gpu_id: 0 };
  const deepExtract = extracted('HYPOTHETICAL.deep.workload.json');
  const parallelExtract = extracted('HYPOTHETICAL.parallel.workload.json');
  const serialExtract = extracted('HYPOTHETICAL.serial.workload.json');
  const headContext = { ...environment(), MODEL_PATH: 'fixture.model', SERVING_GPU: 0, WORKLOAD: {}, NOISE_BAND: 0,
    WORKFLOW_DIR: 'fixture.workflow.dir', ACCURACY_INPUTS: {}, curOverlay: '', curFlags: '', curEnv: {}, curTput: 100,
    c: { head, ext: deepExtract, best: 1.25, lang: 'triton', lastEval: 'fixture.eval' },
    st: { ext: parallelExtract }, h: head, ext: serialExtract, j: { ext: parallelExtract, h: head },
    cand: { isolated: 1.25 }, headIntegrateInputs: {}, reason: 'fixable', dreason: 'fixable', rejectClass: () => 'integration' };
  const correctiveLiterals = Array.from(headSource.matchAll(/tryCorrectiveReauthor\((\{[\s\S]*?\n\s*\})\)/g), match => match[1]);
  check(correctiveLiterals.length, 4, 'all four actual head corrective literal objects are located');
  for (const [literalIndex, literal] of correctiveLiterals.entries()) {
    const spec = vm.runInNewContext(`(${literal})`, headContext, { timeout: 1000 });
    const expectedPath = [deepExtract, parallelExtract, serialExtract, serialExtract][literalIndex].workload_path;
    check(spec.workload_path, expectedPath, 'actual head corrective literal forwards its own extracted weighted path');
    const routed = await correction(HYPOTHETICAL, spec.workload_path, spec.workload_aligned);
    check(routed.calls.surgeon, 0, 'forwarded head weighted path prevents legacy surgeon');
    check(routed.calls.native[0].op_spec.workload_path, expectedPath, 'head weighted path reaches corrective native op_spec');
  }
  const initialLiterals = Array.from(headSource.matchAll(/op_spec: (\{[^\n]+\}),\r?\n/g), match => match[1]);
  check(initialLiterals.length, 2, 'both existing inline initial head workflow op_spec literals are located');
  for (const [literalIndex, literal] of initialLiterals.entries()) {
    const opSpec = vm.runInNewContext(`(${literal})`, headContext, { timeout: 1000 });
    check(JSON.parse(JSON.stringify(opSpec)), { op_kind: 'fixture.op', shapes: { batch: 2 }, dtype: 'bf16',
      regime: 'fixture.regime', cuda_graph_safe: true, workload_path: [parallelExtract, serialExtract][literalIndex].workload_path },
    'initial head workflow literal retains its existing shape and optional workload contract');
  }
  console.log(`PASS: ${assertions} assertions; HYPOTHETICAL fixtures only; source execution with stubbed workflow/agent/A-B calls.`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
