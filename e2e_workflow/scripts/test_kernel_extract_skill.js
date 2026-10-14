#!/usr/bin/env node
'use strict';
// Exercise the real workload detection and role prompt funnel without a GPU.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'e2e_workflow.js'), 'utf8');
const settings = src.slice(src.indexOf('const ISL_DECLARED ='), src.indexOf('// Seed config:'));
const measurementSettings = src.slice(src.indexOf('// One lifecycle for EVERY throughput number'),
  src.indexOf('// CUDA/HIP-graph deployment requirement'));
const hook = src.match(/function kernelExtractBlock\(role\) \{[\s\S]*?\n\}/)[0];
const role = src.match(/function roleAgent\(role, phase, intro, inputs\) \{[\s\S]*?\n\}/)[0];
const legacyRole = role.replace('function roleAgent(', 'function legacyRoleAgent(')
  .replace(' + kernelExtractBlock(role)', '');
const carry = src.match(/const carryState = \{[\s\S]*?\n\};/)[0];
const carryFor = new Function('WORKLOAD_KIND', 'WORKLOAD_SPEC', `
  const KERNEL_EXTRACT_ON = WORKLOAD_KIND === 'agentx_trace_replay';
  const MEASUREMENT_MODE = 'isolated_server', VALIDATION_MEASUREMENT_MODE = 'isolated_server';
  const VALIDATION_REPLICAS = 1;
  const BACKEND = 'sglang', EVAL_DIR = '/eval', MODEL_NAME = 'model', BASELINE_TPUT = 100;
  const NOISE_BAND = 0.5, curFlags = '', curEnv = '', curOverlay = '', curTput = 100;
  const profile = null, strategy = null, tuning = null, history = {};
  const headQueue = [], kernelQueue = [], acceptedHeads = [], flaggedHeads = [];
  const acceptedKernels = [], pendingIntegrations = [];
  ${carry}
  return carryState;
`);
const make = new Function('A', 'process', `
  const WORKFLOW_DIR = ${JSON.stringify(root)};
  const MODEL_PATH = A.kernel_path && !A.model_path ? '' : (A.model_path || '/models/test');
  const logs = [], log = msg => logs.push(msg), BACKEND = A.backend || 'sglang', SERVING_TP = 8, SERVING_GPU = '0,1,2,3,4,5,6,7';
  const GPU_IDS = SERVING_GPU, EFFECTIVE_CONFIG_DIGEST = '';
  const E2E_LEARNED_KB_ENABLED = false, cfg = o => JSON.stringify(o);
  const expertSkillsBlock = () => '', warmStartBlock = () => '';
  const correctnessEnvBlock = () => '', workloadIdentityBlock = () => '';
  ${settings}
  ${measurementSettings}
  ${hook}
  ${role}
  ${legacyRole}
  return { active: KERNEL_EXTRACT_ON, workload: WORKLOAD, roleAgent, legacyRoleAgent, logs,
    block: kernelExtractBlock, measurement: MEASUREMENT_MODE,
    validationMeasurement: VALIDATION_MEASUREMENT_MODE, validationSamples: VALIDATION_SAMPLES };
`);
const consumers = [
  ['profiler', 'baseline'], ['profiler', 'reprofile'],
  ['system_architect', 'strategize'], ['system_architect', 'plan_milestone'],
  ['kernel_extractor', 'extract_op'], ['kernel_extractor', 'extract'],
  ['op_benchmarker', 'bakeoff'],
];
for (const [args, env] of [
  [{workload_spec: {kind: 'agentx_trace_replay', concurrency: 10}}, {}],
  [{state: {workload_spec: {kind: 'agentx_trace_replay', concurrency: 10}}}, {}],
  [{bench_client: 'agentx'}, {}],
  [{}, {BENCH_CLIENT: 'agentx'}],
  [{}, {GEAK_WORKLOAD_KIND: 'agentx_trace_replay'}],
  [{workload_shape_provenance: 'agentx_observed'}, {}],
  [{workload_shape_provenance: 'synthetic_fallback_on_agentx'}, {}],
]) {
  const run = make({...args, analysis_skill: 'none', use_expert_skills: 'false'}, {env});
  assert(run.active);
  assert.equal(run.workload.kind, 'agentx_trace_replay');
  assert.equal(run.measurement, 'warm_server');
  assert.equal(run.validationMeasurement, 'warm_server');
  assert.equal(run.validationSamples, 1);
  for (const [name, phase] of consumers) {
    const prompt = run.roleAgent(name, phase, 'Run the phase.', {WORKLOAD: run.workload});
    assert(prompt.includes(`PHASE=${phase}`));
    const skill = prompt.match(/Read (\S+kernel-extract\/SKILL.md)/)[1];
    assert(fs.existsSync(skill), 'injected skill must ship at the referenced path');
  }
  assert.equal(run.block('director'), '');
  assert.equal(run.block('config_tuner'), '');
}
// AgentX uses the same core method across model/backend features, with no model whitelist.
for (let [backend, specification] of [
  ['sglang', {kind: 'synthetic_isl_osl', model: {architecture: 'dense', speculative: false}}],
  ['vllm', {kind: 'dynamic_serving', model: {architecture: 'moe', speculative: true}}],
  ['atom', {kind: 'trace_replay', cache: {prefix: false}}],
  ['vllm', {kind: 'embedding', phases: ['encode', 'pool']}],
  ['custom', {kind: 'offline_batch', phases: ['forward'], shapes: [[3, 224, 224]]}],
  ['custom', {kind: 'previously_unknown_kind', custom_axis: [1, 17, 63]}],
]) {
  specification = {...specification, scenario: specification.kind, kind: 'agentx_trace_replay'};
  const run = make({backend, workload_spec: specification}, undefined);
  assert(run.active);
  assert.equal(run.workload.kind, specification.kind);
  assert.deepEqual(run.workload.spec, specification);
  for (const [name, phase] of consumers) {
    const prompt = run.roleAgent(name, phase, '', {});
    assert(prompt.includes('kernel-extract/SKILL.md'));
    assert(prompt.includes(JSON.stringify(specification)));
    assert(prompt.includes(`BACKEND=${backend}`));
  }
  const saved = carryFor(specification.kind, specification);
  const resumed = make({state: saved}, undefined);
  assert(resumed.active);
  assert.deepEqual(resumed.workload.spec, specification);
}
// Every non-AgentX call preserves the old workload, role prompts, and carried state.
for (const args of [
  {}, {bench_client: 'native'}, {bench_client: 'inferencex'},
  {workload_spec: {kind: 'synthetic_isl_osl'}},
  {workload_spec: {kind: 'dynamic_serving'}},
  {workload_spec: {kind: 'custom_workload', shape_axes: [1, 17]}},
  {state: {workload_spec: {kind: 'synthetic_isl_osl'}}},
]) {
  const run = make(args, undefined);
  assert(!run.active);
  assert.deepEqual(run.workload, {isl: 1024, osl: 1024, conc: 64, shape_provenance: 'synthetic_default'});
  assert.deepEqual(run.logs, []);
  for (const [name, phase] of consumers) {
    const inputs = {WORKLOAD: run.workload, PROFILE_TOPN: '/eval/profile_topN.json'};
    assert.equal(run.block(name), '');
    assert.equal(run.roleAgent(name, phase, 'Original phase.', inputs),
      run.legacyRoleAgent(name, phase, 'Original phase.', inputs), 'fixed-load prompt is byte-identical');
  }
  const spec = args.workload_spec || args.state?.workload_spec || {};
  assert.deepEqual(carryFor(spec.kind || 'unknown', spec), carryFor('unknown', {}));
}
// An explicit fixed workload wins over stale inherited replay signals and state.
const fixed = make({workload_spec: {kind: 'synthetic_isl_osl'},
  state: {workload_spec: {kind: 'agentx_trace_replay'}}},
  {env: {GEAK_WORKLOAD_KIND: 'agentx_trace_replay', BENCH_CLIENT: 'agentx'}});
assert(!fixed.active);
assert(!Object.hasOwn(fixed.workload, 'kind'));
for (const client of ['native', 'inferencex']) {
  const run = make({}, {env: {GEAK_WORKLOAD_KIND: 'agentx_trace_replay', BENCH_CLIENT: client}});
  assert(!run.active, 'current fixed-load client outranks a leftover AgentX environment signal');
}
const standalone = make({kernel_path: '/task/kernel.py', workload_spec: {kind: 'agentx_trace_replay'}}, undefined);
assert(!standalone.active);
for (const [name] of consumers) assert.equal(standalone.block(name), '');
assert.equal(fixed.measurement, 'warm_server');
assert.equal(fixed.validationMeasurement, 'warm_server');
// Extraction does not change the caller's benchmark lifecycle or replica defaults.
const replaySpec = {kind: 'agentx_trace_replay'};
const explicit = make({workload_spec: replaySpec, measurement_mode: 'isolated_server',
  validation_measurement_mode: 'isolated_server', validation_replicas: 2}, undefined);
assert.equal(explicit.measurement, 'isolated_server');
assert.equal(explicit.validationMeasurement, 'isolated_server');
assert.equal(explicit.validationSamples, 2);
console.log('PASS: AgentX-only activation across models and resumes; fixed-load prompts, workload and state stay unchanged.');
