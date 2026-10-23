#!/usr/bin/env node
'use strict';
// Exercise the real workload detection and role prompt funnel without a GPU.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'e2e_workflow.js'), 'utf8');
const settings = src.slice(src.indexOf('const ISL_DECLARED ='), src.indexOf('// Seed config:'));
const captureSettings = src.slice(src.indexOf('const CAPTURE_BYTE_BUDGET ='), src.indexOf('const TASK ='));
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
  const MN_SPEC = null, pdConfigFields = () => ({}), pdDirectionGaps = [];
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
  const SERVING_TOPOLOGY = '', MN = null, pdVars = () => ({}), pdFragmentBlock = () => '';
  ${settings}
  ${measurementSettings}
  ${captureSettings}
  ${hook}
  ${role}
  ${legacyRole}
  return { active: KERNEL_EXTRACT_ON, workload: WORKLOAD, spec: WORKLOAD_SPEC,
    roleAgent, legacyRoleAgent, logs,
    block: kernelExtractBlock, measurement: MEASUREMENT_MODE,
    validationMeasurement: VALIDATION_MEASUREMENT_MODE, validationSamples: VALIDATION_SAMPLES };
`);
const consumers = [
  ['system_architect', 'update_experience'],
  ['kernel_extractor', 'prepare'], ['kernel_extractor', 'repair_delivery'],
  ['op_benchmarker', 'bakeoff'],
];
for (const args of [
  {workload_kind: 'agentx_trace_replay'},
  {workload_spec: {kind: 'agentx_trace_replay', concurrency: 10}},
  {state: {workload_spec: {kind: 'agentx_trace_replay', concurrency: 10}}},
]) {
  const run = make({...args, analysis_skill: 'none', use_expert_skills: 'false'}, undefined);
  assert(run.active);
  assert.equal(run.workload.kind, 'agentx_trace_replay');
  assert.equal(run.measurement, 'warm_server');
  assert.equal(run.validationMeasurement, 'warm_server');
  assert.equal(run.validationSamples, 1);
  for (const [name, phase] of consumers) {
    const prompt = run.roleAgent(name, phase, 'Run the phase.', {WORKLOAD: run.workload});
    assert(prompt.includes(`PHASE=${phase}`));
    if (name === 'kernel_extractor') {
      const skill = prompt.match(/Read (\S+kernel-extract\/SKILL.md)/)[1];
      assert(fs.existsSync(skill), 'injected skill must ship at the referenced path');
      assert(!prompt.includes(`First Read ${root}/roles/kernel_extractor.md`));
      assert(prompt.includes(`${path.dirname(skill)}/references/workflow-handoff.md`));
      assert(prompt.includes('"SKILL_ROOT":' + JSON.stringify(path.dirname(skill))));
      assert(prompt.includes('"CAPTURE_PERSIST_POLICY":"structure_only"'));
      assert(prompt.includes('"SHARED_TOOLS":' + JSON.stringify(path.join(root, 'scripts'))));
      assert(prompt.includes('"PARSE_PROFILE_STRICT_PHASES":"1"'));
      assert(prompt.includes('"KERNEL_ENV_SKIP_ENUM_REAP":"1"'));
    } else {
      const handoff = prompt.match(/Read (\S+kernel-extract\/references\/workflow-handoff.md)/)[1];
      assert(fs.existsSync(handoff));
      assert(!prompt.includes('kernel-extract/SKILL.md'), 'downstream roles load only the handoff');
      assert(prompt.includes('do not repeat discovery, sampling or UT authoring'));
    }
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
    assert(prompt.includes(name === 'kernel_extractor'
      ? 'kernel-extract/SKILL.md' : 'kernel-extract/references/workflow-handoff.md'));
    if (name !== 'kernel_extractor') assert(prompt.includes(JSON.stringify(specification)));
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
  {state: {workload_spec: {kind: 'custom_workload', concurrency: 7}}},
]) {
  const run = make(args, undefined);
  assert(!run.active);
  assert.deepEqual(run.workload, {isl: 1024, osl: 1024, conc: 64, shape_provenance: 'synthetic_default'});
  assert.deepEqual(run.logs, []);
  if (!args.workload_spec) assert.deepEqual(run.spec, {}, 'only AgentX declarations resume from state');
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
const fixedKind = make({workload_kind: 'synthetic',
  state: {workload_spec: {kind: 'agentx_trace_replay', concurrency: 10}}}, undefined);
assert(!fixedKind.active);
assert.deepEqual(fixedKind.spec, {}, 'explicit non-AgentX kind does not inherit replay configuration');
// This integration must not expand main's workload identity from incidental metadata.
for (const args of [{}, {bench_client: 'agentx'}, {workload_shape_provenance: 'agentx_observed'}]) {
  const run = make(args, {env: {GEAK_WORKLOAD_KIND: 'agentx_trace_replay', BENCH_CLIENT: 'agentx',
    WORKLOAD_ENV: 'agentx', WORKLOAD_CLIENT: 'agentx'}});
  assert(!run.active, 'AgentX extraction requires an explicit workload declaration');
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

const override = make({workload_kind: 'agentx_trace_replay', capture_persist_policy: 'full'}, undefined);
assert(override.roleAgent('kernel_extractor', 'prepare', '', {}).includes('"CAPTURE_PERSIST_POLICY":"full"'));

// Verify instruction paths after moving common tools out of the skill package.
const skillRoot = path.join(root, 'knowledge/kernel_extraction_skills/kernel-extract');
const documents = [path.join(skillRoot, 'SKILL.md'),
  ...fs.readdirSync(path.join(skillRoot, 'references')).filter(n => n.endsWith('.md'))
    .map(n => path.join(skillRoot, 'references', n))];
for (const file of documents) {
  const text = fs.readFileSync(file, 'utf8');
  for (const [, href] of text.matchAll(/\[[^\]]+\]\(([^)]+)\)/g)) {
    if (/^(https?:|#)/.test(href)) continue;
    assert(fs.existsSync(path.resolve(path.dirname(file), href.split('#')[0])), `${file}: ${href}`);
  }
  for (const [, variable, name] of text.matchAll(/\$(SKILL_ROOT|SHARED_TOOLS)\/([\w./-]+\.(?:py|mjs|sh))/g)) {
    assert(fs.existsSync(path.resolve(variable === 'SKILL_ROOT' ? skillRoot : path.join(root, 'scripts'), name)),
      `${file}: ${variable}/${name}`);
  }
}
