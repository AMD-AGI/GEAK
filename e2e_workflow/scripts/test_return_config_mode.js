#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Execute the shipped setup/resume, launch-role inputs, carry-state and return emitters.
'use strict';
const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
function between(start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a);
  assert(a >= 0 && b > a, `workflow source block: ${start}`);
  return source.slice(a, b);
}
const init = between('const INIT_FLAGS =', '// Schema-v2 handoffs');
const setupInputs = between("    roleAgent('director', 'setup',", "    { phase: 'Setup'");
const profilerInputs = between("    roleAgent('profiler', 'baseline',", "    { phase: 'Profile'");
const integratorInputs = between('    const mileIntegrateInputs = {', '    const integ = await runIntegrateBothLegs(');
const pendingInputs = between('      { ...p.inputs, CURRENT_OVERLAY:', '      `finish-integrate');
const finalizeInputs = between("    roleAgent('e2e_integrator', 'finalize',", "    { phase: 'Finalize'");
const validationInputs = between("    roleAgent('director', 'validate',", "    { phase: 'Validate'");
const setup = between('  // An explicitly complete seed', '  curOverlay = INIT_BASE_OVERLAY;');
const resume = between("  curFlags = ST.flags || '';", '  curOverlay = ST.overlay || INIT_BASE_OVERLAY;');
const state = between('const carryState = {', '// What this run got from the KB');
const accepted = between('\n  accepted_config: { flags: curFlags', '  accepted_kernels: acceptedKernels,');
const sweepAcceptance = between('  if (sweep && sweep.best_throughput_tok_s > curTput) {',
  '    log(`Config sweep accepted.');
const warmAcceptance = between('        const parity = String(trial.parity', '          log(`[kb] ADOPTED');
const afterWarmThroughput = between('let curTput = ST.throughput || kbSeedTput || BASELINE_TPUT;',
  "if (want('config')");
const cfgSource = between('const cfg =', '// Workflow scripts have no filesystem API.');
const roleAgentSource = between('function roleAgent(role, phase, intro, inputs) {', '// Resilient agent wrapper:');
const validationDecision = between('  validation = await safeAgent(', "} else {\n  log(`Phase(s)");

function makeContext(args, carried, setupFlags) {
  return {
    A: args, ST: carried || {}, setup: { server_flags: { extra: setupFlags }, server_env: 'RECIPE=1' },
    curArgsMode: 'append', curFlags: '', curEnv: '', curUnsetEnvs: [], curRemoveArgs: [], curOverlay: '', curTput: 100,
    BACKEND: 'sglang', EVAL_DIR: '/eval', MODEL_NAME: 'test', BASELINE_TPUT: 100,
    NOISE_BAND: 0.5, profile: {}, strategy: {}, headQueue: [], kernelQueue: [],
    acceptedHeads: [], flaggedHeads: [], acceptedKernels: [], tuning: null,
    pendingIntegrations: [], history: {},
    LAUNCH_SCRIPT: '/recipe', MODEL_PATH: '/model', EXP_ROOT: '/exp', EVAL_DIR_OVERRIDE: '',
    MODEL_NAME_HINT: 'test', TASK: '', GPU_IDS: '0', WORKLOAD: {}, INIT_BASE_OVERLAY: '',
    roleAgent: (_role, _phase, _task, inputs) => inputs,
    PARITY_REPLICAS: 2, WORKFLOW_DIR: '/workflow', GPU_LIST: ['0'],
    TRACELENS_INPUTS: {}, ANALYSIS_SKILL_INPUTS: {}, TUNING_FINALIZE_INPUTS: {},
    c: { gpu_id: '0', short_name: 'kernel', pct_gpu_time: 10 },
    ext: { task_dir: '/task', source_path_in_sglang: 'kernel.py', target_callable: 'kernel.run' },
    kl: { final_patch: '/patch', final_geomean: 1.2 }, allAccepted: [],
    p: { inputs: { KEPT_INPUT: 'pending', CURRENT_FLAGS: '--stale', CURRENT_ENV: 'STALE=1',
      CURRENT_OVERLAY: '/stale', CURRENT_REMOVE_ARGS: ['--stale'], CURRENT_UNSET_ENVS: ['STALE'],
      CURRENT_THROUGHPUT: 1 } },
    finalize: null, finalTput: 100, APPLY_TO_ORIGINAL: true,
    VALIDATION_MEASUREMENT_MODE: 'fresh_server', VALIDATION_SAMPLES: 2, report: null,
  };
}

function run(args = {}, carried = null, setupFlags = '--disable-cuda-graph') {
  const context = makeContext(args, carried, setupFlags);
  const script = `${init}\n${carried ? resume : setup}\n${integratorInputs}\n${state}\n` +
    `JSON.stringify({ state: carryState, ${accepted}
      setup_inputs: ${carried ? 'null' : `([${setupInputs}][0])`},
      launch_inputs: {
        profiler: [${profilerInputs}][0], integrator: mileIntegrateInputs,
        pending_integrator: [${pendingInputs}][0],
        finalize: [${finalizeInputs}][0], validation: [${validationInputs}][0],
      } });`;
  return JSON.parse(vm.runInNewContext(script, context, { timeout: 1000 }));
}

function expectedLaunchInputs(flags, env, removeArgs = [], unsetEnvs = [], argsMode = 'append', clearSeedRemovals = false) {
  return {
    profiler: {
      EVAL_DIR: '/eval', MODEL_PATH: '/model', GPU_ID: '0', WORKLOAD: {}, ROUND: 0,
      OVERLAY_PYTHONPATH: '', EXTRA_SERVER_ARGS: flags, EXTRA_ENV: env,
      GEAK_UNSET_ENVS: JSON.stringify(unsetEnvs), SKILL_DIR: '/workflow',
      ...(removeArgs.length || clearSeedRemovals ? { GEAK_REMOVE_ARGS: JSON.stringify(removeArgs) } : {}),
    },
    integrator: {
      EVAL_DIR: '/eval', MODEL_PATH: '/model', GPU_ID: '0', WORKLOAD: {}, NOISE_BAND_PCT: 0.5,
      KERNEL_RESULT: {
        short_name: 'kernel', task_dir: '/task', source_path_in_sglang: 'kernel.py',
        target_callable: 'kernel.run', final_patch: '/patch', verified_isolated_speedup: 1.2,
        pct_gpu_time: 10,
      },
      CURRENT_OVERLAY: '', CURRENT_FLAGS: flags, CURRENT_ENV: env,
      CURRENT_UNSET_ENVS: unsetEnvs, CURRENT_THROUGHPUT: 100, SKILL_DIR: '/workflow',
      ...(removeArgs.length || clearSeedRemovals ? { CURRENT_REMOVE_ARGS: removeArgs } : {}),
    },
    finalize: {
      EVAL_DIR: '/eval', FINAL_OVERLAY: '', ACCEPTED_FLAGS: flags, ACCEPTED_ENV: env,
      ACCEPTED_UNSET_ENVS: unsetEnvs, ACCEPTED_KERNELS: [], BASELINE_THROUGHPUT: 100,
      SKILL_DIR: '/workflow',
      ...(removeArgs.length || clearSeedRemovals ? { ACCEPTED_REMOVE_ARGS: removeArgs } : {}),
    },
    pending_integrator: {
      KEPT_INPUT: 'pending', CURRENT_OVERLAY: '', CURRENT_FLAGS: flags, CURRENT_ENV: env,
      CURRENT_REMOVE_ARGS: removeArgs, CURRENT_UNSET_ENVS: unsetEnvs, CURRENT_THROUGHPUT: 100,
    },
    validation: {
      EVAL_DIR: '/eval', MODEL_PATH: '/model', GPU_ID: '0', BASELINE_THROUGHPUT: 100,
      NOISE_BAND_PCT: 0.5, BASELINE_OVERLAY: '', FINAL_OVERLAY: '',
      FINAL_FLAGS: { flags, env, unset_envs: unsetEnvs, remove_args: removeArgs, args_mode: argsMode },
      CLAIMED_THROUGHPUT: 100, WORKLOAD: {}, APPLY_TO_ORIGINAL: true,
      MEASUREMENT_MODE: 'fresh_server', MEASUREMENT_PURPOSE: 'validation', REPLICAS: 2,
      SKILL_DIR: '/workflow', ARCHITECT_REPORT: '/eval/architect_report.md',
      FINAL_REPORT: '/eval/final_report.md',
    },
  };
}

async function runSweep(sweep, warmStart = false) {
  const context = makeContext({ initial_extra_server_args: '--prior 1', initial_args_mode: 'replace',
    initial_extra_env: 'OLD=1', initial_env_complete: true,
    initial_remove_args: ['--deleted'], initial_unset_envs: ['REMOVED_ENV'] }, null, '--recipe');
  const checkpoints = [];
  Object.assign(context, {
    sweep, trial: { kept: true, parity: 'pass' }, measured: sweep.best_throughput_tok_s,
    mf: { merged: '--merged 2' }, me: { merged: 'MERGED=1' }, kbSeedTput: 0,
    MEASUREMENT_MODE: 'fresh_server', EFFECTIVE_CONFIG_DIGEST: '', checkpoints,
    requireE2EValidationCheckpoint: async (file, receipt) => checkpoints.push({ file, receipt }),
  });
  // Stop each shipped acceptance block after its checkpoint and close its if.
  // The remaining block only logs/reprofiles; no acceptance statements are replaced.
  const script = `(async () => { ${init}\n${setup}\n` +
    `${warmStart ? warmAcceptance : sweepAcceptance}\n}\n` +
    `${warmStart ? afterWarmThroughput : ''}\n${state}\n` +
    `return JSON.stringify({ state: carryState, ${accepted} checkpoints }); })()`;
  return JSON.parse(await vm.runInNewContext(script, context, { timeout: 1000 }));
}

function assertRenderedRole(inputs, removalJson) {
  const defaults = {
    BACKEND: 'sglang', SERVING_TP: 2, SERVING_GPU: '0,1', MEASUREMENT_MODE: 'fresh_server',
    PARITY_REPLICAS: 2, SEARCH_REPLICAS: 1, VALIDATION_REPLICAS: 3, EFFECTIVE_CONFIG_DIGEST: 'digest',
  };
  const original = JSON.stringify(inputs);
  const prompt = vm.runInNewContext(`${cfgSource}\n${roleAgentSource}\n` +
    `roleAgent('director', 'test', 'Check the removal transport.', inputs)`, {
    ...defaults, inputs, WORKFLOW_DIR: '/workflow', GPU_IDS: '0,1',
    expertSkillsBlock: () => '', warmStartBlock: () => '',
  }, { timeout: 1000 });
  assert.equal(JSON.stringify(inputs), original, 'rendering cannot mutate the carried role inputs');
  const renderedInputs = prompt.split('\n## Inputs\n')[1].split('\n\nReturn ONLY')[0];
  const expectedInputs = { ...defaults, ...inputs };
  if (removalJson !== undefined) expectedInputs.GEAK_REMOVE_ARGS = removalJson;
  const expectedLines = Object.entries(expectedInputs).map(([key, value]) =>
    `- ${key}: ${typeof value === 'string' ? value : JSON.stringify(value)}`).join('\n');
  assert.equal(renderedInputs, expectedLines,
    'the actual shared roleAgent renders the exact launch controls and preserved inputs');
  if (removalJson === undefined) {
    assert(!prompt.includes('GEAK_REMOVE_ARGS'), 'legacy inputs add no removal field or instruction');
    assert(!prompt.includes('server_args_unverified'), 'legacy inputs keep the previous prompt body');
  } else {
    assert(prompt.includes('Pass GEAK_REMOVE_ARGS from Inputs verbatim to every bench_e2e.sh launch, including [].'));
    assert(prompt.includes('do not bypass it or report throughput'));
    assert(prompt.includes('Preserve GEAK_REMOVE_ARGS in the final launch bundle.'));
  }
}

async function runValidationDecision(directorResult) {
  const context = makeContext({}, null, '--recipe');
  const calls = [], logs = [];
  Object.assign(context, {
    allAccepted: [{ short_name: 'kernel' }], history: { ledger: [] },
    validation: null, validatedOk: 'not_run', finalTput: 120, finalSpeedup: 1.2,
    VALIDATE_SCHEMA: {}, EXPERIENCE_SCHEMA: {}, calls, logs,
    log: (message) => logs.push(message),
    safeAgent: async (inputs, options) => {
      calls.push({ label: options.label, inputs });
      return options.label === 'director:validate' ? directorResult : {};
    },
  });
  const script = `(async () => { ${validationDecision}\n` +
    `return JSON.stringify({ validatedOk, finalSpeedup, calls, logs }); })()`;
  try {
    return JSON.parse(await vm.runInNewContext(script, context, { timeout: 1000 }));
  } catch (error) {
    return { error: error.message, calls, logs, validatedOk: context.validatedOk, finalSpeedup: context.finalSpeedup };
  }
}

async function test() {
  const flags = '--context-length 9728 --cuda-graph-max-bs 64';
  const complete = run({ initial_extra_server_args: flags, initial_args_mode: 'replace' });
  assert.deepEqual(complete.accepted_config, { flags, env: 'RECIPE=1', args_mode: 'replace', remove_args: [] });
  assert(!complete.accepted_config.flags.includes('--disable-cuda-graph'));
  assert.equal(complete.state.args_mode, 'replace');
  assert.equal(complete.setup_inputs.INIT_ARGS_MODE, 'replace');
  assert.equal(complete.setup_inputs.INIT_FLAGS, flags);

  const resumed = run({ initial_extra_server_args: '--different-seed', initial_args_mode: 'append' }, complete.state);
  assert.deepEqual(resumed.accepted_config, complete.accepted_config,
    'a later invocation uses the carried flags and their completeness together');

  const empty = run({ initial_extra_server_args: '', initial_args_mode: 'replace' });
  assert.equal(empty.accepted_config.flags, '', 'a complete empty base cannot restore setup recipe flags');
  assert.equal(empty.accepted_config.args_mode, 'replace');
  assert.equal(empty.setup_inputs.INIT_ARGS_MODE, 'replace');
  assert.equal(empty.setup_inputs.INIT_FLAGS, '', 'the baseline-measuring role receives the explicit empty seed');

  const legacy = run({ initial_extra_server_args: '--candidate-only' });
  assert.deepEqual(legacy.accepted_config, { flags: '--candidate-only', env: 'RECIPE=1' });
  assert(!Object.hasOwn(legacy.state, 'args_mode'));
  assert(!Object.hasOwn(legacy.setup_inputs, 'INIT_ARGS_MODE'));
  const legacyResume = run({ initial_args_mode: 'replace' }, { flags: '--saved-delta', env: '' });
  assert.deepEqual(legacyResume.accepted_config, { flags: '--saved-delta', env: '' },
    'a newly complete handoff must not relabel an older carried delta');
  assert(!Object.hasOwn(legacyResume.state, 'args_mode'));

  const standalone = run();
  assert.deepEqual(standalone.accepted_config, { flags: '--disable-cuda-graph', env: 'RECIPE=1' });
  const invalidMode = run({ initial_args_mode: 'unknown', initial_extra_server_args: '--candidate-only' });
  assert(!Object.hasOwn(invalidMode.accepted_config, 'args_mode'));
  const literal = run({ initial_args_mode: 'replace', initial_extra_env: "'JSON={\"x\": \"space value\"}' EMPTY=" });
  assert.equal(literal.accepted_config.env, "'JSON={\"x\": \"space value\"}' EMPTY=",
    'argument completeness does not reinterpret environment assignments');
  const unset = run({ initial_extra_env: '', initial_env_complete: true,
    initial_unset_envs: ['RECIPE'], initial_args_mode: 'replace' });
  assert.equal(unset.accepted_config.env, '', 'complete empty env cannot restore setup assignments');
  assert.deepEqual(unset.accepted_config.unset_envs, ['RECIPE']);
  assert.deepEqual(unset.state.unset_envs, ['RECIPE']);
  assert.equal(unset.setup_inputs.INIT_ENV_COMPLETE, true);
  assert.deepEqual(unset.setup_inputs.INIT_UNSET_ENVS, ['RECIPE']);
  assert.deepEqual(run({}, unset.state).accepted_config, unset.accepted_config);
  assert(!Object.hasOwn(run({ initial_unset_envs: ['NEW'] }, { flags: '', env: '' }).accepted_config, 'unset_envs'),
    'older carried state does not acquire a new handoff removal');
  const removed = run({ initial_extra_server_args: '--keep 1', initial_args_mode: 'replace',
    initial_remove_args: ['--disable-radix-cache'] });
  assert.deepEqual(removed.accepted_config.remove_args, ['--disable-radix-cache']);
  assert.deepEqual(run({}, removed.state).accepted_config, removed.accepted_config);
  assert(!Object.hasOwn(run({ initial_remove_args: ['--new'] }, { flags: '', env: '' }).accepted_config, 'remove_args'));

  const controls = run({ initial_extra_server_args: '--keep 1', initial_args_mode: 'replace',
    initial_extra_env: 'KEEP_ENV=1', initial_env_complete: true,
    initial_remove_args: ['--disable-radix-cache', '--limit 8'], initial_unset_envs: ['OLD_ENV'] });
  assert.deepEqual(controls.setup_inputs, {
    LAUNCH_SCRIPT: '/recipe', MODEL_PATH: '/model', EXP_ROOT: '/exp', EVAL_DIR_OVERRIDE: '',
    MODEL_NAME_HINT: 'test', TASK: '', GPU_IDS: '0', WORKLOAD: {},
    INIT_FLAGS: '--keep 1', INIT_ENV: 'KEEP_ENV=1', INIT_BASE_OVERLAY: '',
    INIT_ARGS_MODE: 'replace', INIT_ENV_COMPLETE: true,
    INIT_UNSET_ENVS: ['OLD_ENV'], INIT_REMOVE_ARGS: ['--disable-radix-cache', '--limit 8'],
    MEASUREMENT_PURPOSE: 'parity', REPLICAS: 2, SKILL_DIR: '/workflow',
  }, 'the baseline-measuring setup role receives both removal controls with the exact seed');
  assert.deepEqual(controls.launch_inputs, expectedLaunchInputs('--keep 1', 'KEEP_ENV=1',
    ['--disable-radix-cache', '--limit 8'], ['OLD_ENV'], 'replace'),
  'profiling, integration, final bundle and validation receive the same effective configuration');

  const changedSeed = {
    initial_extra_server_args: '--new-seed', initial_args_mode: 'append',
    initial_extra_env: 'NEW_ENV=1', initial_remove_args: ['--new-removal'],
    initial_unset_envs: ['NEW_UNSET'],
  };
  const resumedControls = run(changedSeed, controls.state);
  assert.equal(resumedControls.setup_inputs, null);
  assert.deepEqual(resumedControls.accepted_config, controls.accepted_config);
  assert.deepEqual(resumedControls.launch_inputs, controls.launch_inputs,
    'every downstream launch uses carried removals, never the new handoff seed');

  const emptyControls = run(changedSeed, {
    flags: '', env: '', args_mode: 'replace', remove_args: [], unset_envs: [],
  });
  assert.deepEqual(emptyControls.accepted_config, { flags: '', env: '', args_mode: 'replace', remove_args: [] });
  assert.deepEqual(emptyControls.state.remove_args, []);
  assert.deepEqual(emptyControls.launch_inputs, expectedLaunchInputs('', '', [], [], 'replace', true),
    'explicit empty carried controls and values do not revive the new seed');
  assert.equal(emptyControls.launch_inputs.profiler.GEAK_REMOVE_ARGS, '[]',
    'an explicit empty role input clears the seed control inherited through the process environment');
  assert.deepEqual(emptyControls.launch_inputs.pending_integrator.CURRENT_REMOVE_ARGS, [],
    'a resumed pending integration cannot restore its captured stale removal controls');

  const emptySeedControls = run({ initial_extra_server_args: '', initial_args_mode: 'replace',
    initial_extra_env: '', initial_env_complete: true, initial_remove_args: [], initial_unset_envs: [] });
  assert.deepEqual(emptySeedControls.accepted_config, { flags: '', env: '', args_mode: 'replace', remove_args: [] });
  assert(!Object.hasOwn(emptySeedControls.setup_inputs, 'INIT_REMOVE_ARGS'));
  assert(!Object.hasOwn(emptySeedControls.setup_inputs, 'INIT_UNSET_ENVS'));
  assert.deepEqual(emptySeedControls.launch_inputs, expectedLaunchInputs('', '', [], [], 'replace'));

  const legacyControls = run(changedSeed, { flags: '--saved 2', env: 'SAVED=1' });
  assert.deepEqual(legacyControls.accepted_config, { flags: '--saved 2', env: 'SAVED=1' });
  assert.deepEqual(legacyControls.launch_inputs, expectedLaunchInputs('--saved 2', 'SAVED=1', [], [], 'append', true),
    'legacy carried state without controls stays a delta and acquires no seed removals');
  assert.deepEqual(run({}, { flags: '--saved 2', env: 'SAVED=1' }).launch_inputs,
    expectedLaunchInputs('--saved 2', 'SAVED=1'),
    'without inherited seed removals, legacy roles keep their previous input shape');
  assert.deepEqual(standalone.launch_inputs, expectedLaunchInputs('--disable-cuda-graph', 'RECIPE=1'),
    'runs without removals preserve the existing launch-role input shape');
  assert(!Object.hasOwn(standalone.setup_inputs, 'INIT_REMOVE_ARGS'));

  // Follow actual shipped input objects through the real common prompt builder,
  // including controls that must clear the handoff's inherited environment.
  assertRenderedRole(controls.setup_inputs, '["--disable-radix-cache","--limit 8"]');
  for (const inputs of Object.values(controls.launch_inputs)) {
    assertRenderedRole(inputs, '["--disable-radix-cache","--limit 8"]');
  }
  for (const inputs of Object.values(emptyControls.launch_inputs)) assertRenderedRole(inputs, '[]');
  assertRenderedRole({ INIT_REMOVE_ARGS: [] }, '[]');
  assertRenderedRole({ CURRENT_REMOVE_ARGS: [] }, '[]');
  assertRenderedRole({ ACCEPTED_REMOVE_ARGS: [] }, '[]');
  assertRenderedRole({ FINAL_FLAGS: { flags: '--keep 1', remove_args: [] } }, '[]');
  assertRenderedRole({ GEAK_REMOVE_ARGS: '[ "--raw", "--limit 8" ]' }, '[ "--raw", "--limit 8" ]');
  assertRenderedRole({ GEAK_REMOVE_ARGS: '[]' }, '[]');
  assertRenderedRole({ CURRENT_REMOVE_ARGS: [], FINAL_FLAGS: { remove_args: ['--stale'] },
    GEAK_REMOVE_ARGS: '["--stale-env"]' }, '[]');
  assertRenderedRole({});
  assertRenderedRole({ CURRENT_FLAGS: '--legacy 1' });
  for (const warmStart of [false, true]) {
    const phase = warmStart ? 'WarmStart' : 'ConfigSweep';
    const emptyWin = await runSweep({ accepted_flags: '', accepted_env: '', best_throughput_tok_s: 120 }, warmStart);
    const emptyAccepted = { flags: '', env: '', args_mode: 'replace', remove_args: ['--deleted'], unset_envs: ['REMOVED_ENV'] };
    assert.deepEqual(emptyWin.accepted_config, emptyAccepted,
      `${phase}: an empty winning configuration cannot restore prior or merged values`);
    assert.equal(emptyWin.state.flags, '');
    assert.equal(emptyWin.state.env, '');
    assert.equal(emptyWin.state.throughput, 120);
    assert.equal(emptyWin.checkpoints.length, 1);
    assert.equal(emptyWin.checkpoints[0].file, 'config/e2e_validation.json');
    assert.equal(emptyWin.checkpoints[0].receipt.phase, phase);
    assert.equal(emptyWin.checkpoints[0].receipt.final_throughput_tok_s, 120);
    assert.deepEqual(emptyWin.checkpoints[0].receipt.accepted_config,
      { ...emptyAccepted, effective_config_digest: '' },
      `${phase}: checkpoint and returned configuration must describe the same measured winner`);
    assert.deepEqual(run({}, emptyWin.state).accepted_config, emptyAccepted,
      `${phase}: resume preserves the empty measured winner`);

    const omittedEnv = await runSweep({ accepted_flags: '--winner 2', best_throughput_tok_s: 120 }, warmStart);
    assert.deepEqual(omittedEnv.accepted_config, {
      ...emptyAccepted, flags: '--winner 2', env: warmStart ? 'MERGED=1' : 'OLD=1',
    }, `${phase}: omitted optional env retains the existing fallback`);
    assert.equal(omittedEnv.checkpoints[0].receipt.accepted_config.env, omittedEnv.accepted_config.env);

    const noWin = await runSweep({ accepted_flags: '', accepted_env: '', best_throughput_tok_s: 99 }, warmStart);
    assert.deepEqual(noWin.accepted_config, { ...emptyAccepted, flags: '--prior 1', env: 'OLD=1' });
    assert.equal(noWin.state.throughput, 100);
    assert.deepEqual(noWin.checkpoints, [], `${phase}: a losing result cannot replace or checkpoint the current best`);
  }

  for (const numbers of [{}, { director_verified_throughput_tok_s: 999, throughput_speedup: 9.99 }]) {
    const failed = await runValidationDecision({ validation_status: 'server_args_unverified', ...numbers });
    assert.match(failed.error, /Final server launch failed argument verification/);
    assert.equal(failed.validatedOk, 'not_run', 'typed launch failure throws before numeric fallback or success grading');
    assert.equal(failed.finalSpeedup, 1.2);
    assert.deepEqual(failed.calls.map((call) => call.label), ['director:validate'],
      'failed final argument verification cannot reach experience curation');
    assert.deepEqual(failed.logs, [], 'typed launch failure cannot log a carried-throughput success');
  }
  for (const directorResult of [null, { validation_status: 'timeout' }]) {
    const fallback = await runValidationDecision(directorResult);
    assert(!fallback.error);
    assert.equal(fallback.validatedOk, false);
    assert.equal(fallback.finalSpeedup, 1.2);
    assert.deepEqual(fallback.calls.map((call) => call.label), ['director:validate', 'architect:experience final']);
    assert.equal(fallback.calls[1].inputs.FINAL_THROUGHPUT, 120);
    assert.equal(fallback.calls[1].inputs.FINAL_SPEEDUP, 1.2,
      'an ordinary missing-number result preserves the accepted same-session fallback');
  }
  const valid = await runValidationDecision({ validation_status: 'passed', output_parity: 'pass',
    director_verified_throughput_tok_s: 130, throughput_speedup: 1.3 });
  assert(!valid.error);
  assert.equal(valid.validatedOk, true);
  assert.equal(valid.finalSpeedup, 1.3);
  assert.equal(valid.calls[1].inputs.FINAL_THROUGHPUT, 130);
  assert.equal(valid.calls[1].inputs.VALIDATION_STATUS, 'passed');
  console.log('Shipped workflow configuration, role rendering, launch controls, sweep and validation checks passed');
}

module.exports = { run, runSweep, runValidationDecision };
if (require.main === module) test().catch((error) => { console.error(error); process.exitCode = 1; });
