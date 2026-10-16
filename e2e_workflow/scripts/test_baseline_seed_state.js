#!/usr/bin/env node
// Execute the real setup/tuning/state/input fragments without models or GPUs.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const parts = path.resolve(__dirname, '../src/e2e_workflow');
const read = (name) => fs.readFileSync(path.join(parts, name), 'utf8');
const between = (src, start, end) => {
  const i = src.indexOf(start), j = src.indexOf(end, i);
  assert.ok(i >= 0 && j > i, `source fragment exists: ${start}`);
  return src.slice(i, j);
};
const setup = read('12_setup.js'), finalize = read('18_finalize.js');
const seed = between(setup, '  curFlags = INIT_FLAGS', '  log(`Setup done.');
const resume = between(setup, "  curFlags = ST.flags", '  profile = { profile_topN_json: ST.');
const tune = between(read('15_tuning_skillset.js'), '    if (tuning.apply_env)', '    log(`Tuning skillset ACCEPTED:');
const carry = between(finalize, 'const carryState = {', '\n// What this run got from the KB');
const integrate = between(read('09_runtime_helpers.js'), '  const withTuning = {', '  let integ =');
const run = (code, ctx) => vm.runInNewContext(code, ctx);
const plain = (value) => JSON.parse(JSON.stringify(value));
const ctx = {
  INIT_FLAGS: '', INIT_ENV: '', INIT_BASE_OVERLAY: '/seed/overlay',
  setup: { server_flags: { extra: '--attention-backend seed' }, server_env: 'SEED=1' },
  tuning: { apply_env: 'TUNED=1', apply_flags: '--tuned', apply_overlay: '/accepted/overlay', post_tune_throughput_tok_s: 120 },
  BACKEND: 'sglang', EVAL_DIR: '/eval', MODEL_NAME: 'model', BASELINE_TPUT: 100,
  NOISE_BAND: 1, profile: null, strategy: null, headQueue: [], kernelQueue: [],
  acceptedHeads: [], flaggedHeads: [], acceptedKernels: [], pendingIntegrations: [], history: {},
};
run(seed + tune + carry + '\nglobalThis.saved = carryState;', ctx);
const saved = plain(ctx.saved);
assert.deepEqual([saved.baseline_flags, saved.baseline_env, saved.baseline_overlay],
  ['--attention-backend seed', 'SEED=1', '/seed/overlay']);
assert.deepEqual([saved.flags, saved.env, saved.overlay],
  ['--attention-backend seed --tuned', 'SEED=1 TUNED=1', '/accepted/overlay']);

const brief = (state) => {
  const restored = { ST: state, INIT_BASE_OVERLAY: '/different-resume-overlay' };
  run(resume, restored);
  restored.inputs = {
    CURRENT_FLAGS: restored.curFlags, CURRENT_ENV: restored.curEnv,
    CURRENT_OVERLAY: restored.curOverlay, BASELINE_FLAGS: 'stale pending input',
  };
  Object.assign(restored, { SEARCH_REPLICAS: 2, ACCURACY_INPUTS: {}, tuningIntegrateInputs: () => ({}) });
  run(integrate + '\nglobalThis.brief = withTuning;', restored);
  return plain(restored.brief);
};
const next = brief(saved);
assert.deepEqual([next.BASELINE_FLAGS, next.BASELINE_ENV, next.BASELINE_OVERLAY],
  ['--attention-backend seed', 'SEED=1', '/seed/overlay']);
assert.deepEqual([next.CURRENT_FLAGS, next.CURRENT_ENV, next.CURRENT_OVERLAY],
  [saved.flags, saved.env, saved.overlay]);

const legacy = brief({ flags: '--current', env: 'CURRENT=1', overlay: '/current' });
assert.deepEqual([legacy.BASELINE_FLAGS, legacy.BASELINE_ENV, legacy.BASELINE_OVERLAY], [null, null, null]);
const pristine = brief({ ...saved, baseline_flags: '', baseline_env: '', baseline_overlay: '' });
assert.deepEqual([pristine.BASELINE_FLAGS, pristine.BASELINE_ENV, pristine.BASELINE_OVERLAY], ['', '', '']);
const partial = brief({ ...saved, baseline_env: undefined });
assert.equal(partial.BASELINE_ENV, null);
assert.equal(partial.BASELINE_OVERLAY, '/seed/overlay');

const explicit = {
  INIT_FLAGS: '--caller', INIT_ENV: 'CALLER=1', INIT_BASE_OVERLAY: '',
  setup: { server_flags: { extra: '--caller --resolved' }, server_env: 'CALLER=1 RESOLVED=1' },
};
run(seed, explicit);
assert.deepEqual([explicit.BASELINE_FLAGS, explicit.BASELINE_ENV, explicit.BASELINE_OVERLAY],
  ['--caller --resolved', 'CALLER=1 RESOLVED=1', '']);
assert.equal(explicit.curFlags, '--caller');
const fallback = { ...explicit, setup: {} };
run(seed, fallback);
assert.deepEqual([fallback.BASELINE_FLAGS, fallback.BASELINE_ENV], ['--caller', 'CALLER=1']);
const missing = { ...explicit, INIT_FLAGS: '', INIT_ENV: '', setup: {} };
run(seed, missing);
assert.deepEqual([missing.BASELINE_FLAGS, missing.BASELINE_ENV], [null, null]);
const measuredEmpty = { ...explicit, setup: { server_flags: { extra: '' }, server_env: '' } };
run(seed, measuredEmpty);
assert.deepEqual([measuredEmpty.BASELINE_FLAGS, measuredEmpty.BASELINE_ENV], ['', '']);
for (const block of [
  between(read('16_head_kernel.js'), '        const deepInputs = {', '        const integ ='),
  between(finalize, "roleAgent('director', 'validate'", "{ phase: 'Validate'"),
]) assert.match(block, /BASELINE_FLAGS, BASELINE_ENV, BASELINE_OVERLAY/);
console.log('PASS: resolved seed survives tuning, state/resume and integration; legacy missing seed stays unknown.');
