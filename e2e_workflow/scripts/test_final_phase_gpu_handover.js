#!/usr/bin/env node
// Regression guard for handing the GPUs to the final phase (no GPU, no model needed).
//
// A 24h GLM-5.2 run lost its validated result to the search it had already closed. TuningSkillset started
// ~1.5h before the reserve boundary and its loop (a tuner sweep over every GPU, then a two-leg A/B) took
// ~5h; the boundary abandoned the agent, not its legs, so Finalize and Validate measured on GPUs the
// tuning legs still held, and a detached waiter was still polling to start the next one when the run
// returned. Pinned here:
//
//   1. TuningSkillset starts only with tuning_min_window_s (default 6h) left before the reserve.
//   2. Finalize and Validate each begin with the run_reaper.py fence, bounded on its own clock and
//      non-fatal; PROCESS SAFETY tells every agent what the fence means.
//
// The gate and the fence are EXTRACTED from the real source and run against stubs; their placement in
// the final phase is checked structurally.
//
// Run:  node e2e_workflow/scripts/test_final_phase_gpu_handover.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const WORKFLOW = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const REAPER = path.join(ROOT, 'e2e_workflow', 'scripts', 'run_reaper.py');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const src = fs.readFileSync(WORKFLOW, 'utf8');
const reaperSrc = fs.readFileSync(REAPER, 'utf8');
const H = 3600 * 1000;

function slice(startMarker, endMarker) {
  const start = src.indexOf(startMarker);
  const end = src.indexOf(endMarker, start);
  ok(start !== -1 && end > start, `located ${JSON.stringify(startMarker)} .. ${JSON.stringify(endMarker)}`);
  if (failures) process.exit(1);
  return src.slice(start, end);
}

// ---------------------------------------------------------------- the tuning window
const gateSrc = slice('const _tuningMinWindowS =', 'function deepBoundedWorkflow(');
function gate({ budgetMs = 20 * H, remaining, reserve = 8 * H, windowS }) {
  const logs = [];
  const A = windowS === undefined ? {} : { tuning_min_window_s: windowS };
  const g = new Function('A', 'TIME_BUDGET_MS', 'remainingMs', 'FINAL_RESERVE_MS', 'log',
    `${gateSrc}\nreturn { tuningWindowShort, TUNING_MIN_WINDOW_MS };`)(
    A, budgetMs, () => remaining, reserve, (m) => logs.push(String(m)));
  return { short: g.tuningWindowShort(), windowMs: g.TUNING_MIN_WINDOW_MS, logs };
}

console.log('\n# TuningSkillset needs its whole loop before the reserve');
ok(gate({ remaining: 20 * H }).short === false, '12h before the reserve: tuning runs');
ok(gate({ remaining: 14 * H }).short === false, 'exactly 6h before the reserve: tuning runs');
const late = gate({ remaining: 13 * H });
ok(late.short === true, '5h before the reserve: tuning is skipped');
ok(late.logs.length === 1 && /TuningSkillset SKIPPED — ~300min left before the final reserve/.test(late.logs[0])
  && /under the 360min/.test(late.logs[0]), 'the skip says how much time was left and how much it needed');
const inside = gate({ remaining: 7 * H });
ok(inside.short === true && /~0min left/.test(inside.logs[0]), 'already inside the reserve: skipped, never negative');
ok(gate({ budgetMs: null, remaining: Infinity }).short === false, 'no time budget: the gate is inert');

console.log('\n# tuning_min_window_s');
ok(gate({ remaining: 8 * H, windowS: '0' }).short === false, '0 disables the gate');
ok(gate({ remaining: 10 * H, windowS: '3600' }).short === false, 'a 1h window admits tuning 2h before the reserve');
ok(gate({ remaining: 8.5 * H, windowS: '3600' }).short === true, 'and refuses it 30min before');
ok(gate({ remaining: 20 * H, windowS: 'abc' }).windowMs === 6 * H, 'junk falls back to the 6h default');
ok(gate({ remaining: 20 * H, windowS: '-5' }).windowMs === 6 * H, 'a negative window falls back to the default');
ok(gate({ remaining: 20 * H }).windowMs === 6 * H, 'unset means 6h');

console.log('\n# the gate guards the TuningSkillset dispatch');
const tuneIf = src.split('\n').find((l) => /^if \(want\('tune'\) && TUNING_SKILLSET_ENABLED/.test(l)) || '';
ok(/!searchClosed\('TuningSkillset'\) && !tuningWindowShort\(\)\) \{$/.test(tuneIf),
  'the tune phase checks the reserve boundary, then the tuning window');

// ---------------------------------------------------------------- the fence
const fenceSrc = slice('const RUN_REAPER =', "if (want('final')) {");
function fence(answer, evalDir = '/ws/sess/geak/e2e_cycle0') {
  const calls = [];
  const logs = [];
  const shq = (s) => "'" + String(s == null ? '' : s).replace(/'/g, "'\\''") + "'";
  const obj = (props, required) => ({ type: 'object', properties: props, required: required || [] });
  const arrObj = { type: 'array' };
  const safeAgent = async (prompt, opts, tries, timeoutFor) => {
    calls.push({ prompt, opts, tries, timeoutMs: timeoutFor() });
    return answer;
  };
  const run = new Function('WORKFLOW_DIR', 'EVAL_DIR', 'shq', 'obj', 'arrObj', 'safeAgent', 'log',
    `${fenceSrc}\nreturn finalPhaseFence;`)(
    '/geak/e2e_workflow', evalDir, shq, obj, arrObj, safeAgent, (m) => logs.push(String(m)));
  return run('finalize', 'Finalize').then((r) => ({ r, calls, logs }));
}

(async () => {
  console.log('\n# the fence runs the reaper once, on its own clock');
  const receipt = { ok: true, stage: 'finalize', note: '',
    stopped: [{ pid: 7, name: 'bash wait_then_post.sh', why: 'detached' }, { pid: 8, name: 'python3 gemm_tuner.py', why: 'gpu' }],
    survivors: [] };
  const done = await fence(receipt);
  ok(done.calls.length === 1, 'one agent call');
  const call = done.calls[0] || {};
  ok(String(call.prompt).includes(
    "python3 '/geak/e2e_workflow/scripts/run_reaper.py' fence --eval-dir '/ws/sess/geak/e2e_cycle0' --stage finalize"),
  'it runs exactly the reaper command for this eval dir and stage');
  ok(call.tries === 1, 'a fence is never retried');
  ok(call.timeoutMs === 10 * 60000, 'bounded at 10min, not the final-phase agent timeout');
  ok(call.opts && call.opts.phase === 'Finalize' && call.opts.label === 'fence:finalize', 'charged to its phase');
  ok(done.r === receipt, 'the receipt is returned');
  ok(/stopped 2 process\(es\) still running from the search \(bash wait_then_post\.sh, python3 gemm_tuner\.py\)/
    .test(done.logs.join('\n')), 'the log names what was stopped');

  const failed = await fence(null);
  ok(failed.r === null && /NON-FATAL/.test(failed.logs.join('\n')), 'a fence that returns nothing does not stop the phase');
  const noDir = await fence(receipt, '');
  ok(noDir.r === null && noDir.calls.length === 0, 'no eval dir: nothing to fence');

  console.log('\n# both final stages open with the fence');
  const finalBlock = src.slice(src.indexOf("if (want('final')) {"));
  const at = (s) => finalBlock.indexOf(s);
  ok(at('FINAL_PHASE_STARTED = true;') !== -1
    && at('FINAL_PHASE_STARTED = true;') < at("phase('Finalize');")
    && at("phase('Finalize');") < at("await finalPhaseFence('finalize', 'Finalize');")
    && at("await finalPhaseFence('finalize', 'Finalize');") < at("roleAgent('e2e_integrator', 'finalize'"),
  'Finalize: the reserve exemption, the phase, the fence, then the integrator');
  ok(at("phase('Validate');") !== -1
    && at("phase('Validate');") < at("await finalPhaseFence('validate', 'Validate');")
    && at("await finalPhaseFence('validate', 'Validate');") < at("roleAgent('director', 'validate'"),
  'Validate: the phase, the fence, then the director');

  console.log('\n# PROCESS SAFETY says what the fence means');
  const fenceFile = (reaperSrc.match(/^FENCE_FILE = "([^"]+)"$/m) || [])[1];
  ok(fenceFile === '.final_phase_fence', 'the reaper writes .final_phase_fence');
  const safety = slice('## ONCE THE FINAL PHASE STARTS, IT OWNS THE GPUS', '`;');
  ok(safety.includes(`$EVAL_DIR/${fenceFile}`), 'the rule names the file the reaper writes');
  ok(/e2e_integrator finalizing or the director validating/.test(safety), 'only the final phase\'s own roles are exempt');

  console.log(failures ? `\n${failures} FAILURE(S)` : '\nall final-phase GPU handover checks passed');
  process.exit(failures ? 1 : 0);
})();
