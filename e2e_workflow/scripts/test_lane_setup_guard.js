#!/usr/bin/env node
// Regression guard for a kernel lane's benchmark setup under the hung-agent guard (no GPU, no model).
//
// kernel_lane.js: a setup the guard cuts gets one resume (SETUP_RESUME) that reads back what the
// still-running attempt left on disk; a setup that fails again names the guard in its error.
// e2e_workflow.js: that error is not retried by the author loop, and an explicit agent_timeout_ms
// reaches the lanes through laneArgs.
//
// Run:  node e2e_workflow/scripts/test_lane_setup_guard.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const laneSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), 'utf8');
const e2eSrc = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const slice = (src, startMarker, endMarker) => {
  const start = src.indexOf(startMarker);
  const end = src.indexOf(endMarker, start);
  return start !== -1 && end > start ? src.slice(start, end + endMarker.length) : '';
};
const coreSrc = slice(laneSrc, 'async function agentTCore(p, o) {', '\n  return null;\n}\n');
const setupSrc = slice(laneSrc, 'const benchInputs = {',
  "(hung-agent guard expired after ${Math.round(AGENT_TIMEOUT_MS / 60000)}min)` : ''));\n}\n");
const retrySrc = slice(e2eSrc, 'const authorRetryable = ', '(r.authored === false && r.final_geomean == null));\n');
const laneArgsSrc = slice(e2eSrc, 'const laneArgs = (wfArgs) => {', '\n  return out;\n};\n');
ok(coreSrc !== '' && setupSrc !== '', 'agentTCore and the benchmark setup located in kernel_lane.js');
ok(retrySrc !== '' && laneArgsSrc !== '', 'authorRetryable and laneArgs located in e2e_workflow.js');

const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const BASELINE = [{ name: 'c0', latency_ms: 1.25 }];
const GOOD = { commandment_path: '/eval/COMMANDMENT.md', baseline_per_case: BASELINE, baseline_geomean_ms: 1.25 };

// Runs the lane's setup block on the real agentTCore. `plan` maps a label to 'hang', 'throw', or a result.
async function runSetup(plan) {
  const logs = [];
  const prompts = {};
  const agent = (p, o) => {
    prompts[o.label] = p;
    const step = plan[o.label];
    if (step === 'hang') return new Promise(() => {});
    if (step === 'throw') return Promise.reject(new Error('gateway 503'));
    return Promise.resolve(step === undefined ? null : step);
  };
  const run = new AsyncFunction('agent', 'log', 'roleAgent',
    `const AGENT_TIMEOUT_MS = 30, AGENT_RETRIES = 1, ABL_B6 = false;
     const guardExpired = new Set();
     const tlAgent = () => null, ablDrainPrelude = () => '';
     ${coreSrc}
     const agentT = (p, o) => agentTCore(p, o);
     const CANONICAL = '/eval/ws', EVAL_DIR = '/eval', WORKFLOW_DIR = '/wf', GPU_POOL = '1', analysis = {};
     const HARNESS_ADDENDUM = '', WORKLOAD_SPEC_PATH = '', WORKLOAD_SPEC = null, BENCH_SCHEMA = {};
     ${setupSrc}
     return bench;`);
  const roleAgent = (role, phase, intro, inputs) => JSON.stringify({ role, phase, intro, inputs });
  let bench = null, error = '';
  try { bench = await run(agent, (m) => logs.push(m), roleAgent); } catch (e) { error = String(e && e.message); }
  return { bench, error, logs, prompts };
}

(async () => {
  let r = await runSetup({ benchmark_engineer: GOOD });
  ok(r.bench === GOOD && !('benchmark_engineer:resume' in r.prompts), 'a setup that returns its baseline is not resumed');

  r = await runSetup({ benchmark_engineer: 'hang', 'benchmark_engineer:resume': GOOD });
  ok(r.bench === GOOD && r.error === '', 'a setup cut by the guard is resumed and the lane continues');
  ok(/"SETUP_RESUME":true/.test(r.prompts['benchmark_engineer:resume'] || ''), 'the resume carries SETUP_RESUME');
  ok(r.logs.some((l) => /\[setup-resume\]/.test(l)), 'the resume is logged');

  r = await runSetup({ benchmark_engineer: 'hang', 'benchmark_engineer:resume': 'hang' });
  ok(/no baseline recorded \(hung-agent guard expired after \d+min\)/.test(r.error), 'a resume the guard also cuts fails naming the guard');

  r = await runSetup({ benchmark_engineer: 'throw' });
  ok(/no baseline recorded$/.test(r.error) && !('benchmark_engineer:resume' in r.prompts),
    'an API failure is not resumed and does not name the guard');

  r = await runSetup({ benchmark_engineer: { commandment_path: '/eval/COMMANDMENT.md' } });
  ok(/no baseline recorded$/.test(r.error) && !('benchmark_engineer:resume' in r.prompts),
    'a returned setup without a baseline is not resumed');

  const authorRetryable = new Function(`${retrySrc}\nreturn authorRetryable;`)();
  const cut = { authored: false, validation_status: 'error',
    reason: 'Error: Benchmark setup failed: no baseline recorded (hung-agent guard expired after 60min)' };
  ok(authorRetryable(cut) === false, 'the author loop does not retry a lane whose setup the guard cut');
  ok(authorRetryable({ authored: false, validation_status: 'error', reason: 'Error: Benchmark setup failed: no baseline recorded' }) === true,
    'it still retries any other lane error');
  ok(authorRetryable(null) === true && authorRetryable({ authored: false, final_geomean: null }) === true,
    'it still retries a lane that returned nothing');
  ok(authorRetryable({ authored: true, final_geomean: 0.9, validation_status: 'pass' }) === false,
    'it never retries a completed no-speedup');

  const mkLaneArgs = (A, isAgentx = false) => new Function('A', 'LANE_USE_LEARNED_KB', 'OPTIMIZER', 'ablLaneArgs',
    'GPU_FAMILY', 'EXPECTED_GFX', 'EXPECTED_TARGET', 'EXPECTED_DEVICE_NAME', 'EXPECTED_PHYSICAL_CU_COUNT', 'IS_AGENTX',
    `${laneArgsSrc}\nreturn laneArgs;`)(A, 'false', '', () => ({}), {}, '', '', '', 0, isAgentx);
  ok(!('agent_timeout_ms' in mkLaneArgs({})({ budget: 6 })), 'an unset agent_timeout_ms leaves the lane default');
  ok(mkLaneArgs({ agent_timeout_ms: 21600000 })({ budget: 6 }).agent_timeout_ms === '21600000',
    'an explicit agent_timeout_ms reaches the lane');
  ok(mkLaneArgs({ agent_timeout_ms: 21600000 })({ agent_timeout_ms: '60000' }).agent_timeout_ms === '60000',
    'a per-call value still wins');
  ok(mkLaneArgs({ agent_timeout_ms: 21600000 }, true)({ budget: 6 }).agent_timeout_ms === '21600000',
    'the timeout also reaches a trace replay lane');
  ok(!('min_improve' in mkLaneArgs({})({ budget: 6 })), 'a fixed-load lane keeps its own commit gate');
  ok(mkLaneArgs({}, true)({ budget: 6 }).min_improve === 'noise' &&
    mkLaneArgs({}, true)({ min_improve: '0.01' }).min_improve === '0.01',
    'a trace replay lane gets the noise commit gate unless the call sets one');

  console.log(failures === 0 ? '\nPASS: lane setup survives the hung-agent guard.' : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})();
