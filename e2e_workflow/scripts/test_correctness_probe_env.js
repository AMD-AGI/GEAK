#!/usr/bin/env node
// Regression guard for two run-level policies a long trace-replay run depends on (no GPU, no model):
//
//   1. Simulated speculative acceptance (SGLANG_SIMULATE_ACC_*) keeps throughput comparable but makes
//      the unmodified reference irreproducible, so every greedy-parity probe fails and every candidate
//      is rejected on the simulator rather than on the change. run_e2e forwards the switch names as
//      correctness_unset_env; the roles that run correctness probes are told to serve them without.
//   2. The hung-agent guard is a Promise.race: it stops WAITING, it cannot stop the agent. Retrying
//      after it fires starts a second agent beside the first, on the same GPUs and eval dirs.
//      agent_timeout_retry=false returns null instead, and inside the final reserve so does the default,
//      keeping the running attempt for the caller to adopt.
//
// Both are lifted from the real source and executed; a run that sets neither must see no change.
//
// Run:  node e2e_workflow/scripts/test_correctness_probe_env.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const between = (start, end) => {
  const i = src.indexOf(start);
  const j = i === -1 ? -1 : src.indexOf(end, i);
  return i === -1 || j === -1 ? null : src.slice(i, j);
};

console.log('\n# correctness_unset_env is parsed into plain variable names');
const envSrc = between('const CORRECTNESS_UNSET_ENV', '// The AMD authoring knowledge base');
ok(!!envSrc, 'CORRECTNESS_UNSET_ENV / CORRECTNESS_PROBE_ROLES located');
if (!envSrc) process.exit(1);
const parse = (A) => new Function('A', `${envSrc}\nreturn { CORRECTNESS_UNSET_ENV, CORRECTNESS_PROBE_ROLES };`)(A);
ok(parse({}).CORRECTNESS_UNSET_ENV.length === 0, 'unset => no names');
ok(JSON.stringify(parse({ correctness_unset_env: 'SGLANG_SIMULATE_ACC_LEN, $(touch x),SGLANG_SIMULATE_ACC_METHOD,' })
  .CORRECTNESS_UNSET_ENV) === JSON.stringify(['SGLANG_SIMULATE_ACC_LEN', 'SGLANG_SIMULATE_ACC_METHOD']),
'only identifier-shaped names survive, so nothing else can reach an `env -u` command line');

console.log('\n# the probe block reaches the probing roles only, and only when the switches are set');
const blockSrc = (src.match(/function correctnessEnvBlock\(role\) \{[\s\S]*?\n\}/) || [])[0];
ok(!!blockSrc, 'correctnessEnvBlock(role) located');
if (!blockSrc) process.exit(1);
const blockFor = (A) => {
  const { CORRECTNESS_UNSET_ENV, CORRECTNESS_PROBE_ROLES } = parse(A);
  return new Function('CORRECTNESS_UNSET_ENV', 'CORRECTNESS_PROBE_ROLES', `${blockSrc}\nreturn correctnessEnvBlock;`)(
    CORRECTNESS_UNSET_ENV, CORRECTNESS_PROBE_ROLES);
};
const on = blockFor({ correctness_unset_env: 'SGLANG_SIMULATE_ACC_LEN,SGLANG_SIMULATE_ACC_METHOD' });
for (const role of ['e2e_integrator', 'director', 'config_tuner', 'tuning_specialist']) {
  ok(on(role).includes('env -u SGLANG_SIMULATE_ACC_LEN -u SGLANG_SIMULATE_ACC_METHOD'),
    `${role} is told to serve its correctness probes with both switches unset`);
}
ok(/Throughput legs keep them/.test(on('e2e_integrator')),
  'the block keeps the throughput legs on the simulated config (they ARE the measurement)');
for (const role of ['profiler', 'system_architect', 'kernel_author']) {
  ok(on(role) === '', `${role} runs no correctness probe and sees nothing`);
}
const off = blockFor({});
ok(['e2e_integrator', 'director', 'config_tuner', 'tuning_specialist', 'profiler'].every((r) => off(r) === ''),
  'no switches => every role prompt is byte-identical');
ok(/return base \+ expertSkillsBlock\(role\)[^;]*\+ correctnessEnvBlock\(role\);/.test(src),
  'roleAgent appends the block after base (additive, never rewrites it)');
ok(/\.\.\.\(correctnessEnvBlock\(role\) \? \{ CORRECTNESS_UNSET_ENV: /.test(src),
  'the Inputs block names the switches only where the prompt explains them');

console.log('\n# the task-accuracy gate reaches every integration A/B, the tuning agent and the Director');
const withTuning = between('const withTuning = {', '};');
ok(!!withTuning && withTuning.includes('...ACCURACY_INPUTS'),
  'runIntegrateBothLegs passes the gate to every integration (WarmStart replay included)');
ok(/ACCURACY_GATE, \.\.\.ACCURACY_INPUTS,/.test(src), 'the tuning agent gets the gate inputs');
ok((src.match(/SKILL_DIR: WORKFLOW_DIR, \.\.\.ACCURACY_INPUTS,/g) || []).length >= 2,
  'both Director validation briefs get the gate inputs');
const accSrc = between('const ACCURACY_GATE =', '// Simulated speculative acceptance');
ok(!!accSrc, 'ACCURACY_* block located');
const accFor = (A) => new Function('A', 'WORKFLOW_DIR', `${accSrc}\nreturn ACCURACY_INPUTS;`)(A, '/wf');
ok(Object.keys(accFor({})).length === 0, "accuracy_gate unset => ACCURACY_INPUTS is empty, the spread adds nothing");
ok(accFor({ accuracy_gate: 'gsm8k', accuracy_limit: 400, accuracy_tol: 0.02 }).ACCURACY_LIMIT === 400,
  'accuracy_gate=gsm8k carries the operator limit through');

console.log('\n# a hung agent is not retried beside itself when agent_timeout_retry=false');
const retrySrc = (src.match(/const AGENT_TIMEOUT_RETRY = [^\n]*\n/) || [])[0];
ok(!!retrySrc, 'AGENT_TIMEOUT_RETRY located');
const retryFor = (A) => new Function('A', `${retrySrc}\nreturn AGENT_TIMEOUT_RETRY;`)(A);
ok(retryFor({}) === true, 'retry stays the default');
ok(retryFor({ agent_timeout_retry: 'false' }) === false, "agent_timeout_retry='false' turns it off");
ok(retryFor({ agent_timeout_retry: 'true' }) === true, "agent_timeout_retry='true' keeps it on");

const loopSrc = between('function agentBounded(', '// --- FlyDSL provisioning gate');
ok(!!loopSrc && loopSrc.includes('async function safeAgent('), 'agentBounded + safeAgent located');
if (!loopSrc) process.exit(1);
const harness = (retry, agentImpl, inReserve = false) => {
  const calls = { agent: 0 };
  const logs = [];
  const { safeAgent, ABANDONED_RUNS } = new Function(
    'withProcessSafety', 'agentTimeoutFor', 'agent', 'log', 'tlAgent',
    'LLM_TL', 'AGENT_TIMEOUT_RETRY', 'atReserveFloor',
    `${loopSrc}\nreturn { safeAgent, ABANDONED_RUNS };`)(
    (p) => p, () => 5,
    (...a) => { calls.agent++; return agentImpl(...a); },
    (m) => logs.push(m), () => null, { nested: [] }, retry, () => inReserve);
  return { safeAgent, ABANDONED_RUNS, calls, logs };
};
const never = () => new Promise(() => {});

(async () => {
  const noRetry = harness(false, never);
  ok(await noRetry.safeAgent('p', { label: 'integrate x' }) === null && noRetry.calls.agent === 1,
    `retry off: a hung attempt is reported null after ONE dispatch (got ${noRetry.calls.agent})`);
  ok(noRetry.logs.some((m) => /without a retry/.test(m)), 'and the log says why');

  ok(noRetry.ABANDONED_RUNS.size === 0, 'outside the reserve the abandoned attempt is not kept for adoption');

  const retry = harness(true, never);
  await retry.safeAgent('p', { label: 'integrate x' });
  ok(retry.calls.agent === 3, `retry on (default): unchanged, 3 dispatches (got ${retry.calls.agent})`);

  // Inside the reserve every retry gets the 2min floor and runs beside the attempt it replaced.
  const reserve = harness(true, never, true);
  ok(await reserve.safeAgent('p', { label: 'warm_start integrate k' }) === null && reserve.calls.agent === 1,
    `retry on, inside the reserve: ONE dispatch, no retry (got ${reserve.calls.agent})`);
  ok(reserve.ABANDONED_RUNS.has('warm_start integrate k') &&
    typeof reserve.ABANDONED_RUNS.get('warm_start integrate k').then === 'function',
  'the still-running attempt is kept under its label, so its caller can adopt the verdict');
  ok(reserve.logs.some((m) => /final reserve has begun/.test(m)), 'and the log says why');

  const empty = harness(false, async () => null);
  await empty.safeAgent('p', { label: 'x' });
  ok(empty.calls.agent === 3, `retry off still retries an attempt that RETURNED empty (got ${empty.calls.agent})`);

  const fine = harness(false, async () => ({ ok: 1 }));
  const r = await fine.safeAgent('p', { label: 'x' });
  ok(r && r.ok === 1 && fine.calls.agent === 1, 'a returning agent is passed through untouched');

  console.log(failures ? `\nFAILED (${failures})` : '\nPASS');
  process.exit(failures ? 1 : 0);
})();
