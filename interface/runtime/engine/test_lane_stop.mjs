// Exercise the actual lane helper, runtime semaphore and process reaper without a GPU/model.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { createRuntime } from './run_workflow.mjs';

const kernelDir = fileURLToPath(new URL('../../../kernel_workflow/', import.meta.url));
const source = await readFile(join(kernelDir, 'kernel_lane.js'), 'utf8');
const start = source.indexOf('const STOP_LANE_SCHEMA =');
const end = source.indexOf('// WALL-CLOCK DEADLINE', start);
assert.ok(start >= 0 && end > start);
const helperSource = source.slice(start, end);
const receipt = { ok: true, blocked: true, stopped: [], survivors: [] };

function laneHelper(agent, fallback = async () => { throw new Error('unexpected cleanup agent'); },
  timedOut = new Set()) {
  const logs = [];
  const obj = (properties, required) => ({ type: 'object', properties, required });
  const { stop, stopAfter } = new Function('agent', 'agentT', 'log', 'obj', 'WORKFLOW_DIR', 'TIMED_OUT_LABELS',
    `${helperSource}\nreturn { stop: stopLaneWork, stopAfter: stopAfterTimeout };`)(
    agent, fallback, m => logs.push(m), obj, kernelDir, timedOut);
  return { stop, stopAfter, logs };
}

function deferred() {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
}

async function bounded(promise) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('cleanup blocked behind occupied agent slot')), 5000);
    })]);
  } finally { clearTimeout(timer); }
}

test('cleanup stops lane work while every agent slot remains occupied', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'geak-lane-stop-'));
  // Also exercise literal argv handling; these characters must never become shell code.
  const lane = join(temp, "lane with 'quote' $(literal)");
  await mkdir(join(lane, 'workspace'), { recursive: true });
  const work = spawn('sleep', ['300'], { cwd: join(lane, 'workspace'), stdio: 'ignore' });
  const exited = once(work, 'exit');
  await once(work, 'spawn');
  const entered = deferred();
  const finish = deferred();
  const calls = [];
  const rt = createRuntime({ concurrency: 1, agentTimeoutMs: 0, log: () => {}, backend: {
    name: 'fake',
    runAgent: async ({ label }) => {
      calls.push(label);
      if (label === 'benchmark') { entered.resolve(); await finish.promise; }
      return { text: 'done' };
    },
  } });
  const abandoned = rt.agent('benchmark', { label: 'benchmark' });
  await entered.promise; // The lane guard may stop waiting; this agent still owns the sole slot.
  const queued = rt.agent('next', { label: 'next' });
  try {
    const { stop } = laneHelper(rt.agent);
    const result = await bounded(stop(lane, 'benchmark returned no baseline', 'Benchmark'));
    assert.equal(result.ok, true);
    assert.equal(result.blocked, true);
    assert.ok(result.stopped.some(p => p.pid === work.pid));
    await bounded(exited);
    assert.match(await readFile(join(lane, '.geak_stop'), 'utf8'), /no baseline/);
    assert.deepEqual(calls, ['benchmark'], 'cleanup did not enter or free the ordinary agent queue');
    assert.equal(rt.state.spawned, 2, 'cleanup did not consume a model call');
  } finally {
    finish.resolve();
    await Promise.all([abandoned, queued]);
    if (work.exitCode === null && work.signalCode === null) work.kill('SIGKILL');
    await exited;
    await rm(temp, { recursive: true, force: true });
  }
  assert.deepEqual(calls, ['benchmark', 'next']);
});

test('concurrent stops for the same lane share one host command', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'geak-lane-stop-'));
  const rt = createRuntime({ backend: { name: 'unused' }, log: () => {} });
  try {
    const first = rt.agent.stopLaneWork(dir, 'first reason');
    const second = rt.agent.stopLaneWork(dir, 'second reason');
    assert.equal(first, second);
    assert.equal((await bounded(first)).blocked, true);
    assert.equal(await readFile(join(dir, '.geak_stop'), 'utf8'), 'first reason\n');
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('host failure is nonfatal and never falls back into the saturated queue', async () => {
  const rt = createRuntime({ backend: { name: 'unused' }, log: () => {} });
  const dir = await mkdtemp(join(tmpdir(), 'geak-lane-stop-'));
  try {
    const { stop, logs } = laneHelper(rt.agent);
    assert.equal(await stop(join(dir, 'missing'), 'failed', 'Benchmark'), null);
    assert.match(logs.join('\n'), /host cleanup failed.*NON-FATAL/s);
    assert.equal(rt.state.spawned, 0);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test('native Workflow without the hook retains its cleanup-agent fallback', async () => {
  const calls = [];
  const { stop } = laneHelper(async () => {}, async (prompt, opts) => {
    calls.push({ prompt, opts });
    return receipt;
  });
  assert.equal(await stop('/review/exp/lane', 'no result', 'Author'), receipt);
  assert.equal(calls.length, 1);
  assert.match(calls[0].prompt, /stop_lane_work\.py.*--eval-dir '\/review\/exp\/lane'/);
  assert.equal(calls[0].opts.label, 'stop:lane');
  assert.equal(calls[0].opts.phase, 'Author');
  assert.equal(await stop('', 'no dir', 'Author'), null);
  assert.equal(calls.length, 1);
});

test('a timed-out agent the lane carries on without is stopped in its own dir only', async () => {
  const temp = await mkdtemp(join(tmpdir(), 'geak-lane-stop-'));
  const out = join(temp, 'round_1', 'engineer_0');
  await mkdir(join(out, 'workspace'), { recursive: true });
  await mkdir(join(out, 'verify'), { recursive: true });
  const eng = spawn('sleep', ['300'], { cwd: join(out, 'workspace'), stdio: 'ignore' });
  const ver = spawn('sleep', ['300'], { cwd: join(out, 'verify'), stdio: 'ignore' });
  const engExited = once(eng, 'exit');
  await Promise.all([once(eng, 'spawn'), once(ver, 'spawn')]);
  const rt = createRuntime({ backend: { name: 'unused' }, log: () => {} });
  const timedOut = new Set(['eng r1_d0:memory']);
  try {
    const { stopAfter, logs } = laneHelper(rt.agent, undefined, timedOut);
    // A result, or a null the guard did not cause (an API fault), stops nothing.
    assert.deepEqual(await stopAfter({ ok: 1 }, 'eng r1_d0:memory', join(out, 'workspace'), 'Optimize'), { ok: 1 });
    assert.equal(await stopAfter(null, 'verify r1_d0', join(out, 'verify'), 'Verify'), null);
    assert.equal(logs.length, 0);
    assert.equal(await stopAfter(null, 'eng r1_d0:memory', join(out, 'workspace'), 'Optimize'), null);
    await bounded(engExited);
    assert.match(await readFile(join(out, 'workspace', '.geak_stop'), 'utf8'), /eng r1_d0:memory timed out/);
    assert.equal(timedOut.size, 0, 'each timeout is handled once');
    assert.equal(ver.exitCode, null, 'the sibling verify dir is left running');
    await assert.rejects(readFile(join(out, 'verify', '.geak_stop')), /ENOENT/);
    assert.equal(rt.state.spawned, 0);
  } finally {
    for (const p of [eng, ver]) if (p.exitCode === null && p.signalCode === null) p.kill('SIGKILL');
    await rm(temp, { recursive: true, force: true });
  }
});

test('the guard records its timeouts and the engineer, verify and integrate sites stop their own dirs', () => {
  const guard = source.slice(source.indexOf('[hung-agent guard]'), source.indexOf('resolve(null);', source.indexOf('[hung-agent guard]')));
  assert.match(guard, /TIMED_OUT_LABELS\.add\(label\);/);
  assert.match(source, /await stopAfterTimeout\(eng, `\$\{d\.specialty === 'deep_explore' \? 'deep' : 'eng'\} \$\{d\.id\}:\$\{d\.specialty\}`,\s*`\$\{d\.out_dir\}\/workspace`, 'Optimize'\);/);
  assert.match(source, /\.then\(\(ver\) => stopAfterTimeout\(ver, vlabel, `\$\{d\.out_dir\}\/verify`, 'Verify'\)\)/);
  assert.match(source, /await stopAfterTimeout\(integrate, `integrate r\$\{round\}`, `\$\{EVAL_DIR\}\/round_\$\{round\}\/integrate`, 'Merge'\);/);
});
