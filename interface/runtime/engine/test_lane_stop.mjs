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

function laneHelper(agent, fallback = async () => { throw new Error('unexpected cleanup agent'); }) {
  const logs = [];
  const obj = (properties, required) => ({ type: 'object', properties, required });
  const stop = new Function('agent', 'agentT', 'log', 'obj', 'WORKFLOW_DIR',
    `${helperSource}\nreturn stopLaneWork;`)(agent, fallback, m => logs.push(m), obj, kernelDir);
  return { stop, logs };
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
