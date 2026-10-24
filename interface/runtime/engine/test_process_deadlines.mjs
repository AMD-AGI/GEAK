// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import test from 'node:test';
import { createRuntime } from './run_workflow.mjs';
import { runProcess } from './backends/base.mjs';
import { schemaInstruction } from './schema.mjs';

const guard = await readFile(new URL('../../../kernel_workflow/src/shared/agent_guard.js', import.meta.url), 'utf8');
const boundedArgs = () => ({
  deadline_epoch: Date.now() / 1000 + 30,
  search_reserve_s: 10,
  final_cleanup_reserve_s: 1,
});
const quiet = { log: () => {} };

async function fixture(body, action) {
  const dir = await mkdtemp(join(tmpdir(), 'geak-process-deadlines-'));
  const path = join(dir, 'workflow.js');
  try {
    await writeFile(path, body);
    return await action(path);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test('bounded guard waits for the real role process to terminate and settle', async () => {
  const runs = [];
  const rt = createRuntime({ ...quiet, backend: {
    name: 'test',
    async runAgent({ timeoutMs }) {
      const result = await runProcess({ cmd: process.execPath, args: ['-e', `
        process.on('SIGTERM', () => setTimeout(() => {
          process.stdout.write('settled\\n'); process.exit(0);
        }, 120));
        process.stdout.write(process.pid + '\\n');
        setInterval(() => {}, 1000);
      `], timeoutMs });
      runs.push(result);
      if (result.status === 'timeout') throw new Error('role timeout');
      return { text: result.stdout };
    },
  } });
  await fixture(`
    const AGENT_TIMEOUT_MS = 300, AGENT_RETRIES = 3;
    const tlAgent = () => null;
    ${guard}
    return await agentT('role', {label: 'role'});
  `, async path => {
    assert.equal(await rt.runScript(path, boundedArgs(), 0), null);
    assert.equal(runs.length, 1, 'the guard must await backend cleanup before returning');
    const result = runs[0];
    assert.equal(result.status, 'timeout');
    assert.equal(result.group_cleared, true);
    assert.equal(result.stdio_held, false);
    assert.match(result.stdout, /settled/);
    assert.ok(result.duration_ms >= 400, `returned before signal handler finished: ${result.duration_ms}`);
    const pid = Number(result.stdout.split('\n')[0]);
    assert.throws(() => process.kill(-pid, 0), { code: 'ESRCH' });
  });
});

test('queued calls and schema retries recheck their cutoff before spawning', async () => {
  for (const retry of [false, true]) {
    let calls = 0;
    const rt = createRuntime({ ...quiet, concurrency: 1, schemaRetries: 2, backend: {
      name: 'test',
      async runAgent() {
        calls++;
        await delay(250);
        return { text: retry ? 'invalid JSON' : 'done' };
      },
    } });
    const options = { processDeadlineEpoch: Date.now() / 1000 + 7.15 };
    if (retry) {
      await assert.rejects(rt.agent('retry', { ...options, schema: { type: 'object' } }),
        { code: 'GEAK_PROCESS_DEADLINE' });
      assert.equal(rt.state.schemaFails, 1);
    } else {
      const settled = await Promise.allSettled([rt.agent('first', options), rt.agent('queued', options)]);
      assert.equal(settled[0].value, 'done');
      assert.equal(settled[1].status, 'rejected');
      assert.equal(settled[1].reason.code, 'GEAK_PROCESS_DEADLINE');
    }
    assert.equal(calls, 1);
  }
});

test('expired bounded guards do not retry or launch a backend', async () => {
  let calls = 0;
  const events = [];
  const rt = createRuntime({ ...quiet, backend: { name: 'test', async runAgent() {
    calls++; return { text: 'unexpected' };
  } } });
  await fixture(`
    const AGENT_TIMEOUT_MS = 300, AGENT_RETRIES = 3;
    const tlAgent = () => args.events.push('dispatch');
    ${guard}
    setProcessDeadline(args.cutoff);
    return await agentT('expired');
  `, async path => {
    assert.equal(await rt.runScript(path, {
      ...boundedArgs(), cutoff: Date.now() / 1000 - 1, events,
    }, 0), null);
    assert.equal(calls, 0);
    assert.deepEqual(events, ['dispatch']);
  });
});

test('schema retries share one cumulative role timeout', async () => {
  const timeouts = [];
  const rt = createRuntime({ ...quiet, schemaRetries: 3, backend: {
    name: 'test', async runAgent({ timeoutMs }) {
      timeouts.push(timeoutMs);
      await delay(90);
      return { text: 'invalid JSON' };
    },
  } });
  await assert.rejects(rt.agent('retry', {
    processDeadlineEpoch: Date.now() / 1000 + 30, processTimeoutMs: 150,
    schema: { type: 'object' },
  }), { code: 'GEAK_AGENT_TIMEOUT' });
  assert.equal(timeouts.length, 2);
  assert.ok(timeouts[0] <= 150 && timeouts[1] > 0 && timeouts[1] < 70, String(timeouts));
});

test('raw commands settle at the scoped deadline and expired commands never spawn', async () => {
  const rt = createRuntime({ ...quiet, backend: { name: 'unused' } });
  await fixture(`
    setProcessDeadline(args.cutoff);
    const result = await runProcess({cmd: args.node, args: ['-e', 'setInterval(() => {}, 1000)'],
      timeoutMs: 0, graceMs: 50, settleMs: 100});
    setProcessDeadline(args.expired);
    let expired;
    try { await runProcess({cmd: '/this-must-not-spawn'}); }
    catch (error) { expired = error.code; }
    return {result, expired};
  `, async path => {
    const result = await rt.runScript(path, {
      ...boundedArgs(), node: process.execPath,
      cutoff: Date.now() / 1000 + 0.6, expired: Date.now() / 1000 - 1,
    }, 0);
    assert.equal(result.result.status, 'timeout');
    assert.equal(result.result.group_cleared, true);
    assert.equal(result.result.stdio_held, false);
    assert.ok(result.result.duration_ms < 1000);
    assert.equal(result.expired, 'GEAK_PROCESS_DEADLINE');
  });
});

test('scoped allowances retain the original ceiling and ignore later wall-clock jumps', async () => {
  const timeouts = [];
  const args = boundedArgs();
  const rt = createRuntime({ ...quiet, backend: { name: 'test', async runAgent({ timeoutMs }) {
    timeouts.push(timeoutMs); return { text: 'done' };
  } } });
  await fixture(`
    setProcessDeadline(args.deadline_epoch - args.search_reserve_s);
    await agent('search');
    setProcessDeadline(args.deadline_epoch - 5);
    await agent('report');
    setProcessDeadline(args.deadline_epoch - args.final_cleanup_reserve_s);
    await agent('validate');
    try { setProcessDeadline(args.deadline_epoch); }
    catch (error) { return error.message; }
  `, async path => {
    const originalNow = Date.now;
    try {
      Date.now = () => originalNow() + 3_600_000;
      assert.match(await rt.runScript(path, args, 0), /exceeds original workflow allowance/);
    } finally { Date.now = originalNow; }
    assert.equal(timeouts.length, 3);
    for (const [i, expected] of [13_000, 18_000, 22_000].entries()) {
      assert.ok(timeouts[i] <= expected && timeouts[i] > expected - 1000, String(timeouts));
    }
  });
});

test('without cutoff arguments the runtime preserves the configured agent timeout', async () => {
  const timeouts = [];
  const rt = createRuntime({ ...quiet, agentTimeoutMs: 12345, backend: {
    name: 'test', async runAgent({ timeoutMs }) { timeouts.push(timeoutMs); return { text: 'done' }; },
  } });
  await fixture(`return {active: agent.processDeadlineActive, result: await agent('default')};`, async path => {
    assert.deepEqual(await rt.runScript(path, {}, 0), { active: false, result: 'done' });
    assert.deepEqual(timeouts, [12345]);
  });
});

test('budget guidance matches the bound after queue, schema retry and phase clipping', async () => {
  const seen = [];
  const schema = { type: 'object' };
  const rt = createRuntime({ ...quiet, concurrency: 1, schemaRetries: 1, agentTimeoutMs: 4000, backend: {
    name: 'test', async runAgent({ label, prompt, timeoutMs }) {
      const match = prompt.match(/\n\nBackend call allowance at dispatch: (\d+) ms\. Backend cutoff \(UTC\): (\S+)\./);
      assert.ok(match, 'finite backend calls must receive their time bound');
      assert.equal(prompt.slice(0, match.index), label + (label === 'queued' ? schemaInstruction(schema) : ''),
        'original prompt and schema prefix are unchanged');
      assert.equal(Number(match[1]), timeoutMs, 'prompt and enforced timeout must agree');
      const entry = { label, timeoutMs, cutoff: Date.parse(match[2]) };
      assert.ok(Number.isFinite(entry.cutoff));
      seen.push(entry);
      await delay(label === 'first' ? 120 : 60);
      return { text: label === 'queued' && seen.filter(e => e.label === label).length === 1
        ? 'invalid JSON' : label === 'queued' ? '{}' : 'done' };
    },
  } });
  const options = { processDeadlineEpoch: Date.now() / 1000 + 30, processTimeoutMs: 800 };
  await Promise.all([
    rt.agent('first', { ...options, label: 'first' }),
    rt.agent('queued', { ...options, label: 'queued', schema }),
  ]);
  assert.equal(seen.length, 3);
  assert.ok(seen[1].timeoutMs < seen[0].timeoutMs - 80, 'queue time is charged');
  assert.ok(seen[2].timeoutMs < seen[1].timeoutMs - 40, 'schema retry time is charged');
  assert.ok(Math.abs(seen[2].cutoff - seen[1].cutoff) < 100, 'retry cannot reset the visible cutoff');

  const phaseCutoff = Date.now() / 1000 + 9;
  await fixture(`setProcessDeadline(args.cutoff); return await agent('phase', {label: 'phase'});`,
    path => rt.runScript(path, { ...boundedArgs(), cutoff: phaseCutoff }, 0));
  const phase = seen.at(-1);
  assert.ok(phase.timeoutMs > 0 && phase.timeoutMs < 2100, 'phase cutoff leaves existing cleanup reserve');
  assert.ok(Math.abs(phase.cutoff - (phaseCutoff * 1000 - 7000)) < 100);

  const unlimited = createRuntime({ ...quiet, agentTimeoutMs: 0, backend: {
    name: 'test', async runAgent({ prompt, timeoutMs }) {
      assert.equal(timeoutMs, 0);
      assert.equal(prompt, 'unlimited', 'no invented limit for an unlimited call');
      return { text: 'done' };
    },
  } });
  assert.equal(await unlimited.agent('unlimited'), 'done');
});
