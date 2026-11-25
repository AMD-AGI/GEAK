import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  AGENT_TAG_VAR, pidsWithTag, reapTagged, waitTagGone,
} from '../lib/agent_reaper.mjs';

// A fake /proc, so the scanning rules can be asserted without racing real processes.
function fakeProc(processes) {
  const root = mkdtempSync(join(tmpdir(), 'avo-proc-'));
  for (const [pid, environ] of Object.entries(processes)) {
    mkdirSync(join(root, pid), { recursive: true });
    if (environ !== null) {
      writeFileSync(join(root, pid, 'environ'), `${environ.join('\0')}\0`);
    }
  }
  mkdirSync(join(root, 'self'), { recursive: true });
  writeFileSync(join(root, 'sys'), '');
  return root;
}

const sleep = (ms) => new Promise((done) => { setTimeout(done, ms); });
const alive = (pid) => {
  try { process.kill(pid, 0); return true; } catch { return false; }
};

test('the tag finds descendants and ignores everything else', () => {
  const proc = fakeProc({
    100: [`${AGENT_TAG_VAR}=abc`, 'PATH=/usr/bin'],
    101: ['PATH=/usr/bin'],
    102: [`${AGENT_TAG_VAR}=other`],
    // A substring of the tag must not match: tags are compared whole.
    103: [`${AGENT_TAG_VAR}=abcdef`],
    // Another user's process, unreadable rather than absent.
    104: null,
    105: [`${AGENT_TAG_VAR}=abc`],
  });
  assert.deepEqual(pidsWithTag('abc', { proc, self: 0 }).sort((a, b) => a - b), [100, 105]);
  assert.deepEqual(pidsWithTag('', { proc, self: 0 }), []);
  assert.deepEqual(pidsWithTag('abc', { proc, self: 100 }), [105]);
});

test('a missing /proc yields no victims rather than throwing', () => {
  assert.deepEqual(pidsWithTag('abc', { proc: join(tmpdir(), 'avo-no-such-proc') }), []);
});

test('reaping escalates to SIGKILL only for what survived the grace period', () => {
  const proc = fakeProc({ 200: [`${AGENT_TAG_VAR}=t`], 201: [`${AGENT_TAG_VAR}=t`] });
  const signals = [];
  let escalate = null;
  const count = reapTagged('t', {
    proc,
    self: 0,
    kill: (pid, signal) => signals.push([pid, signal]),
    schedule: (fn) => { escalate = fn; return { unref() {} }; },
  });
  assert.equal(count, 2);
  assert.deepEqual(signals, [[200, 'SIGTERM'], [201, 'SIGTERM']]);
  escalate();
  assert.deepEqual(signals.slice(2), [[200, 'SIGKILL'], [201, 'SIGKILL']]);
});

test('nothing tagged means no signals and no escalation is scheduled', () => {
  const proc = fakeProc({ 300: ['PATH=/usr/bin'] });
  let scheduled = false;
  const count = reapTagged('t', {
    proc,
    self: 0,
    kill: () => assert.fail('signalled an untagged process'),
    schedule: () => { scheduled = true; return { unref() {} }; },
  });
  assert.equal(count, 0);
  assert.equal(scheduled, false);
});

test('a background loop that escapes the agent process group is still reaped', async () => {
  // The real failure: a worker wrote its own driver script and launched it with
  // setsid, so it left the agent's process group. Killing that group could not
  // reach it, and it relaunched a benchmark after the run had been torn down.
  const tag = `test-${process.pid}-${Date.now()}`;
  const script = join(mkdtempSync(join(tmpdir(), 'avo-escape-')), 'loop.sh');
  writeFileSync(script, 'while true; do sleep 0.2; done\n');

  // The escapee's own stdio goes to /dev/null: inheriting the pipe would hold it
  // open and the agent would never look finished.
  const pidFile = `${script}.pid`;
  const agent = spawn('/bin/sh', [
    '-c', `setsid /bin/sh ${script} >/dev/null 2>&1 & echo $! > ${pidFile}`,
  ], {
    env: { ...process.env, [AGENT_TAG_VAR]: tag },
    stdio: 'ignore',
    detached: true,
  });
  await new Promise((done) => agent.on('exit', done));
  const out = readFileSync(pidFile, 'utf8');

  // The agent itself is gone, exactly as after a normal verdict.
  const escapee = Number(out.trim());
  assert.ok(escapee > 0, `no background pid captured (${JSON.stringify(out)})`);
  await sleep(200);
  assert.equal(alive(escapee), true, 'the escapee should outlive its agent');

  let group = -1;
  try { group = process.getpgid(escapee); } catch { /* raced */ }
  assert.notEqual(group, agent.pid, 'setsid should have left the agent process group');

  assert.ok(pidsWithTag(tag).includes(escapee), 'the tag should still find it');
  assert.equal(reapTagged(tag, { graceMs: 500 }) >= 1, true);

  for (let waited = 0; waited < 40 && alive(escapee); waited += 1) {
    await sleep(50);
  }
  assert.equal(alive(escapee), false, 'the escaped loop should have been reaped');
});

test('waiting returns 0 as soon as no tagged process is left', async () => {
  const proc = fakeProc({ 300: ['PATH=/usr/bin'] });
  const started = Date.now();
  assert.equal(await waitTagGone('t', { proc, self: 0, timeoutMs: 2000, pollMs: 20 }), 0);
  assert.ok(Date.now() - started < 500);
});

test('waiting gives up at the deadline and reports what is still alive', async () => {
  const proc = fakeProc({ 400: [`${AGENT_TAG_VAR}=t`], 401: [`${AGENT_TAG_VAR}=t`] });
  const started = Date.now();
  assert.equal(await waitTagGone('t', { proc, self: 0, timeoutMs: 200, pollMs: 20 }), 2);
  assert.ok(Date.now() - started >= 200);
});
