// End-to-end check of how run_avo.mjs ends an agent call: the real shim drives a
// fake `claude` (tests/fixtures/fake_claude.mjs) that replays `result` events and
// then stays alive, the way the real CLI does while an agent still holds a
// background task or a scheduled wakeup.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import {
  chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const WORKFLOW_DIR = path.resolve(HERE, '..');
const SHIM = path.join(WORKFLOW_DIR, 'run_avo.mjs');
const FAKE = path.join(HERE, 'fixtures', 'fake_claude.mjs');
const SCRIPT = path.join(HERE, 'fixtures', 'one_agent_workflow.js');

const ANSWER = { status: 'ok', probe: 'a' };
const REQUIRED = ['status', 'probe'];
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };

// Runs the shim once. `plan` drives the fake claude; returns the workflow's result
// and the shim's stdout.
async function runShim(plan, { timeoutMs, straggler = false } = {}) {
  const root = mkdtempSync(path.join(tmpdir(), 'avo-early-'));
  const kernel = path.join(root, 'kernel');
  mkdirSync(kernel);
  const pidfile = path.join(root, 'straggler.pid');
  const planFile = path.join(root, 'plan.json');
  writeFileSync(planFile, JSON.stringify({
    ...plan, ...(straggler ? { straggler_pidfile: pidfile } : {}),
  }));
  // claude_bin must be one executable; do not rely on `node` being on PATH.
  const bin = path.join(root, 'claude');
  writeFileSync(bin, `#!/bin/sh\nexec "${process.execPath}" "${FAKE}" "$@"\n`);
  chmodSync(bin, 0o755);
  const config = path.join(root, 'config.json');
  writeFileSync(config, JSON.stringify({
    scriptPath: SCRIPT,
    claude_bin: bin,
    claude_home: root,
    root_dir: root,
    agent_retries: 0,
    ...(timeoutMs ? { agent_timeout_ms: timeoutMs } : {}),
    args: {
      kernel_path: kernel,
      workflow_dir: WORKFLOW_DIR,
      exp_root: path.join(root, 'exp'),
      probe_required: REQUIRED,
      ...(straggler ? { straggler_pidfile: pidfile } : {}),
    },
  }));
  let stdout = '';
  const child = spawn(process.execPath, [SHIM, config], {
    env: { ...process.env, FAKE_CLAUDE_PLAN: planFile },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.stdout.on('data', (d) => { stdout += d; });
  child.stderr.on('data', (d) => { stdout += d; });
  await new Promise((done) => { child.on('close', done); });
  try {
    const result = JSON.parse(
      readFileSync(path.join(root, 'exp', '_shim_logs', 'result.json'), 'utf8'));
    return { result, stdout };
  } finally {
    if (existsSync(pidfile)) {
      const pid = Number(readFileSync(pidfile, 'utf8'));
      if (alive(pid)) process.kill(pid, 'SIGKILL');
    }
    rmSync(root, { recursive: true, force: true });
  }
}

test('an agent that exits right after its answer takes the unchanged path', async () => {
  const { result, stdout } = await runShim({
    events: [{ t: 0.3, result: JSON.stringify(ANSWER) }], exit_at: 0.6,
  });
  assert.deepEqual(result.answer, ANSWER);
  assert.ok(result.seconds < 2.5, `took ${result.seconds}s`);
  assert.doesNotMatch(stdout, /✂/);
  assert.match(stdout, /✓ probe/);
});

test('an answered agent that stays alive is stopped after the grace', async () => {
  const { result, stdout } = await runShim({
    events: [{ t: 0.3, result: JSON.stringify(ANSWER) }],
  });
  assert.deepEqual(result.answer, ANSWER);
  assert.ok(result.seconds >= 3 && result.seconds < 7, `took ${result.seconds}s`);
  assert.match(stdout, /✂ final answer received/);
  assert.match(stdout, /✓ probe/);
  assert.doesNotMatch(stdout, /still alive after SIGKILL/);
});

test('progress text does not end the call; the later answer does', async () => {
  const { result, stdout } = await runShim({
    events: [
      { t: 0.2, result: 'still waiting for the background task' },
      { t: 1.0, result: JSON.stringify(ANSWER) },
    ],
  });
  assert.deepEqual(result.answer, ANSWER);
  assert.ok(result.seconds >= 4 && result.seconds < 8, `took ${result.seconds}s`);
  assert.equal((stdout.match(/✂/g) || []).length, 1);
});

test('JSON quoted inside prose is not taken as the answer', async () => {
  const { result } = await runShim({
    events: [
      { t: 0.2, result: 'Draft so far: {"status":"ok","probe":"draft"} - will confirm' },
      { t: 4.5, result: JSON.stringify(ANSWER) },
    ],
  });
  // Had the draft counted, the call would have ended at 3.2s with probe "draft".
  assert.deepEqual(result.answer, ANSWER);
  assert.ok(result.seconds >= 7.5, `took ${result.seconds}s`);
});

test('error results and results missing a key do not end the call', async () => {
  const { result } = await runShim({
    events: [
      { t: 0.2, result: JSON.stringify({ status: 'ok', probe: 'error' }), is_error: true },
      { t: 0.5, result: JSON.stringify({ status: 'ok' }) },
      { t: 4.5, result: `\`\`\`json\n${JSON.stringify(ANSWER)}\n\`\`\`` },
    ],
  });
  assert.deepEqual(result.answer, ANSWER);
  assert.ok(result.seconds >= 7.5, `took ${result.seconds}s`);
});

test('without a final answer the call still ends on process exit', async () => {
  const { result, stdout } = await runShim({
    events: [{ t: 0.3, result: JSON.stringify({ status: 'ok' }) }], exit_at: 2,
  });
  assert.equal(result.answer, null);
  assert.ok(result.seconds >= 2 && result.seconds < 4, `took ${result.seconds}s`);
  assert.doesNotMatch(stdout, /✂/);
  assert.match(stdout, /✗ probe/);
});

test('a background process that ignores SIGTERM is gone when the call returns', async () => {
  const { result, stdout } = await runShim({
    events: [{ t: 0.3, result: JSON.stringify(ANSWER) }],
  }, { straggler: true });
  assert.deepEqual(result.answer, ANSWER);
  assert.equal(result.straggler_alive, false);
  // grace (3s) + SIGKILL after reapTagged's 5s
  assert.ok(result.seconds >= 8 && result.seconds < 12, `took ${result.seconds}s`);
  assert.doesNotMatch(stdout, /still alive after SIGKILL/);
});

test('an agent that never answers still ends on the wall-clock timeout', async () => {
  const { result, stdout } = await runShim({
    events: [{ t: 0.2, result: 'still working' }],
  }, { timeoutMs: 1500 });
  assert.equal(result.answer, null);
  assert.match(stdout, /agent wall-clock timeout/);
  assert.doesNotMatch(stdout, /✂/);
});
