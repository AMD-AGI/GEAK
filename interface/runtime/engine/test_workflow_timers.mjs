// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const runtime = new URL('./run_workflow.mjs', import.meta.url).href;
const child = `
  import assert from 'node:assert/strict';
  const [runtime, failure, waiting, late, scenario] = process.argv.slice(1);
  const { createRuntime } = await import(runtime);
  let started;
  const ready = new Promise(resolve => { started = resolve; });
  const rt = createRuntime({
    backend: {
      name: 'test',
      async runAgent({ label }) {
        if (label === 'fail') {
          if (scenario === 'deadline') await ready;
          throw new Error('expected failure');
        }
        if (label === 'slow') return ready.then(() => ({ text: 'done' }));
        return new Promise(() => {});
      },
    },
    log: () => {},
  });
  if (scenario === 'late') {
    let created;
    const timerCreated = new Promise(resolve => { created = resolve; });
    assert.equal(await rt.runScript(late, { created }, 0), 'early');
    started();
    await timerCreated;
  } else {
    const failed = rt.runScript(failure, {}, 0).then(
      () => { throw new Error('workflow should fail'); },
      error => assert.equal(error.message, 'expected failure'),
    );
    if (scenario === 'deadline') {
      assert.equal(await rt.runScript(waiting, { started }, 0), 'deadline fired');
    }
    await failed;
  }
  console.log(scenario);
`;

test('settled workflows release timers without disabling a live workflow deadline', () => {
  const dir = mkdtempSync(join(tmpdir(), 'geak-workflow-timers-'));
  try {
    const failure = join(dir, 'failure.js');
    const waiting = join(dir, 'waiting.js');
    const late = join(dir, 'late.js');
    writeFileSync(failure, `
      setTimeout(() => {}, 30_000);
      await agent('fail', { label: 'fail' });
    `);
    writeFileSync(waiting, `
      setTimeout(() => {}, 30_000);
      const deadline = new Promise(resolve => setTimeout(resolve, 150, 'deadline fired'));
      args.started();
      return await Promise.race([
        agent('unresolved', { label: 'unresolved' }),
        deadline,
      ]);
    `);
    writeFileSync(late, `
      (async () => {
        await agent('slow', { label: 'slow' });
        setTimeout(() => {}, 30_000);
        args.created();
      })();
      return 'early';
    `);
    for (const scenario of ['failure', 'deadline', 'late']) {
      // No process.exit in the child: an abandoned timer must fail this bound.
      const result = spawnSync(process.execPath,
        ['--input-type=module', '-e', child, runtime, failure, waiting, late, scenario],
        { encoding: 'utf8', timeout: 5_000 });
      assert.ifError(result.error);
      assert.equal(result.status, 0, result.stderr);
      assert.equal(result.stdout.trim(), scenario);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
