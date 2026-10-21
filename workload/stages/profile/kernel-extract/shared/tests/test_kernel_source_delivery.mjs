import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { sourceSnapshot, fileRef, verifySourceDelivery } from '../kernel_source_delivery.mjs';

async function fixture(t, nested = false) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'source-delivery-'));
  const task = nested ? path.join(root, 'task') : root;
  if (nested) await fs.mkdir(task);
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await fs.mkdir(path.join(task, 'kernel_src'));
  await fs.mkdir(path.join(task, 'run'));
  const write = async (name, value) => {
    await fs.writeFile(path.join(task, name), JSON.stringify(value));
    return fileRef(task, name);
  };
  await write('meta.json', { target_callable: 'pkg:op' });
  for (const name of ['cases.py', 'unittest.py', 'kernel_src/score.py', 'kernel_src/topk.cu'])
    await fs.writeFile(path.join(task, name), '# original source\n');
  const snapshot = await sourceSnapshot(task);
  const m = { contract: 'kernel_source_delivery/v1', target_callable: 'pkg:op', ...snapshot, kernels: [] };
  m.ut_result_ref = await write('run/ut.json', { ok: true, exit_code: 0,
    target_callable: m.target_callable, snapshot_sha256: m.snapshot_sha256 });
  for (const device_kernel of ['score', 'topk']) {
    const row = { device_kernel, source_files: [`kernel_src/${device_kernel === 'score' ? 'score.py' : 'topk.cu'}`] };
    for (const [field, kind] of [['execution_ref', 'candidate_execution'], ['negative_control_ref', 'numerical_rejection']]) {
      const log = `run/${device_kernel}_${kind}.log`;
      await fs.writeFile(path.join(task, log), 'fixture GPU trace\n');
      row[field] = await write(`run/${device_kernel}_${kind}.json`, {
        kind, ok: true, device_kernel, target_callable: m.target_callable, snapshot_sha256: m.snapshot_sha256,
        command: 'python unittest.py', gpu: '0', log_ref: await fileRef(task, log),
        invocations: 1, launch_succeeded: true, numerical_mismatches: 1,
      });
    }
    m.kernels.push(row);
  }
  const kernel = { source_delivery_required: true, selected_device_kernels: ['score', 'topk'] };
  const result = { task_dir: task, target_callable: 'pkg:op' };
  const pack = async () => ({ ...result, source_delivery_ref: await write('run/source.json', m) });
  return { root, task, kernel, result, m, write, pack };
}
test('source delivery covers all selected implementations and remains opt-in for legacy calls', async t => {
  const f = await fixture(t);
  assert.equal((await verifySourceDelivery(f.kernel, await f.pack())).ok, true);
  assert.equal((await verifySourceDelivery({}, f.result)).ok, true);
  assert.equal((await verifySourceDelivery(f.kernel, f.result)).ok, false);
  f.m.kernels.pop();
  f.kernel.source_device_kernels = ['score'];
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /source record for topk/);
});
test('an entry change cannot inherit an older UT verdict', async t => {
  const f = await fixture(t);
  assert.equal((await verifySourceDelivery(f.kernel, { ...await f.pack(), target_callable: 'pkg:deeper' })).ok, false);
  const ut = JSON.parse(await fs.readFile(path.join(f.task, 'run/ut.json')));
  ut.target_callable = 'pkg:previous';
  f.m.ut_result_ref = await f.write('run/ut.json', ut);
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /current entry/);
});
test('changing or adding source, cases or UT after a pass invalidates the recorded snapshot', async t => {
  const f = await fixture(t);
  for (const name of ['kernel_src/topk.cu', 'cases.py', 'unittest.py']) {
    const file = path.join(f.task, name), before = await fs.readFile(file);
    await fs.appendFile(file, '# changed\n');
    assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /changed after/);
    await fs.writeFile(file, before);
  }
  await fs.writeFile(path.join(f.task, 'kernel_src/new.h'), '// new dependency');
  assert.equal((await verifySourceDelivery(f.kernel, await f.pack())).ok, false);
});
test('compile-only failure is not a numerical negative control', async t => {
  const f = await fixture(t), row = f.m.kernels[0];
  const evidence = JSON.parse(await fs.readFile(path.join(f.task, row.negative_control_ref.path)));
  evidence.launch_succeeded = false; evidence.numerical_mismatches = 0;
  row.negative_control_ref = await f.write(row.negative_control_ref.path, evidence);
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /did not numerically reject/);
});
test('tampered logs and source symlinks outside the task cannot certify execution', async t => {
  const f = await fixture(t);
  await fs.appendFile(path.join(f.task, 'run/score_candidate_execution.log'), 'changed');
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /changed.*log/);
  await fs.symlink(import.meta.filename, path.join(f.task, 'kernel_src/external.py'));
  await assert.rejects(sourceSnapshot(f.task), /outside task/);
});
