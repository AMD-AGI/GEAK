import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { sourceSnapshot, fileRef, verifySourceDelivery } from './kernel_source_delivery.mjs';

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

// Recorded evidence fixtures exercise admission/reuse; they are not GPU tests.
import { readPreparedKernels, reusePreparedUT } from './agentx_kernel_preparation.mjs';
import { checkPreparedUT } from './kernel_extraction_delivery.mjs';

async function preparationFixture(t) {
  const f = await fixture(t, true);
  const seams = [{ target_callable: 'pkg:op', depth: 1, role: 'op_seam', device_kernels: ['score'] }];
  f.kernel = { device_kernel: 'score', short_name: 'score', entity_kind: 'gpu_kernel',
    selected_device_kernels: ['score'], seam_candidates: seams, source_delivery_required: true };
  const result = { ...await f.pack(), unittest_smoke: 'pass', baseline_frozen: true,
    source: { path: 'kernel_src/score.py' }, notes: ['structured notes'], regime_summary: { DECODE: 3 },
    selection_validation: { contract: 'kernel_selection', ok: true, target_callable: 'pkg:op',
      device_kernel: 'score', total_calls_observed: 1, target_marker_calls: 1, matched_kernel_calls: 1,
      deepest_verified: true, live_candidate_targets: ['pkg:op'], deeper_live_candidates: [],
      candidate_targets_tested: ['pkg:op'] } };
  await f.write('run/RETURN.json', result);
  await f.write('run/workload.json', {});
  await f.write('run/ranking.json', { ranking: [{ device_kernel: 'score', selection_pct_gpu_time: 7.2 }] });
  const delivery = { workload_json: 'task/run/workload.json', ranking_json: 'task/run/ranking.json',
    tasks: [{ kernel: f.kernel, phase: 'extract', result_path: 'task/run/RETURN.json' }],
    failed_tasks: [{ short_name: 'previous', reason: 'missing operands' }],
    deferred_tasks: [{ short_name: 'later', reason: 'budget' }], notes: { coverage: 'limited' } };
  const save = async () => { await f.write('run/delivery.json', delivery); return readPreparedKernels(f.root, 'task/run/delivery.json'); };
  return { ...f, result, delivery, save };
}

test('complete-stage handoff accepts structured descriptions and reuses the validated task', async t => {
  const f = await preparationFixture(t), ready = await f.save();
  assert.equal(ready.kernel_candidates.length, 1, JSON.stringify(ready.failed_tasks));
  assert.equal(ready.kernel_candidates[0].pct_gpu_time, 7.2);
  assert.equal(ready.failed_tasks.length, 1);
  assert.equal(ready.deferred_tasks.length, 1);
  const reused = await reusePreparedUT(ready.kernel_candidates[0]);
  assert.equal(reused.unittest_smoke, 'pass');
  assert.deepEqual(reused.notes, ['structured notes']);
  const source = await fs.readFile(new URL('../e2e_workflow.js', import.meta.url), 'utf8');
  const block = source.slice(source.indexOf('async function extractWithBaseline('), source.indexOf('// abDone =='));
  const dispatch = new Function('WORKFLOW_DIR', `
    const KERNEL_EXTRACT_ON = true;
    const safeAgent = () => { throw new Error('must not invoke a second extractor'); };
    ${block}\nreturn extractWithBaseline;
  `)(path.resolve(import.meta.dirname, '..'));
  assert.equal((await dispatch('kernel_extractor', 'extract', '', { KERNEL: ready.kernel_candidates[0] })).smoke, 'pass');
});

test('failed UT remains visible and never enters an optimization queue', async t => {
  const f = await preparationFixture(t);
  f.result.unittest_smoke = 'fail';
  await f.write('run/RETURN.json', f.result);
  const ready = await f.save();
  assert.equal(ready.kernel_candidates.length, 0);
  assert.equal(ready.failed_tasks.length, 2);
  assert.equal(ready.failed_tasks[1].checks.ut_status.ok, false);
});

test('prepared reuse detects changed result files and changed candidate source', async t => {
  const f = await preparationFixture(t), ready = await f.save(), kernel = ready.kernel_candidates[0];
  const file = path.join(f.task, 'run/RETURN.json'), before = await fs.readFile(file);
  await fs.appendFile(file, '\n');
  assert.match((await reusePreparedUT(kernel)).smoke_detail, /result changed/);
  await fs.writeFile(file, before);
  await fs.appendFile(path.join(f.task, 'kernel_src/score.py'), '# edited after UT\n');
  assert.equal((await reusePreparedUT(kernel)).smoke, 'fail');
});

test('duplicate UT ownership does not schedule the same selected implementation twice', async t => {
  const f = await preparationFixture(t);
  f.delivery.tasks.push({ ...f.delivery.tasks[0], kernel: { ...f.kernel, short_name: 'duplicate' } });
  const ready = await f.save();
  assert.equal(ready.kernel_candidates.length, 1);
  assert.match(ready.failed_tasks[1].reason, /already owned/);
});

test('delivery requires UT files within eval and complete declared selected boundaries', async t => {
  const f = await preparationFixture(t);
  assert.equal((await checkPreparedUT({ ...f.kernel, selected_device_kernels: ['score', 'topk'] }, f.result, 'extract')).ok, false);
  f.delivery.tasks[0].result_path = '/etc/hosts';
  const ready = await f.save();
  assert.equal(ready.kernel_candidates.length, 0);
  assert.match(ready.failed_tasks[1].reason, /inside EVAL_DIR/);
});

test('full extract-only workflow resumes from baseline and returns UT state without optimization', async t => {
  const f = await preparationFixture(t);
  await f.save();
  const { createRuntime } = await import('../../interface/runtime/engine/run_workflow.mjs');
  const calls = [], logs = [];
  const runtime = createRuntime({ backend: { name: 'fixture', runAgent: async input => {
    calls.push(input.prompt);
    if (!input.prompt.includes('PHASE=prepare')) {
      assert.match(input.prompt, /workflow_return.json|PHASE=persist_llm_stats/);
      return { text: JSON.stringify({ written: true, path: f.root, ok: true }) };
    }
    return { text: JSON.stringify({ delivery_path: path.join(f.task, 'run/delivery.json') }) };
  } }, log: line => logs.push(line) });
  const workload = { kind: 'agentx_trace_replay', concurrency: 10 };
  const result = await runtime.runScript(path.resolve(import.meta.dirname, '../e2e_workflow.js'), {
    model_path: '/model', workflow_dir: path.resolve(import.meta.dirname, '..'),
    expected_gfx: 'gfx950', expected_target: 'mi355x', expected_physical_cu_count: 256,
    phases: 'extract', workload_spec: workload, config_tune: 'false', tuning_skillset: 'false',
    analysis_skill: 'none', use_expert_skills: 'false', use_learned_kb: 'false', warm_start: 'false',
    gpu_ids: '0', final_reserve_s: 0,
    state: { eval_dir: f.root, model_name: 'fixture', baseline_throughput_tok_s: 1,
      throughput: 1, flags: '--tp 1', env: '', overlay: '', workload_spec: workload },
  }, 0);
  assert.equal(calls.filter(p => p.includes('PHASE=prepare')).length, 1, logs.join('\n'));
  assert.deepEqual(result.phases_run, ['extract']);
  assert.equal(result.state.kernelQueue.length, 1);
  assert.equal(result.kernel_extraction.failed_tasks.length, 1);
  assert.equal(result.kernel_extraction.deferred_tasks.length, 1);
  assert.equal(result.validation_status, 'phase_partial');
});
