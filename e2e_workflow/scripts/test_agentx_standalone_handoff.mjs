// CPU fixtures test evidence transport; they do not claim real GPU execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { sourceSnapshot, fileRef } from './kernel_source_delivery.mjs';

const cli = fileURLToPath(new URL('./kernel_extraction_delivery.mjs', import.meta.url));

async function fixture(t) {
  const evalDir = await fs.mkdtemp(path.join(os.tmpdir(), 'standalone handoff fixture '));
  t.after(() => fs.rm(evalDir, {recursive: true, force: true}));
  const task = path.join(evalDir, 'tasks', 'leaf');
  await fs.mkdir(task, {recursive: true});
  await fs.mkdir(path.join(task, 'kernel_src'));
  await fs.mkdir(path.join(task, 'run'));
  const write = async (name, value) => {
    await fs.writeFile(path.join(task, name), JSON.stringify(value));
    return fileRef(task, name);
  };
  await write('meta.json', {target_callable: 'pkg:leaf'});
  for (const name of ['cases.py', 'unittest.py', 'kernel_src/leaf.cu'])
    await fs.writeFile(path.join(task, name), '// CPU-only source fixture\n');
  const snapshot = await sourceSnapshot(task);
  const kernel = {entity_kind: 'gpu_kernel', device_kernel: 'leaf_kernel',
    selected_device_kernels: ['leaf_kernel'], source_delivery_required: true,
    seam_candidates: [{target_callable: 'pkg:leaf', role: 'inner_launcher', depth: 1,
      device_kernels: ['leaf_kernel']}]};
  const selection = await write('run/selection.json', {
    contract: 'kernel_selection', ok: true, target_callable: 'pkg:leaf',
    device_kernel: 'leaf_kernel', total_calls_observed: 1, target_marker_calls: 1,
    matched_kernel_calls: 1, deepest_verified: true, candidate_targets_tested: ['pkg:leaf'],
    live_candidate_targets: ['pkg:leaf'], deeper_live_candidates: [], failed: [],
  });
  const ut = await write('run/ut.json', {ok: true, exit_code: 0,
    target_callable: 'pkg:leaf', snapshot_sha256: snapshot.snapshot_sha256});
  const leaf = {device_kernel: 'leaf_kernel', source_files: ['kernel_src/leaf.cu']};
  for (const [field,kind] of [['execution_ref','candidate_execution'],
                             ['negative_control_ref','numerical_rejection']]) {
    await fs.writeFile(path.join(task, `run/${kind}.log`), 'CPU TEST FIXTURE, not GPU evidence\n');
    leaf[field] = await write(`run/${kind}.json`, {kind, ok: true,
      device_kernel: 'leaf_kernel', target_callable: 'pkg:leaf',
      snapshot_sha256: snapshot.snapshot_sha256,
      command: 'fixture-only', gpu: 'synthetic', invocations: 1,
      launch_succeeded: true, numerical_mismatches: 1,
      log_ref: await fileRef(task, `run/${kind}.log`)});
  }
  const source = await write('run/source.json', {contract:'kernel_source_delivery/v1',
    target_callable:'pkg:leaf', ...snapshot, ut_result_ref:ut, kernels:[leaf]});
  const result = {editable:true, task_dir:task, target_callable:'pkg:leaf',
    unittest_smoke:'pass', baseline_frozen:true,
    shape_coverage: { ledger: 'run/coverage.json', target_pct: 90, phases: {
      DECODE: { basis: 'mass', coverage_pct: 93.1, meets_target: true } } },
    selection_validation_ref:selection, source_delivery_ref:source};
  await write('run/requested_kernel.json',kernel);
  await write('run/draft.json',result);
  const args=[cli,'--kernel',path.join(task,'run/requested_kernel.json'),'--result',path.join(task,'run/draft.json'),
    '--phase','extract','--out',path.join(task,'run/RETURN.json')];
  const run=()=>spawnSync(process.execPath,args,{cwd:os.tmpdir(),encoding:'utf8'});
  return {evalDir,task,kernel,result,write,run};
}


import { packStandaloneTasks, readPreparedKernels, reusePreparedUT } from './agentx_kernel_preparation.mjs';
const adapter = fileURLToPath(new URL('./agentx_kernel_preparation.mjs', import.meta.url));

async function delivered(t) {
  const f = await fixture(t);
  // Execute the actual shared preflight used by the skill before exercising the workflow adapter.
  const checked = f.run();
  assert.equal(checked.status, 0, checked.stdout + checked.stderr);
  const write = (file, data) => fs.writeFile(path.join(f.evalDir, file), JSON.stringify(data));
  await write('workload.json', { kind: 'agentx_trace_replay' });
  await write('ranking.json', { ranking: [{name: 'leaf_kernel', selection_pct_gpu_time: 7.5}] });
  const manifest = {workload_json: 'workload.json', ranking_json: 'ranking.json',
    tasks: [{task_dir: 'tasks/leaf', phase: 'extract'}],
    failed_tasks: [{selected_device_kernels: ['failed_kernel'], reason: 'recorded GPU failure'}],
    deferred_tasks: [{selected_device_kernels: ['later_kernel'], reason: 'budget exhausted'}],
    coverage_summary: {covered_pct: 71, uncovered_pct: 29}, notes: 'CPU transport fixture only'};
  await write('manifest.json', manifest);
  return {...f, manifest, writeEval: write,
    pack: () => packStandaloneTasks(f.evalDir, 'manifest.json', 'delivery.json')};
}

test('skill preflight -> workflow packaging -> optimization queue preserves evidence and dispositions', async t => {
  const f = await delivered(t);
  const before = await fs.readFile(path.join(f.task, 'run/RETURN.json'), 'utf8');
  const proc = spawnSync(process.execPath, [adapter, '--eval-dir', f.evalDir,
    '--manifest', 'manifest.json', '--out', 'delivery.json'], {encoding: 'utf8', cwd: os.tmpdir()});
  assert.equal(proc.status, 0, proc.stdout + proc.stderr);
  const {delivery_path} = JSON.parse(proc.stdout);
  const prepared = await readPreparedKernels(f.evalDir, delivery_path);
  assert.equal(prepared.kernel_candidates.length, 1);
  assert.equal(prepared.kernel_candidates[0].device_kernel, 'leaf_kernel');
  assert.equal(prepared.kernel_candidates[0].pct_gpu_time, 7.5);
  assert.deepEqual(prepared.failed_tasks, f.manifest.failed_tasks);
  assert.deepEqual(prepared.deferred_tasks, f.manifest.deferred_tasks);
  assert.deepEqual(prepared.coverage_summary, f.manifest.coverage_summary);
  const reused = await reusePreparedUT(prepared.kernel_candidates[0]);
  assert.equal(reused.unittest_smoke, 'pass');
  assert.equal(reused.task_dir, f.task);
  assert.equal(await fs.readFile(path.join(f.task, 'run/RETURN.json'), 'utf8'), before);
});

test('changed source invalidates handoff and downstream reuse without overwriting accepted delivery', async t => {
  const f = await delivered(t);
  await f.pack();
  const before = await fs.readFile(path.join(f.evalDir, 'delivery.json'), 'utf8');
  const prepared = await readPreparedKernels(f.evalDir, 'delivery.json');
  await fs.appendFile(path.join(f.task, 'kernel_src/leaf.cu'), '// changed after validation\n');
  await assert.rejects(f.pack(), /invalid evidence/);
  assert.equal((await reusePreparedUT(prepared.kernel_candidates[0])).smoke, 'fail');
  assert.equal(await fs.readFile(path.join(f.evalDir, 'delivery.json'), 'utf8'), before);
});

test('duplicate completed tasks and missing attributed cost cannot silently enter queues', async t => {
  const f = await delivered(t);
  await f.writeEval('manifest.json', {...f.manifest, tasks: [...f.manifest.tasks, ...f.manifest.tasks]});
  await assert.rejects(f.pack(), /duplicate UT task/);
  await f.writeEval('manifest.json', f.manifest);
  await f.writeEval('ranking.json', {ranking: []});
  await assert.rejects(f.pack(), /no attributed cost/);
});

test('all-failed or deferred extraction remains explicit and produces no optimization tasks', async t => {
  const f = await delivered(t);
  await f.writeEval('manifest.json', {...f.manifest, tasks: []});
  await f.pack();
  const prepared = await readPreparedKernels(f.evalDir, 'delivery.json');
  assert.equal(prepared.kernel_candidates.length + prepared.head_candidates.length, 0);
  assert.deepEqual(prepared.failed_tasks, f.manifest.failed_tasks);
  assert.deepEqual(prepared.deferred_tasks, f.manifest.deferred_tasks);
});

test('packaging validates task ownership, output containment and source evidence preservation', async t => {
  const f = await delivered(t);
  await assert.rejects(packStandaloneTasks(f.evalDir, 'manifest.json', 'ranking.json'), /overwrite source/);
  await assert.rejects(packStandaloneTasks(f.evalDir, 'manifest.json', '../escape.json'), /inside EVAL_DIR/);
  await fs.symlink(path.join(f.evalDir, 'ranking.json'), path.join(f.evalDir, 'link.json'));
  await assert.rejects(packStandaloneTasks(f.evalDir, 'manifest.json', 'link.json'), /symlink/);
  await f.write('run/RETURN.json', {...f.result, task_dir: f.evalDir});
  await assert.rejects(f.pack(), /differs from its declared task/);
});

// Execute each production call site with real saved artifacts, including the
// serial, fast and deep head paths. No agent or GPU execution is needed here.
test('all optimization dispatch sites reuse UTs and fail locally without authoring replacements', async t => {
  const f = await delivered(t);
  await f.pack();
  const prepared = (await readPreparedKernels(f.evalDir, 'delivery.json')).kernel_candidates[0];
  const workflowDir = fileURLToPath(new URL('..', import.meta.url));
  const src = await fs.readFile(path.join(workflowDir, 'e2e_workflow.js'), 'utf8');
  const helper = src.slice(src.indexOf('async function extractWithBaseline('), src.indexOf('\n// abDone =='));
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const sites = [...src.matchAll(/const ext = await extractWithBaseline\(/g)];
  assert.equal(sites.length, 4, 'three head modes and the milestone path');
  const dispatchers = sites.map(site => {
    const start = site.index, end = src.indexOf('});', src.indexOf('schema:', start)) + 3;
    const consume = src.lastIndexOf('if (KERNEL_EXTRACT_ON) headQueue =', start);
    const isHead = src.slice(start, end).includes("'extract_op'");
    const body = (isHead ? src.slice(consume, src.indexOf('\n', consume)) + '\n' : '') + src.slice(start, end);
    return async kernel => {
      const queued = {...kernel, prepared_ut: kernel.prepared_ut};
      const pending = {...prepared, short_name: 'pending', prepared_ut: {...prepared.prepared_ut}};
      const env = {KERNEL_EXTRACT_ON: true, WORKFLOW_DIR: workflowDir, EVAL_DIR: f.evalDir,
        MODEL_PATH: '/model', GPU_LIST: ['0'], gpu: '0', WORKLOAD: {}, profile: {},
        curOverlay: '', curFlags: '', curEnv: '', CONC: 32, EXTRACT_SCHEMA: {}, EXTRACT_OP_SCHEMA: {},
        h: {...queued}, c: {...queued}, headQueue: [queued, pending], gemmSynthFor: () => false,
        safeAgent: () => { throw new Error('downstream authoring is forbidden'); }};
      return new AsyncFunction('env', `let {${Object.keys(env).join(',')}} = env;
        ${helper}\n${body}\nreturn {ext, headQueue};`)(env);
    };
  });
  const before = await fs.readFile(path.join(f.task, 'run/RETURN.json'), 'utf8');
  for (const [i, dispatch] of dispatchers.entries()) {
    const {ext, headQueue} = await dispatch(prepared);
    assert.equal(ext.task_dir, f.task);
    assert.equal(ext.smoke, 'pass');
    if (i < 3) assert.deepEqual(headQueue.map(c => c.short_name), ['pending'],
      'dispatched heads are consumed while undispatched heads remain resumable');
    const missing = await dispatch({...prepared, prepared_ut: undefined});
    assert.equal(missing.ext.smoke, 'fail');
    assert.match(missing.ext.smoke_detail, /missing prepared UT/);
  }
  assert.equal(await fs.readFile(path.join(f.task, 'run/RETURN.json'), 'utf8'), before);
  await fs.appendFile(path.join(f.task, 'cases.py'), '# invalidates the frozen UT\n');
  for (const dispatch of dispatchers) assert.equal((await dispatch(prepared)).ext.smoke, 'fail');
});

test('whole-op metadata routes attention aliases to head and preserves the consumer op kind', async t => {
  const f = await delivered(t);
  await f.write('run/requested_kernel.json', {...f.kernel, op_kind: 'attention'});
  await f.write('run/RETURN.json', {...f.result, smoke: 'pass'});
  await f.writeEval('manifest.json', {...f.manifest, tasks: [{task_dir: 'tasks/leaf', phase: 'extract_op'}]});
  await f.pack();
  const queues = await readPreparedKernels(f.evalDir, 'delivery.json');
  assert.equal(queues.kernel_candidates.length, 0);
  assert.equal(queues.head_candidates[0].op_kind, 'attn');
  assert.equal((await reusePreparedUT(queues.head_candidates[0])).op_kind, 'attn');
  await f.write('run/RETURN.json', {...f.result, smoke: 'pass', op_kind: 'moe'});
  await assert.rejects(f.pack(), /op_kind differs/);
});

test('completed multi-rank UTs retain evidence but do not enter single-GPU optimization lanes', async t => {
  const f = await delivered(t);
  for (const declaration of [{required_gpu_count: 8}, {op_kind: 'collective'}]) {
    await f.write('run/requested_kernel.json', {...f.kernel, ...declaration});
    await f.pack();
    const queues = await readPreparedKernels(f.evalDir, 'delivery.json');
    assert.equal(queues.head_candidates.length + queues.kernel_candidates.length, 0);
    const task = queues.deferred_tasks.at(-1);
    assert.equal(task.task_dir, f.task);
    assert.match(task.reason, /UT validated.*multi-GPU lane/);
    assert.equal((await reusePreparedUT(task)).unittest_smoke, 'pass');
  }
});
