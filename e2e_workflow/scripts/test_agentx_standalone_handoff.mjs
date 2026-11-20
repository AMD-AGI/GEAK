// CPU fixtures test evidence transport; they do not claim real GPU execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { sourceSnapshot, fileRef } from './kernel_source_delivery.mjs';

const cli = fileURLToPath(new URL('./kernel_extraction_delivery.mjs', import.meta.url));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const GENERATED = '# GEAK-GENERATED unittest.py (generate_ut.py v1): do not edit; regenerate with generate_ut.py\n';
const receipt = world => ({version: 'geak_timing_receipt/v2', all_primed: true, timer_unprimed: false,
  cache_mode: 'read-evict', world_size: world, timer: 'cuda_event_graph',
  alignment: world > 1 ? {per_sample: 'flush, synchronize, gloo barrier, release', merge: 'max over ranks'} : null,
  cases: {decode_m1: Object.fromEntries(['baseline', 'current'].map(leg => [leg, {ms: 0.05,
    timer: 'cuda_event_graph', primed: true, host_ms: 0.01,
    ...(world > 1 ? {rank_ms: Array(world).fill(0.05)} : {})}]))}});
// Shape of /ci/kernel_workflow/wf_neg_mimo_tp4/task/run/ut_result.json `timing`: an older
// hand-written TP4 UT reporting per-rank microseconds and its alignment.
const wfNegTiming = {cases: [{sig: 'swa_moe_r1_ctx128k', weight: 1, weight_source: 'regime', layers_per_step: 8,
  baseline: {per_rank_us: [54.764, 54.764, 54.764, 54.764], step_us: 54.764, max_rank_skew_us: 0.002, windows: 10},
  candidate: {per_rank_us: [54.881, 54.88, 54.881, 54.881], step_us: 54.881, max_rank_skew_us: 0.004, windows: 10},
  speedup: 0.9979}], weighted_speedup: 0.9979,
  alignment: {host: 'torch.cuda.synchronize() then a gloo dist.barrier() over the 4 ranks before every timed window',
    counts: 'identical warmup / aligning / timed replay counts on every rank',
    aggregate: 'per window step time = max over ranks; case time = median over windows and rounds'}};

// `world` > 0 builds a generate_ut.py task of that world size; `gpus` declares required_gpu_count
// in the request and result; `ut` and `meta` add fields to the UT result and meta.json.
async function fixture(t, {world = 0, gpus = world > 1 ? world : undefined, ut: utFields = {}, meta = {}} = {}) {
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
  await write('meta.json', {target_callable: 'pkg:leaf', ...meta,
    ...(world ? {capabilities: {multi_rank: {world_size: world}}} : {})});
  for (const name of ['cases.py', 'unittest.py', 'kernel_src/leaf.cu'])
    await fs.writeFile(path.join(task, name), '// CPU-only source fixture\n');
  let contract;
  if (world) {
    await fs.writeFile(path.join(task, 'unittest.py'), GENERATED);
    for (const name of ['ut_runner.py', 'harness_lib.py', 'leg_runner.py', 'overlay_setup.py'])
      await fs.writeFile(path.join(task, name), '# CPU-only runtime fixture\n');
    const files = [];
    for (const name of ['cases.py', 'harness_lib.py', 'leg_runner.py', 'meta.json', 'overlay_setup.py',
      'unittest.py', 'ut_runner.py']) files.push(await fileRef(task, name));
    contract = {contract: 'geak_ut_contract/v1', files, contract_sha256: sha(JSON.stringify(files))};
    await write('run/ut_contract.json', contract);
  }
  const snapshot = await sourceSnapshot(task);
  const declared = gpus === undefined ? {} : {required_gpu_count: gpus};
  const kernel = {entity_kind: 'gpu_kernel', device_kernel: 'leaf_kernel',
    selected_device_kernels: ['leaf_kernel'], source_delivery_required: true,
    seam_candidates: [{target_callable: 'pkg:leaf', role: 'inner_launcher', depth: 1,
      device_kernels: ['leaf_kernel']}], ...declared};
  const selection = await write('run/selection.json', {
    contract: 'kernel_selection', ok: true, target_callable: 'pkg:leaf',
    device_kernel: 'leaf_kernel', total_calls_observed: 1, target_marker_calls: 1,
    matched_kernel_calls: 1, deepest_verified: true, candidate_targets_tested: ['pkg:leaf'],
    live_candidate_targets: ['pkg:leaf'], deeper_live_candidates: [], failed: [],
  });
  const ut = await write('run/ut.json', {ok: true, exit_code: 0,
    target_callable: 'pkg:leaf', snapshot_sha256: snapshot.snapshot_sha256,
    ...(world ? {contract_sha256: contract.contract_sha256, world_size: world, receipt: receipt(world)} : {}),
    ...utFields});
  const leaf = {device_kernel: 'leaf_kernel', source_files: ['kernel_src/leaf.cu']};
  for (const [field,kind] of [['execution_ref','candidate_execution'],
                             ['negative_control_ref','numerical_rejection']]) {
    await fs.writeFile(path.join(task, `run/${kind}.log`), 'CPU TEST FIXTURE, not GPU evidence\n');
    leaf[field] = await write(`run/${kind}.json`, {kind, ok: true,
      device_kernel: 'leaf_kernel', target_callable: 'pkg:leaf',
      snapshot_sha256: snapshot.snapshot_sha256,
      command: 'fixture-only', gpu: 'synthetic', invocations: 1,
      launch_succeeded: true, numerical_mismatches: 1,
      log_ref: await fileRef(task, `run/${kind}.log`),
      ...(world > 1 && kind === 'numerical_rejection' ? {ranks: Array.from({length: world},
        (_, rank) => ({rank, launch_succeeded: true, numerical_mismatches: 1}))} : {})});
  }
  if (world) await write('run/verify_ut.json', {contract: 'geak_verify_ut/v1', verdict: 'PASS',
    contract_sha256: contract.contract_sha256, snapshot_sha256: snapshot.snapshot_sha256,
    world_size: world, gaps: [], errors: []});
  const source = await write('run/source.json', {contract:'kernel_source_delivery/v1',
    target_callable:'pkg:leaf', ...snapshot, ut_result_ref:ut, kernels:[leaf]});
  const result = {editable:true, task_dir:task, target_callable:'pkg:leaf',
    unittest_smoke:'pass', baseline_frozen:true,
    selection_validation_ref:selection, source_delivery_ref:source, ...declared};
  await write('run/requested_kernel.json',kernel);
  await write('run/draft.json',result);
  const args=[cli,'--kernel',path.join(task,'run/requested_kernel.json'),'--result',path.join(task,'run/draft.json'),
    '--phase','extract','--out',path.join(task,'run/RETURN.json')];
  const run=()=>spawnSync(process.execPath,args,{cwd:os.tmpdir(),encoding:'utf8'});
  return {evalDir,task,kernel,result,write,run};
}


import { packStandaloneTasks, readPreparedKernels } from './agentx_kernel_preparation.mjs';
const adapter = fileURLToPath(new URL('./agentx_kernel_preparation.mjs', import.meta.url));

async function delivered(t, options) {
  const f = await fixture(t, options);
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
  // The validated RETURN.json rides along with the envelope -- the workflow sandbox has no
  // filesystem, so a result_path it cannot open would not be evidence it can dispatch on.
  const carried = prepared.kernel_candidates[0].prepared_ut;
  assert.equal(carried.result.unittest_smoke, 'pass');
  assert.equal(carried.result.task_dir, f.task);
  assert.equal(await fs.readFile(path.join(f.task, 'run/RETURN.json'), 'utf8'), before);
});

test('changed source invalidates handoff and downstream reuse without overwriting accepted delivery', async t => {
  const f = await delivered(t);
  await f.pack();
  const before = await fs.readFile(path.join(f.evalDir, 'delivery.json'), 'utf8');
  const prepared = await readPreparedKernels(f.evalDir, 'delivery.json');
  await fs.appendFile(path.join(f.task, 'kernel_src/leaf.cu'), '// changed after validation\n');
  await assert.rejects(f.pack(), /invalid evidence/);
  // Re-reading is the ONLY gate now: packing is where changed source is caught, and it refuses to
  // emit a queue at all rather than emitting one the dispatcher would have to re-verify.
  const stale = await readPreparedKernels(f.evalDir, 'delivery.json');
  assert.deepEqual([...stale.head_candidates, ...stale.kernel_candidates], []);
  assert.equal(prepared.kernel_candidates.length, 1, 'the pre-edit envelope was a real queue');
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
  // preparedUtDefect + reusePreparedUT sit immediately above extractWithBaseline and are what the
  // AgentX branch returns, so the slice starts there: the workflow sandbox has no filesystem, so
  // reuse is pure JS over the envelope rather than a module the dispatcher could import.
  const helper = src.slice(src.indexOf('function preparedUtDefect('), src.indexOf('\n// abDone =='));
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
  // An edited UT is caught at PREPARATION, which is the single gate now: the dispatcher cannot
  // re-hash kernel_src because the workflow sandbox has no filesystem, so readPreparedKernels is
  // where a mutated task stops being a candidate at all.
  await fs.appendFile(path.join(f.task, 'cases.py'), '# invalidates the frozen UT\n');
  const stale = await readPreparedKernels(f.evalDir, 'delivery.json');
  assert.deepEqual([...stale.head_candidates, ...stale.kernel_candidates], [],
    'an edited UT cannot be re-prepared into an optimization queue');
  assert(stale.failed_tasks.some(t => t.short_name === prepared.short_name),
    'the edited task is reported as failed, not silently dropped');
});

// The envelope reaches the workflow through an AGENT, not through the file the packer wrote: the
// sandbox has no filesystem to open it with. So "the packer validated this" is a claim, and the
// intake re-asserts every invariant it can without reading bytes. Forge the reply and it must
// THROW -- which is what arms the repair_delivery retry. Dropping the bad candidate instead would
// shorten the queue silently and the run would end looking like there was nothing to optimize.
test('a forged or degraded envelope is refused at intake, not taken on trust', async t => {
  const f = await delivered(t);
  await f.pack();
  const good = await readPreparedKernels(f.evalDir, 'delivery.json');
  const src = await fs.readFile(new URL('../e2e_workflow.js', import.meta.url), 'utf8');
  const cut = (from, to) => src.slice(src.indexOf(from), src.indexOf(to, src.indexOf(from)));
  const reader = cut('function preparedUtDefect(', '\n// Hand an optimization lane')
    + cut('function readAgentXPreparation(', '\nasync function prepareAgentXKernels(');
  const read = new Function(`${reader}\nreturn readAgentXPreparation;`)();
  const reply = {delivery_path: '/eval/delivery.json', prepared: good};
  assert.equal(read(structuredClone(reply)).kernel_candidates.length, 1, 'the real envelope passes');
  const mutate = (fn) => { const r = structuredClone(reply); fn(r.prepared.kernel_candidates[0]); return r; };
  for (const [why, fn] of [
    ['no result at all', (c) => { delete c.prepared_ut.result; }],
    ['a failing UT copied forward', (c) => { c.prepared_ut.result.unittest_smoke = 'fail'; }],
    ['no frozen baseline', (c) => { delete c.prepared_ut.result.baseline_frozen; }],
    ['no task_dir', (c) => { delete c.prepared_ut.result.task_dir; }],
    ['an invented GPU share', (c) => { c.pct_gpu_time = 420; }],
    ['an op_kind that contradicts its own result', (c) => { c.op_kind = 'moe'; c.prepared_ut.result.op_kind = 'gemm'; }],
  ]) assert.throws(() => read(mutate(fn)), /failed re-validation/, why);
  assert.throws(() => read({delivery_path: '/eval/delivery.json'}), /no prepared envelope/);
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
  await f.write('run/RETURN.json', {...f.result, smoke: 'pass', op_kind: 'moe'});
  await assert.rejects(f.pack(), /op_kind differs/);
});

// main's deferred entry is {...candidate, task_dir, result_path, required_gpu_count, reason}; a
// multi-GPU candidate is the single-GPU candidate plus required_gpu_count and nothing else (A17).
// Both are compared with the single-GPU candidate `single` of the same task (fixture `s`).
function assertSingleGpuShape(entry, f, single, s, extra) {
  const keys = Object.keys(single);
  assert.deepEqual(Object.keys(entry), [...keys, ...extra.filter(k => !keys.includes(k))]);
  const drop = ['required_gpu_count', 'prepared_ut', 'task_dir', 'result_path', 'reason'];
  const rest = (o, dir) => JSON.parse(JSON.stringify(Object.fromEntries(Object.entries(o)
    .filter(([k]) => !drop.includes(k)))).split(dir).join('<EVAL>'));
  assert.deepEqual(rest(entry, f.evalDir), rest(single, s.evalDir));
  assert.deepEqual(Object.keys(entry.prepared_ut), Object.keys(single.prepared_ut));
}
const DEFERRED = /^UT validated; multi-GPU optimization needs a declared GPU count and per-rank time and alignment for every timed case: /;

test('completed multi-rank UTs without per-rank timing retain evidence and stay deferred', async t => {
  const missingAlignment = {...wfNegTiming, alignment: undefined};
  for (const [options, declaration, gpus, missing] of [
    [{}, {required_gpu_count: 8}, 8, /: no timing receipt; old format: no timing$/],
    [{}, {op_kind: 'collective'}, 2, /: no required_gpu_count in the request or the result \(the collective default of 2 GPUs is not a declaration\); no timing receipt; old format: no timing$/],
    [{ut: {timing: missingAlignment}}, {required_gpu_count: 4}, 4, /: no timing receipt; old format: timing has no alignment$/],
    [{ut: {timing: wfNegTiming}}, {required_gpu_count: 8}, 8, /: no timing receipt; old format: timing case swa_moe_r1_ctx128k: baseline.per_rank_us is not 8 finite per-rank times$/],
  ]) {
    const f = await delivered(t, options);
    await f.write('run/requested_kernel.json', {...f.kernel, ...declaration});
    await f.pack();
    const queues = await readPreparedKernels(f.evalDir, 'delivery.json');
    assert.equal(queues.head_candidates.length + queues.kernel_candidates.length, 0);
    assert.deepEqual(queues.failed_tasks, f.manifest.failed_tasks, 'a GPU-count problem never fails the task');
    const task = queues.deferred_tasks.at(-1);
    assert.equal(task.task_dir, f.task);
    assert.equal(task.result_path, await fs.realpath(path.join(f.task, 'run/RETURN.json')));
    assert.equal(task.required_gpu_count, gpus);
    assert.match(task.reason, DEFERRED);
    assert.match(task.reason, missing);
    assert.deepEqual(Object.keys(task).slice('required_gpu_count' in declaration ? -3 : -4),
      ['task_dir', 'result_path', ...('required_gpu_count' in declaration ? [] : ['required_gpu_count']), 'reason'],
      'main\'s deferred entry fields');
    assert.equal(task.prepared_ut.result.unittest_smoke, 'pass', 'deferred tasks keep their evidence');
  }
});

test('a multi-rank UT with per-rank timing and one declared GPU count enters its queue with required_gpu_count only', async t => {
  // A generated 4-rank task (timing receipt) and an older hand-written TP4 UT (wf_neg format), with
  // the count in the request and the result, or (`only`) in one of them; the single-GPU twin of
  // each case declares 1 the same way.
  for (const [options, twin, only] of [
    [{world: 4}, {world: 1, gpus: 1}, ''],
    [{world: 4}, {world: 1, gpus: 1}, 'result'],
    [{gpus: 4, ut: {timing: wfNegTiming}}, {gpus: 1, ut: {timing: wfNegTiming}}, ''],
    [{gpus: 4, ut: {timing: wfNegTiming}}, {gpus: 1, ut: {timing: wfNegTiming}}, 'request'],
  ]) {
    const declared = async (fx, n) => {
      const drop = ({required_gpu_count, ...rest}) => rest;
      if (only === 'result') fx.kernel = drop(fx.kernel);
      if (only === 'request') fx.result = drop(fx.result);
      await fx.write('run/requested_kernel.json', fx.kernel);
      await fx.write('run/RETURN.json', fx.result);
      assert.equal(fx.kernel.required_gpu_count ?? fx.result.required_gpu_count, n);
      return fx;
    };
    const f = await declared(await delivered(t, options), 4);
    const {prepared} = await f.pack();
    assert.deepEqual(prepared.failed_tasks, f.manifest.failed_tasks);
    assert.deepEqual(prepared.deferred_tasks, f.manifest.deferred_tasks, 'a timed, declared multi-GPU task is not deferred');
    assert.equal(prepared.kernel_candidates.length, 1);
    const candidate = prepared.kernel_candidates[0];
    assert.equal(candidate.required_gpu_count, 4);
    // Exactly the single-GPU candidate of the same task, plus required_gpu_count.
    const s = await declared(await delivered(t, twin), 1);
    const single = (await s.pack()).prepared.kernel_candidates[0];
    assertSingleGpuShape(candidate, f, single, s, ['required_gpu_count']);
    // Whole-op GEMM/attention/MoE tasks route to the head queue the same way.
    await f.write('run/requested_kernel.json', {...f.kernel, op_kind: 'moe'});
    await f.write('run/RETURN.json', {...f.result, smoke: 'pass'});
    await f.writeEval('manifest.json', {...f.manifest, tasks: [{task_dir: 'tasks/leaf', phase: 'extract_op'}]});
    const head = (await f.pack()).prepared;
    assert.deepEqual(head.head_candidates.map(c => [c.op_kind, c.required_gpu_count]), [['moe', 4]]);
    assert.equal(head.kernel_candidates.length, 0);
  }
});

test('single-GPU candidates keep their shape', async t => {
  const f = await delivered(t);
  const candidate = (await f.pack()).prepared.kernel_candidates[0];
  assert(!('required_gpu_count' in candidate));
  await f.write('run/requested_kernel.json', {...f.kernel, required_gpu_count: 1});
  const declared = (await f.pack()).prepared.kernel_candidates[0];
  assert.deepEqual(Object.keys(declared).filter(k => !(k in candidate)), ['required_gpu_count']);
  assert.deepEqual(Object.keys(candidate).filter(k => !(k in declared)), []);
});

// ---------------------------------------------------------------------------------------------
// The 31 packing scenarios of review-08 (table 2.1), built as its probe builds them: a hand-written
// ("old") task, or with `world` a generate_ut.py task of that world size. Single-GPU tasks (by main's
// count) and invalid counts keep main's result; a multi-GPU task is a candidate with
// required_gpu_count only when it is timed per rank and its count is declared, else main's
// deferred entry; a generated task breaking one of its own validity rules fails with main's keys.
import { loadContract, checkDelivery } from './kernel_extraction_delivery.mjs';
const legacyTiming = n => ({cases: [{sig: 's', baseline: {per_rank_us: Array(n).fill(54.7)}}], alignment: {host: 'barrier'}});
async function scenario(t, {kernel: kx = {}, result: rx = {}, ut: ux = {}, meta: mx = {}, world = 0, phase = 'extract',
  verify = {}, rejectionExtra = {}} = {}) {
  const evalDir = await fs.mkdtemp(path.join(os.tmpdir(), 'packing scenario '));
  t.after(() => fs.rm(evalDir, {recursive: true, force: true}));
  const task = path.join(evalDir, 'tasks', 'leaf');
  await fs.mkdir(path.join(task, 'kernel_src'), {recursive: true});
  await fs.mkdir(path.join(task, 'run'));
  const write = async (name, value) => { await fs.writeFile(path.join(task, name), JSON.stringify(value)); return fileRef(task, name); };
  await write('meta.json', {target_callable: 'pkg:leaf', ...mx, ...(world ? {capabilities: {multi_rank: {world_size: world}}} : {})});
  for (const n of ['cases.py', 'unittest.py', 'kernel_src/leaf.cu']) await fs.writeFile(path.join(task, n), '// fixture\n');
  let contract;
  if (world) {
    await fs.writeFile(path.join(task, 'unittest.py'), GENERATED);
    for (const n of ['ut_runner.py', 'harness_lib.py', 'leg_runner.py', 'overlay_setup.py']) await fs.writeFile(path.join(task, n), '# rt\n');
    const files = [];
    for (const n of ['cases.py', 'harness_lib.py', 'leg_runner.py', 'meta.json', 'overlay_setup.py', 'unittest.py', 'ut_runner.py'])
      files.push(await fileRef(task, n));
    contract = {contract: 'geak_ut_contract/v1', files, contract_sha256: sha(JSON.stringify(files))};
    await write('run/ut_contract.json', contract);
  }
  const snapshot = await sourceSnapshot(task);
  const kernel = {entity_kind: 'gpu_kernel', device_kernel: 'leaf_kernel', selected_device_kernels: ['leaf_kernel'],
    source_delivery_required: true, seam_candidates: [{target_callable: 'pkg:leaf', role: 'inner_launcher', depth: 1,
      device_kernels: ['leaf_kernel']}], ...kx};
  const selection = await write('run/selection.json', {contract: 'kernel_selection', ok: true, target_callable: 'pkg:leaf',
    device_kernel: 'leaf_kernel', total_calls_observed: 1, target_marker_calls: 1, matched_kernel_calls: 1, deepest_verified: true,
    candidate_targets_tested: ['pkg:leaf'], live_candidate_targets: ['pkg:leaf'], deeper_live_candidates: [], failed: []});
  const ut = await write('run/ut.json', {ok: true, exit_code: 0, target_callable: 'pkg:leaf', snapshot_sha256: snapshot.snapshot_sha256,
    ...(world ? {contract_sha256: contract.contract_sha256, receipt: receipt(world)} : {}), ...ux});
  const leaf = {device_kernel: 'leaf_kernel', source_files: ['kernel_src/leaf.cu']};
  for (const [field, kind] of [['execution_ref', 'candidate_execution'], ['negative_control_ref', 'numerical_rejection']]) {
    await fs.writeFile(path.join(task, `run/${kind}.log`), 'log\n');
    leaf[field] = await write(`run/${kind}.json`, {kind, ok: true, device_kernel: 'leaf_kernel', target_callable: 'pkg:leaf',
      snapshot_sha256: snapshot.snapshot_sha256, command: 'x', gpu: 'g', invocations: 1, launch_succeeded: true,
      numerical_mismatches: 1, log_ref: await fileRef(task, `run/${kind}.log`),
      ...(world > 1 && kind === 'numerical_rejection' ? {ranks: Array.from({length: world},
        (_, rank) => ({rank, launch_succeeded: true, numerical_mismatches: 1}))} : {}),
      ...(kind === 'numerical_rejection' ? rejectionExtra : {})});
  }
  if (world) await write('run/verify_ut.json', {verdict: 'PASS', contract_sha256: contract.contract_sha256,
    snapshot_sha256: snapshot.snapshot_sha256, gaps: [], ...verify});
  const source = await write('run/source.json', {contract: 'kernel_source_delivery/v1', target_callable: 'pkg:leaf', ...snapshot,
    ut_result_ref: ut, kernels: [leaf]});
  const result = {editable: true, task_dir: task, target_callable: 'pkg:leaf', unittest_smoke: 'pass', smoke: 'pass',
    baseline_frozen: true, selection_validation_ref: selection, source_delivery_ref: source, ...rx};
  await write('run/requested_kernel.json', kernel);
  await write('run/RETURN.json', result);
  const writeEval = (file, data) => fs.writeFile(path.join(evalDir, file), JSON.stringify(data));
  await writeEval('workload.json', {kind: 'w'});
  await writeEval('ranking.json', {ranking: [{name: 'leaf_kernel', selection_pct_gpu_time: 7.5}]});
  await writeEval('manifest.json', {workload_json: 'workload.json', ranking_json: 'ranking.json', tasks: [{task_dir: 'tasks/leaf', phase}]});
  await writeEval('delivery2.json', {workload_json: 'workload.json', ranking_json: 'ranking.json',
    tasks: [{kernel, phase, result_path: 'tasks/leaf/run/RETURN.json'}]});
  return {evalDir, task, kernel, result, phase, snapshot,
    pack: () => packStandaloneTasks(evalDir, 'manifest.json', 'delivery.json'),
    read: () => readPreparedKernels(evalDir, 'delivery2.json')};
}
// The same task on one GPU: each written count becomes 1 (a collective without one declares 1 in
// its result, which adds no candidate field), a generated task gets world_size 1.
function singleGpuTwin({kernel = {}, result = {}, world = 0, rejectionExtra, ...rest}) {
  const one = o => ('required_gpu_count' in o ? {...o, required_gpu_count: 1} : {...o});
  const k = one(kernel), r = one(result);
  if (!('required_gpu_count' in k) && !('required_gpu_count' in r)) r.required_gpu_count = 1;
  return {...rest, kernel: k, result: r, world: Math.min(world, 1)};
}
const SCENARIOS = [
  // Single-GPU by main's count, or an invalid count: main's result.
  ['old-1gpu-plain', {}, {queue: 'kernel_candidates'}],
  ["old-k'4'", {kernel: {required_gpu_count: '4'}}, {invalid: true}],
  ['old-k0', {kernel: {required_gpu_count: 0}}, {invalid: true}],
  ['old-k-null', {kernel: {required_gpu_count: null}}, {queue: 'kernel_candidates'}],
  ['old-k-true', {kernel: {required_gpu_count: true}}, {invalid: true}],
  ['old-k1-r2', {kernel: {required_gpu_count: 1}, result: {required_gpu_count: 2}}, {queue: 'kernel_candidates'}],
  ["old-r'4'", {result: {required_gpu_count: '4'}}, {invalid: true}],
  ['old-k1-r0', {kernel: {required_gpu_count: 1}, result: {required_gpu_count: 0}}, {queue: 'kernel_candidates'}],
  ['old-collective-k1', {kernel: {op_kind: 'collective', required_gpu_count: 1}}, {queue: 'kernel_candidates'}],
  ['old-1gpu-ut-fail', {result: {unittest_smoke: 'fail'}}, {failed: 'ut_status'}],
  ['gen-1gpu', {world: 1}, {queue: 'kernel_candidates'}],
  ['gen-1gpu-k1', {world: 1, kernel: {required_gpu_count: 1}}, {queue: 'kernel_candidates'}],
  // Multi-GPU without per-rank timing or a declared count: main's deferred bucket and entry fields.
  ['old-k4-r2', {kernel: {required_gpu_count: 4}, result: {required_gpu_count: 2}},
    {deferred: 4, why: /: the request and the result do not declare one positive-integer GPU count: request required_gpu_count 4, result required_gpu_count 2; no timing receipt/}],
  ["old-k4-r'4'", {kernel: {required_gpu_count: 4}, result: {required_gpu_count: '4'}},
    {deferred: 4, why: /: the request and the result do not declare one positive-integer GPU count: request required_gpu_count 4, result required_gpu_count "4"; no timing receipt/}],
  ['old-collective-nocount', {kernel: {op_kind: 'collective'}},
    {deferred: 2, why: /: no required_gpu_count in the request or the result \(the collective default of 2 GPUs is not a declaration\); no timing receipt/}],
  ['old-collective-in-result-only', {result: {op_kind: 'collective'}},
    {deferred: 2, why: /: no required_gpu_count in the request or the result \(the collective default of 2 GPUs is not a declaration\); no timing receipt/}],
  ['old-collective-k2-no-timing', {kernel: {op_kind: 'collective', required_gpu_count: 2}}, {deferred: 2, why: /: no timing receipt; old format: no timing$/}],
  ['old-k8-no-timing', {kernel: {required_gpu_count: 8}}, {deferred: 8, why: /: no timing receipt; old format: no timing$/}],
  // Multi-GPU, timed per rank and declared: a candidate with required_gpu_count (D1).
  ['old-k4-legacy-timing', {kernel: {required_gpu_count: 4}, ut: {timing: legacyTiming(4)}}, {queue: 'kernel_candidates', gpus: 4}],
  ['old-r4-only-legacy-timing', {result: {required_gpu_count: 4}, ut: {timing: legacyTiming(4)}}, {queue: 'kernel_candidates', gpus: 4}],
  ['old-k4-legacy-timing-bad-timeouts', {kernel: {required_gpu_count: 4}, meta: {timeouts: 'x'}, ut: {timing: legacyTiming(4)}},
    {queue: 'kernel_candidates', gpus: 4}],
  ['old-k4-receipt-format', {kernel: {required_gpu_count: 4},
    ut: {receipt: {alignment: {x: 1}, cases: {a: {baseline: {rank_ms: [1, 1, 1, 1]}}}}}}, {queue: 'kernel_candidates', gpus: 4}],
  ['old-extract_op-moe-k4-legacy', {phase: 'extract_op', kernel: {op_kind: 'moe', required_gpu_count: 4}, ut: {timing: legacyTiming(4)}},
    {queue: 'head_candidates', gpus: 4}],
  ['gen-4gpu-k4', {world: 4, kernel: {required_gpu_count: 4}}, {queue: 'kernel_candidates', gpus: 4}],
  ['gen-4gpu-r4-only', {world: 4, result: {required_gpu_count: 4}}, {queue: 'kernel_candidates', gpus: 4}],
  // Generated tasks breaking a rule of their own (A17 point 4): failed_tasks with main's check keys.
  ['gen-1gpu-k2', {world: 1, kernel: {required_gpu_count: 2}},
    {failed: 'source_delivery', why: /^declared GPU counts disagree: requested required_gpu_count 2, returned required_gpu_count absent, meta world_size 1$/}],
  ['gen-1gpu-static-pass', {world: 1, verify: {verdict: 'STATIC_PASS'}}, {failed: 'source_delivery', why: /^verify_ut verdict is STATIC_PASS, not PASS$/}],
  ['gen-1gpu-no-receipt', {world: 1, ut: {receipt: null}}, {failed: 'source_delivery', why: /^UT result carries no timing receipt$/}],
  ['gen-1gpu-disabled-noreason', {world: 1, meta: {controls: {disabled: {noop: ''}}}},
    {failed: 'source_delivery', why: /^disabled controls without a reason: noop$/}],
  ['gen-4gpu-undeclared', {world: 4}, {failed: 'source_delivery',
    why: /^declared GPU counts disagree: requested required_gpu_count absent, returned required_gpu_count absent, meta world_size 4; a 4-rank task writes required_gpu_count in its request or result$/}],
  ['gen-4gpu-k4-norankrows', {world: 4, kernel: {required_gpu_count: 4}, rejectionExtra: {ranks: null}},
    {failed: 'source_delivery', why: /^numerical_rejection has no per-rank results \(leaf_kernel\)$/}],
];
const DELIVERY_CHECKS = ['schema', 'selection', 'baseline_declaration', 'task', 'ut_status', 'source_delivery'];
const PREPARED_CHECKS = ['baseline_declaration', 'task', 'ut_status', 'source_delivery', 'boundary'];
const QUEUES = ['head_candidates', 'kernel_candidates', 'deferred_tasks', 'failed_tasks'];

test('packing and delivery checks over the 31 review-08 scenarios (table 2.1)', async t => {
  assert.equal(SCENARIOS.length, 31);
  const contract = await loadContract();
  for (const [name, options, want] of SCENARIOS) {
    const f = await scenario(t, options);
    const ctx = `${name}`;
    // The delivery check is main's for every task: same keys, and a pass is main's exact object.
    const report = await checkDelivery(f.kernel, structuredClone(f.result), f.phase, contract);
    assert.deepEqual(Object.keys(report.checks), DELIVERY_CHECKS, ctx);
    // (The probe's extract_op fixture has no result op_kind or seam evidence: main fails those two too.)
    assert.deepEqual(Object.entries(report.checks).filter(([, c]) => !c.ok).map(([k]) => k),
      want.failed ? [want.failed] : f.phase === 'extract_op' ? ['schema', 'selection'] : [], ctx);
    const mainPass = {ok: true, why: 'source and recorded GPU evidence match 1 requested kernels', snapshot_sha256: f.snapshot.snapshot_sha256};
    if (want.failed === 'source_delivery') {
      assert.equal(report.checks.source_delivery.ok, false, ctx);
      assert.match(report.checks.source_delivery.why, want.why, ctx);
    } else assert.deepEqual(report.checks.source_delivery, mainPass, ctx);
    if (want.failed === 'ut_status') assert.deepEqual(report.checks.ut_status, {ok: false, reported: 'fail'}, ctx);

    const queues = await f.read();
    const only = queue => QUEUES.filter(q => q !== queue).every(q => queues[q].length === 0)
      && queues[queue].length === 1;
    if (want.invalid) {
      // main: the invalid count throws inside the loop and lands in failed_tasks with this exact entry.
      const entry = {short_name: 'leaf', result_path: 'tasks/leaf/run/RETURN.json', reason: 'invalid required_gpu_count'};
      assert(only('failed_tasks'), ctx);
      assert.deepEqual(queues.failed_tasks, [entry], ctx);
      await assert.rejects(f.pack(), {message: `completed tasks have invalid evidence: ${JSON.stringify([entry])}`}, ctx);
      continue;
    }
    if (want.failed) {
      assert(only('failed_tasks'), ctx);
      const entry = queues.failed_tasks[0];
      assert.deepEqual(Object.keys(entry), ['short_name', 'task_dir', 'result_path', 'checks'], ctx);
      assert.deepEqual(Object.keys(entry.checks), PREPARED_CHECKS, ctx);
      assert.deepEqual(Object.entries(entry.checks).filter(([, c]) => !c.ok).map(([k]) => k), [want.failed], ctx);
      if (want.why) assert.match(entry.checks.source_delivery.why, want.why, ctx);
      await assert.rejects(f.pack(), /completed tasks have invalid evidence/, ctx);
      continue;
    }
    const {prepared} = await f.pack();
    for (const q of QUEUES) assert.equal(prepared[q].length, queues[q].length, `${ctx}: ${q}`);
    if (want.deferred) {
      assert(only('deferred_tasks'), `${ctx}: ${JSON.stringify(queues.failed_tasks)}`);
      const entry = queues.deferred_tasks[0];
      assert.equal(entry.required_gpu_count, want.deferred, ctx);
      assert.equal(entry.task_dir, f.task, ctx);
      assert.equal(entry.result_path, await fs.realpath(path.join(f.task, 'run/RETURN.json')), ctx);
      assert.match(entry.reason, DEFERRED, ctx);
      assert.match(entry.reason, want.why, ctx);
      const s = await scenario(t, singleGpuTwin(options));
      const single = (await s.read()).kernel_candidates[0];
      assertSingleGpuShape(entry, f, single, s, ['task_dir', 'result_path', 'required_gpu_count', 'reason']);
      continue;
    }
    assert(only(want.queue), `${ctx}: ${JSON.stringify(queues.failed_tasks)} ${JSON.stringify(queues.deferred_tasks.map(d => d.reason))}`);
    const candidate = queues[want.queue][0];
    if (want.gpus) {
      assert.equal(candidate.required_gpu_count, want.gpus, ctx);
      const s = await scenario(t, singleGpuTwin(options));
      const single = (await s.read())[want.queue][0];
      assertSingleGpuShape(candidate, f, single, s, ['required_gpu_count']);
    } else {
      // A single-GPU candidate carries exactly what its request declared (main's `...kernel`).
      assert.equal('required_gpu_count' in candidate, 'required_gpu_count' in f.kernel, ctx);
      assert.equal(candidate.required_gpu_count, f.kernel.required_gpu_count, ctx);
    }
  }
});
