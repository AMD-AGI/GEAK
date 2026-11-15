import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { sourceSnapshot, fileRef, verifySourceDelivery, isGeneratedTask, utContract,
  utContractDefect, verifyUtDefect, rankReceiptDefect, legacyRankTimingDefect, perRankTimingDefect,
  utReceiptDefect, rankRejectionDefect, metaWorldSize, gpuCountDefect, disabledControlsDefect,
} from './kernel_source_delivery.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const GENERATED = '# GEAK-GENERATED unittest.py (generate_ut.py v1): do not edit; regenerate with generate_ut.py\n';
const CONTRACT_FILES = ['cases.py', 'harness_lib.py', 'leg_runner.py', 'meta.json', 'overlay_setup.py',
  'unittest.py', 'ut_runner.py'];
const ranks = (world, row = {}) => Array.from({ length: world }, (_, rank) =>
  ({ rank, launch_succeeded: true, numerical_mismatches: 3, ...row }));
const receipt = world => ({ version: 'geak_timing_receipt/v2', all_primed: true, timer_unprimed: false,
  cache_mode: 'read-evict', world_size: world, timer: 'cuda_event_graph',
  alignment: world > 1 ? { per_sample: 'flush, synchronize, gloo barrier, release', merge: 'max over ranks' } : null,
  cases: { m1: Object.fromEntries(['baseline', 'current'].map(leg => [leg, { ms: 0.1, timer: 'cuda_event_graph',
    primed: true, host_ms: 0.01, ...(world > 1 ? { rank_ms: Array(world).fill(0.1) } : {}) }])) } });

// `world` > 0 builds a task as generate_ut.py freezes it (marker, vendored runtime,
// run/ut_contract.json, run/verify_ut.json, timing receipt); 0 keeps a hand-written task.
async function fixture(t, nested = false, { world = 0, meta = {} } = {}) {
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
  const edit = async (name, mutate) => {
    const value = JSON.parse(await fs.readFile(path.join(task, name), 'utf8'));
    mutate(value);
    return write(name, value);
  };
  await write('meta.json', { target_callable: 'pkg:op', ...meta,
    ...(world ? { capabilities: { multi_rank: { world_size: world } } } : {}) });
  for (const name of ['cases.py', 'unittest.py', 'kernel_src/score.py', 'kernel_src/topk.cu'])
    await fs.writeFile(path.join(task, name), '# original source\n');
  let contract = null;
  if (world) {
    await fs.writeFile(path.join(task, 'unittest.py'), `${GENERATED}from ut_runner import main\n`);
    for (const name of ['ut_runner.py', 'harness_lib.py', 'leg_runner.py', 'overlay_setup.py'])
      await fs.writeFile(path.join(task, name), `# vendored ${name}\n`);
    const files = [];
    for (const name of CONTRACT_FILES) files.push(await fileRef(task, name));
    contract = { contract: 'geak_ut_contract/v1', generated_by: 'generate_ut.py v1', files,
      contract_sha256: sha(JSON.stringify(files)), world_size: world };
    await write('run/ut_contract.json', contract);
  }
  const snapshot = await sourceSnapshot(task);
  const m = { contract: 'kernel_source_delivery/v1', target_callable: 'pkg:op', ...snapshot, kernels: [] };
  m.ut_result_ref = await write('run/ut.json', { ok: true, exit_code: 0,
    target_callable: m.target_callable, snapshot_sha256: m.snapshot_sha256,
    ...(world ? { contract_sha256: contract.contract_sha256, world_size: world, receipt: receipt(world) } : {}) });
  for (const device_kernel of ['score', 'topk']) {
    const row = { device_kernel, source_files: [`kernel_src/${device_kernel === 'score' ? 'score.py' : 'topk.cu'}`] };
    for (const [field, kind] of [['execution_ref', 'candidate_execution'], ['negative_control_ref', 'numerical_rejection']]) {
      const log = `run/${device_kernel}_${kind}.log`;
      await fs.writeFile(path.join(task, log), 'fixture GPU trace\n');
      row[field] = await write(`run/${device_kernel}_${kind}.json`, {
        kind, ok: true, device_kernel, target_callable: m.target_callable, snapshot_sha256: m.snapshot_sha256,
        command: 'python unittest.py', gpu: '0', log_ref: await fileRef(task, log),
        invocations: 1, launch_succeeded: true, numerical_mismatches: 1,
        ...(world > 1 && kind === 'numerical_rejection' ? { ranks: ranks(world) } : {}),
      });
    }
    m.kernels.push(row);
  }
  if (world) await write('run/verify_ut.json', { contract: 'geak_verify_ut/v1', verdict: 'PASS',
    contract_sha256: contract.contract_sha256, snapshot_sha256: m.snapshot_sha256, world_size: world,
    gaps: [], errors: [] });
  const kernel = { source_delivery_required: true, selected_device_kernels: ['score', 'topk'],
    ...(world > 1 ? { required_gpu_count: world } : {}) };
  const result = { task_dir: task, target_callable: 'pkg:op' };
  const pack = async () => ({ ...result, source_delivery_ref: await write('run/source.json', m) });
  return { root, task, kernel, result, m, write, edit, pack, contract };
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

// Generated tasks (unittest.py written by generate_ut.py) carry stricter evidence.
const check = async f => verifySourceDelivery(f.kernel, await f.pack());
// Each case starts from the saved bytes, so one failure never masks the next.
async function eachFails(f, name, cases, rebind = ref => ref) {
  const file = path.join(f.task, name), saved = await fs.readFile(file);
  for (const [mutate, pattern] of cases) {
    await fs.writeFile(file, saved);
    rebind(await f.edit(name, mutate));
    assert.match((await check(f)).why, pattern);
  }
  await fs.writeFile(file, saved);
  rebind(await fileRef(f.task, name));
  assert.equal((await check(f)).ok, true, 'restored evidence passes again');
}

test('a hand-written task keeps the original checks and result', async t => {
  const f = await fixture(t);
  f.kernel.required_gpu_count = 4;
  const result = { ...await f.pack(), required_gpu_count: 2 };
  assert.deepEqual(await verifySourceDelivery(f.kernel, result), { ok: true,
    why: 'source and recorded GPU evidence match 2 requested kernels', snapshot_sha256: f.m.snapshot_sha256 });
  assert.equal(await isGeneratedTask(f.task), false);
  await fs.writeFile(path.join(f.task, 'unittest.py'), `#!/usr/bin/env python3\n${GENERATED}`);
  assert.equal(await isGeneratedTask(f.task), false, 'the marker counts only as the first line');
});

test('a generated task passes with its frozen contract, PASS verification and timing receipt', async t => {
  for (const world of [1, 4]) {
    const f = await fixture(t, false, { world });
    assert.equal(await isGeneratedTask(f.task), true);
    // Its extra rules only add failures: a pass is main's result object, single- or multi-GPU.
    assert.deepEqual(await check(f), { ok: true, why: 'source and recorded GPU evidence match 2 requested kernels',
      snapshot_sha256: f.m.snapshot_sha256 });
  }
});

test('a generated task requires run/ut_contract.json to list the runtime and match its files', async t => {
  const f = await fixture(t, false, { world: 1 });
  await fs.appendFile(path.join(f.task, 'harness_lib.py'), '# edited after freeze\n');
  assert.match((await check(f)).why, /changed after freeze: harness_lib.py/);
  await fs.writeFile(path.join(f.task, 'harness_lib.py'), '# vendored harness_lib.py\n');
  await eachFails(f, 'run/ut_contract.json', [
    [c => { c.files = c.files.filter(x => x.path !== 'ut_runner.py'); c.contract_sha256 = sha(JSON.stringify(c.files)); },
      /does not list ut_runner.py/],
    [c => { c.contract_sha256 = '0'.repeat(64); }, /contract_sha256 does not match/],
    [c => { c.files[0].sha256 = '0'.repeat(64); c.contract_sha256 = sha(JSON.stringify(c.files)); },
      /changed after freeze: cases.py/],
    [c => { c.files.push({ path: 'reference.pt', sha256: '0'.repeat(64) }); }, /UT contract file reference.pt/],
  ]);
  await fs.rm(path.join(f.task, 'run/ut_contract.json'));
  assert.match((await check(f)).why, /missing run\/ut_contract.json/);
});

test('a generated task requires a PASS run/verify_ut.json bound to the current contract and snapshot', async t => {
  const f = await fixture(t, false, { world: 1 });
  await eachFails(f, 'run/verify_ut.json', [
    [v => { v.verdict = 'FAIL'; }, /verdict is FAIL/],
    [v => { v.verdict = 'INCONCLUSIVE'; }, /verdict is INCONCLUSIVE/],
    [v => { v.contract_sha256 = '0'.repeat(64); }, /current UT contract/],
    [v => { delete v.snapshot_sha256; }, /current source snapshot/],
    [v => { v.gaps = ['within_tol control not exercised']; }, /coverage gaps/],
  ]);
  await fs.rm(path.join(f.task, 'run/verify_ut.json'));
  assert.match((await check(f)).why, /missing run\/verify_ut.json/);
});

test('a generated UT result names its contract and times every case per rank with alignment', async t => {
  const one = await fixture(t, false, { world: 1 });
  await eachFails(one, 'run/ut.json', [
    [u => { delete u.contract_sha256; }, /does not identify the current UT contract/],
    [u => { delete u.receipt; }, /no timing receipt/],
  ], ref => { one.m.ut_result_ref = ref; });
  const f = await fixture(t, false, { world: 4 });
  await eachFails(f, 'run/ut.json', [
    [u => { u.receipt.alignment = null; }, /timing receipt has no alignment/],
    [u => { u.receipt.cases = {}; }, /timing receipt has no timed case/],
    [u => { u.receipt.cases.m1.baseline.rank_ms.pop(); }, /case m1: baseline.rank_ms is not 4 finite per-rank times/],
    [u => { u.receipt.cases.m1.baseline.rank_ms[2] = null; }, /baseline.rank_ms is not 4/],
    [u => { delete u.receipt.cases.m1.current.rank_ms; }, /current.rank_ms is not 4/],
  ], ref => { f.m.ut_result_ref = ref; });
  f.m.ut_result_ref = await f.edit('run/ut.json', u => { delete u.receipt.cases.m1.current; });
  assert.equal((await check(f)).ok, true, 'a receipt without a current leg checks the baseline only');
});

test('a generated multi-rank task needs every rank to reject the source mutation', async t => {
  const f = await fixture(t, false, { world: 4 });
  const row = f.m.kernels[1];
  await eachFails(f, row.negative_control_ref.path, [
    [e => { delete e.ranks; }, /no per-rank results \(topk\)/],
    [e => { e.ranks.pop(); }, /3 rank rows for world_size 4/],
    [e => { e.ranks[3].rank = 0; }, /one row for rank 0/],
    [e => { e.ranks[2].launch_succeeded = false; }, /rank 2 did not numerically reject/],
    [e => { e.ranks[1].numerical_mismatches = 0; }, /rank 1 did not numerically reject/],
  ], ref => { row.negative_control_ref = ref; });
});

test('a generated task declares one GPU count in its request, result and meta', async t => {
  // world_size > 1: the request or the result (at least one) writes it, and every written one is world_size.
  const f = await fixture(t, false, { world: 4 });
  const result = await f.pack(), bare = { ...f.kernel, required_gpu_count: undefined };
  for (const [kernel, returned] of [[f.kernel, 4], [f.kernel, undefined], [bare, 4], [{ ...f.kernel, required_gpu_count: null }, 4]])
    assert.equal((await verifySourceDelivery(kernel, { ...result, required_gpu_count: returned })).ok, true,
      `request ${kernel.required_gpu_count}, result ${returned}`);
  for (const [kernel, returned, why] of [
    [bare, undefined, /requested required_gpu_count absent, returned required_gpu_count absent, meta world_size 4; a 4-rank task writes required_gpu_count in its request or result$/],
    [f.kernel, 2, /requested required_gpu_count 4, returned required_gpu_count 2, meta world_size 4$/],
    [{ ...f.kernel, required_gpu_count: 2 }, 2, /requested required_gpu_count 2, returned required_gpu_count 2/],
    [bare, '4', /returned required_gpu_count "4"/],
    [{ ...f.kernel, required_gpu_count: 8 }, undefined, /requested required_gpu_count 8, returned required_gpu_count absent/],
  ]) assert.match((await verifySourceDelivery(kernel, { ...result, required_gpu_count: returned })).why, why);
  // world_size 1: both may be absent; a written count must be 1.
  const single = await fixture(t, false, { world: 1 });
  const one = await single.pack();
  for (const [kernel, returned] of [[{}, undefined], [{ required_gpu_count: 1 }, undefined], [{}, 1], [{ required_gpu_count: 1 }, 1]])
    assert.equal((await verifySourceDelivery({ ...single.kernel, ...kernel }, { ...one, required_gpu_count: returned })).ok, true);
  assert.match((await verifySourceDelivery({ ...single.kernel, required_gpu_count: 2 }, one)).why,
    /^declared GPU counts disagree: requested required_gpu_count 2, returned required_gpu_count absent, meta world_size 1$/);
  assert.match((await verifySourceDelivery(single.kernel, { ...one, required_gpu_count: 2 })).why,
    /requested required_gpu_count absent, returned required_gpu_count 2, meta world_size 1$/);
});

// A28: a source mutation never overwrites the delivered run/ut_result.json. After the clean source
// is restored with the same snapshot and contract, the last full UT's result is delivered as is.
test('a generated task delivers the last full UT reused after a restored source mutation', async t => {
  for (const world of [1, 4]) {
    const f = await fixture(t, false, { world });
    const full = JSON.parse(await fs.readFile(path.join(f.task, 'run/ut.json'), 'utf8'));
    f.m.ut_result_ref = await f.write('run/ut_result.json', full);
    const delivered = await fs.readFile(path.join(f.task, 'run/ut_result.json'));
    assert.equal((await check(f)).ok, true);
    // The reversible mutation runs correctness-only and writes under run/mutations/<leaf>/.
    const leaf = path.join(f.task, 'kernel_src/topk.cu'), clean = await fs.readFile(leaf);
    await fs.writeFile(leaf, '# mutated: drops the output scale\n');
    const mutated = await sourceSnapshot(f.task);
    assert.notEqual(mutated.snapshot_sha256, f.m.snapshot_sha256);
    await fs.mkdir(path.join(f.task, 'run/mutations/topk'), { recursive: true });
    await f.write('run/mutations/topk/ut_result.json', { ...full, ok: false, exit_code: 1,
      snapshot_sha256: mutated.snapshot_sha256, mode: 'correctness_only' });
    assert.match((await check(f)).why, /changed after recorded validation/, 'not delivered while mutated');
    await fs.writeFile(leaf, clean);
    // Restored: same snapshot and contract as the last full UT, so its result is accepted unchanged.
    assert.deepEqual(await check(f), { ok: true, why: 'source and recorded GPU evidence match 2 requested kernels',
      snapshot_sha256: f.m.snapshot_sha256 });
    assert.deepEqual(await fs.readFile(path.join(f.task, 'run/ut_result.json')), delivered);
    // A restore that does not give back the snapshot is refused ...
    await fs.writeFile(leaf, Buffer.concat([clean, Buffer.from('\n')]));
    assert.match((await check(f)).why, /source, cases, harness or meta changed after recorded validation/);
    await fs.writeFile(leaf, clean);
    // ... and so is a reused result recorded for another snapshot or another UT contract.
    f.m.ut_result_ref = await f.write('run/ut_result.json', { ...full, snapshot_sha256: mutated.snapshot_sha256 });
    assert.match((await check(f)).why, /full GPU UT result must pass and identify the current entry and source snapshot/);
    f.m.ut_result_ref = await f.write('run/ut_result.json', { ...full, contract_sha256: '0'.repeat(64) });
    assert.match((await check(f)).why, /UT result does not identify the current UT contract/);
    f.m.ut_result_ref = await f.write('run/ut_result.json', full);
    assert.equal((await check(f)).ok, true, 'the last full UT result passes again');
  }
});

test('a generated task names a reason for every disabled negative control', async t => {
  const reason = 'the kernel allocates a fresh output buffer on every call';
  const bare = await fixture(t, false, { world: 1, meta: { controls: { disabled: { stale_output: reason, scale_out: ' ' } } } });
  assert.match((await check(bare)).why, /disabled controls without a reason: scale_out/);
  const named = await fixture(t, false, { world: 1, meta: { controls: { disabled: { stale_output: reason } } } });
  assert.equal((await check(named)).ok, true);
});

test('generated-task helpers are pure and accept the receipt or the older per-rank format', () => {
  const legacy = { alignment: { host: 'gloo barrier before every timed window' }, cases: [{ sig: 'decode',
    baseline: { per_rank_us: [54.76, 54.76, 54.77, 54.76] }, candidate: { per_rank_us: [54.88, 54.88, 54.88, 54.88] } }] };
  assert.equal(perRankTimingDefect({ receipt: receipt(4) }, 4), '');
  assert.equal(perRankTimingDefect({ timing: legacy }, 4), '');
  assert.equal(perRankTimingDefect({ ok: true }, 4), 'no timing receipt; old format: no timing');
  assert.equal(perRankTimingDefect({ receipt: receipt(1), timing: { cases: [] } }, 4),
    'timing receipt has no alignment; old format: timing has no alignment');
  assert.match(legacyRankTimingDefect(legacy, 2), /decode: baseline.per_rank_us is not 2/);
  assert.match(legacyRankTimingDefect({ ...legacy, cases: [] }, 4), /no timed case/);
  assert.match(rankReceiptDefect(receipt(4), 8), /rank_ms is not 8/);
  assert.equal(utReceiptDefect({ contract_sha256: 'c', receipt: receipt(1) }, 'c', 1), '');
  assert.match(utReceiptDefect({ receipt: receipt(1) }, undefined, 1), /current UT contract/);
  assert.equal(rankRejectionDefect({ ranks: ranks(2) }, 2), '');
  assert.match(rankRejectionDefect({ ranks: ranks(2, { numerical_mismatches: '3' }) }, 2), /rank 0 did not/);
  assert.equal(metaWorldSize({}), 1);
  assert.equal(metaWorldSize({ capabilities: { multi_rank: { world_size: 4 } } }), 4);
  for (const bad of [0, 2.5, '4'])
    assert.throws(() => metaWorldSize({ capabilities: { multi_rank: { world_size: bad } } }), /positive integer/);
  assert.equal(gpuCountDefect({}, {}, 1), '');
  assert.equal(gpuCountDefect({ required_gpu_count: 4 }, {}, 4), '');
  assert.equal(gpuCountDefect({}, { required_gpu_count: 4 }, 4), '');
  assert.equal(gpuCountDefect({ required_gpu_count: null }, { required_gpu_count: 4 }, 4), '');
  assert.match(gpuCountDefect({}, {}, 4), /requested required_gpu_count absent, returned required_gpu_count absent/);
  assert.match(gpuCountDefect({ required_gpu_count: 4 }, { required_gpu_count: '4' }, 4), /returned required_gpu_count "4"/);
  assert.match(gpuCountDefect({}, { required_gpu_count: 2 }, 1), /returned required_gpu_count 2, meta world_size 1/);
  assert.equal(verifyUtDefect({ verdict: 'PASS', contract_sha256: 'c', snapshot_sha256: 's' }, 'c', 's'), '');
  assert.match(verifyUtDefect({ verdict: 'PASS' }, undefined, undefined), /current UT contract/);
  assert.equal(disabledControlsDefect({}), '');
  assert.match(disabledControlsDefect({ controls: { disabled: ['noop'] } }), /map each disabled control/);
  assert.match(utContractDefect({ files: [] }, { files: [] }), /does not list unittest.py, cases.py, meta.json/);
});

test('the contract CLI recomputes run/ut_contract.json from its file list', async t => {
  const f = await fixture(t, false, { world: 1 });
  const cli = fileURLToPath(new URL('./kernel_source_delivery.mjs', import.meta.url));
  const run = () => spawnSync(process.execPath, [cli, 'contract', f.task], { encoding: 'utf8' });
  const out = run();
  assert.equal(out.status, 0, out.stderr);
  assert.deepEqual(JSON.parse(out.stdout), { files: f.contract.files, contract_sha256: f.contract.contract_sha256 });
  await fs.appendFile(path.join(f.task, 'leg_runner.py'), '# changed\n');
  assert.notEqual(JSON.parse(run().stdout).contract_sha256, f.contract.contract_sha256);
  assert.notEqual((await utContract(f.task)).contract_sha256, f.contract.contract_sha256);
});

// Recorded evidence fixtures exercise admission/reuse; they are not GPU tests.
import { readPreparedKernels } from './agentx_kernel_preparation.mjs';
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

test('complete-stage handoff accepts validated UTs without a separate selection report', async t => {
  const f = await preparationFixture(t);
  delete f.result.selection_validation;
  await f.write('run/RETURN.json', f.result);
  const ready = await f.save();
  assert.equal(ready.kernel_candidates.length, 1, JSON.stringify(ready.failed_tasks));
  assert.equal(ready.kernel_candidates[0].pct_gpu_time, 7.2);
  assert.equal(ready.failed_tasks.length, 1);
  assert.equal(ready.deferred_tasks.length, 1);
  const carried = ready.kernel_candidates[0].prepared_ut.result;
  assert.equal(carried.unittest_smoke, 'pass');
  assert.deepEqual(carried.notes, ['structured notes']);
  const source = await fs.readFile(new URL('../e2e_workflow.js', import.meta.url), 'utf8');
  // From preparedUtDefect: the AgentX branch reuses the prepared UT through it, and both are
  // plain functions in the workflow source -- the sandbox has no filesystem to import from.
  const block = source.slice(source.indexOf('function preparedUtDefect('), source.indexOf('// abDone =='));
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

// Preparation is the single gate: dispatch replays its verdict and cannot re-hash anything, so
// a result or source edited after validation has to stop a candidate HERE or not at all.
test('preparation refuses a task whose result or candidate source changed after validation', async t => {
  const f = await preparationFixture(t);
  assert.equal((await f.save()).kernel_candidates.length, 1, 'the clean task is queued');
  await fs.appendFile(path.join(f.task, 'kernel_src/score.py'), '# edited after UT\n');
  const edited = await f.save();
  assert.equal(edited.kernel_candidates.length, 0);
  assert.match(JSON.stringify(edited.failed_tasks.at(-1)), /source|snapshot|changed/i);
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
  const prepared = await f.save();
  const { createRuntime } = await import('../../interface/runtime/engine/run_workflow.mjs');
  const calls = [], logs = [];
  const runtime = createRuntime({ backend: { name: 'fixture', runAgent: async input => {
    calls.push(input.prompt);
    if (!input.prompt.includes('PHASE=prepare')) {
      assert.match(input.prompt, /workflow_return.json|PHASE=persist_llm_stats/);
      return { text: JSON.stringify({ written: true, path: f.root, ok: true }) };
    }
    // The envelope travels in the reply, not as a path: this is the stdout of
    // `agentx_kernel_preparation.mjs --emit-prepared`, which the workflow cannot read itself.
    return { text: JSON.stringify({ delivery_path: path.join(f.task, 'run/delivery.json'), prepared }) };
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

test('joint execution preserves explicitly requested source scope for unchanged dependencies', async t => {
  const f = await preparationFixture(t);
  const phase = { regime: 'DECODE', device_kernel: 'score', required_device_kernels: ['score', 'topk'],
    seam_candidates: [{ ...f.kernel.seam_candidates[0], device_kernels: ['score', 'topk'] }] };
  f.kernel.joint_ut = { phases: [phase] };
  f.result.selection_validation = { ...f.result.selection_validation, selection_scope: 'joint_chain',
    required_device_kernels: ['score', 'topk'],
    chain_members: ['score', 'topk'].map(device_kernel => ({ device_kernel, matched_kernel_calls: 1 })),
    joint_call_coverage: [{ call: 0, missing_members: [] }] };
  f.result.phase_selection_validations = [{ regime: 'DECODE', target_callable: f.result.target_callable,
    selection_validation: f.result.selection_validation }];
  f.m.kernels = f.m.kernels.filter(k => k.device_kernel === 'score');
  f.result.source_delivery_ref = (await f.pack()).source_delivery_ref;
  await f.write('run/RETURN.json', f.result);
  const ready = await f.save();
  assert.equal(ready.kernel_candidates.length, 1, JSON.stringify(ready.failed_tasks));
  f.kernel.source_device_kernels = ['topk'];
  const missingRequested = await f.save();
  assert.equal(missingRequested.kernel_candidates.length, 0);
  assert.match(missingRequested.failed_tasks[1].checks.source_delivery.why, /source record for topk/);
});
