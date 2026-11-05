import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { sourceSnapshot, utDriverSnapshot, UT_DRIVER_SNAPSHOT_CONTRACT,
  parsePythonUtArtifact, fileRef, verifySourceDelivery } from '../kernel_source_delivery.mjs';

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

const gates = ['contract', 'inputs', 'binding', 'launches', 'correctness', 'server_calls',
  'compile_variants', 'isolation', 'replay', 'negative_controls', 'restore', 'timing'];

async function driverFixture(t) {
  const f = await fixture(t, true);
  for (const name of ['ut_driver.py', 'harness_lib.py', 'compile_watch.py'])
    await fs.writeFile(path.join(f.task, name), '# frozen helper\n');
  await f.write('run/ut_cases.json', { cases: ['one', 'two'] });
  await f.write('run/ut_contract.json', { family: 'test' });
  await f.write('run/server_calls.json', { calls: ['entry'] });
  await fs.mkdir(path.join(f.task, 'run/server_calls'));
  await f.write('run/server_calls/input.json', { indices: [1, 2] });
  await fs.writeFile(path.join(f.task, 'kernel_src/tuning.csv'), 'm,tile\n8,16\n');
  const snapshot = await utDriverSnapshot(f.task);
  f.report = { started: 1700000000, verdict: 'pass', exit_code: 0,
    target_callable: f.m.target_callable, snapshot_sha256: snapshot.snapshot_sha256,
    reuse_key: snapshot.reuse_key, family: 'test', contract_sha256: 'a'.repeat(64),
    gates: Object.fromEntries(gates.map(g => [g, { status: g === 'replay' ? 'not_required' : 'pass' }])) };
  f.ut = { ...f.report, ok: true,
    gates: Object.fromEntries(Object.entries(f.report.gates).map(([g, r]) => [g, { ...r, ok: true }])) };
  delete f.ut.reuse_key;
  f.persist = async () => {
    f.m.ut_result_ref = await f.write('run/ut_result.json', f.ut);
    f.m.ut_snapshot = { contract: UT_DRIVER_SNAPSHOT_CONTRACT,
      report_ref: await f.write('run/ut_report.json', f.report) };
  };
  await f.persist();
  return f;
}

test('UT driver hashes match the independent Python _snapshot and _reuse_key implementations', async t => {
  const f = await driverFixture(t);
  await fs.mkdir(path.join(f.task, 'kernel_src/nested'));
  await fs.mkdir(path.join(f.task, 'kernel_src/__pycache__'));
  await fs.writeFile(path.join(f.task, 'kernel_src/nested/launch.dat'), 'not a source extension');
  await fs.writeFile(path.join(f.task, 'kernel_src/__pycache__/ignored.pyc'), 'ignored');
  await fs.writeFile(path.join(f.task, '\ue000.py'), '# BMP\n');
  await fs.writeFile(path.join(f.task, '\u{10000}.py'), '# supplementary code point\n');
  await fs.symlink('score.py', path.join(f.task, 'kernel_src/alias.py'));
  await fs.symlink('nested', path.join(f.task, 'kernel_src/dir_alias'));
  const driver = path.resolve(import.meta.dirname, '../../ut_contracts/ut_driver.py');
  const python = spawnSync('python3', ['-c', `
import ast, hashlib, json, os, sys, types
tree = ast.parse(open(sys.argv[1]).read())
names = {'_sha', '_tree_hashes', '_snapshot', '_reuse_key'}
subset = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
assert len(subset.body) == 4
namespace = {'hashlib': hashlib, 'os': os}
exec(compile(subset, sys.argv[1], 'exec'), namespace)
task = types.SimpleNamespace(dir=sys.argv[2])
print(json.dumps({k: namespace[f](task) for k, f in [('snapshot_sha256', '_snapshot'), ('reuse_key', '_reuse_key')]}))
`, driver, f.task], { encoding: 'utf8' });
  assert.equal(python.status, 0, python.stderr);
  const actual = await utDriverSnapshot(f.task);
  assert.deepEqual({ snapshot_sha256: actual.snapshot_sha256, reuse_key: actual.reuse_key }, JSON.parse(python.stdout));
});

test('legacy UT digest requires explicit compatibility and a hashed full report', async t => {
  const f = await driverFixture(t);
  assert.notEqual(f.ut.snapshot_sha256, f.m.snapshot_sha256);
  assert.equal((await verifySourceDelivery(f.kernel, await f.pack())).ok, true);
  const spec = f.m.ut_snapshot;
  delete f.m.ut_snapshot;
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /current entry and source snapshot/);
  f.m.ut_snapshot = { ...spec, contract: 'accept_any_snapshot' };
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /unsupported UT snapshot contract/);
  f.m.ut_snapshot = { contract: spec.contract };
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /path and SHA-256/);
  f.m.ut_snapshot = spec;
  await fs.appendFile(path.join(f.task, spec.report_ref.path), ' ');
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /SHA-256 mismatch/);
});

test('legacy reuse key binds frozen helpers and all specified server evidence', async t => {
  const f = await driverFixture(t), before = await utDriverSnapshot(f.task);
  for (const name of ['ut_driver.py', 'harness_lib.py', 'compile_watch.py', 'run/server_calls.json',
    'run/server_calls/input.json']) {
    const file = path.join(f.task, name), bytes = await fs.readFile(file);
    await fs.appendFile(file, ' ');
    assert.equal((await utDriverSnapshot(f.task)).snapshot_sha256, before.snapshot_sha256);
    assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /reuse_key does not match/, name);
    await fs.writeFile(file, bytes);
  }
  for (const name of ['new_helper.py', 'run/server_new.json', 'run/server_calls/new.bin']) {
    const file = path.join(f.task, name);
    await fs.writeFile(file, 'new dependency');
    assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /reuse_key does not match/, name);
    await fs.unlink(file);
  }
  assert.equal((await verifySourceDelivery(f.kernel, await f.pack())).ok, true);
});

test('legacy snapshot covers case manifests, contracts and non-source kernel data', async t => {
  const f = await driverFixture(t);
  for (const name of ['run/ut_cases.json', 'run/ut_contract.json', 'kernel_src/tuning.csv']) {
    const file = path.join(f.task, name), bytes = await fs.readFile(file);
    await fs.appendFile(file, ' ');
    // Canonical source snapshot is unchanged; accepting it alone would miss this.
    assert.deepEqual(await sourceSnapshot(f.task), { files: f.m.files, snapshot_sha256: f.m.snapshot_sha256 });
    assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /UT driver snapshot does not match/, name);
    await fs.writeFile(file, bytes);
  }
});

test('compatibility refuses partial, quick, skipped or incomplete full UT evidence', async t => {
  const f = await driverFixture(t), report = structuredClone(f.report), ut = structuredClone(f.ut);
  const mutations = [
    () => { f.report.verdict = 'partial'; },
    () => { f.report.quick = { cases: ['one'] }; },
    () => { delete f.report.gates.timing; },
    () => { f.report.gates.timing.status = 'skipped'; },
    () => { f.ut.gates.correctness.ok = false; },
    () => { f.ut.gates.correctness.status = 'fail'; },
    () => { delete f.ut.gates.compile_variants; },
    () => { f.ut.ok = false; f.ut.smoke = 'pass'; },
    () => { f.report.gates.extra = { status: 'fail' }; },
  ];
  for (const mutate of mutations) {
    f.report = structuredClone(report); f.ut = structuredClone(ut); mutate(); await f.persist();
    assert.equal((await verifySourceDelivery(f.kernel, await f.pack())).ok, false);
  }
});

test('compatibility binds the full report to the same run, target and contract as ut_result', async t => {
  const f = await driverFixture(t), report = structuredClone(f.report);
  for (const [key, value] of [['started', report.started + 1], ['family', 'another_family'],
    ['contract_sha256', 'b'.repeat(64)], ['target_callable', 'pkg:other'], ['reuse_key', 'c'.repeat(64)],
    ['snapshot_sha256', 'd'.repeat(64)]]) {
    f.report = { ...structuredClone(report), [key]: value }; await f.persist();
    assert.equal((await verifySourceDelivery(f.kernel, await f.pack())).ok, false, key);
  }
});

test('legacy compatibility retains canonical per-leaf snapshot checks and task-local dependency bounds', async t => {
  const f = await driverFixture(t), row = f.m.kernels[0];
  const evidence = JSON.parse(await fs.readFile(path.join(f.task, row.execution_ref.path)));
  evidence.snapshot_sha256 = f.ut.snapshot_sha256;
  row.execution_ref = await f.write(row.execution_ref.path, evidence);
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /incomplete candidate_execution/);
  evidence.snapshot_sha256 = f.m.snapshot_sha256;
  row.execution_ref = await f.write(row.execution_ref.path, evidence);
  await fs.writeFile(path.join(f.root, 'outside.py'), '# outside dependency');
  await fs.symlink('../outside.py', path.join(f.task, 'external_helper.py'));
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /outside task/);
});

function infiniteMarginReport(report = {}) {
  return { ...report, gates: { ...report.gates, negative_controls: { status: 'pass',
    controls: [{ caught: true, ok: true, restored: true, margin: 0 }] } } };
}
const marginText = (document, token = 'Infinity') =>
  JSON.stringify(document).replace('"margin":0', `"margin":${token}`);

test('Python margin normalization leaves quoted Infinity and escaped strings unchanged', () => {
  const document = infiniteMarginReport({ note: 'Infinity "NaN" -Infinity \\ \n',
    Infinity: 'quoted key and value: Infinity', escaped: 'Infinity',
    markerCollision: '\u0000geak-positive-infinity:0' });
  const text = marginText(document).replace('"escaped":"Infinity"', '"escaped":"\\u0049nfinity"');
  const parsed = parsePythonUtArtifact(text, 'ut_report');
  assert.equal(parsed.note, document.note);
  assert.equal(parsed.Infinity, document.Infinity);
  assert.equal(parsed.escaped, 'Infinity');
  assert.equal(parsed.markerCollision, document.markerCollision);
  assert.equal(parsed.gates.negative_controls.controls[0].margin, Infinity);
});

test('only successful known negative-control margin paths accept positive Infinity', () => {
  const report = infiniteMarginReport();
  for (const token of ['NaN', '-Infinity', '1e999', '-1e999'])
    assert.throws(() => parsePythonUtArtifact(marginText(report, token), 'ut_report'),
      /NaN|negative Infinity|non-finite JSON number/, token);
  for (const extra of ['"invocations":Infinity', '"device_kernel":Infinity',
    '"started":Infinity', '"margin":Infinity', '"timing":{"latency":Infinity}']) {
    const text = marginText(report).replace(/^\{/, `{${extra},`);
    assert.throws(() => parsePythonUtArtifact(text, 'ut_report'), /not allowed/, extra);
  }
  const failed = structuredClone(report);
  failed.gates.negative_controls.controls[0].caught = false;
  assert.throws(() => parsePythonUtArtifact(marginText(failed), 'ut_report'), /not allowed/);
  const objectControls = { gates: { negative_controls: { status: 'pass', controls: {
    0: { margin: 0, caught: true, ok: true, restored: true } } } } };
  assert.throws(() => parsePythonUtArtifact(marginText(objectControls), 'ut_report'), /not allowed/);
  assert.throws(() => parsePythonUtArtifact('{Infinity: 1}', 'ut_report'), /object key/);
  assert.throws(() => parsePythonUtArtifact('{"invocations":Infinity,"invocations":1}', 'ut_report'), /discarded/);
  assert.throws(() => parsePythonUtArtifact(marginText(report), 'arbitrary_diagnostics'), /unsupported/);
});

test('explicit driver compatibility accepts original hashed Python reports and leaf margin artifacts', async t => {
  const f = await driverFixture(t);
  const reportPath = 'run/ut_report.json';
  const reportBytes = marginText(infiniteMarginReport(f.report));
  await fs.writeFile(path.join(f.task, reportPath), reportBytes);
  f.m.ut_snapshot.report_ref = await fileRef(f.task, reportPath);
  const row = f.m.kernels[0], negativePath = row.negative_control_ref.path;
  const negative = JSON.parse(await fs.readFile(path.join(f.task, negativePath)));
  negative.ut_negative_controls_gate = { caught: true, ok: true, restored: true, margin: 0 };
  const negativeBytes = marginText(negative);
  await fs.writeFile(path.join(f.task, negativePath), negativeBytes);
  row.negative_control_ref = await fileRef(f.task, negativePath);
  assert.equal(parsePythonUtArtifact(negativeBytes, 'negative_control').ut_negative_controls_gate.margin, Infinity);
  assert.equal((await verifySourceDelivery(f.kernel, await f.pack())).ok, true);
  assert.equal(await fs.readFile(path.join(f.task, reportPath), 'utf8'), reportBytes);
  assert.equal(await fs.readFile(path.join(f.task, negativePath), 'utf8'), negativeBytes);
  // Integrity is checked against original Python bytes before lexical parsing.
  await fs.appendFile(path.join(f.task, reportPath), ' ');
  assert.match((await verifySourceDelivery(f.kernel, await f.pack())).why, /SHA-256 mismatch/);
});

test('non-finite leaf counts and canonical/default Python constants are never allowed', async t => {
  const f = await driverFixture(t), row = f.m.kernels[0], name = row.negative_control_ref.path;
  const negative = JSON.parse(await fs.readFile(path.join(f.task, name)));
  negative.ut_negative_controls_gate = { caught: true, ok: true, restored: true, margin: 0 };
  for (const token of ['Infinity', '-Infinity', 'NaN', '1e999']) {
    const text = JSON.stringify(negative).replace('"numerical_mismatches":1', `"numerical_mismatches":${token}`);
    await fs.writeFile(path.join(f.task, name), text);
    row.negative_control_ref = await fileRef(f.task, name);
    assert.equal((await verifySourceDelivery(f.kernel, await f.pack())).ok, false, token);
  }
  const canonical = await fixture(t), originalRow = canonical.m.kernels[0];
  const original = JSON.parse(await fs.readFile(path.join(canonical.task, originalRow.negative_control_ref.path)));
  original.ut_negative_controls_gate = { caught: true, ok: true, restored: true, margin: 0 };
  await fs.writeFile(path.join(canonical.task, originalRow.negative_control_ref.path), marginText(original));
  originalRow.negative_control_ref = await fileRef(canonical.task, originalRow.negative_control_ref.path);
  assert.equal((await verifySourceDelivery(canonical.kernel, await canonical.pack())).ok, false);
});
