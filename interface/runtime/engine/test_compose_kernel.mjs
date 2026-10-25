// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Real child processes, source staging and native request/report readers. The
// full-runtime child and UT timing evidence are synthetic; no model or GPU runs.
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { runProcess } from './backends/base.mjs';
import { composeKernel } from './compose_kernel.mjs';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const PYTHON = process.env.PYTHON || 'python3';
const json = async path => JSON.parse(await readFile(path, 'utf8'));

const fakeRuntime = String.raw`
import { appendFile, readFile, writeFile } from 'node:fs/promises';
const argv = process.argv.slice(2);
const value = key => argv[argv.indexOf(key) + 1];
const args = argv.includes('--args-file')
  ? JSON.parse(await readFile(value('--args-file'), 'utf8')) : JSON.parse(value('--args'));
const event = kind => appendFile(process.env.TEST_EVENTS, JSON.stringify({
  kind, pid: process.pid, at: Date.now(), cwd: process.cwd(), argv, args,
  context: process.env.TEST_CONTEXT}) + '\n');
await event('start');
await new Promise(resolve => setTimeout(resolve, Number(process.env.TEST_DELAY || 50)));
if (process.env.TEST_FAIL === '1') { await event('failed'); process.exit(7); }
const result = JSON.parse(await readFile(process.env.TEST_TEMPLATE, 'utf8'));
await writeFile(value('--result-file'), JSON.stringify(result));
await writeFile(value('--metrics-file'), JSON.stringify({ok: true, agents_spawned: 3, wall_ms: 50}));
await event('finish');
`;

// Setup writes an exact task plus child terminal artifacts. During common
// measurement only run_native is replaced; the existing adapter still captures
// source, creates operation IDs, acquires its lock and checks raw reports.
const nativeFixture = String.raw`
import difflib, hashlib, json, os, re, shutil, sys
from pathlib import Path
from geak.native._util import canonical_json
from geak.native.external import _gate
from scripts.native_runtime import direct_external_command as adapter
from tests.native.fixtures import fixture, raw_reports

def evidence(request, factor):
    artifacts = raw_reports(request)
    history = []
    for index in range(1, 4):
        prefix = f'process_{index:02d}/'
        report = json.loads(artifacts[prefix+'ut_result.json'])
        for row in report['per_case']:
            row['optimized_ms'] /= factor
            row['optimized_legs'] = [v/factor for v in row['optimized_legs']]
            row['speedup'] *= factor
        for fold in report['folds'].values():
            fold['weighted'] *= factor
        report['call_weighted'] *= factor
        current = json.loads(artifacts[prefix+'_runs_v2.jsonl'].splitlines()[-1])
        current.update(per_case=report['per_case'], weighted=report['call_weighted'],
            folds={key: value['weighted'] for key, value in report['folds'].items()},
            buckets={row['name']: {'sp': row['speedup'], 'mde': row['mde_frac']} for row in report['per_case']})
        history.append(current)
        gain, regression = _gate(history)
        report.update(admissible=gain, reproducible_gain=gain, reproducible_regression=regression,
            repro={'runs': index, 'enough': index >= 2, 'reproducible_gain': gain,
                   'reproducible_regression': regression})
        artifacts[prefix+'ut_result.json'] = canonical_json(report)
        artifacts[prefix+'_runs_v2.jsonl'] = b'\n'.join(map(canonical_json, history))+b'\n'
        receipt = json.loads(artifacts[prefix+'RECEIPT.json'])
        for name in ('ut_result.json', '_runs_v2.jsonl'):
            receipt[name+'_sha256'] = hashlib.sha256(artifacts[prefix+name]).hexdigest()
        artifacts[prefix+'RECEIPT.json'] = canonical_json(receipt)
    return artifacts

def write_evidence(request, workspace, output):
    raw = (workspace/'kernel_src/module.py').read_text()
    name = re.search(r'synthetic (geak|evo) factor=', raw)
    name = name.group(1) if name else 'original'
    if os.environ.get('TEST_COMMON_FAIL') in (name, '*'):
        raise ValueError('synthetic common evaluator failure')
    factor = re.search(r'factor=([0-9.]+)', raw)
    factor = float(factor.group(1)) if factor else None
    overrides = json.loads(os.environ.get('TEST_COMMON_FACTORS', '{}'))
    if name in overrides:
        factor = overrides[name]
    output.mkdir(parents=True, exist_ok=False)
    artifacts = raw_reports(request, original=True) if factor is None else evidence(request, factor)
    for artifact, data in artifacts.items():
        path = output/artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return 0

if sys.argv[1] == '--setup':
    root, options = Path(sys.argv[2]), json.loads(sys.argv[3])
    task = root/'task'
    contract = fixture(task)
    (task/'task.json').write_bytes(canonical_json(contract))
    for name, factor in [('geak', options.get('geak_factor', 1.2)), ('evo', options.get('evo_factor', 1.4))]:
        lane = root/name
        workspace = lane/'workspace'
        shutil.copytree(task, workspace)
        original = (task/'kernel_src/module.py').read_text()
        candidate = original if options.get(name+'_original') else '# synthetic '+name+' factor='+str(factor)+'\n'
        (workspace/'kernel_src/module.py').write_text(candidate)
        patch = lane/'final.diff'
        patch.write_text(''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
            fromfile='a/kernel_src/module.py', tofile='b/kernel_src/module.py')))
        captured = adapter.snapshot(workspace, contract['source_paths'])
        request = adapter.create_request(contract, task, 'child', name+'-final', captured, workspace)
        operation = lane/'state'/'evaluations'/request['operation_id']
        operation.mkdir(parents=True)
        (operation/'request.json').write_bytes(canonical_json(request))
        write_evidence(request, workspace, operation/'native')
        checked = adapter.checked_result(request, operation/'native', task, patch)
        result = {**checked, 'eval_dir': str(lane/'eval'), 'final_speedup': checked['primary_speedup'],
            'final_patch': str(patch), 'final_patch_sha256': checked['patch_sha256'],
            'final_selection': {'schema': 'geak-native-measurement-identity-v1', 'capture_only': True,
                'candidate_id': request['binding']['candidate_id'], 'evaluation_id': request['operation_id'],
                'source_id': request['binding']['candidate_source_id'], 'source_paths': contract['source_paths']}}
        (lane/'eval').mkdir()
        (lane/'eval'/'host_final_patch.diff').write_bytes(patch.read_bytes())
        if not checked['native_reproducible_gain']:
            result.update(final_patch='', final_patch_sha256=None)
        (lane/'terminal.json').write_bytes(canonical_json(result))
    print(hashlib.sha256((task/'task.json').read_bytes()).hexdigest())
else:
    adapter.run_native = write_evidence
    sys.exit(adapter.main(sys.argv[1:]))
`;

async function fixture(t, options = {}) {
  const root = await mkdtemp(join(tmpdir(), 'geak-compose-test-'));
  t.after(() => rm(root, {recursive: true, force: true}));
  const runtime = join(root, 'fake_runtime.mjs'), native = join(root, 'fake_native.py');
  await writeFile(runtime, fakeRuntime);
  await writeFile(native, `import sys\nsys.path.insert(0, ${JSON.stringify(REPO)})\n` + nativeFixture);
  const setup = await runProcess({cmd: PYTHON, args: [native, '--setup', root, JSON.stringify(options)],
    env: {PYTHONPATH: REPO}, timeoutMs: 10000});
  assert.equal(setup.code, 0, setup.stderr);
  const task = join(root, 'task');
  const direct = join(REPO, 'scripts/native_runtime/direct_external_command.py');
  const prefix = (state, epoch) => ['--task-root', task, '--state-root', state,
    '--gpu-lock-file', join(root, 'native.lock'), '--epoch', epoch];
  const lanes = [];
  const deadline = Date.now() / 1000 + 30;
  for (const id of ['geak', 'evo']) {
    const args = {optimizer: id === 'geak' ? 'greedy' : 'evo', kernel_path: task,
      workflow_dir: join(REPO, 'kernel_workflow'), exp_root: join(root, id, 'exp'), eval_dir: join(root, id, 'eval'),
      gpu_ids: id === 'geak' ? '0' : '1',
      budget: id === 'geak' ? 6 : 12, task: 'Synthetic exact callable contract',
      objective: {kind: 'external_primary', primary_regime: 'extend'},
      deadline_epoch: deadline, search_reserve_s: 1,
      final_validation_reserve_s: 0.5, final_cleanup_reserve_s: 0.1,
      external_measurement_command: [PYTHON, native, ...prefix(join(root, id, 'state'), 'child'), 'measure'],
      external_result_command: [PYTHON, direct, ...prefix(join(root, id, 'state'), 'child'), 'read'],
      apply_to_original: 'false', warm_start: 'off', use_learned_kb: 'false'};
    const args_file = join(root, id, 'args.json');
    await writeFile(args_file, JSON.stringify(args));
    lanes.push({id, policy: {finalist: id === 'geak' ? 'required' : 'eligible', imports: 'none'}, args_file, runtime_command: [process.execPath, runtime],
      runtime_options: {profile: 'synthetic', concurrency: id === 'geak' ? 2 : 4},
      cwd: join(root, id), env: {PYTHONPATH: REPO, TEST_CONTEXT: id,
        TEST_TEMPLATE: join(root, id, 'terminal.json'), TEST_EVENTS: join(root, 'events.jsonl')},
      timeout_ms: 60000, resources: {context_id: id, device_ids: [id === 'geak' ? 'physical-A' : 'physical-B'],
        gpu_ids: [id === 'geak' ? '0' : '1']}});
  }
  const manifest = {schema: 'geak.kernel-composition/2', task_root: task, deadline_epoch: deadline,
    task_sha256: setup.stdout.trim(), out_dir: join(root, 'composed'),
    evaluation: {measurement_command: [PYTHON, native, ...prefix(join(root, 'common'), 'common'), 'measure'],
      result_command: [PYTHON, native, ...prefix(join(root, 'common'), 'common'), 'read'], timeout_ms: 10000}, lanes};
  return {root, manifest, lanes, async terminal(id, change) {
    const path = join(root, id, 'terminal.json');
    const result = await json(path); change(result); await writeFile(path, JSON.stringify(result));
  }};
}

test('separate full runtime launches preserve args and select fresh common native evidence', async t => {
  const f = await fixture(t);
  const input = await Promise.all(f.lanes.map(lane => json(lane.args_file)));
  const old = await json(join(f.root, 'evo/terminal.json'));
  // Child summaries are not the score of the new common comparison.
  await f.terminal('geak', result => { result.final_speedup = 900; });
  const result = await composeKernel(f.manifest);
  assert.equal(result.winner?.id, 'evo');
  assert.equal(result.required_finalist.retained, true);
  assert.equal(result.children.length, 2);
  assert.ok(result.children.every(child => child.status === 'qualified'));
  assert.notEqual(result.winner.final_selection.evaluation_id, old.final_selection.evaluation_id);
  assert.notEqual(result.winner.final_selection.candidate_id, old.final_selection.candidate_id,
    'the common epoch gets a newly captured candidate identity');
  assert.equal(result.winner.final_selection.source_id, old.final_selection.source_id);
  assert.equal(result.winner.final_patch_sha256, old.final_patch_sha256);
  assert.ok(result.winner.primary_speedup < 10, 'self-reported child score is not selected');
  const events = (await readFile(join(f.root, 'events.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  const starts = events.filter(event => event.kind === 'start');
  assert.equal(starts.length, 2);
  assert.notEqual(starts[0].pid, starts[1].pid);
  for (const [i, lane] of f.lanes.entries()) {
    const event = starts.find(event => event.context === lane.id);
    assert.equal(event.cwd, lane.cwd);
    assert.deepEqual(event.args, input[i], 'all original task and greedy/EVO settings reach the worker');
    assert.equal(event.argv[event.argv.indexOf('--concurrency') + 1], String(lane.runtime_options.concurrency));
    assert.equal(event.argv[event.argv.indexOf('--profile') + 1], 'synthetic');
  }
  assert.deepEqual(await json(join(f.manifest.out_dir, 'result.json')), result);
});

test('an additional complete greedy control can win without changing the protected GEAK lane', async t => {
  const f = await fixture(t);
  const geakArgs = await json(f.lanes[0].args_file);
  const controlArgs = {...await json(f.lanes[1].args_file), optimizer: 'greedy'};
  await writeFile(f.lanes[1].args_file, JSON.stringify(controlArgs));
  f.lanes[1].id = 'control';
  f.lanes[1].env.TEST_CONTEXT = 'control';
  f.lanes[1].resources.context_id = 'control';
  const result = await composeKernel(f.manifest);
  assert.equal(result.winner?.id, 'control');
  assert.equal(result.selection_basis, 'common_native_measurement');
  assert.equal(result.required_finalist.qualified, true);
  assert.equal(result.comparison_complete, true);
  const events = (await readFile(join(f.root, 'events.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  const geak = events.find(event => event.kind === 'start' && event.context === 'geak');
  const control = events.find(event => event.kind === 'start' && event.context === 'control');
  assert.notEqual(geak.pid, control.pid);
  assert.deepEqual(geak.args, geakArgs, 'protected GEAK retains its complete original greedy configuration');
  assert.deepEqual(control.args, controlArgs, 'extra resources may use the existing greedy optimizer');
  assert.equal(geak.args.budget, 6);
  assert.equal(control.args.budget, 12);
  for (const lane of f.lanes) {
    assert.deepEqual(result.children.find(child => child.id === lane.id).resources, lane.resources);
  }
});

test('slower and tied extra finalists preserve the actual GEAK candidate', async t => {
  for (const evo_factor of [1.05, 1.2]) await t.test(String(evo_factor), async t => {
    const f = await fixture(t, {geak_factor: 1.2, evo_factor});
    const result = await composeKernel(f.manifest);
    assert.equal(result.winner?.id, 'geak');
    assert.equal(result.winner.final_patch_sha256, (await json(join(f.root, 'geak/terminal.json'))).final_patch_sha256);
  });
});

test('failed, vetoed or changed-source extras cannot replace completed GEAK', async t => {
  for (const fault of ['process', 'director', 'patch']) await t.test(fault, async t => {
    const f = await fixture(t, {evo_factor: 2});
    if (fault === 'process') {
      f.lanes[1].env.TEST_FAIL = '1';
      f.lanes[0].env.TEST_DELAY = '200';
    }
    if (fault === 'director') await f.terminal('evo', result => { result.validation_status = 'flagged'; });
    if (fault === 'patch') await writeFile(join(f.root, 'evo/final.diff'), 'changed after terminal capture');
    const result = await composeKernel(f.manifest);
    assert.equal(result.winner?.id, 'geak');
    assert.equal(result.required_finalist.retained, true);
    assert.equal(result.children.find(child => child.id === 'evo').status, 'failed');
    assert.equal((await json(join(f.manifest.out_dir, 'geak/candidate.json'))).status, 'qualified');
  });
});

test('missing GEAK remains explicit even when an extra qualifies', async t => {
  const f = await fixture(t);
  f.lanes[0].env.TEST_FAIL = '1';
  const result = await composeKernel(f.manifest);
  assert.equal(result.winner?.id, 'evo');
  assert.equal(result.required_finalist.retained, false);
  assert.equal(result.required_finalist.status, 'failed');
  assert.equal(result.children.find(child => child.id === 'geak').process.code, 7);
});

test('foreign task, objective and shared resource grants fail before worker launch', async t => {
  for (const fault of ['task', 'regime', 'device']) await t.test(fault, async t => {
    const f = await fixture(t);
    if (fault === 'task') f.manifest.task_sha256 = 'd'.repeat(64);
    if (fault === 'device') f.lanes[1].resources.device_ids = ['physical-A'];
    if (fault === 'regime') {
      const args = await json(f.lanes[1].args_file);
      args.objective.primary_regime = 'foreign';
      await writeFile(f.lanes[1].args_file, JSON.stringify(args));
    }
    await assert.rejects(composeKernel(f.manifest), /task|objective|disjoint/);
    await assert.rejects(readFile(join(f.root, 'events.jsonl')), {code: 'ENOENT'});
  });
});

test('a forged final source identity cannot survive common source capture', async t => {
  const f = await fixture(t, {evo_factor: 2});
  await f.terminal('evo', result => { result.final_selection.source_id = 'a'.repeat(64); });
  const result = await composeKernel(f.manifest);
  assert.equal(result.winner?.id, 'geak');
  const extra = result.children.find(child => child.id === 'evo');
  assert.equal(extra.status, 'failed');
  assert.match(extra.error, /candidate|source/);
});

test('a fresh common comparison can reverse child ranking', async t => {
  const f = await fixture(t, {geak_factor: 1.1, evo_factor: 1.5});
  f.manifest.evaluation.env = {TEST_COMMON_FACTORS: JSON.stringify({geak: 1.5, evo: 1.05})};
  const result = await composeKernel(f.manifest);
  assert.equal(result.winner?.id, 'geak');
  const geak = result.children.find(child => child.id === 'geak');
  const evo = result.children.find(child => child.id === 'evo');
  assert.ok(geak.child_evidence.primary_speedup < evo.child_evidence.primary_speedup);
  assert.ok(geak.common_evidence.primary_speedup > evo.common_evidence.primary_speedup);
});

test('common measurement failures preserve artifacts and distinguish unqualified GEAK', async t => {
  for (const failed of ['evo', 'geak', '*']) await t.test(failed, async t => {
    const f = await fixture(t);
    f.manifest.evaluation.env = {TEST_COMMON_FAIL: failed};
    const result = await composeKernel(f.manifest);
    assert.equal(result.winner?.id, 'geak');
    assert.equal(result.required_finalist.retained, true, 'a failed replay must not erase GEAK source/evidence');
    assert.equal(result.required_finalist.qualified, failed === 'evo');
    for (const child of result.children) {
      assert.equal(await readFile(child.patch, 'utf8'), await readFile(join(f.root, child.id, 'final.diff'), 'utf8'));
      if (failed === child.id || failed === '*') assert.equal(child.status, 'qualification_failed');
    }
    if (failed !== 'evo') {
      assert.equal(result.selection_basis, 'retained_required_fallback');
      assert.equal(result.outcome, 'comparison_incomplete_required_retained');
      assert.equal(result.winner.primary_speedup, null, 'retention must not invent a fresh score');
      assert.equal(result.winner.comparison_unavailable, true);
      assert.ok(result.winner.child_evidence.primary_speedup > 1, 'old evidence stays explicitly separate');
    }
  });
});

test('a reserved staging collision preserves exported evidence and the GEAK fallback', async t => {
  const f = await fixture(t);
  const extraDirectory = join(f.manifest.out_dir, 'geak_final_workspace');
  f.lanes[1].output_dir = extraDirectory;
  const result = await composeKernel(f.manifest);
  const geak = result.children.find(child => child.id === 'geak');
  const extra = result.children.find(child => child.id === 'evo');
  assert.equal(geak.status, 'qualification_failed');
  assert.match(geak.qualification_error, /EEXIST/);
  assert.equal(extra.status, 'qualified');
  assert.equal(extra.directory, extraDirectory);
  assert.equal(result.required_finalist.retained, true);
  assert.equal(result.required_finalist.qualified, false);
  assert.equal(result.comparison_complete, false);
  assert.equal(result.winner?.id, 'geak');
  assert.equal(result.selection_basis, 'retained_required_fallback');
  assert.equal(result.outcome, 'comparison_incomplete_required_retained');
  assert.equal(result.winner.primary_speedup, null);
  assert.equal(result.winner.final_patch, geak.patch);
  for (const child of result.children) {
    assert.equal(await readFile(child.patch, 'utf8'), await readFile(join(f.root, child.id, 'final.diff'), 'utf8'));
    assert.deepEqual(await json(child.result_file), await json(join(f.root, child.id, 'terminal.json')));
    assert.equal((await json(child.metrics_file)).ok, true);
    assert.deepEqual(await json(join(child.directory, 'candidate.json')), child);
  }
});

test('valid ORIGINAL/no-gain GEAK is retained and freshly qualified as no improvement', async t => {
  const f = await fixture(t, {geak_original: true, evo_factor: 0.1});
  const result = await composeKernel(f.manifest);
  assert.equal(result.winner, null);
  assert.equal(result.required_finalist.retained, true);
  assert.equal(result.required_finalist.qualified, true);
  assert.equal(result.required_finalist.status, 'no_improvement');
  const geak = result.children.find(child => child.id === 'geak');
  assert.equal(await readFile(geak.patch, 'utf8'), '');
  assert.equal(geak.common_evidence.primary_speedup, 1);
  assert.equal(geak.common_evidence.native_reproducible_gain, false);
});

test('child launch uses captured args even if the caller file changes afterward', async t => {
  const f = await fixture(t);
  const original = await json(f.lanes[0].args_file);
  let changed = false;
  const run = async command => {
    if (!changed && command.cmd === process.execPath) {
      changed = true;
      await writeFile(f.lanes[0].args_file, JSON.stringify({...original, budget: 999, optimizer: 'evo'}));
    }
    return runProcess(command);
  };
  const result = await composeKernel(f.manifest, {run});
  assert.equal(result.required_finalist.qualified, true);
  const events = (await readFile(join(f.root, 'events.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  const geak = events.find(event => event.kind === 'start' && event.context === 'geak');
  assert.deepEqual(geak.args, original);
  assert.equal((await json(f.lanes[0].args_file)).budget, 999, 'fixture really changed the caller file');
});

test('one arbitrarily named lineage preserves the direct child invocation and accepted stop', async t => {
  const f = await fixture(t);
  f.manifest.lanes = [f.lanes[0]];
  const lane = f.manifest.lanes[0];
  lane.id = 'reference';
  lane.runtime_options.backend_events_file = join(f.root, 'geak/backend_calls.jsonl');
  const originalArgs = await json(lane.args_file);
  originalArgs.max_no_improve = 7;
  originalArgs.max_forced_replans = 3;
  await writeFile(lane.args_file, JSON.stringify(originalArgs));
  await f.terminal('geak', result => { result.stopped_by = 'no_improve'; result.rounds = 4; });
  const directResult = join(f.root, 'direct_result.json'), directMetrics = join(f.root, 'direct_metrics.json');
  const direct = await runProcess({cmd: lane.runtime_command[0], args: [...lane.runtime_command.slice(1),
    join(REPO, 'kernel_workflow/kernel_workflow.js'),
    ...Object.entries(lane.runtime_options).flatMap(([key, value]) => ['--' + key.replaceAll('_', '-'), String(value)]),
    '--args-file', lane.args_file, '--result-file', directResult, '--metrics-file', directMetrics],
    cwd: lane.cwd, env: lane.env, timeoutMs: lane.timeout_ms});
  assert.equal(direct.code, 0);
  const result = await composeKernel(f.manifest);
  const child = result.children[0];
  assert.equal(result.winner.id, 'reference');
  assert.equal(result.required_finalist.id, 'reference');
  assert.deepEqual(await json(child.result_file), await json(directResult));
  assert.deepEqual(await json(child.metrics_file), await json(directMetrics));
  assert.equal(child.workflow_summary.stopped_by, 'no_improve');
  assert.ok(child.process.remaining_allowance_at_settlement_s > 0, 'accepted early stop leaves allowance unused');
  const events = (await readFile(join(f.root, 'events.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(events.map(event => event.kind), ['start', 'finish', 'start', 'finish'], 'one workflow call, no wrapper restart');
  const starts = events.filter(event => event.kind === 'start');
  assert.deepEqual(starts[0].args, starts[1].args);
  assert.deepEqual(starts[1].args, originalArgs);
  // Only each invocation's input snapshot and output destinations differ.
  const withoutPaths = argv => argv.slice(0, argv.indexOf('--args-file'));
  assert.deepEqual(withoutPaths(starts[0].argv), withoutPaths(starts[1].argv));
  assert.ok(starts[1].argv.includes('--backend-events-file'));
});

test('required finalist policy is independent of name and optimizer', async t => {
  const f = await fixture(t, {geak_factor: 1.2, evo_factor: 1.2});
  f.lanes[0].policy.finalist = 'eligible';
  f.lanes[1].policy.finalist = 'required';
  f.lanes[1].id = 'reference';
  const result = await composeKernel(f.manifest);
  assert.equal(result.required_finalist.id, 'reference');
  assert.equal(result.winner.id, 'reference', 'tie retains the declared required member, even with evo optimizer');
});

test('inconsistent policies, clocks, private paths and local grants fail before any child', async t => {
  for (const fault of ['policy', 'import', 'deadline', 'eval', 'exp', 'root', 'symlink', 'state', 'native', 'gpu', 'recorder']) {
    await t.test(fault, async t => {
      const f = await fixture(t);
      const args = await json(f.lanes[1].args_file);
      const first = await json(f.lanes[0].args_file);
      if (fault === 'policy') f.lanes[1].policy.finalist = 'required';
      if (fault === 'import') f.lanes[1].policy.imports = 'selected';
      if (fault === 'deadline') args.deadline_epoch += 1;
      if (fault === 'eval') args.eval_dir = first.eval_dir;
      if (fault === 'exp') args.exp_root = join(first.eval_dir, 'nested');
      if (fault === 'root') args.exp_root = '/';
      if (fault === 'symlink') {
        await symlink(first.eval_dir, join(f.root, 'alias'));
        args.exp_root = join(f.root, 'alias', 'fresh');
      }
      if (fault === 'state') args.state_dir = first.eval_dir;
      if (fault === 'native') args.external_measurement_command = first.external_measurement_command;
      if (fault === 'gpu') args.gpu_ids = '0';
      if (fault === 'recorder') f.lanes[1].runtime_options.backend_events_file = join(first.eval_dir, 'calls.jsonl');
      await writeFile(f.lanes[1].args_file, JSON.stringify(args));
      await assert.rejects(composeKernel(f.manifest), /policy|deadline|private path|gpu_ids/);
      await assert.rejects(readFile(join(f.root, 'events.jsonl')), {code: 'ENOENT'});
    });
  }
});

test('a child returning another evaluation directory is ineligible', async t => {
  const f = await fixture(t);
  await f.terminal('evo', result => { result.eval_dir = join(f.root, 'geak/eval'); });
  const result = await composeKernel(f.manifest);
  assert.equal(result.winner.id, 'geak');
  assert.match(result.children[1].error, /different private evaluation directory/);
});

test('a dangling symlink ancestor is rejected before it can alias a fresh private directory', async t => {
  const f = await fixture(t);
  const fresh = join(f.root, 'fresh'), alias = join(f.root, 'alias');
  await symlink(fresh, alias);
  for (const [i, lane] of f.lanes.entries()) {
    const args = await json(lane.args_file);
    args.state_dir = join(i === 0 ? fresh : alias, 'state');
    await writeFile(lane.args_file, JSON.stringify(args));
  }
  await assert.rejects(composeKernel(f.manifest), /unresolved symbolic link/);
  await assert.rejects(readFile(join(f.root, 'events.jsonl')), {code: 'ENOENT'});
});
