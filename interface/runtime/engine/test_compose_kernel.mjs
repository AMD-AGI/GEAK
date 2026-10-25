// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Real child processes, source staging and native request/report readers. The
// full-runtime child and UT timing evidence are synthetic; no model or GPU runs.
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile, symlink, mkdir, cp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
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
import difflib, hashlib, json, os, re, shutil, sys, uuid
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
    # Joined CPU parity uses an explicit deterministic opaque-ID fixture. The
    # adapter still derives and verifies every source/candidate/receipt binding.
    if os.environ.get('TEST_NATIVE_ID_COUNTER'):
        counter = Path(os.environ['TEST_NATIVE_ID_COUNTER'])
        def next_uuid():
            n = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(n + 1))
            return uuid.UUID(hex=hashlib.sha256(('joined-parity-'+str(n)).encode()).hexdigest()[:32])
        adapter.uuid.uuid4 = next_uuid
    adapter.run_native = write_evidence
    sys.exit(adapter.main(sys.argv[1:]))
`;

async function fixture(t, options = {}) {
  const root = await mkdtemp(join(tmpdir(), 'geak-compose-test-'));
  t.after(async () => {
    if (options.joined_record_name && process.env.GEAK_JOINED_PARITY_ARTIFACT_DIR) {
      await cp(root, join(process.env.GEAK_JOINED_PARITY_ARTIFACT_DIR,
        options.joined_record_name + '-' + basename(root)), {recursive: true});
    }
    await rm(root, {recursive: true, force: true});
  });
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
      apply_to_original: 'false', warm_start: 'off', use_learned_kb: 'false',
      learned_dir: join(root, id, 'learned'), kb_artifacts_dir: join(root, id, 'kb_artifacts'),
      kb_store_dir: join(root, id, 'kb_store'), kb_remote: 'off'};
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
    let result;
    if (failed === 'geak') {
      const manifest = join(f.root, 'composition.json');
      await writeFile(manifest, JSON.stringify(f.manifest));
      const cli = await runProcess({cmd: process.execPath,
        args: [join(REPO, 'interface/runtime/engine/compose_kernel.mjs'), manifest], timeoutMs: 10000});
      assert.equal(cli.code, 1, 'failed required qualification keeps a nonzero CLI exit');
      result = JSON.parse(cli.stdout);
    } else result = await composeKernel(f.manifest);
    assert.equal(result.winner?.id, failed === 'geak' ? 'evo' : 'geak');
    assert.equal(result.comparison_complete, false);
    assert.equal(result.required_finalist.retained, true, 'a failed replay must not erase GEAK source/evidence');
    assert.equal(result.required_finalist.qualified, failed === 'evo');
    for (const child of result.children) {
      assert.equal(await readFile(child.patch, 'utf8'), await readFile(join(f.root, child.id, 'final.diff'), 'utf8'));
      if (failed === child.id || failed === '*') assert.equal(child.status, 'qualification_failed');
    }
    if (failed === '*') {
      assert.equal(result.selection_basis, 'retained_required_fallback');
      assert.equal(result.outcome, 'comparison_incomplete_required_retained');
      assert.equal(result.winner.primary_speedup, null, 'retention must not invent a fresh score');
      assert.equal(result.winner.comparison_unavailable, true);
      assert.ok(result.winner.child_evidence.primary_speedup > 1, 'old evidence stays explicitly separate');
    } else {
      assert.equal(result.selection_basis, 'common_native_measurement');
      assert.equal(result.outcome, 'improved');
      assert.ok(result.winner.primary_speedup > 1);
    }
  });
});

test('a reserved staging collision preserves failed required evidence and the common winner', async t => {
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
  assert.equal(result.winner?.id, 'evo');
  assert.equal(result.selection_basis, 'common_native_measurement');
  assert.equal(result.outcome, 'improved');
  assert.ok(result.winner.primary_speedup > 1);
  assert.equal(result.winner.final_patch, extra.patch);
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

test('one-member composition forwards declared external peer sources without changing clocks', async t => {
  const f = await fixture(t), lane = f.lanes[0], args = await json(lane.args_file);
  f.manifest.lanes = [lane];
  args.optimizer = 'evo';
  lane.policy.imports = 'selected';
  const root = join(f.root, 'external-peer-archive');
  await mkdir(root); // Mount isolation belongs to the caller, outside this fixture.
  args.evolution_peers = [{id: 'peer', root, native_epoch: 'peer-native'}];
  await writeFile(lane.args_file, JSON.stringify(args));
  const result = await composeKernel(f.manifest);
  assert.equal(result.deadline_epoch, f.manifest.deadline_epoch);
  assert.deepEqual((await json(join(f.manifest.out_dir, lane.id, 'args.json'))).evolution_peers, args.evolution_peers);
});

test('one-member composition rejects a peer overlapping its own writable state before launch', async t => {
  const f = await fixture(t), lane = f.lanes[0], args = await json(lane.args_file);
  f.manifest.lanes = [lane];
  args.optimizer = 'evo';
  lane.policy.imports = 'selected';
  args.evolution_peers = [{id: 'peer', root: join(args.eval_dir, 'evolution'), native_epoch: 'peer-native'}];
  await writeFile(lane.args_file, JSON.stringify(args));
  await assert.rejects(composeKernel(f.manifest), /peer root overlaps recipient writable state/);
  await assert.rejects(readFile(join(f.root, 'events.jsonl')), {code: 'ENOENT'});
});

test('a child returning another evaluation directory is ineligible', async t => {
  const f = await fixture(t);
  await f.terminal('evo', result => { result.eval_dir = join(f.root, 'geak/eval'); });
  const result = await composeKernel(f.manifest);
  assert.equal(result.winner.id, 'geak');
  assert.match(result.children[1].error, /different private evaluation directory/);
});

test('every lineage requires explicit nonempty local knowledge roots', async t => {
  for (const index of [0, 1]) for (const key of ['learned_dir', 'kb_artifacts_dir', 'kb_store_dir']) {
    for (const value of [undefined, '']) await t.test(`${index}:${key}:${value === '' ? 'empty' : 'missing'}`, async t => {
      const f = await fixture(t), lane = f.lanes[index], args = await json(lane.args_file);
      args[key] = value;
      await writeFile(lane.args_file, JSON.stringify(args));
      await assert.rejects(composeKernel(f.manifest), /private paths must be absolute/);
      await assert.rejects(readFile(join(f.root, 'events.jsonl')), {code: 'ENOENT'});
    });
  }
});

test('knowledge roots and optional memory hooks cannot alias another lineage', async t => {
  for (const key of ['learned_dir', 'kb_artifacts_dir', 'kb_store_dir', 'shared_kb', 'global_kb']) {
    for (const alias of [false, true]) await t.test(`${key}:${alias ? 'symlink' : 'equal'}`, async t => {
      const f = await fixture(t), [first, second] = await Promise.all(f.lanes.map(lane => json(lane.args_file)));
      const owned = first[key] || join(f.root, 'geak', key);
      await mkdir(owned, {recursive: true});
      first[key] = owned;
      if (alias) await symlink(owned, join(f.root, 'alias'));
      second[key] = alias ? join(f.root, 'alias') : owned;
      for (const [i, args] of [first, second].entries()) await writeFile(f.lanes[i].args_file, JSON.stringify(args));
      await assert.rejects(composeKernel(f.manifest), /overlaps another owner's private path/);
      await assert.rejects(readFile(join(f.root, 'events.jsonl')), {code: 'ENOENT'});
    });
  }
});

test('remote knowledge defaults are rejected and explicit private roots share workflow code', async t => {
  for (const remote of [undefined, '', 'auto']) await t.test(String(remote), async t => {
    const f = await fixture(t), lane = f.lanes[0], args = await json(lane.args_file);
    args.kb_remote = remote;
    await writeFile(lane.args_file, JSON.stringify(args));
    await assert.rejects(composeKernel(f.manifest), /requires kb_remote=off/);
    await assert.rejects(readFile(join(f.root, 'events.jsonl')), {code: 'ENOENT'});
  });
  await t.test('private nested roots and normalized off', async t => {
    const f = await fixture(t);
    for (const lane of f.lanes) {
      const args = await json(lane.args_file);
      for (const key of ['learned_dir', 'kb_artifacts_dir', 'kb_store_dir', 'shared_kb', 'global_kb'])
        args[key] = join(args.exp_root, key);
      args.kb_remote = ' OFF ';
      await writeFile(lane.args_file, JSON.stringify(args));
    }
    const result = await composeKernel(f.manifest);
    assert.equal(result.comparison_complete, true);
    const records = await Promise.all(result.children.map(child => json(join(child.directory, 'args.json'))));
    assert.equal(records[0].workflow_dir, records[1].workflow_dir, 'workflow source remains shared');
    const binding = await json(join(f.manifest.out_dir, 'binding.json'));
    for (const lane of f.lanes) for (const key of ['learned_dir', 'kb_artifacts_dir', 'kb_store_dir', 'shared_kb', 'global_kb'])
      assert.ok(binding.private_paths.some(path => path.owner === lane.id && path.label === key));
  });
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

// Test-only complete-workflow child. The production dispatcher, lane, runtime,
// deadline primitives, source capture and native readers execute unchanged.
const joinedRuntime = String.raw`
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, cp } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { createHash } from 'node:crypto';
import childProcess from 'node:child_process';
import { syncBuiltinESMExports } from 'node:module';
const argv = process.argv.slice(2), value = key => argv[argv.indexOf(key) + 1];
const args = JSON.parse(await readFile(value('--args-file'), 'utf8'));
const scenario = process.env.TEST_JOINED_ENDING;
const raw = { scenario, args, calls: [], tools: [], logs: [], clock: [], effects: [] };
const originalSpawn = childProcess.spawn;
// Observe actual host commands without changing arguments, streams or timers.
childProcess.spawn = function(cmd, commandArgs, options) {
  const row = {cmd, args: commandArgs, cwd: options.cwd, stdin: '', stdout: '', stderr: '', code: null, signal: null};
  raw.tools.push(row);
  const child = originalSpawn.call(this, cmd, commandArgs, options);
  const write = child.stdin.write.bind(child.stdin), end = child.stdin.end.bind(child.stdin);
  let insideEnd = false;
  child.stdin.write = (input, ...rest) => {
    if (!insideEnd) row.stdin += input.toString();
    return write(input, ...rest);
  };
  child.stdin.end = (input, ...rest) => {
    if (typeof input === 'string' || ArrayBuffer.isView(input)) row.stdin += input.toString();
    insideEnd = true;
    try { return end(input, ...rest); }
    finally { insideEnd = false; }
  };
  child.stdout.on('data', data => { row.stdout += data.toString(); });
  child.stderr.on('data', data => { row.stderr += data.toString(); });
  child.on('close', (code, signal) => { row.code = code; row.signal = signal; });
  return child;
};
syncBuiltinESMExports();
const { createRuntime } = await import(process.env.TEST_RUNTIME_MODULE);
const { runProcess } = await import(process.env.TEST_PROCESS_MODULE);
const realNow = Date.now, realHrtime = process.hrtime.bigint;
const wallStart = realNow(), epoch = (args.deadline_epoch - 600) * 1000;
let logicalMs = 0;
Date.now = () => epoch + logicalMs;
process.hrtime.bigint = () => BigInt(logicalMs) * 1000000n;
const advance = (label, next = logicalMs + 25) => {
  assert.ok(next >= logicalMs);
  raw.clock.push({label, from_ms: logicalMs, to_ms: next}); logicalMs = next;
};
// This watchdog and all runtime process/grace timers retain ordinary real time.
const watchdog = setTimeout(() => process.exit(98), 20000);
const hash = value => createHash('sha256').update(value).digest('hex');
const field = (prompt, key) => {
  const line = prompt.split('\n').find(line => line.startsWith('- ' + key + ': '));
  assert.ok(line, 'missing fixture input ' + key);
  const value = line.slice(key.length + 4);
  try { return JSON.parse(value); } catch { return value; }
};
const E = args.eval_dir, W = join(E, 'workspace');
const gitEnv = {
  GIT_AUTHOR_NAME: 'GEAK fixture', GIT_AUTHOR_EMAIL: 'fixture@example.com',
  GIT_COMMITTER_NAME: 'GEAK fixture', GIT_COMMITTER_EMAIL: 'fixture@example.com',
  GIT_AUTHOR_DATE: '2026-01-01T00:00:00Z', GIT_COMMITTER_DATE: '2026-01-01T00:00:00Z',
};
const tool = async (cmd, commandArgs, extra = {}) => {
  const result = await runProcess({cmd, args: commandArgs, timeoutMs: 10000, ...extra});
  assert.equal(result.status, 'completed'); assert.equal(result.code, 0, result.stderr);
  assert.equal(result.group_cleared, true); assert.equal(result.strays_killed, false);
  return result.stdout;
};
const git = (path, ...commandArgs) => tool('git', ['-C', path, ...commandArgs], {env: gitEnv});
const write = async (path, text) => {
  await mkdir(dirname(path), {recursive: true}); await writeFile(path, text);
  raw.effects.push({kind: 'write', path, sha256: hash(text)});
};
const native = async workspace => {
  const cmd = args.external_measurement_command;
  const text = await tool(cmd[0], [...cmd.slice(1), '--workspace', workspace]);
  return JSON.parse(text.trim().split('\n').at(-1));
};
const stop = {stop: true, directions: [], reasoning: 'fixture accepted stop'};
const profile = {bottleneck: 'memory', top_opportunities: [], summary_path: join(E, 'profile.json')};
const expected = ['director:setup', 'tech_lead:analyze', 'benchmark_engineer', 'profile_engineer:baseline',
  'warm_start:resolve', 'tech_lead:plan r1',
  ...(scenario === 'improve' ? ['eng r1_d1:memory', 'verify r1_d1', 'commit r1', 'reprofile r1',
    'tech_lead:memory r1', 'storage:reclaim r1'] : scenario === 'self_stop'
    ? ['tech_lead:replan r1#1', 'tech_lead:replan r1#2'] : []),
  'tech_lead:report', 'director:validate',
  ...(scenario === 'improve' ? ['update_experience', 'kb:write'] : []), 'file_writer:persist_llm_stats'];
const failures = [];
let rt;
try {
  rt = createRuntime({concurrency: Number(value('--concurrency')), agentTimeoutMs: Number(value('--agent-timeout-ms')),
    schemaRetries: 0, backendEventsFile: value('--backend-events-file'), log: message => raw.logs.push(message),
    backend: {name: 'joined-fixture', async runAgent(options) {
      const row = {options, returned: null}; raw.calls.push(row);
      try {
        assert.equal(options.label, expected[raw.calls.length - 1], 'unexpected or missing response step');
        assert.ok(!options.label.startsWith('clock '), 'process-bounded mode must use the runtime clock');
        const {label, prompt} = options;
        let result;
        if (label === 'director:setup') {
          await cp(args.kernel_path, W, {recursive: true});
          await git(W, 'init', '-q'); await git(W, 'add', '.'); await git(W, 'commit', '-q', '-m', 'ORIGINAL');
          raw.initial_commit = (await git(W, 'rev-parse', 'HEAD')).trim();
          result = {eval_dir: E, workspace: W, kernel_name: 'fixture', device_gfx: 'gfx950',
            device_target: 'mi355x', physical_cu_count: 256, baseline_frozen: true, source_files: ['kernel_src/module.py']};
        } else if (label === 'tech_lead:analyze') {
          result = {kernel_type: 'gemm', roadmap_summary: 'fixture question', modifiable_files: ['kernel_src/module.py']};
        } else if (label === 'benchmark_engineer') {
          const measured = await native(W);
          await write(join(E, 'COMMANDMENT.md'), 'Synthetic full-task native contract.\n');
          result = {commandment_path: join(E, 'COMMANDMENT.md'), baseline_per_case: measured.per_case,
            external_evaluation: measured.external_evaluation, reliable: true};
        } else if (label === 'profile_engineer:baseline' || label === 'reprofile r1') {
          await write(profile.summary_path, JSON.stringify(profile)); result = profile;
        } else if (label === 'warm_start:resolve') {
          const path = join(args.kb_store_dir, 'initial.json');
          const bytes = await readFile(path); raw.effects.push({kind: 'read', path, sha256: hash(bytes)});
          result = JSON.parse(bytes);
        } else if (label === 'tech_lead:plan r1') {
          const path = join(args.workflow_dir, 'knowledge/learned/INDEX.md');
          const bytes = await readFile(path); raw.effects.push({kind: 'read', path, sha256: hash(bytes)});
          assert.equal(field(prompt, 'LEARNED_KB'), 'on');
          result = scenario === 'improve' ? {stop: false, directions: [{id: 'r1_d1', title: 'fixture source change',
            specialty: 'memory', focus_files: ['kernel_src/module.py'], prompt: 'synthetic test mutation', learned_refs: []}]} : stop;
        } else if (label.startsWith('tech_lead:replan ')) result = stop;
        else if (label === 'eng r1_d1:memory') {
          const workspace = field(prompt, 'KERNEL_PATH'), output = field(prompt, 'OUTPUT_DIR');
          await cp(W, workspace, {recursive: true});
          await write(join(workspace, 'kernel_src/module.py'), '# synthetic geak factor=1.4\n');
          const patch = await git(workspace, 'diff', '--binary');
          await write(join(output, 'best_patch.diff'), patch);
          const measured = await native(workspace);
          result = {status: 'ok', measurement_valid: true, external_evaluation: measured.external_evaluation};
        } else if (label === 'verify r1_d1') {
          const workspace = join(field(prompt, 'VERIFY_DIR'), 'workspace');
          await cp(W, workspace, {recursive: true}); await git(workspace, 'apply', field(prompt, 'PATCH'));
          const measured = await native(workspace);
          result = {status: 'verified', correctness: 'pass', external_evaluation: measured.external_evaluation};
        } else if (label === 'commit r1') {
          await git(W, 'apply', join(E, 'round_1/engineer_0/best_patch.diff'));
          await git(W, 'add', '.'); await git(W, 'commit', '-q', '-m', 'fixture accepted round');
          await write(join(E, 'current_best.diff'), await git(W, 'diff', 'HEAD~1', 'HEAD'));
          result = {committed: true, current_best_diff: join(E, 'current_best.diff')};
        } else if (label === 'tech_lead:memory r1') {
          result = {insights: ['fixture source was measured'], ledger: [{question: 'fixture', outcome: 'measured'}]};
          await write(join(E, 'memory.json'), JSON.stringify(result));
        } else if (label === 'storage:reclaim r1') result = {ok: true, note: 'no heavy fixture artifacts'};
        else if (label === 'tech_lead:report') {
          const history = field(prompt, 'HISTORY');
          result = {report_path: join(E, 'report.json'), final_patch: join(E, 'host_final_patch.diff'), rounds: history.rounds.length};
          await write(result.report_path, JSON.stringify({history, result}));
        } else if (label === 'director:validate') {
          result = {validation_status: scenario === 'cutoff_veto' ? 'flagged' : 'accepted',
            correctness: 'pass', timing_basis: 'synthetic'};
        } else if (label === 'update_experience') {
          const learned = field(prompt, 'LEARNED_DIR');
          for (const name of ['README.md', 'INDEX.md']) {
            const path = join(learned, name), bytes = await readFile(path);
            raw.effects.push({kind: 'read', path, sha256: hash(bytes)});
          }
          const path = join(learned, 'fixture-card.md');
          await write(path, 'Synthetic qualified source evidence.\n'); result = {action: 'created', card_path: path};
        } else if (label === 'kb:write') {
          const path = join(args.kb_store_dir, 'written.json');
          await write(path, JSON.stringify({source_commit: (await git(W, 'rev-parse', 'HEAD')).trim()}));
          result = {written: true, slug: 'fixture', dir: args.kb_store_dir};
        } else if (label === 'file_writer:persist_llm_stats') {
          await write(join(E, 'accounting-prompt.txt'), prompt); result = {written: true, ok: true, path: E};
        } else throw new Error('unhandled fixture step ' + label);
        row.returned = result;
        advance(label, label === 'tech_lead:plan r1' && scenario === 'cutoff_veto' ? 420000 : undefined);
        return {text: JSON.stringify(result)};
      } catch (error) { failures.push(String(error.stack || error)); throw error; }
    }},
  });
  raw.result = await rt.runScript(argv[0], args, 0);
  raw.runtime_state = rt.state;
  raw.final_commit = (await git(W, 'rev-parse', 'HEAD')).trim();
  raw.final_patch = await readFile(join(E, 'host_final_patch.diff'), 'utf8');
  raw.recorder = (await readFile(value('--backend-events-file'), 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(failures, []);
  assert.deepEqual(raw.calls.map(row => row.options.label), expected);
  assert.equal(rt.state.schemaFails, 0);
  assert.equal(rt.state.backendTrace.open_calls, 0); assert.equal(rt.state.backendTrace.write_error, null);
  assert.equal(raw.recorder.length, 2 * expected.length);
  await writeFile(value('--result-file'), JSON.stringify(raw.result));
  await writeFile(value('--metrics-file'), JSON.stringify({ok: true, ...rt.state}));
} catch (error) { raw.error = String(error.stack || error); process.exitCode = 1; }
finally {
  raw.fixture_failures = failures;
  await writeFile(join(args.exp_root, 'child_trace.json'), JSON.stringify(raw, null, 2));
  Date.now = realNow; process.hrtime.bigint = realHrtime;
  clearTimeout(watchdog);
  await writeFile(join(args.exp_root, 'real_wall.json'), JSON.stringify({elapsed_ms: realNow() - wallStart}));
}
`;

test('joined one-member parity executes the real process-bounded workflow through its postlude', async t => {
  for (const ending of ['improve', 'self_stop', 'cutoff_veto']) await t.test(ending, async t => {
    const f = await fixture(t, {joined_record_name: ending});
    const lane = f.lanes[0], privateRoot = join(f.root, 'geak');
    f.manifest.lanes = [lane]; lane.id = 'reference';
    const runtime = join(f.root, 'joined_runtime.mjs'); await writeFile(runtime, joinedRuntime);
    lane.runtime_command = [process.execPath, runtime];
    lane.runtime_options = {concurrency: 2, agent_timeout_ms: 600000,
      backend_events_file: join(privateRoot, 'backend_calls.jsonl')};
    lane.timeout_ms = 660000;
    const args = {...await json(lane.args_file), workflow_dir: join(privateRoot, 'kernel_workflow'),
      deadline_epoch: Math.floor(Date.now() / 1000) + 600, search_reserve_s: 180,
      final_validation_reserve_s: 120, final_cleanup_reserve_s: 60,
      agent_timeout_ms: 600000, agent_retries: 1, budget: ending === 'improve' ? 1 : 6,
      max_forced_replans: 2, no_stop_s: 10, warm_start: 'on', use_learned_kb: 'true',
      kb_remote: 'off', kb_store_dir: join(privateRoot, 'kb_store'), kb_artifacts_dir: join(privateRoot, 'kb_artifacts')};
    f.manifest.deadline_epoch = args.deadline_epoch;
    lane.env = {...lane.env, TEST_JOINED_ENDING: ending, GEAK_LIVE_TRACE: '0',
      TEST_RUNTIME_MODULE: new URL('./run_workflow.mjs', import.meta.url).href,
      TEST_PROCESS_MODULE: new URL('./backends/base.mjs', import.meta.url).href,
      TEST_NATIVE_ID_COUNTER: join(privateRoot, 'native_counter')};
    const reset = async () => {
      await rm(privateRoot, {recursive: true, force: true});
      await mkdir(args.exp_root, {recursive: true});
      await mkdir(join(args.workflow_dir, 'knowledge/learned'), {recursive: true});
      await cp(join(REPO, 'kernel_workflow/kernel_lane.js'), join(args.workflow_dir, 'kernel_lane.js'));
      await writeFile(join(args.workflow_dir, 'knowledge/learned/INDEX.md'), 'Synthetic empty learned index.\n');
      await mkdir(args.learned_dir, {recursive: true});
      await cp(join(args.workflow_dir, 'knowledge/learned/INDEX.md'), join(args.learned_dir, 'INDEX.md'));
      await writeFile(join(args.learned_dir, 'README.md'), 'Synthetic private learned baseline.\n');
      await mkdir(args.kb_store_dir, {recursive: true});
      await writeFile(join(args.kb_store_dir, 'initial.json'), JSON.stringify({read_reason: 'empty', candidates: []}));
      await writeFile(lane.args_file, JSON.stringify(args));
    };
    await reset();
    const direct = await runProcess({cmd: process.execPath, args: [runtime, join(REPO, 'kernel_workflow/kernel_workflow.js'),
      ...Object.entries(lane.runtime_options).flatMap(([key, value]) => ['--' + key.replaceAll('_', '-'), String(value)]),
      '--args-file', lane.args_file, '--result-file', join(privateRoot, 'direct_result.json'),
      '--metrics-file', join(privateRoot, 'direct_metrics.json')], cwd: lane.cwd, env: lane.env, timeoutMs: 25000});
    await writeFile(join(f.root, 'direct_process.json'), JSON.stringify(direct));
    const first = await json(join(args.exp_root, 'child_trace.json'));
    assert.equal(direct.code, 0, first.error || direct.stderr);
    await cp(privateRoot, join(f.root, 'direct_saved'), {recursive: true});
    await reset();
    const result = await composeKernel(f.manifest);
    const second = await json(join(args.exp_root, 'child_trace.json'));
    // Same absolute private paths and deterministic opaque IDs: no field scrub.
    assert.deepEqual(second, first);
    assert.equal(result.children.length, 1);
    assert.deepEqual(await json(result.children[0].result_file), first.result);
    assert.equal(first.result.forced_replans, ending === 'self_stop' ? 2 : 0);
    assert.equal(first.result.stopped_by, ending === 'improve' ? 'budget' : ending === 'self_stop' ? 'tech_lead_stop' : 'deadline');
    assert.equal(first.result.deadline_hit, ending === 'cutoff_veto');
    assert.equal(first.final_commit === first.initial_commit, ending !== 'improve');
    assert.equal(first.final_patch.length > 0, ending === 'improve');
    if (ending === 'improve') {
      const patchWrites = first.tools.filter(row => row.cmd === 'sh' && row.args[1] === 'cat > "$1"'
        && row.args.at(-1) === join(args.eval_dir, 'host_final_patch.diff'));
      assert.ok(patchWrites.length > 0, 'the postlude writes the retained final patch through host stdin');
      for (const row of patchWrites) assert.equal(row.stdin, first.final_patch);
      assert.equal(first.result.learned_card.card_path, join(args.learned_dir, 'fixture-card.md'));
      for (const name of ['README.md', 'INDEX.md'])
        assert.ok(first.effects.some(row => row.kind === 'read' && row.path === join(args.learned_dir, name)));
    }
    assert.equal(first.result.warm_start.read_reason, 'empty');
    assert.equal(first.result.warm_start.adopted, false);
    assert.equal(!!first.result.kb_written && !!first.result.learned_card, ending === 'improve');
    assert.equal(first.recorder.filter(row => row.event === 'start').length, first.calls.length);
    const call = label => first.calls.find(row => row.options.label === label).options;
    if (ending === 'cutoff_veto') {
      assert.equal(call('tech_lead:report').timeoutMs, 53000, 'Report uses its original reserved cutoff');
      assert.equal(call('director:validate').timeoutMs, 112975, 'validation uses its original reserved cutoff');
    }
    assert.ok(first.calls.some(row => row.options.label === 'file_writer:persist_llm_stats'));
    if (ending === 'cutoff_veto') {
      assert.equal(result.required_finalist.qualified, false);
      assert.equal(result.children[0].status, 'failed');
      assert.equal(result.children[0].common_evidence, undefined, 'Director veto cannot enter common selection');
    } else {
      const child = result.children[0];
      assert.equal(result.required_finalist.qualified, true);
      assert.equal(child.common_selection.source_id, first.result.final_selection.source_id);
      assert.notEqual(child.common_selection.candidate_id, first.result.final_selection.candidate_id);
      assert.notEqual(child.common_selection.evaluation_id, first.result.final_selection.evaluation_id);
      assert.equal(child.status, ending === 'improve' ? 'qualified' : 'no_improvement');
    }
    await writeFile(join(f.root, 'joined_parity_result.json'), JSON.stringify({ending, raw_child_equal: true,
      role_calls: first.calls.length, host_commands: first.tools.length, source_commit: first.final_commit,
      child_final_selection: first.result.final_selection, composed: result}, null, 2));
  });
});
