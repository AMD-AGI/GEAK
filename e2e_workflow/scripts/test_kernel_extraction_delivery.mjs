import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { loadContract, checkDelivery, attachVerdict, hydrateExtractionVerdicts } from './kernel_extraction_delivery.mjs';
import { createRuntime } from '../../interface/runtime/engine/run_workflow.mjs';

const contract = await loadContract();
const cli = fileURLToPath(new URL('./kernel_extraction_delivery.mjs', import.meta.url));

async function fixture(t) {
  const task = await fs.mkdtemp(path.join(os.tmpdir(), 'extraction-delivery-'));
  t.after(() => fs.rm(task, { recursive: true, force: true }));
  const phases = ['DECODE', 'PREFILL'].map(regime => {
    const target = `pkg.indexer:${regime.toLowerCase()}`;
    const kernels = [`${regime}_score`, `${regime}_topk`];
    const seams = [{ target_callable: target, role: 'op_seam', depth: 1, device_kernels: kernels }];
    return {
      spec: { regime, device_kernel: kernels[0], required_device_kernels: kernels, seam_candidates: seams },
      row: { regime, target_callable: target, seam_candidates: seams, selection_validation: {
        contract: 'kernel_selection', ok: true, target_callable: target, device_kernel: kernels[0],
        selection_scope: 'joint_chain', required_device_kernels: kernels,
        total_calls_observed: 2, target_marker_calls: 2, matched_kernel_calls: 2,
        candidate_targets_tested: [target], live_candidate_targets: [target],
        deeper_live_candidates: [], deepest_verified: true, failed: [],
        chain_members: kernels.map(device_kernel => ({ device_kernel, matched_kernel_calls: 2 })),
        joint_call_coverage: [0, 1].map(call => ({ call, missing_members: [] })),
      } },
    };
  });
  const kernel = { entity_kind: 'gpu_kernel', device_kernel: phases[0].spec.device_kernel,
    op_kind: 'attn', joint_ut: { phases: phases.map(p => p.spec) } };
  const result = { ...phases[0].row, task_dir: task, editable: true, baseline_frozen: true,
    op_kind: 'attn', smoke: 'pass', unittest_smoke: 'pass',
    phase_selection_validations: phases.map(p => p.row) };
  for (const p of phases) await fs.writeFile(path.join(task, `${p.spec.regime}.json`), JSON.stringify(p.row.selection_validation));
  const pack = async (value = result) => {
    let out = await attachVerdict(value, 'DECODE.json');
    for (const p of phases) out = await attachVerdict(out, `${p.spec.regime}.json`, p.spec.regime);
    return out;
  };
  return { task, kernel, result, phases, pack };
}

test('original inline returns and hashed raw verdicts use the same production gate', async t => {
  const f = await fixture(t);
  const original = JSON.stringify(f.result);
  for (const phase of ['extract', 'extract_op']) {
    assert.equal((await checkDelivery(f.kernel, f.result, phase, contract)).ok, true);
    const packed = await f.pack();
    assert(!packed.selection_validation);
    assert(packed.phase_selection_validations.every(p => p.selection_validation_ref));
    assert.equal((await checkDelivery(f.kernel, packed, phase, contract)).ok, true);
    const hydrated = await hydrateExtractionVerdicts(packed);
    assert.deepEqual(hydrated.selection_validation, f.result.selection_validation);
    assert.deepEqual(hydrated.phase_selection_validations[0].selection_validation, f.result.selection_validation);
  }
  assert.equal(JSON.stringify(f.result), original, 'packing cannot mutate agent input or verdicts');
});

test('reproduce abbreviated indexer phase return; raw file transport repairs delivery only', async t => {
  const f = await fixture(t);
  const broken = structuredClone(f.result);
  broken.phase_selection_validations[0].selection_validation = {
    _same_as: 'top-level selection_validation', ok: true, selection_scope: 'joint_chain',
  };
  assert.equal((await checkDelivery(f.kernel, broken, 'extract_op', contract)).ok, false);
  const repaired = await f.pack(broken);
  assert.equal((await checkDelivery(f.kernel, repaired, 'extract_op', contract)).ok, true);
  assert.equal(repaired.smoke, broken.smoke);
});

test('file references do not conceal failed or incomplete execution evidence', async t => {
  const f = await fixture(t);
  for (const mutate of [
    v => { v.ok = false; v.failed = ['joint_member_missing_in_call']; },
    v => { v.joint_call_coverage[1].missing_members = ['DECODE_topk']; },
    v => { v.chain_members[1].matched_kernel_calls = 0; },
    v => { v.device_kernel = 'another_kernel'; },
    v => { v.candidate_targets_tested = []; },
    v => { v.deeper_live_candidates = ['pkg:deeper_whole_chain']; },
  ]) {
    const raw = structuredClone(f.result.selection_validation); mutate(raw);
    await fs.writeFile(path.join(f.task, 'DECODE.json'), JSON.stringify(raw));
    const packed = await f.pack();
    // The raw file is authoritative even if an inline claim was also supplied.
    packed.phase_selection_validations[0].selection_validation = f.result.selection_validation;
    assert.equal((await checkDelivery(f.kernel, packed, 'extract_op', contract)).ok, false);
  }
});

test('missing joint input declaration remains a selector error, not auto-repaired from return', async t => {
  const f = await fixture(t);
  const kernel = { ...f.kernel, joint_ut: undefined, seam_candidates: f.phases[0].spec.seam_candidates };
  const result = await f.pack();
  result.joint_ut = f.kernel.joint_ut;
  const report = await checkDelivery(kernel, result, 'extract', contract);
  assert.equal(report.ok, false);
  assert.match(report.checks.selection.why, /requires a declared joint UT boundary/);
});

test('alternative dispatch branches verify one whole-op entry without simultaneous membership', async t => {
  const f = await fixture(t), target = 'pkg:whole_op';
  const phases = f.phases.map((p, i) => ({ regime: `DECODE_branch_${i}`, device_kernel: p.spec.device_kernel,
    seam_candidates: [{ target_callable: target, role: 'op_seam', depth: 0, device_kernels: [p.spec.device_kernel] }] }));
  const rows = phases.map(p => ({ regime: p.regime, target_callable: target, seam_candidates: p.seam_candidates,
    selection_validation: { contract: 'kernel_selection', ok: true, target_callable: target,
      device_kernel: p.device_kernel, total_calls_observed: 1, target_marker_calls: 1, matched_kernel_calls: 1,
      candidate_targets_tested: [target], live_candidate_targets: [target], deepest_verified: true,
      deeper_live_candidates: [] } }));
  const kernel = { op_kind: 'moe', entity_kind: 'gpu_kernel', device_kernel: phases[0].device_kernel,
    selected_device_kernels: phases.map(p => p.device_kernel), dispatch_ut: { phases } };
  const result = { ...f.result, target_callable: target, phase_selection_validations: rows };
  assert.equal((await checkDelivery(kernel, result, 'extract', contract)).ok, true);
  for (const mutate of [
    r => { r.phase_selection_validations.pop(); },
    r => { r.phase_selection_validations[1].selection_validation.device_kernel = 'wrong'; },
    r => { r.phase_selection_validations[1].target_callable = 'pkg:gemm_leaf'; },
    r => { r.phase_selection_validations[1].selection_validation.matched_kernel_calls = 0; },
    r => { r.selection_validation_ref = { path: 'missing.json', sha256: '0'.repeat(64) }; },
  ]) {
    const bad = structuredClone(result); mutate(bad);
    assert.equal((await checkDelivery(kernel, bad, 'extract', contract)).ok, false);
  }
});

test('file integrity errors fail closed and cannot fall back to inline PASS', async t => {
  const f = await fixture(t);
  const packed = await f.pack();
  for (const mutate of [
    r => { r.selection_validation_ref.sha256 = '0'.repeat(64); },
    r => { r.selection_validation_ref.path = 'missing.json'; },
    r => { r.selection_validation_ref.sha256 = ''; },
    r => { r.phase_selection_validations[1].selection_validation_ref.path = 'missing.json'; },
  ]) {
    const bad = structuredClone(packed); mutate(bad);
    bad.selection_validation = f.result.selection_validation;
    const report = await checkDelivery(f.kernel, bad, 'extract_op', contract);
    assert.equal(report.ok, false);
    assert.match(report.checks.selection.why, /selection evidence file invalid/);
  }
  await fs.appendFile(path.join(f.task, 'DECODE.json'), ' ');
  assert.match((await checkDelivery(f.kernel, packed, 'extract_op', contract)).checks.selection.why, /SHA-256 mismatch/);
});

test('a verdict symlink outside the task is rejected', async t => {
  const f = await fixture(t);
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), 'other-task-'));
  t.after(() => fs.rm(outside, { recursive: true, force: true }));
  await fs.writeFile(path.join(outside, 'verdict.json'), JSON.stringify(f.result.selection_validation));
  await fs.symlink(path.join(outside, 'verdict.json'), path.join(f.task, 'escaped.json'));
  await assert.rejects(attachVerdict(f.result, 'escaped.json'), /inside task_dir/);
  const packed = await f.pack(); packed.selection_validation_ref.path = 'escaped.json';
  assert.equal((await checkDelivery(f.kernel, packed, 'extract_op', contract)).ok, false);
});

test('schema, baseline declaration and UT status still apply after selection succeeds', async t => {
  const f = await fixture(t);
  for (const mutate of [
    r => { r.smoke = 'PASS'; },
    r => { r.smoke = 'fail'; },
    r => { r.baseline_frozen = false; },
    r => { r.editable = false; },
    r => { r.smoke_detail = { ok: true }; },
    r => { r.phase_selection_validations.pop(); },
    r => { r.phase_selection_validations[1].regime = 'DECODE'; },
  ]) {
    const bad = await f.pack(); mutate(bad);
    assert.equal((await checkDelivery(f.kernel, bad, 'extract_op', contract)).ok, false);
  }
});

test('CLI writes only a passing delivery; a failed draft preserves the existing output', async t => {
  const f = await fixture(t);
  await fs.writeFile(path.join(f.task, 'KERNEL.json'), JSON.stringify(f.kernel));
  await fs.writeFile(path.join(f.task, 'draft.json'), JSON.stringify(f.result));
  const args = [cli, '--kernel', path.join(f.task, 'KERNEL.json'), '--result', path.join(f.task, 'draft.json'),
    '--phase', 'extract_op', '--verdict', 'DECODE.json', '--phase-verdict', 'DECODE=DECODE.json',
    '--phase-verdict', 'PREFILL=PREFILL.json', '--out', path.join(f.task, 'RETURN.json')];
  const good = spawnSync(process.execPath, args, { encoding: 'utf8' });
  assert.equal(good.status, 0, good.stdout + good.stderr);
  const saved = await fs.readFile(path.join(f.task, 'RETURN.json'), 'utf8');
  const bad = { ...f.result, smoke: 'fail' };
  await fs.writeFile(path.join(f.task, 'draft.json'), JSON.stringify(bad));
  const failed = spawnSync(process.execPath, args, { encoding: 'utf8' });
  assert.equal(failed.status, 1);
  assert.equal(await fs.readFile(path.join(f.task, 'RETURN.json'), 'utf8'), saved);
});

test('runtime preserves legacy file verdicts and refuses unprepared AgentX dispatch', async t => {
  const f = await fixture(t);
  const kernel = { entity_kind: 'gpu_kernel', device_kernel: 'leaf_kernel',
    seam_candidates: [{ target_callable: 'pkg:leaf', role: 'inner_launcher', depth: 1,
      device_kernels: ['leaf_kernel'] }] };
  const verdict = { contract: 'kernel_selection', ok: true, target_callable: 'pkg:leaf',
    device_kernel: 'leaf_kernel', total_calls_observed: 1, target_marker_calls: 1,
    matched_kernel_calls: 1, deepest_verified: true, candidate_targets_tested: ['pkg:leaf'],
    live_candidate_targets: ['pkg:leaf'], deeper_live_candidates: [] };
  await fs.writeFile(path.join(f.task, 'leaf.json'), JSON.stringify(verdict));
  const result = await attachVerdict({ editable: true, task_dir: f.task, unittest_smoke: 'pass',
    baseline_frozen: true, target_callable: 'pkg:leaf' }, 'leaf.json');
  const workflow = await fs.readFile(new URL('../e2e_workflow.js', import.meta.url), 'utf8');
  // Start at preparedUtDefect: the AgentX branch of extractWithBaseline reuses the prepared UT
  // through it, and both are plain functions in the workflow source -- the sandbox has no
  // filesystem for the dispatcher to import a module from.
  const start = workflow.indexOf('function preparedUtDefect(');
  const extraction = workflow.slice(start, workflow.indexOf('\n// abDone ==', start));
  const script = path.join(f.task, 'driver.js');
  await fs.writeFile(script, `
    const {loadContract} = await import(${JSON.stringify(new URL('./kernel_extraction_delivery.mjs', import.meta.url).href)});
    const {guardUnhydratedVerdicts, kernelSelectionVerified, hasFrozenBaseline, EXTRACT_SCHEMA} = await loadContract();
    const CAPTURE_STORAGE_ENV = '', CAPTURE_BYTE_BUDGET = 1, CAPTURE_CASE_BYTE_LIMIT = 1;
    const CAPTURE_PERSIST_POLICY = '', BASELINE_EXTRACT_RETRIES = 0, KERNEL_EXTRACT_ON = !!args.agentx;
    const WORKFLOW_DIR = ${JSON.stringify(fileURLToPath(new URL('..', import.meta.url)))};
    const roleAgent = () => 'fixture';
    const safeAgent = async () => agent('fixture', {schema: EXTRACT_SCHEMA});
    const withDiskSelection = async ext => ext;
    ${extraction}
    return extractWithBaseline('kernel_extractor', 'extract', '', {KERNEL: args.kernel});
  `);
  const logs = [];
  let agentCalls = 0;
  // Hydration is the EXTRACTOR's preflight, not the workflow's: resolving a
  // selection_validation_ref means reading and hashing a file, and the workflow sandbox has no
  // filesystem. So the agent returns what kernel_extraction_delivery.mjs would have produced --
  // recomputed per call, so a verdict file corrupted mid-test is still observed.
  const runtime = createRuntime({ backend: { name: 'fixture',
    runAgent: async () => { agentCalls++;
      return { text: JSON.stringify(await hydrateExtractionVerdicts(result)) }; } },
    log: line => logs.push(line) });
  const accepted = await runtime.runScript(script, { kernel }, 0);
  assert.equal(accepted.unittest_smoke, 'pass');
  assert.deepEqual(accepted.selection_validation, verdict);
  // A head that demands source evidence can only get it from the AgentX preparation stage, where
  // verifySourceDelivery runs with a filesystem. Demanding it on the legacy path -- which has no
  // way to produce it, the workflow sandbox having no filesystem -- fails closed rather than
  // letting the requirement lapse. The REAL verification is covered in test_kernel_source_delivery
  // and test_agentx_standalone_handoff; this only pins that the requirement is never ignored.
  const missingSource = await runtime.runScript(script, { kernel: { ...kernel, source_delivery_required: true } }, 0);
  assert.equal(missingSource.unittest_smoke, 'fail');
  assert.equal(missingSource.source_delivery_failed, true);
  assert.match(missingSource.smoke_detail, /outside the AgentX preparation stage/);
  await fs.appendFile(path.join(f.task, 'leaf.json'), ' ');
  const rejected = await runtime.runScript(script, { kernel }, 0);
  assert.equal(rejected.unittest_smoke, 'fail');
  assert.match(rejected.smoke_detail, /SHA-256 mismatch/);
  const before = agentCalls;
  const unprepared = await runtime.runScript(script, {kernel, agentx: true}, 0);
  assert.equal(unprepared.unittest_smoke, 'fail');
  assert.match(unprepared.smoke_detail, /missing prepared UT/);
  assert.equal(agentCalls, before, 'AgentX cannot fall back to legacy UT authoring');
});
