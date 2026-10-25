#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Opt-in composition of complete existing native GEAK/EVO runs. The caller owns
// isolated execution contexts and device grants; this is not a resource scheduler.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, rename, realpath, lstat } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runProcess } from './backends/base.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const WORKFLOW = join(ROOT, 'kernel_workflow/kernel_workflow.js');
const MATERIALIZE = join(ROOT, 'kernel_workflow/scripts/materialize_workspace.sh');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const canonical = value => JSON.stringify(Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b))));
const candidateId = (job, epoch, source) => sha(canonical({ kind: 'candidate', job_id: job,
  measurement_epoch_id: epoch, source_id: source, parents: [], donors: [], contribution_claims: [] }));
const need = (condition, message) => { if (!condition) throw new Error(message); };
const hex = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const argv = value => Array.isArray(value) && value.length > 0 && value.every(x => typeof x === 'string' && x && !x.includes('\0'));
const clean = p => p.status === 'completed' && p.code === 0 && p.group_cleared && !p.stdio_held && !p.strays_killed;
const positive = n => Number.isFinite(n) && n > 0;
const json = async path => JSON.parse(await readFile(path, 'utf8'));
// Resolve existing ancestors too: fresh output directories need not exist yet,
// but spelling the same parent through a symlink must not create a second owner.
async function resolvedPath(path) {
  need(isAbsolute(path || ''), 'private paths must be absolute');
  try { return await realpath(path); }
  catch (error) {
    if (error.code !== 'ENOENT') throw error;
    // A fresh component is allowed; an existing link to a missing target is not
    // a distinct lexical path and could alias another owner once created.
    const entry = await lstat(path).catch(error => {
      if (error.code !== 'ENOENT') throw error;
      return null;
    });
    need(!entry?.isSymbolicLink(), 'private path has an unresolved symbolic link');
    return join(await resolvedPath(dirname(path)), basename(path));
  }
}
const contains = (parent, path) => parent === path || path.startsWith(parent.endsWith(sep) ? parent : parent + sep);
const overlaps = (a, b) => contains(a, b) || contains(b, a);
async function save(path, value) {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2) + '\n');
  await rename(temporary, path);
}
function option(command, name) {
  need(!command.some(part => part.startsWith(name + '=')), `use the separate ${name} argument form`);
  const indices = command.flatMap((part, i) => part === name ? [i] : []);
  need(indices.length === 1 && indices[0] + 1 < command.length, `native command requires one ${name}`);
  return command[indices[0] + 1];
}
function nativeCommand(command, action) {
  need(argv(command) && command.at(-1) === action, `native ${action} command must be an argv ending in ${action}`);
  const task = option(command, '--task-root');
  need(isAbsolute(task), 'native task root must be absolute');
  option(command, '--state-root');
  option(command, '--gpu-lock-file');
  // Explicit epochs prevent a default change from changing candidate identities.
  option(command, '--epoch');
  return task;
}
const TASK_CHECK = `import hashlib,json,sys
from pathlib import Path
from geak.native.external import validate_external_task
from geak.native.contract import job_digest
p=Path(sys.argv[1]); raw=(p/'task.json').read_bytes()
c=validate_external_task(json.loads(raw),p)
print(json.dumps({'task_sha256':hashlib.sha256(raw).hexdigest(),'job_id':job_digest(c),'primary_regime':c['report_contract']['primary_regime']}))`;
const APPLY_PATCH = `import sys
from pathlib import Path
from geak.native.source import _snapshot,_patched,_read
import json
base,patch,destination=map(Path,sys.argv[1:])
c=json.loads((base/'task.json').read_bytes())
source=_patched(_snapshot(base,c['source_paths']),_read(patch.parent,patch.name)[0])
for name,(raw,mode) in source.items():
 p=destination/name; p.write_bytes(raw); p.chmod(0o644|mode)`;

function checked(read, objective, identity, patchHash) {
  need(read?.external_measurement_valid === true && read.measurement_valid === true
    && read.correctness === 'pass' && read.objective?.kind === 'external_primary'
    && read.objective.primary_regime === objective.primary_regime
    && read.primary_regime === objective.primary_regime && positive(read.primary_speedup)
    && read.external_evaluation?.candidate_id === identity.candidate_id
    && read.external_evaluation?.evaluation_id === identity.evaluation_id
    && read.patch_sha256 === patchHash, 'native evidence, objective or patch identity differs');
  need(typeof read.native_reproducible_gain === 'boolean' && typeof read.native_reproducible_regression === 'boolean'
    && ['accepted', 'no_improvement'].includes(read.validation_status), 'native gain decision is missing');
  return read;
}

export async function composeKernel(manifest, { run = runProcess } = {}) {
  need(manifest?.schema === 'geak.kernel-composition/2', 'use composition schema /2 with explicit lineage policies');
  const { task_root: taskRoot, task_sha256: taskHash, out_dir: outDir, evaluation, lanes, deadline_epoch: deadline } = manifest;
  need(isAbsolute(taskRoot || '') && isAbsolute(outDir || '') && hex(taskHash), 'absolute task/output paths and task SHA256 required');
  need(Array.isArray(lanes) && lanes.length >= 1, 'at least one lineage required');
  need(lanes.every(l => /^[a-z][a-z0-9_-]*$/.test(l.id)) && new Set(lanes.map(l => l.id)).size === lanes.length,
    'distinct safe lane IDs required');
  need(lanes.every(l => l.policy?.imports === 'none' && ['required', 'eligible'].includes(l.policy.finalist)
    && Object.keys(l.policy).sort().join(',') === 'finalist,imports'), 'supported lineage policy requires finalist=required|eligible and imports=none');
  need(lanes.filter(l => l.policy.finalist === 'required').length === 1, 'exactly one required finalist policy required');
  const requiredId = lanes.find(l => l.policy.finalist === 'required').id;
  need(positive(deadline), 'one common absolute deadline required');
  need(evaluation && positive(evaluation.timeout_ms), 'common evaluation timeout required');
  const commonRoot = nativeCommand(evaluation.measurement_command, 'measure');
  nativeCommand(evaluation.result_command, 'read');
  need(JSON.stringify(evaluation.measurement_command.slice(0, -1)) === JSON.stringify(evaluation.result_command.slice(0, -1)),
    'common measurement and reader must use the same native command prefix');
  need(await realpath(commonRoot) === await realpath(taskRoot), 'common evaluator must use the pinned task root');
  need(sha(await readFile(join(taskRoot, 'task.json'))) === taskHash, 'task.json SHA256 differs');

  const execute = async (command, extra = [], settings = {}) => {
    const p = await run({ cmd: command[0], args: [...command.slice(1), ...extra], timeoutMs: 30000, ...settings });
    need(clean(p), `command did not settle cleanly: ${command[0]} (${p.status}, exit ${p.code})`);
    return p;
  };
  const pythonEnv = { PYTHONPATH: [ROOT, process.env.PYTHONPATH].filter(Boolean).join(':') };
  const binding = JSON.parse((await execute(['python3', '-c', TASK_CHECK, taskRoot], [], { env: pythonEnv })).stdout);
  need(binding.task_sha256 === taskHash, 'validated task identity differs');
  const objective = { kind: 'external_primary', primary_regime: binding.primary_regime };
  const contextIds = new Set(), devices = new Set(), argsFiles = new Set(), outputPaths = new Set();
  const privatePaths = [];
  const ownPath = async (owner, label, path) => {
    const resolved = await resolvedPath(path);
    need(!privatePaths.some(other => other.owner !== owner && overlaps(resolved, other.path)),
      `${owner} ${label} overlaps another owner's private path`);
    privatePaths.push({ owner, label, path: resolved });
  };
  await ownPath(null, 'common native state', option(evaluation.result_command, '--state-root'));
  const prepared = [];
  for (const lane of lanes) {
    need(argv(lane.runtime_command) && isAbsolute(lane.args_file || '') && positive(lane.timeout_ms), 'complete lane runtime/args/timeout required');
    const directory = lane.output_dir || join(outDir, lane.id);
    need(isAbsolute(directory) && !outputPaths.has(resolve(directory)) && resolve(directory) !== resolve(outDir),
      'lane output directories must be distinct absolute paths');
    outputPaths.add(resolve(directory));
    await ownPath(lane.id, 'output_dir', directory);
    need(lane.runtime_options && Number.isSafeInteger(lane.runtime_options.concurrency) && lane.runtime_options.concurrency > 0,
      'each lane needs its own positive runtime concurrency');
    need(lane.resources?.context_id && Array.isArray(lane.resources.device_ids) && lane.resources.device_ids.length,
      'each lane needs an isolated context and physical device grant');
    need(!contextIds.has(lane.resources.context_id), 'lane contexts must be distinct');
    contextIds.add(lane.resources.context_id);
    for (const device of lane.resources.device_ids) {
      need(typeof device === 'string' && device && !devices.has(device), 'lane physical device grants must be disjoint');
      devices.add(device);
    }
    const argsPath = await realpath(lane.args_file);
    need(!argsFiles.has(argsPath), 'lane argument files must be separate');
    argsFiles.add(argsPath);
    const bytes = await readFile(argsPath), args = JSON.parse(bytes);
    need(['greedy', 'evo'].includes(args.optimizer), 'lineages must use the existing greedy or evo workflow');
    const gpuIds = lane.resources.gpu_ids;
    need(Array.isArray(gpuIds) && gpuIds.length === lane.resources.device_ids.length
      && gpuIds.every(id => typeof id === 'string' && /^\d+$/.test(id)) && new Set(gpuIds).size === gpuIds.length,
      'each physical device needs one distinct local GPU ID');
    need(args.gpu_ids != null && JSON.stringify(String(args.gpu_ids).split(',').map(id => id.trim())) === JSON.stringify(gpuIds),
      'workflow gpu_ids differ from the declared local-to-physical grant');
    for (const key of ['exp_root', 'eval_dir']) await ownPath(lane.id, key, args[key]);
    if (args.state_dir) await ownPath(lane.id, 'state_dir', args.state_dir);
    need((args.mode || 'optimize') === 'optimize' && String(args.apply_to_original ?? 'false') === 'false',
      'composition requires optimize-mode result handoffs without applying to the original');
    need(args.objective?.kind === objective.kind && args.objective.primary_regime === objective.primary_regime,
      'lane objective differs from the pinned native task');
    need(positive(args.deadline_epoch) && positive(args.search_reserve_s), 'each full lane requires its existing process-bounded deadline/reserves');
    need(args.deadline_epoch === deadline, 'lineage deadline differs from the common deadline');
    need(Date.now() + lane.timeout_ms >= args.deadline_epoch * 1000 + 7000,
      'outer lane timeout must include its full deadline and cleanup');
    need(isAbsolute(args.kernel_path || '') && sha(await readFile(join(args.kernel_path, 'task.json'))) === taskHash,
      'lane ORIGINAL task differs from the pinned task');
    const reader = args.external_final_result_command || args.external_result_command;
    const readerRoot = nativeCommand(reader, 'read');
    need(sha(await readFile(join(readerRoot, 'task.json'))) === taskHash, 'lane final reader task differs');
    for (const key of ['external_measurement_command', 'external_result_command',
      'external_final_measurement_command', 'external_final_result_command']) {
      if (!args[key]) continue;
      nativeCommand(args[key], key.includes('measurement') ? 'measure' : 'read');
      await ownPath(lane.id, key, option(args[key], '--state-root'));
    }
    const options = [];
    for (const [key, value] of Object.entries(lane.runtime_options)) {
      need(['agent', 'profile', 'model', 'registry', 'concurrency', 'agent_timeout_ms', 'backend_events_file'].includes(key), `unsupported runtime option: ${key}`);
      need(typeof value === 'string' || typeof value === 'number', 'runtime options must be scalar');
      if (key === 'backend_events_file') await ownPath(lane.id, key, value);
      options.push('--' + key.replaceAll('_', '-'), String(value));
    }
    prepared.push({ lane, directory, args, argsBytes: bytes, argsHash: sha(bytes), reader, options });
  }
  await mkdir(outDir); // A fresh output directory prevents stale results from being adopted.
  await save(join(outDir, 'binding.json'), { ...binding, objective, deadline_epoch: deadline, private_paths: privatePaths, evaluation: {
    task_root: commonRoot, state_root: option(evaluation.result_command, '--state-root'),
    epoch: option(evaluation.result_command, '--epoch'), gpu_lock_file: option(evaluation.result_command, '--gpu-lock-file') } });

  async function runLane({ lane, directory, args, argsBytes, argsHash, reader, options }) {
    await mkdir(directory);
    const resultFile = join(directory, 'runtime_result.json'), metricsFile = join(directory, 'runtime_metrics.json');
    const argsFile = join(directory, 'args.json');
    await writeFile(argsFile, argsBytes, { flag: 'wx' });
    const record = { id: lane.id, directory, status: 'failed', args_sha256: argsHash, resources: lane.resources,
      policy: lane.policy, deadline_epoch: deadline, runtime_options: lane.runtime_options,
      result_file: resultFile, metrics_file: metricsFile };
    try {
      const started = Date.now();
      const p = await run({ cmd: lane.runtime_command[0], args: [...lane.runtime_command.slice(1), WORKFLOW,
        ...options, '--args-file', argsFile, '--result-file', resultFile, '--metrics-file', metricsFile],
        cwd: lane.cwd, env: lane.env, timeoutMs: lane.timeout_ms });
      const settled = Date.now();
      await writeFile(join(directory, 'stdout.log'), p.stdout || '');
      await writeFile(join(directory, 'stderr.log'), p.stderr || '');
      record.process = { status: p.status, code: p.code, duration_ms: p.duration_ms,
        group_cleared: p.group_cleared, strays_killed: p.strays_killed, stdio_held: p.stdio_held,
        started_at_utc: new Date(started).toISOString(), settled_at_utc: new Date(settled).toISOString(),
        remaining_allowance_at_settlement_s: Math.max(0, deadline - settled / 1000) };
      need(clean(p), 'full worker did not settle cleanly');
      const [result, metrics] = await Promise.all([json(resultFile), json(metricsFile)]);
      record.metrics = metrics;
      need(metrics.ok === true, 'worker runtime reported failure');
      need(isAbsolute(result.eval_dir || '') && await resolvedPath(result.eval_dir) === await resolvedPath(args.eval_dir),
        'worker returned a different private evaluation directory');
      record.workflow_summary = Object.fromEntries(['stopped_by', 'deadline_hit', 'rounds', 'budget_used', 'budget_total']
        .map(key => [key, result[key] ?? null]));
      need(['accepted', 'no_improvement'].includes(result.validation_status) && result.external_measurement_valid === true,
        'worker did not return a valid native result (or the director rejected it)');
      const identity = result.final_selection;
      need(identity && hex(identity.candidate_id) && hex(identity.source_id) && identity.evaluation_id
        && result.external_evaluation?.candidate_id === identity.candidate_id
        && result.external_evaluation?.evaluation_id === identity.evaluation_id, 'worker final identity differs');
      need(identity.candidate_id === candidateId(binding.job_id, option(reader, '--epoch'), identity.source_id),
        'worker candidate is not bound to the common task and its declared epoch');
      // Native no-gain results intentionally suppress export but preserve the
      // host's captured patch. Read it through the same source-bound reader.
      const patchPath = result.final_patch || (isAbsolute(result.eval_dir || '') && join(result.eval_dir, 'host_final_patch.diff'));
      need(isAbsolute(patchPath || ''), 'worker has no captured final patch');
      const patch = await readFile(patchPath);
      if (result.final_patch) need(hex(result.final_patch_sha256) && sha(patch) === result.final_patch_sha256,
        'worker final patch SHA256 differs');
      record.patch = join(directory, 'final_patch.diff');
      record.patch_sha256 = sha(patch);
      await writeFile(record.patch, patch, { flag: 'wx' });
      const read = JSON.parse((await execute(reader, ['--workspace', args.kernel_path, '--patch', record.patch,
        '--candidate-id', identity.candidate_id, '--evaluation-id', identity.evaluation_id], { cwd: lane.cwd, env: lane.env })).stdout);
      record.child_evidence = checked(read, objective, identity, record.patch_sha256);
      record.child_selection = identity;
      record.source_id = identity.source_id;
      record.status = 'retained';
    } catch (error) { record.error = String(error.message || error); }
    await save(join(directory, 'candidate.json'), record);
    return record;
  }
  // Each existing workflow runs once, including its own stop policy and postlude.
  // A peer cannot retire it, restart it, or borrow its unused allowance.
  const settled = await Promise.allSettled(prepared.map(runLane));
  const children = settled.map((entry, i) => entry.status === 'fulfilled' ? entry.value
    : { id: prepared[i].lane.id, status: 'failed', error: String(entry.reason) });

  // All search workers have settled. Finalists use one common evaluator in a
  // deterministic sequence, with a fresh ORIGINAL+patch workspace for each.
  const ordered = [...children].sort((a, b) => (a.id === requiredId ? -1 : b.id === requiredId ? 1 : 0));
  for (const child of ordered) {
    if (child.status !== 'retained') continue;
    try {
      const workspace = join(outDir, `${child.id}_final_workspace`);
      await mkdir(workspace); // Never let the materializer clear an existing export.
      await execute(['bash', MATERIALIZE, '--src', taskRoot, '--dst', workspace], [], { timeoutMs: 300000 });
      await execute(['python3', '-c', APPLY_PATCH, taskRoot, child.patch, workspace], [], { env: pythonEnv });
      const evaluationSettings = { timeoutMs: evaluation.timeout_ms, cwd: evaluation.cwd, env: evaluation.env };
      const captured = JSON.parse((await execute(evaluation.measurement_command,
        ['--workspace', workspace, '--capture-only'], evaluationSettings)).stdout);
      need(captured.capture_only === true && hex(captured.candidate_id) && captured.source_id === child.source_id
        && captured.evaluation_id, 'common capture differs from the retained source');
      need(captured.candidate_id === candidateId(binding.job_id, option(evaluation.result_command, '--epoch'), captured.source_id),
        'common candidate is not bound to the pinned task and final epoch');
      await execute(evaluation.measurement_command, ['--workspace', workspace,
        '--candidate-id', captured.candidate_id, '--evaluation-id', captured.evaluation_id], evaluationSettings);
      const measured = JSON.parse((await execute(evaluation.result_command, ['--workspace', taskRoot,
        '--patch', child.patch, '--candidate-id', captured.candidate_id, '--evaluation-id', captured.evaluation_id], evaluationSettings)).stdout);
      child.common_evidence = checked(measured, objective, captured, child.patch_sha256);
      child.common_selection = captured;
      child.status = measured.native_reproducible_gain && !measured.native_reproducible_regression ? 'qualified' : 'no_improvement';
    } catch (error) { child.status = 'qualification_failed'; child.qualification_error = String(error.message || error); }
    await save(join(child.directory, 'candidate.json'), child);
  }
  let winner = null;
  for (const child of ordered) if (child.status === 'qualified'
    && (!winner || child.common_evidence.primary_speedup > winner.common_evidence.primary_speedup)) winner = child;
  const required = children.find(child => child.id === requiredId);
  const comparisonComplete = children.every(child => ['qualified', 'no_improvement'].includes(child.status));
  // Retain earlier required evidence only when no common-qualified gain exists.
  const fallback = !winner && required.status === 'qualification_failed' && required.child_evidence?.native_reproducible_gain === true
    && required.child_evidence.native_reproducible_regression === false ? required : null;
  const anyMeasured = children.some(child => ['qualified', 'no_improvement'].includes(child.status));
  const result = { schema: 'geak.kernel-composition-result/2', binding, objective, deadline_epoch: deadline,
    required_finalist: { id: requiredId, status: required.status, retained: !!required.child_evidence,
      qualified: ['qualified', 'no_improvement'].includes(required.status) },
    comparison_complete: comparisonComplete,
    outcome: fallback ? 'comparison_incomplete_required_retained' : winner ? 'improved'
      : comparisonComplete ? 'no_improvement' : anyMeasured ? 'comparison_incomplete' : 'comparison_unavailable',
    selection_basis: fallback ? 'retained_required_fallback' : winner ? 'common_native_measurement' : 'original_retained',
    winner: fallback ? { id: fallback.id, final_patch: fallback.patch, final_patch_sha256: fallback.patch_sha256,
      final_selection: fallback.child_selection, child_evidence: fallback.child_evidence,
      primary_speedup: null, comparison_unavailable: true }
      : winner ? { id: winner.id, final_patch: winner.patch, final_patch_sha256: winner.patch_sha256,
      final_selection: winner.common_selection, ...winner.common_evidence } : null,
    children };
  await save(join(outDir, 'result.json'), result);
  return result;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    need(process.argv.length === 3, 'usage: node compose_kernel.mjs /absolute/composition.json');
    const result = await composeKernel(await json(resolve(process.argv[2])));
    process.stdout.write(JSON.stringify(result) + '\n');
    if (!result.required_finalist.qualified) process.exitCode = 1;
  } catch (error) { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; }
}
