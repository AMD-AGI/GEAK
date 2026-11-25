#!/usr/bin/env node
// Verify recorded GPU evidence against the current source/entry. This checks
// consistency and coverage; it does not replace execution or audit GPU logs.
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const sourceExt = /\.(py|c|cc|cpp|cxx|h|hpp|cu|cuh|hip|s|mlir)$/i;
async function localFile(task, name) {
  const file = await fs.realpath(path.resolve(task, name));
  const relative = path.relative(task, file);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
    throw new Error(`file outside task: ${name}`);
  return file;
}
export async function fileRef(taskDir, name) {
  const task = await fs.realpath(taskDir), file = await localFile(task, name);
  return { path: path.relative(task, file), sha256: sha(await fs.readFile(file)) };
}
async function readRef(task, ref) {
  if (!ref || typeof ref.path !== 'string' || !/^[a-f0-9]{64}$/.test(ref.sha256 || ''))
    throw new Error('evidence requires a task-local path and SHA-256');
  const bytes = await fs.readFile(await localFile(task, ref.path));
  if (sha(bytes) !== ref.sha256) throw new Error(`SHA-256 mismatch: ${ref.path}`);
  return JSON.parse(bytes.toString('utf8'));
}
export async function sourceSnapshot(taskDir) {
  const task = await fs.realpath(taskDir), names = ['meta.json', 'cases.py', 'unittest.py'];
  const walk = async dir => {
    for (const e of await fs.readdir(path.join(task, dir), { withFileTypes: true })) {
      const name = path.join(dir, e.name);
      if (e.isDirectory() && e.name !== '__pycache__') await walk(name);
      else if (sourceExt.test(e.name)) names.push(name);
    }
  };
  await walk('kernel_src');
  // Additional task-local harness/build adapters can be declared by the author.
  const meta = JSON.parse(await fs.readFile(path.join(task, 'meta.json'), 'utf8'));
  for (const name of meta.kernel_extraction?.source_delivery_files || []) names.push(name);
  const files = [];
  for (const name of [...new Set(names)].sort()) files.push(await fileRef(task, name));
  if (!files.some(f => f.path.startsWith(`kernel_src${path.sep}`)))
    throw new Error('kernel_src contains no source files');
  return { files, snapshot_sha256: sha(JSON.stringify(files)) };
}

// A UT written by generate_ut.py starts with this line. Only such "new" tasks get the checks
// below; hand-written (old) tasks keep the original checks.
export const GENERATED_UT_MARKER = '# GEAK-GENERATED unittest.py';
export const UT_CONTRACT_CORE_FILES = ['unittest.py', 'cases.py', 'meta.json', 'ut_runner.py',
  'harness_lib.py', 'leg_runner.py', 'overlay_setup.py'];
const isObject = v => !!v && typeof v === 'object' && !Array.isArray(v);
const present = v => v !== undefined && v !== null;
const rankTimes = (v, n) => Array.isArray(v) && v.length === n && v.every(x => Number.isFinite(x));
export async function isGeneratedTask(taskDir) {
  try {
    const text = await fs.readFile(path.join(taskDir, 'unittest.py'), 'utf8');
    return text.split('\n', 1)[0].startsWith(GENERATED_UT_MARKER);
  } catch { return false; }
}
async function readTaskJson(task, name) {
  try { return JSON.parse(await fs.readFile(await localFile(task, name), 'utf8')); }
  catch (e) { throw new Error(e.code === 'ENOENT' ? `missing ${name}` : `${name}: ${e.message}`); }
}
// Recompute {files, contract_sha256} over the files listed in run/ut_contract.json.
export async function utContract(taskDir, recorded = null) {
  const task = await fs.realpath(taskDir);
  recorded ||= await readTaskJson(task, 'run/ut_contract.json');
  if (!isObject(recorded) || !Array.isArray(recorded.files)) throw new Error('run/ut_contract.json lists no files');
  const files = [];
  for (const f of recorded.files) {
    if (typeof f?.path !== 'string' || !f.path) throw new Error('run/ut_contract.json has a file without a path');
    try { files.push({ path: f.path, sha256: (await fileRef(task, f.path)).sha256 }); }
    catch (e) { throw new Error(`UT contract file ${f.path}: ${e.message}`); }
  }
  return { files, contract_sha256: sha(JSON.stringify(files)) };
}

// Pure checks for generated tasks. Each returns '' when satisfied, else the reason.
// `recorded` is run/ut_contract.json, `current` its utContract() recomputation.
export function utContractDefect(recorded, current) {
  if (!isObject(recorded) || !Array.isArray(recorded.files)) return 'run/ut_contract.json lists no files';
  const listed = new Set(recorded.files.map(f => f?.path));
  const missing = UT_CONTRACT_CORE_FILES.filter(name => !listed.has(name));
  if (missing.length) return `UT contract does not list ${missing.join(', ')}`;
  const now = new Map((current?.files || []).map(f => [f.path, f.sha256]));
  const changed = recorded.files.filter(f => typeof f?.sha256 !== 'string' || now.get(f.path) !== f.sha256);
  if (changed.length) return `UT contract files changed after freeze: ${changed.map(f => f?.path).join(', ')}`;
  if (recorded.contract_sha256 !== sha(JSON.stringify(recorded.files)))
    return 'UT contract_sha256 does not match its file list';
  return '';
}
export function verifyUtDefect(verify, contractSha, snapshotSha) {
  if (!isObject(verify)) return 'run/verify_ut.json is not a verification report';
  if (verify.verdict !== 'PASS') return `verify_ut verdict is ${verify.verdict ?? 'missing'}, not PASS`;
  if (typeof verify.contract_sha256 !== 'string' || verify.contract_sha256 !== contractSha)
    return 'verify_ut.json does not certify the current UT contract';
  if (typeof verify.snapshot_sha256 !== 'string' || verify.snapshot_sha256 !== snapshotSha)
    return 'verify_ut.json does not certify the current source snapshot';
  if (present(verify.gaps) && !(Array.isArray(verify.gaps) && !verify.gaps.length))
    return 'verify_ut.json declares coverage gaps';
  return '';
}
// Timing receipt: alignment plus `worldSize` per-rank times for every timed case (baseline, and
// current when present).
export function rankReceiptDefect(receipt, worldSize) {
  if (!isObject(receipt)) return 'no timing receipt';
  if (!isObject(receipt.alignment)) return 'timing receipt has no alignment';
  const cases = isObject(receipt.cases) ? Object.entries(receipt.cases) : [];
  if (!cases.length) return 'timing receipt has no timed case';
  for (const [sig, c] of cases)
    for (const leg of ['baseline', 'current'])
      if ((leg === 'baseline' || present(c?.current)) && !rankTimes(c?.[leg]?.rank_ms, worldSize))
        return `timing receipt case ${sig}: ${leg}.rank_ms is not ${worldSize} finite per-rank times`;
  return '';
}
// Older hand-written multi-rank UTs: timing.cases[].baseline.per_rank_us and timing.alignment.
export function legacyRankTimingDefect(timing, worldSize) {
  if (!isObject(timing)) return 'no timing';
  if (!isObject(timing.alignment)) return 'timing has no alignment';
  const cases = Array.isArray(timing.cases) ? timing.cases : [];
  if (!cases.length) return 'timing has no timed case';
  for (const [i, c] of cases.entries())
    if (!rankTimes(c?.baseline?.per_rank_us, worldSize))
      return `timing case ${c?.sig ?? i}: baseline.per_rank_us is not ${worldSize} finite per-rank times`;
  return '';
}
// Per-rank time and alignment for every timed case, in either format.
export function perRankTimingDefect(ut, worldSize) {
  const receipt = rankReceiptDefect(ut?.receipt, worldSize);
  if (!receipt) return '';
  const legacy = legacyRankTimingDefect(ut?.timing, worldSize);
  return legacy ? `${receipt}; old format: ${legacy}` : '';
}
export function utReceiptDefect(ut, contractSha, worldSize) {
  if (typeof ut?.contract_sha256 !== 'string' || ut.contract_sha256 !== contractSha)
    return 'UT result does not identify the current UT contract';
  if (!isObject(ut.receipt)) return 'UT result carries no timing receipt';
  return worldSize > 1 ? rankReceiptDefect(ut.receipt, worldSize) : '';
}
// A source mutation counts as rejected only when every rank launched and saw mismatches.
export function rankRejectionDefect(record, worldSize) {
  const rows = record?.ranks;
  if (!Array.isArray(rows)) return 'numerical_rejection has no per-rank results';
  if (rows.length !== worldSize) return `numerical_rejection has ${rows.length} rank rows for world_size ${worldSize}`;
  for (let rank = 0; rank < worldSize; rank++) {
    const row = rows.filter(r => r?.rank === rank);
    if (row.length !== 1) return `numerical_rejection needs one row for rank ${rank}`;
    if (row[0].launch_succeeded !== true || !(Number.isFinite(row[0].numerical_mismatches) && row[0].numerical_mismatches > 0))
      return `rank ${rank} did not numerically reject the mutation`;
  }
  return '';
}
// meta.capabilities.multi_rank.world_size; absent => 1.
export function metaWorldSize(meta) {
  const n = meta?.capabilities?.multi_rank?.world_size ?? 1;
  if (!Number.isInteger(n) || n < 1) throw new Error(`meta world_size must be a positive integer: ${JSON.stringify(n)}`);
  return n;
}
// The GPU count a generated task declares. world_size > 1: the request or the result (at least one)
// writes required_gpu_count, and every written one equals world_size. world_size 1: both may be
// absent; a written one must be 1. (null counts as absent, as in packing's `??` default.)
export function gpuCountDefect(kernel, result, worldSize) {
  const written = [kernel?.required_gpu_count, result?.required_gpu_count].filter(present);
  if (written.every(n => n === worldSize) && (worldSize === 1 || written.length)) return '';
  const show = v => present(v) ? JSON.stringify(v) : 'absent';
  return `declared GPU counts disagree: requested required_gpu_count ${show(kernel?.required_gpu_count)}, ` +
    `returned required_gpu_count ${show(result?.required_gpu_count)}, meta world_size ${show(worldSize)}` +
    (worldSize > 1 && !written.length ? `; a ${worldSize}-rank task writes required_gpu_count in its request or result` : '');
}
// Every negative control runs unless meta names it with a reason.
export function disabledControlsDefect(meta) {
  const disabled = meta?.controls?.disabled;
  if (!present(disabled)) return '';
  if (!isObject(disabled)) return 'meta.controls.disabled must map each disabled control to its reason';
  const bare = Object.entries(disabled).filter(([, why]) => typeof why !== 'string' || !why.trim());
  return bare.length ? `disabled controls without a reason: ${bare.map(([k]) => k).join(', ')}` : '';
}
// Throws the first unmet rule; returns nothing, so a passing generated task gets main's result.
async function verifyGeneratedTask(task, kernel, result, meta, snapshotSha, ut, rejections) {
  const fail = why => { if (why) throw new Error(why); };
  const worldSize = metaWorldSize(meta);
  fail(gpuCountDefect(kernel, result, worldSize));
  fail(disabledControlsDefect(meta));
  const contract = await readTaskJson(task, 'run/ut_contract.json');
  fail(utContractDefect(contract, await utContract(task, contract)));
  fail(verifyUtDefect(await readTaskJson(task, 'run/verify_ut.json'), contract.contract_sha256, snapshotSha));
  fail(utReceiptDefect(ut, contract.contract_sha256, worldSize));
  if (worldSize > 1) for (const [k, e] of rejections) {
    const why = rankRejectionDefect(e, worldSize);
    if (why) throw new Error(`${why} (${k})`);
  }
}

// The UT result a delivery certifies: source_delivery_ref -> ut_result_ref, else run/ut_result.json.
export async function readUtResult(result) {
  const task = await fs.realpath(result.task_dir);
  const m = result.source_delivery_ref ? await readRef(task, result.source_delivery_ref) : null;
  return m?.ut_result_ref ? readRef(task, m.ut_result_ref) : readTaskJson(task, 'run/ut_result.json');
}

export async function verifySourceDelivery(kernel, result) {
  if (kernel?.source_delivery_required !== true && !result?.source_delivery_ref)
    return { ok: true, why: 'source evidence contract not requested' };
  try {
    const task = await fs.realpath(result.task_dir);
    const m = await readRef(task, result.source_delivery_ref);
    if (m.contract !== 'kernel_source_delivery/v1') throw new Error('wrong source delivery contract');
    const snapshot = await sourceSnapshot(task);
    if (m.snapshot_sha256 !== snapshot.snapshot_sha256 || JSON.stringify(m.files) !== JSON.stringify(snapshot.files))
      throw new Error('source, cases, harness or meta changed after recorded validation');
    const meta = JSON.parse(await fs.readFile(path.join(task, 'meta.json'), 'utf8'));
    if (!m.target_callable || m.target_callable !== result.target_callable || meta.target_callable !== m.target_callable)
      throw new Error('source evidence does not certify the current target_callable');
    const ut = await readRef(task, m.ut_result_ref);
    if (!(ut.ok === true || ut.smoke === 'pass') || ut.exit_code !== 0 ||
        ut.snapshot_sha256 !== m.snapshot_sha256 || ut.target_callable !== m.target_callable)
      throw new Error('full GPU UT result must pass and identify the current entry and source snapshot');
    const required = [...new Set([...(kernel.selected_device_kernels || [kernel.device_kernel]),
      ...(kernel.source_device_kernels || [])])];
    if (!required.length || required.some(k => !k)) throw new Error('source delivery has no requested kernels');
    if (!Array.isArray(m.kernels)) throw new Error('source delivery lacks kernel evidence');
    const rejections = [];
    for (const k of required) {
      const rows = m.kernels.filter(r => r.device_kernel === k);
      if (rows.length !== 1) throw new Error(`expected one source record for ${k}`);
      const row = rows[0];
      if (!Array.isArray(row.source_files) || !row.source_files.length || row.source_files.some(p =>
          !p.startsWith('kernel_src/') || !sourceExt.test(p) || !m.files.some(f => f.path === p)))
        throw new Error(`missing native/JIT implementation files for ${k}`);
      for (const [field, kind] of [['execution_ref', 'candidate_execution'], ['negative_control_ref', 'numerical_rejection']]) {
        const e = await readRef(task, row[field]);
        if (e.kind !== kind || e.ok !== true || e.device_kernel !== k ||
            e.snapshot_sha256 !== m.snapshot_sha256 || e.target_callable !== m.target_callable ||
            !e.command || !e.gpu || !e.log_ref)
          throw new Error(`incomplete ${kind} evidence for ${k}`);
        // Hash the actual log as well; it need not be JSON.
        const actual = await fileRef(task, e.log_ref.path);
        if (actual.sha256 !== e.log_ref.sha256) throw new Error(`changed ${kind} log for ${k}`);
        if (kind === 'candidate_execution' && !(e.invocations > 0))
          throw new Error(`no candidate leaf invocation for ${k}`);
        if (kind === 'numerical_rejection' && (e.launch_succeeded !== true || !(e.numerical_mismatches > 0)))
          throw new Error(`negative control did not numerically reject ${k}`);
        if (kind === 'numerical_rejection') rejections.push([k, e]);
      }
    }
    // A generated task passes the same way as any other: its extra rules only add failures.
    if (await isGeneratedTask(task))
      await verifyGeneratedTask(task, kernel, result, meta, snapshot.snapshot_sha256, ut, rejections);
    return { ok: true, why: `source and recorded GPU evidence match ${required.length} requested kernels`,
      snapshot_sha256: m.snapshot_sha256 };
  } catch (e) { return { ok: false, why: e.message }; }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [mode, task, name] = process.argv.slice(2);
  if (mode === 'snapshot' && task) console.log(JSON.stringify(await sourceSnapshot(task), null, 2));
  else if (mode === 'ref' && task && name) console.log(JSON.stringify(await fileRef(task, name), null, 2));
  else if (mode === 'contract' && task) console.log(JSON.stringify(await utContract(task), null, 2));
  else { console.error('Usage: kernel_source_delivery.mjs snapshot TASK | ref TASK FILE | contract TASK'); process.exitCode = 2; }
}
