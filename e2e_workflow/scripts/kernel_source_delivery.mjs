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
      }
    }
    return { ok: true, why: `source and recorded GPU evidence match ${required.length} requested kernels`,
      snapshot_sha256: m.snapshot_sha256 };
  } catch (e) { return { ok: false, why: e.message }; }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [mode, task, name] = process.argv.slice(2);
  if (mode === 'snapshot' && task) console.log(JSON.stringify(await sourceSnapshot(task), null, 2));
  else if (mode === 'ref' && task && name) console.log(JSON.stringify(await fileRef(task, name), null, 2));
  else { console.error('Usage: kernel_source_delivery.mjs snapshot TASK | ref TASK FILE'); process.exitCode = 2; }
}
