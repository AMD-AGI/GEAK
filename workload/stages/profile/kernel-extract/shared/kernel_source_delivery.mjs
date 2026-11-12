#!/usr/bin/env node
// Verify recorded GPU evidence against the current source/entry. This checks
// consistency and coverage; it does not replace execution or audit GPU logs.
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const sourceExt = /\.(py|c|cc|cpp|cxx|h|hpp|cu|cuh|hip|s|mlir)$/i;
export const UT_DRIVER_SNAPSHOT_CONTRACT = 'ut_driver/path_sha256_concat/v1';
const fullUtGates = ['contract', 'inputs', 'binding', 'launches', 'correctness', 'server_calls',
  'compile_variants', 'isolation', 'replay', 'negative_controls', 'restore', 'timing'];
const hexDigest = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
// Python sorts Unicode code points, whereas Array.sort() compares UTF-16 units.
const pythonSorted = names => [...names].sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)));
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
function allowedInfiniteMargin(kind, parts, document) {
  const caught = c => c?.caught === true && c.ok === true && c.restored === true;
  if (kind === 'ut_report' && parts.length === 5 && parts[0] === 'gates' &&
      parts[1] === 'negative_controls' && parts[2] === 'controls' && parts[4] === 'margin' &&
      /^(0|[1-9][0-9]*)$/.test(parts[3])) {
    const gate = document?.gates?.negative_controls;
    return gate?.status === 'pass' && Array.isArray(gate.controls) && caught(gate.controls[Number(parts[3])]);
  }
  return kind === 'negative_control' && parts.length === 2 &&
    parts[0] === 'ut_negative_controls_gate' && parts[1] === 'margin' &&
    document?.kind === 'numerical_rejection' && document.ok === true && document.launch_succeeded === true &&
    Number.isFinite(document.numerical_mismatches) && document.numerical_mismatches > 0 &&
    caught(document.ut_negative_controls_gate);
}

function finiteEvidence(document, markers = new Set(), kind = null) {
  let consumed = 0;
  const visit = (value, parts) => {
    if (typeof value === 'string' && markers.has(value)) {
      if (!allowedInfiniteMargin(kind, parts, document))
        throw new Error(`positive Infinity is not allowed at /${parts.join('/')}`);
      consumed++;
      return Infinity;
    }
    if (typeof value === 'number' && !Number.isFinite(value))
      throw new Error(`non-finite JSON number at /${parts.join('/')}`);
    if (Array.isArray(value)) {
      for (let i = 0; i < value.length; i++) value[i] = visit(value[i], [...parts, String(i)]);
    } else if (value && typeof value === 'object') {
      for (const key of Object.keys(value)) {
        if (markers.has(key)) throw new Error('Infinity cannot be an object key');
        value[key] = visit(value[key], [...parts, key]);
      }
    }
    return value;
  };
  const result = visit(document, []);
  // JSON.parse normally discards duplicate object properties. Do not allow an
  // overwritten Infinity token to evade its path check, even in diagnostics.
  if (consumed !== markers.size) throw new Error('Infinity token was discarded by duplicate JSON properties');
  return result;
}

export function parsePythonUtArtifact(text, kind) {
  if (!['ut_report', 'negative_control'].includes(kind)) throw new Error('unsupported Python UT artifact kind');
  const tokens = [], strings = new Set();
  for (let i = 0; i < text.length;) {
    if (text[i] === '"') {
      const start = i++;
      while (i < text.length && text[i] !== '"') i += text[i] === '\\' ? 2 : 1;
      if (i >= text.length) throw new Error('unterminated JSON string');
      i++;
      strings.add(JSON.parse(text.slice(start, i))); // validate escapes; never eval
    } else if (text.startsWith('-Infinity', i) || text.startsWith('NaN', i)) {
      throw new Error('NaN and negative Infinity are not supported in Python UT evidence');
    } else if (text.startsWith('Infinity', i)) {
      tokens.push(i); i += 'Infinity'.length;
    } else i++;
  }
  const chunks = [], markers = new Set();
  let previous = 0;
  for (const [index, position] of tokens.entries()) {
    // Each marker is absent from ALL original decoded strings (including keys),
    // so quoted or unicode-escaped text can never masquerade as a numeric token.
    let marker = `\u0000geak-positive-infinity:${index}`;
    while (strings.has(marker)) marker += ':';
    markers.add(marker);
    chunks.push(text.slice(previous, position), JSON.stringify(marker));
    previous = position + 'Infinity'.length;
  }
  chunks.push(text.slice(previous));
  return finiteEvidence(JSON.parse(chunks.join('')), markers, kind);
}

async function readRef(task, ref, pythonArtifact = null) {
  if (!ref || typeof ref.path !== 'string' || !/^[a-f0-9]{64}$/.test(ref.sha256 || ''))
    throw new Error('evidence requires a task-local path and SHA-256');
  const bytes = await fs.readFile(await localFile(task, ref.path));
  if (sha(bytes) !== ref.sha256) throw new Error(`SHA-256 mismatch: ${ref.path}`);
  // Always hash the ORIGINAL bytes before any compatibility normalization.
  const text = bytes.toString('utf8');
  return pythonArtifact ? parsePythonUtArtifact(text, pythonArtifact) : finiteEvidence(JSON.parse(text));
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

export async function utDriverSnapshot(taskDir) {
  // Reproduce the versioned ut_driver.py _snapshot/_reuse_key algorithms without
  // importing or executing a task's Python. This intentionally differs from the
  // canonical sourceSnapshot above: fixed concatenation order and ALL kernel_src
  // files (including build data), plus helpers/server evidence in the reuse key.
  const task = await fs.realpath(taskDir), tree = [];
  const isFile = async name => {
    try { return (await fs.stat(path.join(task, name))).isFile(); }
    catch (e) { if (e.code === 'ENOENT') return false; throw e; }
  };
  const exists = async name => {
    try { await fs.stat(path.join(task, name)); return true; }
    catch (e) { if (e.code === 'ENOENT') return false; throw e; }
  };
  const list = async name => {
    try { return await fs.readdir(path.join(task, name)); }
    catch (e) { if (e.code === 'ENOENT') return []; throw e; }
  };
  const walk = async dir => {
    for (const e of await fs.readdir(path.join(task, dir), { withFileTypes: true })) {
      const name = path.join(dir, e.name);
      if (e.isDirectory()) await walk(name);
      // os.walk does not follow symlink directories. It does hash symlink files;
      // localFile additionally refuses any target outside the task boundary.
      else if (!(e.isSymbolicLink() && (await fs.stat(path.join(task, name))).isDirectory()) &&
               !path.join(task, name).includes('__pycache__')) tree.push(name);
    }
  };
  await walk('kernel_src');
  const snapshotNames = ['meta.json', 'cases.py', 'unittest.py', 'run/ut_cases.json', 'run/ut_contract.json',
    ...pythonSorted(tree)];
  const reuseNames = [
    ...pythonSorted((await list('')).filter(n => n.endsWith('.py') || n.endsWith('.json'))),
    ...pythonSorted((await list('run')).filter(n => n === 'ut_cases.json' || n === 'ut_contract.json' ||
      n.startsWith('server_')).map(n => `run/${n}`)),
    ...pythonSorted((await list('run/server_calls')).map(n => `run/server_calls/${n}`)),
    ...pythonSorted(tree),
  ];
  const digest = async (names, included) => {
    const h = createHash('sha256');
    for (const name of names) {
      if (!(await included(name))) continue;
      // Hash the original relative spelling, not fileRef's resolved symlink path.
      const file = await localFile(task, name);
      h.update(name, 'utf8'); h.update(sha(await fs.readFile(file)), 'utf8');
    }
    return h.digest('hex');
  };
  return { contract: UT_DRIVER_SNAPSHOT_CONTRACT,
    snapshot_sha256: await digest(snapshotNames, exists), reuse_key: await digest(reuseNames, isFile) };
}

async function verifyUtSnapshot(task, manifest, ut) {
  if (manifest.ut_snapshot === undefined) {
    if (ut.validation_contract !== undefined || ut.validation_mode !== undefined)
      throw new Error('scoped UT validation requires the hashed UT driver report');
    if (!(ut.ok === true || ut.smoke === 'pass') || ut.exit_code !== 0 ||
        ut.snapshot_sha256 !== manifest.snapshot_sha256 || ut.target_callable !== manifest.target_callable)
      throw new Error('full GPU UT result must pass and identify the current entry and source snapshot');
    return;
  }
  const spec = manifest.ut_snapshot;
  if (spec?.contract !== UT_DRIVER_SNAPSHOT_CONTRACT)
    throw new Error('unsupported UT snapshot contract');
  const report = await readRef(task, spec.report_ref, 'ut_report');
  const current = await utDriverSnapshot(task);
  if (ut.snapshot_sha256 !== current.snapshot_sha256 || report.snapshot_sha256 !== current.snapshot_sha256)
    throw new Error('UT driver snapshot does not match current task inputs/source');
  if (!hexDigest(report.reuse_key) || report.reuse_key !== current.reuse_key)
    throw new Error('UT driver reuse_key does not match current helpers, server evidence and source');
  if (ut.ok !== true || ut.verdict !== 'pass' || ut.exit_code !== 0 || report.verdict !== 'pass' ||
      report.exit_code !== 0 || report.quick || ut.target_callable !== manifest.target_callable ||
      report.target_callable !== manifest.target_callable)
    throw new Error('UT driver compatibility requires a full passing UT result and report for the current entry');
  if (!Number.isFinite(report.started) || report.started <= 0 || report.started !== ut.started ||
      typeof report.family !== 'string' || !report.family || report.family !== ut.family ||
      !hexDigest(report.contract_sha256) || report.contract_sha256 !== ut.contract_sha256)
    throw new Error('UT result and full report do not identify the same validation run and contract');
  const passed = r => r && ['pass', 'not_required'].includes(r.status);
  let required = fullUtGates;
  const scoped = [report, ut].some(d => d.validation_contract !== undefined || d.validation_mode !== undefined);
  if (scoped) {
    const mode = report.validation_mode;
    if (!['correctness', 'performance'].includes(mode) || ut.validation_mode !== mode ||
        [report, ut].some(d => d.validation_contract !== 'kernel_ut_validation/v1'))
      throw new Error('UT result/report must declare the same supported validation mode');
    required = mode === 'correctness' ? fullUtGates.filter(g => g !== 'timing') : fullUtGates;
    const sameGates = a => Array.isArray(a) && a.length === required.length &&
      new Set(a).size === required.length && required.every(g => a.includes(g));
    if ([report, ut].some(d => !sameGates(d.required_gates) || !sameGates(Object.keys(d.gates || {})) ||
        d.performance_measured !== (mode === 'performance') ||
        mode === 'performance' && d.gates?.timing?.status !== 'pass'))
      throw new Error('UT validation scope or performance claim is inconsistent with its required gates');
  }
  if (!required.every(g => passed(report.gates?.[g]) && passed(ut.gates?.[g]) &&
      ut.gates[g].ok === true && ut.gates[g].status === report.gates[g].status) ||
      !Object.values(report.gates || {}).every(passed) || !Object.values(ut.gates || {}).every(passed))
    throw new Error(scoped ? 'UT validation requires every gate in its declared mode to pass' :
      'UT driver compatibility requires every full UT gate, including timing, to pass');
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
    await verifyUtSnapshot(task, m, ut);
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
        const pythonArtifact = kind === 'numerical_rejection' &&
          m.ut_snapshot?.contract === UT_DRIVER_SNAPSHOT_CONTRACT ? 'negative_control' : null;
        const e = await readRef(task, row[field], pythonArtifact);
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
      snapshot_sha256: m.snapshot_sha256,
      ...(ut.validation_contract ? { validation_mode: ut.validation_mode,
        performance_measured: ut.performance_measured } : {}) };
  } catch (e) { return { ok: false, why: e.message }; }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [mode, task, name] = process.argv.slice(2);
  if (mode === 'snapshot' && task) console.log(JSON.stringify(await sourceSnapshot(task), null, 2));
  else if (mode === 'ut-driver-snapshot' && task) console.log(JSON.stringify(await utDriverSnapshot(task), null, 2));
  else if (mode === 'ref' && task && name) console.log(JSON.stringify(await fileRef(task, name), null, 2));
  else { console.error('Usage: kernel_source_delivery.mjs snapshot TASK | ut-driver-snapshot TASK | ref TASK FILE'); process.exitCode = 2; }
}
