#!/usr/bin/env node
// CPU-only preparation/preflight of extractor returns. The workflow remains the
// authority: delivery_contract.mjs is its schemas, identity gate and evidence-file
// reader, extracted once; pass a GEAK e2e_workflow.js to loadContract() to re-extract.
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { validate } from './schema.mjs';
import { verifySourceDelivery } from './kernel_source_delivery.mjs';

export async function loadContract(filename = null) {
  if (!filename) return await import('./delivery_contract.mjs');
  const source = await fs.readFile(filename, 'utf8');
  const block = (start, end) => {
    const a = source.indexOf(start), b = source.indexOf(end, a);
    if (a < 0 || b <= a) throw new Error(`workflow contract block unavailable: ${start}`);
    return source.slice(a, b);
  };
  const code = [
    block('const obj =', '// Result of the ensure_flydsl'),
    block('const SMOKE_STATUS =', 'const OPBENCH_SCHEMA ='),
    block('const EXTRACT_SCHEMA =', 'const KERNEL_LAYER_SCHEMA ='),
    block('const CALLABLE_SPEC_RX =', 'const PRE_FLAGGED_HEADS ='),
    block('const hasFrozenBaseline =', '// Why an extraction failed'),
  ].join('\n');
  return new Function(`${code}\nreturn {
    EXTRACT_SCHEMA, EXTRACT_OP_SCHEMA, kernelSelectionVerified,
    hydrateExtractionVerdicts, hasFrozenBaseline
  };`)();
}

export async function checkDelivery(kernel, result, phase, contract = null) {
  if (!['extract', 'extract_op'].includes(phase)) throw new Error('phase must be extract or extract_op');
  contract ||= await loadContract();
  const schema = validate(result, phase === 'extract' ? contract.EXTRACT_SCHEMA : contract.EXTRACT_OP_SCHEMA);
  const hydrated = await contract.hydrateExtractionVerdicts(result);
  let selection;
  try { selection = contract.kernelSelectionVerified(kernel, hydrated); }
  catch (e) { selection = { ok: false, why: `malformed selection evidence: ${e.message}` }; }
  const smoke = phase === 'extract' ? hydrated?.unittest_smoke : hydrated?.smoke;
  const checks = {
    schema, selection,
    baseline_declaration: { ok: contract.hasFrozenBaseline(hydrated) },
    task: { ok: !!hydrated?.task_dir && hydrated?.editable !== false },
    ut_status: { ok: smoke === 'pass', reported: smoke ?? null },
    source_delivery: await verifySourceDelivery(kernel, hydrated),
  };
  return { ok: Object.values(checks).every(c => c.ok), phase, checks,
    scope: 'delivery contract only; reads recorded evidence, does not run or independently certify GPU UTs' };
}

// The complete AgentX skill owns selection and its intermediate metadata.
// Handoff verifies the completed UT against its source/binding evidence; it
// does not reapply the legacy selection-only contract to a passing task.
export async function checkPreparedUT(kernel, result, phase, contract = null) {
  if (!['extract', 'extract_op'].includes(phase)) throw new Error('phase must be extract or extract_op');
  contract ||= await loadContract();
  const smoke = phase === 'extract' ? result?.unittest_smoke : result?.smoke;
  const selected = kernel.selected_device_kernels || [kernel.device_kernel];
  const phases = kernel.joint_ut?.phases || kernel.dispatch_ut?.phases || [];
  const declared = new Set(phases.flatMap(p => p.required_device_kernels || [p.device_kernel]));
  const checks = {
    baseline_declaration: { ok: contract.hasFrozenBaseline(result) },
    task: { ok: typeof result?.task_dir === 'string' && !!result.task_dir && result.editable !== false },
    ut_status: { ok: smoke === 'pass', reported: smoke ?? null },
    source_delivery: await verifySourceDelivery({ ...kernel, source_delivery_required: true }, result),
    boundary: { ok: Array.isArray(selected) && selected.length > 0 &&
      selected.every(k => typeof k === 'string' && k.trim()) &&
      (phases.length ? selected.every(k => declared.has(k)) : selected.length === 1) },
  };
  return { ok: Object.values(checks).every(check => check.ok), phase, checks,
    scope: 'completed UT/source evidence consistency; selection stays within the skill; does not rerun GPU tests' };
}

export async function attachVerdict(result, file, regime = null) {
  const out = structuredClone(result);
  const task = await fs.realpath(out.task_dir);
  const filename = await fs.realpath(path.resolve(task, file));
  const relative = path.relative(task, filename);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
    throw new Error('selection verdict must be inside task_dir');
  const bytes = await fs.readFile(filename);
  const verdict = JSON.parse(bytes.toString('utf8'));
  if (verdict?.contract !== 'kernel_selection') throw new Error('expected raw kernel_selection.py JSON');
  const rows = regime === null ? [out] : (out.phase_selection_validations || []).filter(r => r.regime === regime);
  if (rows.length !== 1) throw new Error(`expected exactly one existing phase result for ${regime}`);
  const row = rows[0];
  row.selection_validation_ref = { path: relative, sha256: createHash('sha256').update(bytes).digest('hex') };
  delete row.selection_validation;
  return out;
}

async function main(argv) {
  const options = {}, phaseFiles = [];
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === '--help') {
      console.log('node kernel_extraction_delivery.mjs --kernel KERNEL.json --result draft.json --phase extract|extract_op [--prepared] [--verdict task-relative.json] [--phase-verdict REGIME=task-relative.json ...] [--out RETURN.json]');
      return;
    }
    if (flag === '--prepared') { options.prepared = true; continue; }
    if (!['--kernel', '--result', '--phase', '--verdict', '--phase-verdict', '--out'].includes(flag) || !argv[i + 1])
      throw new Error(`unknown or incomplete option: ${flag}`);
    const value = argv[++i];
    if (flag === '--phase-verdict') phaseFiles.push(value);
    else options[flag.slice(2)] = value;
  }
  if (!options.kernel || !options.result || !options.phase)
    throw new Error('--kernel, --result and --phase are required; --kernel is the original KERNEL input');
  const kernel = JSON.parse(await fs.readFile(options.kernel, 'utf8'));
  if (kernel?.entity_kind !== 'gpu_kernel' || !kernel.device_kernel)
    throw new Error('--kernel must contain the original GPU KERNEL object, not a task meta or result');
  let result = JSON.parse(await fs.readFile(options.result, 'utf8'));
  if (options.verdict) result = await attachVerdict(result, options.verdict);
  for (const value of phaseFiles) {
    const split = value.indexOf('=');
    if (split <= 0 || split === value.length - 1) throw new Error('--phase-verdict requires REGIME=path');
    result = await attachVerdict(result, value.slice(split + 1), value.slice(0, split));
  }
  const report = await (options.prepared ? checkPreparedUT : checkDelivery)(kernel, result, options.phase);
  // Never replace a previous deliverable with a failed draft.
  if (report.ok && options.out) await fs.writeFile(options.out, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
  process.exitCode = report.ok ? 0 : 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => {
    console.error(JSON.stringify({ ok: false, error: error.message }));
    process.exitCode = 1;
  });
}
