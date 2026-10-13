#!/usr/bin/env node
// Standalone adaptation, 2026-10-12: CPU preflight uses only bundled modules.
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { validate } from './schema.mjs';
import { verifySourceDelivery } from './kernel_source_delivery.mjs';

import * as standaloneContract from './delivery_contract.mjs';

export async function loadContract() { return standaloneContract; }

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
      console.log('node kernel_extraction_delivery.mjs --kernel KERNEL.json --result draft.json --phase extract|extract_op [--verdict task-relative.json] [--phase-verdict REGIME=task-relative.json ...] [--out RETURN.json]');
      return;
    }
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
  if (kernel.source_delivery_required !== true)
    throw new Error('standalone extraction requires source_delivery_required=true on the saved request');
  let result = JSON.parse(await fs.readFile(options.result, 'utf8'));
  if (options.verdict) result = await attachVerdict(result, options.verdict);
  for (const value of phaseFiles) {
    const split = value.indexOf('=');
    if (split <= 0 || split === value.length - 1) throw new Error('--phase-verdict requires REGIME=path');
    result = await attachVerdict(result, value.slice(split + 1), value.slice(0, split));
  }
  const report = await checkDelivery(kernel, result, options.phase);
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
