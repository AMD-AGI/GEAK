import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { checkPreparedUT } from './kernel_extraction_delivery.mjs';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
async function within(root, name) {
  const file = await fs.realpath(path.resolve(root, name));
  const relative = path.relative(root, file);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
    throw new Error(`preparation artifact must be inside EVAL_DIR: ${name}`);
  return file;
}

// Selection is an internal artifact. Read the skill's completed UT deliveries
// once, then expose only validated tasks to the ordinary optimization queues.
export async function readPreparedKernels(evalDir, filename) {
  const root = await fs.realpath(evalDir);
  const deliveryPath = await within(root, filename);
  const delivery = JSON.parse(await fs.readFile(deliveryPath, 'utf8'));
  return readPreparedDocument(root, delivery, deliveryPath);
}

async function readPreparedDocument(root, delivery, deliveryPath) {
  if (!Array.isArray(delivery.tasks)) throw new Error('UT delivery requires tasks[]');
  const workloadPath = await within(root, delivery.workload_json);
  const rankingPath = await within(root, delivery.ranking_json);
  const ranking = JSON.parse(await fs.readFile(rankingPath, 'utf8'));
  const output = { head_candidates: [], kernel_candidates: [], config_directions: [],
    profile_workload_json: workloadPath, kernel_ranking_json: rankingPath,
    kernel_extraction_delivery_json: deliveryPath,
    failed_tasks: Array.isArray(delivery.failed_tasks) ? delivery.failed_tasks.slice() : [],
    deferred_tasks: Array.isArray(delivery.deferred_tasks) ? delivery.deferred_tasks.slice() : [],
    coverage_summary: delivery.coverage_summary ?? {}, notes: delivery.notes ?? '' };
  const seen = new Set();
  const selectedOwners = new Map();
  for (const task of delivery.tasks) {
    let label = task?.kernel?.short_name || task?.kernel?.ut_task || task?.result_path || '(unnamed)';
    try {
      const kernel = task.kernel;
      if (!kernel || typeof kernel.device_kernel !== 'string' || !kernel.device_kernel.trim())
        throw new Error('completed UT requires its selected device_kernel');
      if (!['extract', 'extract_op'].includes(task.phase)) throw new Error('UT phase must be extract or extract_op');
      const resultPath = await within(root, task.result_path);
      const bytes = await fs.readFile(resultPath);
      const result = JSON.parse(bytes.toString('utf8'));
      await within(root, result.task_dir);
      label = kernel.short_name || kernel.ut_task || path.basename(result.task_dir);
      if (seen.has(label)) throw new Error(`duplicate UT task: ${label}`);
      seen.add(label);
      // Cost/link fields are carried from existing evidence, not a second
      // selection decision or a reason to regenerate a passing UT.
      const anchor = (ranking.ranking || []).find(row =>
        (row.device_kernel || row.name) === kernel.device_kernel);
      const evidence = { selection_pct_gpu_time: anchor?.selection_pct_gpu_time,
        ...(kernel.kernel_extraction || {}), workload_json: workloadPath, ranking_json: rankingPath };
      const share = evidence.selection_pct_gpu_time;
      if (!Number.isFinite(share) || share < 0 || share > 100)
        throw new Error('selected kernel has no attributed cost in its saved ranking');
      const candidate = { ...kernel, short_name: label, entity_kind: 'gpu_kernel',
        target_callable: result.target_callable,
        op_kind: kernel.op_kind || result.op_kind, kernel_extraction: evidence,
        pct_gpu_time: share, source_delivery_required: true };
      const checked = await checkPreparedUT(candidate, result, task.phase);
      if (!checked.ok) {
        output.failed_tasks.push({ short_name: label, task_dir: result.task_dir,
          result_path: resultPath, checks: checked.checks });
        continue;
      }
      for (const id of candidate.selected_device_kernels || [candidate.device_kernel]) {
        if (selectedOwners.has(id)) throw new Error(`selected kernel already owned by ${selectedOwners.get(id)}: ${id}`);
      }
      for (const id of candidate.selected_device_kernels || [candidate.device_kernel]) selectedOwners.set(id, label);
      candidate.prepared_ut = { phase: task.phase, result_path: resultPath, sha256: sha256(bytes) };
      const queue = task.phase === 'extract_op' && ['gemm', 'attn', 'moe'].includes(candidate.op_kind)
        ? 'head_candidates' : 'kernel_candidates';
      output[queue].push(candidate);
    } catch (error) {
      output.failed_tasks.push({ short_name: label, result_path: task?.result_path, reason: error.message });
    }
  }
  return output;
}

// Downstream code consumes a prepared task; it does not invoke the extractor
// again. Editing candidate code or its recorded result invalidates this reuse.
export async function reusePreparedUT(kernel) {
  try {
    const ref = kernel.prepared_ut;
    const bytes = await fs.readFile(ref.result_path);
    if (sha256(bytes) !== ref.sha256) throw new Error('prepared UT result changed after validation');
    const result = JSON.parse(bytes.toString('utf8'));
    const checked = await checkPreparedUT({ ...kernel, source_delivery_required: true }, result, ref.phase);
    if (!checked.ok) throw new Error(JSON.stringify(checked.checks));
    return { ...result, smoke: 'pass', unittest_smoke: 'pass' };
  } catch (error) {
    return { smoke: 'fail', unittest_smoke: 'fail',
      smoke_detail: `prepared UT is no longer valid: ${error.message}` };
  }
}

// Translate the standalone skill's saved task declarations and returns into the
// existing workflow envelope. Selection and GPU tests remain owned by the skill.
export async function packStandaloneTasks(evalDir, manifestPath, outputPath) {
  const root = await fs.realpath(evalDir);
  const input = await within(root, manifestPath);
  const manifest = JSON.parse(await fs.readFile(input, 'utf8'));
  if (!Array.isArray(manifest.tasks)) throw new Error('manifest requires tasks[]');
  for (const field of ['failed_tasks', 'deferred_tasks']) {
    if (manifest[field] !== undefined && !Array.isArray(manifest[field]))
      throw new Error(`${field} must be an array`);
  }
  const output = path.resolve(root, outputPath);
  const parent = await fs.realpath(path.dirname(output));
  const relative = path.relative(root, parent);
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
    throw new Error('handoff output must stay inside EVAL_DIR');
  const destination = path.join(parent, path.basename(output));
  const protectedFiles = new Set([input,
    await within(root, manifest.workload_json), await within(root, manifest.ranking_json)]);
  const delivery = { ...manifest, tasks: [] };
  for (const row of manifest.tasks) {
    if (!['extract', 'extract_op'].includes(row.phase))
      throw new Error('task phase must be extract or extract_op');
    const task = await within(root, row.task_dir);
    const requestPath = await within(root, path.join(task, 'run/requested_kernel.json'));
    const resultPath = await within(root, path.join(task, 'run/RETURN.json'));
    protectedFiles.add(requestPath);
    protectedFiles.add(resultPath);
    const kernel = JSON.parse(await fs.readFile(requestPath, 'utf8'));
    const result = JSON.parse(await fs.readFile(resultPath, 'utf8'));
    if (await fs.realpath(result.task_dir) !== task)
      throw new Error(`RETURN.json task_dir differs from its declared task: ${task}`);
    delivery.tasks.push({ kernel, phase: row.phase, result_path: path.relative(root, resultPath) });
  }
  if (protectedFiles.has(destination)) throw new Error('handoff output would overwrite source evidence');
  // Refuse symlinks as output: never follow a link out of the evaluation root or
  // replace a referenced input through an alias.
  try {
    if ((await fs.lstat(destination)).isSymbolicLink()) throw new Error('handoff output must not be a symlink');
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const prepared = await readPreparedDocument(root, delivery, destination);
  const invalid = prepared.failed_tasks.slice((manifest.failed_tasks || []).length);
  if (invalid.length) throw new Error(`completed tasks have invalid evidence: ${JSON.stringify(invalid)}`);
  await fs.writeFile(destination, JSON.stringify(delivery, null, 2) + '\n');
  return { delivery_path: destination };
}

async function main(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (key === '--help') {
      console.log('node agentx_kernel_preparation.mjs --eval-dir EVAL --manifest workflow_tasks.json --out ut_delivery.json');
      return;
    }
    if (!['--eval-dir', '--manifest', '--out'].includes(key) || !argv[i + 1])
      throw new Error(`unknown or incomplete option: ${key}`);
    options[key] = argv[++i];
  }
  if (!options['--eval-dir'] || !options['--manifest'] || !options['--out'])
    throw new Error('--eval-dir, --manifest and --out are required');
  console.log(JSON.stringify(await packStandaloneTasks(options['--eval-dir'], options['--manifest'], options['--out'])));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => {
    console.error(JSON.stringify({ ok: false, error: error.message }));
    process.exitCode = 1;
  });
}
