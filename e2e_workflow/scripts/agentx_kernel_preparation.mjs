import * as fs from 'node:fs/promises';
import path from 'node:path';
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
        pct_gpu_time: share, source_delivery_required: true,
        source_device_kernels: [...new Set([...(kernel.source_device_kernels || []),
          ...(kernel.joint_ut?.phases || []).flatMap(p => p.required_device_kernels || [])])] };
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
