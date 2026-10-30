import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { checkPreparedUT } from './kernel_extraction_delivery.mjs';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
// Keep whole-op routing consistent with the head consumer's vocabulary.
function opKind(kernel, result) {
  const canonical = value => value === 'attention' ? 'attn' : value;
  const requested = canonical(kernel.op_kind), returned = canonical(result.op_kind);
  if (requested && returned && requested !== returned)
    throw new Error(`op_kind differs between declaration and result: ${requested} / ${returned}`);
  return returned || requested;
}
async function within(root, name) {
  const file = await fs.realpath(path.resolve(root, name));
  const relative = path.relative(root, file);
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
    throw new Error(`preparation artifact must be inside EVAL_DIR: ${name}`);
  return file;
}

// PD: the cross-engine denominator is `mn.py pd-mass --apply`'s (both engines' GPU-time
// mass x ranks) -- never one the extraction agent chose, and never a graded axis.
function pdEngineWeights(ranking) {
  const weights = ranking.denominators || {};
  if (weights.rule !== 'gpu_time_mass')
    throw new Error('PD kernel_ranking.json denominators were not written by `mn.py pd-mass --apply`; ' +
      'run it on the ranking before select_targets.py');
  return weights;
}

// Selection is an internal artifact. Read the skill's completed UT deliveries
// once, then expose only validated tasks to the ordinary optimization queues.
// engines: a PD run's engine names; each task then names the one its kernel was traced on.
export async function readPreparedKernels(evalDir, filename, { engines = [] } = {}) {
  const root = await fs.realpath(evalDir);
  const deliveryPath = await within(root, filename);
  const delivery = JSON.parse(await fs.readFile(deliveryPath, 'utf8'));
  return readPreparedDocument(root, delivery, deliveryPath, engines);
}

async function readPreparedDocument(root, delivery, deliveryPath, engines = []) {
  if (!Array.isArray(delivery.tasks)) throw new Error('UT delivery requires tasks[]');
  const workloadPath = await within(root, delivery.workload_json);
  const rankingPath = await within(root, delivery.ranking_json);
  const ranking = JSON.parse(await fs.readFile(rankingPath, 'utf8'));
  const engineWeights = engines.length ? pdEngineWeights(ranking) : undefined;
  const output = { head_candidates: [], kernel_candidates: [], config_directions: [],
    ...(engineWeights ? { engine_weights: engineWeights } : {}),
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
      if (engines.length && !engines.includes(kernel.engine))
        throw new Error(`requested_kernel.json engine must name the engine it was traced on (${engines.join(' | ')}), got ${JSON.stringify(kernel.engine)}`);
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
        op_kind: opKind(kernel, result), kernel_extraction: evidence,
        pct_gpu_time: share, source_delivery_required: true };
      if (task.phase === 'extract_op' && !candidate.op_kind) throw new Error('whole-op task requires op_kind');
      const checked = await checkPreparedUT(candidate, result, task.phase);
      if (!checked.ok) {
        output.failed_tasks.push({ short_name: label, task_dir: result.task_dir,
          result_path: resultPath, checks: checked.checks });
        continue;
      }
      // Under PD the same kernel on the other engine is another machine's, another task.
      const owned = (candidate.selected_device_kernels || [candidate.device_kernel])
        .map(id => engines.length ? `${kernel.engine}:${id}` : id);
      for (const id of owned) {
        if (selectedOwners.has(id)) throw new Error(`selected kernel already owned by ${selectedOwners.get(id)}: ${id}`);
      }
      for (const id of owned) selectedOwners.set(id, label);
      // `result` travels WITH the envelope, not just its path: the workflow script runs in a
      // sealed sandbox with no filesystem (see e2e_workflow.js "RUNTIME SANDBOX CONTRACT"), so a
      // path it cannot open is not evidence it can act on. This object is the one this function
      // just validated through checkPreparedUT, so carrying it forward hands the optimization
      // lanes exactly what passed the gate instead of a promise to re-read it later.
      candidate.prepared_ut = { phase: task.phase, result_path: resultPath,
        sha256: sha256(bytes), result };
      // Existing optimization lanes lease one GPU. A validated multi-rank UT
      // remains a deliverable, but cannot be dispatched to that single-card lane.
      const gpuCount = kernel.required_gpu_count ?? result.required_gpu_count ??
        (candidate.op_kind === 'collective' ? 2 : 1);
      if (!Number.isInteger(gpuCount) || gpuCount < 1) throw new Error('invalid required_gpu_count');
      if (gpuCount > 1) {
        output.deferred_tasks.push({ ...candidate, task_dir: result.task_dir,
          result_path: resultPath, required_gpu_count: gpuCount,
          reason: 'UT validated; optimization requires a multi-GPU lane, current head/milestone lanes lease one GPU' });
        continue;
      }
      const queue = task.phase === 'extract_op' && ['gemm', 'attn', 'moe'].includes(candidate.op_kind)
        ? 'head_candidates' : 'kernel_candidates';
      output[queue].push(candidate);
    } catch (error) {
      output.failed_tasks.push({ short_name: label, result_path: task?.result_path, reason: error.message });
    }
  }
  return output;
}

// NOTE: there is deliberately no re-reading reuse helper here any more. Dispatch happens inside
// e2e_workflow.js, which runs in a sandbox with no filesystem (see its "RUNTIME SANDBOX CONTRACT"),
// so it cannot call one -- it replays this module's verdict via the result carried in
// prepared_ut.result and re-asserts the filesystem-free invariants itself (preparedUtDefect).
// A helper here that nothing could call would be an untestable second opinion: readPreparedKernels
// and packStandaloneTasks are the single gate, and a task edited after validation must fail there.

// Translate the standalone skill's saved task declarations and returns into the
// existing workflow envelope. Selection and GPU tests remain owned by the skill.
export async function packStandaloneTasks(evalDir, manifestPath, outputPath, { engines = [] } = {}) {
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
  const prepared = await readPreparedDocument(root, delivery, destination, engines);
  const invalid = prepared.failed_tasks.slice((manifest.failed_tasks || []).length);
  if (invalid.length) throw new Error(`completed tasks have invalid evidence: ${JSON.stringify(invalid)}`);
  await fs.writeFile(destination, JSON.stringify(delivery, null, 2) + '\n');
  // `prepared` is the SAME document the invalid-task gate above was computed from — returning it
  // costs nothing and is the only way the sandboxed workflow can see it (it cannot read the file).
  return { delivery_path: destination, prepared };
}

async function main(argv) {
  const options = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (key === '--help') {
      console.log('node agentx_kernel_preparation.mjs --eval-dir EVAL --manifest workflow_tasks.json --out ut_delivery.json [--emit-prepared] [--engines prefill,decode]');
      return;
    }
    // --emit-prepared also prints the workflow envelope this packing already computed. The
    // workflow script cannot read the delivery file itself, so this is how it receives it.
    if (key === '--emit-prepared') { options['--emit-prepared'] = true; continue; }
    if (!['--eval-dir', '--manifest', '--out', '--engines'].includes(key) || !argv[i + 1])
      throw new Error(`unknown or incomplete option: ${key}`);
    options[key] = argv[++i];
  }
  if (!options['--eval-dir'] || !options['--manifest'] || !options['--out'])
    throw new Error('--eval-dir, --manifest and --out are required');
  const engines = options['--engines'] ? options['--engines'].split(',').filter(Boolean) : [];
  const packed = await packStandaloneTasks(options['--eval-dir'], options['--manifest'], options['--out'], { engines });
  console.log(JSON.stringify(options['--emit-prepared']
    ? packed
    : { delivery_path: packed.delivery_path }));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => {
    console.error(JSON.stringify({ ok: false, error: error.message }));
    process.exitCode = 1;
  });
}
