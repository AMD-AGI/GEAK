// Orchestrator: run kernel_workflow.js on a LIST of kernels sequentially, all pinned to ONE GPU.
// Launch one of these per GPU (4 total) for 4-way parallelism. Each kernel's kernel_workflow runs
// as a nested child workflow (its agents are subagents). Output goes to <exp_base>/<kernel_name>.
//
// args = {
//   workflow_dir: "<abs path to kernel_workflow/>",   // holds kernel_workflow.js + roles/ + knowledge/
//   exp_base:     "<abs shared exp dir>",        // all kernels share this; each writes a <name>/ subdir
//   gpu:          "0",                           // single GPU id this batch is pinned to
//   budget:       6,                             // per-kernel optimization budget
//   kernels:      ["/abs/kernel_dir", ...]       // kernel dirs to process on this GPU, in order
// }
export const meta = {
  name: 'team-workflow-bmk-batch',
  description: 'Run kernel_workflow on a list of kernels sequentially on one GPU (one batch per GPU).',
  phases: [{ title: 'Optimize', detail: 'nested kernel_workflow per kernel, sequential on one GPU' }],
};

const A = args || {};
const WF = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WF) throw new Error('args.workflow_dir is required');
const EXP_BASE = String(A.exp_base || '').replace(/\/+$/, '');
if (!EXP_BASE) throw new Error('args.exp_base is required');
const GPU = String(A.gpu != null ? A.gpu : '0');
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
const TASK = A.task != null ? String(A.task) : '';
const KERNELS = Array.isArray(A.kernels) ? A.kernels : [];
if (!KERNELS.length) throw new Error('args.kernels must be a non-empty array');

const TEAM = WF + '/kernel_workflow.js';

phase('Optimize');
log(`[gpu ${GPU}] ${KERNELS.length} kernels: ${KERNELS.map(k => k.split('/').pop()).join(', ')}`);

const results = [];
for (let i = 0; i < KERNELS.length; i++) {
  const k = String(KERNELS[i]).replace(/\/+$/, '');
  const name = k.split('/').pop();
  log(`[gpu ${GPU}] (${i + 1}/${KERNELS.length}) START ${name}`);
  let res = null;
  try {
    res = await workflow({ scriptPath: TEAM }, {
      kernel_path: k,
      workflow_dir: WF,
      budget: BUDGET,
      gpu_ids: GPU,
      eval_dir: EXP_BASE + '/' + name,
      apply_to_original: 'false',
      task: TASK,
    });
  } catch (e) {
    res = { error: String(e && e.message || e) };
    log(`[gpu ${GPU}] ERROR ${name}: ${res.error}`);
  }
  results.push({ name, eval_dir: EXP_BASE + '/' + name, result: res });
  log(`[gpu ${GPU}] (${i + 1}/${KERNELS.length}) DONE ${name}`);
}

return { gpu: GPU, count: KERNELS.length, results };
