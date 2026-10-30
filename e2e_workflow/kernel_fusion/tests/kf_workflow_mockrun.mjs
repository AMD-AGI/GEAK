// Runs kf_workflow.js against scripted agent / nested-workflow answers (no GPU, no LLM).
//   node kf_workflow_mockrun.mjs <scenario> <workflow_dir>
// Prints one JSON line: {result, calls: [{label, phase, prompt}], workflows: [args]}.
import { readFile } from 'node:fs/promises';

const [scenario, WF_DIR] = process.argv.slice(2);
const KF = '/tmp/kf_mock/kf';
const calls = [];
const workflows = [];

// c1 (difficulty 3) is recommended; c1s is its smaller exclusive alternative (the fallback).
const CANDS = [
  { id: 'c1', rank: 1, level: 'L2', region: 'route epilogue + prep', kernels: ['joint', 'prep', 'scatter'], difficulty: 3,
    exclusive_with: ['c1s', 'c3'], seam: { target_callable: 'pkg.moe:MoE.forward' }, region_pct_device_time: 4,
    bitwise_exact: true, est_gain_pct_expected: 2.7 },
  { id: 'c2', rank: 2, level: 'L1', region: 'gemm+quant', kernels: ['gemm', 'quant'], difficulty: 2,
    seam: { target_callable: 'pkg.linear:apply' }, region_pct_device_time: 6, est_gain_pct_expected: 1.5 },
  { id: 'c1s', rank: 4, level: 'L1', region: 'prep + scatter', kernels: ['prep', 'scatter'], difficulty: 1,
    exclusive_with: ['c1'], seam: { target_callable: 'pkg.moe:compact' }, est_gain_pct_expected: 1.6 },
  { id: 'c3', rank: 5, level: 'L3', region: 'moe stage', kernels: ['a', 'b'], difficulty: 2, seam: {}, est_gain_pct_expected: 9 },
];
const ISO = { c1: scenario === 'fallback' ? 0.97 : 1.4, c2: 0.95, c1s: 1.6 };
const won = (gate, d) => ({ gate, ab_complete: true, e2e_throughput_tok_s: 100 + d, ref_med: 100, cand_med: 100 + d,
  e2e_delta_pct: d, accepted_overlay: `${KF}/overlay/cand_${gate}` });

const answers = {
  'kf:director:setup': () => (scenario === 'setup_fail' ? null : { eval_dir: KF, baseline_throughput_tok_s: 100, noise_band_pct: 0.5 }),
  'kf:profiler': () => (scenario === 'no_trace' || scenario === 'trace_retry'
    ? { profile_topN_json: '', top_kernels: [], notes: 'capture still running' }
    : { profile_topN_json: `${KF}/profile/round_0/profile_topN.json`, top_kernels: [{ name: 'k' }] }),
  'kf:profiler (retry)': () => (scenario === 'no_trace' ? null
    : { profile_topN_json: `${KF}/profile/round_0/profile_topN.json`, top_kernels: [{ name: 'k' }] }),
  'kf:analyst': () => ({ topk_json: `${KF}/analysis/topk.json`, report_path: `${KF}/analysis/report.md`,
    candidates: CANDS, recommended: ['c1', 'c2'], report_md: '# report', model_notes_md: '# notes' }),
  'kf:integrate c1': () => ({
    accept: won('accepted', 3), stack: won('stack', 0.2),
    retry: { gate: 'incomplete', ab_complete: false, e2e_throughput_tok_s: 0, ref_med: 100 },
  }[scenario] || { gate: 'rejected', e2e_throughput_tok_s: 99, reason: 'parity_regression' }),
  'kf:integrate c1 (finish 1)': () => won('accepted', 2),
  'kf:integrate c1s': () => won('accepted', 1.5),
  'kf:director:validate': () => (scenario === 'stack'
    ? { validation_status: 'validated_no_win', throughput_speedup: 1.001, output_parity: 'pass' }
    : { validation_status: 'validated_win', throughput_speedup: 1.025, director_verified_throughput_tok_s: 102.5, output_parity: 'pass' }),
};
for (const id of Object.keys(ISO)) {
  answers[`kf:extract ${id}`] = () => ({ editable: true, task_dir: `${KF}/kernels/${id}_task`, unittest_smoke: 'pass', short_name: id,
    target_callable: CANDS.find((c) => c.id === id).seam.target_callable,
    baseline_frozen: !(scenario === 'unfrozen' && id === 'c1') });
}

async function agent(prompt, opts) {
  calls.push({ label: opts.label, phase: opts.phase, prompt });
  const f = answers[opts.label];
  if (!f) throw new Error(`unscripted agent ${opts.label}`);
  return f();
}
async function workflow(ref, a) {
  workflows.push({ scriptPath: ref.scriptPath, ...a });
  const id = a.kernel_path.split('/').pop().replace(/_task$/, '');
  return { eval_dir: `${KF}/kernels/_exp/${id}`, final_patch: ISO[id] > 1 ? `diff ${id}` : '', final_geomean: ISO[id],
    validation_status: ISO[id] > 1 ? 'validated' : 'no_gain' };
}
const parallel = (ts) => Promise.all(ts.map((t) => Promise.resolve().then(t).catch(() => null)));
const pipeline = null;
const phase = () => {};
const log = () => {};

const args = { workflow_dir: WF_DIR, kf_dir: KF, model_path: '/models/M', gpu_ids: '0,1', tp: 2,
  kf_prompt: '/prompts/p.txt,/prompts/deploy.txt', agent_timeout_ms: 0, kf_task: 'Do not read /answers.',
  initial_extra_env: 'SGLANG_USE_AITER=1 SGLANG_SIMULATE_ACC_LEN=4.05 SGLANG_SIMULATE_ACC_METHOD=match-expected HF_HUB_OFFLINE=1',
  correctness_unset_env: 'SGLANG_SIMULATE_ACC_LEN,SGLANG_SIMULATE_ACC_METHOD' };
if (scenario === 'analysis_only') Object.assign(args, { kf_steps: 'topk', kf_trace_dir: '/traces/x' });

const src = (await readFile(`${WF_DIR}/kernel_fusion/kf_workflow.js`, 'utf8'))
  .replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', src);
const result = await fn(agent, parallel, pipeline, phase, log, workflow, args, {});
process.stdout.write(JSON.stringify({ result, calls, workflows }) + '\n');
