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
const ISO = { c1: scenario === 'fallback' ? 0.97 : 1.4, c2: (scenario === 'accept' || scenario === 'stack_fail' || scenario === 'reuse') ? 1.2 : 0.95, c1s: 1.6 };
// the select agent's picks (it sees every UT result; the script enforces only the exclusivity line)
// default: every UT result that beats stock, in the order given
const SELECT = { select_conflict: ['c1', 'c1s', 'c2'], select_slow: ['c1', 'c2'] }[scenario] || null;
const won = (gate, d) => ({ gate, ab_complete: true, e2e_throughput_tok_s: 100 + d, ref_med: 100, cand_med: 100 + d,
  e2e_delta_pct: d, accepted_overlay: `${KF}/overlay/cand_${gate}` });

const AGX = scenario.startsWith('agentx');
const prepared = (id, ok = true) => ({ short_name: id, op_kind: 'fusion', pct_gpu_time: 3,
  prepared_ut: { phase: 'extract', result: { task_dir: `${KF}/tasks/${id}`, editable: true, unittest_smoke: ok ? 'pass' : 'fail',
    baseline_frozen: true, target_callable: `pkg.${id}:f` } } });
const answers = {
  'kf:kernel-extract trace': () => (scenario === 'agentx_no_trace' ? { trace_dir: '', workload_json: '', points: [] }
    : { trace_dir: `${KF}/kernel_extraction/round_kf/capture/traces`, workload_json: `${KF}/kernel_workload.json`,
        ranking_json: `${KF}/kernel_ranking.json`, points: [{ point_id: 'de_1', phase: 'DECODE', verified: true }] }),
  'kf:kernel-extract trace (retry)': () => ({ trace_dir: '', workload_json: '', points: [] }),
  // one prepare per region: c1 / c1s pass; c2 fails twice (the retry is told why)
  // agentx_flat: a hand-assembled delivery (packer unavailable) carries the UT result on the entry itself
  'kf:kernel-extract prepare c1': () => ({ delivery_path: `${KF}/d_c1.json`, prepared: scenario === 'agentx_flat'
    ? { head_candidates: [{ short_name: 'c1', phase: 'extract_op', result: { task_dir: `${KF}/tasks/c1`, editable: true,
        smoke: 'pass', baseline_frozen: true, target_callable: 'pkg.c1:f' } }], kernel_candidates: [] }
    : { head_candidates: [], kernel_candidates: [prepared('c1')] } }),
  'kf:kernel-extract prepare c1s': () => ({ delivery_path: `${KF}/d_c1s.json`, prepared: { head_candidates: [], kernel_candidates: [] } }),
  'kf:kernel-extract prepare c1s (retry)': () => ({ delivery_path: `${KF}/d_c1s.json`, prepared: { head_candidates: [], kernel_candidates: [prepared('c1s')] } }),
  'kf:kernel-extract prepare c2': () => ({ delivery_path: `${KF}/d_c2.json`,
    prepared: { head_candidates: [], kernel_candidates: [], failed_tasks: [{ task: 'c2', reason: 'operands not reconstructed' }] } }),
  'kf:kernel-extract prepare c2 (retry)': () => ({ delivery_path: `${KF}/d_c2.json`, prepared: { head_candidates: [], kernel_candidates: [prepared('c2', false)] } }),
  'kf:director:setup': () => (scenario === 'setup_fail' ? null
    : { eval_dir: KF, baseline_throughput_tok_s: scenario === 'detached_baseline' ? 0 : 100, noise_band_pct: 0.5 }),
  'kf:baseline wait 1': () => ({ status: 'running' }),
  'kf:baseline wait 2': () => ({ status: 'done', baseline_throughput_tok_s: 286.9 }),
  'kf:profiler': () => (scenario === 'no_trace' || scenario === 'trace_retry'
    ? { profile_topN_json: '', top_kernels: [], notes: 'capture still running' }
    : { profile_topN_json: `${KF}/profile/round_0/profile_topN.json`, top_kernels: [{ name: 'k' }] }),
  'kf:profiler (retry)': () => (scenario === 'no_trace' ? null
    : { profile_topN_json: `${KF}/profile/round_0/profile_topN.json`, top_kernels: [{ name: 'k' }] }),
  // string_seam: the analyst's structured result gives seam as the bare "pkg.mod:fn" (topk.json on disk keeps the dict)
  'kf:opportunity-analysis': () => ({ topk_json: `${KF}/analysis/topk.json`, report_path: `${KF}/analysis/report.md`,
    candidates: scenario.includes('string_seam')
      ? CANDS.map((c) => (c.seam && c.seam.target_callable ? { ...c, seam: c.seam.target_callable } : c)) : CANDS,
    recommended: ['c1', 'c2'], report_md: '# report', model_notes_md: '# notes' }),
  // which patterns get a kernel + UT: the agent's call (c3 has no seam, the script drops it)
  'kf:feasibility-review': () => (scenario === 'triage_none' ? null
    : scenario === 'no_seam' ? { selected: [{ id: 'c3' }], skipped: [] }
    : { selected: [{ id: 'c1' }, { id: 'c1s' }, { id: 'c2' }, { id: 'c3' }], skipped: [] }),
  'kf:integration-review': (prompt) => {
    const ut = JSON.parse(prompt.split('## UT results\n')[1].split('\n')[0]);
    const ids = SELECT || ut.filter((u) => u.isolated_speedup > 1).map((u) => u.id);
    return { selected: ids.map((id) => ({ id, reason: 'mock' })), skipped: [{ id: 'c1s', reason: 'overlaps c1' }] };
  },
  // only after the whole stack fails accuracy: each kernel alone, correctness only
  'kf:integrate screen': () => ({ members: [{ short_name: 'c1', correctness: 'pass' },
    { short_name: 'c2', correctness: scenario === 'stack_fail' ? 'fail' : 'pass', reason: 'garbage output' }] }),
  'kf:integrate kf_stack (screened)': () => ({ ...won('accepted', 1.5), accepted_overlay: `${KF}/overlay/cand_kf_stack` }),
  // ONE combined A/B of every winner
  'kf:integrate kf_stack': () => ({
    reject: { gate: 'rejected', e2e_throughput_tok_s: 99, ref_med: 100, cand_med: 99, e2e_delta_pct: -1, ab_complete: true, reason: 'slower' },
    parity_reject: { gate: 'rejected', e2e_throughput_tok_s: 100, ref_med: 100, ab_complete: false, output_parity: 'fail', reason: 'parity' },
    stack_fail: { gate: 'rejected', e2e_throughput_tok_s: 0, ref_med: 100, ab_complete: false, reason: 'correctness: gsm8k 0.80 < 0.93' },
    stack: { ...won('stack', -0.1), accepted_overlay: `${KF}/overlay/cand_kf_stack` },
    small_gain: { ...won('stack', 0.22), accepted_overlay: `${KF}/overlay/cand_kf_stack` },
    retry: { gate: 'incomplete', ab_complete: false, e2e_throughput_tok_s: 0, ref_med: 100 },
  }[scenario] || { ...won('accepted', 2.5), accepted_overlay: `${KF}/overlay/cand_kf_stack` }),
  'kf:integrate kf_stack (finish 1)': () => ({ ...won('accepted', 2), accepted_overlay: `${KF}/overlay/cand_kf_stack` }),
};
for (const id of Object.keys(ISO)) {
  answers[`kf:extract ${id}`] = () => ({ editable: true, task_dir: `${KF}/kernels/${id}_task`, unittest_smoke: 'pass', short_name: id,
    target_callable: CANDS.find((c) => c.id === id).seam.target_callable,
    baseline_frozen: !(scenario === 'unfrozen' && id === 'c1') });
}

async function agent(prompt, opts) {
  if ('timeoutMs' in opts) throw new Error('timeoutMs must not reach the agent() call');
  calls.push({ label: opts.label, phase: opts.phase, prompt, ...(opts.label === 'kf:opportunity-analysis' ? { schema: opts.schema } : {}) });
  if (opts.label.startsWith('kf:author ')) {
    // the fused kernel is written by one author agent against the region UT
    const kernel_path = prompt.match(/TASK_DIR=(\S+)/)[1];
    workflows.push({ kernel_path, task: prompt });
    const id = kernel_path.split('/').pop().replace(/_task$/, '');
    return { final_patch: `diff ${id}`, isolated_speedup: ISO[id], correctness: 'pass', notes: '',
      overlay_dir: `${KF}/kernels/${id}/overlay`, banner: `[kf-${id}] ENGAGED` };
  }
  const f = answers[opts.label];
  if (!f) throw new Error(`unscripted agent ${opts.label}`);
  return f(prompt);
}
async function workflow() { throw new Error('KernelFusion must not start a nested GEAK workflow'); }
const parallel = (ts) => Promise.all(ts.map((t) => Promise.resolve().then(t).catch(() => null)));
const pipeline = null;
const phase = () => {};
const log = () => {};

const args = { workflow_dir: WF_DIR, kf_dir: KF, model_path: '/models/M', gpu_ids: '0,1', tp: 2,
  ...(AGX ? { workload_spec: { kind: 'agentx_trace_replay', concurrency: 4 } } : {}),
  kf_prompt: '/prompts/p.txt,/prompts/deploy.txt', agent_timeout_ms: 0, kf_task: 'Do not read /answers.',
  initial_extra_env: 'SGLANG_USE_AITER=1 SGLANG_SIMULATE_ACC_LEN=4.05 SGLANG_SIMULATE_ACC_METHOD=match-expected HF_HUB_OFFLINE=1',
  correctness_unset_env: 'SGLANG_SIMULATE_ACC_LEN,SGLANG_SIMULATE_ACC_METHOD' };
if (scenario === 'analysis_only') Object.assign(args, { kf_steps: 'topk', kf_trace_dir: '/traces/x' });
if (scenario === 'reuse') args.kf_reuse = { 'kf:author c1': { final_patch: 'old c1', isolated_speedup: 1.3, correctness: 'pass',
  overlay_dir: '/old/c1/overlay', banner: '[kf-c1] ENGAGED' } };
if (scenario === 'reuse_topk') Object.assign(args, { kf_baseline_dir: '/old/kf/baseline', kf_trace_dir: '/traces/x',
  kf_topk_json: '/old/kf/analysis/topk.json', kf_topk: { candidates: CANDS, recommended: ['c1', 'c2'] } });
if (scenario === 'reuse_baseline') Object.assign(args, { kf_baseline_dir: '/old/kf/baseline', kf_trace_dir: '/traces/x' });

const src = (await readFile(`${WF_DIR}/kernel_fusion/kf_workflow.js`, 'utf8'))
  .replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', src);
const result = await fn(agent, parallel, pipeline, phase, log, workflow, args, {});
process.stdout.write(JSON.stringify({ result, calls, workflows }) + '\n');
