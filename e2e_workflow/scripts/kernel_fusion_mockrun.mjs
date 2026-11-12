// Runs e2e_workflow.js end to end on GEAK's own runtime (interface/runtime/engine) with a scripted
// backend: every agent answers by label and nested kernel_lane.js runs are stubbed. No GPU, no LLM.
//   node kernel_fusion_mockrun.mjs <scenario> [workflow script]
// KF_MOCK_STATE=<file> hands a previous run's returned state in as args.state (on_resume).
// Prints one JSON line: {ok, error, result, calls: [{label, prompt}], lanes: [{script, ...args}], logs}.
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRuntime } from '../../interface/runtime/engine/run_workflow.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WF_DIR = path.resolve(HERE, '..');
const [scenario = 'off', script = path.join(WF_DIR, 'e2e_workflow.js')] = process.argv.slice(2);
const EV = '/tmp/kf_mock/eval';
const KF_ON = scenario.startsWith('on');
const AGX = scenario.includes('agentx');
const HEAD = scenario.includes('head');
const calls = [];
const lanes = [];
const logs = [];

// The fusion profile: c1 and c1s cover overlapping regions (exclusive); c2 is independent; c3 has no seam.
// A kf_topk handed in by the caller skips the analyst's schema, so on_topk gives c2's seam as a bare string.
const C2_SEAM = 'pkg.norm:fused_norm_quant';
const CANDS = [
  { id: 'c1', rank: 1, level: 'L2', region: 'route epilogue + prep', kernels: ['moe_align_block_size_kernel', 'prep_kernel'],
    exclusive_with: ['c1s'], seam: { target_callable: 'pkg.moe:MoE.forward', source_file: 'pkg/moe.py' },
    region_pct_device_time: 4, bitwise_exact: true, est_gain_pct_expected: 2.7, phases: ['DECODE'] },
  { id: 'c2', rank: 2, level: 'L1', region: 'rmsnorm + quant', kernels: ['rms_norm_kernel', 'quant_kernel'],
    seam: { target_callable: C2_SEAM }, region_pct_device_time: 6, est_gain_pct_expected: 1.5, phases: ['DECODE', 'EXTEND'],
    fused_design: 'one pass: normalize in registers, quantize before the store' },
  { id: 'c1s', rank: 3, level: 'L1', region: 'prep only', kernels: ['prep_kernel'], exclusive_with: ['c1'],
    seam: { target_callable: 'pkg.moe:prep' }, region_pct_device_time: 1, est_gain_pct_expected: 0.4 },
  { id: 'c3', rank: 4, level: 'L3', region: 'moe stage', kernels: ['a', 'b'], seam: {}, est_gain_pct_expected: 9 },
];
// The head scenarios add c4, an L2 region on the head h_gemm's own kernel (GEMM + bias + GELU).
const C4 = { id: 'c4', rank: 5, level: 'L2', region: 'gemm + bias + gelu', kernels: ['gemm_bf16_kernel', 'gelu_kernel'],
  seam: { target_callable: 'pkg.mlp:MLP.forward' }, region_pct_device_time: 22, est_gain_pct_expected: 3 };
const REGIONS = HEAD ? [...CANDS, C4] : CANDS;
const ext = (id) => ({ short_name: id, editable: true, task_dir: `${EV}/tasks/${id}`, unittest_smoke: 'pass',
  baseline_frozen: true, target_callable: `pkg.${id}:f`, source_path_in_sglang: `pkg/${id}.py` });
// A head GEAK's own admission and kernel-selection checks accept: a profiled gpu_kernel with one inner launcher.
const H_SEAM = { target_callable: 'pkg.gemm:gemm', role: 'inner_launcher', depth: 1, device_kernels: ['gemm_bf16_kernel'] };
const H_GEMM = { short_name: 'h_gemm', pct_gpu_time: 20, entity_kind: 'gpu_kernel', device_kernel: 'gemm_bf16_kernel',
  candidate_backends: ['triton'], seam_candidates: [H_SEAM] };
const H_EXT = { short_name: 'h_gemm', op_kind: 'gemm', editable: true, task_dir: `${EV}/tasks/h_gemm`, smoke: 'pass',
  target_callable: H_SEAM.target_callable, baseline_frozen: true, device_kernel: 'gemm_bf16_kernel', dtype: 'bf16',
  shapes: { N: 4096, K: 4096 }, seam_candidates: [H_SEAM], selection_validation: { contract: 'kernel_selection', ok: true,
    target_callable: H_SEAM.target_callable, device_kernel: 'gemm_bf16_kernel', total_calls_observed: 8,
    target_marker_calls: 8, matched_kernel_calls: 8, candidate_targets_tested: [H_SEAM.target_callable],
    live_candidate_targets: [H_SEAM.target_callable], deepest_verified: true } };
const AX_HEAD = { ...H_GEMM, op_kind: 'gemm', prepared_ut: { phase: 'extract_op', result: H_EXT } };
// AgentX: the kernel-extract skill's {delivery_path, prepared} envelope, one validated UT per [name, pct, op_kind].
const prepared = (round, rows, heads = []) => ({ delivery_path: `${EV}/kernel_extraction/round_${round}/delivery.json`,
  prepared: { head_candidates: heads, kernel_candidates: rows.map(([n, pct, kind]) => ({ short_name: n, pct_gpu_time: pct,
    op_kind: kind, prepared_ut: { phase: 'extract', result: { ...ext(n), op_kind: kind } } })) } });
const integ = (name, gate, d, extra = {}) => ({ short_name: name, gate, ab_complete: gate !== 'incomplete',
  e2e_throughput_tok_s: 100 + d, e2e_delta_pct: d, ref_med: 100, cand_med: 100 + d, output_parity: 'pass',
  parity_kind: 'byte_exact', accepted_overlay: ['accepted', 'stack'].includes(gate) ? `${EV}/overlay/cand_${name}` : '',
  reason: '', ...extra });
const fusedInteg = (name, d, extra = {}) => integ(name, 'accepted', d, { parity_kind: 'accuracy', ...extra });

// What the architect routes: a fused region (fusion_id) beside a single kernel. A failed fusion profile
// (on_error, on_config_error) gets exactly off_kernels' answers: its run must be off_kernels' run.
const LIKE_OFF = ['off_kernels', 'on_error', 'on_config_error'].includes(scenario);
const MK_NORM = { short_name: 'mk_norm', pct_gpu_time: 10, device_kernel: 'rms_norm_kernel', candidate_backends: ['triton'] };
const MK_GEMM = { short_name: 'mk_gemm', pct_gpu_time: 12, device_kernel: 'gemm_kernel', candidate_backends: ['triton'] };
const KF_C2 = { short_name: 'kf_c2', fusion_id: 'c2', pct_gpu_time: 6, candidate_backends: ['triton'] };
const KF_C4 = { short_name: 'kf_c4', fusion_id: 'c4', pct_gpu_time: 22, candidate_backends: ['triton'] };
const routed = LIKE_OFF ? [MK_NORM, MK_GEMM] : {
  off: [], off_false: [], off_head: [MK_GEMM],
  on_unknown: [{ short_name: 'k_x', fusion_id: 'zz', classification: 'fusion', pct_gpu_time: 7, candidate_backends: ['triton'] }],
}[scenario] || (HEAD ? [KF_C4, KF_C2] : [KF_C2, MK_GEMM]);
// h_gemm's candidates: the fused kf_c4 (on), the bake-off's env winner and an authored triton kernel.
const headInteg = (src, gate, d, extra = {}) => integ('h_gemm', gate, d,
  { accepted_overlay: gate === 'accepted' ? `${EV}/overlay/cand_h_gemm_${src}` : '', ...extra });

const answers = {
  'director:setup': () => ({ eval_dir: EV, model_name: 'mock-model', baseline_throughput_tok_s: 100, noise_band_pct: 0.5,
    gfx: 'gfx942', device_target: 'mi300x', physical_cu_count: 304 }),
  'profiler:baseline': () => ({ profile_topN_json: `${EV}/profile/round_0/profile_topN.json`,
    profile_workload_json: `${EV}/profile/round_0/profile_workload.json`, top_kernels: [] }),
  'profiler:reprofile m1': () => ({ profile_topN_json: `${EV}/profile/round_1/profile_topN.json`, top_kernels: [] }),
  'profiler:post-head': () => ({ profile_topN_json: `${EV}/profile/round_head/profile_topN.json`, top_kernels: [] }),
  'architect:strategize': () => ({ head_candidates: HEAD ? [H_GEMM] : [], config_directions: [], kernel_candidates: routed }),
  // on_head_unrun: the head's extraction fails before its A/B, so kf_c4 stays queued for the Milestone loop.
  'extract_op h_gemm': () => (scenario === 'on_head_unrun' ? { op_kind: 'gemm', task_dir: '', smoke: 'fail',
    smoke_detail: 'capture failed' } : H_EXT),
  'bakeoff h_gemm': () => ({ gate: 'have_winner', isolated_speedup: 1.2, measured: true, winner_backend: 'aiter',
    winner_kind: 'env', apply_env: 'AITER_TUNED=1', best_known_ms: 0.5, author_plan: [{ language: 'triton', route: 'author' }] }),
  'extract kf_c4': () => ext('kf_c4'),
  // The fused kernel wins (on_head), clears the gate below the bake-off's winner (on_head_loses), does not
  // finish its A/B until the Finalize-gate (on_head_pending), or fails live in a fixable way (on_head_fix).
  'integrate h_gemm:kf_c4': () => ({
    on_head_loses: headInteg('kf_c4', 'accepted', 1, { parity_kind: 'accuracy' }),
    on_head_pending: headInteg('kf_c4', 'incomplete', 0, { reason: 'server never healthy' }),
    on_head_fix: headInteg('kf_c4', 'rejected', 0, { reason: 'cuda_graph_capture_unsafe: host sync on the hot path' }),
  }[scenario] || headInteg('kf_c4', 'accepted', 6, { parity_kind: 'accuracy' })),
  'integrate h_gemm:kf_c4 (finish 1)': () => headInteg('kf_c4', 'incomplete', 0),
  'integrate h_gemm:kf_c4 (finish 2)': () => headInteg('kf_c4', 'incomplete', 0),
  'integrate h_gemm:kf_c4 (finish 3)': () => headInteg('kf_c4', 'incomplete', 0),
  'finish-integrate h_gemm': () => headInteg('kf_c4', 'accepted', 6, { parity_kind: 'accuracy', e2e_throughput_tok_s: 112 }),
  'integrate h_gemm corrective': () => headInteg('fix', 'accepted', 5, { parity_kind: 'accuracy' }),
  'integrate h_gemm:aiter': () => (['on_head_pending', 'on_head_fix'].includes(scenario)
    ? headInteg('aiter', 'rejected', 0, { reason: 'no e2e gain' }) : headInteg('aiter', 'accepted', 3)),
  'integrate h_gemm:triton': () => headInteg('triton', 'rejected', 0, { reason: 'no e2e gain' }),
  'overlay-check h_gemm': () => ({ ok: true, problems: [], manifest_sha256: 'd'.repeat(64), files: 2 }),
  'integrate kf_c4': () => fusedInteg('kf_c4', 3, { e2e_throughput_tok_s: 103 }),
  'overlay-check kf_c4': () => ({ ok: true, problems: [], manifest_sha256: 'e'.repeat(64), files: 2 }),
  'extract mk_norm': () => ext('mk_norm'),
  'extract mk_gemm': () => ext('mk_gemm'),
  'extract kf_c2': () => ext('kf_c2'),
  'extract k_x': () => ext('k_x'),
  'integrate mk_norm': () => integ('mk_norm', 'accepted', 10),
  // off_kernels: mk_gemm's A/B only finishes at the Finalize-gate.
  'integrate mk_gemm': () => (LIKE_OFF ? integ('mk_gemm', 'incomplete', 0, { reason: 'server never healthy' })
    : integ('mk_gemm', 'accepted', 2, { e2e_throughput_tok_s: 104 })),
  'integrate mk_gemm (finish 1)': () => integ('mk_gemm', 'incomplete', 0),
  'integrate mk_gemm (finish 2)': () => integ('mk_gemm', 'incomplete', 0),
  'integrate mk_gemm (finish 3)': () => integ('mk_gemm', 'incomplete', 0),
  'finish-integrate mk_gemm': () => integ('mk_gemm', 'accepted', 20),
  'integrate kf_c2': () => (scenario === 'on_pending' ? integ('kf_c2', 'incomplete', 0, { reason: 'server never healthy' })
    : fusedInteg('kf_c2', 2, HEAD ? { e2e_throughput_tok_s: 110 } : {})),
  'integrate kf_c2 (finish 1)': () => integ('kf_c2', 'incomplete', 0),
  'integrate kf_c2 (finish 2)': () => integ('kf_c2', 'incomplete', 0),
  'integrate kf_c2 (finish 3)': () => integ('kf_c2', 'incomplete', 0),
  'finish-integrate kf_c2': () => fusedInteg('kf_c2', 2, { e2e_throughput_tok_s: 106 }),
  'integrate k_x': () => integ('k_x', 'rejected', 0, { reason: 'no e2e gain' }),
  'overlay-check kf_c2': () => (scenario === 'on_bad_overlay'
    ? { ok: false, problems: ['_kf_c2.py does not compile: SyntaxError'], manifest_sha256: '', files: 3 }
    : { ok: true, problems: [], manifest_sha256: 'b'.repeat(64), files: 3 }),
  'scan-incomplete-ab': () => ({ incomplete: [] }),
  'e2e_integrator:finalize': () => ({ final_overlay: `${EV}/final/overlay`, final_throughput_tok_s: 100 }),
  'architect:report': () => ({ throughput_speedup: 1, report_path: `${EV}/architect_report.md` }),
  'director:validate': () => ({ director_verified_throughput_tok_s: 100, throughput_speedup: 1, validation_status: 'validated',
    output_parity: 'pass' }),
  clock: () => ({ epoch: 1700000000 }),
  // On AgentX the analyst also recommends c3, which has no seam: it gets no region UT.
  'fusion_analyst:analyze': () => (scenario === 'on_error' ? null : { topk_json: `${EV}/kernel_fusion/analysis/topk.json`,
    report_path: `${EV}/kernel_fusion/analysis/REPORT.md`, report_md: '# report', candidates: REGIONS,
    recommended: [...(AGX ? ['c1', 'c2', 'c3'] : ['c1', 'c2']), ...(HEAD ? ['c4'] : [])] }),
  // --- AgentX: one prepared single kernel; the recommended regions get their own region UT ---
  'kernel_extractor:prepare baseline': () => prepared('baseline', [['k_norm', 5, 'norm']], HEAD ? [AX_HEAD] : []),
  'kernel_extractor:prepare kf_c1': () => prepared('kf_c1', [['kf_c1', 4, 'fusion']]),
  'kernel_extractor:prepare kf_c2': () => prepared('kf_c2', [['kf_c2', 6, 'fusion']]),
  'kernel_extractor:prepare kf_c4': () => prepared('kf_c4', [['kf_c4', 22, 'fusion']]),
  'integrate k_norm': () => integ('k_norm', 'accepted', 2, { e2e_throughput_tok_s: 104 }),
  'integrate kf_c1': () => fusedInteg('kf_c1', 1, { e2e_throughput_tok_s: 105 }),
  'overlay-check kf_c1': () => ({ ok: true, problems: [], manifest_sha256: 'c'.repeat(64), files: 4 }),
};

const backend = {
  name: 'claude',
  async runAgent({ prompt, label }) {
    calls.push({ label, prompt });
    if (String(label).startsWith('persist-e2e-checkpoint:')) {
      const target = (/at "([^"]+)"/.exec(prompt) || [])[1] || '';
      return { text: JSON.stringify({ written: true, path: target, checkpoint_sha256: 'a'.repeat(64) }) };
    }
    const a = answers[label];
    return { text: JSON.stringify(a ? a(prompt) : null) };
  },
};
const rt = createRuntime({ backend, concurrency: 16, schemaRetries: 0, log: (m) => { logs.push(String(m)); } });
const workflow = async (ref, wfArgs) => {
  lanes.push({ script: (ref && ref.scriptPath) || '', ...wfArgs });
  const short = String(wfArgs.kernel_path || '').split('/').pop();
  const iso = { kf_c1: 1.4, kf_c2: 1.3, kf_c4: 1.5 }[short] || 1.1;
  return { eval_dir: `${EV}/kernels/_exp/${short}`, final_patch: `${EV}/kernels/_exp/${short}/final_patch.diff`,
    final_geomean: iso, final_weighted: iso, final_speedup: iso, validation_status: 'accepted', winner: { source: 'triton' },
    report_path: `${EV}/kernels/_exp/${short}/tech_lead_report.md` };
};
const args = {
  workflow_dir: WF_DIR, kernel_workflow_dir: path.resolve(WF_DIR, '..', 'kernel_workflow'),
  model_path: '/models/mock', launch_script: '/models/launch.sh', gpu_ids: '0,1', expected_gfx: 'gfx942',
  expected_target: 'mi300x', backend: 'sglang', isl: 1024, osl: 256, conc: 16,
  ...(KF_ON ? { kernel_fusion: 'true' } : {}),
  ...(AGX ? { workload_kind: 'agentx_trace_replay' } : {}),
  ...(scenario === 'off_false' ? { kernel_fusion: 'false' } : {}),
  ...(scenario === 'on_fast' ? { fast_mode: 'true' } : {}),
  ...(scenario === 'on_deep' ? { deep_mode: 'true' } : {}),
  ...(scenario === 'on_config_error' ? { kf_config_errors: 'kf_trace_dir does not exist: /nope' } : {}),
  ...(scenario === 'on_topk' ? { kf_topk_json: '/data/topk.json', kf_topk: { recommended: ['c1', 'c2'],
    candidates: CANDS.map((c) => (c.id === 'c2' ? { ...c, seam: C2_SEAM } : c)) } } : {}),
  ...(scenario === 'on_setup' ? { phases: 'setup' } : {}),
  ...(scenario === 'on_resume' ? { phases: 'kernel,final',
    state: JSON.parse(await readFile(process.env.KF_MOCK_STATE, 'utf8')) } : {}),
};

const body = (await readFile(script, 'utf8')).replace(/^\uFEFF/, '')
  .replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
let result = null, error = '';
try {
  result = await new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', body)(
    rt.agent, rt.parallel, rt.pipeline, rt.phase, rt.log, workflow, args, rt.budget);
} catch (e) {
  error = String((e && e.stack) || e);
}
// Exit (pending agent-timeout timers would keep the loop alive) only once a piped stdout has flushed.
process.stdout.write(JSON.stringify({ ok: !error, error, result, calls, lanes, logs }) + '\n', () => process.exit(0));
