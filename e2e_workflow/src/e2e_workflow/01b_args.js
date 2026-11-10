const RDNA4_ISA = EXPECTED_GFX === 'gfx1201';
// Products with a `product:` section in roofline/peaks.md; keep in sync with it.
const RDNA4_SERVING_PRODUCTS = ['r9700', 'r9600d'];
const RDNA4_PRODUCT_E2E = RDNA4_SERVING_PRODUCTS.includes(EXPECTED_TARGET);
// ISA isolation and product policy are separate: any gfx1201 target avoids CDNA priors, but only
// an exact product identity selects calibrated peaks, serving backend and image policy.
const RDNA4_ISOLATE = RDNA4_ISA;
const E2E_LEARNED_KB_ENABLED = !RDNA4_ISOLATE;
const MATRIX_CORE_NAME = RDNA4_ISOLATE ? 'WMMA' : 'MFMA';

// Lanes launched from here default the learned KB OFF (kernel_workflow alone defaults it ON): e2e
// extracts whatever the profiler surfaces, where another run's card for a similar-looking op is an
// unwanted prior. Override with args.use_learned_kb=true.
const LANE_USE_LEARNED_KB = String(A.use_learned_kb != null ? A.use_learned_kb : 'false');
// Select the policy inside the existing worker, without another workflow level.
const LANE_OPTIMIZER = A.optimizer == null || A.optimizer === '' ? 'evo' : A.optimizer;
if (!['greedy', 'evo'].includes(LANE_OPTIMIZER)) {
  throw new Error('optimizer must be greedy or evo');
}
const EXPLICIT_KERNEL_BUDGET = typeof A.kernel_budget === 'string' && /^\+?\d+$/.test(A.kernel_budget.trim())
  ? Number(A.kernel_budget) : A.kernel_budget;
if (A.kernel_budget !== undefined && (!Number.isSafeInteger(EXPLICIT_KERNEL_BUDGET) || EXPLICIT_KERNEL_BUDGET < 1)) {
  throw new Error('kernel_budget must be a positive safe integer');
}

// Every nested lane call builds its args here, so none silently takes the lane's own defaults;
// test_e2e_lane_defaults.py fails on a `scriptPath: KERNEL_WF_SCRIPT` call that doesn't.
// llm_stats is forwarded only when the parent got one explicitly. wfArgs spreads last so a
// per-call value wins (RDNA4 isolation still forces KB off).
const laneArgs = (wfArgs) => {
  const out = {
    use_learned_kb: LANE_USE_LEARNED_KB,
    optimizer: LANE_OPTIMIZER,
    // The worker enables these only for boolean true; do not coerce caller values.
    ...(A.evolution_case_evidence_view !== undefined
      ? { evolution_case_evidence_view: A.evolution_case_evidence_view } : {}),
    ...(A.evolution_evidence_handoff !== undefined
      ? { evolution_evidence_handoff: A.evolution_evidence_handoff } : {}),
    ...(A.evolution_preserve_founder_attempts !== undefined
      ? { evolution_preserve_founder_attempts: A.evolution_preserve_founder_attempts } : {}),
    // The worker validates the EVO schedule selector itself.
    ...(A.evolution_start !== undefined ? { evolution_start: A.evolution_start } : {}),
    ...Object.fromEntries(['evolution_return_development', 'evolution_import_stall_sources',
      'record_planner_inputs', 'case_measurement_view'].filter(key => A[key] !== undefined)
      .map(key => [key, A[key]])),
    ...(A.llm_stats != null ? { llm_stats: String(A.llm_stats) } : {}),
    ...wfArgs,
  };
  if (RDNA4_ISOLATE) {
    out.use_learned_kb = 'false';
    out.use_expert_skills = 'false';
    out.perf_knowledge_dir = '';
    out.warm_start = 'off';
  }
  if (EXPECTED_GFX) out.expected_gfx = out.expected_gfx || EXPECTED_GFX;
  if (EXPECTED_TARGET) out.expected_target = out.expected_target || EXPECTED_TARGET;
  if (EXPECTED_DEVICE_NAME) out.expected_device_name = out.expected_device_name || EXPECTED_DEVICE_NAME;
  if (EXPECTED_PHYSICAL_CU_COUNT > 0) {
    out.expected_physical_cu_count =
      out.expected_physical_cu_count || EXPECTED_PHYSICAL_CU_COUNT;
  }
  return out;
};

// EXP_ROOT = where timestamped run dirs go. Default: sibling "exp/" next to this workflow dir.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

// ---- Live execution tracker (read-only observer; see interface/geak_trace_collector.py) ----
// Started as soon as the run has a durable identity, so the timeline covers the first agent.
// Detached and non-fatal (a failed start is logged and ignored); it never influences the run.
// GEAK_LIVE_TRACE=0 disables it.
startLiveTracker(EXP_ROOT, WORKFLOW_DIR);
function startLiveTracker(expRoot, wfDir) {
  try {
    if (String(process.env.GEAK_LIVE_TRACE || '1') === '0') return;
    const { spawn } = require('child_process');
    const interval = String(process.env.GEAK_LIVE_TRACE_INTERVAL_S || '30');
    const child = spawn('python3', [
      '-B', `${wfDir}/../interface/geak_trace_collector.py`,
      '--exp-root', expRoot,
      // The runtime's wf_*.json record carries these same args, so they identify this launch.
      '--identity-args', JSON.stringify(A),
      '--script-dir', wfDir,
      // Per-run filename: two runs under one exp_root must not overwrite each other.
      '--out-dir', expRoot,
      '--watch', '--interval', interval,
      '--resolve-timeout', '600',
      '--max-seconds', String(process.env.GEAK_LIVE_TRACE_MAX_S || '172800'),
    ], { detached: true, stdio: 'ignore' });
    // spawn reports a missing executable (ENOENT) asynchronously; try/catch can't see it.
    child.on('error', (err) => {
      try { log(`Live execution tracker failed to start (non-fatal): ${err && err.message}`); }
      catch (_) {}
    });
    child.unref();
    log(`Live execution tracker started -> ${expRoot}/geak_trace_<runId>.json`);
  } catch (e) {
    log(`Live execution tracker not started (non-fatal): ${e && e.message}`);
  }
}

// ---- Profile-analysis skill (OPTIONAL, pluggable; see knowledge/analysis_skills/INDEX.md) ----
// The Profiler may run ONE analysis skill on parse_profile.py's Top-N to add a headroom estimate
// (default `roofline`: % of hardware ceiling -> attainable e2e gain), so budget goes to headroom, not
// just size. Advisory only: never prunes or overrides pct_gpu_time. `none`/`false` disables it.
const ANALYSIS_SKILL = String(A.analysis_skill != null ? A.analysis_skill : 'roofline').trim();
const ANALYSIS_SKILL_ON = !!ANALYSIS_SKILL && ANALYSIS_SKILL !== 'none' && ANALYSIS_SKILL !== 'false';
const ANALYSIS_SKILL_INPUTS = ANALYSIS_SKILL_ON ? {
  ANALYSIS_SKILL: ANALYSIS_SKILL,
  ANALYSIS_SKILL_DIR: `${WORKFLOW_DIR}/knowledge/analysis_skills/${ANALYSIS_SKILL}`,
} : { ANALYSIS_SKILL: '', ANALYSIS_SKILL_DIR: '' };
if (ANALYSIS_SKILL_ON) log(`Profile-analysis skill: ${ANALYSIS_SKILL} (advisory; annotates + reorders, never prunes).`);

// ---- Upstream TraceLens / kernel-agent prior (OPTIONAL; forwarded by run_e2e.py as args.tracelens) ----
// Only non-null paths are sent. With analysis_md the Profiler builds the Top-N from TraceLens instead
// of its own trace (plus a parse_profile pass on trace_file if present); the Architect uses
// kernel_candidates as a routing prior. Absent => every TRACELENS_* input is ''.
const TL = (A.tracelens && typeof A.tracelens === 'object') ? A.tracelens : {};
const TRACELENS_INPUTS = {
  TRACELENS_ANALYSIS_MD: String(TL.analysis_md || ''),
  TRACELENS_KERNEL_CANDIDATES_JSON: String(TL.kernel_candidates_json || ''),
  TRACELENS_REPORT_JSON: String(TL.tracelens_report_json || ''),
  TRACELENS_TRACE_FILE: String(TL.trace_file || ''),
};
if (TL && Object.keys(TL).length) log(`TraceLens prior present: ${Object.keys(TL).filter(k => TL[k]).join(', ') || '(none non-null)'}.`);

// ---- single-kernel pass-through: if kernel_path (and no model_path), just run the kernel layer ----
const KERNEL_PATH = A.kernel_path || '';
const MODEL_PATH = A.model_path || '';
if (!MODEL_PATH && !KERNEL_PATH) {
  throw new Error('Provide args.model_path (e2e mode) OR args.kernel_path (single-kernel pass-through).');
}
if (MODEL_PATH && (!EXPECTED_GFX || !EXPECTED_TARGET)) {
  throw new Error(
    'E2E serving requires structured GPU identity before policy selection: pass expected_gfx and ' +
    'expected_target (run interface/run_e2e.py for auto-detection).');
}
if (MODEL_PATH && RDNA4_ISA && !RDNA4_PRODUCT_E2E) {
  throw new Error(
    'gfx1201 architecture detected but the product is not confirmed as one of ' +
    `${RDNA4_SERVING_PRODUCTS.join('/')}; refusing the product-scoped serving ` +
    'policy for an unknown gfx1201 product.');
}

const LAUNCH_SCRIPT = A.launch_script || '';
const BACKEND = String(A.backend != null ? A.backend : (RDNA4_PRODUCT_E2E ? 'vllm' : 'sglang')).trim()
  || (RDNA4_PRODUCT_E2E ? 'vllm' : 'sglang');  // serving adapter
if (RDNA4_PRODUCT_E2E && BACKEND !== 'vllm') {
  throw new Error(`${EXPECTED_TARGET.toUpperCase()} E2E is vLLM-only (got backend=${BACKEND}): there is no validated ${EXPECTED_TARGET.toUpperCase()} image for it; pass args.backend=vllm`);
}
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Serving TP + GPU set, used for EVERY e2e serving launch (baseline, sweep, integrate, validate,
// profile) and kept fixed so deltas stay comparable. Distinct from GPU_LIST, the pool for isolated
// benchmarks and kernel lanes. Default TP=1; SERVING_GPU defaults to the first TP ids of GPU_LIST.
const SERVING_TP = parseInt(A.tp != null ? A.tp : (A.serving_tp != null ? A.serving_tp : 1), 10);
const SERVING_GPU = String(A.serving_gpu != null ? A.serving_gpu
  : GPU_LIST.slice(0, Math.max(1, SERVING_TP)).join(',') || '0');
// Expert-parallel degree of the BASELINE launch; run_e2e.py always sends it (_resolve_ep). It is a
// KB addressing dimension beside TP; the Config Tuner may change it mid-run, and the accepted value
// lives in accepted_config. With launch details but no args.ep, fail rather than guess: a wrong ep
// files the results under another deployment.
function resolveServingEp() {
  if (A.ep != null) {
    const ep = Number(A.ep);
    if (!Number.isInteger(ep) || ep < 1) throw new Error('args.ep must be a positive integer');
    return ep;
  }
  if (!A.initial_extra_server_args && !A.initial_extra_env && !A.launch_recipe) return 1;
  throw new Error('args.ep is required when initial_extra_server_args / initial_extra_env / ' +
    'launch_recipe are passed: the baseline expert-parallel degree cannot be derived from them ' +
    'here (no subprocess in the workflow runtime), and it addresses the KB page beside tp. ' +
    'Launch through interface/run_e2e.py, which resolves it with _resolve_ep, or pass args.ep.');
}
const SERVING_EP = resolveServingEp();
// ---- WALL-CLOCK BUDGET (opt-in) ----
// time_budget_s is the orchestrator's hard kill budget (run_e2e.py GEAK_E2E_TIMEOUT_S), so GEAK can
// pace itself and finish Finalize/Report/Validate before the SIGKILL. Interpreted only here; absent
// => TIME_BUDGET_MS is null and every budget branch short-circuits.
const TIME_BUDGET_MS = A.time_budget_s != null ? parseInt(A.time_budget_s, 10) * 1000 : null;
// ---- ELAPSED CLOCK ----
// Date.now() is unavailable (breaks resume), so elapsed time comes from timers armed all at once at
// absolute offsets: a self-rearming chain compounds scheduler lateness and drifts behind real time.
// No time_budget_s => remainingMs() is Infinity.
let ELAPSED_MS = 0;
const CLOCK_TICK_MS = 60000;
const CLOCK_MAX_RUNGS = 2048;   // an absurd budget widens the step instead of flooding the event loop
if (TIME_BUDGET_MS != null && typeof setTimeout === 'function') {
  const step = Math.ceil(TIME_BUDGET_MS / CLOCK_TICK_MS) <= CLOCK_MAX_RUNGS
    ? CLOCK_TICK_MS
    : Math.ceil(TIME_BUDGET_MS / CLOCK_MAX_RUNGS);
  // One rung PAST the budget so remainingMs() can reach 0.
  for (let at = step; at <= TIME_BUDGET_MS + step; at += step) {
    const mark = at;
    const t = setTimeout(() => { if (mark > ELAPSED_MS) ELAPSED_MS = mark; }, mark);   // max: stays monotonic
    if (t && t.unref) t.unref();
  }
}
const remainingMs = () => (TIME_BUDGET_MS == null ? Infinity : Math.max(0, TIME_BUDGET_MS - ELAPSED_MS));
const remainingMin = () => (TIME_BUDGET_MS == null ? Infinity : Math.round(remainingMs() / 60000));
// ---- FINAL-PHASE RESERVE ----
// Wall-clock held back for the whole final phase (Finalize + Report + Validate + workflow_return);
// no mode starts new work inside it. Default 60min, capped at 20% of the budget (binds below 5h).
// An overrun only risks Validate's re-measure (Report already wrote the reports; run_e2e falls back).
// run_e2e.py env overrides: GEAK_FINAL_RESERVE_S, GEAK_FINAL_RESERVE_CAP_FRAC (clamped to 5-50%).
const FINAL_RESERVE_CAP_FRAC = Math.min(0.5, Math.max(0.05,
  parseFloat(A.final_reserve_cap_frac != null ? A.final_reserve_cap_frac : 0.2) || 0.2));
// A trace replay's final phase (Finalize's search leg, Validate's canonical legs, 40min for the rest)
// outranks a smaller reserve and the cap. Read from raw args: it must resolve AgentX as IS_AGENTX does.
const RESERVE_WORKLOAD_SPEC = A.workload_spec && typeof A.workload_spec === 'object'
  ? A.workload_spec
  : ((!A.workload_kind || A.workload_kind === 'agentx_trace_replay') &&
      A.state?.workload_spec?.kind === 'agentx_trace_replay'
    ? A.state.workload_spec : {});
const RESERVE_IS_AGENTX =
  String(A.workload_kind || RESERVE_WORKLOAD_SPEC.kind || 'synthetic') === 'agentx_trace_replay';
const AGENTX_LEG_OVERHEAD_S = Math.max(0,
  parseInt(A.agentx_leg_overhead_s != null ? A.agentx_leg_overhead_s : 1200, 10) || 0);
const agentxLegS = (windowS) => windowS + AGENTX_LEG_OVERHEAD_S;
const _declaredWindowS = (v, dflt) => (Number(v) > 0 ? Number(v) : dflt);
const RESERVE_DURATION_S = _declaredWindowS(RESERVE_WORKLOAD_SPEC.duration_s, 3600);
const RESERVE_LOOP_S = _declaredWindowS(RESERVE_WORKLOAD_SPEC.geak_loop_duration_s, 900);
const AGENTX_VALIDATION_LEGS = 2 * (String(A.validation_measurement_mode || 'warm_server') === 'warm_server'
  ? 1 : Math.max(1, parseInt(A.validation_replicas != null ? A.validation_replicas : 3, 10) || 1));
const AGENTX_FINAL_PHASE_MS = !RESERVE_IS_AGENTX ? 0
  : 1000 * (2400 + agentxLegS(RESERVE_LOOP_S) + AGENTX_VALIDATION_LEGS * agentxLegS(RESERVE_DURATION_S));
// Validate's timed windows alone: the least time in which it can return a number.
const AGENTX_VALIDATE_WINDOWS_MS = !RESERVE_IS_AGENTX ? 0 : 1000 * AGENTX_VALIDATION_LEGS * RESERVE_DURATION_S;
// The run's minimum: Setup's leg (+10min), the KB warm start's A/B, then the final phase.
const AGENTX_AB_OVERHEAD_S = 600;   // the integrator's work around an A/B's legs
const AGENTX_PRE_FINAL_MIN_MS = !RESERVE_IS_AGENTX ? 0
  : 1000 * (600 + agentxLegS(RESERVE_DURATION_S) + 2 * agentxLegS(RESERVE_LOOP_S) + AGENTX_AB_OVERHEAD_S);
const AGENTX_MIN_BUDGET_MS = AGENTX_PRE_FINAL_MIN_MS + AGENTX_FINAL_PHASE_MS;
const FINAL_RESERVE_MS = TIME_BUDGET_MS != null
  ? (() => {
    const configured = Math.min(
      parseInt(A.final_reserve_s != null ? A.final_reserve_s : 3600, 10) * 1000, // 60min
      Math.floor(TIME_BUDGET_MS * FINAL_RESERVE_CAP_FRAC));
    return RESERVE_IS_AGENTX
      ? Math.max(configured, Math.min(AGENTX_FINAL_PHASE_MS, TIME_BUDGET_MS - AGENTX_PRE_FINAL_MIN_MS))
      : configured;
  })()
  : null;
// Effective budget = budget minus the reserve: one definition of time available to optimize.
const TIME_BUDGET_EFFECTIVE_MS = TIME_BUDGET_MS != null
  ? Math.max(60000, TIME_BUDGET_MS - FINAL_RESERVE_MS)
  : null;
// Cap on the dispatch-deadline tail (budget − deadline). Deadlines are max(60% of budget,
// budget − cap): small budgets keep the 60% rule, large ones don't idle ~40%. Default 3h.
const TIME_TAIL_CAP_MS = parseInt(A.time_tail_cap_s != null ? A.time_tail_cap_s : 10800, 10) * 1000; // 3h
// Point after which no mode starts new head/milestone/wave work. Defined here (not at the deadline timer)
// because deep mode references it far earlier; TIME_DEADLINE_HIT is armed on it below.
const TIME_HEAD_DEADLINE_MS = TIME_BUDGET_EFFECTIVE_MS != null
  ? Math.max(Math.floor(TIME_BUDGET_EFFECTIVE_MS * 0.6), TIME_BUDGET_EFFECTIVE_MS - TIME_TAIL_CAP_MS) : null;
if (RESERVE_IS_AGENTX && TIME_BUDGET_MS != null) {
  log(`[time-budget] trace replay: final phase ~${Math.round(AGENTX_FINAL_PHASE_MS / 60000)}min, reserving ` +
    `${Math.round(FINAL_RESERVE_MS / 60000)}min of ${Math.round(TIME_BUDGET_MS / 60000)}min; the run's minimum ` +
    `is ~${Math.round(AGENTX_MIN_BUDGET_MS / 60000)}min.` +
    (TIME_BUDGET_MS < AGENTX_MIN_BUDGET_MS
      ? ' WARNING: the budget is under it, so this run may not return a validated gain — raise time_budget_s.'
      : ''));
}
// ---- FAST MODE (opt-in, default OFF) ----
// Time-boxed run (default 5h) that optimizes only through the HeadKernel track: it skips ConfigSweep,
// TuningSkillset and Milestone. Every fast knob is a `FAST_MODE ? fast : default` ternary, so with
// fast_mode off nothing below changes.
const FAST_MODE = String(A.fast_mode != null ? A.fast_mode : 'false') === 'true';
// Fast-run wall-clock budget (default 5h), enforced by timers: a deadline flag stops new head ops,
// and each nested head workflow has its own time bound.
let FAST_BUDGET_MS = parseInt(A.fast_budget_ms != null ? A.fast_budget_ms : 18000000, 10); // 5h
// An orchestrator budget (time_budget_s minus the final reserve) replaces the default and fast_budget_ms.
if (TIME_BUDGET_EFFECTIVE_MS != null) FAST_BUDGET_MS = TIME_BUDGET_EFFECTIVE_MS;
// No new head ops after this point, so in-flight heads and the final phase fit in FAST_BUDGET_MS.
// Default max(60% of the budget, budget − TIME_TAIL_CAP_MS).
const FAST_HEAD_DEADLINE_MS = parseInt(A.fast_head_deadline_ms != null ? A.fast_head_deadline_ms
  : Math.max(Math.floor(FAST_BUDGET_MS * 0.6), FAST_BUDGET_MS - TIME_TAIL_CAP_MS), 10);
// Per-head nested workflow time cap (fast mode only; per parallel GPU lane) so one recursive kernel
// run can't eat the budget. Default 90min: a non-trivial DSL author can need ~80min.
const FAST_HEAD_WF_MS = parseInt(A.fast_head_workflow_ms != null ? A.fast_head_workflow_ms : 5400000, 10);
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);       // max kernel-optimization tasks
// Dispatch at least this many editable-kernel tasks before the loop may stop on no-improve or an
// empty queue. Capped by BUDGET.
const MIN_KERNEL_TASKS = Math.min(parseInt(A.min_kernel_tasks != null ? A.min_kernel_tasks : 4, 10), BUDGET);
// Milestone skips editable kernels below this pct_gpu_time (Amdahl: a few % can't beat the e2e noise
// band). Overrides MIN_KERNEL_TASKS: if no candidate clears it, Milestone stops.
const MILESTONE_MIN_PCT = parseFloat(A.milestone_min_pct != null ? A.milestone_min_pct : 5);
const KERNEL_BUDGET = EXPLICIT_KERNEL_BUDGET !== undefined ? EXPLICIT_KERNEL_BUDGET : (FAST_MODE ? 3 : 6); // budget passed DOWN per kernel (fewer rounds in fast mode)
const CONFIG_TUNE_ENABLED = String(A.config_tune != null ? A.config_tune : 'true') === 'true';
// ---- TUNING SKILLSET phase (default ON) ----
// The vendored tuning skillset (hash-pinned whole by scripts/tuning_skillset_sync.py) is run by one
// role (tuning_specialist) as one phase between ConfigSweep and HeadKernel (that phase says why).
// tuning_skillset="false" disables it entirely.
const TUNING_SKILLSET_ENABLED = !RDNA4_ISOLATE && String(A.tuning_skillset != null ? A.tuning_skillset : 'true') === 'true';
// Under expert_skills/ so its skills are selected through the same index.yaml as every other expert
// skill; the index entries live outside the pinned tree.
const TUNING_SKILLSET_DIR = String(A.tuning_skillset_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge/expert_skills/tuning')).replace(/\/+$/, '');
// tuning-kb/ is the skillset's per-model answer key (verified wins + artifacts): fine in production,
// contaminating in a blind eval. tuning_kb="false" tells the role not to read it.
const TUNING_KB_ENABLED = !RDNA4_ISOLATE && String(A.tuning_kb != null ? A.tuning_kb : 'true') === 'true';
// No tuning op budget on purpose: tuning ops are cheap and cumulative; the role decides when to stop.
// Head-kernel track (GEMM/attention) — the highest-pct_gpu_time ops, optimized regardless of edit flag.
const HEAD_THRESHOLD_PCT = parseFloat(A.head_threshold_pct != null ? A.head_threshold_pct : 5);
// Max head-op bake-offs (default 3; fast mode: max(3, GPU count) to keep every parallel lane busy).
const HEAD_BUDGET = parseInt(A.head_budget != null ? A.head_budget : (FAST_MODE ? Math.max(3, GPU_LIST.length) : 3), 10);
// Languages to author per head op, in the Op Benchmarker's ROI order (GEMM: flydsl, then triton).
// Default 2 in every mode; raise it to try hip/ck when the headroom justifies the cost.
const HEAD_AUTHOR_MAX = parseInt(A.head_author_max != null ? A.head_author_max : 2, 10);
// An op at or above this pct_gpu_time is never silently skipped: a bake-off fault, no-win or failed
// extraction is flagged loudly, and the author route still runs when a plan exists.
const HEAD_PROTECT_PCT = parseFloat(A.head_protect_pct != null ? A.head_protect_pct : 30);
// Corrective re-author: a head winner rejected at the e2e gate for a fixable integration posture
// (e.g. lazy JIT compile in TP>1 warmup, a host sync that breaks graph capture) gets a targeted fix of
// the existing kernel and one more A/B. Not charged to HEAD_BUDGET; skipped after the kernel deadline.
// Per-head cap; 0 disables.
const HEAD_CORRECTIVE_MAX = parseInt(A.head_corrective_max != null ? A.head_corrective_max : 2, 10);
// SURGICAL-FIX tier (default ON): before the hours-long kernel_workflow re-author, one kernel_surgeon
// agent makes the smallest fix for the reject diagnosis, self-verifies on the immutable unittest and
// re-gates (most rejects are tiny seam bugs). Escalates only if that fails; surgical_fix=false skips it.
const SURGICAL_FIX = String(A.surgical_fix != null ? A.surgical_fix : 'true') === 'true';
const FIXABLE_REJECT_RX = /cuda_graph_capture_unsafe|no[_ ]?binary|NO_BINARY_FOR_GPU|hipErrorNoBinaryForGpu|capture[_ ]?(unsafe|hang)|host[_ ]?sync|graph[_ ]?capture|no[_ ]?rebind[_ ]?seam|no[_ ]?engagement|not[_ ]?engaged|signature[_ ]?mismatch|wrong[_ ]?seam/i;
// ---- CORRECTNESS-class reject (auto-correct) ----
// The candidate engaged and beat the isolated oracle but gives wrong output on the live path
// (parity/accuracy failure), typically by over-fitting the captured snapshot (e.g. a data_ptr()-keyed
// cache, a stale index/mask), which a single-snapshot unittest can't catch. The corrective loop feeds
// this diagnosis to the re-author. A surprising speedup is a measurement review, not wrong output.
const CORRECTNESS_REJECT_RX = /parit|corrupt|mismatch|diverge|garbage|degener|\bnan\b|\binf\b|accuracy[_ ]?regress|wrong[_ ]?output|incorrect/i;
// Amdahl is only an estimate (GPU time is not wall time; an overlay can touch several profile rows),
// so exceeding it by this margin requests a measurement review, not a reject.
const IMPLAUSIBLE_SPEEDUP_MARGIN = parseFloat(A.implausible_speedup_margin != null ? A.implausible_speedup_margin : 1.0);
function amdahlCeilingPct(pct_gpu_time, isolated) {
  const p = Math.max(0, Math.min(1, (pct_gpu_time || 0) / 100));
  const s = (isolated && isolated > 1) ? isolated : 1;
  if (p <= 0 || s <= 1) return Infinity;   // unknown inputs -> never flag (fail-open)
  return (1 / (1 - p * (1 - 1 / s)) - 1) * 100;
}
// Only accuracy-gated accepts get an independent A/B review. The estimate is advisory: exceeding it
// never rejects a candidate by itself and never triggers a kernel correctness rewrite.
function parityIsSoft(integ) {
  const pk = integ && integ.parity_kind;               // 'byte_exact' | 'accuracy' | 'none' (optional)
  if (pk === 'accuracy') return true;
  if (pk === 'byte_exact' || pk === 'none') return false;
  return ACCURACY_GATE !== 'none';                      // unknown -> soft only when the run uses an accuracy gate
}
function speedupEstimate(pct_gpu_time, isolated, integ) {
  const fallback = { pct_gpu_time, estimate_pct: amdahlCeilingPct(pct_gpu_time, isolated), source: 'head_estimate' };
  const coverage = integ && integ.affected_gpu_time;
  const m = integ && integ.measurement_evidence;
  if (!coverage || !m || !coverage.profile_path || !m.candidate_digest || !m.reference_digest || !m.workload_digest
      || coverage.candidate_digest !== m.candidate_digest
      || coverage.reference_digest !== m.reference_digest || coverage.workload_digest !== m.workload_digest
      || !Array.isArray(coverage.kernels) || !coverage.kernels.length) return fallback;
  // IDs identify disjoint rows in ONE profile/denominator (kernel + shape + phase). Do not count a
  // module total together with its children, or duplicate a split-K reduction already in a group.
  const seen = new Map();
  let pct = 0, saved = 0, uniform = false;
  for (const row of coverage.kernels) {
    if (!row || typeof row.id !== 'string' || !row.id || typeof row.pct_gpu_time !== 'number'
        || !Number.isFinite(row.pct_gpu_time) || row.pct_gpu_time < 0 || row.pct_gpu_time > 100) return fallback;
    const signature = JSON.stringify([row.pct_gpu_time, row.isolated_speedup, row.removed === true]);
    if (seen.has(row.id)) {
      if (seen.get(row.id) !== signature) return fallback;
      continue;
    }
    seen.set(row.id, signature);
    pct += row.pct_gpu_time;
    if (row.removed === true) saved += row.pct_gpu_time / 100;
    else {
      let s = row.isolated_speedup;
      if (!(typeof s === 'number' && Number.isFinite(s) && s > 0)) { s = isolated; uniform = true; }
      if (!(typeof s === 'number' && Number.isFinite(s) && s > 0)) return fallback;
      saved += row.pct_gpu_time / 100 * (1 - 1 / s);
    }
  }
  if (!(pct > 0) || pct > 100 + 1e-9) return fallback;
  return { pct_gpu_time: pct, estimate_pct: saved >= 1 ? Infinity : (1 / (1 - saved) - 1) * 100,
    source: uniform ? 'coverage_uniform_estimate' : 'coverage_per_path' };
}
function speedupNeedsReview(pct_gpu_time, isolated, integ) {
  if (!parityIsSoft(integ)) return false;
  const ceilPct = speedupEstimate(pct_gpu_time, isolated, integ).estimate_pct;
  if (!Number.isFinite(ceilPct)) return false;
  return ((integ && integ.e2e_delta_pct) || 0) > ceilPct * (1 + IMPLAUSIBLE_SPEEDUP_MARGIN) + 1e-9;
}
function logAmdahlDiagnostic(pct_gpu_time, isolated, integ) {
  const ceilPct = speedupEstimate(pct_gpu_time, isolated, integ).estimate_pct;
  const deltaPct = integ && integ.e2e_delta_pct;
  if (Number.isFinite(ceilPct) && Number.isFinite(deltaPct)
      && deltaPct > ceilPct * (1 + IMPLAUSIBLE_SPEEDUP_MARGIN) + 1e-9) {
    log(`  [amdahl diagnostic] measured +${deltaPct.toFixed(1)}% exceeds conditional +${ceilPct.toFixed(1)}% estimate. ` +
      'Check timing boundary, overlap, work completeness and quality evidence; model discrepancy alone does not change the gate.');
  }
}
function confirmationEvidenceMatches(integ) {
  const review = integ && integ.speedup_review;
  const before = review && review.original_measurement;
  const after = integ && integ.measurement_evidence;
  if (!review || review.status !== 'confirmed' || !before || !after) return false;
  // Equal numbers are allowed; re-reading the same artifacts is not independent evidence.
  if (!before.measurement_id || !after.measurement_id || before.measurement_id === after.measurement_id
      || !before.ref_summary || !before.cand_summary || !after.ref_summary || !after.cand_summary
      || new Set([before.ref_summary, before.cand_summary, after.ref_summary, after.cand_summary]).size !== 4
      || after.fresh_pair !== true || after.configs_matched !== true || after.workload_matched !== true
      || !Number.isInteger(after.ref_replicas) || !Number.isInteger(after.cand_replicas)
      || after.ref_replicas < 2 || after.cand_replicas < 2) return false;
  // Older evidence may lack protocol_digest on both sides; if either side has it, they must match.
  if ((before.protocol_digest || after.protocol_digest) && before.protocol_digest !== after.protocol_digest) return false;
  return ['candidate_digest', 'reference_digest', 'workload_digest'].every(
    (key) => typeof before[key] === 'string' && before[key].length > 0 && before[key] === after[key]);
}
function confirmedSpeedup(integ) {
  return !!(integ && integ.gate === 'accepted' && integ.ab_complete === true
    && integ.output_parity === 'pass' && ['accuracy', 'byte_exact'].includes(integ.parity_kind)
    && confirmationEvidenceMatches(integ));
}
// Classify a reject reason into a fix-and-retry class ('' = terminal, not auto-correctable).
// Posture is tested first: CORRECTNESS_REJECT_RX's bare `mismatch` would otherwise claim
// `signature_mismatch`, which needs the integration fix. The posture tokens are specific enough not to
// capture a genuine output-correctness reject.
function rejectClass(reason) {
  const r = reason || '';
  // Review diagnostics (and implausible_speedup rejects) may mention accuracy or a config mismatch;
  // they never warrant a kernel re-author.
  if (/^(speedup_review_|implausible_speedup)/i.test(r)) return '';
  if (FIXABLE_REJECT_RX.test(r)) return 'integration';
  if (CORRECTNESS_REJECT_RX.test(r)) return 'correctness';
  return '';
}
// Measured eligibility: a complete accepted/stack A/B without a parity failure. A speedup review
// withholds it only when it ran and returned 'rejected'; pending, unreviewed or inconclusive
// reviews are recorded evidence. The estimate itself is advisory for every parity kind.
function integEligible(integ) {
  return !!(integ && (integ.gate === 'accepted' || integ.gate === 'stack')
    && integ.ab_complete !== false && integ.output_parity !== 'fail'
    && !(integ.speedup_review && integ.speedup_review.status === 'rejected'));
}
// integEligible, logging the advisory estimate discrepancy of an eligible result.
function integAccepted(integ, pct_gpu_time, isolated) {
  const accepted = integEligible(integ);
  if (accepted) logAmdahlDiagnostic(pct_gpu_time, isolated, integ);
  return accepted;
}
// The integrator's evidence-backed reason; neither the estimate nor a review fabricates a failure.
function gateRejectReason(integ, pct_gpu_time, isolated) {
  if (integ && integ.output_parity === 'fail') return 'parity_regression';
  return integ ? (integ.reason || integ.gate || '') : '';
}
// ---- DEEP MODE (opt-in, default OFF) ----
// Long HeadKernel mode: backends co-optimize each head op across waves (kernel_workflow STATE_DIR keeps
// progress), sharing a curated blackboard KB, toward a roofline target; an adaptive, batched e2e gate
// validates the best candidates between waves. Every knob is gated on DEEP_MODE; excludes fast mode.
const DEEP_MODE = String(A.deep_mode != null ? A.deep_mode : 'false') === 'true';
let DEEP_HEAD_BUDGET_MS = parseInt(A.deep_head_budget_ms != null ? A.deep_head_budget_ms : 86400000, 10); // 24h for the whole HeadKernel module
// An orchestrator budget replaces the 24h default. Deep uses the shared dispatch deadline so its burst
// drain, final e2e gate and the final phase still fit before the kill.
if (TIME_HEAD_DEADLINE_MS != null) DEEP_HEAD_BUDGET_MS = TIME_HEAD_DEADLINE_MS;
const DEEP_HEAD_WF_MS = parseInt(A.deep_head_workflow_ms != null ? A.deep_head_workflow_ms : 4500000, 10); // per-burst nested kernel_workflow time cap (75min); bounds the burst, not the gate
const DEEP_E2E_GAIN_TRIGGER = parseFloat(A.deep_e2e_gain_trigger != null ? A.deep_e2e_gain_trigger : 0.08); // isolated-best improvement since last e2e gate that triggers a new (batched) gate
const DEEP_E2E_MAX_INTERVAL_MS = parseInt(A.deep_e2e_max_interval_ms != null ? A.deep_e2e_max_interval_ms : 7200000, 10); // force an e2e gate at least this often when a new candidate exists (default 2h)
const DEEP_PLATEAU_STREAK = parseInt(A.deep_plateau_streak != null ? A.deep_plateau_streak : 2, 10); // consecutive non-improving waves before a backend lane is parked (frees its GPU)
const DEEP_BACKENDS_OVERRIDE = String(A.deep_backends || '').trim(); // optional CSV "triton:optimize,hip:author,..."; default = derive from the bake-off
// ---- DEEP MODE (the canonical deep orchestrator; runs when deep_mode=true) ----
// The HeadKernel roles run deeper (more rounds per lane), broader (a lane per viable backend) and in
// parallel. All (head op, backend) lanes share one pool, picked by Amdahl mass x ceiling gap x recent
// gain; the controller re-profiles as the bottleneck moves. Co-opt uses the cards serving doesn't need
// (time-slices with exactly TP cards). High-ceiling lanes get more patience; a parked lane revives on
// a fresh cross-backend borrow. SHARED_KB is per op, GLOBAL_KB run-wide.
const DEEP = DEEP_MODE;
const DEEP_E2E_TARGET = parseFloat(A.deep_e2e_target != null ? A.deep_e2e_target : 1.5); // stretch goal: +50% e2e vs the true baseline
const DEEP_PLATEAU_STREAK_HIGH = parseInt(A.deep_plateau_streak_high != null ? A.deep_plateau_streak_high : 4, 10); // plateau patience for high-ceiling lanes
const DEEP_REPROFILE_GAIN = parseFloat(A.deep_reprofile_gain != null ? A.deep_reprofile_gain : 0.10); // cumulative e2e gain since last profile that triggers a re-profile (bottleneck moved)
// When all lanes plateau they are re-seeded with fresh authoring directions (weighted to the
// dominant-Amdahl head) instead of exiting, so the run uses its whole budget.
const DEEP_WAVE_BUDGET = parseInt(A.deep_wave_budget != null ? A.deep_wave_budget : 3, 10); // kernel_workflow rounds per burst (short bursts, more waves)
const DEEP_MAX_RESEEDS = parseInt(A.deep_max_reseeds != null ? A.deep_max_reseeds : 12, 10); // max fresh-direction re-seeds per lane
// Convergence stop + agent-budget backstop: after K waves with no new isolated or e2e gain the loop
// finalizes, and DEEP_AGENT_BUDGET keeps nested-agent spend under the runtime's hard agent cap (which
// throws and skips finalize).
const DEEP_CONVERGE_STREAK = parseInt(A.deep_converge_streak != null ? A.deep_converge_streak : 3, 10); // consecutive zero-gain waves before declaring convergence and finalizing
const DEEP_AGENT_BUDGET = parseInt(A.deep_agent_budget != null ? A.deep_agent_budget : 820, 10); // no new waves once projected nested-agent spend reaches this (runtime cap: 1000)
const DEEP_AGENTS_PER_BURST = parseInt(A.deep_agents_per_burst != null ? A.deep_agents_per_burst : 20, 10); // nested agents per burst, for the budget projection
const DEEP_FINAL_ACCURACY_LIMIT = parseInt(A.deep_final_accuracy_limit != null ? A.deep_final_accuracy_limit : 500, 10); // larger gsm8k sample at the Finalize gate so a boundary candidate is decided on signal
// ---- ACCURACY GATE (opt-in switch) ----
// Byte-exact parity over-rejects quantized kernels (in-tolerance rounding flips borderline argmaxes).
// accuracy_gate=gsm8k instead accepts iff cand_score >= baseline_score - accuracy_tol on a sampled
// gsm8k subset (scripts/gsm8k_eval.py, 5-shot greedy exact_match). Default 'none' = byte/greedy parity.
const ACCURACY_GATE = String(A.accuracy_gate || 'none').trim();          // 'none' | 'gsm8k'
const ACCURACY_LIMIT = parseInt(A.accuracy_limit != null ? A.accuracy_limit : 200, 10); // sampled gsm8k subset size
const ACCURACY_TOL = parseFloat(A.accuracy_tol != null ? A.accuracy_tol : 0.01);        // allowed absolute exact_match drop vs baseline
const ACCURACY_INPUTS = (ACCURACY_GATE !== 'none')
  ? { ACCURACY_GATE, ACCURACY_LIMIT, ACCURACY_TOL, GSM8K_EVAL_SCRIPT: `${WORKFLOW_DIR}/scripts/gsm8k_eval.py` }
  : {};
// Simulated speculative acceptance (run_e2e forwards the switch names it finds, e.g.
// SGLANG_SIMULATE_ACC_*) serves output that is neither deterministic nor model-faithful, so roles
// that run parity/accuracy probes are told to serve those probes with these names unset.
const CORRECTNESS_UNSET_ENV = String(A.correctness_unset_env || '').split(',')
  .map((s) => s.trim()).filter((s) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(s));
const CORRECTNESS_PROBE_ROLES = new Set(['e2e_integrator', 'director', 'config_tuner', 'tuning_specialist']);
// The AMD authoring knowledge base (REFERENCE ONLY — facts/how-to, never decisions; agents always
// measure). Default: sibling perf_knowledge/. Workflows enumerate candidates from
// index/capability_index.yaml; status/perf in cards are dated evidence, not routing inputs.
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge')).replace(/\/+$/, '');
if (RDNA4_ISOLATE) KERNEL_KNOWLEDGE_DIR = '';
// Kernel-layer warm start: before round 1 a lane re-validates its kernel's top stored patches from
// kb_artifacts/ through the normal verify gate. e2e only forwards the knobs, so every lane shares one
// store and an orchestrator-supplied store reaches them; unset values equal the lane defaults.
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts')).replace(/\/+$/, '');
const KB_ARGS = {
  kb_artifacts_dir: KB_ARTIFACTS_DIR,
  warm_start: RDNA4_ISOLATE ? 'off' : String(A.warm_start != null ? A.warm_start : 'on'),
  ...(A.warm_start_match != null ? { warm_start_match: String(A.warm_start_match) } : {}),
  ...(A.warm_start_min_speedup != null ? { warm_start_min_speedup: A.warm_start_min_speedup } : {}),
  // The KB plane lanes read and write (one per run); unset leaves the lane default `store`
  // (kernel_lane.js:KB_MODE).
  ...(A.kb_mode != null ? { kb_mode: String(A.kb_mode) } : {}),
  ...(A.kb_store_dir != null ? { kb_store_dir: String(A.kb_store_dir) } : {}),
  ...(A.kb_framework_version != null ? { kb_framework_version: String(A.kb_framework_version) } : {}),
};
// Same off-spelling set the kernel layer accepts; an off run must stay off everywhere.
const WARM_START_OFF = ['off', 'false', 'none'].includes(KB_ARGS.warm_start.trim().toLowerCase());
