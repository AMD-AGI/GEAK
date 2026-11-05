export const meta = {
  name: 'e2e-workflow',

  description: 'End-to-end LLM inference-throughput optimizer for AMD Instinct MI-series GPUs (CDNA gfx942/gfx950) and the validated RDNA4 product Radeon AI PRO R9700 (gfx1201, vLLM only). Generic gfx1201 is not a product identity. The serving stack is pluggable via scripts/adapters/<backend>.sh (sglang + vllm + ATOM shipped on Instinct; R9700 is vLLM-only). A system layer (e2e Director / System Architect / Profiler / Config Tuner / Kernel Extractor / e2e Integrator) wraps the UNCHANGED single-kernel kernel_workflow: it preflights the env, profiles a running server, triages hot kernels by Amdahl, tunes config/backends, extracts hot editable kernels into standalone unittests, recursively optimizes them with kernel_workflow.js, overlays them back, and re-validates serving throughput. Also still optimizes a single kernel (pass-through).',
  whenToUse: 'Optimize the serving throughput of an LLM on AMD Instinct MI GPUs, or on Radeon AI PRO R9700 with vLLM. Use interface/run_e2e.py for structured auto-detection; a direct model-mode call must pass expected_gfx and expected_target before backend/knowledge policy is selected. Pass args.model_path (required) + optional args.backend (sglang|vllm|atom on Instinct, default sglang; vllm only on confirmed R9700) + args.launch_script (optional). For a single kernel, pass args.kernel_path instead and it delegates straight to the kernel layer.',

  phases: [
    { title: 'Setup', detail: 'e2e Director builds the isolated eval dir + records baseline throughput' },
    { title: 'KernelExtract', detail: 'AgentX only: select kernels, extract source, and author and validate UTs before optimization' },
    { title: 'Profile', detail: 'Profiler captures a warm trace -> standardized Top-N' },
    { title: 'Strategize', detail: 'System Architect routes kernels by Amdahl (config vs kernel vs host)' },
    { title: 'ConfigSweep', detail: 'Config Tuner sweeps flags/env/backends FIRST (default ON)' },
    { title: 'TuningSkillset', detail: 'Tuning Specialist runs the VENDORED tuning skillset whole + standalone (its own pre/post A/B) BEFORE HeadKernel, so its share of the gain is attributable' },
    { title: 'HeadKernel', detail: 'highest-%GPU ops (GEMM/attn): extract_op -> backend bake-off (incl. FlyDSL) + aiter-DB/author tune -> e2e gate' },
    { title: 'Milestone', detail: 'loop over editable kernels ABOVE milestone_min_pct% GPU (default 5): plan -> extract -> recursive kernel optimize -> overlay -> e2e gate -> reprofile' },
    { title: 'Finalize-gate', detail: 'finish any e2e A/B left incomplete before the final phase (still OPTIMIZATION work: budget-capped, never runs inside the final reserve)' },
    { title: 'Finalize', detail: 'e2e Integrator assembles the overlay + patch + launch bundle' },
    { title: 'Report', detail: 'System Architect writes the throughput report + grows the playbook' },
    { title: 'Validate', detail: 'e2e Director independently re-measures throughput + arbitrates' },
  ],
};

// RUNTIME SANDBOX CONTRACT -- read before adding any Node API to this file.
// The script runs on two runtimes with different globals:
//                                             process   require   import()
//   Claude Code's native Workflow tool          no        no      REFUSED AT LAUNCH
//   interface/runtime/engine/run_workflow.mjs   yes       no      yes
// The native sandbox has only the JS built-ins plus agent/parallel/pipeline/phase/log/workflow/
// args/budget/console/setTimeout: no filesystem, no child_process, and one `import(` in code
// rejects the whole script at launch. So: no `import(` and no load-bearing `require(`; file work
// goes in a scripts/ CLI that an agent runs. test_llm_ledger.py enforces the `import(` rule.

// Args + defaults. A JS workflow can't read its own path, so workflow_dir is passed in.
let _rawArgs = args;
let _parseErr = '';
// The driver agent may pass `args` as a complete JSON string with stray trailing characters
// (e.g. an extra '}'). Parse tolerantly: retry while trimming the tail, bounded so a truly
// malformed payload still fails loudly.
if (typeof _rawArgs === 'string') {
  let _s = _rawArgs.trim();
  for (let _i = 0; _i < 8 && _s.length > 1; _i++) {
    try { _rawArgs = JSON.parse(_s); _parseErr = ''; break; }
    catch (_e) { _parseErr = String((_e && _e.message) || _e); _s = _s.slice(0, -1).trim(); }
  }
}
const A = _rawArgs || {};
// LLM accounting (args.llm_stats="false" disables): records each agent call's role/phase/attempt.
// No timestamps (no Date.now() here); scripts/llm_ledger.py takes durations from the transcripts.
const LLM_STATS = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
const LLM_TL = { schema: 'geak.agent_timeline/1', workflow: 'e2e_workflow', events: [], nested: [] };
const TL_ROLE_RE = /You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\./;
// Called at dispatch (before the await): events stay in dispatch order, which the ledger's
// positional join relies on, and hung or throwing attempts are still recorded. Returns the event
// (null when off); the caller sets `ok` once the attempt resolves.
function tlAgent(prompt, o, attempt) {
  if (!LLM_STATS) return null;
  // Identity comes from the prompt's `You are the <role>. PHASE=<sub_phase>.` line (also in the
  // transcript), not from opts.label, which is free-form display text.
  const m = TL_ROLE_RE.exec(String(prompt || ''));
  const e = {
    seq: LLM_TL.events.length,
    phase: (o && o.phase) || '',
    label: (o && o.label) || 'agent',
    role: m ? m[1] : '',
    sub_phase: m ? m[2] : '',
    attempt: attempt,
    ok: false,
  };
  LLM_TL.events.push(e);
  return e;
}
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  let _preview;
  try {
    _preview = typeof _rawArgs === 'object' && _rawArgs !== null
      ? 'object keys=[' + Object.keys(_rawArgs).join(',') + ']'
      : JSON.stringify(_rawArgs);
  } catch (_e) { _preview = '<unstringifiable>'; }
  const _s = typeof args === 'string' ? args : '';
  throw new Error('args.workflow_dir is required. DIAGNOSTIC: typeof args=' + (typeof args) +
    ' typeof _rawArgs=' + (typeof _rawArgs) +
    ' rawStrLen=' + _s.length +
    ' parseErr=' + _parseErr +
    ' tail=' + JSON.stringify(_s.slice(-200)) +
    ' preview=' + String(_preview).slice(0, 200));
}
// The UNCHANGED single-kernel workflow. Default: sibling "kernel_workflow" dir next to this one.
const KERNEL_WF_DIR = String(A.kernel_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kernel_workflow')).replace(/\/+$/, '');
// Call the single-language worker lane directly: going through the kernel_workflow.js dispatcher
// would nest e2e -> dispatcher -> worker (3 levels), which the runtime forbids.
const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/kernel_lane.js`;

const EXPECTED_GFX = String(A.expected_gfx || '').trim().toLowerCase();
const EXPECTED_TARGET = String(A.expected_target || '').trim().toLowerCase();
const EXPECTED_DEVICE_NAME = String(A.expected_device_name || '').trim();
const EXPECTED_PHYSICAL_CU_COUNT = Number(A.expected_physical_cu_count || 0);
if (['r9700', 'r9600d'].includes(EXPECTED_TARGET) && EXPECTED_GFX !== 'gfx1201') {
  throw new Error(`expected_target=${EXPECTED_TARGET} requires expected_gfx=gfx1201`);
}
if (EXPECTED_GFX === 'gfx1200' || EXPECTED_TARGET === 'gfx1200') {
  throw new Error('gfx1200 is not supported: this workflow is hardware-validated only on R9700 / gfx1201');
}
const RDNA4_ISA = EXPECTED_GFX === 'gfx1201';
// Products with a `product:` section in perf_knowledge/hardware/product_peaks.md; keep in sync with it.
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

// Every nested lane call builds its args here, so none silently takes the lane's own defaults;
// test_e2e_lane_defaults.py fails on a `scriptPath: KERNEL_WF_SCRIPT` call that doesn't.
// ablLaneArgs() forwards the ablation arm; llm_stats is forwarded only when the parent got one
// explicitly. wfArgs spreads last so a per-call value wins (RDNA4 isolation still forces KB off).
const laneArgs = (wfArgs) => {
  const out = {
    use_learned_kb: LANE_USE_LEARNED_KB,
    ...ablLaneArgs(),
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
// The Profiler may run ONE analysis skill after parse_profile.py's Top-N (default `kernel-roofline`:
// native TraceLens models, call stacks and coverage). Advisory only: never prunes or overrides
// pct_gpu_time. `none`/`false` disables it; the legacy `roofline` name selects the replacement.
const ANALYSIS_SKILL = String(A.analysis_skill != null ? A.analysis_skill : 'kernel-roofline').trim().replace(/^roofline$/, 'kernel-roofline');
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
const KERNEL_BUDGET = parseInt(A.kernel_budget != null ? A.kernel_budget : (FAST_MODE ? 3 : 6), 10); // budget passed DOWN per kernel (fewer rounds in fast mode)
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
// Only accuracy-gated accepts can need an independent A/B confirmation; exceeding the estimate alone
// never triggers a kernel correctness rewrite.
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
// runIntegrateBothLegs resolves the review before any caller can bank the candidate. A confirmed
// measurement can exceed the estimate by any amount; actual parity failures still reject.
function integAccepted(integ, pct_gpu_time, isolated) {
  return !!(integ && (integ.gate === 'accepted' || integ.gate === 'stack')
    && integ.ab_complete !== false && integ.output_parity !== 'fail'
    && !(integ.speedup_review && ['pending', 'inconclusive', 'rejected'].includes(integ.speedup_review.status))
    && (!speedupNeedsReview(pct_gpu_time, isolated, integ) || confirmedSpeedup(integ)));
}
function gateRejectReason(integ, pct_gpu_time, isolated) {
  if (integ && integ.output_parity === 'fail') return 'parity_regression';
  if (integ && (integ.gate === 'accepted' || integ.gate === 'stack')
      && !integAccepted(integ, pct_gpu_time, isolated))
    return 'speedup_review_pending: independent A/B confirmation required';
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

// E2E-LEVEL WARM START — the DEPLOYMENT knowledge base: was this deployment (model, gfx, framework
// + version, precision, tp, isl/osl/conc) tuned before? Module A reads the store and validates on
// this box before believing it; Module B records this run at the end. A.warm_start governs both
// this and the kernel layer:
//   on (default)      | read the ladder, validate the top candidate on this box, ADOPT if it wins.
//   reference         | read only; never bench, never adopt — the offer is prose for later agents.
//   return_after_read | resolve + validate, then return before the optimization flow runs.
//   off/false/none    | nothing: no phase, no agent, no log.
const E2E_WARM_START = KB_ARGS.warm_start.trim().toLowerCase() || 'on';
const E2E_WARM_START_ON = !WARM_START_OFF;
// Fast mode forces reference-only: it skips config work, so it won't spend a 20-40min server launch
// benching a stored config.
const E2E_WARM_START_REF_ONLY = E2E_WARM_START === 'reference' || FAST_MODE;
const E2E_WARM_START_RETURN_AFTER = E2E_WARM_START === 'return_after_read';
// Deployment-store plane, derived from kb_mode once here so reader and writer agree:
//   local   on-disk store only (offline rehearsal; no network, nothing permanent).
//   remote  the KB Store service only.
//   both    local first, remote as well — a network failure is reported, not fatal.
// Default `both`: the local copy is free and survives an unreachable service.
const E2E_KB_PLANE = ['local', 'remote', 'both'].includes(String(A.e2e_kb_plane || '').trim())
  ? String(A.e2e_kb_plane).trim()
  : (String(A.kb_mode || '').trim().toLowerCase() === 'local' ? 'local' : 'both');
const E2E_KB_STORE_DIR = String(A.kb_store_dir ||
  KB_ARTIFACTS_DIR.replace(/\/[^/]*$/, '') + '/kb_store_local').replace(/\/+$/, '');
const E2E_STORE_SCRIPT = `${WORKFLOW_DIR}/scripts/e2e_store.py`;
// Where the warm-start read writes this run's KB address; run_e2e.py reads it to write the record
// if this process never does. Must match interface/run_e2e.py KB_IDENTITY_FILE.
const KB_IDENTITY_BASENAME = 'kb_identity.json';
// Two floors: this read floor ("worth knowing about"; drops only records that lost) and the bench
// floor below ("worth a 20-40min launch"). `e2e_`-prefixed because the kernel lane's
// warm_start_min_speedup means its own thing; the bare name is still honoured.
const E2E_WARM_START_MIN_SPEEDUP = Number.isFinite(parseFloat(A.e2e_warm_start_min_speedup))
  ? parseFloat(A.e2e_warm_start_min_speedup)
  : Number.isFinite(parseFloat(A.warm_start_min_speedup))
    ? parseFloat(A.warm_start_min_speedup) : 1.0;
// Bench floor: below it the offer stays a reference (a ~1.01x claim is within session drift). Not
// applied to kernel replay, whose stored speedup is isolated; E2E_WARM_START_KERNELS_N budgets that.
const E2E_WARM_START_BENCH_MIN_SPEEDUP =
  Number.isFinite(parseFloat(A.e2e_warm_start_bench_min_speedup))
    ? parseFloat(A.e2e_warm_start_bench_min_speedup) : 1.02;

// How a local verdict is filed against the record. The local bench only decides what THIS run
// adopts; no single outcome retires a record:
//   rejected        applied, took effect, lost here -> `failed`; the only bucket that accumulates
//                   toward a retire (kb/attest.py:should_retire, run by `e2e_store.py curate`).
//   not_reproduced  never ran, or ran without taking effect -> the record is missing something.
//   inapplicable    collided with a baseline this run did not choose; kept out of the arithmetic.
// Every value must be one of kb/attest.py:OUTCOMES.
const KB_ATTEST_OUTCOME = {
  adopted: 'validated',
  rejected: 'failed',
  not_reproduced: 'not_reproduced',
  inapplicable: 'inapplicable',
};
// Read broadly, bench narrowly: offer 3, measure 1 (reading is free; benching costs a launch).
const E2E_WARM_START_TOP_N = parseInt(A.e2e_warm_start_top_n != null ? A.e2e_warm_start_top_n : 3, 10);
const E2E_WARM_START_VALIDATE_N = parseInt(
  A.e2e_warm_start_validate_n != null ? A.e2e_warm_start_validate_n : 1, 10);
// The same budget for a COARSE rung (`tp_any`/`workload_any`: measured at another tp or workload).
// The bench measures the config on this box, so the stored number only orders the tries. 0 makes
// coarse hits reference-only.
const E2E_WARM_START_VALIDATE_N_COARSE = parseInt(
  A.e2e_warm_start_validate_n_coarse != null ? A.e2e_warm_start_validate_n_coarse : 1, 10);
// How many stored KERNELS get replayed through a fresh integrate A/B. Each one is another two server
// launches, so this is a budget, not a completeness target — the rest stay references.
const E2E_WARM_START_KERNELS_N = parseInt(
  A.e2e_warm_start_kernels_n != null ? A.e2e_warm_start_kernels_n : 2, 10);
// Win-kinds rebuildable from a record alone: `patch` re-applies a diff through the overlay;
// `env`/`flag` re-route to an implementation already installed. Other kinds (e.g. `authored`, whose
// build workspace dies with its run) replay via the record's prebuilt overlay.tar.gz when present,
// else are demoted to a reference (never reported as a failure).
const E2E_REPLAYABLE_KINDS = new Set(['patch', 'env', 'flag']);
// Roles told about the warm start. Excludes e2e_integrator and director:validate, which produce the
// authoritative numbers. tuning_specialist is included: a stored tuned artifact saves hours of search.
const WARM_START_ROLES = new Set(['system_architect', 'config_tuner', 'tuning_specialist']);
// Credentials and trust for every emitted KB command come from scripts/kb_env.sh, which run_e2e.py
// sources too, so both programs share one definition. (shq is declared later, so quoting is inline.)
const KB_ENV_PRELUDE = `. "${WORKFLOW_DIR}/scripts/kb_env.sh"; `;
// Expert skills: human-authored, validated recipes (perf_knowledge/expert_skills/) used as advisory
// high priors that roles reproduce and gate by the usual on-box A/B; never override measurement.
// Default off (use_expert_skills="true" enables); flag and dir are passed to the kernel layer.
const USE_EXPERT_SKILLS = !RDNA4_ISOLATE && String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR + '/expert_skills')).replace(/\/+$/, '');
// Only routing/bake-off/integration roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['system_architect', 'op_benchmarker', 'e2e_integrator']);
const GEMM_SYNTH = String(A.gemm_synth != null ? A.gemm_synth : 'true');     // synth GEMM inputs (cheap)
const ENABLE_FP8 = String(A.enable_fp8 != null ? A.enable_fp8 : 'false');    // Tier-D quant (parity-breaking)
const FAST_PATH_FIRST = String(A.fast_path_first != null ? A.fast_path_first : 'true') === 'true';
// The caller's declared ISL/OSL, kept apart from the resolved target shape (they differ on a trace
// replay); the resolved ISL/OSL/CONC are defined after the workload kind is parsed.
const ISL_DECLARED = A.isl != null ? parseInt(A.isl, 10) : null;
const OSL_DECLARED = A.osl != null ? parseInt(A.osl, 10) : null;

// Workload IDENTITY — the synthetic sweep, or a trace replay that owns its load. Under Hyperloom,
// run_e2e.py exports BENCH_CLIENT / E2E_METRIC / AGENTX_* from the handoff. Standalone, declare it
// as args.workload_spec (the handoff's object) or args.workload_kind="agentx_trace_replay" (the
// canonical setup below; workload_spec overrides any part). AgentX extraction resumes its saved
// declaration when no new workload is supplied.
const WORKLOAD_SPEC = A.workload_spec && typeof A.workload_spec === 'object'
  ? A.workload_spec
  : ((!A.workload_kind || A.workload_kind === 'agentx_trace_replay') &&
      A.state?.workload_spec?.kind === 'agentx_trace_replay'
    ? A.state.workload_spec : {});
const WORKLOAD_KIND = String(A.workload_kind || WORKLOAD_SPEC.kind || 'synthetic');
const IS_AGENTX = WORKLOAD_KIND === 'agentx_trace_replay';
// The canonical AgentX submission setup: leaderboard values, not tuning knobs. A deviating run is
// stamped non-canonical and never KEPT; the inner search legs run the scenario's 900s floor, so
// they are always non-canonical.
const AGENTX_DEFAULTS = {
  scenario: 'inferencex-agentx-mvp',
  // The full-context corpus, not the _256k sibling (which drops requests over 256k tokens);
  // the campaign serves at --max-model-len 1048576.
  corpus: 'semianalysis_cc_traces_weka_062126',
  num_entries: 393,
  duration_s: 3600,           // canonical/parity window
  geak_loop_duration_s: 900,  // inner search legs (scenario floor)
  warmup_requests_per_lane: 10,
  warmup_grace_period_s: 1800,
  // Not defaulted: the client picks the tolerance per leg (MEASUREMENT_PURPOSE: exploring vs
  // compared). A declared value overrides it for the whole run.
  failed_request_threshold: null,
  // InferenceX P90 interactivity, 1000 / P90(ITL) tok/s/user (decode-only; bench_summarize.py
  // also reports total tok/s as a guard). Hyperloom declares its own basis; this is the default.
  metric_basis: 'p90_intvty_inferencex',
};
const AGENTX = IS_AGENTX ? Object.assign({}, AGENTX_DEFAULTS, WORKLOAD_SPEC) : null;
// ONE concurrency for the run: roles pass CONC=<WORKLOAD.conc> on every bench line, which outranks
// bench_env.sh, so workload_spec.concurrency must win here.
const CONC = parseInt(AGENTX && AGENTX.concurrency != null ? AGENTX.concurrency
  : (A.conc != null ? A.conc : 64), 10);

// Kernel-targeting shape — the shape roles synthesize kernel inputs against. On a trace replay
// isl/osl aren't measured (aiperf owns lengths), but roles size tiles and GEMM shapes from them, so a
// 1024/1024 placeholder would aim the search far below the real load. Mirrors
// interface/run_e2e.py::_targeting_shape so both entries resolve the same shape and provenance.
function resolveTargetShape() {
  const haveDeclared = ISL_DECLARED > 0 && OSL_DECLARED > 0;
  const declaredOr = (v) => (v != null ? v : 1024);
  // Hyperloom resolves this upstream and passes it down; honour it verbatim.
  if (A.workload_shape_provenance) {
    return [declaredOr(ISL_DECLARED), declaredOr(OSL_DECLARED),
            String(A.workload_shape_provenance)];
  }
  if (!IS_AGENTX) {
    // Fixed ISL/OSL path: declared args, else the synthetic default.
    return [declaredOr(ISL_DECLARED), declaredOr(OSL_DECLARED),
            haveDeclared ? 'declared_args' : 'synthetic_default'];
  }
  // Use an explicit observed shape if given; never guess or hard-code a corpus average (it moves
  // with corpus, tokenizer and window). Otherwise the baseline bench measures it and
  // adoptMeasuredShape() installs it, which is why no kernel work starts before the baseline.
  const obsIsl = parseInt(AGENTX.observed_isl, 10);
  const obsOsl = parseInt(AGENTX.observed_osl, 10);
  if (obsIsl > 0 && obsOsl > 0) return [obsIsl, obsOsl, 'agentx_observed'];
  if (haveDeclared) return [ISL_DECLARED, OSL_DECLARED, 'agentx_declared_args'];
  return [1024, 1024, 'agentx_pending_baseline'];
}
// Mutable: on a trace replay the baseline installs the measured shape.
let [ISL, OSL, WORKLOAD_SHAPE_PROVENANCE] = resolveTargetShape();
const WORKLOAD = { isl: ISL, osl: OSL, conc: CONC, shape_provenance: WORKLOAD_SHAPE_PROVENANCE };
// Whether ISL/OSL describe the served load or a placeholder; prompts tell roles which, since a
// placeholder would misdirect kernel choices.
let SHAPE_IS_MEASURED = !/^(agentx_pending_baseline|synthetic_fallback_on_agentx)$/
  .test(WORKLOAD_SHAPE_PROVENANCE);

// Install the shape the baseline served (bench_e2e.sh observed_isl/observed_osl: per-request mean,
// median across replicas). Called right after Setup, before any kernel work.
function adoptMeasuredShape(isl, osl, where) {
  if (!(isl > 0 && osl > 0)) return false;
  ISL = Math.round(isl);
  OSL = Math.round(osl);
  WORKLOAD_SHAPE_PROVENANCE = 'agentx_measured_this_run';
  SHAPE_IS_MEASURED = true;
  WORKLOAD.isl = ISL;
  WORKLOAD.osl = OSL;
  WORKLOAD.shape_provenance = WORKLOAD_SHAPE_PROVENANCE;
  log(`Kernel-targeting shape measured from ${where}: isl=${ISL} osl=${OSL} `
    + `(ratio ${(ISL / Math.max(1, OSL)).toFixed(1)}:1). Roles size kernels against this.`);
  return true;
}
// The baseline summary can land after Setup returns (a leg relaunched after a hardware fault), so
// the prompt builder also retries adoption lazily: the first prompt built after the summary
// appears installs the real shape. A no-op once measured.
function ensureMeasuredShape() {
  if (SHAPE_IS_MEASURED || !IS_AGENTX) return SHAPE_IS_MEASURED;
  // EVAL_DIR is assigned by Setup; before that there is no baseline to read.
  const evalDir = typeof EVAL_DIR === 'undefined' ? '' : (EVAL_DIR || '');
  if (!evalDir) return SHAPE_IS_MEASURED;
  let shape = null;
  try {
    // Same execFileSync/python3 route the Setup probe uses: this JS has no fs.
    const { execFileSync } = require('child_process');
    const out = execFileSync('python3', ['-c', `
import json, os, sys
p = os.path.join(sys.argv[1], 'baseline', 'bench_summary.json')
def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
try:
    with open(p) as fh:
        s = json.load(fh) or {}
    print(json.dumps([_num(s.get('observed_isl')), _num(s.get('observed_osl'))]))
except Exception:
    print('[null, null]')
`, evalDir], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 30000 });
    shape = JSON.parse(out);
  } catch (err) {
    return SHAPE_IS_MEASURED;   // unreadable right now; a later prompt retries
  }
  if (shape && shape[0] > 0 && shape[1] > 0) {
    adoptMeasuredShape(shape[0], shape[1],
      'the baseline bench (read after Setup returned, which had no summary yet)');
  }
  return SHAPE_IS_MEASURED;
}
// The axis bench_e2e.sh medians must be selected here: its default (output tok/s) barely moves on a
// prefill-heavy trace. This is bench_summarize.py's _BASES flattened (E2E_METRIC -> metric_basis);
// test_bench_env_channel.py keeps the two in sync.
const METRIC_BASIS_OF_AXIS = {
  output: 'aggregate_output_tok_s',
  total: 'aggregate_total_token_tok_s',
  total_token: 'aggregate_total_token_tok_s',
  total_throughput: 'aggregate_total_token_tok_s',
  intvty: 'e2e_norm_intvty_p90',
  interactivity: 'e2e_norm_intvty_p90',
  e2e_norm_intvty_p50: 'e2e_norm_intvty_p50',
  e2e_norm_intvty_p90: 'e2e_norm_intvty_p90',
  p90_intvty_inferencex: 'p90_intvty_inferencex',
};
// The E2E_METRIC the declaration sets for each basis: the basis itself wherever _BASES takes it.
const AXIS_OF_METRIC_BASIS = {
  aggregate_output_tok_s: 'output',
  aggregate_total_token_tok_s: 'total',
  e2e_norm_intvty_p50: 'e2e_norm_intvty_p50',
  e2e_norm_intvty_p90: 'e2e_norm_intvty_p90',
  p90_intvty_inferencex: 'p90_intvty_inferencex',
};
const AXIS_LABEL = {
  aggregate_output_tok_s: 'OUTPUT tok/s',
  aggregate_total_token_tok_s: 'TOTAL (input+output) tok/s',
  e2e_norm_intvty_p50: 'TTFT-inclusive P50 interactivity: 1 / P50(E2EL/OSL) in tok/s/user',
  p90_intvty_inferencex: 'InferenceX P90 interactivity: 1000 / P90(ITL) in tok/s/user, decode only',
  e2e_norm_intvty_p90: 'TTFT-inclusive P90 interactivity: 1 / P90(E2EL/OSL) in tok/s/user',
};
// Precedence: the declared basis; else an exported E2E_METRIC (it outranks bench_env.sh); else
// the standalone default. An axis bench_summarize.py can't measure fails here, before the baseline.
function resolveAgentxMetric() {
  if (!AGENTX) return { basis: '', source: '' };
  const unknown = (what, v) => new Error(`AgentX: ${what} '${v}' is not an axis ` +
    `bench_summarize.py measures (known: ${Object.keys(METRIC_BASIS_OF_AXIS).join(', ')}).`);
  let inherited = '';
  try { inherited = String(process.env.E2E_METRIC || '').trim().toLowerCase(); } catch (_) { /* no env */ }
  // The bench reads an exported value as an axis token, so that is the only way it is looked up.
  const inheritedBasis = inherited ? METRIC_BASIS_OF_AXIS[inherited] : '';
  if (inherited && !inheritedBasis) throw unknown('the exported E2E_METRIC', inherited);
  const declared = String(WORKLOAD_SPEC.metric_basis || '').trim().toLowerCase();
  if (!declared) {
    return inheritedBasis ? { basis: inheritedBasis, source: 'the exported E2E_METRIC' }
      : { basis: AGENTX_DEFAULTS.metric_basis, source: 'the standalone default' };
  }
  const basis = AXIS_OF_METRIC_BASIS[declared] ? declared : METRIC_BASIS_OF_AXIS[declared];
  if (!basis) throw unknown('workload_spec.metric_basis', declared);
  if (inheritedBasis && inheritedBasis !== basis) {
    throw new Error(`AgentX: workload_spec.metric_basis=${basis}, but E2E_METRIC=${inherited} is ` +
      'exported in this environment and outranks bench_env.sh, so every bench would grade a ' +
      'different axis than the one declared. Unset E2E_METRIC or declare the basis it names.');
  }
  return { basis, source: 'workload_spec.metric_basis' };
}
const { basis: AGENTX_METRIC_BASIS, source: AGENTX_METRIC_SOURCE } = resolveAgentxMetric();
const AGENTX_E2E_METRIC = AGENTX ? AXIS_OF_METRIC_BASIS[AGENTX_METRIC_BASIS] : '';
// KB recall can use its own basis (e.g. only p50 candidates while the bench grades output/total);
// it never changes the bench axis or the writer's basis.
function resolveKbMetric() {
  let env = '';
  try { env = String(process.env.GEAK_E2E_KB_METRIC_BASIS || '').trim(); } catch (_) { /* no env */ }
  const arg = String(A.e2e_kb_metric_basis || '').trim();
  const requested = (arg || env).toLowerCase();
  if (!requested) return { basis: AGENTX_METRIC_BASIS, strict: false, source: 'measurement' };
  const basis = AXIS_OF_METRIC_BASIS[requested] ? requested : METRIC_BASIS_OF_AXIS[requested];
  if (!basis) throw new Error(`Unknown e2e KB metric basis '${requested}'. Known bases: ` +
    Object.keys(AXIS_OF_METRIC_BASIS).join(', '));
  return { basis, strict: true, source: arg ? 'e2e_kb_metric_basis' : 'GEAK_E2E_KB_METRIC_BASIS' };
}
const E2E_KB_METRIC = resolveKbMetric();
function kbMeasurementBasis() {
  if (AGENTX) return AGENTX_METRIC_BASIS;
  let axis = '';
  try { axis = String(process.env.E2E_METRIC || '').trim().toLowerCase(); } catch (_) { /* no env */ }
  return METRIC_BASIS_OF_AXIS[axis] || 'aggregate_output_tok_s';
}
function kbCanAttestMetric(candidate) {
  const measured = kbMeasurementBasis();
  // A fresh output/total A/B is not evidence for or against the stored p50 claim.
  return candidate.metric_basis ? candidate.metric_basis === measured : !E2E_KB_METRIC.strict;
}
// Body of the per-run bench_env.sh the Director drops beside the bench script (roles/director.md
// PHASE=setup); empty when synthetic. Each line assigns only if the name is unset or empty, so an
// exported value always wins. Spelled `[ -n "${K:-}" ] || K='v'`, not `: "${K:='v'}"`, whose quotes
// would become part of the value.
const _shq = (v) => `'${String(v).replace(/'/g, "'\\''")}'`;
const _agentxEnvPairs = () => {
  if (!AGENTX) return [];
  const pairs = [
    ['GEAK_WORKLOAD_KIND', WORKLOAD_KIND],
    ['GEAK_ISL_OSL_INACTIVE', '1'],
    ['BENCH_CLIENT', 'agentx'],
    ['E2E_METRIC', AGENTX_E2E_METRIC],
    ['GEAK_METRIC_BASIS', AGENTX_METRIC_BASIS],
    // No REPEATS: an explicit one outranks REPLICAS in bench_e2e.sh, which sets trace-replay rounds
    // itself after resolving the replica count.
    ['CONC', String(CONC)],
    ['GEAK_AGENTX_SCENARIO', AGENTX.scenario],
    ['AGENTX_DATASET', AGENTX.corpus],
    // The reference the run is compared against; overriding the corpus or window doesn't move it
    // (so such a replay is non-canonical). Declare canonical_corpus / canonical_duration_s to move it.
    ['AGENTX_CANONICAL_DATASET', AGENTX.canonical_corpus || AGENTX_DEFAULTS.corpus],
    ['AGENTX_NUM_ENTRIES', AGENTX.num_entries],
    ['GEAK_AGENTX_DURATION_S', AGENTX.duration_s],
    ['AGENTX_CANONICAL_DURATION', AGENTX.canonical_duration_s || AGENTX_DEFAULTS.duration_s],
    ['GEAK_AGENTX_LOOP_DURATION_S', AGENTX.geak_loop_duration_s],
    ['AGENTX_WARMUP_REQUESTS_PER_LANE', AGENTX.warmup_requests_per_lane],
    ['AGENTX_WARMUP_GRACE_PERIOD', AGENTX.warmup_grace_period_s],
  ];
  // Optional, only when supplied: aiperf binary, InferenceX checkout (may hold map_aiperf.py; GEAK
  // vendors a fallback), and the profile-window placement.
  const opt = [
    ['AGENTX_FAILED_REQUEST_THRESHOLD', AGENTX.failed_request_threshold],
    ['AIPERF_BIN', AGENTX.aiperf_bin],
    ['INFERENCEX_PATH', AGENTX.inferencex_path],
    ['AGENTX_PROFILE_WARMUP_S', AGENTX.profile_warmup_s],
    ['AGENTX_PROFILE_WINDOW_S', AGENTX.profile_window_s],
    ['AGENTX_MAX_CTX', AGENTX.max_context_length],
  ];
  for (const [k, v] of opt) if (v != null && String(v).trim() !== '') pairs.push([k, v]);
  return pairs;
};
const AGENTX_ENV = !AGENTX ? '' : [
  '#!/usr/bin/env bash',
  '# Per-run workload declaration, written by e2e_workflow.js at Setup.',
  '# Sourced by bench_e2e.sh / bench_replica.sh from beside themselves.',
  '# Assign-if-unset throughout: an inherited/exported value always wins here.',
  ...(_agentxEnvPairs().map(([k, v]) => `[ -n "\${${k}:-}" ] || ${k}=${_shq(v)}`)),
  `export ${_agentxEnvPairs().map(([k]) => k).join(' ')}`,
  '',
].join('\n');
// Kernel extraction is enabled only by an AgentX workload; model/backend features select its
// branches, never its activation.
const KERNEL_EXTRACT_ON = !!MODEL_PATH && WORKLOAD_KIND === 'agentx_trace_replay';
const KERNEL_EXTRACT_DIR = `${WORKFLOW_DIR}/knowledge/kernel_extraction_skills/kernel-extract`;
if (KERNEL_EXTRACT_ON) {
  WORKLOAD.kind = WORKLOAD_KIND;
  WORKLOAD.spec = WORKLOAD_SPEC;
}
if (KERNEL_EXTRACT_ON) log(`Kernel extraction skill enabled (workload=${WORKLOAD_KIND}; verify features from current evidence).`);
// Seed config: an orchestrator that already searched config passes its best flags/env, so the
// baseline is measured on that config, not the stack default (TP/GPU: SERVING_TP / SERVING_GPU).
const INIT_FLAGS = String(A.initial_extra_server_args || '');
const INIT_ENV = String(A.initial_extra_env || '');
// Schema-v2 handoffs may carry the overlay stack behind Hyperloom's current best. It is part of the
// baseline configuration, so it stays underneath every later candidate overlay.
const INIT_BASE_OVERLAY = String(A.initial_overlay_pythonpath || '');
const EFFECTIVE_CONFIG_DIGEST = String(A.effective_config_digest || '');
// One lifecycle for EVERY throughput number (baseline, search A/B, parity, validation), since cold
// and warm numbers aren't comparable. warm_server = Hyperloom's protocol: one server per leg, round
// 1 is a discarded warmup, round 2 IS the number (a trace replay warms each lane itself).
// isolated_server (REPLICAS fresh servers per leg) gates on boot-to-boot variance. The bench
// dispatcher owns retry/degraded aggregation in both modes.
const MEASUREMENT_MODE = String(A.measurement_mode || 'warm_server');
const PARITY_REPLICAS = parseInt(A.parity_replicas != null ? A.parity_replicas : 1, 10);
const SEARCH_REPLICAS = parseInt(A.search_replicas != null ? A.search_replicas : 1, 10);
const VALIDATION_REPLICAS = parseInt(A.validation_replicas != null ? A.validation_replicas : 3, 10);
// Validation has its own mode so the final re-measure can be stricter (isolated_server +
// validation_replicas=3 adds the boot-to-boot dispersion test). Default warm_server: isolated
// validation is 6-12 serialized cold boots and can overrun FINAL_RESERVE_MS.
const VALIDATION_MEASUREMENT_MODE = String(A.validation_measurement_mode || 'warm_server');
// Samples follow the mode: warm_server is one timed round per leg; isolated uses validation_replicas.
const VALIDATION_SAMPLES = VALIDATION_MEASUREMENT_MODE === 'warm_server' ? 1 : VALIDATION_REPLICAS;
// CUDA/HIP-graph deployment requirement (derived from the serving config). vllm/sglang/atom capture
// the steady-state decode path into a full graph unless --enforce-eager is set, and a win that relies
// on its own per-call graph wrapper falls back to eager inside it and vanishes e2e. So when graphs are
// on, every kernel-optimize task is told the win must be intrinsic and graph-capture-safe.
const CUDA_GRAPH_DEPLOY = (BACKEND === 'vllm' || BACKEND === 'sglang' || BACKEND === 'atom') && !/enforce[-_]eager/i.test(INIT_FLAGS);
const GRAPH_REQ = CUDA_GRAPH_DEPLOY ? (
  ' DEPLOYMENT REQUIREMENT — the server captures the steady-state decode path into a FULL CUDA/HIP graph, ' +
  'so this kernel runs INSIDE that captured graph. Your speedup MUST be INTRINSIC: better tiles/algorithm, ' +
  `fused quant (one fp8 ${MATRIX_CORE_NAME}, kill the dequant), or fewer ops/launches that reduce work INSIDE the captured ` +
  'region. Do NOT rely on a per-call CUDA/HIP-graph capture+replay WRAPPER for the speedup — inside the ' +
  "server's graph that wrapper falls back to eager and the win vanishes, and the e2e integrate gate WILL " +
  'reject a wrapper-only win (this already happened: a 1.22x isolated MoE GEMM gave 0% e2e because only its ' +
  'static tile change survived the graph). The steady-state decode call must be graph-capture-safe: ' +
  'host-sync-free (no .item()/.cpu()/.tolist()/.synchronize(), no Python branch on a GPU scalar), shape-stable, ' +
  'and prep/compile ONCE (cache by data_ptr) so the captured region only LAUNCHES the kernel. VERIFY your ' +
  'speedup holds when the op is replayed under a CUDA graph, not just in eager timing.'
) : '';
// Acceptance noise band (%): 0.5 for a fixed-shape workload (identical work per replica, isolated
// measurements, non-overlap and engagement proof). A trace replay varies with which trajectories a
// replica completes, so it gets a 4% floor; a TTFT-inclusive axis needs an explicit noise_band_pct.
const NOISE_BAND_AGENTX = 4.0;
const NOISE_BAND_DEFAULT = parseFloat(
  A.noise_band_pct != null ? A.noise_band_pct : (IS_AGENTX ? NOISE_BAND_AGENTX : 0.5));
// The gate band from the Director's value. The Director starts at 0.5, so on a trace replay it may
// only raise the band above the workload floor; a fixed-shape run uses the Director's value.
function gateNoiseBand(reported) {
  const r = parseFloat(reported);
  const director = Number.isFinite(r) && r > 0 ? r : 0;
  if (!IS_AGENTX) return director || NOISE_BAND_DEFAULT;
  return Math.max(director, NOISE_BAND_DEFAULT);
}
// No timed-repeat knob: bench_e2e.sh derives rounds from GEAK_REPEAT_MODE + MEASUREMENT_PURPOSE.
// Every integrate A/B must measure both legs: on gate:'incomplete' / ab_complete:false the
// integrator is re-invoked up to this many times to finish the missing leg, then the candidate is
// carried as a pending integration.
const AB_FINISH_RETRIES = parseInt(A.ab_finish_retries != null ? A.ab_finish_retries : 3, 10);
// A resolvable FROZEN baseline (seeded baseline_overlay/ + meta.candidate_bind, or an importable
// meta.baseline_callable on the op track) is the mandatory speedup denominator; without it the
// unittest would time the candidate against its own naive scaffold. Re-extract up to this many
// times, then treat the extraction as failed (flag if dominant, else skip).
const BASELINE_EXTRACT_RETRIES = parseInt(A.baseline_extract_retries != null ? A.baseline_extract_retries : 3, 10);
// Capture storage bounds forced into every extraction EXTRA_ENV (override via args).
const CAPTURE_BYTE_BUDGET = String(A.capture_byte_budget != null ? A.capture_byte_budget : '8GiB');
const CAPTURE_CASE_BYTE_LIMIT = String(A.capture_case_byte_limit != null ? A.capture_case_byte_limit : '2GiB');
const CAPTURE_PERSIST_POLICY = String(A.capture_persist_policy != null ? A.capture_persist_policy : (KERNEL_EXTRACT_ON ? 'structure_only' : 'share_large'));
const CAPTURE_WORKSPACE_BUDGET = String(A.capture_workspace_budget != null ? A.capture_workspace_budget : '32GiB');
const CAPTURE_STORAGE_ENV = `CAPTURE_BYTE_BUDGET=${CAPTURE_BYTE_BUDGET} CAPTURE_CASE_BYTE_LIMIT=${CAPTURE_CASE_BYTE_LIMIT} CAPTURE_PERSIST_POLICY=${CAPTURE_PERSIST_POLICY}`;
const TASK = A.task || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
// The caller reads workflow_return.json and recovers results only from a pinned eval_dir, so a Director
// that builds a sibling directory leaves the whole run where the caller never looks.
function evalDirDivergence(override, returned) {
  const norm = (p) => String(p || '').trim().replace(/(.)\/+$/, '$1');
  if (!norm(override) || norm(returned) === norm(override)) return '';
  return `Director returned eval_dir ${norm(returned) || '(none)'}, but the caller pinned ${norm(override)}`;
}
const MODEL_NAME_HINT = (MODEL_PATH || KERNEL_PATH).replace(/\/+$/, '').split('/').pop();

// Phase-scoped driving (so a long run survives host-session compaction): args.phases = subset of
// {setup,config,tune,head,kernel,final}, default 'all'. State crosses phases via the return value (no
// fs): pass the prior return back as args.state. Unrequested phases load carried state.
const PHASES = String(A.phases || 'all').split(',').map(s => s.trim()).filter(Boolean);
const RUN_ALL = PHASES.includes('all');
// Fast mode skips ConfigSweep, TuningSkillset and Milestone: everything comes from HeadKernel within
// the wall-clock budget.
const FAST_SKIP = FAST_MODE ? new Set(['config', 'tune', 'kernel']) : null;
// Deep mode skips only Milestone; ConfigSweep and TuningSkillset are cheap and give co-opt a better
// starting stack.
const DEEP_SKIP = DEEP_MODE ? new Set(['kernel']) : null;
// AgentX prepares UTs once, then optimizes them against the same serving baseline.
const AGENTX_SKIP = KERNEL_EXTRACT_ON ? new Set(['profile', 'config', 'tune']) : null;
const want = (p) => (RUN_ALL || PHASES.includes(p)) && !(FAST_SKIP && FAST_SKIP.has(p)) && !(DEEP_SKIP && DEEP_SKIP.has(p)) && !(AGENTX_SKIP && AGENTX_SKIP.has(p));
const ST = A.state || {};   // carried state from a prior phase invocation
// Hoisted: WarmStart's integrate legs reach tuningIntegrateInputs() before the TuningSkillset phase
// runs, so `tuning` must already be declared; that phase reassigns it.
let tuning = ST.tuning || null;
if (FAST_MODE) log(`[fast-mode] ON: skipping ConfigSweep + Milestone; HeadKernel-only; budget ${Math.round(FAST_BUDGET_MS / 60000)}min (stop new heads at ${Math.round(FAST_HEAD_DEADLINE_MS / 60000)}min, per-head workflow cap ${Math.round(FAST_HEAD_WF_MS / 60000)}min).`);

// Schema fragments.
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
const arrStr = { type: 'array', items: { type: 'string' } };

const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, model_name: { type: 'string' },
  baseline_throughput_tok_s: { type: 'number' }, baseline_spread_pct: { type: 'number' },
  noise_band_pct: { type: 'number' }, baseline_summary_path: { type: 'string' },
  server_flags: { type: 'object', additionalProperties: true }, server_env: { type: 'string' },
  tp: { type: 'number' }, workload: { type: 'object', additionalProperties: true },
  bench_script: { type: 'string' }, notes: { type: 'string' },
  // Deployment dimensions the KB addresses a page by, from the Director's preflight. gfx, product
  // target and physical CU count are required (device_name is optional telemetry). Never guess: the
  // service has no DELETE, so a wrong page is permanent.
  gfx: { type: 'string' }, device_target: { type: 'string' },
  device_name: { type: 'string' }, physical_cu_count: { type: 'number' },
  precision: { type: 'string' },
  framework_version: { type: 'string' }, rocm_version: { type: 'string' },
}, ['eval_dir', 'baseline_throughput_tok_s', 'gfx', 'device_target', 'physical_cu_count']);

// admitHeads adopts entity_kind from these rows, so a row without it silently refuses every head.
// device_kernel is required on gpu_kernel rows too; admitHeads checks that (no conditional schema here).
const PROFILE_ROW_SCHEMA = obj({
  entity_kind: { type: 'string',
    enum: ['gpu_kernel', 'memory_op', 'dispatcher_op', 'python_launcher', 'unresolved'] },
  device_kernel: { type: 'string' },
}, ['entity_kind']);
const PROFILE_SCHEMA = obj({
  round: { type: 'number' }, profile_topN_json: { type: 'string' }, profile_topN_md: { type: 'string' },
  profile_workload_json: { type: 'string' }, // per-(shape,dtype) weighted workload model (optional)
  source: { type: 'string' }, total_gpu_time_ms: { type: 'number' },
  top_kernels: { type: 'array', items: PROFILE_ROW_SCHEMA },
  shift_note: { type: 'string' }, notes: { type: 'string' },
}, ['profile_topN_json', 'top_kernels']);

const STRATEGY_SCHEMA = obj({
  regime_summary: { type: 'string' }, config_directions: arrObj,
  head_candidates: { type: 'array', items: obj({
    short_name: { type: 'string' }, entity_kind: { type: 'string' }, device_kernel: { type: 'string' },
  }, ['short_name', 'entity_kind', 'device_kernel']) },
  kernel_candidates: arrObj,
  drop_list: arrObj, order_of_work: arrStr, strategy_path: { type: 'string' },
}, ['kernel_candidates']);

// Returned after source extraction and GPU UT validation. Ranking metadata stays in files, but the
// finished envelope travels in the reply (this script can't read files; see RUNTIME SANDBOX
// CONTRACT): `prepared` is the verbatim, gated stdout of `agentx_kernel_preparation.mjs
// --emit-prepared`, shaped like STRATEGY_SCHEMA's queues.
const AGENTX_PREPARATION_SCHEMA = obj({
  delivery_path: { type: 'string' },
  prepared: obj({
    head_candidates: arrObj, kernel_candidates: arrObj,
    failed_tasks: arrObj, deferred_tasks: arrObj,
    profile_workload_json: { type: 'string' }, kernel_ranking_json: { type: 'string' },
    kernel_extraction_delivery_json: { type: 'string' },
  }, ['head_candidates', 'kernel_candidates']),
}, ['delivery_path', 'prepared']);

// Result of the ensure_flydsl provisioning gate (build-on-demand). ok=true iff flydsl is importable
// (flydsl + kernels.moe_gemm_2stage) after sourcing env_file; built distinguishes a fresh build from reuse.
const FLYDSL_GATE_SCHEMA = obj({
  ok: { type: 'boolean' }, built: { type: 'boolean' },
  env_file: { type: 'string' }, version: { type: 'string' }, reason: { type: 'string' },
}, ['ok']);

const SWEEP_SCHEMA = obj({
  trials: arrObj, accepted_flags: { type: 'string' }, accepted_env: { type: 'string' },
  best_throughput_tok_s: { type: 'number' }, throughput_speedup_vs_baseline: { type: 'number' },
  summary: { type: 'string' },
  // Filled only by the warm-start repair pass (so not `required`): the structured "why did this not
  // run here", read instead of regexing the notes.
  repair: obj({
    attempted: { type: 'boolean' },
    classification: { type: 'string' },   // upstream_gone | conflicts_with_baseline | env_unsupported | mixed | unknown
    dropped_flags: { type: 'array', items: { type: 'string' } },
    reason_by_flag: { type: 'object', additionalProperties: true },
    launch_error: { type: 'string' },
  }, []),
}, ['accepted_flags', 'best_throughput_tok_s']);

// `e2e_store.py resolve` output, passed through verbatim. Nothing is required or typed as a number:
// an unwritten page answers with empty candidates + `read_reason`, and `speedup` can be null.
const KB_RESOLVE_SCHEMA = obj({
  tried: arrStr, canonical_id: { type: 'string' }, match_tier: { type: 'string' },
  ranked_by: { type: 'string' }, candidates: arrObj, read_reason: { type: 'string' },
  // `plane` is what was asked for; `read_plane` is which one answered (they differ under `both`).
  plane: { type: 'string' }, read_plane: { type: 'string' },
  // How the offer is ordered (throughput, high to low) vs the metric the rung crowns its champion
  // on; a coarse rung is ordered by throughput but still promotes on speedup.
  sorted_by: { type: 'string' }, champion_metric: { type: 'string' },
  // Why a rung came back empty, in counts ("never recorded" vs "filtered by the floor"). Its
  // `read_plane` is the first rung that saw anything, not necessarily the one that answered.
  curation: { type: 'object', additionalProperties: true },
}, []);
// Result of the tuning-skillset phase. pre/post are its own isolated-server A/B legs (not the run
// baseline), so tuning_delta_pct is an attributable share. No accept is banked without
// engagement_verified.
const TUNING_SCHEMA = obj({
  ran: { type: 'boolean' }, mode: { type: 'string' }, skills_used: arrStr,
  preflight: { type: 'object', additionalProperties: true },
  ops_tuned: arrObj, artifacts: arrStr,
  // Deploy bundle for tuned DATA (config tables a library reads from its package dir, caches to
  // drop), which the PYTHONPATH overlay can't carry: EVAL_DIR/tuning/deploy/{MANIFEST.json,
  // tuning_patch.diff,files/,deploy.sh}, folded into EVAL_DIR/final/ by Finalize.
  deploy_bundle: { type: 'string' }, deploy_verified: { type: 'boolean' },
  cache_invalidation: arrStr,
  // DATA paths the deploy writes inside an installed package tree (e.g. aiter/configs/*.csv): the
  // carve-out from the Integrator's clean-live-tree check. Data only; code goes in apply_overlay.
  live_tree_files: arrStr,
  // Code a tuned artifact needs in order to bind (e.g. a routing switch), shipped as a reversible
  // overlay like a head env-winner's routing patch, never as a live-tree .py edit.
  apply_overlay: { type: 'string' },
  apply_env: { type: 'string' }, apply_flags: { type: 'string' },
  correctness_gate: { type: 'string' }, accuracy_note: { type: 'string' },
  engagement_verified: { type: 'boolean' }, engagement_evidence: { type: 'string' },
  pre_tune_throughput_tok_s: { type: 'number' }, post_tune_throughput_tok_s: { type: 'number' },
  noise_floor_pct: { type: 'number' }, tuning_delta_pct: { type: 'number' },
  tuning_speedup: { type: 'number' },
  ab_interleaved: { type: 'boolean' }, ab_complete: { type: 'boolean' },
  gate: { type: 'string', enum: ['accepted', 'no_win', 'rejected', 'incomplete', 'skipped'] },
  report_path: { type: 'string' }, summary: { type: 'string' }, reason: { type: 'string' },
}, ['gate']);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  config_directions: arrObj, head_candidates: arrObj, kernel_candidates: arrObj,
}, ['stop']);

// Extraction smoke gate: exactly lowercase "pass" or "fail" (every gate tests `=== 'pass'`); the
// enum makes StructuredOutput reject free text, which belongs in smoke_detail / notes.
const SMOKE_STATUS = { type: 'string', enum: ['pass', 'fail'] };
// Large machine verdicts travel as immutable task-local files, not LLM summaries.
const SELECTION_VERDICT_REF_SCHEMA = obj({
  path: { type: 'string' }, sha256: { type: 'string' },
}, ['path', 'sha256']);
const EXTRACTION_SELECTION_FIELDS = {
  selection_validation_ref: SELECTION_VERDICT_REF_SCHEMA,
  source_delivery_ref: SELECTION_VERDICT_REF_SCHEMA,
  joint_ut: { type: 'object', additionalProperties: true },
  phase_selection_validations: { type: 'array', items: obj({
    regime: { type: 'string' }, target_callable: { type: 'string' }, seam_candidates: arrObj,
    selection_validation: { type: 'object', additionalProperties: true },
    selection_validation_ref: SELECTION_VERDICT_REF_SCHEMA,
  }, ['regime', 'target_callable']) },
};

const EXTRACT_OP_SCHEMA = obj({
  ...EXTRACTION_SELECTION_FIELDS,
  short_name: { type: 'string' }, op_kind: { type: 'string' }, editable: { type: 'boolean' },
  task_dir: { type: 'string' }, shapes: { type: 'object', additionalProperties: true },
  workload_path: { type: 'string' }, // per-(shape,dtype) weighted workload model for this kernel (optional)
  dtype: { type: 'string' }, synthesized: { type: 'boolean' }, regimes_captured: arrStr,
  candidate_backends: arrStr, reference_io_sha256: { type: 'string' },
  target_callable: { type: 'string' }, // module:attr rebind seam for an authored kernel ('' if none)
  baseline_callable: { type: 'string' }, // module:attr of the FROZEN real online kernel (the speedup denominator)
  baseline_frozen: { type: 'boolean' }, // true only when baseline_callable resolves outside the task dir
  device_kernel: { type: 'string' },
  seam_candidates: arrObj,
  selection_validation: { type: 'object', additionalProperties: true },
  smoke: SMOKE_STATUS, smoke_detail: { type: 'string' }, notes: { type: 'string' },
}, ['op_kind', 'task_dir', 'smoke']);

const OPBENCH_SCHEMA = obj({
  short_name: { type: 'string' }, op_kind: { type: 'string' }, provenance_ok: { type: 'boolean' },
  winner_backend: { type: 'string' }, winner_kind: { type: 'string' },
  // null when nothing was timed (0.0 would read as "not faster"); gate on `measured`, not the number.
  isolated_speedup: { type: ['number', 'null'] }, measured: { type: 'boolean' },
  winner_editable: { type: 'boolean' },
  best_known_ms: { type: 'number' },
  recommend_tier_c: { type: 'boolean' }, author_plan: arrObj, tuning_artifact: { type: 'string' },
  apply_env: { type: 'string' }, apply_flags: { type: 'string' }, code_patch: { type: 'string' },
  per_backend: arrObj, parity_note: { type: 'string' },
  gate: { type: 'string', enum: ['have_winner', 'author_recommended', 'no_win', 'harness_error', 'tamper'] },
  harness_suspect: { type: 'boolean' }, reason: { type: 'string' },
}, ['gate', 'isolated_speedup']);

const EXTRACT_SCHEMA = obj({
  ...EXTRACTION_SELECTION_FIELDS,
  short_name: { type: 'string' }, editable: { type: 'boolean' }, task_dir: { type: 'string' },
  source_path_in_sglang: { type: 'string' }, target_callable: { type: 'string' },
  num_cases: { type: 'number' }, regimes_captured: arrStr, candidate_backends: arrStr,
  build: { type: 'boolean' }, unittest_smoke: SMOKE_STATUS, smoke_detail: { type: 'string' },
  // the baseline leg is an ENVIRONMENT (baseline_overlay/ = a frozen CURRENT_OVERLAY snapshot), and
  // candidate_bind is the ONE entry layered on top of it to make the candidate leg.
  candidate_bind: { type: 'object', additionalProperties: true },
  baseline_overlay: { type: 'string' },
  baseline_frozen: { type: 'boolean' }, // true only when baseline_overlay/ was seeded AND candidate_bind is declared
  device_kernel: { type: 'string' },
  seam_candidates: arrObj,
  selection_validation: { type: 'object', additionalProperties: true },
  reference_io_sha256: { type: 'string' }, notes: { type: 'string' },
}, ['editable', 'task_dir', 'unittest_smoke']);

const KERNEL_LAYER_SCHEMA = obj({
  ran: { type: 'boolean' }, kernel_eval_dir: { type: 'string' },
  final_patch: { type: 'string' }, final_geomean: { type: 'number' },
  validation_status: { type: 'string' }, note: { type: 'string' },
}, ['ran', 'final_patch', 'final_geomean']);

const MEASUREMENT_EVIDENCE_SCHEMA = obj({
  measurement_id: { type: 'string' }, candidate_digest: { type: 'string' },
  reference_digest: { type: 'string' }, workload_digest: { type: 'string' }, protocol_digest: { type: 'string' },
  ref_summary: { type: 'string' }, cand_summary: { type: 'string' },
  ref_replicas: { type: 'number' }, cand_replicas: { type: 'number' },
  fresh_pair: { type: 'boolean' }, configs_matched: { type: 'boolean' }, workload_matched: { type: 'boolean' },
}, []);
const INTEGRATE_SCHEMA = obj({
  short_name: { type: 'string' }, provenance_ok: { type: 'boolean' },
  isolated_speedup: { type: 'number' }, pct_gpu_time: { type: 'number' },
  affected_gpu_time: obj({
    candidate_digest: { type: 'string' },
    profile_path: { type: 'string' }, reference_digest: { type: 'string' }, workload_digest: { type: 'string' },
    kernels: { type: 'array', items: obj({
      id: { type: 'string' }, pct_gpu_time: { type: 'number' }, isolated_speedup: { type: 'number' },
      removed: { type: 'boolean' },
    }, ['id', 'pct_gpu_time']) },
  }, ['profile_path', 'candidate_digest', 'reference_digest', 'workload_digest', 'kernels']),
  measurement_evidence: MEASUREMENT_EVIDENCE_SCHEMA,
  speedup_review: obj({
    status: { type: 'string', enum: ['pending', 'confirmed', 'inconclusive', 'rejected'] },
    original_measurement: MEASUREMENT_EVIDENCE_SCHEMA,
    estimate: { type: 'object', additionalProperties: true },
  }, ['status']),
  e2e_throughput_tok_s: { type: 'number' }, e2e_delta_pct: { type: 'number' },
  // ab_complete=true only when both serving legs were measured; a timed-out or hung gate returns
  // gate:'incomplete' (ab_complete=false) with any partial ref_med / cand_med — not a rejection.
  ref_med: { type: 'number' }, cand_med: { type: 'number' },
  ab_complete: { type: 'boolean' },
  output_parity: { type: 'string' },
  // How an accept's correctness was established: 'byte_exact' (greedy byte parity vs the true
  // baseline), 'accuracy' (sampled task-accuracy probe) or 'none'.
  parity_kind: { type: 'string' },
  gate: { type: 'string', enum: ['accepted', 'stack', 'rejected', 'incomplete'] },
  accepted_overlay: { type: 'string' }, reason: { type: 'string' },
}, ['gate', 'e2e_throughput_tok_s']);

const FINALIZE_SCHEMA = obj({
  final_overlay: { type: 'string' }, final_patch: { type: 'string' }, final_launch_script: { type: 'string' },
  final_throughput_tok_s: { type: 'number' }, throughput_speedup: { type: 'number' },
  accepted_kernels: arrStr, accepted_config: { type: 'object', additionalProperties: true }, note: { type: 'string' },
  // Whether the tuning deploy bundle reached final/ and still engages there (reported, not assumed:
  // a silently failed data install reproduces none of the tuning gain).
  tuning_in_bundle: { type: 'boolean' }, tuning_engagement_recheck: { type: 'string' },
}, ['final_throughput_tok_s']);

const EXPERIENCE_SCHEMA = obj({
  playbook_appended: { type: 'boolean' }, insights: arrStr, ledger: arrObj,
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const REPORT_SCHEMA = obj({
  baseline_throughput_tok_s: { type: 'number' }, final_throughput_tok_s: { type: 'number' },
  throughput_speedup: { type: 'number' }, accepted_config: { type: 'object', additionalProperties: true },
  accepted_kernels: arrObj, milestones: { type: 'number' }, report_path: { type: 'string' },
}, ['throughput_speedup', 'report_path']);

const VALIDATE_SCHEMA = obj({
  model_name: { type: 'string' }, baseline_throughput_tok_s: { type: 'number' },
  director_verified_throughput_tok_s: { type: 'number' }, throughput_speedup: { type: 'number' },
  claimed_throughput_tok_s: { type: 'number' }, validation_status: { type: 'string' },
  output_parity: { type: 'string' }, applied_to_original: { type: 'string' },
  final_overlay: { type: 'string' }, final_launch_script: { type: 'string' },
  arbitration_note: { type: 'string' },
}, ['director_verified_throughput_tok_s', 'validation_status']);

const CHECKPOINT_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, path: { type: 'string' }, checkpoint_sha256: { type: 'string' },
}, ['written', 'path', 'checkpoint_sha256']);

// Prompt helpers (mirror the single-kernel workflow).
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// No filesystem here, so a writer agent snapshots mutable replay assets atomically while the
// measured stack still exists, from a structured schema-v2 intent shared by every phase.
let lastCommittedCheckpoint = null;
async function persistE2EValidationCheckpoint(relativePath, intent, timeoutFor = agentTimeoutFor) {
  if (!EVAL_DIR) return null;
  const target = `${EVAL_DIR}/${relativePath}`;
  const payload = {
    schema_version: 2, checkpoint_type: 'e2e_validation', committed: true,
    eval_dir: EVAL_DIR,
    ...(lastCommittedCheckpoint ? { parent_checkpoint: lastCommittedCheckpoint } : {}),
    ...intent,
  };
  const written = await safeAgent(
    `You are an E2E checkpoint writer. Materialize the schema-v2 checkpoint intent below at ` +
    `"${target}". Before writing, inspect the referenced measured artifacts and populate ` +
    `measurement.legs, replay commands, stack digests, and integrity.checkpoint_assets. Copy every ` +
    `script/patch/overlay/table/manifest used to reproduce this measurement beneath the checkpoint's ` +
    `checkpoint_assets/ directory; record each relative snapshot path and SHA-256. Canonicalize JSON ` +
    `(sorted keys, compact separators), calculate checkpoint_sha256 excluding that field, then write via ` +
    `a temporary file, fsync it, os.replace(), and fsync the parent directory. Never infer a missing ` +
    `A/B leg, engagement, correctness, asset, or selected kernel slot. Return the written path and digest.\n\n` +
    '```json\n' + JSON.stringify(payload, null, 2) + '\n```',
    { phase: intent.phase || 'Finalize', label: `persist-e2e-checkpoint:${relativePath}`,
      schema: CHECKPOINT_WRITE_SCHEMA },
    2,
    timeoutFor
  );
  // Candidate-local A/B snapshots are diagnostic evidence, never a replay
  // parent; only a complete accepted stack may advance the digest chain.
  if (
    payload.committed === true && written && written.written === true
    && !relativePath.includes("/candidate_")
  ) {
    lastCommittedCheckpoint = {
      path: relativePath, checkpoint_sha256: written.checkpoint_sha256,
    };
  }
  return written;
}

// Candidate-local A/B records are best-effort, but an accepted config/tuning/overlay/final result
// must not continue without the checkpoint that makes it replayable.
async function requireE2EValidationCheckpoint(relativePath, intent) {
  const written = await persistE2EValidationCheckpoint(relativePath, intent, checkpointTimeoutFor);
  if (!written || written.written !== true || !written.checkpoint_sha256) {
    throw new Error(
      `checkpoint_write_failed: ${relativePath}; accepted result cannot be committed without a replayable checkpoint`
    );
  }
  return written;
}

// Expert-skills prompt injection. '' when the feature is off or the role is not a skills consumer;
// otherwise an advisory pointer telling the agent to Read the fragment and query the skills index.
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role)) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits the current bottleneck/op and whose ` +
    `validation_status is \`validated\`, and treat each as a HIGH-PRIOR candidate to reproduce — advisory ` +
    `only, never overriding your on-box A/B, never reducing a result below the measured baseline.`;
}

// e2e warm start: identity formatting + the prompt hook. These `let`s stay at their off-value
// (null / '') unless Module A runs, and every consumer keys off them.
let KB_DIMS = null;        // the deployment dimensions the Director established during preflight
let KB_REF_DIR = '';       // where Module A left its references; '' when it never ran
let KB_CACHE_DIR = '';     // where Module A materialized the records' FILES (patches, tuned tables)
let KB_REF_VERDICT = '';   // what we MEASURED about the offer, threaded into later roles' Inputs
let KB_READ_PLANE = '';    // which plane answered the read (not implied by `both`)
// Everything this run recalled, report-ready: the address asked, what came back, what re-measuring
// it here produced, and whether it was accepted (fed to the Report role). `asked` is filled even
// with zero candidates: a miss and a never-recorded page are the same 404, so the ladder asked is
// the finding.
let KB_RECALL = { e2e: null, kernel: [] };

// One collection point for the KERNEL plane's recall: every lane return (its `warm_start` block,
// kernel_lane.js) passes through here, so the report shows stored patches a lane adopted.
function noteKernelKB(r, label) {
  // Absorb the nested run's agent timeline (llm_timeline) for the ledger's nested-run accounting.
  if (r && r.llm_timeline) LLM_TL.nested.push(r.llm_timeline);
  const w = r && r.warm_start;
  if (w && typeof w === 'object') {
    KB_RECALL.kernel.push({
      lane: String(label || ''),
      slug: String(w.slug || ''),
      read_reason: String(w.read_reason || ''), match_tier: String(w.match_tier || ''),
      filtered: w.filtered || null,
      candidates: Array.isArray(w.candidates) ? w.candidates.length : 0,
      adopted: !!w.adopted,
      adopted_speedup: w.adopted ? (w.adopted_speedup != null ? w.adopted_speedup : null) : null,
      // Carry the split, not just the geomean: an adopted patch with no own rounds flatters the lane.
      incremental_speedup: w.incremental_speedup != null ? w.incremental_speedup : null,
      rounds_committed: w.rounds_committed != null ? w.rounds_committed : null,
      no_rounds_after_adopt: !!w.no_rounds_after_adopt,
      final_geomean: r.final_geomean != null ? r.final_geomean : null,
      validation_status: String(r.validation_status || ''),
    });
  }
  return r;
}

const shq = (s) => "'" + String(s == null ? '' : s).replace(/'/g, "'\\''") + "'";

// The ONE place e2e KB identity argv is formatted, for both reader (Module A) and writer (Module B):
// a one-segment disagreement silently addresses a different page (see kb/identity.py).
function kbIdentityFlags() {
  if (!KB_DIMS) return '';
  return [`--model ${shq(KB_DIMS.model)}`, `--gfx ${shq(KB_DIMS.gfx)}`,
    `--framework ${shq(BACKEND)}`, `--framework-version ${shq(KB_DIMS.framework_version)}`,
    `--precision ${shq(KB_DIMS.precision)}`, `--rocm-version ${shq(KB_DIMS.rocm_version)}`,
    `--tp ${SERVING_TP}`, `--ep ${SERVING_EP}`,
    // The declared kind, never the shape provenance (whose fallback values would mis-file
    // degraded AgentX runs).
    `--workload-kind ${shq(WORKLOAD_KIND)}`,
    `--isl ${ISL}`, `--osl ${OSL}`, `--conc ${CONC}`].join(' ');
}

// Metadata shared by the persisted return and the KB writer.
function kbResultMetadata() {
  return {
    // Only tp/ep/conc address AgentX. Observed shape is retained outside the content digest.
    workload: Object.assign({}, WORKLOAD, {
      kind: WORKLOAD_KIND, tp: SERVING_TP, ep: SERVING_EP,
    }, AGENTX ? {
      scenario: AGENTX.scenario, corpus: AGENTX.corpus, num_entries: AGENTX.num_entries,
      duration_s: AGENTX.duration_s, metric_basis: AGENTX_METRIC_BASIS,
    } : {}),
    metric_basis: AGENTX ? AGENTX_METRIC_BASIS : '',
    comparability: A.comparability || {},
  };
}

// --store is what the local plane writes into and is meaningless to a remote-only run; open_plane()
// hard-errors without it on plane local|both, so it is not optional there.
function kbPlaneFlags(plane) {
  plane = plane || E2E_KB_PLANE;
  return `--plane ${plane}` + (plane === 'remote' ? '' : ` --store ${shq(E2E_KB_STORE_DIR)}`);
}

// A READ takes one plane: `cmd_resolve` picks it (kb.plane:read_planes; service first, local mirror
// as fallback) and reports it as `read_plane`. The graded axis is not part of the address (one page
// holds several bases), so the reader states it, or mixed units would be ranked together. An
// explicit recall policy requires an exact basis; otherwise the measured basis filters (unlabelled
// records still match).
function kbMetricBasisFlag() {
  return E2E_KB_METRIC.basis ? `--metric-basis ${shq(E2E_KB_METRIC.basis)} ` +
    (E2E_KB_METRIC.strict ? '--require-metric-basis ' : '') : '';
}

function kbResolveScript(args) {
  return KB_ENV_PRELUDE + '\\\n' +
    `python3 ${shq(E2E_STORE_SCRIPT)} resolve ${kbIdentityFlags()} \\\n` +
    `  ${kbMetricBasisFlag()}${kbPlaneFlags(E2E_KB_PLANE)} ${args}`;
}

// Warm-start prompt injection, like expertSkillsBlock: '' when the feature is off, the role is not a
// consumer, or Module A found nothing (KB_REF_DIR is set only by Module A).
function warmStartBlock(role) {
  if (!KB_REF_DIR || !WARM_START_ROLES.has(role)) return '';
  // `tuning_kb=false` is the blind-eval switch; the tuning track's prior lives in the shared KB, so
  // the switch also withholds this block.
  if (role === 'tuning_specialist' && !TUNING_KB_ENABLED) return '';
  return `\n\n## Warm start (a PRIOR run's record — already measured on this box)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/warm_start.md and follow it. The knowledge base ` +
    `offered prior configurations for this exact deployment; they are in ${KB_REF_DIR}/, and ` +
    `${KB_REF_DIR}/measured_on_this_box.md records what happened when THIS run benched them — that ` +
    `file OVERRIDES the stored claims in its siblings wherever the two disagree. A stored number is ` +
    `a hypothesis; only the measured column is evidence.` +
    // The tuning track reads the same pages to check whether its search was already run once, so it
    // gets an extra paragraph rather than a second block.
    (role === 'tuning_specialist' && KB_CACHE_DIR
      ? `\n\nFor YOUR track specifically: an accepted-kernel entry marked \`from tuning skillset\` is a ` +
        `tuned artifact a prior run produced and proved engaged. Its file was downloaded with the ` +
        `record — look under ${KB_CACHE_DIR}/*/files/tuning/ — and the entry names the env var that ` +
        `binds it. Before you start searching a shape, check whether it is already there. If it is, ` +
        `install the artifact, prove engagement the same way you would for your own, and measure it ` +
        `in your pre/post A/B; a recalled artifact still has to earn its accept on THIS box, but it ` +
        `costs one measurement instead of a search. Search only the shapes that came back empty.`
      : '');
}

// Run-wide workload-identity invariant for EVERY role prompt (a trace replay changes what a
// benchmark means); '' unless the caller declared a trace replay.
function workloadIdentityBlock() {
  if (!IS_AGENTX) return '';
  // Late-adopt here too: every role prompt passes through this block.
  ensureMeasuredShape();
  return `
## WORKLOAD IDENTITY — this run measures a TRACE REPLAY, not a fixed ISL/OSL sweep
The served load is the AgentX corpus \`${AGENTX.corpus}\` replayed by aiperf under scenario
\`${AGENTX.scenario}\` (${AGENTX.num_entries} entries, concurrency ${CONC}).
The CLIENT owns the load: it chooses the request mix, the arrival pattern, its own warmup, and the
measurement duration. Consequences you must respect:

* **ISL=${ISL} / OSL=${OSL} is a KERNEL-TARGETING shape, not a benchmark** (provenance:
  ${WORKLOAD_SHAPE_PROVENANCE}${SHAPE_IS_MEASURED ? '' : ' — A PLACEHOLDER, NOT THIS CORPUS'}).
  ${SHAPE_IS_MEASURED
    ? 'It is an average of what was actually served, so it is the right regime to size tiles and\n  synthesize GEMM/attention inputs against. Read it as a centre of mass, not as a shape every\n  request has.'
    : 'It is the synthetic default and does NOT describe this load: the real shape is read off the\n  baseline once it has run, so nothing here has a corpus average compiled into it.\n  Do NOT size tiles or synthesize GEMM shapes from this number — if your phase needs the served\n  shape, take it from the baseline summary (observed_isl/observed_osl), not from this line.'}
  A trace corpus spans orders of magnitude in request length, so there is no single ISL: read the
  real distribution off the run's own export rather than assuming one. NEVER build a synthetic sweep
  at these lengths and NEVER report one as an e2e number — measured on this stack, a synthetic sweep
  came out ~2.76x the trace-replay throughput, so such a number is not a smaller-scale version of
  the real one, it is a different workload.
* **Keep passing \`ISL=<isl> OSL=<osl> CONC=<conc>\` on the bench line exactly as your role file
  shows.** The trace client ignores them. Do not restructure your bench invocation for this workload.
* **Client selection and the metric axis are ALREADY configured** in \`$EVAL_DIR/bench_env.sh\`, which
  \`bench_e2e.sh\` sources beside itself on every invocation. Do NOT set \`BENCH_CLIENT\`,
  \`E2E_METRIC\` or any \`AGENTX_*\` variable yourself, and never delete or edit that file.
* **The graded axis is ${AXIS_LABEL[AGENTX_METRIC_BASIS]}.**
  \`bench_summary.json\` reports it as \`throughput_tok_s_median\` with
  \`metric_basis=${AGENTX_METRIC_BASIS}\`. Always read the metric-neutral key; ${/intvty/.test(AGENTX_METRIC_BASIS)
    ? 'on this axis it is tok/s/user, higher is better. Report `guard_aggregate_output_tok_s_median`\n  beside it.' + (AGENTX_METRIC_BASIS === 'e2e_norm_intvty_p50'
      ? ' Also report `guard_e2e_norm_intvty_p90_median`, the tail guard.' : '')
    : 'on a ~140:1\n  prefill:output trace the output-only axis barely moves for a large real change in work.'}
* **A measured window is LONG**: ${AGENTX.geak_loop_duration_s}s per search leg,
  ${AGENTX.duration_s}s for parity/validation. Budget your phase around that and do not retry a
  timed-out leg blindly.
* **Search legs are deliberately non-canonical.** They run the scenario's ${AGENTX.geak_loop_duration_s}s
  floor rather than the canonical ${AGENTX.duration_s}s, so the client stamps them
  \`submission_valid=false\` with the deviations attached. That is EXPECTED and does not invalidate an
  A/B comparison — both legs deviate identically. It does mean a search-leg number may never be
  reported as a submittable result.
`;
}

// The reserve holds ONE search leg for Finalize, and a search-leg claim is no yardstick for Validate's
// legs (a claim equal to the baseline is canonical). Both are '' off a trace replay.
function agentxFinalizeNote() {
  if (!IS_AGENTX) return '';
  return ` On this trace replay the final bench of step 2 is ONE leg on the ${AGENTX.geak_loop_duration_s}s ` +
    `search window. Do NOT also bench the bundle on the ${AGENTX.duration_s}s canonical window: Validate ` +
    'measures the stock and the final stack there right after this phase, and the final reserve holds time ' +
    "for those legs only. Return the search leg's median as final_throughput_tok_s and say in note that it " +
    'is a search-window number. Omit throughput_speedup: BASELINE_THROUGHPUT was measured on the canonical ' +
    'window, so the ratio of the two is not a speedup.';
}
function agentxClaimNote(claimed, baseline) {
  if (!IS_AGENTX || !(claimed > 0) || claimed === baseline) return '';
  return ` On this trace replay CLAIMED_THROUGHPUT comes from ${AGENTX.geak_loop_duration_s}s search legs, ` +
    `which replay a different, shorter slice of the corpus than your ${AGENTX.duration_s}s legs, so it is ` +
    'not comparable with your numbers. In step 4 drop both claim conditions (validated_win\'s "within the ' +
    'noise band of the claim" and flagged\'s "lower than the claim by more than the band") and arbitrate on ' +
    'your same-session A/B, the baseline and parity. Still report claimed_throughput_tok_s, and say in ' +
    'arbitration_note that it is a search-window number.';
}
// Every signal must agree, so no single adoption path can pass a changed bundle off as Setup's.
function agentxAdoptedNothing(s) {
  return IS_AGENTX && !!s.setup && s.baseline > 0 && !s.accepted && s.tput === s.baseline
    && s.flags === s.setup.flags && s.env === s.setup.env && s.overlay === s.setup.overlay;
}
// Why a trace replay skips Validate, or '': nothing to compare, or less time than its timed windows.
function agentxValidateSkip(adoptedNothing, leftMs) {
  if (!IS_AGENTX) return '';
  if (adoptedNothing) return 'final_is_baseline';
  return leftMs < AGENTX_VALIDATE_WINDOWS_MS ? 'no_time' : '';
}

function correctnessEnvBlock(role) {
  if (!CORRECTNESS_UNSET_ENV.length || !CORRECTNESS_PROBE_ROLES.has(role)) return '';
  const unset = CORRECTNESS_UNSET_ENV.map((v) => `-u ${v}`).join(' ');
  return `\n\n## CORRECTNESS PROBES RUN WITHOUT SIMULATED ACCEPTANCE\n` +
    `This stack serves with simulated speculative-decoding acceptance (${CORRECTNESS_UNSET_ENV.join(', ')}). ` +
    `It keeps throughput comparable, but what it serves is neither deterministic nor the model's own ` +
    `output: the unmodified reference does not reproduce itself, so a greedy-parity or task-accuracy probe ` +
    `run under it measures the simulator, not the change. For EVERY correctness probe, launch BOTH servers ` +
    `(the TRUE baseline and the candidate) with these variables removed, from the process env ` +
    `(\`env ${unset} ...\`) and from EXTRA_ENV, and change nothing else: same flags, same overlay, same ` +
    `TP/GPU. Throughput legs keep them, because those legs ARE the measurement, so a correctness server ` +
    `is never one of the throughput servers. Say in your notes that the probe ran with simulated ` +
    `acceptance off.`;
}

// Optimization roles consume the skill's delivery; only the extractor loads its full procedure.
function kernelExtractBlock(role) {
  if (!KERNEL_EXTRACT_ON || !['system_architect', 'op_benchmarker'].includes(role)) return '';
  return `\n\n## AgentX prepared UTs\n` +
    `Read ${KERNEL_EXTRACT_DIR}/references/workflow-handoff.md. Use the supplied prepared tasks ` +
    `and saved workload/ranking evidence. Selection, cases, weights and source bindings are already ` +
    `validated; do not repeat discovery, sampling or UT authoring, or replace them with fixed ISL/OSL ` +
    `assumptions. Run the existing immutable UT against optimization candidates, then E2E A/B.\n` +
    `Declared workload kind: ${WORKLOAD_KIND}; specification: ${JSON.stringify(WORKLOAD_SPEC)}`;
}

function roleAgent(role, phase, intro, inputs) {
  // BACKEND is injected for every role: any role that calls bench_e2e.sh must forward it
  // (BACKEND=<backend>) so the right serving adapter (scripts/adapters/<backend>.sh) is used.
  const standaloneExtraction = KERNEL_EXTRACT_ON && role === 'kernel_extractor';
  const inall = {
    BACKEND, SERVING_TP, SERVING_GPU,
    MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS, VALIDATION_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST,
    E2E_LEARNED_KB: E2E_LEARNED_KB_ENABLED ? 'on' : 'off',
    // Only surfaced on a trace replay.
    ...(IS_AGENTX ? { WORKLOAD_KIND } : {}),
    ...(correctnessEnvBlock(role) ? { CORRECTNESS_UNSET_ENV: CORRECTNESS_UNSET_ENV.join(' ') } : {}),
    ...inputs,
    ...(standaloneExtraction ? {
      SKILL_ROOT: KERNEL_EXTRACT_DIR, SHARED_TOOLS: `${WORKFLOW_DIR}/scripts`, WORKFLOW_DIR,
      PARSE_PROFILE_STRICT_PHASES: '1', KERNEL_ENV_SKIP_ENUM_REAP: '1',
      CAPTURE_PERSIST_POLICY, CAPTURE_STORAGE_ENV,
    } : {}),
  };
  const instruction = standaloneExtraction
    ? `First Read ${KERNEL_EXTRACT_DIR}/SKILL.md and ${KERNEL_EXTRACT_DIR}/references/workflow-handoff.md.
Use the complete extraction procedure and its shared and skill-specific tools. The handoff reference supplies only workflow inputs and output packaging.
Do not load roles/kernel_extractor.md; it is the legacy procedure for other workloads.`
    : `First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.`;
  const base = `You are the ${role}. PHASE=${phase}.
${instruction}
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
When you invoke bench_e2e.sh, pass BACKEND=${BACKEND} in its env so the correct serving adapter is used.

## SERVING CONFIG INVARIANT (do not violate — all e2e numbers must be comparable)
Every e2e SERVING benchmark in this run (baseline, config sweep, integrate ref/cand, validation,
profiler trace) MUST use the SAME serving config: tensor-parallel TP=${SERVING_TP} on the GPU set
GPU=${SERVING_GPU}. Whenever you invoke bench_e2e.sh for a SERVING throughput/profile measurement, pass
exactly these in its env:
    BACKEND=${BACKEND} TP=${SERVING_TP} GPU=${SERVING_GPU}
NEVER change TP or the GPU set between the baseline, a candidate, and validation — a TP/GPU mismatch
makes every delta meaningless. (If SERVING_TP=1, GPU=${SERVING_GPU} is a single id; if SERVING_TP>1 it
is a comma-separated set spanning exactly TP GPUs.)
GPU_IDS=${GPU_IDS} is a SEPARATE OPTIMIZATION-PARALLELISM pool: it is used ONLY for single-GPU isolated
work (op_bench bake-offs, shape-capture, the recursive kernel layer), where each task pins ONE id from
the pool via GPU_ID. Do NOT use the serving TP/GPU set for that isolated work, and do NOT use a single
optimization-pool id for a serving launch — keep the two separate.
${workloadIdentityBlock()}
## Inputs
${cfg(inall)}

Return ONLY the structured JSON the ${standaloneExtraction ? 'workflow handoff reference' : 'role file'} specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role) + warmStartBlock(role) + kernelExtractBlock(role) + correctnessEnvBlock(role);
}

// Resilient agent wrapper: retry a failed agent a few times, then degrade to null for the callers'
// null-guards (setup re-throws). Each attempt races a timeout that resolves null (the harness turns
// errors into null, not hangs): 120min (45min fast), since agents launch servers and run ~30min
// benches. args.agent_timeout_ms=0 disables it; without setTimeout it is a raw agent().
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : (FAST_MODE ? 2700000 : 7200000), 10);
// The guard abandons the PROMISE, not the agent: a timed-out attempt keeps running its servers and legs,
// so a retry runs beside it on the same GPUs and files. agent_timeout_retry=false degrades to null instead.
const AGENT_TIMEOUT_RETRY = String(A.agent_timeout_retry != null ? A.agent_timeout_retry : 'true') !== 'false';
// Process-safety rules prepended to EVERY agent prompt at the one `agent()` funnel: no
// pattern-matched kills (they can TERM the caller's orchestrator), detach long legs from the tool's
// shell manager (it group-reaps them), and never start a command with `sleep` (Claude Code 2.1.310+
// refuses one of 25s+ with background tasks on; run_e2e.py turns those off too).
const PROCESS_SAFETY = `## PROCESS SAFETY (a violation can kill the caller's orchestrator, failing the whole task)
This container's PID 1 is the CALLER's orchestrator process, not yours, and it is NOT restartable.
NEVER run global or pattern-matched process cleanup: no \`pkill -f\` / \`pgrep -f ... | xargs kill\` /
\`killall\` / \`ps aux | grep ... | xargs kill\`, and never \`kill -- -PGID\` for a group you did not
create. A pattern as innocent as \`-f vllm\` matches the orchestrator's own command line and TERMs it.
Manage ONLY processes you started, by the pid you captured at launch. When a script of yours launches a
server, do NOT hand-roll the shutdown — source the shared teardown contract, which verifies process
identity (pid, pgid, /proc start time) before signalling anything:
    source ${WORKFLOW_DIR}/scripts/server_teardown.sh
    trap server_teardown EXIT
    \${SERVER_LAUNCH_PREFIX:-} <launch server> &   # own session => pgid == pid
    SERVER_PID=\$!; server_record_identity "\$SERVER_PID"

## LONG MEASUREMENTS OUTLIVE YOUR TOOL CALL — DETACH THEM FROM THE SHELL MANAGER
A serving measurement is longer than any bound your Bash tool can give it. One canonical trace-replay
leg is a 3600s window PLUS warmup and finalization (~70min wall), and a cold server boot alone is ~9min.
Your foreground bound is BASH_MAX_TIMEOUT_MS, and your tool's OWN background mode is worse: a shell the
tool manages for you is reaped on the manager's schedule, and the reap is a GROUP kill that takes the
server down with the client — you get a truncated log, an unloaded GPU, and no summary, from a leg that
was measuring perfectly. Do NOT run a bench through your tool's background/\`run_in_background\` mode.
Launch it so it is orphaned to init, OUTSIDE the manager's reach, then wait on its log:
    cd "\$EVAL_DIR" && setsid nohup bash run_bench.sh > run_bench.out 2>&1 < /dev/null &
    # returns immediately; the leg's parent becomes PID 1 and no tool-call bound applies
Then wait in SEPARATE foreground calls, each a bounded loop that checks BEFORE it sleeps and ends inside
the timeout you give that Bash call (the tool caps it at BASH_MAX_TIMEOUT_MS, 10 min by default):
    timeout 540 bash -c 'until [ -s <leg_dir>/bench_summary.json ]; do sleep 30; done'; tail -5 run_bench.out
Never START a command with a \`sleep\` (Claude Code refuses one of 25s or more), and never wait by ending
your turn: your turn ending IS your return, so a Monitor, background-watcher or \`run_in_background\`
notification lands after you are gone. Keep waiting until the summary lands or the leg is provably dead.
Before you conclude a leg died, check whether it is still alive (\`ps -o pid,stat,etime -p <pid>\`) and
whether the GPUs still hold weights — a reaped wrapper with a live leg underneath looks identical to a crash.

## ONCE THE FINAL PHASE STARTS, IT OWNS THE GPUS
When \$EVAL_DIR/.final_phase_fence exists the search is over: the workflow stopped every leg, tuner and
waiter still running from it, on purpose, so Finalize and Validate measure on quiet GPUs. Unless you are
the e2e_integrator finalizing or the director validating, start nothing more on the GPUs (no server, leg,
tuner or profiler), do not relaunch anything of yours that was stopped, and return what you already have.
`;
function withProcessSafety(prompt) {
  return typeof prompt === 'string' ? PROCESS_SAFETY + prompt : prompt;
}

// Near the deadline, cap each OPTIMIZATION agent's hung-guard at (remaining − reserve), so a chain
// of agents in one loop iteration can't overrun the reserve. Final-phase agents are exempt by
// position (FINAL_PHASE_STARTED, set at the phase('Finalize') call site), not label: the
// Finalize-gate tags its agents 'Finalize'. 2min floor; inert without time_budget_s.
let FINAL_PHASE_STARTED = false;
const FINALIZE_GATE_PHASE = 'Finalize-gate';   // display grouping only — not load-bearing for the cap
const AGENT_TIMEOUT_FLOOR_MS = 120000;
function agentTimeoutFor() {
  if (TIME_BUDGET_MS == null || FINAL_PHASE_STARTED) return AGENT_TIMEOUT_MS;
  return Math.max(AGENT_TIMEOUT_FLOOR_MS, Math.min(AGENT_TIMEOUT_MS, remainingMs() - FINAL_RESERVE_MS));
}
// True when a new agent would only get the floor (reserve begun or under one floor away, final phase
// not started): nothing measured fits, and the abandoned agent would keep using the GPUs.
function atReserveFloor() {
  return TIME_BUDGET_MS != null && !FINAL_PHASE_STARTED &&
    remainingMs() - FINAL_RESERVE_MS <= AGENT_TIMEOUT_FLOOR_MS;
}
// A required checkpoint write (copy + hash, no GPU, 4-8min) is not bounded by the reserve cap: up to
// 30min, passed only at requireE2EValidationCheckpoint. A hung writer is not retried beside itself
// (two writers would leave digests disagreeing with assets).
const CHECKPOINT_COMMIT_TIMEOUT_MS = 30 * 60000;
function checkpointTimeoutFor() {
  return Math.max(agentTimeoutFor(), Math.min(AGENT_TIMEOUT_MS, CHECKPOINT_COMMIT_TIMEOUT_MS));
}

// ABLATION ARMS — phase-resumed cost ablations (research/ablations/README.md).
// Inert unless args.ablation_arm names an arm; `''` and `A1` are the control (every helper returns
// its off-value). The audit draw is a seeded hash of (seed, head), reproducible across arms
// (Math.random()/Date.now() throw here). B4-TIME caps wall-clock (the runtime never sees prices);
// B5-EFFORT routes to a lower effort tier of the same model. B4/B5/B45 are accepted aliases.
const ABL_RAW_ARM = String(A.ablation_arm || '').trim().toUpperCase();
const ABL_ALIASES = { B4: 'B4-TIME', B5: 'B5-EFFORT', B45: 'B45-TIME-EFFORT' };
const ABLATION_ARM = ABL_ALIASES[ABL_RAW_ARM] || ABL_RAW_ARM;
const ABL_ACTIVE = ABLATION_ARM !== '' && ABLATION_ARM !== 'A1';
const ABL = (arm) => ABLATION_ARM === arm || ABLATION_ARM === ABL_ALIASES[arm] ||
  (ABLATION_ARM === 'B45-TIME-EFFORT' && (arm === 'B4' || arm === 'B5'));
// The kernel lane is a separate workflow with its own args; B6 lives there, so forward the arm.
function ablLaneArgs() {
  // The control (A1 / unset) forwards nothing.
  if (!ABL_ACTIVE) return {};
  return { ablation_arm: ABL_RAW_ARM, ablation_seed: String(A.ablation_seed || '') };
}
const ABL_SEED = String(A.ablation_seed || 'geak-ablation-2026');
// Pilot wall-clock limits, not calibrated thresholds (see study.json): preparation, the
// implementation trial and an audited continuation each have their own cap. Validation is never
// cut; it is the run's FINAL_RESERVE.
const ABL_IMPL_MS = parseInt(A.ablation_head_trial_ms != null ? A.ablation_head_trial_ms : 1200000, 10);
const ABL_PREP_MS = parseInt(A.ablation_prep_ms != null ? A.ablation_prep_ms : 1200000, 10);
const ABL_AUDIT_MS = parseInt(A.ablation_audit_ms != null ? A.ablation_audit_ms : ABL_IMPL_MS, 10);
const ABL_HEAD_TRIAL_MS = ABL_IMPL_MS;                 // alias used by the event log
const ABL_AUDIT_P = Number(A.ablation_audit_probability != null ? A.ablation_audit_probability : 0.1);
const ABL_EVENTS = [];
function ablEvent(o) { if (ABLATION_ARM) ABL_EVENTS.push(o); }
// B4 needs the elapsed clock, which only time_budget_s arms; without it B4 would silently
// match A1, so refuse to run.
if (ABL('B4') && TIME_BUDGET_MS == null) {
  throw new Error('ablation_arm=' + ABLATION_ARM + ' requires time_budget_s: without it the ' +
    'elapsed clock is never armed and the head trial gate can never fire.');
}
function ablDraw(key) {
  let h = 2166136261; const s = ABL_SEED + ':' + key;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return ((h >>> 0) % 1000000) / 1000000;
}
// B4 readiness (not a gain claim; adoption still needs the A/B): an exact candidate identity hash
// that differs from the frozen incumbent, plus the compile/correctness records the authoring layer
// produced. Integration is reported, never inferred.
function ablCandId(c) {
  const body = [c && c.final_patch, c && c.code_patch, c && c.apply_env, c && c.apply_flags,
    c && c.tuning_artifact, c && c.kernel_eval_dir].map((x) => String(x || '')).join('\u0000');
  let h1 = 2166136261, h2 = 2246822519;
  for (let i = 0; i < body.length; i++) {
    h1 ^= body.charCodeAt(i); h1 = Math.imul(h1, 16777619);
    h2 ^= body.charCodeAt(body.length - 1 - i); h2 = Math.imul(h2, 2654435761);
  }
  return body.trim() ? ((h1 >>> 0).toString(16) + (h2 >>> 0).toString(16)) : '';
}
function ablCandEvidence(c, incumbentId) {
  const id = ablCandId(c);
  return {
    candidate_id: id,
    has_identity: !!id,
    non_incumbent: !!id && id !== incumbentId,
    // final_patch exists only after the kernel layer's oracle passed, and a bake-off winner only
    // after it ran: the only compile/correctness records available here.
    compile_ok: !!(c && (c.final_patch || c.code_patch || c.tuning_artifact || c.apply_env || c.apply_flags)),
    correctness_ok: !!(c && (c.kind === 'authored' ? c.final_patch : c.isolated > 0)),
    integration_ok: !!(c && c.integration_ok),      // only the e2e A/B can set this
    isolated: (c && c.isolated) || 0,
  };
}
function ablHeadReady(cands, ext, incumbentId) {
  if (!(ext && ext.task_dir)) return false;
  return (cands || []).some((c) => {
    const e = ablCandEvidence(c, incumbentId);
    return e.has_identity && e.non_incumbent && e.compile_ok && e.correctness_ok;
  });
}
// B5 router: low effort only for bounded tasks with an objective validator (never authoring,
// oracles or adjudication) and never on a retry, which escalates with the failure attached.
const ABL_CHEAP_LABELS = /^(storage:reclaim|bakeoff |extract_op |roofline )/;
function ablEffortFor(opts, attempt) {
  if (!ABL('B5') || attempt > 0) return null;
  const label = String((opts && opts.label) || '');
  return ABL_CHEAP_LABELS.test(label) ? 'low' : null;
}

function agentBounded(rawPrompt, opts, ablAttempt, onHung, timeoutFor = agentTimeoutFor) {
  const prompt = withProcessSafety(rawPrompt);
  const ablEffort = ablEffortFor(opts, ablAttempt || 0);
  if (ablEffort) {
    opts = { ...(opts || {}), effort: ablEffort };
    ablEvent({ event: 'route', label: (opts && opts.label) || '', tier: 'cheap', effort: ablEffort });
  }
  const timeoutMs = timeoutFor();
  if (typeof setTimeout !== 'function' || !(timeoutMs > 0)) return agent(prompt, opts);
  let to;
  const run = agent(prompt, opts);
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [hung-agent guard] ${(opts && opts.label) || 'agent'} exceeded ${Math.round(timeoutMs / 60000)}min with no return — treating as a failed attempt.`);
      if (onHung) onHung(run);
      resolve(null);
    }, timeoutMs);
  });
  return Promise.race([
    run.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
    guard,
  ]);
}

// Attempts the hung-agent guard abandoned inside the reserve, by label. Each is still running; an
// integrator's eventual return is its A/B verdict (adoptAbandonedVerdict).
const ABANDONED_RUNS = new Map();

async function safeAgent(prompt, opts, tries = 3, timeoutFor = agentTimeoutFor) {
  let lastErr = 'unknown';
  for (let i = 0; i < tries; i++) {
    try {
      // B5-EFFORT is a cascade: the escalated retry sees the cheap attempt's failure (both are
      // charged to the task). Inert for every other arm.
      const p2 = (ABL('B5') && i > 0 && typeof prompt === 'string')
        ? prompt + `\n\n## PREVIOUS ATTEMPT FAILED — do not repeat it\nA cheaper-effort attempt at this exact task failed with:\n\`\`\`\n${String(lastErr).slice(0, 2000)}\n\`\`\`\nDiagnose that failure before acting.\n`
        : prompt;
      const ev = tlAgent(prompt, opts, i + 1);   // record AT DISPATCH; ok=false until it resolves
      let hung = null;
      const r = await agentBounded(p2, opts, i, (run) => { hung = run; }, timeoutFor);
      if (r) {
        if (r.llm_timeline) LLM_TL.nested.push(r.llm_timeline);
        if (ev) ev.ok = true;
        return r;
      }
      const inReserve = !!hung && atReserveFloor();
      if (hung && (!AGENT_TIMEOUT_RETRY || inReserve)) {
        if (inReserve) ABANDONED_RUNS.set((opts && opts.label) || '', hung);
        log(`agent[${(opts && opts.label) || '?'}] DEGRADED to null without a retry: the timed-out attempt ` +
          `may still be running (${inReserve ? 'the final reserve has begun' : 'agent_timeout_retry=false'}).`);
        return null;
      }
      lastErr = 'null/empty result';
    } catch (e) { lastErr = String(e); }
    log(`agent[${(opts && opts.label) || '?'}] attempt ${i + 1}/${tries} failed: ${String(lastErr).slice(0, 160)}`);
  }
  log(`agent[${(opts && opts.label) || '?'}] DEGRADED to null after ${tries} tries (${String(lastErr).slice(0, 120)})`);
  return null;
}

// --- FlyDSL provisioning gate ----
// Once strategize routes a flydsl backend onto any candidate, the orchestrator (not the architect
// LLM) runs the ensure_flydsl skill before any consumer can select it (roles/system_architect.md
// §0a). Reuse-or-build is decided only by ensure_flydsl.sh's own version gate.
let flydslProvisioned = false;   // guard so the two call sites (strategize / re-strategize) provision at most once
const _wantsFlydsl = (c) => ((c && c.candidate_backends) || []).some((b) => String(b).toLowerCase() === 'flydsl');
const flydslRouted = (...queues) => queues.some((q) => (q || []).some(_wantsFlydsl));
function stripFlydslFromQueues(...queues) {
  let n = 0;
  for (const q of queues) for (const c of (q || [])) {
    if (_wantsFlydsl(c)) { c.candidate_backends = (c.candidate_backends || []).filter((b) => String(b).toLowerCase() !== 'flydsl'); n++; }
  }
  return n;
}
async function ensureFlydslGate() {
  if (flydslProvisioned) return;                      // already provisioned this run
  if (!flydslRouted(headQueue, kernelQueue)) return;  // strategize did not route flydsl -> nothing to do
  log('[flydsl-gate] strategize routed a flydsl backend -> running ensure_flydsl (detect -> reuse-or-build) BEFORE any consumer.');
  const g = await safeAgent(
    `You are the flydsl provisioner. Run the ensure_flydsl expert skill EXACTLY as ` +
    `${WORKFLOW_DIR}/../perf_knowledge/expert_skills/skills/ensure_flydsl/skill.md specifies: launch ` +
    `${WORKFLOW_DIR}/../perf_knowledge/expert_skills/skills/ensure_flydsl/ensure_flydsl.sh DETACHED and poll ` +
    `(success = flydsl_env.sh written; failure = .flydsl_build.failed). The script's own version gate ` +
    `(import flydsl AND kernels.moe_gemm_2stage AND __version__ >= FLYDSL_MIN_VERSION, default 0.2.2) is the ` +
    `SOLE authority: reuse an ambient flydsl if satisfied, else clone+build the pin (~15-18min). Do NOT use ` +
    `is_flydsl_available() to decide anything. Return {ok,built,env_file,version,reason}; ok=true iff, after ` +
    `sourcing flydsl_env.sh, "import flydsl, kernels.moe_gemm_2stage" works.`,
    { phase: 'Strategize', label: 'architect:ensure_flydsl', schema: FLYDSL_GATE_SCHEMA });
  if (g && g.ok) {
    flydslProvisioned = true;
    log(`[flydsl-gate] FlyDSL ready (${g.built ? 'built' : 'reused'} ${g.version || '?'}); env=${g.env_file || '(default /opt/flydsl/FlyDSL/flydsl_env.sh)'}.`);
  } else {
    const n = stripFlydslFromQueues(headQueue, kernelQueue);
    log(`[flydsl-gate] ensure_flydsl FAILED (${(g && g.reason) || 'no result'}) -> stripped 'flydsl' from ${n} candidate backend list(s); those heads fall back to their other backends (no silent flydsl selection).`);
  }
}

// A profile identity is a device symbol; the replacement target is a live Python callable. Keep the
// two machine fields separate and require runtime evidence before bake-off or authoring.
const CALLABLE_SPEC_RX = /^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$/;
const validCallableSpec = (s) => CALLABLE_SPEC_RX.test(String(s || '').trim());
// parse_profile.short_name truncates display names at 60 chars; a declared name that hit the limit
// is our own doing, so a prefix match at it is accepted rather than read as a mismatch.
const SHORT_NAME_LIMIT = 60;
// Removes balanced groups innermost-first until the symbol stops changing (a single greedy pass
// would span `k<a>(t<b>)` from the first opener to the last closer).
const stripBalanced = (symbol, opener, closer) => {
  const esc = (c) => c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const rx = new RegExp(`${esc(opener)}[^${esc(opener)}${esc(closer)}]*${esc(closer)}`, 'g');
  let text = symbol;
  for (let previous = null; previous !== text;) {
    previous = text;
    text = text.replace(rx, ' ');
  }
  return text;
};
// Must stay identical to canonical_kernel_name in kernel_selection.py (the JS gate and the Python
// verdict compare the same symbols). Brackets, templates and parentheses are stripped as balanced
// groups, not cut at the first '(' (ROCm's `(anonymous namespace)` precedes the kernel name, and
// annotations like `[clone .1]`, `[prefill]` or `(Device -> Device)` follow it); then the return type
// is dropped and the FIRST identifier taken, as parse_profile.short_name does.
function canonicalDeviceKernel(s) {
  const stripped = stripBalanced(stripBalanced(
    stripBalanced(String(s || ''), '[', ']'), '<', '>'), '(', ')');
  // A leftover opener never closes (the name was elided mid-template), so the name is what precedes
  // it. Gaps left by removed groups are closed around `::` (`at::native:: ::CatArray...`).
  const cut = stripped.split(/[<([]/)[0].trim().replace(/\s*::\s*/g, '::');
  const match = /^[\w:]+/.exec(cut.replace(/^void\s+/, ''));
  return match ? match[0].split('::').pop().toLowerCase().replace(/[^a-z0-9_]+/g, '') : '';
}
// A name at the display limit ends mid-token (no word boundary follows). Called both ways: either
// side may be the shortened one (heads carry short_name, traces the full symbol).
const truncatedPrefix = (needle, haystack) =>
  needle.length >= SHORT_NAME_LIMIT && haystack.startsWith(needle);
// [arguments, closed] for the symbol's first `<...>`, independent of return type and namespace.
// `closed` is false when the name was elided inside the template. Separators fold to `_`, so
// `<128, 4, ...>` is not a prefix of `<128, 48, ...>`.
function templateArguments(value) {
  const text = String(value || ''), start = text.indexOf('<');
  const fold = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_');
  if (start < 0) return ['', true];
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === '<') depth++;
    else if (text[i] === '>' && --depth === 0) return [fold(text.slice(start + 1, i)), true];
  }
  return [fold(text.slice(start + 1)), false];
}
function kernelIdentitiesMatch(a, b) {
  const x = canonicalDeviceKernel(a), y = canonicalDeviceKernel(b);
  if (!x || !y) return false;
  if (x !== y && !truncatedPrefix(x, y) && !truncatedPrefix(y, x)) return false;
  // The base token drops template arguments so a bare name matches its decorated spelling; when
  // BOTH sides carry them they must agree (many kernels differ only by their template functor).
  const [xa, xc] = templateArguments(a), [ya, yc] = templateArguments(b);
  if (!xa || !ya) return true;
  if (xc && yc) return xa === ya;
  // An elided list still has to agree as far as both sides actually spell it out.
  return xa.startsWith(ya) || ya.startsWith(xa);
}
function requiredDeviceKernel(h) {
  if (!h || h.entity_kind !== 'gpu_kernel') return '';
  // entity_evidence.matched_profiled_kernel is what parse_profile.py --annotate stamped on the row,
  // so it is the profiler's own name for this entity and outranks the display short_name.
  return String(h.device_kernel ||
    (h.entity_evidence && h.entity_evidence.matched_profiled_kernel) ||
    h.short_name || h.name || '').trim();
}
const candidateSpec = (c) =>
  String((c && (c.target_callable || c.callable || c.spec)) || '').trim();
// A declared depth orders candidates before any runtime evidence; a missing or non-finite one
// returns null so the candidate is refused (NaN would silently break every comparison).
const candidateDepth = (c) => {
  const depth = Number(c && c.depth);
  return Number.isFinite(depth) ? depth : null;
};
function selectionCandidatesForHead(h, ext) {
  const required = requiredDeviceKernel(h);
  const fused = !!(h && (h.is_fused_kernel === true || h.op_kind === 'moe'));
  const bySpec = new Map();
  for (const candidate of [
    ...((h && h.seam_candidates) || []),
    ...((ext && ext.seam_candidates) || []),
  ]) {
    const spec = candidateSpec(candidate);
    // Architect classifications win for an existing spec; Extractor may append missing inner launchers.
    if (spec && !bySpec.has(spec)) bySpec.set(spec, candidate);
  }
  return Array.from(bySpec.values()).filter((candidate) => {
    if (!candidate || typeof candidate !== 'object' || !validCallableSpec(candidateSpec(candidate)))
      return false;
    const role = String(candidate.role || '').toLowerCase();
    if (role === 'kernel_entry') return false; // native/JIT object identity is unsafe to monkeypatch
    if (candidateDepth(candidate) === null) return false;
    const kernels = Array.isArray(candidate.device_kernels) ? candidate.device_kernels : [];
    if (required && !kernels.some((kernel) => kernelIdentitiesMatch(required, kernel))) return false;
    return (fused
      ? ['outer_wrapper', 'dispatcher', 'op_seam']
      : ['outer_wrapper', 'dispatcher', 'op_seam', 'inner_launcher']).includes(role);
  });
}
function prepareHeadSelection(h) {
  if (!h || typeof h !== 'object') return h;
  // A completed UT already verified its callable; do not replace it with an
  // untested nomination when the optimization queue is admitted.
  if (typeof KERNEL_EXTRACT_ON !== 'undefined' && KERNEL_EXTRACT_ON && h.prepared_ut) return { ...h };
  const out = { ...h };
  const required = requiredDeviceKernel(out);
  if (required && !out.device_kernel) out.device_kernel = required;
  // live_call_seam and malformed target_callable values are prose, never machine targets.
  if (!validCallableSpec(out.target_callable)) out.target_callable = '';
  const fused = out.is_fused_kernel === true || out.op_kind === 'moe';
  const jointPhase = out.joint_ut && (out.joint_ut.phases || []).find(p =>
    (p.required_device_kernels || []).some(k => kernelIdentitiesMatch(k, required)));
  const deployable = selectionCandidatesForHead(jointPhase ? {
    ...out, seam_candidates: jointPhase.seam_candidates || out.seam_candidates,
  } : out).filter((candidate) => {
    if (jointPhase && !(jointPhase.required_device_kernels || []).every(k =>
        (candidate.device_kernels || []).some(v => kernelIdentitiesMatch(k, v)))) return false;
    const role = String(candidate.role || '').toLowerCase();
    return fused ? role === 'op_seam' : ['inner_launcher', 'op_seam'].includes(role);
  });
  deployable.sort((a, b) => candidateDepth(b) - candidateDepth(a));
  out.target_callable = deployable.length ? candidateSpec(deployable[0]) : '';
  out.selection_status = out.target_callable
    ? 'candidate_selected_needs_runtime_verification'
    : 'extractor_must_discover_live_launcher';
  return out;
}
function kernelSelectionVerified(h, ext, jointScopeMembers = null) {
  if (ext && ext.selection_evidence_error) return { ok: false, why: ext.selection_evidence_error };
  if (h && h.dispatch_ut) {
    if (h.joint_ut) return { ok: false, why: 'choose joint_ut or dispatch_ut, not both' };
    const phases = h.dispatch_ut.phases, rows = ext && ext.phase_selection_validations;
    if (!Array.isArray(phases) || !phases.length || !Array.isArray(rows) || rows.length !== phases.length)
      return { ok: false, why: 'dispatch UT requires one selection verdict per declared branch' };
    const seen = new Set();
    for (const spec of phases) {
      if (!spec.regime || !spec.device_kernel || !Array.isArray(spec.seam_candidates) ||
          !spec.seam_candidates.length || seen.has(spec.regime))
        return { ok: false, why: 'dispatch branches must have distinct nonempty regime labels' };
      seen.add(spec.regime);
      const matches = rows.filter(r => r.regime === spec.regime);
      if (matches.length !== 1 || matches[0].target_callable !== ext.target_callable)
        return { ok: false, why: `${spec.regime}: dispatch branch must verify the same whole-op callable` };
      const check = kernelSelectionVerified({ ...h, dispatch_ut: undefined,
        is_fused_kernel: true, device_kernel: spec.device_kernel, seam_candidates: spec.seam_candidates }, matches[0]);
      if (!check.ok) return { ok: false, why: `${spec.regime}: ${check.why}` };
    }
    if ([requiredDeviceKernel(h), ...(h.selected_device_kernels || [])].some(k =>
        !phases.some(p => kernelIdentitiesMatch(k, p.device_kernel))))
      return { ok: false, why: 'dispatch branches omit a selected device kernel' };
    return { ok: true, why: 'verified whole-op dispatch branches independently' };
  }
  if (h && h.joint_ut) {
    const phases = h.joint_ut.phases;
    const results = ext && ext.phase_selection_validations;
    if (!Array.isArray(phases) || !phases.length || !Array.isArray(results))
      return { ok: false, why: 'joint UT requires declared phases and per-phase selection verdicts' };
    if (!phases.some(p => (p.required_device_kernels || []).some(k =>
        kernelIdentitiesMatch(k, requiredDeviceKernel(h)))))
      return { ok: false, why: 'joint UT does not contain its profiled anchor kernel' };
    const regimes = new Set();
    for (const spec of phases) {
      const regime = String(spec.regime || '').trim();
      if (!regime || regimes.has(regime))
        return { ok: false, why: 'joint UT phases must have distinct nonempty regime labels' };
      regimes.add(regime);
      const requiredMembers = spec.required_device_kernels;
      if (!Array.isArray(requiredMembers) || requiredMembers.length < 2 ||
          !requiredMembers.every(k => typeof k === 'string' && k.trim()) ||
          requiredMembers.some((k, i) => requiredMembers.slice(0, i).some(v => kernelIdentitiesMatch(k, v))) ||
          !requiredMembers.some(k => kernelIdentitiesMatch(k, spec.device_kernel)))
        return { ok: false, why: `${regime}: joint phase must declare its anchor and companion kernels` };
      const matches = results.filter(r => r.regime === regime);
      if (matches.length !== 1)
        return { ok: false, why: `${regime}: expected one joint selection verdict` };
      const result = matches[0], verdict = result.selection_validation;
      if (!verdict || verdict.selection_scope !== 'joint_chain')
        return { ok: false, why: `${regime}: a leaf verdict cannot certify a joint chain` };
      const reported = verdict.required_device_kernels || [];
      if (reported.length !== requiredMembers.length ||
          requiredMembers.some(k => !reported.some(v => kernelIdentitiesMatch(k, v))))
        return { ok: false, why: `${regime}: joint verdict has the wrong member kernel set` };
      if (requiredMembers.some(k => !(verdict.chain_members || []).some(m =>
          kernelIdentitiesMatch(k, m.device_kernel) && Number(m.matched_kernel_calls) > 0)))
        return { ok: false, why: `${regime}: missing positive member-kernel execution evidence` };
      if (!Array.isArray(verdict.joint_call_coverage) || !verdict.joint_call_coverage.length ||
          verdict.joint_call_coverage.some(c => !Array.isArray(c.missing_members) || c.missing_members.length))
        return { ok: false, why: `${regime}: incomplete joint invocation coverage` };
      const check = kernelSelectionVerified({ ...h, joint_ut: undefined,
        device_kernel: spec.device_kernel, seam_candidates: spec.seam_candidates || result.seam_candidates },
        result, requiredMembers);
      if (!check.ok) return { ok: false, why: `${regime}: ${check.why}` };
    }
    if (results.length !== phases.length)
      return { ok: false, why: 'joint UT returned undeclared or duplicate phase results' };
    if (!results.some(r => r.target_callable === ext.target_callable))
      return { ok: false, why: 'joint UT primary callable is not one of its verified phase entries' };
    return { ok: true, why: `verified complete joint chains for ${Array.from(regimes).join(', ')}` };
  }
  const required = requiredDeviceKernel(h);
  if (!required) return { ok: true, why: 'not a profiled GPU-head extraction' };
  const target = String((ext && ext.target_callable) || '').trim();
  if (!validCallableSpec(target))
    return { ok: false, why: `target_callable '${target}' is not an exact module:attr spec` };
  const verdict = ext && ext.selection_validation;
  if (!verdict || verdict.contract !== 'kernel_selection')
    return { ok: false, why: 'kernel_selection.py verdict is missing' };
  if (verdict.selection_scope === 'joint_chain' && !jointScopeMembers)
    return { ok: false, why: 'joint-chain selection requires a declared joint UT boundary' };
  if (verdict.ok !== true)
    return { ok: false,
      why: `kernel_selection.py failed: ${(verdict.failed || []).join(', ') || 'unknown'}` };
  if (String(verdict.target_callable || '').trim() !== target)
    return { ok: false,
      why: `selection verdict certifies '${verdict.target_callable || ''}', not '${target}'` };
  if (!kernelIdentitiesMatch(required, verdict.device_kernel || ext.device_kernel))
    return { ok: false,
      why: `selection verdict certifies '${verdict.device_kernel || ext.device_kernel || ''}', ` +
        `not profiled kernel '${required}'` };
  if (Number(verdict.total_calls_observed || 0) <= 0 ||
      Number(verdict.target_marker_calls || 0) <= 0 ||
      Number(verdict.matched_kernel_calls || 0) <= 0)
    return { ok: false, why: 'selection verdict lacks positive call/marker/kernel evidence' };
  const candidates = selectionCandidatesForHead(h, ext);
  if (!candidates.length)
    return { ok: false, why: 'no relevant structured seam_candidates were returned' };
  const tested = new Set((verdict.candidate_targets_tested || []).map((value) => String(value || '').trim()));
  const omitted = candidates.map(candidateSpec).filter((spec) => !tested.has(spec));
  if (omitted.length)
    return { ok: false, why: `selection probe omitted candidate(s): ${omitted.join(', ')}` };
  const selected = candidates.find((candidate) => candidateSpec(candidate) === target);
  const selectedRole = String((selected && selected.role) || '').toLowerCase();
  const fused = h.is_fused_kernel === true || h.op_kind === 'moe';
  if (!selected || (fused ? selectedRole !== 'op_seam'
    : !['inner_launcher', 'op_seam'].includes(selectedRole)))
    return { ok: false,
      why: fused
        ? `fused head must select the whole-operation op_seam, got '${selectedRole || 'unknown'}'`
        : `selected target role '${selectedRole || 'unknown'}' is not a deployable inner/op seam` };
  // "Deepest" is settled by the profiler's OBSERVED nesting (kernel_selection.py's
  // deeper_live_candidates), not the agent-declared `depth`, which can only add a failure.
  const observedDeeper = (verdict.deeper_live_candidates || [])
    .map((value) => String(value || '').trim()).filter(Boolean);
  if (observedDeeper.length)
    return { ok: false,
      why: `deeper live candidate(s) observed across calls/ranks: ${observedDeeper.join(', ')}` };
  const live = new Set((verdict.live_candidate_targets || []).map((value) => String(value || '').trim()));
  const selectedDepth = candidateDepth(selected);
  const deeper = candidates.filter((candidate) =>
    candidateSpec(candidate) !== target && live.has(candidateSpec(candidate)) &&
    candidateDepth(candidate) > selectedDepth);
  if (deeper.length)
    return { ok: false,
      why: `declared-deeper live candidate(s): ${deeper.map(candidateSpec).join(', ')}` };
  if (verdict.deepest_verified !== true)
    return { ok: false, why: 'selected callable is not machine-verified as deepest' };
  return { ok: true, why: `${target} launches profiled kernel ${required}` };
}

// The profiled row, not the Architect's restated entity_kind, is the authority: a head adopts
// gpu_kernel only if every row its device_kernel matches is a named gpu_kernel row (exact names
// first; a canonical match folds `aten::mul` with `mul`). Rows are identified by their most
// specific name, since one short_name is shared across instantiations.
function adoptProfiledEntityKind(head, rows) {
  if (!head || head.entity_kind === 'gpu_kernel' || !head.device_kernel) return false;
  const want = String(head.device_kernel).trim();
  const identity = (r) => [r.device_kernel, r.name, r.short_name]
    .map((v) => String(v || '').trim()).find(Boolean) || '';
  const profiled = (rows || []).filter((r) => r && identity(r));
  let matched = profiled.filter((r) => identity(r) === want);
  if (!matched.length) matched = profiled.filter((r) => kernelIdentitiesMatch(identity(r), want));
  if (!matched.length || matched.some((r) => r.entity_kind !== 'gpu_kernel' || !r.device_kernel)) return false;
  head.entity_kind = 'gpu_kernel';
  return true;
}

// A selection verdict may arrive as `selection_validation_ref` (path + SHA-256). This script cannot
// read or hash files (see RUNTIME SANDBOX CONTRACT), so the extractor's preflight
// (scripts/kernel_extraction_delivery.mjs::hydrateExtractionVerdicts) resolves it. Backstop: a ref
// still unresolved here becomes an explicit evidence error, never an inline PASS.
function guardUnhydratedVerdicts(ext) {
  if (!ext || typeof ext !== 'object') return ext;
  // A hydration that ran and failed already gave the more specific diagnosis, so it wins.
  if (ext.selection_evidence_error) return ext;
  const phases = Array.isArray(ext.phase_selection_validations) ? ext.phase_selection_validations : [];
  const unresolved = (row) => !!(row && row.selection_validation_ref && !row.selection_validation);
  if (!unresolved(ext) && !phases.some(unresolved)) return ext;
  return { ...ext, selection_evidence_error:
    'selection_validation_ref arrived unresolved: the workflow sandbox cannot read or hash files. ' +
    'Run scripts/kernel_extraction_delivery.mjs as the extraction preflight so the verdict is ' +
    'hydrated into selection_validation before it is returned.' };
}

const PRE_FLAGGED_HEADS = [];
function admitHeads(queue, stage) {
  const admitted = [];
  const profiledRows = (profile && profile.top_kernels) || [];
  const unnamed = profiledRows.filter((r) => r && r.entity_kind === 'gpu_kernel' && !r.device_kernel);
  if (unnamed.length)
    log(`  ⚠️ profile contract: ${unnamed.length} gpu_kernel row(s) carry no device_kernel ` +
      `(${unnamed.map((r) => r.short_name || r.name || '?').join(', ')}); heads cannot adopt ` +
      `entity_kind from them (${stage}).`);
  for (const head of (queue || []).filter(Boolean)) {
    const label = head.short_name || head.name || '(unnamed)';
    const restated = head.entity_kind;
    if (adoptProfiledEntityKind(head, profiledRows))
      log(`  [admit] ${label}: entity_kind=${restated || 'missing'} -> gpu_kernel from the profiled row (${stage}).`);
    if (head.entity_kind !== 'gpu_kernel') {
      log(`  ⚠️ FLAG ${label}: entity_kind=${head.entity_kind || 'missing'}; ` +
        `the head track requires a profiler-confirmed gpu_kernel (${stage}).`);
      PRE_FLAGGED_HEADS.push({ short_name: label,
        pct_gpu_time: head.pct_gpu_time, stage, gate: 'wrong_head_granularity',
        reason: `entity_kind=${head.entity_kind || 'missing'}; gpu_kernel required` });
      continue;
    }
    // A gpu_kernel head with no device identity would pass kernelSelectionVerified vacuously
    // ("not a profiled GPU-head extraction"), so refuse it at admission.
    if (!requiredDeviceKernel(head)) {
      log(`  ⚠️ FLAG ${label}: entity_kind=gpu_kernel but no device_kernel/short_name/name; ` +
        `there is no profiled symbol to verify a seam against (${stage}).`);
      PRE_FLAGGED_HEADS.push({ short_name: label,
        pct_gpu_time: head.pct_gpu_time, stage, gate: 'missing_kernel_identity',
        reason: 'gpu_kernel head carries no device_kernel/short_name/name' });
      continue;
    }
    admitted.push(prepareHeadSelection(head));
  }
  return admitted;
}

// A FROZEN baseline is resolvable when the extractor seeded baseline_overlay/ + declared
// meta.candidate_bind (kernel track), or set an importable meta.baseline_callable (op track).
// That is the language-independent speedup denominator.
const hasFrozenBaseline = (ext) =>
  !!(ext && (ext.baseline_frozen === true ||
             (ext.candidate_bind && typeof ext.candidate_bind === 'object') ||
             (typeof ext.baseline_callable === 'string' && ext.baseline_callable.trim() !== '')));

// Why an extraction failed, for logs / ledger / flagged heads. The status field is an enum
// ("pass"/"fail"), so the prose lives in smoke_detail; notes may be unrelated, so read both.
const extractFailWhy = (e) => !e ? 'none'
  : [e.smoke_detail, e.notes].filter(Boolean).join(' | ') || e.smoke || e.unittest_smoke || 'none';

// Re-asserts in pure JS what the shipped envelope must still prove. agentx_kernel_preparation.mjs
// gated the on-disk RETURN.json while packing, but its output reaches us through an agent (see
// RUNTIME SANDBOX CONTRACT), so a truncated or hand-built reply must not pass. Checked at intake
// (readAgentXPreparation) and at dispatch (reusePreparedUT); on-disk hashes stay preparation's job.
function preparedUtDefect(kernel, result, phase) {
  if (!result || typeof result !== 'object')
    return 'prepared UT carries no validated result; re-pack the delivery with --emit-prepared';
  if (typeof result.task_dir !== 'string' || !result.task_dir.trim())
    return 'prepared UT result names no task_dir';
  if (result.editable === false) return 'prepared UT task is not editable';
  // extract reports unittest_smoke, extract_op reports smoke -- same split checkPreparedUT uses.
  const status = phase === 'extract' ? result.unittest_smoke : result.smoke;
  if (status !== 'pass') return `prepared UT did not pass (${phase}: ${status ?? 'no status'})`;
  if (result.baseline_frozen !== true)
    return 'prepared UT has no frozen baseline, so there is no speedup denominator';
  if (!kernel || typeof kernel.pct_gpu_time !== 'number' ||
      !Number.isFinite(kernel.pct_gpu_time) || kernel.pct_gpu_time < 0 || kernel.pct_gpu_time > 100)
    return 'prepared candidate has no attributed GPU-time share in [0, 100]';
  // `attention` and `attn` are one kind (agentx_kernel_preparation.mjs::opKind); a declaration that
  // disagrees with its own result is a defect, not a routing choice.
  const canonical = (v) => v === 'attention' ? 'attn' : v;
  const requested = canonical(kernel.op_kind), returned = canonical(result.op_kind);
  if (requested && returned && requested !== returned)
    return `op_kind differs between declaration and result: ${requested} / ${returned}`;
  return '';
}

// Hand an optimization lane the UT preparation validated, after re-asserting the invariants above;
// no prepared_ut.result fails closed. Known limit: a task dir mutated after packing is not detected.
function reusePreparedUT(kernel) {
  const fail = (why) => ({ smoke: 'fail', unittest_smoke: 'fail',
    smoke_detail: `prepared UT is no longer valid: ${why}` });
  const ref = kernel && kernel.prepared_ut;
  if (!ref) return fail('missing prepared UT; complete the extraction stage before optimization');
  const defect = preparedUtDefect(kernel, ref.result, ref.phase);
  if (defect) return fail(defect);
  const canonical = (v) => v === 'attention' ? 'attn' : v;
  return { ...ref.result, op_kind: canonical(ref.result.op_kind) || canonical(kernel.op_kind),
    smoke: 'pass', unittest_smoke: 'pass' };
}

// AgentX reuses its prepared UT. Otherwise run kernel_extractor and, while smoke passes but the
// baseline is not frozen or the kernel selection is unverified, re-invoke it with a corrective (up to
// BASELINE_EXTRACT_RETRIES); if still incomplete, return an extract failure so the head is flagged
// or skipped, never timed against its own scaffold. Failures stay local to the caller; GPU work
// uses the skill's per-device locks.
async function extractWithBaseline(role, phase, intro, inputs, opts) {
  const smokeOk = (e) => !!(e && e.task_dir && (e.smoke === 'pass' || e.unittest_smoke === 'pass'));
  const head = (inputs && inputs.KERNEL) || {};
  if (typeof KERNEL_EXTRACT_ON !== 'undefined' && KERNEL_EXTRACT_ON) return reusePreparedUT(head);
  const captureIntro = `${intro} CAPTURE STORAGE BOUNDS (issue #429): every capture EXTRA_ENV MUST include ` +
    `\`${CAPTURE_STORAGE_ENV}\`. Use kernel_selection.py with --task-dir "$TASK" so the selected oracle is ` +
    `promoted and all capture.pid-* dirs are reclaimed. Unittests for large MoE oracles MUST use ` +
    `h.iter_eager_cases_from_oracle / h.check_correct_multi_lazy.`;
  let ext = await safeAgent(roleAgent(role, phase, captureIntro, {
    ...(inputs || {}),
    CAPTURE_STORAGE_ENV,
    CAPTURE_BYTE_BUDGET,
    CAPTURE_CASE_BYTE_LIMIT,
    CAPTURE_PERSIST_POLICY,
  }), opts);
  ext = guardUnhydratedVerdicts(ext);
  let tries = 0;
  const attemptedTargets = [];
  const complete = (e) => hasFrozenBaseline(e) && kernelSelectionVerified(head, e).ok;
  while (smokeOk(ext) && !complete(ext) && tries < BASELINE_EXTRACT_RETRIES) {
    tries++;
    const selection = kernelSelectionVerified(head, ext);
    const needBaseline = !hasFrozenBaseline(ext);
    const priorTarget = String((ext && ext.target_callable) || '').trim();
    if (priorTarget && !attemptedTargets.includes(priorTarget)) attemptedTargets.push(priorTarget);
    const selectionCorrective = selection.ok ? '' :
      ` PRIOR ATTEMPT DID NOT SELECT THE PROFILED GPU KERNEL: ${selection.why}. ` +
      'Treat KERNEL.live_call_seam as prose only. Merge KERNEL.seam_candidates with any missing inner ' +
      'launcher found from source/runtime inspection; preserve existing candidate classifications. ' +
      'BEFORE re-running capture, reclaim prior process-local artifacts: ' +
      '`python3 "$SKILL_DIR/scripts/capture_shapes.py" --cleanup-task-dir "$TASK" --no-promote` ' +
      '(issue #429 — never accumulate capture.pid-* oracles across retries). ' +
      'Install safe markers for every relevant candidate, never native/JIT kernel_entry objects. Run ' +
      'kernel_selection.py over every process-local capture and all root-call traces with --task-dir "$TASK". ' +
      'Return its JSON verbatim as selection_validation. Select the deepest live inner_launcher/op_seam across all ' +
      'calls/ranks; a fused head must select the whole-op op_seam. Rejecting the previous outer wrapper ' +
      'is not success. Do not return any ATTEMPTED_TARGET_CALLABLES value again.';
    const baselineCorrective = needBaseline ?
      ' PRIOR ATTEMPT DID NOT FREEZE A BASELINE. You MUST seed baseline_overlay/ from ' +
      'CURRENT_OVERLAY (the live serving stack = the speedup denominator), declare meta.candidate_bind ' +
      '(the ONE overlay entry built from kernel_src/), prove both legs differ via h.assert_legs_differ, ' +
      'then return baseline_frozen:true. An extraction with no frozen baseline is INVALID and will be discarded.' : '';
    log(`  ${(opts && opts.label) || role}: extraction contract incomplete ` +
      `(${selection.ok ? 'kernel selected' : selection.why}; ` +
      `${needBaseline ? 'baseline missing (baseline_overlay/ + meta.candidate_bind)' : 'baseline frozen'}). ` +
      `RE-EXTRACTING (retry ${tries}/${BASELINE_EXTRACT_RETRIES}).`);
    // Best-effort capture reclaim before the agent retries.
    if (ext && ext.task_dir) {
      try {
        const { execFileSync } = require('child_process');
        execFileSync('python3', [
          `${WORKFLOW_DIR}/scripts/capture_shapes.py`,
          '--cleanup-task-dir', String(ext.task_dir),
          '--no-promote',
        ], { stdio: 'pipe', timeout: 120000 });
        log(`  ${(opts && opts.label) || role}: reclaimed capture artifacts under ${ext.task_dir}`);
      } catch (cleanupErr) {
        log(`  ${(opts && opts.label) || role}: capture reclaim skipped (${cleanupErr && cleanupErr.message ? cleanupErr.message : cleanupErr})`);
      }
    }
    ext = await safeAgent(
      roleAgent(role, phase, captureIntro + selectionCorrective + baselineCorrective, {
        ...(inputs || {}),
        CAPTURE_STORAGE_ENV,
        CAPTURE_BYTE_BUDGET,
        CAPTURE_CASE_BYTE_LIMIT,
        CAPTURE_PERSIST_POLICY,
        PRIOR_TARGET_CALLABLE: priorTarget,
        PRIOR_SELECTION_VALIDATION: (ext && ext.selection_validation) || {},
        ATTEMPTED_TARGET_CALLABLES: attemptedTargets.slice(),
      }),
      opts);
    ext = guardUnhydratedVerdicts(ext);
  }
  const finalSelection = kernelSelectionVerified(head, ext);
  if (smokeOk(ext) && !finalSelection.ok) {
    log(`  ${(opts && opts.label) || role}: kernel selection still unverified after ` +
      `${BASELINE_EXTRACT_RETRIES} re-extractions — ABORTING (${finalSelection.why}).`);
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail', selection_failed: true,
      smoke_detail: `workflow forced fail: kernel selection failed: ${finalSelection.why}`,
      notes: `kernel selection failed: ${finalSelection.why} — ${ext.notes || ''}` };
  }
  if (smokeOk(ext) && !hasFrozenBaseline(ext)) {
    log(`  ${(opts && opts.label) || role}: STILL no frozen baseline after ${BASELINE_EXTRACT_RETRIES} ` +
      `re-extractions — ABORTING this extraction (refusing a fake speedup vs the candidate's own scaffold).`);
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail',
      smoke_detail: `workflow forced fail: no frozen baseline after ${BASELINE_EXTRACT_RETRIES} re-extractions`,
      notes: `no frozen baseline after ${BASELINE_EXTRACT_RETRIES} re-extractions ` +
        `(baseline_overlay/ + meta.candidate_bind required as the speedup denominator) — ${ext.notes || ''}` };
  }
  // source_delivery_required is set only by the AgentX preparation reader, whose mode returned at
  // reusePreparedUT (verifySourceDelivery ran at packing). Reaching here means source evidence is
  // required on a path that cannot hash kernel_src (no filesystem), so fail closed.
  if (smokeOk(ext) && head.source_delivery_required === true) {
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail', source_delivery_failed: true,
      smoke_detail: 'source delivery incomplete: this head requires source evidence, but it was ' +
        'extracted outside the AgentX preparation stage, which is what verifies it' };
  }
  return ext;
}

// abDone == the integrator measured BOTH legs (ref + cand) and emitted a real
// verdict. gate:'incomplete' or ab_complete:false means a leg is still missing.
const abDone = (integ) => !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);

// Publish an e2e delta with its two medians from the same A/B (ref_med before the win, cand_med
// after), so consumers can restate wins against one fixed baseline; bare percentages don't compose.
const e2eFrom = (integ) => ({
  e2e_delta_pct: integ.e2e_delta_pct,
  base_tput: integ.ref_med,
  new_tput: integ.cand_med,
});
// The KERNEL_RESULT a given integrate A/B gated. Call sites read both patch spellings
// (head=code_patch, milestone=final_patch; tryCorrectiveReauthor sets both).
const krOf = (inputs) => (inputs && inputs.KERNEL_RESULT) || {};

// Bank an accepted win with the fields a KB reader needs to replay it elsewhere. e2e_store.py's
// _accepted_kernels keeps every field but looks up `name`, `language`, `isolated_speedup` and
// `patch`, so those aliases sit beside short_name/backend/isolated (read by the report, resume state
// and run_e2e.py). `e.patch` wins: a corrective re-author supplies the fixed patch.
const bankAccepted = (list, e, kr) => {
  const k = kr || {};
  list.push({
    ...e,
    name: e.short_name || '',
    language: e.backend || '',
    isolated_speedup: Number(e.isolated) || 0,
    winner_kind: e.kind || k.winner_kind || '',
    patch: e.patch || k.code_patch || k.final_patch || '',
    target_callable: k.target_callable || '',
    source_path_in_sglang: k.source_path_in_sglang || '',
    apply_env: e.apply_env || k.apply_env || '',
    apply_flags: e.apply_flags || k.apply_flags || '',
    pct_gpu_time: e.pct_gpu_time != null ? e.pct_gpu_time : (k.pct_gpu_time || 0),
  });
};

// An integrator abandoned at the reserve boundary is still measuring; with no FINISH retry allowed
// inside the reserve, its late return is this A/B's only verdict. Wait for it through at most the
// first LATE_VERDICT_RESERVE_FRAC of the reserve, never into AGENTX_FINAL_PHASE_MS.
const LATE_VERDICT_RESERVE_FRAC = 0.25;
async function adoptAbandonedVerdict(integ, label) {
  const run = ABANDONED_RUNS.get(label);
  ABANDONED_RUNS.delete(label);
  if (integ || !run || !atReserveFloor() || typeof setTimeout !== 'function') return integ;
  const waitMs = remainingMs() - Math.max(Math.ceil(FINAL_RESERVE_MS * (1 - LATE_VERDICT_RESERVE_FRAC)),
    AGENTX_FINAL_PHASE_MS);
  if (!(waitMs > 0)) return integ;
  log(`  ${label}: abandoned at the reserve boundary — waiting up to ${Math.round(waitMs / 60000)}min ` +
    'for its own verdict instead of starting a FINISH retry.');
  let to;
  const late = await Promise.race([
    run.then((r) => r || null, () => null),
    new Promise((resolve) => { to = setTimeout(() => resolve(null), waitMs); }),
  ]);
  clearTimeout(to);
  if (!late) {
    log(`  ${label}: no late verdict within ${Math.round(waitMs / 60000)}min; the A/B stays incomplete.`);
    return integ;
  }
  if (late.llm_timeline) LLM_TL.nested.push(late.llm_timeline);
  log(`  ${label}: ADOPTED the late verdict (${late.gate || '?'}${late.e2e_delta_pct != null ? `, ${late.e2e_delta_pct}%` : ''}).`);
  return late;
}

// Run ONE integrate A/B and GUARANTEE both legs complete: if the integrator returns incomplete,
// re-invoke it in resume mode for the missing leg (up to AB_FINISH_RETRIES). Every integrate site
// uses this, so an A/B never ends after the reference leg alone.
const SPEEDUP_REVIEW_RESULTS = new Map();
async function reviewSpeedup(integ, inputs, label, phaseName) {
  const kr = inputs.KERNEL_RESULT || {};
  const pct = kr.pct_gpu_time || 0, isolated = kr.verified_isolated_speedup || 0;
  if (!abDone(integ) || !['accepted', 'stack'].includes(integ.gate) || integ.output_parity === 'fail'
      || !speedupNeedsReview(pct, isolated, integ) || confirmedSpeedup(integ)) return integ;
  if (integ.speedup_review && ['pending', 'inconclusive', 'rejected'].includes(integ.speedup_review.status)) return integ;
  const original = integ.measurement_evidence || {};
  const estimate = speedupEstimate(pct, isolated, integ);
  // Cache by measurement AND candidate/reference inputs. Returning the same cached A/B cannot
  // restart a review, including when Finalize revisits a pending integration.
  const key = JSON.stringify([original, kr, inputs.CURRENT_OVERLAY, inputs.CURRENT_FLAGS,
    inputs.CURRENT_ENV, inputs.WORKLOAD, integ.ref_med, integ.cand_med]);
  if (SPEEDUP_REVIEW_RESULTS.has(key)) return SPEEDUP_REVIEW_RESULTS.get(key);
  const pending = { ...integ, speedup_review: { status: 'inconclusive', original_measurement: original, estimate },
    reason: 'speedup_review_inconclusive: independent A/B confirmation unavailable' };
  SPEEDUP_REVIEW_RESULTS.set(key, pending);
  log(`  ${label}: speedup review (+${integ.e2e_delta_pct}% vs estimate +${estimate.estimate_pct.toFixed(2)}%, ` +
    `${estimate.pct_gpu_time}% GPU, ${estimate.source}); one independent A/B, no kernel rewrite.`);
  if (!['measurement_id', 'candidate_digest', 'reference_digest', 'workload_digest', 'ref_summary', 'cand_summary']
      .every((field) => typeof original[field] === 'string' && original[field].length > 0)) {
    pending.reason = 'speedup_review_inconclusive: original A/B provenance missing; cannot bind a confirmation';
    return pending;
  }
  if (searchClosed(`${label}: speedup review`)) return pending;
  const reviewInputs = { ...inputs, FORCE_FRESH_AB: true, RESUME_AB: false, REUSE_REF: false,
    REPLICAS: Math.max(2, Number(inputs.REPLICAS) || 1),
    CAND_OVERLAY_DIR: integ.accepted_overlay || '',
    SPEEDUP_REVIEW: { original_measurement: original, estimate },
  };
  delete reviewInputs.SHARED_REF_MED;
  // A confirmation abandoned at the boundary confirms nothing.
  if (abWindowShort(abLegs(reviewInputs), `${label}: speedup review`, reviewInputs.REPLICAS)) {
    pending.reason = 'speedup_review_inconclusive: deferred, the confirmation A/B cannot finish before the final reserve';
    return pending;
  }
  const fresh = await safeAgent(roleAgent('e2e_integrator', 'integrate',
    'Confirm this surprising speedup with ONE independent A/B. Follow FORCE_FRESH_AB in your role: ' +
    'use separate ref/cand artifact directories, preserve the exact candidate and current reference, ' +
    'measure at least two replicas per leg and repeat the configured correctness and engagement gates. ' +
    'Do not reuse old benchmark legs or rewrite the kernel. Amdahl is an estimate, never a correctness verdict. ' +
    'Return the fresh measurements and their provenance; if the gain reproduces, accept even above the estimate.',
    reviewInputs), { phase: phaseName, label: `${label} speedup review`, schema: INTEGRATE_SCHEMA }, 1);
  let result = pending;
  if (fresh && fresh.ab_complete === true && fresh.gate !== 'incomplete') {
    const checked = { ...fresh, speedup_review: { status: 'confirmed', original_measurement: original, estimate } };
    if (confirmationEvidenceMatches(checked)) {
      if (fresh.gate === 'accepted' && fresh.output_parity === 'pass'
          && ['accuracy', 'byte_exact'].includes(fresh.parity_kind)) result = checked;
      else if (fresh.gate === 'rejected' || fresh.output_parity === 'fail') {
        result = { ...checked, gate: 'rejected', speedup_review: { ...checked.speedup_review, status: 'rejected' } };
      }
    }
  }
  SPEEDUP_REVIEW_RESULTS.set(key, result);
  return result;
}
async function runIntegrateBothLegs(intro, inputs, label, phaseName) {
  // Incomplete, so callers keep it PENDING, not rejected.
  if (abWindowShort(abLegs(inputs), label)) {
    return { gate: 'incomplete', ab_complete: false,
      reason: 'deferred: the trace-replay A/B cannot finish before the final reserve' };
  }
  // Caller inputs win; the accuracy gate and the tuning carve-out only ever ADD keys (each is {} when
  // off / when no tuning was banked).
  const withTuning = {
    MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
    PROFILE_TOPN_JSON: (profile && profile.profile_topN_json) || '',
    ...ACCURACY_INPUTS, ...tuningIntegrateInputs(), ...inputs,
  };
  let integ = await adoptAbandonedVerdict(await safeAgent(
    roleAgent('e2e_integrator', 'integrate', intro, withTuning),
    { phase: phaseName, label, schema: INTEGRATE_SCHEMA }), label);
  let tries = 0;
  while (!abDone(integ) && tries < AB_FINISH_RETRIES) {
    if (searchClosed(`${label}: FINISH retry`)) break;
    if (abWindowShort(1, `${label}: FINISH retry`)) break;
    tries++;
    log(`  ${label}: A/B incomplete (${integ ? (integ.reason || integ.gate) : 'null/timeout'}); ` +
      `re-invoking integrator to FINISH the missing leg (retry ${tries}/${AB_FINISH_RETRIES}).`);
    const finishLabel = `${label} (finish ${tries})`;
    integ = await adoptAbandonedVerdict(await safeAgent(
      roleAgent('e2e_integrator', 'integrate',
        'FINISH this e2e A/B. A reference leg may already exist on disk under $CB/ref — reuse it and ' +
        'run ONLY the missing candidate leg (or re-run both if no usable ref exists), then return ' +
        'accepted/stack/rejected with ab_complete:true. Do NOT return gate:"incomplete" unless a hard ' +
        'hardware/harness fault persists after this retry. If short on time, keep one isolated search ' +
        'replica per leg so BOTH legs still run.',
        { ...withTuning, RESUME_AB: true }),
      { phase: phaseName, label: finishLabel, schema: INTEGRATE_SCHEMA }), finishLabel);
  }
  integ = await reviewSpeedup(integ, withTuning, label, phaseName);
  if (integ && integ.short_name) {
    // Preserve every candidate's review when several backends share one head short_name.
    const measurementId = (integ.measurement_evidence && integ.measurement_evidence.measurement_id) || withTuning.CAND_TAG || '';
    const candidateId = [integ.short_name, measurementId].filter(Boolean).join('_').replace(/[^A-Za-z0-9_.-]+/g, '_');
    const kr = withTuning.KERNEL_RESULT || {};
    const accepted = integAccepted(integ, kr.pct_gpu_time, kr.verified_isolated_speedup);
    const reviewPending = integ.speedup_review && ['pending', 'inconclusive'].includes(integ.speedup_review.status);
    await persistE2EValidationCheckpoint(`overlay/candidate_${candidateId}/e2e_validation.json`, {
      phase: 'Overlay', validation_level: 'integrator', gate: reviewPending ? 'incomplete' : (integ.gate || 'incomplete'),
      committed: accepted, validation_status: accepted ? 'accepted_intermediate' : (reviewPending ? 'speedup_review_pending' : 'incomplete'),
      integration_result: integ,
      baseline_throughput_tok_s: integ.ref_med || 0,
      final_throughput_tok_s: integ.cand_med || integ.e2e_throughput_tok_s || 0,
      throughput_speedup: (integ.ref_med && integ.cand_med) ? integ.cand_med / integ.ref_med : 0,
      baseline_config: { flags: withTuning.EXTRA_SERVER_ARGS || '', env: withTuning.EXTRA_ENV || '' },
      accepted_config: { flags: withTuning.EXTRA_SERVER_ARGS || '', env: withTuning.EXTRA_ENV || '' },
      accepted_kernels: accepted ? [{ short_name: integ.short_name,
        kernel_slot: integ.target_callable || integ.short_name }] : [],
      accepted_heads: [], final_patch: [], final_overlay: { path: integ.accepted_overlay || '' },
      final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
        workload: WORKLOAD, source_artifact: `${EVAL_DIR}/overlay`,
        acceptance: { gain_exceeds_noise: accepted, correctness_passed: integ.output_parity !== 'fail' } },
      stack: { kernel_slots: accepted ? [{ kernel_slot: integ.target_callable || integ.short_name,
        selected: true, candidate_id: candidateId }] : [] },
      replay: { asset_paths: [`${EVAL_DIR}/bench_e2e.sh`, integ.accepted_overlay].filter(Boolean) },
      integrity: { checkpoint_assets: [] },
    });
  }
  return integ;
}

// Lightweight SURGICAL FIX (Tier 1 of the corrective): one kernel_surgeon agent makes the smallest
// fix that keeps the isolated win, self-verifies on the immutable unittest, and returns a
// kernel_workflow-shaped result, or null to escalate to the heavy re-author.
const SURGEON_SCHEMA = obj({
  fixed: { type: 'boolean' }, final_patch: { type: 'string' }, final_geomean: { type: 'number' },
  eval_dir: { type: 'string' }, root_cause: { type: 'string' }, note: { type: 'string' },
}, ['fixed']);
async function trySurgicalFix(spec, reason, fixClass, attempt) {
  if (!SURGICAL_FIX) return null;
  const intro = `A verified-isolated ${spec.language || 'triton'} kernel winner for op "${spec.short_name}" ` +
    `(${(spec.isolated || 0).toFixed(2)}x isolated) ENGAGED live but was REJECTED at the e2e gate. ` +
    `Reject class = ${fixClass}. Reject reason: "${reason}". Make the SMALLEST possible fix that clears it ` +
    `while KEEPING the algorithm + the isolated win; do NOT re-optimize or re-explore. Self-verify on the ` +
    `IMMUTABLE unittest, then emit a minimal final_patch.`;
  const s = await safeAgent(
    roleAgent('kernel_surgeon', 'surgical_fix', intro, {
      TASK_DIR: spec.task_dir || '', KERNEL_EVAL_DIR: spec.kernel_eval_dir || '',
      CURRENT_PATCH: (spec.base_inputs && spec.base_inputs.KERNEL_RESULT &&
        (spec.base_inputs.KERNEL_RESULT.code_patch || spec.base_inputs.KERNEL_RESULT.final_patch)) || '',
      REJECT_REASON: reason, FIX_CLASS: fixClass, ISOLATED: spec.isolated || 0,
      LANGUAGE: spec.language || 'triton', GPU_ID: spec.gpu_id, KERNEL_WF_DIR,
    }),
    { phase: spec.phase_name || 'HeadKernel', label: `surgeon ${spec.short_name} (${attempt})`, schema: SURGEON_SCHEMA });
  if (s && s.fixed && s.final_patch && (s.final_geomean || 0) > 1.0) {
    return { authored: true, final_patch: s.final_patch, final_geomean: s.final_geomean,
      eval_dir: s.eval_dir || spec.kernel_eval_dir || '', reason: s.root_cause || 'surgical fix' };
  }
  log(`  ${spec.short_name}: surgical fix did not produce a verified patch (${s ? (s.note || s.root_cause || 'not fixed') : 'null'}) — escalating to re-author.`);
  return null;
}

// CORRECTIVE RE-AUTHOR, used by every head-integration site. A candidate that passed the isolated
// oracle and engaged live but was rejected at the e2e gate for a fixable class (rejectClass) gets up
// to HEAD_CORRECTIVE_MAX fix-and-retries of the EXISTING kernel, each re-gated by a both-legs A/B.
// Not charged to HEAD_BUDGET; skipped once the deadline fired. spec.base_inputs is the integrate
// inputs template; spec.cur = {overlay,flags,env,tput}. Returns { banked, integ, isolated, ... }.
async function tryCorrectiveReauthor(spec) {
  let reason = spec.reason || '';
  // Which fix-and-retry class is this reject? '' = terminal (not auto-correctable).
  let fixClass = spec.fix_class || rejectClass(reason);
  const eligible = HEAD_CORRECTIVE_MAX > 0 && !((FAST_MODE && FAST_DEADLINE_HIT) || TIME_DEADLINE_HIT)
    && (spec.kernel_eval_dir || spec.task_dir) && (spec.isolated || 0) > 1.0 && fixClass !== '';
  if (!eligible) return { banked: false };
  const curTput = (spec.cur && spec.cur.tput) || 0;
  // Class-specific, generic instructions: `integration` = wrong posture (JIT/capture/host-sync);
  // `correctness` = wrong live output. Both keep the algorithm and isolated win.
  const INTEGRATION_FIX_TASK =
    `Fix ONLY the integration posture per knowledge/learned/method-cudagraph-safe-integration: ` +
    `precompile/register EVERY (shape-bucket × config) the LIVE workload hits — PREFILL buckets AND decode buckets, ` +
    `every per-bucket tile/config the kernel selects — at WARMUP before capture (an *_overlay_precompile(weights, ` +
    `scales, buckets) hook the integrator calls once, pre-capture) so ALL TP workers load a prebuilt code object ` +
    `instead of lazily compiling in the multiproc warmup (the cause of NO_BINARY_FOR_GPU / capture hang). If the cause ` +
    `is a host-sync (.item()/.cpu()/.sum().item()) on the hot path, remove it and cache weight prep by data_ptr(). ` +
    `Keep the steady-state hot path host-sync-free. ` +
    `If the reject is no_rebind_seam / no_engagement (the overlay bound but never ran on the live path): FIND the ` +
    `method the live server ACTUALLY dispatches (grep the cand server.log for which entry engaged — e.g. the ` +
    `modular 'TritonExperts.apply'/'invoke_fused_moe_triton_kernel' path, NOT a dead legacy '*_impl'), then rebind ` +
    `the overlay at THAT seam and MATCH its call signature so the kernel is invoked. Prove it re-engages ` +
    `(engagement_check > 0) before returning.`;
  const CORRECTNESS_FIX_TASK =
    `The kernel is WRONG on the LIVE path (reject: "${reason}") even though it passed the isolated oracle — the ` +
    `classic single-snapshot OVER-FIT. Live serving calls this op every step with CHANGING contents (routing / ` +
    `token→expert assignment / gather-scatter indices / masks) while REUSING the same buffers (stable data_ptr, ` +
    `new contents); the fixed-tensor unittest hides this. Fix the CORRECTNESS without giving up the speedup: ` +
    `(1) NEVER key a cache on tensor.data_ptr()/id() for any tensor whose CONTENTS vary per call (routing/gather/ ` +
    `scatter/mask/index) — recompute it each call (cheap, and CUDA-graph-capture-safe) or key on a value hash; ` +
    `only cache things that depend purely on SHAPE/DTYPE (e.g. tile/occupancy config). (2) Do not carry per-call ` +
    `state across calls or assume the previous call's indices/mask still apply. (3) If you use atomics/scatter-add ` +
    `into a reused output, zero it each call inside the captured region. VALIDATE your fix adversarially: call the ` +
    `kernel on TWO different routing snapshots back-to-back REUSING the same input buffers, and on repeated calls ` +
    `with fresh contents, asserting bit/tol correctness on EACH — a data_ptr cache or stale reuse must fail this.`;
  for (let cAttempt = 1; cAttempt <= HEAD_CORRECTIVE_MAX; cAttempt++) {
    log(`  ${spec.short_name}: ${fixClass.toUpperCase()} reject (${reason}) — corrective ${cAttempt}/${HEAD_CORRECTIVE_MAX} (fix-and-retry the iso winner; NOT a new head, NOT charged to head_budget).`);
    // TIER 1: lightweight surgical patch (one agent, minutes).
    let fix = await trySurgicalFix(spec, reason, fixClass, cAttempt);
    const viaSurgical = !!(fix && fix.final_patch && (fix.final_geomean || 0) > 1.0);
    // TIER 2: the heavyweight kernel_workflow re-author, only if the surgical patch failed.
    if (!viaSurgical) {
      try {
        fix = await fastBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
          kernel_path: spec.kernel_eval_dir || spec.task_dir, workflow_dir: KERNEL_WF_DIR,
          mode: 'optimize', target_language: spec.language || 'triton',
          op_spec: { op_kind: spec.op_kind, shapes: spec.shapes || {}, dtype: spec.dtype || 'bf16', regime: spec.regime || '', cuda_graph_safe: true, ...(spec.workload_path ? { workload_path: spec.workload_path } : {}) },
          perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
          use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
          // Keep the isolated win: KB history is readable but never auto-applied here.
          ...KB_ARGS, warm_start: WARM_START_OFF ? KB_ARGS.warm_start : 'reference',
          budget: KERNEL_BUDGET, gpu_ids: spec.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
          task: `CORRECTIVE FIX — do NOT re-discover the algorithm; KEEP the ${(spec.isolated || 0).toFixed(2)}x isolated win. ` +
            `This kernel PASSED the isolated oracle and ENGAGED on all live workers but was REJECTED at the e2e serving gate ` +
            `for: "${reason}". ` + (fixClass === 'correctness' ? CORRECTNESS_FIX_TASK : INTEGRATION_FIX_TASK) +
            ` Emit a fixed final_patch. ` + GRAPH_REQ + (TASK || ''),
          apply_to_original: 'false',
        }, `${spec.short_name}:corrective`);
      } catch (e) { fix = { authored: false, validation_status: 'error', reason: String(e) }; }
    }
    if (!fix || fix.authored === false || !(fix.final_geomean > 1.0) || !fix.final_patch) {
      log(`  ${spec.short_name}: corrective produced no usable kernel (${fix ? fix.reason || fix.validation_status : 'null'}).`);
      return { banked: false };
    }
    log(`  ${spec.short_name}: fix via ${viaSurgical ? 'SURGICAL patch (light)' : 'kernel_workflow re-author (heavy)'}; iso geomean ${(fix.final_geomean || 0).toFixed(2)}. Re-gating e2e.`);
    // Re-gate the fixed kernel in the caller's KERNEL_RESULT shape, swapping in the corrected patch,
    // eval_dir and speedup. Both patch fields get the new patch; a stale one would re-apply the bug.
    const base = spec.base_inputs || {};
    const fixInputs = { ...base,
      KERNEL_RESULT: { ...(base.KERNEL_RESULT || {}),
        code_patch: fix.final_patch, final_patch: fix.final_patch,
        authored_kernel_eval_dir: fix.eval_dir || spec.kernel_eval_dir || (base.KERNEL_RESULT && base.KERNEL_RESULT.authored_kernel_eval_dir) || '',
        verified_isolated_speedup: fix.final_geomean,
        pct_gpu_time: spec.pct_gpu_time || (base.KERNEL_RESULT && base.KERNEL_RESULT.pct_gpu_time) || 0,
        corrective_fix_of: reason } };
    if (spec.cur) {
      fixInputs.CURRENT_OVERLAY = spec.cur.overlay; fixInputs.CURRENT_FLAGS = spec.cur.flags;
      fixInputs.CURRENT_ENV = spec.cur.env; fixInputs.CURRENT_THROUGHPUT = spec.cur.tput;
    }
    const integ2 = await runIntegrateBothLegs(
      'Apply the CORRECTIVELY-FIXED kernel winner; gate on e2e throughput.', fixInputs,
      `integrate ${spec.short_name} corrective`, spec.phase_name || 'HeadKernel');
    const ab2 = !!(integ2 && integ2.gate !== 'incomplete' && integ2.ab_complete !== false);
    const pctForGuard = spec.pct_gpu_time || (spec.base_inputs && spec.base_inputs.KERNEL_RESULT && spec.base_inputs.KERNEL_RESULT.pct_gpu_time) || 0;
    if (ab2 && integAccepted(integ2, pctForGuard, fix.final_geomean) && integ2.e2e_throughput_tok_s > curTput) {
      // Carry the corrected patch out to bankAccepted: the call site's KERNEL_RESULT still holds the
      // rejected one.
      return { banked: true, integ: integ2, isolated: fix.final_geomean, patch: fix.final_patch,
        kernel_eval_dir: fix.eval_dir || spec.kernel_eval_dir || '' };
    }
    reason = gateRejectReason(integ2, pctForGuard, fix.final_geomean) || reason;
    // Structured stop, not a re-classification of the prose reason: once parity no longer fails, a
    // throughput or review issue cannot justify another correctness rewrite.
    if (ab2 && integ2.output_parity !== 'fail') {
      log(`  ${spec.short_name}: corrective has no bankable e2e win (${reason}) — stopping kernel rewrites.`);
      break;
    }
    log(`  ${spec.short_name}: corrective still rejected (${reason}).`);
    fixClass = rejectClass(reason);
    if (fixClass === '') break;   // new failure not auto-correctable -> stop retrying
    // Progressive: the NEXT attempt builds on this attempt's (partially) fixed kernel, not the original.
    spec.kernel_eval_dir = fix.eval_dir || spec.kernel_eval_dir;
  }
  return { banked: false };
}

// --- FAST-MODE wall-clock control (no-op unless FAST_MODE) ----
// Date.now() is unavailable here (it would break resume), so budgets use setTimeout: a one-shot flag
// stops the head loop STARTING new ops, and fastBoundedWorkflow() caps each nested kernel workflow.
let FAST_DEADLINE_HIT = false;
if (FAST_MODE && typeof setTimeout === 'function' && FAST_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    FAST_DEADLINE_HIT = true;
    log(`[fast-mode] head-dispatch deadline (${Math.round(FAST_HEAD_DEADLINE_MS / 60000)}min) reached — no NEW head ops will start; finishing the in-flight head then proceeding to Finalize/Validate within the ${Math.round(FAST_BUDGET_MS / 60000)}min budget.`);
  }, FAST_HEAD_DEADLINE_MS);
}
// A nested kernel workflow raced against the tightest of the fast-mode cap, B4's remaining
// implementation cap (ABL_NESTED_CAP_MS, set by the head loop; null = none) and the trace-replay
// reserve bound. On expiry it resolves null, which callers treat as "no kernel".
let ABL_NESTED_CAP_MS = null;
async function fastBoundedWorkflow(ref, wfArgs, label) {
  if (laneWindowShort(`nested kernel workflow ${label || ''}`)) return null;
  wfArgs = { ...wfArgs, ...(await laneDeadlineArgs(label, 'HeadKernel')) };
  // noteKernelKB returns `r` unchanged, so the caller's null-guards are untouched.
  const p = workflow(ref, laneArgs(wfArgs)).then((r) => noteKernelKB(r, label));
  const ablCap = ABL('B4') && ABL_NESTED_CAP_MS != null ? ABL_NESTED_CAP_MS : null;
  const fastCap = FAST_MODE && FAST_HEAD_WF_MS > 0 ? FAST_HEAD_WF_MS : null;
  const resCap = laneReserveCapMs();
  const capMs = [ablCap, fastCap, resCap].reduce((m, c) => (c != null && (m == null || c < m) ? c : m), null);
  if (typeof setTimeout !== 'function' || !(capMs > 0)) return p;
  return raceLaneCap(p, capMs, () => {
    if (capMs === resCap) return logLaneAtReserve(label);
    log(`  [${ablCap != null && capMs === ablCap ? ABLATION_ARM : 'fast-mode'}] nested kernel workflow ${label || ''} exceeded ${Math.round(capMs / 60000)}min — abandoning (null) to stay on budget.`);
    if (ablCap != null && capMs === ablCap) {
      ablEvent({ event: 'nested_cap', label: label || '', cap_ms: capMs, action: 'abandon' });
    }
  });
}
// Milestone lanes: the trace-replay reserve bound alone.
async function reserveBoundedWorkflow(ref, wfArgs, label) {
  if (laneWindowShort(`nested kernel workflow ${label || ''}`)) return null;
  wfArgs = { ...wfArgs, ...(await laneDeadlineArgs(label, 'Milestone')) };
  const p = workflow(ref, laneArgs(wfArgs)).then((r) => noteKernelKB(r, label));
  const capMs = laneReserveCapMs();
  if (typeof setTimeout !== 'function' || !(capMs > 0)) return p;
  return raceLaneCap(p, capMs, () => logLaneAtReserve(label));
}
function raceLaneCap(p, capMs, onCap) {
  let to;
  const guard = new Promise((resolve) => {
    to = setTimeout(() => { onCap(); resolve(null); }, capMs);
  });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}
function logLaneAtReserve(label) {
  log(`  [time-budget] nested kernel workflow ${label || ''} still running at the final reserve — abandoning (null).`);
}

// --- DEEP-MODE wall-clock control (no-op unless DEEP_MODE) ----
// Same mechanism: a one-shot flag stops NEW co-opt waves once the HeadKernel budget is spent, and
// deepBoundedWorkflow() caps each nested kernel_workflow burst so one backend can't stall a wave.
let DEEP_DEADLINE_HIT = false;
if (DEEP_MODE && typeof setTimeout === 'function' && DEEP_HEAD_BUDGET_MS > 0) {
  setTimeout(() => {
    DEEP_DEADLINE_HIT = true;
    log(`[deep-mode] HeadKernel budget (${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h) reached — no NEW co-opt waves will start; finishing the in-flight wave then proceeding to Finalize/Validate.`);
  }, DEEP_HEAD_BUDGET_MS);
}
// --- DEFAULT-MODE wall-clock control (no-op unless time_budget_s passed) ----
// One flag for every mode: stop STARTING new head/milestone work at TIME_HEAD_DEADLINE_MS, leaving
// the tail for in-flight work and Finalize/Report/Validate.
let TIME_DEADLINE_HIT = false;
if (TIME_HEAD_DEADLINE_MS != null && typeof setTimeout === 'function' && TIME_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    TIME_DEADLINE_HIT = true;
    log(`[time-budget] dispatch deadline (${Math.round(TIME_HEAD_DEADLINE_MS / 60000)}min of ${Math.round(TIME_BUDGET_MS / 60000)}min budget) reached — no NEW head/milestone work will start; finishing in-flight then proceeding to Finalize/Report/Validate before the hard kill.`);
  }, TIME_HEAD_DEADLINE_MS);
}
// --- RESERVE BOUNDARY ----
// The one flag saying "the reserve has begun" (it also bounds the Finalize-gate drain): a single
// absolute setTimeout, so unlike remainingMs() it cannot drift. Inert without time_budget_s.
let TIME_FINAL_DEADLINE_HIT = false;
if (TIME_BUDGET_EFFECTIVE_MS != null && typeof setTimeout === 'function' && TIME_BUDGET_EFFECTIVE_MS > 0) {
  setTimeout(() => {
    TIME_FINAL_DEADLINE_HIT = true;
    log(`[time-budget] RESERVE BOUNDARY reached (${Math.round(TIME_BUDGET_EFFECTIVE_MS / 60000)}min of the ${Math.round(TIME_BUDGET_MS / 60000)}min budget spent) — the remaining ~${Math.round(FINAL_RESERVE_MS / 60000)}min belongs to Finalize/Report/Validate. No NEW A/B or measured work starts from here.`);
  }, TIME_BUDGET_EFFECTIVE_MS);
}
// The same rule for everything ahead of the Finalize-gate: the search phases, their re-profiles, the
// KB replays and FINISH retries. Each checks before it dispatches; see atReserveFloor() for why.
function searchClosed(what) {
  if (FINAL_PHASE_STARTED || (!TIME_FINAL_DEADLINE_HIT && !atReserveFloor())) return false;
  log(`[time-budget] ${what} SKIPPED — the final reserve has begun.`);
  return true;
}
// The tuning loop is hours of GPU work under ONE agent, which the reserve can abandon but not stop,
// so it starts only with this much left before the reserve. tuning_min_window_s=0 disables.
const _tuningMinWindowS = parseInt(A.tuning_min_window_s, 10);
const TUNING_MIN_WINDOW_MS = (Number.isFinite(_tuningMinWindowS) && _tuningMinWindowS >= 0
  ? _tuningMinWindowS : 21600) * 1000;   // 6h
function tuningWindowShort() {
  if (TIME_BUDGET_MS == null || !(TUNING_MIN_WINDOW_MS > 0)) return false;
  const windowMs = remainingMs() - FINAL_RESERVE_MS;
  if (windowMs >= TUNING_MIN_WINDOW_MS) return false;
  log(`[time-budget] TuningSkillset SKIPPED — ~${Math.max(0, Math.round(windowMs / 60000))}min left before ` +
    `the final reserve, under the ${Math.round(TUNING_MIN_WINDOW_MS / 60000)}min the tuning loop needs to ` +
    'finish its A/B before Finalize and Validate need the GPUs.');
  return true;
}
function deepBoundedWorkflow(ref, wfArgs, label) {
  const p = workflow(ref, laneArgs(wfArgs)).then((r) => noteKernelKB(r, label));
  if (!DEEP_MODE || typeof setTimeout !== 'function' || !(DEEP_HEAD_WF_MS > 0)) return p;
  let to;
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [deep-mode] nested kernel_workflow burst ${label || ''} exceeded ${Math.round(DEEP_HEAD_WF_MS / 60000)}min — abandoning (null) so the wave barrier proceeds; STATE_DIR keeps its progress for the next wave.`);
      resolve(null);
    }, DEEP_HEAD_WF_MS);
  });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}
// --- TRACE-REPLAY SEARCH vs THE RESERVE ----
// Trace-replay lanes and A/Bs start only if they fit before the boundary (inert off AgentX/time_budget_s).
const AGENTX_LANE_ROUND_S = 3600;    // a lane round in flight at its stop, plus the lane's report
const AGENTX_EXTRACT_S = 4200;       // kernel extraction
const agentxGuardsOn = () => IS_AGENTX && TIME_BUDGET_MS != null && !FINAL_PHASE_STARTED;
const searchWindowMs = () => remainingMs() - FINAL_RESERVE_MS;
// A reused or resumed reference leg is not run again.
const abLegs = (inputs) => (inputs && (inputs.REUSE_REF || inputs.RESUME_AB) ? 1 : 2);
// A leg timed over several passes costs that many legs (an upper bound under warm_server).
const agentxAbMs = (legs, passes = 1) =>
  1000 * (legs * passes * agentxLegS(_declaredWindowS(AGENTX.geak_loop_duration_s, 900)) + AGENTX_AB_OVERHEAD_S);
const agentxLaneTailMs = () => agentxAbMs(2) + AGENTX_LANE_ROUND_S * 1000;
const agentxLaneStartMs = () => agentxLaneTailMs() + AGENTX_LANE_ROUND_S * 1000;
// need() reads AGENTX, so it runs only once the guard holds.
function windowShort(need, what, verdict, cost) {
  if (!agentxGuardsOn()) return false;
  const windowMs = searchWindowMs(), needMs = need();
  if (windowMs >= needMs) return false;
  log(`[time-budget] ${what} ${verdict} — ~${Math.max(0, Math.round(windowMs / 60000))}min left before the ` +
    `final reserve, under the ~${Math.round(needMs / 60000)}min ${cost}.`);
  return true;
}
// An A/B that cannot finish is deferred (incomplete, pending), not abandoned mid-leg at the boundary.
function abWindowShort(legs, what, passes = 1) {
  return windowShort(() => agentxAbMs(legs, passes), what, 'DEFERRED',
    `a ${legs}-leg trace-replay A/B${passes > 1 ? ` with ${passes} timed passes a leg` : ''} takes`);
}
// A lane needs one round before its stop plus the A/B for what it found.
function laneWindowShort(what) {
  return windowShort(agentxLaneStartMs, what, 'NOT STARTED', 'a kernel lane and its A/B need');
}
// Every search candidate needs kernel extraction and a lane; without both the run ends early.
function searchWindowShort(what) {
  return windowShort(() => AGENTX_EXTRACT_S * 1000 + agentxLaneStartMs(), what, 'SKIPPED',
    'kernel extraction plus one kernel lane and its A/B need. Going straight to Finalize and Validate');
}
// kernel_lane.js starts no round after deadline_epoch, so the stop is a lane tail ahead of the boundary;
// no_stop_s past the window leaves the TechLead free to end it early. No clock (Date.now() breaks
// resume): ONE `date +%s` read, anchored to ELAPSED_MS; unreadable, only the backstop bounds the lane.
const CLOCK_SCHEMA = obj({ epoch: { type: 'number' } }, ['epoch']);
let EPOCH_AT_T0 = null;
async function epochNow(phaseName) {
  if (EPOCH_AT_T0 == null) {
    EPOCH_AT_T0 = safeAgent('Run EXACTLY this command and nothing else:\n```bash\ndate +%s\n```\n' +
      'Return {"epoch": <the integer it printed>}. Do NOT modify any file and do NOT run anything else.',
      { phase: phaseName, label: 'clock', effort: 'low', schema: CLOCK_SCHEMA }, 1, () => 5 * 60000)
      .then((r) => (r && Number.isFinite(r.epoch) ? Math.floor(r.epoch) - Math.floor(ELAPSED_MS / 1000) : null));
  }
  const t0 = await EPOCH_AT_T0;
  if (t0 == null) EPOCH_AT_T0 = null;
  return t0 == null ? null : t0 + Math.floor(ELAPSED_MS / 1000);
}
async function laneDeadlineArgs(label, phaseName) {
  if (!agentxGuardsOn()) return {};
  const stopInS = Math.floor((searchWindowMs() - agentxLaneTailMs()) / 1000);
  if (!(stopInS > 0)) return {};
  const now = await epochNow(phaseName);
  if (now == null) {
    log(`  [time-budget] ${label}: wall clock unreadable, no round deadline.`);
    return {};
  }
  log(`  [time-budget] ${label}: no new round after ~${Math.round(stopInS / 60)}min ` +
    `(${Math.round(agentxLaneTailMs() / 60000)}min before the final reserve).`);
  return { deadline_epoch: now + stopInS, no_stop_s: stopInS + 86400 };
}
// Backstop: never wait for a lane past the boundary.
const laneReserveCapMs = () => (agentxGuardsOn() ? Math.max(AGENT_TIMEOUT_FLOOR_MS, searchWindowMs()) : null);

// GPU semaphore (FAST MODE only for head jobs; deep mode's co-opt pools also use one): exclusive
// leases of physical card ids, so concurrent isolated measurements never share a GPU. Deadlock-free:
// a waiter holds no cards and acquires its full count atomically.
function makeSem(ids) {
  const free = ids.slice(); const waiters = [];
  const pump = () => { while (waiters.length && waiters[0].n <= free.length) {
    const w = waiters.shift(); w.resolve(free.splice(0, w.n)); } };
  return {
    size: ids.length,
    acquire(n = 1) { if (n <= free.length) return Promise.resolve(free.splice(0, n));
      return new Promise((resolve) => { waiters.push({ n, resolve }); }); },
    release(got) { free.push(...got); pump(); },
    async with(n, fn) { const g = await this.acquire(n); try { return await fn(g); } finally { this.release(g); } },
  };
}

// ---- SINGLE-KERNEL PASS-THROUGH: delegate straight to the kernel layer ----
if (!MODEL_PATH && KERNEL_PATH) {
  phase('Setup');
  log(`Single-kernel pass-through -> ${KERNEL_WF_SCRIPT} on ${KERNEL_PATH}`);
  // Recurse into the kernel layer via workflow() (one allowed level of nesting); it returns
  // {eval_dir, final_geomean, final_patch, validation_status, ...}.
  let passthru;
  try {
    const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, laneArgs({
      kernel_path: KERNEL_PATH, workflow_dir: KERNEL_WF_DIR,
      // The lane defaults to triton; a pass-through caller naming another backend must reach that
      // backend's own warm-start page (the slug is per-language), not triton's.
      ...(A.target_language ? { target_language: String(A.target_language) } : {}),
      use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
      ...KB_ARGS,
      budget: KERNEL_BUDGET, gpu_ids: GPU_IDS, task: TASK, exp_root: EXP_ROOT,
      apply_to_original: APPLY_TO_ORIGINAL,
    }));
    noteKernelKB(r, 'single-kernel pass-through');
    passthru = { ran: true, kernel_eval_dir: r.eval_dir, final_patch: r.final_patch,
      final_geomean: r.final_geomean, validation_status: r.validation_status,
      note: (r.winner && r.winner.source) || '' };
  } catch (e) {
    passthru = { ran: false, kernel_eval_dir: '', final_patch: '', final_geomean: 0,
      validation_status: 'error', note: String(e) };
  }
  log(`Single-kernel done. geomean=${passthru ? passthru.final_geomean : '?'}x`);
  return { mode: 'single_kernel', kernel_path: KERNEL_PATH, ...(passthru || {}) };
}

// OP-IDENTITY GUARD: a fused-MoE / grouped-expert GEMM is optimized AS the fused op at its live
// dispatcher seam (a dense GEMM candidate has no live call site), so tag it op_kind='moe' (keeps
// gemmSynthFor's dense synth off); it is never skipped. Detected from is_fused_kernel or the
// profile class/name. Shared by every strategize call site, and it also admits the queue.
const _isFusedOp = (c) => (c && c.is_fused_kernel === true) ||
  /(?:^|[^a-z])moe(?:[^a-z]|$)|group(?:ed)?[_ ]?gemm|ck_moe|expert|fused[_ ]?moe|fmoe|asm_moe|fused_custom/i
    .test(`${(c && c.op_kind) || ''} ${(c && c.short_name) || ''} ${(c && c.name) || ''} ${(c && c.classification) || ''} ${(c && c.class) || ''} ${(c && c.backend) || ''}`);
function applyOpIdentityGuard(queue, stage) {
  let tagged = 0;
  for (const c of (queue || [])) {
    if (!_isFusedOp(c)) continue;
    c.op_kind = 'moe';
    tagged++;
  }
  const admitted = admitHeads(queue, stage);
  if (tagged) log(`[op-identity] ${tagged} fused/grouped head(s): op_kind=moe (never dense-GEMM), bound at live seam — optimized as the fused op, never skipped.`);
  return admitted;
}

// ---- MERGING A RECOVERED CONFIGURATION INTO THE ONE THIS RUN WAS HANDED ----
// A stored e2e record holds a WHOLE launch configuration. The launcher's flag/env expansion is
// last-wins, so a key both sides pin would apply in join order; merge here instead, naming every key
// once and reporting what changed.

// Flags that may legitimately repeat (deduping `--lora-path` would drop adapters); every other flag
// here is a scalar. Grow it only with evidence.
const REPEATABLE_FLAGS = new Set(['--lora-path', '--lora-paths']);

const _looksLikeFlag = (t) => /^--?[A-Za-z]/.test(t);

/** `--a 1 --b=2 --c` -> [{name,value,eq}]. Unrecognized tokens ride along under name=null. */
function parseFlagString(s) {
  const toks = String(s || '').trim().split(/\s+/).filter(Boolean);
  const items = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    if (!_looksLikeFlag(t)) { items.push({ name: null, value: t, eq: false }); continue; }
    const eq = t.indexOf('=');
    if (eq > 0) { items.push({ name: t.slice(0, eq), value: t.slice(eq + 1), eq: true }); continue; }
    // Everything up to the next flag is this flag's value; `--foo a b` stays one item so a
    // multi-valued flag is overridden as a unit rather than half-overridden.
    const vals = [];
    while (i + 1 < toks.length && !_looksLikeFlag(toks[i + 1])) vals.push(toks[++i]);
    items.push({ name: t, value: vals.length ? vals.join(' ') : null, eq: false });
  }
  return items;
}

const renderFlagItems = (items) => items.map(
  (it) => (it.name == null ? it.value
    : it.value == null ? it.name
      : it.eq ? `${it.name}=${it.value}` : `${it.name} ${it.value}`)).join(' ').trim();

/** `K=V K2=V2` -> [{name,value}]. A token that is not an assignment rides along under name=null. */
function parseEnvString(s) {
  return String(s || '').trim().split(/\s+/).filter(Boolean).map((t) => {
    const m = /^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/.exec(t);
    return m ? { name: m[1], value: m[2] } : { name: null, value: t };
  });
}

const renderEnvItems = (items) => items.map(
  (it) => (it.name == null ? it.value : `${it.name}=${it.value}`)).join(' ').trim();

/**
 * Key-by-key merge of a recovered configuration onto this run's baseline. The recovered value wins
 * a collision IN PLACE (it is what is being tested), so baseline order survives and each key appears
 * once. Returns `{ merged, overrides, added, unchanged }`; `overrides` lists the knobs where this box
 * disagreed with the record.
 */
function mergeConfig(baseline, recovered, { parse, render, repeatable }) {
  const out = parse(baseline).map((it) => ({ ...it }));
  const at = new Map();
  out.forEach((it, i) => { if (it.name && !repeatable.has(it.name)) at.set(it.name, i); });
  const overrides = [], added = [], unchanged = [];
  for (const it of parse(recovered)) {
    if (it.name == null || repeatable.has(it.name) || !at.has(it.name)) {
      out.push({ ...it });
      if (it.name != null) {
        added.push(it.name);
        if (!repeatable.has(it.name)) at.set(it.name, out.length - 1);
      }
      continue;
    }
    const i = at.get(it.name), prev = out[i];
    if (String(prev.value == null ? '' : prev.value) === String(it.value == null ? '' : it.value)) {
      unchanged.push(it.name);
      continue;
    }
    overrides.push({ flag: it.name, baseline_value: prev.value, recalled_value: it.value });
    out[i] = { ...it };
  }
  return { merged: render(out), overrides, added, unchanged };
}

const mergeFlags = (baseFlags, storedFlags) => mergeConfig(baseFlags, storedFlags,
  { parse: parseFlagString, render: renderFlagItems, repeatable: REPEATABLE_FLAGS });
const mergeEnv = (baseEnv, storedEnv) => mergeConfig(baseEnv, storedEnv,
  { parse: parseEnvString, render: renderEnvItems, repeatable: new Set() });

/**
 * Which trial actually ran, and what it measured: the LAST trial with a positive throughput (after a
 * repair, trials[0] is the failed verbatim replay), and `best_throughput_tok_s` only when it differs
 * from the baseline (the role returns the baseline there when nothing was kept).
 */
function measuredTrialOf(sweep, baselineTput) {
  const trials = (sweep && Array.isArray(sweep.trials)) ? sweep.trials : [];
  const ran = trials.filter((t) => t && Number(t.throughput_tok_s) > 0);
  const trial = ran.length ? ran[ran.length - 1] : (trials[0] || {});
  const best = Number(sweep && sweep.best_throughput_tok_s) || 0;
  const measured = ran.length ? Number(trial.throughput_tok_s)
    : (best && best !== baselineTput ? best : 0);
  return { trial, measured };
}

/**
 * Knobs the merged configuration asked for that the trial that ran does not carry, recovered by
 * subtraction (`repair.dropped_flags` is optional and often empty), so `applied_partial` is right.
 */
function droppedByDiff(mergedFlags, mergedEnv, applied) {
  const app = String(applied || '').trim();
  if (!app) return [];
  const keyOf = (t) => {
    const f = /^(--?[A-Za-z][^=\s]*)/.exec(t);
    if (f) return f[1];
    const e = /^([A-Za-z_][A-Za-z0-9_]*)=/.exec(t);
    return e ? e[1] : null;
  };
  const have = new Set();
  app.split(/\s+/).forEach((t) => { const k = keyOf(t); if (k) have.add(k); });
  const want = [];
  parseFlagString(mergedFlags).forEach((it) => { if (it.name) want.push(it.name); });
  parseEnvString(mergedEnv).forEach((it) => { if (it.name) want.push(it.name); });
  return want.filter((k) => !have.has(k));
}

/** `--context-length 11264 -> 13312; MAX_MODEL_LEN 13312 -> 16384`, or "". For prompts and logs. */
const describeOverrides = (overrides) => (overrides || []).map(
  (o) => `${o.flag} ${o.baseline_value == null ? '(unset)' : o.baseline_value} -> ` +
    `${o.recalled_value == null ? '(set)' : o.recalled_value}`).join('; ');

// PHASE: Setup + Baseline profile + Strategize  (gated; else load carried state)
// Module A's outputs, folded into the ordinary state variables at their declarations below (Module A
// runs BEFORE Profile so the Top-N is taken on the adopted config). Off-values make each fold a no-op.
let kbSeedTput = 0;
let kbSeedOverlay = '';
const kbSeedKernels = [];
// Spread into the Strategize and ConfigSweep inputs; stays {} unless Module A measured something.
let KB_REF_INPUTS = {};

let EVAL_DIR, MODEL_NAME, BASELINE_TPUT, NOISE_BAND, curFlags, curEnv, curOverlay, profile, strategy, kernelQueue, headQueue;
let SETUP_CONFIG = null;   // what BASELINE_TPUT was measured on, if Setup ran here

// The envelope arrives in the reply: this script cannot read EVAL_DIR (RUNTIME SANDBOX CONTRACT).
// Every malformed reply THROWS, which arms the caller's repair_delivery retry; an empty queue would
// read as "nothing to optimize".
function readAgentXPreparation(reply) {
  const prepared = reply && reply.prepared;
  if (!prepared || typeof prepared !== 'object')
    throw new Error('reply carries no prepared envelope; run agentx_kernel_preparation.mjs with ' +
      '--emit-prepared and return its stdout verbatim');
  for (const key of ['head_candidates', 'kernel_candidates'])
    if (!Array.isArray(prepared[key])) throw new Error(`prepared.${key} must be an array`);
  // Re-assert each candidate's invariants (preparedUtDefect); a bad one THROWS rather than being
  // silently dropped.
  const defects = [];
  for (const c of [...prepared.head_candidates, ...prepared.kernel_candidates]) {
    const why = !c || !c.prepared_ut ? 'carries no prepared_ut'
      : preparedUtDefect(c, c.prepared_ut.result, c.prepared_ut.phase);
    if (why) defects.push(`${(c && c.short_name) || '(unnamed)'}: ${why}`);
  }
  if (defects.length)
    throw new Error(`${defects.length} prepared candidate(s) failed re-validation — ${defects.join('; ')}`);
  if (!prepared.kernel_extraction_delivery_json && reply.delivery_path)
    prepared.kernel_extraction_delivery_json = reply.delivery_path;
  for (const key of ['failed_tasks', 'deferred_tasks'])
    if (!Array.isArray(prepared[key])) prepared[key] = [];
  prepared.config_directions = Array.isArray(prepared.config_directions) ? prepared.config_directions : [];
  return prepared;
}

async function prepareAgentXKernels(round, throughput) {
  phase('KernelExtract');
  const inputs = {
    EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, TASK, ROUND: round,
    BASELINE_LOG: `${EVAL_DIR}/baseline/server.log`, BASELINE_BENCH_DIR: `${EVAL_DIR}/baseline`,
    BASELINE_SUMMARY: `${EVAL_DIR}/baseline_summary.md`,
    EVIDENCE_DIR: `${EVAL_DIR}/kernel_extraction/round_${round}`,
    BASELINE_THROUGHPUT: BASELINE_TPUT, CURRENT_THROUGHPUT: throughput,
    CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
    PREVIOUS_WORKLOAD_JSON: profile ? profile.profile_workload_json || '' : '',
    PREVIOUS_RANKING_JSON: profile ? profile.kernel_ranking_json || '' : '',
    PREVIOUS_UT_DELIVERY: profile ? profile.kernel_extraction_delivery_json || '' : '',
    EXTRACTION_SELECTION_POLICY: { phases: ['PREFILL', 'DECODE'], top_k_per_phase: 5,
      overall_pct_gt: 3, combination: 'phase_top_k_and_overall_threshold', backfill: false },
    SOURCE_DELIVERY_REQUIRED: true,
    BUDGET, HEAD_BUDGET, REMAINING_MIN: remainingMin(),
    SKILL_DIR: WORKFLOW_DIR, ...TRACELENS_INPUTS, ...ANALYSIS_SKILL_INPUTS, ...KB_REF_INPUTS,
  };
  let reply = await safeAgent(roleAgent('kernel_extractor', 'prepare',
    'Complete the kernel-extract skill from the saved baseline through executable, GPU-validated UTs. ' +
    'Selection, task grouping and bootstrap replay are INTERNAL steps: save their artifacts and continue ' +
    'to source extraction, UT authoring and validation in this invocation. Prefer fixed-seed input factories ' +
    'from saved shape/layout metadata; UT execution must not start serving or capture shapes. ' +
    'Capture only missing structural state needed for faithful cases. Do not return a candidate list ' +
    'before UTs have been attempted. Reuse compatible evidence and passing UTs. Never rerun the full ' +
    'AgentX baseline for discovery. Follow TASK scope and the default per-phase selection rule. ' +
    'Prove that edits to each opt GPU leaf affect candidate correctness while the frozen baseline passes; ' +
    'restore the clean source and rerun the UT. Keep failures and budget-deferred tasks in the delivery. ' +
    'Read references/workflow-handoff.md in the skill. Pack with ' +
    '`agentx_kernel_preparation.mjs --emit-prepared` and return ONLY {delivery_path, prepared}: ' +
    'delivery_path names that file under EVAL_DIR, and prepared is that command\'s stdout envelope ' +
    'VERBATIM — the workflow cannot open the file, so an envelope you summarise or rebuild by hand ' +
    'is a corrupted handoff. Return it after all scheduled UT work is finished. Do not optimize or ' +
    'integrate.', inputs),
    { phase: 'KernelExtract', label: `kernel_extractor:prepare ${round}`, schema: AGENTX_PREPARATION_SCHEMA });
  for (let attempt = 0; ; attempt++) {
    try {
      if (!reply?.delivery_path) throw new Error('completed skill did not return delivery_path');
      const prepared = await readAgentXPreparation(reply);
      log(`Kernel extraction ready: ${prepared.head_candidates.length + prepared.kernel_candidates.length} validated UTs, ` +
        `${prepared.failed_tasks.length} failed, ${prepared.deferred_tasks.length} deferred.`);
      return prepared;
    } catch (error) {
      if (attempt >= 1) throw new Error(`AgentX UT delivery unreadable: ${error.message}`);
      reply = await safeAgent(roleAgent('kernel_extractor', 'repair_delivery',
        'Repair only the completed-stage handoff using saved files. Do not repeat selection, captures or ' +
        'passing GPU tests. Preserve every real failure and all task results. The reader failed with: ' +
        error.message + '. Re-pack with `agentx_kernel_preparation.mjs --emit-prepared` and return ' +
        '{delivery_path, prepared} for the repaired file, prepared being that stdout verbatim.',
        { ...inputs, PREVIOUS_DELIVERY_PATH: reply?.delivery_path || '' }),
        { phase: 'KernelExtract', label: `kernel_extractor:repair delivery ${round}`, schema: AGENTX_PREPARATION_SCHEMA });
    }
  }
}

if (want('setup')) {
  phase('Setup');
  const setup = await safeAgent(
    roleAgent('director', 'setup', 'Build the isolated e2e eval dir and record the baseline throughput.', {
      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT, EVAL_DIR_OVERRIDE, MODEL_NAME_HINT, TASK,
      GPU_IDS, WORKLOAD, INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY,
      MEASUREMENT_PURPOSE: 'parity', REPLICAS: PARITY_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR, EXPECTED_GFX, EXPECTED_TARGET,
      EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
      // Only on a declared trace replay: the Director writes it to $EVAL_DIR/bench_env.sh before
      // the baseline bench, so the baseline uses the real client and metric axis.
      ...(AGENTX_ENV ? { BENCH_ENV_CONTENT: AGENTX_ENV } : {}),
    }),
    { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
  if (!setup || !setup.eval_dir) throw new Error('Setup failed: no eval_dir');
  const detectedGfx = String(setup.gfx || '').trim().toLowerCase();
  const detectedTarget = String(setup.device_target || '').trim().toLowerCase();
  const detectedPhysicalCuCount = Number(setup.physical_cu_count);
  if (!detectedGfx || !detectedTarget) {
    throw new Error('Setup failed: Director did not return structured gfx and device_target identity');
  }
  if (!Number.isFinite(detectedPhysicalCuCount) || detectedPhysicalCuCount <= 0) {
    throw new Error('Setup failed: Director did not return a positive physical_cu_count');
  }
  if (EXPECTED_GFX && detectedGfx !== EXPECTED_GFX) {
    throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${detectedGfx}`);
  }
  if (EXPECTED_TARGET && detectedTarget !== EXPECTED_TARGET) {
    throw new Error(`GPU product mismatch: expected ${EXPECTED_TARGET}, detected ${detectedTarget}`);
  }
  const normalizeDeviceName = (name) =>
    String(name || '').trim().replace(/\s+/g, ' ').toLowerCase();
  if (EXPECTED_DEVICE_NAME &&
      normalizeDeviceName(setup.device_name) !== normalizeDeviceName(EXPECTED_DEVICE_NAME)) {
    log(`Setup identity telemetry differs from the deterministic probe: device_name expected ` +
        `${EXPECTED_DEVICE_NAME}, Director returned ` +
        `${String(setup.device_name || '').trim() || 'unknown'}; using expected value.`);
  }
  if (EXPECTED_PHYSICAL_CU_COUNT > 0 &&
      detectedPhysicalCuCount !== EXPECTED_PHYSICAL_CU_COUNT) {
    log(`Setup identity telemetry differs from the deterministic probe: physical_cu_count expected ` +
        `${EXPECTED_PHYSICAL_CU_COUNT}, Director returned ${detectedPhysicalCuCount}; using expected value.`);
  }
  const divergence = evalDirDivergence(EVAL_DIR_OVERRIDE, setup.eval_dir);
  if (divergence) throw new Error(`Setup failed: ${divergence}`);
  EVAL_DIR = setup.eval_dir;
  MODEL_NAME = setup.model_name || MODEL_NAME_HINT;
  BASELINE_TPUT = setup.baseline_throughput_tok_s;
  NOISE_BAND = gateNoiseBand(setup.noise_band_pct);
  // Seed flags/env win when provided (baseline was measured on them); else fall
  // back to whatever the director resolved.
  curFlags = INIT_FLAGS || (setup.server_flags && setup.server_flags.extra) || '';
  curEnv = INIT_ENV || (setup.server_env || '');
  curOverlay = INIT_BASE_OVERLAY;
  SETUP_CONFIG = { flags: curFlags, env: curEnv, overlay: curOverlay };
  log(`Setup done. EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT} tok/s (noise band ${NOISE_BAND}%)`);
  if (IS_AGENTX) {
    log(`Setup: AgentX graded on metric_basis=${AGENTX_METRIC_BASIS} (E2E_METRIC=${AGENTX_E2E_METRIC}, ` +
      `from ${AGENTX_METRIC_SOURCE}), concurrency ${CONC}, noise band ${NOISE_BAND}% ` +
      `(Director reported ${setup.noise_band_pct != null ? setup.noise_band_pct : 'none'}%, ` +
      `workload floor ${NOISE_BAND_DEFAULT}%).`);
  }

  // ── The trace-replay declaration must have TAKEN EFFECT, not merely been sent ──
  // BASELINE_TPUT is the denominator of every reported gain, and the declaration travels through an
  // agent, so confirm bench_env.sh exists and the baseline summary carries the requested metric
  // basis. A mismatch aborts now; an unreadable summary warns and continues.
  if (IS_AGENTX) {
    let verdict = null;
    try {
      const { execFileSync } = require('child_process');
      const out = execFileSync('python3', ['-c', `
import json, os, sys
eval_dir = sys.argv[1]
want = sys.argv[2]
env_file = os.path.join(eval_dir, 'bench_env.sh')
summary = os.path.join(eval_dir, 'baseline', 'bench_summary.json')
res = {'env_file': os.path.isfile(env_file) and os.path.getsize(env_file) > 0}
def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
try:
    with open(summary) as fh:
        _s = json.load(fh) or {}
    res['basis'] = _s.get('metric_basis') or ''
    # The request shape the baseline actually served. Reading it back is what
    # lets kernel work target the served regime without this file, or the
    # workflow, hardcoding a corpus average that would go stale.
    res['observed_isl'] = _num(_s.get('observed_isl'))
    res['observed_osl'] = _num(_s.get('observed_osl'))
except Exception as exc:
    res['basis'] = None
    res['why'] = f'{type(exc).__name__}: {exc}'
res['want'] = want
print(json.dumps(res))
`, EVAL_DIR, AGENTX_METRIC_BASIS], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 60000 });
      verdict = JSON.parse(out);
    } catch (probeErr) {
      log(`Setup: WARNING could not verify the trace-replay declaration ` +
        `(${probeErr && probeErr.message ? probeErr.message : probeErr}). ` +
        `Proceeding, but confirm ${EVAL_DIR}/bench_env.sh and the baseline metric_basis by hand.`);
    }
    if (verdict) {
      if (!verdict.env_file) {
        throw new Error(
          `Setup declared workload_kind=${WORKLOAD_KIND} but ${EVAL_DIR}/bench_env.sh was never ` +
          `written, so the baseline was measured with the synthetic client on the wrong metric axis. ` +
          `Every later delta would be relative to the wrong number. Re-run Setup.`);
      }
      if (verdict.basis === null) {
        log(`Setup: WARNING baseline/bench_summary.json unreadable (${verdict.why || 'unknown'}); ` +
          `cannot confirm metric_basis=${AGENTX_METRIC_BASIS}.`);
      } else if (verdict.basis !== AGENTX_METRIC_BASIS) {
        throw new Error(
          `Setup measured the baseline on metric_basis=${verdict.basis || '<absent>'} but this run ` +
          `declared ${AGENTX_METRIC_BASIS}. The two axes are not comparable, so the baseline is ` +
          `unusable. Check that ${EVAL_DIR}/bench_env.sh sets E2E_METRIC=${AGENTX_E2E_METRIC} and ` +
          `that the bench did not inherit another E2E_METRIC (an exported value outranks the ` +
          `file), then re-run Setup.`);
      } else {
        log(`Setup: trace-replay declaration verified (bench_env.sh present, ` +
          `baseline metric_basis=${verdict.basis}).`);
      }
      // Take the kernel-targeting shape from what the baseline served, unless the caller declared
      // one explicitly.
      if (SHAPE_IS_MEASURED) {
        log(`Setup: keeping the declared kernel-targeting shape isl=${ISL} osl=${OSL} ` +
          `(provenance ${WORKLOAD_SHAPE_PROVENANCE}); the baseline served ` +
          `isl=${verdict.observed_isl || '?'} osl=${verdict.observed_osl || '?'}.`);
      } else if (!adoptMeasuredShape(verdict.observed_isl, verdict.observed_osl,
                                     'the baseline bench')) {
        log(`Setup: the baseline summary carried no served request shape yet, so kernel ` +
          `targeting is still the synthetic ${ISL}/${OSL} placeholder — far below a trace ` +
          `replay's real lengths. This is expected when a relaunched leg has not written its ` +
          `summary yet: every later role prompt retries the read and installs the shape as ` +
          `soon as it appears (watch for "Kernel-targeting shape measured"). If that line ` +
          `never comes, kernel choices are aimed at the wrong regime (the MEASUREMENT is ` +
          `unaffected) — pass workload_spec.observed_isl/observed_osl to set it explicitly.`);
      }
    }
  }

  // =========================================================================
  // MODULE A — read the deployment KB, then VALIDATE what it says on this box.
  // After Setup (needs EVAL_DIR, bench_e2e.sh, BASELINE_TPUT, NOISE_BAND), before Profile (Top-N on
  // the adopted config), inside want('setup') (a phase-partial resume cold-starts). A stored speedup
  // counts only once this run re-measures it through the same gate.
  if (E2E_WARM_START_ON) {
    phase('WarmStart');
    KB_DIMS = {
      model: MODEL_NAME,
      gfx: String(setup.gfx || '').trim(),
      framework_version: String(setup.framework_version || A.kb_framework_version || '').trim(),
      precision: String(setup.precision || '').trim(),
      rocm_version: String(setup.rocm_version || '').trim(),
    };
    if (!KB_DIMS.gfx) {
      // Refuse to read rather than read the wrong page: an arch-less identity resolves to the
      // `gfx=unknown` rung, measured on unknown hardware.
      log('[kb] warm start skipped: read_reason=missing_arch (the Director reported no gfx; a ' +
        'cross-arch config is not a candidate, it is a guess).');
      // Still reported: "did not look, and why" differs from "looked and found nothing".
      KB_RECALL.e2e = { read_reason: 'missing_arch', tried: [], answered: '', match_tier: '',
        plane: E2E_KB_PLANE, mode: E2E_WARM_START, candidates: 0, configs: [], kernels: [] };
    } else {
      const refsDir = `${EVAL_DIR}/kb_references`;
      const cacheDir = `${EVAL_DIR}/kb_cache`;
      const KB_IDENTITY_FILE = `${EVAL_DIR}/${KB_IDENTITY_BASENAME}`;
      const resolved = await safeAgent(
        `You are the e2e warm-start resolver. Run EXACTLY this command and return its JSON stdout ` +
        `verbatim as StructuredOutput — do not add, drop, reorder, or reinterpret any field. The ` +
        `command pretty-prints its JSON over several lines; return the whole object, not the first ` +
        `line. A non-zero exit or an empty candidate list is a VALID answer (the page has never been ` +
        `written): return what it printed, do not retry with different flags, and do not invent ` +
        `candidates. It may consult the shared KB Store service first and fall back to the on-disk ` +
        `store by itself; run it as one script and do not split it into separate commands.\n` +
        '```bash\n' +
        kbResolveScript(
          `--top-n ${E2E_WARM_START_TOP_N} \\\n` +
          `  --min-speedup ${E2E_WARM_START_MIN_SPEEDUP} \\\n` +
          `  --refs-dir ${shq(refsDir)} --cache-dir ${shq(cacheDir)} \\\n` +
          // The read also leaves this run's KB address on disk, so run_e2e.py can salvage results
          // onto the right pages if this process dies before Module B. --store/--identity-plane are
          // only recorded (so a `both` run's salvage keeps its local mirror).
          `  --identity-out ${shq(KB_IDENTITY_FILE)} \\\n` +
          `  --store ${shq(E2E_KB_STORE_DIR)} --identity-plane ${E2E_KB_PLANE}`) + '\n' +
        '```',
        { phase: 'WarmStart', label: 'warm_start:resolve', schema: KB_RESOLVE_SCHEMA }) || {};
      const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
      // Log the ladder VERBATIM: it is the only record of which address was asked, so identity drift
      // isn't mistaken for an empty store. `read_plane` is the plane that ANSWERED (`plane` if absent).
      KB_READ_PLANE = String(resolved.read_plane || resolved.plane || '');
      log(`[kb] e2e read: plane=${KB_READ_PLANE || '?'} tried=[${(resolved.tried || []).join(' | ')}] ` +
        `answered=${resolved.canonical_id || '?'} tier=${resolved.match_tier || '-'} ` +
        `sorted_by=${resolved.sorted_by || resolved.ranked_by || '-'} ` +
        `champion_metric=${resolved.champion_metric || '-'} reason=${resolved.read_reason || '?'} ` +
        `candidates=${cands.length}`);
      log(`[kb] recall_metric_basis=${E2E_KB_METRIC.basis || 'any'} ` +
        `strict=${E2E_KB_METRIC.strict} source=${E2E_KB_METRIC.source} ` +
        `measurement_metric_basis=${kbMeasurementBasis()}`);
      // The counts BEHIND a zero: nobody wrote this page, everything was retracted, everything was
      // under the floor — all `candidates=0` above. Logged separately so that line stays greppable.
      const curation = (resolved.curation && typeof resolved.curation === 'object') ? resolved.curation : {};
      if (!cands.length && Number(curation.scanned) > 0) {
        log(`[kb] the page was NOT empty: ${curation.scanned} record(s) scanned on plane ` +
          `'${curation.read_plane || KB_READ_PLANE || '?'}' at ` +
          `'${curation.canonical_id || resolved.canonical_id || '?'}' (tier ${curation.tier || '-'}), ` +
          `all curated away — retired=${curation.retired || 0} ` +
          `same_direction_collapsed=${curation.same_direction_collapsed || 0} ` +
          `below_min_speedup=${curation.below_min_speedup || 0} (floor ${curation.min_speedup}). ` +
          'This is a FLOOR result, not a cold deployment.');
      }
      // Record the ask before anything is benched, so a mid-warm-start death still reports which
      // ladder was queried.
      KB_RECALL.e2e = {
        read_reason: String(resolved.read_reason || ''),
        tried: Array.isArray(resolved.tried) ? resolved.tried : [],
        answered: String(resolved.canonical_id || ''),
        match_tier: String(resolved.match_tier || ''),
        plane: E2E_KB_PLANE, read_plane: KB_READ_PLANE, mode: E2E_WARM_START,
        recall_metric_basis: E2E_KB_METRIC.basis,
        recall_metric_strict: E2E_KB_METRIC.strict, recall_metric_source: E2E_KB_METRIC.source,
        measurement_metric_basis: kbMeasurementBasis(),
        candidates: cands.length, configs: [], kernels: [],
        // Both floors, always: what this run declined to see, and at what threshold.
        read_min_speedup: E2E_WARM_START_MIN_SPEEDUP,
        bench_min_speedup: E2E_WARM_START_BENCH_MIN_SPEEDUP,
        curation,
      };
      if (cands.length) KB_REF_DIR = refsDir;   // arms warmStartBlock() for the consumer roles
      // Armed on the same condition: a path to an empty cache reads to an agent as "the file is
      // missing" rather than "there was no record".
      if (cands.length) KB_CACHE_DIR = cacheDir;

      // How many offers are worth a server launch; a coarse rung gets its own smaller budget
      // (E2E_WARM_START_VALIDATE_N_COARSE), not zero.
      const exactTier = (resolved.match_tier || '') === 'exact';
      const benchN = E2E_WARM_START_REF_ONLY ? 0
        : exactTier ? E2E_WARM_START_VALIDATE_N
        : A.e2e_warm_start_validate_n != null ? E2E_WARM_START_VALIDATE_N
        : E2E_WARM_START_VALIDATE_N_COARSE;
      // WHICH offers: on a coarse rung the stored throughput was measured elsewhere, so re-order by
      // speedup (the rung's own metric). The top-N itself was still fetched by throughput.
      const benchOrder = exactTier ? cands : [...cands].sort(
        (x, y) => (Number(y.speedup) || 0) - (Number(x.speedup) || 0));
      // The bench floor: a record under it stays an offered reference and loses only the launch.
      // A record with no stored speedup stays benchable.
      const aboveBenchFloor = (c) =>
        c.speedup == null || Number(c.speedup) >= E2E_WARM_START_BENCH_MIN_SPEEDUP;
      const benchable = benchOrder.filter(aboveBenchFloor);
      const belowBenchFloor = benchN ? benchOrder.filter(c => !aboveBenchFloor(c)) : [];
      if (belowBenchFloor.length) {
        log(`[kb] ${belowBenchFloor.length} offer(s) stay references, not benched: stored speedup ` +
          `below the bench floor of ${E2E_WARM_START_BENCH_MIN_SPEEDUP}x — ` +
          belowBenchFloor.map(c => `${String(c.session_id || '?').slice(-12)}=${c.speedup}x`).join(' ') +
          '. A claim that size is this box\'s own drift; the config and any tuned artifact it ' +
          'carries are still offered to the roles that can use them without a launch.');
      }
      if (cands.length && !benchN) {
        log(`[kb] not benching: ${E2E_WARM_START_REF_ONLY
          ? (FAST_MODE ? 'fast mode — all optimization comes from the head track' : 'warm_start=reference')
          : `the caller set e2e_warm_start_validate_n_coarse=0 and match tier ` +
            `'${resolved.match_tier}' is not exact`}. The offers stay as references.`);
      } else if (benchable.length && !exactTier) {
        log(`[kb] benching ${Math.min(benchN, benchable.length)} of ${benchable.length} offer(s) from coarse ` +
          `tier '${resolved.match_tier}', picked by stored speedup. The stored throughput is not ` +
          `comparable to this baseline; the A/B below is measured on this box and is what counts.`);
      }

      // Seeded with the records the bench floor declined; `skipped` is never attested (not tried).
      const verdicts = belowBenchFloor.map(c => ({
        ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a', outcome: 'skipped',
        why: `stored speedup ${c.speedup}x is below the bench floor of ` +
          `${E2E_WARM_START_BENCH_MIN_SPEEDUP}x — offered as a reference, not measured`,
      }));
      for (const c of benchable.slice(0, benchN)) {
        if (searchClosed(`[kb] replay of config ${c.session_id || '?'}`)) {
          verdicts.push({ ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a',
            outcome: 'skipped', why: 'the final reserve began before its turn' });
          continue;
        }
        // VALIDATE THROUGH THE ORIGINAL GATE (config_tuner:sweep), so a stored config and a fresh
        // one are judged by identical evidence.
        const stored = (c.accepted_config && typeof c.accepted_config === 'object') ? c.accepted_config : {};
        const storedFlags = String(stored.flags || '');
        const storedEnv = String(stored.env || '');
        if (!storedFlags && !storedEnv) {
          verdicts.push({ ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a',
            outcome: 'skipped', why: 'the record carries no config to apply' });
          continue;
        }
        // Key-by-key merge (mergeConfig): the recorded value wins every collision.
        const mf = mergeFlags(curFlags, storedFlags);
        const me = mergeEnv(curEnv, storedEnv);
        const overrides = mf.overrides.concat(me.overrides);
        if (overrides.length) {
          log(`[kb] merge ${c.session_id || '?'}: the recorded config overrides ${overrides.length} ` +
            `knob(s) this run's baseline already pins — ${describeOverrides(overrides)}. ` +
            'The recorded value is the one under test, so it wins.');
        }
        // Nothing to test: the baseline already IS the recorded configuration (a bench would file a
        // ~0% self-comparison as a loss). Left un-attested.
        if (mf.merged === renderFlagItems(parseFlagString(curFlags)) &&
            me.merged === renderEnvItems(parseEnvString(curEnv))) {
          verdicts.push({ ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a',
            outcome: 'skipped',
            why: "this run's baseline already carries the whole recorded configuration" });
          continue;
        }
        const runValidation = (brief, label) => safeAgent(
          roleAgent('config_tuner', 'sweep', brief,
            {
              EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, BASELINE_THROUGHPUT: BASELINE_TPUT,
              NOISE_BAND_PCT: NOISE_BAND,
              CONFIG_DIRECTIONS: [{
                rank: 1,
                direction: 'kb_warm_start:' + (c.direction || 'unlabeled'),
                axis: 'compound (recovered configuration — do not split)',
                // ALREADY MERGED with this run's baseline. Use verbatim; do not re-combine with
                // CURRENT_FLAGS/CURRENT_ENV, or the collisions the merge just resolved come back.
                flags: mf.merged, env: me.merged,
                rationale: `Recorded under ${c.canonical_id || resolved.canonical_id} (session ` +
                  `${c.session_id || '?'}), where it measured ${c.throughput_tok_s != null ? c.throughput_tok_s : '?'}` +
                  ` tok/s (${c.speedup != null ? c.speedup + 'x' : 'speedup not recorded'}) against a ` +
                  `baseline of ${c.baseline_throughput_tok_s != null ? c.baseline_throughput_tok_s : '?'} tok/s. ` +
                  'That number is a HYPOTHESIS about this box, not a measurement of it.' +
                  (overrides.length
                    ? ` NOTE: ${overrides.length} knob(s) in it disagree with this run's baseline and ` +
                      `the recorded value was taken — ${describeOverrides(overrides)}. If the result is ` +
                      'neutral, check these first: they are where the recorded config and this box ' +
                      'were asked to be two different things.'
                    : ''),
              }],
              MERGE_OVERRIDES: overrides, MERGE_ADDED: mf.added.concat(me.added),
              CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
              MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
              SKILL_DIR: WORKFLOW_DIR,
            }),
          { phase: 'WarmStart', label, schema: SWEEP_SCHEMA });

        let sweep = await runValidation(
          'Validate ONE historical configuration recovered from the knowledge base. Treat it exactly ' +
          'as you would a fresh direction: same isolated-server measurement, same parity check, same ' +
          'swap-took-effect verification. THREE deviations from your role file, all deliberate:\n' +
          '(1) Do NOT decompose this direction into one-axis-at-a-time trials. A stored config is an ' +
          'ALREADY-COMPOUNDED whole that was accepted together on another box; benching its knobs ' +
          'separately measures a configuration nobody has ever run, and the parts can each be ' +
          'neutral while the whole is a win (or the reverse). Apply all of it, once, as a single ' +
          'trial.\n' +
          "(2) The direction's `flags`/`env` are ALREADY MERGED with this run's baseline, key by key. " +
          'Pass them VERBATIM to bench_e2e.sh. Do NOT append CURRENT_FLAGS/CURRENT_ENV to them — those ' +
          'are shown to you only so you can see what was overridden, and re-combining them puts the ' +
          'duplicate keys back and hands the outcome to argparse last-wins.\n' +
          '(3) Verify the swap TOOK EFFECT before you believe a null result. This config came from a ' +
          'different framework_version: a flag that was renamed or removed upstream is accepted ' +
          'silently on the command line and then ignored, which is indistinguishable from "the ' +
          'config made no difference". Grep the server log for each flag/env actually being ' +
          'honoured, and if one is not, say so in the trial notes rather than reporting a clean no-op. ' +
          'MERGE_OVERRIDES lists the knobs where the record and this box disagreed — check those in ' +
          'the log first.',
          `warm_start:validate:${c.session_id || 'cand'}`);

        // ONE repair attempt, only when the server produced no number: the tuner classifies the
        // launch failure (a stale record vs a conflict with this baseline), adapts the minimum, and
        // benches once more through the same accept gate.
        let repair = null;
        if (!measuredTrialOf(sweep, BASELINE_TPUT).measured &&
            !searchClosed(`[kb] repair of config ${c.session_id || '?'}`)) {
          const repaired = await runValidation(
            'A recovered configuration was applied to this box and the server produced NO ' +
            'measurement — it failed to launch, failed to become healthy, or died before the bench. ' +
            'This is the ONE repair attempt; there is no second.\n\n' +
            'Read your own launch log from the attempt you just made. Then, for each knob that came ' +
            'from the record (MERGE_OVERRIDES and MERGE_ADDED name them), classify it:\n' +
            '  - `upstream_gone`: this build does not have the flag/env at all — argparse rejected ' +
            'it, or it is absent from `--help`. This is the RECORD being stale.\n' +
            "  - `conflicts_with_baseline`: the build has it, but the value cannot hold together with " +
            "something this run's baseline pins (a length above the served context, a memory " +
            'fraction the rest of the config cannot fit under, a backend the baseline excludes). ' +
            'This is the PAIRING failing, and says nothing about the record.\n' +
            '  - `env_unsupported`: the build ignores or rejects the env var.\n' +
            'Then drop or adapt the MINIMUM needed to get a running server — keep as much of the ' +
            'recorded configuration as will run, and never drop a knob you did not have to — and ' +
            'bench that once, exactly as before.\n\n' +
            'Return the same JSON, plus a `repair` object: `{"attempted": true, "classification": ' +
            '"upstream_gone|conflicts_with_baseline|env_unsupported|mixed|unknown", ' +
            '"dropped_flags": ["--flag", ...], "reason_by_flag": {"--flag": "why"}, ' +
            '"launch_error": "<the line that actually killed it>"}`. If the server still will not ' +
            'come up, say so and return `best_throughput_tok_s: 0` — an honest "cannot run here" is ' +
            'worth more than a number from a configuration that is no longer the record.',
            `warm_start:repair:${c.session_id || 'cand'}`);
          if (repaired) {
            repair = (repaired.repair && typeof repaired.repair === 'object') ? repaired.repair : {};
            sweep = repaired;
          }
          log(`[kb] repair ${c.session_id || '?'}: ` + (repair
            ? `classified ${repair.classification || 'unknown'}` +
              `${(repair.dropped_flags || []).length ? `, dropped ${(repair.dropped_flags || []).join(' ')}` : ''}` +
              `${measuredTrialOf(sweep, BASELINE_TPUT).measured ? ', server came up' : ', still no measurement'}`
            : 'the repair attempt itself produced nothing'));
        }
        const { trial, measured } = measuredTrialOf(sweep, BASELINE_TPUT);
        // The role's own drop list, else what was asked for minus what ran.
        const dropped = (repair && Array.isArray(repair.dropped_flags) && repair.dropped_flags.length)
          ? repair.dropped_flags
          : (repair ? droppedByDiff(mf.merged, me.merged,
            String(trial.change || [trial.flags, trial.env].filter(Boolean).join(' | '))) : []);
        const parity = String(trial.parity || trial.output_parity || '');
        const deltaPct = BASELINE_TPUT ? ((measured - BASELINE_TPUT) / BASELINE_TPUT) * 100 : 0;
        // Both the tuner's judgement AND our arithmetic: the ratio is re-derived from the raw number
        // rather than trusting `kept: true`.
        const accept = trial.kept === true && measured > BASELINE_TPUT &&
          deltaPct > NOISE_BAND && parity !== 'fail';
        if (accept) {
          curFlags = sweep.accepted_flags || mf.merged || curFlags;
          curEnv = sweep.accepted_env || me.merged || curEnv;
          kbSeedTput = measured;
          await requireE2EValidationCheckpoint('config/e2e_validation.json', {
            phase: 'WarmStart', validation_level: 'config_sweep', gate: 'accepted',
            validation_status: 'accepted_config',
            baseline_throughput_tok_s: BASELINE_TPUT, final_throughput_tok_s: measured,
            throughput_speedup: BASELINE_TPUT ? measured / BASELINE_TPUT : 1,
            baseline_config: { flags: INIT_FLAGS, env: INIT_ENV },
            accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
            accepted_kernels: [], accepted_heads: [], final_patch: [],
            final_overlay: { path: curOverlay },
            final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
            bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
            measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
              workload: WORKLOAD, source_artifact: `${EVAL_DIR}/config/sweep_results.json`,
              acceptance_source: 'kb_warm_start',
              acceptance: { gain_exceeds_noise: true, correctness_passed: true, noise_floor_pct: NOISE_BAND } },
            stack: { config_layers: (sweep.trials || []).filter((t) => t && t.kept === true) },
            replay: { asset_paths: [`${EVAL_DIR}/bench_e2e.sh`, `${EVAL_DIR}/config/sweep_results.json`] },
            integrity: { checkpoint_assets: [] },
          });
          log(`[kb] ADOPTED ${c.session_id || '?'} (${c.direction || 'unlabeled'}): ` +
            `${measured} tok/s, +${deltaPct.toFixed(2)}% vs baseline ${BASELINE_TPUT} (noise band ${NOISE_BAND}%)` +
            `${dropped.length ? `, with ${dropped.join(' ')} dropped to make it run here` : ''}.`);
        }
        // FOUR outcomes (see KB_ATTEST_OUTCOME), counted apart by kb/attest.py; only a loss and a
        // no-effect count against the record, as evidence toward a threshold. The role file spells
        // the per-trial text `note`.
        const notes = String(trial.notes || trial.note || (sweep && sweep.summary) || '');
        const inert = /\b(?:not (?:honou?red|recognized|recognised|applied|supported)|unrecognized|unrecognised|ignored|no such option|unknown (?:option|argument|flag)|renamed|removed upstream|did not take effect)\b/i.test(notes);
        // The repair's structured classification outranks the free-text regex.
        const conflicted = String((repair && repair.classification) || '') === 'conflicts_with_baseline';
        const outcome = accept ? 'adopted'
          : !measured ? (conflicted ? 'inapplicable' : 'not_reproduced')
            : inert ? 'not_reproduced' : 'rejected';
        if (!accept) {
          log(`[kb] ${{ not_reproduced: 'NOT REPRODUCED', inapplicable: 'INAPPLICABLE HERE' }[outcome] || 'rejected'} ` +
            `${c.session_id || '?'} (${c.direction || 'unlabeled'}): ` +
            `measured ${measured || 'n/a'} tok/s vs baseline ${BASELINE_TPUT}` +
            `${measured ? ` (${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(2)}%)` : ''}` +
            `${parity ? `, parity=${parity}` : ''}${inert ? ', the config never took effect' : ''}` +
            `${outcome === 'inapplicable' ? ", it collides with this run's baseline" : ''}` +
            ` — kept as a reference, not applied. This is this box's result, not a verdict on the ` +
            `record: it is filed as evidence toward a threshold, and retires nothing by itself.`);
        }
        verdicts.push({ ...c, measured_tok_s: measured || null, delta_pct: measured ? deltaPct : null,
          parity: parity || 'unknown', outcome, why: notes,
          // What was actually put on the box vs what the record asked for.
          overrides, applied_partial: dropped.length > 0, dropped_flags: dropped,
          repair_classification: String((repair && repair.classification) || '') });
        if (accept) break;   // adopt the first that passes; the rest stay references
      }
      // ---- Replay the stored KERNELS through the ordinary integrate gate ----
      // Same machinery as the Milestone track (runIntegrateBothLegs, INTEGRATE_SCHEMA, the isolated
      // A/B with parity probe, integAccepted). A candidate lives in its own overlay directory, active
      // only on PYTHONPATH, so rejecting one is simply not adopting it.
      const kbKernels = [];
      const seenKernel = new Set();
      for (const c of cands) {
        const bundlePath = (c.bundle && c.bundle.path) || '';
        // The record's prebuilt overlay (`overlay.tar.gz` from e2e_store._pack_overlay), found via
        // the bundle manifest (no filesystem here); without it the entry stays a reference.
        const bundleFiles = Array.isArray(c.bundle && c.bundle.files) ? c.bundle.files : [];
        const overlayTar = (bundlePath && bundleFiles.includes('overlay.tar.gz'))
          ? `${bundlePath}/files/overlay.tar.gz` : '';
        for (const k of (Array.isArray(c.accepted_kernels) ? c.accepted_kernels : [])) {
          const name = String((k && (k.name || k.short_name)) || '').trim();
          if (!name || seenKernel.has(name)) continue;   // one op recorded by several runs is one op
          seenKernel.add(name);
          kbKernels.push({ ...k, name, bundle: bundlePath, overlay_tar: overlayTar,
            kb_session_id: c.session_id || '' });
        }
      }
      let replayed = 0;
      // One overlay tarball is the whole RECORD's final overlay, so it is benched ONCE and the
      // record's other kernels are marked as covered by it.
      const overlayReplayed = new Set();
      const kernelVerdicts = [];
      for (const k of kbKernels) {
        const kind = String(k.winner_kind || k.kind || '').trim().toLowerCase();
        // `patch` lives in the record's bundle (`kernels/<name>.patch`); with no bundle there is no
        // file to hand the integrator.
        const patchPath = (kind === 'patch' && k.patch && k.bundle)
          ? `${k.bundle}/files/${k.patch}` : '';
        // An env/flag win is replayable only if the record names the env or flag that routed the op;
        // otherwise the A/B would compare two identical configurations.
        const hasRouting = !!(String(k.apply_env || '').trim() || String(k.apply_flags || '').trim());
        // A kind this lane cannot rebuild from a record (e.g. `authored`, whose workspace is gone) can
        // still be replayed from the record's built overlay; its manifest names the rebind seams, so
        // the A/B shows whether they bind here.
        const viaOverlay = !E2E_REPLAYABLE_KINDS.has(kind) && !!k.overlay_tar;
        const unreplayable =
          E2E_WARM_START_REF_ONLY ? 'read-only mode'
          : (!E2E_REPLAYABLE_KINDS.has(kind) && !k.overlay_tar)
            ? `winner_kind '${kind || 'unrecorded'}' cannot be rebuilt from a record, and this record ships no overlay.tar.gz to replay instead`
          : (viaOverlay && overlayReplayed.has(k.bundle))
            ? "already covered by this record's overlay replay — one tarball is the whole record's overlay, not one kernel's"
          : (!viaOverlay && kind === 'patch' && !patchPath) ? 'the record names a patch but its bundle holds no file'
          : (!viaOverlay && kind !== 'patch' && !hasRouting) ? `recorded as a '${kind}' win but carries no apply_env/apply_flags, so there is nothing to re-apply`
          : replayed >= E2E_WARM_START_KERNELS_N ? `replay budget of ${E2E_WARM_START_KERNELS_N} already spent`
          : searchClosed(`[kb] replay of kernel ${k.name}`) ? 'the final reserve began before its turn'
          : '';
        if (unreplayable) {
          kernelVerdicts.push({ ...k, kind, outcome: 'reference', measured_delta_pct: null,
            why: unreplayable });
          continue;
        }
        replayed++;
        if (viaOverlay) overlayReplayed.add(k.bundle);
        const isolated = Number(k.isolated_speedup) || 0;
        const kbIntegrateInputs = {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
          KERNEL_RESULT: {
            short_name: k.name, winner_kind: kind,
            // Both patch spellings: the two tracks read different ones.
            code_patch: patchPath, final_patch: patchPath,
            apply_env: String(k.apply_env || ''), apply_flags: String(k.apply_flags || ''),
            target_callable: String(k.target_callable || ''),
            source_path_in_sglang: String(k.source_path_in_sglang || ''),
            verified_isolated_speedup: isolated, pct_gpu_time: Number(k.pct_gpu_time) || 0,
            // task_dir is deliberately EMPTY and the provenance is declared foreign — see the intro.
            task_dir: '', provenance: 'knowledge_base_replay',
          },
          // Only on the prebuilt-overlay path: unpack the candidate rather than assemble it.
          ...(viaOverlay ? { KB_OVERLAY_TARBALL: k.overlay_tar } : {}),
          CURRENT_OVERLAY: kbSeedOverlay || curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
          CURRENT_THROUGHPUT: kbSeedTput || BASELINE_TPUT, SKILL_DIR: WORKFLOW_DIR,
        };
        const integ = await runIntegrateBothLegs(
          'Overlay a kernel RECOVERED FROM THE KNOWLEDGE BASE and gate it on e2e throughput. Run your ' +
          'normal isolated-server A/B — same fresh-server legs, same parity probe. Two things are different ' +
          'and you must honour both:\n' +
          (viaOverlay
            ? '(0) DO NOT BUILD THE CANDIDATE OVERLAY. This kernel was authored by another run, whose ' +
              'workspace does not exist here, so your `authored` recipe (git apply into ' +
              'authored_kernel_eval_dir/workspace, copy kernel_src, add-rebind) cannot run and must not ' +
              'be attempted. The record ships the FINISHED overlay instead. Unpack it and bench THAT ' +
              'directory as the candidate. Note the tarball is a STANDALONE overlay from another run: ' +
              'untarring it over CURRENT_OVERLAY would overwrite `_overlay_manifest.json` and drop every ' +
              'rebind already accepted here, which turns the A/B into a comparison against a candidate ' +
              'that is missing part of its own reference. Merge all manifest entries, including deferred ' +
              'installers, with the checked merge command below. Use a fresh CAND directory; a conflict ' +
              'must be reported as inapplicable, never resolved by dropping existing hooks:\n' +
              '```bash\n' +
              'CAND="$EVAL_DIR/overlay/cand_<short_name>"\n' +
              `KBO=$(mktemp -d) && tar xzf ${shq(k.overlay_tar)} -C "$KBO" --strip-components=1\n` +
              'python3 "$SKILL_DIR/scripts/overlay_setup.py" merge --overlay "$CAND" \\\n' +
              '  --from "$CURRENT_OVERLAY" --with "$KBO"\n' +
              '```\n' +
              'The manifest declares rebinds, injected modules and installers with their target sites. ' +
              'VERIFY EVERY TARGET ACTUALLY TOOK on the candidate server (installer verification API, ' +
              'load banner, and an executed seam call) before you believe ' +
              'a null result: this overlay was built against a different framework_version, and a rebind ' +
              'whose target module was renamed upstream binds nothing, silently, which is indistinguishable ' +
              'from "the kernel made no difference". If no rebind takes, report gate:"rejected" with ' +
              'reason `no_rebind_seam` rather than a clean no-op. An authored/JIT kernel on the decode ' +
              'path still has to satisfy your CUDA-graph-safety rule — the packed source is rebuilt by ' +
              "this box's own torch load(), and a build failure is a rejection, not a retry.\n"
            : '') +
          '(1) There is NO task_dir and therefore NO immutable oracle for this kernel. Your step-1 ' +
          'provenance re-check cannot run, because the workspace that produced this patch does not ' +
          'exist on this box. Do NOT claim provenance was verified. The FRESH correctness probe against ' +
          'the TRUE baseline (greedy parity, or the task-accuracy gate when ACCURACY_GATE is in your ' +
          'inputs) is the only correctness evidence available here, so run it and report its result ' +
          'plainly — if it fails or cannot be established, REJECT.\n' +
          '(2) verified_isolated_speedup is a FOREIGN number, measured on another box against another ' +
          'baseline. It is context for what to expect, never evidence. Gate on what YOU measure.\n' +
          'As always: never mutate the installed package. The overlay on PYTHONPATH is the only ' +
          'mechanism, so a rejection costs nothing to undo — you simply do not adopt the directory.',
          kbIntegrateInputs, `warm_start integrate ${k.name}`, 'WarmStart');
        const base = kbSeedTput || BASELINE_TPUT;
        if (abDone(integ) && integAccepted(integ, Number(k.pct_gpu_time) || 0, isolated) &&
            integ.e2e_throughput_tok_s > base) {
          kbSeedOverlay = integ.accepted_overlay || kbSeedOverlay;
          kbSeedTput = integ.e2e_throughput_tok_s;
          bankAccepted(kbSeedKernels, {
            short_name: k.name, backend: String(k.language || k.backend || ''), kind,
            // On the prebuilt-overlay path there is no patch: the accepted overlay (packed by
            // Module B) is what reproduces the win.
            e2e_delta_pct: integ.e2e_delta_pct, isolated, patch: patchPath,
            replayed_from_overlay: viaOverlay ? k.overlay_tar : '',
            pct_gpu_time: Number(k.pct_gpu_time) || 0,
            apply_env: String(k.apply_env || ''), apply_flags: String(k.apply_flags || ''),
            // Provenance stays on the entry: a recovered win re-verified here, not a discovery.
            from_knowledge_base: true, kb_session_id: k.kb_session_id,
          }, krOf(kbIntegrateInputs));
          kernelVerdicts.push({ ...k, kind, outcome: 'adopted',
            measured_delta_pct: integ.e2e_delta_pct, why: integ.reason || '' });
          log(`[kb] ADOPTED kernel ${k.name} (${kind}): e2e now ${kbSeedTput} tok/s (+${integ.e2e_delta_pct}%).`);
        } else {
          const reason = gateRejectReason(integ, Number(k.pct_gpu_time) || 0, isolated);
          kernelVerdicts.push({ ...k, kind, outcome: abDone(integ) ? 'rejected' : 'incomplete',
            measured_delta_pct: integ ? integ.e2e_delta_pct : null,
            why: reason || 'A/B did not complete' });
          // No corrective re-author: a stored patch that fails its fresh gate is handed to the
          // Architect as a lead, not repaired.
          log(`[kb] kernel ${k.name} (${kind}) not adopted: ${reason || 'A/B did not complete'} — kept as a reference.`);
        }
      }
      // ---- Demote what did not survive into a REFERENCE for the roles that come next ----
      // The store's `e2e_reference_<sha7>.md` stays as its claim; this box's measurements go in a
      // sibling file that overrides it, surfaced three ways: the reference dir, the prompt block
      // (WARM_START_ROLES) and the `## Inputs` entry, which always lands.
      // ---- Tell the STORE what happened ----
      // `e2e_store.py attest` counts each attempt onto the record (no score or champion moves;
      // KB_ATTEST_OUTCOME sets the buckets, kb/attest.py:should_retire the curate threshold). Only
      // config verdicts benched on this box count; a record whose bundle also claims kernels is
      // reported as config-half-only.
      const configHalfOnly = v => Array.isArray(v.accepted_kernels) && v.accepted_kernels.length > 0;
      const attestable = verdicts.filter(v => v.session_id && kbCanAttestMetric(v) &&
        ['adopted', 'rejected', 'not_reproduced', 'inapplicable'].includes(v.outcome));
      const crossMetric = verdicts.filter(v => v.session_id && !kbCanAttestMetric(v));
      if (crossMetric.length) log(`[kb] ${crossMetric.length} record(s) not attested: ` +
        'the local measurement axis does not establish their stored metric claim.');
      if (attestable.length) {
        const cmds = attestable.map(v =>
          `python3 ${shq(E2E_STORE_SCRIPT)} attest ${kbIdentityFlags()} ${kbPlaneFlags()} ` +
          `--session-id ${shq(v.session_id)} ` +
          `--outcome ${KB_ATTEST_OUTCOME[v.outcome] || 'inapplicable'} ` +
          (v.measured_tok_s ? `--measured-tok-s ${v.measured_tok_s} ` : '') +
          `--baseline-tok-s ${BASELINE_TPUT} --parity ${shq(v.parity || 'n/a')} ` +
          // The note says what actually ran (overrides, drops), not just why it ended.
          `--note ${shq([
            // The local verdict leads the note: the filed bucket (KB_ATTEST_OUTCOME) may not show it.
            `local verdict: ${v.outcome}` +
              (v.measured_tok_s ? ` (${v.measured_tok_s} tok/s vs baseline ${BASELINE_TPUT})` : ' (no measurement)'),
            configHalfOnly(v)
              ? `config half only — the record also claims ${v.accepted_kernels.length} kernel(s), ` +
                'benched in the kernel lane and NOT covered by this verdict' : '',
            (v.overrides || []).length ? `overrode ${describeOverrides(v.overrides)}` : '',
            (v.dropped_flags || []).length
              ? `dropped ${(v.dropped_flags || []).join(' ')} to make it run here` +
                `${v.repair_classification ? ` (${v.repair_classification})` : ''}` : '',
            String(v.why || ''),
          ].filter(Boolean).join('; ').slice(0, 400))} ` +
          `--measured-by ${shq('e2e_workflow:' + BACKEND)} --apply || true`);
        try {
          await safeAgent(
            `You are the e2e knowledge-base attestor. Run EXACTLY these commands in order and ` +
            `return {"ran": <how many you ran>, "note": "<anything that failed>"}. Each records ` +
            `what THIS box saw when it benched a stored record. Do NOT edit them, do NOT add or ` +
            `drop any, and do NOT retry a failure — a repeat would double-count the attempt, and ` +
            `the ledger is meant to say how often this record was tried here, not how often the ` +
            `attestor pressed the button.\n` +
            '```bash\n' + KB_ENV_PRELUDE + cmds.join('\n') + '\n```',
            { phase: 'WarmStart', label: 'kb:attest',
              schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) },
            1);
          log(`[kb] attested ${attestable.length} benched record(s): ` +
            attestable.map(v => `${v.session_id.slice(-12)}=${v.outcome}`).join(' '));
        } catch (e) {
          // Non-fatal, like every other KB write.
          log(`[kb] attest failed (NON-FATAL): ${String(e).slice(0, 200)}`);
        }
      }

      const allVerdicts = verdicts.concat(kernelVerdicts);
      // The measured_on_this_box.md rows in machine form, for the Report role. Set outside the
      // `if (allVerdicts.length)` guard: "offered but not benched" is itself reportable.
      if (KB_RECALL.e2e) {
        KB_RECALL.e2e.configs = verdicts.map(v => ({
          direction: String(v.direction || 'unlabeled'), session_id: String(v.session_id || ''),
          stored_tok_s: v.throughput_tok_s != null ? v.throughput_tok_s : null,
          stored_speedup: v.speedup != null ? v.speedup : null,
          // Without the stored baseline a lower `delta_pct` reads as decay when the two baselines
          // were never comparable; without the half-flag it reads as a failed kernel.
          stored_baseline_tok_s: v.baseline_throughput_tok_s != null ? v.baseline_throughput_tok_s : null,
          config_half_only: configHalfOnly(v),
          measured_tok_s: v.measured_tok_s != null ? v.measured_tok_s : null,
          delta_pct: v.delta_pct != null ? v.delta_pct : null,
          parity: String(v.parity || ''), outcome: String(v.outcome || ''),
          // What the merge and repair did, beside the number they produced.
          overrides: Array.isArray(v.overrides) ? v.overrides : [],
          applied_partial: v.applied_partial === true,
          dropped_flags: Array.isArray(v.dropped_flags) ? v.dropped_flags : [],
          why: String(v.why || '').slice(0, 300),
        }));
        KB_RECALL.e2e.kernels = kernelVerdicts.map(v => ({
          name: String(v.name || ''), kind: String(v.kind || ''),
          kb_session_id: String(v.kb_session_id || ''),
          stored_isolated: v.isolated_speedup != null ? v.isolated_speedup : null,
          measured_delta_pct: v.measured_delta_pct != null ? v.measured_delta_pct : null,
          outcome: String(v.outcome || ''), why: String(v.why || '').slice(0, 300),
        }));
      }
      if (allVerdicts.length) {
        const adoptedCfg = verdicts.filter(v => v.outcome === 'adopted');
        const adoptedKer = kernelVerdicts.filter(v => v.outcome === 'adopted');
        const rejected = allVerdicts.filter(v => v.outcome !== 'adopted');
        // Also counted inside `rejected`: to the next roles both mean "do not re-propose verbatim".
        const notReproduced = verdicts.filter(
          v => v.outcome === 'not_reproduced' || v.outcome === 'inapplicable');
        const md = [
          '# Warm start — MEASURED ON THIS BOX',
          '',
          'This file OVERRIDES the stored claims in its sibling `e2e_reference_*.md` wherever the two',
          'disagree. Those files record what another box reported; this one records what happened when',
          'this run applied the same thing here, through the same gate a fresh idea faces.',
          '',
          `- baseline: **${BASELINE_TPUT} tok/s** (noise band ${NOISE_BAND}%)`,
          `- serving: BACKEND=${BACKEND} TP=${SERVING_TP} GPU=${SERVING_GPU}, workload isl=${ISL} osl=${OSL} conc=${CONC}`,
          `- identity read: \`${resolved.canonical_id || '?'}\` (match tier \`${resolved.match_tier || '-'}\`)`,
          // `exact` covers the identity only (no flags, env or kv-cache dtype), so a record measured
          // against another baseline can reproduce smaller here for that reason alone.
          ...(verdicts.some(v => v.baseline_throughput_tok_s != null && BASELINE_TPUT &&
              Math.abs(v.baseline_throughput_tok_s - BASELINE_TPUT) / BASELINE_TPUT > NOISE_BAND / 100)
            ? ['- CAUTION: the match tier covers the identity only — not server flags, env or ' +
               'kv-cache dtype. At least one offer below was recorded against a different baseline ' +
               '(column `stored baseline`), so its stored percentage and the one measured here are ' +
               'not the same quantity.']
            : []),
          '',
          '## Configurations',
          '',
          '| stored direction | stored claim | stored baseline | measured here | delta vs baseline | parity | outcome |',
          '|---|---|---|---|---|---|---|',
          ...(verdicts.length ? verdicts.map(v =>
            `| ${v.direction || 'unlabeled'} | ${v.throughput_tok_s != null ? v.throughput_tok_s + ' tok/s' : '?'}` +
            `${v.speedup != null ? ` (${v.speedup}x)` : ''}` +
            `${configHalfOnly(v) ? ` — WHOLE BUNDLE, incl. ${v.accepted_kernels.length} kernel(s); only the config half ran in this row` : ''} | ` +
            `${v.baseline_throughput_tok_s != null ? v.baseline_throughput_tok_s + ' tok/s' : '—'} | ` +
            `${v.measured_tok_s != null ? v.measured_tok_s + ' tok/s' : 'not benched'} | ` +
            `${v.delta_pct != null ? (v.delta_pct >= 0 ? '+' : '') + v.delta_pct.toFixed(2) + '%' : '—'} | ` +
            `${v.parity || '—'} | **${v.outcome}** |`)
            : ['| _(none offered)_ | | | | | | |']),
          '',
          '## Kernels',
          '',
          '| kernel | kind | stored isolated | measured e2e delta | outcome | why |',
          '|---|---|---|---|---|---|',
          ...(kernelVerdicts.length ? kernelVerdicts.map(v =>
            `| ${v.name} | ${v.kind || '?'} | ${v.isolated_speedup ? v.isolated_speedup + 'x' : '—'} | ` +
            `${v.measured_delta_pct != null ? (v.measured_delta_pct >= 0 ? '+' : '') + v.measured_delta_pct + '%' : '—'} | ` +
            `**${v.outcome}** | ${String(v.why || '').slice(0, 160).replace(/[\\|]/g, '\\$&')} |`)
            : ['| _(none recorded)_ | | | | | |']),
          '',
          // A top-N entry that could not be reproduced is handed forward WITH its reproduction
          // material (launch script, patch paths).
          ...(notReproduced.length ? [
            '## REFERENCE ONLY — recalled but NOT reproduced here',
            '',
            'These were offered by the store and benched on this box, and either produced no number',
            'at all, never took effect (a flag renamed upstream is accepted silently and then',
            "ignored), or collided with a knob this run's baseline already pins and could not be",
            'applied here at all. They are NOT results — and the last of those is not evidence',
            'against the record either, only against the pairing. They are the closest thing this',
            'deployment has to a record of what someone else got working, and their material is',
            'below so you can read',
            'what they actually did rather than guess from a direction label.',
            '',
            ...notReproduced.flatMap(v => {
              const repro = (v.repro && typeof v.repro === 'object') ? v.repro : {};
              const root = (v.bundle && v.bundle.path) ? `${v.bundle.path}/files` : '';
              const patches = (Array.isArray(repro.kernels) ? repro.kernels : [])
                .filter(k => k && k.patch).map(k => root ? `${root}/${k.patch}` : k.patch);
              return [
                `### ${v.direction || 'unlabeled'} (session \`${v.session_id || '?'}\`)`,
                `- why it did not reproduce: ${v.outcome === 'inapplicable'
                  ? "it could not be applied to this run's baseline at all — " : ''}` +
                  `${String(v.why || 'no measurement came back').slice(0, 300)}`,
                ...((v.overrides || []).length
                  ? [`- knobs where it disagreed with this run's baseline (recorded value was ` +
                     `taken): ${describeOverrides(v.overrides)}`] : []),
                ...((v.dropped_flags || []).length
                  ? [`- dropped to get it running here: ${v.dropped_flags.join(' ')}` +
                     `${v.repair_classification ? ` (${v.repair_classification})` : ''} — so the ` +
                     'number above, if any, is for LESS than the record asked for'] : []),
                `- stored claim: ${v.throughput_tok_s != null ? v.throughput_tok_s + ' tok/s' : '?'}` +
                  `${v.speedup != null ? ` (${v.speedup}x)` : ''}, recorded elsewhere`,
                `- launch script: ${repro.launch
                  ? `\`${root ? `${root}/${repro.launch}` : repro.launch}\`` +
                    (repro.launch_origin === 'synthesized'
                      ? ' — SYNTHESIZED from the stored config, never executed as-is'
                      : ' — captured from the original run')
                  : 'none recorded (this record predates `repro`)'}`,
                `- kernel patches: ${patches.length ? patches.map(p => `\`${p}\``).join(', ') : 'none carried'}` +
                  `${repro.kernels_without_patch ? ` (${repro.kernels_without_patch} accepted kernel(s) carry no patch)` : ''}`,
                `- flags: \`${String((v.accepted_config || {}).flags || '(none)')}\``,
                `- env: \`${String((v.accepted_config || {}).env || '(none)')}\``,
                '',
              ];
            }),
          ] : []),
          '## How to use this',
          '',
          adoptedCfg.length
            ? `The adopted configuration is ALREADY in CURRENT_FLAGS / CURRENT_ENV. Do not re-propose it — ` +
              `propose only things that COMPOUND on top of it.`
            : `Nothing from the store was adopted as configuration, so CURRENT_FLAGS / CURRENT_ENV are ` +
              `unchanged from the baseline.`,
          '',
          adoptedKer.length
            ? `${adoptedKer.length} recovered kernel(s) are already in the active overlay and already ` +
              `reflected in the profile you are routing from. Their ops are DONE; look elsewhere.`
            : `No recovered kernel was adopted, so the overlay is empty and every op in the profile is ` +
              `still open.`,
          '',
          rejected.length
            ? `The ${rejected.length} rejected/unreplayed entries above are LEADS, not dead ends. A ` +
              `rejection here means the whole compounded thing did not beat this baseline on this box — ` +
              `it does NOT mean each knob inside it is worthless, and it does not mean the DIRECTION is ` +
              `wrong. Do not re-propose any of them verbatim; do feel free to propose an individual axis ` +
              `from one, or the same idea approached differently.`
            : `Nothing was rejected.`,
          '',
        ].join('\n');
        await safeAgent(
          `You are a file writer. Use the Write tool to create the file ` +
          `"${refsDir}/measured_on_this_box.md" with EXACTLY the content below, verbatim. ` +
          `Do NOT reformat, summarize, re-order rows, or change any number:\n\n` +
          '````markdown\n' + md + '\n````\n\n' +
          `Then return {"written": true, "path": "${refsDir}/measured_on_this_box.md"}.`,
          { phase: 'WarmStart', label: 'warm_start:record-measurements',
            schema: obj({ written: { type: 'boolean' }, path: { type: 'string' } }, []) },
          2);
        KB_REF_DIR = refsDir;   // the measurements exist even if the offer list was thin

        // The channel that always fires (`## Inputs` of two roles); terse on purpose.
        KB_REF_INPUTS = {
          KB_REFERENCE_DIR: refsDir,
          KB_REFERENCE_VERDICT:
            `The knowledge base offered ${allVerdicts.length} prior result(s) for this deployment ` +
            `(${resolved.canonical_id || '?'}, tier ${resolved.match_tier || '-'}). This run benched them ` +
            `and recorded what actually happened in ${refsDir}/measured_on_this_box.md — read that file, ` +
            `and treat it as overriding the stored claims in its siblings. ` +
            (adoptedCfg.length
              ? `ADOPTED config: ${adoptedCfg.map(v => v.direction || 'unlabeled').join(', ')} — it is ALREADY ` +
                `in CURRENT_FLAGS/CURRENT_ENV, so propose only things that COMPOUND on top of it, never it again. `
              : `No stored config was adopted. `) +
            (adoptedKer.length
              ? `ADOPTED kernels: ${adoptedKer.map(v => v.name).join(', ')} — already in the overlay and already ` +
                `reflected in the profile, so those ops are done. `
              : `No stored kernel was adopted. `) +
            (rejected.length
              ? `REJECTED/unreplayed: ${rejected.map(v => v.direction || v.name || '?').join(', ')}. Do not ` +
                `re-propose any of them verbatim — the compounded whole lost on this box. Their individual ` +
                `knobs may each still be a valid axis, and their declared directions are still legitimate ` +
                `ideas to reach a different way.`
              : '') +
            (notReproduced.length
              ? ` ${notReproduced.length} of those could not be reproduced AT ALL here (no number, or the ` +
                `config never took effect) — they are in the REFERENCE ONLY section of that file with their ` +
                `launch script and kernel patches attached. Read them as evidence of what worked somewhere ` +
                `else, not as something to re-run.`
              : ''),
        };
      }
    }
  }

  // A resumed overlay wins; otherwise a freshly replayed KB overlay stacks on the Hyperloom underlay.
  // Fold before Profile and ConfigSweep so every downstream measurement sees the active source stack.
  curOverlay = ST.overlay || kbSeedOverlay || curOverlay || '';

  if (searchClosed('Profile and Strategize') || searchWindowShort('Profile, kernel extraction and the kernel search')) {
    kernelQueue = [];
    headQueue = [];
  } else {
    if (KERNEL_EXTRACT_ON) {
      strategy = await prepareAgentXKernels('baseline', kbSeedTput || BASELINE_TPUT);
      profile = strategy;
    } else {
      phase('Profile');
      profile = await safeAgent(
        roleAgent('profiler', 'baseline', 'Capture a warm trace and emit the standardized Top-N.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 0,
          OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          ...TRACELENS_INPUTS, ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Profile', label: 'profiler:baseline', schema: PROFILE_SCHEMA });
      log(`Baseline profiled. ${profile ? (profile.top_kernels || []).length : 0} top kernels.`);

      phase('Strategize');
      strategy = await safeAgent(
        roleAgent('system_architect', 'strategize', 'Route the Top-N into config/kernel/host tracks by Amdahl.', {
          EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: BASELINE_TPUT,
          WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED, SKILL_DIR: WORKFLOW_DIR,
          ...TRACELENS_INPUTS, ...ANALYSIS_SKILL_INPUTS, ...KB_REF_INPUTS,
        }),
        { phase: 'Strategize', label: 'architect:strategize', schema: STRATEGY_SCHEMA });
    }
    kernelQueue = (strategy && strategy.kernel_candidates) ? strategy.kernel_candidates.slice() : [];
    headQueue = (strategy && strategy.head_candidates) ? strategy.head_candidates.slice() : [];
    headQueue = applyOpIdentityGuard(headQueue, 'strategize');
    log(`Strategy: ${headQueue.length} head candidates, ${kernelQueue.length} kernel candidates, ${(strategy && strategy.config_directions || []).length} config directions.`);
    // strategize decided the backends -> if any candidate routed flydsl, provision it now (blocking).
    await ensureFlydslGate();
  }
} else {
  // Load carried state from a prior phase invocation (args.state).
  EVAL_DIR = ST.eval_dir || EVAL_DIR_OVERRIDE;
  if (!EVAL_DIR) throw new Error('Non-setup phase requires args.state.eval_dir (or args.eval_dir)');
  MODEL_NAME = ST.model_name || MODEL_NAME_HINT;
  BASELINE_TPUT = ST.baseline_throughput_tok_s || 0;
  NOISE_BAND = gateNoiseBand(ST.noise_band_pct);
  curFlags = ST.flags || '';
  curEnv = ST.env || '';
  curOverlay = ST.overlay || INIT_BASE_OVERLAY;
  profile = { profile_topN_json: ST.profile_topn_json || '',
    ...(KERNEL_EXTRACT_ON ? { profile_workload_json: ST.profile_workload_json || '',
      kernel_ranking_json: ST.kernel_ranking_json || '',
      kernel_extraction_delivery_json: ST.kernel_extraction_delivery_json || '',
      failed_tasks: ST.kernel_extraction_summary?.failed_tasks || [],
      deferred_tasks: ST.kernel_extraction_summary?.deferred_tasks || [],
      coverage_summary: ST.kernel_extraction_summary?.coverage_summary || {} } : {}) };
  strategy = { config_directions: ST.config_directions || [] };
  kernelQueue = ST.kernelQueue || [];
  headQueue = admitHeads(ST.headQueue || [], 'resume');
  // Dispatch replays prepared_ut.result (it cannot read disk), so a candidate without it re-runs
  // preparation, which re-packs the saved evidence rather than redoing GPU work.
  const unusable = (k) => !k || !k.prepared_ut || !k.prepared_ut.result;
  if (KERNEL_EXTRACT_ON &&
      (PHASES.includes('extract') ||
       (want('head') && (!Array.isArray(ST.headQueue) || headQueue.some(unusable))) ||
       (want('kernel') && (!Array.isArray(ST.kernelQueue) || kernelQueue.some(unusable))))) {
    // Finish the skill through GPU UT validation. Existing selection-only queues
    // are evidence to reuse, not completed tasks ready for optimization.
    strategy = await prepareAgentXKernels('resume', ST.throughput || BASELINE_TPUT);
    profile = strategy;
    headQueue = applyOpIdentityGuard(strategy.head_candidates.slice(), 'resume selection');
    kernelQueue = strategy.kernel_candidates.slice();
  }
  log(`Loaded carried state: EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT}, flags='${curFlags}', env='${curEnv}', ${headQueue.length} head + ${kernelQueue.length} kernel candidates.`);
}

// ===========================================================================
// PHASE: Config sweep (Config Tuner) — FIRST, reshapes the profile
// ===========================================================================
// A carried phase-partial state wins (already measured); kbSeedTput is 0 unless Module A adopted.
let curTput = ST.throughput || kbSeedTput || BASELINE_TPUT;
if (want('config') && CONFIG_TUNE_ENABLED && strategy && (strategy.config_directions || []).length &&
    !searchClosed('ConfigSweep')) {
  phase('ConfigSweep');
  const sweep = await safeAgent(
    roleAgent('config_tuner', 'sweep', 'Sweep the ranked config axes one at a time; keep wins.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, BASELINE_THROUGHPUT: BASELINE_TPUT,
      NOISE_BAND_PCT: NOISE_BAND, CONFIG_DIRECTIONS: strategy.config_directions,
      CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
      MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR, ...KB_REF_INPUTS,
    }),
    { phase: 'ConfigSweep', label: 'config_tuner:sweep', schema: SWEEP_SCHEMA });
  if (sweep && sweep.best_throughput_tok_s > curTput) {
    curFlags = sweep.accepted_flags || curFlags;
    curEnv = sweep.accepted_env || curEnv;
    curTput = sweep.best_throughput_tok_s;
    await requireE2EValidationCheckpoint('config/e2e_validation.json', {
      phase: 'ConfigSweep', validation_level: 'config_sweep', gate: 'accepted',
      validation_status: 'accepted_config',
      baseline_throughput_tok_s: BASELINE_TPUT, final_throughput_tok_s: curTput,
      throughput_speedup: BASELINE_TPUT ? curTput / BASELINE_TPUT : 1,
      baseline_config: { flags: INIT_FLAGS, env: INIT_ENV },
      accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
      accepted_kernels: [], accepted_heads: [], final_patch: [], final_overlay: { path: curOverlay },
      final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
        workload: WORKLOAD, source_artifact: `${EVAL_DIR}/config/sweep_results.json`,
        acceptance: { gain_exceeds_noise: true, correctness_passed: true, noise_floor_pct: NOISE_BAND } },
      stack: { config_layers: (sweep.trials || []).filter((t) => t && t.kept === true) },
      replay: { asset_paths: [`${EVAL_DIR}/bench_e2e.sh`, `${EVAL_DIR}/config/sweep_results.json`] },
      integrity: { checkpoint_assets: [] },
    });
    log(`Config sweep accepted. throughput ${curTput} tok/s (${(curTput / BASELINE_TPUT).toFixed(3)}x). Re-profiling.`);
    if (!searchClosed('post-config re-profile and re-strategize')) {
      // Re-profile: config changed which kernels dominate.
      profile = await safeAgent(
        roleAgent('profiler', 'reprofile', 'Re-profile after the config sweep.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 'config',
          OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Profile', label: 'profiler:post-config', schema: PROFILE_SCHEMA });
      // Re-strategize the kernel queue against the new profile.
      const restrat = await safeAgent(
        roleAgent('system_architect', 'strategize', 'Re-route after config changed the landscape.', {
          EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: curTput,
          WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED: false, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Strategize', label: 'architect:re-strategize', schema: STRATEGY_SCHEMA });
      if (restrat && restrat.kernel_candidates) kernelQueue = restrat.kernel_candidates.slice();
      if (restrat && restrat.head_candidates)
        headQueue = applyOpIdentityGuard(restrat.head_candidates.slice(), 're-strategize');
      // re-strategize may have (re)routed flydsl -> provision it (idempotent; no-op if already done).
      await ensureFlydslGate();
    }
  } else {
    log(`Config sweep found no win above the noise band.`);
  }
}

// Shared state for the head + kernel tracks (carried across phase invocations via args.state);
// declared before the HeadKernel block that uses it (temporal dead zone).
let dispatched = 0;                        // counts ONLY kernel-optimization tasks (the budget)
let milestone = 0;
let noImprove = 0;
// kbSeedKernels: stored kernels Module A replayed and this run's own A/B accepted.
const acceptedKernels = (ST.accepted_kernels || []).slice().concat(kbSeedKernels);
const acceptedHeads = (ST.accepted_heads || []).slice();
// Verified-isolated wins whose e2e A/B did not complete. Not rejections: Finalize finishes them
// (Fix C) and the return surfaces them as pending_integrations.
const pendingIntegrations = (ST.pending_integrations || []).slice();
const flaggedHeads = (ST.flagged_heads || []).concat(PRE_FLAGGED_HEADS);   // heads that could NOT be optimized (loudly surfaced, never silently skipped)
let headDispatched = 0;
const history = ST.history || { insights: [], ledger: [], milestones: [], bottleneck_now: '', suggest_next: '' };

// A fused op (op_kind='moe', set by the op-identity guard OR the Architect) is extracted AS the fused op,
// never decomposed into a standalone dense GEMM — so dense-GEMM synth is off for it.
function gemmSynthFor(h) { return (h && h.op_kind === 'moe') ? 'false' : GEMM_SYNTH; }

// ===========================================================================
// PHASE: TuningSkillset — the VENDORED tuning skillset, run WHOLE and STANDALONE, BEFORE HeadKernel.
// TUNING_SKILLSET_DIR is one hash-pinned tree (scripts/tuning_skillset_sync.py). tuning_specialist
// runs its full loop with its own pre/post A/B, so the head track then works on what is still hot,
// against a tuned reference. An accept folds into curFlags/curEnv and the profile is re-taken.
// ===========================================================================
if (want('tune') && TUNING_SKILLSET_ENABLED && !searchClosed('TuningSkillset') && !tuningWindowShort()) {
  phase('TuningSkillset');
  log(`Tuning skillset: ${TUNING_SKILLSET_DIR} (whole, standalone, pre-HeadKernel); ` +
    `no op cap, tuning-kb ${TUNING_KB_ENABLED ? 'ENABLED' : 'DISABLED (blind eval)'}.`);
  tuning = await safeAgent(
    roleAgent('tuning_specialist', 'tune',
      'Tune the live stack with the skillset, measure your OWN in-session isolated-server pre/post A/B, ' +
      'prove engagement, and hand back a deploy bundle that reaches production through EVAL_DIR/final/.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD,
      BASELINE_THROUGHPUT: BASELINE_TPUT, CURRENT_THROUGHPUT: curTput,
      CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
      MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
      NOISE_BAND_PCT: NOISE_BAND, ACCURACY_GATE, ...ACCURACY_INPUTS,
      PROFILE_TOPN: profile ? profile.profile_topN_json : '',
      TUNING_TARGETS: (strategy && strategy.head_candidates) || headQueue || [],
      TUNING_SKILLSET_DIR, TUNING_KB_ENABLED, SKILL_DIR: WORKFLOW_DIR,
      // The always-fires channel (as for the Architect and config_tuner), gated like warmStartBlock()
      // so a blind eval stays blind through both channels.
      ...(TUNING_KB_ENABLED ? KB_REF_INPUTS : {}),
      ...(TUNING_KB_ENABLED && KB_CACHE_DIR ? { KB_CACHE_DIR } : {}),
      // The read side of the tuned-table write below (kernel store, keyed per op/backend/gfx); the
      // role queries per op itself. Read and write must name the same PLANE; a read takes exactly
      // one (kbResolveScript), so `both` reads `remote` with the local mirror as fallback.
      ...(TUNING_KB_ENABLED && KB_DIMS && KB_DIMS.gfx ? {
        TUNED_KB_PLANE: E2E_KB_PLANE === 'both' ? 'remote' : E2E_KB_PLANE,
        TUNED_KB_STORE: E2E_KB_STORE_DIR,
        TUNED_KB_GFX: KB_DIMS.gfx,
        // Pages are keyed on arch and op, not dtype; this filters the read to one dtype's rows
        // (empty offers every dtype).
        TUNED_KB_PRECISION: KB_DIMS.precision || '',
        TUNED_KB_SCRIPT: KERNEL_WF_DIR + '/scripts/experience_store.py',
        TUNED_KB_ENV_PRELUDE: KB_ENV_PRELUDE,
      } : {}),
    }),
    { phase: 'TuningSkillset', label: 'tuning_specialist:tune', schema: TUNING_SCHEMA });

  // An accept must clear every bar the skillset sets: proven engagement (an artifact never proven
  // live proves nothing) and a completed both-leg A/B (a post-only number is not a measurement).
  const tuned = tuning && tuning.gate === 'accepted';
  const tuneOk = tuned && tuning.engagement_verified === true && tuning.ab_complete !== false &&
    tuning.correctness_gate !== 'fail' && tuning.post_tune_throughput_tok_s > 0 &&
    tuning.post_tune_throughput_tok_s > (tuning.pre_tune_throughput_tok_s || 0);

  // Every op the skillset named: read by the banking (gated on `tuneOk`) and by the attestation
  // (not gated). `tuning &&` because safeAgent can return null.
  const tunedOps = ((tuning && tuning.ops_tuned) || [])
    .filter((o) => String(o.op || o.short_name || '').trim());

  // Attest recalled tuned ops (the write below files only wins). Outside `tuneOk`: a per-op verdict
  // stands even when the phase banks nothing. Before the write, which carries the ledger across its
  // rewrite (experience_store.py `_carry_remote`). Only recalled ops that reached the GPU (artifact,
  // engagement or a measurement) count: `recalls` counts attempts on hardware (kb/attest.py).
  const tuningAttestable = KB_DIMS && KB_DIMS.gfx ? tunedOps.filter((o) =>
    String(o.session_id || '').trim() &&
    /recall|kb|knowledge/i.test(String(o.source || o.origin || ''))) : [];
  const tuningRecalls = tuningAttestable.filter((o) =>
    o.engaged === true || Number(o.isolated_speedup) > 0 || String(o.artifact || '').trim());
  const tuningUnattempted = tuningAttestable.length - tuningRecalls.length;
  if (tuningUnattempted) {
    log(`[kernel-kb] ${tuningUnattempted} recalled op(s) NOT attested: no artifact, engagement, or ` +
      `measurement, so they never reached the GPU — an offer nobody benched is not evidence.`);
  }
  if (tuningRecalls.length) {
    const storeScript = KERNEL_WF_DIR + '/scripts/experience_store.py';
    // One plane per verdict (`both` would count one attempt twice): the plane that answered, as the
    // op reports it (the role retries a remote miss locally), else the requested one.
    const askedRemote = E2E_KB_PLANE !== 'local';
    const planeOf = (o) => {
      const said = String(o.read_plane || o.recall_plane || o.plane || '').trim().toLowerCase();
      return said === 'local' || said === 'remote' ? said : (askedRemote ? 'remote' : 'local');
    };
    const cmds = tuningRecalls.map((o) => {
      const sp = Number(o.isolated_speedup) || 0;
      const outcome = o.engaged !== true ? 'not_reproduced' : sp > 1.0 ? 'validated' : 'failed';
      const plane = planeOf(o);
      return `python3 ${shq(storeScript)} attest --plane ${plane} ` +
        (plane === 'local' && E2E_KB_STORE_DIR ? `--store ${shq(E2E_KB_STORE_DIR)} ` : '') +
        `--session-id ${shq(String(o.session_id).trim())} ` +
        `--kernel-name ${shq(String(o.op || o.short_name).trim())} ` +
        `--language ${shq(String(o.backend || 'tuned').trim())} --gfx ${shq(KB_DIMS.gfx)} ` +
        (KB_DIMS.framework_version ? `--framework-version ${shq(KB_DIMS.framework_version)} ` : '') +
        `--outcome ${outcome} --measured-speedup ${sp} ` +
        // `claimed_gate`: the skillset's own claim (an unproven accept is downgraded below this);
        // `banked` is recorded beside the outcome, not as a gate on it.
        `--note ${shq(`e2e tuning recall: engaged=${o.engaged === true}; banked=${!!tuneOk}` +
          `; claimed_gate=${tuning.gate}` +
          `${o.note ? '; ' + String(o.note) : ''}`.slice(0, 300))} ` +
        `--measured-by ${shq('e2e_workflow:tuning:' + BACKEND)} --apply || true`;
    });
    // The prelude authenticates the service; needed if ANY op attests remotely.
    const anyRemote = tuningRecalls.some((o) => planeOf(o) === 'remote');
    try {
      await safeAgent(
        `You are the tuning knowledge-base attestor. Run EXACTLY these commands in order and ` +
        `return {"ran": <how many you ran>, "note": "<anything that failed>"}. Each records what ` +
        `this box saw when it installed a RECALLED tuned artifact. Do NOT edit them, do NOT add ` +
        `or drop any, and do NOT retry a failure — a repeat would double-count the attempt.\n` +
        '```bash\n' + (anyRemote ? KB_ENV_PRELUDE + '\n' : '') + cmds.join('\n') + '\n```',
        { phase: 'TuningSkillset', label: 'kernel-kb:attest-tuned',
          schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) },
        1);
      const planes = tuningRecalls.map(planeOf);
      log(`[kernel-kb] attested ${tuningRecalls.length} recalled tuned op(s) ` +
        `(${planes.filter((p) => p === 'remote').length} remote, ` +
        `${planes.filter((p) => p === 'local').length} local; claimed gate=${tuning.gate}` +
        `${tuneOk ? '' : ', not banked by this phase'}).`);
    } catch (e) {
      log(`[kernel-kb] tuned attest failed (NON-FATAL): ${String(e).slice(0, 200)}`);
    }
  }

  if (tuneOk) {
    const tuningBaselineConfig = { flags: curFlags, env: curEnv };
    if (tuning.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + tuning.apply_env;
    if (tuning.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + tuning.apply_flags;
    // Carry the routing/enabling overlay forward as an accepted head patch does; otherwise the code
    // half of a tuning win is dropped and its tuned data binds to nothing.
    if (tuning.apply_overlay) curOverlay = tuning.apply_overlay;
    curTput = tuning.post_tune_throughput_tok_s;
    log(`Tuning skillset ACCEPTED: ${tuning.pre_tune_throughput_tok_s} -> ${curTput} tok/s ` +
      `(+${(tuning.tuning_delta_pct || 0).toFixed(2)}% attributable to tuning; ` +
      `${(tuning.ops_tuned || []).length} op(s), engagement proven` +
      `${tuning.apply_overlay ? `, carrying routing overlay ${tuning.apply_overlay}` : ''}). Re-profiling.`);
    // An unverified deploy bundle does not withhold the accept (the in-run numbers stay valid); it
    // risks DELIVERY (final_launch.sh reproducing less). Finalize re-checks; warn early here too.
    if (tuning.deploy_verified !== true) {
      log(`WARNING: tuning accepted WITHOUT a verified deploy bundle (deploy_bundle=` +
        `${tuning.deploy_bundle || '<none>'}). The in-run measurements stand, but this win may not ` +
        `reproduce from EVAL_DIR/final/ — Finalize will re-check and report tuning_in_bundle.`);
    }
    history.ledger.push({
      direction: 'tuning_skillset', verdict: 'confirmed',
      e2e_delta_pct: tuning.tuning_delta_pct || 0,
      lesson: `tuning skillset (standalone, pre-head): ${tuning.summary || 'accepted'}`,
    });

    // Bank each tuned op as an accepted kernel, deterministically here rather than via Finalize's
    // accepted_kernels, so the KB record carries the lever itself and not just the launch script.
    for (const o of tunedOps) {
      bankAccepted(acceptedKernels, {
        short_name: String(o.op || o.short_name).trim(),
        backend: o.backend || '',
        // A tuned table is applied through env, a first-class winner_kind in the store.
        kind: 'env',
        isolated: Number(o.isolated_speedup) || 0,
        // One A/B covers all ops: the delta is per-op only when there is one op; otherwise 0 (the
        // phase figure is tuning_skillset.tuning_delta_pct) so consumers that sum do not double-count.
        e2e_delta_pct: tunedOps.length === 1 ? (tuning.tuning_delta_pct || 0) : 0,
        apply_env: tuning.apply_env || '',
        apply_flags: tuning.apply_flags || '',
        patch: '',
        engaged: o.engaged === true,
        from_tuning_skillset: true,
        tuning_artifact: o.artifact || '',
        // SEARCHED here or RECALLED: a recall re-banked as a discovery would look like an
        // independent confirmation to anyone counting records.
        tuning_source: String(o.source || o.origin || '').trim() || 'search',
      }, {});
    }
    if (tunedOps.length) {
      const recalled = tunedOps.filter((o) => /recall|kb|knowledge/i.test(String(o.source || o.origin || ''))).length;
      log(`Banked ${tunedOps.length} tuned op(s) as accepted kernels (kind=env) so the KB record ` +
        `carries the lever: ${tunedOps.map((o) => o.op || o.short_name).join(', ')}` +
        `${recalled ? ` (${recalled} recalled from the KB rather than searched)` : ''}.`);
    }

    // ---- Write each tuned op into the KERNEL lane's experience store, one entry per op ----
    // Unlike the end-of-run e2e record (whole deployment, gated on final throughput), this ranks on
    // isolated per-op speedup per (kernel, language, gfx), so the tuned table survives the run's
    // verdict. Filed only if tuneOk, and per op only if `isolated_speedup > 1.0` and `engaged`.
    // Best-effort: a failure loses a record, never a measurement.
    const kernelKbOps = KB_DIMS && KB_DIMS.gfx ? tunedOps.filter((o) =>
      Number(o.isolated_speedup) > 1.0 && o.engaged === true && String(o.artifact || '').trim()) : [];
    if (kernelKbOps.length) {
      const storeScript = KERNEL_WF_DIR + '/scripts/experience_store.py';
      // `both` mirrors to the shared service; without a store dir it degrades to the plain write
      // (as at the kernel lane's own write site).
      const remoteOn = E2E_KB_PLANE !== 'local' && !!E2E_KB_STORE_DIR;
      // `--precision` and `--serving-*` are recorded in `value.upstream`, not keyed; precision lets
      // the reader's `--precision` filter drop another dtype's table before it skews the ranking.
      const cmds = kernelKbOps.map((o) => {
        const op = String(o.op || o.short_name).trim();
        return `python3 ${shq(storeScript)} ${remoteOn
            ? `write-remote --plane both --store ${shq(E2E_KB_STORE_DIR)}`
            : 'write'} \\
  --root ${shq(KB_ARTIFACTS_DIR)} --kernel-name ${shq(op)} \\
  --language ${shq(String(o.backend || 'tuned').trim())} --gfx ${shq(KB_DIMS.gfx)} \\
  --kernel-class tuning --speedup ${Number(o.isolated_speedup)} \\
  --carrier tuned_artifact --artifact ${shq(String(o.artifact).trim())} \\
  --tuner ${shq(String(o.tuner || '').trim())} --apply-env ${shq(tuning.apply_env || '')} \\
  --cache-invalidation ${shq((tuning.cache_invalidation || []).join(' && '))} \\
  --metric-kind tuning_isolated --case-names ${shq(String(o.shapes || '').replace(/,/g, ';'))} \\
  --precision ${shq(KB_DIMS.precision || '')} --serving-framework ${shq(BACKEND)} \\
  --serving-framework-version ${shq(KB_DIMS.framework_version || '')} \\
  --direction ${shq('tuning-' + (String(o.backend || 'op').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-')))} \\
  --eval-dir ${shq(EVAL_DIR)}${tuning.report_path ? ` --report ${shq(tuning.report_path)}` : ''}`;
      });
      try {
        const kw = await safeAgent(
          `You are the tuning experience writer. Run EACH of the commands below EXACTLY as written, ` +
          `in order. Each applies its own gates and prints one line of JSON; collect those lines in ` +
          `order into \`results\`. Do NOT edit a command, do NOT add or drop flags, and do NOT retry ` +
          `one that fails` + (remoteOn
            ? ` — these may reach the shared KB Store service, and every write it accepts is ` +
              `PERMANENT (it exposes no delete), so a retry creates a second permanent record ` +
              `instead of fixing the first`
            : '') + `. For a command that errors, put ` +
          `{"written": false, "reason": "io_error"} in its place so the list stays aligned with the ` +
          `command order.\n\n` +
          `THEN, last, write \`{"results": [...]}\` — the same list — to ` +
          `\`${EVAL_DIR}/tuning/kb_write_tuned.json\`. That file is the receipt saying this filing ` +
          `already happened: run_e2e.py refiles these ops from disk when the workflow is killed ` +
          `before reaching this step, and it skips when the receipt is present. Without it, a run ` +
          `that dies AFTER this step gets every op written a second time, and the store counts the ` +
          `copy as an independent reproduction — which can promote a candidate on one measurement ` +
          `seen twice. Write the receipt even if every command failed.\n` +
          '```bash\n' + (remoteOn ? KB_ENV_PRELUDE + '\n' : '') + cmds.join('\n\n') + '\n```',
          { phase: 'TuningSkillset', label: 'kernel-kb:write-tuned',
            schema: obj({ results: arrObj }, ['results']) },
          1);
        const rows = (kw && kw.results) || [];
        const wrote = rows.filter((r) => r && r.written).length;
        const dup = rows.filter((r) => r && !r.written && /duplicate/.test(String(r.reason || ''))).length;
        log(`[kernel-kb] tuned ops filed as isolated wins: ${wrote}/${kernelKbOps.length} written` +
          `${dup ? `, ${dup} already known (counted as reproductions)` : ''}` +
          `${rows.filter((r) => r && !r.written && !/duplicate/.test(String(r.reason || '')))
            .map((r) => ` [${r.reason}]`).join('')}. These survive the run's own e2e verdict.`);
      } catch (e) {
        log(`[kernel-kb] tuned-op write failed (NON-FATAL — the tuning result itself is unaffected): ` +
          `${String(e).slice(0, 200)}`);
      }
    } else if (tunedOps.length && (!KB_DIMS || !KB_DIMS.gfx)) {
      log(`[kernel-kb] tuned ops NOT filed: no gfx established, and an arch-less entry is ` +
        `unattributable (a tuned table is valid for exactly one arch).`);
    }
    await requireE2EValidationCheckpoint('tuning/e2e_validation.json', {
      phase: 'TuningSkillset', validation_level: 'tuning_skillset', gate: 'accepted',
      validation_status: 'accepted_tuning',
      baseline_throughput_tok_s: tuning.pre_tune_throughput_tok_s,
      final_throughput_tok_s: tuning.post_tune_throughput_tok_s,
      throughput_speedup: tuning.tuning_speedup ||
        (tuning.post_tune_throughput_tok_s / tuning.pre_tune_throughput_tok_s),
      baseline_config: tuningBaselineConfig,
      accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
      accepted_kernels: tunedOps.map((o) => ({
        kernel_id: o.kernel_id || o.op || o.short_name || '',
        kernel_slot: o.kernel_slot || o.target_callable || o.target_file || o.op || o.short_name || '',
        short_name: o.op || o.short_name || '', backend: o.backend || 'geak',
        from_tuning_skillset: true, isolated: o.isolated_speedup || 0,
      })),
      accepted_heads: [], final_patch: [], final_overlay: { path: curOverlay },
      final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
        workload: WORKLOAD, source_artifact: `${EVAL_DIR}/tuning/ab/ab_summary.json`,
        acceptance: { gain_exceeds_noise: true, correctness_passed: tuning.correctness_gate !== 'fail',
          noise_floor_pct: tuning.noise_floor_pct || NOISE_BAND } },
      stack: { kernel_slots: tunedOps.map((o) => ({
        kernel_slot: o.kernel_slot || o.target_callable || o.target_file || o.op || o.short_name || '',
        selected: true,
      })), deployment_layers: [tuning.deploy_bundle].filter(Boolean) },
      replay: { asset_paths: [tuning.deploy_bundle, ...(tuning.artifacts || [])].filter(Boolean),
        cache_invalidation: tuning.cache_invalidation || [], requires_server_restart: true },
      integrity: { checkpoint_assets: [] },
      tuning_skillset: tuning,
    });

    // Tuning changed which kernels dominate: re-profile + re-strategize, as after ConfigSweep.
    if (!searchClosed('post-tuning re-profile and re-strategize')) {
      profile = await safeAgent(
        roleAgent('profiler', 'reprofile', 'Re-profile after the tuning skillset deployed its artifacts.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 'tuning',
          // curOverlay, not '': without a banked routing overlay the tuned artifact binds to nothing.
          OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Profile', label: 'profiler:post-tuning', schema: PROFILE_SCHEMA });
      const retune = await safeAgent(
        roleAgent('system_architect', 'strategize', 'Re-route after tuning changed the landscape.', {
          EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: curTput,
          WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED: false, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Strategize', label: 'architect:post-tuning-strategize', schema: STRATEGY_SCHEMA });
      if (retune && retune.kernel_candidates) kernelQueue = retune.kernel_candidates.slice();
      if (retune && retune.head_candidates)
        headQueue = applyOpIdentityGuard(retune.head_candidates.slice(), 'post-tuning re-strategize');
      if (retune) await ensureFlydslGate();
    }
  } else if (tuned) {
    // Claimed accepted but failed a hard bar: not banked (an unproven artifact poisons later A/Bs).
    log(`Tuning skillset claimed ACCEPTED but did not clear the bar ` +
      `(engagement_verified=${tuning.engagement_verified}, ab_complete=${tuning.ab_complete}, ` +
      `correctness=${tuning.correctness_gate}, pre=${tuning.pre_tune_throughput_tok_s}, ` +
      `post=${tuning.post_tune_throughput_tok_s}) — NOT banked; continuing on the pre-tuning config.`);
    history.ledger.push({ direction: 'tuning_skillset', verdict: 'flagged',
      lesson: 'tuning claimed a win it could not prove (engagement/A-B/correctness) — not banked' });
    tuning = { ...tuning, gate: 'no_win', reason: (tuning.reason || '') + ' [orchestrator: accept withheld — unproven]' };
  } else {
    log(`Tuning skillset produced no banked win (gate=${tuning ? tuning.gate : 'null/degraded'}` +
      `${tuning && tuning.reason ? `: ${tuning.reason}` : ''}).`);
    history.ledger.push({ direction: 'tuning_skillset', verdict: tuning && tuning.gate === 'skipped' ? 'skipped' : 'dead_end',
      lesson: (tuning && (tuning.reason || tuning.summary)) || 'tuning phase produced no result' });
  }
} else if (want('tune') && !TUNING_SKILLSET_ENABLED) {
  log('Tuning skillset DISABLED (tuning_skillset=false) — skipping the phase entirely.');
}
// Report inputs for the tuning phase; {} when off/absent, so the Report prompt is unchanged.
const TUNING_REPORT_INPUTS = (TUNING_SKILLSET_ENABLED && tuning) ? { TUNING_RESULT: tuning } : {};
// Finalize inputs: a tuned DATA artifact cannot ride the PYTHONPATH overlay, so Finalize folds this
// bundle into EVAL_DIR/final/ (diff, files, deploy.sh before server start). {} unless tuning banked.
const TUNING_FINALIZE_INPUTS = (TUNING_SKILLSET_ENABLED && tuning && tuning.gate === 'accepted')
  ? {
    TUNING_DEPLOY_BUNDLE: tuning.deploy_bundle || `${EVAL_DIR}/tuning/deploy`,
    TUNING_APPLY_ENV: tuning.apply_env || '',
    TUNING_CACHE_INVALIDATION: tuning.cache_invalidation || [],
    TUNING_ARTIFACTS: tuning.artifacts || [],
    TUNING_LIVE_TREE_FILES: tuning.live_tree_files || [],
    TUNING_OVERLAY: tuning.apply_overlay || '',
  }
  : {};

// The live-tree carve-out for EVERY integrate leg (roles/e2e_integrator.md, "never mutate
// site-packages"): an accepted tuning deploy legitimately writes into the installed tree, so these
// paths are exempt from the clean-tree check (otherwise a leg "restores" them and drops the tuning).
// A function, so it reflects a downgrade to no_win and a resumed run's carried state; {} otherwise.
function tuningIntegrateInputs() {
  if (!(TUNING_SKILLSET_ENABLED && tuning && tuning.gate === 'accepted')) return {};
  return {
    TUNING_LIVE_TREE_FILES: tuning.live_tree_files || [],
    TUNING_DEPLOY_BUNDLE: tuning.deploy_bundle || `${EVAL_DIR}/tuning/deploy`,
  };
}

// ===========================================================================
// PHASE: HeadKernel — the highest-pct_gpu_time ops (GEMM / attention), optimized regardless of the
// edit flag, via the bake-off ladder. Runs BEFORE the editable-kernel loop. The op-identity guard has
// already forced fused/monolithic heads to op_kind=moe (dense-GEMM synth off, live seam kept), so
// each head is optimized AS the op the live kernel dispatches, never as an un-integrable dense GEMM.
// ===========================================================================
if (want('head') && headQueue.length && HEAD_BUDGET > 0) {
  phase('HeadKernel');
  log(`Head-kernel track: ${headQueue.length} candidate op(s), head_budget=${HEAD_BUDGET}, ${KERNEL_EXTRACT_ON ? 'prepared UT inventory' : `threshold=${HEAD_THRESHOLD_PCT}%`}.`);
  // Preserve the prepared inventory order for AgentX, Architect order for fixed-load runs.
  const heads = headQueue.slice(0, HEAD_BUDGET).map((c, i) => ({
    ...c, idx: i, gpu_id: GPU_LIST[i % GPU_LIST.length],
    short_name: c.short_name || `${c.op_kind || 'op'}${i}`,
  }));
  if (DEEP) {
    // ==== DEEP-MODE: global cross-kernel × cross-backend co-optimization ====
    // One lane pool over ALL (head op × backend) lanes; a serial e2e gate overlapping co-opt; full
    // backend roster with ceiling-aware patience + revive; per-op SHARED_KB + run-global GLOBAL_KB;
    // budget-driven EV scheduling toward DEEP_E2E_TARGET.
    const ROOFLINE_SCHEMA = { type: 'object', additionalProperties: true, properties: { roofline_note: { type: 'string' }, target_geomean: { type: 'number' } }, required: ['roofline_note'] };
    const OKV = { type: 'object', additionalProperties: true, properties: { ok: { type: 'boolean' }, summary: { type: 'string' }, feedback_path: { type: 'string' }, addendum_path: { type: 'string' } }, required: ['ok'] };
    const HARVEST_SCHEMA = { type: 'object', additionalProperties: true, properties: { lanes: { type: 'array', items: { type: 'object', additionalProperties: true, properties: { uid: { type: 'string' }, has_state: { type: 'boolean' }, cumulative: { type: 'number' }, best_ms: { type: 'number' }, vs_live: { type: 'number' }, eval_dir: { type: 'string' }, patch: { type: 'string' } }, required: ['uid'] } } }, required: ['lanes'] };

    // ---- GPU partition (N-adaptive, elastic; serial gate) ----
    // 'main' = cards outside the serving slot (never paused); 'serve' = the serving cards, used for
    // co-opt only in waves with no due gate. A gate wave runs the e2e A/B on the serving slot while
    // 'main' keeps co-optimizing; with exactly TP cards 'main' is empty and the gate runs alone.
    const servingCards = SERVING_GPU.split(',').map(s => s.trim()).filter(Boolean);
    const servingSet = new Set(servingCards);
    const dedicatedCoopt = GPU_LIST.filter(g => !servingSet.has(g));
    const haveSpare = dedicatedCoopt.length > 0;
    const cooptMain = haveSpare ? makeSem(dedicatedCoopt) : null;          // dedicated cards (never paused)
    const cooptServe = makeSem(haveSpare ? servingCards : GPU_LIST);       // serving cards (idle during a gate wave)
    const mainSlots = haveSpare ? dedicatedCoopt.length : 0;
    const serveSlots = haveSpare ? servingCards.length : GPU_LIST.length;
    log(`[deep] partition: serving {${servingCards.join(',')}} TP=${SERVING_TP}; dedicated co-opt {${dedicatedCoopt.join(',') || '(none)'}}; mode=${haveSpare ? 'OVERLAP gate||co-opt' : 'TIME-SLICE (N==TP)'}; budget ${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h; e2e target ×${DEEP_E2E_TARGET} (${Math.round(BASELINE_TPUT * DEEP_E2E_TARGET)} tok/s).`);

    // ---- ceiling prior: by lane ROLE, not model/shape ----
    const ceilingPrior = (lang, mode) => (mode === 'author' ? 2.2 : (lang === 'triton' && mode === 'optimize' ? 1.6 : 1.8));

    // ---- per-head prep: extract + bake-off + roofline + lane roster (cheap agents on GPU_LIST[0]) ----
    const GLOBAL_KB = `${EVAL_DIR}/deep_head/GLOBAL_KB.md`;
    const prepHead = async (h) => {
      if (KERNEL_EXTRACT_ON) headQueue = headQueue.filter(c => c.prepared_ut !== h.prepared_ut);
      const ext = await extractWithBaseline(
        'kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, KERNEL: h, GEMM_SYNTH: gemmSynthFor(h),
          ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
          CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          REQUIRE_DECODE_BUCKET: true, DECODE_M_BUCKETS: [1, CONC],
          PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
        },
        { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (!ext || ext.smoke !== 'pass' || !ext.task_dir) {
        const why = extractFailWhy(ext);
        log(`  [deep] ${h.short_name}: op extraction failed (${why})${isDominant ? ' [DOMINANT — flagged]' : ''}; skipping.`);
        if (isDominant) flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'extract', gate: 'extract_failed', reason: why });
        history.ledger.push({ direction: h.short_name, verdict: isDominant ? 'flagged' : 'dead_end', lesson: `op extraction failed (${why})` });
        return null;
      }
      const bake = await safeAgent(
        roleAgent('op_benchmarker', 'bakeoff', 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.', {
          EVAL_DIR, OP_TASK_DIR: ext.task_dir, OP_KIND: ext.op_kind, PCT_GPU_TIME: h.pct_gpu_time,
          CANDIDATE_BACKENDS: ext.candidate_backends || h.candidate_backends || [],
          GPU_ID: GPU_LIST[0], ENABLE_FP8, KERNEL_WF_DIR, KERNEL_BUDGET: DEEP_WAVE_BUDGET, SKILL_DIR: WORKFLOW_DIR,
        }),
        { phase: 'HeadKernel', label: `bakeoff ${h.short_name}`, schema: OPBENCH_SCHEMA });
      // Lane roster: always tune the live kernel, author EVERY backend the bake-off proposes, then add
      // distinct authoring directions (generic steers, no magic numbers).
      let lanesSpec = [];
      if (DEEP_BACKENDS_OVERRIDE) {
        lanesSpec = DEEP_BACKENDS_OVERRIDE.split(',').map(s => s.trim()).filter(Boolean).map(s => {
          const [lang, mode] = s.split(':'); const L = (lang || '').trim();
          return { key: L, lang: L, mode: (mode || 'author').trim(), steer: '' };
        }).filter(b => b.lang);
      } else {
        const planLangs = (bake && Array.isArray(bake.author_plan) ? bake.author_plan : [])
          .map(ap => ({ lang: (ap.language || '').trim().toLowerCase(), mode: ap.route === 'rewrite' ? 'optimize' : 'author' }))
          .filter(x => x.lang);
        const liveLang = (ext.live_backend || 'triton').toLowerCase();
        const otherLangs = [...new Set(planLangs.map(x => x.lang).filter(l => l && l !== liveLang))];
        lanesSpec = [{ key: `${liveLang}-opt`, lang: liveLang, mode: 'optimize',
          steer: ` DIRECTION=tile-tune: per-shape AUTOTUNE the live ${liveLang} kernel (block sizes, warps, stages, scheduling, grid swizzle; split-K for small-M decode). Stay graph-capture-safe.` }];
        for (const l of otherLangs) lanesSpec.push({ key: l, lang: l, mode: (planLangs.find(x => x.lang === l) || {}).mode || 'author',
          steer: ` AUTHOR a ${l} implementation that beats the LIVE kernel (not just your own first port); read SHARED_KB + GLOBAL_KB and borrow the winning decomposition other lanes/kernels found.` });
        const extra = [
          { key: `${liveLang}-fused`, lang: liveLang, mode: 'author', steer: ` DIRECTION=fused-author: author a fresh single-pass FUSED kernel (fold pre/post ops + scaling into the main ${MATRIX_CORE_NAME} core; epilogue-fuse activation). Beat the LIVE kernel.` },
          { key: `${liveLang}-splitk`, lang: liveLang, mode: 'author', steer: ' DIRECTION=split-K: author a split-K + accumulate variant for the large-M prefill shapes, with a per-shape launch selector that uses the non-split path for small-M decode.' },
          { key: `${liveLang}-deep`, lang: liveLang, mode: 'optimize', steer: ' DIRECTION=deep-explore: combine persistent kernel + epilogue fusion + grid swizzle + aggressive tiling in one coherent rewrite; push toward the roofline SOTA bar.' },
        ];
        for (const t of extra) lanesSpec.push(t);   // global pool — no per-card truncation here
      }
      const liveBaselineMs = (bake && Number.isFinite(bake.best_known_ms) && bake.best_known_ms > 0) ? bake.best_known_ms : 0;
      const deepDir = `${EVAL_DIR}/deep_head/${h.short_name}`;
      const sharedKb = `${deepDir}/SHARED_KB.md`;
      const opSpec = { op_kind: ext.op_kind, shapes: ext.shapes || {}, dtype: ext.dtype || 'bf16', regime: h.regime || 'both', cuda_graph_safe: true, ...(ext.workload_path ? { workload_path: ext.workload_path } : {}) };
      const anchor = await safeAgent(
        `You are the ROOFLINE ANCHOR + shared-KB bootstrapper for DEEP cross-backend optimization of head op ${h.short_name} (${ext.op_kind}). ` +
        `Inputs: OP_TASK_DIR=${ext.task_dir}; shapes=${JSON.stringify(ext.shapes || {})}; dtype=${ext.dtype || '?'}; read ${EVAL_DIR}/env_report.json for the on-box device peak (FLOP/s + HBM bandwidth). ` +
        `DO: (a) mkdir -p ${deepDir}; (b) compute the ROOFLINE ceiling per case (compute- vs memory-bound, target ms/case + an overall SOTA geomean ~80-90% of roofline); ` +
        `(c) bootstrap ${sharedKb} (markdown) with sections: Roofline target; Current best per backend (table backend|best geomean|technique|wave — empty now); Techniques that WORK (technique -> measured effect -> source); Dead-ends (scoped, evidence); Cross-backend assignments (borrow); Open hypotheses. ` +
        (E2E_LEARNED_KB_ENABLED
          ? `Cite relevant ${KERNEL_KNOWLEDGE_DIR} cards (read INDEX in knowledge/learned/ first) for ${ext.op_kind}. `
          : `E2E_LEARNED_KB=off: do not read or cite knowledge/learned/; derive the bootstrap from this run's measurements only. `) +
        `Return {roofline_note, target_geomean}.`,
        { phase: 'HeadKernel', label: `roofline ${h.short_name}`, schema: ROOFLINE_SCHEMA });
      const rooflineTarget = anchor && Number.isFinite(anchor.target_geomean) ? anchor.target_geomean : 0;
      const lanes = lanesSpec.map((b) => ({
        uid: `${h.short_name}::${b.key || b.lang}`, key: b.key || b.lang, lang: b.lang, mode: b.mode, steer: b.steer || '',
        state_dir: `${deepDir}/state/${b.key || b.lang}`, best: 1.0, noImprove: 0, active: true, ran: 0, lastEval: '', patch: '',
        ceiling: Math.max(ceilingPrior(b.lang, b.mode), rooflineTarget || 0),
        head: h, ext, deepDir, sharedKb, opSpec, liveBaselineMs, rooflineTarget,
      }));
      log(`[deep] ${h.short_name} (${(h.pct_gpu_time || 0).toFixed(1)}% GPU): ${lanes.length} lanes [${lanes.map(l => l.key + ':' + l.mode).join(', ')}]; roofline×${(rooflineTarget || 0).toFixed(2)}.`);
      return { h, ext, lanes, deepDir, sharedKb, liveBaselineMs };
    };

    const dHeads = heads.slice().sort((a, b) => (b.pct_gpu_time || 0) - (a.pct_gpu_time || 0));   // dominant-Amdahl first
    log(`[deep] cross-kernel × cross-backend co-opt over ${dHeads.length} head op(s); GPU pool {${GPU_LIST.join(',')}}.`);
    headDispatched += dHeads.length;
    const preps = [];
    for (const h of dHeads) { if (DEEP_DEADLINE_HIT) break; const p = await prepHead(h); if (p) preps.push(p); }
    const allLanes = [];
    for (const p of preps) for (const l of p.lanes) allLanes.push(l);
    if (!allLanes.length) { log('[deep] no viable head lanes; nothing to optimize.'); }

    // ---- EV: Amdahl mass × remaining ceiling gap × recent improvement (exploration floor) ----
    const evOf = (l) => {
      const amdahl = Math.max(0.01, (l.head.pct_gpu_time || 1) / 100);
      const gap = Math.max(0.02, (l.ceiling || 1.5) - l.best);
      const rate = l.ran === 0 ? 0.6 : Math.max(0.03, (l.lastGain || 0));   // unrun lanes get an exploration bonus
      return amdahl * gap * rate;
    };

    // ---- batched HARVEST of lanes from DISK truth (immune to a nulled burst) ----
    const harvestLanes = async (set, tag) => {
      if (!set.length) return;
      const byHead = {}; for (const l of set) (byHead[l.head.short_name] = byHead[l.head.short_name] || []).push(l);
      for (const sn of Object.keys(byHead)) {
        const ls = byHead[sn]; const base = ls[0].liveBaselineMs;
        const metric = KERNEL_EXTRACT_ON
          ? 'Use the prepared UT workload and frozen baseline. vs_live is the recorded workload-weighted speedup (sum of baseline_ms * effective_calls divided by sum of candidate_ms * effective_calls); best_ms is diagnostic only. Missing weighted evidence gives vs_live=1, never a scalar geomean fallback.'
          : `LIVE baseline geomean = ${base || '?'} ms. best_ms is the absolute geomean across cases; vs_live=${base || 0}/best_ms when both>0, else 1.0.`;
        const harvest = await safeAgent(
          `Deep-v2 co-opt harvest [${tag}] for head ${sn} (${ls[0].ext.op_kind}). ${metric} A lane's own cumulative gain is self-relative, not vs-live. Read DISK, do not guess. Lanes:\n` +
          ls.map(l => `- uid=${l.uid}: STATE.json=${l.state_dir}/STATE.json; cumulative-best workspace=${l.state_dir}/best; newest finished run under ${l.deepDir}/runs/${l.key}/team_*/*/ with a final_patch.diff (+ baseline_timing.json / tech_lead_report.md for the optimized per-case ms)`).join('\n') + '\n' +
          `For EACH uid return: uid; has_state (STATE.json AND best/kernel_src exist); best_ms and vs_live using the metric above; cumulative (self-relative, ref); eval_dir (newest runs/<key>/team_*/<op> with non-empty final_patch.diff, else ""); patch (that diff path, else ""). Return {lanes:[{uid,has_state,best_ms,vs_live,cumulative,eval_dir,patch}]}.`,
          { phase: 'HeadKernel', label: `harvest ${sn} ${tag}`, schema: HARVEST_SCHEMA });
        const hmap = {}; for (const e of (harvest && Array.isArray(harvest.lanes) ? harvest.lanes : [])) hmap[e.uid] = e;
        const anyState = Object.values(hmap).some(e => e && e.has_state);
        for (const l of ls) {
          const e = hmap[l.uid]; if (!e) continue;
          const g = Number.isFinite(e.vs_live) && e.vs_live > 0 ? e.vs_live : (Number.isFinite(e.cumulative) && !base ? e.cumulative : null);
          if (g != null && g > l.best * 1.001) { l.lastGain = g - l.best; l.best = g; l.noImprove = 0; }
          else if (l.ran > 0) { l.lastGain = 0; l.noImprove++; }
          if (e.eval_dir) l.lastEval = e.eval_dir;
          if (e.patch) l.patch = e.patch;
          if (e.has_state) l.noStateStreak = 0;   // produced a result -> not infeasible
          // ceiling-aware patience: lanes still far from their ceiling get MORE waves before parking.
          const farFromCeiling = (l.ceiling - l.best) > 0.3 * Math.max(l.ceiling, 1e-9);
          const streakCap = farFromCeiling ? DEEP_PLATEAU_STREAK_HIGH : DEEP_PLATEAU_STREAK;
          if (e.has_state === false && anyState && l.ran > 0) {
            l.active = false; l.noStateStreak = (l.noStateStreak || 0) + 1;
            // 2 consecutive bursts with no persisted result while peers produced => infeasible
            // backend×op (no primitive for it); dead lanes are never reseeded or revived.
            if (l.noStateStreak >= 2) { l.dead = true; log(`  [deep] DEAD ${l.uid} \u2014 infeasible (no result ${l.noStateStreak}x while peers produced); won't reseed/revive.`); }
            else log(`  [deep] park ${l.uid} \u2014 no persisted result while peers produced.`);
          }
          else if (l.noImprove >= streakCap) { l.active = false; log(`  [deep] park ${l.uid} (plateau ${l.best.toFixed(3)}x, ceiling ${l.ceiling.toFixed(2)}x).`); }
        }
      }
    };

    // ---- CURATE per-op SHARED_KB + run-global GLOBAL_KB; REVIVE high-ceiling parked lanes ----
    const curateAndRevive = async (tag) => {
      for (const p of preps) {
        const ls = allLanes.filter(l => l.head.short_name === p.h.short_name && (l.lastEval || l.ran > 0));
        if (!ls.length) continue;
        await safeAgent(
          `KB CURATOR [deep ${tag}] for ${p.h.short_name} (${p.lanes[0] ? p.lanes[0].ext.op_kind : ''}). Per-lane vs-live best: ${JSON.stringify(ls.map(l => ({ uid: l.uid, lang: l.lang, vs_live: l.best, active: l.active, eval_dir: l.lastEval })))}. ` +
          `For each lane with an eval_dir, READ its insight_log.md + tech_lead_report.md; extract what WORKED (measured), what FAILED (scoped dead-end), and any technique another lane should borrow. ` +
          `REWRITE ${p.sharedKb} (keep sections; "Current best per backend" one row per lane; every technique a MEASURED effect + SOURCE; disproven -> Dead-ends; fill "Cross-backend assignments (borrow)" with concrete "lane X found Y -> lane Z try W"). ` +
          `ALSO append cross-KERNEL transferable techniques to ${GLOBAL_KB} (techniques that should generalize to OTHER head ops/backends in this run; one line each, with the source uid). Keep both concise/high-signal. Return {ok,summary}.`,
          { phase: 'HeadKernel', label: `curate ${p.h.short_name} ${tag}`, schema: OKV });
      }
      // Revive: a parked lane with a big remaining ceiling gap gets another shot (at most 2) with the
      // freshly curated borrows.
      let revived = 0;
      for (const l of allLanes) {
        if (l.active || l.dead || (l.revives || 0) >= 2) continue;
        if ((l.ceiling - l.best) > 0.4 * Math.max(l.ceiling, 1e-9)) { l.active = true; l.noImprove = 0; l.revives = (l.revives || 0) + 1; revived++; }
      }
      if (revived) log(`  [deep] revived ${revived} high-ceiling parked lane(s) with fresh borrows.`);
    };

    // ---- BATCHED serial e2e GATE on the serving slot (concurrent with co-opt on dedicated cards) ----
    const runGate = async (opts = {}) => {
      // At FINALIZE, sweep ALL patched lanes (incl. isolated ~1.0) onto the cumulative overlay: the e2e
      // gate is the arbiter, so an oracle-invisible e2e win still banks. Per-wave gates take the top 2.
      const cands = opts.final
        ? allLanes.filter(l => (l.lastEval || l.patch)).sort((a, b) => b.best - a.best)
        : allLanes.filter(l => (l.lastEval || l.patch) && l.best > 1.0).sort((a, b) => b.best - a.best).slice(0, 2);
      if (!cands.length) return;
      const bankedHeads = new Set();   // one kernel per head: same-head lanes edit the SAME module
      if (opts.final) log(`[deep] FINALIZE gate: sweeping ${cands.length} patched lane(s) onto the cumulative overlay (combined cross-kernel deliverable).`);
      e2eGateCount++;
      log(`[deep] E2E GATE #${e2eGateCount} on serving {${SERVING_GPU}} TP=${SERVING_TP}: [${cands.map(c => c.uid + ' ' + c.best.toFixed(3) + 'x').join(', ')}] (overlapping co-opt on dedicated cards).`);
      for (const c of cands) {
        if (opts.final && bankedHeads.has(c.head.short_name)) { log(`  [deep] FINALIZE: skip ${c.uid} -- head ${c.head.short_name} already banked (same module, cannot stack).`); continue; }
        const deepInputs = {
          EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
          KERNEL_RESULT: {
            short_name: c.head.short_name, task_dir: c.ext.task_dir, op_kind: c.ext.op_kind, lane: c.key,
            winner_kind: 'patch', winner_backend: c.lang,
            target_callable: c.ext.target_callable || c.head.target_callable || '',
            authored_language: c.lang, authored_kernel_eval_dir: c.lastEval,
            apply_env: '', apply_flags: '', code_patch: c.patch || (c.lastEval ? `${c.lastEval}/final_patch.diff` : ''), tuning_artifact: '',
            verified_isolated_speedup: c.best, pct_gpu_time: c.head.pct_gpu_time, parity_note: 'expected_close',
          },
          CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
          CURRENT_THROUGHPUT: curTput,
          MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
          SKILL_DIR: WORKFLOW_DIR, DEEP_FEEDBACK: true,
          ...tuningIntegrateInputs(), ...ACCURACY_INPUTS,
          ...(opts.final && ACCURACY_GATE !== 'none' ? { ACCURACY_LIMIT: DEEP_FINAL_ACCURACY_LIMIT } : {}),   // de-noise the finalize accuracy decision
        };
        const integ = await runIntegrateBothLegs(
          'Apply a deep head candidate; gate on e2e throughput; report engagement/cudagraph/mem/decode for feedback.',
          deepInputs, `integrate ${c.uid} g${e2eGateCount}`, 'HeadKernel');
        if (integ && integ.output_parity === 'fail') {
          log(`  [deep] ${c.uid}: REJECTED — output_parity=fail vs true baseline.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'dead_end', lesson: 'parity fail vs true baseline' });
        } else if (integAccepted(integ, c.head.pct_gpu_time, c.best) && integ.e2e_throughput_tok_s > curTput) {
          curOverlay = integ.accepted_overlay || curOverlay; curTput = integ.e2e_throughput_tok_s; bankedHeads.add(c.head.short_name);
          bankAccepted(acceptedHeads, { short_name: c.head.short_name, op_kind: c.ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', ...e2eFrom(integ), isolated: c.best }, krOf(deepInputs));
          log(`  [deep] ${c.uid}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%); target ${Math.round(BASELINE_TPUT * DEEP_E2E_TARGET)} tok/s.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
        } else {
          // Only actual correctness/integration failures enter the corrective helper.
          // An inconclusive speedup review stops without re-authoring the candidate.
          const dreason = gateRejectReason(integ, c.head.pct_gpu_time, c.best);
          const dcorr = await tryCorrectiveReauthor({
            short_name: c.head.short_name, op_kind: c.ext.op_kind, shapes: c.ext.shapes, dtype: c.ext.dtype, regime: c.head.regime,
            gpu_id: SERVING_GPU, kernel_eval_dir: c.lastEval, task_dir: c.ext.task_dir, language: c.lang,
            isolated: c.best, reason: dreason, fix_class: rejectClass(dreason), pct_gpu_time: c.head.pct_gpu_time,
            base_inputs: {
              EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
              KERNEL_RESULT: {
                short_name: c.head.short_name, task_dir: c.ext.task_dir, op_kind: c.ext.op_kind, lane: c.key,
                winner_kind: 'patch', winner_backend: c.lang,
                target_callable: c.ext.target_callable || c.head.target_callable || '',
                authored_language: c.lang, authored_kernel_eval_dir: c.lastEval, apply_env: '', apply_flags: '',
                code_patch: c.patch || '', tuning_artifact: '', verified_isolated_speedup: c.best,
                pct_gpu_time: c.head.pct_gpu_time, parity_note: 'expected_close' },
              SKILL_DIR: WORKFLOW_DIR, ...ACCURACY_INPUTS,
            },
            cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
          });
          if (dcorr.banked) {
            curOverlay = dcorr.integ.accepted_overlay || curOverlay; curTput = dcorr.integ.e2e_throughput_tok_s; bankedHeads.add(c.head.short_name);
            bankAccepted(acceptedHeads, { short_name: c.head.short_name, op_kind: c.ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', ...e2eFrom(dcorr.integ), isolated: dcorr.isolated, corrective: true, patch: dcorr.patch || '' }, krOf(deepInputs));
            log(`  [deep] ${c.uid}: ACCEPTED after corrective re-author. e2e now ${curTput} tok/s (+${dcorr.integ.e2e_delta_pct}%).`);
            history.ledger.push({ direction: c.uid, isolated_speedup: dcorr.isolated, e2e_delta_pct: dcorr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${dreason}` });
          } else {
            log(`  [deep] ${c.uid}: e2e gate ${integ ? integ.gate : 'none'} (${integ ? integ.reason || '' : 'integrate failed'}).`);
            history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: integ ? integ.reason || 'no e2e gain' : 'integrate failed' });
          }
        }
      }
      const fb = await safeAgent(
        `You are the e2e FEEDBACK + HARNESS refiner [deep g${e2eGateCount}]. For the candidates just gated, write ${EVAL_DIR}/deep_head/e2e_feedback.md (per candidate: e2e delta; ENGAGED-live vs eager-fell-back under cudagraph; cudagraph/memory/decode behavior; parity/accuracy; ROOT CAUSE of any isolated->e2e gap). ` +
        `Then refresh ${EVAL_DIR}/deep_head/HARNESS_ADDENDUM.md so the isolated target ALIGNS with e2e WITHOUT touching the frozen oracle: (a) e2e-critical decode M-buckets to weight; (b) whether a cudagraph capture/replay measurement wrapper is needed; (c) hard gates (decode-no-regress, memory cap, graph-safe) so an isolated "win" that is all-NUL/eager under graph is caught EARLY. Return {ok, feedback_path, addendum_path}.`,
        { phase: 'HeadKernel', label: `feedback g${e2eGateCount}`, schema: OKV });
      gateFeedbackPath = (fb && fb.feedback_path) || `${EVAL_DIR}/deep_head/e2e_feedback.md`;
      gateHarnessPath = (fb && fb.addendum_path) || `${EVAL_DIR}/deep_head/HARNESS_ADDENDUM.md`;
      // re-profile if the stack moved enough — chase the new dominant bottleneck (Amdahl shifted).
      if (!KERNEL_EXTRACT_ON && want('profile') && curTput > lastReprofileTput * (1 + DEEP_REPROFILE_GAIN)) {
        log(`[deep] e2e +${((curTput / lastReprofileTput - 1) * 100).toFixed(1)}% since last profile — re-profiling to chase the moving bottleneck.`);
        const rp = await safeAgent(
          roleAgent('profiler', 'reprofile', 'Re-profile the CURRENT overlaid server; return refreshed head pct_gpu_time so EV re-weights toward the new bottleneck.', {
            EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          }),
          { phase: 'HeadKernel', label: `reprofile g${e2eGateCount}`, schema: { type: 'object', additionalProperties: true, properties: { heads: { type: 'array', items: { type: 'object', additionalProperties: true } } } } });
        if (rp && Array.isArray(rp.heads)) {
          for (const nh of rp.heads) {
            const tgt = allLanes.filter(l => l.head.short_name === (nh.short_name || nh.name));
            for (const l of tgt) if (Number.isFinite(nh.pct_gpu_time)) {
              l.head.pct_gpu_time = nh.pct_gpu_time;
            }
          }
          log(`[deep] re-profile updated head Amdahl weights.`);
        }
        lastReprofileTput = curTput;
      }
    };

    // ---- DEPTH: fresh authoring directions to RE-SEED plateaued lanes ----
    const DEEP_STEERS = {
      triton: [
        ` DIRECTION=persistent-kernel: a persistent / grid-stride kernel that keeps tiles resident and overlaps global load with ${MATRIX_CORE_NAME}.`,
        ` DIRECTION=warp-specialization: split warps into a producer (async global->LDS copy) and a consumer (${MATRIX_CORE_NAME}) for software pipelining.`,
        ' DIRECTION=epilogue-fusion: fuse the scale/activation/cast epilogue into the GEMM to remove a memory round-trip.',
        RDNA4_ISOLATE
          ? ' DIRECTION=wmma-layout: re-tune WMMA tile shape / LDS swizzle / num_warps / waves_per_eu / GROUP_SIZE_M for this exact (N,K,M-bucket).'
          : ' DIRECTION=mfma-layout: re-tune matrix_instr_nonkdim / kpack / LDS swizzle / waves_per_eu / GROUP_SIZE_M for this exact (N,K,M-bucket).',
        ' DIRECTION=double-buffer: deepen num_stages and LDS double-buffering to hide HBM latency on the K loop.',
        ' DIRECTION=split-K-atomic: split the K reduction across CUs with atomic accumulate for the large-M prefill shapes; non-split for small-M decode.',
        ' DIRECTION=fresh-rewrite: abandon the current tiling and try a fundamentally different decomposition than your best so far.',
      ],
      _default: [
        ' DIRECTION=new-decomposition: try a fundamentally different tiling/decomposition than your current best.',
        ' DIRECTION=fuse-prologue-epilogue: fold the pre/post ops + scaling into the main compute kernel.',
        ' DIRECTION=retune-shapes: per (N,K,M-bucket) re-search the launch-config space from scratch.',
        ' DIRECTION=pipeline: add software pipelining / double buffering across the reduction loop.',
      ],
    };
    const nextSteer = (l) => { const lib = DEEP_STEERS[l.lang] || DEEP_STEERS._default; const s = lib[(l.steerIdx || 0) % lib.length]; l.steerIdx = (l.steerIdx || 0) + 1; return s; };
    // When all lanes are parked but budget remains, revive them (dominant head first) with a fresh
    // direction instead of exiting at the first global plateau.
    const reseedForDepth = () => {
      let n = 0;
      const order = allLanes.slice().sort((a, b) => (b.head.pct_gpu_time || 0) - (a.head.pct_gpu_time || 0) || b.best - a.best);
      for (const l of order) {
        if (l.dead || (l.reseeds || 0) >= DEEP_MAX_RESEEDS) continue;
        l.active = true; l.noImprove = 0; l.reseeds = (l.reseeds || 0) + 1; l.steer = nextSteer(l); n++;
      }
      if (n) log(`[deep] global plateau but budget remains -> RE-SEED ${n} lane(s) with fresh DEEP directions (depth pass; dominant head first). Keeps compounding until the ${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h budget.`);
      return n > 0;
    };

    // ---- GLOBAL WAVE LOOP (runs until the budget; re-seeds on plateau) ----
    let wave = 0, e2eGateCount = 0, lastE2eIsoBest = 1.0, lastReprofileTput = curTput, gateFeedbackPath = '', gateHarnessPath = '', e2eIntervalHit = false;
    const armInterval = () => { if (typeof setTimeout === 'function' && DEEP_E2E_MAX_INTERVAL_MS > 0) setTimeout(() => { e2eIntervalHit = true; }, DEEP_E2E_MAX_INTERVAL_MS); };
    armInterval();
    let convergeStreak = 0, deepBurstsSpent = 0;
    try {
    // TIME_DEADLINE_HIT too (same deadline), so the wave loop never runs past the reserve.
    while (!DEEP_DEADLINE_HIT && !TIME_DEADLINE_HIT) {
      if (convergeStreak >= DEEP_CONVERGE_STREAK) { log(`[deep] CONVERGED \u2014 ${convergeStreak} consecutive zero-gain waves; stopping to finalize.`); break; }
      const projAgents = (deepBurstsSpent + mainSlots + serveSlots) * DEEP_AGENTS_PER_BURST + wave * 6;   // project a FULL next wave of bursts + per-wave overhead (harvest/curate/gate)
      if (projAgents >= DEEP_AGENT_BUDGET) { log(`[deep] agent budget reached (proj ~${projAgents}/${DEEP_AGENT_BUDGET}, ${deepBurstsSpent} bursts); stopping to finalize with margin for the cap.`); break; }
      if (!allLanes.some(l => l.active)) { if (!reseedForDepth()) break; continue; }   // depth: don't exit on plateau while budget remains
      wave++;
      const prevTput = curTput;
      const globalIsoBest = Math.max(1.0, ...allLanes.map(l => l.best));
      const gained = globalIsoBest / Math.max(lastE2eIsoBest, 1e-9) - 1;
      const haveCand = allLanes.some(l => (l.lastEval || l.patch) && l.best > 1.0);
      const gateDue = haveCand && (e2eGateCount === 0 || gained >= DEEP_E2E_GAIN_TRIGGER || e2eIntervalHit);
      // pick ready lanes by EV; a gate wave reserves the serving cards (only dedicated cards co-opt).
      const slots = gateDue ? mainSlots : (mainSlots + serveSlots);
      const ready = allLanes.filter(l => l.active).sort((a, b) => evOf(b) - evOf(a)).slice(0, Math.max(0, slots));
      const onMain = ready.slice(0, mainSlots);
      const onServe = gateDue ? [] : ready.slice(mainSlots);
      log(`[deep] WAVE ${wave}: ${allLanes.filter(l => l.active).length} active; running ${ready.length} [${ready.map(l => l.uid).join(', ') || '-'}]${gateDue ? ' + E2E GATE (overlap)' : ''}; e2e ${curTput} tok/s.`);
      const runBurst = (l, pool) => pool.with(1, async (g) => {
        l.ran++;
        await deepBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
          kernel_path: l.ext.task_dir, workflow_dir: KERNEL_WF_DIR, mode: l.mode, target_language: l.lang, op_spec: l.opSpec,
          perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR, use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
          ...KB_ARGS,
          budget: DEEP_WAVE_BUDGET, max_no_improve: DEEP_WAVE_BUDGET, gpu_ids: g[0],
          state_dir: l.state_dir, shared_kb: l.sharedKb, global_kb: GLOBAL_KB,
          incremental_analyze: l.ran > 1 ? 'true' : 'false',   // 2nd+ burst = continuation -> skip cold re-analysis
          ...(gateFeedbackPath ? { e2e_feedback: gateFeedbackPath } : {}),
          ...(gateHarnessPath ? { harness_addendum: gateHarnessPath } : {}),
          exp_root: `${l.deepDir}/runs/${l.key}`, apply_to_original: 'false',
          task: `deep lane '${l.key}' of ${l.head.short_name} (${l.ext.op_kind}), backend=${l.lang}, mode=${l.mode}.${l.steer} Build STRICTLY beyond this lane's cumulative best (vs-live ${l.best.toFixed(3)}x); roofline SOTA ~${(l.rooflineTarget || 0).toFixed(2)}x. Beat the LIVE kernel, not just your own first port. Read SHARED_KB + GLOBAL_KB and BORROW transferable techniques (incl. from OTHER kernels); write findings back.` + GRAPH_REQ + (TASK || ''),
        }, l.uid);
        return null;
      });
      await Promise.all([
        gateDue ? runGate() : Promise.resolve(),
        parallel(onMain.map(l => () => runBurst(l, cooptMain))),
        parallel(onServe.map(l => () => runBurst(l, cooptServe))),
      ]);
      if (gateDue) { lastE2eIsoBest = globalIsoBest; e2eIntervalHit = false; armInterval(); }   // e2eGateCount is bumped inside runGate()
      deepBurstsSpent += ready.length;
      await harvestLanes(ready, `w${wave}`);
      await curateAndRevive(`w${wave}`);
      const newIsoBest = Math.max(1.0, ...allLanes.map(l => l.best));   // convergence: did e2e OR any isolated best move?
      convergeStreak = (curTput > prevTput * 1.001 || newIsoBest > globalIsoBest * 1.001) ? 0 : convergeStreak + 1;
    }
    } catch (e) { log(`[deep] wave loop aborted (${(e && e.message) || e}); proceeding to finalize so progress is still banked + reported.`); }
    // FINALIZE always runs (convergence, agent budget, deadline or a throw); if it is cut off mid-sweep,
    // what is banked so far (curOverlay/curTput) holds.
    try {
      await harvestLanes(allLanes.filter(l => l.lastEval || l.ran > 0), 'final');
      await runGate({ final: true });   // every patched lane -> cumulative overlay
    } catch (e) { log(`[deep] finalize partial (${(e && e.message) || e}); best banked overlay so far: e2e ${curTput} tok/s.`); }
    log(`[deep] done after ${wave} wave(s), ${e2eGateCount} gate(s). e2e ${curTput} tok/s (${(curTput / Math.max(BASELINE_TPUT, 1e-9)).toFixed(3)}× baseline; target ×${DEEP_E2E_TARGET}). per-lane vs-live: ${allLanes.map(l => l.uid + '=' + l.best.toFixed(2) + 'x').join(', ')}.`);
  } else if (FAST_MODE && GPU_LIST.length > 1) {
    // ==== FAST-MODE PARALLEL HEAD TRACK (FAST_MODE, >1 card; else the serial branch below) ====
    // Per-head extract + bake-off, then every (operator, direction) author job in parallel, each on ONE
    // exclusively leased card. After a BARRIER (all cards released), integration runs SERIALLY on the
    // serving slot {SERVING_GPU} (TP=SERVING_TP), one op at a time, so nothing disturbs the e2e A/B.
    const ISO = makeSem(GPU_LIST);
    log(`[fast-mode] PARALLEL head track: ISO lanes={${GPU_LIST.join(',')}} (${GPU_LIST.length}); ` +
      `serving slot={${SERVING_GPU}} TP=${SERVING_TP} reserved for the serial e2e gate (hard barrier between).`);

    // ---- opt-A: per-head extract + bake-off, parallel, exclusive 1-card lease each ----
    const prepared = await parallel(heads.map((h) => async () => {
      if (FAST_DEADLINE_HIT) return { h, dead: 'deadline' };
      return ISO.with(1, async (g) => {
        const gpu = g[0];
        if (KERNEL_EXTRACT_ON) headQueue = headQueue.filter(c => c.prepared_ut !== h.prepared_ut);
        const ext = await extractWithBaseline(
          'kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
            EVAL_DIR, MODEL_PATH, GPU_ID: gpu, WORKLOAD, KERNEL: h, GEMM_SYNTH: gemmSynthFor(h),
            ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
            CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
            REQUIRE_DECODE_BUCKET: true, DECODE_M_BUCKETS: [1, CONC],
            PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
          },
          { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
        if (!ext || ext.smoke !== 'pass' || !ext.task_dir) return { h, gpu, ext, dead: 'extract' };
        const bake = await safeAgent(
          roleAgent('op_benchmarker', 'bakeoff', 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.', {
            EVAL_DIR, OP_TASK_DIR: ext.task_dir, OP_KIND: ext.op_kind, PCT_GPU_TIME: h.pct_gpu_time,
            CANDIDATE_BACKENDS: ext.candidate_backends || h.candidate_backends || [],
            GPU_ID: gpu, ENABLE_FP8, KERNEL_WF_DIR, KERNEL_BUDGET, SKILL_DIR: WORKFLOW_DIR,
          }),
          { phase: 'HeadKernel', label: `bakeoff ${h.short_name}`, schema: OPBENCH_SCHEMA });
        return { h, gpu, ext, bake };
      });
    }));

    // ---- process opt-A: flag dominant heads (never silently skip); seed direct_light + author jobs ----
    const headState = new Map();   // short_name -> { h, ext, cands: [] }
    const authorJobs = [];         // flattened (operator × direction) author directions
    for (const p of prepared) {
      if (!p) continue;
      const h = p.h;
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (p.dead === 'deadline') { log(`  [fast-mode] ${h.short_name}: skipped (dispatch deadline).`); continue;
      }
      if (p.dead === 'extract' || !p.ext || !p.ext.task_dir) {
        const why = extractFailWhy(p.ext);
        if (isDominant) { log(`  ⚠️ FLAG ${h.short_name}: DOMINANT head op extraction FAILED (${why}) — flagged, NOT skipped.`);
          flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'extract', gate: 'extract_failed', reason: why }); }
        else log(`  ${h.short_name}: op extraction failed (${why}); skipping.`);
        history.ledger.push({ direction: h.short_name, verdict: isDominant ? 'flagged' : 'dead_end', lesson: `op extraction failed (${why})` });
        continue;
      }
      const ext = p.ext, bake = p.bake;
      const harness = !!(bake && (bake.gate === 'harness_error' || bake.harness_suspect));
      const hasPlan = !!(bake && Array.isArray(bake.author_plan) && bake.author_plan.length);
      if (!bake || (bake.gate !== 'have_winner' && bake.gate !== 'author_recommended')) {
        if (isDominant || harness) {
          log(`  ⚠️ FLAG ${h.short_name}: bake-off gate=${bake ? bake.gate : 'null'}${harness ? ' (HARNESS ERROR — not a real no-win)' : ''}.${hasPlan ? ' Proceeding to author route.' : ''}`);
          flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'bakeoff', gate: bake ? bake.gate : 'null', harness_error: harness, had_author_plan: hasPlan, reason: bake ? bake.reason || bake.gate : 'bakeoff null' });
          history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: harness ? 'harness_error' : 'flagged', lesson: bake ? bake.reason || bake.gate : 'bakeoff null' });
          if (!hasPlan) continue;
        } else {
          log(`  ${h.short_name}: no win and nothing worth authoring (${bake ? bake.reason || bake.gate : 'none'}); skipping.`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: 'dead_end', lesson: bake ? bake.reason || 'no op win' : 'bakeoff failed' });
          continue;
        }
      }
      const st = { h, ext, cands: [] };
      headState.set(h.short_name, st);
      if (bake && bake.gate === 'have_winner' && bake.isolated_speedup > 1.0)
        st.cands.push({ kind: 'direct_light', source: bake.winner_backend, winner_kind: bake.winner_kind,
          apply_env: bake.apply_env || '', apply_flags: bake.apply_flags || '', code_patch: bake.code_patch || '',
          tuning_artifact: bake.tuning_artifact || '', isolated: bake.isolated_speedup, parity_note: bake.parity_note || 'expected_close' });
      for (const ap of (bake && bake.author_plan ? bake.author_plan.slice(0, HEAD_AUTHOR_MAX) : []))
        authorJobs.push({ short_name: h.short_name, h, ext, ap, best_known_ms: bake.best_known_ms });
    }

    // ---- opt-B: ALL (operator × direction) author jobs in ONE parallel pool, exclusive 1-card lease ----
    log(`[fast-mode] author fan-out: ${authorJobs.length} (operator × direction) job(s) across ${GPU_LIST.length} GPU lanes.`);
    const authored = await parallel(authorJobs.map((j) => async () => {
      if (FAST_DEADLINE_HIT) return null;
      return ISO.with(1, async (g) => {
        const lang = j.ap.language || 'triton';
        let al;
        try {
          al = await fastBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
            kernel_path: j.ext.task_dir, workflow_dir: KERNEL_WF_DIR,
            mode: j.ap.route === 'rewrite' ? 'optimize' : 'author', target_language: lang,
            op_spec: { op_kind: j.ext.op_kind, shapes: j.ext.shapes || {}, dtype: j.ext.dtype || 'bf16', regime: j.h.regime || '', cuda_graph_safe: true, ...(j.ext.workload_path ? { workload_path: j.ext.workload_path } : {}) },
            perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
            use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
            ...KB_ARGS,
            budget: KERNEL_BUDGET, gpu_ids: g[0], exp_root: `${EVAL_DIR}/kernels/_exp`,
            task: `Author+optimize a ${lang} implementation of this op vs the immutable oracle (beat ${j.best_known_ms || '?'} ms). ` +
              `This kernel will be overlaid onto the LIVE decode path (CUDA-graph captured): its STEADY-STATE hot path MUST be ` +
              `host-sync-free (NO .item()/.cpu()/.tolist()/.sum().item()/torch.cuda.synchronize(), no Python branch on a GPU scalar). ` +
              `Cache any weight prep (transpose/requant/preshuffle) by weight.data_ptr() done ONCE, not per call. ` +
              `MEMORY FOOTPRINT IS A HARD CONSTRAINT: use the FUSED fp8 path (fold the block-scale into the operand scale, one fp8 ${MATRIX_CORE_NAME} ` +
              `GEMM) and cache only COMPACT fp8/preshuffled weights (never a bf16 expansion); the integrated kernel MUST fit at the ` +
              `accepted config's mem-fraction. ` + GRAPH_REQ + (TASK || ''),
            apply_to_original: 'false',
          }, `${j.short_name}:${lang}`);
        } catch (e) { al = { authored: false, validation_status: 'error', reason: String(e) }; }
        return { j, al };
      });
    }));
    // ---- BARRIER: all isolated optimize done; ISO pool idle; serving slot now contention-free ----
    for (const r of authored) {
      if (!r || !r.al) continue;
      const j = r.j, al = r.al, lang = j.ap.language || 'triton';
      const st = headState.get(j.short_name); if (!st) continue;
      // Rank on the WORKLOAD-WEIGHTED speedup (the kernel layer's primary metric, comparable with
      // bake.isolated_speedup), not the geomean, which low-weight prefill buckets dominate.
      const alIso = KERNEL_EXTRACT_ON ? (al.final_weighted ?? al.final_speedup ?? 0)
        : (al.final_weighted != null ? al.final_weighted : al.final_geomean);
      if (al.authored !== false && alIso > 1.0 && al.final_patch) {
        st.cands.push({ kind: 'authored', source: lang, winner_kind: 'authored', language: lang,
          final_patch: al.final_patch, kernel_eval_dir: al.eval_dir, isolated: alIso });
        log(`  ${j.short_name}: authored ${lang} ${alIso.toFixed(2)}x weighted (vs the frozen online kernel, not the seed).`);
      } else {
        log(`  ${j.short_name}: author ${lang} produced no usable kernel (${al ? al.reason || al.validation_status : 'none'}).`);
        history.ledger.push({ direction: `${j.short_name}:${lang}`, verdict: 'dead_end', lesson: al ? al.reason || 'author no speedup' : 'author failed' });
      }
    }

    // ---- integrate SERIALLY on the serving slot, in head order, ISO quiesced ----
    for (const h of heads) {
      const st = headState.get(h.short_name); if (!st) continue;
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (!st.cands.length) {
        if (isDominant) { log(`  ⚠️ FLAG ${h.short_name}: DOMINANT head produced NO candidate — flagged, NOT skipped.`);
          if (!flaggedHeads.some((f) => f.short_name === h.short_name)) flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'no_candidate', gate: 'no_candidate', reason: 'bake-off + author route both empty' });
          history.ledger.push({ direction: h.short_name, verdict: 'flagged', lesson: 'DOMINANT head: no candidate to integrate' }); }
        else log(`  ${h.short_name}: no candidate to integrate; skipping.`);
        continue;
      }
      st.cands.sort((a, b) => (b.isolated || 0) - (a.isolated || 0));
      const cand = st.cands[0];
      log(`  ${h.short_name}: best candidate=${cand.source} (${(cand.isolated || 0).toFixed(2)}x, ${cand.kind}). Integrating to e2e (serial, slot {${SERVING_GPU}}).`);
      const headWinnerInputs = {
        EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
        KERNEL_RESULT: { short_name: h.short_name, task_dir: st.ext.task_dir, op_kind: st.ext.op_kind,
          winner_kind: cand.winner_kind, winner_backend: cand.source,
          target_callable: st.ext.target_callable || h.target_callable || '',
          authored_language: cand.language || '', authored_kernel_eval_dir: cand.kernel_eval_dir || '',
          apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
          code_patch: cand.code_patch || cand.final_patch || '', tuning_artifact: cand.tuning_artifact || '',
          verified_isolated_speedup: cand.isolated || 0, pct_gpu_time: h.pct_gpu_time,
          // Live seam + engagement check: an overlay that does not bind on the live path is rejected
          // (no_engagement) before a full e2e A/B is spent.
          live_call_seam: h.live_call_seam || '', engagement_check: h.engagement_check || '',
          parity_note: cand.parity_note || 'expected_close' },
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
        CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
        ENGAGEMENT_CHECK: h.engagement_check || '',
      };
      const integ = await runIntegrateBothLegs(
        'Apply the head-op winner; gate on e2e throughput.', headWinnerInputs,
        `integrate ${h.short_name}`, 'HeadKernel');
      if (integAccepted(integ, h.pct_gpu_time, cand.isolated) && integ.e2e_throughput_tok_s > curTput) {
        curOverlay = integ.accepted_overlay || curOverlay;
        if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
        if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
        curTput = integ.e2e_throughput_tok_s;
        bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: st.ext.op_kind, backend: cand.source, kind: cand.winner_kind, ...e2eFrom(integ), isolated: cand.isolated }, krOf(headWinnerInputs));
        log(`  ${h.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
        history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
      } else {
        // A measurement review never routes to the correctness corrective.
        const reason = gateRejectReason(integ, h.pct_gpu_time, cand.isolated);
        const corr = (cand.kind === 'authored' && rejectClass(reason) !== '')
          ? await tryCorrectiveReauthor({
              short_name: h.short_name, op_kind: st.ext.op_kind, shapes: st.ext.shapes, dtype: st.ext.dtype, regime: h.regime,
              gpu_id: SERVING_GPU, kernel_eval_dir: cand.kernel_eval_dir, task_dir: st.ext.task_dir, language: cand.language,
              isolated: cand.isolated, reason, fix_class: rejectClass(reason), pct_gpu_time: h.pct_gpu_time, phase_name: 'HeadKernel',
              base_inputs: {
                EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
                KERNEL_RESULT: { short_name: h.short_name, task_dir: st.ext.task_dir, op_kind: st.ext.op_kind,
                  winner_kind: cand.winner_kind, winner_backend: cand.source,
                  target_callable: st.ext.target_callable || h.target_callable || '',
                  authored_language: cand.language || '', authored_kernel_eval_dir: cand.kernel_eval_dir || '',
                  apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
                  code_patch: cand.code_patch || cand.final_patch || '', tuning_artifact: cand.tuning_artifact || '',
                  verified_isolated_speedup: cand.isolated || 0, pct_gpu_time: h.pct_gpu_time, parity_note: 'expected_close' },
                SKILL_DIR: WORKFLOW_DIR,
              },
              cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
            })
          : { banked: false };
        if (corr.banked) {
          curOverlay = corr.integ.accepted_overlay || curOverlay; curTput = corr.integ.e2e_throughput_tok_s;
          bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: st.ext.op_kind, backend: cand.source, kind: 'authored', ...e2eFrom(corr.integ), isolated: corr.isolated, corrective: true, patch: corr.patch || '' }, krOf(headWinnerInputs));
          log(`  ${h.short_name}: ACCEPTED after corrective re-author (${reason}). e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${reason}` });
        } else {
          log(`  ${h.short_name}: REJECTED at e2e gate (${reason}).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: reason || 'no e2e gain' });
        }
      }
    }
  } else {
  for (const h of heads) {
    // Budget guard: start no new head op after a dispatch deadline (FAST_DEADLINE_HIT in fast mode;
    // TIME_DEADLINE_HIT in all modes when time_budget_s is set).
    if ((FAST_MODE && FAST_DEADLINE_HIT) || TIME_DEADLINE_HIT) {
      log(`[budget] dispatch deadline reached — stopping head dispatch before ${h.short_name} (${headDispatched}/${heads.length} heads done).`);
      break;
    }
    headDispatched++;
    if (KERNEL_EXTRACT_ON) headQueue = headQueue.filter(c => c.prepared_ut !== h.prepared_ut);
    // B4 measures THIS head's trial against its own start, not the run clock.
    const ablHeadStart = ELAPSED_MS;
    // (h1) Extract the op into a standalone immutable unittest (fused heads stay fused: gemmSynthFor).
    const ext = await extractWithBaseline(
      'kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: h.gpu_id, WORKLOAD, KERNEL: h, GEMM_SYNTH: gemmSynthFor(h),
        ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        // The unittest MUST span BOTH regimes (kernel_extractor.md "Shapes must span BOTH regimes"):
        // decode M (running batch ≈ conc) and M=1 are passed explicitly so they are never dropped.
        REQUIRE_DECODE_BUCKET: true,
        DECODE_M_BUCKETS: [1, CONC],
        PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
      },
      { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
    const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
    if (!ext || ext.smoke !== 'pass' || !ext.task_dir) {
      const why = extractFailWhy(ext);
      if (isDominant) {
        log(`  ⚠️ FLAG ${h.short_name}: DOMINANT head (${(h.pct_gpu_time || 0).toFixed(1)}% GPU) op extraction FAILED (${why}) — flagged, NOT silently skipped.`);
        flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'extract', gate: 'extract_failed', reason: why });
        history.ledger.push({ direction: h.short_name, verdict: 'flagged', lesson: `DOMINANT head extraction failed (${why})` });
      } else {
        log(`  ${h.short_name}: op extraction failed (${why}); skipping.`);
        history.ledger.push({ direction: h.short_name, verdict: 'dead_end', lesson: 'op extraction failed' });
      }
      continue;
    }
    // (h2) DISCOVER existing impls + tune cheap levers + DECIDE an author_plan.
    const bake = await safeAgent(
      roleAgent('op_benchmarker', 'bakeoff', 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.', {
        EVAL_DIR, OP_TASK_DIR: ext.task_dir, OP_KIND: ext.op_kind, PCT_GPU_TIME: h.pct_gpu_time,
        CANDIDATE_BACKENDS: ext.candidate_backends || h.candidate_backends || [],
        GPU_ID: h.gpu_id, ENABLE_FP8, KERNEL_WF_DIR, KERNEL_BUDGET, SKILL_DIR: WORKFLOW_DIR,
      }),
      { phase: 'HeadKernel', label: `bakeoff ${h.short_name}`, schema: OPBENCH_SCHEMA });
    if (!bake || (bake.gate !== 'have_winner' && bake.gate !== 'author_recommended')) {
      const gate = bake ? bake.gate : 'null';
      const harness = !!(bake && (bake.gate === 'harness_error' || bake.harness_suspect));
      const hasPlan = !!(bake && Array.isArray(bake.author_plan) && bake.author_plan.length);
      // A DOMINANT head or a HARNESS fault is flagged, never silently skipped; with an author_plan it
      // still falls through to the author route (judged by the immutable unittest).
      if (isDominant || harness) {
        log(`  ⚠️ FLAG ${h.short_name}: ${isDominant ? `DOMINANT head (${(h.pct_gpu_time || 0).toFixed(1)}% GPU)` : 'head'} bake-off gate=${gate}${harness ? ' (HARNESS ERROR — bake-off could not measure; NOT a real no-win)' : ''}. ${hasPlan ? 'Proceeding to author route anyway.' : 'No author_plan to fall back on.'}`);
        flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'bakeoff', gate, harness_error: harness, had_author_plan: hasPlan, reason: bake ? bake.reason || gate : 'bakeoff returned null' });
        history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: harness ? 'harness_error' : 'flagged', lesson: bake ? bake.reason || gate : 'bakeoff null' });
        if (!hasPlan) continue;       // can't author -> FLAGGED (surfaced in report), not a silent skip
        // else: fall through to the author route below (do NOT continue)
      } else {
        log(`  ${h.short_name}: no win and nothing worth authoring (${bake ? bake.reason || gate : 'none'}); skipping.`);
        history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: 'dead_end', lesson: bake ? bake.reason || 'no op win' : 'bakeoff failed' });
        continue;
      }
    }

    // Build the candidate list: the cheap direct_light winner (if any) + any authored implementations.
    const headCands = [];
    if (bake.gate === 'have_winner' && bake.isolated_speedup > 1.0) {
      headCands.push({ kind: 'direct_light', source: bake.winner_backend, winner_kind: bake.winner_kind,
        apply_env: bake.apply_env || '', apply_flags: bake.apply_flags || '', code_patch: bake.code_patch || '',
        tuning_artifact: bake.tuning_artifact || '', isolated: bake.isolated_speedup,
        parity_note: bake.parity_note || 'expected_close' });
    }
    // Author/rewrite route per planned language via the kernel layer: mode=author writes from scratch
    // then optimizes, mode=optimize rewrites an existing impl; the oracle in ext.task_dir judges both.
    let plan = (bake.author_plan || []).slice(0, HEAD_AUTHOR_MAX);
    // ---- B4 gate ----
    // One bounded trial for the author route (the expensive part of a head): at the deadline it
    // continues only if the head already holds a non-incumbent candidate, else it pauses, with a
    // seeded audit continuation (ABL_AUDIT_P, default 10%) so late winners stay estimable.
    // The incumbent's identity: a candidate equal to it is not a candidate.
    const ablIncumbentId = ablCandId({ apply_env: curEnv, apply_flags: curFlags, code_patch: curOverlay });
    let ablImplStart = ELAPSED_MS;      // the implementation clock starts AFTER preparation
    let ablImplCap = ABL_IMPL_MS;
    let ablAudited = false;
    if (ABL('B4')) {
      const prepMs = ELAPSED_MS - ablHeadStart;
      const ready = ablHeadReady(headCands, ext, ablIncumbentId);
      // GATE 1 -- preparation: extraction + bake-off past ABL_PREP_MS with no candidate stops the
      // head (charged to preparation, not to the implementation trial).
      if (!ready && prepMs >= ABL_PREP_MS) {
        const draw = ablDraw(h.short_name);
        ablAudited = draw < ABL_AUDIT_P;
        ablEvent({ event: 'prep_gate', head: h.short_name, pct_gpu_time: h.pct_gpu_time,
          action: ablAudited ? 'audit_continue' : 'pause', ready: false, draw,
          audit_probability: ABL_AUDIT_P, prep_ms: prepMs, prep_cap_ms: ABL_PREP_MS,
          candidates: headCands.length,
          reason: 'preparation budget exhausted with no evidence-bearing non-incumbent candidate' });
        log(`  [${ABLATION_ARM}] ${h.short_name}: PREP budget spent (${Math.round(prepMs / 60000)}min), no candidate -> ${ablAudited ? 'AUDIT-CONTINUE' : 'PAUSE'} (draw ${draw.toFixed(4)} vs p=${ABL_AUDIT_P}).`);
        if (!ablAudited) {
          history.ledger.push({ direction: h.short_name, verdict: 'paused',
            lesson: `${ABLATION_ARM}: preparation budget expired before any candidate existed` });
          plan = [];
        } else {
          ablImplCap = ABL_AUDIT_MS;    // the audit continuation is a SEPARATE budget line
        }
      } else {
        ablEvent({ event: 'prep_gate', head: h.short_name, pct_gpu_time: h.pct_gpu_time,
          action: 'admit', ready, prep_ms: prepMs, prep_cap_ms: ABL_PREP_MS,
          candidates: headCands.length, impl_cap_ms: ablImplCap });
      }
    }
    for (const ap of plan) {
      // GATE 2 -- implementation: the cap is checked before every language AND handed to the
      // nested workflow, the only boundary that can contain work already in flight.
      if (ABL('B4')) {
        const implMs = ELAPSED_MS - ablImplStart;
        if (implMs >= ablImplCap) {
          ablEvent({ event: 'impl_gate', head: h.short_name, action: 'stop',
            impl_ms: implMs, impl_cap_ms: ablImplCap, audited: ablAudited,
            remaining_plan: plan.length, reason: 'implementation trial budget exhausted' });
          log(`  [${ABLATION_ARM}] ${h.short_name}: implementation budget spent (${Math.round(implMs / 60000)}min of ${Math.round(ablImplCap / 60000)}min) — stopping the author route.`);
          history.ledger.push({ direction: h.short_name, verdict: 'paused',
            lesson: `${ABLATION_ARM}: implementation trial budget expired` });
          break;
        }
        ABL_NESTED_CAP_MS = Math.max(60000, ablImplCap - implMs);
      }
      const lang = ap.language || 'triton';
      if (laneWindowShort(`author lane ${h.short_name}:${lang}`)) {
        history.ledger.push({ direction: `${h.short_name}:${lang}`, verdict: 'paused',
          lesson: 'time-budget: too close to the final reserve for a kernel lane plus its A/B' });
        break;
      }
      let al;
      // Retry a TRANSIENT failure (threw, or no final_geomean) so a language is not silently dropped;
      // never retry a completed no-speedup (final_geomean <= 1.0), which is a real result.
      const AUTHOR_TRIES = parseInt(A.head_author_tries != null ? A.head_author_tries : (FAST_MODE ? 1 : 2), 10);
      for (let attempt = 1; attempt <= AUTHOR_TRIES; attempt++) {
        try {
          al = await fastBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
            kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
            mode: ap.route === 'rewrite' ? 'optimize' : 'author', target_language: lang,
            op_spec: { op_kind: ext.op_kind, shapes: ext.shapes || {}, dtype: ext.dtype || 'bf16', regime: h.regime || '', cuda_graph_safe: true, ...(ext.workload_path ? { workload_path: ext.workload_path } : {}) },
            perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
            use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
            ...KB_ARGS,
            budget: KERNEL_BUDGET, gpu_ids: h.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
            task: `Author+optimize a ${lang} implementation of this op vs the immutable oracle (beat ${bake.best_known_ms || '?'} ms). ` +
              `This kernel will be overlaid onto the LIVE sglang decode path, which is CUDA-graph captured: its STEADY-STATE hot ` +
              `path (2nd call onward) MUST be host-sync-free — NO .item()/.cpu()/.tolist()/.sum().item()/torch.cuda.synchronize() ` +
              `and no Python branch on a GPU scalar (a host sync DEADLOCKS graph capture → 0 live forwards → e2e rejected). ` +
              `Cache any weight prep (transpose/requant/preshuffle) by weight.data_ptr() done ONCE, not per call. ` +
              `MEMORY FOOTPRINT IS A HARD CONSTRAINT: the persistent weight cache is kept for ALL layers at once, so do NOT ` +
              `re-materialize full bf16 weights (raw+preshuffled bf16 across every layer = tens of GB → forces mem-fraction ` +
              `down → starves the KV-cache pool → net e2e REGRESSION even when the GEMM is faster). Use the FUSED fp8 path ` +
              `(fold the block-scale into the operand scale, run ONE fp8 ${MATRIX_CORE_NAME} GEMM — the "kill the dequant" lever) and cache ` +
              `only COMPACT fp8/preshuffled weights (~the model's own fp8 weight size), never a bf16 expansion. The integrated ` +
              `kernel MUST fit at the same mem-fraction the accepted config uses. ` + GRAPH_REQ + (TASK || ''),
            apply_to_original: 'false',
          }, `${h.short_name}:${lang}`);
        } catch (e) { al = { authored: false, validation_status: 'error', reason: String(e) }; }
        const transient = !al || al.validation_status === 'error' || (al.authored === false && al.final_geomean == null);
        if (!transient || attempt === AUTHOR_TRIES) break;
        log(`  ${h.short_name}: author ${lang} attempt ${attempt}/${AUTHOR_TRIES} died transiently (${al ? al.reason || al.validation_status : 'null'}) — retrying so this language isn't dropped.`);
      }
      // Rank on the WORKLOAD-WEIGHTED speedup (same basis as bake.isolated_speedup), not the
      // geomean; only fixed-load runs fall back to the geomean.
      const alIso = al && (KERNEL_EXTRACT_ON ? (al.final_weighted ?? al.final_speedup ?? 0)
        : (al.final_weighted != null ? al.final_weighted : al.final_geomean));
      if (al && al.authored !== false && alIso > 1.0 && al.final_patch) {
        headCands.push({ kind: 'authored', source: lang, winner_kind: 'authored', language: lang,
          final_patch: al.final_patch, kernel_eval_dir: al.eval_dir, isolated: alIso });
        log(`  ${h.short_name}: authored ${lang} ${alIso.toFixed(2)}x weighted (vs the frozen online kernel, not the seed).`);
      } else {
        log(`  ${h.short_name}: author ${lang} produced no usable kernel (${al ? al.reason || al.validation_status : 'none'}).`);
        history.ledger.push({ direction: `${h.short_name}:${lang}`, verdict: 'dead_end', lesson: al ? al.reason || 'author no speedup' : 'author failed' });
      }
    }
    if (!headCands.length) {
      if (isDominant) {
        log(`  ⚠️ FLAG ${h.short_name}: DOMINANT head (${(h.pct_gpu_time || 0).toFixed(1)}% GPU) produced NO candidate (bake-off + author route both empty) — flagged, NOT silently skipped.`);
        if (!flaggedHeads.some(f => f.short_name === h.short_name)) {
          flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'no_candidate', gate: 'no_candidate', reason: 'bake-off harness/no-win and author route produced no usable kernel' });
        }
        history.ledger.push({ direction: h.short_name, verdict: 'flagged', lesson: 'DOMINANT head: no candidate to integrate' });
      } else {
        log(`  ${h.short_name}: no candidate to integrate; skipping.`);
      }
      continue;
    }
    headCands.sort((a, b) => (b.isolated || 0) - (a.isolated || 0));
    log(`  ${h.short_name}: ${headCands.length} candidate(s) [${headCands.map(c => `${c.source} ${(c.isolated || 0).toFixed(2)}x`).join(', ')}] — e2e-testing EACH; reference leg measured ONCE and reused.`);

    // (h3) e2e gate on EVERY candidate (isolated speedup is a weak predictor of e2e). All A/B against
    // the same config, so the reference leg is measured once and reused (REUSE_REF/SHARED_REF_MED);
    // CAND_TAG keeps each candidate's bench/overlay apart. The best MEASURED e2e that clears the gate
    // wins; if none clears, the top candidate goes to corrective/pending/reject. Inputs are built per
    // candidate so Fix C can re-issue the same A/B at Finalize.
    const mkIntegrateInputs = (cand, ci, sharedRefMed) => ({
      EVAL_DIR, MODEL_PATH, GPU_ID: h.gpu_id, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
      KERNEL_RESULT: { short_name: h.short_name, task_dir: ext.task_dir, op_kind: ext.op_kind,
        winner_kind: cand.winner_kind, winner_backend: cand.source,
        target_callable: ext.target_callable || h.target_callable || '',
        authored_language: cand.language || '', authored_kernel_eval_dir: cand.kernel_eval_dir || '',
        apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
        code_patch: cand.code_patch || cand.final_patch || '', tuning_artifact: cand.tuning_artifact || '',
        verified_isolated_speedup: cand.isolated || 0, pct_gpu_time: h.pct_gpu_time,
        parity_note: cand.parity_note || 'expected_close' },
      CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
      CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
      CAND_TAG: `c${ci}_${cand.source}`,
      ...(sharedRefMed != null ? { REUSE_REF: true, SHARED_REF_MED: sharedRefMed } : {}),
    });

    let sharedRefMed = null;      // reference leg measured once per head, then reused by later candidates
    let bestPick = null;          // { cand, integ } with the highest MEASURED e2e that cleared the gate
    let top = null;               // top-iso candidate's {cand, inputs, integ, abc} — the corrective fallback target
    for (let ci = 0; ci < headCands.length; ci++) {
      const cand = headCands[ci];
      const inputs = mkIntegrateInputs(cand, ci, sharedRefMed);
      const integ = await runIntegrateBothLegs(
        'Apply a head-op candidate; gate on e2e throughput (the reference config is shared across this head\'s candidates — reuse it when REUSE_REF/SHARED_REF_MED are set).',
        inputs, `integrate ${h.short_name}:${cand.source}`, 'HeadKernel');
      // A/B completion: gate:'incomplete' / ab_complete:false means a leg is still missing (NOT a rejection).
      const abc = !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);
      if (sharedRefMed == null && abc && integ.ref_med) sharedRefMed = integ.ref_med;   // lock the shared ref
      if (ci === 0) top = { cand, inputs, integ, abc };
      const passed = abc && integAccepted(integ, h.pct_gpu_time, cand.isolated) && integ.e2e_throughput_tok_s > curTput;
      // One ledger row per candidate (short_name:backend), so the report sees every backend's e2e.
      history.ledger.push({ direction: `${h.short_name}:${cand.source}`, isolated_speedup: cand.isolated || 0,
        e2e_delta_pct: integ ? (integ.e2e_delta_pct || 0) : 0,
        verdict: passed ? 'candidate_passed' : (abc ? 'candidate_rejected' : 'candidate_incomplete'),
        lesson: `${cand.winner_kind} ${cand.source}: ${integ && integ.e2e_throughput_tok_s ? `${integ.e2e_throughput_tok_s.toFixed(0)} tok/s (${(integ.e2e_delta_pct || 0).toFixed(2)}%)` : (integ ? integ.reason || integ.gate : 'null/timeout')}` });
      if (passed) {
        if (!bestPick || integ.e2e_throughput_tok_s > bestPick.integ.e2e_throughput_tok_s) bestPick = { cand, integ, inputs };
        log(`  ${h.short_name}: candidate ${cand.source} PASSED e2e gate (${integ.e2e_throughput_tok_s} tok/s, +${integ.e2e_delta_pct}%).`);
      } else {
        log(`  ${h.short_name}: candidate ${cand.source} ${abc ? `rejected (${gateRejectReason(integ, h.pct_gpu_time, cand.isolated)})` : `A/B incomplete (${integ ? integ.reason || integ.gate : 'null/timeout'})`}.`);
      }
    }

    if (bestPick) {
      // A head winner may carry an overlay (authored/patch) and/or config (env/flag): capture both.
      const cand = bestPick.cand, integ = bestPick.integ;
      curOverlay = integ.accepted_overlay || curOverlay;
      if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
      if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
      curTput = integ.e2e_throughput_tok_s;
      bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: cand.winner_kind, ...e2eFrom(integ), isolated: cand.isolated }, krOf(bestPick.inputs));
      log(`  ${h.short_name}: ACCEPTED best candidate=${cand.source} (${(cand.isolated || 0).toFixed(2)}x iso). e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else {
      // None cleared: on the TOP candidate, corrective re-author for a fixable reject/crash, else an
      // incomplete A/B stays PENDING, else a plain reject.
      const cand = top.cand, integ = top.integ, headIntegrateInputs = top.inputs;
      if (!top.abc) {
        // A FIXABLE warmup crash (zero A/B samples: graph-capture-unsafe, host sync, NO_BINARY_FOR_GPU)
        // tries the corrective re-author first; PENDING only if not fixable or the fix did not land.
        const reason = integ ? (integ.reason || integ.gate || '') : '';
        const corr = (cand.kind === 'authored' && FIXABLE_REJECT_RX.test(reason))
          ? await tryCorrectiveReauthor({
              short_name: h.short_name, op_kind: ext.op_kind, shapes: ext.shapes, dtype: ext.dtype, regime: h.regime,
              gpu_id: h.gpu_id, kernel_eval_dir: cand.kernel_eval_dir, task_dir: ext.task_dir, language: cand.language,
              isolated: cand.isolated, base_inputs: headIntegrateInputs, reason,
              cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
            })
          : { banked: false };
        if (corr.banked) {
          curOverlay = corr.integ.accepted_overlay || curOverlay; curTput = corr.integ.e2e_throughput_tok_s;
          bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: 'authored', ...e2eFrom(corr.integ), isolated: corr.isolated, corrective: true, patch: corr.patch || '' }, krOf(headIntegrateInputs));
          log(`  ${h.short_name}: ACCEPTED after corrective re-author (was crash/incomplete: ${reason}). e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed crash: ${reason}` });
        } else {
          pendingIntegrations.push({ track: 'head', short_name: h.short_name, isolated: cand.isolated || 0,
            pct_gpu_time: h.pct_gpu_time, inputs: headIntegrateInputs,
            winner_kind: cand.winner_kind, apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
            op_kind: ext.op_kind, backend: cand.source,
            partial: integ ? { gate: integ.gate, ref_med: integ.ref_med, cand_med: integ.cand_med, reason: integ.reason } : null });
          log(`  ${h.short_name}: INTEGRATE INCOMPLETE — A/B not finished (${integ ? integ.reason || integ.gate : 'null/timeout'}); kept as PENDING (not a rejection).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, verdict: 'incomplete', lesson: integ ? integ.reason || 'A/B not finished' : 'integrate timed out/null before A/B completed' });
        }
      } else {
        // Route actual parity/accuracy failures to correctness fixes and JIT/capture failures
        // to integration fixes; an inconclusive speedup review is not a kernel defect.
        const reason = gateRejectReason(integ, h.pct_gpu_time, cand.isolated);
        const corr = (cand.kind === 'authored')
          ? await tryCorrectiveReauthor({
              short_name: h.short_name, op_kind: ext.op_kind, shapes: ext.shapes, dtype: ext.dtype, regime: h.regime,
              gpu_id: h.gpu_id, kernel_eval_dir: cand.kernel_eval_dir, task_dir: ext.task_dir, language: cand.language,
              isolated: cand.isolated, base_inputs: headIntegrateInputs, reason, fix_class: rejectClass(reason), pct_gpu_time: h.pct_gpu_time,
              cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
            })
          : { banked: false };
        if (corr.banked) {
          curOverlay = corr.integ.accepted_overlay || curOverlay; curTput = corr.integ.e2e_throughput_tok_s;
          bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: 'authored', ...e2eFrom(corr.integ), isolated: corr.isolated, corrective: true, patch: corr.patch || '' }, krOf(headIntegrateInputs));
          log(`  ${h.short_name}: ACCEPTED after corrective re-author. e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${reason}` });
        } else {
          log(`  ${h.short_name}: REJECTED at e2e gate (${reason}).`);
          history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct || 0, verdict: 'dead_end', lesson: reason || 'no e2e gain' });
        }
      }
    }
  }
  } // end serial head track (default path; runs for normal mode and fast-mode-single-GPU)
  // Head wins reshape the profile: re-profile before the kernel loop.
  if (!KERNEL_EXTRACT_ON && acceptedHeads.length && !searchClosed('post-head re-profile')) {
    profile = await safeAgent(
      roleAgent('profiler', 'reprofile', 'Re-profile after head-kernel wins.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 'head',
        OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        ...ANALYSIS_SKILL_INPUTS,
      }),
      { phase: 'Profile', label: 'profiler:post-head', schema: PROFILE_SCHEMA });
  }
  log(`Head-kernel track done. ${acceptedHeads.length} accepted, throughput ${curTput} tok/s (${(curTput / BASELINE_TPUT).toFixed(3)}x).`);
  if (flaggedHeads.length) {
    log(`⚠️ ${flaggedHeads.length} DOMINANT head(s) FLAGGED (not optimized, NOT silently skipped): ` +
      flaggedHeads.map(f => `${f.short_name} [${(f.pct_gpu_time || 0).toFixed(1)}% GPU, ${f.gate}${f.harness_error ? '/harness' : ''}]`).join('; ') +
      `. These carry the most headroom — see the report's FLAGGED section.`);
  }
}

// ===========================================================================
// PHASE: Milestone loop — extract -> recursive kernel optimize -> overlay -> e2e gate
// ===========================================================================
// Floor: below MIN_KERNEL_TASKS dispatched tasks the noImprove early-stop is off (an empty plan still
// stops). TIME_DEADLINE_HIT (set only with time_budget_s) stops new milestones, even below the floor.
while (want('kernel') && !TIME_DEADLINE_HIT && dispatched < BUDGET && (dispatched < MIN_KERNEL_TASKS || noImprove < 2)) {
  milestone++;
  const remaining = BUDGET - dispatched;
  const belowFloor = dispatched < MIN_KERNEL_TASKS;

  // --- (a) Plan this milestone (Architect): below the floor it MUST nominate, never re-using a
  // confirmed e2e-null direction verbatim.
  phase('Milestone');
  let plan;
  if (KERNEL_EXTRACT_ON) {
    plan = { stop: !kernelQueue.length, kernel_candidates: kernelQueue };
    kernelQueue = [];
  } else {
    plan = (milestone === 1 && kernelQueue.length)
      ? { stop: false, kernel_candidates: kernelQueue }
      : await safeAgent(
        roleAgent('system_architect', 'plan_milestone', `Nominate next kernels — ONLY editable kernels with pct_gpu_time >= ${MILESTONE_MIN_PCT}% (below that, Amdahl says they can't move e2e; do not nominate them even to meet the floor). Each candidate MUST carry its pct_gpu_time.`, {
          EVAL_DIR, ROUND: milestone, BUDGET_REMAINING: remaining, CURRENT_THROUGHPUT: curTput,
          BASELINE_THROUGHPUT: BASELINE_TPUT, NOISE_BAND_PCT: NOISE_BAND, MILESTONE_MIN_PCT,
          MIN_KERNEL_TASKS, DISPATCHED_SO_FAR: dispatched, BELOW_MIN_FLOOR: belowFloor,
          PROFILE_TOPN: profile ? profile.profile_topN_json : '', HISTORY: history, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Milestone', label: `architect:plan m${milestone}`, schema: PLAN_SCHEMA });

  }

  const planCandsRaw = (plan && plan.kernel_candidates) ? plan.kernel_candidates : [];
  // AgentX keeps the skill-selected inventory; only execution budgets apply here.
  // Fixed-load pct_gpu_time gate (overrides the floor): only kernels at or above MILESTONE_MIN_PCT;
  // a missing pct is kept.
  const planCands = KERNEL_EXTRACT_ON ? planCandsRaw : planCandsRaw.filter(c => c.pct_gpu_time == null || c.pct_gpu_time >= MILESTONE_MIN_PCT);
  const skipped = KERNEL_EXTRACT_ON ? [] : planCandsRaw.filter(c => c.pct_gpu_time != null && c.pct_gpu_time < MILESTONE_MIN_PCT);
  if (skipped.length) log(`Milestone ${milestone}: skipped ${skipped.length} kernel(s) below ${MILESTONE_MIN_PCT}% GPU [${skipped.map(c => `${c.short_name || '?'}@${(+c.pct_gpu_time).toFixed(1)}%`).join(', ')}].`);
  if (!planCands.length) {
    if (KERNEL_EXTRACT_ON) log(`Milestone ${milestone}: prepared UT queue exhausted; stopping.`);
    else if (planCandsRaw.length) log(`Milestone ${milestone}: stop — no remaining kernel clears the ${MILESTONE_MIN_PCT}% GPU bar (Amdahl: sub-threshold kernels can't move e2e). Floor is overridden by the pct gate.`);
    else if (belowFloor) log(`Milestone ${milestone}: below floor (${dispatched}/${MIN_KERNEL_TASKS}) but Architect nominated nothing — cannot fabricate candidates; stopping.`);
    else log(`Milestone ${milestone}: stop (floor ${MIN_KERNEL_TASKS} met). ${plan ? plan.reasoning || '' : ''}`);
    break;
  }

  const cands = planCands.slice(0, remaining).map((c, i) => ({
    ...c, idx: i, gpu_id: GPU_LIST[i % GPU_LIST.length],
    short_name: c.short_name || `k${milestone}_${i}`,
  }));
  // AgentX selection inventories more work than this execution budget can run.
  // Preserve undispatched eligible candidates in carried state for resume.
  if (KERNEL_EXTRACT_ON) kernelQueue = planCands.slice(cands.length);
  dispatched += cands.length;
  log(`Milestone ${milestone}: ${cands.length} kernel candidate(s) [${cands.map(c => c.short_name).join(', ')}], dispatched ${dispatched}/${BUDGET} (floor ${MIN_KERNEL_TASKS})`);

  // --- (b) PARALLEL optimize (extract + kernel layer per candidate, on distinct GPUs), then SERIAL
  // integrate: e2e A/Bs must not overlap, and accepted overlays compound in order.
  const optimized = await parallel(cands.map((c) => async () => {
    const ext = await extractWithBaseline(
      'kernel_extractor', 'extract', 'Capture shapes + oracle; emit an immutable unittest task dir.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: c.gpu_id, WORKLOAD, KERNEL: c,
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
      },
      { phase: 'Milestone', label: `extract ${c.short_name}`, schema: EXTRACT_SCHEMA });
    if (!ext || ext.editable === false || ext.unittest_smoke !== 'pass' || !ext.task_dir) {
      return { c, skip: true, reason: `extraction failed/non-editable (${extractFailWhy(ext)})` };
    }
    // RECURSIVE kernel layer on the IMMUTABLE task dir (one allowed nesting level via workflow()).
    let kl;
    try {
      const r = await reserveBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
        kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
        use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
        ...KB_ARGS,
        budget: KERNEL_BUDGET, gpu_ids: c.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
        task: 'Compare candidate backends ' + JSON.stringify(c.candidate_backends || []) +
          ' for this kernel; pick the fastest that passes the immutable unittest. ' + GRAPH_REQ + (TASK || ''),
        apply_to_original: 'false',
      }, String(c.short_name || ext.op_kind || 'milestone'));
      if (r == null && IS_AGENTX) {
        return { c, skip: true, paused: true, reason: 'kernel lane not run to completion before the final reserve' };
      }
      kl = { ran: true, kernel_eval_dir: r.eval_dir, final_patch: r.final_patch,
        final_geomean: r.final_geomean,
        isolated_speedup: KERNEL_EXTRACT_ON ? (r.final_weighted ?? r.final_speedup ?? 0) : r.final_geomean,
        validation_status: r.validation_status,
        note: (r.winner && r.winner.source) || '' };
    } catch (e) {
      kl = { ran: false, final_patch: '', final_geomean: 0, isolated_speedup: 0, validation_status: 'error', note: String(e) };
    }
    return { c, ext, kl };
  }));

  // --- serial integrate (compounding overlay, isolated measurement) ---
  let milestoneImproved = false;
  for (const o of optimized) {
    if (!o) continue;
    const c = o.c;
    if (o.skip) {
      log(`  ${c.short_name}: ${o.reason}; skipping.`);
      history.ledger.push({ direction: c.short_name, verdict: o.paused ? 'paused' : 'dead_end', lesson: o.reason });
      continue;
    }
    const { ext, kl } = o;
    if (!kl || !kl.ran || !(kl.isolated_speedup > 1.0) || !kl.final_patch) {
      log(`  ${c.short_name}: kernel layer produced no speedup (${kl ? kl.isolated_speedup : '?'}x); skipping integrate.`);
      history.ledger.push({ direction: c.short_name, isolated_speedup: kl ? kl.isolated_speedup : 0, verdict: 'dead_end', lesson: 'no isolated speedup' });
      continue;
    }
    log(`  ${c.short_name}: kernel layer ${kl.isolated_speedup.toFixed(2)}x isolated. Integrating to e2e.`);
    // Build inputs ONCE so Fix C can re-issue the SAME A/B for a pending win at Finalize.
    const mileIntegrateInputs = {
      EVAL_DIR, MODEL_PATH, GPU_ID: c.gpu_id, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
      KERNEL_RESULT: { short_name: c.short_name, task_dir: ext.task_dir,
        source_path_in_sglang: ext.source_path_in_sglang, target_callable: ext.target_callable,
        final_patch: kl.final_patch, verified_isolated_speedup: kl.isolated_speedup, pct_gpu_time: c.pct_gpu_time },
      CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
      CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
    };
    const integ = await runIntegrateBothLegs(
      'Overlay the optimized kernel back; gate on e2e throughput.', mileIntegrateInputs,
      `integrate ${c.short_name}`, 'Milestone');

    // Three-state gate: null (timeout/hang) or an explicit incomplete => PENDING, not rejected.
    const abDone = !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);
    if (abDone && integAccepted(integ, c.pct_gpu_time, kl.isolated_speedup) && integ.e2e_throughput_tok_s > curTput) {
      curOverlay = integ.accepted_overlay || curOverlay;
      curTput = integ.e2e_throughput_tok_s;
      bankAccepted(acceptedKernels, { short_name: c.short_name, backend: kl.note || '', kind: 'patch', ...e2eFrom(integ), isolated: kl.isolated_speedup }, krOf(mileIntegrateInputs));
      milestoneImproved = true;
      log(`  ${c.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: c.short_name, isolated_speedup: kl.isolated_speedup, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else {
      // As in the head track: a fixable reject or warmup crash earns a corrective re-author (not
      // charged to HEAD_BUDGET); a non-fixable incomplete A/B stays PENDING.
      const reason = gateRejectReason(integ, c.pct_gpu_time, kl.isolated_speedup);
      const corr = (rejectClass(reason) !== '')
        ? await tryCorrectiveReauthor({
            short_name: c.short_name, op_kind: ext.op_kind, shapes: ext.shapes, dtype: ext.dtype, regime: c.regime,
            gpu_id: c.gpu_id, kernel_eval_dir: kl.kernel_eval_dir, task_dir: ext.task_dir, language: kl.language || '',
            isolated: kl.isolated_speedup, base_inputs: mileIntegrateInputs, reason, fix_class: rejectClass(reason), pct_gpu_time: c.pct_gpu_time, phase_name: 'Milestone',
            cur: { overlay: curOverlay, flags: curFlags, env: curEnv, tput: curTput },
          })
        : { banked: false };
      if (corr.banked) {
        curOverlay = corr.integ.accepted_overlay || curOverlay; curTput = corr.integ.e2e_throughput_tok_s;
        bankAccepted(acceptedKernels, { short_name: c.short_name, backend: kl.note || '', kind: 'patch', ...e2eFrom(corr.integ), isolated: corr.isolated, corrective: true, patch: corr.patch || '' }, krOf(mileIntegrateInputs));
        milestoneImproved = true;
        log(`  ${c.short_name}: ACCEPTED after corrective re-author (${reason}). e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
        history.ledger.push({ direction: c.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${reason}` });
      } else if (!abDone) {
        pendingIntegrations.push({ track: 'milestone', short_name: c.short_name, isolated: kl.isolated_speedup,
          pct_gpu_time: c.pct_gpu_time, inputs: mileIntegrateInputs, backend: kl.note || '',
          partial: integ ? { gate: integ.gate, ref_med: integ.ref_med, cand_med: integ.cand_med, reason: integ.reason } : null });
        log(`  ${c.short_name}: INTEGRATE INCOMPLETE — A/B not finished (${integ ? integ.reason || integ.gate : 'null/timeout'}); kept as PENDING (not a rejection).`);
        history.ledger.push({ direction: c.short_name, isolated_speedup: kl.isolated_speedup, verdict: 'incomplete', lesson: integ ? integ.reason || 'A/B not finished' : 'integrate timed out/null before A/B completed' });
      } else {
        log(`  ${c.short_name}: REJECTED at e2e gate (${reason}).`);
        history.ledger.push({ direction: c.short_name, isolated_speedup: kl.isolated_speedup, e2e_delta_pct: integ.e2e_delta_pct || 0, verdict: 'dead_end', lesson: reason || 'no e2e gain' });
      }
    }
  }

  // --- (c) If improved: re-profile
  if (milestoneImproved) {
    noImprove = 0;
    if (!KERNEL_EXTRACT_ON) {
      profile = await safeAgent(
        roleAgent('profiler', 'reprofile', 'Re-profile the new best server.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: milestone,
          OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          ...ANALYSIS_SKILL_INPUTS,
        }),
        { phase: 'Profile', label: `profiler:reprofile m${milestone}`, schema: PROFILE_SCHEMA });
    }
  } else {
    noImprove++;
  }

  // --- (d) Update the persistent experience library + in-run memory
  const exp = E2E_LEARNED_KB_ENABLED ? await safeAgent(
    roleAgent('system_architect', 'update_experience', 'Curate knowledge/learned/ (merge/insert >=2-star / archive contradicted) per learned/README.md.', {
      ROUND: milestone, EVAL_DIR, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
      MILESTONE_RESULTS: history.ledger.slice(-cands.length),
      REPROFILE_SHIFT: profile ? profile.shift_note : '', PRIOR_HISTORY: history,
      ...ANALYSIS_SKILL_INPUTS,
    }),
    { phase: 'Milestone', label: `architect:experience m${milestone}`, schema: EXPERIENCE_SCHEMA })
    : null;
  if (exp) {
    if (exp.insights) history.insights = exp.insights;
    if (exp.bottleneck_now) history.bottleneck_now = exp.bottleneck_now;
    if (exp.suggest_next) history.suggest_next = exp.suggest_next;
  }
  history.milestones.push({ milestone, accepted: acceptedKernels.length, throughput: curTput, improved: milestoneImproved });
  log(`Milestone ${milestone} done. throughput=${curTput} tok/s (${(curTput / BASELINE_TPUT).toFixed(3)}x), noImprove=${noImprove}`);
}

// ===========================================================================
// PHASE: Finalize + Report + Validate  (gated)
// ===========================================================================
let allAccepted = acceptedHeads.concat(acceptedKernels);
let finalize = null, report = null, validation = null;
let finalTput = curTput, finalSpeedup = BASELINE_TPUT ? curTput / BASELINE_TPUT : 1.0;
let validatedOk = false;   // did the independent Validate produce a usable (positive) number?
let validateSkip = '', finalFromAb = false;

// --- Fix C: EVERY incomplete A/B must be finished (both legs) before final ----
// A reference-only A/B is not a result. (1) Reconstruct from disk (overlay/cand_*/
// integrate_result.json) every A/B that never completed, including from a killed or resumed run;
// (2) finish each pending one (runIntegrateBothLegs runs the missing leg). Bounded by
// AB_FINISH_RETRIES, agentBounded and TIME_FINAL_DEADLINE_HIT: never spends the final reserve.
if (want('final')) {
  // Reclaim leftover process-local capture dirs before Finalize uses more disk; runs even past
  // TIME_FINAL_DEADLINE_HIT (cheap local FS work, not a bench).
  try {
    const { execFileSync } = require('child_process');
    const out = execFileSync('python3', [
      `${WORKFLOW_DIR}/scripts/capture_shapes.py`,
      '--reclaim-workspace', EVAL_DIR,
      '--workspace-budget', CAPTURE_WORKSPACE_BUDGET,
    ], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 300000 });
    log(`Finalize: capture workspace reclaim done (${CAPTURE_WORKSPACE_BUDGET} budget).`);
    try {
      const tel = JSON.parse(out);
      if (tel.over_budget) {
        log(`Finalize: WARNING authoritative oracles over workspace budget ` +
          `(${tel.authoritative_oracle_bytes} bytes > ${CAPTURE_WORKSPACE_BUDGET}).`);
      }
    } catch (_) { /* telemetry parse best-effort */ }
  } catch (reclaimErr) {
    log(`Finalize: capture workspace reclaim skipped (${reclaimErr && reclaimErr.message ? reclaimErr.message : reclaimErr})`);
  }
  // (0) Past the final-reserve boundary skip the whole gate (scan included); pending entries reach
  // the caller via pendingIntegrations: deferred, not discarded.
  if (TIME_FINAL_DEADLINE_HIT) {
    log(`Finalize-gate: SKIPPED entirely — the final reserve has already begun. ` +
      `${pendingIntegrations.length} pending A/B(s) deferred to the caller (surfaced in pending_integrations).`);
  } else {
  // (1) Disk-reconstruct incomplete A/Bs via a read-only agent (the JS has no fs).
  const RECON_SCHEMA = obj({ incomplete: { type: 'array', items: obj({
    short_name: { type: 'string' }, overlay_dir: { type: 'string' },
    winner_kind: { type: 'string' }, apply_env: { type: 'string' }, apply_flags: { type: 'string' },
    op_kind: { type: 'string' }, target_callable: { type: 'string' },
    isolated: { type: 'number' }, pct_gpu_time: { type: 'number' },
    ref_present: { type: 'boolean' }, cand_present: { type: 'boolean' },
  }, ['short_name', 'overlay_dir']) } }, ['incomplete']);
  const scan = await safeAgent(
    `You are a read-only filesystem scanner. Do NOT run any benchmark, server, or A/B. ` +
    `Enumerate the directories ${EVAL_DIR}/overlay/cand_* and (if it exists) ` +
    `${EVAL_DIR}/final/overlay/cand_*. For EACH such cand_<name> dir, read its ` +
    `integrate_result.json if present. Classify the A/B as INCOMPLETE when: the file is ` +
    `MISSING, OR gate=="incomplete", OR (ab_complete is false/absent AND there is no usable ` +
    `cand/bench_summary.json in the dir). A summary is usable only when status=="complete" and ` +
    `usable_for_acceptance==true. Treat it as COMPLETE (skip it) when ab_complete:true ` +
    `OR gate is one of {accepted,stack,rejected} AND the candidate summary is usable. ` +
    `For each INCOMPLETE dir, emit one object with: short_name (the dir name minus the ` +
    `"cand_" prefix), overlay_dir (absolute path), and from integrate_result.json (use "" / ` +
    `null when absent): winner_kind, apply_env, apply_flags, op_kind, ` +
    `target_callable (or target_file), isolated (= isolated_speedup), pct_gpu_time, plus ` +
    `ref_present (true only if ref/bench_summary.json is usable) and cand_present (true only if ` +
    `cand/bench_summary.json is usable). Return ONLY compact JSON {"incomplete":[...]} (empty array if none).`,
    { phase: FINALIZE_GATE_PHASE, label: 'scan-incomplete-ab', schema: RECON_SCHEMA });
  const known = new Set(pendingIntegrations.map((p) => p.short_name));
  for (const it of ((scan && scan.incomplete) || [])) {
    if (!it || !it.short_name || known.has(it.short_name)) continue;
    known.add(it.short_name);
    pendingIntegrations.push({
      short_name: it.short_name, track: 'head', winner_kind: it.winner_kind || '',
      apply_env: it.apply_env || '', apply_flags: it.apply_flags || '',
      op_kind: it.op_kind || '', backend: '', isolated: it.isolated || 0,
      inputs: {
        EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND,
        KERNEL_RESULT: {
          short_name: it.short_name, op_kind: it.op_kind || '', winner_kind: it.winner_kind || '',
          target_callable: it.target_callable || '', apply_env: it.apply_env || '',
          apply_flags: it.apply_flags || '', code_patch: '', tuning_artifact: '',
          verified_isolated_speedup: it.isolated || 0, pct_gpu_time: it.pct_gpu_time,
          parity_note: 'expected_close',
        },
        // The overlay is already built: the integrator benches it directly (RESUME_AB), no rebuild.
        CAND_OVERLAY_DIR: it.overlay_dir, RESUME_AB: true, SKILL_DIR: WORKFLOW_DIR,
      },
    });
    log(`Finalize-gate: disk-reconstructed incomplete A/B "${it.short_name}" ` +
      `(winner_kind=${it.winner_kind || '?'}, ref=${!!it.ref_present}, cand=${!!it.cand_present}); queued to finish.`);
  }

  // (2) Finish every pending A/B, never past the reserve boundary; highest isolated speedup first.
  pendingIntegrations.sort((a, b) => (b.isolated || 0) - (a.isolated || 0));
  const stillIncomplete = [];
  while (pendingIntegrations.length) {
    // Break, not shift, so unfinished entries stay in pendingIntegrations for wfReturn.
    if (TIME_FINAL_DEADLINE_HIT) {
      log(`Finalize-gate: STOPPING with ${pendingIntegrations.length} A/B(s) unfinished ` +
        `(${pendingIntegrations.map((q) => q.short_name).join(', ')}) — the final reserve has begun and ` +
        `Finalize/Report/Validate must run. Deferred, not discarded: they are surfaced in pending_integrations.`);
      break;
    }
    if (abWindowShort(abLegs(pendingIntegrations[0].inputs), `Finalize-gate: ${pendingIntegrations[0].short_name}`)) {
      log(`Finalize-gate: STOPPING with ${pendingIntegrations.length} A/B(s) unfinished ` +
        `(${pendingIntegrations.map((q) => q.short_name).join(', ')}): the next cannot finish before the final ` +
        'reserve. Deferred to pending_integrations.');
      break;
    }
    const p = pendingIntegrations.shift();   // pop one per iteration => guaranteed termination
    log(`Finalize-gate: finishing pending A/B (${p.short_name}, ${(p.isolated || 0).toFixed(2)}x isolated); ` +
      `${pendingIntegrations.length} more queued.`);
    const integ = await runIntegrateBothLegs(
      'Finish the e2e A/B for this verified-isolated win. Run BOTH legs (reference + candidate) to ' +
      'completion, then return accepted/stack/rejected with ab_complete:true.',
      // Measured against the latest accepted baseline (current overlay/flags/env/throughput).
      { ...p.inputs, CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_THROUGHPUT: curTput },
      `finish-integrate ${p.short_name}`, FINALIZE_GATE_PHASE);
    if (abDone(integ) && integAccepted(integ, p.pct_gpu_time, p.isolated) && integ.e2e_throughput_tok_s > curTput) {
      curOverlay = integ.accepted_overlay || curOverlay;
      if (p.track === 'head') {
        if (p.winner_kind === 'env' && p.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + p.apply_env;
        if (p.winner_kind === 'flag' && p.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + p.apply_flags;
        bankAccepted(acceptedHeads, { short_name: p.short_name, op_kind: p.op_kind, backend: p.backend, kind: p.winner_kind, ...e2eFrom(integ), isolated: p.isolated, pct_gpu_time: p.pct_gpu_time }, krOf(p.inputs));
      } else {
        bankAccepted(acceptedKernels, { short_name: p.short_name, backend: p.backend || '', kind: 'patch', ...e2eFrom(integ), isolated: p.isolated, pct_gpu_time: p.pct_gpu_time }, krOf(p.inputs));
      }
      curTput = integ.e2e_throughput_tok_s;
      finalTput = curTput; finalSpeedup = BASELINE_TPUT ? curTput / BASELINE_TPUT : 1.0;
      log(`Finalize-gate: pending win ${p.short_name} ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: p.short_name, isolated_speedup: p.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: 'finished at finalize-gate' });
    } else if (abDone(integ)) {
      log(`Finalize-gate: pending win ${p.short_name} REJECTED after a COMPLETED A/B (${integ.reason || integ.gate}).`);
      history.ledger.push({ direction: p.short_name, isolated_speedup: p.isolated, e2e_delta_pct: integ.e2e_delta_pct || 0, verdict: 'dead_end', lesson: integ.reason || 'no e2e gain' });
    } else {
      stillIncomplete.push(p.short_name);
      log(`Finalize-gate: pending win ${p.short_name} A/B STILL incomplete after ${AB_FINISH_RETRIES} finish-retries ` +
        `(${integ ? integ.reason || integ.gate : 'null/timeout'}); hard fault — surfacing it.`);
      history.ledger.push({ direction: p.short_name, isolated_speedup: p.isolated, verdict: 'incomplete', lesson: integ ? integ.reason || 'A/B could not complete both legs' : 'integrate degraded/timed out' });
    }
  }
  if (stillIncomplete.length) log(`Finalize-gate: ${stillIncomplete.length} A/B(s) could not complete both legs even after retries: ${stillIncomplete.join(', ')}.`);
  }
}
allAccepted = acceptedHeads.concat(acceptedKernels);   // refresh after Fix C may have banked a pending win
if (allAccepted.length) {
  await requireE2EValidationCheckpoint('overlay/accepted_stack/e2e_validation.json', {
    phase: 'Overlay', validation_level: 'integrator', gate: 'accepted',
    validation_status: 'accepted_intermediate',
    baseline_throughput_tok_s: BASELINE_TPUT, final_throughput_tok_s: curTput,
    throughput_speedup: BASELINE_TPUT ? curTput / BASELINE_TPUT : 1,
    baseline_config: { flags: INIT_FLAGS, env: INIT_ENV },
    accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
    accepted_kernels: acceptedKernels, accepted_heads: acceptedHeads, final_patch: [],
    final_overlay: { path: curOverlay },
    final_launch_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
    bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
    measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
      workload: WORKLOAD, source_artifact: `${EVAL_DIR}/overlay`,
      acceptance: { gain_exceeds_noise: true, correctness_passed: true, noise_floor_pct: NOISE_BAND } },
    stack: { kernel_slots: allAccepted.map((k) => ({
      kernel_slot: k.target_callable || k.short_name || k.name || '',
      selected: true, candidate_id: k.short_name || k.name || '',
    })) },
    replay: { asset_paths: [`${EVAL_DIR}/bench_e2e.sh`, curOverlay].filter(Boolean) },
    integrity: { checkpoint_assets: [] },
  });
}
// The reserve boundary abandons a search agent's promise, not the agent, so detached search work may
// still hold GPUs. Finalize and Validate each start with run_reaper.py (selects by this run's marker,
// not by name), bounded on its own clock; a failed fence is non-fatal.
const RUN_REAPER = `${WORKFLOW_DIR}/scripts/run_reaper.py`;
const FENCE_TIMEOUT_MS = 10 * 60000;
const FENCE_SCHEMA = obj({ ok: { type: 'boolean' }, stage: { type: 'string' }, stopped: arrObj,
  survivors: arrObj, note: { type: 'string' } }, []);
async function finalPhaseFence(stage, phaseName) {
  if (!EVAL_DIR) return null;
  const cmd = `python3 ${shq(RUN_REAPER)} fence --eval-dir ${shq(EVAL_DIR)} --stage ${stage}`;
  const r = await safeAgent(
    `You are the final-phase GPU fence. Run EXACTLY this command, once, and return its JSON stdout verbatim ` +
    `as StructuredOutput; it pretty-prints over several lines, so return the whole object. It is this ` +
    `workflow's own cleanup and the sanctioned way to do what PROCESS SAFETY asks — stop only what this run ` +
    `started: it selects processes by this run's marker and eval dir, never by a name pattern, and never ` +
    `signals PID 1, the caller, or your own shell. Do not edit it, retry it, or act on what it reports; a ` +
    `non-zero exit is a valid answer, so return what it printed.\n` +
    '```bash\n' + cmd + '\n```',
    { phase: phaseName, label: `fence:${stage}`, schema: FENCE_SCHEMA }, 1, () => FENCE_TIMEOUT_MS);
  if (!r) {
    log(`[final-phase fence] ${stage}: the fence returned nothing — continuing without it (NON-FATAL).`);
    return null;
  }
  const stopped = Array.isArray(r.stopped) ? r.stopped : [];
  const survivors = Array.isArray(r.survivors) ? r.survivors : [];
  const names = [...new Set(stopped.map((p) => String((p && p.name) || (p && p.pid) || '?')))].slice(0, 8);
  log(`[final-phase fence] ${stage}: stopped ${stopped.length} process(es) still running from the search` +
    `${names.length ? ` (${names.join(', ')})` : ''}${survivors.length ? `; ${survivors.length} still up` : ''}` +
    `${r.note ? ` — ${r.note}` : ''}.`);
  return r;
}

if (want('final')) {
  if (TIME_BUDGET_MS != null) log(`[time-budget] entering the final phase with ~${remainingMin()}min of the ${Math.round(TIME_BUDGET_MS / 60000)}min budget left (reserve was ${Math.round(FINAL_RESERVE_MS / 60000)}min).`);
  FINAL_PHASE_STARTED = true;   // the one place the reserve exemption is granted — by position, not label
  phase('Finalize');
  await finalPhaseFence('finalize', 'Finalize');
  finalize = await safeAgent(
    roleAgent('e2e_integrator', 'finalize', 'Assemble the final overlay + patch + launch script bundle.' + agentxFinalizeNote(), {
      EVAL_DIR, FINAL_OVERLAY: curOverlay, ACCEPTED_FLAGS: curFlags, ACCEPTED_ENV: curEnv,
      ACCEPTED_KERNELS: allAccepted, BASELINE_THROUGHPUT: BASELINE_TPUT, SKILL_DIR: WORKFLOW_DIR,
      ...TUNING_FINALIZE_INPUTS,
    }),
    { phase: 'Finalize', label: 'e2e_integrator:finalize', schema: FINALIZE_SCHEMA });
  finalTput = (finalize && finalize.final_throughput_tok_s) || curTput;
  // The stock bundle's number is Setup's canonical one, not Finalize's search leg.
  const adoptedNothing = agentxAdoptedNothing({ setup: SETUP_CONFIG, accepted: allAccepted.length, tput: curTput,
    baseline: BASELINE_TPUT, flags: curFlags, env: curEnv, overlay: curOverlay });
  if (adoptedNothing) finalTput = BASELINE_TPUT;
  if (finalize && finalTput > BASELINE_TPUT * (1 + NOISE_BAND / 100)) {
    await requireE2EValidationCheckpoint('final/e2e_validation.json', {
      phase: 'Finalize', validation_level: 'final_pair', gate: 'accepted',
      validation_status: 'provisional_final_pair',
      baseline_throughput_tok_s: BASELINE_TPUT, final_throughput_tok_s: finalTput,
      throughput_speedup: BASELINE_TPUT ? finalTput / BASELINE_TPUT : 1,
      baseline_config: { flags: INIT_FLAGS, env: INIT_ENV },
      accepted_config: { flags: curFlags, env: curEnv, effective_config_digest: EFFECTIVE_CONFIG_DIGEST },
      accepted_kernels: acceptedKernels, accepted_heads: acceptedHeads,
      final_patch: [finalize.final_patch].filter(Boolean),
      final_overlay: { path: finalize.final_overlay || curOverlay },
      final_launch_script: { path: finalize.final_launch_script || '' },
      bench_script: { path: `${EVAL_DIR}/bench_e2e.sh` },
      measurement: { measurement_mode: MEASUREMENT_MODE, metric_basis: 'aggregate_output_tok_s',
        workload: WORKLOAD, source_artifact: `${EVAL_DIR}/final/bench/bench_summary.json`,
        acceptance: { gain_exceeds_noise: true, correctness_passed: true, noise_floor_pct: NOISE_BAND } },
      stack: { kernel_slots: allAccepted.map((k) => ({
        kernel_slot: k.target_callable || k.short_name || k.name || '',
        selected: true, candidate_id: k.short_name || k.name || '',
      })) },
      replay: { asset_paths: [finalize.final_patch, finalize.final_overlay,
        finalize.final_launch_script, `${EVAL_DIR}/bench_e2e.sh`].filter(Boolean) },
      integrity: { checkpoint_assets: [] },
    });
  }

  phase('Report');
  report = await safeAgent(
    roleAgent('system_architect', 'report',
      'Write architect_report.md AND the full final_report.md in English (with the Phases tree + ' +
      'artifacts tree modules). final_report.md MUST contain a "## Knowledge-base recall" section ' +
      'built from KB_RECALL — see your role file. Write it even when nothing was recalled: report ' +
      'the exact canonical ids that were tried and the read_reason. On an exact-lookup store a miss ' +
      'and a never-recorded page are the same 404, so the address asked is the finding, and a reader ' +
      'who cannot see it cannot tell "no prior art" from "prior art one segment away". Zero candidates ' +
      'is NOT the same as an empty page: report KB_RECALL.e2e.curation (scanned/retired/' +
      'same_direction_collapsed/below_min_speedup, and its own read_plane) with both floors.', {
      EVAL_DIR, HISTORY: history, BASELINE_THROUGHPUT: BASELINE_TPUT, FINAL_THROUGHPUT: finalTput,
      KB_RECALL,
      ACCEPTED_CONFIG: { flags: curFlags, env: curEnv }, ACCEPTED_KERNELS: allAccepted,
      ACCEPTED_HEADS: acceptedHeads, FLAGGED_HEADS: flaggedHeads, MILESTONES: milestone, BUDGET_USED: dispatched, BUDGET, MIN_KERNEL_TASKS,
      PROFILE_TOPN: profile ? profile.profile_topN_json : '', WORKLOAD, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
      ...ANALYSIS_SKILL_INPUTS, ...TUNING_REPORT_INPUTS,
    }),
    { phase: 'Report', label: 'architect:report', schema: REPORT_SCHEMA });

  // Validate is sized into FINAL_RESERVE_MS and not time-gated (skips only per agentxValidateSkip);
  // if a hard kill lands mid-bench, Report's files and run_e2e.py's fallback
  // (director_e2e_validation.json, then the best overlay's integrate_result.json) still carry a win.
  phase('Validate');
  validateSkip = agentxValidateSkip(adoptedNothing, remainingMs());
  if (validateSkip === 'final_is_baseline') {
    log(`Validate SKIPPED: nothing adopted, so the final bundle is the stock one (Setup's ${BASELINE_TPUT} tok/s).`);
  } else if (validateSkip === 'no_time') {
    log(`Validate SKIPPED: ~${remainingMin()}min left, under its ${Math.round(AGENTX_VALIDATE_WINDOWS_MS / 60000)}min of timed windows.`);
  } else {
    await finalPhaseFence('validate', 'Validate');
  }
  validation = validateSkip ? null : await safeAgent(
    roleAgent('director', 'validate', 'Independently re-measure throughput + parity; arbitrate; then reconcile the report with the validated numbers.' +
      agentxClaimNote(finalTput, BASELINE_TPUT), {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], BASELINE_THROUGHPUT: BASELINE_TPUT, NOISE_BAND_PCT: NOISE_BAND,
      BASELINE_OVERLAY: INIT_BASE_OVERLAY,
      FINAL_OVERLAY: (finalize && finalize.final_overlay) || curOverlay,
      FINAL_FLAGS: { flags: curFlags, env: curEnv },
      CLAIMED_THROUGHPUT: finalTput, WORKLOAD, APPLY_TO_ORIGINAL,
      MEASUREMENT_MODE: VALIDATION_MEASUREMENT_MODE,
      MEASUREMENT_PURPOSE: 'validation', REPLICAS: VALIDATION_SAMPLES,
      SKILL_DIR: WORKFLOW_DIR, ...ACCURACY_INPUTS,
      // Report wrote these before Validate ran: the Director rewrites their headline numbers (and
      // status/parity) to its own, so they never disagree. Defaults are the standard EVAL_DIR names.
      ARCHITECT_REPORT: (report && report.report_path) || `${EVAL_DIR}/architect_report.md`,
      FINAL_REPORT: `${EVAL_DIR}/final_report.md`,
    }),
    { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });
  // Trust the Director's number only when it is a real positive measurement; a Validate with no
  // usable number never erases the accepted A/B win carried in finalTput.
  validatedOk = !!(validation && validation.director_verified_throughput_tok_s > 0 && validation.throughput_speedup > 0);
  finalSpeedup = validatedOk ? validation.throughput_speedup : (BASELINE_TPUT ? finalTput / BASELINE_TPUT : finalSpeedup);
  // A search leg over a canonical baseline is no speedup: run_e2e.py reports the accepted A/B's own pair.
  finalFromAb = IS_AGENTX && !validatedOk && allAccepted.length > 0;
  log(`COMPLETE. ${MODEL_NAME}: ${BASELINE_TPUT} -> ${validatedOk ? validation.director_verified_throughput_tok_s : finalTput} tok/s ` +
    `(${finalSpeedup ? finalSpeedup.toFixed(3) : '?'}x, status ${validationStatusOf()}` +
    `${validation && !validatedOk ? '; Validate produced no number — using carried accepted A/B win' : ''}). Results in ${EVAL_DIR}`);
  if (finalFromAb) {
    log("Validate gave no number: returning no final or speedup, so run_e2e.py reports the accepted A/B's own pair.");
  }

  // --- FINAL experience reconciliation (post-gate) ----
  // Cards written as "pending" before the Finalize-gate (Fix C) and Validate confirmed them get the
  // verified post-Validate numbers here, in one re-curate.
  if (allAccepted.length && E2E_LEARNED_KB_ENABLED) {
    const verifiedTput = validatedOk ? validation.director_verified_throughput_tok_s : finalTput;
    await safeAgent(
      roleAgent('system_architect', 'update_experience',
        'FINAL RECONCILIATION (post-gate): re-curate knowledge/learned/ per learned/README.md. For every ' +
        'entry in ACCEPTED_HEADS / ACCEPTED_KERNELS whose card was written earlier THIS run but still reads ' +
        '"Integrator pending" / "e2e gate pending" (or carries no e2e number), MERGE the VERIFIED gated ' +
        'result into that card: set effect from the entry\'s e2e_delta_pct plus the run FINAL_THROUGHPUT / ' +
        'FINAL_SPEEDUP / VALIDATION_STATUS / OUTPUT_PARITY, and raise confidence per learned/README.md now ' +
        'that the e2e gate passed. Do NOT blind-append and do NOT touch rejected/dead-end cards.', {
        ROUND: 'final', FINAL_GATE: true, EVAL_DIR, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
        ACCEPTED_HEADS: acceptedHeads, ACCEPTED_KERNELS: acceptedKernels,
        BASELINE_THROUGHPUT: BASELINE_TPUT, FINAL_THROUGHPUT: verifiedTput, FINAL_SPEEDUP: finalSpeedup,
        VALIDATION_STATUS: validatedOk ? validation.validation_status
          : (validation ? validation.validation_status || 'flagged' : 'unknown'),
        OUTPUT_PARITY: validation ? validation.output_parity : 'unknown',
        MILESTONE_RESULTS: history.ledger.filter((e) => e && e.lesson === 'finished at finalize-gate'),
        PRIOR_HISTORY: history,
        ...ANALYSIS_SKILL_INPUTS,
      }),
      { phase: 'Validate', label: 'architect:experience final', schema: EXPERIENCE_SCHEMA });
    log('Finalize: re-curated knowledge/learned/ with verified post-gate numbers (pending -> verified).');
  }
} else {
  log(`Phase(s) [${PHASES.join(',')}] done. Carried throughput ${curTput} tok/s. Pass the returned 'state' to the next phase invocation.`);
}

// State to carry into the NEXT phase invocation (args.state) when driving phase-by-phase.
const carryState = {
  backend: BACKEND,
  ...(KERNEL_EXTRACT_ON
    ? { workload_spec: { ...WORKLOAD_SPEC, kind: WORKLOAD_KIND },
        profile_workload_json: profile ? profile.profile_workload_json || '' : '',
        kernel_ranking_json: profile ? profile.kernel_ranking_json || '' : '',
        kernel_extraction_delivery_json: profile ? profile.kernel_extraction_delivery_json || '' : '',
        kernel_extraction_summary: {
          failed_tasks: profile?.failed_tasks || [], deferred_tasks: profile?.deferred_tasks || [],
          coverage_summary: profile?.coverage_summary || {},
        } } : {}),
  eval_dir: EVAL_DIR, model_name: MODEL_NAME, baseline_throughput_tok_s: BASELINE_TPUT,
  noise_band_pct: NOISE_BAND, flags: curFlags, env: curEnv, overlay: curOverlay, throughput: curTput,
  profile_topn_json: profile ? profile.profile_topN_json : '',
  config_directions: (strategy && strategy.config_directions) || [],
  // AgentX candidates carry prepared_ut.result (dispatch cannot re-read it from disk), so an
  // undispatched task survives a resume (references/workflow-handoff.md).
  headQueue, kernelQueue, accepted_heads: acceptedHeads, flagged_heads: flaggedHeads, accepted_kernels: acceptedKernels,
  // Full tuning result: a phase-by-phase resume neither re-runs tuning nor loses its attribution.
  ...(tuning ? { tuning } : {}),
  // Pending wins WITH their inputs, so a resumed run can finish their A/B.
  pending_integrations: pendingIntegrations,
  history,
};

// What this run got from the KB, and separately what it added (a warm start's headline speedup is
// partly inherited). BASELINE_TPUT stays the true cold baseline throughout.
const kbWarmStart = (E2E_WARM_START_ON && KB_DIMS) ? {
  identity: KB_DIMS, plane: E2E_KB_PLANE, read_plane: KB_READ_PLANE, mode: E2E_WARM_START,
  reference_dir: KB_REF_DIR,
  adopted_throughput_tok_s: kbSeedTput || null,
  adopted_kernels: kbSeedKernels.map(k => k.name).filter(Boolean),
  // This run's own gain past the warm start; null when nothing was adopted.
  incremental_speedup: kbSeedTput ? Number((finalTput / kbSeedTput).toFixed(6)) : null,
  // The full recall ledger (both planes), the same object the Report role was handed.
  recall: KB_RECALL,
} : null;
// Attribution for the standalone tuning phase (its own pre/post legs): delta% and its SHARE of the
// run's total gain (null, not 0, when there is no net gain to apportion).
function tuningReturn() {
  if (!TUNING_SKILLSET_ENABLED) return { enabled: false };
  const base = {
    enabled: true, skillset_dir: TUNING_SKILLSET_DIR, kb_enabled: TUNING_KB_ENABLED, ran: !!tuning,
  };
  if (!tuning) return { ...base, gate: want('tune') ? 'not_run' : 'phase_not_selected' };
  const finalT = validatedOk ? validation.director_verified_throughput_tok_s : finalTput;
  const totalGain = (finalT || 0) - (BASELINE_TPUT || 0);
  const tuningGain = (tuning.post_tune_throughput_tok_s || 0) - (tuning.pre_tune_throughput_tok_s || 0);
  const banked = tuning.gate === 'accepted';
  return {
    ...base,
    gate: tuning.gate,
    mode: tuning.mode || '',
    skills_used: tuning.skills_used || [],
    ops_tuned: (tuning.ops_tuned || []).map((o) => ({
      op: o.op || o.short_name || '', backend: o.backend || '',
      isolated_speedup: o.isolated_speedup || 0, engaged: o.engaged === true, artifact: o.artifact || '',
    })),
    artifacts: tuning.artifacts || [],
    // To reproduce the tuning elsewhere: deploy_bundle (idempotent deploy.sh, run by final_launch.sh).
    deploy_bundle: tuning.deploy_bundle || '', deploy_verified: tuning.deploy_verified === true,
    cache_invalidation: tuning.cache_invalidation || [], live_tree_files: tuning.live_tree_files || [],
    apply_overlay: tuning.apply_overlay || '',
    // Finalize's check that the tuning is in the shipped bundle and engages; null if it did not run.
    in_final_bundle: finalize ? (finalize.tuning_in_bundle === true) : null,
    final_bundle_engagement_recheck: (finalize && finalize.tuning_engagement_recheck) || '',
    apply_env: tuning.apply_env || '', apply_flags: tuning.apply_flags || '',
    correctness_gate: tuning.correctness_gate || 'unknown',
    engagement_verified: tuning.engagement_verified === true,
    engagement_evidence: tuning.engagement_evidence || '',
    pre_tune_throughput_tok_s: tuning.pre_tune_throughput_tok_s || 0,
    post_tune_throughput_tok_s: tuning.post_tune_throughput_tok_s || 0,
    noise_floor_pct: tuning.noise_floor_pct || 0,
    tuning_delta_pct: tuning.tuning_delta_pct || 0,
    tuning_speedup: tuning.tuning_speedup ||
      (tuning.pre_tune_throughput_tok_s ? (tuning.post_tune_throughput_tok_s / tuning.pre_tune_throughput_tok_s) : 1.0),
    ab_interleaved: tuning.ab_interleaved === true,
    ab_complete: tuning.ab_complete !== false,
    // Share of the run's TOTAL gain attributable to the tuning phase (banked wins only).
    share_of_total_gain_pct: (banked && totalGain > 0) ? +((tuningGain / totalGain) * 100).toFixed(2) : null,
    report_path: tuning.report_path || `${EVAL_DIR}/tuning/tuning_report.md`,
    summary: tuning.summary || '', reason: tuning.reason || '',
  };
}

// A Validate that returned nothing is flagged, so no KB writer records the carried number.
function validationStatusOf() {
  return validatedOk ? validation.validation_status
    : (validation ? `${validation.validation_status || 'flagged'}_no_number_used_carried_ab`
      : (!want('final') ? 'phase_partial'
        : validateSkip === 'final_is_baseline' ? 'skipped_final_is_baseline'
          : (IS_AGENTX ? 'flagged_no_number_used_carried_ab' : 'unknown')));
}

const wfReturn = {
  // The contract shape run_e2e.py reads; bump only on a breaking change to the keys below.
  schema_version: 1,
  mode: 'e2e',
  fast_mode: FAST_MODE,   // true => ConfigSweep, TuningSkillset and Milestone skipped (HeadKernel-only)
  deep_mode: DEEP_MODE,   // true => HeadKernel runs the long cross-backend co-optimization scheduler
  backend: BACKEND,
  phases_run: PHASES,
  ablation: { arm: ABLATION_ARM || 'A1', requested_arm: ABL_RAW_ARM || 'A1',
    active: ABL_ACTIVE, seed: ABL_SEED, patch_version: 'GEAK-ABLATION-ARMS-v2',
    budgets_ms: { prep: ABL_PREP_MS, implementation: ABL_IMPL_MS, audit_continuation: ABL_AUDIT_MS,
      validation_reserve: FINAL_RESERVE_MS, total_time_budget: TIME_BUDGET_MS },
    head_trial_ms: ABL_HEAD_TRIAL_MS, audit_probability: ABL_AUDIT_P, events: ABL_EVENTS },
  eval_dir: EVAL_DIR,
  model_name: MODEL_NAME,
  baseline_throughput_tok_s: BASELINE_TPUT,
  // The Director's number only when validatedOk; else the carried accepted A/B win, never 0 (except
  // finalFromAb's empty pair, which run_e2e.py fills from disk).
  final_throughput_tok_s: validatedOk ? validation.director_verified_throughput_tok_s : (finalFromAb ? 0 : finalTput),
  throughput_speedup: finalFromAb ? 0 : finalSpeedup,
  validation_status: validationStatusOf(),
  output_parity: validation ? validation.output_parity : 'unknown',
  accepted_config: { flags: curFlags, env: curEnv },
  accepted_kernels: acceptedKernels,
  accepted_heads: acceptedHeads,
  // Tuning phase attribution, reported apart from accepted_heads/accepted_kernels.
  tuning_skillset: tuningReturn(),
  ...(KERNEL_EXTRACT_ON ? { kernel_extraction: {
    delivery_path: profile?.kernel_extraction_delivery_json || '',
    failed_tasks: profile?.failed_tasks || [], deferred_tasks: profile?.deferred_tasks || [],
    coverage_summary: profile?.coverage_summary || {},
  } } : {}),
  // Verified-isolated wins whose e2e A/B never completed, surfaced (slim) for the caller to finish.
  pending_integrations: pendingIntegrations.map(p => ({
    short_name: p.short_name, track: p.track, isolated: p.isolated,
    pct_gpu_time: p.pct_gpu_time, partial: p.partial || null,
  })),
  flagged_heads: flaggedHeads,   // dominant heads not optimized (harness/extract/no-candidate)
  config_tune_enabled: CONFIG_TUNE_ENABLED,
  head_budget: HEAD_BUDGET,
  head_used: headDispatched,
  milestones: milestone,
  budget_used: dispatched,
  budget_total: BUDGET,
  final_overlay: (validation && validation.final_overlay) || (finalize && finalize.final_overlay) || curOverlay,
  // final_patch is the KB's reproducible code (e2e_store.py _ARTIFACT_KEYS reads
  // result["final_patch"]); final_overlay, a directory, cannot stand in for it.
  ...kbResultMetadata(),
  final_patch: (finalize && finalize.final_patch) || '',
  final_launch_script: (validation && validation.final_launch_script) || (finalize && finalize.final_launch_script) || '',
  report_path: report ? report.report_path : `${EVAL_DIR}/architect_report.md`,
  // Conditional spread: with the feature off the key is absent.
  ...(kbWarmStart ? { kb_warm_start: kbWarmStart } : {}),
  // Independent of kb_warm_start (null without KB_DIMS, while kernel lanes may still have recalled);
  // absent when nothing was recalled on either plane.
  ...((KB_RECALL.e2e || KB_RECALL.kernel.length) ? { kb_recall: KB_RECALL } : {}),
  state: carryState,
};

// ---- Canonical handoff to run_e2e.py ----
// The return value may never reach the transcript (the Workflow can run as a background task) and
// the script has no fs, so an agent persists `wfReturn` to <eval_dir>/workflow_return.json, which
// run_e2e.py reads first. Best-effort: on failure run_e2e.py recovers from disk.
if (EVAL_DIR) {
  try {
    const json = JSON.stringify(wfReturn, null, 2);
    await safeAgent(
      `You are a file writer. Use the Write tool to create the file ` +
      `"${EVAL_DIR}/workflow_return.json" with EXACTLY the content below, verbatim. ` +
      `Do NOT reformat, truncate, summarize, or add/remove any keys or values:\n\n` +
      '```json\n' + json + '\n```\n\n' +
      `Then return {"written": true, "path": "${EVAL_DIR}/workflow_return.json"}.`,
      { phase: 'Validate', label: 'persist-workflow-return',
        schema: obj({ written: { type: 'boolean' }, path: { type: 'string' } }, []) },
      2);
    log(`Persisted canonical workflow_return.json -> ${EVAL_DIR}/workflow_return.json (run_e2e.py reads this first).`);
  } catch (e) {
    log(`persist workflow_return.json failed (NON-FATAL — run_e2e.py will recover from disk): ${String(e)}`);
  }
}

// ===========================================================================
// MODULE B — record this run in the deployment knowledge base.
// Strictly after the workflow_return.json persist (Module B can be SIGKILLed while on the network);
// that same file is the --result, so record and report match. EVERY --apply IS PERMANENT (no
// DELETE for /v1/kb/*), hence the conservative gates.
// ===========================================================================
// The Director's verdict, not the raw ratio, decides a real win (a low base-leg outlier can lift a
// no-win above 1.0); `startsWith` also catches the `..._no_number_used_carried_ab` spelling.
const kbNoWinVerdict = ['validated_no_win', 'recovered_no_gain']
  .some((s) => String(wfReturn.validation_status || '').startsWith(s));
if (E2E_WARM_START_ON && KB_DIMS && KB_DIMS.gfx && want('final') && EVAL_DIR &&
    !A.kb_defer_to_interface &&
    wfReturn.throughput_speedup > 1.0 && wfReturn.final_throughput_tok_s > 0 && !kbNoWinVerdict) {
  // Computed deterministically here, never by an agent: `direction` is in _content_digest, so an
  // unstable label would mint duplicate sessions. '' (unlabeled) when nothing distinct happened.
  const kbDirection = [
    (curFlags !== INIT_FLAGS || curEnv !== INIT_ENV) ? 'config' : '',
    (acceptedKernels.length || acceptedHeads.length) ? 'kernels' : '',
    FAST_MODE ? 'fast' : (DEEP_MODE ? 'deep' : ''),
  ].filter(Boolean).join('+');
  const writeCmd =
    KB_ENV_PRELUDE +
    `python3 ${shq(E2E_STORE_SCRIPT)} write ${kbIdentityFlags()} ` +
    `${kbPlaneFlags()} --result ${shq(EVAL_DIR + '/workflow_return.json')} ` +
    // --require-win repeats the `if` above inside the writer: run_e2e.py's salvage path is a second
    // writer, and both must refuse the same results via one gate (e2e_store.win_gate).
    `--require-win --direction ${shq(kbDirection)} ` +
    `--measured-by ${shq('e2e_workflow:' + BACKEND)} --apply`;
  try {
    const written = await safeAgent(
      `You are the e2e knowledge-base writer. Run EXACTLY this command and return its JSON stdout ` +
      `verbatim as StructuredOutput. It pretty-prints over several lines; return the whole object. ` +
      `Do NOT edit the command, do NOT re-run it with different flags, and do NOT retry it on ` +
      `failure — every write it performs is PERMANENT (the store has no delete), so a second ` +
      `attempt with adjusted arguments creates a second permanent record rather than fixing the ` +
      `first. If it fails, return what it printed along with the error.\n` +
      '```bash\n' + writeCmd + ` | tee ${shq(EVAL_DIR + '/kb_write.json')}\n` + '```',
      { phase: 'Validate', label: 'kb:write',
        schema: obj({ ok: { type: 'boolean' }, applied: { type: 'boolean' },
          session_id: { type: 'string' }, files: arrStr, rungs: arrObj }, []) },
      1);
    wfReturn.kb_written = written || { ok: false, error: 'writer agent returned nothing' };
    const rungs = (written && written.rungs) || [];
    log(`[kb] wrote this run: session=${(written && written.session_id) || '?'} ` +
      `direction='${kbDirection}' rungs=${rungs.filter(r => r.written).length}/${rungs.length} ` +
      `promoted=${rungs.filter(r => r.promoted).length}` +
      `${rungs.filter(r => r.error).map(r => ` [${r.tier}: ${r.error}]`).join('')}`);
  } catch (e) {
    // Non-fatal, like the persist above: the result is already on disk and returned.
    wfReturn.kb_written = { ok: false, error: String(e).slice(0, 200) };
    log(`[kb] write failed (NON-FATAL — the run's own artifacts are unaffected): ${String(e)}`);
  }
} else if (E2E_WARM_START_ON && KB_DIMS) {
  const why = !KB_DIMS.gfx ? 'no gfx (an arch-less record is permanent and unattributable)'
    : !want('final') ? 'this is a phase-partial run, so the number is not final'
    : !(wfReturn.throughput_speedup > 1.0) ? `no win to record (${wfReturn.throughput_speedup}x)`
    : kbNoWinVerdict ? `Director declared no win (${wfReturn.validation_status}) — the ${wfReturn.throughput_speedup}x same-session ratio is box-drift, not a gain`
    : 'no final throughput measured';
  log(`[kb] not recording this run: ${why}.`);
}

// Last step, best-effort: an agent persists the agent timeline (per-role token/time attribution) and
// renders the run report (role-execution-tree HTML + MD).
if (EVAL_DIR && LLM_STATS) {
  try {
    // `instance` (this run's eval dir) lets the parser dedupe a nested timeline reached twice
    // without collapsing distinct lanes of the same shape; needs no Date/random.
    const tlJson = JSON.stringify({ ...LLM_TL, instance: EVAL_DIR });
    const tlPath = `${EVAL_DIR}/reports/trace/agent_timeline.json`;
    await safeAgent(
      `You are the file_writer. PHASE=persist_llm_stats.\n` +
      `Do exactly two things, in order:\n` +
      `1. Use the Write tool to create "${tlPath}" with EXACTLY the JSON below, verbatim ` +
      `(create parent directories if needed; do NOT reformat, truncate or summarize it):\n\n` +
      '```json\n' + tlJson + '\n```\n\n' +
      `2. Then run this Bash command (best-effort; if it fails, carry on and report ok=false).\n` +
      `   It runs the token/time/cost ledger AND renders the role-execution-tree report (HTML + MD):\n` +
      `   python3 -B "${WORKFLOW_DIR}/../interface/geak_report.py" --eval-dir "${EVAL_DIR}"\n\n` +
      `Then return {"written": true, "path": "${tlPath}", "ok": <true if the command exited 0 else false>}.`,
      { phase: 'Validate', label: 'file_writer:persist_llm_stats',
        schema: obj({ written: { type: 'boolean' }, path: { type: 'string' }, ok: { type: 'boolean' } }, []) },
      2);
    const nestedNote = LLM_TL.nested.length ? ' plus ' + LLM_TL.nested.length + ' nested kernel run(s)' : '';
    log(`LLM token+time+cost ledger -> ${EVAL_DIR}/reports/trace/ and run report (HTML+MD) -> ` +
        `${EVAL_DIR}/report/. ${LLM_TL.events.length} agent attempts recorded${nestedNote}.`);
  } catch (e) {
    log(`LLM stats emit failed (NON-FATAL — the run is unaffected): ${String(e)}`);
  }
}

return wfReturn;
