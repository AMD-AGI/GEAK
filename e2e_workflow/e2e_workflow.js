export const meta = {
  name: 'e2e-workflow',
  description: 'End-to-end LLM inference-throughput optimizer for AMD Instinct MI-series GPUs (CDNA gfx942/gfx950, the target card is auto-detected on-box). The serving stack is pluggable via scripts/adapters/<backend>.sh (sglang + vllm shipped; pass args.backend). A system layer (e2e Director / System Architect / Profiler / Config Tuner / Kernel Extractor / e2e Integrator) wraps the UNCHANGED single-kernel kernel_workflow: it preflights the env, profiles a running server, triages hot kernels by Amdahl, tunes config/backends, extracts hot editable kernels into standalone unittests, recursively optimizes them with kernel_workflow.js, overlays them back, and re-validates serving throughput. Also still optimizes a single kernel (pass-through).',
  whenToUse: 'Optimize the serving throughput of an LLM on AMD Instinct MI GPUs. Pass args.model_path (required) + optional args.backend (sglang|vllm, default sglang) + args.launch_script (optional). For a single kernel, pass args.kernel_path instead and it delegates straight to the kernel layer.',
  phases: [
    { title: 'Setup', detail: 'e2e Director builds the isolated eval dir + records baseline throughput' },
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

// ---------------------------------------------------------------------------
// Args + defaults. A JS workflow can't read its own path, so workflow_dir is passed in.
// ---------------------------------------------------------------------------
const A = args || {};
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the dir holding e2e_workflow.js, ' +
    'roles/, knowledge/, scripts/ (the dirname of this script).');
}
// The UNCHANGED single-kernel workflow. Default: sibling "kernel_workflow" dir next to this one.
const KERNEL_WF_DIR = String(A.kernel_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kernel_workflow')).replace(/\/+$/, '');
// The single-language WORKER lane. kernel_workflow.js is now a dispatcher (mode=optimize/author ->
// this worker; mode=bakeoff -> multi-language fan-out), so e2e MUST call the worker DIRECTLY
// (kernel_lane.js) — routing through the dispatcher would add a nesting level (e2e -> dispatcher ->
// worker = 3 levels) and the runtime forbids it. The worker's behavior/args are unchanged.
const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/kernel_lane.js`;

// The kernel workflow's learned KB is OFF by default for lanes launched from here, and ON by default
// when kernel_workflow is driven directly. Same worker, different prior: a kernel_workflow campaign
// re-optimizes a fixed benchmark set, where a card distilled from a past run of the same kernel is
// the point; e2e extracts whatever the profiler happens to surface from a live server, and a card
// carrying another run's conclusions about a superficially similar op is a prior nobody asked for.
// Off is also the conservative default for the layer that has never been measured with it on.
// Override per run with args.use_learned_kb=true.
const LANE_USE_LEARNED_KB = String(A.use_learned_kb != null ? A.use_learned_kb : 'false');

// EVERY nested lane invocation goes through here. Setting the flag at each call site instead would
// be the defect this repo keeps re-making — there are seven call sites today, and the eighth would
// silently take the lane's own default (on) with nothing to catch it. test_e2e_lane_defaults.py
// fails if a `scriptPath: KERNEL_WF_SCRIPT` call is added that does not route through this.
const laneArgs = (wfArgs) => ({ use_learned_kb: LANE_USE_LEARNED_KB, ...wfArgs });

// EXP_ROOT = where timestamped run dirs go. Default: sibling "exp/" next to this workflow dir.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

// ---- Profile-analysis skill (OPTIONAL, pluggable; see knowledge/analysis_skills/INDEX.md) ----
// After parse_profile.py emits the standardized Top-N, the Profiler may run ONE analysis skill to
// enrich it with a headroom estimate the Architect can route on. Default `roofline`: per-kernel % of
// the hardware ceiling -> attainable speedup -> expected e2e gain, so budget goes to the kernel with
// HEADROOM rather than merely the biggest one. STRICTLY ADVISORY: it annotates and suggests an
// ordering, never prunes a candidate and never overrides the measured pct_gpu_time.
// `analysis_skill=none` (or a missing/unreadable skill dir) disables the step and the run behaves
// exactly as it did before the feature existed -> ANALYSIS_SKILL_* are '' and nothing is injected.
const ANALYSIS_SKILL = String(A.analysis_skill != null ? A.analysis_skill : 'roofline').trim();
const ANALYSIS_SKILL_ON = !!ANALYSIS_SKILL && ANALYSIS_SKILL !== 'none' && ANALYSIS_SKILL !== 'false';
const ANALYSIS_SKILL_INPUTS = ANALYSIS_SKILL_ON ? {
  ANALYSIS_SKILL: ANALYSIS_SKILL,
  ANALYSIS_SKILL_DIR: `${WORKFLOW_DIR}/knowledge/analysis_skills/${ANALYSIS_SKILL}`,
} : { ANALYSIS_SKILL: '', ANALYSIS_SKILL_DIR: '' };
if (ANALYSIS_SKILL_ON) log(`Profile-analysis skill: ${ANALYSIS_SKILL} (advisory; annotates + reorders, never prunes).`);

// ---- Upstream TraceLens / kernel-agent prior (OPTIONAL; forwarded by run_e2e.py as args.tracelens) ----
// run_e2e.py resolves these paths beside the geak handoff and forwards ONLY the non-null ones.
// They are a PRIOR for the Profile/Strategize/Extract phases: if analysis_md exists the Profiler skips
// its own trace collection and builds the Top-N from TraceLens (and runs an EXTRA parse_profile pass on
// trace_file when present); the Architect uses kernel_candidates as a routing prior. ENTIRELY ADDITIVE:
// when args.tracelens is absent every TRACELENS_* input is '' and the run is byte-identical.
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

const LAUNCH_SCRIPT = A.launch_script || '';
const BACKEND = String(A.backend != null ? A.backend : 'sglang').trim() || 'sglang';  // serving adapter
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Serving tensor-parallel: TP size + the GPU set used for EVERY e2e SERVING launch (baseline, config
// sweep, integrate ref/cand, validation, profiler). This is DISTINCT from GPU_LIST (the
// optimization-parallelism pool used for isolated op benchmarks + the recursive kernel layer). For TP>1
// the SAME (TP, GPU set) must be used for every e2e measurement or deltas are incomparable. Default
// TP=1 on GPU_LIST[0] (backward compatible). args.tp (or args.serving_tp) sets TP; args.serving_gpu
// overrides the GPU set (default = first TP ids of GPU_LIST, comma-joined).
const SERVING_TP = parseInt(A.tp != null ? A.tp : (A.serving_tp != null ? A.serving_tp : 1), 10);
const SERVING_GPU = String(A.serving_gpu != null ? A.serving_gpu
  : GPU_LIST.slice(0, Math.max(1, SERVING_TP)).join(',') || '0');
// ---- WALL-CLOCK BUDGET (opt-in; default OFF when absent => byte-identical) ---------------------------
// time_budget_s is the EXTERNAL orchestrator's HARD kill budget (run_e2e.py GEAK_E2E_TIMEOUT_S),
// forwarded so GEAK can self-pace and FINISH (Finalize/Report/Validate + workflow_return flush) BEFORE
// the SIGKILL — instead of being torn down mid-flight (the deep 24h-budget-vs-12h-kill failure). This is
// the SINGLE place the orchestrator budget is interpreted. When the arg is ABSENT (GEAK invoked directly,
// not via the interface) TIME_BUDGET_MS is null and EVERY budget branch below short-circuits, so the run
// is byte-identical to a build without this feature. No model/run specifics; pure time arithmetic.
const TIME_BUDGET_MS = A.time_budget_s != null ? parseInt(A.time_budget_s, 10) * 1000 : null;
// ---- ELAPSED CLOCK ----
// Date.now() is unavailable in Workflow scripts (breaks resume), so elapsed time is derived from timers.
// No time_budget_s => remainingMs() is Infinity and every guard short-circuits.
//
// Rungs are armed all at once at ABSOLUTE offsets from t0, not as a self-rearming chain: a chain re-arms
// inside its own callback, so scheduler lateness compounds and ELAPSED_MS drifts behind real time. That
// under-count is what spent the reserve in the 20260823 sessions (#1202). A late rung here delays only
// itself.
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
// Hard floor of wall-clock held back for the WHOLE final phase (Finalize + Report + Validate + the
// workflow_return write); no mode starts new work inside it. Without it, three sessions shipped no final
// report (Hyperloom #1202). 60min covers all but the tail of 85 historical final phases (p50 40 / p75 50
// / p90 77 / max 86min), and the tail is the case this exists for: big models spend the final phase
// waiting on server boots. A longer tail still ships the reports (Report writes them before Validate; only
// the re-measure is at risk and run_e2e falls back). The 20% cap binds below a 5h budget.
// Env override: GEAK_FINAL_RESERVE_S.
const FINAL_RESERVE_CAP_FRAC = 0.2;
const FINAL_RESERVE_MS = TIME_BUDGET_MS != null
  ? Math.min(parseInt(A.final_reserve_s != null ? A.final_reserve_s : 3600, 10) * 1000, // 60min
             Math.floor(TIME_BUDGET_MS * FINAL_RESERVE_CAP_FRAC))
  : null;
// Effective budget = budget minus the reserve: one definition of time available to optimize.
const TIME_BUDGET_EFFECTIVE_MS = TIME_BUDGET_MS != null
  ? Math.max(60000, TIME_BUDGET_MS - FINAL_RESERVE_MS)
  : null;
// Cap on the dispatch-deadline TAIL (budget − deadline). A flat 60% deadline reserves 40%, which is wasteful
// on large budgets (24h → ~9h reserved though a few hours suffice). Deadlines below are max(60%, budget − this
// cap): the 60% floor keeps small budgets unchanged; the cap bounds the reserve on large ones. Default 3h.
const TIME_TAIL_CAP_MS = parseInt(A.time_tail_cap_s != null ? A.time_tail_cap_s : 10800, 10) * 1000; // 3h
// Point after which no mode starts new head/milestone/wave work. Defined here (not at the deadline timer)
// because deep mode references it far earlier; TIME_DEADLINE_HIT is armed on it below.
const TIME_HEAD_DEADLINE_MS = TIME_BUDGET_EFFECTIVE_MS != null
  ? Math.max(Math.floor(TIME_BUDGET_EFFECTIVE_MS * 0.6), TIME_BUDGET_EFFECTIVE_MS - TIME_TAIL_CAP_MS) : null;
// ---- FAST MODE (opt-in, default OFF) ----------------------------------------------------------------
// A time-boxed run that takes ALL its optimization from the HeadKernel track: it SKIPS ConfigSweep AND
// the editable-kernel Milestone loop, and completes within a wall-clock budget (default 5h). It exists
// for "give me the best head-kernel wins you can in 5 hours" runs.
// CRITICAL: when fast_mode is OFF (the default) NOTHING below changes the full pipeline — every fast-mode
// knob is selected by a `FAST_MODE ? fast : original` ternary that resolves to the ORIGINAL value, and
// the phase skips / deadline timers are gated on FAST_MODE — so a non-fast run is byte-identical (same
// prompts, same budgets, same control flow) to a build without this feature. No default-mode regression.
const FAST_MODE = String(A.fast_mode != null ? A.fast_mode : 'false') === 'true';
// Total wall-clock budget for a fast run (default 5h). Enforced with setTimeout (Date.now() is NOT
// available in workflow scripts): a global deadline flag stops dispatching NEW head ops, and each nested
// head author-workflow is independently time-bounded so no single op can overrun the budget.
let FAST_BUDGET_MS = parseInt(A.fast_budget_ms != null ? A.fast_budget_ms : 18000000, 10); // 5h
// When the orchestrator passes a wall-clock budget it is the SOURCE OF TRUTH — use it directly (replace the
// 5h default) so a fast run fills the granted time and still finalizes before the external SIGKILL. The 5h
// default (and any explicit fast_budget_ms) applies only when time_budget_s is absent (direct invocation).
if (TIME_BUDGET_EFFECTIVE_MS != null) FAST_BUDGET_MS = TIME_BUDGET_EFFECTIVE_MS;
// Stop STARTING new head ops after this point so the in-flight head + Finalize/Report/Validate still land
// inside FAST_BUDGET_MS. Default 60% of the budget (3h at 5h) leaves ~40% for the last head to finish +
// the deliverable/validation tail.
const FAST_HEAD_DEADLINE_MS = parseInt(A.fast_head_deadline_ms != null ? A.fast_head_deadline_ms
  : Math.max(Math.floor(FAST_BUDGET_MS * 0.6), FAST_BUDGET_MS - TIME_TAIL_CAP_MS), 10);
// Per-head nested author/optimize workflow() bound (fast mode only): a single recursive kernel run can't
// eat the whole budget. Default 90min. (Heads run in PARALLEL on exclusive GPU lanes, so this is a
// per-lane wall-clock cap, not summed — 90min/lane + serial integrate + Finalize still lands inside the 5h
// FAST_BUDGET_MS. 35min was too short: a non-trivial MXFP8 FlyDSL fused-fp8 author needs ~80min and was
// being abandoned mid-flight — the head track got a null/empty patch even though the author later finished.)
const FAST_HEAD_WF_MS = parseInt(A.fast_head_workflow_ms != null ? A.fast_head_workflow_ms : 5400000, 10);
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);       // max kernel-optimization tasks
// MIN floor: dispatch at LEAST this many editable-kernel tasks before the loop may stop on no-improve /
// empty queue (prompt-tunable). Prevents the milestone track from never firing. Capped by BUDGET.
const MIN_KERNEL_TASKS = Math.min(parseInt(A.min_kernel_tasks != null ? A.min_kernel_tasks : 4, 10), BUDGET);
// Milestone only optimizes editable kernels whose profiled share is worth it: skip any candidate with
// pct_gpu_time below this threshold (Amdahl — a kernel a few % of GPU can't move e2e past the noise band).
// Configurable via args.milestone_min_pct (default 5). This OVERRIDES the MIN_KERNEL_TASKS floor: if no
// candidate clears the bar, the milestone stops rather than grinding low-value kernels.
const MILESTONE_MIN_PCT = parseFloat(A.milestone_min_pct != null ? A.milestone_min_pct : 5);
const KERNEL_BUDGET = parseInt(A.kernel_budget != null ? A.kernel_budget : (FAST_MODE ? 3 : 6), 10); // budget passed DOWN per kernel (fewer rounds in fast mode)
const CONFIG_TUNE_ENABLED = String(A.config_tune != null ? A.config_tune : 'true') === 'true';
// ---- TUNING SKILLSET phase (default ON) --------------------------------------------------------------
// The tuning skillset is vendored WHOLE into this repo (default
// <repo>/perf_knowledge/expert_skills/tuning, hash-pinned as ONE unit by
// e2e_workflow/scripts/tuning_skillset_sync.py) and is run by ONE role (tuning_specialist) as ONE phase
// that sits AFTER ConfigSweep and BEFORE HeadKernel. It is deliberately a standalone phase rather than a
// prompt fragment sprinkled into the head bake-off: (a) the skillset is a complete six-step loop
// (scope -> baseline -> search -> correctness -> deploy -> engagement verify) and only runs to completion
// when it owns a phase; (b) taking its own pre/post e2e A/B is what makes "how much did tuning buy us"
// an attributable number in the final report instead of an unsplittable blob of total gain. Whatever it
// accepts is folded into curFlags/curEnv, so HeadKernel optimizes ON TOP of a tuned stack and its own
// A/B reference leg already contains the tuning.
// tuning_skillset="false" disables the phase entirely: no prompt injection, no report inputs, no state
// keys -> the run is byte-identical to a build without this feature.
const TUNING_SKILLSET_ENABLED = String(A.tuning_skillset != null ? A.tuning_skillset : 'true') === 'true';
// Lives under expert_skills/ so the tuning skills sit in the same hierarchy, and are selected through
// the same index.yaml, as every other expert skill. The tree itself stays VENDORED and pinned whole —
// the index entries that describe it are outside it, which is what lets both things be true at once.
const TUNING_SKILLSET_DIR = String(A.tuning_skillset_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge/expert_skills/tuning')).replace(/\/+$/, '');
// tuning-kb/ is the skillset's per-model ANSWER KEY (verified wins + deployable artifacts). Useful in
// production, contaminating in a blind evaluation — the skillset says so itself. Default ON; pass
// tuning_kb="false" for eval runs and the role is told not to read it.
const TUNING_KB_ENABLED = String(A.tuning_kb != null ? A.tuning_kb : 'true') === 'true';
// NOTE: there is deliberately NO op budget here. The head track caps its ops because each one spends a
// recursive kernel-authoring run; tuning ops are cheap by comparison and their value is cumulative, so
// a cap would just leave measurable wins on the table. The role decides where the returns stop.
// Head-kernel track (GEMM/attention) — the highest-pct_gpu_time ops, optimized regardless of edit flag.
const HEAD_THRESHOLD_PCT = parseFloat(A.head_threshold_pct != null ? A.head_threshold_pct : 5);
// max head-op bake-offs. FAST MODE: the head track is parallelized across the GPU pool (one exclusive
// lane per card), so scale the default up to the lane count (>=3) to keep every card busy in opt-A.
// Default mode is UNCHANGED (3).
const HEAD_BUDGET = parseInt(A.head_budget != null ? A.head_budget : (FAST_MODE ? Math.max(3, GPU_LIST.length) : 3), 10);
// Author route: how many languages to author per head op. The Op Benchmarker orders author_plan by ROI
// (for a GEMM head: flydsl first — SOTA GEMM DSL — then triton). Default 2 covers flydsl+triton per head
// while keeping the kernel-layer cost bounded; bump to try hip/ck too when the headroom justifies it.
// FAST MODE used to drop this to 1 to bound the (serial) wall-clock. Now the head track runs author
// directions in PARALLEL across the GPU pool, so a 2nd direction (e.g. flydsl + triton per op) is nearly
// free in wall-clock — keep 2 in both modes so different optimization directions actually fan out.
const HEAD_AUTHOR_MAX = parseInt(A.head_author_max != null ? A.head_author_max : 2, 10);
// Dominant-head protection: an op whose pct_gpu_time >= this is NEVER silently skipped. If its bake-off
// hits a harness fault / no-win / extraction failure, the orchestrator LOUDLY flags it (and still tries
// the author route when a plan exists) instead of dropping the biggest lever on the floor. Default 30%.
const HEAD_PROTECT_PCT = parseFloat(A.head_protect_pct != null ? A.head_protect_pct : 30);
// Corrective re-author: when a verified-isolated head winner is REJECTED at the e2e gate for a FIXABLE
// integration reason (it ENGAGED live + beat the isolated oracle, only the integration POSTURE is wrong —
// e.g. a JIT/DSL kernel lazily compiling in the TP>1 warmup -> NO_BINARY_FOR_GPU / cuda_graph_capture_unsafe,
// or a host-sync that breaks capture), spend ONE cheap targeted fix-and-retry: optimize the EXISTING kernel
// (keep the algorithm/iso win, fix only the integration), then re-run the e2e A/B once. This is NOT a new
// head discovery, so it does NOT consume HEAD_BUDGET; it is bounded per head by HEAD_CORRECTIVE_MAX and is
// skipped once the kernel-phase wall-clock deadline has fired. head_corrective_max=0 disables it.
const HEAD_CORRECTIVE_MAX = parseInt(A.head_corrective_max != null ? A.head_corrective_max : 2, 10);
// SURGICAL-FIX tier (default ON): before escalating a corrective to the HEAVYWEIGHT kernel_workflow
// re-author (multi-round, multi-engineer, ~hours), first try a LIGHTWEIGHT single-agent targeted patch
// (the kernel_surgeon role): read the reject diagnosis + the failing kernel + the live call seam, make
// the SMALLEST edit that fixes the defect, self-verify on the immutable unittest (correctness + isolated
// win preserved), re-gate. Most integration/correctness rejects are a tiny seam bug (write into y=
// instead of returning; stop caching a per-call tensor by data_ptr) — minutes, not hours. Escalates to
// the heavy re-author only when the surgical patch fails. surgical_fix=false => old heavy-only behavior.
const SURGICAL_FIX = String(A.surgical_fix != null ? A.surgical_fix : 'true') === 'true';
const FIXABLE_REJECT_RX = /cuda_graph_capture_unsafe|no[_ ]?binary|NO_BINARY_FOR_GPU|hipErrorNoBinaryForGpu|capture[_ ]?(unsafe|hang)|host[_ ]?sync|graph[_ ]?capture|no[_ ]?rebind[_ ]?seam|no[_ ]?engagement|not[_ ]?engaged|signature[_ ]?mismatch|wrong[_ ]?seam/i;
// ---- CORRECTNESS-class reject (auto-correct) --------------------------------------------------------
// A SECOND fix-and-retryable class: the candidate ENGAGED and beat the isolated oracle but produces the
// WRONG output on the LIVE path (parity/accuracy failure) — OR posts an IMPLAUSIBLE e2e speedup (faster
// only because it computes degenerate/less work). Distinct from the integration-POSTURE class above: the
// kernel over-fit the single captured snapshot, assuming input-buffer identity / contents / routing are
// stable across calls (e.g. a cache keyed by data_ptr(), a stale reused index/mask, a persistent-state
// assumption). A single-snapshot isolated unittest can NEVER catch this (fixed tensors → any such cache
// is always "correct"), so a blind re-run reproduces the bug; the corrective loop below feeds this
// diagnosis back so the re-author actually removes the over-fit. GENERIC — no kernel/model specifics.
const CORRECTNESS_REJECT_RX = /parit|corrupt|mismatch|diverge|garbage|degener|\bnan\b|\binf\b|accuracy[_ ]?regress|wrong[_ ]?output|incorrect|implausible/i;
// Amdahl ceiling: the MOST e2e speedup an op that is `pct`% of GPU time can yield at isolated speedup S is
// 1/(1 - (pct/100)(1 - 1/S)). A measured e2e delta far above this ceiling can be the fingerprint of a
// kernel doing degenerate/less work (corruption). Uses ONLY the profile pct + isolated speedup.
const IMPLAUSIBLE_SPEEDUP_MARGIN = parseFloat(A.implausible_speedup_margin != null ? A.implausible_speedup_margin : 1.0); // headroom over the theoretical ceiling before flagging (1.0 = must exceed 2x the ceiling)
function amdahlCeilingPct(pct_gpu_time, isolated) {
  const p = Math.max(0, Math.min(1, (pct_gpu_time || 0) / 100));
  const s = (isolated && isolated > 1) ? isolated : 1;
  if (p <= 0 || s <= 1) return Infinity;   // unknown inputs -> never flag (fail-open)
  return (1 / (1 - p * (1 - 1 / s)) - 1) * 100;
}
// PARITY-AWARE guard (fixes the false-positive that would DROP real wins). The Amdahl ceiling is derived
// from an imperfect profile `pct_gpu_time`; a genuine win can legitimately exceed it when the profile
// under-counts the op, or when the change has system-wide effects (memory pressure / batching / a config
// swap). So the implausible-speedup verdict is applied ONLY when the acceptance rests on the SOFT gate —
// a sampled task-accuracy probe (quant / `accuracy_gate=gsm8k`), where a degenerate output could squeak
// past a small sample. A BYTE-EXACT parity pass is a HARD correctness guarantee → its speedup is real
// even above the ceiling → never flagged. This removes the false-positive on byte-exact (incl. config
// env/flag) wins while keeping the backstop exactly where byte-parity is waived. Backward-safe: an
// integrator that doesn't report `parity_kind` only trips the guard when the RUN uses an accuracy gate.
function parityIsSoft(integ) {
  const pk = integ && integ.parity_kind;               // 'byte_exact' | 'accuracy' | 'none' (optional)
  if (pk === 'accuracy') return true;
  if (pk === 'byte_exact' || pk === 'none') return false;
  return ACCURACY_GATE !== 'none';                      // unknown -> soft only when the run uses an accuracy gate
}
function isImplausibleSpeedup(pct_gpu_time, isolated, integ) {
  if (!parityIsSoft(integ)) return false;              // hard byte-exact correctness -> trust the speedup
  const ceilPct = amdahlCeilingPct(pct_gpu_time, isolated);
  if (!Number.isFinite(ceilPct)) return false;
  return ((integ && integ.e2e_delta_pct) || 0) > ceilPct * (1 + IMPLAUSIBLE_SPEEDUP_MARGIN) + 1e-9;
}
// Classify a reject reason into a fix-and-retry class ('' = terminal, not auto-correctable).
// POSTURE IS TESTED FIRST. CORRECTNESS_REJECT_RX contains a bare `mismatch`, which used to swallow
// the `signature_mismatch` that FIXABLE_REJECT_RX names explicitly: a seam whose signature does not
// match the live call site was routed to the correctness corrective ("your kernel computes the wrong
// thing on the live path, look for a data_ptr over-fit") when what it needs is the integration one
// ("find the method the live server actually dispatches and match its call signature"). The posture
// tokens are specific (signature/seam/engagement/capture), so testing them first cannot capture a
// genuine output-correctness reject; and until the candidate is installed at the right seam, any
// statement about its output is meaningless anyway.
function rejectClass(reason) {
  const r = reason || '';
  if (FIXABLE_REJECT_RX.test(r)) return 'integration';
  if (CORRECTNESS_REJECT_RX.test(r)) return 'correctness';
  return '';
}
// A gate 'accept'/'stack' only counts as a REAL win if the measured e2e delta is not an implausible
// (corruption) speedup. Centralizes the guard so every integrate site treats a too-good-to-be-true
// delta as a reject instead of banking it. GENERIC (uses only pct_gpu_time + isolated speedup).
function integAccepted(integ, pct_gpu_time, isolated) {
  return !!(integ && (integ.gate === 'accepted' || integ.gate === 'stack')
    && !isImplausibleSpeedup(pct_gpu_time, isolated, integ));
}
// The reason string to feed the corrective loop: if the gate "passed" but the delta is impossible, emit
// an implausible_speedup verdict (routes to the correctness corrective); else the integrator's own reason.
function gateRejectReason(integ, pct_gpu_time, isolated) {
  if (integ && (integ.gate === 'accepted' || integ.gate === 'stack')
      && isImplausibleSpeedup(pct_gpu_time, isolated, integ))
    return `implausible_speedup (+${(integ.e2e_delta_pct || 0).toFixed(1)}% >> Amdahl ceiling +${amdahlCeilingPct(pct_gpu_time, isolated).toFixed(1)}% on a soft/accuracy-gated accept — likely corruption/degenerate work)`;
  return integ ? (integ.reason || integ.gate || '') : '';
}
// ---- DEEP MODE (opt-in, default OFF) ----------------------------------------------------------------
// A long, thorough HeadKernel mode that pursues SOTA per head op via CROSS-BACKEND CO-OPTIMIZATION:
// N backends optimize the SAME head op in parallel (one exclusive GPU lane each), continuously
// CONTINUING across waves (kernel_workflow STATE_DIR persistence — no lost experience / no re-explored
// directions), sharing a live blackboard KB (a curator distills each wave's findings + assigns directed
// cross-backend borrows), anchored to a roofline SOTA target. Between waves an ADAPTIVE, BATCHED e2e
// gate validates the best candidate(s) end-to-end and feeds the result + a refined harness addendum back
// so the isolated target stays aligned with e2e. GPU scheduling: a single semaphore over GPU_LIST gives
// co-opt lanes exclusive cards; the e2e gate leases ALL cards (TP serving) so it never overlaps co-opt.
// EVERY knob is `DEEP_MODE ? deep : original`-gated; with deep_mode off the whole block is dead code and
// normal/fast are byte-identical. deep_mode is mutually exclusive with the fast parallel track.
const DEEP_MODE = String(A.deep_mode != null ? A.deep_mode : 'false') === 'true';
let DEEP_HEAD_BUDGET_MS = parseInt(A.deep_head_budget_ms != null ? A.deep_head_budget_ms : 86400000, 10); // 24h for the whole HeadKernel module (deep mode does deep exploration)
// When the orchestrator passes a wall-clock budget it is the SOURCE OF TRUTH — use it directly (replace the
// 24h default) so deep fills the granted time and self-finalizes before the external SIGKILL (fixing the
// deep 24h-budget-vs-real-kill failure where it was torn down mid-wave). The 24h default applies only when
// time_budget_s is absent (direct invocation) => byte-identical to today.
// Use the SHARED dispatch deadline, not the raw effective budget: deep used to start waves until 92% of
// the hard kill, then couldn't fit its burst drain + final e2e gate + Finalize/Report/Validate, so it
// shipped no final report (Hyperloom #1202). Fast/default modes have always carved this same tail.
if (TIME_HEAD_DEADLINE_MS != null) DEEP_HEAD_BUDGET_MS = TIME_HEAD_DEADLINE_MS;
const DEEP_HEAD_WF_MS = parseInt(A.deep_head_workflow_ms != null ? A.deep_head_workflow_ms : 4500000, 10); // per-burst nested kernel_workflow time cap (75min) — bounds the per-wave barrier. Harvest+gate run at the TOP of each wave on disk truth, so gate latency is decoupled from this; the cap only bounds how long a burst runs.
const DEEP_E2E_GAIN_TRIGGER = parseFloat(A.deep_e2e_gain_trigger != null ? A.deep_e2e_gain_trigger : 0.08); // isolated-best improvement since last e2e gate that triggers a new (batched) gate
const DEEP_E2E_MAX_INTERVAL_MS = parseInt(A.deep_e2e_max_interval_ms != null ? A.deep_e2e_max_interval_ms : 7200000, 10); // force an e2e gate at least this often when a new candidate exists (default 2h)
const DEEP_PLATEAU_STREAK = parseInt(A.deep_plateau_streak != null ? A.deep_plateau_streak : 2, 10); // consecutive non-improving waves before a backend lane is parked (frees its GPU)
const DEEP_BACKENDS_OVERRIDE = String(A.deep_backends || '').trim(); // optional CSV "triton:optimize,hip:author,..."; default = derive from the bake-off
// ---- DEEP MODE (the canonical deep orchestrator; runs when deep_mode=true) --------------------------
// Deep is a DISCIPLINED DELTA on the normal HeadKernel paradigm (same extract→bakeoff→author→gate→overlay
// role agents), made DEEPER (more kernel_workflow rounds per lane via STATE_DIR + reseed), BROADER (a lane
// per viable backend), and FASTER (lanes run in parallel on dedicated cards; only timing is serialized).
// It is a GLOBAL, GPU-elastic, budget-driven orchestrator:
//   • GLOBAL lane pool — every (head op × backend) lane competes in ONE pool, so multiple KERNELS and
//     multiple BACKENDS optimize concurrently (no more "head A fully done before head B starts").
//   • GPU-elastic & N-adaptive — co-opt runs on cards NOT needed by the serving slot; the serial e2e
//     gate runs on the fixed serving slot CONCURRENTLY (overlap, no idle cards). With exactly TP cards
//     (no spare) it degrades gracefully to time-slice (gate pauses co-opt). Derived from gpu_ids+TP;
//     nothing about card count is hard-coded.
//   • Full-backend roster + ceiling-aware patience + revive — every bake-off backend gets a lane; a
//     high-ceiling-but-slow-start backend is NOT parked early (patience scales with remaining ceiling
//     gap); a parked lane can be REVIVED when the curator hands it a fresh cross-backend borrow.
//   • Budget controller — picks the highest-EV lane next (EV = Amdahl mass × remaining-ceiling gap ×
//     recent improvement rate); re-profiles to chase the moving bottleneck; kills dead directions fast.
//   • Cross-pollination — per-op SHARED_KB (cross-backend) PLUS a run-global GLOBAL_KB (cross-KERNEL,
//     same-backend technique transfer).
// Everything here is gated by DEEP (≡ DEEP_MODE). DEEP_MODE off → default/fast are byte-identical
// (this whole block is dead code). No model/kernel/backend specifics are hard-coded.
const DEEP = DEEP_MODE;   // deep_mode=true runs the global cross-kernel×backend orchestrator (the canonical deep mode)
const DEEP_E2E_TARGET = parseFloat(A.deep_e2e_target != null ? A.deep_e2e_target : 1.5); // stretch goal: +50% e2e vs the TRUE baseline (keep pushing toward it within budget)
const DEEP_PLATEAU_STREAK_HIGH = parseInt(A.deep_plateau_streak_high != null ? A.deep_plateau_streak_high : 4, 10); // patience for HIGH-ceiling lanes (still far from their potential) before parking
const DEEP_REPROFILE_GAIN = parseFloat(A.deep_reprofile_gain != null ? A.deep_reprofile_gain : 0.10); // cumulative e2e gain since last profile that triggers a re-profile (bottleneck moved)
// DEPTH knobs (v2): the first runs were too SHALLOW — 10 parallel lanes split the budget so each lane
// got only ~2 bursts, and the wave loop EXITED as soon as all lanes plateau-parked (abandoning most of
// the 24h). v1 reached +31.5% by going DEEP: ~15h, stacking 4 generations on the winning kernel. Fix:
//   (a) deeper bursts (more kernel_workflow rounds per burst);
//   (b) RUN UNTIL THE BUDGET — when all lanes plateau, RE-SEED them with FRESH authoring directions
//       (concentrated on the dominant-Amdahl head) instead of exiting, so depth keeps compounding.
const DEEP_WAVE_BUDGET = parseInt(A.deep_wave_budget != null ? A.deep_wave_budget : 3, 10); // kernel_workflow rounds per burst. Was 6 but deeper-per-burst SLOWED exploration and underperformed; 3 = faster bursts + more waves (depth now comes from run-until-budget + reseed, not bigger bursts).
const DEEP_MAX_RESEEDS = parseInt(A.deep_max_reseeds != null ? A.deep_max_reseeds : 12, 10); // max fresh-direction re-seeds per lane before it is truly exhausted (budget usually stops first)
// P1: convergence-stop + agent-budget backstop. The wave loop re-seeds for depth, but once optimization
// has demonstrably CONVERGED (K consecutive waves with no new isolated OR e2e gain) it STOPS and finalizes
// rather than burning cheap zero-winner waves into the runtime's hard agent cap (which throws and skips
// finalize). DEEP_AGENT_BUDGET keeps projected nested-agent spend safely below that cap so finalize runs.
const DEEP_CONVERGE_STREAK = parseInt(A.deep_converge_streak != null ? A.deep_converge_streak : 3, 10); // consecutive zero-gain waves before declaring convergence and finalizing
const DEEP_AGENT_BUDGET = parseInt(A.deep_agent_budget != null ? A.deep_agent_budget : 820, 10); // stop STARTING new waves once projected nested-agent spend reaches this (~180 margin under the 1000 runtime cap for finalize)
const DEEP_AGENTS_PER_BURST = parseInt(A.deep_agents_per_burst != null ? A.deep_agents_per_burst : 20, 10); // realistic nested-agents-per-burst (setup+bench[+analyze+profile if cold] + ~3 rounds*~7) for the budget projection
const DEEP_FINAL_ACCURACY_LIMIT = parseInt(A.deep_final_accuracy_limit != null ? A.deep_final_accuracy_limit : 500, 10); // larger gsm8k sample at the FINALIZE gate so a boundary candidate (e.g. ~tol-1pt) is decided on signal, not n=200 noise
// ---- ACCURACY GATE (opt-in switch) ------------------------------------------------------------------
// For QUANTIZED kernels (MXFP8/fp8) byte-exact e2e parity is the WRONG bar — a kernel within the unittest
// tolerance rounds differently and flips borderline greedy argmaxes, so byte-parity over-rejects valid
// kernels. The RIGHT bar is TASK ACCURACY. When accuracy_gate=gsm8k, the e2e_integrator scores the
// candidate vs the true baseline on a sampled gsm8k subset (scripts/gsm8k_eval.py, 5-shot greedy
// exact_match, InferenceX-style) and accepts iff cand_score >= baseline_score - accuracy_tol — instead of
// (over-strict) byte-parity. Default 'none' => unchanged byte/greedy parity (normal/fast untouched).
const ACCURACY_GATE = String(A.accuracy_gate || 'none').trim();          // 'none' | 'gsm8k'
const ACCURACY_LIMIT = parseInt(A.accuracy_limit != null ? A.accuracy_limit : 200, 10); // sampled gsm8k subset size
const ACCURACY_TOL = parseFloat(A.accuracy_tol != null ? A.accuracy_tol : 0.01);        // allowed absolute exact_match drop vs baseline
const ACCURACY_INPUTS = (ACCURACY_GATE !== 'none')
  ? { ACCURACY_GATE, ACCURACY_LIMIT, ACCURACY_TOL, GSM8K_EVAL_SCRIPT: `${WORKFLOW_DIR}/scripts/gsm8k_eval.py` }
  : {};
// The AMD authoring knowledge base (REFERENCE ONLY — facts/how-to, never decisions; agents always
// measure). Default: sibling perf_knowledge/. Workflows enumerate candidates from
// index/capability_index.yaml; status/perf in cards are dated evidence, not routing inputs.
const KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge')).replace(/\/+$/, '');
// Warm start = the kernel layer's LOCAL experience reuse: before round 1 a lane resolves this kernel's
// own history in kb_artifacts/ and re-validates the top stored patches through the same verify gate as
// a fresh candidate. The knobs live in the kernel layer; e2e only forwards them, so that (a) one store
// is shared by every recursive lane in a run and (b) an orchestrator-supplied store reaches them at all
// (a lane's own default resolves next to its workflow dir and would ignore the e2e arg). The forwarded
// values equal the lane defaults when nothing is passed, so an unparameterized run is unchanged.
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts')).replace(/\/+$/, '');
const KB_ARGS = {
  kb_artifacts_dir: KB_ARTIFACTS_DIR,
  warm_start: String(A.warm_start != null ? A.warm_start : 'on'),
  ...(A.warm_start_match != null ? { warm_start_match: String(A.warm_start_match) } : {}),
  ...(A.warm_start_min_speedup != null ? { warm_start_min_speedup: A.warm_start_min_speedup } : {}),
  // Which plane the lanes read and write. Forwarded like the rest so one run uses one plane; omitted
  // when unset, which leaves each lane on its own `local` default.
  ...(A.kb_mode != null ? { kb_mode: String(A.kb_mode) } : {}),
  ...(A.kb_store_dir != null ? { kb_store_dir: String(A.kb_store_dir) } : {}),
  ...(A.kb_framework_version != null ? { kb_framework_version: String(A.kb_framework_version) } : {}),
};
// Same off-spelling set the kernel layer accepts; an off run must stay off everywhere.
const WARM_START_OFF = ['off', 'false', 'none'].includes(KB_ARGS.warm_start.trim().toLowerCase());

// ---------------------------------------------------------------------------
// E2E-LEVEL WARM START — the DEPLOYMENT knowledge base.
//
// KB_ARGS above forwards the KERNEL layer's warm start (per-kernel patches, resolved by the nested
// lane). This block is the layer above it: "has anyone already tuned THIS deployment — this (model,
// gfx, framework, framework_version, precision, tp, isl/osl/conc)?" That question was rediscovered
// from scratch on every run, at a cost of hours of server launches, because nothing here ever asked
// it. Two new modules answer it: Module A reads the store and VALIDATES what it gets on this box
// before believing any of it, and Module B records this run at the end. Everything between them —
// the entire original optimization flow — is untouched.
//
// One knob governs both layers (A.warm_start, already defaulted to 'on' by KB_ARGS), with the same
// vocabulary the kernel lane accepts, so an `off` run is off everywhere:
//   on (default)      | read the ladder, validate the top candidate on this box, ADOPT if it wins.
//   reference         | read only; never bench, never adopt — the offer is prose for later agents.
//   return_after_read | resolve + validate, then return before the optimization flow runs.
//   off/false/none    | nothing: no phase, no agent, no log, and every role prompt byte-identical.
const E2E_WARM_START = KB_ARGS.warm_start.trim().toLowerCase() || 'on';
const E2E_WARM_START_ON = !WARM_START_OFF;
// Fast mode's premise is that all optimization comes from the head track within a wall-clock budget
// (FAST_SKIP already drops 'config'), so a warm-start config bench — a 20-40min server launch that
// this mode has explicitly declined to spend anywhere else — is forced down to reference-only.
const E2E_WARM_START_REF_ONLY = E2E_WARM_START === 'reference' || FAST_MODE;
const E2E_WARM_START_RETURN_AFTER = E2E_WARM_START === 'return_after_read';
// Which plane the DEPLOYMENT store is read from and written to. Translated from the kernel layer's
// kb_mode vocabulary ONCE, here, so the reader and the writer can never disagree about it.
//   local   on-disk store only (offline rehearsal; no network, nothing permanent).
//   remote  the KB Store service only.
//   both    local first, remote as well — a network failure is reported, not fatal.
// Default `both`: the local plane costs nothing and gives the run a durable copy even when the
// service is unreachable, which on this box it intermittently is.
const E2E_KB_PLANE = ['local', 'remote', 'both'].includes(String(A.e2e_kb_plane || '').trim())
  ? String(A.e2e_kb_plane).trim()
  : (String(A.kb_mode || '').trim().toLowerCase() === 'local' ? 'local' : 'both');
const E2E_KB_STORE_DIR = String(A.kb_store_dir ||
  KB_ARTIFACTS_DIR.replace(/\/[^/]*$/, '') + '/kb_store_local').replace(/\/+$/, '');
const E2E_STORE_SCRIPT = `${WORKFLOW_DIR}/scripts/e2e_store.py`;
// The file the warm-start read drops this run's KB address into, and the one run_e2e.py looks for
// when it has to write the record this process never got to. Named in both programs; spelled once
// here and once in interface/run_e2e.py (KB_IDENTITY_FILE), which is the same arrangement
// workflow_return.json already has.
const KB_IDENTITY_BASENAME = 'kb_identity.json';
// Every candidate costs a full server launch to reject, so a recorded near-tie is not worth benching.
const E2E_WARM_START_MIN_SPEEDUP = Number.isFinite(parseFloat(A.warm_start_min_speedup))
  ? parseFloat(A.warm_start_min_speedup) : 1.05;
// Read broadly, bench narrowly. Reading is free and breadth is exactly what makes the demoted
// reference path useful to the Architect; benching is the expensive half. So the default is
// "offer 3, measure 1".
const E2E_WARM_START_TOP_N = parseInt(A.e2e_warm_start_top_n != null ? A.e2e_warm_start_top_n : 3, 10);
const E2E_WARM_START_VALIDATE_N = parseInt(
  A.e2e_warm_start_validate_n != null ? A.e2e_warm_start_validate_n : 1, 10);
// The same budget for a COARSE rung (`tp_any`/`workload_any`), where the record was measured at a
// different tp or a different isl/osl/conc. This used to be hard zero, on the argument that a
// non-comparable stored number is not worth a server launch. That argument confuses the PRIOR with
// the MEASUREMENT: the bench below runs the recovered config on THIS box against THIS run's
// baseline, so what the record claimed never enters the verdict — it only decides what is worth
// trying first. And a config's transferability is exactly what the coarse rungs are for: the store
// already ranks them by SPEEDUP rather than throughput (e2e_store.rung_metric) because the ratio is
// the part that survives a workload change. Zero there meant every `tp_any` hit was filed as a lead
// and then never followed by anyone, which is what the 20260826 gpt-oss-120b and Qwen3.5-122B runs
// did. Budgeted rather than unbounded because the launch cost is real; set to 0 to restore the old
// reference-only behaviour.
const E2E_WARM_START_VALIDATE_N_COARSE = parseInt(
  A.e2e_warm_start_validate_n_coarse != null ? A.e2e_warm_start_validate_n_coarse : 1, 10);
// How many stored KERNELS get replayed through a fresh integrate A/B. Each one is another two server
// launches, so this is a budget, not a completeness target — the rest stay references.
const E2E_WARM_START_KERNELS_N = parseInt(
  A.e2e_warm_start_kernels_n != null ? A.e2e_warm_start_kernels_n : 2, 10);
// Which recorded win-kinds can be REBUILT from a record alone. `patch` re-applies a diff through the
// overlay; `env`/`flag` re-route the op to an implementation that already exists in this install.
// `authored` cannot be rebuilt: the integrator's authored recipe git-applies the diff inside
// `authored_kernel_eval_dir/workspace/` and assembles the module from `kernel_src/`, and that
// workspace dies with the run that made it.
//
// Not-rebuildable is NOT not-replayable, and reading it that way is what kept every authored kernel
// in the store permanently unverified. The overlay directory — the OUTPUT of that build — is carried
// in the record as `overlay.tar.gz` (e2e_store._pack_overlay), and it is the complete mechanism:
// manifest, sitecustomize, the authored source, PYTHONPATH-activated and structurally reversible. So
// a kind outside this set falls through to the prebuilt-overlay path below when the bundle has one,
// and is demoted to a reference only when it has neither. A demotion is never reported as a failure —
// it is still a real datum about what worked on this deployment.
const E2E_REPLAYABLE_KINDS = new Set(['patch', 'env', 'flag']);
// Which roles are TOLD about the warm start. Deliberately excludes e2e_integrator and
// director:validate — those two produce the run's authoritative numbers, and a stored prior in their
// context is contamination with no upside.
// tuning_specialist is a consumer for the same reason the other two are, and one more: the tuning
// track is the ONLY one whose output previously had no way back in. A tuned artifact is bulky,
// deploys outside the git tree, and takes hours of search to rediscover — so a run that re-derives
// one it already has in the store is the most expensive avoidable thing this workflow does.
const WARM_START_ROLES = new Set(['system_architect', 'config_tuner', 'tuning_specialist']);
// Credentials and trust for every emitted KB command. The six lines that do the work live in
// scripts/kb_env.sh (which explains them), not here, because run_e2e.py runs KB commands too — the
// salvage write for a run whose workflow died before Module B — and a Python copy of the token path
// and CA fallback list is exactly the kind of duplicate that drifts without ever raising. Sourced,
// so both programs get the same answer from the same file.
// (shq is declared further down, so the quoting is written out here rather than called.)
const KB_ENV_PRELUDE = `. "${WORKFLOW_DIR}/scripts/kb_env.sh"; `;
// Expert skills = human-authored, validated optimization recipes (perf_knowledge/expert_skills/). They
// are ADVISORY priors: a matched `validated` skill is a HIGH-PRIOR candidate that routing/integration
// roles reproduce, then gate by the usual on-box A/B — it NEVER overrides measurement and NEVER reduces
// a result below the measured baseline. Default OFF (opt-in): pass use_expert_skills="true" to enable.
// When OFF (the default) NOTHING is injected into any role prompt -> the prompt (and thus the whole run)
// is byte-identical to a build without this feature. The flag + dir are passed DOWN to the kernel layer.
const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR + '/expert_skills')).replace(/\/+$/, '');
// Only routing/bake-off/integration roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['system_architect', 'op_benchmarker', 'e2e_integrator']);
const GEMM_SYNTH = String(A.gemm_synth != null ? A.gemm_synth : 'true');     // synth GEMM inputs (cheap)
const ENABLE_FP8 = String(A.enable_fp8 != null ? A.enable_fp8 : 'false');    // Tier-D quant (parity-breaking)
const FAST_PATH_FIRST = String(A.fast_path_first != null ? A.fast_path_first : 'true') === 'true';
const ISL = parseInt(A.isl != null ? A.isl : 1024, 10);
const OSL = parseInt(A.osl != null ? A.osl : 1024, 10);
const CONC = parseInt(A.conc != null ? A.conc : 64, 10);
const WORKLOAD = { isl: ISL, osl: OSL, conc: CONC };
// Seed config: when an external orchestrator (e.g. Hyperloom) already did
// config/param search, it passes its accepted best flags/env so the GEAK
// baseline is measured ON that config (fair engagement start), not the stack
// default. Serving TP/GPU are handled by SERVING_TP / SERVING_GPU above.
const INIT_FLAGS = String(A.initial_extra_server_args || '');
const INIT_ENV = String(A.initial_extra_env || '');
// Schema-v2 handoffs may carry the exact Python overlay/source snapshot stack
// that produced Hyperloom's current best.  This is part of the baseline
// configuration, not a GEAK-authored candidate, so it must remain underneath
// every later candidate overlay instead of being dropped at Setup.
const INIT_BASE_OVERLAY = String(A.initial_overlay_pythonpath || '');
const EFFECTIVE_CONFIG_DIGEST = String(A.effective_config_digest || '');
// Throughput measurements use independent server replicas.  Search/parity use
// one Hyperloom-equivalent replica; final validation uses three replicas to
// estimate variance.  The bench dispatcher owns retry/degraded aggregation.
const MEASUREMENT_MODE = String(A.measurement_mode || 'isolated_server');
const PARITY_REPLICAS = parseInt(A.parity_replicas != null ? A.parity_replicas : 1, 10);
const SEARCH_REPLICAS = parseInt(A.search_replicas != null ? A.search_replicas : 1, 10);
const VALIDATION_REPLICAS = parseInt(A.validation_replicas != null ? A.validation_replicas : 3, 10);
// CUDA/HIP-graph deployment requirement (general; derived from the serving config, NOT hardcoded).
// vllm/sglang capture the steady-state decode path into a FULL CUDA graph UNLESS --enforce-eager is set.
// A kernel that wins only via its OWN per-call graph-capture+replay wrapper falls back to eager inside the
// server's graph, so the isolated win evaporates e2e (observed on M3: MoE 1.22x isolated -> 0% e2e). When
// graphs are on we inject an EXPLICIT requirement into every kernel-optimize task: the win must be intrinsic
// and graph-capture-safe. Detection is config-driven (enforce-eager absent + graph-capable backend), so it
// auto-disables for an enforce-eager run and applies to any future graph-capturing backend.
const CUDA_GRAPH_DEPLOY = (BACKEND === 'vllm' || BACKEND === 'sglang') && !/enforce[-_]eager/i.test(INIT_FLAGS);
const GRAPH_REQ = CUDA_GRAPH_DEPLOY ? (
  ' DEPLOYMENT REQUIREMENT — the server captures the steady-state decode path into a FULL CUDA/HIP graph, ' +
  'so this kernel runs INSIDE that captured graph. Your speedup MUST be INTRINSIC: better tiles/algorithm, ' +
  'fused quant (one fp8 MFMA, kill the dequant), or fewer ops/launches that reduce work INSIDE the captured ' +
  'region. Do NOT rely on a per-call CUDA/HIP-graph capture+replay WRAPPER for the speedup — inside the ' +
  "server's graph that wrapper falls back to eager and the win vanishes, and the e2e integrate gate WILL " +
  'reject a wrapper-only win (this already happened: a 1.22x isolated MoE GEMM gave 0% e2e because only its ' +
  'static tile change survived the graph). The steady-state decode call must be graph-capture-safe: ' +
  'host-sync-free (no .item()/.cpu()/.tolist()/.synchronize(), no Python branch on a GPU scalar), shape-stable, ' +
  'and prep/compile ONCE (cache by data_ptr) so the captured region only LAUNCHES the kernel. VERIFY your ' +
  'speedup holds when the op is replayed under a CUDA graph, not just in eager timing.'
) : '';
// Acceptance noise band (%). Isolated-server ref/candidate measurements, non-overlap, and engagement
// proof (see e2e_integrator) make a 0.5% default trustworthy. Prompt-tunable.
const NOISE_BAND_DEFAULT = parseFloat(A.noise_band_pct != null ? A.noise_band_pct : 0.5);
// Legacy timed-repeat input retained for callers that explicitly select the legacy protocol.
// Isolated-server runs use the purpose-specific replica counts above.
const E2E_REPEATS = parseInt(A.e2e_repeats != null ? A.e2e_repeats : 2, 10);
// Every integrate A/B MUST measure BOTH legs (reference + candidate). When the
// integrator returns gate:'incomplete'/ab_complete:false (it only ran ref, hung,
// or degraded), the orchestrator RE-INVOKES it to finish the missing leg up to
// this many times before falling back to carrying it as a pending integration.
// This is what guarantees "every A/B runs ref AND cand to completion regardless
// of pass/fail" — general, not per-kernel. Bump via args.ab_finish_retries.
const AB_FINISH_RETRIES = parseInt(A.ab_finish_retries != null ? A.ab_finish_retries : 3, 10);
// A resolvable FROZEN baseline (a seeded baseline_overlay/ + a declared meta.candidate_bind, or an
// importable meta.baseline_callable on the op track) is the speedup DENOMINATOR and is MANDATORY.
// If an extraction smoke-passes but froze no baseline, the
// unittest would silently time the candidate against its own naive same-language scaffold (the
// "optimized-HIP vs naive-HIP = fake 15.7× isolated, ~0% e2e" bug). When that happens we RE-EXTRACT
// up to this many times; if still missing, the extraction is treated as a FAILURE (flag dominant /
// skip others) — never a fake speedup. Bump via args.baseline_extract_retries.
const BASELINE_EXTRACT_RETRIES = parseInt(A.baseline_extract_retries != null ? A.baseline_extract_retries : 3, 10);
// Issue #429: force capture storage bounds into every extraction EXTRA_ENV (override via args).
const CAPTURE_BYTE_BUDGET = String(A.capture_byte_budget != null ? A.capture_byte_budget : '8GiB');
const CAPTURE_CASE_BYTE_LIMIT = String(A.capture_case_byte_limit != null ? A.capture_case_byte_limit : '2GiB');
const CAPTURE_PERSIST_POLICY = String(A.capture_persist_policy != null ? A.capture_persist_policy : 'share_large');
const CAPTURE_WORKSPACE_BUDGET = String(A.capture_workspace_budget != null ? A.capture_workspace_budget : '32GiB');
const CAPTURE_STORAGE_ENV = `CAPTURE_BYTE_BUDGET=${CAPTURE_BYTE_BUDGET} CAPTURE_CASE_BYTE_LIMIT=${CAPTURE_CASE_BYTE_LIMIT} CAPTURE_PERSIST_POLICY=${CAPTURE_PERSIST_POLICY}`;
const TASK = A.task || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const MODEL_NAME_HINT = (MODEL_PATH || KERNEL_PATH).replace(/\/+$/, '').split('/').pop();

// ---------------------------------------------------------------------------
// Phase-scoped driving (robustness): a long single background run can be orphaned if the host session
// context compacts mid-run. To avoid that, the orchestration can be driven phase-by-phase: invoke with
// args.phases = subset of {setup,config,tune,head,kernel,final} (default 'all' = run everything in one go).
// Cross-phase state flows through the RETURN value (the script has no fs); pass the prior return back as
// args.state for the next phase. Each phase only RUNS if requested; otherwise it loads carried state.
// ---------------------------------------------------------------------------
const PHASES = String(A.phases || 'all').split(',').map(s => s.trim()).filter(Boolean);
const RUN_ALL = PHASES.includes('all');
// Fast mode SKIPS ConfigSweep ('config'), the tuning skillset ('tune') and the editable-kernel Milestone
// ('kernel') so all optimization comes from HeadKernel within the wall-clock budget. ('tune' joins the
// skip set for the same reason 'config' does: fast mode's contract is "best head-kernel wins in N hours",
// and the tuning loop's tuner sweeps + its own pre/post A/B do not fit that budget.) Default mode:
// FAST_SKIP is null → want() is the original `RUN_ALL || PHASES.includes(p)`, unchanged.
const FAST_SKIP = FAST_MODE ? new Set(['config', 'tune', 'kernel']) : null;
// Deep mode concentrates its (20h) HeadKernel budget on cross-backend co-opt: skip the editable-kernel
// Milestone ('kernel') but KEEP ConfigSweep ('config' — cheap and it stabilizes the baseline) and KEEP
// the tuning skillset ('tune' — same rationale, and a tuned stack is a strictly better starting point
// for the co-opt lanes). Null in every other mode, so normal + fast are unchanged.
const DEEP_SKIP = DEEP_MODE ? new Set(['kernel']) : null;
const want = (p) => (RUN_ALL || PHASES.includes(p)) && !(FAST_SKIP && FAST_SKIP.has(p)) && !(DEEP_SKIP && DEEP_SKIP.has(p));
const ST = A.state || {};   // carried state from a prior phase invocation
if (FAST_MODE) log(`[fast-mode] ON: skipping ConfigSweep + Milestone; HeadKernel-only; budget ${Math.round(FAST_BUDGET_MS / 60000)}min (stop new heads at ${Math.round(FAST_HEAD_DEADLINE_MS / 60000)}min, per-head workflow cap ${Math.round(FAST_HEAD_WF_MS / 60000)}min).`);

// ---------------------------------------------------------------------------
// Schema fragments.
// ---------------------------------------------------------------------------
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
  // The four deployment dimensions the KB addresses a page by, beyond model/framework/workload which
  // this script already knows. The Director establishes all four during its existing preflight (it
  // has to, to launch the server at all) and previously just discarded them. They are OPTIONAL here
  // and `required` is unchanged, so a director that returns none of them is still valid — the run
  // simply files itself under a coarse `unknown` page, which is recoverable. A GUESS is not: the
  // service has no DELETE, so a wrong-but-authoritative-looking page is permanent.
  gfx: { type: 'string' }, precision: { type: 'string' },
  framework_version: { type: 'string' }, rocm_version: { type: 'string' },
}, ['eval_dir', 'baseline_throughput_tok_s']);

const PROFILE_SCHEMA = obj({
  round: { type: 'number' }, profile_topN_json: { type: 'string' }, profile_topN_md: { type: 'string' },
  profile_workload_json: { type: 'string' }, // per-(shape,dtype) weighted workload model (optional)
  source: { type: 'string' }, total_gpu_time_ms: { type: 'number' }, top_kernels: arrObj,
  shift_note: { type: 'string' }, notes: { type: 'string' },
}, ['profile_topN_json', 'top_kernels']);

const STRATEGY_SCHEMA = obj({
  regime_summary: { type: 'string' }, config_directions: arrObj,
  head_candidates: arrObj, kernel_candidates: arrObj,
  drop_list: arrObj, order_of_work: arrStr, strategy_path: { type: 'string' },
}, ['kernel_candidates']);

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
  // Only the warm-start repair pass fills this, and it stays OUT of `required` so every ordinary
  // sweep keeps validating unchanged. It is the structured half of "why did this not run here" —
  // read in place of guessing at the free-text notes with a regex.
  repair: obj({
    attempted: { type: 'boolean' },
    classification: { type: 'string' },   // upstream_gone | conflicts_with_baseline | env_unsupported | mixed | unknown
    dropped_flags: { type: 'array', items: { type: 'string' } },
    reason_by_flag: { type: 'object', additionalProperties: true },
    launch_error: { type: 'string' },
  }, []),
}, ['accepted_flags', 'best_throughput_tok_s']);

// `e2e_store.py resolve` output, passed through VERBATIM. Nothing is `required` and no field is
// typed as a number: a page that was never written answers with empty candidates and a
// `read_reason`, and a recorded run legitimately has `speedup: null` when it only ever measured
// absolute throughput. Typing either would turn a truthful answer into a schema failure and a
// cold start.
const KB_RESOLVE_SCHEMA = obj({
  tried: arrStr, canonical_id: { type: 'string' }, match_tier: { type: 'string' },
  ranked_by: { type: 'string' }, candidates: arrObj, read_reason: { type: 'string' },
  plane: { type: 'string' },
  // How the offer was ORDERED (throughput, high to low, on every rung) versus which metric that
  // rung crowns its champion on. Two different things that used to share one word: a coarse rung
  // is ranked by absolute throughput here but still promotes on speedup, and a reader told only
  // 'speedup' would mis-explain the order it is looking at.
  sorted_by: { type: 'string' }, champion_metric: { type: 'string' },
}, []);
// Result of the standalone tuning-skillset phase. pre/post are ITS OWN in-session isolated-server A/B
// legs (NOT the run baseline), which makes tuning_delta_pct an attributable share of the total gain.
// engagement_verified is load-bearing: the skillset's own thesis is that an unproven artifact is not a
// win, so the orchestrator refuses to bank an accept without it.
const TUNING_SCHEMA = obj({
  ran: { type: 'boolean' }, mode: { type: 'string' }, skills_used: arrStr,
  preflight: { type: 'object', additionalProperties: true },
  ops_tuned: arrObj, artifacts: arrStr,
  // Deploy bundle: how a tuned DATA artifact reaches production. GEAK's overlay is a PYTHONPATH
  // mechanism for code, but tuned config tables are data a library reads from its own package dir, and
  // some need a derived cache dropped before they take effect. Neither travels in the overlay — so the
  // role writes EVAL_DIR/tuning/deploy/{MANIFEST.json,tuning_patch.diff,files/,deploy.sh} and Finalize
  // folds it into EVAL_DIR/final/ (diff concatenated, deploy.sh invoked from final_launch.sh).
  deploy_bundle: { type: 'string' }, deploy_verified: { type: 'boolean' },
  cache_invalidation: arrStr,
  // DATA paths the deploy writes INSIDE an installed package tree (e.g. aiter/configs/*.csv). The
  // Integrator forbids live-tree edits and asserts `git status --porcelain` is empty before/after every
  // A/B leg; this list is the carve-out that keeps a banked tuning from being cleaned away. Data only —
  // a source change belongs in apply_overlay, see below.
  live_tree_files: arrStr,
  // A tuned artifact binds only if the live seam dispatches the backend that consumes it, so tuning
  // sometimes needs a CODE change to land (a routing switch, or a wrapper that drops the kernel
  // selection — tuning-aiter/SKILL.md §2b measures a correctly-tuned row going 85.7% SLOWER on a wrapper
  // that ignores kernelName). That is in scope, and it travels as a reversible overlay, exactly like a
  // head env-winner's routing patch: never as a live-tree .py edit, which cannot be varied per A/B leg.
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

const EXTRACT_OP_SCHEMA = obj({
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
  smoke: { type: 'string' }, notes: { type: 'string' },
}, ['op_kind', 'task_dir', 'smoke']);

const OPBENCH_SCHEMA = obj({
  short_name: { type: 'string' }, op_kind: { type: 'string' }, provenance_ok: { type: 'boolean' },
  winner_backend: { type: 'string' }, winner_kind: { type: 'string' },
  // null when nothing was timed -- a speedup is a measurement, and 0.0 read as
  // 'benched, not faster'. `measured` is the discriminator; gate on it, not on the number.
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
  short_name: { type: 'string' }, editable: { type: 'boolean' }, task_dir: { type: 'string' },
  source_path_in_sglang: { type: 'string' }, target_callable: { type: 'string' },
  num_cases: { type: 'number' }, regimes_captured: arrStr, candidate_backends: arrStr,
  build: { type: 'boolean' }, unittest_smoke: { type: 'string' },
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

const INTEGRATE_SCHEMA = obj({
  short_name: { type: 'string' }, provenance_ok: { type: 'boolean' },
  isolated_speedup: { type: 'number' }, pct_gpu_time: { type: 'number' },
  e2e_throughput_tok_s: { type: 'number' }, e2e_delta_pct: { type: 'number' },
  // A/B completion signals: the integrator MUST set ab_complete=true ONLY when
  // BOTH the reference and candidate serving legs were measured. When it ran out
  // of time / hung mid-gate it returns gate:'incomplete' (ab_complete=false) with
  // whatever partial legs it has (ref_med / cand_med) — that is NOT a rejection.
  ref_med: { type: 'number' }, cand_med: { type: 'number' },
  ab_complete: { type: 'boolean' },
  output_parity: { type: 'string' },
  // How the correctness of an ACCEPT was established: 'byte_exact' (hard greedy byte-parity vs the true
  // baseline), 'accuracy' (soft sampled task-accuracy probe — quant / accuracy_gate), or 'none'. The
  // implausible-speedup guard only distrusts an 'accuracy'/soft accept; a byte_exact accept is trusted.
  parity_kind: { type: 'string' },
  gate: { type: 'string', enum: ['accepted', 'stack', 'rejected', 'incomplete'] },
  accepted_overlay: { type: 'string' }, reason: { type: 'string' },
}, ['gate', 'e2e_throughput_tok_s']);

const FINALIZE_SCHEMA = obj({
  final_overlay: { type: 'string' }, final_patch: { type: 'string' }, final_launch_script: { type: 'string' },
  final_throughput_tok_s: { type: 'number' }, throughput_speedup: { type: 'number' },
  accepted_kernels: arrStr, accepted_config: { type: 'object', additionalProperties: true }, note: { type: 'string' },
  // Did the tuning deploy bundle make it into final/ AND still engage on the assembled bundle? Reported
  // rather than assumed: a data artifact that silently fails to install would leave a bundle that looks
  // complete and reproduces none of the tuning gain.
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

// ---------------------------------------------------------------------------
// Prompt helpers (mirror the single-kernel workflow).
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// Expert-skills prompt injection. PURELY ADDITIVE: returns '' whenever the feature is OFF or the role
// is not a skills consumer, so roleAgent's output is byte-identical to the pre-feature build in those
// cases. When ON, it appends a short advisory pointer that tells the agent to Read the fragment file
// and query the skills index. (Workflow scripts have no fs access; the agent does the reading.)
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role)) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits the current bottleneck/op and whose ` +
    `validation_status is \`validated\`, and treat each as a HIGH-PRIOR candidate to reproduce — advisory ` +
    `only, never overriding your on-box A/B, never reducing a result below the measured baseline.`;
}

// ---------------------------------------------------------------------------
// e2e warm start: identity formatting + the prompt hook.
//
// These three `let`s are the ONLY state the feature adds above Module A, and all three stay at their
// off-value (null / '') for the whole run unless Module A actually executes. Every consumer below
// keys off them, which is what makes `warm_start=off` produce a byte-identical run rather than one
// that merely behaves the same.
// ---------------------------------------------------------------------------
let KB_DIMS = null;        // the deployment dimensions the Director established during preflight
let KB_REF_DIR = '';       // where Module A left its references; '' when it never ran
let KB_CACHE_DIR = '';     // where Module A materialized the records' FILES (patches, tuned tables)
let KB_REF_VERDICT = '';   // what we MEASURED about the offer, threaded into later roles' Inputs
let KB_READ_PLANE = '';    // which plane ANSWERED the read, which `both` alone does not tell you
// Everything this run RECALLED, in one report-ready object: which address was asked, what came back,
// what re-measuring it HERE produced, and whether it was accepted. Threaded into the Report role's
// Inputs because final_report.md was previously silent about recall altogether — a run that recalled
// nothing and a run that recalled a 1.12x record and then rejected it wrote the same report, and the
// second is the one a reader most needs to see.
//
// `asked` is filled in even when zero candidates come back. On an exact-lookup scheme a miss and a
// never-recorded page are the SAME 404, so the ladder that was asked IS the finding: it is the only
// artifact that distinguishes "no prior art" from "prior art exists one segment away".
let KB_RECALL = { e2e: null, kernel: [] };

// One collection point for the KERNEL plane's recall. Every nested kernel_workflow returns its own
// `warm_start` block (kernel_lane.js), and until now all of it died inside the lane: the e2e report
// could say a kernel was authored from scratch while the lane had in fact adopted a stored patch.
// Called from the two bounded wrappers and the two direct workflow() sites, i.e. every lane call.
function noteKernelKB(r, label) {
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
      // A lane that adopted a stored patch and then committed nothing of its own is where the kernel
      // plane most often flatters itself, so carry the split rather than just the headline geomean.
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

// The ONE place e2e KB identity argv is formatted — called by both the reader (Module A) and the
// writer (Module B). kb/identity.py's own header names the failure this prevents: a reader and a
// writer that disagree by a single segment do not raise, they address two different pages, and the
// only symptom is that history quietly stops existing. Two call sites formatting the same flags
// independently is exactly how that drift starts.
function kbIdentityFlags() {
  if (!KB_DIMS) return '';
  return [`--model ${shq(KB_DIMS.model)}`, `--gfx ${shq(KB_DIMS.gfx)}`,
    `--framework ${shq(BACKEND)}`, `--framework-version ${shq(KB_DIMS.framework_version)}`,
    `--precision ${shq(KB_DIMS.precision)}`, `--rocm-version ${shq(KB_DIMS.rocm_version)}`,
    `--tp ${SERVING_TP}`, `--isl ${ISL}`, `--osl ${OSL}`, `--conc ${CONC}`].join(' ');
}

// --store is what the local plane writes into and is meaningless to a remote-only run; open_plane()
// hard-errors without it on plane local|both, so it is not optional there.
function kbPlaneFlags(plane) {
  plane = plane || E2E_KB_PLANE;
  return `--plane ${plane}` + (plane === 'remote' ? '' : ` --store ${shq(E2E_KB_STORE_DIR)}`);
}

// A READ takes exactly one plane — `open_plane()` returns (local, remote) for `both` and cmd_resolve
// uses only the first, deliberately: merging two rankings needs a cross-plane comparability rule
// that nothing here has, and silently preferring one would let a stale local mirror shadow the
// service without saying so. So `--plane both` on a read means LOCAL, which is the opposite of what
// this workflow wants. The choice is made here instead, in the open: try the service, and fall back
// to the local store only when it has no answer. `read_reason` in the returned JSON says which one
// spoke. The credentials cannot be tested from this process (non-interactive shells never source the
// profile that sets them), so the branch lives in the emitted bash, after the prelude has exported
// whatever the box actually has.
function kbResolveScript(args) {
  const invoke = (plane) =>
    `python3 ${shq(E2E_STORE_SCRIPT)} resolve ${kbIdentityFlags()} \\\n` +
    `  ${kbPlaneFlags(plane)} ${args}`;
  if (E2E_KB_PLANE !== 'both') return KB_ENV_PRELUDE + '\\\n' + invoke(E2E_KB_PLANE);
  return KB_ENV_PRELUDE + `
REMOTE_OUT=''
if [ -n "$KB_STORE_TOKEN" ]; then
  REMOTE_OUT=$(${invoke('remote')} 2>/dev/null || true)
  if printf '%s' "$REMOTE_OUT" | python3 -c 'import json,sys; sys.exit(0 if (json.load(sys.stdin).get("candidates") or []) else 1)' 2>/dev/null; then
    printf '%s\\n' "$REMOTE_OUT"; exit 0
  fi
fi
${invoke('local')}`;
}

// Warm-start prompt injection, mirroring expertSkillsBlock exactly: returns '' whenever the feature
// is off or the role is not a consumer, so roleAgent's output is byte-identical to the pre-feature
// build in those cases. KB_REF_DIR is assigned by Module A and nowhere else, so an off run — or one
// whose read found nothing — never reaches the template at all.
function warmStartBlock(role) {
  if (!KB_REF_DIR || !WARM_START_ROLES.has(role)) return '';
  // `tuning_kb=false` is the blind-evaluation switch: it exists so a run can measure what the tuning
  // track finds with NO prior knowledge in its context. Now that the tuning track's prior knowledge
  // lives in the shared KB rather than in a private store, the switch has to reach here too — one
  // flag, one meaning, whichever store the knowledge happens to sit in. Without this, "blind" would
  // have quietly stopped being blind the moment the two stores merged.
  if (role === 'tuning_specialist' && !TUNING_KB_ENABLED) return '';
  return `\n\n## Warm start (a PRIOR run's record — already measured on this box)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/warm_start.md and follow it. The knowledge base ` +
    `offered prior configurations for this exact deployment; they are in ${KB_REF_DIR}/, and ` +
    `${KB_REF_DIR}/measured_on_this_box.md records what happened when THIS run benched them — that ` +
    `file OVERRIDES the stored claims in its siblings wherever the two disagree. A stored number is ` +
    `a hypothesis; only the measured column is evidence.` +
    // The tuning track reads the same pages for a different purpose, so it gets the extra paragraph
    // here rather than a second block: the others are shopping for a CONFIG to adopt, it is checking
    // whether the search it is about to spend hours on has already been run to completion once.
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

function roleAgent(role, phase, intro, inputs) {
  // BACKEND is injected for every role: any role that calls bench_e2e.sh must forward it
  // (BACKEND=<backend>) so the right serving adapter (scripts/adapters/<backend>.sh) is used.
  const inall = {
    BACKEND, SERVING_TP, SERVING_GPU,
    MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS, VALIDATION_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST,
    ...inputs,
  };
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
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

## Inputs
${cfg(inall)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role) + warmStartBlock(role);
}

// Resilient agent wrapper: a single agent failure (transient API 502 / didn't emit StructuredOutput)
// must NOT kill a multi-hour run. Retry a few times, then DEGRADE to null so the caller's existing
// null-guards skip/continue gracefully (critical phases like setup re-throw on null themselves).
// Bound each attempt: an agent LLM call that HANGS (no response, no terminal error) would block this
// await forever — the harness resolves terminal errors to null but not an indefinite hang. Race the
// call against a VERY generous timeout that resolves null, which the loop below treats as a failed
// attempt (retry, then degrade). A true hang never returns, so a generous bound still catches it while
// NEVER killing a legitimately-long agent. The OUTER e2e agents orchestrate the serving stack (Director
// launches sglang + runs the baseline bench ~30min; ConfigTuner does multiple server-launch+bench
// cycles; the head e2e gate overlays + launches + A/B benches) — these run far longer than a kernel
// agent, so the bound must be large (default 120min). Too-short a bound here causes the long setup
// agent to be killed and retried, spawning duplicate exp dirs. args.agent_timeout_ms=0 disables;
// falls back to raw agent() if setTimeout is unavailable.
// Default 120min (generous — outer e2e agents launch servers + run ~30min benches). Fast mode tightens
// it to 45min so a single hung/slow agent can't blow the wall-clock budget (still ample for the director
// baseline + the head e2e A/B). Default mode keeps 120min → unchanged.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : (FAST_MODE ? 2700000 : 7200000), 10);
// Process-safety rule prepended to EVERY agent prompt. It is injected at this funnel
// (the one place `agent()` is ever called) rather than in roleAgent(), because several
// prompts are built inline and would otherwise never see it — and any future call site
// inherits it automatically instead of having to remember it. Bash-capable agents own
// server lifecycles, and a pattern-matched kill is indistinguishable from a correct one
// until it TERMs the caller's orchestrator and fails the whole task.
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

`;
function withProcessSafety(prompt) {
  return typeof prompt === 'string' ? PROCESS_SAFETY + prompt : prompt;
}

// Option A (Hyperloom #1202): near the deadline, tighten each OPTIMIZATION agent's hung-guard so no
// in-flight step finishes later than (budget − reserve). The loop-tops stop STARTING new work at
// TIME_HEAD_DEADLINE_MS, but one iteration is a SEQUENCE of agents (extract → bakeoff → author → verify)
// each armed with the full AGENT_TIMEOUT_MS, and that sequence could overrun the reserve and starve
// Finalize/Report/Validate (the #1202 stall was a 62min extract_op running into the SIGKILL). Final-phase
// agents are EXEMPT — they run INSIDE the reserve by design. The 2min floor lets a deadline-killed agent's
// retries self-limit instead of burning the reserve. Inert when time_budget_s is absent (byte-identical).
//
// Exemption is by POSITION, not by a caller-supplied label. Keying it on opts.phase ∈ ['Finalize',
// 'Report','Validate'] failed OPEN: the Finalize-GATE runs before phase('Finalize') yet tagged its
// integrator agents 'Finalize', so optimization work claimed unlimited time. FINAL_PHASE_STARTED is set
// once, at the phase('Finalize') call site. Agents in flight when it flips keep the cap they were armed
// with — work started before the final phase stays bounded.
let FINAL_PHASE_STARTED = false;
const FINALIZE_GATE_PHASE = 'Finalize-gate';   // display grouping only — not load-bearing for the cap
function agentTimeoutFor() {
  if (TIME_BUDGET_MS == null || FINAL_PHASE_STARTED) return AGENT_TIMEOUT_MS;
  return Math.max(120000, Math.min(AGENT_TIMEOUT_MS, remainingMs() - FINAL_RESERVE_MS));
}

function agentBounded(rawPrompt, opts) {
  const prompt = withProcessSafety(rawPrompt);
  const timeoutMs = agentTimeoutFor();
  if (typeof setTimeout !== 'function' || !(timeoutMs > 0)) return agent(prompt, opts);
  let to;
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [hung-agent guard] ${(opts && opts.label) || 'agent'} exceeded ${Math.round(timeoutMs / 60000)}min with no return — treating as a failed attempt.`);
      resolve(null);
    }, timeoutMs);
  });
  return Promise.race([
    agent(prompt, opts).then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
    guard,
  ]);
}

async function safeAgent(prompt, opts, tries = 3) {
  let lastErr = 'unknown';
  for (let i = 0; i < tries; i++) {
    try {
      const r = await agentBounded(prompt, opts);
      if (r) return r;
      lastErr = 'null/empty result';
    } catch (e) { lastErr = String(e); }
    log(`agent[${(opts && opts.label) || '?'}] attempt ${i + 1}/${tries} failed: ${String(lastErr).slice(0, 160)}`);
  }
  log(`agent[${(opts && opts.label) || '?'}] DEGRADED to null after ${tries} tries (${String(lastErr).slice(0, 120)})`);
  return null;
}

// --- FlyDSL provisioning gate -------------------------------------------------
// Design intent (roles/system_architect.md §0a): the MOMENT strategize routes a flydsl backend onto
// any head/kernel candidate, FlyDSL must be provisioned via the ensure_flydsl expert skill BEFORE any
// consumer can select it. We enforce it in the ORCHESTRATOR — not by trusting the architect LLM to have
// run it — so "decide to use flydsl" and "ensure_flydsl" are one inseparable step. The detect->reuse-or-build
// decision lives ENTIRELY inside ensure_flydsl.sh (its own version gate: import flydsl + kernels.moe_gemm_2stage
// AND __version__ >= FLYDSL_MIN_VERSION, default 0.2.2). is_flydsl_available() is NOT used to decide anything.
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
// Balanced groups are removed innermost-first until the symbol stops changing. One greedy pass
// spans from the FIRST opener to the LAST closer, so `k<a>(t<b>)` loses the '(' that marks the end
// of the name, and a nested `k<pair<a,b>>` leaves the leftover delimiter behind.
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
// Must stay identical to canonical_kernel_name in kernel_selection.py -- the JS gate and the Python
// verdict compare the same two symbols, so any drift lets a kernel pass one side and fail the other.
// Parentheses are stripped as balanced groups rather than by cutting at the first '(': ROCm spells
// the unnamed namespace `(anonymous namespace)`, which puts a parenthesis BEFORE the kernel, so
// cutting there collapsed every such symbol to the return type `void`, and two unrelated kernels
// that both collapsed to `void` then certified each other. Brackets go the same way, which subsumes
// the `[clone .1]` rule: this pipeline appends its own annotations to a kernel name
// (`_fwd_grouped_kernel_stage1 [sliding_attention]`, `main_kernel[prefill]`), and rocprof spells
// memory ops `Memcpy DtoD (Device -> Device)`. The return type is then dropped by name and the FIRST
// identifier taken, as parse_profile.short_name does; taking the last whitespace token also drops
// `void`, but on any of those annotated spellings it answers with the annotation.
function canonicalDeviceKernel(s) {
  const stripped = stripBalanced(stripBalanced(
    stripBalanced(String(s || ''), '[', ']'), '<', '>'), '(', ')');
  // Whatever opener is left never closes, so the symbol was cut off inside it -- profile artifacts
  // store kernel names elided mid-template. The name is what precedes that opener; reading on
  // instead lifts a fragment out of the template arguments and states it with the same confidence
  // as a real name, which then matches the wrong kernel rather than refusing.
  // Removing a group also leaves a gap where it stood, and an unnamed namespace sits mid-
  // qualification: `at::native::(anonymous namespace)::CatArrayBatchedCopy` becomes
  // `at::native:: ::CatArray...`, where the identifier probe stops at the space.
  const cut = stripped.split(/[<([]/)[0].trim().replace(/\s*::\s*/g, '::');
  const match = /^[\w:]+/.exec(cut.replace(/^void\s+/, ''));
  return match ? match[0].split('::').pop().toLowerCase().replace(/[^a-z0-9_]+/g, '') : '';
}
// A name that hit the display limit ends mid-token, so no word boundary can follow it. Tested both
// ways because either side may be the shortened one: heads carry a short_name while traces carry
// the full symbol, and which is which is not fixed.
const truncatedPrefix = (needle, haystack) =>
  needle.length >= SHORT_NAME_LIMIT && haystack.startsWith(needle);
// [arguments, closed] for the symbol's first `<...>`. Reading only what follows the `<` keeps this
// independent of the return type and namespace, which one side routinely spells and the other does
// not. `closed` is false when the symbol was cut off inside the template -- profile artifacts elide
// long names, and only the visible part of an elided argument list can be held against anything.
// Separators become `_` rather than vanishing, so `<128, 4, ...>` cannot read as a prefix of
// `<128, 48, ...>`.
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
  // The base token deliberately drops template arguments so a bare declared name can match its
  // decorated spelling. When BOTH sides carry them the information is present on both, and ignoring
  // it certifies the wrong kernel: one capture here held 20 distinct kernels named
  // at::native::vectorized_elementwise_kernel, separated only by their functor.
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
// A declared depth is a STRUCTURAL claim about the call chain. It orders candidates before any
// runtime evidence exists, so a value that is absent or not a finite number is unusable: `Number()`
// turns it into NaN, every comparison against it is silently false, and a genuinely deeper candidate
// stops being able to outrank anything. Refuse the candidate instead of ranking it as depth 0.
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
  const out = { ...h };
  const required = requiredDeviceKernel(out);
  if (required && !out.device_kernel) out.device_kernel = required;
  // live_call_seam and malformed target_callable values are prose, never machine targets.
  if (!validCallableSpec(out.target_callable)) out.target_callable = '';
  const fused = out.is_fused_kernel === true || out.op_kind === 'moe';
  const deployable = selectionCandidatesForHead(out).filter((candidate) => {
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
function kernelSelectionVerified(h, ext) {
  const required = requiredDeviceKernel(h);
  if (!required) return { ok: true, why: 'not a profiled GPU-head extraction' };
  const target = String((ext && ext.target_callable) || '').trim();
  if (!validCallableSpec(target))
    return { ok: false, why: `target_callable '${target}' is not an exact module:attr spec` };
  const verdict = ext && ext.selection_validation;
  if (!verdict || verdict.contract !== 'kernel_selection')
    return { ok: false, why: 'kernel_selection.py verdict is missing' };
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
  // "Deepest" is settled by the profiler's OBSERVED host-span nesting (kernel_selection.py computes
  // deeper_live_candidates from it), not by the declared `depth`. The declared integer is a claim by
  // the same agent whose choice is under review, and an appended candidate carrying a large depth
  // would otherwise outrank a genuinely deeper one. The structural claim is still checked, but only
  // as an ADDITIONAL way to fail: it can never buy a pass that the observed nesting did not grant.
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

const PRE_FLAGGED_HEADS = [];
function admitHeads(queue, stage) {
  const admitted = [];
  for (const head of (queue || []).filter(Boolean)) {
    const label = head.short_name || head.name || '(unnamed)';
    if (head.entity_kind !== 'gpu_kernel') {
      log(`  ⚠️ FLAG ${label}: entity_kind=${head.entity_kind || 'missing'}; ` +
        `the head track requires a profiler-confirmed gpu_kernel (${stage}).`);
      PRE_FLAGGED_HEADS.push({ short_name: label,
        pct_gpu_time: head.pct_gpu_time, stage, gate: 'wrong_head_granularity',
        reason: `entity_kind=${head.entity_kind || 'missing'}; gpu_kernel required` });
      continue;
    }
    // A gpu_kernel head with NO device identity at all makes requiredDeviceKernel return '', and
    // kernelSelectionVerified then passes it as "not a profiled GPU-head extraction". That vacuous
    // pass is the whole contract switched off for this head, so refuse it at admission instead.
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

// Run a kernel_extractor agent and GUARANTEE it froze a real baseline. safeAgent already retries
// transient failures; this wraps it to ALSO re-extract when the extraction succeeds (smoke passed,
// task dir present) but produced NO frozen baseline — re-invoking with a corrective instruction up
// to BASELINE_EXTRACT_RETRIES times. If a baseline still can't be frozen, we force the caller's
// existing extract-failure path (smoke/unittest_smoke -> 'fail' + a reason) so the head is flagged
// (if dominant) or skipped, NEVER timed against its own scaffold. `role`/`phase`/`intro`/`inputs`
// are the roleAgent args; `opts` is the safeAgent opts (phase/label/schema). Used by every extract
// site (deep, opt-A, milestone/head extract_op, and the non-op milestone extract).
async function extractWithBaseline(role, phase, intro, inputs, opts) {
  const smokeOk = (e) => !!(e && e.task_dir && (e.smoke === 'pass' || e.unittest_smoke === 'pass'));
  const head = (inputs && inputs.KERNEL) || {};
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
    // Best-effort reclaim before the agent retries (issue #429 storage amplification).
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
  }
  const finalSelection = kernelSelectionVerified(head, ext);
  if (smokeOk(ext) && !finalSelection.ok) {
    log(`  ${(opts && opts.label) || role}: kernel selection still unverified after ` +
      `${BASELINE_EXTRACT_RETRIES} re-extractions — ABORTING (${finalSelection.why}).`);
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail', selection_failed: true,
      notes: `kernel selection failed: ${finalSelection.why} — ${ext.notes || ''}` };
  }
  if (smokeOk(ext) && !hasFrozenBaseline(ext)) {
    log(`  ${(opts && opts.label) || role}: STILL no frozen baseline after ${BASELINE_EXTRACT_RETRIES} ` +
      `re-extractions — ABORTING this extraction (refusing a fake speedup vs the candidate's own scaffold).`);
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail',
      notes: `no frozen baseline after ${BASELINE_EXTRACT_RETRIES} re-extractions ` +
        `(baseline_overlay/ + meta.candidate_bind required as the speedup denominator) — ${ext.notes || ''}` };
  }
  return ext;
}

// abDone == the integrator measured BOTH legs (ref + cand) and emitted a real
// verdict. gate:'incomplete' or ab_complete:false means a leg is still missing.
const abDone = (integ) => !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);

// An e2e delta is a RATIO. These are the two numbers it is a ratio of, taken
// off the same A/B that produced it: ref_med is where the workload stood
// before this win, cand_med is where it stood after. Published together, a
// consumer can restate the win in points of one fixed baseline. Published
// alone, the only thing a consumer can do is add percentages that were each
// measured against a different denominator, and those do not compose.
const e2eFrom = (integ) => ({
  e2e_delta_pct: integ.e2e_delta_pct,
  base_tput: integ.ref_med,
  new_tput: integ.cand_med,
});
// The KERNEL_RESULT that a given integrate A/B actually gated. Both patch spellings are read at the
// call sites because the two tracks fill different ones (head=code_patch, milestone=final_patch) and
// tryCorrectiveReauthor deliberately sets BOTH to the corrected patch.
const krOf = (inputs) => (inputs && inputs.KERNEL_RESULT) || {};

// Bank an accepted win, carrying the fields a KB reader needs to REPLAY it on another box.
// e2e_store.py:_accepted_kernels starts from dict(item) so anything here survives verbatim into the
// record — but it LOOKS UP `name`, `language`, `isolated_speedup` and `patch`, while this workflow
// only ever pushed `short_name`, `backend` and `isolated`. Every accepted kernel was therefore
// matched by nothing and silently dropped on the way into the store, which is why both e2e records
// already in the remote KB were written with accepted_kernels:[] and no patch at all. The old
// spellings stay (the report, the resume state and run_e2e.py all read them); the aliases are added
// alongside. `e.patch` wins over the KERNEL_RESULT's because a corrective re-author supplies the
// FIXED patch and the gated one is the broken kernel.
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

// Run ONE integrate A/B and GUARANTEE both legs complete. The first call does a
// normal apply+gate; if the integrator returns incomplete (ran only ref, hung,
// or degraded mid-A/B) we RE-INVOKE it in resume mode to run the MISSING leg,
// up to AB_FINISH_RETRIES times. General contract used by every integrate site
// (head, milestone, finalize-gate, disk-recovered): an A/B never ends after the
// reference leg alone — it is driven to a complete ref+cand measurement.
async function runIntegrateBothLegs(intro, inputs, label, phaseName) {
  // Caller inputs win; the tuning carve-out only ever ADDS keys (and is {} when no tuning was banked).
  const withTuning = {
    MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
    ...tuningIntegrateInputs(), ...inputs,
  };
  let integ = await safeAgent(
    roleAgent('e2e_integrator', 'integrate', intro, withTuning),
    { phase: phaseName, label, schema: INTEGRATE_SCHEMA });
  let tries = 0;
  while (!abDone(integ) && tries < AB_FINISH_RETRIES) {
    tries++;
    log(`  ${label}: A/B incomplete (${integ ? (integ.reason || integ.gate) : 'null/timeout'}); ` +
      `re-invoking integrator to FINISH the missing leg (retry ${tries}/${AB_FINISH_RETRIES}).`);
    integ = await safeAgent(
      roleAgent('e2e_integrator', 'integrate',
        'FINISH this e2e A/B. A reference leg may already exist on disk under $CB/ref — reuse it and ' +
        'run ONLY the missing candidate leg (or re-run both if no usable ref exists), then return ' +
        'accepted/stack/rejected with ab_complete:true. Do NOT return gate:"incomplete" unless a hard ' +
        'hardware/harness fault persists after this retry. If short on time, keep one isolated search ' +
        'replica per leg so BOTH legs still run.',
        { ...withTuning, RESUME_AB: true }),
      { phase: phaseName, label: `${label} (finish ${tries})`, schema: INTEGRATE_SCHEMA });
  }
  return integ;
}

// Lightweight SURGICAL FIX (Tier 1 of the corrective). One focused kernel_surgeon agent reads the reject
// diagnosis + the failing kernel + the live call seam (from the task's meta.json), makes the SMALLEST
// edit that fixes the defect WITHOUT giving up the isolated win, self-verifies on the IMMUTABLE unittest
// (correctness PASS + geomean preserved), and emits a final_patch. Returns a fix object shaped like the
// kernel_workflow result ({final_patch, final_geomean, eval_dir, authored}) so the corrective re-gate path
// is identical, or null if it could not produce a verified patch (=> escalate to the heavy re-author).
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

// Reusable CORRECTIVE RE-AUTHOR (general; used by every head-integration site, any mode). A head
// candidate that PASSED the isolated oracle and ENGAGED live but was REJECTED at the e2e gate for a
// FIXABLE integration reason (JIT/DSL kernel lazily compiling in TP>1 warmup -> NO_BINARY_FOR_GPU /
// cuda_graph_capture_unsafe / capture hang / host-sync) earns up to HEAD_CORRECTIVE_MAX cheap fix-and-
// retries: re-OPTIMIZE the EXISTING kernel (keep the algorithm + isolated win; fix only the integration
// posture), then re-run the both-legs e2e A/B. This is NOT a new head discovery, so it does NOT consume
// HEAD_BUDGET; it is bounded per head and skipped once the kernel-phase wall-clock deadline has fired.
// spec: { short_name, op_kind, shapes, dtype, regime, gpu_id, kernel_eval_dir, task_dir, language,
//         isolated, base_inputs (the integrate inputs template, carries KERNEL_RESULT), reason,
//         cur:{overlay,flags,env,tput} }.  Returns { banked, integ, isolated } (banked=false if
// ineligible or still rejected). See knowledge/learned/method-cudagraph-safe-integration.
async function tryCorrectiveReauthor(spec) {
  let reason = spec.reason || '';
  // Which fix-and-retry class is this reject? '' = terminal (not auto-correctable).
  let fixClass = spec.fix_class || rejectClass(reason);
  const eligible = HEAD_CORRECTIVE_MAX > 0 && !((FAST_MODE && FAST_DEADLINE_HIT) || TIME_DEADLINE_HIT)
    && (spec.kernel_eval_dir || spec.task_dir) && (spec.isolated || 0) > 1.0 && fixClass !== '';
  if (!eligible) return { banked: false };
  const curTput = (spec.cur && spec.cur.tput) || 0;
  // The corrective instruction is CLASS-SPECIFIC. `integration` = the posture is wrong (JIT/capture/
  // host-sync); `correctness` = the output is wrong on the live path (parity/accuracy fail or an
  // implausible speedup), i.e. the kernel over-fit the single captured snapshot. Both KEEP the
  // algorithm + isolated win and fix only the defect. GENERIC — the text describes the failure CLASS,
  // never this specific kernel/model.
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
    `with fresh contents, asserting bit/tol correctness on EACH — a data_ptr cache or stale reuse must fail this. ` +
    `An implausible speedup (far above the op's Amdahl ceiling) means it is doing less/degenerate work — treat as wrong.`;
  for (let cAttempt = 1; cAttempt <= HEAD_CORRECTIVE_MAX; cAttempt++) {
    log(`  ${spec.short_name}: ${fixClass.toUpperCase()} reject (${reason}) — corrective ${cAttempt}/${HEAD_CORRECTIVE_MAX} (fix-and-retry the iso winner; NOT a new head, NOT charged to head_budget).`);
    // TIER 1 — lightweight SURGICAL patch (one agent, minutes). Most rejects are a tiny seam bug.
    let fix = await trySurgicalFix(spec, reason, fixClass, cAttempt);
    const viaSurgical = !!(fix && fix.final_patch && (fix.final_geomean || 0) > 1.0);
    // TIER 2 — escalate to the HEAVYWEIGHT kernel_workflow re-author ONLY if the surgical patch failed.
    if (!viaSurgical) {
      try {
        fix = await fastBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
          kernel_path: spec.kernel_eval_dir || spec.task_dir, workflow_dir: KERNEL_WF_DIR,
          mode: 'optimize', target_language: spec.language || 'triton',
          op_spec: { op_kind: spec.op_kind, shapes: spec.shapes || {}, dtype: spec.dtype || 'bf16', regime: spec.regime || '', cuda_graph_safe: true, ...(spec.workload_path ? { workload_path: spec.workload_path } : {}) },
          perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
          use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
          // A corrective must KEEP the isolated win; adopting a stored patch over it would be exactly the
          // re-discovery the task forbids. History is still readable, just never auto-applied.
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
    // Re-gate the FIXED kernel. Preserve the caller's KERNEL_RESULT SHAPE (head=authored/code_patch,
    // milestone=editable/final_patch) and only swap in the corrected patch + eval_dir + iso speedup, so the
    // helper is track-agnostic. Set BOTH patch fields to the new patch — the integrator reads whichever
    // matches its apply mode; leaving the other stale would re-apply the broken kernel.
    const base = spec.base_inputs || {};
    const fixInputs = { ...base,
      KERNEL_RESULT: { ...(base.KERNEL_RESULT || {}),
        code_patch: fix.final_patch, final_patch: fix.final_patch,
        authored_kernel_eval_dir: fix.eval_dir || spec.kernel_eval_dir || (base.KERNEL_RESULT && base.KERNEL_RESULT.authored_kernel_eval_dir) || '',
        verified_isolated_speedup: fix.final_geomean,
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
    // Implausible-speedup guard: a "win" whose e2e delta blows past the op's Amdahl ceiling is corruption
    // masquerading as a win (does less/degenerate work) — never bank it; treat as a correctness reject and
    // let the next corrective attempt fix it. GENERIC (uses only pct_gpu_time + isolated).
    const implausible2 = ab2 && (integ2.gate === 'accepted' || integ2.gate === 'stack')
      && isImplausibleSpeedup(pctForGuard, fix.final_geomean, integ2);
    if (ab2 && (integ2.gate === 'accepted' || integ2.gate === 'stack') && integ2.e2e_throughput_tok_s > curTput && !implausible2) {
      // Carry the CORRECTED patch out to the caller's bankAccepted. The gated KERNEL_RESULT at the
      // call site still holds the broken kernel, and recording that into the KB would publish a
      // patch that was rejected on this very box as if it were the win.
      return { banked: true, integ: integ2, isolated: fix.final_geomean, patch: fix.final_patch,
        kernel_eval_dir: fix.eval_dir || spec.kernel_eval_dir || '' };
    }
    reason = implausible2
      ? `implausible_speedup (+${(integ2.e2e_delta_pct || 0).toFixed(1)}% >> Amdahl ceiling +${amdahlCeilingPct(pctForGuard, fix.final_geomean).toFixed(1)}% — corruption)`
      : ((integ2 && (integ2.reason || integ2.gate)) || reason);
    // STRUCTURED stop (do NOT re-classify from the prose reason — it may MENTION "corruption" while saying
    // it was RESOLVED, which would loop wastefully). If this attempt made the kernel CORRECT (parity no
    // longer failing) and it was not an implausible speedup, the ONLY remaining problem is throughput/
    // do-no-harm — more re-authoring cannot create Amdahl headroom that isn't there, so STOP now.
    if (ab2 && integ2.output_parity !== 'fail' && !implausible2) {
      log(`  ${spec.short_name}: corrective produced a CORRECT kernel with no e2e win (do-no-harm: ${reason}) — stopping; throughput headroom for this op is exhausted.`);
      break;
    }
    log(`  ${spec.short_name}: corrective still rejected (${reason}).`);
    fixClass = implausible2 ? 'correctness' : rejectClass(reason);
    if (fixClass === '') break;   // new failure not auto-correctable -> stop retrying
    // Progressive: the NEXT attempt builds on this attempt's (partially) fixed kernel, not the original.
    spec.kernel_eval_dir = fix.eval_dir || spec.kernel_eval_dir;
  }
  return { banked: false };
}

// --- FAST-MODE wall-clock control (no-op unless FAST_MODE) -------------------------------------------
// Date.now()/new Date() are unavailable in workflow scripts (they would break resume), so the budget is
// enforced with setTimeout: (1) a one-shot deadline flag that stops the head loop from STARTING new ops,
// and (2) fastBoundedWorkflow() which races each nested kernel workflow against a per-head cap so the
// in-flight op can't overrun. Both are inert when FAST_MODE is off → default path is byte-identical.
let FAST_DEADLINE_HIT = false;
if (FAST_MODE && typeof setTimeout === 'function' && FAST_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    FAST_DEADLINE_HIT = true;
    log(`[fast-mode] head-dispatch deadline (${Math.round(FAST_HEAD_DEADLINE_MS / 60000)}min) reached — no NEW head ops will start; finishing the in-flight head then proceeding to Finalize/Validate within the ${Math.round(FAST_BUDGET_MS / 60000)}min budget.`);
  }, FAST_HEAD_DEADLINE_MS);
}
// Run a nested kernel workflow with a fast-mode time cap. When FAST_MODE is off it returns the raw
// workflow() promise (identical to a direct call); on cap-expiry it resolves null so the caller's
// existing null-guards treat it as "no kernel" and continue.
function fastBoundedWorkflow(ref, wfArgs, label) {
  // noteKernelKB is a pass-through recorder: it returns `r` unchanged, so the caller's null-guards
  // and result handling are untouched. It is attached here (and in deepBoundedWorkflow, and at the
  // two direct workflow() sites) so EVERY lane call reports its kernel-plane recall exactly once.
  const p = workflow(ref, laneArgs(wfArgs)).then((r) => noteKernelKB(r, label));
  if (!FAST_MODE || typeof setTimeout !== 'function' || !(FAST_HEAD_WF_MS > 0)) return p;
  let to;
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [fast-mode] nested kernel workflow ${label || ''} exceeded ${Math.round(FAST_HEAD_WF_MS / 60000)}min — abandoning (null) to stay on budget.`);
      resolve(null);
    }, FAST_HEAD_WF_MS);
  });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}

// --- DEEP-MODE wall-clock control (no-op unless DEEP_MODE) -------------------------------------------
// Same mechanism as fast mode: a one-shot deadline flag stops the deep head scheduler from starting NEW
// co-opt waves once the 20h HeadKernel budget is spent (the in-flight wave + Finalize/Validate still run),
// and deepBoundedWorkflow() caps each nested kernel_workflow BURST so a slow backend can't stall the
// per-wave barrier. Both inert when DEEP_MODE is off → default/fast paths byte-identical.
let DEEP_DEADLINE_HIT = false;
if (DEEP_MODE && typeof setTimeout === 'function' && DEEP_HEAD_BUDGET_MS > 0) {
  setTimeout(() => {
    DEEP_DEADLINE_HIT = true;
    log(`[deep-mode] HeadKernel budget (${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h) reached — no NEW co-opt waves will start; finishing the in-flight wave then proceeding to Finalize/Validate.`);
  }, DEEP_HEAD_BUDGET_MS);
}
// --- DEFAULT-MODE wall-clock control (no-op unless time_budget_s passed) -----------------------------
// fast/deep have their own deadline flags (capped above); the DEFAULT pipeline had NO time awareness at
// all — its Milestone editable-kernel loop and head bake-off track could run past the orchestrator's kill.
// Mirror the fast/deep mechanism with ONE general flag: stop STARTING new head/milestone work after 60% of
// the effective budget (same 60/40 carve as fast — leaves the tail for the in-flight task + Finalize/
// Report/Validate). Active in ALL modes when time_budget_s is set (harmless for fast/deep, which stop even
// earlier on their own flags); fully inert (never registered) when time_budget_s is absent => byte-identical.
let TIME_DEADLINE_HIT = false;
if (TIME_HEAD_DEADLINE_MS != null && typeof setTimeout === 'function' && TIME_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    TIME_DEADLINE_HIT = true;
    log(`[time-budget] dispatch deadline (${Math.round(TIME_HEAD_DEADLINE_MS / 60000)}min of ${Math.round(TIME_BUDGET_MS / 60000)}min budget) reached — no NEW head/milestone work will start; finishing in-flight then proceeding to Finalize/Report/Validate before the hard kill.`);
  }, TIME_HEAD_DEADLINE_MS);
}
// --- RESERVE BOUNDARY ---------------------------------------------------------------------------
// TIME_DEADLINE_HIT above only stops STARTING new head/milestone work; it does not bound the
// Finalize-gate that sits between it and phase('Finalize'), whose unbounded drain loop ate the whole
// reserve in the 20260823 sessions. This is the one flag saying "the reserve has begun": a single
// absolute setTimeout, so unlike remainingMs() it cannot drift. Inert without time_budget_s.
let TIME_FINAL_DEADLINE_HIT = false;
if (TIME_BUDGET_EFFECTIVE_MS != null && typeof setTimeout === 'function' && TIME_BUDGET_EFFECTIVE_MS > 0) {
  setTimeout(() => {
    TIME_FINAL_DEADLINE_HIT = true;
    log(`[time-budget] RESERVE BOUNDARY reached (${Math.round(TIME_BUDGET_EFFECTIVE_MS / 60000)}min of the ${Math.round(TIME_BUDGET_MS / 60000)}min budget spent) — the remaining ~${Math.round(FINAL_RESERVE_MS / 60000)}min belongs to Finalize/Report/Validate. No NEW A/B or measured work starts from here.`);
  }, TIME_BUDGET_EFFECTIVE_MS);
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

// GPU semaphore (FAST MODE only — the default path never constructs one). Hands out EXCLUSIVE leases of
// physical card ids from a pool so two concurrent isolated jobs never share a GPU -> their op-bench /
// kernel-layer speed measurements never contend. Deadlock-free: a waiter holds 0 cards while queued and
// acquires its full count atomically (no hold-and-wait). Uses only Promises/arrays (no Date.now/Math.random,
// which the Workflow runtime forbids).
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

// ===========================================================================
// SINGLE-KERNEL PASS-THROUGH: delegate straight to the unchanged kernel layer.
// ===========================================================================
if (!MODEL_PATH && KERNEL_PATH) {
  phase('Setup');
  log(`Single-kernel pass-through -> ${KERNEL_WF_SCRIPT} on ${KERNEL_PATH}`);
  // Recurse into the UNCHANGED kernel layer via the native workflow() primitive (one allowed level of
  // nesting). kernel_workflow.js returns {eval_dir, final_geomean, final_patch, validation_status, ...}.
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

// OP-IDENTITY GUARD — a fused-MoE / grouped-expert GEMM must be optimized AS the fused op at its live
// dispatcher seam, never decomposed into standalone dense GEMMs (a dense candidate has no live call site
// → no_rebind_seam). So force op_kind='moe' (the grouped-GEMM branch; gemmSynthFor keys on this to keep
// dense synth OFF). The head is never SKIPPED — editability is irrelevant, since a non-editable fused
// kernel is still backend-swapped at its (editable) dispatcher. GENERIC: detects via the Architect's
// is_fused_kernel OR the profile class/name; never keys on a backend name.
// Module-scoped so EVERY strategize/re-strategize call site applies it identically — the initial
// Strategize, the post-config re-strategize, and the post-tuning one — instead of only the first.
// Admission runs here too, so a call site cannot tag identity and then forget to admit.
const _isFusedOp = (c) => (c && c.is_fused_kernel === true) ||
  /(?:^|[^a-z])moe(?:[^a-z]|$)|group(?:ed)?[_ ]?gemm|ck_moe|expert|fused[_ ]?moe|fmoe|asm_moe|fused_custom/i
    .test(`${(c && c.op_kind) || ''} ${(c && c.short_name) || ''} ${(c && c.name) || ''} ${(c && c.classification) || ''} ${(c && c.class) || ''} ${(c && c.backend) || ''}`);
function applyOpIdentityGuard(queue, stage) {
  let tagged = 0;
  for (const c of (queue || [])) {
    if (!_isFusedOp(c)) continue;
    c.op_kind = 'moe';                                                                // grouped-GEMM branch (gemmSynthFor → no dense synth)
    tagged++;
  }
  const admitted = admitHeads(queue, stage);
  if (tagged) log(`[op-identity] ${tagged} fused/grouped head(s): op_kind=moe (never dense-GEMM), bound at live seam — optimized as the fused op, never skipped.`);
  return admitted;
}

// ===========================================================================
// MERGING A RECOVERED CONFIGURATION INTO THE ONE THIS RUN WAS HANDED
// ===========================================================================
// A stored e2e record holds a WHOLE launch configuration, not a delta, because that is the only
// form that can be replayed on a box whose baseline nobody recorded. Replaying it means combining
// it with whatever baseline configuration THIS run was handed — under Hyperloom that is a full
// flag string the run does not own (interface/run_e2e.py seeds `initial_extra_server_args` from
// the EXPLORE result and then sets `config_tune: "false"`, so there is no later sweep to correct
// anything either).
//
// That combination used to be a string concatenation performed by an agent, and it was wrong in a
// way nothing could see. `adapters/sglang.sh` expands `$EXTRA_SERVER_ARGS` straight into argparse
// and `$EXTRA_ENV` straight into `env`, and both resolve a repeated key by last-wins. So when the
// baseline and the record each pin `--context-length`, WHICH ONE APPLIES is decided by the order
// the agent happened to write them in. Lose that coin flip and the server runs the baseline
// configuration twice: ~0% delta, no complaint in any log — the flag WAS honoured, just not with
// the recorded value — and the record is filed as `rejected`. A record that wins gets recorded as
// a loss, which is the one outcome the whole warm start exists to avoid.
//
// So the merge is arithmetic, done here, and it reports what it did. Same rule the launcher
// already applies to one flag by hand (`adapters/sglang.sh`: don't add `--watchdog-timeout` if the
// caller set one), generalized and moved to where the string is built instead of where it is used.
// The output names every key exactly once, so last-wins never gets a vote.

// Flags that legitimately appear more than once. Deliberately tiny: `--lora-path` is a list and
// deduping it would silently drop adapters, while everything else in these configs is a scalar
// where a second occurrence is a mistake we are here to remove. Grow it only with evidence.
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
 * Key-by-key merge of a recovered configuration onto this run's baseline.
 *
 * The recovered value WINS on a collision — it is the thing being tested, and applying the
 * baseline's value instead would test nothing. It wins IN PLACE, so the baseline's ordering
 * survives and the output contains each key once.
 *
 * Returns `{ merged, overrides, added, unchanged }`. `overrides` is the part that has to reach a
 * human and the tuner: it is the list of knobs where this box disagreed with the record, and it is
 * the first thing to look at when a replay comes back neutral.
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

/** `--context-length 11264 -> 13312; MAX_MODEL_LEN 13312 -> 16384`, or "". For prompts and logs. */
const describeOverrides = (overrides) => (overrides || []).map(
  (o) => `${o.flag} ${o.baseline_value == null ? '(unset)' : o.baseline_value} -> ` +
    `${o.recalled_value == null ? '(set)' : o.recalled_value}`).join('; ');

// ===========================================================================
// PHASE: Setup + Baseline profile + Strategize  (gated; else load carried state)
// ===========================================================================
// Module A's outputs. They are folded into the ORDINARY state variables at those variables' real
// declaration sites further down. Module A must run BEFORE Profile so the Top-N is taken on the
// adopted config. Their off-values are 0 / '' / [], which makes each fold a no-op when the feature is off.
let kbSeedTput = 0;
let kbSeedOverlay = '';
const kbSeedKernels = [];
// Spread into the Strategize and ConfigSweep inputs. Stays `{}` unless Module A measured something:
// `{...{}}` contributes no own-properties and preserves key order, so those two prompts are
// byte-identical to the pre-feature build on an off run.
let KB_REF_INPUTS = {};

let EVAL_DIR, MODEL_NAME, BASELINE_TPUT, NOISE_BAND, curFlags, curEnv, curOverlay, profile, strategy, kernelQueue, headQueue;
if (want('setup')) {
  phase('Setup');
  const setup = await safeAgent(
    roleAgent('director', 'setup', 'Build the isolated e2e eval dir and record the baseline throughput.', {
      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT, EVAL_DIR_OVERRIDE, MODEL_NAME_HINT, TASK,
      GPU_IDS, WORKLOAD, INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY,
      MEASUREMENT_PURPOSE: 'parity', REPLICAS: PARITY_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR,
    }),
    { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
  if (!setup || !setup.eval_dir) throw new Error('Setup failed: no eval_dir');
  EVAL_DIR = setup.eval_dir;
  MODEL_NAME = setup.model_name || MODEL_NAME_HINT;
  BASELINE_TPUT = setup.baseline_throughput_tok_s;
  NOISE_BAND = setup.noise_band_pct || NOISE_BAND_DEFAULT;
  // Seed flags/env win when provided (baseline was measured on them); else fall
  // back to whatever the director resolved.
  curFlags = INIT_FLAGS || (setup.server_flags && setup.server_flags.extra) || '';
  curEnv = INIT_ENV || (setup.server_env || '');
  curOverlay = INIT_BASE_OVERLAY;
  log(`Setup done. EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT} tok/s (noise band ${NOISE_BAND}%)`);

  // =========================================================================
  // MODULE A — read the deployment KB, then VALIDATE what it says on this box.
  //
  // Placed HERE, and not earlier, for three reasons that are each load-bearing:
  //   * EVAL_DIR, bench_e2e.sh, BASELINE_TPUT and NOISE_BAND do not exist until Setup returns, and a
  //     candidate cannot be judged without a baseline to judge it against.
  //   * it is before Profile, so the Top-N is captured on the ADOPTED config. This is the same
  //     discipline the flow already asserts by re-profiling after ConfigSweep: a config change moves
  //     which kernels dominate, and routing off a stale profile optimizes the wrong ops.
  //   * it is INSIDE want('setup'), so a phase-partial resume cold-starts automatically rather than
  //     re-reading and re-benching a store on every phase invocation.
  //
  // Nothing below is trusted on the store's word. A stored speedup was measured on another box, on
  // another day, against another baseline; the only thing that makes it actionable here is that this
  // run re-measured it through the same gate a fresh idea would face.
  // =========================================================================
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
      // Refuse to read rather than read the wrong page. An arch-less identity resolves to the
      // `gfx=unknown` rung, whose records were measured on hardware we cannot establish — and the
      // cost of believing one is a 20-40min server launch, not a wasted lookup.
      log('[kb] warm start skipped: read_reason=missing_arch (the Director reported no gfx; a ' +
        'cross-arch config is not a candidate, it is a guess).');
      // Still a recall FACT worth reporting: "we did not look, and here is why" is not the same
      // report as "we looked and the store was empty", and the reader cannot tell them apart from
      // an absent section.
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
          // The read leaves this run's KB ADDRESS on disk as a side effect, at the one moment the
          // dimensions are known and the process is still healthy. Module B below is the happy
          // path; when this process dies before reaching it (6 of the 9 measured runs on
          // 20260821-22), run_e2e.py salvages the numbers from disk and needs this file to know
          // which pages they belong on. Free: no extra agent, no second formatting of the dims.
          // --store/--identity-plane are recorded, not used, by the read: a `both` run reads
          // remote-first, so without them the file would tell the salvage writer "remote" and lose
          // the local mirror this run was configured to keep.
          `  --identity-out ${shq(KB_IDENTITY_FILE)} \\\n` +
          `  --store ${shq(E2E_KB_STORE_DIR)} --identity-plane ${E2E_KB_PLANE}`) + '\n' +
        '```',
        { phase: 'WarmStart', label: 'warm_start:resolve', schema: KB_RESOLVE_SCHEMA }) || {};
      const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
      // Log the ladder VERBATIM. On a scheme with no search, "never recorded" and "recorded under an
      // address one segment different" are the same 404, and this line is the only record of which
      // question was actually asked — without it a silent identity drift looks like an empty store.
      KB_READ_PLANE = String(resolved.plane || '');
      log(`[kb] e2e read: plane=${resolved.plane || '?'} tried=[${(resolved.tried || []).join(' | ')}] ` +
        `answered=${resolved.canonical_id || '?'} tier=${resolved.match_tier || '-'} ` +
        `sorted_by=${resolved.sorted_by || resolved.ranked_by || '-'} ` +
        `champion_metric=${resolved.champion_metric || '-'} reason=${resolved.read_reason || '?'} ` +
        `candidates=${cands.length}`);
      // Record the ASK before anything is benched. If this process dies mid-warm-start the report
      // still knows which ladder was queried, and on a zero-candidate read this is the entire
      // finding — see KB_RECALL's declaration for why the address matters more than the count.
      KB_RECALL.e2e = {
        read_reason: String(resolved.read_reason || ''),
        tried: Array.isArray(resolved.tried) ? resolved.tried : [],
        answered: String(resolved.canonical_id || ''),
        match_tier: String(resolved.match_tier || ''),
        plane: E2E_KB_PLANE, read_plane: KB_READ_PLANE, mode: E2E_WARM_START,
        candidates: cands.length, configs: [], kernels: [],
      };
      if (cands.length) KB_REF_DIR = refsDir;   // arms warmStartBlock() for the consumer roles
      // Armed on the same condition and never separately: the cache holds nothing until a candidate
      // is materialized, and a path to an empty directory reads to an agent as "the file is missing"
      // rather than "there was no record", which are very different things to act on.
      if (cands.length) KB_CACHE_DIR = cacheDir;

      // How many of the offers are worth a server launch. A coarse rung gets its own (smaller)
      // budget rather than zero — see E2E_WARM_START_VALIDATE_N_COARSE for why the stored number's
      // incomparability bounds how much we spend guessing, not whether the guess can be measured.
      const exactTier = (resolved.match_tier || '') === 'exact';
      const benchN = E2E_WARM_START_REF_ONLY ? 0
        : exactTier ? E2E_WARM_START_VALIDATE_N
        : A.e2e_warm_start_validate_n != null ? E2E_WARM_START_VALIDATE_N
        : E2E_WARM_START_VALIDATE_N_COARSE;
      // WHICH offers, once we know how many. The page came back ordered by absolute throughput (the
      // resolve default), and on a coarse rung that throughput was measured at some other tp or
      // workload point, so it ranks the offers by the one quantity that did not survive the move.
      // Re-order by the rung's own metric — speedup — so the launch is spent on the record that
      // claims the biggest RATIO, which is what the store promotes its coarse champions on. Note the
      // page was still FETCHED by throughput, so this reorders a throughput-biased top-N; making the
      // fetch itself follow the rung metric belongs in e2e_store.resolve, not here.
      const benchOrder = exactTier ? cands : [...cands].sort(
        (x, y) => (Number(y.speedup) || 0) - (Number(x.speedup) || 0));
      if (cands.length && !benchN) {
        log(`[kb] not benching: ${E2E_WARM_START_REF_ONLY
          ? (FAST_MODE ? 'fast mode — all optimization comes from the head track' : 'warm_start=reference')
          : `the caller set e2e_warm_start_validate_n_coarse=0 and match tier ` +
            `'${resolved.match_tier}' is not exact`}. The offers stay as references.`);
      } else if (cands.length && !exactTier) {
        log(`[kb] benching ${Math.min(benchN, cands.length)} of ${cands.length} offer(s) from coarse ` +
          `tier '${resolved.match_tier}', picked by stored speedup. The stored throughput is not ` +
          `comparable to this baseline; the A/B below is measured on this box and is what counts.`);
      }

      const verdicts = [];
      for (const c of benchOrder.slice(0, benchN)) {
        // VALIDATE THROUGH THE ORIGINAL GATE. This is config_tuner:sweep — the same role, the same
        // schema, the same bench_e2e.sh at the same TP/GPU, the same delta-vs-median arithmetic, the
        // same parity check and the same swap-took-effect log grep the flow already trusts for a
        // fresh idea. Reusing it rather than writing a warm-start harness is the whole reason a
        // stored config and a proposed config are judged by identical evidence.
        const stored = (c.accepted_config && typeof c.accepted_config === 'object') ? c.accepted_config : {};
        const storedFlags = String(stored.flags || '');
        const storedEnv = String(stored.env || '');
        if (!storedFlags && !storedEnv) {
          verdicts.push({ ...c, measured_tok_s: null, delta_pct: null, parity: 'n/a',
            outcome: 'skipped', why: 'the record carries no config to apply' });
          continue;
        }
        // Key-by-key, not string-concatenated — see mergeConfig. The recorded value wins every
        // collision; `overrides` is the list of knobs where this box and the record disagreed.
        const mf = mergeFlags(curFlags, storedFlags);
        const me = mergeEnv(curEnv, storedEnv);
        const overrides = mf.overrides.concat(me.overrides);
        if (overrides.length) {
          log(`[kb] merge ${c.session_id || '?'}: the recorded config overrides ${overrides.length} ` +
            `knob(s) this run's baseline already pins — ${describeOverrides(overrides)}. ` +
            'The recorded value is the one under test, so it wins.');
        }
        // Nothing left to test. Not a verdict on the record and not worth a server launch: this
        // run's baseline already IS the recorded configuration, so a bench would compare the
        // baseline with itself and file the ~0% result as a loss. Left un-attested on purpose —
        // `recalls` counts attempts on hardware, and this one never reached any.
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
              NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
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

        // ONE repair attempt, and only when the server produced no number at all.
        //
        // A recovered config is replayed on a box whose baseline it has never met, and the first
        // launch is where that shows up: a flag deleted upstream makes argparse exit before the
        // model loads, and a knob this run's baseline pins for a reason it did not record can be
        // incompatible with the recorded value. Both used to end the same way — measured 0, one
        // shot spent, `not_reproduced` filed against the record. The first of those really is the
        // record's fault. The second is not, and a record that keeps meeting incompatible
        // baselines was being retired for it.
        //
        // So: hand the launch failure back, make the tuner say WHICH of the two it is, let it
        // adapt the minimum needed, and bench once more. Exactly once — a repair loop against a
        // config that cannot run here burns server launches to learn nothing new — and through the
        // SAME accept gate, with no allowance made for having been repaired.
        let repair = null;
        if (!((sweep && sweep.best_throughput_tok_s) || 0)) {
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
              `${(sweep && sweep.best_throughput_tok_s) ? ', server came up' : ', still no measurement'}`
            : 'the repair attempt itself produced nothing'));
        }
        const dropped = (repair && Array.isArray(repair.dropped_flags)) ? repair.dropped_flags : [];
        const trial = (sweep && (sweep.trials || [])[0]) || {};
        const measured = (sweep && sweep.best_throughput_tok_s) || 0;
        const parity = String(trial.parity || trial.output_parity || '');
        const deltaPct = BASELINE_TPUT ? ((measured - BASELINE_TPUT) / BASELINE_TPUT) * 100 : 0;
        // Both the tuner's own judgement AND our arithmetic. A warm-start candidate is precisely
        // where an agent is most tempted to ratify a stored claim it did not establish, so the
        // orchestrator re-derives the ratio from the raw number rather than accepting `kept: true`.
        const accept = trial.kept === true && measured > BASELINE_TPUT &&
          deltaPct > NOISE_BAND && parity !== 'fail';
        if (accept) {
          curFlags = sweep.accepted_flags || mf.merged || curFlags;
          curEnv = sweep.accepted_env || me.merged || curEnv;
          kbSeedTput = measured;
          log(`[kb] ADOPTED ${c.session_id || '?'} (${c.direction || 'unlabeled'}): ` +
            `${measured} tok/s, +${deltaPct.toFixed(2)}% vs baseline ${BASELINE_TPUT} (noise band ${NOISE_BAND}%)` +
            `${dropped.length ? `, with ${dropped.join(' ')} dropped to make it run here` : ''}.`);
        }
        // FOUR outcomes. "ran and lost", "could not be made to run" and "does not fit this box"
        // mean three different things to the record: a loss is one box's verdict on a real
        // configuration; a config that never took effect says the record is missing something — a
        // flag renamed upstream, an env the build does not honour; a config that collides with a
        // baseline this run did not choose says nothing about the record at all. Only the middle
        // one is evidence for retiring it, and the store counts all three apart (kb/attest.py).
        //
        // `note` and `notes` are both read: the role file's return schema spells it `note`, and
        // reading only `notes` meant the per-trial text never reached this regex at all.
        const notes = String(trial.notes || trial.note || (sweep && sweep.summary) || '');
        const inert = /\b(?:not (?:honou?red|recognized|recognised|applied|supported)|unrecognized|unrecognised|ignored|no such option|unknown (?:option|argument|flag)|renamed|removed upstream|did not take effect)\b/i.test(notes);
        // The repair's structured verdict outranks the regex: it was reached by reading the launch
        // error, while the regex is guessing at free text written for a human.
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
            `${outcome === 'inapplicable' ? ", it collides with this run's baseline and is no evidence against the record" : ''}` +
            ` — kept as a reference, not applied.`);
        }
        verdicts.push({ ...c, measured_tok_s: measured || null, delta_pct: measured ? deltaPct : null,
          parity: parity || 'unknown', outcome, why: notes,
          // What was actually put on the box, as opposed to what the record asked for. A reader
          // comparing this run's number with the stored one needs both, and `applied_partial`
          // is the difference between "the record lost" and "most of the record lost".
          overrides, applied_partial: dropped.length > 0, dropped_flags: dropped,
          repair_classification: String((repair && repair.classification) || '') });
        if (accept) break;   // adopt the first that passes; the rest stay references
      }
      // ---------------------------------------------------------------------
      // Replay the stored KERNELS through the ordinary integrate gate.
      //
      // Same machinery as the Milestone track, verbatim: runIntegrateBothLegs, the same
      // INTEGRATE_SCHEMA, the same isolated ref/candidate A/B with the parity probe, the same integAccepted
      // predicate. The only thing that differs is where the patch came from, and that is exactly the
      // thing the A/B is there to make irrelevant.
      //
      // Reverting is structural rather than an operation: a candidate lives in its own overlay
      // directory and is activated only by being on PYTHONPATH, so rejecting one means not adopting
      // it. Nothing is ever mutated in the install.
      // ---------------------------------------------------------------------
      const kbKernels = [];
      const seenKernel = new Set();
      for (const c of cands) {
        const bundlePath = (c.bundle && c.bundle.path) || '';
        // The record's PREBUILT overlay directory, packed by e2e_store._pack_overlay as
        // `overlay.tar.gz` (manifest + sitecustomize + `_patched/` + every overlay-root module the
        // rebinds transitively import). Named from the manifest rather than probed, because this
        // script has no filesystem: a bundle whose upload never carried the tarball simply does not
        // list it, and the entry stays a reference instead of costing two server launches to find out.
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
      // One overlay tarball is the whole RECORD's final overlay, not one kernel's. If a record banked
      // three authored kernels, restoring it restores all three at once, so it is benched ONCE and the
      // remaining kernels of that record are marked as covered by it rather than each buying their own
      // pair of server launches to measure the identical directory.
      const overlayReplayed = new Set();
      const kernelVerdicts = [];
      for (const k of kbKernels) {
        const kind = String(k.winner_kind || k.kind || '').trim().toLowerCase();
        // `patch` is stored inside the record's own bundle as `kernels/<name>.patch`, so the path only
        // resolves if the artifacts actually came down. A record whose manifest was never committed
        // downloads a knowledge document and no files, and handing the integrator a path that opens
        // nothing would burn two server launches to discover it.
        const patchPath = (kind === 'patch' && k.patch && k.bundle)
          ? `${k.bundle}/files/${k.patch}` : '';
        // An env/flag win is only replayable if the record actually says WHICH env or flag routed the
        // op. Several records in the store name the kind but carry empty apply_env/apply_flags — they
        // were written before those fields were banked. Sending one to the integrator produces an A/B
        // between two identical configurations: two full server launches to measure nothing, and a
        // 'rejected' verdict that then libels a win which may well have been real.
        const hasRouting = !!(String(k.apply_env || '').trim() || String(k.apply_flags || '').trim());
        // A kind this lane cannot BUILD from a record can still be REPLAYED when the record ships the
        // built artifact. `authored` is the case that matters: the integrator's authored path needs
        // `authored_kernel_eval_dir/workspace/` to git-apply the diff and assemble the module, and that
        // workspace is gone the moment the run ends — which is what the old blanket exclusion was
        // about. But the OUTPUT of that build, the overlay directory, is in the record: manifest,
        // sitecustomize, and the authored source (e.g. `geak_hip_extend/extend_attention_hip.hip`,
        // rebuilt by the reader's own torch load()). The rebind seam is named in the manifest
        // (`sglang...extend_attention:extend_attention_fwd` -> `geak_hip_extend:extend_attention_fwd`),
        // so it is checkable rather than unprovable, and if it does not bind here the A/B says so.
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
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
          KERNEL_RESULT: {
            short_name: k.name, winner_kind: kind,
            // Both spellings, because the two tracks read different ones and this synthetic result
            // has to satisfy whichever the integrator reaches for.
            code_patch: patchPath, final_patch: patchPath,
            apply_env: String(k.apply_env || ''), apply_flags: String(k.apply_flags || ''),
            target_callable: String(k.target_callable || ''),
            source_path_in_sglang: String(k.source_path_in_sglang || ''),
            verified_isolated_speedup: isolated, pct_gpu_time: Number(k.pct_gpu_time) || 0,
            // task_dir is deliberately EMPTY and the provenance is declared foreign — see the intro.
            task_dir: '', provenance: 'knowledge_base_replay',
          },
          // Set only on the prebuilt-overlay path, so the integrator can tell "assemble the candidate
          // from this patch" from "the candidate already exists, unpack it and bench it".
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
              'that is missing part of its own reference. So graft its rebinds ON TOP of the current ' +
              'overlay rather than replacing it:\n' +
              '```bash\n' +
              'CAND="$EVAL_DIR/overlay/cand_<short_name>"\n' +
              'cp -r "$CURRENT_OVERLAY"/. "$CAND"/ 2>/dev/null || mkdir -p "$CAND"   # empty CURRENT_OVERLAY is the normal case\n' +
              `KBO=$(mktemp -d) && tar xzf ${shq(k.overlay_tar)} -C "$KBO" --strip-components=1\n` +
              'cp -r "$KBO"/. "$CAND"/                      # modules + _patched/; manifest handled next\n' +
              'if [ -s "$CURRENT_OVERLAY/_overlay_manifest.json" ]; then\n' +
              '  cp "$CURRENT_OVERLAY/_overlay_manifest.json" "$CAND/_overlay_manifest.json"   # restore, then re-add\n' +
              '  # for each {target,impl_module,impl_attr} in "$KBO/_overlay_manifest.json":\n' +
              '  python3 "$SKILL_DIR/scripts/overlay_setup.py" add-rebind --overlay "$CAND" \\\n' +
              '    --target "<target>" --impl-module "<impl_module>" --impl-attr "<impl_attr>"\n' +
              'fi\n' +
              'PYTHONPATH="$CAND" python3 "$SKILL_DIR/scripts/overlay_setup.py" check --module "<impl_module>"\n' +
              '```\n' +
              'The manifest names every rebind as `target -> impl_module:impl_attr`. VERIFY EACH REBIND ' +
              'ACTUALLY TOOK on the candidate server (load banner, or the check above) before you believe ' +
              'a null result: this overlay was built against a different framework_version, and a rebind ' +
              'whose target module was renamed upstream binds nothing, silently, which is indistinguishable ' +
              'from "the kernel made no difference". If no rebind takes, report gate:"rejected" with ' +
              'reason `no_rebind_seam` rather than a clean no-op. An authored/JIT kernel on the decode ' +
              'path still has to satisfy your CUDA-graph-safety rule — the packed source is rebuilt by ' +
              "this box's own torch load(), and a build failure is a rejection, not a retry.\n"
            : '') +
          '(1) There is NO task_dir and therefore NO immutable oracle for this kernel. Your step-1 ' +
          'provenance re-check cannot run, because the workspace that produced this patch does not ' +
          'exist on this box. Do NOT claim provenance was verified. The FRESH parity probe against the ' +
          'reference leg is the only correctness evidence available here, so run it and report its ' +
          'result plainly — if parity fails or cannot be established, REJECT.\n' +
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
            // On the prebuilt-overlay path there is no patch to point at; what reproduces this win is
            // the accepted overlay the integrator just built from the tarball, and Module B packs that
            // directory the same way the record we replayed was packed.
            e2e_delta_pct: integ.e2e_delta_pct, isolated, patch: patchPath,
            replayed_from_overlay: viaOverlay ? k.overlay_tar : '',
            pct_gpu_time: Number(k.pct_gpu_time) || 0,
            apply_env: String(k.apply_env || ''), apply_flags: String(k.apply_flags || ''),
            // Provenance stays ON the banked entry: this kernel is a recovered win re-verified here,
            // not something this run discovered, and Module B must not later claim otherwise.
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
          // No corrective re-author here. That path re-optimizes a kernel this run authored and owns;
          // a stored patch that fails its fresh gate is simply not this box's win, and the honest
          // outcome is to hand it to the Architect as a lead rather than to repair someone else's diff.
          log(`[kb] kernel ${k.name} (${kind}) not adopted: ${reason || 'A/B did not complete'} — kept as a reference.`);
        }
      }
      // ---------------------------------------------------------------------
      // Demote what did not survive into a REFERENCE for the roles that come next.
      //
      // The resolver's own `e2e_reference_<sha7>.md` is deliberately left alone. It is the pristine
      // record of what the STORE claimed, and the disagreement between that and what this box
      // measured is the interesting datum — editing the claim to match the measurement destroys the
      // only evidence that the two differ. So the measurements go in a sibling file that says, in as
      // many words, that it overrides its neighbours.
      //
      // Three channels, because the first two can each be missed. The reference DIRECTORY is only
      // read if an agent chooses to; the prompt BLOCK only reaches roles in WARM_START_ROLES; the
      // INPUTS entry lands in `## Inputs` unconditionally and is the one that always fires.
      // ---------------------------------------------------------------------
      // ---------------------------------------------------------------------
      // Tell the STORE what happened. Until now this loop's verdicts died with the run: the next
      // box to resolve this identity saw the same optimistic record, benched it, and failed the
      // same way, forever. `e2e_store.py attest` counts the attempt onto the record itself, at
      // every rung, so `validations / recalls` becomes something a later curation pass can retire
      // on. It moves no score and no champion — one box's failure is evidence, not a verdict.
      //
      // Only candidates that were actually PUT ON THIS BOX are counted. A record listed in the
      // offer and never benched (`skipped`, or below benchN) has learned nothing about itself, and
      // counting it would decay the very ratio the retire pass reads.
      //
      // Config verdicts only. A replayed kernel that failed is evidence against the KERNEL lane's
      // record, not against the e2e run that once used it, and the two have separate ledgers —
      // `experience_store.py attest` is where that verdict belongs.
      const attestable = verdicts.filter(v => v.session_id &&
        ['adopted', 'rejected', 'not_reproduced', 'inapplicable'].includes(v.outcome));
      if (attestable.length) {
        const cmds = attestable.map(v =>
          `python3 ${shq(E2E_STORE_SCRIPT)} attest ${kbIdentityFlags()} ${kbPlaneFlags()} ` +
          `--session-id ${shq(v.session_id)} ` +
          `--outcome ${v.outcome === 'adopted' ? 'validated' : v.outcome} ` +
          (v.measured_tok_s ? `--measured-tok-s ${v.measured_tok_s} ` : '') +
          `--baseline-tok-s ${BASELINE_TPUT} --parity ${shq(v.parity || 'n/a')} ` +
          // The note carries what was actually run, not just why it ended: a bare "no win" against
          // a partially-dropped config would read, three months later, as a verdict on the whole
          // record. Overrides first — they are what a reader has to know to interpret the number.
          `--note ${shq([
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
            `an over-counted failure retires a record that may still be right elsewhere.\n` +
            '```bash\n' + KB_ENV_PRELUDE + cmds.join('\n') + '\n```',
            { phase: 'WarmStart', label: 'kb:attest',
              schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) },
            1);
          log(`[kb] attested ${attestable.length} benched record(s): ` +
            attestable.map(v => `${v.session_id.slice(-12)}=${v.outcome}`).join(' '));
        } catch (e) {
          // Non-fatal, like every other KB write. The measurements are already on disk and already
          // in the reference file; losing the counter costs a future reader context, not this run.
          log(`[kb] attest failed (NON-FATAL): ${String(e).slice(0, 200)}`);
        }
      }

      const allVerdicts = verdicts.concat(kernelVerdicts);
      // Same rows the measured_on_this_box.md tables carry, in machine form, so the Report role
      // states the recall outcome from the orchestrator's own record rather than from whether an
      // agent happened to open a reference file. Set OUTSIDE the `if (allVerdicts.length)` guard:
      // an empty verdict list against a non-empty offer means "offered but not benched", which is
      // itself a reportable outcome.
      if (KB_RECALL.e2e) {
        KB_RECALL.e2e.configs = verdicts.map(v => ({
          direction: String(v.direction || 'unlabeled'), session_id: String(v.session_id || ''),
          stored_tok_s: v.throughput_tok_s != null ? v.throughput_tok_s : null,
          stored_speedup: v.speedup != null ? v.speedup : null,
          measured_tok_s: v.measured_tok_s != null ? v.measured_tok_s : null,
          delta_pct: v.delta_pct != null ? v.delta_pct : null,
          parity: String(v.parity || ''), outcome: String(v.outcome || ''),
          // What the merge and the repair did, in the same row as the number they produced. A
          // reader who sees a neutral `delta_pct` needs to know whether the recorded value was the
          // one that ran (`overrides`) and whether all of it ran (`applied_partial`); without them
          // the row reads as a verdict on the record when it may be a verdict on the pairing.
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
        // Split out of `rejected` for the reference section below, but deliberately still counted
        // inside it: for the roles that come next, "lost the A/B" and "never ran" are both "do not
        // re-propose this verbatim". The distinction matters to the STORE, not to the Architect.
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
          '',
          '## Configurations',
          '',
          '| stored direction | stored claim | measured here | delta vs baseline | parity | outcome |',
          '|---|---|---|---|---|---|',
          ...(verdicts.length ? verdicts.map(v =>
            `| ${v.direction || 'unlabeled'} | ${v.throughput_tok_s != null ? v.throughput_tok_s + ' tok/s' : '?'}` +
            `${v.speedup != null ? ` (${v.speedup}x)` : ''} | ` +
            `${v.measured_tok_s != null ? v.measured_tok_s + ' tok/s' : 'not benched'} | ` +
            `${v.delta_pct != null ? (v.delta_pct >= 0 ? '+' : '') + v.delta_pct.toFixed(2) + '%' : '—'} | ` +
            `${v.parity || '—'} | **${v.outcome}** |`)
            : ['| _(none offered)_ | | | | | |']),
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
          // The plan's fourth requirement, made concrete: a top-N entry that could not be
          // reproduced is not discarded, it is handed forward WITH its reproduction material. A
          // lead the next role cannot open is a lead it will not use, so the launch script and the
          // patch paths are spelled out here rather than left to be rediscovered in the bundle.
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

        // The channel that always fires. Terse on purpose — it goes into `## Inputs` of two roles,
        // and a verdict that has to be waded through is a verdict that gets skimmed.
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
  kernelQueue = (strategy && strategy.kernel_candidates) ? strategy.kernel_candidates.slice() : [];
  headQueue = (strategy && strategy.head_candidates) ? strategy.head_candidates.slice() : [];
  headQueue = applyOpIdentityGuard(headQueue, 'strategize');
  log(`Strategy: ${headQueue.length} head candidates, ${kernelQueue.length} kernel candidates, ${(strategy && strategy.config_directions || []).length} config directions.`);
  // strategize decided the backends -> if any candidate routed flydsl, provision it now (blocking).
  await ensureFlydslGate();
} else {
  // Load carried state from a prior phase invocation (args.state).
  EVAL_DIR = ST.eval_dir || EVAL_DIR_OVERRIDE;
  if (!EVAL_DIR) throw new Error('Non-setup phase requires args.state.eval_dir (or args.eval_dir)');
  MODEL_NAME = ST.model_name || MODEL_NAME_HINT;
  BASELINE_TPUT = ST.baseline_throughput_tok_s || 0;
  NOISE_BAND = ST.noise_band_pct || NOISE_BAND_DEFAULT;
  curFlags = ST.flags || '';
  curEnv = ST.env || '';
  curOverlay = ST.overlay || INIT_BASE_OVERLAY;
  profile = { profile_topN_json: ST.profile_topn_json || '' };
  strategy = { config_directions: ST.config_directions || [] };
  kernelQueue = ST.kernelQueue || [];
  headQueue = admitHeads(ST.headQueue || [], 'resume');
  log(`Loaded carried state: EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT}, flags='${curFlags}', env='${curEnv}', ${headQueue.length} head + ${kernelQueue.length} kernel candidates.`);
}

// ===========================================================================
// PHASE: Config sweep (Config Tuner) — FIRST, reshapes the profile
// ===========================================================================
// A carried phase-partial state still wins: it is a measurement this run already made. kbSeedTput is
// 0 when Module A did not run or adopted nothing, so `0 || BASELINE_TPUT === BASELINE_TPUT`.
let curTput = ST.throughput || kbSeedTput || BASELINE_TPUT;
if (want('config') && CONFIG_TUNE_ENABLED && strategy && (strategy.config_directions || []).length) {
  phase('ConfigSweep');
  const sweep = await safeAgent(
    roleAgent('config_tuner', 'sweep', 'Sweep the ranked config axes one at a time; keep wins.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, BASELINE_THROUGHPUT: BASELINE_TPUT,
      NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS, CONFIG_DIRECTIONS: strategy.config_directions,
      CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_OVERLAY: curOverlay,
      MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR, ...KB_REF_INPUTS,
    }),
    { phase: 'ConfigSweep', label: 'config_tuner:sweep', schema: SWEEP_SCHEMA });
  if (sweep && sweep.best_throughput_tok_s > curTput) {
    curFlags = sweep.accepted_flags || curFlags;
    curEnv = sweep.accepted_env || curEnv;
    curTput = sweep.best_throughput_tok_s;
    log(`Config sweep accepted. throughput ${curTput} tok/s (${(curTput / BASELINE_TPUT).toFixed(3)}x). Re-profiling.`);
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
  } else {
    log(`Config sweep found no win above the noise band.`);
  }
}

// ---------------------------------------------------------------------------
// Shared state carried across the head + kernel tracks (and across phase invocations via args.state).
// MUST be declared BEFORE the HeadKernel block that uses them (else temporal-dead-zone ReferenceError).
// ---------------------------------------------------------------------------
let dispatched = 0;                        // counts ONLY kernel-optimization tasks (the budget)
let milestone = 0;
let noImprove = 0;
// kbSeedKernels is empty unless Module A REPLAYED a stored kernel and its fresh A/B accepted it —
// these are this run's own measurements of a recovered patch, not the store's claims about it.
const acceptedKernels = (ST.accepted_kernels || []).slice().concat(kbSeedKernels);
const acceptedHeads = (ST.accepted_heads || []).slice();
// Verified-isolated wins whose e2e A/B did NOT complete (integrate agent timed
// out / hung / degraded to null mid-gate). These are NOT rejections — keep them
// so Finalize can finish the best one's A/B (Fix C) and so a real isolated win
// is surfaced (return.pending_integrations) instead of being silently dropped.
const pendingIntegrations = (ST.pending_integrations || []).slice();
const flaggedHeads = (ST.flagged_heads || []).concat(PRE_FLAGGED_HEADS);   // heads that could NOT be optimized (loudly surfaced, never silently skipped)
let headDispatched = 0;
const history = ST.history || { insights: [], ledger: [], milestones: [], bottleneck_now: '', suggest_next: '' };

// A fused op (op_kind='moe', set by the op-identity guard OR the Architect) is extracted AS the fused op,
// never decomposed into a standalone dense GEMM — so dense-GEMM synth is off for it.
function gemmSynthFor(h) { return (h && h.op_kind === 'moe') ? 'false' : GEMM_SYNTH; }

// ===========================================================================
// PHASE: TuningSkillset — the VENDORED tuning skillset, run WHOLE and STANDALONE, BEFORE HeadKernel.
//
// The skillset lives at TUNING_SKILLSET_DIR as one intact, hash-pinned tree (it is validated standalone,
// so GEAK must run the copy that was validated — see e2e_workflow/scripts/tuning_skillset_sync.py). It is
// NOT decomposed into knowledge/ files and NOT injected as an advisory fragment into other roles: ONE role
// (tuning_specialist) runs its complete six-step loop as ONE phase.
//
// Position is deliberate — after ConfigSweep, before HeadKernel:
//   * BEFORE HeadKernel, because tuning is the cheap lever that reshapes which kernels dominate. Running
//     it first means the head track spends its (expensive) budget on the ops that are still hot AFTER
//     tuning, and every head A/B measures against a reference leg that already contains the tuning.
//   * STANDALONE, because a tuning loop folded into the head bake-off never runs to completion (it
//     collapses into "one more candidate config") and its contribution becomes unattributable. With its
//     own in-session pre/post isolated-server A/B, the final report can state tuning's share of the gain.
//
// An accept is folded into curFlags/curEnv (the deploy's required env IS the engagement mechanism), then
// the profile is re-taken exactly as it is after a config win, because tuning changes the landscape too.
// ===========================================================================
let tuning = ST.tuning || null;
if (want('tune') && TUNING_SKILLSET_ENABLED) {
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
      NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS, ACCURACY_GATE,
      PROFILE_TOPN: profile ? profile.profile_topN_json : '',
      TUNING_TARGETS: (strategy && strategy.head_candidates) || headQueue || [],
      TUNING_SKILLSET_DIR, TUNING_KB_ENABLED, SKILL_DIR: WORKFLOW_DIR,
      // The always-fires channel, same as the Architect's and the config_tuner's. The prompt BLOCK
      // above can be empty (no candidates, or the read never ran); these Inputs entries are how the
      // role learns the store exists at all. Gated by TUNING_KB_ENABLED for the same reason
      // warmStartBlock() is: blind eval has to stay blind through BOTH channels or through neither.
      ...(TUNING_KB_ENABLED ? KB_REF_INPUTS : {}),
      ...(TUNING_KB_ENABLED && KB_CACHE_DIR ? { KB_CACHE_DIR } : {}),
      // The other half of the write below: the kernel store is where THIS phase files its tuned
      // tables, keyed per (op, backend, gfx), so it is also where a later run finds them — and it
      // finds them there even when the run that produced them ended below its own e2e baseline and
      // wrote no deployment-level record at all. Handing over the root and the arch rather than a
      // pre-computed candidate list is deliberate: the role knows which ops it is about to work on
      // (that is the judgement the phase exists for) and can ask per op, which no read done up here
      // before the profile is consulted could do. Same TUNING_KB_ENABLED switch as everything else,
      // so blind evaluation stays blind through every channel.
      //
      // PLANE, not root. The write below is `write-remote --plane both` whenever a service is
      // configured, so the record lands behind a key on the shared store; a directory read against
      // KB_ARTIFACTS_DIR would look at this run's own checkout, which HL creates empty per run and
      // deletes with the run, and would report the miss as `kernel_page_not_found` — indistinguishable
      // from a page that genuinely has nothing. Read and write have to name the same plane or the
      // lane writes into one store and reads from another forever. A read takes exactly ONE plane
      // (see kbResolveScript), so `both` is resolved to `remote` here and the local mirror is offered
      // as the named fallback rather than silently shadowing the service.
      ...(TUNING_KB_ENABLED && KB_DIMS && KB_DIMS.gfx ? {
        TUNED_KB_PLANE: E2E_KB_PLANE === 'both' ? 'remote' : E2E_KB_PLANE,
        TUNED_KB_STORE: E2E_KB_STORE_DIR,
        TUNED_KB_GFX: KB_DIMS.gfx,
        // The page is keyed on arch and op, NOT on dtype — one `fused_moe` page holds the bf16 and
        // the fp8_w8a8 tables side by side, ranked against each other on speedup. Passing this makes
        // the read drop the other dtype's rows instead of ranking them; leaving it empty is the old
        // behaviour (everything offered), which is why an unstated precision is never an error.
        TUNED_KB_PRECISION: KB_DIMS.precision || '',
        TUNED_KB_SCRIPT: KERNEL_WF_DIR + '/scripts/experience_store.py',
        TUNED_KB_ENV_PRELUDE: KB_ENV_PRELUDE,
      } : {}),
    }),
    { phase: 'TuningSkillset', label: 'tuning_specialist:tune', schema: TUNING_SCHEMA });

  // An accept must clear EVERY bar the skillset itself sets. Engagement is the load-bearing one: its
  // stated thesis is that a speedup whose artifact was never proven live has proven nothing, and that
  // failure mode is silent (the artifact lands where nothing reads it and the timing still moves). A
  // completed BOTH-leg A/B is required for the same reason it is on the head track — a post-only number
  // is not a measurement.
  const tuned = tuning && tuning.gate === 'accepted';
  const tuneOk = tuned && tuning.engagement_verified === true && tuning.ab_complete !== false &&
    tuning.correctness_gate !== 'fail' && tuning.post_tune_throughput_tok_s > 0 &&
    tuning.post_tune_throughput_tok_s > (tuning.pre_tune_throughput_tok_s || 0);
  if (tuneOk) {
    if (tuning.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + tuning.apply_env;
    if (tuning.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + tuning.apply_flags;
    // Carry the routing/enabling overlay forward exactly as an accepted head patch does. Without this
    // the code half of a tuning win is silently dropped at the phase boundary and the data half is left
    // bound to nothing — the failure the skillset spends its longest section on.
    if (tuning.apply_overlay) curOverlay = tuning.apply_overlay;
    curTput = tuning.post_tune_throughput_tok_s;
    log(`Tuning skillset ACCEPTED: ${tuning.pre_tune_throughput_tok_s} -> ${curTput} tok/s ` +
      `(+${(tuning.tuning_delta_pct || 0).toFixed(2)}% attributable to tuning; ` +
      `${(tuning.ops_tuned || []).length} op(s), engagement proven` +
      `${tuning.apply_overlay ? `, carrying routing overlay ${tuning.apply_overlay}` : ''}). Re-profiling.`);
    // A missing/unverified deploy bundle does NOT withhold the accept: the gain was measured and proven
    // live, and the artifacts are already installed in this container, so every in-run number stays
    // valid. What breaks is DELIVERY — final_launch.sh would reproduce less than the headline. Finalize
    // re-checks and reports it, but warn here too, because this is the point at which it is still cheap
    // to fix and the easiest failure in this phase to not notice.
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

    // Bank each tuned op as an accepted kernel, so the KB record carries the LEVER and not just the
    // launch script that happens to reference it. This is deliberately done HERE, deterministically,
    // rather than left to Finalize's own accepted_kernels: the DeepSeek-V4-Pro 20260823 run banked a
    // 3.29x tuned a8w8 blockscale table, returned accepted_kernels:[], and wrote a KB record whose
    // files were [final.patch, launch.sh, report.md] — the lever itself was never recorded, and the
    // next run at that canonical id recalls a configuration it cannot reproduce. Gated on tuneOk, so
    // an unproven tuning claim banks nothing, exactly as it folds nothing into curEnv/curFlags.
    const tunedOps = (tuning.ops_tuned || []).filter((o) => String(o.op || o.short_name || '').trim());
    for (const o of tunedOps) {
      bankAccepted(acceptedKernels, {
        short_name: String(o.op || o.short_name).trim(),
        backend: o.backend || '',
        // A tuned table is applied through env (the deploy's env var IS the engagement mechanism),
        // which is already a first-class winner_kind in the store alongside patch/authored.
        kind: 'env',
        isolated: Number(o.isolated_speedup) || 0,
        // The phase measured ONE pre/post A/B covering all of its ops, so its delta is attributable
        // to a single op and only to a single op. With several, the per-op number is left at 0 and
        // the phase-level figure stands alone in tuning_skillset.tuning_delta_pct — stamping the
        // whole delta onto each op would double-count it in any consumer that sums the list.
        e2e_delta_pct: tunedOps.length === 1 ? (tuning.tuning_delta_pct || 0) : 0,
        apply_env: tuning.apply_env || '',
        apply_flags: tuning.apply_flags || '',
        patch: '',
        engaged: o.engaged === true,
        from_tuning_skillset: true,
        tuning_artifact: o.artifact || '',
        // Whether this op was SEARCHED here or RECALLED from a prior record. Now that the tuning
        // track reads the store it also writes, the two are indistinguishable downstream without
        // this field — and a recall re-banked as a discovery would make one original search look
        // like N independent confirmations to anyone counting records.
        tuning_source: String(o.source || o.origin || '').trim() || 'search',
      }, {});
    }
    if (tunedOps.length) {
      const recalled = tunedOps.filter((o) => /recall|kb|knowledge/i.test(String(o.source || o.origin || ''))).length;
      log(`Banked ${tunedOps.length} tuned op(s) as accepted kernels (kind=env) so the KB record ` +
        `carries the lever: ${tunedOps.map((o) => o.op || o.short_name).join(', ')}` +
        `${recalled ? ` (${recalled} recalled from the KB rather than searched)` : ''}.`);
    }

    // ---------------------------------------------------------------------------------------
    // Write each tuned op into the KERNEL lane's experience store, one entry per op.
    //
    // This is not a duplicate of the e2e KB write at the end of the run, and it exists because that
    // write cannot carry this result. The e2e record is keyed on the whole deployment and gated on
    // the run's FINAL throughput_speedup: a tuning win of a few percent that a later phase erases,
    // or a run that ends below its own baseline for reasons that have nothing to do with tuning,
    // takes the tuned table down with it — the search happened, it was proven engaged, and the next
    // run at the same identity searches it again from scratch. The kernel store has no such
    // coupling. It ranks on ISOLATED per-op speedup, which is exactly what a tuned op has, and it is
    // addressed per (kernel, language, gfx), which is exactly the granularity at which a tuned table
    // is reusable. So the durable home for this knowledge is here, and this write is deliberately
    // independent of the run's e2e outcome.
    //
    // Gated on tuneOk, so nothing unproven is filed. Per-op: `isolated_speedup > 1.0` and `engaged`,
    // because an op the phase could not prove live is not knowledge about that op regardless of what
    // the phase-level A/B measured. Best-effort throughout — a failure here loses a record, never a
    // measurement, and must not touch the run.
    const kernelKbOps = KB_DIMS && KB_DIMS.gfx ? tunedOps.filter((o) =>
      Number(o.isolated_speedup) > 1.0 && o.engaged === true && String(o.artifact || '').trim()) : [];
    if (kernelKbOps.length) {
      const storeScript = KERNEL_WF_DIR + '/scripts/experience_store.py';
      // `both` mirrors to the shared service; without a store dir there is nothing to mirror INTO,
      // so it degrades to the plain directory write rather than erroring. Same selection the kernel
      // lane makes at its own write site.
      const remoteOn = E2E_KB_PLANE !== 'local' && !!E2E_KB_STORE_DIR;
      // `--precision` and the two `--serving-*` flags are RECORDED, not keyed — they land in
      // `value.upstream` and never move the entry's address. A tuned table's destination filename
      // encodes its dtype (`...dtype=fp8_w8a8.json`), so a table measured under one precision offered
      // to a run on another installs under a name that runtime never looks up: wasted, not wrong. What
      // it does corrupt is RANKING, by putting an unusable table above a usable one on the same page.
      // Stating precision here is what lets the reader's `--precision` filter drop it first.
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

    // Tuning changed which kernels dominate — re-profile + re-strategize so the head track works the
    // POST-tuning landscape, not the pre-tuning one. Same contract as the post-ConfigSweep re-profile.
    profile = await safeAgent(
      roleAgent('profiler', 'reprofile', 'Re-profile after the tuning skillset deployed its artifacts.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 'tuning',
        // curOverlay, not '': tuning may have banked a routing overlay, and profiling without it would
        // measure a stack where the tuned artifact binds to nothing, then hand the head track a Top-N
        // for the wrong landscape.
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
  } else if (tuned) {
    // Claimed accepted but failed a hard bar. Do NOT bank it — an unproven artifact silently poisons
    // every downstream A/B, which is the exact failure the skillset exists to prevent.
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
// Report inputs for the tuning phase. Empty object when the phase is off/absent, so the Report prompt is
// byte-identical to a build without this feature.
const TUNING_REPORT_INPUTS = (TUNING_SKILLSET_ENABLED && tuning) ? { TUNING_RESULT: tuning } : {};
// Finalize inputs. A tuned DATA artifact cannot ride the PYTHONPATH overlay, so the ONLY way it reaches
// production is for Finalize to fold this bundle into EVAL_DIR/final/ (concatenate its diff into
// final_patch.diff, copy its files, and invoke its deploy.sh from final_launch.sh before the server
// starts). Passed only when the tuning phase actually banked a win, so an unaccepted/absent tuning
// leaves the Finalize prompt byte-identical.
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

// The live-tree carve-out, handed to EVERY integrate leg (see roles/e2e_integrator.md, "never mutate
// site-packages"). That rule asserts `git status --porcelain` is EMPTY in the installed tree before and
// after every gate leg — but an accepted tuning deploy legitimately writes into that tree, because a
// config table only takes effect from inside the package that reads it. Without this list the head track
// sees a dirty tree and either fails the assertion or "restores" it, silently deleting the banked tuning
// out from under the reference leg of every later A/B. The rule's rationale does not apply to these
// paths: they are recorded, reproducible from the deploy bundle, and INTENDED to be in both legs, exactly
// like accepted env. A function, not a const, so it reflects a downgrade-to-no_win and so a resumed run
// picks the carve-out up from carried state. Returns {} otherwise => prompt byte-identical without tuning.
function tuningIntegrateInputs() {
  if (!(TUNING_SKILLSET_ENABLED && tuning && tuning.gate === 'accepted')) return {};
  return {
    TUNING_LIVE_TREE_FILES: tuning.live_tree_files || [],
    TUNING_DEPLOY_BUNDLE: tuning.deploy_bundle || `${EVAL_DIR}/tuning/deploy`,
  };
}

// ===========================================================================
// PHASE: HeadKernel — the highest-pct_gpu_time ops (GEMM / attention), optimized
// regardless of edit flag, via the bake-off ladder. This is the lever the old
// design missed for GEMM (~78% of GPU time). Runs BEFORE the editable-kernel loop.
// The op-identity guard (see Strategize) has already forced fused/monolithic heads to op_kind=moe with
// dense-GEMM synth OFF + the live seam preserved, so each head is optimized AS the op the live kernel
// dispatches (backend-swap / tune / author-fused) — never decomposed into an un-integrable dense GEMM.
// ===========================================================================
if (want('head') && headQueue.length && HEAD_BUDGET > 0) {
  phase('HeadKernel');
  log(`Head-kernel track: ${headQueue.length} candidate op(s), head_budget=${HEAD_BUDGET}, threshold=${HEAD_THRESHOLD_PCT}%.`);
  // Head ops are taken in the Architect's Amdahl-ranked order — no forced GEMM-first reordering.
  const heads = headQueue.slice(0, HEAD_BUDGET).map((c, i) => ({
    ...c, idx: i, gpu_id: GPU_LIST[i % GPU_LIST.length],
    short_name: c.short_name || `${c.op_kind || 'op'}${i}`,
  }));
  if (DEEP) {
    // ============ DEEP-MODE v2: GLOBAL cross-kernel × cross-backend co-optimization ====================
    // One global lane pool over ALL (head op × backend) lanes (kernels + backends optimize concurrently);
    // GPU-elastic serial e2e gate overlapping co-opt; full-backend roster with ceiling-aware patience +
    // revive; per-op SHARED_KB + run-global GLOBAL_KB; budget-driven EV scheduling toward a +50% e2e goal.
    const ROOFLINE_SCHEMA = { type: 'object', additionalProperties: true, properties: { roofline_note: { type: 'string' }, target_geomean: { type: 'number' } }, required: ['roofline_note'] };
    const OKV = { type: 'object', additionalProperties: true, properties: { ok: { type: 'boolean' }, summary: { type: 'string' }, feedback_path: { type: 'string' }, addendum_path: { type: 'string' } }, required: ['ok'] };
    const HARVEST_SCHEMA = { type: 'object', additionalProperties: true, properties: { lanes: { type: 'array', items: { type: 'object', additionalProperties: true, properties: { uid: { type: 'string' }, has_state: { type: 'boolean' }, cumulative: { type: 'number' }, best_ms: { type: 'number' }, vs_live: { type: 'number' }, eval_dir: { type: 'string' }, patch: { type: 'string' } }, required: ['uid'] } } }, required: ['lanes'] };

    // ---- GPU partition (N-adaptive, elastic; serial gate) -------------------------------------------
    // Co-opt runs on cards NOT needed by the serving slot ('main', never paused). The serving cards form
    // a second co-opt pool ('serve') used only in waves with NO due gate; a gate wave runs the e2e A/B on
    // the serving slot CONCURRENTLY with co-opt on the dedicated cards (overlap, no idle). With exactly TP
    // cards (no dedicated), 'main' is empty and a gate wave runs the gate alone (graceful time-slice).
    const servingCards = SERVING_GPU.split(',').map(s => s.trim()).filter(Boolean);
    const servingSet = new Set(servingCards);
    const dedicatedCoopt = GPU_LIST.filter(g => !servingSet.has(g));
    const haveSpare = dedicatedCoopt.length > 0;
    const cooptMain = haveSpare ? makeSem(dedicatedCoopt) : null;          // dedicated cards (never paused)
    const cooptServe = makeSem(haveSpare ? servingCards : GPU_LIST);       // serving cards (idle during a gate wave)
    const mainSlots = haveSpare ? dedicatedCoopt.length : 0;
    const serveSlots = haveSpare ? servingCards.length : GPU_LIST.length;
    log(`[deep] partition: serving {${servingCards.join(',')}} TP=${SERVING_TP}; dedicated co-opt {${dedicatedCoopt.join(',') || '(none)'}}; mode=${haveSpare ? 'OVERLAP gate||co-opt' : 'TIME-SLICE (N==TP)'}; budget ${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h; e2e target ×${DEEP_E2E_TARGET} (${Math.round(BASELINE_TPUT * DEEP_E2E_TARGET)} tok/s).`);

    // ---- ceiling prior: GENERAL (by lane ROLE, not model/shape) -------------------------------------
    const ceilingPrior = (lang, mode) => (mode === 'author' ? 2.2 : (lang === 'triton' && mode === 'optimize' ? 1.6 : 1.8));

    // ---- per-head prep: extract + bake-off + roofline + lane roster (cheap agents on GPU_LIST[0]) ----
    const GLOBAL_KB = `${EVAL_DIR}/deep_head/GLOBAL_KB.md`;
    const prepHead = async (h) => {
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
        const why = ext ? ext.notes || ext.smoke : 'none';
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
      // Lane roster: ALWAYS tune the live editable kernel + author EVERY backend the bake-off proposes
      // (no backend dropped a priori — unbiased), then fill remaining diversity with distinct authoring
      // directions. Steers are DIRECTIONS (generic), never hard-coded magic numbers.
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
          { key: `${liveLang}-fused`, lang: liveLang, mode: 'author', steer: ' DIRECTION=fused-author: author a fresh single-pass FUSED kernel (fold pre/post ops + scaling into the main MFMA core; epilogue-fuse activation). Beat the LIVE kernel.' },
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
        `(c) bootstrap ${sharedKb} (markdown) with sections: Roofline target; Current best per backend (table backend|best geomean|technique|wave — empty now); Techniques that WORK (technique -> measured effect -> source); Dead-ends (scoped, evidence); Cross-backend assignments (borrow); Open hypotheses. Cite relevant ${KERNEL_KNOWLEDGE_DIR} cards (read INDEX in knowledge/learned/ first) for ${ext.op_kind}. ` +
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

    // ---- EV: Amdahl mass × remaining ceiling gap × recent improvement (with exploration floor) -------
    const evOf = (l) => {
      const amdahl = Math.max(0.01, (l.head.pct_gpu_time || 1) / 100);
      const gap = Math.max(0.02, (l.ceiling || 1.5) - l.best);
      const rate = l.ran === 0 ? 0.6 : Math.max(0.03, (l.lastGain || 0));   // unrun lanes get an exploration bonus
      return amdahl * gap * rate;
    };

    // ---- batched HARVEST of a set of lanes from DISK truth (immune to a nulled burst) ----------------
    const harvestLanes = async (set, tag) => {
      if (!set.length) return;
      const byHead = {}; for (const l of set) (byHead[l.head.short_name] = byHead[l.head.short_name] || []).push(l);
      for (const sn of Object.keys(byHead)) {
        const ls = byHead[sn]; const base = ls[0].liveBaselineMs;
        const harvest = await safeAgent(
          `Deep-v2 co-opt harvest [${tag}] for head ${sn} (${ls[0].ext.op_kind}). LIVE baseline geomean = ${base || '?'} ms (the bar; a lane's own "cumulative" is self-relative and NOT comparable — compute speedup vs THIS baseline). Read DISK, do not guess. Lanes:\n` +
          ls.map(l => `- uid=${l.uid}: STATE.json=${l.state_dir}/STATE.json; cumulative-best workspace=${l.state_dir}/best; newest finished run under ${l.deepDir}/runs/${l.key}/team_*/*/ with a final_patch.diff (+ baseline_timing.json / tech_lead_report.md for the optimized per-case ms)`).join('\n') + '\n' +
          `For EACH uid return: uid; has_state (STATE.json AND best/kernel_src exist); best_ms (ABSOLUTE geomean ms of the lane's cumulative-best across cases); vs_live (= ${base || 0}/best_ms when both>0, else 1.0); cumulative (self-relative, ref); eval_dir (newest runs/<key>/team_*/<op> with non-empty final_patch.diff, else ""); patch (that diff path, else ""). Return {lanes:[{uid,has_state,best_ms,vs_live,cumulative,eval_dir,patch}]}.`,
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
          if (e.has_state) l.noStateStreak = 0;   // P3: produced a result -> not infeasible
          // ceiling-aware patience: lanes still far from their ceiling get MORE waves before parking.
          const farFromCeiling = (l.ceiling - l.best) > 0.3 * Math.max(l.ceiling, 1e-9);
          const streakCap = farFromCeiling ? DEEP_PLATEAU_STREAK_HIGH : DEEP_PLATEAU_STREAK;
          if (e.has_state === false && anyState && l.ran > 0) {
            l.active = false; l.noStateStreak = (l.noStateStreak || 0) + 1;
            // P3: 2 consecutive bursts with NO persisted result while peers produced => structurally
            // infeasible backend\u00d7op (the backend has no primitive for this op, e.g. flydsl on grouped-MoE).
            // Mark dead so reseed/revive never waste budget re-trying it.
            if (l.noStateStreak >= 2) { l.dead = true; log(`  [deep] DEAD ${l.uid} \u2014 infeasible (no result ${l.noStateStreak}x while peers produced); won't reseed/revive.`); }
            else log(`  [deep] park ${l.uid} \u2014 no persisted result while peers produced.`);
          }
          else if (l.noImprove >= streakCap) { l.active = false; log(`  [deep] park ${l.uid} (plateau ${l.best.toFixed(3)}x, ceiling ${l.ceiling.toFixed(2)}x).`); }
        }
      }
    };

    // ---- CURATE per-op SHARED_KB + run-global GLOBAL_KB, and REVIVE high-ceiling parked lanes ---------
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
      // revive: a parked lane that is still HIGH-ceiling (big remaining gap) gets one more shot with the
      // freshly-curated cross-backend borrows — so an initially-poor high-ceiling backend is not abandoned.
      let revived = 0;
      for (const l of allLanes) {
        if (l.active || l.dead || (l.revives || 0) >= 2) continue;
        if ((l.ceiling - l.best) > 0.4 * Math.max(l.ceiling, 1e-9)) { l.active = true; l.noImprove = 0; l.revives = (l.revives || 0) + 1; revived++; }
      }
      if (revived) log(`  [deep] revived ${revived} high-ceiling parked lane(s) with fresh borrows.`);
    };

    // ---- BATCHED serial e2e GATE on the serving slot (runs concurrently with co-opt on dedicated cards)
    const runGate = async (opts = {}) => {
      // P4: at FINALIZE, sweep ALL patched lanes (incl. isolated~1.0) onto the CUMULATIVE overlay \u2014 the
      // e2e gate is the true arbiter, so a host-level / oracle-invisible win (isolated~1.0 but real e2e
      // gain, e.g. fused_moe big-M coarsen) still gets banked. Per-wave gates keep the cheap top-2-by-iso.
      const cands = opts.final
        ? allLanes.filter(l => (l.lastEval || l.patch)).sort((a, b) => b.best - a.best)
        : allLanes.filter(l => (l.lastEval || l.patch) && l.best > 1.0).sort((a, b) => b.best - a.best).slice(0, 2);
      if (!cands.length) return;
      const bankedHeads = new Set();   // P4: one kernel per head in the combined overlay (same-head lanes edit the SAME module -> cannot stack; keep the first lane that converts e2e)
      if (opts.final) log(`[deep] FINALIZE gate: sweeping ${cands.length} patched lane(s) onto the cumulative overlay (combined cross-kernel deliverable).`);
      e2eGateCount++;
      log(`[deep] E2E GATE #${e2eGateCount} on serving {${SERVING_GPU}} TP=${SERVING_TP}: [${cands.map(c => c.uid + ' ' + c.best.toFixed(3) + 'x').join(', ')}] (overlapping co-opt on dedicated cards).`);
      for (const c of cands) {
        if (opts.final && bankedHeads.has(c.head.short_name)) { log(`  [deep] FINALIZE: skip ${c.uid} -- head ${c.head.short_name} already banked (same module, cannot stack).`); continue; }
        const deepInputs = {
          EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
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
        const integ = await safeAgent(
          roleAgent('e2e_integrator', 'integrate', 'Apply a deep head candidate; gate on e2e throughput; report engagement/cudagraph/mem/decode for feedback.', deepInputs),
          { phase: 'HeadKernel', label: `integrate ${c.uid} g${e2eGateCount}`, schema: INTEGRATE_SCHEMA });
        if (integ && integ.output_parity === 'fail') {
          log(`  [deep] ${c.uid}: REJECTED — output_parity=fail vs true baseline.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'dead_end', lesson: 'parity fail vs true baseline' });
        } else if (integAccepted(integ, c.head.pct_gpu_time, c.best) && integ.e2e_throughput_tok_s > curTput) {
          curOverlay = integ.accepted_overlay || curOverlay; curTput = integ.e2e_throughput_tok_s; bankedHeads.add(c.head.short_name);
          bankAccepted(acceptedHeads, { short_name: c.head.short_name, op_kind: c.ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', ...e2eFrom(integ), isolated: c.best }, krOf(deepInputs));
          log(`  [deep] ${c.uid}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%); target ${Math.round(BASELINE_TPUT * DEEP_E2E_TARGET)} tok/s.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
        } else {
          // gateRejectReason converts an implausible "pass" into a corruption reject so it routes to the
          // correctness corrective (the deep site already calls the corrective helper for every reject).
          const dreason = gateRejectReason(integ, c.head.pct_gpu_time, c.best);
          const dcorr = await tryCorrectiveReauthor({
            short_name: c.head.short_name, op_kind: c.ext.op_kind, shapes: c.ext.shapes, dtype: c.ext.dtype, regime: c.head.regime,
            gpu_id: SERVING_GPU, kernel_eval_dir: c.lastEval, task_dir: c.ext.task_dir, language: c.lang,
            isolated: c.best, reason: dreason, fix_class: rejectClass(dreason), pct_gpu_time: c.head.pct_gpu_time,
            base_inputs: {
              EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
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
      if (want('profile') && curTput > lastReprofileTput * (1 + DEEP_REPROFILE_GAIN)) {
        log(`[deep] e2e +${((curTput / lastReprofileTput - 1) * 100).toFixed(1)}% since last profile — re-profiling to chase the moving bottleneck.`);
        const rp = await safeAgent(
          roleAgent('profiler', 'reprofile', 'Re-profile the CURRENT overlaid server; return refreshed head pct_gpu_time so EV re-weights toward the new bottleneck.', {
            EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          }),
          { phase: 'HeadKernel', label: `reprofile g${e2eGateCount}`, schema: { type: 'object', additionalProperties: true, properties: { heads: { type: 'array', items: { type: 'object', additionalProperties: true } } } } });
        if (rp && Array.isArray(rp.heads)) {
          for (const nh of rp.heads) {
            const tgt = allLanes.filter(l => l.head.short_name === (nh.short_name || nh.name));
            for (const l of tgt) if (Number.isFinite(nh.pct_gpu_time)) l.head.pct_gpu_time = nh.pct_gpu_time;
          }
          log(`[deep] re-profile updated head Amdahl weights.`);
        }
        lastReprofileTput = curTput;
      }
    };

    // ---- DEPTH: fresh authoring directions to RE-SEED a plateaued lane (so it keeps going DEEPER) ----
    // When a lane plateaus we don't abandon it (and its share of the budget) — we hand it a NEW direction
    // it hasn't tried, biased to the dominant-Amdahl head, so the search compounds depth instead of exiting.
    const DEEP_STEERS = {
      triton: [
        ' DIRECTION=persistent-kernel: a persistent / grid-stride kernel that keeps tiles resident and overlaps global load with MFMA.',
        ' DIRECTION=warp-specialization: split warps into a producer (async global->LDS copy) and a consumer (MFMA) for software pipelining.',
        ' DIRECTION=epilogue-fusion: fuse the scale/activation/cast epilogue into the GEMM to remove a memory round-trip.',
        ' DIRECTION=mfma-layout: re-tune matrix_instr_nonkdim / kpack / LDS swizzle / waves_per_eu / GROUP_SIZE_M for this exact (N,K,M-bucket).',
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
    // RE-SEED when all lanes parked but budget remains: revive lanes (dominant head first) with a FRESH
    // direction so deep optimization uses the FULL budget instead of exiting at the first global plateau.
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

    // ---- GLOBAL WAVE LOOP (runs until the budget; re-seeds on plateau for depth) ---------------------
    let wave = 0, e2eGateCount = 0, lastE2eIsoBest = 1.0, lastReprofileTput = curTput, gateFeedbackPath = '', gateHarnessPath = '', e2eIntervalHit = false;
    const armInterval = () => { if (typeof setTimeout === 'function' && DEEP_E2E_MAX_INTERVAL_MS > 0) setTimeout(() => { e2eIntervalHit = true; }, DEEP_E2E_MAX_INTERVAL_MS); };
    armInterval();
    let convergeStreak = 0, deepBurstsSpent = 0;
    try {
    // Honour TIME_DEADLINE_HIT as well as DEEP_DEADLINE_HIT (both armed off the same deadline) so the deep
    // wave loop can never become the one dispatcher that runs past the reserve.
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
          incremental_analyze: l.ran > 1 ? 'true' : 'false',   // P2: 2nd+ burst of a lane = continuation -> skip cold re-analysis
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
      const newIsoBest = Math.max(1.0, ...allLanes.map(l => l.best));   // P1 convergence: did this wave move e2e OR any lane's isolated best?
      convergeStreak = (curTput > prevTput * 1.001 || newIsoBest > globalIsoBest * 1.001) ? 0 : convergeStreak + 1;
    }
    } catch (e) { log(`[deep] wave loop aborted (${(e && e.message) || e}); proceeding to finalize so progress is still banked + reported.`); }
    // FINALIZE \u2014 always runs (convergence / agent-budget / deadline / throw all land here). Wrapped so
    // that even if it hits the runtime cap mid-sweep, the overlay banked so far (curOverlay/curTput) holds.
    try {
      await harvestLanes(allLanes.filter(l => l.lastEval || l.ran > 0), 'final');
      await runGate({ final: true });   // P4 FINALIZE: every patched lane -> cumulative overlay (the combined cross-kernel deliverable; banks oracle-invisible host/e2e wins like the manual +21.8%)
    } catch (e) { log(`[deep] finalize partial (${(e && e.message) || e}); best banked overlay so far: e2e ${curTput} tok/s.`); }
    log(`[deep] done after ${wave} wave(s), ${e2eGateCount} gate(s). e2e ${curTput} tok/s (${(curTput / Math.max(BASELINE_TPUT, 1e-9)).toFixed(3)}× baseline; target ×${DEEP_E2E_TARGET}). per-lane vs-live: ${allLanes.map(l => l.uid + '=' + l.best.toFixed(2) + 'x').join(', ')}.`);
  } else if (FAST_MODE && GPU_LIST.length > 1) {
    // ========================= FAST-MODE PARALLEL HEAD TRACK (fast-mode only) =========================
    // The default behavior is the byte-identical serial `else` branch below — this whole block only runs
    // when FAST_MODE is on AND there is more than one card. Design (FAST_PLAN §4, STRICT timing):
    //   opt-A  parallel: per-head extract + bake-off, each leasing ONE card exclusively
    //   opt-B  parallel: ALL (operator × direction) author jobs in one pool, each leasing ONE card
    //   BARRIER: every isolated job has released its card -> ISO pool fully idle
    //   integrate SERIAL on the fixed serving slot {SERVING_GPU} (TP=SERVING_TP), one op at a time
    // Why this satisfies the three requirements:
    //   (1) operators AND optimization directions both fan out (flattened (head,language) job pool);
    //   (2) the GPU semaphore gives every op-bench / kernel-layer job an EXCLUSIVE card, so no two
    //       speed measurements ever share a GPU -> no timing contention while optimizing;
    //   (3) integration happens only AFTER the barrier and SERIALLY on the serving slot (which itself
    //       spans all TP cards), so no isolated work can preempt the e2e A/B -> the gate number is clean.
    const ISO = makeSem(GPU_LIST);
    log(`[fast-mode] PARALLEL head track: ISO lanes={${GPU_LIST.join(',')}} (${GPU_LIST.length}); ` +
      `serving slot={${SERVING_GPU}} TP=${SERVING_TP} reserved for the serial e2e gate (hard barrier between).`);

    // ---- opt-A: per-head extract + bake-off, parallel, exclusive 1-card lease each ----
    const prepared = await parallel(heads.map((h) => async () => {
      if (FAST_DEADLINE_HIT) return { h, dead: 'deadline' };
      return ISO.with(1, async (g) => {
        const gpu = g[0];
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

    // ---- process opt-A: dominant-head flagging (never silently skip), seed direct_light + author jobs ----
    const headState = new Map();   // short_name -> { h, ext, cands: [] }
    const authorJobs = [];         // flattened (operator × direction) author directions
    for (const p of prepared) {
      if (!p) continue;
      const h = p.h;
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (p.dead === 'deadline') { log(`  [fast-mode] ${h.short_name}: skipped (dispatch deadline).`); continue;
      }
      if (p.dead === 'extract' || !p.ext || !p.ext.task_dir) {
        const why = p.ext ? p.ext.notes || p.ext.smoke : 'none';
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
              `MEMORY FOOTPRINT IS A HARD CONSTRAINT: use the FUSED fp8 path (fold the block-scale into the operand scale, one fp8 MFMA ` +
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
      // Rank candidates on the WORKLOAD-WEIGHTED speedup (kernel layer's PRIMARY metric, = the env/
      // direct_light candidate's bake.isolated_speedup), NOT the unweighted geomean. In a decode-bound
      // serving regime the geomean is dominated by zero/low-weight prefill buckets and buries the decode
      // bucket that governs throughput — ranking by geomean can drop the truly-best authored kernel.
      const alIso = al.final_weighted != null ? al.final_weighted : al.final_geomean;
      if (al.authored !== false && alIso > 1.0 && al.final_patch) {
        st.cands.push({ kind: 'authored', source: lang, winner_kind: 'authored', language: lang,
          final_patch: al.final_patch, kernel_eval_dir: al.eval_dir, isolated: alIso });
        log(`  ${j.short_name}: authored ${lang} ${alIso.toFixed(2)}x weighted (vs the frozen online kernel, not the seed).`);
      } else {
        log(`  ${j.short_name}: author ${lang} produced no usable kernel (${al ? al.reason || al.validation_status : 'none'}).`);
        history.ledger.push({ direction: `${j.short_name}:${lang}`, verdict: 'dead_end', lesson: al ? al.reason || 'author no speedup' : 'author failed' });
      }
    }

    // ---- integrate SERIAL on the fixed serving slot, in head order, ISO quiesced (no GPU preemption) ----
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
        EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
        KERNEL_RESULT: { short_name: h.short_name, task_dir: st.ext.task_dir, op_kind: st.ext.op_kind,
          winner_kind: cand.winner_kind, winner_backend: cand.source,
          target_callable: st.ext.target_callable || h.target_callable || '',
          authored_language: cand.language || '', authored_kernel_eval_dir: cand.kernel_eval_dir || '',
          apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
          code_patch: cand.code_patch || cand.final_patch || '', tuning_artifact: cand.tuning_artifact || '',
          verified_isolated_speedup: cand.isolated || 0, pct_gpu_time: h.pct_gpu_time,
          // Pass the Architect's live seam + a concrete engagement assertion so the Integrator can
          // VERIFY the overlay actually binds on the live path BEFORE spending a full e2e A/B — an
          // unreachable lever is then rejected in minutes (no_engagement), not hours.
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
        // gateRejectReason injects an implausible_speedup verdict when the gate "passed" but the delta is
        // impossible (corruption) — so a fake win routes to the correctness corrective instead of banking.
        const reason = gateRejectReason(integ, h.pct_gpu_time, cand.isolated);
        const corr = (cand.kind === 'authored' && rejectClass(reason) !== '')
          ? await tryCorrectiveReauthor({
              short_name: h.short_name, op_kind: st.ext.op_kind, shapes: st.ext.shapes, dtype: st.ext.dtype, regime: h.regime,
              gpu_id: SERVING_GPU, kernel_eval_dir: cand.kernel_eval_dir, task_dir: st.ext.task_dir, language: cand.language,
              isolated: cand.isolated, reason, fix_class: rejectClass(reason), pct_gpu_time: h.pct_gpu_time, phase_name: 'HeadKernel',
              base_inputs: {
                EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
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
    // Budget guard: stop STARTING new head ops once a dispatch deadline has fired, so the in-flight work +
    // Finalize/Validate still land inside the wall-clock budget. FAST_DEADLINE_HIT covers fast mode;
    // TIME_DEADLINE_HIT covers ALL modes when time_budget_s was passed (inert otherwise => no-op default).
    if ((FAST_MODE && FAST_DEADLINE_HIT) || TIME_DEADLINE_HIT) {
      log(`[budget] dispatch deadline reached — stopping head dispatch before ${h.short_name} (${headDispatched}/${heads.length} heads done).`);
      break;
    }
    headDispatched++;
    // (h1) Extract the op into a standalone immutable unittest. The op-identity guard already forced a
    // fused/monolithic head to op_kind=moe with GEMM_SYNTH off (gemmSynthFor) so it is extracted as the
    // fused op bound at its live seam — never decomposed into a standalone dense GEMM. Nothing is skipped.
    const ext = await extractWithBaseline(
      'kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: h.gpu_id, WORKLOAD, KERNEL: h, GEMM_SYNTH: gemmSynthFor(h),
        ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        // The unittest MUST span BOTH regimes. Steady-state serving is decode/TPOT-bound, so a
        // head GEMM tuned only on GPU-time-dominant prefill M regresses decode and loses e2e.
        // Pass the decode M explicitly (= running batch ≈ conc) so it is never dropped, plus a
        // per-step M=1. See kernel_extractor.md "Shapes must span BOTH regimes".
        REQUIRE_DECODE_BUCKET: true,
        DECODE_M_BUCKETS: [1, CONC],
        PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
      },
      { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
    const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
    if (!ext || ext.smoke !== 'pass' || !ext.task_dir) {
      const why = ext ? ext.notes || ext.smoke : 'none';
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
      // A DOMINANT head, or a HARNESS fault (not a real no-win), must NEVER be silently skipped.
      // Flag it loudly; and if there is an author_plan, STILL try the author route (it is judged by the
      // immutable unittest, independent of the broken bake-off probe) — so fall through instead of skip.
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
    // Author/rewrite route: write (+optimize) a fresh impl per planned language via the recursive kernel
    // layer. mode=author writes a from-scratch baseline then optimizes it; mode=optimize rewrites an
    // existing editable impl. The immutable oracle in ext.task_dir is the judge for both.
    const plan = (bake.author_plan || []).slice(0, HEAD_AUTHOR_MAX);
    for (const ap of plan) {
      const lang = ap.language || 'triton';
      let al;
      // Retry the nested author on a TRANSIENT/early failure (threw, or returned with no real
      // optimization: no final_geomean) — a transient nested-workflow death must NOT silently drop a
      // language (it dropped FlyDSL in the 2026-06-12 run). Do NOT retry a COMPLETED no-speedup
      // (final_geomean present but <=1.0) — that's a real result, retrying just wastes budget.
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
              `(fold the block-scale into the operand scale, run ONE fp8 MFMA GEMM — the "kill the dequant" lever) and cache ` +
              `only COMPACT fp8/preshuffled weights (~the model's own fp8 weight size), never a bf16 expansion. The integrated ` +
              `kernel MUST fit at the same mem-fraction the accepted config uses. ` + GRAPH_REQ + (TASK || ''),
            apply_to_original: 'false',
          }, `${h.short_name}:${lang}`);
        } catch (e) { al = { authored: false, validation_status: 'error', reason: String(e) }; }
        const transient = !al || al.validation_status === 'error' || (al.authored === false && al.final_geomean == null);
        if (!transient || attempt === AUTHOR_TRIES) break;
        log(`  ${h.short_name}: author ${lang} attempt ${attempt}/${AUTHOR_TRIES} died transiently (${al ? al.reason || al.validation_status : 'null'}) — retrying so this language isn't dropped.`);
      }
      // Rank on the WORKLOAD-WEIGHTED speedup (kernel layer's PRIMARY metric, same basis as the
      // env/direct_light candidate's bake.isolated_speedup) — NOT the unweighted geomean, which in a
      // decode-bound regime is dragged down by low/zero-weight prefill buckets and can bury the decode
      // bucket that actually governs serving throughput. Fall back to geomean if no weighted number.
      const alIso = al && (al.final_weighted != null ? al.final_weighted : al.final_geomean);
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

    // (h3) e2e gate. Test EVERY candidate, not just the top-iso one: isolated speedup is a weak predictor
    // of e2e (a decode-bound run had an iso-1.055 env tune deliver +14.5% e2e), so the truly-best candidate
    // can only be found by measuring each on the live server. All of a head's candidates A/B against the
    // SAME current config, so the REFERENCE leg is identical for all of them — measure it ONCE (first
    // candidate) and reuse it for the rest via REUSE_REF/SHARED_REF_MED (the integrator then skips the
    // reference server launch and only benches the candidate leg). CAND_TAG keeps each candidate's cand
    // bench/overlay in its own dir so they don't clobber each other. Pick the candidate with the highest
    // MEASURED e2e that clears the gate; only if NONE clears do we fall back to corrective/pending/reject on
    // the top candidate. Inputs are built per-candidate so Fix C can re-issue the SAME A/B at Finalize.
    const mkIntegrateInputs = (cand, ci, sharedRefMed) => ({
      EVAL_DIR, MODEL_PATH, GPU_ID: h.gpu_id, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
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
      // Persist EVERY candidate's MEASURED e2e (one ledger row per backend, keyed short_name:backend) so the
      // report + experience library keep the full per-backend picture — not just the winning backend.
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
      // Best MEASURED e2e among all candidates that cleared the gate. A head winner may be carried as an
      // overlay (authored/patch) AND/OR config (env/flag) — capture both.
      const cand = bestPick.cand, integ = bestPick.integ;
      curOverlay = integ.accepted_overlay || curOverlay;
      if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
      if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
      curTput = integ.e2e_throughput_tok_s;
      bankAccepted(acceptedHeads, { short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: cand.winner_kind, ...e2eFrom(integ), isolated: cand.isolated }, krOf(bestPick.inputs));
      log(`  ${h.short_name}: ACCEPTED best candidate=${cand.source} (${(cand.isolated || 0).toFixed(2)}x iso). e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else {
      // No candidate cleared the gate. Operate on the TOP (highest-weighted) candidate: corrective re-author
      // for a fixable reject/crash, else keep an incomplete A/B as PENDING, else a plain reject. Same
      // three-state handling as before, now applied to the best candidate after testing them all.
      const cand = top.cand, integ = top.integ, headIntegrateInputs = top.inputs;
      if (!top.abc) {
        // A FIXABLE crash-during-warmup (cand server died -> ZERO A/B samples) is the most common corrective
        // case (cuda_graph_capture_unsafe / host-sync / NO_BINARY_FOR_GPU). Try the corrective re-author
        // first; only keep PENDING if it is not fixable (a real transient timeout/hang) or the fix didn't land.
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
        // gateRejectReason injects an implausible_speedup verdict when the gate "passed" but the delta is
        // impossible (corruption); rejectClass then routes parity/accuracy/implausible rejects to the
        // correctness corrective and JIT/capture/host-sync rejects to the integration corrective.
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
  // Head wins reshape the profile massively (GEMM mass shrinks) — re-profile before the kernel loop.
  if (acceptedHeads.length) {
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
// Floor: keep dispatching until >= MIN_KERNEL_TASKS editable-kernel tasks have run, THEN allow the
// noImprove early-stop. While below the floor the loop never stops on no-improve / empty plan.
// Budget: when time_budget_s was passed, TIME_DEADLINE_HIT stops STARTING new milestones (even below the
// floor) so the in-flight work + Finalize/Validate finish before the orchestrator's hard kill. The guard
// is inert when time_budget_s is absent (TIME_DEADLINE_HIT never set) => byte-identical default behavior.
while (want('kernel') && !TIME_DEADLINE_HIT && dispatched < BUDGET && (dispatched < MIN_KERNEL_TASKS || noImprove < 2)) {
  milestone++;
  const remaining = BUDGET - dispatched;
  const belowFloor = dispatched < MIN_KERNEL_TASKS;

  // --- (a) Plan this milestone (Architect): nominate next kernels. While BELOW the floor the Architect
  // MUST nominate (it may not stop); it draws fresh editable candidates from the re-profile + the broad
  // candidate pool, never re-using a confirmed e2e-null direction verbatim. ---
  phase('Milestone');
  const plan = (milestone === 1 && kernelQueue.length)
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

  const planCandsRaw = (plan && plan.kernel_candidates) ? plan.kernel_candidates : [];
  // pct_gpu_time gate: only optimize kernels above MILESTONE_MIN_PCT (a candidate missing pct is kept,
  // not silently dropped — but logged). This gate OVERRIDES the min-floor: low-pct kernels are not worth it.
  const planCands = planCandsRaw.filter(c => c.pct_gpu_time == null || c.pct_gpu_time >= MILESTONE_MIN_PCT);
  const skipped = planCandsRaw.filter(c => c.pct_gpu_time != null && c.pct_gpu_time < MILESTONE_MIN_PCT);
  if (skipped.length) log(`Milestone ${milestone}: skipped ${skipped.length} kernel(s) below ${MILESTONE_MIN_PCT}% GPU [${skipped.map(c => `${c.short_name || '?'}@${(+c.pct_gpu_time).toFixed(1)}%`).join(', ')}].`);
  if (!planCands.length) {
    if (planCandsRaw.length) log(`Milestone ${milestone}: stop — no remaining kernel clears the ${MILESTONE_MIN_PCT}% GPU bar (Amdahl: sub-threshold kernels can't move e2e). Floor is overridden by the pct gate.`);
    else if (belowFloor) log(`Milestone ${milestone}: below floor (${dispatched}/${MIN_KERNEL_TASKS}) but Architect nominated nothing — cannot fabricate candidates; stopping.`);
    else log(`Milestone ${milestone}: stop (floor ${MIN_KERNEL_TASKS} met). ${plan ? plan.reasoning || '' : ''}`);
    break;
  }

  const cands = planCands.slice(0, remaining).map((c, i) => ({
    ...c, idx: i, gpu_id: GPU_LIST[i % GPU_LIST.length],
    short_name: c.short_name || `k${milestone}_${i}`,
  }));
  dispatched += cands.length;
  log(`Milestone ${milestone}: ${cands.length} kernel candidate(s) [${cands.map(c => c.short_name).join(', ')}], dispatched ${dispatched}/${BUDGET} (floor ${MIN_KERNEL_TASKS})`);

  // --- (b) PARALLEL optimize (extract + recursive kernel layer per candidate, on distinct GPUs), then
  // SERIAL integrate. The optimize stage is independent per kernel -> run concurrently. The e2e integrate
  // stage MEASURES throughput and COMPOUNDS the overlay, so it must be serial: no two servers benched at
  // once (no timing conflict) and accepted overlays carry forward in order.
  const optimized = await parallel(cands.map((c) => async () => {
    const ext = await extractWithBaseline(
      'kernel_extractor', 'extract', 'Capture shapes + oracle; emit an immutable unittest task dir.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: c.gpu_id, WORKLOAD, KERNEL: c,
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
      },
      { phase: 'Milestone', label: `extract ${c.short_name}`, schema: EXTRACT_SCHEMA });
    if (!ext || ext.editable === false || ext.unittest_smoke !== 'pass' || !ext.task_dir) {
      return { c, skip: true, reason: `extraction failed/non-editable (${ext ? ext.notes || ext.unittest_smoke : 'none'})` };
    }
    // RECURSIVE kernel layer on the IMMUTABLE task dir (one allowed nesting level via workflow()).
    let kl;
    try {
      const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, laneArgs({
        kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
        use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
        ...KB_ARGS,
        budget: KERNEL_BUDGET, gpu_ids: c.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
        task: 'Compare candidate backends ' + JSON.stringify(c.candidate_backends || []) +
          ' for this kernel; pick the fastest that passes the immutable unittest. ' + GRAPH_REQ + (TASK || ''),
        apply_to_original: 'false',
      }));
      noteKernelKB(r, String(c.short_name || ext.op_kind || 'milestone'));
      kl = { ran: true, kernel_eval_dir: r.eval_dir, final_patch: r.final_patch,
        final_geomean: r.final_geomean, validation_status: r.validation_status,
        note: (r.winner && r.winner.source) || '' };
    } catch (e) {
      kl = { ran: false, final_patch: '', final_geomean: 0, validation_status: 'error', note: String(e) };
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
      history.ledger.push({ direction: c.short_name, verdict: 'dead_end', lesson: o.reason });
      continue;
    }
    const { ext, kl } = o;
    if (!kl || !kl.ran || !(kl.final_geomean > 1.0) || !kl.final_patch) {
      log(`  ${c.short_name}: kernel layer produced no speedup (${kl ? kl.final_geomean : '?'}x); skipping integrate.`);
      history.ledger.push({ direction: c.short_name, isolated_speedup: kl ? kl.final_geomean : 0, verdict: 'dead_end', lesson: 'no isolated speedup' });
      continue;
    }
    log(`  ${c.short_name}: kernel layer ${kl.final_geomean.toFixed(2)}x isolated. Integrating to e2e.`);
    // Build inputs ONCE so Fix C can re-issue the SAME A/B for a pending win at Finalize.
    const mileIntegrateInputs = {
      EVAL_DIR, MODEL_PATH, GPU_ID: c.gpu_id, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
      KERNEL_RESULT: { short_name: c.short_name, task_dir: ext.task_dir,
        source_path_in_sglang: ext.source_path_in_sglang, target_callable: ext.target_callable,
        final_patch: kl.final_patch, verified_isolated_speedup: kl.final_geomean, pct_gpu_time: c.pct_gpu_time },
      CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
      CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
    };
    const integ = await runIntegrateBothLegs(
      'Overlay the optimized kernel back; gate on e2e throughput.', mileIntegrateInputs,
      `integrate ${c.short_name}`, 'Milestone');

    // Three-state gate (see head track): incomplete A/B => PENDING, not rejected.
    // Backward-compatible: null (timeout/hang/degrade) or an explicit incomplete
    // flag => not done; a legacy return without ab_complete behaves as before.
    const abDone = !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);
    if (abDone && integAccepted(integ, c.pct_gpu_time, kl.final_geomean) && integ.e2e_throughput_tok_s > curTput) {
      curOverlay = integ.accepted_overlay || curOverlay;
      curTput = integ.e2e_throughput_tok_s;
      bankAccepted(acceptedKernels, { short_name: c.short_name, backend: kl.note || '', kind: 'patch', ...e2eFrom(integ), isolated: kl.final_geomean }, krOf(mileIntegrateInputs));
      milestoneImproved = true;
      log(`  ${c.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: c.short_name, isolated_speedup: kl.final_geomean, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else {
      // Unified reject/incomplete handling (same as the head track): a FIXABLE reject OR fixable
      // crash-during-warmup (ab_complete=false) of an iso-verified editable kernel earns a corrective
      // re-author (re-optimize the EXISTING kernel; NOT charged to HEAD_BUDGET); only keep PENDING if the
      // A/B was merely incomplete for a NON-fixable reason (real transient timeout/hang). See tryCorrectiveReauthor.
      const reason = gateRejectReason(integ, c.pct_gpu_time, kl.final_geomean);
      const corr = (rejectClass(reason) !== '')
        ? await tryCorrectiveReauthor({
            short_name: c.short_name, op_kind: ext.op_kind, shapes: ext.shapes, dtype: ext.dtype, regime: c.regime,
            gpu_id: c.gpu_id, kernel_eval_dir: kl.kernel_eval_dir, task_dir: ext.task_dir, language: kl.language || '',
            isolated: kl.final_geomean, base_inputs: mileIntegrateInputs, reason, fix_class: rejectClass(reason), pct_gpu_time: c.pct_gpu_time, phase_name: 'Milestone',
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
        pendingIntegrations.push({ track: 'milestone', short_name: c.short_name, isolated: kl.final_geomean,
          pct_gpu_time: c.pct_gpu_time, inputs: mileIntegrateInputs, backend: kl.note || '',
          partial: integ ? { gate: integ.gate, ref_med: integ.ref_med, cand_med: integ.cand_med, reason: integ.reason } : null });
        log(`  ${c.short_name}: INTEGRATE INCOMPLETE — A/B not finished (${integ ? integ.reason || integ.gate : 'null/timeout'}); kept as PENDING (not a rejection).`);
        history.ledger.push({ direction: c.short_name, isolated_speedup: kl.final_geomean, verdict: 'incomplete', lesson: integ ? integ.reason || 'A/B not finished' : 'integrate timed out/null before A/B completed' });
      } else {
        log(`  ${c.short_name}: REJECTED at e2e gate (${reason}).`);
        history.ledger.push({ direction: c.short_name, isolated_speedup: kl.final_geomean, e2e_delta_pct: integ.e2e_delta_pct || 0, verdict: 'dead_end', lesson: reason || 'no e2e gain' });
      }
    }
  }

  // --- (c) If improved: re-profile + grow the experience library ----------
  if (milestoneImproved) {
    noImprove = 0;
    profile = await safeAgent(
      roleAgent('profiler', 'reprofile', 'Re-profile the new best server.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: milestone,
        OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        ...ANALYSIS_SKILL_INPUTS,
      }),
      { phase: 'Profile', label: `profiler:reprofile m${milestone}`, schema: PROFILE_SCHEMA });
  } else {
    noImprove++;
  }

  // --- (d) Update the persistent experience library + in-run memory -------
  const exp = await safeAgent(
    roleAgent('system_architect', 'update_experience', 'Curate knowledge/learned/ (merge/insert >=2-star / archive contradicted) per learned/README.md.', {
      ROUND: milestone, EVAL_DIR, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
      MILESTONE_RESULTS: history.ledger.slice(-cands.length),
      REPROFILE_SHIFT: profile ? profile.shift_note : '', PRIOR_HISTORY: history,
      ...ANALYSIS_SKILL_INPUTS,
    }),
    { phase: 'Milestone', label: `architect:experience m${milestone}`, schema: EXPERIENCE_SCHEMA });
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

// --- Fix C: EVERY incomplete A/B must be finished (both legs) before final ----
// Completeness guarantee (general, not per-kernel): an A/B that measured only the
// reference leg (gate:'incomplete' / ab_complete:false) is NOT a result. Before
// finalizing we
//   (1) RECONSTRUCT from disk every overlay whose integrate_result shows the A/B
//       never completed both legs — so incompletes from ANY integrate branch, or
//       from a prior crashed/killed/resumed run (the JS has no fs and no carried
//       state across a process restart), are recovered, then
//   (2) drive EVERY pending integration to a complete ref+cand measurement
//       (runIntegrateBothLegs re-invokes the integrator to run the missing leg).
// This replaces the old "finish only the single best, and only when nothing else
// was accepted" gate, which stranded sibling incompletes (e.g. an env-config
// head) at ref-only whenever any other head/kernel had been accepted. Keys ONLY
// off the stable overlay/cand_*/integrate_result.json layout. Bounded by
// AB_FINISH_RETRIES + each agent's agentBounded guard + the outer runner budget
// + TIME_FINAL_DEADLINE_HIT (this gate must NEVER spend the final reserve; see below).
if (want('final')) {
  // Issue #429: reclaim leftover process-local capture dirs before Finalize burns more disk.
  // Run even when TIME_FINAL_DEADLINE_HIT — reclaim is cheap local FS work, not a bench.
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
  // (0) The reserve is not ours. Each iteration below boots a server and runs two benches, so past
  // the boundary there is no time to finish even one: skip the whole gate, scan agent included.
  // Pending entries stay in pendingIntegrations and reach the caller — deferred, not discarded.
  if (TIME_FINAL_DEADLINE_HIT) {
    log(`Finalize-gate: SKIPPED entirely — the final reserve has already begun. ` +
      `${pendingIntegrations.length} pending A/B(s) deferred to the caller (surfaced in pending_integrations).`);
  } else {
  // (1) Disk-reconstruct incomplete A/Bs. The JS has no fs, so a read-only Bash
  // agent enumerates the overlay layout and reports which A/Bs never completed.
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
        EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
        KERNEL_RESULT: {
          short_name: it.short_name, op_kind: it.op_kind || '', winner_kind: it.winner_kind || '',
          target_callable: it.target_callable || '', apply_env: it.apply_env || '',
          apply_flags: it.apply_flags || '', code_patch: '', tuning_artifact: '',
          verified_isolated_speedup: it.isolated || 0, pct_gpu_time: it.pct_gpu_time,
          parity_note: 'expected_close',
        },
        // The candidate overlay is already built on disk; the integrator benches
        // it directly on resume (RESUME_AB) — it does NOT rebuild it.
        CAND_OVERLAY_DIR: it.overlay_dir, RESUME_AB: true, SKILL_DIR: WORKFLOW_DIR,
      },
    });
    log(`Finalize-gate: disk-reconstructed incomplete A/B "${it.short_name}" ` +
      `(winner_kind=${it.winner_kind || '?'}, ref=${!!it.ref_present}, cand=${!!it.cand_present}); queued to finish.`);
  }

  // (2) Drive EVERY pending integration to a complete ref+cand measurement — but never
  // past the reserve boundary. Highest isolated speedup first, so if the clock cuts the
  // queue short the most valuable A/B is the one that got finished.
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
    const p = pendingIntegrations.shift();   // pop one per iteration => guaranteed termination
    log(`Finalize-gate: finishing pending A/B (${p.short_name}, ${(p.isolated || 0).toFixed(2)}x isolated); ` +
      `${pendingIntegrations.length} more queued.`);
    const integ = await runIntegrateBothLegs(
      'Finish the e2e A/B for this verified-isolated win. Run BOTH legs (reference + candidate) to ' +
      'completion, then return accepted/stack/rejected with ab_complete:true.',
      // Pin the CURRENT carried overlay/flags/env/throughput so the A/B is
      // measured against the latest accepted baseline.
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
if (want('final')) {
  if (TIME_BUDGET_MS != null) log(`[time-budget] entering the final phase with ~${remainingMin()}min of the ${Math.round(TIME_BUDGET_MS / 60000)}min budget left (reserve was ${Math.round(FINAL_RESERVE_MS / 60000)}min).`);
  FINAL_PHASE_STARTED = true;   // the one place the reserve exemption is granted — by position, not label
  phase('Finalize');
  finalize = await safeAgent(
    roleAgent('e2e_integrator', 'finalize', 'Assemble the final overlay + patch + launch script bundle.', {
      EVAL_DIR, FINAL_OVERLAY: curOverlay, ACCEPTED_FLAGS: curFlags, ACCEPTED_ENV: curEnv,
      ACCEPTED_KERNELS: allAccepted, BASELINE_THROUGHPUT: BASELINE_TPUT, SKILL_DIR: WORKFLOW_DIR,
      ...TUNING_FINALIZE_INPUTS,
    }),
    { phase: 'Finalize', label: 'e2e_integrator:finalize', schema: FINALIZE_SCHEMA });
  finalTput = (finalize && finalize.final_throughput_tok_s) || curTput;

  phase('Report');
  report = await safeAgent(
    roleAgent('system_architect', 'report',
      'Write architect_report.md AND the full final_report.md in English (with the Phases tree + ' +
      'artifacts tree modules). final_report.md MUST contain a "## Knowledge-base recall" section ' +
      'built from KB_RECALL — see your role file. Write it even when nothing was recalled: report ' +
      'the exact canonical ids that were tried and the read_reason. On an exact-lookup store a miss ' +
      'and a never-recorded page are the same 404, so the address asked is the finding, and a reader ' +
      'who cannot see it cannot tell "no prior art" from "prior art one segment away".', {
      EVAL_DIR, HISTORY: history, BASELINE_THROUGHPUT: BASELINE_TPUT, FINAL_THROUGHPUT: finalTput,
      KB_RECALL,
      ACCEPTED_CONFIG: { flags: curFlags, env: curEnv }, ACCEPTED_KERNELS: allAccepted,
      ACCEPTED_HEADS: acceptedHeads, FLAGGED_HEADS: flaggedHeads, MILESTONES: milestone, BUDGET_USED: dispatched, BUDGET, MIN_KERNEL_TASKS,
      PROFILE_TOPN: profile ? profile.profile_topN_json : '', WORKLOAD, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
      ...ANALYSIS_SKILL_INPUTS, ...TUNING_REPORT_INPUTS,
    }),
    { phase: 'Report', label: 'architect:report', schema: REPORT_SCHEMA });

  // Validate is sized into FINAL_RESERVE_MS, so there should be room for it here; it is not time-GATED
  // (attempting it beats skipping when the clock has drifted). If the hard kill lands mid-bench the loss is
  // bounded — Report already wrote architect_report.md + final_report.md, and run_e2e.py falls back
  // (director_e2e_validation.json → best overlay's integrate_result.json), so a real win still reaches the caller.
  phase('Validate');
  validation = await safeAgent(
    roleAgent('director', 'validate', 'Independently re-measure throughput + parity; arbitrate; then reconcile the report with the validated numbers.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], BASELINE_THROUGHPUT: BASELINE_TPUT, NOISE_BAND_PCT: NOISE_BAND,
      BASELINE_OVERLAY: INIT_BASE_OVERLAY,
      FINAL_OVERLAY: (finalize && finalize.final_overlay) || curOverlay,
      FINAL_FLAGS: { flags: curFlags, env: curEnv },
      CLAIMED_THROUGHPUT: finalTput, WORKLOAD, APPLY_TO_ORIGINAL, E2E_REPEATS,
      MEASUREMENT_PURPOSE: 'validation', REPLICAS: VALIDATION_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR,
      // The Report phase already wrote these files with the Finalize-bundle bench (the Director had not
      // run yet). After validation the Director MUST review + rewrite their headline throughput / speedup
      // / TTFT / TPOT (and status/parity) to its authoritative same-session numbers, so report-vs-director
      // can never disagree. Paths default to the standard EVAL_DIR names if the report result is absent.
      ARCHITECT_REPORT: (report && report.report_path) || `${EVAL_DIR}/architect_report.md`,
      FINAL_REPORT: `${EVAL_DIR}/final_report.md`,
    }),
    { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });
  // A Validate that did NOT produce a usable number (e.g. its server crashed in
  // engine-core init) must NEVER erase the accepted same-session A/B win we
  // already carry in finalTput/curTput. Trust the Director's independent number
  // ONLY when it is a real positive measurement; otherwise fall back to the
  // carried best-accepted throughput so a real, parity-checked win is never
  // reported as 0 / no_gain downstream.
  validatedOk = !!(validation && validation.director_verified_throughput_tok_s > 0 && validation.throughput_speedup > 0);
  finalSpeedup = validatedOk ? validation.throughput_speedup : (BASELINE_TPUT ? finalTput / BASELINE_TPUT : finalSpeedup);
  log(`COMPLETE. ${MODEL_NAME}: ${BASELINE_TPUT} -> ${validatedOk ? validation.director_verified_throughput_tok_s : finalTput} tok/s ` +
    `(${finalSpeedup ? finalSpeedup.toFixed(3) : '?'}x, status ${validation ? validation.validation_status : '?'}` +
    `${validation && !validatedOk ? '; Validate produced no number — using carried accepted A/B win' : ''}). Results in ${EVAL_DIR}`);

  // --- FINAL experience reconciliation (post-gate) -----------------------
  // The per-milestone update_experience (in the Milestone loop above) is the ONLY
  // curate step, and it runs on that milestone's ledger. But a head/kernel whose
  // e2e A/B was still INCOMPLETE at milestone time is recorded as "Integrator
  // pending", and the Finalize-gate (Fix C) that actually CONFIRMS its e2e win runs
  // AFTER the last milestone curate — as does the Director's Validate. Without a
  // pass here, those cards stay frozen at "pending" and never receive the verified
  // gated number (see the 20260812 Qwen3.5-27B run: the +16.1% head card was left
  // at "e2e transfer NOT yet gated"). Re-curate ONCE now, with the authoritative
  // post-Validate numbers, so finalize-gate confirmations are written back.
  if (allAccepted.length) {
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
  eval_dir: EVAL_DIR, model_name: MODEL_NAME, baseline_throughput_tok_s: BASELINE_TPUT,
  noise_band_pct: NOISE_BAND, flags: curFlags, env: curEnv, overlay: curOverlay, throughput: curTput,
  profile_topn_json: profile ? profile.profile_topN_json : '',
  config_directions: (strategy && strategy.config_directions) || [],
  headQueue, kernelQueue, accepted_heads: acceptedHeads, flagged_heads: flaggedHeads, accepted_kernels: acceptedKernels,
  // Full tuning-phase result, so a phase-by-phase resume does not re-run the tuning loop and the Report
  // phase still has the attribution numbers when it runs in a later invocation.
  ...(tuning ? { tuning } : {}),
  // Carry pending (verified-isolated, A/B-incomplete) wins WITH their inputs so a
  // resumed phase run can finish their A/B instead of re-discovering them.
  pending_integrations: pendingIntegrations,
  history,
};

// What this run got from the KB, and — separately — what it added on its own. The split is the
// point: a warm-started run's headline speedup is partly inherited, and reporting the whole of it as
// this run's work is how a knowledge base starts flattering itself. BASELINE_TPUT stays the true
// cold baseline throughout, so `throughput_speedup` above is still measured against the real floor.
const kbWarmStart = (E2E_WARM_START_ON && KB_DIMS) ? {
  identity: KB_DIMS, plane: E2E_KB_PLANE, read_plane: KB_READ_PLANE, mode: E2E_WARM_START,
  reference_dir: KB_REF_DIR,
  adopted_throughput_tok_s: kbSeedTput || null,
  adopted_kernels: kbSeedKernels.map(k => k.name).filter(Boolean),
  // The gain that is genuinely THIS run's: everything after the warm start's own contribution. Null
  // when nothing was adopted, because then the ordinary speedup already answers the question.
  incremental_speedup: kbSeedTput ? Number((finalTput / kbSeedTput).toFixed(6)) : null,
  // The full recall ledger, both planes. Same object the Report role was handed, so the prose in
  // final_report.md and whatever a downstream consumer reads cannot disagree about what was
  // recalled or how it fared.
  recall: KB_RECALL,
} : null;
// Attribution block for the standalone tuning phase. Because the phase measured its OWN interleaved
// pre/post legs, its contribution can be expressed both absolutely (delta%) and as a SHARE of the run's
// total gain — which is the number "how much did the tuning actually help?" is asking for. share is null
// (not 0) when the run had no net gain to apportion, so a caller never divides by a meaningless total.
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
    // How the win reaches production. Consumers that need to reproduce the tuning outside this run read
    // deploy_bundle (its deploy.sh is idempotent); final_launch.sh already invokes it.
    deploy_bundle: tuning.deploy_bundle || '', deploy_verified: tuning.deploy_verified === true,
    cache_invalidation: tuning.cache_invalidation || [], live_tree_files: tuning.live_tree_files || [],
    apply_overlay: tuning.apply_overlay || '',
    // Finalize's independent answer to "did the tuning actually make it into the shipped bundle, and
    // does it still engage there?". null when Finalize did not run (a phase-scoped invocation).
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

const wfReturn = {
  // schema_version pins the CONTRACT shape run_e2e.py reads. Bump only on a
  // breaking change to the keys below; run_e2e.py keys its canonical-artifact
  // read off this so it never silently mis-parses a future shape.
  schema_version: 1,
  mode: 'e2e',
  fast_mode: FAST_MODE,   // true => ConfigSweep + Milestone skipped; HeadKernel-only within the time budget
  deep_mode: DEEP_MODE,   // true => HeadKernel runs the long cross-backend co-optimization scheduler (20h)
  backend: BACKEND,
  phases_run: PHASES,
  eval_dir: EVAL_DIR,
  model_name: MODEL_NAME,
  baseline_throughput_tok_s: BASELINE_TPUT,
  // Trust the Director's independent number ONLY when it is a real positive
  // measurement (validatedOk). A crashed/degenerate Validate falls back to the
  // carried accepted same-session A/B win — never 0 — so a real, parity-checked
  // gain is not silently demoted to no_gain by run_e2e.py.
  final_throughput_tok_s: validatedOk ? validation.director_verified_throughput_tok_s : finalTput,
  throughput_speedup: finalSpeedup,
  validation_status: validatedOk ? validation.validation_status
    : (validation ? `${validation.validation_status || 'flagged'}_no_number_used_carried_ab`
       : (want('final') ? 'unknown' : 'phase_partial')),
  output_parity: validation ? validation.output_parity : 'unknown',
  accepted_config: { flags: curFlags, env: curEnv },
  accepted_kernels: acceptedKernels,
  accepted_heads: acceptedHeads,
  // Standalone tuning-skillset phase: its own measured pre/post legs plus the share of the run's TOTAL
  // gain those legs account for. Reported separately from accepted_heads/accepted_kernels precisely
  // because the phase is standalone — "how much did tuning buy" is answerable, not folded into the total.
  tuning_skillset: tuningReturn(),
  // Verified-isolated wins whose e2e A/B never completed (timeout/hang mid-gate).
  // A REAL isolated speedup that simply ran out of A/B time — surfaced (slim) so
  // the caller can resume/finish it, never silently discarded as a rejection.
  pending_integrations: pendingIntegrations.map(p => ({
    short_name: p.short_name, track: p.track, isolated: p.isolated,
    pct_gpu_time: p.pct_gpu_time, partial: p.partial || null,
  })),
  flagged_heads: flaggedHeads,   // dominant heads surfaced but not optimized (harness/extract/no-candidate) — never silently dropped
  config_tune_enabled: CONFIG_TUNE_ENABLED,
  head_budget: HEAD_BUDGET,
  head_used: headDispatched,
  milestones: milestone,
  budget_used: dispatched,
  budget_total: BUDGET,
  final_overlay: (validation && validation.final_overlay) || (finalize && finalize.final_overlay) || curOverlay,
  // FINALIZE_SCHEMA has always defined final_patch and the Finalizer has always returned it, but it
  // was never surfaced here — so the single diff carrying the whole run's win reached neither the
  // report nor the KB (e2e_store.py's _ARTIFACT_KEYS looks up exactly result["final_patch"]).
  // final_overlay cannot stand in for it: that is a DIRECTORY, and the store's os.path.isfile filter
  // drops it, so a run with no final_patch uploads no reproducible code at all.
  final_patch: (finalize && finalize.final_patch) || '',
  final_launch_script: (validation && validation.final_launch_script) || (finalize && finalize.final_launch_script) || '',
  report_path: report ? report.report_path : `${EVAL_DIR}/architect_report.md`,
  // Conditional spread, never an unconditional key: with the feature off this contributes no own
  // property and both the returned object and the persisted file are byte-identical to before.
  ...(kbWarmStart ? { kb_warm_start: kbWarmStart } : {}),
  // Emitted independently of kb_warm_start. kbWarmStart is null whenever KB_DIMS was never
  // established (a preflight that produced no gfx), but the KERNEL lanes can still have recalled by
  // then — folding recall only into kbWarmStart would drop exactly the runs where knowing what was
  // recalled matters most. Absent entirely when nothing was recalled on either plane.
  ...((KB_RECALL.e2e || KB_RECALL.kernel.length) ? { kb_recall: KB_RECALL } : {}),
  state: carryState,
};

// ---------------------------------------------------------------------------
// Canonical handoff to run_e2e.py — write the SCHEMA-VALIDATED return to a FIXED
// on-disk path from INSIDE the workflow.
//
// WHY (the perfect-cooperation contract): run_e2e.py must produce a TRUTHFUL
// result.json regardless of HOW the agent ran this Workflow. Newer Claude CLIs
// run a Workflow invocation as a NON-BLOCKING background task: the orchestrating
// turn ends before completion, so this return value is NEVER echoed to the
// transcript for run_e2e.py to scrape. The script itself has no fs, so we have an
// agent (the only thing with Write) persist `wfReturn` to
// <eval_dir>/workflow_return.json as the workflow's FINAL act. run_e2e.py reads
// THAT one contracted file first (deterministic, no transcript scrape, no parsing
// of free-form per-candidate files).
//
// Best-effort by design: if this persist fails (agent variance), the returned
// value is UNCHANGED and run_e2e.py still falls back to its layered, schema-robust
// disk recovery (director_e2e_validation.json -> accepted intermediate win ->
// measured-baseline no_gain). So correctness never DEPENDS on this single write —
// it is the preferred fast path, not the only path.
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
//
// STRICTLY AFTER the workflow_return.json persist above, and that ordering is not stylistic.
// workflow_return.json is the pinned contract run_e2e.py depends on, documented as the workflow's
// final act precisely because this process gets SIGKILLed. Module B is the one operation in the
// whole workflow that waits on a NETWORK rather than a GPU, and putting it in front of the persist
// would trade the run's canonical artifact for a knowledge-base entry. Feeding the KB that same file
// as --result has a second benefit: the record cannot drift from the report, because they are the
// same bytes.
//
// EVERY --apply IS PERMANENT. The service exposes no DELETE for /v1/kb/*, so a wrong record cannot
// be cleaned up, only outranked. Hence the gates below are conservative by design.
// ===========================================================================
// The Director's verdict, not the raw ratio, decides whether a same-session A/B is a real win: a
// declared no-win can still carry a ratio ABOVE 1.0 when a low outlier in the base leg depresses its
// median, so the two mechanically-identical legs' ratio squeaks over 1.0 while the run bought
// nothing (see the 20260822 gemma-4-26B-A4B run: 1.0215x raw same-session, but 0.9453x vs the
// provided baseline, validation_status=validated_no_win, empty overlay + empty patch). Keying the
// write on the ratio alone wrote that BELOW-baseline number in as the exact-id champion and buried
// the real framework-layer win beneath it. A KB write is PERMANENT (no delete), so a verdict the
// Director already computed to mean "no win" must never mint a champion record, whatever the ratio
// rounded to. `startsWith` also catches the `..._no_number_used_carried_ab` fallback spelling.
const kbNoWinVerdict = ['validated_no_win', 'recovered_no_gain']
  .some((s) => String(wfReturn.validation_status || '').startsWith(s));
if (E2E_WARM_START_ON && KB_DIMS && KB_DIMS.gfx && want('final') && EVAL_DIR &&
    wfReturn.throughput_speedup > 1.0 && wfReturn.final_throughput_tok_s > 0 && !kbNoWinVerdict) {
  // Computed HERE, deterministically, from facts this script already holds — never asked of an
  // agent. `direction` is inside _content_digest, so a label that varies between two runs of the
  // same configuration mints a second session instead of replacing the first, and the page fills
  // with near-identical entries that outrank each other by noise. '' (unlabeled) is the honest
  // answer when nothing distinct happened; the store collapses unlabeled records with nothing.
  const kbDirection = [
    (curFlags !== INIT_FLAGS || curEnv !== INIT_ENV) ? 'config' : '',
    (acceptedKernels.length || acceptedHeads.length) ? 'kernels' : '',
    FAST_MODE ? 'fast' : (DEEP_MODE ? 'deep' : ''),
  ].filter(Boolean).join('+');
  const writeCmd =
    KB_ENV_PRELUDE +
    `python3 ${shq(E2E_STORE_SCRIPT)} write ${kbIdentityFlags()} ` +
    `${kbPlaneFlags()} --result ${shq(EVAL_DIR + '/workflow_return.json')} ` +
    // --require-win re-asks, inside the writer, the same question the `if` above already answered.
    // Not redundancy for its own sake: run_e2e.py's salvage path is a SECOND writer, and the two
    // must refuse the same results. One gate implementation (e2e_store.win_gate), two callers —
    // the arrangement that stops the next gate fix from reaching only half the write paths.
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
    // Non-fatal, exactly like the persist above. The run's result is already on disk and already
    // returned; failing to record it is a lost opportunity, not a lost measurement.
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

return wfReturn;
