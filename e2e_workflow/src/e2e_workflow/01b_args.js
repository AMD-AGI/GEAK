const RDNA4_ISA = EXPECTED_GFX === 'gfx1201';
const R9700_E2E = EXPECTED_TARGET === 'r9700';
// ISA isolation and product policy are intentionally separate. Any gfx1201
// target must avoid CDNA priors, but only exact R9700 identity selects its
// calibrated peaks, serving backend, and image policy.
const RDNA4_ISOLATE = RDNA4_ISA;
const E2E_LEARNED_KB_ENABLED = !RDNA4_ISOLATE;
const MATRIX_CORE_NAME = RDNA4_ISOLATE ? 'WMMA' : 'MFMA';

// The kernel workflow's learned KB is OFF by default for lanes launched from here, and ON by default
// when kernel_workflow is driven directly. Same worker, different prior: a kernel_workflow campaign
// re-optimizes a fixed benchmark set, where a card distilled from a past run of the same kernel is
// the point; e2e extracts whatever the profiler happens to surface from a live server, and a card
// carrying another run's conclusions about a superficially similar op is a prior nobody asked for.
// Off is also the conservative default for the layer that has never been measured with it on.
// Override per run with args.use_learned_kb=true.
const LANE_USE_LEARNED_KB = String(A.use_learned_kb != null ? A.use_learned_kb : 'false');
// Select the policy inside the existing worker, without another workflow level.
if (A.optimizer !== undefined && !['greedy', 'evo'].includes(A.optimizer)) {
  throw new Error('optimizer must be greedy or evo');
}
const EXPLICIT_KERNEL_BUDGET = typeof A.kernel_budget === 'string' && /^\+?\d+$/.test(A.kernel_budget.trim())
  ? Number(A.kernel_budget) : A.kernel_budget;
if (A.kernel_budget !== undefined && (!Number.isSafeInteger(EXPLICIT_KERNEL_BUDGET) || EXPLICIT_KERNEL_BUDGET < 1)) {
  throw new Error('kernel_budget must be a positive safe integer');
}

// EVERY nested lane invocation goes through here. Setting the flag at each call site instead would
// be the defect this repo keeps re-making — there are seven call sites today, and the eighth would
// silently take the lane's own default (on) with nothing to catch it. test_e2e_lane_defaults.py
// fails if a `scriptPath: KERNEL_WF_SCRIPT` call is added that does not route through this.
// llm_stats is forwarded ONLY when the parent was given one explicitly, so an unset parent keeps the
// original argument shape (child defaults on) and a parent opt-out (`llm_stats:"false"`) reaches the
// lane instead of silently reverting to the lane's default. wfArgs spreads last so a per-call override wins.
const laneArgs = (wfArgs) => {
  const out = {
    use_learned_kb: LANE_USE_LEARNED_KB,
    ...(A.optimizer !== undefined ? { optimizer: A.optimizer } : {}),
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
// Started HERE, at the earliest point the run has a durable identity, so the
// delegation graph/timeline is being recorded from the first agent rather than
// reconstructed only at the end. It resolves its own workflow run from EXP_ROOT
// via the runtime's workflow record (no eval-dir substring matching), polls the
// journal + transcripts, and republishes a snapshot atomically.
//
// It is detached, read-only, and strictly non-fatal: it never influences model
// choice, prompts, budgets or optimization decisions, and any failure to start
// is logged and ignored. Set GEAK_LIVE_TRACE=0 to disable.
startLiveTracker(EXP_ROOT, WORKFLOW_DIR);
function startLiveTracker(expRoot, wfDir) {
  try {
    if (String(process.env.GEAK_LIVE_TRACE || '1') === '0') return;
    const { spawn } = require('child_process');
    const interval = String(process.env.GEAK_LIVE_TRACE_INTERVAL_S || '30');
    const child = spawn('python3', [
      '-B', `${wfDir}/../interface/geak_trace_collector.py`,
      '--exp-root', expRoot,
      // Runtime-supplied invocation identity: the wf_*.json record carries this
      // same args object, so matching it identifies THIS launch deterministically
      // instead of guessing from timing or from being the only run around.
      '--identity-args', JSON.stringify(A),
      '--script-dir', wfDir,
      // Per-run filename: two runs under one exp_root must not overwrite each other.
      '--out-dir', expRoot,
      '--watch', '--interval', interval,
      '--resolve-timeout', '600',
      '--max-seconds', String(process.env.GEAK_LIVE_TRACE_MAX_S || '172800'),
    ], { detached: true, stdio: 'ignore' });
    // spawn reports a missing executable ASYNCHRONOUSLY: try/catch cannot see it,
    // and without this listener the ENOENT is an unhandled error event.
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
if (MODEL_PATH && (!EXPECTED_GFX || !EXPECTED_TARGET)) {
  throw new Error(
    'E2E serving requires structured GPU identity before policy selection: pass expected_gfx and ' +
    'expected_target (run interface/run_e2e.py for auto-detection).');
}
if (MODEL_PATH && RDNA4_ISA && !R9700_E2E) {
  throw new Error(
    'gfx1201 architecture detected but the product is not confirmed as r9700; refusing R9700-only ' +
    'serving policy for an unknown gfx1201 product.');
}

const LAUNCH_SCRIPT = A.launch_script || '';
const BACKEND = String(A.backend != null ? A.backend : (R9700_E2E ? 'vllm' : 'sglang')).trim()
  || (R9700_E2E ? 'vllm' : 'sglang');  // serving adapter
if (R9700_E2E && BACKEND !== 'vllm') {
  throw new Error(`R9700 E2E is vLLM-only (got backend=${BACKEND}): there is no validated R9700 image for it; pass args.backend=vllm`);
}
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
// Expert-parallel degree AS THE BASELINE WAS LAUNCHED, resolved upstream (run_e2e._resolve_ep) and
// passed down rather than re-derived from flags here. It is a KB addressing dimension beside TP:
// the same model on the same stack with the experts sharded differently is a different deployment.
// Unlike TP it is also a knob the Config Tuner may move mid-run (--ep-size rides EXTRA_SERVER_ARGS),
// so this constant states the baseline's value and the accepted one lives in accepted_config.
// Standalone runs resolve through the same Python parser as interface dispatch.
function resolveServingEp() {
  if (A.ep != null) {
    const ep = Number(A.ep);
    if (!Number.isInteger(ep) || ep < 1) throw new Error('args.ep must be a positive integer');
    return ep;
  }
  if (!A.initial_extra_server_args && !A.initial_extra_env && !A.launch_recipe) return 1;
  const { execFileSync } = require('child_process');
  const value = execFileSync('python3', ['-c', `
import json, os, sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
from interface.run_e2e import _resolve_ep
a = json.loads(sys.argv[2])
print(_resolve_ep(a, None, int(sys.argv[3])))
`, WORKFLOW_DIR, JSON.stringify({ accepted_flags: A.initial_extra_server_args || '',
    accepted_env: A.initial_extra_env || '', launch_recipe: A.launch_recipe || '' }),
  String(SERVING_TP)], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 30000 });
  const ep = Number(value.trim());
  if (!Number.isInteger(ep) || ep < 1) throw new Error('could not resolve baseline expert parallelism');
  return ep;
}
const SERVING_EP = resolveServingEp();
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
// Env override: GEAK_FINAL_RESERVE_S. The cap is GEAK_FINAL_RESERVE_CAP_FRAC (clamped to 5-50%): a trace
// replay's paired validation is two ~70min legs plus their boots, which 20% of most budgets cannot hold.
const FINAL_RESERVE_CAP_FRAC = Math.min(0.5, Math.max(0.05,
  parseFloat(A.final_reserve_cap_frac != null ? A.final_reserve_cap_frac : 0.2) || 0.2));
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
const KERNEL_BUDGET = EXPLICIT_KERNEL_BUDGET !== undefined ? EXPLICIT_KERNEL_BUDGET : (FAST_MODE ? 3 : 6); // budget passed DOWN per kernel (fewer rounds in fast mode)
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
const TUNING_SKILLSET_ENABLED = !RDNA4_ISOLATE && String(A.tuning_skillset != null ? A.tuning_skillset : 'true') === 'true';
// Lives under expert_skills/ so the tuning skills sit in the same hierarchy, and are selected through
// the same index.yaml, as every other expert skill. The tree itself stays VENDORED and pinned whole —
// the index entries that describe it are outside it, which is what lets both things be true at once.
const TUNING_SKILLSET_DIR = String(A.tuning_skillset_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge/expert_skills/tuning')).replace(/\/+$/, '');
// tuning-kb/ is the skillset's per-model ANSWER KEY (verified wins + deployable artifacts). Useful in
// production, contaminating in a blind evaluation — the skillset says so itself. Default ON; pass
// tuning_kb="false" for eval runs and the role is told not to read it.
const TUNING_KB_ENABLED = !RDNA4_ISOLATE && String(A.tuning_kb != null ? A.tuning_kb : 'true') === 'true';
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
// Simulated speculative acceptance (run_e2e forwards the switch names it finds, e.g. SGLANG_SIMULATE_ACC_*)
// keeps throughput comparable but serves neither deterministic nor model-faithful output, so a parity or
// accuracy probe under it measures the simulator. The roles that run such probes are told to serve them
// with these names unset; nothing else changes, and a run without the switches never sees the block.
const CORRECTNESS_UNSET_ENV = String(A.correctness_unset_env || '').split(',')
  .map((s) => s.trim()).filter((s) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(s));
const CORRECTNESS_PROBE_ROLES = new Set(['e2e_integrator', 'director', 'config_tuner', 'tuning_specialist']);
// The AMD authoring knowledge base (REFERENCE ONLY — facts/how-to, never decisions; agents always
// measure). Default: sibling perf_knowledge/. Workflows enumerate candidates from
// index/capability_index.yaml; status/perf in cards are dated evidence, not routing inputs.
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge')).replace(/\/+$/, '');
if (RDNA4_ISOLATE) KERNEL_KNOWLEDGE_DIR = '';
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
  warm_start: RDNA4_ISOLATE ? 'off' : String(A.warm_start != null ? A.warm_start : 'on'),
  ...(A.warm_start_match != null ? { warm_start_match: String(A.warm_start_match) } : {}),
  ...(A.warm_start_min_speedup != null ? { warm_start_min_speedup: A.warm_start_min_speedup } : {}),
  // Which plane the lanes read and write. Forwarded like the rest so one run uses one plane; omitted
  // when unset, which leaves each lane on its own default — `store`, the canonical-id plane, since
  // the kernel read converged on the same addressing the service uses (kernel_lane.js:KB_MODE).
  ...(A.kb_mode != null ? { kb_mode: String(A.kb_mode) } : {}),
  ...(A.kb_store_dir != null ? { kb_store_dir: String(A.kb_store_dir) } : {}),
  ...(A.kb_framework_version != null ? { kb_framework_version: String(A.kb_framework_version) } : {}),
};
// Same off-spelling set the kernel layer accepts; an off run must stay off everywhere.
const WARM_START_OFF = ['off', 'false', 'none'].includes(KB_ARGS.warm_start.trim().toLowerCase());

