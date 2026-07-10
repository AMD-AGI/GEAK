export const meta = {
  name: 'e2e-workflow',
  description: 'End-to-end LLM inference-throughput optimizer for AMD Instinct MI-series GPUs (CDNA gfx942/gfx950, the target card is auto-detected on-box). The serving stack is pluggable via scripts/adapters/<backend>.sh (sglang + vllm shipped; pass args.backend). A system layer (e2e Director / System Architect / Profiler / Config Tuner / Kernel Extractor / e2e Integrator) wraps the UNCHANGED single-kernel kernel_workflow: it preflights the env, profiles a running server, triages hot kernels by Amdahl, tunes config/backends, extracts hot editable kernels into standalone unittests, recursively optimizes them with kernel_workflow.js, overlays them back, and re-validates serving throughput. Also still optimizes a single kernel (pass-through).',
  whenToUse: 'Optimize the serving throughput of an LLM on AMD Instinct MI GPUs. Pass args.model_path (required) + optional args.backend (sglang|vllm, default sglang) + args.launch_script (optional). For a single kernel, pass args.kernel_path instead and it delegates straight to the kernel layer.',
  phases: [
    { title: 'Setup', detail: 'e2e Director builds the isolated eval dir + records baseline throughput' },
    { title: 'Profile', detail: 'Profiler captures a warm trace -> standardized Top-N' },
    { title: 'Strategize', detail: 'System Architect routes kernels by Amdahl (config vs kernel vs host)' },
    { title: 'ConfigSweep', detail: 'Config Tuner sweeps flags/env/backends FIRST (default ON)' },
    { title: 'HeadKernel', detail: 'highest-%GPU ops (GEMM/attn): extract_op -> backend bake-off (incl. FlyDSL) + aiter-DB/author tune -> e2e gate' },
    { title: 'Milestone', detail: 'loop over editable kernels ABOVE milestone_min_pct% GPU (default 5): plan -> extract -> recursive kernel optimize -> overlay -> e2e gate -> reprofile' },
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
const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/kernel_workflow.js`;

// EXP_ROOT = where timestamped run dirs go. Default: sibling "exp/" next to this workflow dir.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

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
// Reserve a tail for the post-deadline finish: the in-flight wave/head completes, then Finalize + Report +
// the final Validate bench + the workflow_return write must all land before the hard kill. Carve 8% (min
// 20min) off the top here, ONCE, so every mode below shares one definition of "effective budget".
const TIME_BUDGET_EFFECTIVE_MS = TIME_BUDGET_MS != null
  ? Math.max(60000, TIME_BUDGET_MS - Math.max(1200000, Math.floor(TIME_BUDGET_MS * 0.08)))
  : null;
// Cap on the dispatch-deadline TAIL (budget − deadline). A flat 60% deadline reserves 40%, which is wasteful
// on large budgets (24h → ~9h reserved though a few hours suffice). Deadlines below are max(60%, budget − this
// cap): the 60% floor keeps small budgets unchanged; the cap bounds the reserve on large ones. Default 3h.
const TIME_TAIL_CAP_MS = parseInt(A.time_tail_cap_s != null ? A.time_tail_cap_s : 10800, 10) * 1000; // 3h
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
function rejectClass(reason) {
  const r = reason || '';
  if (CORRECTNESS_REJECT_RX.test(r)) return 'correctness';
  if (FIXABLE_REJECT_RX.test(r)) return 'integration';
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
if (TIME_BUDGET_EFFECTIVE_MS != null) DEEP_HEAD_BUDGET_MS = TIME_BUDGET_EFFECTIVE_MS;
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
// Acceptance noise band (%). Tight measurement (interleaved A/B, E2E_REPEATS repeats, non-overlap +
// engagement proof — see e2e_integrator) makes a 0.5% default trustworthy. Prompt-tunable.
const NOISE_BAND_DEFAULT = parseFloat(A.noise_band_pct != null ? A.noise_band_pct : 0.5);
// Repeats per timed e2e measurement (the integrator/validator pass this to bench_e2e.sh; the shared
// bench script is NOT edited — interleaving is driven from the eval dir). Prompt-tunable.
// Default 2: with <0.5% spreads, 2 reps + the non-overlap (cand_min>ref_max) check is sufficient to
// judge a win; 7 was overkill and ~3x slower. Bump via args.e2e_repeats for a noisy box.
const E2E_REPEATS = parseInt(A.e2e_repeats != null ? A.e2e_repeats : 2, 10);
// Every integrate A/B MUST measure BOTH legs (reference + candidate). When the
// integrator returns gate:'incomplete'/ab_complete:false (it only ran ref, hung,
// or degraded), the orchestrator RE-INVOKES it to finish the missing leg up to
// this many times before falling back to carrying it as a pending integration.
// This is what guarantees "every A/B runs ref AND cand to completion regardless
// of pass/fail" — general, not per-kernel. Bump via args.ab_finish_retries.
const AB_FINISH_RETRIES = parseInt(A.ab_finish_retries != null ? A.ab_finish_retries : 3, 10);
// A resolvable FROZEN baseline (baseline_src/ frozen OR importable meta.baseline_callable) is the
// speedup DENOMINATOR and is MANDATORY. If an extraction smoke-passes but froze no baseline, the
// unittest would silently time the candidate against its own naive same-language scaffold (the
// "optimized-HIP vs naive-HIP = fake 15.7× isolated, ~0% e2e" bug). When that happens we RE-EXTRACT
// up to this many times; if still missing, the extraction is treated as a FAILURE (flag dominant /
// skip others) — never a fake speedup. Bump via args.baseline_extract_retries.
const BASELINE_EXTRACT_RETRIES = parseInt(A.baseline_extract_retries != null ? A.baseline_extract_retries : 3, 10);
const TASK = A.task || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const MODEL_NAME_HINT = (MODEL_PATH || KERNEL_PATH).replace(/\/+$/, '').split('/').pop();

// ---------------------------------------------------------------------------
// Phase-scoped driving (robustness): a long single background run can be orphaned if the host session
// context compacts mid-run. To avoid that, the orchestration can be driven phase-by-phase: invoke with
// args.phases = subset of {setup,config,head,kernel,final} (default 'all' = run everything in one go).
// Cross-phase state flows through the RETURN value (the script has no fs); pass the prior return back as
// args.state for the next phase. Each phase only RUNS if requested; otherwise it loads carried state.
// ---------------------------------------------------------------------------
const PHASES = String(A.phases || 'all').split(',').map(s => s.trim()).filter(Boolean);
const RUN_ALL = PHASES.includes('all');
// Fast mode SKIPS ConfigSweep ('config') and the editable-kernel Milestone ('kernel') so all optimization
// comes from HeadKernel within the wall-clock budget. Default mode: FAST_SKIP is null → want() is the
// original `RUN_ALL || PHASES.includes(p)`, unchanged.
const FAST_SKIP = FAST_MODE ? new Set(['config', 'kernel']) : null;
// Deep mode concentrates its (20h) HeadKernel budget on cross-backend co-opt: skip the editable-kernel
// Milestone ('kernel') but KEEP ConfigSweep ('config' — cheap and it stabilizes the baseline). Null in
// every other mode, so normal + fast are unchanged.
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

const SWEEP_SCHEMA = obj({
  trials: arrObj, accepted_flags: { type: 'string' }, accepted_env: { type: 'string' },
  best_throughput_tok_s: { type: 'number' }, throughput_speedup_vs_baseline: { type: 'number' },
  summary: { type: 'string' },
}, ['accepted_flags', 'best_throughput_tok_s']);

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
  baseline_frozen: { type: 'boolean' }, // true only when baseline_src/ was frozen OR baseline_callable resolves
  smoke: { type: 'string' }, notes: { type: 'string' },
}, ['op_kind', 'task_dir', 'smoke']);

const OPBENCH_SCHEMA = obj({
  short_name: { type: 'string' }, op_kind: { type: 'string' }, provenance_ok: { type: 'boolean' },
  winner_backend: { type: 'string' }, winner_kind: { type: 'string' },
  isolated_speedup: { type: 'number' }, winner_editable: { type: 'boolean' },
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
  baseline_callable: { type: 'string' }, // module:attr of the FROZEN real online kernel (the speedup denominator)
  baseline_frozen: { type: 'boolean' }, // true only when baseline_src/ was frozen OR baseline_callable resolves
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

function roleAgent(role, phase, intro, inputs) {
  // BACKEND is injected for every role: any role that calls bench_e2e.sh must forward it
  // (BACKEND=<backend>) so the right serving adapter (scripts/adapters/<backend>.sh) is used.
  const inall = { BACKEND, SERVING_TP, SERVING_GPU, ...inputs };
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
  return base + expertSkillsBlock(role);
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
function agentBounded(prompt, opts) {
  if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) return agent(prompt, opts);
  let to;
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [hung-agent guard] ${(opts && opts.label) || 'agent'} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — treating as a failed attempt.`);
      resolve(null);
    }, AGENT_TIMEOUT_MS);
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

// A FROZEN baseline is resolvable when the extractor either froze baseline_src/ (baseline_frozen)
// OR set an importable meta.baseline_callable. That is the language-independent speedup denominator.
const hasFrozenBaseline = (ext) =>
  !!(ext && (ext.baseline_frozen === true ||
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
  let ext = await safeAgent(roleAgent(role, phase, intro, inputs), opts);
  let tries = 0;
  while (smokeOk(ext) && !hasFrozenBaseline(ext) && tries < BASELINE_EXTRACT_RETRIES) {
    tries++;
    log(`  ${(opts && opts.label) || role}: extraction froze NO baseline ` +
      `(baseline_src/ or meta.baseline_callable) — the speedup denominator would fall back to the ` +
      `candidate's own scaffold (fake-win). RE-EXTRACTING (retry ${tries}/${BASELINE_EXTRACT_RETRIES}).`);
    ext = await safeAgent(
      roleAgent(role, phase,
        intro + ' PRIOR ATTEMPT DID NOT FREEZE A BASELINE. You MUST freeze the real online kernel into ' +
        'an immutable baseline_src/ and set meta.baseline_callable (the speedup denominator), bind the ' +
        "unittest's baseline leg to it, then return baseline_frozen:true. An extraction with no frozen " +
        'baseline is INVALID and will be discarded.',
        inputs),
      opts);
  }
  if (smokeOk(ext) && !hasFrozenBaseline(ext)) {
    log(`  ${(opts && opts.label) || role}: STILL no frozen baseline after ${BASELINE_EXTRACT_RETRIES} ` +
      `re-extractions — ABORTING this extraction (refusing a fake speedup vs the candidate's own scaffold).`);
    return { ...ext, smoke: 'fail', unittest_smoke: 'fail',
      notes: `no frozen baseline after ${BASELINE_EXTRACT_RETRIES} re-extractions ` +
        `(baseline_src/ / meta.baseline_callable required as the speedup denominator) — ${ext.notes || ''}` };
  }
  return ext;
}

// abDone == the integrator measured BOTH legs (ref + cand) and emitted a real
// verdict. gate:'incomplete' or ab_complete:false means a leg is still missing.
const abDone = (integ) => !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);

// Run ONE integrate A/B and GUARANTEE both legs complete. The first call does a
// normal apply+gate; if the integrator returns incomplete (ran only ref, hung,
// or degraded mid-A/B) we RE-INVOKE it in resume mode to run the MISSING leg,
// up to AB_FINISH_RETRIES times. General contract used by every integrate site
// (head, milestone, finalize-gate, disk-recovered): an A/B never ends after the
// reference leg alone — it is driven to a complete ref+cand measurement.
async function runIntegrateBothLegs(intro, inputs, label, phaseName) {
  let integ = await safeAgent(
    roleAgent('e2e_integrator', 'integrate', intro, inputs),
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
        'hardware/harness fault persists after this retry. If short on time, shrink E2E_REPEATS toward 1 ' +
        'so BOTH legs still run.',
        { ...inputs, RESUME_AB: true }),
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
      return { banked: true, integ: integ2, isolated: fix.final_geomean };
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
  const p = workflow(ref, wfArgs);
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
const TIME_HEAD_DEADLINE_MS = TIME_BUDGET_EFFECTIVE_MS != null
  ? Math.max(Math.floor(TIME_BUDGET_EFFECTIVE_MS * 0.6), TIME_BUDGET_EFFECTIVE_MS - TIME_TAIL_CAP_MS) : null;
if (TIME_HEAD_DEADLINE_MS != null && typeof setTimeout === 'function' && TIME_HEAD_DEADLINE_MS > 0) {
  setTimeout(() => {
    TIME_DEADLINE_HIT = true;
    log(`[time-budget] dispatch deadline (${Math.round(TIME_HEAD_DEADLINE_MS / 60000)}min of ${Math.round(TIME_BUDGET_MS / 60000)}min budget) reached — no NEW head/milestone work will start; finishing in-flight then proceeding to Finalize/Report/Validate before the hard kill.`);
  }, TIME_HEAD_DEADLINE_MS);
}
function deepBoundedWorkflow(ref, wfArgs, label) {
  const p = workflow(ref, wfArgs);
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
    const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, {
      kernel_path: KERNEL_PATH, workflow_dir: KERNEL_WF_DIR,
      use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
      budget: KERNEL_BUDGET, gpu_ids: GPU_IDS, task: TASK, exp_root: EXP_ROOT,
      apply_to_original: APPLY_TO_ORIGINAL,
    });
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

// ===========================================================================
// PHASE: Setup + Baseline profile + Strategize  (gated; else load carried state)
// ===========================================================================
let EVAL_DIR, MODEL_NAME, BASELINE_TPUT, NOISE_BAND, curFlags, curEnv, profile, strategy, kernelQueue, headQueue;
if (want('setup')) {
  phase('Setup');
  const setup = await safeAgent(
    roleAgent('director', 'setup', 'Build the isolated e2e eval dir and record the baseline throughput.', {
      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT, EVAL_DIR_OVERRIDE, MODEL_NAME_HINT, TASK,
      GPU_IDS, WORKLOAD, INIT_FLAGS, INIT_ENV, SKILL_DIR: WORKFLOW_DIR,
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
  log(`Setup done. EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT} tok/s (noise band ${NOISE_BAND}%)`);

  phase('Profile');
  profile = await safeAgent(
    roleAgent('profiler', 'baseline', 'Capture a warm trace and emit the standardized Top-N.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 0,
      OVERLAY_PYTHONPATH: '', EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
      ...TRACELENS_INPUTS,
    }),
    { phase: 'Profile', label: 'profiler:baseline', schema: PROFILE_SCHEMA });
  log(`Baseline profiled. ${profile ? (profile.top_kernels || []).length : 0} top kernels.`);

  phase('Strategize');
  strategy = await safeAgent(
    roleAgent('system_architect', 'strategize', 'Route the Top-N into config/kernel/host tracks by Amdahl.', {
      EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: BASELINE_TPUT,
      WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED, SKILL_DIR: WORKFLOW_DIR,
      ...TRACELENS_INPUTS,
    }),
    { phase: 'Strategize', label: 'architect:strategize', schema: STRATEGY_SCHEMA });
  kernelQueue = (strategy && strategy.kernel_candidates) ? strategy.kernel_candidates.slice() : [];
  headQueue = (strategy && strategy.head_candidates) ? strategy.head_candidates.slice() : [];
  // OP-IDENTITY GUARD — a fused-MoE / grouped-expert GEMM must be optimized AS the fused op at its live
  // dispatcher seam, never decomposed into standalone dense GEMMs (a dense candidate has no live call site
  // → no_rebind_seam). So force op_kind='moe' (the grouped-GEMM branch; gemmSynthFor keys on this to keep
  // dense synth OFF) and preserve the live seam as target_callable, so ANY lever (backend-swap / tune /
  // author-fused) binds. The head is never SKIPPED — editability is irrelevant, since a non-editable fused
  // kernel is still backend-swapped at its (editable) dispatcher. GENERIC: detects via the Architect's
  // is_fused_kernel OR the profile class/name; never keys on a backend name.
  const _isFusedOp = (c) => (c && c.is_fused_kernel === true) ||
    /(?:^|[^a-z])moe(?:[^a-z]|$)|group(?:ed)?[_ ]?gemm|ck_moe|expert|fused[_ ]?moe|fmoe|asm_moe|fused_custom/i
      .test(`${(c && c.op_kind) || ''} ${(c && c.short_name) || ''} ${(c && c.name) || ''} ${(c && c.classification) || ''} ${(c && c.class) || ''} ${(c && c.backend) || ''}`);
  let _fusedTagged = 0;
  for (const c of headQueue) {
    if (!_isFusedOp(c)) continue;
    c.op_kind = 'moe';                                                                // grouped-GEMM branch (gemmSynthFor → no dense synth)
    if (!c.target_callable && c.live_call_seam) c.target_callable = c.live_call_seam;  // bind at the live seam
    _fusedTagged++;
  }
  if (_fusedTagged) log(`[op-identity] ${_fusedTagged} fused/grouped head(s): op_kind=moe (never dense-GEMM), bound at live seam — optimized as the fused op, never skipped.`);
  log(`Strategy: ${headQueue.length} head candidates, ${kernelQueue.length} kernel candidates, ${(strategy && strategy.config_directions || []).length} config directions.`);
} else {
  // Load carried state from a prior phase invocation (args.state).
  EVAL_DIR = ST.eval_dir || EVAL_DIR_OVERRIDE;
  if (!EVAL_DIR) throw new Error('Non-setup phase requires args.state.eval_dir (or args.eval_dir)');
  MODEL_NAME = ST.model_name || MODEL_NAME_HINT;
  BASELINE_TPUT = ST.baseline_throughput_tok_s || 0;
  NOISE_BAND = ST.noise_band_pct || NOISE_BAND_DEFAULT;
  curFlags = ST.flags || '';
  curEnv = ST.env || '';
  profile = { profile_topN_json: ST.profile_topn_json || '' };
  strategy = { config_directions: ST.config_directions || [] };
  kernelQueue = ST.kernelQueue || [];
  headQueue = ST.headQueue || [];
  log(`Loaded carried state: EVAL_DIR=${EVAL_DIR}, baseline ${BASELINE_TPUT}, flags='${curFlags}', env='${curEnv}', ${headQueue.length} head + ${kernelQueue.length} kernel candidates.`);
}

// ===========================================================================
// PHASE: Config sweep (Config Tuner) — FIRST, reshapes the profile
// ===========================================================================
let curTput = ST.throughput || BASELINE_TPUT;
if (want('config') && CONFIG_TUNE_ENABLED && strategy && (strategy.config_directions || []).length) {
  phase('ConfigSweep');
  const sweep = await safeAgent(
    roleAgent('config_tuner', 'sweep', 'Sweep the ranked config axes one at a time; keep wins.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, BASELINE_THROUGHPUT: BASELINE_TPUT,
      NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS, CONFIG_DIRECTIONS: strategy.config_directions,
      CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
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
        OVERLAY_PYTHONPATH: '', EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
      }),
      { phase: 'Profile', label: 'profiler:post-config', schema: PROFILE_SCHEMA });
    // Re-strategize the kernel queue against the new profile.
    const restrat = await safeAgent(
      roleAgent('system_architect', 'strategize', 'Re-route after config changed the landscape.', {
        EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: curTput,
        WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED: false, SKILL_DIR: WORKFLOW_DIR,
      }),
      { phase: 'Strategize', label: 'architect:re-strategize', schema: STRATEGY_SCHEMA });
    if (restrat && restrat.kernel_candidates) kernelQueue = restrat.kernel_candidates.slice();
    if (restrat && restrat.head_candidates) headQueue = restrat.head_candidates.slice();
  } else {
    log(`Config sweep found no win above the noise band.`);
  }
}

// ---------------------------------------------------------------------------
// Shared state carried across the head + kernel tracks (and across phase invocations via args.state).
// MUST be declared BEFORE the HeadKernel block that uses them (else temporal-dead-zone ReferenceError).
// ---------------------------------------------------------------------------
let curOverlay = ST.overlay || '';        // the accepted overlay carried forward
let dispatched = 0;                        // counts ONLY kernel-optimization tasks (the budget)
let milestone = 0;
let noImprove = 0;
const acceptedKernels = (ST.accepted_kernels || []).slice();
const acceptedHeads = (ST.accepted_heads || []).slice();
// Verified-isolated wins whose e2e A/B did NOT complete (integrate agent timed
// out / hung / degraded to null mid-gate). These are NOT rejections — keep them
// so Finalize can finish the best one's A/B (Fix C) and so a real isolated win
// is surfaced (return.pending_integrations) instead of being silently dropped.
const pendingIntegrations = (ST.pending_integrations || []).slice();
const flaggedHeads = (ST.flagged_heads || []).slice();   // dominant heads that could NOT be optimized (loudly surfaced, never silently skipped)
let headDispatched = 0;
const history = ST.history || { insights: [], ledger: [], milestones: [], bottleneck_now: '', suggest_next: '' };

// A fused op (op_kind='moe', set by the op-identity guard OR the Architect) is extracted AS the fused op,
// never decomposed into a standalone dense GEMM — so dense-GEMM synth is off for it.
function gemmSynthFor(h) { return (h && h.op_kind === 'moe') ? 'false' : GEMM_SYNTH; }

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
          CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
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
        const integ = await safeAgent(
          roleAgent('e2e_integrator', 'integrate', 'Apply a deep head candidate; gate on e2e throughput; report engagement/cudagraph/mem/decode for feedback.', {
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
            CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR, DEEP_FEEDBACK: true,
            ...ACCURACY_INPUTS,
            ...(opts.final && ACCURACY_GATE !== 'none' ? { ACCURACY_LIMIT: DEEP_FINAL_ACCURACY_LIMIT } : {}),   // de-noise the finalize accuracy decision
          }),
          { phase: 'HeadKernel', label: `integrate ${c.uid} g${e2eGateCount}`, schema: INTEGRATE_SCHEMA });
        if (integ && integ.output_parity === 'fail') {
          log(`  [deep] ${c.uid}: REJECTED — output_parity=fail vs true baseline.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'dead_end', lesson: 'parity fail vs true baseline' });
        } else if (integAccepted(integ, c.head.pct_gpu_time, c.best) && integ.e2e_throughput_tok_s > curTput) {
          curOverlay = integ.accepted_overlay || curOverlay; curTput = integ.e2e_throughput_tok_s; bankedHeads.add(c.head.short_name);
          acceptedHeads.push({ short_name: c.head.short_name, op_kind: c.ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', e2e_delta_pct: integ.e2e_delta_pct, isolated: c.best });
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
            acceptedHeads.push({ short_name: c.head.short_name, op_kind: c.ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', e2e_delta_pct: dcorr.integ.e2e_delta_pct, isolated: dcorr.isolated, corrective: true });
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
    while (!DEEP_DEADLINE_HIT) {
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
            CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
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
      if (al.authored !== false && al.final_geomean > 1.0 && al.final_patch) {
        st.cands.push({ kind: 'authored', source: lang, winner_kind: 'authored', language: lang,
          final_patch: al.final_patch, kernel_eval_dir: al.eval_dir, isolated: al.final_geomean });
        log(`  ${j.short_name}: authored ${lang} ${al.final_geomean.toFixed(2)}x (vs the frozen online kernel, not the seed).`);
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
      const integ = await runIntegrateBothLegs(
        'Apply the head-op winner; gate on e2e throughput.', {
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
        },
        `integrate ${h.short_name}`, 'HeadKernel');
      if (integAccepted(integ, h.pct_gpu_time, cand.isolated) && integ.e2e_throughput_tok_s > curTput) {
        curOverlay = integ.accepted_overlay || curOverlay;
        if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
        if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
        curTput = integ.e2e_throughput_tok_s;
        acceptedHeads.push({ short_name: h.short_name, op_kind: st.ext.op_kind, backend: cand.source, kind: cand.winner_kind, e2e_delta_pct: integ.e2e_delta_pct, isolated: cand.isolated });
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
          acceptedHeads.push({ short_name: h.short_name, op_kind: st.ext.op_kind, backend: cand.source, kind: 'authored', e2e_delta_pct: corr.integ.e2e_delta_pct, isolated: corr.isolated, corrective: true });
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
        CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
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
      if (al && al.authored !== false && al.final_geomean > 1.0 && al.final_patch) {
        headCands.push({ kind: 'authored', source: lang, winner_kind: 'authored', language: lang,
          final_patch: al.final_patch, kernel_eval_dir: al.eval_dir, isolated: al.final_geomean });
        log(`  ${h.short_name}: authored ${lang} ${al.final_geomean.toFixed(2)}x (vs the frozen online kernel, not the seed).`);
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
    const cand = headCands[0];
    log(`  ${h.short_name}: best candidate=${cand.source} (${(cand.isolated || 0).toFixed(2)}x, ${cand.kind}). Integrating to e2e.`);

    // (h3) e2e gate on the chosen candidate. direct_light env/flag → config; authored/patch → overlay.
    // Build inputs ONCE so Fix C can re-issue the SAME A/B for a pending win at Finalize.
    const headIntegrateInputs = {
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
    };
    const integ = await runIntegrateBothLegs(
      'Apply the head-op winner; gate on e2e throughput.', headIntegrateInputs,
      `integrate ${h.short_name}`, 'HeadKernel');

    // Three-state gate: an integrate that did NOT complete its A/B (null /
    // gate:'incomplete' / ab_complete!==true) is NOT a rejection — keep it as a
    // pending verified-isolated win so Finalize (Fix C) can finish/surface it.
    // Backward-compatible: treat the return as a COMPLETED A/B unless it is null
    // (timeout/hang/degrade — the actual incident cause) or EXPLICITLY flags
    // itself incomplete (gate:'incomplete' or ab_complete===false). A legacy
    // integrator that omits ab_complete still accepts/rejects exactly as before.
    const abDone = !!(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false);
    if (abDone && integAccepted(integ, h.pct_gpu_time, cand.isolated) && integ.e2e_throughput_tok_s > curTput) {
      // a head winner may be carried as overlay (authored/patch) AND/OR config (env/flag) — capture both.
      curOverlay = integ.accepted_overlay || curOverlay;
      if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
      if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
      curTput = integ.e2e_throughput_tok_s;
      acceptedHeads.push({ short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: cand.winner_kind, e2e_delta_pct: integ.e2e_delta_pct, isolated: cand.isolated });
      log(`  ${h.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else if (!abDone) {
      // A FIXABLE crash-during-warmup (the cand server died -> ZERO A/B samples -> ab_complete=false) is the
      // MOST COMMON corrective case (cuda_graph_capture_unsafe / host-sync / NO_BINARY_FOR_GPU poisons the
      // HIP context at capture). The integrator still names the cause, so treat a fixable+named crash like a
      // fixable reject: try the corrective re-author FIRST; only keep PENDING if it is not fixable (a real
      // transient timeout/hang) or the fix didn't land.
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
        acceptedHeads.push({ short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: 'authored', e2e_delta_pct: corr.integ.e2e_delta_pct, isolated: corr.isolated, corrective: true });
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
      // (h3-fix) Corrective re-author: a fix-and-retryable reject of an authored, iso-verified winner gets up
      // to HEAD_CORRECTIVE_MAX cheap fix-and-retries (optimize the EXISTING kernel, no re-discovery; re-gate).
      // Not a new head -> not charged to HEAD_BUDGET. See tryCorrectiveReauthor.
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
        acceptedHeads.push({ short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: 'authored', e2e_delta_pct: corr.integ.e2e_delta_pct, isolated: corr.isolated, corrective: true });
        log(`  ${h.short_name}: ACCEPTED after corrective re-author. e2e now ${curTput} tok/s (+${corr.integ.e2e_delta_pct}%).`);
        history.ledger.push({ direction: h.short_name, isolated_speedup: corr.isolated, e2e_delta_pct: corr.integ.e2e_delta_pct, verdict: 'confirmed_corrective', lesson: `fixed: ${reason}` });
      } else {
        log(`  ${h.short_name}: REJECTED at e2e gate (${reason}).`);
        history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct || 0, verdict: 'dead_end', lesson: reason || 'no e2e gain' });
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
        CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        ...(profile && profile.profile_workload_json ? { PROFILE_WORKLOAD_JSON: profile.profile_workload_json } : {}),
      },
      { phase: 'Milestone', label: `extract ${c.short_name}`, schema: EXTRACT_SCHEMA });
    if (!ext || ext.editable === false || ext.unittest_smoke !== 'pass' || !ext.task_dir) {
      return { c, skip: true, reason: `extraction failed/non-editable (${ext ? ext.notes || ext.unittest_smoke : 'none'})` };
    }
    // RECURSIVE kernel layer on the IMMUTABLE task dir (one allowed nesting level via workflow()).
    let kl;
    try {
      const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, {
        kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
        use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
        budget: KERNEL_BUDGET, gpu_ids: c.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
        task: 'Compare candidate backends ' + JSON.stringify(c.candidate_backends || []) +
          ' for this kernel; pick the fastest that passes the immutable unittest. ' + GRAPH_REQ + (TASK || ''),
        apply_to_original: 'false',
      });
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
      acceptedKernels.push({ short_name: c.short_name, backend: kl.note || '', e2e_delta_pct: integ.e2e_delta_pct, isolated: kl.final_geomean });
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
        acceptedKernels.push({ short_name: c.short_name, backend: kl.note || '', e2e_delta_pct: corr.integ.e2e_delta_pct, isolated: corr.isolated, corrective: true });
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
// AB_FINISH_RETRIES + each agent's agentBounded guard + the outer runner budget.
if (want('final')) {
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
    `MISSING, OR gate=="incomplete", OR (ab_complete is false/absent AND there is no ` +
    `cand/bench_runs.jsonl in the dir). Treat it as COMPLETE (skip it) when ab_complete:true ` +
    `OR gate is one of {accepted,stack,rejected} AND a cand/bench_runs.jsonl exists. ` +
    `For each INCOMPLETE dir, emit one object with: short_name (the dir name minus the ` +
    `"cand_" prefix), overlay_dir (absolute path), and from integrate_result.json (use "" / ` +
    `null when absent): winner_kind, apply_env, apply_flags, op_kind, ` +
    `target_callable (or target_file), isolated (= isolated_speedup), pct_gpu_time, plus ` +
    `ref_present (true if ref/bench_runs.jsonl exists) and cand_present (true if ` +
    `cand/bench_runs.jsonl exists). Return ONLY compact JSON {"incomplete":[...]} (empty array if none).`,
    { phase: 'Finalize', label: 'scan-incomplete-ab', schema: RECON_SCHEMA });
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

  // (2) Drive EVERY pending integration to a complete ref+cand measurement.
  pendingIntegrations.sort((a, b) => (b.isolated || 0) - (a.isolated || 0));
  const stillIncomplete = [];
  while (pendingIntegrations.length) {
    const p = pendingIntegrations.shift();   // pop one per iteration => guaranteed termination
    log(`Finalize-gate: finishing pending A/B (${p.short_name}, ${(p.isolated || 0).toFixed(2)}x isolated); ` +
      `${pendingIntegrations.length} more queued.`);
    const integ = await runIntegrateBothLegs(
      'Finish the e2e A/B for this verified-isolated win. Run BOTH legs (reference + candidate) to ' +
      'completion, then return accepted/stack/rejected with ab_complete:true.',
      // Pin the CURRENT carried overlay/flags/env/throughput so the A/B is
      // measured against the latest accepted baseline.
      { ...p.inputs, CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, CURRENT_THROUGHPUT: curTput },
      `finish-integrate ${p.short_name}`, 'Finalize');
    if (abDone(integ) && integAccepted(integ, p.pct_gpu_time, p.isolated) && integ.e2e_throughput_tok_s > curTput) {
      curOverlay = integ.accepted_overlay || curOverlay;
      if (p.track === 'head') {
        if (p.winner_kind === 'env' && p.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + p.apply_env;
        if (p.winner_kind === 'flag' && p.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + p.apply_flags;
        acceptedHeads.push({ short_name: p.short_name, op_kind: p.op_kind, backend: p.backend, kind: p.winner_kind, e2e_delta_pct: integ.e2e_delta_pct, isolated: p.isolated });
      } else {
        acceptedKernels.push({ short_name: p.short_name, backend: p.backend || '', e2e_delta_pct: integ.e2e_delta_pct, isolated: p.isolated });
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
allAccepted = acceptedHeads.concat(acceptedKernels);   // refresh after Fix C may have banked a pending win
if (want('final')) {
  phase('Finalize');
  finalize = await safeAgent(
    roleAgent('e2e_integrator', 'finalize', 'Assemble the final overlay + patch + launch script bundle.', {
      EVAL_DIR, FINAL_OVERLAY: curOverlay, ACCEPTED_FLAGS: curFlags, ACCEPTED_ENV: curEnv,
      ACCEPTED_KERNELS: allAccepted, BASELINE_THROUGHPUT: BASELINE_TPUT, SKILL_DIR: WORKFLOW_DIR,
    }),
    { phase: 'Finalize', label: 'e2e_integrator:finalize', schema: FINALIZE_SCHEMA });
  finalTput = (finalize && finalize.final_throughput_tok_s) || curTput;

  phase('Report');
  report = await safeAgent(
    roleAgent('system_architect', 'report', 'Write architect_report.md AND the full final_report.md in English (with the Phases tree + artifacts tree modules).', {
      EVAL_DIR, HISTORY: history, BASELINE_THROUGHPUT: BASELINE_TPUT, FINAL_THROUGHPUT: finalTput,
      ACCEPTED_CONFIG: { flags: curFlags, env: curEnv }, ACCEPTED_KERNELS: allAccepted,
      ACCEPTED_HEADS: acceptedHeads, FLAGGED_HEADS: flaggedHeads, MILESTONES: milestone, BUDGET_USED: dispatched, BUDGET, MIN_KERNEL_TASKS,
      PROFILE_TOPN: profile ? profile.profile_topN_json : '', WORKLOAD, MODEL_NAME, SKILL_DIR: WORKFLOW_DIR,
    }),
    { phase: 'Report', label: 'architect:report', schema: REPORT_SCHEMA });

  phase('Validate');
  validation = await safeAgent(
    roleAgent('director', 'validate', 'Independently re-measure throughput + parity; arbitrate; then reconcile the report with the validated numbers.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], BASELINE_THROUGHPUT: BASELINE_TPUT, NOISE_BAND_PCT: NOISE_BAND,
      FINAL_OVERLAY: (finalize && finalize.final_overlay) || curOverlay,
      FINAL_FLAGS: { flags: curFlags, env: curEnv },
      CLAIMED_THROUGHPUT: finalTput, WORKLOAD, APPLY_TO_ORIGINAL, E2E_REPEATS, SKILL_DIR: WORKFLOW_DIR,
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
  // Carry pending (verified-isolated, A/B-incomplete) wins WITH their inputs so a
  // resumed phase run can finish their A/B instead of re-discovering them.
  pending_integrations: pendingIntegrations,
  history,
};

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
  final_launch_script: (validation && validation.final_launch_script) || (finalize && finalize.final_launch_script) || '',
  report_path: report ? report.report_path : `${EVAL_DIR}/architect_report.md`,
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

return wfReturn;
