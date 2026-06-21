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
const FAST_BUDGET_MS = parseInt(A.fast_budget_ms != null ? A.fast_budget_ms : 18000000, 10); // 5h
// Stop STARTING new head ops after this point so the in-flight head + Finalize/Report/Validate still land
// inside FAST_BUDGET_MS. Default 60% of the budget (3h at 5h) leaves ~40% for the last head to finish +
// the deliverable/validation tail.
const FAST_HEAD_DEADLINE_MS = parseInt(A.fast_head_deadline_ms != null ? A.fast_head_deadline_ms
  : Math.floor(FAST_BUDGET_MS * 0.6), 10);
// Per-head nested author/optimize workflow() bound (fast mode only): a single recursive kernel run can't
// eat the whole budget. Default 35min.
const FAST_HEAD_WF_MS = parseInt(A.fast_head_workflow_ms != null ? A.fast_head_workflow_ms : 2100000, 10);
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
const DEEP_HEAD_BUDGET_MS = parseInt(A.deep_head_budget_ms != null ? A.deep_head_budget_ms : 86400000, 10); // 24h for the whole HeadKernel module (deep mode does deep exploration)
const DEEP_WAVE_KERNEL_BUDGET = parseInt(A.deep_wave_kernel_budget != null ? A.deep_wave_kernel_budget : 3, 10); // direction-budget per backend per wave (a bounded burst, then a curator+maybe-e2e step)
const DEEP_HEAD_WF_MS = parseInt(A.deep_head_workflow_ms != null ? A.deep_head_workflow_ms : 4500000, 10); // per-burst nested kernel_workflow time cap (75min) — bounds the per-wave barrier. Harvest+gate run at the TOP of each wave on disk truth, so gate latency is decoupled from this; the cap only bounds how long a burst runs.
const DEEP_E2E_GAIN_TRIGGER = parseFloat(A.deep_e2e_gain_trigger != null ? A.deep_e2e_gain_trigger : 0.08); // isolated-best improvement since last e2e gate that triggers a new (batched) gate
const DEEP_E2E_MAX_INTERVAL_MS = parseInt(A.deep_e2e_max_interval_ms != null ? A.deep_e2e_max_interval_ms : 7200000, 10); // force an e2e gate at least this often when a new candidate exists (default 2h)
const DEEP_PLATEAU_STREAK = parseInt(A.deep_plateau_streak != null ? A.deep_plateau_streak : 2, 10); // consecutive non-improving waves before a backend lane is parked (frees its GPU)
// HARD per-head wave cap — a budget backstop. The harvest's vs_live is an LLM estimate (noisy), so the
// noImprove-based plateau-park can fail to fire and a head would otherwise spin until its ~12h time-slice.
// This bounds each head to a fixed number of co-opt waves regardless, then advances to the next head.
const DEEP_MAX_WAVES_PER_HEAD = parseInt(A.deep_max_waves_per_head != null ? A.deep_max_waves_per_head : 4, 10);
const DEEP_BACKENDS_OVERRIDE = String(A.deep_backends || '').trim(); // optional CSV "triton:optimize,hip:author,..."; default = derive from the bake-off
// ---- DEEP MODE v2 (default ON when deep_mode=true) --------------------------------------------------
// v2 supersedes the v1 per-head-serial loop with a GLOBAL, GPU-elastic, budget-driven orchestrator:
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
// Everything here is gated by DEEP_V2 (⊆ DEEP_MODE). DEEP_V2 off → the v1 deep block runs unchanged;
// DEEP_MODE off → default/fast are byte-identical. No model/kernel/backend specifics are hard-coded.
const DEEP_V2 = DEEP_MODE && String(A.deep_v2 != null ? A.deep_v2 : 'true') === 'true';
const DEEP_E2E_TARGET = parseFloat(A.deep_e2e_target != null ? A.deep_e2e_target : 1.5); // stretch goal: +50% e2e vs the TRUE baseline (keep pushing toward it within budget)
const DEEP_PLATEAU_STREAK_HIGH = parseInt(A.deep_plateau_streak_high != null ? A.deep_plateau_streak_high : 4, 10); // patience for HIGH-ceiling lanes (still far from their potential) before parking
const DEEP_V2_GATE_BURST_MS = parseInt(A.deep_v2_gate_burst_ms != null ? A.deep_v2_gate_burst_ms : 1500000, 10); // shorter burst cap (25min) for co-opt bursts that share the serving slot, so a due gate waits at most one short burst
const DEEP_V2_REPROFILE_GAIN = parseFloat(A.deep_v2_reprofile_gain != null ? A.deep_v2_reprofile_gain : 0.10); // cumulative e2e gain since last profile that triggers a re-profile (bottleneck moved)
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
// config/param search, it passes its accepted best flags/env so the PerfSkills
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
  dtype: { type: 'string' }, synthesized: { type: 'boolean' }, regimes_captured: arrStr,
  candidate_backends: arrStr, reference_io_sha256: { type: 'string' },
  target_callable: { type: 'string' }, // module:attr rebind seam for an authored kernel ('' if none)
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
  output_parity: { type: 'string' }, gate: { type: 'string' },
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
    }),
    { phase: 'Profile', label: 'profiler:baseline', schema: PROFILE_SCHEMA });
  log(`Baseline profiled. ${profile ? (profile.top_kernels || []).length : 0} top kernels.`);

  phase('Strategize');
  strategy = await safeAgent(
    roleAgent('system_architect', 'strategize', 'Route the Top-N into config/kernel/host tracks by Amdahl.', {
      EVAL_DIR, PROFILE_TOPN: profile ? profile.profile_topN_json : '', BASELINE_THROUGHPUT: BASELINE_TPUT,
      WORKLOAD, BUDGET, HEAD_THRESHOLD_PCT, CONFIG_TUNE_ENABLED, SKILL_DIR: WORKFLOW_DIR,
    }),
    { phase: 'Strategize', label: 'architect:strategize', schema: STRATEGY_SCHEMA });
  kernelQueue = (strategy && strategy.kernel_candidates) ? strategy.kernel_candidates.slice() : [];
  headQueue = (strategy && strategy.head_candidates) ? strategy.head_candidates.slice() : [];
  log(`Strategy: ${headQueue.length} head (GEMM/attn) candidates, ${kernelQueue.length} kernel candidates, ${(strategy && strategy.config_directions || []).length} config directions.`);
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
const flaggedHeads = (ST.flagged_heads || []).slice();   // dominant heads that could NOT be optimized (loudly surfaced, never silently skipped)
let headDispatched = 0;
const history = ST.history || { insights: [], ledger: [], milestones: [], bottleneck_now: '', suggest_next: '' };

// ===========================================================================
// PHASE: HeadKernel — the highest-pct_gpu_time ops (GEMM / attention), optimized
// regardless of edit flag, via the bake-off ladder. This is the lever the old
// design missed for GEMM (~78% of GPU time). Runs BEFORE the editable-kernel loop.
// ===========================================================================
if (want('head') && headQueue.length && HEAD_BUDGET > 0) {
  phase('HeadKernel');
  log(`Head-kernel track: ${headQueue.length} candidate op(s), head_budget=${HEAD_BUDGET}, threshold=${HEAD_THRESHOLD_PCT}%.`);
  const heads = headQueue.slice(0, HEAD_BUDGET).map((c, i) => ({
    ...c, idx: i, gpu_id: GPU_LIST[i % GPU_LIST.length],
    short_name: c.short_name || `${c.op_kind || 'op'}${i}`,
  }));
  if (DEEP_V2) {
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
    log(`[deep-v2] partition: serving {${servingCards.join(',')}} TP=${SERVING_TP}; dedicated co-opt {${dedicatedCoopt.join(',') || '(none)'}}; mode=${haveSpare ? 'OVERLAP gate||co-opt' : 'TIME-SLICE (N==TP)'}; budget ${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h; e2e target ×${DEEP_E2E_TARGET} (${Math.round(BASELINE_TPUT * DEEP_E2E_TARGET)} tok/s).`);

    // ---- ceiling prior: GENERAL (by lane ROLE, not model/shape) -------------------------------------
    const ceilingPrior = (lang, mode) => (mode === 'author' ? 2.2 : (lang === 'triton' && mode === 'optimize' ? 1.6 : 1.8));

    // ---- per-head prep: extract + bake-off + roofline + lane roster (cheap agents on GPU_LIST[0]) ----
    const GLOBAL_KB = `${EVAL_DIR}/deep_head/GLOBAL_KB.md`;
    const prepHead = async (h) => {
      const ext = await safeAgent(
        roleAgent('kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, KERNEL: h, GEMM_SYNTH,
          CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          REQUIRE_DECODE_BUCKET: true, DECODE_M_BUCKETS: [1, CONC],
          PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
        }),
        { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (!ext || ext.smoke !== 'pass' || !ext.task_dir) {
        const why = ext ? ext.notes || ext.smoke : 'none';
        log(`  [deep-v2] ${h.short_name}: op extraction failed (${why})${isDominant ? ' [DOMINANT — flagged]' : ''}; skipping.`);
        if (isDominant) flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'extract', gate: 'extract_failed', reason: why });
        history.ledger.push({ direction: h.short_name, verdict: isDominant ? 'flagged' : 'dead_end', lesson: `op extraction failed (${why})` });
        return null;
      }
      const bake = await safeAgent(
        roleAgent('op_benchmarker', 'bakeoff', 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.', {
          EVAL_DIR, OP_TASK_DIR: ext.task_dir, OP_KIND: ext.op_kind, PCT_GPU_TIME: h.pct_gpu_time,
          CANDIDATE_BACKENDS: ext.candidate_backends || h.candidate_backends || [],
          GPU_ID: GPU_LIST[0], ENABLE_FP8, KERNEL_WF_DIR, KERNEL_BUDGET: DEEP_WAVE_KERNEL_BUDGET, SKILL_DIR: WORKFLOW_DIR,
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
      const opSpec = { op_kind: ext.op_kind, shapes: ext.shapes || {}, dtype: ext.dtype || 'bf16', regime: h.regime || 'both', cuda_graph_safe: true };
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
      log(`[deep-v2] ${h.short_name} (${(h.pct_gpu_time || 0).toFixed(1)}% GPU): ${lanes.length} lanes [${lanes.map(l => l.key + ':' + l.mode).join(', ')}]; roofline×${(rooflineTarget || 0).toFixed(2)}.`);
      return { h, ext, lanes, deepDir, sharedKb, liveBaselineMs };
    };

    headDispatched += dHeads.length;
    const preps = [];
    for (const h of dHeads) { if (DEEP_DEADLINE_HIT) break; const p = await prepHead(h); if (p) preps.push(p); }
    const allLanes = [];
    for (const p of preps) for (const l of p.lanes) allLanes.push(l);
    if (!allLanes.length) { log('[deep-v2] no viable head lanes; nothing to optimize.'); }

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
          // ceiling-aware patience: lanes still far from their ceiling get MORE waves before parking.
          const farFromCeiling = (l.ceiling - l.best) > 0.3 * Math.max(l.ceiling, 1e-9);
          const streakCap = farFromCeiling ? DEEP_PLATEAU_STREAK_HIGH : DEEP_PLATEAU_STREAK;
          if (e.has_state === false && anyState && l.ran > 0) { l.active = false; log(`  [deep-v2] park ${l.uid} — no persisted result while peers produced.`); }
          else if (l.noImprove >= streakCap) { l.active = false; log(`  [deep-v2] park ${l.uid} (plateau ${l.best.toFixed(3)}x, ceiling ${l.ceiling.toFixed(2)}x).`); }
        }
      }
    };

    // ---- CURATE per-op SHARED_KB + run-global GLOBAL_KB, and REVIVE high-ceiling parked lanes ---------
    const curateAndRevive = async (tag) => {
      for (const p of preps) {
        const ls = allLanes.filter(l => l.head.short_name === p.h.short_name && (l.lastEval || l.ran > 0));
        if (!ls.length) continue;
        await safeAgent(
          `KB CURATOR [deep-v2 ${tag}] for ${p.h.short_name} (${p.lanes[0] ? p.lanes[0].ext.op_kind : ''}). Per-lane vs-live best: ${JSON.stringify(ls.map(l => ({ uid: l.uid, lang: l.lang, vs_live: l.best, active: l.active, eval_dir: l.lastEval })))}. ` +
          `For each lane with an eval_dir, READ its insight_log.md + tech_lead_report.md; extract what WORKED (measured), what FAILED (scoped dead-end), and any technique another lane should borrow. ` +
          `REWRITE ${p.sharedKb} (keep sections; "Current best per backend" one row per lane; every technique a MEASURED effect + SOURCE; disproven -> Dead-ends; fill "Cross-backend assignments (borrow)" with concrete "lane X found Y -> lane Z try W"). ` +
          `ALSO append cross-KERNEL transferable techniques to ${GLOBAL_KB} (techniques that should generalize to OTHER head ops/backends in this run; one line each, with the source uid). Keep both concise/high-signal. Return {ok,summary}.`,
          { phase: 'HeadKernel', label: `curate ${p.h.short_name} ${tag}`, schema: OKV });
      }
      // revive: a parked lane that is still HIGH-ceiling (big remaining gap) gets one more shot with the
      // freshly-curated cross-backend borrows — so an initially-poor high-ceiling backend is not abandoned.
      let revived = 0;
      for (const l of allLanes) {
        if (l.active || (l.revives || 0) >= 2) continue;
        if ((l.ceiling - l.best) > 0.4 * Math.max(l.ceiling, 1e-9)) { l.active = true; l.noImprove = 0; l.revives = (l.revives || 0) + 1; revived++; }
      }
      if (revived) log(`  [deep-v2] revived ${revived} high-ceiling parked lane(s) with fresh borrows.`);
    };

    // ---- BATCHED serial e2e GATE on the serving slot (runs concurrently with co-opt on dedicated cards)
    const runGate = async () => {
      const cands = allLanes.filter(l => (l.lastEval || l.patch) && l.best > 1.0).sort((a, b) => b.best - a.best).slice(0, 2);
      if (!cands.length) return;
      e2eGateCount++;
      log(`[deep-v2] E2E GATE #${e2eGateCount} on serving {${SERVING_GPU}} TP=${SERVING_TP}: [${cands.map(c => c.uid + ' ' + c.best.toFixed(3) + 'x').join(', ')}] (overlapping co-opt on dedicated cards).`);
      for (const c of cands) {
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
          }),
          { phase: 'HeadKernel', label: `integrate ${c.uid} g${e2eGateCount}`, schema: INTEGRATE_SCHEMA });
        if (integ && integ.output_parity === 'fail') {
          log(`  [deep-v2] ${c.uid}: REJECTED — output_parity=fail vs true baseline.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'dead_end', lesson: 'parity fail vs true baseline' });
        } else if (integ && (integ.gate === 'accepted' || integ.gate === 'stack') && integ.e2e_throughput_tok_s > curTput) {
          curOverlay = integ.accepted_overlay || curOverlay; curTput = integ.e2e_throughput_tok_s;
          acceptedHeads.push({ short_name: c.head.short_name, op_kind: c.ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', e2e_delta_pct: integ.e2e_delta_pct, isolated: c.best });
          log(`  [deep-v2] ${c.uid}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%); target ${Math.round(BASELINE_TPUT * DEEP_E2E_TARGET)} tok/s.`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
        } else {
          log(`  [deep-v2] ${c.uid}: e2e gate ${integ ? integ.gate : 'none'} (${integ ? integ.reason || '' : 'integrate failed'}).`);
          history.ledger.push({ direction: c.uid, isolated_speedup: c.best, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: integ ? integ.reason || 'no e2e gain' : 'integrate failed' });
        }
      }
      const fb = await safeAgent(
        `You are the e2e FEEDBACK + HARNESS refiner [deep-v2 g${e2eGateCount}]. For the candidates just gated, write ${EVAL_DIR}/deep_head/e2e_feedback.md (per candidate: e2e delta; ENGAGED-live vs eager-fell-back under cudagraph; cudagraph/memory/decode behavior; parity/accuracy; ROOT CAUSE of any isolated->e2e gap). ` +
        `Then refresh ${EVAL_DIR}/deep_head/HARNESS_ADDENDUM.md so the isolated target ALIGNS with e2e WITHOUT touching the frozen oracle: (a) e2e-critical decode M-buckets to weight; (b) whether a cudagraph capture/replay measurement wrapper is needed; (c) hard gates (decode-no-regress, memory cap, graph-safe) so an isolated "win" that is all-NUL/eager under graph is caught EARLY. Return {ok, feedback_path, addendum_path}.`,
        { phase: 'HeadKernel', label: `feedback g${e2eGateCount}`, schema: OKV });
      gateFeedbackPath = (fb && fb.feedback_path) || `${EVAL_DIR}/deep_head/e2e_feedback.md`;
      gateHarnessPath = (fb && fb.addendum_path) || `${EVAL_DIR}/deep_head/HARNESS_ADDENDUM.md`;
      // re-profile if the stack moved enough — chase the new dominant bottleneck (Amdahl shifted).
      if (want('profile') && curTput > lastReprofileTput * (1 + DEEP_V2_REPROFILE_GAIN)) {
        log(`[deep-v2] e2e +${((curTput / lastReprofileTput - 1) * 100).toFixed(1)}% since last profile — re-profiling to chase the moving bottleneck.`);
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
          log(`[deep-v2] re-profile updated head Amdahl weights.`);
        }
        lastReprofileTput = curTput;
      }
    };

    // ---- GLOBAL WAVE LOOP ---------------------------------------------------------------------------
    let wave = 0, e2eGateCount = 0, lastE2eIsoBest = 1.0, lastReprofileTput = curTput, gateFeedbackPath = '', gateHarnessPath = '', e2eIntervalHit = false;
    const armInterval = () => { if (typeof setTimeout === 'function' && DEEP_E2E_MAX_INTERVAL_MS > 0) setTimeout(() => { e2eIntervalHit = true; }, DEEP_E2E_MAX_INTERVAL_MS); };
    armInterval();
    while (!DEEP_DEADLINE_HIT && allLanes.some(l => l.active)) {
      wave++;
      const globalIsoBest = Math.max(1.0, ...allLanes.map(l => l.best));
      const gained = globalIsoBest / Math.max(lastE2eIsoBest, 1e-9) - 1;
      const haveCand = allLanes.some(l => (l.lastEval || l.patch) && l.best > 1.0);
      const gateDue = haveCand && (e2eGateCount === 0 || gained >= DEEP_E2E_GAIN_TRIGGER || e2eIntervalHit);
      // pick ready lanes by EV; a gate wave reserves the serving cards (only dedicated cards co-opt).
      const slots = gateDue ? mainSlots : (mainSlots + serveSlots);
      const ready = allLanes.filter(l => l.active).sort((a, b) => evOf(b) - evOf(a)).slice(0, Math.max(0, slots));
      const onMain = ready.slice(0, mainSlots);
      const onServe = gateDue ? [] : ready.slice(mainSlots);
      log(`[deep-v2] WAVE ${wave}: ${allLanes.filter(l => l.active).length} active; running ${ready.length} [${ready.map(l => l.uid).join(', ') || '-'}]${gateDue ? ' + E2E GATE (overlap)' : ''}; e2e ${curTput} tok/s.`);
      const runBurst = (l, pool) => pool.with(1, async (g) => {
        l.ran++;
        await deepBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
          kernel_path: l.ext.task_dir, workflow_dir: KERNEL_WF_DIR, mode: l.mode, target_language: l.lang, op_spec: l.opSpec,
          perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR, use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
          budget: DEEP_WAVE_KERNEL_BUDGET, max_no_improve: DEEP_WAVE_KERNEL_BUDGET, gpu_ids: g[0],
          state_dir: l.state_dir, shared_kb: l.sharedKb, global_kb: GLOBAL_KB,
          ...(gateFeedbackPath ? { e2e_feedback: gateFeedbackPath } : {}),
          ...(gateHarnessPath ? { harness_addendum: gateHarnessPath } : {}),
          exp_root: `${l.deepDir}/runs/${l.key}`, apply_to_original: 'false',
          task: `DEEP-v2 lane '${l.key}' of ${l.head.short_name} (${l.ext.op_kind}), backend=${l.lang}, mode=${l.mode}.${l.steer} Build STRICTLY beyond this lane's cumulative best (vs-live ${l.best.toFixed(3)}x); roofline SOTA ~${(l.rooflineTarget || 0).toFixed(2)}x. Beat the LIVE kernel, not just your own first port. Read SHARED_KB + GLOBAL_KB and BORROW transferable techniques (incl. from OTHER kernels); write findings back.` + GRAPH_REQ + (TASK || ''),
        }, l.uid);
        return null;
      });
      await Promise.all([
        gateDue ? runGate() : Promise.resolve(),
        parallel(onMain.map(l => () => runBurst(l, cooptMain))),
        parallel(onServe.map(l => () => runBurst(l, cooptServe))),
      ]);
      if (gateDue) { lastE2eIsoBest = globalIsoBest; e2eIntervalHit = false; armInterval(); }   // e2eGateCount is bumped inside runGate()
      await harvestLanes(ready, `w${wave}`);
      await curateAndRevive(`w${wave}`);
    }
    await harvestLanes(allLanes.filter(l => l.lastEval || l.ran > 0), 'final');
    await runGate();   // final gate on accumulated best
    log(`[deep-v2] done after ${wave} wave(s), ${e2eGateCount} gate(s). e2e ${curTput} tok/s (${(curTput / Math.max(BASELINE_TPUT, 1e-9)).toFixed(3)}× baseline; target ×${DEEP_E2E_TARGET}). per-lane vs-live: ${allLanes.map(l => l.uid + '=' + l.best.toFixed(2) + 'x').join(', ')}.`);
  } else if (DEEP_MODE) {
    // ===================== DEEP-MODE CROSS-BACKEND CO-OPTIMIZATION HEAD TRACK (deep-mode only) ========
    // Per head op (dominant first), N backends optimize the SAME op IN PARALLEL on exclusive GPU lanes,
    // each CONTINUING across waves via kernel_workflow STATE_DIR (no lost experience), sharing a live
    // blackboard KB (curator distills + assigns cross-backend borrows), anchored to a roofline target.
    // Between waves an ADAPTIVE, BATCHED e2e gate validates the best candidate(s) and feeds the result +
    // a refined harness addendum back so the isolated target tracks e2e. The GPU semaphore gives co-opt
    // lanes exclusive cards; the e2e gate leases ALL cards (TP serving) so it never overlaps co-opt.
    const ISO = makeSem(GPU_LIST);
    const ROOFLINE_SCHEMA = { type: 'object', additionalProperties: true, properties: { roofline_note: { type: 'string' }, target_geomean: { type: 'number' } }, required: ['roofline_note'] };
    const OK_SCHEMA = { type: 'object', additionalProperties: true, properties: { ok: { type: 'boolean' }, summary: { type: 'string' }, feedback_path: { type: 'string' }, addendum_path: { type: 'string' } }, required: ['ok'] };
    const HARVEST_SCHEMA = { type: 'object', additionalProperties: true, properties: { lanes: { type: 'array', items: { type: 'object', additionalProperties: true, properties: { key: { type: 'string' }, backend: { type: 'string' }, has_state: { type: 'boolean' }, cumulative: { type: 'number' }, best_ms: { type: 'number' }, vs_live: { type: 'number' }, eval_dir: { type: 'string' }, patch: { type: 'string' } }, required: ['key'] } } }, required: ['lanes'] };
    const dHeads = heads.slice().sort((a, b) => (b.pct_gpu_time || 0) - (a.pct_gpu_time || 0));
    const pctSum = dHeads.reduce((s, h) => s + (h.pct_gpu_time || 1), 0) || 1;
    log(`[deep-mode] cross-backend co-opt over ${dHeads.length} head op(s); GPU pool {${GPU_LIST.join(',')}}; serving slot {${SERVING_GPU}} TP=${SERVING_TP}; HeadKernel budget ${Math.round(DEEP_HEAD_BUDGET_MS / 3600000)}h.`);

    for (const h of dHeads) {
      if (DEEP_DEADLINE_HIT) { log(`[deep-mode] budget hit — skipping remaining head ${h.short_name}.`); break; }
      headDispatched++;
      // (1) extract the op into a standalone immutable unittest (spans decode + prefill regimes).
      const ext = await safeAgent(
        roleAgent('kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, KERNEL: h, GEMM_SYNTH,
          CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
          REQUIRE_DECODE_BUCKET: true, DECODE_M_BUCKETS: [1, CONC],
          PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
        }),
        { phase: 'HeadKernel', label: `extract_op ${h.short_name}`, schema: EXTRACT_OP_SCHEMA });
      const isDominant = (h.pct_gpu_time || 0) >= HEAD_PROTECT_PCT;
      if (!ext || ext.smoke !== 'pass' || !ext.task_dir) {
        const why = ext ? ext.notes || ext.smoke : 'none';
        log(`  [deep-mode] ${h.short_name}: op extraction failed (${why})${isDominant ? ' [DOMINANT — flagged]' : ''}; skipping.`);
        if (isDominant) flaggedHeads.push({ short_name: h.short_name, pct_gpu_time: h.pct_gpu_time, stage: 'extract', gate: 'extract_failed', reason: why });
        history.ledger.push({ direction: h.short_name, verdict: isDominant ? 'flagged' : 'dead_end', lesson: `op extraction failed (${why})` });
        continue;
      }
      // (2) bake-off: discover candidate backends + author_plan + best_known_ms.
      const bake = await safeAgent(
        roleAgent('op_benchmarker', 'bakeoff', 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.', {
          EVAL_DIR, OP_TASK_DIR: ext.task_dir, OP_KIND: ext.op_kind, PCT_GPU_TIME: h.pct_gpu_time,
          CANDIDATE_BACKENDS: ext.candidate_backends || h.candidate_backends || [],
          GPU_ID: GPU_LIST[0], ENABLE_FP8, KERNEL_WF_DIR, KERNEL_BUDGET: DEEP_WAVE_KERNEL_BUDGET, SKILL_DIR: WORKFLOW_DIR,
        }),
        { phase: 'HeadKernel', label: `bakeoff ${h.short_name}`, schema: OPBENCH_SCHEMA });

      // Derive the co-opt LANES. Key lesson from the first run: cross-backend parallelism BACKFIRES when
      // only one backend is viable (e.g. MXFP8-E8M0 where only Triton has the primitive) — it wastes GPUs
      // on structurally-dead authors (tilelang/hip/flydsl 3x slower) while the one viable lane only gets
      // shallow tuning, so deep_mode lost to fast_mode (which stacked Triton fused+split-K kernels to +21%).
      // Fix: CONCENTRATE. Always run the editable live backend (triton) PLUS each genuinely-viable other
      // backend the bake-off proposed; then FILL the remaining GPU lanes with DIVERSE TRITON DIRECTIONS
      // (fused author, split-K author, deep-explore) so every card amplifies the winner instead of a dead end.
      // Each lane has a UNIQUE `key` (multiple triton lanes) + a `steer` appended to its task.
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
        const otherLangs = [...new Set(planLangs.map(x => x.lang).filter(l => l && l !== 'triton'))];
        // lane 0: tune the editable live Triton kernel
        lanesSpec = [{ key: 'triton-opt', lang: 'triton', mode: 'optimize',
          steer: ' DIRECTION=tile-tune: per-shape AUTOTUNE the live tl.dot_scaled — BLOCK_M/N/K, num_warps{4,8}, num_stages{1,2}, waves_per_eu, matrix_instr_nonkdim, GROUP_SIZE_M L2-swizzle; split-K for decode small-M. Stay cudagraph-safe.' }];
        // genuine cross-backend lanes (only backends the bake-off actually deemed worth authoring)
        for (const l of otherLangs) lanesSpec.push({ key: l, lang: l, mode: (planLangs.find(x => x.lang === l) || {}).mode || 'author',
          steer: ` AUTHOR a ${l} implementation that beats the LIVE native-Triton kernel (not just your own first port); read SHARED_KB and borrow the winning decomposition other lanes found.` });
        // FILL remaining cards with diverse Triton authoring directions (amplify the always-viable backend)
        const extraTriton = [
          { key: 'triton-fused', lang: 'triton', mode: 'author',
            steer: ' DIRECTION=fused-author: AUTHOR a fresh FUSED single-pass Triton kernel — fold the per-token requant + the E8M0 microscale into ONE tl.dot_scaled MFMA core (kill any dequant/requant materialization), epilogue-fuse activation/scale. Beat the LIVE kernel.' },
          { key: 'triton-splitk', lang: 'triton', mode: 'author',
            steer: ' DIRECTION=split-K: AUTHOR a SPLIT-K + atomic-accumulate Triton variant for the prefill-heavy large-M shapes (parallelize the K reduction across CUs), with a per-shape launch selector that uses the non-split path for decode small-M.' },
          { key: 'triton-deep', lang: 'triton', mode: 'optimize',
            steer: ' DIRECTION=deep-explore: combine persistent-kernel + epilogue fusion + grid-swizzle + aggressive tiling in one coherent rewrite; push toward the roofline SOTA bar.' },
        ];
        for (const t of extraTriton) { if (lanesSpec.length >= GPU_LIST.length) break; lanesSpec.push(t); }
      }
      const backends = lanesSpec.slice(0, Math.max(2, GPU_LIST.length));  // at most one lane per card
      // The LIVE serving baseline = the op's frozen-oracle geomean ms (the bar an authored kernel must
      // beat). CRITICAL for cross-backend ranking: a kernel_workflow's own "cumulative speedup" is
      // SELF-RELATIVE (vs that lane's first port), so an author lane reporting 5x can still be 3x SLOWER
      // than the live kernel. We rank/gate by speedup-VS-LIVE = liveBaselineMs / lane_best_ms instead.
      const liveBaselineMs = (bake && Number.isFinite(bake.best_known_ms) && bake.best_known_ms > 0) ? bake.best_known_ms : 0;
      const deepDir = `${EVAL_DIR}/deep_head/${h.short_name}`;
      const sharedKb = `${deepDir}/SHARED_KB.md`;
      const opSpec = { op_kind: ext.op_kind, shapes: ext.shapes || {}, dtype: ext.dtype || 'bf16', regime: h.regime || 'both', cuda_graph_safe: true };
      log(`[deep-mode] ${h.short_name} (${(h.pct_gpu_time || 0).toFixed(1)}% GPU): co-opt lanes [${backends.map(b => b.key + ':' + b.mode).join(', ')}].`);

      // (3) roofline anchor + shared-KB bootstrap (one cheap agent, no GPU).
      const anchor = await safeAgent(
        `You are the ROOFLINE ANCHOR + shared-KB bootstrapper for DEEP cross-backend optimization of head op ${h.short_name} (${ext.op_kind}). ` +
        `Inputs: OP_TASK_DIR=${ext.task_dir}; shapes=${JSON.stringify(ext.shapes || {})}; dtype=${ext.dtype || '?'}; read ${EVAL_DIR}/env_report.json for the on-box device peak (FLOP/s + HBM bandwidth). ` +
        `DO: (a) mkdir -p ${deepDir}; (b) compute the ROOFLINE ceiling per case (classify compute- vs memory-bound, give a target ms per case and an overall SOTA target geomean ~80-90% of roofline an authored kernel should approach); ` +
        `(c) bootstrap ${sharedKb} with these sections (markdown): Roofline target (per-case ceiling + the SOTA geomean bar); Current best per backend (table backend|best geomean|technique|wave — empty now); Techniques that WORK (evidence: technique -> measured effect -> source); Dead-ends (do NOT retry, with evidence); Cross-backend assignments (borrow) (concrete "backend X -> backend Z try W"); Open hypotheses. Cite relevant ${KERNEL_KNOWLEDGE_DIR} cards for ${ext.op_kind}. ` +
        `Return {roofline_note, target_geomean}.`,
        { phase: 'HeadKernel', label: `roofline ${h.short_name}`, schema: ROOFLINE_SCHEMA });
      const rooflineTarget = anchor && Number.isFinite(anchor.target_geomean) ? anchor.target_geomean : 0;

      // per-lane state (continuing across waves via STATE_DIR; key is unique even for multiple triton lanes)
      const lanes = backends.map((b) => ({ ...b, key: b.key || b.lang, steer: b.steer || '', state_dir: `${deepDir}/state/${b.key || b.lang}`, best: 1.0, noImprove: 0, active: true, lastEval: '', patch: '' }));
      let e2eFeedbackPath = '';
      let harnessAddPath = '';
      let lastE2eIsoBest = 1.0;
      let e2eGateCount = 0;
      let wave = 0;
      // per-head time slice (Amdahl-weighted) layered UNDER the global 20h deadline.
      let waveDeadlineHit = false;
      const headSlice = Math.max(1800000, Math.floor(DEEP_HEAD_BUDGET_MS * ((h.pct_gpu_time || 1) / pctSum)));
      const sliceTimer = (typeof setTimeout === 'function') ? setTimeout(() => { waveDeadlineHit = true; }, headSlice) : null;
      // adaptive e2e interval (re-armed after each gate)
      let e2eIntervalHit = false;
      const armInterval = () => { if (typeof setTimeout === 'function' && DEEP_E2E_MAX_INTERVAL_MS > 0) setTimeout(() => { e2eIntervalHit = true; }, DEEP_E2E_MAX_INTERVAL_MS); };
      armInterval();

      // harvestAndGate: read DISK TRUTH (each lane's STATE.json/best/patch, immune to the cap nulling a
      // burst's return) → update bests → park dead/plateaued lanes → curate SHARED_KB → fire the adaptive
      // batched e2e gate on the ACCUMULATED best. Called at the TOP of each wave (so the gate tests
      // completed work and is DECOUPLED from this wave's slowest burst — on a resumed run it tests the
      // prior best, e.g. tilelang's 2.7x, immediately) and once after the loop for the final wave's bursts.
      const harvestAndGate = async (tag) => {
        const harvest = await safeAgent(
          `Deep co-opt of ${h.short_name} (${ext.op_kind}): report the AUTHORITATIVE persisted state per ACTIVE lane (read from disk, do NOT guess). ` +
          `The LIVE serving baseline geomean for this op = ${liveBaselineMs || '?'} ms (the bar to beat; a lane's own "cumulative" is SELF-RELATIVE to its first port and is NOT comparable — you MUST compute speedup vs THIS live baseline). Lanes (id=key):\n` +
          lanes.filter(l => l.active).map(l => `- key=${l.key} (lang=${l.lang}): STATE.json=${l.state_dir}/STATE.json; cumulative-best workspace=${l.state_dir}/best; newest finished run=newest ${deepDir}/runs/${l.key}/team_*/*/ with a final_patch.diff (its baseline_timing.json + tech_lead_report.md give the OPTIMIZED absolute per-case ms)`).join('\n') + '\n' +
          `For EACH active lane return: key (the lane id above); has_state (does ${'$'}state_dir/STATE.json exist AND ${'$'}state_dir/best/kernel_src exist); best_ms (the ABSOLUTE geomean ms of this lane's cumulative-BEST kernel across the op's cases — read the lane's measured optimized per-case ms, NOT a relative number); vs_live (= ${liveBaselineMs || 0} / best_ms when both > 0, i.e. speedup vs the LIVE baseline; 1.0 if unknown); cumulative (the self-relative number, for reference); eval_dir (newest runs/<key>/team_*/<op> dir with a non-empty final_patch.diff, else ""); patch (that final_patch.diff path, else ""). Return {lanes:[{key,has_state,best_ms,vs_live,cumulative,eval_dir,patch}]}.`,
          { phase: 'HeadKernel', label: `harvest ${h.short_name} ${tag}`, schema: HARVEST_SCHEMA });
        const hmap = {}; for (const e of (harvest && Array.isArray(harvest.lanes) ? harvest.lanes : [])) hmap[e.key || e.backend] = e;
        const anyState = Object.values(hmap).some(e => e && e.has_state);   // some lane has produced a result
        for (const l of lanes) {
          if (!l.active) continue;
          const e = hmap[l.key];
          // Rank by speedup VS LIVE (comparable across lanes), NOT the self-relative cumulative.
          const g = e && Number.isFinite(e.vs_live) && e.vs_live > 0 ? e.vs_live
                  : (e && Number.isFinite(e.cumulative) && !liveBaselineMs ? e.cumulative : null);
          if (g != null && g > l.best * 1.001) { l.best = g; l.noImprove = 0; } else if (anyState) l.noImprove++;
          if (e && e.eval_dir) l.lastEval = e.eval_dir;
          if (e && e.patch) l.patch = e.patch;
          // Park a structurally-dead lane (peers produced but this one has NO STATE) → frees its GPU.
          // Guarded by anyState so a fresh first wave parks nothing.
          if (e && e.has_state === false && anyState) { l.active = false; log(`  [deep-mode] parking ${h.short_name}/${l.key} — no persisted result while peers produced; GPU freed.`); }
          else if (l.noImprove >= DEEP_PLATEAU_STREAK) { l.active = false; log(`  [deep-mode] parking ${h.short_name}/${l.key} (plateau at ${l.best.toFixed(3)}x); GPU freed.`); }
        }
        // CURATOR — distill the latest persisted findings into SHARED_KB for the upcoming bursts.
        await safeAgent(
          `You are the KB CURATOR for deep co-opt of ${h.short_name} (${ext.op_kind}) [${tag}]. Per-lane state (key:lang, vs-live best): ${JSON.stringify(lanes.map(l => ({ key: l.key, lang: l.lang, vs_live_best: l.best, active: l.active, eval_dir: l.lastEval })))}. ` +
          `For each lane WITH an eval_dir, READ its insight_log.md + tech_lead_report.md and extract what WORKED (measured effect), what FAILED (dead-end), and any technique another lane should borrow. ` +
          `REWRITE ${sharedKb} keeping its sections: update "Current best per backend" (one row per lane key); every technique needs a MEASURED effect + a SOURCE; move disproven items to "Dead-ends"; fill "Cross-backend assignments (borrow)" with concrete next directives ("lane X found Y -> lane Z try W"). Keep it concise/high-signal. Return {ok, summary}.`,
          { phase: 'HeadKernel', label: `curate ${h.short_name} ${tag}`, schema: OK_SCHEMA });
        // ADAPTIVE, BATCHED e2e GATE on the accumulated best.
        const globalIsoBest = Math.max(1.0, ...lanes.map(l => l.best));
        const gained = globalIsoBest / Math.max(lastE2eIsoBest, 1e-9) - 1;
        const wantGate = globalIsoBest > 1.0 && (e2eGateCount === 0 || gained >= DEEP_E2E_GAIN_TRIGGER || e2eIntervalHit);
        if (!wantGate) return;
        const gateCands = lanes.filter(l => (l.lastEval || l.patch) && l.best > 1.0).sort((a, b) => b.best - a.best).slice(0, 2);
        if (!gateCands.length) return;
        e2eGateCount++; e2eIntervalHit = false; lastE2eIsoBest = globalIsoBest; armInterval();
        log(`[deep-mode] ${h.short_name} E2E GATE #${e2eGateCount}: batch-testing [${gateCands.map(c => c.key + ' ' + c.best.toFixed(3) + 'x').join(', ')}] on serving slot {${SERVING_GPU}} TP=${SERVING_TP}.`);
        await ISO.with(GPU_LIST.length, async () => {  // lease ALL cards -> serving never overlaps co-opt
          for (const c of gateCands) {
            const integ = await safeAgent(
              roleAgent('e2e_integrator', 'integrate', 'Apply a deep head candidate; gate on e2e throughput; report engagement/cudagraph/mem/decode problems for feedback.', {
                EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
                KERNEL_RESULT: {
                  short_name: h.short_name, task_dir: ext.task_dir, op_kind: ext.op_kind, lane: c.key,
                  winner_kind: 'patch', winner_backend: c.lang,
                  target_callable: ext.target_callable || h.target_callable || '',
                  authored_language: c.lang, authored_kernel_eval_dir: c.lastEval,
                  apply_env: '', apply_flags: '', code_patch: c.patch || (c.lastEval ? `${c.lastEval}/final_patch.diff` : ''), tuning_artifact: '',
                  verified_isolated_speedup: c.best, pct_gpu_time: h.pct_gpu_time, parity_note: 'expected_close',
                },
                CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
                CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR, DEEP_FEEDBACK: true,
                ...ACCURACY_INPUTS,
              }),
              { phase: 'HeadKernel', label: `integrate ${h.short_name}/${c.key} g${e2eGateCount}`, schema: INTEGRATE_SCHEMA });
            // Belt-and-suspenders: never accept a parity-failing candidate even if the gate says so —
            // this is what let cumulative MXFP8 drift ride through before (7/12 diverged vs true baseline).
            if (integ && integ.output_parity === 'fail') {
              log(`  [deep-mode] ${h.short_name}/${c.key}: REJECTED — output_parity=fail vs true baseline (no cumulative drift allowed).`);
              history.ledger.push({ direction: `${h.short_name}/${c.key}`, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'dead_end', lesson: 'parity fail vs true baseline' });
            } else if (integ && (integ.gate === 'accepted' || integ.gate === 'stack') && integ.e2e_throughput_tok_s > curTput) {
              curOverlay = integ.accepted_overlay || curOverlay;
              curTput = integ.e2e_throughput_tok_s;
              acceptedHeads.push({ short_name: h.short_name, op_kind: ext.op_kind, backend: c.lang, lane: c.key, kind: 'patch', e2e_delta_pct: integ.e2e_delta_pct, isolated: c.best });
              log(`  [deep-mode] ${h.short_name}/${c.key}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
              history.ledger.push({ direction: `${h.short_name}/${c.key}`, isolated_speedup: c.best, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
            } else {
              log(`  [deep-mode] ${h.short_name}/${c.key}: e2e gate ${integ ? integ.gate : 'none'} (${integ ? integ.reason || '' : 'integrate failed'}).`);
              history.ledger.push({ direction: `${h.short_name}/${c.key}`, isolated_speedup: c.best, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: integ ? integ.reason || 'no e2e gain' : 'integrate failed' });
            }
          }
        });
        const fb = await safeAgent(
          `You are the e2e FEEDBACK + HARNESS refiner for ${h.short_name} (${ext.op_kind}). e2e gate #${e2eGateCount} just ran. ` +
          `Write ${deepDir}/e2e_feedback.md: per tested candidate, the e2e delta, whether the kernel ENGAGED live vs eager-fell-back under cudagraph, cudagraph behavior, memory-footprint impact, decode regression, parity — and the ROOT CAUSE of any isolated->e2e gap. ` +
          `Then write/refresh ${deepDir}/HARNESS_ADDENDUM.md so the isolated target ALIGNS with e2e WITHOUT touching the immutable oracle (unittest.py/meta.json/reference_io.pt stay frozen): (a) which cases to weight (e2e-critical decode M-buckets), (b) whether a cudagraph capture/replay measurement wrapper is required, (c) hard constraint gates (decode-no-regress, memory cap, cudagraph-safe). Source from integrator outputs under ${EVAL_DIR} + candidate eval dirs. Return {ok, feedback_path, addendum_path}.`,
          { phase: 'HeadKernel', label: `feedback ${h.short_name} g${e2eGateCount}`, schema: OK_SCHEMA });
        e2eFeedbackPath = (fb && fb.feedback_path) || `${deepDir}/e2e_feedback.md`;
        harnessAddPath = (fb && fb.addendum_path) || `${deepDir}/HARNESS_ADDENDUM.md`;
      };

      while (!DEEP_DEADLINE_HIT && !waveDeadlineHit && wave < DEEP_MAX_WAVES_PER_HEAD && lanes.some(l => l.active)) {
        wave++;
        // (A) GATE FIRST on accumulated disk truth (prior bursts / resumed state) — decoupled from this
        // wave's slowest burst; on a resumed run this tests the prior best immediately.
        await harvestAndGate(`w${wave}-pre`);
        const live = lanes.filter(l => l.active);
        if (!live.length) break;
        log(`[deep-mode] ${h.short_name} WAVE ${wave}: ${live.length} active lane(s) [${live.map(l => l.key).join(',')}].`);
        // (B) PRODUCE: one bounded burst per active lane (exclusive card) — improves disk for next gate.
        await parallel(live.map((l) => async () => ISO.with(1, async (g) => {
          await deepBoundedWorkflow({ scriptPath: KERNEL_WF_SCRIPT }, {
            kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
            mode: l.mode, target_language: l.lang, op_spec: opSpec,
            perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR, use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
            budget: DEEP_WAVE_KERNEL_BUDGET, max_no_improve: DEEP_WAVE_KERNEL_BUDGET, gpu_ids: g[0],
            state_dir: l.state_dir, shared_kb: sharedKb,
            ...(e2eFeedbackPath ? { e2e_feedback: e2eFeedbackPath } : {}),
            ...(harnessAddPath ? { harness_addendum: harnessAddPath } : {}),
            exp_root: `${deepDir}/runs/${l.key}`, apply_to_original: 'false',
            task: `DEEP co-opt lane '${l.key}' of ${h.short_name} (${ext.op_kind}), backend=${l.lang}, mode=${l.mode}.${l.steer} Build STRICTLY beyond this lane's cumulative best (vs-live ${l.best.toFixed(3)}x); SOTA roofline target ~${rooflineTarget || '?'}x. The LIVE native kernel is the bar — beat IT, not just your own first port. Read SHARED_KB and BORROW transferable techniques from other lanes; write your findings back.` + GRAPH_REQ + (TASK || ''),
          }, `${h.short_name}:${l.key}`);
          return null;
        })));
      } // end wave loop
      await harvestAndGate('final');   // (C) gate the final wave's bursts (loop exits before a top-harvest)
      if (sliceTimer && typeof clearTimeout === 'function') clearTimeout(sliceTimer);
      log(`[deep-mode] ${h.short_name} done after ${wave} wave(s), ${e2eGateCount} e2e gate(s). best vs-live: ${lanes.map(l => l.key + '=' + l.best.toFixed(3) + 'x').join(', ')}. e2e now ${curTput} tok/s.`);
    } // end per-head deep loop
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
        const ext = await safeAgent(
          roleAgent('kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
            EVAL_DIR, MODEL_PATH, GPU_ID: gpu, WORKLOAD, KERNEL: h, GEMM_SYNTH,
            CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
            REQUIRE_DECODE_BUCKET: true, DECODE_M_BUCKETS: [1, CONC],
            PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
          }),
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
      if (p.dead === 'deadline') { log(`  [fast-mode] ${h.short_name}: skipped (dispatch deadline).`); continue; }
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
            op_spec: { op_kind: j.ext.op_kind, shapes: j.ext.shapes || {}, dtype: j.ext.dtype || 'bf16', regime: j.h.regime || '', cuda_graph_safe: true },
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
        log(`  ${j.short_name}: authored ${lang} ${al.final_geomean.toFixed(2)}x (vs its own baseline).`);
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
      const integ = await safeAgent(
        roleAgent('e2e_integrator', 'integrate', 'Apply the head-op winner; gate on e2e throughput.', {
          EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
          KERNEL_RESULT: { short_name: h.short_name, task_dir: st.ext.task_dir, op_kind: st.ext.op_kind,
            winner_kind: cand.winner_kind, winner_backend: cand.source,
            target_callable: st.ext.target_callable || h.target_callable || '',
            authored_language: cand.language || '', authored_kernel_eval_dir: cand.kernel_eval_dir || '',
            apply_env: cand.apply_env || '', apply_flags: cand.apply_flags || '',
            code_patch: cand.code_patch || cand.final_patch || '', tuning_artifact: cand.tuning_artifact || '',
            verified_isolated_speedup: cand.isolated || 0, pct_gpu_time: h.pct_gpu_time,
            parity_note: cand.parity_note || 'expected_close' },
          CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
          CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
        }),
        { phase: 'HeadKernel', label: `integrate ${h.short_name}`, schema: INTEGRATE_SCHEMA });
      if (integ && (integ.gate === 'accepted' || integ.gate === 'stack') && integ.e2e_throughput_tok_s > curTput) {
        curOverlay = integ.accepted_overlay || curOverlay;
        if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
        if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
        curTput = integ.e2e_throughput_tok_s;
        acceptedHeads.push({ short_name: h.short_name, op_kind: st.ext.op_kind, backend: cand.source, kind: cand.winner_kind, e2e_delta_pct: integ.e2e_delta_pct, isolated: cand.isolated });
        log(`  ${h.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
        history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
      } else {
        log(`  ${h.short_name}: REJECTED at e2e gate (${integ ? integ.reason || integ.gate : 'none'}).`);
        history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: integ ? integ.reason || 'no e2e gain' : 'integrate failed' });
      }
    }
  } else {
  for (const h of heads) {
    // Fast-mode budget guard: stop STARTING new head ops once the dispatch deadline has fired, so the
    // in-flight work + Finalize/Validate still land inside the wall-clock budget. (No-op in default mode.)
    if (FAST_MODE && FAST_DEADLINE_HIT) {
      log(`[fast-mode] budget deadline reached — stopping head dispatch before ${h.short_name} (${headDispatched}/${heads.length} heads done).`);
      break;
    }
    headDispatched++;
    // (h1) Extract the op into a standalone immutable unittest (GEMM synth / attn capture).
    const ext = await safeAgent(
      roleAgent('kernel_extractor', 'extract_op', 'Build a standalone op unittest for a head kernel.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: h.gpu_id, WORKLOAD, KERNEL: h, GEMM_SYNTH,
        CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
        // The unittest MUST span BOTH regimes. Steady-state serving is decode/TPOT-bound, so a
        // head GEMM tuned only on GPU-time-dominant prefill M regresses decode and loses e2e.
        // Pass the decode M explicitly (= running batch ≈ conc) so it is never dropped, plus a
        // per-step M=1. See kernel_extractor.md "Shapes must span BOTH regimes".
        REQUIRE_DECODE_BUCKET: true,
        DECODE_M_BUCKETS: [1, CONC],
        PREFILL_M_NOTE: 'also include the profiled large prefill M (chunk size, ~thousands) per (N,K)',
      }),
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
            op_spec: { op_kind: ext.op_kind, shapes: ext.shapes || {}, dtype: ext.dtype || 'bf16', regime: h.regime || '', cuda_graph_safe: true },
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
        log(`  ${h.short_name}: authored ${lang} ${al.final_geomean.toFixed(2)}x (vs its own baseline).`);
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
    const integ = await safeAgent(
      roleAgent('e2e_integrator', 'integrate', 'Apply the head-op winner; gate on e2e throughput.', {
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
      }),
      { phase: 'HeadKernel', label: `integrate ${h.short_name}`, schema: INTEGRATE_SCHEMA });

    if (integ && (integ.gate === 'accepted' || integ.gate === 'stack') && integ.e2e_throughput_tok_s > curTput) {
      // a head winner may be carried as overlay (authored/patch) AND/OR config (env/flag) — capture both.
      curOverlay = integ.accepted_overlay || curOverlay;
      if (cand.winner_kind === 'env' && cand.apply_env) curEnv = (curEnv ? curEnv + ' ' : '') + cand.apply_env;
      if (cand.winner_kind === 'flag' && cand.apply_flags) curFlags = (curFlags ? curFlags + ' ' : '') + cand.apply_flags;
      curTput = integ.e2e_throughput_tok_s;
      acceptedHeads.push({ short_name: h.short_name, op_kind: ext.op_kind, backend: cand.source, kind: cand.winner_kind, e2e_delta_pct: integ.e2e_delta_pct, isolated: cand.isolated });
      log(`  ${h.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else {
      log(`  ${h.short_name}: REJECTED at e2e gate (${integ ? integ.reason || integ.gate : 'none'}).`);
      history.ledger.push({ direction: h.short_name, isolated_speedup: cand.isolated, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: integ ? integ.reason || 'no e2e gain' : 'integrate failed' });
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
while (want('kernel') && dispatched < BUDGET && (dispatched < MIN_KERNEL_TASKS || noImprove < 2)) {
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
    const ext = await safeAgent(
      roleAgent('kernel_extractor', 'extract', 'Capture shapes + oracle; emit an immutable unittest task dir.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: c.gpu_id, WORKLOAD, KERNEL: c,
        CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv, SKILL_DIR: WORKFLOW_DIR,
      }),
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
    const integ = await safeAgent(
      roleAgent('e2e_integrator', 'integrate', 'Overlay the optimized kernel back; gate on e2e throughput.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: c.gpu_id, WORKLOAD, NOISE_BAND_PCT: NOISE_BAND, E2E_REPEATS,
        KERNEL_RESULT: { short_name: c.short_name, task_dir: ext.task_dir,
          source_path_in_sglang: ext.source_path_in_sglang, target_callable: ext.target_callable,
          final_patch: kl.final_patch, verified_isolated_speedup: kl.final_geomean, pct_gpu_time: c.pct_gpu_time },
        CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
        CURRENT_THROUGHPUT: curTput, SKILL_DIR: WORKFLOW_DIR,
      }),
      { phase: 'Milestone', label: `integrate ${c.short_name}`, schema: INTEGRATE_SCHEMA });

    if (integ && (integ.gate === 'accepted' || integ.gate === 'stack') && integ.e2e_throughput_tok_s > curTput) {
      curOverlay = integ.accepted_overlay || curOverlay;
      curTput = integ.e2e_throughput_tok_s;
      acceptedKernels.push({ short_name: c.short_name, backend: kl.note || '', e2e_delta_pct: integ.e2e_delta_pct, isolated: kl.final_geomean });
      milestoneImproved = true;
      log(`  ${c.short_name}: ACCEPTED. e2e now ${curTput} tok/s (+${integ.e2e_delta_pct}%).`);
      history.ledger.push({ direction: c.short_name, isolated_speedup: kl.final_geomean, e2e_delta_pct: integ.e2e_delta_pct, verdict: 'confirmed', lesson: integ.reason || '' });
    } else {
      log(`  ${c.short_name}: REJECTED at e2e gate (${integ ? integ.reason || integ.gate : 'none'}).`);
      history.ledger.push({ direction: c.short_name, isolated_speedup: kl.final_geomean, e2e_delta_pct: integ ? integ.e2e_delta_pct : 0, verdict: 'dead_end', lesson: integ ? integ.reason || 'no e2e gain' : 'integrate failed' });
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
const allAccepted = acceptedHeads.concat(acceptedKernels);
let finalize = null, report = null, validation = null;
let finalTput = curTput, finalSpeedup = BASELINE_TPUT ? curTput / BASELINE_TPUT : 1.0;
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
    roleAgent('director', 'validate', 'Independently re-measure throughput + parity; arbitrate.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], BASELINE_THROUGHPUT: BASELINE_TPUT, NOISE_BAND_PCT: NOISE_BAND,
      FINAL_OVERLAY: (finalize && finalize.final_overlay) || curOverlay,
      FINAL_FLAGS: { flags: curFlags, env: curEnv },
      CLAIMED_THROUGHPUT: finalTput, WORKLOAD, APPLY_TO_ORIGINAL, E2E_REPEATS, SKILL_DIR: WORKFLOW_DIR,
    }),
    { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });
  finalSpeedup = validation ? validation.throughput_speedup : (finalTput / BASELINE_TPUT);
  log(`COMPLETE. ${MODEL_NAME}: ${BASELINE_TPUT} -> ${validation ? validation.director_verified_throughput_tok_s : finalTput} tok/s ` +
    `(${finalSpeedup ? finalSpeedup.toFixed(3) : '?'}x, status ${validation ? validation.validation_status : '?'}). Results in ${EVAL_DIR}`);
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
  headQueue, kernelQueue, accepted_heads: acceptedHeads, flagged_heads: flaggedHeads, accepted_kernels: acceptedKernels, history,
};

return {
  mode: 'e2e',
  fast_mode: FAST_MODE,   // true => ConfigSweep + Milestone skipped; HeadKernel-only within the time budget
  deep_mode: DEEP_MODE,   // true => HeadKernel runs the long cross-backend co-optimization scheduler (20h)
  backend: BACKEND,
  phases_run: PHASES,
  eval_dir: EVAL_DIR,
  model_name: MODEL_NAME,
  baseline_throughput_tok_s: BASELINE_TPUT,
  final_throughput_tok_s: validation ? validation.director_verified_throughput_tok_s : finalTput,
  throughput_speedup: finalSpeedup,
  validation_status: validation ? validation.validation_status : (want('final') ? 'unknown' : 'phase_partial'),
  output_parity: validation ? validation.output_parity : 'unknown',
  accepted_config: { flags: curFlags, env: curEnv },
  accepted_kernels: acceptedKernels,
  accepted_heads: acceptedHeads,
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
