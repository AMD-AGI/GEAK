export const meta = {
  name: 'team-perf-workflow-e2e',
  description: 'End-to-end LLM inference-throughput optimizer for AMD MI300X. The serving stack is pluggable via scripts/adapters/<backend>.sh (sglang + vllm shipped; pass args.backend). A system layer (e2e Director / System Architect / Profiler / Config Tuner / Kernel Extractor / e2e Integrator) wraps the UNCHANGED single-kernel team_workflow: it preflights the env, profiles a running server, triages hot kernels by Amdahl, tunes config/backends, extracts hot editable kernels into standalone unittests, recursively optimizes them with team_workflow.js, overlays them back, and re-validates serving throughput. Also still optimizes a single kernel (pass-through).',
  whenToUse: 'Optimize the serving throughput of an LLM on MI300X. Pass args.model_path (required) + optional args.backend (sglang|vllm, default sglang) + args.launch_script (optional). For a single kernel, pass args.kernel_path instead and it delegates straight to the kernel layer.',
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
  throw new Error('args.workflow_dir is required: absolute path to the dir holding team_workflow_e2e.js, ' +
    'roles/, knowledge/, scripts/ (the dirname of this script).');
}
// The UNCHANGED single-kernel workflow. Default: sibling "workflows" dir next to this one.
const KERNEL_WF_DIR = String(A.kernel_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/workflows')).replace(/\/+$/, '');
const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/team_workflow.js`;

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
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);       // max kernel-optimization tasks
// MIN floor: dispatch at LEAST this many editable-kernel tasks before the loop may stop on no-improve /
// empty queue (prompt-tunable). Prevents the milestone track from never firing. Capped by BUDGET.
const MIN_KERNEL_TASKS = Math.min(parseInt(A.min_kernel_tasks != null ? A.min_kernel_tasks : 4, 10), BUDGET);
// Milestone only optimizes editable kernels whose profiled share is worth it: skip any candidate with
// pct_gpu_time below this threshold (Amdahl — a kernel a few % of GPU can't move e2e past the noise band).
// Configurable via args.milestone_min_pct (default 5). This OVERRIDES the MIN_KERNEL_TASKS floor: if no
// candidate clears the bar, the milestone stops rather than grinding low-value kernels.
const MILESTONE_MIN_PCT = parseFloat(A.milestone_min_pct != null ? A.milestone_min_pct : 5);
const KERNEL_BUDGET = parseInt(A.kernel_budget != null ? A.kernel_budget : 6, 10); // budget passed DOWN per kernel
const CONFIG_TUNE_ENABLED = String(A.config_tune != null ? A.config_tune : 'true') === 'true';
// Head-kernel track (GEMM/attention) — the highest-pct_gpu_time ops, optimized regardless of edit flag.
const HEAD_THRESHOLD_PCT = parseFloat(A.head_threshold_pct != null ? A.head_threshold_pct : 5);
const HEAD_BUDGET = parseInt(A.head_budget != null ? A.head_budget : 3, 10); // max head-op bake-offs
// Author route: how many languages to author per head op. The Op Benchmarker orders author_plan by ROI
// (for a GEMM head: flydsl first — SOTA GEMM DSL — then triton). Default 2 covers flydsl+triton per head
// while keeping the kernel-layer cost bounded; bump to try hip/ck too when the headroom justifies it.
const HEAD_AUTHOR_MAX = parseInt(A.head_author_max != null ? A.head_author_max : 2, 10);
// The AMD authoring knowledge base (REFERENCE ONLY — facts/how-to, never decisions; agents always
// measure). Default: sibling kernel_knowledge/. Workflows enumerate candidates from
// index/capability_index.yaml; status/perf in cards are dated evidence, not routing inputs.
const KERNEL_KNOWLEDGE_DIR = String(A.kernel_knowledge_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kernel_knowledge')).replace(/\/+$/, '');
const GEMM_SYNTH = String(A.gemm_synth != null ? A.gemm_synth : 'true');     // synth GEMM inputs (cheap)
const ENABLE_FP8 = String(A.enable_fp8 != null ? A.enable_fp8 : 'false');    // Tier-D quant (parity-breaking)
const FAST_PATH_FIRST = String(A.fast_path_first != null ? A.fast_path_first : 'true') === 'true';
const ISL = parseInt(A.isl != null ? A.isl : 1024, 10);
const OSL = parseInt(A.osl != null ? A.osl : 1024, 10);
const CONC = parseInt(A.conc != null ? A.conc : 64, 10);
const WORKLOAD = { isl: ISL, osl: OSL, conc: CONC };
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
const want = (p) => RUN_ALL || PHASES.includes(p);
const ST = A.state || {};   // carried state from a prior phase invocation

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
  server_flags: { type: 'object', additionalProperties: true }, workload: { type: 'object', additionalProperties: true },
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
  per_backend: arrObj, parity_note: { type: 'string' }, gate: { type: 'string' }, reason: { type: 'string' },
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

function roleAgent(role, phase, intro, inputs) {
  // BACKEND is injected for every role: any role that calls bench_e2e.sh must forward it
  // (BACKEND=<backend>) so the right serving adapter (scripts/adapters/<backend>.sh) is used.
  const inall = { BACKEND, ...inputs };
  return `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
When you invoke bench_e2e.sh, pass BACKEND=${BACKEND} in its env so the correct serving adapter is used.

## SERVING CONFIG INVARIANT (do not violate — all e2e numbers must be comparable)
Every e2e benchmark in this run (baseline, config sweep, integrate ref/cand, validation) MUST use the
SAME serving config: tensor-parallel TP=1 on a SINGLE GPU. GPU_IDS=${GPU_IDS} is the
OPTIMIZATION-PARALLELISM pool (spread recursive kernel/head optimization across GPUs) — it is NOT the
serving tensor-parallel size. Launch every serving server on ONE GPU (a single id from the pool) with
TP=1 (the bench_e2e.sh default). NEVER set TP>1 or GPU=0,1,2,3 for a serving launch — a TP=4 baseline
vs a TP=1 candidate makes every delta meaningless.

## Inputs
${cfg(inall)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
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
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 7200000, 10);
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

// ===========================================================================
// SINGLE-KERNEL PASS-THROUGH: delegate straight to the unchanged kernel layer.
// ===========================================================================
if (!MODEL_PATH && KERNEL_PATH) {
  phase('Setup');
  log(`Single-kernel pass-through -> ${KERNEL_WF_SCRIPT} on ${KERNEL_PATH}`);
  // Recurse into the UNCHANGED kernel layer via the native workflow() primitive (one allowed level of
  // nesting). team_workflow.js returns {eval_dir, final_geomean, final_patch, validation_status, ...}.
  let passthru;
  try {
    const r = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, {
      kernel_path: KERNEL_PATH, workflow_dir: KERNEL_WF_DIR,
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
      GPU_IDS, WORKLOAD, SKILL_DIR: WORKFLOW_DIR,
    }),
    { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
  if (!setup || !setup.eval_dir) throw new Error('Setup failed: no eval_dir');
  EVAL_DIR = setup.eval_dir;
  MODEL_NAME = setup.model_name || MODEL_NAME_HINT;
  BASELINE_TPUT = setup.baseline_throughput_tok_s;
  NOISE_BAND = setup.noise_band_pct || NOISE_BAND_DEFAULT;
  curFlags = (setup.server_flags && setup.server_flags.extra) || '';
  curEnv = '';
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
  for (const h of heads) {
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
    if (!ext || ext.smoke !== 'pass' || !ext.task_dir) {
      log(`  ${h.short_name}: op extraction failed (${ext ? ext.notes || ext.smoke : 'none'}); skipping.`);
      history.ledger.push({ direction: h.short_name, verdict: 'dead_end', lesson: 'op extraction failed' });
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
      log(`  ${h.short_name}: no win and nothing worth authoring (${bake ? bake.reason || bake.gate : 'none'}); skipping.`);
      history.ledger.push({ direction: h.short_name, isolated_speedup: bake ? bake.isolated_speedup : 0, verdict: 'dead_end', lesson: bake ? bake.reason || 'no op win' : 'bakeoff failed' });
      continue;
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
      const AUTHOR_TRIES = parseInt(A.head_author_tries != null ? A.head_author_tries : 2, 10);
      for (let attempt = 1; attempt <= AUTHOR_TRIES; attempt++) {
        try {
          al = await workflow({ scriptPath: KERNEL_WF_SCRIPT }, {
            kernel_path: ext.task_dir, workflow_dir: KERNEL_WF_DIR,
            mode: ap.route === 'rewrite' ? 'optimize' : 'author', target_language: lang,
            op_spec: { op_kind: ext.op_kind, shapes: ext.shapes || {}, dtype: ext.dtype || 'bf16', regime: h.regime || '', cuda_graph_safe: true },
            kernel_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
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
              `kernel MUST fit at the same mem-fraction the accepted config uses. ` + (TASK || ''),
            apply_to_original: 'false',
          });
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
      log(`  ${h.short_name}: no candidate to integrate; skipping.`);
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
        budget: KERNEL_BUDGET, gpu_ids: c.gpu_id, exp_root: `${EVAL_DIR}/kernels/_exp`,
        task: 'Compare candidate backends ' + JSON.stringify(c.candidate_backends || []) +
          ' for this kernel; pick the fastest that passes the immutable unittest. ' + (TASK || ''),
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
    roleAgent('system_architect', 'update_experience', 'Append durable findings to the backend playbook.', {
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
      ACCEPTED_HEADS: acceptedHeads, MILESTONES: milestone, BUDGET_USED: dispatched, BUDGET, MIN_KERNEL_TASKS,
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
  headQueue, kernelQueue, accepted_heads: acceptedHeads, accepted_kernels: acceptedKernels, history,
};

return {
  mode: 'e2e',
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
