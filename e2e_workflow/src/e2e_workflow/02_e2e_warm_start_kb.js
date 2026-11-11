
// E2E-LEVEL WARM START — the DEPLOYMENT knowledge base: was this deployment (model, gfx, framework
// + version, precision, tp, isl/osl/conc) tuned before? Module A reads the store and validates on
// this box before believing it; Module B records this run at the end. A.warm_start governs both
// this and the kernel layer:
//   on (default)      | read the ladder, validate the top candidate on this box, ADOPT if it wins.
//   reference         | read only; never bench, never adopt — the offer is prose for later agents.
//   off/false/none    | nothing: no phase, no agent, no log.
const E2E_WARM_START = KB_ARGS.warm_start.trim().toLowerCase() || 'on';
const E2E_WARM_START_ON = !WARM_START_OFF;
// Fast mode forces reference-only: it skips config work, so it won't spend a 20-40min server launch
// benching a stored config.
const E2E_WARM_START_REF_ONLY = E2E_WARM_START === 'reference' || FAST_MODE;
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
// The caller's declared ISL/OSL, kept apart from the resolved target shape (they differ on a trace
// replay); the resolved ISL/OSL/CONC are defined after the workload kind is parsed.
const ISL_DECLARED = A.isl != null ? parseInt(A.isl, 10) : null;
const OSL_DECLARED = A.osl != null ? parseInt(A.osl, 10) : null;

