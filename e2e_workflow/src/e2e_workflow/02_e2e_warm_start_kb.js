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
//   off/false/none    | nothing: no phase, no agent, no log, and every role prompt byte-identical.
const E2E_WARM_START = KB_ARGS.warm_start.trim().toLowerCase() || 'on';
const E2E_WARM_START_ON = !WARM_START_OFF;
// Fast mode's premise is that all optimization comes from the head track within a wall-clock budget
// (FAST_SKIP already drops 'config'), so a warm-start config bench — a 20-40min server launch that
// this mode has explicitly declined to spend anywhere else — is forced down to reference-only.
const E2E_WARM_START_REF_ONLY = E2E_WARM_START === 'reference' || FAST_MODE;
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
// TWO questions, TWO floors: "worth KNOWING about" and "worth a 20-40min server launch". A single
// floor dropped a small stored win from the offer entirely, losing the tuning table and env var it
// carried, not just the launch. Reading is free, so the read floor only drops records that LOST.
// `e2e_`-prefixed because the kernel lane's `warm_start_min_speedup` means its own thing; the bare
// name stays honoured for callers raising this floor.
const E2E_WARM_START_MIN_SPEEDUP = Number.isFinite(parseFloat(A.e2e_warm_start_min_speedup))
  ? parseFloat(A.e2e_warm_start_min_speedup)
  : Number.isFinite(parseFloat(A.warm_start_min_speedup))
    ? parseFloat(A.warm_start_min_speedup) : 1.0;
// The floor that spends money. Session-to-session drift on one box is the size of a near-tie
// claim, so benching a 1.01x is a coin flip against its own noise; below this the offer stays a
// reference. Not applied to the kernel replay below — an accepted-kernel entry carries an ISOLATED
// speedup measured on a different harness, budgeted by E2E_WARM_START_KERNELS_N instead.
const E2E_WARM_START_BENCH_MIN_SPEEDUP =
  Number.isFinite(parseFloat(A.e2e_warm_start_bench_min_speedup))
    ? parseFloat(A.e2e_warm_start_bench_min_speedup) : 1.02;

// How a local verdict is filed AGAINST THE RECORD. The local bench decides what THIS run adopts
// and nothing else — a different box means a different baseline, image, driver and neighbours — so
// each outcome goes to the bucket that describes it, and none retires anything by itself:
//
//   rejected        applied, took effect, lost here -> `failed`. The only bucket that can
//                   accumulate toward a retire decision, which is why it must not be discarded.
//   not_reproduced  never ran, or ran without taking effect (a flag renamed upstream) -> the
//                   record is missing something.
//   inapplicable    collided with a baseline this run did not choose -> a verdict on the PAIRING,
//                   which kb/attest.py keeps out of the arithmetic.
//
// Mapping all three to `inapplicable` protected records too well: `tried = recalls - inapplicable`
// is then identically zero and nothing can ever accumulate a retire signal. The judgement stays a
// separate act (kb/attest.py:should_retire, run by `e2e_store.py curate`, dry-run by default).
// Every value must be one of kb/attest.py:OUTCOMES.
const KB_ATTEST_OUTCOME = {
  adopted: 'validated',
  rejected: 'failed',
  not_reproduced: 'not_reproduced',
  inapplicable: 'inapplicable',
};
// Read broadly, bench narrowly. Reading is free and breadth is exactly what makes the demoted
// reference path useful to the Architect; benching is the expensive half. So the default is
// "offer 3, measure 1".
const E2E_WARM_START_TOP_N = parseInt(A.e2e_warm_start_top_n != null ? A.e2e_warm_start_top_n : 3, 10);
const E2E_WARM_START_VALIDATE_N = parseInt(
  A.e2e_warm_start_validate_n != null ? A.e2e_warm_start_validate_n : 1, 10);
// The same budget for a COARSE rung (`tp_any`/`workload_any`), where the record was measured at a
// different tp or workload point. It was hard zero, on the argument that a non-comparable stored
// number is not worth a launch — which confuses the PRIOR with the MEASUREMENT: the bench runs the
// config on THIS box against THIS baseline, so the stored number only decides what is tried first.
// Zero meant every `tp_any` hit was filed as a lead nobody followed (the 20260826 gpt-oss-120b and
// Qwen3.5-122B runs). Budgeted, not unbounded, because the launch cost is real; 0 restores the old
// reference-only behaviour.
const E2E_WARM_START_VALIDATE_N_COARSE = parseInt(
  A.e2e_warm_start_validate_n_coarse != null ? A.e2e_warm_start_validate_n_coarse : 1, 10);
// How many stored KERNELS get replayed through a fresh integrate A/B. Each one is another two server
// launches, so this is a budget, not a completeness target — the rest stay references.
const E2E_WARM_START_KERNELS_N = parseInt(
  A.e2e_warm_start_kernels_n != null ? A.e2e_warm_start_kernels_n : 2, 10);
// Which recorded win-kinds can be REBUILT from a record alone. `patch` re-applies a diff through the
// overlay; `env`/`flag` re-route the op to an implementation that already exists in this install.
// `authored` cannot be rebuilt: its recipe git-applies the diff inside
// `authored_kernel_eval_dir/workspace/`, and that workspace dies with the run that made it.
//
// Not-rebuildable is NOT not-replayable, and reading it that way kept every authored kernel in the
// store permanently unverified. The OUTPUT of that build travels in the record as `overlay.tar.gz`
// (e2e_store._pack_overlay) and is the complete, reversible mechanism, so a kind outside this set
// falls through to the prebuilt-overlay path below when the bundle has one, and is demoted to a
// reference only when it has neither. A demotion is never reported as a failure — it is still a
// real datum about what worked on this deployment.
const E2E_REPLAYABLE_KINDS = new Set(['patch', 'env', 'flag']);
// Which roles are TOLD about the warm start. Deliberately excludes e2e_integrator and
// director:validate — those two produce the run's authoritative numbers, and a stored prior in their
// context is contamination with no upside.
// tuning_specialist is a consumer for the same reason, plus one: a tuned artifact deploys outside
// the git tree and takes hours of search to rediscover, so re-deriving one the store already holds
// is the most expensive avoidable thing this workflow does.
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
const USE_EXPERT_SKILLS = !RDNA4_ISOLATE && String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR + '/expert_skills')).replace(/\/+$/, '');
// Only routing/bake-off/integration roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['system_architect', 'op_benchmarker', 'e2e_integrator']);
const GEMM_SYNTH = String(A.gemm_synth != null ? A.gemm_synth : 'true');     // synth GEMM inputs (cheap)
const ENABLE_FP8 = String(A.enable_fp8 != null ? A.enable_fp8 : 'false');    // Tier-D quant (parity-breaking)
// What the CALLER declared, kept apart from the shape roles finally optimize
// against. On a trace replay those two are not the same thing, and telling them
// apart needs the workload kind, which is parsed further down -- so the resolved
// ISL/OSL/CONC/WORKLOAD are defined after it (CONC beside the declaration, the
// shape by resolveTargetShape()).
const ISL_DECLARED = A.isl != null ? parseInt(A.isl, 10) : null;
const OSL_DECLARED = A.osl != null ? parseInt(A.osl, 10) : null;

