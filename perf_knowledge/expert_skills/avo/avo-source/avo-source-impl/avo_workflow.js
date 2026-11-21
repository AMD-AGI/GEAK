export const meta = {
  name: 'avo-perf-workflow',
  description: 'GEAK-v4 integrated source-level AVO: continuous single-lineage kernel evolution with a wall-clock budget, deterministic commit gate, anti-stall supervision, and git-tagged lineage. This integration intentionally excludes standalone ASM and IR artifact optimization.',
  whenToUse: 'Use run_geak_avo.py to squeeze one editable source-level kernel over hours or days. Pass kernel_path, budget_s, gpu_ids, task, and optionally operator=single|team in the handoff JSON. Triton/HIP/CK/FlyDSL files are treated as source files; standalone .co/.s/LLVM-IR/TTGIR optimization is not supported on this branch.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + avo_repo clone, OR auto-resumes from an existing eval_dir' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + TRUE baseline' },
    { title: 'Seed', detail: 'lineage_engineer commits the baseline as avo-v0' },
    { title: 'Profile', detail: 'profile_engineer classifies the baseline bottleneck' },
    { title: 'Evolve', detail: 'wall-clock loop: variation step (single agent OR a bounded Team round) -> independent verify -> pre-commit director validation -> commit gate -> tag; supervisor on stall' },
    { title: 'Report', detail: 'tech_lead writes the final report + cumulative final_patch.diff' },
    { title: 'Validate', detail: 'director independently re-benchmarks the final patch vs the TRUE baseline' },
    { title: 'Cleanup', detail: 'director prunes leaked worktrees + removes ephemeral workspaces (keeps all artifacts)' },
  ],
};

// ===========================================================================
// avo_workflow.js — Step 2 (native deterministic loop)
//
// Continuous single-lineage kernel evolution (AVO) as a PerfSkills Workflow,
// with the SAME contract as workflows/team_workflow.js:
//     args.kernel_path  ->  verified final_patch.diff + geomean
//
// Native port of GEAK's AVO controller.py: the wall-clock budget loop, commit
// gate, git-tagged lineage (avo-v{N}), stagnation counters, and ESCALATE fan-out
// are DETERMINISTIC JS here; LLM agents are used only for judgement (one
// variation step, supervisor re-plan, benchmark/profile/verify, report). No
// Python dependency (the Step-1 thin wrapper over geak-avo is superseded; the
// avo_runner role is kept as an optional fallback).
//
// Like team_workflow.js: the script never touches the filesystem — agents do all
// FS/shell work; every path is supplied/derived from args.
// ===========================================================================

// Deterministic helpers. Workflow metadata must be the first statement, so
// sibling modules are loaded dynamically after meta instead of using static
// imports before it.
const {
  authoritativeFormalValidation,
  evaluateCommitGate,
  passesCommitGate: _gate,
  classifyFinalValidation,
  formalPrecommitEnabled,
  normalizeDirectorValidation,
} = await import('./lib/commit_gate.mjs');
const {
  authoritativeGeneratedPersistentVerification,
} = await import('./lib/asm_screening_artifacts.mjs');
const { verifyFormalArtifacts } = await import('./lib/formal_artifacts.mjs');
const { archiveFormalReport } = await import('./lib/formal_archive.mjs');
const { isHarnessGeneratedPath } = await import('./lib/harness_generated.mjs');
const { selectStartPoints } = await import('./lib/start_select.mjs');
const { rooflineSummary, recommendLevers, assignSpecialists } = await import('./lib/roofline.mjs');
const { shouldStopRoofline, allocateAttention } = await import('./lib/headroom.mjs');
const {
  affordableRounds, canAffordAnotherRound, roundCostS, shouldHonorTeamStop,
} = await import('./lib/stop_gate.mjs');
const { selectMergeGroup, mergeImproves } = await import('./lib/archive.mjs');
const { irSteer, predictRegression } = await import('./lib/ir_evidence.mjs');
const { normalizeSupervisorDirective } = await import('./lib/supervisor_directive.mjs');
const {
  constrainLanguageDirections, constrainSpecialists, validateLanguagePatchScope,
} = await import('./lib/language_guard.mjs');
const {
  dedupeVerifyItems, selectScreeningFinalist,
} = await import('./lib/verify_plan.mjs');
const {
  normalizeCommitScoreField, verifiedCommitScore, withCommitScore,
} = await import('./lib/score_metric.mjs');
const { readFileSync: readPatchFileSync, existsSync: reportExists } = await import('node:fs');
const { createHash } = await import('node:crypto');
const { execFileSync } = await import('node:child_process');

// ---------------------------------------------------------------------------
// Args + defaults.
// ---------------------------------------------------------------------------
const A = args || {};
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel directory)');

const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'avo_workflow.js, roles/, knowledge/, scripts/.');
}
// <GEAK>/perf_knowledge/expert_skills/avo/avo-source/avo-source-impl -> <GEAK>
const GEAK_ROOT = WORKFLOW_DIR.split('/').slice(0, -5).join('/');
const EXP_ROOT = String(A.exp_root || `${GEAK_ROOT}-runs/avo-source`).replace(/\/+$/, '');
// kernel_base/ holds the kernel roles AVO carries, its own profile_kernel.sh, and
// by-path entries for what it reuses from GEAK's kernel_workflow (gpu_lock,
// knowledge).
const KERNEL_WORKFLOW_DIR = String(A.kernel_workflow_dir ||
  (WORKFLOW_DIR + '/kernel_base')).replace(/\/+$/, '');
const ASM_WORKFLOW_DIR = String(A.asm_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/asm-workflow')).replace(/\/+$/, '');
const IR_WORKFLOW_DIR = String(A.ir_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/ir-workflow')).replace(/\/+$/, '');

const KERNEL_PATH_ORIG = A.kernel_path;
const CAMPAIGN_STARTED_AT = Date.now();
const BUDGET_S = parseInt(A.budget_s != null ? A.budget_s : 3600, 10);   // AVO budget = wall-clock seconds
const requestedFinalReserve = parseInt(
  A.final_phase_reserve_s != null
    ? A.final_phase_reserve_s
    : Math.min(900, Math.max(300, Math.round(BUDGET_S * 0.1))),
  10);
const FINAL_PHASE_RESERVE_S = Math.max(
  0, Math.min(Math.max(0, BUDGET_S - 1),
    Number.isFinite(requestedFinalReserve) ? requestedFinalReserve : 0));
const CAMPAIGN_DEADLINE = CAMPAIGN_STARTED_AT + BUDGET_S * 1000;
const MIN_COMMITS = parseInt(A.min_commits != null ? A.min_commits : 1, 10);
const MAX_STEPS = parseInt(A.max_steps != null ? A.max_steps : 1000, 10); // hard safety cap
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Sequential roles (director / benchmark / profile / final validate) lock from
// the full allocated pool so gpu_lock.sh can pick an idle card. Parallel
// variation/verify workers still bind GPU_LIST[i % n].
const GPU_POOL = GPU_IDS;
const TASK = A.task || '';
const TARGET_LANGUAGE_NORM = String(A.target_language || A.kernel_language || '').toLowerCase();
const REQUESTED_LANGUAGE_LEVEL = String(
  A.avo_language_level || A.language_level || 'source').trim().toLowerCase();
const STANDALONE_LOW_LEVEL_TARGET = [
  'asm', 'amdgcn', 'gcn', 'ir', 'llvm-ir', 'llvm_ir', 'll', 'ttgir',
].includes(TARGET_LANGUAGE_NORM);
if (REQUESTED_LANGUAGE_LEVEL !== 'source'
    || STANDALONE_LOW_LEVEL_TARGET
    || String(A.asm_mode != null ? A.asm_mode : '') === 'true'
    || String(A.ir_mode != null ? A.ir_mode : '') === 'true'
    || String(A.ir_feedback != null ? A.ir_feedback : '') === 'true') {
  throw new Error(
    'GEAK-v4-integrated mode=avo supports source-level optimization only; '
    + 'standalone ASM/IR optimization and lower-level IR feedback are intentionally disabled.');
}
const ASM_MODE = false;
const IR_MODE = false;
const LANGUAGE_MODE = 'source';
const LANGUAGE_SKILL_PATH = ASM_MODE
  ? `${WORKFLOW_DIR}/../skills/asm/SKILL.md`
  : (IR_MODE ? `${WORKFLOW_DIR}/../skills/ir/SKILL.md` : '');
const LANGUAGE_ENGINEER_ADDENDUM = ASM_MODE
  ? `${ASM_WORKFLOW_DIR}/roles/asm_engineer_addendum.md`
  : (IR_MODE ? `${IR_WORKFLOW_DIR}/roles/ir_engineer_addendum.md` : '');
let LANGUAGE_SKILL_SHA256 = '';
let LANGUAGE_ADDENDUM_SHA256 = '';
if (LANGUAGE_SKILL_PATH) {
  try {
    const { readFileSync } = await import('node:fs');
    const { createHash } = await import('node:crypto');
    const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
    LANGUAGE_SKILL_SHA256 = sha256(LANGUAGE_SKILL_PATH);
    LANGUAGE_ADDENDUM_SHA256 = sha256(LANGUAGE_ENGINEER_ADDENDUM);
  } catch (e) {
    throw new Error(`Language knowledge unavailable: ${String(e && e.message ? e.message : e)}`);
  }
}
const LANGUAGE_EDIT_TARGET = ASM_MODE
  ? 'kernel.s or window.s only'
  : (IR_MODE ? 'kernel.ir only' : 'source files allowed by COMMANDMENT');
const LANGUAGE_CONTRACT = LANGUAGE_SKILL_PATH
  ? `This is an AVO ${LANGUAGE_MODE} optimization. Before planning or editing, read ${LANGUAGE_SKILL_PATH} `
    + `and ${LANGUAGE_ENGINEER_ADDENDUM}. `
    + `The editable artifact is ${LANGUAGE_EDIT_TARGET}; obey COMMANDMENT for every build, correctness, benchmark, and profile action.`
  : 'This is a source-level optimization; obey COMMANDMENT and its modifiable-file contract.';
const IR_SHAPES = String(A.ir_shapes || A.shapes || '');
const ASM_PACK = String(A.asm_pack || '');
const ASM_ARCH = String(A.asm_arch || A.arch || A.hw || '');
const ASM_SHAPES = String(A.asm_shapes || A.shapes || '');
const ASM_WINDOW_KERNEL = String(A.asm_window_kernel || '');
const ASM_WINDOW_BEGIN = String(A.asm_window_begin || '');
const ASM_WINDOW_END = String(A.asm_window_end || '');
let GEAK_CAPTURE_DIR = String(
  A.capture_dir || A.geak_capture_dir
    || (typeof process !== 'undefined' ? process.env.GEAK_CAPTURE_DIR : '')
    || '');
let IR_CAPTURE_PROVIDER_ENV = {};
try {
  const rawProvider = typeof process !== 'undefined'
    ? process.env.IR_CAPTURE_PROVIDER_JSON : '';
  IR_CAPTURE_PROVIDER_ENV = rawProvider ? JSON.parse(rawProvider) : {};
} catch {
  IR_CAPTURE_PROVIDER_ENV = {};
}
const IR_CAPTURE_PROVIDER = (
  A.capture_provider && typeof A.capture_provider === 'object'
    ? A.capture_provider : IR_CAPTURE_PROVIDER_ENV);
const IR_CAPTURE_MODE = String(
  IR_CAPTURE_PROVIDER.mode || (GEAK_CAPTURE_DIR ? 'existing' : ''));
const IR_CAPTURE_PATH = String(
  IR_CAPTURE_PROVIDER.capture_dir || IR_CAPTURE_PROVIDER.path
    || GEAK_CAPTURE_DIR || '');
const IR_CAPTURE_KERNEL = String(
  IR_CAPTURE_PROVIDER.kernel_name || IR_CAPTURE_PROVIDER.kernel || '');
const IR_CAPTURE_COMMAND_JSON = JSON.stringify(
  IR_CAPTURE_PROVIDER.application_command
    || IR_CAPTURE_PROVIDER.command || []);
const IR_CAPTURE_DISPATCH = Number(
  IR_CAPTURE_PROVIDER.dispatch_index
    ?? IR_CAPTURE_PROVIDER.dispatch ?? -1);
const IR_CAPTURE_TIMEOUT = Number(
  IR_CAPTURE_PROVIDER.timeout_s ?? IR_CAPTURE_PROVIDER.timeout ?? 300);
const REQUESTED_BENCHMARK_PROTOCOL = String(
  A.benchmark_protocol || '').toLowerCase().replace(/_/g, '-');
// IR validation is always authoritative generated-persistent evidence.
let FORMAL_EVIDENCE_MODE = ASM_MODE
  || IR_MODE
  || REQUESTED_BENCHMARK_PROTOCOL === 'paired-abba';
const VERIFY_REPEATS = parseInt(A.verify_repeats != null ? A.verify_repeats : 5, 10); // Tier-1 noise-robust median (default raised from 3)
const SCREENING_REPEATS = Math.max(1, Math.min(
  VERIFY_REPEATS,
  parseInt(A.screening_repeats != null ? A.screening_repeats : 2, 10)));
const USE_PROFILING = String(A.use_profiling != null ? A.use_profiling : 'true') !== 'false';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();
// Optional environment hook for task packages that need runtime setup before every
// correctness/benchmark/profile command (e.g. source-built FlyDSL + ROCm torch).
const KERNEL_ENV_SCRIPT = String(A.kernel_env_script || A.env_script || '').trim();
// Hard by default after realcap false positives: a candidate may pass the fast verifier,
// but it is not committed/tagged until Director independently replays it from the TRUE baseline.
const REQUESTED_PRECOMMIT_DIRECTOR_VALIDATE = String(A.precommit_director_validate != null
  ? A.precommit_director_validate
  : (A.director_validate_each_commit != null ? A.director_validate_each_commit : 'true')) !== 'false';
// ASM screening measurements are never commit evidence. This hard gate cannot be
// disabled by configuration; non-ASM workflows retain the legacy opt-out.
let PRECOMMIT_DIRECTOR_VALIDATE = formalPrecommitEnabled(
  FORMAL_EVIDENCE_MODE, REQUESTED_PRECOMMIT_DIRECTOR_VALIDATE);
const PRECOMMIT_DIRECTOR_REPEATS = parseInt(A.precommit_director_repeats != null
  ? A.precommit_director_repeats
  : (A.director_commit_repeats != null ? A.director_commit_repeats : VERIFY_REPEATS), 10);
const PRECOMMIT_PAIRED_BLOCKS = parseInt(A.precommit_paired_blocks != null
  ? A.precommit_paired_blocks : 5, 10);
const DIRECTOR_PAIRED_BLOCKS = parseInt(A.director_paired_blocks != null
  ? A.director_paired_blocks : 10, 10);

// Deterministic attempt-ledger backstop. When the runtime provides `recordAttempt`
// (the run_avo.mjs shim), the loop records EVERY attempt (incl. failures) itself —
// no dependency on an LLM role running a shell command. When it is absent (e.g. a
// Workflow runtime without the primitive), the variation_engineer role self-records
// instead (we tell it via RECORD_ATTEMPT), so there is always exactly ONE recorder.
const RUNTIME_RECORDS_ATTEMPTS = typeof recordAttempt === 'function';

// Deterministic per-step worktree teardown (orchestrator-owned). When the runtime
// provides `pruneWorktrees` (the run_avo.mjs shim), each step reclaims the scratch
// worktrees — even if an LLM role errored before removing its own, or mis-placed one —
// so the per-step finally is the single source of truth for the worktree lifecycle.
// Inert when the primitive is absent (the roles' own cleanup + the final Cleanup phase
// remain the only mechanism, as before).
const RUNTIME_PRUNES_WORKTREES = typeof pruneWorktrees === 'function';
// Layer 3: orchestrator pre-creates each step's primary workspace at a fixed absolute
// path (the role just `cd`s in), so the LLM never `git worktree add`s the primary
// workspace and cannot mis-place it. Inert when absent (roles self-create, as before).
const RUNTIME_MAKES_WORKSPACE = typeof addWorktree === 'function';
const RUNTIME_SYNCS_HEARTBEAT = typeof syncAvoHeartbeat === 'function';

// Pre-create a clean workspace at `ref` for a role and return its absolute path, or ''
// when the primitive is absent (the role then self-creates from AVO_REPO@ref). Best-effort.
function makeWorkspace(wsPath, ref) {
  if (!RUNTIME_MAKES_WORKSPACE) return '';
  try { return addWorktree(AVO_REPO, wsPath, ref) || ''; } catch { return ''; }
}

// Verify worktrees are created at BASE_REF. If the candidate patch cannot apply
// there, skip the verify agent instead of burning a GPU round on PATCH_DID_NOT_APPLY.
function patchAppliesOnWorkspace(workspace, patchFile) {
  if (!workspace || !patchFile) return true;
  try {
    execFileSync(
      'git',
      ['-c', `safe.directory=${workspace}`, '-C', workspace, 'apply', '--check', patchFile],
      { stdio: 'pipe' });
    return true;
  } catch {
    return false;
  }
}

// Measurement byproducts (overlay trees, `_runs.jsonl`). Not lineage source;
// restore tracked copies so the clean-worktree guard cannot reject a verified
// candidate.
function restoreHarnessGenerated(execFileSync, gitArgs) {
  const tracked = execFileSync(
    'git', [...gitArgs, 'ls-files', '-z'], { encoding: 'utf8' })
    .split('\0').filter(Boolean).filter(isHarnessGeneratedPath);
  if (!tracked.length) return;
  const roots = [...new Set(tracked.map((f) => f.split('/')[0]))];
  execFileSync(
    'git', [...gitArgs, 'restore', '--source=HEAD', '--staged', '--worktree', '--', ...roots],
    { stdio: 'pipe' });
}

async function checkoutAvoRepo(ref, allowDiscard = false) {
  const { execFileSync } = await import('node:child_process');
  const gitArgs = ['-c', `safe.directory=${AVO_REPO}`, '-C', AVO_REPO];
  // unittest may append `_runs.jsonl` or rebuild overlay trees; those are
  // harness byproducts, not candidate source. Restore them before the
  // clean-worktree guard.
  try { restoreHarnessGenerated(execFileSync, gitArgs); } catch { /* absent from this tree */ }
  const dirty = execFileSync(
    'git', [...gitArgs, 'status', '--porcelain', '--untracked-files=no'],
    { encoding: 'utf8' }).trim();
  if (dirty && !allowDiscard) {
    throw new Error(`avo_repo has tracked changes before checkout: ${dirty}`);
  }
  execFileSync('git', [...gitArgs, 'checkout', '--detach',
    ...(allowDiscard ? ['--force'] : []), ref],
    { stdio: 'pipe' });
}

async function assertCommittedParent(tag, parent) {
  const { execFileSync } = await import('node:child_process');
  const gitArgs = ['-c', `safe.directory=${AVO_REPO}`, '-C', AVO_REPO];
  const rev = (ref) => execFileSync(
    'git', [...gitArgs, 'rev-parse', ref], { encoding: 'utf8' }).trim();
  // All lineage tags are annotated (`git tag -a`). Plain `rev-parse <tag>`
  // returns the tag-object SHA, while HEAD and `<tag>^` resolve to commit SHAs.
  // Peel every ref to a commit before comparing ancestry and identity.
  const commitRev = (ref) => rev(`${ref}^{commit}`);
  if (commitRev(`${tag}^`) !== commitRev(parent)
      || commitRev('HEAD') !== commitRev(tag)) {
    throw new Error(`committed tag ${tag} is not a child of ${parent}`);
  }
}

// Commit-gate + stagnation knobs (ported from GEAK avo defaults, tuned for long-horizon micro-optimization).
const COMMIT_EPSILON = parseFloat(A.commit_epsilon != null ? A.commit_epsilon : 0.001);
// Source/IR runs now default to a small anti-lazy floor so exact ties/no-op patches do not seed the lineage.
// ASM keeps a 1.0 floor because genuine instruction-level wins may be sub-0.2% when measured with realcap.
const DEFAULT_MIN_COMMIT_SPEEDUP = ASM_MODE ? 1.0 : 1.002;
const MIN_COMMIT_SPEEDUP = parseFloat(A.min_commit_speedup != null ? A.min_commit_speedup : DEFAULT_MIN_COMMIT_SPEEDUP);
// AVO is often used after the easy source-level wins are gone; remaining source/IR/ASM wins can be sub-1%.
// Default fixed margin is therefore 0.2% for all modes (matching asm-workflow's micro-gain policy).
// The gate is still variance-aware: required margin = max(significanceMargin, noiseK * cv).
const DEFAULT_SIGNIFICANCE_MARGIN = 0.002;
const SIGNIFICANCE_MARGIN = parseFloat(A.commit_significance_margin != null ? A.commit_significance_margin : DEFAULT_SIGNIFICANCE_MARGIN);
// Source/IR default noise multiplier is relaxed so stable 0.5–1% wins can commit after robust re-measurement.
// ASM retains the historic 0.85 default unless overridden, because realcap already reports true replay COV.
const DEFAULT_COMMIT_NOISE_K = ASM_MODE ? 0.85 : 0.2;
const COMMIT_NOISE_K = parseFloat(A.commit_noise_k != null ? A.commit_noise_k : DEFAULT_COMMIT_NOISE_K);
const COMMIT_SCORE_FIELD = normalizeCommitScoreField(A.commit_score_field);
const COMMIT_CALIBRATED_MARGIN_CAP = A.commit_calibrated_margin_cap != null
  ? parseFloat(A.commit_calibrated_margin_cap) : null;
const STALL_NO_IMPROVE = parseInt(A.stall_no_improve != null ? A.stall_no_improve : 5, 10);
const STALL_PATCH_REPEAT = parseInt(A.stall_patch_repeat != null ? A.stall_patch_repeat : 3, 10);
const SUPERVISOR_CYCLES_TO_ESCALATE = parseInt(A.supervisor_cycles_to_escalate != null ? A.supervisor_cycles_to_escalate : 3, 10);
// Budget-aware supervisor: a redirect is only worth running if there is enough wall-clock
// left to (a) run the supervisor re-plan itself (~this many seconds) AND (b) execute at
// least one redirected step afterwards. Otherwise the (expensive) re-plan is wasted right
// before the deadline (observed: supervisor fired at noImprove=8 with no budget left to use
// its new directions → straight to Report).
const SUPERVISOR_OVERHEAD_S = parseInt(A.supervisor_overhead_s != null ? A.supervisor_overhead_s : 150, 10);
const ESCALATE_WORKERS = parseInt(A.escalate_workers != null ? A.escalate_workers : Math.max(2, GPU_LIST.length), 10);
const MAX_INFRA_FAILURES = parseInt(A.max_infra_failures != null ? A.max_infra_failures : 2, 10);
// Root-cause convergence (anti sterile churn): §5b docs/avo_supervisor_design.md
const CONVERGE_BARREN_CYCLES = parseInt(A.converge_barren_cycles != null ? A.converge_barren_cycles : 4, 10);
const CONVERGE_DX_REPEAT = parseInt(A.converge_dx_repeat != null ? A.converge_dx_repeat : 3, 10);
const DX_SIM_THRESHOLD = parseFloat(A.dx_sim_threshold != null ? A.dx_sim_threshold : 0.55);
// Early-stop (intelligent convergence). DEFAULT OFF — by default Evolve uses the FULL
// wall-clock budget (or the MAX_STEPS / persistent-infra-failure safety caps). Set
// args.early_stop=true (aliases smart_stop / converge_early) to RE-ENABLE the early
// exits: the convergence stop (supervisor converge=true / barren-cycles /
// repeated-diagnosis) AND the roofline early-stop. The supervisor still re-plans on
// stall regardless; this flag only controls whether the run is allowed to STOP early.
// Rationale: running to budget is the safer default (a converge=true on a hard kernel
// is a heuristic; spending the rest of the budget probing is rarely worse and often
// surfaces a late win). Opt into early_stop when you'd rather bank unused budget.
const EARLY_STOP = [A.early_stop, A.smart_stop, A.converge_early]
  .some(v => v === true || v === 1 || String(v != null ? v : '').toLowerCase() === 'true' || String(v) === '1');

// TechLead hard-gate stop (operator=team). With early_stop ON it ends the run at once.
// With early_stop OFF the run is supposed to spend its whole budget, so honoring the stop
// unconditionally silently broke that promise -- but ignoring it outright is not the answer
// either: while the stop was unwired the TechLead just re-issued it every round, and a 12h
// ASM run spent rounds 40-51 re-confirming a stop nobody read before finishing over budget.
// Require N CONSECUTIVE stop requests instead: a genuine convergence re-confirms itself,
// a single pessimistic round does not, and a refused round still runs a real direction.
const TEAM_STOP_CONFIRMATIONS = parseInt(A.team_stop_confirmations != null ? A.team_stop_confirmations : 3, 10);
let teamStopStreak = 0;

// Variation operator: 'single' (default, paper-faithful one agent/step) or 'team'
// (each step = one bounded Team round: tech_lead plans K orthogonal specialist
// directions -> fan-out -> verify each -> commit the best). See docs/avo_on_team.md.
// GEAK integration defaults to breadth-inside-depth: a Team fan-out from a
// diversified set of committed starts. The paper-faithful single/best chain
// remains available only when explicitly requested.
const OPERATOR = String(A.operator != null ? A.operator : 'team').trim() || 'team';
const OPERATOR_DIRECTIONS = parseInt(A.operator_directions != null ? A.operator_directions : Math.max(2, GPU_LIST.length), 10);

// Start-point policy (Diverse Fan-out): where parallel workers begin each step.
// 'best' (default) = all from the single best (chain); 'topk' = top-k versions by
// speedup; 'diverse' = best-per-strategy bucket (escape local optima). Only takes
// effect for fan-out (operator=team / ESCALATE); the global best is always first.
const START_POLICY = String(A.start_policy != null ? A.start_policy : 'diverse').trim() || 'diverse';
const START_K = parseInt(A.start_k != null ? A.start_k : OPERATOR_DIRECTIONS, 10);

// Knowledge base K (§3.1): GEAK-v4's operator/backend reference base.
const KERNEL_KNOWLEDGE_DIR = String(A.kernel_knowledge_dir ||
  (GEAK_ROOT + '/perf_knowledge')).replace(/\/+$/, '');

// Delayed profiling (CuTeGen / §3.2): structural-first for the first N steps (or
// until the first commit), then profiling-guided. 0 = profile from step 1.
const PROFILING_AFTER_STEP = parseInt(A.profiling_after_step != null ? A.profiling_after_step : 3, 10);

// Roofline objective (the hardware-anchored north star). When the COMMANDMENT emits
// per-case flops/bytes, every verified candidate is scored against the MI300X
// roofline: the loop steers at the binding ceiling (memory/compute/basement) and
// STOPS at ROOFLINE_TARGET (diminishing returns) instead of only chasing relative
// gains. With no flops/bytes telemetry this is fully inert (relative-only mode).
const HW = String(A.hw || 'gfx942');
const ROOFLINE_TARGET = parseFloat(A.roofline_target != null ? A.roofline_target : 0.85);
const ROOFLINE_OPTS = { hw: HW };

// IR feedback (read-only): turn a roofline regime into a SPECIFIC source lever using
// lower-level IR (Triton TTGIR / AMDGPU asm) as evidence. Master SWITCH `ir_feedback`
// (default off — opt-in until the rig IR dump is wired): when off, every IR path is
// fully inert and the loop behaves exactly as roofline+relative mode. IR is evidence
// only; the commit gate's verified latency stays authoritative. See docs/avo_ir_feedback_design.md.
const IR_FEEDBACK = false;
const IR_LANG = String(A.ir_lang || (IR_MODE ? TARGET_LANGUAGE_NORM : 'auto')); // auto | triton | hip | off
const IR_TIER = String(A.ir_tier || 'auto');     // auto | ttgir | amdgcn
const IR_PRESCREEN = String(A.ir_prescreen != null ? A.ir_prescreen : 'false') === 'true';
// Deterministic IR capture (IR_FEEDBACK): the kernel symbol to scrape from the Triton
// cache (default derived from the kernel dir name) + the cache dir. Threaded into the
// variation prompt so scripts/ir_dump_triton.sh can scrape the newest IR for THIS
// candidate on EVERY step (win or lose) instead of relying on an ad-hoc agent dump.
const IR_KERNEL_NAME = String(A.ir_kernel_name || '_attn_fwd');
const IR_TRITON_CACHE = String(A.ir_triton_cache || '');

// Autotune sweep (A4): the variation operator may run a BOUNDED best-of-N config
// sweep (tile×waves×unroll×num_stages×precision) instead of a single edit — for the
// last-mile co-tuning toward the roofline. Switch + cap; the bound/selection rule is
// lib/autotune.mjs (single source of truth). Inert when off.
const AUTOTUNE = String(A.autotune != null ? A.autotune : 'false') === 'true';
const AUTOTUNE_MAX = parseInt(A.autotune_max_configs != null ? A.autotune_max_configs : 32, 10);

// ---------------------------------------------------------------------------
// JSON-schema fragments.
// ---------------------------------------------------------------------------
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const perCase = {
  type: 'array',
  items: obj({
    name: { type: 'string' }, baseline_ms: { type: 'number' },
    optimized_ms: { type: 'number' }, speedup: { type: 'number' },
    // Optional roofline telemetry (problem properties). When present, the loop
    // scores the candidate against the MI300X roofline; when absent, relative-only.
    flops: { type: 'number' }, bytes: { type: 'number' }, dtype: { type: 'string' },
    // A2: rocprof-MEASURED HBM bytes (preferred over analytic `bytes`) + counters.
    measured_bytes: { type: 'number' },
    counters: obj({ occupancy_pct: { type: 'number' }, mfma_active_pct: { type: 'number' },
      valu_util_pct: { type: 'number' } }, []),
  }, ['name', 'speedup']),
};

const SETUP_SCHEMA = obj({
  resume: { type: 'boolean' },
  eval_dir: { type: 'string' }, avo_repo: { type: 'string' }, baseline_dir: { type: 'string' },
  kernel_name: { type: 'string' }, source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // resume-only: reconstructed loop state
  best_version: { type: 'string' }, cumulative: { type: 'number' }, commits: { type: 'number' },
  prior_steps: { type: 'number' },
  committed: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' },
  failed: { type: 'array', items: { type: 'string' } },
  insights: { type: 'array', items: { type: 'string' } },
  direction: { type: 'object', additionalProperties: true },
  roofline_fraction: { type: 'number' }, roofline_regime: { type: 'string' },
  best_ir: { type: 'object', additionalProperties: true },
}, ['eval_dir', 'avo_repo', 'kernel_name']);

const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

const SEED_SCHEMA = obj({
  seeded: { type: 'boolean' }, tag: { type: 'string' }, notes: { type: 'string' },
}, ['seeded']);

const PROFILE_SCHEMA = obj({
  bottleneck: { type: 'string' }, profiler_used: { type: 'string' }, dispatch_count: { type: 'number' },
  key_metrics: { type: 'object', additionalProperties: true },
  top_opportunities: { type: 'array', items: { type: 'string' } }, summary_path: { type: 'string' },
}, ['bottleneck', 'top_opportunities']);

const VARIATION_SCHEMA = obj({
  status: { type: 'string' },                       // 'ok' | 'no_change' | 'failed' | 'infra_failure'
  strategy: { type: 'string' },
  patch_file: { type: 'string' },                   // incremental best->candidate diff (empty if no change)
  patch_hash: { type: 'string' },
  candidate_sha256: { type: 'string' },             // built candidate bytes, for pre-verify deduplication
  language_skill_sha256: { type: 'string' },
  language_addendum_sha256: { type: 'string' },
  self_speedup: { type: 'number' },                 // provisional (verify decides)
  per_case: perCase, rationale: { type: 'string' },
  // Optional IR metric vector for this candidate (from ir_probe.py; IR_FEEDBACK only).
  ir: obj({ lang: { type: 'string' }, ok: { type: 'boolean' }, tier: { type: 'string' },
    metrics: { type: 'object', additionalProperties: true },
    config: { type: 'object', additionalProperties: true }, notes: { type: 'string' } }, []),
  marked_failed: { type: 'boolean' }, notes: { type: 'string' },
}, ['status']);

const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  verified_geomean: { type: 'number' }, verified_arithmetic: { type: 'number' },
  verified_weighted: { type: 'number' },
  per_case: perCase, variance_note: { type: 'string' },
  // C1: measured noise level (coefficient of variation of the repeated latencies).
  // The commit gate widens its margin to max(significanceMargin, noiseK*cv).
  cv: { type: 'number' },
  protocol: { type: 'string' },
  formal_status: {
    type: 'string',
    enum: [
      'accepted', 'not_applicable', 'no_significant_gain', 'unstable',
      'correctness_failed', 'artifact_mismatch', 'infrastructure_unhealthy',
      'reproduction_mismatch',
    ],
  },
  infrastructure_status: { type: 'string' }, screening_only: { type: 'boolean' },
  ci_lower: { type: 'number' }, ci_upper: { type: 'number' },
  positive_block_fraction: { type: 'number' }, calibrated_margin: { type: 'number' },
  aa_calibration_passed: { type: 'boolean' },
  bimodal: { type: 'boolean' }, benchmark_report: { type: 'string' },
  notes: { type: 'string' },
}, ['status', 'verified_geomean']);
const ASM_VERIFY_SCHEMA = {
  ...VERIFY_SCHEMA,
  properties: {
    ...VERIFY_SCHEMA.properties,
    benchmark_reports: { type: 'array', items: { type: 'string' } },
  },
  required: [...new Set([
    ...(VERIFY_SCHEMA.required || []), 'correctness', 'protocol',
    'infrastructure_status', 'screening_only', 'benchmark_reports',
  ])],
};

const COMMIT_SCHEMA = obj({
  committed: { type: 'boolean' }, version: { type: 'string' }, tag: { type: 'string' },
  notes: { type: 'string' },
}, ['committed']);

const INTEGRATE_SCHEMA = obj({
  status: { type: 'string' },                       // 'ok' | 'no_change' | 'failed'
  patch_file: { type: 'string' },                   // merged incremental diff (base->merged)
  merged_regimes: { type: 'array', items: { type: 'string' } },
  method: { type: 'string' }, self_geomean: { type: 'number' },
  per_case: perCase, rationale: { type: 'string' }, notes: { type: 'string' },
}, ['status']);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: { type: 'array', items: obj({
    id: { type: 'string' }, title: { type: 'string' },
    specialty: { type: 'string', enum: ASM_MODE
      ? ['algorithm', 'memory', 'compute']
      : ['algorithm', 'memory', 'compute', 'host_runtime'] },
    focus_files: { type: 'array', items: { type: 'string' } },
    expected_speedup: { type: 'number' }, prompt: { type: 'string' },
  }, ['title', 'prompt']) },
}, ['directions']);

const SUPERVISOR_SCHEMA = obj({
  diagnosis: { type: 'string' },
  mark_failed: { type: 'array', items: { type: 'string' } },
  new_directions: { type: 'array', items: obj({
    name: { type: 'string' }, expected: { type: 'string' }, prompt: { type: 'string' },
  }, ['name']) },
  backtrack_to: { type: 'string' }, next_direction: { type: 'string' }, notes: { type: 'string' },
  converge: { type: 'boolean' },
}, ['diagnosis', 'new_directions']);

const REPORT_SCHEMA = obj({
  final_speedup_geomean: { type: 'number' }, final_speedup_arithmetic: { type: 'number' },
  steps: { type: 'number' }, commits: { type: 'number' },
  report_path: { type: 'string' }, final_patch: { type: 'string' }, per_case: perCase, notes: { type: 'string' },
  practical_roofline_fraction: { type: 'number' },
  practical_roofline_regime: { type: 'string' },
  roofline_evidence_path: { type: 'string' },
}, ['final_speedup_geomean', 'report_path', 'final_patch']);

const FINALIZE_SCHEMA = obj({
  finalized: { type: 'boolean' },
  output_dir: { type: 'string' },
  artifacts: { type: 'array', items: { type: 'string' } },
  notes: { type: 'string' },
}, ['finalized', 'output_dir', 'artifacts']);

const VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  per_case: perCase, applied_to_original: { type: 'string' },
  final_patch: { type: 'string' }, cv: { type: 'number' }, notes: { type: 'string' },
  protocol: { type: 'string' }, formal_status: { type: 'string' },
  infrastructure_status: { type: 'string' }, screening_only: { type: 'boolean' },
  ci_lower: { type: 'number' }, ci_upper: { type: 'number' },
  positive_block_fraction: { type: 'number' }, calibrated_margin: { type: 'number' },
  aa_calibration_passed: { type: 'boolean' },
  bimodal: { type: 'boolean' }, benchmark_report: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);
const FORMAL_VALIDATE_SCHEMA = {
  ...VALIDATE_SCHEMA,
  // Agent output is diagnostic only. The orchestrator reads and validates the
  // authoritative realcap_paired_result.json before any gate decision.
  required: ['validation_status'],
};

// ---------------------------------------------------------------------------
// Prompt helpers. roleAgent reads from this workflow's roles/; roleAgentFrom
// reads a REUSED role (benchmark/profile/verify) from KERNEL_WORKFLOW_DIR.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

function promptFor(rolesDir, knowledgeHintDir, role, phase, intro, inputs) {
  return `You are the ${role}. PHASE=${phase}.
First Read ${rolesDir}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to (under ${rolesDir}/knowledge/ or ${knowledgeHintDir}/knowledge/).
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
}
const roleAgent = (role, phase, intro, inputs) => promptFor(WORKFLOW_DIR, KERNEL_WORKFLOW_DIR, role, phase, intro, inputs);
const roleAgentFrom = (dir, role, phase, intro, inputs) => promptFor(dir, WORKFLOW_DIR, role, phase, intro, inputs);
const verifyRole = ASM_MODE
  ? 'avo_verify_asm' : (IR_MODE ? 'avo_verify_ir' : 'avo_verify');
function enforceLanguageKnowledgeEvidence(result) {
  if (!LANGUAGE_SKILL_PATH || !result) return result;
  if (result.language_skill_sha256 === LANGUAGE_SKILL_SHA256
      && result.language_addendum_sha256 === LANGUAGE_ADDENDUM_SHA256) {
    return result;
  }
  return {
    ...result,
    status: 'infra_failure',
    patch_file: '',
    notes: `${result.notes || ''} Language skill/addendum read evidence missing or mismatched.`.trim(),
  };
}

function enforceAsmPatchScope(checked) {
  if (!ASM_MODE || !checked || checked.status !== 'ok' || !checked.patch_file) {
    return checked;
  }
  try {
    const scope = validateLanguagePatchScope(
      readPatchFileSync(checked.patch_file, 'utf8'), LANGUAGE_MODE);
    if (scope.ok) return checked;
    return {
      ...checked,
      status: 'failed',
      notes: `${checked.notes || ''} ASM scope rejected patch files: ${
        scope.violations.join(', ')}.`.trim(),
    };
  } catch (e) {
    return {
      ...checked,
      status: 'infra_failure',
      notes: `${checked.notes || ''} Could not inspect ASM patch scope: ${String(e)}.`.trim(),
    };
  }
}

function enforceWorkerLanguageScope(result) {
  return enforceAsmPatchScope(enforceLanguageKnowledgeEvidence(result));
}
const newPersistentScreeningSession = () => {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  return `${LANGUAGE_MODE}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

function normalizePersistentScreening(
  ver, sessionPrefix, workspace, expectedRepeats = VERIFY_REPEATS,
) {
  if (!(ASM_MODE || IR_MODE)) return ver;
  const normalized = authoritativeGeneratedPersistentVerification(ver, {
    sessionPrefix,
    expectedRepeats,
    workspace,
    languageMode: LANGUAGE_MODE,
    archiveDir: `${AVO_STATE}/screening_evidence`,
  });
  if (normalized?.evidence_rejection) {
    log(`Screening evidence rejected (${sessionPrefix}): ${normalized.evidence_rejection}; `
      + `preserved=${normalized.screening_evidence_dir || 'unavailable'}`);
  }
  return normalized;
}

// Deterministic commit gate — single source of truth in lib/commit_gate.mjs
// (imported at the top of this file; unit-tested by tests/commit_gate.test.mjs).
const GATE_OPTS = {
  epsilon: COMMIT_EPSILON,
  minCommitSpeedup: MIN_COMMIT_SPEEDUP,
  significanceMargin: SIGNIFICANCE_MARGIN,
  noiseK: COMMIT_NOISE_K,
  requireCvForMicroGain: true,
  microGainCeiling: parseFloat(A.micro_gain_ceiling != null ? A.micro_gain_ceiling : 1.02),
  finalReproductionTolerance: parseFloat(
    A.final_reproduction_tolerance != null ? A.final_reproduction_tolerance
      : (ASM_MODE ? 0.03 : 0.10)),
};
const commitScore = (ver) => verifiedCommitScore(ver, COMMIT_SCORE_FIELD);
const scoreForGate = (ver) => {
  const selected = withCommitScore(ver, COMMIT_SCORE_FIELD);
  if (!selected || !Number.isFinite(COMMIT_CALIBRATED_MARGIN_CAP)
      || COMMIT_CALIBRATED_MARGIN_CAP < 0) return selected;
  const reported = Number(selected.calibrated_margin);
  if (!Number.isFinite(reported) || reported <= COMMIT_CALIBRATED_MARGIN_CAP) return selected;
  return { ...selected, calibrated_margin: COMMIT_CALIBRATED_MARGIN_CAP,
    reported_calibrated_margin: reported };
};
const passesCommitGate = (ver, cumulative) => _gate(scoreForGate(ver), cumulative, GATE_OPTS);
const passesFormalCommitGate = (ver, cumulative) => _gate(
  scoreForGate(ver), cumulative,
  { ...GATE_OPTS, requireFormalEvidence: FORMAL_EVIDENCE_MODE });
const evaluateFormalCommitGate = (ver, cumulative) => evaluateCommitGate(
  scoreForGate(ver), cumulative,
  { ...GATE_OPTS, requireFormalEvidence: FORMAL_EVIDENCE_MODE });

// Token-Jaccard on supervisor diagnoses (deterministic “same root cause” proxy).
function diagnosisTokens(s) {
  return new Set(String(s).toLowerCase().split(/[^a-z0-9_]+/).filter(w => w.length > 2));
}
function diagnosisTokenJaccard(a, b) {
  const ta = diagnosisTokens(a);
  const tb = diagnosisTokens(b);
  if (!ta.size && !tb.size) return 1;
  if (!ta.size || !tb.size) return 0;
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter++;
  const u = ta.size + tb.size - inter;
  return u ? inter / u : 0;
}

// Roofline summary from a per-case array. `latencyField` is the absolute-latency
// key to use as each case's runtime (optimized_ms for candidates, baseline_ms for
// the baseline). Returns ok:false when no case carries flops/bytes telemetry — the
// loop then runs in pure relative mode (roofline signals inert). Single source of
// truth: lib/roofline.mjs (mirrors scripts/avo_lineage.py).
function summarizeRoofline(perCase, latencyField) {
  const mapped = (perCase || []).map(c => ({
    name: c.name, dtype: c.dtype, flops: c.flops, bytes: c.bytes,
    latency_ms: c[latencyField] != null ? c[latencyField] : c.latency_ms,
  }));
  return rooflineSummary(mapped, ROOFLINE_OPTS);
}

// Compact steering block injected into the operator/supervisor prompts: where we
// are vs the ceiling, which constraint binds, and the ranked levers for it.
function steerFrom(summary) {
  const regime = summary && summary.ok ? summary.dominant_regime : 'unknown';
  // IR enrichment (Phase 1): turn the regime into specific source levers from the
  // current best's IR. Inert ({ok:false}) when the switch is off or no IR exists.
  const ir = IR_FEEDBACK ? irSteer(regime, bestIr) : { ok: false, hints: [] };
  if (!summary || !summary.ok) {
    return { regime: 'unknown', fraction_of_roofline: null, worst_case: null,
      specialty: 'algorithm', levers: recommendLevers('unknown').levers, ir };
  }
  const rec = recommendLevers(summary.dominant_regime);
  const focus = allocateAttention(summary.analyses).focus;
  return {
    regime: summary.dominant_regime,
    fraction_of_roofline: summary.geomean_fraction,
    worst_case: summary.worst_case, focus_case: focus,
    specialty: rec.specialty, levers: rec.levers, ir,
  };
}

/** Optional pre-verify: skip GPU verify only on high-confidence IR regression vs bestIr. */
function irPrescreenBlocks(irBase, irCand) {
  if (!IR_PRESCREEN || !IR_FEEDBACK) return null;
  if (!irBase || !irBase.ok || !irCand || !irCand.ok) return null;
  const pr = predictRegression(irBase, irCand);
  if (pr.regress && pr.confidence === 'high') return pr;
  return null;
}

// ===========================================================================
// PHASE: Setup — isolated eval dir + the avo_repo lineage clone.
// ===========================================================================
phase('Setup');
const setup = await agent(
  roleAgent('director', 'setup_native', 'Build the isolated eval dir and the avo_repo lineage clone (original is READ-ONLY).', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    TARGET_LANGUAGE: TARGET_LANGUAGE_NORM,
    KERNEL_LANGUAGE: A.kernel_language || '',
    ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR, LANGUAGE_MODE, LANGUAGE_SKILL_PATH, LANGUAGE_CONTRACT,
    IR_LANG,
    IR_ARCH: A.arch || A.hw || HW,
    IR_SHAPES, ASM_PACK, ASM_ARCH, ASM_SHAPES,
    ASM_WINDOW_KERNEL, ASM_WINDOW_BEGIN, ASM_WINDOW_END, GEAK_CAPTURE_DIR,
    IR_CAPTURE_MODE, IR_CAPTURE_PATH, IR_CAPTURE_KERNEL,
    IR_CAPTURE_COMMAND_JSON, IR_CAPTURE_DISPATCH, IR_CAPTURE_TIMEOUT,
    GPU_ID: GPU_LIST[0], GPU_POOL,
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir || !setup.avo_repo) throw new Error('Setup failed: director did not return eval_dir/avo_repo');
const EVAL_DIR = setup.eval_dir;
const AVO_REPO = setup.avo_repo;          // git repo; HEAD == current best; tags avo-v{N}
const KERNEL_NAME = setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
const AVO_STATE = `${EVAL_DIR}/avo_state`;
const RESUME = !!setup.resume;
// IR is strict formal mode on both fresh runs and resumes. Resolve the durable
// setup-time capture from committed metadata rather than trusting a caller's
// temporary provider/source path, and never downgrade to repeated evidence.
if (IR_MODE) {
  let persistedMeta;
  try {
    const { readFileSync } = await import('node:fs');
    persistedMeta = JSON.parse(readFileSync(`${AVO_REPO}/meta.json`, 'utf8'));
  } catch (e) {
    throw new Error(`${RESUME ? 'Resume' : 'Setup'} failed: IR metadata is unreadable (${
      String(e && e.message ? e.message : e)})`);
  }
  const persistedCapture = String(persistedMeta.capture || '');
  if (String(persistedMeta.oracle || '').toLowerCase() !== 'kerncap'
      || !persistedCapture) {
    throw new Error(`${RESUME ? 'Resume' : 'Setup'} failed: strict IR formal metadata lacks a kerncap capture`);
  }
  const { statSync } = await import('node:fs');
  try {
    if (!statSync(persistedCapture).isDirectory()) throw new Error('not a directory');
  } catch {
    throw new Error(`${RESUME ? 'Resume' : 'Setup'} failed: persisted formal IR capture is unavailable: ${persistedCapture}`);
  }
  GEAK_CAPTURE_DIR = persistedCapture;
  FORMAL_EVIDENCE_MODE = true;
  PRECOMMIT_DIRECTOR_VALIDATE = formalPrecommitEnabled(
    FORMAL_EVIDENCE_MODE, REQUESTED_PRECOMMIT_DIRECTOR_VALIDATE);
  log(`Evidence policy: restored captured IR formal mode from ${AVO_REPO}/meta.json`);
}
// On resume, the loop state comes from `avo_lineage.py resume`, which recomputes it
// from lineage.json + baseline.json. The Director reports the same fields in its setup
// JSON, but those are LLM-transcribed: a mistyped `cumulative` becomes the commit-gate
// base, so a regression can be committed as a new best and the whole lineage inherits
// the wrong number. Deterministic values win over the transcription for every field the
// script can derive; the Director's copy is only a fallback for when the script fails.
if (RESUME) {
  try {
    // run_avo.mjs injects this file via AsyncFunction — only relative `./lib/*`
    // imports are rewritten; use dynamic import for Node builtins.
    const { execFileSync } = await import('node:child_process');
    const out = execFileSync('python3', [
      `${WORKFLOW_DIR}/scripts/avo_lineage.py`, 'resume', '--state', AVO_STATE,
    ], { encoding: 'utf8', maxBuffer: 8 << 20 });
    const rj = JSON.parse(out.trim());
    if (rj.resume === true) {
      const corrections = [];
      const adopt = (key, valid) => {
        const next = rj[key];
        if (!valid(next)) return;
        const prev = setup[key];
        setup[key] = next;
        const differs = Array.isArray(next) || (next && typeof next === 'object')
          ? JSON.stringify(prev) !== JSON.stringify(next)
          : prev !== next;
        if (differs) corrections.push(`${key} ${JSON.stringify(prev) ?? 'unset'} -> ${JSON.stringify(next)}`);
      };
      const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
      const isArr = (v) => Array.isArray(v);
      // Gate-critical: these decide what a candidate must beat to commit.
      adopt('cumulative', (v) => isNum(v) && v > 0);
      adopt('best_version', (v) => typeof v === 'string' && v.length > 0);
      adopt('commits', (v) => isNum(v) && v >= 0);
      adopt('committed', (v) => isArr(v) && v.length > 0);
      adopt('baseline_per_case', (v) => isArr(v) && v.length > 0);
      adopt('baseline_geomean_ms', (v) => isNum(v) && v > 0);
      // Steering + memory: durable supervisor/ledger state, not the LLM's recollection.
      adopt('roofline_fraction', isNum);
      adopt('roofline_regime', (v) => typeof v === 'string' && v.length > 0);
      adopt('failed', isArr);
      adopt('insights', isArr);
      // Director-only `wc -l` on variation_* inflated team-mode dirs (step 60 -> 174).
      adopt('prior_steps', (v) => isNum(v) && v >= 0);
      adopt('best_ir', (v) => v != null && typeof v === 'object');
      log(`Resume: state rebuilt from avo_lineage.py (best=${setup.best_version} ${
        Number(setup.cumulative).toFixed(4)}x, prior_steps=${setup.prior_steps})`);
      for (const c of corrections) log(`Resume: corrected ${c} (avo_lineage.py is authoritative)`);
    }
  } catch (e) {
    log(`Resume: deterministic state refresh failed (${String(e && e.message ? e.message : e)}); falling back to Director-reported setup values`);
  }
}
log(`Setup done. EVAL_DIR=${EVAL_DIR} AVO_REPO=${AVO_REPO}${RESUME ? ' (RESUME)' : ''}`);
log(`Commit gate: score=${COMMIT_SCORE_FIELD}, significanceMargin=${SIGNIFICANCE_MARGIN} `
  + `(${ASM_MODE ? 'ASM micro-opt default 0.2%' : 'source default 0.2%'}), `
  + `noiseK=${COMMIT_NOISE_K}, minCommitSpeedup=${MIN_COMMIT_SPEEDUP}, epsilon=${COMMIT_EPSILON}, `
  + `calibratedMarginCap=${Number.isFinite(COMMIT_CALIBRATED_MARGIN_CAP)
    ? COMMIT_CALIBRATED_MARGIN_CAP : 'none'}`);
if (RUNTIME_RECORDS_ATTEMPTS) log('Attempt ledger: runtime backstop ON (every attempt recorded deterministically).');

// Deterministically append ONE attempt to attempts.jsonl when the runtime supports
// it. `committed:false` always — a committed candidate is ALSO recorded with
// committed:true by avo_lineage.py commit (the attempt + the commit are both true
// facts). Inert (and the role self-records) when the primitive is absent.
function logAttempt(rec) {
  if (!RUNTIME_RECORDS_ATTEMPTS) return;
  try { recordAttempt(AVO_STATE, { committed: false, ...rec }); } catch { /* best-effort */ }
}

// Compact IR summary for the attempt ledger (full vector lives on the lineage node).
// Keeps attempts.jsonl small while making the IR trajectory durable + greppable.
function irBrief(ir) {
  if (!IR_FEEDBACK || !ir || typeof ir !== 'object') return undefined;
  return { ok: ir.ok === true, tier: ir.tier, metrics: ir.metrics || {} };
}

// Reclaim every scratch worktree created under this step's dirs (single-mode stepDir
// + its verify/, team workers variation_<step>_<i>/, integrate, and escalate_<step>).
// Deterministic backstop run in the per-step finally so a role that died/errored
// before its own `git worktree remove` cannot leak. Inert without the shim primitive.
function cleanupStepWorktrees() {
  if (!RUNTIME_PRUNES_WORKTREES) return;
  try {
    // all-mode: the evolve loop is strictly sequential, so at a step's teardown EVERY
    // non-main worktree is this-step garbage — including any an LLM mis-placed at a
    // wrong path (git registers them by absolute path, so they are caught anywhere).
    const n = pruneWorktrees(AVO_REPO, null, { all: true });
    if (n) log(`Step ${step}: reclaimed ${n} leaked worktree(s) (deterministic teardown).`);
  } catch { /* best-effort */ }
}

// ---------------------------------------------------------------------------
// Loop state. On a fresh run it is seeded by Benchmark/Seed below; on RESUME it
// is reconstructed from durable state (lineage.json / baseline.json / supervisor_log).
// ---------------------------------------------------------------------------
let BASELINE_PER_CASE, BASELINE_GEOMEAN_MS;
let commits = 0;
let cumulative = 1.0;                 // verified geomean speedup of the best vs TRUE baseline
let bestVersion = 'avo-v0';
// Optional one-round exploration base selected by supervisor backtrack. This is
// deliberately separate from bestVersion/cumulative, which always describe the
// global champion and therefore remain the commit-gate floor.
let explorationStartRef = null;
let committed = [{ id: 'v0', ref: 'avo-v0', speedup: 1.0, strategy: 'baseline' }];
let direction = { name: '', expected: '', prompt: '' };
const history = { insights: [], failed: [], ledger: [] };
// A direction is RETIRED only after repeated failure, and the memory decays.
// "measured but below the commit gate" is the normal outcome of most attempts, so
// retiring on the first occurrence drains the TechLead's plan within a few rounds
// until every team round degenerates into one generic escape worker. The list is
// also embedded verbatim into prompts, so it must stay bounded.
const DIRECTION_RETIRE_AFTER = 2;
const DIRECTION_MEMORY_STEPS = 12;
const MAX_FAILED_DIRECTIONS = 60;
const directionFailures = new Map();
function rememberFailedDirection(name, reason = '', { hard = false } = {}) {
  const value = String(name || '').trim();
  if (!value) return;
  const prior = directionFailures.get(value);
  const carried = prior && (step - prior.lastStep) <= DIRECTION_MEMORY_STEPS
    ? prior.count : 0;
  // "Repetition" means a direction failed again in a LATER round. A team round can
  // hand the same direction to several workers (pairStartsWithDirections cycles
  // dirs when there are more start points than directions) and free-text strategy
  // labels collide across workers outright, so counting each worker separately
  // would retire the direction within its very first round.
  const repeatedWithinStep = !!prior && prior.lastStep === step;
  const increment = hard ? DIRECTION_RETIRE_AFTER : (repeatedWithinStep ? 0 : 1);
  const count = carried + increment;
  directionFailures.set(value, { count, lastStep: step });
  if (count < DIRECTION_RETIRE_AFTER || history.failed.includes(value)) return;
  history.failed.push(value);
  while (history.failed.length > MAX_FAILED_DIRECTIONS) history.failed.shift();
  log(`Direction retired after ${count} failures: ${value}${reason ? ` (${reason})` : ''}`);
}
// Two different failures wear the same "infrastructure_unhealthy" label, and
// they need different responses:
//
//   rig failure      — the harness itself could not measure (broken env, dead
//                      GPU, build failure). Repeating is pointless, so a couple
//                      in a row should abort the run.
//   evidence rejected — the measurement may well have run, but the worker handed
//                      back an unusable report set (wrong session id, wrong
//                      repeat count, artifacts changed mid-session). That is
//                      usually the worker's mistake and the next worker can get
//                      it right, so it must NOT trip the low abort threshold.
//   agent outage     — the agent itself returned nothing, so we learned nothing
//                      about the rig either way. Usually the backend was
//                      unreachable and the worker never ran a measurement at all.
//
// All three still count toward an abort, on separate budgets, so none can silently
// consume the whole run.
//
// Charging an outage to the rig budget aborted a 19.5h run: an API blip returned
// four null workers, which read as "every worker hit a broken harness" and tripped
// the deliberately-low rig threshold on evidence that never existed.
function isAgentOutage(variation) {
  return !variation;
}
// In fan-out a variation can return successfully while its independent verifier
// returns nothing. That is still an agent outage: no measurement verdict exists.
// Verification is not expected for rejected/no-change candidates, budget expiry,
// or deterministic IR pre-screen skips, so those must not consume the outage budget.
function isFanOutAgentOutage(result) {
  if (isAgentOutage(result?.rv)) return true;
  if (result?.dedupeSkipped) return false;
  const verificationExpected = String(result.rv?.status || '').toLowerCase() === 'ok'
    && !!result.rv?.patch_file && !result.budgetExpired && !result.irPrescreen;
  return verificationExpected && (!result.ver || result.ver.agent_outage === true);
}
function isRigFailure(variation, verification) {
  if (!variation) return false;
  if (String(variation.status || '').toLowerCase() === 'infra_failure') return true;
  if (!verification) return false;
  // After a rejection the authoritative check overwrites infrastructure_status,
  // so consult what the harness reported before that.
  const raw = String(verification.evidence_rejection
    ? (verification.reported_infrastructure_status || '')
    : (verification.infrastructure_status || '')).toLowerCase();
  return !!raw && raw !== 'healthy' && raw !== 'unreported';
}
function isEvidenceRejection(verification) {
  return !!(verification && verification.evidence_rejection);
}
// The capture is shared by every worker, so if it drifts nobody can produce valid
// evidence — including workers whose evidence was already banked. That is a
// run-level fault, not a worker's bad report, and charging it to the per-round
// evidence budget spends hours of GPU time to conclude "something moved".
function isCaptureDrift(verification) {
  return !!(verification && verification.evidence_rejection
    && String(verification.evidence_rejection_component || '') === 'capture');
}
function isUnusableMeasurement(variation, verification) {
  return isAgentOutage(variation) || isRigFailure(variation, verification)
    || isEvidenceRejection(verification);
}
function recordDirectionRepairs(guarded, context) {
  for (const repair of guarded.repairs || []) {
    const message = `[language-guard:${context}] ${repair}`;
    history.insights.push(message);
    log(message);
  }
}
// Roofline summary of the CURRENT best (drives steering + the early-stop decision).
let bestRoofline = { ok: false, geomean_fraction: null, dominant_regime: 'unknown', analyses: [] };
// IR metric vector of the CURRENT best (read-only steering evidence; inert unless
// IR_FEEDBACK is on and a worker produced one). Phase-1: threaded into the steer block.
let bestIr = { ok: false };

if (RESUME) {
  BASELINE_PER_CASE = setup.baseline_per_case || [];
  BASELINE_GEOMEAN_MS = setup.baseline_geomean_ms;
  commits = setup.commits || 0;
  cumulative = setup.cumulative || 1.0;
  bestVersion = setup.best_version || 'avo-v0';
  if (Array.isArray(setup.committed) && setup.committed.length) committed = setup.committed;
  history.failed = setup.failed || [];
  history.insights = setup.insights || [];
  if (setup.direction) direction = setup.direction;
  if (!BASELINE_PER_CASE.length) throw new Error('Resume failed: no baseline_per_case in avo_state/baseline.json');
  // Restore the best's roofline standing (from lineage.json via the director).
  if (setup.roofline_fraction != null) {
    bestRoofline = { ok: true, geomean_fraction: setup.roofline_fraction,
      dominant_regime: setup.roofline_regime || 'unknown', worst_case: null, analyses: [] };
  }
  if (setup.best_ir != null && typeof setup.best_ir === 'object') {
    bestIr = setup.best_ir;
    if (IR_FEEDBACK) log(`Resume: best_ir restored (ok=${bestIr.ok === true})`);
  }
  log(`RESUME: best=${bestVersion} (${cumulative.toFixed(4)}x), ${commits} prior commits, ${history.failed.length} failed dirs.`);
} else {
  // ----- PHASE: Benchmark — COMMANDMENT + TRUE baseline (reuse team's role).
  phase('Benchmark');
  const benchReportPath = `${EVAL_DIR}/baseline_report.json`;
  const benchReply = await agent(
    roleAgentFrom(KERNEL_WORKFLOW_DIR, 'benchmark_engineer', 'setup',
      'Build the COMMANDMENT and record a reliable baseline for the avo_repo.', {
      WORKSPACE: AVO_REPO, EVAL_DIR, SKILL_DIR: KERNEL_WORKFLOW_DIR,
      GPU_ID: GPU_POOL, GPU_POOL,
      KERNEL_ENV_SCRIPT, ANALYSIS: { kernel_name: KERNEL_NAME },
      TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH,
      LANGUAGE_ENGINEER_ADDENDUM, LANGUAGE_CONTRACT, ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
      REPORT_PATH: benchReportPath,
    }),
    { phase: 'Benchmark', label: 'benchmark_engineer', schema: BENCH_SCHEMA });
  const bench = benchReply && readAgentReport(benchReportPath);
  if (!bench || !bench.baseline_per_case) throw new Error('Benchmark setup failed: no baseline recorded');
  BASELINE_PER_CASE = bench.baseline_per_case;
  BASELINE_GEOMEAN_MS = bench.baseline_geomean_ms;
  bestRoofline = summarizeRoofline(BASELINE_PER_CASE, 'baseline_ms');   // baseline standing vs ceiling
  if (bestRoofline.ok) log(`Baseline roofline: ${(bestRoofline.geomean_fraction * 100).toFixed(1)}% of peak, regime=${bestRoofline.dominant_regime}`);
  if (bestRoofline.n_invalid) log(`WARNING: ${bestRoofline.n_invalid} baseline case(s) report fraction_of_roofline > 1 (untrustworthy flops/bytes) — excluded from the roofline signal; fix the telemetry recipe.`);
  log(`Benchmark done. ${bench.num_test_cases || BASELINE_PER_CASE.length} cases, baseline geomean ${BASELINE_GEOMEAN_MS} ms, reliable=${bench.reliable}`);

  // ----- PHASE: Seed — commit baseline as avo-v0 (the lineage base).
  phase('Seed');
  const seed = await agent(
    roleAgent('lineage_engineer', 'seed', 'Commit the current avo_repo state as the baseline and tag it avo-v0.', {
      AVO_REPO, AVO_STATE, BASELINE_PER_CASE, BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, HW,
    }),
    { phase: 'Seed', label: 'lineage_engineer:seed', schema: SEED_SCHEMA });
  if (!seed || !seed.seeded) throw new Error('Seed failed: baseline not tagged avo-v0');
  log('Seed done. baseline tagged avo-v0.');
}

if (ASM_MODE) {
  const guarded = constrainLanguageDirections([direction], {
    languageMode: LANGUAGE_MODE, count: 1, step: 0, failed: history.failed,
  });
  recordDirectionRepairs(guarded, 'initial');
  [direction] = guarded.directions;
}

// ===========================================================================
// PHASE: Profile — classify the CURRENT-best bottleneck (fresh: baseline;
// resume: the resumed best). Optional.
// ===========================================================================
let profileSummary = null;
if (USE_PROFILING) {
  phase('Profile');
  profileSummary = await agent(
    roleAgentFrom(KERNEL_WORKFLOW_DIR, 'profile_engineer', 'baseline',
      'Profile the current best (HEAD of avo_repo) and classify the bottleneck.', {
      WORKSPACE: AVO_REPO, EVAL_DIR, SKILL_DIR: KERNEL_WORKFLOW_DIR,
      GPU_ID: GPU_POOL, GPU_POOL, ROUND: 0, COMMANDMENT,
      KERNEL_ENV_SCRIPT, IR_FEEDBACK,
      TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH,
      LANGUAGE_ENGINEER_ADDENDUM, LANGUAGE_CONTRACT, ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
    }),
    { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
  log(`Bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'}`);
}

// ===========================================================================
// PHASE: Evolve — wall-clock single-lineage loop.
// ===========================================================================
phase('Evolve');
const DEADLINE = CAMPAIGN_DEADLINE - FINAL_PHASE_RESERVE_S * 1000;
log(`Campaign budget=${BUDGET_S}s, final-phase reserve=${FINAL_PHASE_RESERVE_S}s, `
  + `evolve remaining=${Math.max(0, Math.round((DEADLINE - Date.now()) / 1000))}s, `
  + `maxSteps=${MAX_STEPS} — ${EARLY_STOP ? 'early_stop ON: convergence + roofline early-stops ENABLED' : 'early_stop OFF (default): run to the available evolve budget; convergence + roofline early-stops DISABLED'}`);
let step = RESUME ? (setup.prior_steps || 0) : 0;   // continue numbering so step dirs don't collide
let bestPerCase = BASELINE_PER_CASE;
let noImprove = 0;
let supervisorCycles = 0;
let infraFailures = 0;
// Unusable evidence gets a looser budget than a broken rig: a worker can hand
// back a bad report set and the next one still succeed, whereas a dead harness
// stays dead. Both are bounded so neither can quietly consume the whole run.
let evidenceRejections = 0;
const MAX_EVIDENCE_REJECTIONS = Math.max(
  4, (MAX_INFRA_FAILURES > 0 ? MAX_INFRA_FAILURES : 2) * 3);
// An unreachable agent backend is transient far more often than a dead rig, and the
// shim already backs off across several minutes before it gives up. Give it the
// loosest budget of the three so a maintenance window costs rounds, not the run.
let agentOutages = 0;
const MAX_AGENT_OUTAGES = parseInt(
  A.max_agent_outages != null ? A.max_agent_outages : 5, 10);
function countAgentOutage(reason) {
  agentOutages++;
  log(`Step ${step}: AGENT OUTAGE (${agentOutages}/${MAX_AGENT_OUTAGES}) — ${reason}`);
  if (MAX_AGENT_OUTAGES > 0 && agentOutages >= MAX_AGENT_OUTAGES) {
    log('Aborting: the agent backend stayed unreachable across rounds; no further work is possible.');
    return true;
  }
  return false;
}
function countRigFailure(reason) {
  infraFailures++;
  log(`Step ${step}: INFRA FAILURE (${infraFailures}/${MAX_INFRA_FAILURES}) — ${reason}`);
  if (MAX_INFRA_FAILURES > 0 && infraFailures >= MAX_INFRA_FAILURES) {
    log('Aborting: persistent infra failure (harness/env cannot run); no commit is possible.');
    return true;
  }
  return false;
}
function countEvidenceRejection(reason) {
  evidenceRejections++;
  log(`Step ${step}: EVIDENCE REJECTED (${evidenceRejections}/${MAX_EVIDENCE_REJECTIONS}) — ${reason}`);
  if (evidenceRejections >= MAX_EVIDENCE_REJECTIONS) {
    log('Aborting: workers keep returning unusable evidence; no commit is possible.');
    return true;
  }
  return false;
}
// Team rounds and ESCALATE use the same fan-out engine, so they must apply the
// same run-level health policy. Returning a disposition keeps break/continue in
// the caller's wall-clock loop while centralizing counters and capture-drift logic.
function handleFanOutRoundHealth(round, mode) {
  if (round.captureDrift) {
    log('Aborting: the formal capture changed during the run, so no worker can '
      + `produce valid evidence — ${round.captureDrift}`);
    history.ledger.push({ step, mode, committed: false,
      reason: 'capture_changed_mid_run' });
    return 'abort';
  }
  if (round.agentOutageCount > 0) {
    const abort = countAgentOutage(
      `${round.agentOutageCount} ${mode} verifier worker(s) timed out or returned no contract`);
    if (abort) return 'abort';
  }
  if (round.allUnusable) {
    const abort = round.allAgentOutage
      ? false
      : (round.allRigFailed
        ? countRigFailure(`all ${mode} workers hit a broken harness/env`)
        : countEvidenceRejection(`no ${mode} worker returned usable evidence`));
    return abort ? 'abort' : 'retry';
  }
  infraFailures = 0;
  evidenceRejections = 0;
  if (!round.agentOutageCount) agentOutages = 0;
  return 'usable';
}
const recentHashes = [];
const stepDurationsMs = [];   // wall time of recent completed steps (budget-aware supervisor)
let redirectsSinceCommit = 0;
let lastSupervisorDiagnosis = '';
let dxDupStreak = 0;

function geomeanLatencyMs(perCase) {
  const xs = (perCase || []).map((c) => {
    const t = c.optimized_ms != null ? c.optimized_ms : c.baseline_ms;
    return Number(t);
  }).filter((t) => Number.isFinite(t) && t > 0);
  if (!xs.length) return null;
  return Math.exp(xs.reduce((a, t) => a + Math.log(t), 0) / xs.length);
}

/** Authoritative `heartbeat.json` when runtime injects `syncAvoHeartbeat` (aligns `step` with the loop). */
function pulseOrchestratorHeartbeat(phase) {
  if (!RUNTIME_SYNCS_HEARTBEAT) return;
  try {
    syncAvoHeartbeat(AVO_STATE, {
      phase,
      step,
      commits,
      best_version: bestVersion,
      best_speedup: cumulative,
      best_latency_ms: geomeanLatencyMs(bestPerCase),
    });
  } catch { /* best-effort */ }
}

function remainingS() { return Math.max(0, Math.round((DEADLINE - Date.now()) / 1000)); }

// Estimated wall-clock seconds for one evolve step, from the trailing average of recent
// steps (0 when nothing has completed yet → guard is inert on the very first iteration).
function estStepS() {
  if (!stepDurationsMs.length) return 0;
  return Math.round(stepDurationsMs.reduce((a, b) => a + b, 0) / stepDurationsMs.length / 1000);
}

// Current wall-clock budget, in the shape lib/stop_gate.mjs consumes.
function budgetNow() {
  return { remainingS: remainingS(), estStepS: estStepS(), overheadS: SUPERVISOR_OVERHEAD_S };
}

// Optimization stage (§3.2 / delayed profiling): structural-first until the kernel
// structure is sound (past PROFILING_AFTER_STEP OR a real commit landed), then
// profiling-guided. Forced 'structural' when profiling is globally off.
function currentStage() {
  if (!USE_PROFILING) return 'structural';
  return (step > PROFILING_AFTER_STEP || commits > 0) ? 'profiling-guided' : 'structural';
}

// Re-profile the new best after a commit (§3.2: the bottleneck moves as the kernel
// evolves). Updates profileSummary so later steps target the *current* bottleneck.
async function reprofileBest(roundNo) {
  if (!USE_PROFILING) return;
  try {
    profileSummary = await agent(
      roleAgentFrom(KERNEL_WORKFLOW_DIR, 'profile_engineer', 'baseline',
        'Re-profile the new current-best (HEAD of avo_repo) and classify the (possibly shifted) bottleneck.', {
        WORKSPACE: AVO_REPO, EVAL_DIR, SKILL_DIR: KERNEL_WORKFLOW_DIR, GPU_ID: GPU_POOL, GPU_POOL,
        ROUND: roundNo, COMMANDMENT, KERNEL_ENV_SCRIPT, IR_FEEDBACK,
        TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH,
        LANGUAGE_ENGINEER_ADDENDUM, LANGUAGE_CONTRACT, ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
      }),
      { phase: 'Profile', label: `profile_engineer:reprofile r${roundNo}`, schema: PROFILE_SCHEMA });
    log(`Re-profiled best: bottleneck=${profileSummary ? profileSummary.bottleneck : '?'}`);
  } catch (e) { log(`Re-profile skipped (non-fatal): ${e}`); }
}

// Measured results come from the report file the agent writes with the code that
// measured them, not from its final reply: a long per_case list can come back
// retyped as a summary string. A missing report means the agent produced no result.
function readAgentReport(path) {
  return reportExists(path) ? JSON.parse(readPatchFileSync(path, 'utf8')) : null;
}

function directorValidationToVerify(validation) {
  return normalizeDirectorValidation(validation);
}

async function ingestFormalDirectorReport(validation, label = '', sessionId = '') {
  if (!FORMAL_EVIDENCE_MODE) return validation;
  const suffix = label ? `_${label}` : '';
  const reportPath = `${EVAL_DIR}/validation_workspace${suffix}/realcap_paired_result.json`;
  const { readFileSync } = await import('node:fs');
  const report = JSON.parse(readFileSync(reportPath, 'utf8'));
  report.artifact = verifyFormalArtifacts(
    report.artifact, reportPath,
    report.correctness?.validation_policy_sha256 || '');
  if (String(report.status || '').toLowerCase() === 'correctness_failed'
      && String(report.correctness?.status || '').toLowerCase() === 'fail') {
    if (!sessionId || String(report.session_id || '') !== String(sessionId)
        || report.correctness?.backend !== 'generated-persistent'
        || report.correctness?.method
          !== 'private_capture_view_baseline_candidate') {
      throw new Error('invalid formal correctness-failure evidence');
    }
    return {
      ...validation,
      correctness: 'fail',
      validation_status: 'correctness_failed',
      formal_status: 'correctness_failed',
      infrastructure_status: 'healthy',
      benchmark_report: reportPath,
      artifact: report.artifact,
    };
  }
  return authoritativeFormalValidation(validation, report, reportPath, sessionId);
}

async function precommitDirectorGate(cand, ver, N, parent) {
  if (!PRECOMMIT_DIRECTOR_VALIDATE) {
    if (FORMAL_EVIDENCE_MODE) {
      log(`PRECOMMIT DIRECTOR REQUIRED for formal avo-v${N}; screening evidence cannot commit`);
      return null;
    }
    return ver;
  }
  if (Date.now() >= DEADLINE) {
    log(`PRECOMMIT SKIPPED avo-v${N}: reserved final-phase window reached`);
    return null;
  }
  try {
    const reportPath = `${AVO_STATE}/director_precommit/v${N}.json`;
    const formalSessionId = `precommit_v${N}_${Date.now()}`;
    let validation = await agent(
      roleAgent('director', 'validate',
        'Pre-commit hard gate: synthesize the cumulative patch for this candidate, then independently re-benchmark it against the TRUE original baseline before lineage commit.', {
        KERNEL_PATH_ORIG, EVAL_DIR, KERNEL_NAME, SKILL_DIR: WORKFLOW_DIR,
        KERNEL_WORKFLOW_DIR, ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR, GPU_ID: GPU_POOL, GPU_POOL,
        TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH, LANGUAGE_CONTRACT,
        IR_LANG, IR_ARCH: A.arch || A.hw || HW, IR_SHAPES,
        ASM_PACK, ASM_ARCH, ASM_SHAPES,
        ASM_WINDOW_KERNEL, ASM_WINDOW_BEGIN, ASM_WINDOW_END, GEAK_CAPTURE_DIR,
        VERIFY_REPEATS: PRECOMMIT_DIRECTOR_REPEATS,
        BENCHMARK_PROTOCOL: FORMAL_EVIDENCE_MODE ? 'paired-abba' : 'repeated',
        PAIRED_BLOCKS: PRECOMMIT_PAIRED_BLOCKS,
        KERNEL_ENV_SCRIPT, APPLY_TO_ORIGINAL: 'false', FINAL_PATCH: '',
        REPORTED_GEOMEAN: ver.verified_geomean,
        REPORTED_WEIGHTED: ver.verified_weighted,
        SCORE_METRIC: COMMIT_SCORE_FIELD,
        CALIBRATED_MARGIN_CAP: COMMIT_CALIBRATED_MARGIN_CAP,
        COMMANDMENT,
        FORMAL_SESSION_ID: formalSessionId,
        AVO_REPO, PARENT_VERSION: parent, CANDIDATE_PATCH: cand.patch_file,
        VALIDATION_LABEL: `precommit_v${N}`, REPORT_PATH: reportPath,
      }),
      { phase: 'Validate', label: `director:precommit avo-v${N}`,
        schema: FORMAL_EVIDENCE_MODE ? FORMAL_VALIDATE_SCHEMA : VALIDATE_SCHEMA });
    if (!FORMAL_EVIDENCE_MODE) validation = validation && readAgentReport(reportPath);
    validation = await ingestFormalDirectorReport(
      validation, `precommit_v${N}`, formalSessionId);
    const directorVer = directorValidationToVerify(validation);
    const gateDecision = evaluateFormalCommitGate(directorVer, cumulative);
    if (!gateDecision.accepted) {
      const d = Number.isFinite(gateDecision.candidate)
        ? gateDecision.candidate.toFixed(4) : '0.0000';
      const threshold = Number.isFinite(gateDecision.threshold)
        ? ` threshold=${gateDecision.threshold.toFixed(4)}x` : '';
      log(`PRECOMMIT DIRECTOR REJECTED avo-v${N}: reason=${gateDecision.reason}; `
        + `director=${d}x current=${cumulative.toFixed(4)}x${threshold} `
        + `status=${validation ? validation.validation_status : 'null'} `
        + `formal_status=${validation ? validation.formal_status || 'missing' : 'null'} `
        + `correctness=${validation ? validation.correctness : 'unknown'}`);
      return null;
    }
    if (validation && validation.validation_status && validation.validation_status !== 'accepted') {
      log(`PRECOMMIT DIRECTOR FLAGGED avo-v${N}: ${validation.validation_status}; committing only director-measured ${directorVer.verified_geomean.toFixed(4)}x, not verifier-reported ${Number(ver.verified_geomean || 0).toFixed(4)}x`);
    }
    return directorVer;
  } catch (e) {
    log(`PRECOMMIT DIRECTOR failed for avo-v${N}; candidate not committed (${String(e && e.message ? e.message : e)})`);
    return null;
  }
}

// Commit a verified candidate through the lineage_engineer; advance lineage state.
// Returns true on success. Mutates the shared loop state (commits/cumulative/best...).
async function commitCandidate(cand, ver, parentStep, parentRef) {
  const N = commits + 1;
  const parent = parentRef || bestVersion;   // Diverse Fan-out: parent = the candidate's OWN start ref (tree edge)
  const globalBestBeforeCommit = bestVersion;
  const commitVer = await precommitDirectorGate(cand, ver, N, parent);
  if (!commitVer) return false;
  if (FORMAL_EVIDENCE_MODE && !passesFormalCommitGate(commitVer, cumulative)) {
    log(`COMMIT REJECTED avo-v${N}: candidate lacks accepted paired formal evidence`);
    return false;
  }
  const committedScore = commitScore(commitVer);
  let irFile = '';
  if (IR_FEEDBACK && cand.ir && cand.ir.ok === true) {
    try {
      const { writeFileSync, mkdirSync } = await import('node:fs');
      mkdirSync(AVO_STATE, { recursive: true });
      irFile = `${AVO_STATE}/._commit_ir_v${N}.json`;
      writeFileSync(irFile, JSON.stringify(cand.ir), 'utf8');
    } catch (e) {
      log(`commit: could not write IR sidecar (${String(e && e.message ? e.message : e)})`);
      irFile = '';
    }
  }
  let commit = null;
  let commitSideEffectClaimed = false;
  try {
    await checkoutAvoRepo(parent);
    commit = await agent(
      roleAgent('lineage_engineer', 'commit',
        'Apply the verified patch onto the candidate\'s checked-out parent version, commit, and tag with a rich message.', {
        AVO_REPO, AVO_STATE, PATCH: cand.patch_file, VERSION_N: N,
        VERIFIED_GEOMEAN: committedScore, VERIFIED_PER_CASE: commitVer.per_case,
        STRATEGY: cand.strategy || '', EXPECTED: cand.expected || '', RATIONALE: cand.rationale || '',
        PARENT_VERSION: parent, GLOBAL_BEST_REF: globalBestBeforeCommit,
        STEP: parentStep, HW, SKILL_DIR: WORKFLOW_DIR,
        IR_FILE: irFile,
      }),
      { phase: 'Evolve', label: `commit avo-v${N}`, schema: COMMIT_SCHEMA });
    if (commit && commit.committed) {
      commitSideEffectClaimed = true;
      await assertCommittedParent(commit.tag || `avo-v${N}`, parent);
    } else {
      await checkoutAvoRepo(globalBestBeforeCommit, true);
    }
  } catch (e) {
    log(`commit avo-v${N} failed parent/integrity checks: ${String(e)}`);
    try { await checkoutAvoRepo(globalBestBeforeCommit, true); } catch { /* fatal state logged above */ }
    if (commitSideEffectClaimed) {
      throw new Error(`fatal lineage integrity failure after commit side effects: ${String(e)}`);
    }
    return false;
  }
  if (commit && commit.committed) {
    commits = N; cumulative = committedScore;
    bestVersion = commit.tag || `avo-v${N}`; bestPerCase = commitVer.per_case || bestPerCase;
    explorationStartRef = null;
    // Roofline standing of the new best (regime feeds the MAP-Elites archive +
    // steering; geomean_fraction feeds the early-stop decision). Inert if no telemetry.
    const candRoofline = summarizeRoofline(commitVer.per_case, 'optimized_ms');
    if (candRoofline.ok) bestRoofline = candRoofline;
    // Capture the new best's IR vector for read-only steering of later steps (Phase 1).
    if (IR_FEEDBACK && cand.ir && cand.ir.ok) bestIr = cand.ir; else if (IR_FEEDBACK) bestIr = { ok: false };
    if (candRoofline.n_invalid) log(`WARNING: ${candRoofline.n_invalid} case(s) report fraction_of_roofline > 1 (untrustworthy flops/bytes) — excluded from the roofline signal.`);
    committed.push({ id: `v${N}`, ref: bestVersion,
      parent_id: String(parent).replace(/^avo-/, ''),
      speedup: committedScore,
      per_case: commitVer.per_case || [],
      formal: FORMAL_EVIDENCE_MODE ? {
        ci_lower: commitVer.ci_lower, ci_upper: commitVer.ci_upper,
        benchmark_report: commitVer.benchmark_report,
        artifact: commitVer.artifact || {},
      } : undefined,
      strategy: cand.strategy || '', regime: candRoofline.ok ? candRoofline.dominant_regime : undefined });
    noImprove = 0; supervisorCycles = 0; recentHashes.length = 0;
    redirectsSinceCommit = 0; lastSupervisorDiagnosis = ''; dxDupStreak = 0;
    const rlStr = candRoofline.ok ? ` ${(candRoofline.geomean_fraction * 100).toFixed(1)}%-roofline ${candRoofline.dominant_regime}` : '';
    log(`COMMITTED ${bestVersion} @ ${cumulative.toFixed(4)}x (parent=${parent}, strategy=${cand.strategy || '?'})${rlStr}`);
    await reprofileBest(parentStep);   // §3.2: bottleneck moves as the kernel evolves
    return true;
  }
  log(`commit failed to apply (${commit ? commit.notes : 'null'})`);
  return false;
}

function candidateIdentity(rv, workspace) {
  const reported = String(rv?.candidate_sha256 || '').trim().toLowerCase();
  if (/^[a-f0-9]{64}$/.test(reported)) return `sha256:${reported}`;
  for (const path of [`${workspace}/candidate.co`, rv?.patch_file]) {
    if (!path) continue;
    try {
      const digest = createHash('sha256').update(readPatchFileSync(path)).digest('hex');
      return `sha256:${digest}`;
    } catch { /* fall through to the next deterministic identity */ }
  }
  const patchHash = String(rv?.patch_hash || '').trim();
  return patchHash ? `patch:${patchHash}` : '';
}

// Run K directions in parallel (variation -> deduplicated staged verify), return the best
// candidate that passes the commit gate, else null. Reused by operator=team and
// ESCALATE. Each worker runs on its own avo_repo@best copy + its own GPU.
// pairs: [{ d (direction), startRef }]; each worker starts from its own startRef
// (Diverse Fan-out) and is verified on that same ref. Returns the best gated
// candidate (carrying its startRef so commit can parent off it → tree).
async function fanOutEvolve(pairs, tagPrefix) {
  const items = pairs.map((p, i) => {
    const guarded = constrainLanguageDirections([p.d], {
      languageMode: LANGUAGE_MODE, count: 1, step: step * 100 + i,
      failed: history.failed,
    });
    recordDirectionRepairs(guarded, `${tagPrefix}:${i}`);
    return { d: guarded.directions[0], startRef: p.startRef || bestVersion, i };
  });
  const variationResults = await pipeline(
    items,
    ({ d, startRef, i }) => {
      const stepDir = `${EVAL_DIR}/${tagPrefix}_${i}`;
      const variationWorkspace = `${stepDir}/workspace`;
      const workspace = makeWorkspace(variationWorkspace, startRef);
      return agent(
        roleAgent('variation_engineer', 'step',
          'One variation step under a distinct assigned direction, starting from BEST_VERSION (your assigned base).', {
          AVO_REPO, AVO_STATE, STEP_DIR: stepDir,
          WORKSPACE: workspace, BEST_VERSION: startRef,
          CUMULATIVE_SPEEDUP: cumulative, DIRECTION: d, GPU_ID: GPU_LIST[i % GPU_LIST.length],
          SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_ENV_SCRIPT, TASK, BASELINE_PER_CASE,
          PROFILE_SUMMARY: profileSummary ? profileSummary.summary_path : '', STAGE: currentStage(),
          KERNEL_KNOWLEDGE_DIR, USE_PROFILING, INSIGHTS: history.insights, FAILED_DIRECTIONS: history.failed,
          RECORD_ATTEMPT: !RUNTIME_RECORDS_ATTEMPTS,
          SYNC_HEARTBEAT: RUNTIME_SYNCS_HEARTBEAT,
          ROOFLINE: steerFrom(bestRoofline),
          IR_FEEDBACK, IR_LANG, IR_TIER, IR_KERNEL_NAME, IR_TRITON_CACHE,
          TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH,
          LANGUAGE_ENGINEER_ADDENDUM, LANGUAGE_EDIT_TARGET, LANGUAGE_CONTRACT,
          ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
          AUTOTUNE: { enabled: AUTOTUNE, max_configs: AUTOTUNE_MAX },
        }),
        { phase: 'Evolve', label: `${tagPrefix} ${i}:${d.name || d.specialty || ''}@${startRef}`, schema: VARIATION_SCHEMA }
      ).then(raw => {
        const rv = enforceWorkerLanguageScope(raw);
        return {
          d, startRef, i, rv, stepDir, variationWorkspace,
          candidateIdentity: candidateIdentity(rv, variationWorkspace),
        };
      });
    });

  const invalid = variationResults.filter(
    ({ rv }) => !rv || rv.status !== 'ok' || !rv.patch_file);
  const eligible = variationResults.filter(
    ({ rv }) => rv && rv.status === 'ok' && rv.patch_file);
  const verifyPlan = dedupeVerifyItems(eligible);
  for (const duplicate of verifyPlan.duplicates) {
    log(`verify ${tagPrefix} ${duplicate.i} skipped: candidate bytes duplicate worker ${
      duplicate.duplicateOf} on ${duplicate.startRef}`);
  }

  const runVerification = (item, repeats, phaseName, dirName, labelPrefix) => {
    const { d, startRef, i, rv } = item;
    if (Date.now() >= DEADLINE) {
      log(`verify ${tagPrefix} ${i} skipped: Evolve wall-clock budget exhausted`);
      return Promise.resolve({
        ...item, ver: null, budgetExpired: true,
      });
    }
    const prSkip = irPrescreenBlocks(bestIr, rv.ir);
    if (prSkip) {
      log(`ir_prescreen: worker ${i} skipped verify — ${prSkip.reasons.join('; ')}`);
      return Promise.resolve({
        ...item, ver: null, irPrescreen: prSkip,
      });
    }
    const verifyDir = `${EVAL_DIR}/${tagPrefix}_${i}/${dirName}`;
    const workspace = makeWorkspace(`${verifyDir}/workspace`, startRef);
    const effectiveWorkspace = workspace || `${verifyDir}/workspace`;
    if (workspace && rv.patch_file && !patchAppliesOnWorkspace(workspace, rv.patch_file)) {
      log(`verify ${tagPrefix} ${i} skipped: PATCH_DID_NOT_APPLY on ${startRef}`);
      return Promise.resolve({
        ...item,
        ver: {
          status: 'failed', correctness: 'fail', verified_geomean: 0,
          notes: 'PATCH_DID_NOT_APPLY',
        },
      });
    }
    const screeningSession = (ASM_MODE || IR_MODE)
      ? newPersistentScreeningSession() : '';
    const label = `${labelPrefix} ${tagPrefix} ${i}`;
    return agent(
      roleAgent(verifyRole, 'verify',
        `Independently ${phaseName} this candidate on its OWN base ref.`, {
        AVO_REPO, BASE_REF: startRef, PATCH: rv.patch_file, VERIFY_DIR: verifyDir,
        WORKSPACE: workspace,
        GPU_ID: GPU_LIST[i % GPU_LIST.length],
        KERNEL_WORKFLOW_DIR, COMMANDMENT, KERNEL_ENV_SCRIPT,
        BASELINE_PER_CASE, VERIFY_REPEATS: repeats, VERIFY_PHASE: phaseName,
        SKILL_DIR: WORKFLOW_DIR, TARGET_LANGUAGE: TARGET_LANGUAGE_NORM,
        LANGUAGE_MODE, LANGUAGE_SKILL_PATH, LANGUAGE_CONTRACT,
        ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
        ...(IR_MODE ? {
          GEAK_CAPTURE_DIR, GEAK_PERF_BACKEND: 'generated-persistent',
        } : {}),
        ...(ASM_MODE
          ? { ASM_SCREENING_SESSION_ID: screeningSession }
          : (IR_MODE
            ? { IR_SCREENING_SESSION_ID: screeningSession } : {})),
      }),
      { phase: 'Verify', label,
        schema: (ASM_MODE || IR_MODE) ? ASM_VERIFY_SCHEMA : VERIFY_SCHEMA }
    ).then(ver => ({
      ...item,
      ver: normalizePersistentScreening(
        ver && readAgentReport(`${verifyDir}/verify_report.json`),
        screeningSession, effectiveWorkspace, repeats),
      agentFailure: !ver && typeof agentFailure === 'function'
        ? agentFailure(label) : null,
    }));
  };

  let verified = [];
  const staged = (ASM_MODE || IR_MODE) && SCREENING_REPEATS < VERIFY_REPEATS;
  if (staged) {
    const screened = await pipeline(
      verifyPlan.unique,
      (item) => runVerification(
        item, SCREENING_REPEATS, 'screen', 'verify_screen', 'verify screen'));
    const finalist = selectScreeningFinalist(screened);
    let full = null;
    if (finalist) {
      log(`verify ${tagPrefix}: worker ${finalist.i} ranked first after ${
        SCREENING_REPEATS} repeats (${Number(finalist.ver.verified_geomean).toFixed(4)}x); `
        + `running full ${VERIFY_REPEATS}-repeat verification`);
      full = await runVerification(
        finalist, VERIFY_REPEATS, 'fully re-measure', 'verify', 'verify');
    }
    verified = screened.map((result) => (
      full && result.i === full.i
        ? { ...full, screeningVer: result.ver }
        : { ...result, screeningOnlyStage: true }
    ));
  } else {
    verified = await pipeline(
      verifyPlan.unique,
      (item) => runVerification(
        item, VERIFY_REPEATS, 're-measure', 'verify', 'verify'));
  }

  const results = [
    ...invalid.map((item) => ({ ...item, ver: null })),
    ...verified,
    ...verifyPlan.duplicates.map((item) => ({
      ...item, ver: null, dedupeSkipped: true,
    })),
  ];
  let best = null;
  // A round in which NO worker produced a usable measurement is not a round of
  // candidates that failed to improve. Counting only the variation side made the
  // abort unreachable in exactly the case it exists for: the harness cannot
  // measure, every candidate comes back rejected, and the run spends its whole
  // budget logging "no improvement" instead of reporting the real cause.
  const healthResults = results.filter((result) => !result.dedupeSkipped);
  let allRigFailed = healthResults.length > 0;
  let allAgentOutage = healthResults.length > 0;
  let allUnusable = healthResults.length > 0;
  let agentOutageCount = 0;
  let captureDrift = null;
  const passers = [];   // gated winners (for the Integrator merge of complementary regimes)
  for (const r of results) {
    const agentOutage = isFanOutAgentOutage(r);
    if (agentOutage) agentOutageCount++;
    if (!r.dedupeSkipped) {
      if (!isRigFailure(r.rv, r.ver)) allRigFailed = false;
      if (!agentOutage) allAgentOutage = false;
      if (!(agentOutage || isRigFailure(r.rv, r.ver)
          || isEvidenceRejection(r.ver))) allUnusable = false;
    }
    if (!captureDrift && isCaptureDrift(r.ver)) captureDrift = r.ver.evidence_rejection;
    // Durable trajectory: record EVERY worker attempt (passed or not), with the
    // facts the orchestrator already holds (rv/ver) — deterministic, no LLM step.
    logAttempt({
      step: parseInt(String(tagPrefix).replace(/\D+/g, ''), 10) || undefined,
      worker: r.i, strategy: (r.rv && r.rv.strategy) || (r.d && r.d.name) || null,
      status: r.irPrescreen ? 'ir_prescreen'
        : ((r.ver && r.ver.status === 'failed') ? 'verification_failed'
          : ((r.rv && r.rv.status) || 'null')),
      candidate_status: (r.rv && r.rv.status) || null,
      verification_status: (r.ver && r.ver.status) || null,
      self_speedup: r.rv && typeof r.rv.self_speedup === 'number' ? r.rv.self_speedup : null,
      verified_geomean: r.ver && typeof r.ver.verified_geomean === 'number' ? r.ver.verified_geomean : null,
      evidence_rejection: (r.ver && r.ver.evidence_rejection) || null,
      screening_evidence_dir: (r.ver && r.ver.screening_evidence_dir) || null,
      patch_hash: (r.rv && r.rv.patch_hash) || null,
      candidate_identity: r.candidateIdentity || null,
      duplicate_of_worker: r.duplicateOf ?? null,
      agent_failure: r.agentFailure || null,
      rationale: (r.rv && r.rv.rationale) ? String(r.rv.rationale).slice(0, 500) : null,
      ir: irBrief(r.rv && r.rv.ir),
    });
    const candidateStatus = String(r.rv?.status || '').toLowerCase();
    if (['failed', 'no_change', 'incorrect'].includes(candidateStatus)) {
      rememberFailedDirection(r.d?.name || r.rv?.strategy,
        `candidate ${candidateStatus}`, { hard: true });
      if (r.rv?.strategy && r.rv.strategy !== r.d?.name) {
        // Free-text strategy labels collide easily; retire them only on repetition.
        rememberFailedDirection(r.rv.strategy, `candidate ${candidateStatus}`);
      }
    } else if (r.ver?.correctness === 'fail' && !r.ver?.evidence_rejection) {
      rememberFailedDirection(r.d?.name || r.rv?.strategy, 'correctness failed',
        { hard: true });
    }
    const clearsGate = !!(!r.screeningOnlyStage && !r.dedupeSkipped
      && r.ver && passesCommitGate(r.ver, cumulative));
    if (clearsGate) {
      const rl = summarizeRoofline(r.ver.per_case, 'optimized_ms');
      passers.push({ rv: r.rv, ver: r.ver, startRef: r.startRef, d: r.d,
        regime: rl.ok ? rl.dominant_regime : undefined, speedup: commitScore(r.ver) });
      if (!best || commitScore(r.ver) > commitScore(best.ver)) best = r;
    } else if (!r.screeningOnlyStage
        && r.ver?.status === 'ok' && r.ver?.correctness === 'pass') {
      rememberFailedDirection(r.d?.name || r.rv?.strategy, 'verified below commit gate');
    }
  }
  return {
    best, passers, allRigFailed, allAgentOutage, allUnusable, captureDrift,
    agentOutageCount,
  };
}

// Commit a fan-out round's outcome (operator=team / ESCALATE). FIRST try an
// Integrator MERGE of complementary winners (P2): if >=2 gated winners share a base
// ref and attack different regimes, the avo_integrator composes them, the merge is
// INDEPENDENTLY re-verified on that base, and it is committed ONLY IF it passes the
// commit gate AND beats the best individual winner (else it is discarded — no fake
// progress). Otherwise (or on merge failure) commit the single best winner, exactly
// as before. Graceful: with <2 complementary winners the merge path never triggers.
async function commitRoundResult(round, parentStep, tagPrefix) {
  if (Date.now() >= DEADLINE) {
    log(`Step ${parentStep}: verified candidates were not committed because the `
      + 'reserved final-phase window has started.');
    history.ledger.push({ step: parentStep, committed: false,
      reason: 'verified_but_final_phase_reserved' });
    return false;
  }
  const mg = selectMergeGroup(round.passers || []);
  if (mg && mg.members.length >= 2) {
    const bestIndividual = Math.max(...mg.members.map(m => m.speedup || 0));
    const PATCHES = mg.members.map((m, j) => ({
      id: `w${j}`, regime: m.regime, strategy: m.rv.strategy || '',
      verified_geomean: commitScore(m.ver), patch_file: m.rv.patch_file,
    }));
    log(`Step ${parentStep}: Integrator merge of ${mg.members.length} complementary winners on ${mg.base} [${mg.members.map(m => m.regime).join('+')}]`);
    let merged = await agent(
      roleAgent('avo_integrator', 'integrate',
        'Merge these complementary verified winners onto their shared BASE_REF into one candidate.', {
        AVO_REPO, BASE_REF: mg.base, PATCHES, BEST_INDIVIDUAL: bestIndividual,
        INTEGRATE_DIR: `${EVAL_DIR}/${tagPrefix}_integrate`,
        WORKSPACE: makeWorkspace(`${EVAL_DIR}/${tagPrefix}_integrate/ws`, mg.base),
        GPU_ID: GPU_POOL, GPU_POOL,
        KERNEL_WORKFLOW_DIR, COMMANDMENT, KERNEL_ENV_SCRIPT, BASELINE_PER_CASE, VERIFY_REPEATS, INSIGHTS: history.insights,
        TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_CONTRACT,
        LANGUAGE_EDIT_TARGET,
      }),
      { phase: 'Evolve', label: `integrate ${tagPrefix}`, schema: INTEGRATE_SCHEMA,
        timeout_ms: Math.max(1000, DEADLINE - Date.now()) });
    merged = enforceWorkerLanguageScope(merged);
    if (Date.now() >= DEADLINE) {
      log(`Step ${parentStep}: integrator reached the final-phase reserve; `
        + 'discarding its output and preserving the reserve.');
      return false;
    }
    if (merged && merged.status === 'ok' && merged.patch_file) {
      const verifyDir = `${EVAL_DIR}/${tagPrefix}_integrate/verify`;
      const workspace = makeWorkspace(`${verifyDir}/workspace`, mg.base);
      const effectiveWorkspace = workspace || `${verifyDir}/workspace`;
      if (workspace && merged.patch_file
          && !patchAppliesOnWorkspace(workspace, merged.patch_file)) {
        log(`Step ${parentStep}: integrate verify skipped — PATCH_DID_NOT_APPLY on ${mg.base}`);
        return false;
      }
      const screeningSession = (ASM_MODE || IR_MODE)
        ? newPersistentScreeningSession() : '';
      const mver = await agent(
        roleAgent(verifyRole, 'verify',
          'Independently re-measure the integrated candidate on the shared base ref.', {
          AVO_REPO, BASE_REF: mg.base, PATCH: merged.patch_file,
          VERIFY_DIR: verifyDir, WORKSPACE: workspace, GPU_ID: GPU_POOL, GPU_POOL,
          KERNEL_WORKFLOW_DIR, COMMANDMENT, KERNEL_ENV_SCRIPT, BASELINE_PER_CASE, VERIFY_REPEATS,
          ...(IR_MODE ? {
            GEAK_CAPTURE_DIR, GEAK_PERF_BACKEND: 'generated-persistent',
          } : {}),
          ...(ASM_MODE
            ? { ASM_SCREENING_SESSION_ID: screeningSession }
            : (IR_MODE
              ? { IR_SCREENING_SESSION_ID: screeningSession } : {})),
        }),
        { phase: 'Verify', label: `verify integrate ${tagPrefix}`,
          schema: (ASM_MODE || IR_MODE) ? ASM_VERIFY_SCHEMA : VERIFY_SCHEMA,
          timeout_ms: Math.max(1000, DEADLINE - Date.now()) });
      const normalizedMver = normalizePersistentScreening(
        mver && readAgentReport(`${verifyDir}/verify_report.json`),
        screeningSession, effectiveWorkspace);
      if (Date.now() >= DEADLINE) {
        log(`Step ${parentStep}: integrated verification reached the final-phase `
          + 'reserve; no commit will be started.');
        return false;
      }
      if (passesCommitGate(normalizedMver, cumulative)
          && mergeImproves(commitScore(normalizedMver), mg.members)) {
        if (await commitCandidate({ patch_file: merged.patch_file,
          strategy: `integrated:${(merged.merged_regimes || mg.members.map(m => m.regime)).join('+')}`,
          expected: '', rationale: merged.rationale || 'integrator merge of complementary winners' },
          normalizedMver, parentStep, mg.base)) {
          log(`Integrator merge COMMITTED (beat best individual ${bestIndividual.toFixed(4)}x)`);
          return true;
        }
      } else {
        log(`Integrator merge did not beat best individual (${bestIndividual.toFixed(4)}x) — falling back to best winner`);
      }
    }
  }
  // Fallback: commit the single best gated winner (today's behavior).
  const b = round.best;
  if (b && await commitCandidate({ patch_file: b.rv.patch_file,
    strategy: b.rv.strategy || (b.d && b.d.name) || '', expected: (b.d && b.d.expected) || '',
    rationale: b.rv.rationale || '', ir: b.rv.ir }, b.ver, parentStep, b.startRef)) return true;
  return false;
}

// Build [{d, startRef}] pairs: K start points (per START_POLICY) × the planned
// directions, zipped (cycling whichever is shorter).
function pairStartsWithDirections(dirs, k) {
  let starts = selectStartPoints(committed, { policy: START_POLICY, k: Math.max(1, k) });
  if (explorationStartRef) {
    starts = [explorationStartRef, ...starts.filter((ref) => ref !== explorationStartRef)]
      .slice(0, Math.max(1, k));
    explorationStartRef = null;
  }
  const n = Math.max(starts.length, dirs.length);
  const pairs = [];
  for (let i = 0; i < n; i++) pairs.push({ d: dirs[i % dirs.length], startRef: starts[i % starts.length] });
  return pairs;
}

// Plan K orthogonal specialist directions via the reused Team tech_lead (operator=team).
// Roofline regime → specialist dispatch (P1): the deterministic SPECIALISTS order puts
// the dominant-ceiling specialist first and spreads the rest over the other ceilings;
// it is passed as a hint to the planner AND used to backfill any direction the planner
// left without a specialty, so the right specialist attacks the binding constraint.
async function planTeamDirections(roundNo) {
  const specialists = constrainSpecialists(assignSpecialists(
    bestRoofline.ok ? bestRoofline.dominant_regime : 'unknown',
    OPERATOR_DIRECTIONS), LANGUAGE_MODE);
  // TechLead's stop gate (tech_lead.md rule 1) is explicitly conditioned on whether budget
  // remains, so it has to be told the wall-clock budget. Passing only the per-round direction
  // cap left that clause unevaluable, and TechLead then judged convergence on its own
  // confidence alone -- ending 12h runs at round 7 with ~10h unspent.
  const budget = budgetNow();
  const affordable = affordableRounds(budget);
  const affordableText = affordable === null
    ? 'unknown (no round has completed yet)' : String(affordable);
  const stopPolicy = EARLY_STOP
    ? 'early_stop=ON -- a justified hard-gate stop=true ends the run immediately.'
    : 'early_stop=OFF -- this run is contracted to spend the FULL wall-clock budget. '
      + 'stop=true is honored ONLY when both hold: (i) no time remains for another round '
      + `(one round costs ~${roundCostS(budget)}s, ${budget.remainingS}s remain, so `
      + `~${affordableText} more round(s) fit), and (ii) ${TEAM_STOP_CONFIRMATIONS} consecutive `
      + `rounds request it (prior consecutive stop requests: ${teamStopStreak}). `
      + (canAffordAnotherRound(budget)
        ? 'Budget remains, so a stop WILL BE REFUSED and would waste this entire round on a '
          + 'generic escape direction. Issue a materially new direction instead.'
        : 'The budget is now effectively spent, so a justified stop will be honored.');
  const planInputs = {
    EVAL_DIR, ROUND: roundNo, DIRECTION_CAP: OPERATOR_DIRECTIONS, CUMULATIVE_SPEEDUP: cumulative,
    BUDGET_TOTAL_S: BUDGET_S, BUDGET_REMAINING_S: remainingS(),
    BUDGET_REMAINING_PCT: BUDGET_S > 0 ? Math.round(100 * remainingS() / BUDGET_S) : 0,
    AVG_ROUND_S: estStepS() || 'unknown (no round has completed yet)',
    ROUNDS_AFFORDABLE: affordableText, MAX_STEPS,
    BASELINE_GEOMEAN_MS, SKILL_DIR: KERNEL_WORKFLOW_DIR,
    PROFILE_SUMMARY: profileSummary, CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
    ROOFLINE: steerFrom(bestRoofline), SPECIALISTS: specialists, IR_FEEDBACK,
    STOP_POLICY: stopPolicy,
    TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH,
    LANGUAGE_ENGINEER_ADDENDUM, LANGUAGE_EDIT_TARGET, LANGUAGE_CONTRACT,
    ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
  };
  const planIntro = 'Plan this round\'s orthogonal specialist directions for the current best. '
    + LANGUAGE_CONTRACT;
  const askPlan = (inputs, label, intro) => agent(
    roleAgentFrom(KERNEL_WORKFLOW_DIR, 'tech_lead', 'plan_round', intro, inputs),
    { phase: 'Evolve', label, schema: PLAN_SCHEMA });
  const usableDirections = (p) => (p && p.directions ? p.directions : [])
    .filter(d => d && d.prompt)
    .map((d, i) => ({ name: d.title || d.id || d.specialty, expected: d.expected_speedup,
      prompt: d.prompt, focus_files: d.focus_files || [],
      specialty: d.specialty || specialists[i % specialists.length] }))
    .filter(d => d.name && !history.failed.includes(d.name))
    .slice(0, OPERATOR_DIRECTIONS);

  const plan = await askPlan(planInputs, `tech_lead:plan r${roundNo}`, planIntro);
  // Do NOT return early on stop. When the stop is refused this round still needs directions,
  // and the re-plan below (or failing that, the escape direction) is what supplies them;
  // returning [] here would make every refused round a no-op.
  const wantsStop = plan?.stop === true;
  let dirs = usableDirections(plan);

  // A refused stop must not cost the round its planning. Answering stop=true normally comes
  // with no directions, so falling straight through to the generic escape direction spent the
  // round on one unfocused probe instead of the 2-3 targeted directions the budget paid for.
  // Tell TechLead the stop was refused, quote the budget that refused it, and give it one
  // chance to plan for real. The stop request itself still stands: the caller keeps counting
  // the confirmation streak, so a genuine stop is still honored once the budget is spent.
  const postPlanBudget = budgetNow();
  const prospective = shouldHonorTeamStop({
    stop: wantsStop, earlyStop: EARLY_STOP, streak: teamStopStreak + 1,
    confirmations: TEAM_STOP_CONFIRMATIONS, ...postPlanBudget,
  });
  if (wantsStop && !prospective.honored && dirs.length === 0) {
    const stillFits = affordableRounds(postPlanBudget);
    log(`Step ${roundNo}: TechLead asked to stop with no directions and the stop will be `
      + `refused [${prospective.reason}] — re-planning once instead of spending the round on a `
      + 'generic escape direction.');
    const replan = await askPlan({
      ...planInputs,
      STOP_REFUSED: `Your stop=true was REFUSED (${prospective.reason}). `
        + `~${postPlanBudget.remainingS}s of the ${BUDGET_S}s budget remain, which still fits `
        + `~${stillFits === null ? 'an unknown number of' : stillFits} more round(s). The run `
        + 'continues whatever you return, so a second stop only forfeits this round to a generic '
        + `probe. Skip rule 1's stop branch this round and return `
        + (OPERATOR_DIRECTIONS > 1
          ? `2-${OPERATOR_DIRECTIONS} materially new directions`
          : 'a materially new direction')
        + ' under rules 2-3, targeting the least-attacked ceiling in ROOFLINE/HISTORY.',
    }, `tech_lead:replan r${roundNo}`,
    `Your stop was refused; the budget is still open. ${planIntro}`);
    const replanned = usableDirections(replan);
    if (replanned.length) {
      dirs = replanned;
      history.insights.push(`[tech_lead:replan r${roundNo}] ${replan?.reasoning || ''}`);
    } else {
      log(`Step ${roundNo}: the re-plan returned no usable direction either — falling back to `
        + 'the escape direction.');
    }
  }
  if (dirs.length === 0) {
    dirs.push({
      name: `team_escape_${roundNo}`,
      expected: 'Find a new direction outside the recorded failed set.',
      // Do NOT inline history.failed here: the direction's own text is what the
      // language guard scans, so retired names like "collapse_launch_overhead"
      // would make the escape direction itself look like host-runtime intent and
      // get dropped. The worker already receives FAILED_DIRECTIONS separately.
      prompt: 'Choose one materially new optimization that is not represented in '
        + 'the FAILED_DIRECTIONS input. Do not repeat a failed mechanism under a '
        + 'new title.',
      focus_files: [],
      specialty: specialists[0],
    });
  }
  const guarded = constrainLanguageDirections(dirs, {
    languageMode: LANGUAGE_MODE, count: OPERATOR_DIRECTIONS,
    step: roundNo, failed: history.failed,
  });
  recordDirectionRepairs(guarded, `team-plan:${roundNo}`);
  return { stop: wantsStop, reasoning: plan?.reasoning || '', directions: guarded.directions };
}

while (Date.now() < DEADLINE && step < MAX_STEPS) {
  if (estStepS() > 0 && remainingS() < estStepS()) {
    log(`Evolve stopping before next round: ${remainingS()}s remain before final-phase reserve, `
      + `below recent round estimate ${estStepS()}s.`);
    break;
  }
  step++;
  const stepT0 = Date.now();
  const gpu = GPU_LIST[(step - 1) % GPU_LIST.length];
  const stepDir = `${EVAL_DIR}/variation_${String(step).padStart(4, '0')}`;
  if (ASM_MODE) {
    const guarded = constrainLanguageDirections([direction], {
      languageMode: LANGUAGE_MODE, count: 1, step, failed: history.failed,
    });
    recordDirectionRepairs(guarded, `step:${step}`);
    [direction] = guarded.directions;
  }

  try {
  if (OPERATOR === 'team') {
    // ===== operator=team: one bounded Team round = the variation operator. =====
    const teamPlan = await planTeamDirections(step);
    teamStopStreak = teamPlan.stop ? teamStopStreak + 1 : 0;
    // With early_stop off the run is contracted to spend the full wall-clock budget, so a
    // TechLead stop must not end it while another round still fits — the same rule the
    // convergence and roofline stops already follow. Confirmations alone were not enough:
    // a confident TechLead just re-issued the stop until the streak was met, exiting 12h
    // runs at round 7 with ~10h unspent.
    const stopBudget = budgetNow();
    const stopDecision = shouldHonorTeamStop({
      stop: teamPlan.stop, earlyStop: EARLY_STOP, streak: teamStopStreak,
      confirmations: TEAM_STOP_CONFIRMATIONS, ...stopBudget,
    });
    if (stopDecision.honored) {
      log(`Step ${step}: TechLead stopped team search after ${teamStopStreak} consecutive `
        + `stop request(s) [${stopDecision.reason}], ~${stopBudget.remainingS}s budget left `
        + `(one round needs ~${roundCostS(stopBudget)}s). ${teamPlan.reasoning}`);
      history.ledger.push({ step, mode: 'team', committed: false, reason: 'tech_lead_stop',
        confirmations: teamStopStreak, honored_because: stopDecision.reason,
        remaining_s: stopBudget.remainingS, detail: teamPlan.reasoning });
      break;
    }
    if (teamPlan.stop) {
      const affordable = affordableRounds(stopBudget);
      log(`Step ${step}: TechLead stop signal REFUSED [${stopDecision.reason}] (early_stop off, `
        + `${teamStopStreak}/${TEAM_STOP_CONFIRMATIONS} confirmations, ~${stopBudget.remainingS}s `
        + `left = ~${affordable === null ? 'unknown' : affordable} more round(s)): `
        + `${teamPlan.reasoning} — continuing to spend the budget.`);
      history.insights.push(
        `[tech_lead:stop_refused ${stopDecision.reason} `
        + `${teamStopStreak}/${TEAM_STOP_CONFIRMATIONS}, ~${stopBudget.remainingS}s budget left] `
        + `${teamPlan.reasoning}`);
    }
    const dirs = teamPlan.directions;
    const pairs = pairStartsWithDirections(dirs, START_K);
    log(`Step ${step} (team): ${pairs.length} workers [${pairs.map(p => `${p.d.specialty || p.d.name}@${p.startRef}`).join(', ')}]`);
    const round = await fanOutEvolve(pairs, `variation_${String(step).padStart(4, '0')}`);
    const roundHealth = handleFanOutRoundHealth(round, 'team');
    if (roundHealth === 'abort') break;
    if (roundHealth === 'retry') continue;
    // Try Integrator merge of complementary winners, else commit the best winner.
    if (!await commitRoundResult(round, step, `variation_${String(step).padStart(4, '0')}`)) {
      noImprove++;
      history.ledger.push({ step, mode: 'team', committed: false });
      log(`Step ${step}: team round produced no committable candidate; noImprove=${noImprove}`);
    }
  } else {
    // ===== operator=single (explicit): one agent = the variation operator. =====
    // --- (1) Variation step (agent): start from best, one focused change. ----
    const singleStartRef = explorationStartRef || bestVersion;
    explorationStartRef = null;
    let v = await agent(
      roleAgent('variation_engineer', 'step',
        'Run ONE variation step: clone avo_repo@best, make one focused change, save_and_test, emit an incremental patch.', {
        AVO_REPO, AVO_STATE, STEP_DIR: stepDir, WORKSPACE: makeWorkspace(`${stepDir}/workspace`, singleStartRef),
        BEST_VERSION: singleStartRef, GLOBAL_BEST_VERSION: bestVersion,
        CUMULATIVE_SPEEDUP: cumulative,
        DIRECTION: direction, GPU_ID: gpu, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_ENV_SCRIPT, TASK,
        BASELINE_PER_CASE, PROFILE_SUMMARY: profileSummary ? profileSummary.summary_path : '', STAGE: currentStage(),
        KERNEL_KNOWLEDGE_DIR, USE_PROFILING, INSIGHTS: history.insights, FAILED_DIRECTIONS: history.failed,
        RECORD_ATTEMPT: !RUNTIME_RECORDS_ATTEMPTS,
        SYNC_HEARTBEAT: RUNTIME_SYNCS_HEARTBEAT,
        ROOFLINE: steerFrom(bestRoofline),
        IR_FEEDBACK, IR_LANG, IR_TIER, IR_KERNEL_NAME, IR_TRITON_CACHE,
        TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH,
        LANGUAGE_ENGINEER_ADDENDUM, LANGUAGE_EDIT_TARGET, LANGUAGE_CONTRACT,
        ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
        AUTOTUNE: { enabled: AUTOTUNE, max_configs: AUTOTUNE_MAX },
      }),
      { phase: 'Evolve', label: `variation s${step}`, schema: VARIATION_SCHEMA });
    v = enforceWorkerLanguageScope(v);

    // Infra-failure guard (broken harness/env): abort after N consecutive.
    if (v && v.status === 'infra_failure') {
      if (countRigFailure(v.notes || 'worker reported a broken harness/env')) break;
      continue;
    }
    // The counter is reset only once this step produced a usable measurement
    // (below), so consecutive verification-side failures still accumulate.

    // No usable candidate → count toward stall, maybe supervise.
    if (!v || v.status !== 'ok' || !v.patch_file) {
      noImprove++;
      if (v && ['failed', 'no_change', 'incorrect'].includes(
        String(v.status || '').toLowerCase())) {
        rememberFailedDirection(direction.name || v.strategy,
          `candidate ${v.status}`, { hard: true });
      }
      logAttempt({ step, strategy: (v && v.strategy) || direction.name || null,
        status: (v && v.status) || 'null',
        self_speedup: v && typeof v.self_speedup === 'number' ? v.self_speedup : null,
        rationale: v && v.rationale ? String(v.rationale).slice(0, 500) : null });
      log(`Step ${step}: no candidate (${v ? v.status : 'null'}); noImprove=${noImprove}`);
    } else {
      // Cycle detection (same patch re-emitted).
      if (v.patch_hash) {
        recentHashes.push(v.patch_hash);
        if (recentHashes.length > STALL_PATCH_REPEAT) recentHashes.shift();
      }

      // --- (2) Independent verification (skipped when ir_prescreen predicts regression).
      const prSkip = irPrescreenBlocks(bestIr, v.ir);
      let ver = null;
      if (prSkip) {
        log(`Step ${step}: ir_prescreen skipped verify — ${prSkip.reasons.join('; ')}`);
        logAttempt({ step, strategy: v.strategy || direction.name || null, status: 'ir_prescreen',
          self_speedup: typeof v.self_speedup === 'number' ? v.self_speedup : null,
          verified_geomean: null, patch_hash: v.patch_hash || null,
          rationale: `ir_prescreen: ${prSkip.reasons.join('; ')}`, ir: irBrief(v.ir) });
        noImprove++;
        rememberFailedDirection(direction.name || v.strategy, 'IR prescreen rejected');
        history.ledger.push({ step, strategy: v.strategy, verified: null, committed: false });
      } else {
        if (Date.now() >= DEADLINE) {
          log(`Step ${step}: verify skipped because the reserved final-phase window has started`);
          noImprove++;
          history.ledger.push({ step, strategy: v.strategy, verified: null, committed: false,
            reason: 'final_phase_reserved_before_verify' });
          continue;
        }
        const verifyDir = `${stepDir}/verify`;
        const workspace = makeWorkspace(`${verifyDir}/workspace`, singleStartRef);
        const effectiveWorkspace = workspace || `${verifyDir}/workspace`;
        if (workspace && v.patch_file && !patchAppliesOnWorkspace(workspace, v.patch_file)) {
          log(`Step ${step}: verify skipped — PATCH_DID_NOT_APPLY on ${singleStartRef}`);
          logAttempt({ step, strategy: v.strategy || direction.name || null,
            status: 'PATCH_DID_NOT_APPLY',
            self_speedup: typeof v.self_speedup === 'number' ? v.self_speedup : null,
            verified_geomean: 0, patch_hash: v.patch_hash || null });
          noImprove++;
          history.ledger.push({ step, strategy: v.strategy, verified: 0, committed: false,
            reason: 'PATCH_DID_NOT_APPLY' });
          continue;
        }
        const screeningSession = (ASM_MODE || IR_MODE)
          ? newPersistentScreeningSession() : '';
        ver = await agent(
          roleAgent(verifyRole, 'verify',
            'Independently re-measure this candidate on its base ref (median over repeats).', {
            AVO_REPO, BASE_REF: singleStartRef, PATCH: v.patch_file, VERIFY_DIR: verifyDir,
            WORKSPACE: workspace,
            GPU_ID: gpu, KERNEL_WORKFLOW_DIR, COMMANDMENT, KERNEL_ENV_SCRIPT, BASELINE_PER_CASE, VERIFY_REPEATS,
            SKILL_DIR: WORKFLOW_DIR, TARGET_LANGUAGE: TARGET_LANGUAGE_NORM,
            LANGUAGE_MODE, LANGUAGE_SKILL_PATH, LANGUAGE_CONTRACT,
            ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
            ...(IR_MODE ? {
              GEAK_CAPTURE_DIR, GEAK_PERF_BACKEND: 'generated-persistent',
            } : {}),
            ...(ASM_MODE
              ? { ASM_SCREENING_SESSION_ID: screeningSession }
              : (IR_MODE
                ? { IR_SCREENING_SESSION_ID: screeningSession } : {})),
          }),
          { phase: 'Verify', label: `verify s${step}`,
            schema: (ASM_MODE || IR_MODE) ? ASM_VERIFY_SCHEMA : VERIFY_SCHEMA });
        ver = normalizePersistentScreening(
          ver && readAgentReport(`${verifyDir}/verify_report.json`),
          screeningSession, effectiveWorkspace);

        // Durable trajectory: record this attempt (pass or fail), deterministically.
        logAttempt({ step, strategy: v.strategy || direction.name || null,
          status: ver && ver.status === 'failed' ? 'verification_failed' : (v.status || 'ok'),
          candidate_status: v.status || null,
          verification_status: ver && ver.status ? ver.status : null,
          self_speedup: typeof v.self_speedup === 'number' ? v.self_speedup : null,
          verified_geomean: ver ? commitScore(ver) : null,
          evidence_rejection: ver && ver.evidence_rejection ? ver.evidence_rejection : null,
          screening_evidence_dir: ver && ver.screening_evidence_dir ? ver.screening_evidence_dir : null,
          patch_hash: v.patch_hash || null,
          rationale: v.rationale ? String(v.rationale).slice(0, 500) : null, ir: irBrief(v.ir) });

        // The verifier returned nothing at all, so this candidate was never
        // measured. Falling through would fail the commit gate on a missing number
        // and retire the direction as "did not improve" on evidence that does not
        // exist, so charge it to the outage budget instead.
        if (!ver) {
          history.ledger.push({ step, strategy: v.strategy, verified: null,
            committed: false, reason: 'verification_agent_outage' });
          if (countAgentOutage(
            `verification returned no result for ${v.strategy || 'candidate'}`)) break;
          continue;
        }
        // An unusable measurement is not a candidate that failed to improve, so
        // it must reach an abort counter rather than only bump noImprove.
        if (isUnusableMeasurement(v, ver)) {
          const rig = isRigFailure(v, ver);
          const detail = ver?.evidence_rejection
            || ver?.infrastructure_status || 'unknown';
          const drifted = isCaptureDrift(ver);
          history.ledger.push({ step, strategy: v.strategy, verified: null,
            committed: false,
            reason: drifted ? 'capture_changed_mid_run'
              : (rig ? 'verification_rig_unhealthy' : 'evidence_rejected') });
          if (drifted) {
            log('Aborting: the formal capture changed during the run, so no worker '
              + `can produce valid evidence — ${detail}`);
            break;
          }
          const abort = rig
            ? countRigFailure(`verification unusable: ${detail}`)
            : countEvidenceRejection(detail);
          if (abort) break;
          continue;
        }
        infraFailures = 0;
        evidenceRejections = 0;
        agentOutages = 0;

        // --- (3) Commit gate (DETERMINISTIC). ---------------------------------
        if (passesCommitGate(ver, cumulative)) {
          if (!await commitCandidate(
            { patch_file: v.patch_file, strategy: v.strategy || direction.name || '',
              expected: direction.expected || '', rationale: v.rationale || '', ir: v.ir },
            ver, step, singleStartRef)) noImprove++;
        } else {
          noImprove++;
          if (ver?.status === 'ok' && ver?.correctness === 'pass') {
            rememberFailedDirection(direction.name || v.strategy, 'verified below commit gate');
          } else if (ver?.correctness === 'fail' && !ver?.evidence_rejection) {
            rememberFailedDirection(direction.name || v.strategy, 'correctness failed',
              { hard: true });
          }
          history.ledger.push({ step, strategy: v.strategy,
            verified: ver ? commitScore(ver) : null, committed: false });
          log(`Step ${step}: not committed (verified=${ver ? commitScore(ver) : '?'} `
            + `via ${COMMIT_SCORE_FIELD} vs best ${cumulative.toFixed(4)}); noImprove=${noImprove}`);
        }
      }
    }
  }

  // Record this step's core wall time (variation+verify+commit) for the budget-aware
  // supervisor estimate — trailing window of the last few steps.
  stepDurationsMs.push(Date.now() - stepT0);
  if (stepDurationsMs.length > 5) stepDurationsMs.shift();

  // --- (4) Stagnation → supervisor re-plan (deterministic trigger). --------
  const cycling = recentHashes.length >= STALL_PATCH_REPEAT &&
    new Set(recentHashes).size === 1;
  const stalled = noImprove >= STALL_NO_IMPROVE || cycling;
  // Budget-aware gate: only re-plan if there is room to run the supervisor AND ≥1
  // redirected step afterwards; otherwise the expensive re-plan lands after the
  // deadline and its new directions are never tried (wasted intervention).
  const needS = estStepS() + SUPERVISOR_OVERHEAD_S;
  const canAffordRedirect = estStepS() === 0 || remainingS() >= needS;
  if (stalled && !canAffordRedirect) {
    log(`Step ${step}: stall detected (noImprove=${noImprove}${cycling ? ', cycling' : ''}) but only ${remainingS()}s left (< ~${needS}s for redirect + 1 step) — skipping supervisor re-plan to bank the budget.`);
  } else if (stalled) {
    const nextSupervisorCycle = supervisorCycles + 1;
    const sup = await agent(
      roleAgent('supervisor', 'redirect',
        'The lineage is stalling. Diagnose, mark dead directions failed, and propose new untried directions.', {
        AVO_STATE, AVO_REPO, BEST_VERSION: bestVersion, CUMULATIVE_SPEEDUP: cumulative,
        STALL_REASON: cycling ? 'patch_cycling' : 'no_improvement', NO_IMPROVE: noImprove,
        FAILED_DIRECTIONS: history.failed, CURRENT_DIRECTION: direction.name,
        PROFILE_SUMMARY: profileSummary ? profileSummary.summary_path : '',
        ROOFLINE: steerFrom(bestRoofline),
        KERNEL_KNOWLEDGE_DIR, SKILL_DIR: WORKFLOW_DIR,
        REDIRECTS_SINCE_COMMIT: redirectsSinceCommit,
        TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH,
        LANGUAGE_ENGINEER_ADDENDUM, LANGUAGE_EDIT_TARGET, LANGUAGE_CONTRACT,
      }),
      { phase: 'Evolve', label: `supervisor c${nextSupervisorCycle}`, schema: SUPERVISOR_SCHEMA });
    let supv = null;
    if (sup) {
      supervisorCycles = nextSupervisorCycle;
      const norm = normalizeSupervisorDirective(sup, { step, failed: history.failed });
      const guarded = constrainLanguageDirections(norm.directive.new_directions, {
        languageMode: LANGUAGE_MODE,
        count: Math.max(1, norm.directive.new_directions.length),
        step, failed: history.failed,
      });
      supv = {
        ...norm.directive,
        new_directions: guarded.directions,
        next_direction: guarded.directions.some(
          (item) => item.name === norm.directive.next_direction)
          ? norm.directive.next_direction : guarded.directions[0].name,
      };
      for (const r of norm.repairs) history.insights.push(`[orchestrator] ${r}`);
      recordDirectionRepairs(guarded, `supervisor:${step}`);
      if (typeof recordSupervision === 'function') {
        try { recordSupervision(AVO_STATE, supv); } catch { /* best-effort */ }
      }
      pulseOrchestratorHeartbeat('supervisor');
    }
    let stopConverged = false;
    let convergeDetail = '';
    if (sup && supv) {
      redirectsSinceCommit++;
      for (const f of (supv.mark_failed || [])) if (!history.failed.includes(f)) history.failed.push(f);
      history.insights.push(`[supervisor] ${supv.diagnosis}`);
      const nds = supv.new_directions;
      let nd = nds.find(d => d.name && !history.failed.includes(d.name));
      if (!nd) {
        history.insights.push('[orchestrator] no novel new_directions name after normalize; escape fallback.');
        const guarded = constrainLanguageDirections([], {
          languageMode: LANGUAGE_MODE, count: 1, step, failed: history.failed,
        });
        recordDirectionRepairs(guarded, `supervisor-fallback:${step}`);
        [nd] = guarded.directions;
        if (!ASM_MODE) {
          nd = { name: `escape_stall_${step}`, expected: '', prompt: 'Orthogonal one-step change from roofline playbook / knowledge base; do not repeat strategies in FAILED_DIRECTIONS.' };
        }
      }
      direction = { name: nd.name, expected: nd.expected || '', prompt: nd.prompt || '' };
      if (supv.backtrack_to) {
        const target = committed.find(node => node.ref === supv.backtrack_to);
        if (!target) {
          log(`Supervisor backtrack ignored: unknown ref ${supv.backtrack_to}`);
        } else {
          explorationStartRef = target.ref;
          log(`Supervisor backtrack scheduled next exploration from ${target.ref} @ `
            + `${Number(target.speedup || 1).toFixed(4)}x; global champion/gate remains `
            + `${bestVersion} @ ${cumulative.toFixed(4)}x`);
        }
      }

      const diag = supv.diagnosis ? String(supv.diagnosis).trim() : '';
      if (diag && lastSupervisorDiagnosis) {
        const sim = diagnosisTokenJaccard(diag, lastSupervisorDiagnosis);
        if (sim >= DX_SIM_THRESHOLD) dxDupStreak++;
        else dxDupStreak = 0;
      } else {
        dxDupStreak = 0;
      }
      if (diag) lastSupervisorDiagnosis = diag;

      if (supv.converge === true) {
        stopConverged = true; convergeDetail = 'supervisor converge=true (structural / no novel angle)';
      } else if (CONVERGE_BARREN_CYCLES > 0 && redirectsSinceCommit >= CONVERGE_BARREN_CYCLES) {
        stopConverged = true;
        convergeDetail = `barren: ${redirectsSinceCommit} supervisor redirect(s) since last commit (threshold ${CONVERGE_BARREN_CYCLES})`;
      } else if (CONVERGE_DX_REPEAT > 0 && dxDupStreak >= CONVERGE_DX_REPEAT) {
        stopConverged = true;
        convergeDetail = `repeated diagnosis: token-Jaccard ≥ ${DX_SIM_THRESHOLD} vs prior for ${dxDupStreak} consecutive pair(s) (threshold ${CONVERGE_DX_REPEAT})`;
      }
    }

    if (stopConverged && EARLY_STOP) {
      log(`Convergence stop: ${convergeDetail} — exiting Evolve (~${remainingS()}s budget left).`);
      history.insights.push(`[converge] ${convergeDetail}`);
      noImprove = 0; recentHashes.length = 0;
      pulseOrchestratorHeartbeat('converge_stop');
      log(`Step ${step} done. best=${bestVersion} (${cumulative.toFixed(4)}x) commits=${commits} remaining=${remainingS()}s`);
      break;
    }
    if (stopConverged) {
      // Default (early_stop OFF): do NOT stop early — keep the supervisor's new directions
      // and fall through to the redirect/escalate path so the remaining budget is used.
      log(`Convergence signal noted but NOT stopping (early_stop off): ${convergeDetail} — continuing with ~${remainingS()}s budget left.`);
      history.insights.push(`[converge:continue] ${convergeDetail}`);
    }

    if (!supv) {
      log('Supervisor returned no usable directive; preserving stall counters for retry.');
      continue;
    }
    noImprove = 0; recentHashes.length = 0;

    // --- (5) ESCALATE: diversified fan-out after repeated fruitless cycles.
    //     (Same fan-out helper as operator=team — just a wider, on-stall rescue.)
    if (supervisorCycles >= SUPERVISOR_CYCLES_TO_ESCALATE) {
      log(`ESCALATE: diversified rescue with ${ESCALATE_WORKERS} workers (after step ${step}).`);
      pulseOrchestratorHeartbeat('escalate');
      let dirs = (supv && supv.new_directions ? supv.new_directions : [])
        .filter(d => d.name && !history.failed.includes(d.name))
        .slice(0, ESCALATE_WORKERS)
        .map(d => ({ name: d.name, expected: d.expected || '', prompt: d.prompt || '' }));
      const guarded = constrainLanguageDirections(dirs, {
        languageMode: LANGUAGE_MODE, count: ESCALATE_WORKERS,
        step, failed: history.failed,
      });
      recordDirectionRepairs(guarded, `escalate:${step}`);
      dirs = guarded.directions;
      while (dirs.length < ESCALATE_WORKERS) dirs.push({ name: `explore_${dirs.length}`, expected: 'diversify', prompt: '' });

      // ESCALATE diversifies starts too (topk when policy=best, else the chosen policy).
      const escStarts = selectStartPoints(committed,
        { policy: START_POLICY === 'best' ? 'topk' : START_POLICY, k: ESCALATE_WORKERS });
      const escPairs = dirs.map((d, i) => ({ d, startRef: escStarts[i % escStarts.length] }));

      const escRound = await fanOutEvolve(escPairs, `escalate_${step}`);
      supervisorCycles = 0;
      const escHealth = handleFanOutRoundHealth(escRound, 'ESCALATE');
      if (escHealth === 'abort') break;
      if (escHealth === 'retry') continue;
      // Same completion as operator=team: try the Integrator merge first, else best.
      if (await commitRoundResult(escRound, step, `escalate_${step}`)) {
        log(`ESCALATE produced a new best ${bestVersion} @ ${cumulative.toFixed(4)}x`);
      } else {
        log('ESCALATE: no rescue candidate beat the current best.');
      }
    }
  }

  log(`Step ${step} done. best=${bestVersion} (${cumulative.toFixed(4)}x) commits=${commits} remaining=${remainingS()}s`);
  pulseOrchestratorHeartbeat('evolve');

  // Roofline early-stop: once we're within ROOFLINE_TARGET of the hardware ceiling
  // (and min_commits satisfied), further steps chase sub-noise gains — stop and
  // bank the budget. Inert when there's no flops/bytes telemetry.
  const stopRL = shouldStopRoofline(bestRoofline, { target: ROOFLINE_TARGET, commits, minCommits: MIN_COMMITS });
  if (stopRL.stop && EARLY_STOP) { log(`Roofline stop: ${stopRL.reason}`); break; }
  else if (stopRL.stop) log(`Roofline stop signal noted but NOT stopping (early_stop off): ${stopRL.reason} — continuing.`);
  } finally {
    // Deterministic teardown: reclaim any worktree this step leaked, regardless of how
    // the body exited (success / continue / break / thrown error). The only uncovered
    // case is the orchestrator process itself being hard-killed mid-step.
    cleanupStepWorktrees();
  }
}

log(`Evolve done. steps=${step} commits=${commits} best=${bestVersion} (${cumulative.toFixed(4)}x)`);
if (commits < MIN_COMMITS) log(`WARNING: only ${commits} commits (< min_commits=${MIN_COMMITS}); budget/kernel may need attention.`);

// ===========================================================================
// PHASE: Report — final report + cumulative baseline->best patch.
// ===========================================================================
phase('Report');
pulseOrchestratorHeartbeat('report');
let irReportContext = '';
if (IR_FEEDBACK) {
  try {
    const { writeFileSync, mkdirSync } = await import('node:fs');
    mkdirSync(EVAL_DIR, { recursive: true });
    irReportContext = `${EVAL_DIR}/ir_report_context.json`;
    const regime = bestRoofline && bestRoofline.ok ? bestRoofline.dominant_regime : 'unknown';
    const payload = {
      ir_feedback: true,
      roofline_regime: regime,
      geomean_fraction_of_roofline: bestRoofline && bestRoofline.ok ? bestRoofline.geomean_fraction : null,
      best_ir: bestIr,
      ir_steer: irSteer(regime, bestIr),
    };
    writeFileSync(irReportContext, JSON.stringify(payload, null, 2), 'utf8');
  } catch (e) {
    log(`Report: could not write ir_report_context (${String(e && e.message ? e.message : e)})`);
    irReportContext = '';
  }
}
const reportInputs = {
  AVO_REPO, AVO_STATE, EVAL_DIR, BEST_VERSION: bestVersion, CUMULATIVE_SPEEDUP: cumulative,
  COMMITS: commits, STEPS: step, BASELINE_PER_CASE, BEST_PER_CASE: bestPerCase,
  ROOFLINE_FRACTION: bestRoofline.ok ? bestRoofline.geomean_fraction : null,
  ROOFLINE_REGIME: bestRoofline.ok ? bestRoofline.dominant_regime : 'unknown', SKILL_DIR: WORKFLOW_DIR,
  IR_FEEDBACK,
};
if (IR_FEEDBACK && irReportContext) reportInputs.IR_REPORT_CONTEXT = irReportContext;
let report = null;
try {
  report = await agent(
    roleAgent('tech_lead', 'report', 'Write the final report and emit the cumulative final_patch.diff (avo-v0 -> best).', reportInputs),
    { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA });
} catch (e) {
  log(`Report: TechLead failed (${String(e)}); attempting deterministic patch synthesis.`);
}
// A measured practical ceiling (for example a streaming microbenchmark) is not
// the same thing as the analytic hardware roofline. Accept it only with an
// in-run evidence artifact and retain the basis/provenance in the result.
let practicalRoofline = { ok: false, fraction: null, regime: 'unknown', evidence_path: '' };
if (report && Number.isFinite(report.practical_roofline_fraction)
    && report.practical_roofline_fraction > 0 && report.practical_roofline_fraction <= 1.05
    && report.roofline_evidence_path) {
  try {
    const { existsSync } = await import('node:fs');
    const { isAbsolute, resolve, sep } = await import('node:path');
    const reportedPath = String(report.roofline_evidence_path);
    const evidencePath = isAbsolute(reportedPath)
      ? resolve(reportedPath) : resolve(EVAL_DIR, reportedPath);
    const evalRoot = resolve(EVAL_DIR);
    if ((evidencePath === evalRoot || evidencePath.startsWith(`${evalRoot}${sep}`))
        && existsSync(evidencePath)) {
      practicalRoofline = {
        ok: true,
        fraction: report.practical_roofline_fraction,
        regime: report.practical_roofline_regime || 'measured-practical',
        evidence_path: evidencePath,
      };
    } else {
      log(`Report: rejected practical roofline evidence outside/missing from EVAL_DIR: ${evidencePath}`);
    }
  } catch (e) {
    log(`Report: could not validate practical roofline evidence (${String(e)})`);
  }
}
// The cumulative patch is just the diff between two lineage tags, so it does not
// need an agent to exist. Throwing here discarded a 19.5h run whose avo-v1 (1.3122x)
// was already committed and tagged, because the reporter could not reach its
// backend. Fall back to the deterministic diff and carry on without the prose.
async function synthesizeFinalPatch() {
  if (!commits || bestVersion === 'avo-v0') return '';
  try {
    const { execFileSync } = await import('node:child_process');
    const { writeFileSync, mkdirSync } = await import('node:fs');
    const diff = execFileSync(
      'git', ['-C', AVO_REPO, 'diff', 'avo-v0', bestVersion],
      { encoding: 'utf8', maxBuffer: 256 << 20 });
    if (!diff.trim()) return '';
    mkdirSync(EVAL_DIR, { recursive: true });
    const out = `${EVAL_DIR}/final_patch.diff`;
    writeFileSync(out, diff, 'utf8');
    return out;
  } catch (e) {
    log(`Report: deterministic final_patch synthesis failed (${String(e && e.message ? e.message : e)})`);
    return '';
  }
}
let FINAL_PATCH = report && report.final_patch ? report.final_patch : '';
if (!FINAL_PATCH) {
  FINAL_PATCH = await synthesizeFinalPatch();
  if (FINAL_PATCH) {
    log(`Report: reporter produced no final_patch; synthesized avo-v0..${bestVersion} `
      + `deterministically at ${FINAL_PATCH}`);
  }
}
if (!FINAL_PATCH) {
  await runFinalCleanup();
  pulseOrchestratorHeartbeat('complete');
  throw new Error('Report failed: no cumulative final_patch was produced'
    + (commits ? ' and avo-v0..best could not be diffed' : ' (nothing was ever committed)'));
}

// Every exit from here on must still hand the caller the run's durable facts —
// lineage tag, commit count, cumulative patch, roofline. Only the independently
// validated numbers depend on the Director actually running.
const runContract = (overrides = {}) => ({
  engine: 'avo-workflow',
  eval_dir: EVAL_DIR,
  kernel_name: KERNEL_NAME,
  final_geomean: 0,
  final_arithmetic: 0,
  final_patch: FINAL_PATCH,
  output_bundle: '',
  output_artifacts: [],
  best_version: bestVersion,
  num_commits: commits,
  steps: step,
  roofline_fraction: bestRoofline.ok ? bestRoofline.geomean_fraction
    : (practicalRoofline.ok ? practicalRoofline.fraction : null),
  roofline_regime: bestRoofline.ok ? bestRoofline.dominant_regime
    : (practicalRoofline.ok ? practicalRoofline.regime : 'unknown'),
  roofline_basis: bestRoofline.ok ? 'analytic'
    : (practicalRoofline.ok ? 'measured_practical' : 'unavailable'),
  roofline_evidence_path: !bestRoofline.ok && practicalRoofline.ok
    ? practicalRoofline.evidence_path : '',
  practical_roofline_fraction: practicalRoofline.ok ? practicalRoofline.fraction : null,
  practical_roofline_regime: practicalRoofline.ok ? practicalRoofline.regime : 'unknown',
  practical_roofline_evidence_path: practicalRoofline.ok
    ? practicalRoofline.evidence_path : '',
  validation_status: 'unknown',
  report_path: '',
  formal_report_path: '',
  ...overrides,
});

async function materializeSourceBundle(validationStatus, benchmarkReport = '', directorReport = '') {
  if (ASM_MODE || IR_MODE) return null;
  try {
    const { copyFileSync, existsSync, mkdirSync, writeFileSync } = await import('node:fs');
    const { basename, extname, join, resolve } = await import('node:path');
    const outputDir = `${EVAL_DIR}/final_artifacts`;
    mkdirSync(outputDir, { recursive: true });
    const artifacts = [];
    const add = (src, preferredName = '') => {
      if (!src || !existsSync(src)) return;
      const dst = join(outputDir, preferredName || basename(src));
      if (resolve(src) !== resolve(dst)) copyFileSync(src, dst);
      artifacts.push(dst);
    };
    add(FINAL_PATCH, 'final_patch.diff');
    add(directorReport, 'director_validation.json');
    add(report && report.report_path, 'tech_lead_report.md');
    if (benchmarkReport) {
      add(benchmarkReport, `benchmark_report${extname(String(benchmarkReport)) || '.json'}`);
    }
    if (practicalRoofline.evidence_path) {
      add(practicalRoofline.evidence_path,
        `roofline_evidence${extname(practicalRoofline.evidence_path) || '.md'}`);
    }
    add(`${AVO_STATE}/lineage.json`, 'lineage.json');
    const manifestPath = join(outputDir, 'manifest.json');
    writeFileSync(manifestPath, `${JSON.stringify({
      engine: 'avo-workflow',
      kernel_name: KERNEL_NAME,
      best_version: bestVersion,
      validation_status: validationStatus,
      artifacts: artifacts.map((p) => basename(p)),
    }, null, 2)}\n`, 'utf8');
    artifacts.push(manifestPath);
    return { finalized: true, output_dir: outputDir, artifacts };
  } catch (e) {
    log(`Finalize source bundle failed (${String(e)})`);
    return null;
  }
}

async function runFinalCleanup() {
  phase('Cleanup');
  pulseOrchestratorHeartbeat('cleanup');
  if (RUNTIME_PRUNES_WORKTREES) {
    try {
      const n = pruneWorktrees(AVO_REPO, null, { all: true });
      if (n) log(`Cleanup: deterministically reclaimed ${n} leaked worktree(s).`);
    } catch (e) {
      log(`Cleanup: deterministic worktree prune failed (${String(e)})`);
    }
  }
  try {
    await agent(
      roleAgent('director', 'cleanup',
        'Prune git worktrees and remove ephemeral workspaces (keep all artifacts).', {
        AVO_REPO, EVAL_DIR,
      }),
      { phase: 'Cleanup', label: 'director:cleanup',
        schema: obj({ cleaned: { type: 'boolean' }, notes: { type: 'string' } }, []) });
    log('Cleanup done.');
  } catch (e) {
    log(`Cleanup role skipped (non-fatal): ${e}`);
  }
  if (RUNTIME_PRUNES_WORKTREES) {
    try { pruneWorktrees(AVO_REPO, null, { all: true }); } catch { /* best-effort */ }
  }
}

// ===========================================================================
// PHASE: Validate — Director independent re-benchmark vs the TRUE baseline.
// ===========================================================================
phase('Validate');
pulseOrchestratorHeartbeat('validate');
const finalFormalSessionId = `final_${Date.now()}`;
const finalReportPath = `${AVO_STATE}/director_final_report.json`;
let rawValidation = null;
try {
  rawValidation = await agent(
    roleAgent('director', 'validate', 'Independently re-benchmark the final patch against the TRUE original baseline.', {
      KERNEL_PATH_ORIG, EVAL_DIR, KERNEL_NAME, SKILL_DIR: WORKFLOW_DIR,
      KERNEL_WORKFLOW_DIR, ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR,
      GPU_ID: GPU_POOL, GPU_POOL, VERIFY_REPEATS,
      TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH, LANGUAGE_CONTRACT,
      IR_LANG, IR_ARCH: A.arch || A.hw || HW, IR_SHAPES,
      ASM_PACK, ASM_ARCH, ASM_SHAPES,
      ASM_WINDOW_KERNEL, ASM_WINDOW_BEGIN, ASM_WINDOW_END, GEAK_CAPTURE_DIR,
      APPLY_TO_ORIGINAL: 'false',
      BENCHMARK_PROTOCOL: FORMAL_EVIDENCE_MODE ? 'paired-abba' : 'repeated',
      PAIRED_BLOCKS: DIRECTOR_PAIRED_BLOCKS,
      KERNEL_ENV_SCRIPT, FINAL_PATCH, REPORTED_GEOMEAN: cumulative, COMMANDMENT,
      FORMAL_SESSION_ID: finalFormalSessionId, REPORT_PATH: finalReportPath,
    }),
    { phase: 'Validate', label: 'director:validate',
      schema: FORMAL_EVIDENCE_MODE ? FORMAL_VALIDATE_SCHEMA : VALIDATE_SCHEMA });
  if (!FORMAL_EVIDENCE_MODE) rawValidation = rawValidation && readAgentReport(finalReportPath);
} catch (e) {
  log(`Validate: Director failed (${String(e)})`);
}
if (!rawValidation) {
  // An unreachable Director means the result is unvalidated, not nonexistent. The
  // committed lineage is a durable fact and the caller can re-validate from the tag,
  // so hand back the contract with an explicit status instead of throwing the run away.
  log('Validate: Director returned no structured result; reporting the committed '
    + `lineage as UNVALIDATED — best=${bestVersion} (${cumulative.toFixed(4)}x screened, `
    + `${commits} commit(s)), patch=${FINAL_PATCH}`);
  const partialBundle = await materializeSourceBundle('validation_unavailable');
  await runFinalCleanup();
  pulseOrchestratorHeartbeat('complete');
  return runContract({
    validation_status: 'validation_unavailable',
    output_bundle: partialBundle && partialBundle.finalized ? partialBundle.output_dir : '',
    output_artifacts: partialBundle && partialBundle.finalized ? partialBundle.artifacts : [],
  });
}

let validation = { ...rawValidation };
try {
  validation = await ingestFormalDirectorReport(
    validation, '', finalFormalSessionId);
  if (FORMAL_EVIDENCE_MODE && validation && validation.benchmark_report) {
    const archived = archiveFormalReport(
      validation.benchmark_report, `${AVO_STATE}/final_validation`);
    validation = {
      ...validation,
      benchmark_report: archived.reportPath,
      artifact: archived.report.artifact,
    };
    log(`Validate: archived authoritative formal evidence at ${archived.reportPath}`);
  }
} catch (e) {
  log(`Validate: authoritative formal report rejected (${String(e)})`);
  validation = validation ? {
    ...validation,
    validation_status: 'infrastructure_unhealthy',
    formal_status: 'infrastructure_unhealthy',
    infrastructure_status: 'infrastructure_unhealthy',
  } : null;
}
// A Director may return a report inside validation_workspace, which Cleanup removes.
// Preserve non-formal evidence too; formal evidence has already been archived above.
if (validation && validation.benchmark_report) {
  try {
    const { copyFileSync, existsSync, mkdirSync } = await import('node:fs');
    const { basename, resolve } = await import('node:path');
    const src = resolve(String(validation.benchmark_report));
    const archiveDir = `${AVO_STATE}/final_validation`;
    mkdirSync(archiveDir, { recursive: true });
    const dst = resolve(archiveDir, basename(src));
    if (!existsSync(src)) {
      log(`Validate: benchmark report does not exist and will not be returned: ${src}`);
      validation.benchmark_report = '';
    } else {
      if (src !== dst) copyFileSync(src, dst);
      validation.benchmark_report = dst;
      log(`Validate: preserved benchmark evidence at ${dst}`);
    }
  } catch (e) {
    log(`Validate: could not preserve benchmark report (${String(e)})`);
    validation.benchmark_report = '';
  }
}
const finalDecision = classifyFinalValidation(
  validation, FORMAL_EVIDENCE_MODE, GATE_OPTS, cumulative);
const finalGateAccepted = finalDecision.accepted;
if (validation && finalGateAccepted) {
  validation.validation_status = 'accepted';
}
if (validation && !finalGateAccepted) {
  validation.validation_status = finalDecision.status;
  validation.applied_to_original = 'false';
  validation.notes = `${validation.notes || ''} Deterministic final gate rejected this result; original not modified.`.trim();
}

let finalAccepted = validation && validation.validation_status === 'accepted';
let finalBundle = null;
if (finalAccepted && (ASM_MODE || IR_MODE)) {
  phase('Finalize');
  pulseOrchestratorHeartbeat('finalize');
  try {
    finalBundle = await agent(
      roleAgent('director', 'finalize_language',
        'Materialize the accepted ASM/IR result into an oracle-verified language output bundle.', {
          EVAL_DIR, KERNEL_NAME, VALIDATION_WORKSPACE: `${EVAL_DIR}/validation_workspace`,
          OUTPUT_DIR: `${EVAL_DIR}/final_artifacts`,
          TARGET_LANGUAGE: TARGET_LANGUAGE_NORM, LANGUAGE_MODE, LANGUAGE_SKILL_PATH,
          ASM_WORKFLOW_DIR, IR_WORKFLOW_DIR, KERNEL_ENV_SCRIPT,
        }),
      { phase: 'Finalize', label: `director:finalize ${LANGUAGE_MODE}`, schema: FINALIZE_SCHEMA });
  } catch (e) {
    log(`Finalize failed (${String(e)})`);
  }
  if (!finalBundle || finalBundle.finalized !== true
      || !Array.isArray(finalBundle.artifacts) || !finalBundle.artifacts.length) {
    finalAccepted = false;
    if (validation) {
      validation.validation_status = 'artifact_mismatch';
      validation.notes = `${validation.notes || ''} Accepted measurement could not be materialized into the required language bundle.`.trim();
    }
  }
}

if (validation && finalAccepted && !ASM_MODE && !IR_MODE
    && String(APPLY_TO_ORIGINAL).toLowerCase() === 'true') {
  const applyResult = await agent(
    roleAgent('director', 'apply',
      'The deterministic final validation and language-finalization gates passed. Apply exactly FINAL_PATCH to the original and re-check integrity.', {
        KERNEL_PATH_ORIG, EVAL_DIR, FINAL_PATCH, KERNEL_NAME,
      }),
    { phase: 'Validate', label: 'director:apply',
      schema: obj({ applied: { type: 'boolean' }, notes: { type: 'string' } }, ['applied']) });
  validation.applied_to_original = applyResult && applyResult.applied ? 'true' : 'false';
  if (!applyResult || !applyResult.applied) {
    validation.validation_status = 'artifact_mismatch';
    finalAccepted = false;
  }
}
if (validation && finalAccepted && (ASM_MODE || IR_MODE)) {
  validation.applied_to_original = 'false';
}

let validationReportPath = '';
async function persistFinalValidation() {
  if (!validation) return false;
  try {
    const { writeFileSync } = await import('node:fs');
    validationReportPath = `${EVAL_DIR}/director_validation.json`;
    writeFileSync(
      validationReportPath,
      `${JSON.stringify(validation, null, 2)}\n`,
      'utf8');
    return true;
  } catch (e) {
    log(`Validate: could not persist deterministic final status (${String(e)})`);
    validationReportPath = '';
    validation.validation_status = 'infrastructure_unhealthy';
    finalAccepted = false;
    return false;
  }
}
await persistFinalValidation();

// Source runs have no language-finalization role, but their accepted patch and
// validation evidence are still a real output bundle. Materialize one
// deterministically so callers do not receive empty output fields.
const mismatchKeep = !!(validation
  && validation.validation_status === 'reproduction_mismatch'
  && Number(validation.director_verified_speedup_geomean) > 0);
if (validation && (finalAccepted || mismatchKeep) && !ASM_MODE && !IR_MODE) {
  finalBundle = await materializeSourceBundle(
    validation.validation_status, validation.benchmark_report, validationReportPath);
  if (finalAccepted && (!finalBundle || finalBundle.finalized !== true)) {
    finalAccepted = false;
    validation.validation_status = 'artifact_mismatch';
    validation.notes = `${validation.notes || ''} Accepted measurement could not be materialized into the source output bundle.`.trim();
    await persistFinalValidation();
  }
}

const reportDirector = finalAccepted || mismatchKeep;
const finalGeomean = reportDirector
  ? validation.director_verified_speedup_geomean : 0;
log(`Validate done. verified geomean=${finalGeomean} status=${validation ? validation.validation_status : '?'}`);

// ===========================================================================
// PHASE: Cleanup — deterministic backstop plus role-owned cache cleanup.
// ===========================================================================
await runFinalCleanup();
pulseOrchestratorHeartbeat('complete');

// ===========================================================================
// Return the team_workflow-compatible contract.
// ===========================================================================
return runContract({
  final_geomean: finalGeomean,
  final_arithmetic: reportDirector ? validation.director_verified_speedup_arithmetic : 0,
  output_bundle: reportDirector && finalBundle && finalBundle.finalized ? finalBundle.output_dir : '',
  output_artifacts: reportDirector && finalBundle && finalBundle.finalized ? finalBundle.artifacts : [],
  validation_status: validation ? validation.validation_status : 'unknown',
  report_path: validationReportPath,
  formal_report_path: validation && validation.benchmark_report
    ? validation.benchmark_report : '',
});
