export const meta = {
  name: 'kernel-lane',
  description: 'SINGLE-LANGUAGE kernel optimization worker (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, and integration. Optimizes ONE kernel in ONE language (mode=optimize) or authors a fresh seed then optimizes it (mode=author). This is the worker invoked per lane by the kernel-workflow dispatcher (kernel_workflow.js) and by e2e_workflow; prefer calling kernel-workflow directly unless you specifically want one unchanged lane. Target: AMD Instinct MI-series GPUs (MI300X/300A/308X/325X on CDNA3 gfx942, MI350X/355X on CDNA4 gfx950 — the target card is auto-detected on-box).',
  whenToUse: 'Internal single-language worker. Prefer the kernel-workflow dispatcher (kernel_workflow.js) as the entry point; invoke this directly only to run one unchanged lane. Pass args.kernel_path (required), args.mode, args.target_language, args.budget, args.gpu_ids, args.gpu_mode, args.task.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies the bottleneck' },
    { title: 'Research', detail: 'OPT-IN (args.dra_enabled): researcher fans research questions out in parallel via native WebSearch/WebFetch, writes a ranked-directions brief the planner seeds from' },
    { title: 'WarmStart', detail: 'search the local experience KB (kb_artifacts/) for the top-3 best patches for this (kernel,language,gfx), validate each through the verify gate, adopt the first that passes [warm_start!=off]' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline, then the TechLead curates ONE distilled card into knowledge/learned/ on a measured win [update_experience!=off]' },
  ],
};

// ---------------------------------------------------------------------------
// Args + defaults. (The script cannot touch the filesystem or read its own
// path; agents do all FS work, and every path is supplied/derived from args —
// nothing about the install location is hard-coded.)
// ---------------------------------------------------------------------------
const A = args || {};
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/model directory)');

// WORKFLOW_DIR = the directory that holds this script + roles/ + knowledge/ + scripts/.
// A JS workflow script can't read its own path, so the caller passes it (it is just the
// dirname of the scriptPath used to launch the workflow).
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js, roles/, knowledge/, scripts/ (i.e. the dirname of this script).');
}
// EXP_ROOT = where timestamped run dirs are written. Default: a sibling "exp/" next to the
// kernel_workflow dir (…/<parent>/kernel_workflow -> …/<parent>/exp). Override with args.exp_root.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

const KERNEL_PATH_ORIG = A.kernel_path;
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// Minimum verified geomean improvement over the cumulative best for a round winner to be COMMITTED
// into the canonical workspace (default 2%). Kept as a knob rather than a hard-coded constant so the
// gate is tunable per run (e.g. raise it on a noisy box, lower it to bank small compounding wins).
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
})();
// Minimum verified speedup for a candidate to enter the round's candidate list (default 1.0 = only a
// candidate that beats the baseline is worth looking at). A knob for the same reason MIN_IMPROVE is:
// a transcription (plain Triton -> Gluon/TileLang/HIP) lands BELOW the comparator by construction, and
// at 1.0 its recovery phase is invisible -- no patch saved, no verify, `winner` null every round. The
// COMMIT gate is separate and still requires beating `cumulative` by MIN_IMPROVE, so a sub-baseline
// candidate can be TRACKED but never BANKED.
const CANDIDATE_FLOOR = (() => {
  const v = parseFloat(A.candidate_floor != null ? A.candidate_floor : 1.0);
  return Number.isFinite(v) && v > 0 ? v : 1.0;
})();
// Rendered into the Optimize prompt, where `${1.0}` would stringify to "1" and silently reword a
// prompt that has always said "geomean>1.0". Keeps the default run byte-identical.
const CANDIDATE_FLOOR_TXT = Number.isInteger(CANDIDATE_FLOOR)
  ? CANDIDATE_FLOOR.toFixed(1) : String(CANDIDATE_FLOOR);
// How far BELOW the best candidate ever seen a round may land and still count as "the search is
// advancing" (default +MIN_IMPROVE = the historical test). A knob because a climb that starts under
// the comparator advances for many rounds without ever clearing `cumulative`, and scoring those as
// stalls ends it at MAX_NO_IMPROVE however much budget was given -- and no static counter substitutes,
// since it would have to pre-guess how many rounds the climb takes. Negative admits a round that gives
// ground (a layout experiment that does not pay is information, not a stall).
const PROGRESS_DELTA = (() => {
  const v = parseFloat(A.progress_delta != null ? A.progress_delta : MIN_IMPROVE);
  return Number.isFinite(v) && v > -1 ? v : MIN_IMPROVE;
})();
// Budget cost of ONE `deep_explore` direction. The deep-explore engineer does far more than a single
// specialist — broad rewrite authority, its own multi-iteration measure→profile→rewrite loop — so it
// is charged more than 1 against the direction budget (default 2). It also always runs in a DEDICATED
// round (no other directions that round), enforced below.
const DEEP_COST = (() => {
  const v = parseInt(A.deep_cost != null ? A.deep_cost : 2, 10);
  return Number.isFinite(v) && v >= 1 ? v : 2;
})();
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Every GPU consumer gets the WHOLE pool rather than a pinned lane. gpu_lock.sh
// resolves a comma spec by flocking whichever lane is free AND idle at acquire
// time, so placement follows what work actually costs instead of an index fixed
// before the cost is known. Measured task costs span 23x on campaign20, and a
// pinned round ends with the slowest LANE, not the slowest task -- at 4-way that
// idles ~43% of lane time. Pinning is also fragile on a shared box: GPU_LIST[0]
// has no fallback when lane 0 has a foreign tenant.
// gpu_mode='pin' restores the pre-scheduler behavior (direction i pinned to GPU_LIST[i % n]).
// It exists ONLY so the scheduler can be A/B'd as a single-variable change against the arm it
// replaced; leaving it out would make the "before" arm unreachable and force the comparison to be
// made across two different scripts, where any other drift would be indistinguishable from the
// scheduler's effect. Default 'pool'.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
const GPU_POOL = GPU_MODE === 'pin' ? GPU_LIST[0] : GPU_LIST.join(',');
const TASK = A.task || '';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();

// --- author mode: when there is NO existing source, write a fresh from-scratch SEED first, then optimize it.
// mode=optimize (default) keeps the exact original behavior (backward compatible). mode=author seeds
// the workspace from an op task dir (immutable oracle + frozen online kernel in baseline_src/), the
// author_engineer writes a passing seed, then the SAME optimize loop runs — always timing against the
// frozen online kernel, never against the seed's own language. KERNEL_KNOWLEDGE_DIR is the AMD authoring
// knowledge base — REFERENCE ONLY (facts/how-to, never decisions; the author always measures regardless). Default:
// sibling perf_knowledge/ so standalone runs use it too; empty if WORKFLOW_DIR is unset (no behavior change).
const MODE = String(A.mode != null ? A.mode : 'optimize').trim() || 'optimize';
const TARGET_LANGUAGE = String(A.target_language != null ? A.target_language : 'triton').trim() || 'triton';
const OP_SPEC = A.op_spec || {};
// When the op will run on the CUDA/HIP-graph-captured decode path (e2e sets op_spec.cuda_graph_safe=true),
// the isolated oracle alone CANNOT catch a kernel that passes iso but host-syncs or lazily-compiles under
// graph capture — the "wins isolated, crashes serving" class (cuda_graph_capture_unsafe / NO_BINARY_FOR_GPU).
// This turns on an OPTIONAL capture+replay smoke in the verify step so that failure is caught at the cheap
// isolated stage. Unset (standalone single-kernel runs / non-graph ops) => byte-identical to before.
const REQUIRE_GRAPH_CAPTURE = !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
// WORKLOAD ALIGNMENT (optional). When the caller supplies the real-workload shape/dtype case
// distribution, the benchmark harness benchmarks EXACTLY those (shape, dtype) cases, weights each
// by its total time contribution in the workload (weight = count * baseline_latency), and the
// optimization target becomes the time-weighted ratio-of-sums instead of an unweighted geomean.
//   workload_spec_path : path to a workload-v1 json (produced by parse_profile.py --workload-out,
//                        or hand-written). The benchmark_engineer reads it (JS can't touch FS).
//   op_spec.workload   : inline cases, same shape as a workload-v1 "kernels[].cases" list (or the
//                        full object). Takes precedence; weight_source becomes "caller".
// Both unset => unweighted behavior, byte-identical to before. Correctness ALWAYS stays on the
// frozen immutable oracle (parity vs baseline_src/, + a recorded reference_io.pt where the task dir
// came from e2e's kernel_extractor); this only shapes the PERFORMANCE measurement.
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const WORKLOAD_SPEC = (OP_SPEC && OP_SPEC.workload) || A.workload || null;
const HAS_WORKLOAD = !!(WORKLOAD_SPEC_PATH ||
  (Array.isArray(WORKLOAD_SPEC) && WORKLOAD_SPEC.length) ||
  (WORKLOAD_SPEC && Array.isArray(WORKLOAD_SPEC.kernels) && WORKLOAD_SPEC.kernels.length));
// PRIMARY-metric selector: prefer the time-weighted number when a workload spec is in play and the
// agent reported one; otherwise fall back to the geomean (unweighted runs => unchanged behavior).
const primSpeedup = (o) => {
  if (!o) return 0;
  const w = o.verified_weighted != null ? o.verified_weighted
          : (o.speedup_weighted != null ? o.speedup_weighted : null);
  if (HAS_WORKLOAD && Number.isFinite(w)) return w;
  const g = o.verified_geomean != null ? o.verified_geomean : o.speedup_geomean;
  return Number.isFinite(g) ? g : 0;
};
// Gate fields like `correctness` / `status` are free strings in the schemas, so an agent legitimately
// answers "PASS - 15/15 draws" and a `=== 'pass'` test silently drops a genuinely verified candidate.
// Match the leading word instead: "PASS - ..." / "passed" gate open, "FAIL"/"did not pass" stay shut.
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);
const KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
// Expert skills = human-authored, validated kernel recipes (perf_knowledge/expert_skills/). ADVISORY
// priors only: a matched `validated` skill is a HIGH-PRIOR author/optimize candidate the planning/author
// roles reproduce, then gate by the isolated A/B vs the oracle — it NEVER overrides measurement. Default
// OFF (opt-in: pass use_expert_skills="true"). When OFF (the default) NOTHING is injected -> byte-identical
// to a build without this feature. When invoked by the e2e layer the flag + dir are passed down.
const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
// Only planning + authoring roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);

// --- Deep Research Agent (DRA) -------------------------------------------------------------------
// OPT-IN: a v4-native research phase that runs AFTER Profile and BEFORE the optimize loop (so the
// COMMANDMENT + baseline profile + analysis exist). The `researcher` persona extracts facts and a
// ranked set of research QUESTIONS; the script fans those out in PARALLEL (each question = one
// hang-guarded agent using native WebSearch/WebFetch), then a synthesis pass writes a ranked
// directions portfolio (deep_search.md / deep_search_brief.md / deep_search.json) into EVAL_DIR that
// the TechLead's plan_round seeds from. DEFAULT OFF: when dra_enabled is not "true" NOTHING runs and
// behavior is byte-identical to a build without this feature (existing runs unchanged).
const DRA_ENABLED = String(A.dra_enabled != null ? A.dra_enabled : 'false') === 'true';
const DRA_MAX_QUESTIONS = (() => {
  const v = parseInt(A.dra_max_questions != null ? A.dra_max_questions : 8, 10);
  return Number.isFinite(v) && v >= 1 ? v : 8;
})();
// Optional Stage 5/6 blindspot-critique pass (one more parallel research wave). Default OFF (budget).
const DRA_BLINDSPOT = String(A.dra_blindspot != null ? A.dra_blindspot : 'false') === 'true';
const DRA_MAX_BLINDSPOTS = (() => {
  const v = parseInt(A.dra_max_blindspots != null ? A.dra_max_blindspots : 4, 10);
  return Number.isFinite(v) && v >= 1 ? v : 4;
})();

// ---------------------------------------------------------------------------
// LEARNED-KNOWLEDGE SINK. Derived from WORKFLOW_DIR and nothing else, so a lane spawned by
// e2e_workflow still curates into kernel_workflow/knowledge/learned/ — e2e's learned/ is a
// separate memory owned by its own system_architect step.
//   update_experience = on (default) | off | false | none
// The bake-off dispatcher passes `off` and curates once centrally instead (N parallel lanes would
// file N near-duplicate cards for one op).
const LEARNED_DIR = `${WORKFLOW_DIR}/knowledge/learned`;
// How much of ONE round the learned KB may steer. Measured, not guessed: an arm whose planner was
// handed at most 3 matched cards and could KB-seed at most one direction reached 4.45x geomean on 16
// kernels; the same 16 under a planner that reads the whole 84-line index and picks freely reached
// 3.44x — a 22.7% regression, with the losses concentrated on kernels where the bounded arm had found
// an unusual win (one went 25.77x -> 2.15x). Both arms averaged ~3 rounds, so it was not lost rounds.
// The index-reading design is better at RECALL; what it lost was the profile-driven direction that
// nothing from the KB had touched — the per-round control. `knowledge/learned/README.md` says the KB
// must be "an accelerant, NOT a cage", and this is the line that keeps that true.
//   KB_DIR_CAP        directions per round that may draw on cards (default 1)
//   KB_COLD_DIRECTION at least one direction per round must be profile-only (default on)
// Whether the planner is being pointed at the learned KB at all. Defined HERE because this lane is
// where it is USED: the constant of this name lives in the dispatcher, and referencing it from the
// lane threw `USE_LEARNED_READ is not defined` at the first plan step of every kernel — 13 of them
// before the run was stopped. A parse test cannot catch that: the file compiles, and the reference
// only resolves when the line runs. Default ON, matching this branch's design where the tech_lead
// role reads INDEX.md unconditionally; `use_learned_kb=false` turns the budget block off with it.
const USE_LEARNED_READ = String(A.use_learned_kb != null ? A.use_learned_kb : 'true') === 'true';
const KB_DIR_CAP = Math.max(0, parseInt(A.kb_dir_cap != null ? A.kb_dir_cap : 1, 10));
const KB_COLD_DIRECTION = String(A.kb_cold_direction != null ? A.kb_cold_direction : 'true') === 'true';
let kbCapBound = 0;      // rounds where the cap actually had to strip something
const UPDATE_EXPERIENCE = String(A.update_experience != null ? A.update_experience : 'on').trim().toLowerCase() || 'on';
const UPDATE_EXPERIENCE_ON = UPDATE_EXPERIENCE !== 'off' && UPDATE_EXPERIENCE !== 'false' && UPDATE_EXPERIENCE !== 'none';

// ---------------------------------------------------------------------------
// WARM-START (local experience KB). The kernel_workflow's machine-produced,
// code-carrying store (kb_artifacts/) — distinct from the human perf_knowledge/
// index. Before the optimize loop, search this store for the top-3 historically
// best patches for THIS (kernel, language, gfx), validate each through the SAME
// verify_engineer gate, and adopt the first that passes as the starting point.
// After Validate, write this run's own win back. Design: KernelForge experience-KB
// lifecycle (../KernelForge/docs/conceptual/experience-kb-lifecycle.md), plan Part 4.
//   warm_start = on (default)   | read + validate top-3, ADOPT the first that passes.
//              = reference       | read top-3 as prose only, never auto-apply.
//              = return_after_read| adopt then RETURN before the optimize loop.
//              = off | false | none | resume-skip | no-arch => cold start (byte-identical).
// kb_artifacts_dir default: sibling of the workflow dir (<repo>/kb_artifacts).
const WARM_START = String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on';
const WARM_START_ON = WARM_START !== 'off' && WARM_START !== 'false' && WARM_START !== 'none';
const WARM_START_REF_ONLY = WARM_START === 'reference';
const WARM_START_RETURN_AFTER = WARM_START === 'return_after_read';
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts' : '')).replace(/\/+$/, '');
const EXPERIENCE_STORE = `${WORKFLOW_DIR}/scripts/experience_store.py`;

// ---------------------------------------------------------------------------
// DEEP-MODE continuation + cross-backend / e2e-feedback hooks. ALL OPTIONAL.
// When none are passed (every normal/fast e2e run, and every standalone run) these are '' / the
// current defaults and are NEVER threaded into a prompt — so behavior is byte-identical to the
// pre-feature build. They are set ONLY by e2e_workflow's deep_mode head scheduler.
//   STATE_DIR        a STABLE dir for THIS (kernel,backend) ACROSS deep waves. When set the run
//                    RESUMES: director seeds the canonical from STATE_DIR/best (the cumulative-best
//                    code) and returns prior_state (cumulative + history) so re-invocation CONTINUES
//                    instead of restarting (no lost experience, no re-explored directions). The frozen
//                    oracle baseline (immutable unittest.py/meta.json) stays the reference, so speedups
//                    remain comparable to the TRUE baseline across waves. update_memory writes STATE.json
//                    + syncs best/ each round.
//   SHARED_KB        cross-backend blackboard file (read by plan+engineers, appended by update_memory).
//   GLOBAL_KB        run-global cross-KERNEL technique blackboard (deep): techniques that generalize
//                    across head ops/backends. Optional; unset (default/fast) => byte-identical prompts.
//   E2E_FEEDBACK     path to the latest end-to-end A/B result + problems from e2e_workflow (engaged?,
//                    cudagraph behavior, mem footprint, decode regression, e2e delta) — steers planning.
//   HARNESS_ADDENDUM path to an e2e-refined harness addendum (timing-weight / cudagraph-capture / hard
//                    constraint gates). The IMMUTABLE oracle is NEVER touched; this only refines what the
//                    perf bench emphasizes so the isolated target aligns with e2e.
//   MAX_NO_IMPROVE   consecutive non-improving rounds before stopping (default 2 = current behavior).
const STATE_DIR = String(A.state_dir || '').replace(/\/+$/, '');
const SHARED_KB = String(A.shared_kb || '').trim();
const GLOBAL_KB = String(A.global_kb || '').trim();   // run-global cross-KERNEL technique blackboard (deep)
const E2E_FEEDBACK = String(A.e2e_feedback || '').trim();
const HARNESS_ADDENDUM = String(A.harness_addendum || '').trim();
// P2 (deep continuation): on a RESUMED wave (STATE_DIR holds prior work) the cold re-derivation of
// Analyze + baseline Profile is largely redundant. INCREMENTAL_RESUME tells those two ADVISORY agents to
// load the prior roadmap/profile and return it with only delta updates instead of re-deriving from
// scratch, so each burst spends its budget on optimization ROUNDS, not re-analysis. Benchmark is NEVER
// incremental (it re-pins a fresh in-window baseline every wave \u2014 the matched-A/B correctness rail).
// Unset (default/fast / first deep burst) => spreading {} adds nothing => byte-identical prompts.
const INCREMENTAL = !!STATE_DIR && String(A.incremental_analyze || '') === 'true';
const RESUME_INPUT = INCREMENTAL ? { INCREMENTAL_RESUME: '1' } : {};
const MAX_NO_IMPROVE = Math.max(1, parseInt(A.max_no_improve != null ? A.max_no_improve : 2, 10));
// Conditional inputs: spreading {} adds NOTHING to a prompt (byte-identical) when a hook is unset.
const KB_INPUTS = {
  ...(SHARED_KB ? { SHARED_KB } : {}),
  ...(GLOBAL_KB ? { GLOBAL_KB } : {}),
  ...(E2E_FEEDBACK ? { E2E_FEEDBACK } : {}),
  ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
};

// ---------------------------------------------------------------------------
// Reusable JSON-schema fragments.
// ---------------------------------------------------------------------------
const perCase = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      baseline_ms: { type: 'number' },
      optimized_ms: { type: 'number' },
      speedup: { type: 'number' },
      // Workload-alignment fields (present only when a WORKLOAD_SPEC drives the harness; absent
      // on a normal unweighted run). weight = this case's baseline time SHARE in the real workload;
      // it is the coefficient of the time-weighted metric Σweight / Σ(weight/speedup). count is
      // optional/informational (regime-attributed cases have no per-call count).
      weight: { type: 'number' },
      count: { type: 'number' },
      dims: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
      dtypes: { type: 'array', items: { type: 'string' } },
      weight_source: { type: 'string' }, // trace | regime | regime_floor | prior | caller
    },
    required: ['name', 'speedup'],
  },
};
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });

const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, workspace: { type: 'string' }, baseline_dir: { type: 'string' },
  kernel_name: { type: 'string' }, source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // Frozen-baseline verdict (BOTH modes). The unittest's timing + random-value parity baseline MUST be
  // the real online kernel — the immutable baseline_src/ dir OR an importable meta.baseline_callable —
  // never kernel_src/ (the candidate's own scaffold). The director sets baseline_frozen=true after it
  // copies baseline_src/ + confirms meta.baseline_callable; the script aborts the run if neither holds.
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  // DEEP-MODE resume only: populated by the director ONLY when STATE_DIR was provided AND a prior best
  // exists there. Lets a continued wave restore its cumulative speedup + insight/ledger history so it
  // does not re-explore dead directions. Absent (undefined) on a fresh run -> no behavior change.
  resumed: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name']);

const AUTHOR_SCHEMA = obj({
  authored: { type: 'boolean' }, target_language: { type: 'string' }, correctness: { type: 'string' },
  baseline_ms: { type: 'number' }, kernel_src_path: { type: 'string' }, entry_point: { type: 'string' },
  build: { type: 'boolean' }, notes: { type: 'string' },
}, ['authored', 'correctness']);

const ANALYZE_SCHEMA = obj({
  kernel_type: { type: 'string' }, kernel_file: { type: 'string' }, entry_point: { type: 'string' },
  modifiable_files: { type: 'array', items: { type: 'string' } },
  bottleneck_guess: { type: 'string' }, roadmap_summary: { type: 'string' },
  candidate_directions: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // perf_knowledge resolution (REFERENCE ONLY): the operator/language this kernel maps to in the
  // AMD perf_knowledge base, plus the most relevant card paths, so engineers read focused context
  // instead of re-navigating the whole base. Empty string / [] / null when no card applies.
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] },
  kk_refs: { type: 'array', items: { type: 'string' } },
}, ['kernel_type', 'roadmap_summary']);

const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  // Workload-aligned outputs: present when a WORKLOAD_SPEC drove case selection + weights.
  // baseline_weighted_total_ms = the baseline time the weights represent (Σ weight_i in time units).
  // The metric is Σ weight_i / Σ (weight_i/speedup_i). workload_aligned flags weights are real (not 1).
  workload_aligned: { type: 'boolean' },
  baseline_weighted_total_ms: { type: 'number' },
  weights_provenance: { type: 'string' }, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

const PROFILE_SCHEMA = obj({
  bottleneck: { type: 'string' }, profiler_used: { type: 'string' }, dispatch_count: { type: 'number' },
  // The accelerator detected on-box (e.g. "MI300X / gfx942 / CDNA3, 304 CU, ~5.3 TB/s"), so the
  // roofline ceiling + grid-sizing advice downstream use the real card instead of an assumed MI300X.
  device: { type: 'string' },
  key_metrics: { type: 'object', additionalProperties: true },
  top_kernels: { type: 'array', items: { type: 'object', additionalProperties: true } },
  top_opportunities: { type: 'array', items: { type: 'string' } },
  summary_path: { type: 'string' }, shift_note: { type: 'string' },
}, ['bottleneck', 'top_opportunities']);

// --- Deep Research Agent (DRA) schemas (opt-in via args.dra_enabled) -------
// Stage 0+1/2: the researcher extracts facts and returns a ranked set of research QUESTIONS the
// script then fans out in parallel. Stages 3/4: one answer per question (run concurrently). Stage 5:
// optional blindspot critique. Stage 7: the ranked-directions portfolio + brief the planner consumes.
const RESEARCH_PLAN_SCHEMA = obj({
  facts: { type: 'object', additionalProperties: true },
  questions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, question: { type: 'string' },
      search_queries: { type: 'array', items: { type: 'string' } },
      rationale: { type: 'string' }, tests_hypothesis: { type: 'string' },
      mode: { type: 'string', enum: ['bottleneck', 'design_space'] },
      rank_score: { type: 'number' },
    }, ['question']),
  },
  notes: { type: 'string' },
}, ['questions']);

const RESEARCH_QUESTION_SCHEMA = obj({
  question_id: { type: 'string' }, question: { type: 'string' },
  mode: { type: 'string' }, tests_hypothesis: { type: 'string' },
  answer: { type: 'string' },
  status: { type: 'string', enum: ['prefer', 'deprioritize', 'reject', 'open'] },
  affected: { type: 'array', items: { type: 'string' } },
  evidence: { type: 'array', items: { type: 'object', additionalProperties: true } },
  taskgen_implications: { type: 'string' }, notes: { type: 'string' },
}, ['answer', 'status']);

const RESEARCH_BLINDSPOT_SCHEMA = obj({
  blindspots: {
    type: 'array',
    items: obj({
      description: { type: 'string' }, why_it_matters: { type: 'string' },
      follow_up_question: { type: 'string' },
    }, ['description']),
  },
}, ['blindspots']);

// Stage 7 final portfolio. `directions` is kept COMPACT here (mirrors deep_search_brief.md): the
// planner reads the brief on disk, so this structured echo is just for logging/validation.
const RESEARCH_SCHEMA = obj({
  num_questions: { type: 'number' }, num_directions: { type: 'number' },
  brief_path: { type: 'string' }, full_path: { type: 'string' }, json_path: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore'] },
      mechanism: { type: 'string' }, expected_upside: { type: 'string' },
      confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
      rank_score: { type: 'number' },
    }, ['title']),
  },
  notes: { type: 'string' },
}, ['num_directions', 'brief_path']);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore'] },
      focus_files: { type: 'array', items: { type: 'string' } },
      expected_speedup: { type: 'number' }, prompt: { type: 'string' },
      kk_refs: { type: 'array', items: { type: 'string' } }, // optional: perf_knowledge card paths for THIS direction (REFERENCE ONLY)
      // Learned cards that SEEDED this direction, by filename. Structural attribution: the planner
      // declares what it opened, the script joins that against what the VERIFIER independently
      // measured. Declared here rather than inferred because the read path is semantic — the planner
      // reads INDEX.md and judges by meaning, so nothing downstream can reconstruct which card it
      // acted on. A citation is not a causal claim (the planner saw the profile too, and one run
      // holds no counterfactual); it is the only way a card can ever LOSE standing.
      learned_refs: { type: 'array', items: { type: 'string' } },
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

const ENG_SCHEMA = obj({
  engineer_id: { type: 'string' }, specialty: { type: 'string' }, task: { type: 'string' },
  strategy: { type: 'string' }, speedup_geomean: { type: 'number' }, speedup_arithmetic: { type: 'number' },
  // Time-weighted ratio-of-sums vs the TRUE baseline (PRIMARY metric when workload_aligned).
  // = Σ weight_i / Σ (weight_i / speedup_i). Omitted on unweighted runs.
  speedup_weighted: { type: 'number' },
  per_case: perCase, status: { type: 'string' }, patch_file: { type: 'string' },
  strategies_tried: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
}, ['status', 'speedup_geomean']);

const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  verified_geomean: { type: 'number' }, verified_arithmetic: { type: 'number' },
  verified_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: { type: 'string' }, notes: { type: 'string' },
  graph_safe: { type: 'string' },
}, ['status', 'verified_geomean']);

const INTEGRATE_SCHEMA = obj({
  attempted: { type: 'boolean' },
  combos_tried: { type: 'array', items: { type: 'object', additionalProperties: true } },
  best: { type: 'object', additionalProperties: true },
  improved_over_best_individual: { type: 'boolean' },
  conclusion: { type: 'string' }, notes: { type: 'string' },
}, ['attempted', 'conclusion']);

const MEMORY_SCHEMA = obj({
  insights: { type: 'array', items: { type: 'string' } },
  ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const COMMIT_SCHEMA = obj({
  committed: { type: 'boolean' }, current_best_diff: { type: 'string' }, note: { type: 'string' },
}, ['committed']);

const REPORT_SCHEMA = obj({
  final_speedup_geomean: { type: 'number' }, final_speedup_arithmetic: { type: 'number' },
  final_speedup_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  rounds: { type: 'number' }, budget_used: { type: 'number' },
  report_path: { type: 'string' }, final_patch: { type: 'string' }, per_case: perCase,
}, ['final_speedup_geomean', 'report_path', 'final_patch']);

const UPDATE_EXPERIENCE_SCHEMA = obj({
  action: { type: 'string' },      // created | merged | skipped
  card_path: { type: 'string' },   // path under knowledge/learned/, or "" if nothing distilled
  key: { type: 'string' }, note: { type: 'string' },
}, []);

const VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' }, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  per_case: perCase, applied_to_original: { type: 'string' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);

// Warm-start resolver output = experience_store.py `resolve` JSON, verbatim.
const WARMSTART_RESOLVE_SCHEMA = obj({
  read_reason: { type: 'string' }, slug: { type: 'string' },
  candidates: {
    type: 'array',
    items: obj({
      rank: { type: 'number' }, slug: { type: 'string' }, speedup: { type: 'number' },
      exp_dir: { type: 'string' }, arch: { type: 'string' },
      patch_path: { type: 'string' }, prose_path: { type: 'string' },
      strategy: { type: 'string' }, status: { type: 'string' },
    }, ['rank', 'patch_path']),
  },
}, ['read_reason']);
// Warm-start writer output = experience_store.py `write` JSON, verbatim.
const WARMSTART_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, reason: { type: 'string' }, slug: { type: 'string' },
  dir: { type: 'string' }, speedup: { type: 'number' },
}, ['written']);

// ---------------------------------------------------------------------------
// Prompt helpers. Every agent reads its role file from WORKFLOW_DIR and the
// relevant knowledge files itself; the script only passes paths + JSON inputs.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// --- Hung-agent guard ------------------------------------------------------
// An agent LLM call that HANGS (no response, no terminal error) blocks a
// parallel()/pipeline() round-barrier forever (observed: engineer agents frozen
// mid-turn wedged the whole optimize round for >30min). The harness resolves
// terminal API errors to null but NOT an indefinite hang. So bound every agent()
// call: if it has not returned after AGENT_TIMEOUT_MS, resolve it to null (which
// every .filter(Boolean)/null-check downstream already tolerates) and let the
// round proceed. VERY generous default (60min): a true hang never returns, so this only fires on a
// hang, NEVER on a legitimately-long agent. Inner agents include benchmark/profile/verify that build
// (hipcc/ninja) and run benches — minutes, well under 60min — plus the LLM-heavy optimize engineers
// (the ones observed hanging). A too-short bound would kill legit long agents (e.g. a slow rocprof or
// build), so keep it large. Cache keys (prompt, opts) are unchanged so resume still works. Falls back
// to raw agent() if setTimeout is unavailable. args.agent_timeout_ms=0 disables.
// API-FAULT TOLERANCE: a transient API failure (gateway 4xx/5xx, rate-limit, dropped connection, the
// model API going down mid-run) must NOT crash the whole workflow. agentT retries the call up to
// AGENT_RETRIES times on a thrown API/agent error, then resolves to null (every .filter(Boolean)/
// null-check downstream — incl. the Director validate + final report — already degrades on null rather
// than exiting). A timeout (hang) resolves null immediately and is NOT retried (a real hang would just
// burn another full timeout window). args.agent_retries tunes the count. If the failure is PERSISTENT
// (e.g. an auth/header requirement the client doesn't send), retries are exhausted then the run
// degrades — re-run with Workflow({resumeFromRunId}) once the client/API is fixed; cached agent results
// make resume cheap.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
async function agentT(p, o) {
  const label = (o && o.label) ? o.label : 'agent';
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    try {
      if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) return await agent(p, o);
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — resolving null so the round proceeds.`);
          resolve(null);
        }, AGENT_TIMEOUT_MS);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      return await Promise.race([
        agent(p, o).then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < AGENT_RETRIES) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${AGENT_RETRIES} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${AGENT_RETRIES} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// WALL-CLOCK DEADLINE (opt-in; absent => byte-identical to a build without it).
//   DEADLINE_EPOCH  unix seconds after which NO new optimization round may start.
//   NO_STOP_S       while more than this many seconds remain, the TechLead may not end the run on
//                   its own convergence gates; it gets exactly one forced re-plan.
// In CODE and not in the caller's prose because the prose version was measured: a "fixed 4h window
// per kernel" comparison whose window had no implementation anywhere realised 1.55h to 15.22h, a
// 9.81x spread, and the comparison it existed to make was void. The loop below is
// `while (dispatched < BUDGET && noImprove < MAX_NO_IMPROVE)` — no time term. Emphatic prompt text
// is not a constraint, it is one more input to a judgement call.
// A workflow script has no clock (Date.now() is unavailable — it would break resume), so the time is
// read by a tiny low-effort agent. If that read FAILS the run is NOT killed: an unreadable clock
// returns Infinity, degrading to the old budget-only behaviour rather than truncating a healthy run.
const DEADLINE_EPOCH = (() => {
  const v = parseInt(A.deadline_epoch != null ? A.deadline_epoch : 0, 10);
  return Number.isFinite(v) && v > 0 ? v : 0;
})();
const NO_STOP_S = Math.max(0, parseInt(A.no_stop_s != null ? A.no_stop_s : 900, 10));
// Run-level cap on how many TechLead stops may be refused (forcedReplans is not reset per round).
// Not 1: the refusal must scale with the window, or a long budget is handed back early.
// Not unbounded: a role that truly cannot name a direction would spend the window planning.
const MAX_FORCED_REPLANS = Math.max(1, parseInt(A.max_forced_replans != null ? A.max_forced_replans : 6, 10));
const CLOCK_SCHEMA = { type: 'object', properties: { epoch: { type: 'number' } },
                       required: ['epoch'], additionalProperties: true };
let deadlineHit = false;
let forcedReplans = 0;

async function secondsLeft(tag) {
  if (!DEADLINE_EPOCH) return Infinity;
  const r = await agentT(
    `Run EXACTLY this command and nothing else:
\`\`\`bash
date +%s
\`\`\`
Return {"epoch": <the integer it printed>}. Do NOT modify any file and do NOT run anything else.`,
    { phase: 'Optimize', label: `clock ${tag}`, effort: 'low', schema: CLOCK_SCHEMA });
  if (!r || !Number.isFinite(r.epoch)) {
    log(`  [deadline] clock read failed at ${tag} — treating time as unlimited for this check.`);
    return Infinity;
  }
  return DEADLINE_EPOCH - r.epoch;
}

// Expert-skills injection. PURELY ADDITIVE: '' when OFF or the role is not a skills consumer, so both
// call sites (roleAgent, and the inline Optimize prompt) are byte-identical to the pre-feature build in
// those cases. When ON, appends an advisory pointer telling the agent to Read the fragment + query the
// skills index (scripts have no fs access).
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills) and whose validation_status is \`validated\`, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.`;
}

function roleAgent(role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role);
}

// ===========================================================================
// PHASE: Setup
// ===========================================================================
phase('Setup');
const setup = await agentT(
  roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC,
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const KERNEL_NAME = setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);

// ---------------------------------------------------------------------------
// Enforce a FROZEN REAL-ONLINE BASELINE in BOTH modes (author AND same-language
// optimize). The immutable unittest times + parity-checks the candidate against
// baseline_src/ / meta.baseline_callable (the live online kernel); if neither
// exists it would silently fall back to timing kernel_src/ against itself — the
// "optimized-HIP vs naive-HIP = fake 15.7×" bug this harness exists to prevent.
// The script has no FS access, so we trust the director's structured verdict
// (it copied baseline_src/ + confirmed the callable). Missing -> abort/re-extract.
// ---------------------------------------------------------------------------
const hasBaseline = setup.baseline_frozen === true ||
  (typeof setup.baseline_callable === 'string' && setup.baseline_callable.trim().length > 0);
if (!hasBaseline) {
  const reason = `no frozen baseline (baseline_src/ or meta.baseline_callable) for ${KERNEL_NAME} — ` +
    `re-extract; refusing to time the candidate against kernel_src/ (fake-win risk)`;
  log(`Setup ABORT: ${reason}`);
  return {
    mode: MODE, authored: false, target_language: TARGET_LANGUAGE,
    eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
    final_geomean: 0, final_patch: '', validation_status: 'no_baseline', reason,
  };
}

// ===========================================================================
// PHASE: Author (mode=author only) — write a fresh from-scratch impl as the
// optimize loop's CODE SEED. On success, HEAD of CANONICAL becomes that seed
// (what the optimize loop diffs its edits against) and the rest of the pipeline
// (Analyze/Benchmark/Profile/optimize loop) runs UNCHANGED on it. The SPEEDUP
// denominator is NEVER the seed — it is the frozen REAL ONLINE kernel in
// baseline_src/ (meta.baseline_callable), regardless of TARGET_LANGUAGE.
// On failure (no correct seed), abort early with a structured result so the
// e2e caller drops this language.
// ===========================================================================
if (MODE === 'author') {
  phase('Author');
  const authored = await agentT(
    roleAgent('author_engineer', 'author', 'Write the simplest correct baseline in the target language.', {
      TARGET_LANGUAGE, OP_SPEC, WORKSPACE: CANONICAL, TASK_DIR: KERNEL_PATH_ORIG,
      GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_KNOWLEDGE_DIR,
      // The author engineer reads the learned index too, so the switch has to reach it. It is the
      // second reader; a switch that covers one of two readers is not a switch.
      LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
    }),
    { phase: 'Author', label: `author:${TARGET_LANGUAGE}`, schema: AUTHOR_SCHEMA });
  if (!authored || !authored.authored || !says(authored.correctness, 'pass')) {
    log(`Author mode FAILED for ${TARGET_LANGUAGE}: ${authored ? authored.notes || authored.correctness : 'no result'}. Aborting (no seed to optimize).`);
    return {
      mode: 'author', authored: false, target_language: TARGET_LANGUAGE,
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_geomean: 0, final_patch: '', validation_status: 'author_failed',
      reason: authored ? authored.notes || 'author produced no correct baseline' : 'author returned nothing',
    };
  }
  log(`Author mode: ${TARGET_LANGUAGE} seed written (correct, seed ${authored.baseline_ms || '?'} ms; denominator = frozen online kernel). Optimizing it now.`);
}

// ===========================================================================
// PHASE: Analyze + Roadmap (TechLead)
// ===========================================================================
phase('Analyze');
const analysis = await agentT(
  roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR,
    ...RESUME_INPUT,
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: ANALYZE_SCHEMA });
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

// perf_knowledge pointers resolved by the TechLead in analyze (REFERENCE ONLY; threaded to the
// planner + engineers so they read focused op/language cards instead of the whole base). Empty when
// no operator card applies (e.g. point-cloud HIP ops) or KERNEL_KNOWLEDGE_DIR is unset → no change.
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || '';
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];

// ===========================================================================
// PHASE: Benchmark setup (Benchmark Engineer)
// ===========================================================================
phase('Benchmark');
const bench = await agentT(
  roleAgent('benchmark_engineer', 'setup', 'Build the COMMANDMENT and record a reliable baseline.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    ANALYSIS: analysis,
    ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
    ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
    ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
  }),
  { phase: 'Benchmark', label: 'benchmark_engineer', schema: BENCH_SCHEMA });
if (!bench || !bench.baseline_per_case) throw new Error('Benchmark setup failed: no baseline recorded');
const BASELINE_PER_CASE = bench.baseline_per_case;
const BASELINE_GEOMEAN_MS = bench.baseline_geomean_ms;
log(`Benchmark done. ${bench.num_test_cases || BASELINE_PER_CASE.length} cases, baseline geomean ${BASELINE_GEOMEAN_MS} ms, reliable=${bench.reliable}`);

// ===========================================================================
// PHASE: Baseline profile (Profile Engineer)
// ===========================================================================
phase('Profile');
let profileSummary = await agentT(
  roleAgent('profile_engineer', 'baseline', 'Profile the baseline and classify the bottleneck.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: 0,
    COMMANDMENT,
    ...RESUME_INPUT,
  }),
  { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
log(`Baseline bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'} (dispatch_count=${profileSummary ? profileSummary.dispatch_count : '?'})`);

// ===========================================================================
// PHASE: Research (Deep Research Agent — OPT-IN via args.dra_enabled)
// Runs AFTER Profile / BEFORE the optimize loop: profile + COMMANDMENT + analysis exist by now, so
// the researcher has the facts it needs. It produces EVAL_DIR/deep_search_brief.md (compact, ranked
// directions) which the TechLead's plan_round seeds directions from. The per-question research is
// fanned out with parallel() and EVERY research agent is wrapped in the agentT() hang-guard, so a
// hung research agent resolves to null and the parallel round-barrier still proceeds (it never wedges
// the run — the known v4 failure mode the hang-guard was built for). DEFAULT OFF → no behavior change.
// ===========================================================================
let researchBriefPath = '';   // EVAL_DIR/deep_search_brief.md when the DRA produced one; '' otherwise
if (DRA_ENABLED) {
  phase('Research');
  const RESEARCH_DIR = `${EVAL_DIR}/research`;

  // --- Stage 0 + 1/2: extract facts, generate + rank research questions (one agent) -------------
  const plan = await agentT(
    roleAgent('researcher', 'research_plan', 'Extract facts and propose ranked research questions.', {
      WORKSPACE: CANONICAL, EVAL_DIR, RESEARCH_DIR, COMMANDMENT, SKILL_DIR: WORKFLOW_DIR,
      ANALYSIS_JSON: `${EVAL_DIR}/analysis.json`, CODEBASE_CONTEXT: `${EVAL_DIR}/codebase_context.md`,
      PROFILING_SUMMARY: profileSummary ? profileSummary.summary_path : '',
      BOTTLENECK: profileSummary ? profileSummary.bottleneck : 'unknown',
      BASELINE_PER_CASE, MAX_QUESTIONS: DRA_MAX_QUESTIONS, TASK,
      KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
    }),
    { phase: 'Research', label: 'researcher:plan', schema: RESEARCH_PLAN_SCHEMA });

  const facts = (plan && plan.facts) || {};
  let questions = (plan && Array.isArray(plan.questions) ? plan.questions : [])
    .slice(0, DRA_MAX_QUESTIONS)
    .map((q, i) => ({ ...q, id: q.id || `q${i}` }));
  log(`Research: ${questions.length} question(s) planned, bottleneck=${facts.bottleneck_type || '?'}`);

  // --- Stages 3/4: research each question IN PARALLEL (native WebSearch/WebFetch), hang-guarded ---
  // parallel() takes an array of zero-arg async thunks and runs them concurrently; each thunk's
  // agentT() bounds the research agent so a hang resolves null instead of blocking the barrier.
  const researchQuestion = (q, stageLabel) => agentT(
    roleAgent('researcher', 'research_question',
      'Research THIS ONE question on the live web and synthesize one judgment.', {
        QUESTION: q, FACTS: facts, RESEARCH_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
        ANSWER_OUT: `${RESEARCH_DIR}/answers/${q.id}.json`,
        CODEBASE_CONTEXT: `${EVAL_DIR}/codebase_context.md`,
        PROFILING_SUMMARY: profileSummary ? profileSummary.summary_path : '',
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
      }),
    { phase: 'Research', label: `researcher:q ${q.id}${stageLabel ? ' ' + stageLabel : ''}`, schema: RESEARCH_QUESTION_SCHEMA });

  let answers = (await parallel(questions.map((q) => () => researchQuestion(q))))
    .filter(Boolean);
  log(`Research: ${answers.length}/${questions.length} first-pass answers returned`);

  // --- Stages 5/6 (optional): blindspot critique + a second parallel research wave --------------
  if (DRA_BLINDSPOT && answers.length) {
    const crit = await agentT(
      roleAgent('researcher', 'research_blindspot', 'Critique the research and surface new blindspots.', {
        FACTS: facts, RESEARCH_DIR, ANSWERS: answers, MAX_BLINDSPOTS: DRA_MAX_BLINDSPOTS,
        SKILL_DIR: WORKFLOW_DIR,
      }),
      { phase: 'Research', label: 'researcher:blindspot', schema: RESEARCH_BLINDSPOT_SCHEMA });
    const followups = (crit && Array.isArray(crit.blindspots) ? crit.blindspots : [])
      .filter(b => b && b.follow_up_question)
      .slice(0, DRA_MAX_BLINDSPOTS)
      .map((b, i) => ({ id: `b${i}`, question: b.follow_up_question, mode: 'bottleneck',
        tests_hypothesis: '', search_queries: [] }));
    if (followups.length) {
      const more = (await parallel(followups.map((q) => () => researchQuestion(q, 'blindspot'))))
        .filter(Boolean);
      answers = answers.concat(more);
      log(`Research: blindspot pass added ${more.length} answer(s) from ${followups.length} follow-up(s)`);
    }
  }

  // --- Stage 7: synthesize the ranked-directions portfolio + the compact planner brief -----------
  const synth = await agentT(
    roleAgent('researcher', 'research_synthesize',
      'Synthesize the ranked portfolio of optimization directions and the compact brief.', {
        FACTS: facts, RESEARCH_DIR, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
        BRIEF_OUT: `${EVAL_DIR}/deep_search_brief.md`, FULL_OUT: `${EVAL_DIR}/deep_search.md`,
        JSON_OUT: `${EVAL_DIR}/deep_search.json`,
      }),
    { phase: 'Research', label: 'researcher:synthesize', schema: RESEARCH_SCHEMA });

  if (synth && synth.brief_path) {
    researchBriefPath = synth.brief_path;
    log(`Research done. ${synth.num_directions || (synth.directions ? synth.directions.length : 0)} ranked direction(s) → ${researchBriefPath}`);
  } else {
    log('Research produced no brief (degraded) — plan_round proceeds without a DRA brief.');
  }
}

// ===========================================================================
// PHASE: Optimization loop (budget-controlled)
// ===========================================================================
let dispatched = 0;          // counts ONLY optimization-direction engineers (the budget)
let round = 0;
let cumulative = 1.0;        // best verified geomean speedup vs the TRUE baseline
let bestSeen = 0;            // best verified geomean of any candidate, committed or not
let noImprove = 0;
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source}
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };
// One row per KB-seeded direction, joined against what the verifier measured (see the push site in
// the round loop). Fed to update_experience and returned to the caller: a driver aggregating these
// is how anyone notices the KB has been cited fifty times and never once carried a round.
const citations = [];

// DEEP-MODE resume: restore cumulative speedup + insight/ledger history from the prior wave so this
// continuation builds ON the cumulative best (canonical was already seeded from STATE_DIR/best by the
// director) and does not re-explore dead directions. No-op on a fresh run (prior_state undefined).
if (setup.resumed && setup.prior_state) {
  const ps = setup.prior_state;
  if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
  if (Array.isArray(ps.insights)) history.insights = ps.insights;
  if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
  if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
  if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
}

// ===========================================================================
// PHASE: WarmStart — search the local experience store for the top-3 best patches
// for THIS (kernel, language, gfx), validate each through the SAME verify_engineer
// gate as a round winner, and ADOPT the first that passes as the starting point.
// The recorded speedup only ranks; adoption is decided by a FRESH measurement here.
// Skipped (cold start; run byte-identical to pre-feature) when warm_start=off, a
// STATE_DIR resume is active, no arch was detected, or the store has nothing.
// gfx is read from the baseline profile's on-box `device` string (no extra probe).
// ===========================================================================
const GFX = (String((profileSummary && profileSummary.device) || '').match(/gfx\d+/i) || [''])[0].toLowerCase();
let warm_start = { adopted: false, read_reason: WARM_START_ON ? 'read' : 'disabled', candidates: [] };
let skipLoop = false;
if (WARM_START_ON && !setup.resumed && KB_ARTIFACTS_DIR) {
  phase('WarmStart');
  if (!GFX) {
    warm_start.read_reason = 'missing_arch';
    log('[kb] warm-start skipped: no gfx detected from the baseline profile device string.');
  } else {
    const resolved = await agentT(
      `You are the warm-start resolver. Run EXACTLY this command and return its single-line JSON stdout ` +
      `verbatim as StructuredOutput — do not add, drop, reorder, or reinterpret any field:
\`\`\`bash
python3 ${EXPERIENCE_STORE} resolve --root ${KB_ARTIFACTS_DIR} \\
  --kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\
  --gfx ${GFX} --top-n 3 --refs-dir ${EVAL_DIR}/kb_references
\`\`\``,
      { phase: 'WarmStart', label: 'warm_start:resolve', schema: WARMSTART_RESOLVE_SCHEMA }) || {};
    warm_start.read_reason = resolved.read_reason || 'read';
    warm_start.slug = resolved.slug || '';
    const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
    warm_start.candidates = cands.map(c => ({ rank: c.rank, slug: c.slug, speedup: c.speedup, status: 'read' }));
    log(`[kb] experience read: slug=${resolved.slug || '?'} reason=${warm_start.read_reason} candidates=${cands.length}`);
    // reference-only: prose is already mirrored to EVAL_DIR/kb_references by the resolver; do not apply.
    if (!WARM_START_REF_ONLY) {
      for (const c of cands) {                              // already rank-ordered (fastest first)
        const rec = warm_start.candidates.find(x => x.rank === c.rank);
        const ver = await agentT(
          roleAgent('verify_engineer', 'verify',
            'Validate a HISTORICAL warm-start patch — the SAME gate as any round candidate. FIRST, before ' +
            'applying, check that EVERY path this patch touches maps into the editable set (see EDITABLE_SET ' +
            'input) at some strip depth; if none maps, return status:"apply_failed", notes "patch_outside_' +
            'editable_set", and DO NOT apply. Apply with `git apply "$PATCH" || git apply --3way "$PATCH"`. ' +
            'On ANY failure restore the working copy fully (git checkout -- . and delete untracked files) ' +
            'before returning so the next candidate starts from a clean tree.', {
            CANONICAL, PATCH: c.patch_path, VERIFY_DIR: `${EVAL_DIR}/warm_start/cand_${c.rank}`,
            EDITABLE_SET: (analysis && analysis.modifiable_files) || [],
            GPU_ID: GPU_LIST[0], SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
            ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
            ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
          }),
          { phase: 'WarmStart', label: `warm_start:verify c${c.rank}`, schema: VERIFY_SCHEMA });
        const sp = primSpeedup(ver);
        if (ver && says(ver.status, 'verified') && says(ver.correctness, 'pass') && sp > 1.0) {
          const adopt = await agentT(
            `You are the TechLead adopting a validated warm-start patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
git apply ${c.patch_path} || git apply --3way ${c.patch_path}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "warm-start adopt: ${c.slug} (${sp.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH applies fail, apply manually to match intent, then add -A + commit and RE-RUN the COMMANDMENT
correctness check; only report committed=true if it still passes. Return JSON {committed, current_best_diff, note}.`,
            { phase: 'WarmStart', label: `warm_start:adopt c${c.rank}`, schema: COMMIT_SCHEMA });
          if (adopt && adopt.committed) {
            // Adopt: the optimize loop now builds ON this patch. cumulative starts at the adopted Nx (vs the
            // pristine frozen baseline), so a run that improves nothing still reports total=Nx — the KB gain
            // is attributed to history, never to this run's own rounds (KernelForge total vs incremental).
            cumulative = sp;
            bestPerCase = (ver.per_case && ver.per_case.length) ? ver.per_case : bestPerCase;
            finalWinner = { source: `warm_start:${c.slug}`, geomean: sp,
              arithmetic: ver.verified_arithmetic || sp, per_case: bestPerCase, patch: c.patch_path };
            if (bestSeen < sp) bestSeen = sp;
            warm_start.adopted = true;
            warm_start.adopted_speedup = sp;
            warm_start.slug = c.slug;
            warm_start.total_speedup = sp;                  // relative to the pristine frozen baseline
            if (rec) rec.status = 'adopted';
            log(`[kb] warm-start ADOPTED ${c.slug} @ ${sp.toFixed(2)}x — optimizing from the patched state.`);
            profileSummary = await agentT(
              roleAgent('profile_engineer', 'reprofile',
                'Re-profile the adopted warm-start state and classify the new bottleneck.', {
                WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_LIST[0], ROUND: 0,
                COMMANDMENT, PREVIOUS_METRICS: profileSummary,
              }),
              { phase: 'WarmStart', label: 'warm_start:reprofile', schema: PROFILE_SCHEMA }) || profileSummary;
            if (history) history.bottleneck_now = profileSummary ? profileSummary.bottleneck : history.bottleneck_now;
            if (WARM_START_RETURN_AFTER) { skipLoop = true; warm_start.returned_after_read_kb = true; }
            break;
          }
        }
        if (rec && rec.status !== 'adopted') rec.status = (ver && ver.status) ? `rejected:${ver.status}` : 'rejected:apply_failed';
        log(`[kb] warm-start candidate c${c.rank} ${rec ? rec.status : 'rejected'} (${sp ? sp.toFixed(2) + 'x' : 'no measure'}).`);
      }
    }
  }
}

while (!skipLoop && dispatched < BUDGET && noImprove < MAX_NO_IMPROVE) {
  // --- (0) HARD STOP: no new round may START past the deadline ----------
  // Checked BEFORE round++ so an expired check does not inflate the round count. The in-flight round
  // is never interrupted — a killed round leaves a half-verified patch and no report.
  let left = await secondsLeft(`pre-r${round + 1}`);
  if (left <= 0) {
    deadlineHit = true;
    log(`WALL-CLOCK DEADLINE reached after ${round} round(s) (budget used ${dispatched}/${BUDGET}) — ` +
        `starting no further rounds; going straight to report + validation.`);
    break;
  }
  if (DEADLINE_EPOCH) log(`[deadline] ${(left / 60).toFixed(0)} min remain before the hard stop.`);

  round++;
  const remaining = BUDGET - dispatched;
  phase('Optimize');

  // --- (a) Plan the round (TechLead) ------------------------------------
  const planInputs = (forbidStop) => ({
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
    KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
    ...KB_INPUTS,
    // DRA brief (REFERENCE), from main. plan_round reads it and seeds directions[] from the ranked
    // DRA directions — see tech_lead.md plan_round. Spread conditionally, so when dra_enabled was off
    // (or the research degraded) the key is absent and the prompt is byte-identical to a build
    // without the feature. Threaded through `planInputs` rather than the call site so the forced
    // re-plan below gets it too: a replan that silently loses the research brief would plan against
    // a different input set than the round it is replacing.
    ...(researchBriefPath ? { DEEP_SEARCH_BRIEF: researchBriefPath } : {}),
    // Stated UNCONDITIONALLY, in both directions. `use_learned_kb=false` used to omit the budget and
    // nothing else, while roles/tech_lead.md still listed knowledge/learned/INDEX.md among the files
    // to read — so the off switch removed the budget that CONSTRAINS the KB and left the instruction
    // that USES it, i.e. the arm it exists to produce (a clean KB-off control) was never clean. The
    // role now gates on this value, so it has to be present whichever way it points.
    LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
    ...(USE_LEARNED_READ ? { LEARNED_KB_BUDGET:
      `At most ${KB_DIR_CAP} of this round's directions may draw on learned cards, and you must cite ` +
      `the cards you used in that direction's \`learned_refs\`. ` +
      (KB_COLD_DIRECTION ? `At least ONE direction must be planned from the profile alone, with an ` +
        `empty \`learned_refs\` — it is this round's control, and without it a round cannot tell you ` +
        `whether the cards helped or merely crowded out what you would have tried anyway. ` : '') +
      `Read as much of the index as you like; the budget is on how much of the ROUND the KB steers, ` +
      `not on what you may look at.` } : {}),
    // Both keys absent unless a deadline was passed => byte-identical prompt when OFF.
    ...(DEADLINE_EPOCH ? { MINUTES_REMAINING: Math.round(left / 60) } : {}),
    ...(forbidStop ? { STOP_NOT_PERMITTED:
      `You returned stop=true with ${Math.round(left / 60)} min of the wall-clock window still ` +
      `unspent. This run is a FIXED-WINDOW measurement: ending early is an invalid data point no ` +
      `matter how good the number or how well argued the rationale, and your role's own convergence ` +
      `gates (launch floor closed, gain under the promotion bar, roofline %, "only a different tech ` +
      `stack is left") do NOT authorise stopping here. Plan at least one direction. Switch lever ` +
      `rather than re-running an exhausted one — algorithm / memory / compute / host_runtime, or a ` +
      `dedicated deep_explore ground-up rewrite. If every direction really is exhausted, spend the ` +
      `round RE-MEASURING the committed best to tighten its error bars and say so. Sizing: keep the ` +
      `round finishable inside the remaining ${Math.round(left / 60)} min.` } : {}),
  });

  let plan = await agentT(
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).',
      planInputs(false)),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });

  // NO EARLY STOP: keep refusing while the window is materially unspent. This used to refuse ONCE,
  // on the reasoning that a role repeating "stop" has run out of ideas. Measured at a 2h budget that
  // cost ~3 min (write_req_to_token_pool_triton stopped at 1.95h/2.00h) and looked harmless; the cost
  // is proportional to the budget, so at 8h the same one-shot escape hands back hours. The caller's
  // rule is "must keep optimizing until the budget is spent", so the refusal has to be as long-lived
  // as the window. Bounded by MAX_FORCED_REPLANS so a role that cannot produce a direction at all
  // degrades to stopping instead of spinning the window away on planning.
  while (DEADLINE_EPOCH && left > NO_STOP_S && forcedReplans < MAX_FORCED_REPLANS &&
         (!plan || plan.stop || !plan.directions || plan.directions.length === 0)) {
    forcedReplans++;
    log(`Round ${round}: stop=true with ${(left / 60).toFixed(0)} min left — refusing (#${forcedReplans}), re-planning.`);
    plan = await agentT(
      roleAgent('tech_lead', 'plan_round', 'Your stop was refused: plan at least one direction.',
        planInputs(true)),
      { phase: 'Optimize', label: `tech_lead:replan r${round}#${forcedReplans}`, schema: PLAN_SCHEMA });
    left = await secondsLeft(`replan-r${round}`);
  }

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    break;
  }

  let directions = plan.directions.slice(0, remaining).map((d, i) => ({
    ...d,
    idx: i,
    id: d.id || `r${round}_d${i}`,
    gpu_id: GPU_MODE === 'pin' ? GPU_LIST[i % GPU_LIST.length] : GPU_POOL,
    out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
  }));
  // deep_explore is a DEDICATED-ROUND, heavyweight mandate: if the plan includes one, run ONLY it this
  // round (its broad ground-up rewrite touches many files and can't be merged with specialist patches),
  // and charge DEEP_COST against the budget. Otherwise each specialist direction costs 1.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  log(`Round ${round}: ${directions.length} direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    (d) => {
      const isDeep = d.specialty === 'deep_explore';
      // deep_explore reads its own role (broad authority + own iteration loop); specialists read engineer.md.
      const readLine = isDeep
        ? `Then Read ${WORKFLOW_DIR}/roles/deep_engineer.md and the knowledge files under ${WORKFLOW_DIR}/knowledge/ ` +
          (USE_LEARNED_READ ? '' : `— EXCLUDING knowledge/learned/, which is off for this run: do not open ` +
            `INDEX.md or any card there. `) +
          `(you have broad authority — combine algorithm + memory + compute + host_runtime levers in one ` +
          `coherent rewrite), and follow them. You MAY edit ANY modifiable source (kernel + Python wrapper ` +
          `+ C++ binding), not just focus_files. Run your OWN multi-iteration measure→(self-)profile→rewrite ` +
          `loop and push to the TARGET; keep the best correct version.`
        : `Then Read ${WORKFLOW_DIR}/roles/engineer.md and ${WORKFLOW_DIR}/knowledge/self_monitoring.md and the ` +
          `knowledge files for your specialty, and follow them.`;
      return agentT(
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
First create YOUR private workspace, then optimize.
\`\`\`bash
# Fresh, ISOLATED workspace via tar-copy that EXCLUDES build artifacts (.git/build/__pycache__/.torch_ext/
# *.so/*.o) — no 'rm' anywhere. Each engineer's out_dir is unique per (round,engineer), so the workspace
# is clean on creation; the tar excludes mean no stale build cache is ever inherited (torch .torch_ext
# stores ABSOLUTE paths, so excluding it forces each workspace to build its own fresh). The big immutable
# golden (reference_io.pt, when present) lives in CANONICAL as an absolute symlink; this tar carries the
# symlink verbatim, so every workspace shares the one physical file — NEVER add -h/--dereference here.
mkdir -p ${d.out_dir}/workspace
( cd ${CANONICAL} && tar --exclude=./.git --exclude='*/.git' --exclude=./build --exclude='*/build' \\
    --exclude=./__pycache__ --exclude='*/__pycache__' --exclude=./.torch_ext --exclude='*/.torch_ext' \\
    --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd ${d.out_dir}/workspace && tar -xf - )
\`\`\`
${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git diff > ${d.out_dir}/best_patch.diff\` when geomean>${CANDIDATE_FLOOR_TXT}.

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
        ...(isDeep ? { TARGET: d.expected_speedup ? `reach ${d.expected_speedup}x (or ~90% of the roofline ceiling), whichever is the harder bar` : 'reach ~90% of the roofline ceiling' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.` +
      // Built inline, not via roleAgent(), so the injection has to be appended here too.
      expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer'),
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA }
    ).then((eng) => ({ d, eng }));
    },

    (prev) => {
      const { d, eng } = prev;
      const patch = `${d.out_dir}/best_patch.diff`;
      // Harvest is MEASUREMENT-anchored, not return-value-anchored. An engineer only writes
      // best_patch.diff when it beat the floor (the Optimize prompt: "Save best_patch.diff ... when
      // geomean>CANDIDATE_FLOOR"), so a lost/failed StructuredOutput does NOT imply there is no winning patch: an
      // engineer that died, timed out, or mis-returned can still have left an applies-clean above-floor diff
      // on disk (observed in a bake-off: a 1.56x Triton patch was silently dropped because its
      // worker_result.json/StructuredOutput never came back). The only return we can TRUST to suppress
      // a patch is a clean below-floor one — the engineer ran, measured, honestly reported <=the floor, and
      // therefore wrote no patch. In every other case (null/failed return, OR a claimed above-floor one) a patch
      // MIGHT be on disk, so we hand the path to verify and let the oracle be the source of truth. We
      // cannot stat the file from the workflow sandbox, so the "is there actually a patch" decision is
      // delegated to verify, which returns apply_failed on an absent/empty patch — dropped by the
      // `verified` filter below, i.e. the same outcome as skipping, but with no false loss.
      const trustworthyBelowBaseline = eng && eng.status !== 'failed' && !(primSpeedup(eng) > CANDIDATE_FLOOR);
      if (trustworthyBelowBaseline) {
        return { d, eng, ver: null };
      }
      const recovered = !eng || eng.status === 'failed';
      if (recovered) log(`Round ${round} dir ${d.id}: engineer return ${eng ? 'failed' : 'missing'} — ` +
        `sending best_patch.diff to verify anyway (oracle decides).`);
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}${recovered ? ' (recovered)' : ''}`, schema: VERIFY_SCHEMA }
      ).then((ver) => ({ d, eng, ver, patch }));
    }
  );

  const clean = results.filter(Boolean);
  const verified = clean.filter(r => r.ver && says(r.ver.status, 'verified') &&
    says(r.ver.correctness, 'pass') && primSpeedup(r.ver) > CANDIDATE_FLOOR);

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the PRIMARY metric used for sorting/gating/cumulative: the time-weighted
  // ratio-of-sums when a workload spec is active, else the unweighted geomean (unchanged behavior).
  // The raw unweighted geomean is retained separately for the report.
  let candidates = verified.map(r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: primSpeedup(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch,
  }));

  let integrate = null;
  if (verified.length >= 2) {
    phase('Merge');
    integrate = await agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          files: r.d.focus_files || [], patch: r.patch })),
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    const integPrim = integrate && integrate.best ? primSpeedup({
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
    }) : 0;
    if (integrate && integrate.conclusion === 'improved' && integrate.best &&
      integPrim > Math.max(...candidates.map(c => c.geomean))) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
      });
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const winner = candidates[0] || null;
  const improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  // Separate question from `improved`: is the SEARCH advancing, not did it beat the incumbent. The
  // `bestSeen > 0` guard keeps round 1 deciding on `improved` alone; from then on bestSeen >= cumulative
  // at the default floor, so at the default PROGRESS_DELTA this implies `improved` and changes nothing.
  // A round with NO candidate is never progress, so a dead round still counts against MAX_NO_IMPROVE.
  const madeProgress = !!(winner && bestSeen > 0 && winner.geomean > bestSeen * (1 + PROGRESS_DELTA));

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH \`git apply\` and \`git apply --3way\` fail, inspect the patch and apply it manually (edit the
files to match the patch's intent), then \`add -A\` + commit. The applied source is NOT guaranteed to
match the patch verbatim after a hand-merge, so after committing, RE-RUN the COMMANDMENT correctness
check (cd ${CANONICAL} && the COMMANDMENT CORRECTNESS cmd via gpu_lock); only report committed=true if
it still passes. (When a clean \`git apply\`/\`--3way\` succeeds, correctness was already verified and a
re-check is not required.) Return JSON {committed, current_best_diff, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });
    cumulative = winner.geomean;
    bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
    finalWinner = winner;

    // --- (f) Re-profile the new best ------------------------------------
    profileSummary = await agentT(
      roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
        COMMANDMENT, PREVIOUS_METRICS: profileSummary,
      }),
      { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
  }

  if (winner && winner.geomean > bestSeen) bestSeen = winner.geomean;
  if (madeProgress || improved) { noImprove = 0; } else { noImprove++; }

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', 'update_memory', 'Distill durable insights + update the hypothesis ledger.', {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ROUND_RESULTS: clean.map(r => ({ id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        expected: r.d.expected_speedup, claimed: r.eng ? r.eng.speedup_geomean : 0,
        verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
        notes: r.eng ? r.eng.notes : '' })),
      INTEGRATE: integrate, WINNER: winner ? { source: winner.source, geomean: winner.geomean } : null,
      IMPROVED: improved, REPROFILE_SHIFT: profileSummary ? profileSummary.shift_note : '',
      PRIOR_HISTORY: history,
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase } : {}),
      ...(SHARED_KB ? { SHARED_KB, TARGET_LANGUAGE } : {}),
    }),
    { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (mem.insights) history.insights = mem.insights;
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  // Enforce the round's KB budget. Honest about what this can and cannot do: stripping a citation
  // does NOT un-influence a direction that was already planned with a card in mind — only the prompt
  // above can do that. What the strip buys is OBSERVABILITY: `kbCapBound` counts the rounds where the
  // planner overran its budget, so "the KB quietly took over planning" becomes a number in the run's
  // return value instead of something you infer from a speedup that got worse.
  if (USE_LEARNED_READ) {
    let seeded = 0; const stripped = [];
    for (const d of directions) {
      if (!d.learned_refs || !d.learned_refs.length) continue;
      if (seeded < KB_DIR_CAP) { seeded++; continue; }
      stripped.push(d.id); d.learned_refs = [];
    }
    // A round where EVERY direction is KB-seeded has no control in it at all.
    if (KB_COLD_DIRECTION && directions.length && directions.every(d => (d.learned_refs || []).length)) {
      const last = directions[directions.length - 1];
      stripped.push(`${last.id}(forced cold)`); last.learned_refs = [];
    }
    if (stripped.length) {
      kbCapBound++;
      log(`Round ${round}: KB budget ${KB_DIR_CAP} exceeded — un-cited ${stripped.join(', ')} ` +
          `(the directions still run). Rounds where the cap bound so far: ${kbCapBound}.`);
    }
  }

  // Join the planner's declared citations against what the verifier independently measured. Derived
  // here rather than self-reported: the planner said which card seeded which direction, the verifier
  // re-measured that direction without knowing, and their join is a fact neither one could fake.
  // `became_winner` and not `verified > 1.0` is the standing test — verified_geomean is measured
  // against the FROZEN baseline, so once a kernel sits at 2.5x cumulative every non-regressing
  // direction clears 1.0 and a card would accrue credit for advancing nothing.
  for (const r of clean) {
    for (const cardRef of (r.d.learned_refs || [])) {
      citations.push({
        card: cardRef, round, direction: r.d.id, specialty: r.d.specialty,
        // null, NOT 0, when the verifier produced nothing. `kb.py` scores <=1.0 as a real loss, so
        // encoding "no evidence" as 0 charged a card for a verifier that crashed or timed out.
        cited_then_verified: r.ver && Number.isFinite(r.ver.verified_geomean)
          ? r.ver.verified_geomean : null,
        // `winner` is only the best candidate OF THIS ROUND; `improved` is whether it beat the
        // incumbent by MIN_IMPROVE. Crediting on `winner` alone let a 1.5x round winner confirm a
        // card while the committed best already sat at 2x — a confirmation for advancing nothing,
        // which is precisely what the three-state scoring was written to avoid. Require both.
        became_winner: !!(winner && winner.id === r.d.id && improved),
      });
    }
  }
  history.rounds.push({
    round,
    directions: directions.map(d => ({ id: d.id, title: d.title, specialty: d.specialty,
      learned_refs: d.learned_refs || [] })),
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none') })),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0 } : null,
    winner: winner ? { source: winner.source, geomean: winner.geomean } : null,
    improved, cumulative,
  });
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}, cumulative=${cumulative.toFixed(2)}x, noImprove=${noImprove}`);
}

// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA });

// ===========================================================================
// PHASE: Director validation + arbitration
// ===========================================================================
phase('Validate');
const validation = await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    APPLY_TO_ORIGINAL, COMMANDMENT,
    FINAL_PATCH: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });

const finalGeomean = validation ? validation.director_verified_speedup_geomean : cumulative;
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = validation && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
const finalPrimary = HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean;
log(`COMPLETE. ${KERNEL_NAME}: verified ${HAS_WORKLOAD ? 'time-weighted' : 'geomean'} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${validation ? validation.validation_status : '?'}). Results in ${EVAL_DIR}`);

// ===========================================================================
// PHASE: UpdateExperience — distill ONE reusable card into this workflow's knowledge/learned/
// (see knowledge/learned/README.md). Runs ONCE, at the very end: never per round, because only
// the final director-verified speedup is evidence a card may cite. `validation` is required for
// the same reason — without it `finalPrimary` falls back to the TechLead's own unverified
// `cumulative`. ADD-only, so a skipped or failing step leaves the run byte-neutral.
// ===========================================================================
let learned_card = null;
// Two runs must never become a card, and neither is a failure. A run that shared its GPU measured
// contention, not the kernel. A run on a HELD-OUT kernel is the instrument for measuring whether the
// KB helps at all — distil it and the next A/B over that kernel reads back its own answer and looks
// spectacular. Defaults are permissive, so the CALLER is the one who has to declare a busy box.
const BOX_QUIET = String(A.box_quiet != null ? A.box_quiet : 'true') === 'true';
const HELD_OUT = String(A.held_out != null ? A.held_out : 'false') === 'true';
const kbGate = !BOX_QUIET ? 'the box was contended — the number measured neighbours, not the kernel'
  : HELD_OUT ? 'this kernel is in the held-out split and is the measuring instrument'
  : '';
if (kbGate) log(`[kb] not distilling: ${kbGate}.`);
// `validation.validation_status === 'accepted'` and not merely "a validation object exists": a
// `flagged` result means the director found a reason not to believe the number (patch install
// no-op, correctness fail, contended box), and curating from it teaches the next run a lesson this
// run did not earn. Reported in review of #411.
const kbAccepted = String((validation && validation.validation_status) || '').toLowerCase() === 'accepted';
if (!kbGate && UPDATE_EXPERIENCE_ON && kbAccepted && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  try {
    learned_card = await agentT(
      roleAgent('update_experience', 'Validate',
        'Curate one distilled learned card from this lane\'s verified win (ADD-only, measured evidence, ' +
        'ratios not wall-clock; record the pitfalls hit; total-then-per-direction for a stacked win).', {
          SCOPE: 'lane', LEARNED_DIR, SKILL_DIR: WORKFLOW_DIR, EVAL_DIR,
          PERF_KNOWLEDGE_DIR: KERNEL_KNOWLEDGE_DIR,
          WINNER: {
            kernel: KERNEL_NAME, language: TARGET_LANGUAGE, mode: MODE, gfx: GFX,
            kernel_class: (analysis && analysis.kernel_type) || '',
            speedup: finalPrimary, validation_status: validation ? validation.validation_status : '',
            bottleneck: profileSummary ? profileSummary.bottleneck : '',
          },
          // `rounds[]` (per-round directions + per-candidate claimed/verified/status + running
          // `cumulative`) is what makes the card's `stack:` attribution and `pitfall:` lines derivable.
          HISTORY: history,
          // What this run's plan cited and what the verifier then measured. CONTEXT ONLY — the
          // counters are applied by `kb.py drain` from the ledger filed below, never by this agent.
          // The comment that used to sit here said the curator merges them, which stopped being
          // true when that arithmetic moved into code.
          CITATIONS: citations,
          PROFILE: profileSummary,
          REPORT_PATH: report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
        }),
      { phase: 'Validate', label: 'update_experience', schema: UPDATE_EXPERIENCE_SCHEMA });
    if (learned_card && learned_card.card_path) {
      log(`[kb] learned card ${learned_card.action || 'written'}: ${learned_card.card_path}`);
    }
  } catch (e) {
    log(`[kb] update_experience skipped: ${e && e.message ? e.message : e}`);
  }
}

// FILE THE CITATION LEDGER — OUTSIDE the curation gate, and deliberately so.
// This is the loss half of the loop. It first shipped INSIDE the `finalPrimary > 1.0` branch above,
// which is the exact bias it exists to remove: the runs that produce losses are the ones that did
// not beat the baseline, were flagged, or ran on a contended box, and every one of them was
// filtered out before it could file anything. Selecting successful runs into the feedback data is
// worse than no feedback, because the resulting counters look earned. Reported in review of #411.
// Curation (writing a NEW card) still requires a verified win — a run with nothing to teach must
// not teach — but the ledger of what the cards ALREADY in the tree did is owed for every run.
// `held_out` is the one exception: that split is the measuring instrument, so it neither writes
// cards nor votes on them.
// A workflow script has no filesystem, hence the one-line agent; the JSON is assembled by
// `kb.py cite` rather than by the agent, because a proposal hand-written by a model is one more
// place for the schema to drift.
if (citations.length && !HELD_OUT) {
  try {
    await agentT(
      `Run EXACTLY this command and nothing else. Do NOT edit any file.
\`\`\`bash
cat <<'CITEJSON' | python3 ${WORKFLOW_DIR}/scripts/kb.py --kb-dir ${LEARNED_DIR} cite \\
--run-id ${JSON.stringify(KERNEL_NAME)} \\
--kernel ${JSON.stringify(KERNEL_NAME)} --citations -
${JSON.stringify(citations)}
CITEJSON
\`\`\`
Return {"filed": <the "citations" number the command printed, or 0>}.`,
      { phase: 'Validate', label: 'kb:cite', effort: 'low',
        schema: { type: 'object', properties: { filed: { type: 'number' } },
                  required: ['filed'], additionalProperties: true } });
    log(`[kb] citation ledger filed: ${citations.length} row(s) for the next drain`);
  } catch (e) {
    log(`[kb] citation ledger NOT filed: ${e && e.message ? e.message : e}`);
  }
}

// ===========================================================================
// Write this run's outcome back to the local experience store (kb_artifacts/) — the
// producer half of the warm-start loop. The script applies its own gate
// (missing_arch / no_improvement / empty_diff) and prints a single-line JSON; the
// whole step is wrapped so a store failure NEVER fails the run (plan Part 4.2). The
// JS-side `finalPrimary > 1.0` pre-check just avoids spending an agent on a run that
// cannot pass the gate anyway.
// ===========================================================================
let kb_written = null;
if (KB_ARTIFACTS_DIR && GFX && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  const kernelClass = (analysis && analysis.kernel_type) || 'unknown';
  const finalPatch = report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`;
  const reportPath = report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`;
  kb_written = await agentT(
    `You are the experience writer. Run EXACTLY this command (it applies its own gates and prints a ` +
    `single-line JSON) and return that JSON verbatim as StructuredOutput. If the command errors, return ` +
    `{"written": false, "reason": "io_error"}.
\`\`\`bash
python3 ${EXPERIENCE_STORE} write --root ${KB_ARTIFACTS_DIR} \\
  --kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\
  --gfx ${GFX} --kernel-class ${JSON.stringify(kernelClass)} \\
  --speedup ${finalPrimary} --baseline-wall-ms ${BASELINE_GEOMEAN_MS} \\
  --patch ${finalPatch} --eval-dir ${EVAL_DIR} --report ${reportPath}
\`\`\``,
    { phase: 'Validate', label: 'kb:write', schema: WARMSTART_WRITE_SCHEMA });
  log(kb_written && kb_written.written
    ? `[kb] experience written: ${kb_written.slug} (speedup ${finalPrimary.toFixed(2)})`
    : `[kb] experience not written: ${kb_written ? kb_written.reason : 'writer returned nothing'}`);
}

// The headline speedup relative to the PRISTINE frozen baseline. When a warm-start patch was adopted,
// this run's own rounds only earned the delta ABOVE the adopted starting point — split them out so a
// KB-derived gain is never reported as this run's work (KernelForge total vs incremental).
const incrementalSpeedup = warm_start.adopted && warm_start.adopted_speedup
  ? (Number.isFinite(finalPrimary) ? finalPrimary / warm_start.adopted_speedup : null)
  : finalPrimary;

return {
  mode: MODE,
  target_language: MODE === 'author' ? TARGET_LANGUAGE : undefined,
  authored: MODE === 'author' ? true : undefined,
  eval_dir: EVAL_DIR,
  kernel_name: KERNEL_NAME,
  workload_aligned: HAS_WORKLOAD,
  final_speedup: finalPrimary,                 // PRIMARY metric (weighted when workload-aligned)
  final_weighted: finalWeighted,
  final_geomean: finalGeomean,
  final_arithmetic: validation ? validation.director_verified_speedup_arithmetic : null,
  tech_lead_reported_geomean: report ? report.final_speedup_geomean : cumulative,
  validation_status: validation ? validation.validation_status : 'unknown',
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
  // null when the step was off, or the run earned nothing worth a card.
  // Which condition actually ended the loop. RECORDED, not inferred: on the previous campaign this
  // existed only in the driver's return value, which the caller sees after ALL its kernels finish —
  // so a parent-process death erased it for every kernel already done, and the realised window had
  // to be reconstructed from queue timestamps. It is the whole point of an enforced window that you
  // can tell a run that used its budget from one that gave up.
  stopped_by: deadlineHit ? 'deadline'
    : (dispatched >= BUDGET) ? 'budget'
    : (noImprove >= MAX_NO_IMPROVE) ? 'no_improve'
    : 'tech_lead_stop',
  deadline_hit: deadlineHit,
  forced_replans: forcedReplans,
  // What the plan cited and whether it carried its round. Returned ALWAYS, including when nothing was
  // cited (an empty array is the finding: the KB was read and nothing in it was worth acting on).
  learned_citations: citations,
  // Rounds where the planner overran the KB budget. A KB that binds every round is one that has taken
  // over planning, which is the regression this budget exists to catch.
  kb_cap_bound_rounds: kbCapBound,
  // The monoculture canary, and it costs no GPU time: how many DISTINCT (specialty, focus_files)
  // directions this run explored vs how many it issued. A KB that helps raises the verified speedup;
  // a KB that cages lowers this without raising that. Reported on KB-less runs too — those are the
  // baseline it has to be read against.
  direction_entropy: (() => {
    const seen = new Set(); let n = 0;
    for (const r of history.rounds) for (const d of (r.directions || [])) {
      n++; seen.add(`${d.specialty}|${(d.focus_files || []).slice().sort().join(',')}`);
    }
    return n ? { distinct: seen.size, issued: n, ratio: +(seen.size / n).toFixed(3) } : null;
  })(),
  learned_card: learned_card && learned_card.card_path
    ? { action: learned_card.action || '', card_path: learned_card.card_path, key: learned_card.key || '' }
    : null,
  // Warm-start (local experience KB) outcome. adopted=false + read_reason on a cold start.
  warm_start: {
    adopted: warm_start.adopted,
    read_reason: warm_start.read_reason,
    slug: warm_start.slug || '',
    adopted_speedup: warm_start.adopted ? warm_start.adopted_speedup : null,
    total_speedup: Number.isFinite(finalPrimary) ? finalPrimary : null,
    incremental_speedup: incrementalSpeedup,
    incremental_improved: !!(warm_start.adopted && Number.isFinite(incrementalSpeedup) && incrementalSpeedup > 1 + MIN_IMPROVE),
    returned_after_read_kb: !!warm_start.returned_after_read_kb,
    candidates: warm_start.candidates,
  },
  kb_written: kb_written && kb_written.written ? { slug: kb_written.slug, dir: kb_written.dir } : null,
};
