export const meta = {
  name: 'kernel-lane',
  description: 'SINGLE-LANGUAGE kernel optimization worker (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, and integration. Optimizes ONE kernel in ONE language (mode=optimize) or authors a fresh seed then optimizes it (mode=author); mode=mega searches a multi-rank operator (e.g. an EP MoE layer) as independent whole-kernel candidates, each verified on-card against the frozen baseline (validated on EP MoE only). This is the worker invoked per lane by the kernel-workflow dispatcher (kernel_workflow.js) and by e2e_workflow; prefer calling kernel-workflow directly unless you specifically want one unchanged lane. Target: AMD Instinct MI-series (CDNA gfx942/gfx950) and the validated RDNA4 product Radeon AI PRO R9700 (gfx1201) — auto-detected on-box; on R9700 use knowledge/amd_rdna4.md (wave32, WMMA). Generic gfx1201 without R9700 identity is not a calibrated target.',
  whenToUse: 'Internal single-language worker. Prefer the kernel-workflow dispatcher (kernel_workflow.js) as the entry point; invoke this directly only to run one unchanged lane. Pass args.kernel_path (required), args.mode, args.target_language, args.budget, args.gpu_ids, args.gpu_mode, args.gpus_per_job, args.job_gpu_ids, args.task; mode=mega also takes its mega_* / guard / control args (see README "Mega mode").',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies the bottleneck (only with args.profile=on; off by default); mode=mega: optional MoE analysis (PHASE=analyze_profile) validates the evidence' },
    { title: 'Research', detail: 'OPT-IN (args.dra_enabled): researcher fans research questions out in parallel via native WebSearch/WebFetch, writes a ranked-directions brief the planner seeds from' },
    { title: 'WarmStart', detail: 'search the experience KB by canonical id (geak:kernel:*) — the shared service when credentialed, else the on-disk store, brought level with kb_artifacts/ first — for the best curated patch per optimization direction for this (kernel,language,gfx), validate each through the verify gate, adopt the first that passes [warm_start!=off]' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile; mode=mega: tech_lead picks one whole-kernel candidate per round, an engineer authors it in its own tree' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline, then the TechLead curates ONE distilled card into knowledge/learned/ on a measured win [update_experience!=off]' },
  ],
};

// ---------------------------------------------------------------------------
// Args + defaults.
// ---------------------------------------------------------------------------
const A = args || {};
if (A.optimizer != null && typeof A.optimizer !== 'string') {
  throw new Error('optimizer must be greedy or evo (a string)');
}
const OPTIMIZER = String(A.optimizer || '').trim().toLowerCase();
if (OPTIMIZER && !['greedy', 'evo'].includes(OPTIMIZER)) {
  throw new Error('optimizer must be greedy or evo');
}
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/model directory)');

// WORKFLOW_DIR = the directory that holds this script + roles/ + knowledge/ + scripts/.
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js, roles/, knowledge/, scripts/ (i.e. the dirname of this script).');
}
// EXP_ROOT = where timestamped run dirs are written.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

let EVO = OPTIMIZER === 'evo' ? createEvolution() : null;
let EVO_FALLBACK = null;   // set when EVO could not start; the lane then runs as the ordinary lane
const KERNEL_PATH_ORIG = A.kernel_path;
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
// ---- GPU families: keep this block identical in kernel_workflow.js, kernel_lane.js and e2e_workflow.js ----
// Per-GPU-family policy, looked up by ISA. The Workflow sandbox has no import, so each workflow carries
// a copy (kernel_workflow/scripts/test_gpu_family_table.js keeps them identical). An ISA no row
// matches gets GPU_FAMILY_DEFAULT, the Instinct (CDNA) policy. Each workflow reads the fields it needs.
//   name        how the bake-off Discover prompt names an isolated family (gfx1201 keeps
//               upstream's 'R9700' label)
//   isolate     withhold the MI-series-calibrated inputs (perf_knowledge cards, expert skills, learned
//               KB, warm start, tuning skillset); the in-tree guide is the hardware reference
//   matrixCore  the matrix ISA the prompts name
const GPU_FAMILIES = [
  { name: 'R9700', isa: /^gfx1201$/, isolate: true, matrixCore: 'WMMA' },
  // RDNA 3.5 APUs (Strix Halo is gfx1151). Only a gfx1151 Radeon 8060S has been run.
  { name: 'RDNA 3.5', isa: /^gfx115\d$/, isolate: true, matrixCore: 'WMMA' },
  // Discrete RDNA3. The workflows accept gfx1100 only as the exact Radeon Pro W7900 (target w7900).
  { name: 'W7900', isa: /^gfx1100$/, isolate: true, matrixCore: 'WMMA' },
];
const GPU_FAMILY_DEFAULT = { name: 'default', isa: null, isolate: false, matrixCore: 'MFMA' };
const gpuFamily = (gfx) => GPU_FAMILIES.find((f) => f.isa.test(String(gfx || ''))) || GPU_FAMILY_DEFAULT;
// ---- end GPU families ----
if (EXPECTED_TARGET === 'w7900' && EXPECTED_GFX !== 'gfx1100') {
  throw new Error('expected_target=w7900 requires expected_gfx=gfx1100');
}
if (EXPECTED_GFX === 'gfx1100' && EXPECTED_TARGET !== 'w7900') {
  throw new Error(
    `expected_gfx=gfx1100 requires expected_target=w7900 (got ${EXPECTED_TARGET || 'none'}): ` +
    'no other gfx1100 product is validated');
}
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// Optional hard cap on optimization rounds (0 / absent = no cap, i.e. unchanged behaviour). Budget
// alone does not bound the round count: the TechLead may spend a budget of 2 as one round of two
// directions or as two rounds of one, which roughly doubles the wall time.
const MAX_ROUNDS = (() => {
  const v = parseInt(A.max_rounds != null ? A.max_rounds : 0, 10);
  return Number.isFinite(v) && v > 0 ? v : Infinity;
})();
// fast_mode (opt-in, for time-boxed runs): after the FINAL round, skip the re-profile and the memory
// update, and give optimize engineers a wall-clock box (engineer_minutes, default 6), which the engineer
// enforces by reading `date +%s` -- a firm instruction, not a hard kill.
const FAST_MODE = String(A.fast_mode != null ? A.fast_mode : 'false').trim().toLowerCase() === 'true';
const ENGINEER_MINUTES = Math.max(1, parseFloat(A.engineer_minutes != null ? A.engineer_minutes : 6));
function fastModeEngineerBlock(outDir) {
  if (!FAST_MODE) return '';
  const secs = Math.round(ENGINEER_MINUTES * 60);
  return `

## FAST MODE -- HARD TIME BOX (overrides the "Iterate" step and the self-monitoring stop rules)
1. Your FIRST command: \`date +%s > ${outDir}/.t0\`
2. Keep your current best kernel IN KERNEL_PATH (the workspace source file), never only in /tmp or a
   scratch copy, and re-save best_patch.diff the moment your best improves.
3. Before starting EACH new experiment run \`echo $(( $(date +%s) - $(cat ${outDir}/.t0) ))\`.
   If it prints >= ${secs} (${ENGINEER_MINUTES} min), STOP experimenting.
4. Also STOP when your last 3 benchmarked attempts together improved your best by < 3%, or after
   3 consecutive attempts that failed correctness or were slower.
5. Iterate with the quick BENCHMARK command; run FULL_BENCHMARK once, on your final best only.
6. On STOP: workspace holds the best kernel, CORRECTNESS passes, best_patch.diff is saved AND non-empty
   (check \`wc -c ${outDir}/best_patch.diff\`; an empty patch is DISCARDED by verify) -> return
   the worker_result immediately. A correct result on time beats a marginally faster one delivered late.`;
}
// Minimum verified geomean improvement over the cumulative best for a round winner to be COMMITTED into
// the canonical workspace (default 2%).
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
})();
// Minimum verified speedup for a candidate to enter the round's candidate list (default 1.0 = only a
// candidate that beats the baseline is worth looking at).
const CANDIDATE_FLOOR = (() => {
  const v = parseFloat(A.candidate_floor != null ? A.candidate_floor : 1.0);
  return Number.isFinite(v) && v > 0 ? v : 1.0;
})();
// Rendered into the Optimize prompt, where `${1.0}` would stringify to "1" and silently reword a
// prompt that has always said "geomean>1.0". Keeps the default run byte-identical.
const CANDIDATE_FLOOR_TXT = Number.isInteger(CANDIDATE_FLOOR)
  ? CANDIDATE_FLOOR.toFixed(1) : String(CANDIDATE_FLOOR);
// How far BELOW the best candidate ever seen a round may land and still count as "the search is
// advancing" (default +MIN_IMPROVE = the historical test).
const PROGRESS_DELTA = (() => {
  const v = parseFloat(A.progress_delta != null ? A.progress_delta : MIN_IMPROVE);
  return Number.isFinite(v) && v > -1 ? v : MIN_IMPROVE;
})();
// Budget cost of ONE `deep_explore` direction.
const DEEP_COST = (() => {
  const v = parseInt(A.deep_cost != null ? A.deep_cost : 2, 10);
  return Number.isFinite(v) && v >= 1 ? v : 2;
})();
// PD gpu_ids `decode:3`: every GPU_ID passed on names the engine again.
const GPU_IDS_ARG = String(A.gpu_ids != null ? A.gpu_ids : '0').trim();
const GPU_ENGINE = (/^([\w.-]+):(?=\d)/.exec(GPU_IDS_ARG) || [])[1] || '';
const onEngine = spec => (GPU_ENGINE && spec ? `${GPU_ENGINE}:${spec}` : spec);
const GPU_IDS = GPU_ENGINE ? GPU_IDS_ARG.slice(GPU_ENGINE.length + 1) : GPU_IDS_ARG;
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Every GPU consumer gets the WHOLE pool rather than a pinned lane.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
// Multi-GPU jobs (multi-rank ops).
function resolveGpuRequest(A, gpuList, detected) {
  const resource = (A.op_spec && A.op_spec.resource) || {};
  const parseIds = (name, value) => {
    const parts = String(value).split(',').map(s => s.trim()).filter(Boolean);
    if (!parts.length) throw new Error(`${name} must contain at least one GPU id`);
    if (parts.some(id => !/^\d+$/.test(id))) throw new Error(`${name} contains a non-numeric GPU id`);
    const ids = parts.map(id => String(Number(id)));
    if (new Set(ids).size !== ids.length) throw new Error(`${name} contains duplicate GPU ids`);
    return ids;
  };
  const pool = parseIds('gpu_ids', gpuList.join(','));
  const fixedValue = String(A.job_gpu_ids || '').trim() || resource.job_gpu_ids;
  const fixedIds = String(fixedValue || '').trim() ? parseIds('job_gpu_ids', fixedValue) : [];
  const explicit = A.gpus_per_job != null || resource.gpus_per_job != null || fixedIds.length > 0;
  const found = (!explicit && detected) || {};
  let foundCount = found.kind === 'scalable' ? pool.length : 1;
  let problem = '';
  if (found.kind === 'unknown') {
    problem = 'Setup found multi-GPU code in this task but could not tell how many GPUs one run needs';
  } else if (found.kind === 'fixed') {
    const n = Number(found.gpus_per_job);
    if (!Number.isSafeInteger(n) || n < 1) {
      problem = `Setup reported no usable GPU count (${found.gpus_per_job})`;
    } else if (n > pool.length) {
      problem = `Setup found ${n} GPUs per run but gpu_ids holds ${pool.length}`;
    } else {
      foundCount = n;
    }
  }
  const warning = problem ? `${problem} (${found.evidence || 'no evidence'}); running single-GPU. ` +
    'Pass gpus_per_job=<N> (or job_gpu_ids) with enough gpu_ids to run it on N GPUs.' : '';
  const countValue = A.gpus_per_job != null ? A.gpus_per_job
    : (resource.gpus_per_job != null ? resource.gpus_per_job : (fixedIds.length || foundCount));
  const countText = String(countValue).trim();
  if (!/^[1-9]\d*$/.test(countText)) throw new Error('gpus_per_job must be a positive integer');
  const gpusPerJob = Number(countText);
  if (!Number.isSafeInteger(gpusPerJob)) throw new Error('gpus_per_job is too large');
  if (fixedIds.length && fixedIds.length !== gpusPerJob) {
    throw new Error(`gpus_per_job=${gpusPerJob} does not match job_gpu_ids size ${fixedIds.length}`);
  }
  if (!fixedIds.length && gpusPerJob > pool.length) {
    throw new Error(`gpus_per_job=${gpusPerJob} exceeds gpu_ids pool size ${pool.length}; ` +
      `pass gpu_ids with at least ${gpusPerJob} GPUs`);
  }
  if (fixedIds.some(id => !pool.includes(id))) throw new Error('job_gpu_ids must be a subset of gpu_ids');
  const jobSpec = fixedIds.length ? `group:${fixedIds.join(',')}`
    : (gpusPerJob > 1 ? `pool:${gpusPerJob}:${pool.join(',')}` : '');
  return { gpusPerJob, fixedIds, jobSpec, explicit, warning };
}
// Validated now so a bad request fails before Setup; re-resolved from Setup's finding when implicit.
let GPU_RESOURCE = null;
let GPU_JOB = '';
let GPU_POOL = '';
let GPU_SINGLE = '';
function applyGpuResource(r) {
  GPU_RESOURCE = r;
  GPU_JOB = onEngine(r.jobSpec);
  GPU_POOL = GPU_JOB || onEngine(GPU_MODE === 'pin' ? GPU_LIST[0] : GPU_LIST.join(','));
  GPU_SINGLE = GPU_JOB || onEngine(GPU_LIST[0]);
}
applyGpuResource(resolveGpuRequest(A, GPU_LIST));
// A multi-GPU task delivered by the UT module (A22): set after Setup, once the GPU count is final,
// when that count is above 1 AND the task's own declaration (Setup's task_declaration) says more than
// one GPU. Only such a job gets the multi-rank prompts, receipts, lease and agent time limits; every
// other job, single- or multi-GPU, runs exactly as before. UT_TOTAL_S is its whole-UT time limit (s).
let UT_MR = false;
let UT_TOTAL_S = 0;
const TASK = A.task || '';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();

// --- author mode: when there is NO existing source, write a fresh from-scratch SEED first, then
// optimize it.
const MODE = String(A.mode != null ? A.mode : 'optimize').trim() || 'optimize';
const MEGA = MODE === 'mega';   // multi-rank candidate search: see the mode=mega backend section
const TARGET_LANGUAGE = String(A.target_language != null ? A.target_language : 'triton').trim() || 'triton';
const OP_SPEC = A.op_spec || {};
// When the op will run on the CUDA/HIP-graph-captured decode path (e2e sets
// op_spec.cuda_graph_safe=true), the isolated oracle alone CANNOT catch a kernel that passes iso but
// host-syncs or lazily-compiles under graph capture — the "wins isolated, crashes serving" class
// (cuda_graph_capture_unsafe / NO_BINARY_FOR_GPU).
const REQUIRE_GRAPH_CAPTURE = !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
// WORKLOAD ALIGNMENT (optional).
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
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
// Expert skills = human-authored, validated kernel recipes (perf_knowledge/expert_skills/).
const USE_EXPERT_SKILLS_REQUESTED = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true' ||
  (MEGA && A.use_expert_skills == null && !!A.expert_skill_id);   // a mega run that names a skill uses it
let USE_EXPERT_SKILLS = USE_EXPERT_SKILLS_REQUESTED;
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
// Only planning + authoring roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);
// Expected-identity isolation must apply before Setup (no CDNA skills/KB on a declared isolated run).
let SKILLS_ISOLATE = gpuFamily(EXPECTED_GFX).isolate;

// --- Deep Research Agent (DRA) -------------------------------------------------------------------
// OPT-IN: a v4-native research phase that runs AFTER Profile and BEFORE the optimize loop (so the
// COMMANDMENT + baseline profile + analysis exist).
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
// LEARNED-KNOWLEDGE SINK.
const LEARNED_DIR = `${WORKFLOW_DIR}/knowledge/learned`;
// How much of ONE round the learned KB may steer.
let USE_LEARNED_READ = String(A.use_learned_kb != null ? A.use_learned_kb : 'true') === 'true';
const KB_DIR_CAP = Math.max(0, parseInt(A.kb_dir_cap != null ? A.kb_dir_cap : 1, 10));
const KB_COLD_DIRECTION = String(A.kb_cold_direction != null ? A.kb_cold_direction : 'true') === 'true';
let kbCapBound = 0;      // rounds where the cap actually had to strip something
const UPDATE_EXPERIENCE = String(A.update_experience != null ? A.update_experience : 'on').trim().toLowerCase() || 'on';
const UPDATE_EXPERIENCE_ON = UPDATE_EXPERIENCE !== 'off' && UPDATE_EXPERIENCE !== 'false' && UPDATE_EXPERIENCE !== 'none';
// Did the CALLER freeze this kernel through oracle_freezer?
const FROZEN_ORACLE = String(A.frozen_oracle != null ? A.frozen_oracle : 'false') === 'true';

// ---------------------------------------------------------------------------
// WARM-START (the experience KB).
const WARM_START = String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on';
let WARM_START_ON = WARM_START !== 'off' && WARM_START !== 'false' && WARM_START !== 'none';
if (SKILLS_ISOLATE) {
  KERNEL_KNOWLEDGE_DIR = '';
  USE_LEARNED_READ = false;
  WARM_START_ON = false;
  USE_EXPERT_SKILLS = false;
}
const WARM_START_REF_ONLY = WARM_START === 'reference';
const WARM_START_RETURN_AFTER = WARM_START === 'return_after_read';
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts' : '')).replace(/\/+$/, '');
const EXPERIENCE_STORE = `${WORKFLOW_DIR}/scripts/experience_store.py`;
// Every candidate costs a full on-box verify to reject, so a recorded near-tie is not worth reading.
const WARM_START_MIN_SPEEDUP = Number.isFinite(parseFloat(A.warm_start_min_speedup))
  ? parseFloat(A.warm_start_min_speedup) : 1.05;
// How hard the resolver may work to map this run's kernel name onto a stored page.
const WARM_START_MATCH = ['exact', 'normalized', 'fuzzy'].includes(String(A.warm_start_match || '').trim())
  ? String(A.warm_start_match).trim() : 'fuzzy';
// Which PLANE the experience is READ from.
//
// `store` is the default because it is the SAME addressing the service uses.
const KB_MODE = String(A.kb_mode || 'store').trim().toLowerCase() === 'local' ? 'local' : 'store';
const KB_STORE_DIR = String(A.kb_store_dir ||
  (KB_ARTIFACTS_DIR ? KB_ARTIFACTS_DIR.replace(/\/[^/]*$/, '') + '/kb_store_local' : '')).replace(/\/+$/, '');
// The key carries a rocm <major>.<minor>; on a box without /opt/rocm the measured stack is empty and
// every record would file under `unspecified`, which splits one kernel's history across two keys.
const KB_FRAMEWORK_VERSION = String(A.kb_framework_version || '').trim();
const KB_VERSION_FLAG = KB_FRAMEWORK_VERSION ? ` --framework-version ${JSON.stringify(KB_FRAMEWORK_VERSION)}` : '';
const KB_ROOT_OK = KB_MODE === 'store' ? !!KB_STORE_DIR : !!KB_ARTIFACTS_DIR;
// Whether this run may also talk to the KB Store SERVICE, on top of whichever local plane KB_MODE
// selected.
const KB_REMOTE = String(A.kb_remote || 'auto').trim().toLowerCase() === 'off' ? 'off' : 'auto';
// The credentials are NOT visible from this process.
const KB_ENV_PRELUDE =
  'export KB_STORE_URL="${GEAK_KB_STORE_URL:-${KB_STORE_URL:-https://global.primus-safe.amd.com/knowledge-base}}"; ' +
  'export KB_STORE_TOKEN="${GEAK_KB_STORE_TOKEN:-${KB_STORE_TOKEN:-$(cat ~/.geak_kb_token 2>/dev/null)}}"; ' +
  'if [ -z "${SSL_CERT_FILE:-}" ]; then for _ca in "${KB_CA_BUNDLE:-}" ' +
  '/shared_nfs/hyperloom/ca/amd-ca-combined.pem "$HOME/amd-extra-ca-bundle.pem"; do ' +
  '[ -n "$_ca" ] && [ -r "$_ca" ] && { export SSL_CERT_FILE="$_ca" REQUESTS_CA_BUNDLE="$_ca" ' +
  'CURL_CA_BUNDLE="$_ca" NODE_EXTRA_CA_CERTS="$_ca"; break; }; done; fi; ';
// Writing in store mode records BOTH planes in one call, so it needs both roots: the directory tree
// stays the source of truth a curation pass edits, and the store is derived from it.
const KB_WRITE_OK = KB_ROOT_OK && !!KB_ARTIFACTS_DIR;
const kebab = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);

// ---------------------------------------------------------------------------
// DEEP-MODE continuation + cross-backend / e2e-feedback hooks.
const STATE_DIR = String(A.state_dir || '').replace(/\/+$/, '');
const SHARED_KB = String(A.shared_kb || '').trim();
const GLOBAL_KB = String(A.global_kb || '').trim();   // run-global cross-KERNEL technique blackboard (deep)
const E2E_FEEDBACK = String(A.e2e_feedback || '').trim();
const HARNESS_ADDENDUM = String(A.harness_addendum || '').trim();
// P2 (deep continuation): on a RESUMED wave (STATE_DIR holds prior work) the cold re-derivation of
// Analyze + baseline Profile is largely redundant.
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
      // Workload-alignment fields (present only when a WORKLOAD_SPEC drives the harness; absent on a
      // normal unweighted run).
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
  kernel_name: { type: 'string' },
  device_gfx: { type: 'string' },
  device_target: { type: 'string' },
  device_name: { type: 'string' },
  physical_cu_count: { type: 'number' },
  source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // Frozen-baseline verdict (BOTH modes).
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  // GPUs one run of the task's test/benchmark needs (roles/director.md "GPU requirement").
  gpu_requirement: obj({
    kind: { type: 'string', enum: ['single', 'fixed', 'scalable', 'unknown'] },
    gpus_per_job: { type: 'number' }, evidence: { type: 'string' },
  }, ['kind']),
  // What the task itself declares, verbatim from `detect_gpu_requirement.py --declaration <task dir>`
  // (always run, also when the GPU count is explicit): its GPU count (null = none) and whole-UT limit.
  task_declaration: obj({
    required_gpu_count: { type: ['number', 'null'] }, ut_total_s: { type: 'number' },
    evidence: { type: 'array', items: { type: 'string' } },
  }, ['required_gpu_count', 'ut_total_s', 'evidence']),
  // DEEP-MODE resume only: populated by the director ONLY when STATE_DIR was provided AND a prior best
  // exists there.
  resumed: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name', 'device_gfx', 'device_target', 'physical_cu_count', 'task_declaration']);

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

// --- Deep Research Agent (DRA) schemas (opt-in via args.dra_enabled) ------- Stage 0+1/2: the
// researcher extracts facts and returns a ranked set of research QUESTIONS the script then fans out in
// parallel.
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

// Stage 7 final portfolio.
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
      // Learned cards that SEEDED this direction, by filename.
      learned_refs: { type: 'array', items: { type: 'string' } },
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

if (EVO) {
  PLAN_SCHEMA.properties.directions.items.properties.parent_id = { type: 'string' };
  PLAN_SCHEMA.properties.directions.items.properties.donor_ids = { type: 'array', items: { type: 'string' } };
}

const ENG_SCHEMA = obj({
  engineer_id: { type: 'string' }, specialty: { type: 'string' }, task: { type: 'string' },
  strategy: { type: 'string' }, speedup_geomean: { type: 'number' }, speedup_arithmetic: { type: 'number' },
  // Time-weighted ratio-of-sums vs the TRUE baseline (PRIMARY metric when workload_aligned).
  speedup_weighted: { type: 'number' },
  per_case: perCase, status: { type: 'string' }, patch_file: { type: 'string' },
  measurement_valid: { type: 'boolean' },
  strategies_tried: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
}, ['status', 'speedup_geomean', 'measurement_valid']);

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

// eikos v2 E1 (R4 round-4 / Astra re-review #4): the round-3 PRECOMMIT_SCHEMA + RECOVERY_SCHEMA were
// removed with the restore-and-continue recovery they served.

// eikos v2 E1 (R9 / Astra re-review #10, Finding 2 — SHORTEST ROUTE): the round-7/8 host-side
// deterministic commit fallback (its fixed-script dispatch, its result schema, and its
// observed-postcondition acceptance predicate) have been REMOVED.

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
  // clean | host_bound | unprimed | unknown | not_applicable.
  timing_basis: { type: 'string' },
  per_case: perCase, applied_to_original: { type: 'string' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);

// ---------------------------------------------------------------------------
// Multi-rank receipts (A01). In a UT-module multi-GPU task (UT_MR) every rank of the UT times the
// case, and one merged number hides how the ranks were aligned and which rank set the pace. These
// variants replace BENCH/REPORT/VALIDATE_SCHEMA only in such a job; any other call still passes
// the objects above, unchanged. The receipt has the shape of the UT's own GEAK_TIMING_RECEIPT
// (geak_timing_receipt/v2): {world_size, alignment, cases: {<sig>: {baseline: {ms, rank_ms},
// current: {ms, rank_ms}}}}; an older multi-rank UT's per_rank_us/step_us/alignment map onto it.
// ---------------------------------------------------------------------------
const rankArr = { type: 'array', items: { type: 'number' } };
// One leg of one case: the merged time (per sample the slowest rank, then the median) + each rank's
// when the UT reports them (null when it reports only the merged time).
const rankLeg = obj({ ms: { type: 'number' }, rank_ms: { type: ['array', 'null'], items: { type: 'number' } } }, ['ms']);
const timingReceipt = {
  type: ['object', 'null'],
  properties: {
    version: { type: 'string' }, world_size: { type: 'number' }, source: { type: 'string' },
    alignment: { type: ['object', 'null'], additionalProperties: true },
    cases: { type: 'object', additionalProperties: obj({ baseline: rankLeg, current: rankLeg }, ['baseline']) },
  },
  required: ['world_size', 'cases'],
  additionalProperties: true,
};
const perCaseMR = {
  ...perCase,
  items: {
    ...perCase.items,
    properties: {
      ...perCase.items.properties,
      // Each rank's time for this case (ms, rank order) and how the ranks were aligned before every
      // sample and merged into baseline_ms / optimized_ms.
      baseline_rank_ms: rankArr, optimized_rank_ms: rankArr,
      alignment: { type: 'object', additionalProperties: true },
    },
  },
};
const BENCH_SCHEMA_MR = obj({
  ...BENCH_SCHEMA.properties,
  // rank_ms / alignment are REQUIRED but nullable: null says the UT gave no per-rank time or no
  // alignment method for the case (then reliable=false), instead of one merged time copied per rank.
  baseline_per_case: { type: 'array', items: obj({
    name: { type: 'string' }, latency_ms: { type: 'number' },
    rank_ms: { type: ['array', 'null'], items: { type: 'number' } },
    alignment: { type: ['object', 'null'], additionalProperties: true },
  }, ['name', 'latency_ms', 'rank_ms', 'alignment']) },
  rank_timing_source: { type: 'string' },
}, [...BENCH_SCHEMA.required, 'reliable']);
const REPORT_SCHEMA_MR = obj({ ...REPORT_SCHEMA.properties, per_case: perCaseMR, timing_receipt: timingReceipt },
  REPORT_SCHEMA.required);
const VALIDATE_SCHEMA_MR = obj({ ...VALIDATE_SCHEMA.properties, per_case: perCaseMR, timing_receipt: timingReceipt },
  [...VALIDATE_SCHEMA.required, 'timing_receipt']);
// The consumer check for all three places (Benchmark record, director receipt, lane return). A merged
// time on every timed case is required; each rank's time and the alignment method are recorded when
// the UT reports them, and their absence is a note, not a failure (a hand-written UT may report only
// the merged, slowest-rank time). benchRankGap / receiptGap return the hard gap ('' when none);
// benchRankNote / receiptNote name what per-rank detail is missing ('' when complete).
const timeOk = (x) => typeof x === 'number' && Number.isFinite(x) && x > 0;
const rankTimesOk = (a, ranks) => Array.isArray(a) && a.length === ranks && a.every(timeOk);
const alignmentOk = (a) => !!a && typeof a === 'object' && !Array.isArray(a) && Object.keys(a).length > 0;
const legGap = (leg, ranks, what) => !leg || !rankTimesOk(leg.rank_ms, ranks) ? `no per-rank ${what} times`
  : !timeOk(leg.ms) ? `no merged ${what} time (ms)` : '';
function benchRankGap(cases) {
  if (!Array.isArray(cases) || !cases.length) return 'no baseline case was recorded';
  for (const c of cases) {
    const name = c && c.name != null ? c.name : '?';
    if (!c || !timeOk(c.latency_ms)) return `baseline case ${name} has no merged time (latency_ms)`;
  }
  return '';
}
function benchRankNote(cases, ranks) {
  const noRanks = [], noAlign = [];
  for (const c of Array.isArray(cases) ? cases : []) {
    const name = c && c.name != null ? c.name : '?';
    if (!c || !rankTimesOk(c.rank_ms, ranks)) noRanks.push(name);
    if (!c || !alignmentOk(c.alignment)) noAlign.push(name);
  }
  return [noRanks.length ? `no per-rank times (rank_ms of ${ranks}) for case(s) ${noRanks.join(', ')}` : '',
    noAlign.length ? `no alignment method for case(s) ${noAlign.join(', ')}` : ''].filter(Boolean).join('; ');
}
function receiptGap(receipt, ranks, needCurrent) {
  if (!receipt || typeof receipt !== 'object') return 'no timing_receipt';
  if (Number(receipt.world_size) !== ranks) return `timing_receipt world_size is ${receipt.world_size}, not ${ranks}`;
  const cases = receipt.cases && typeof receipt.cases === 'object' ? Object.entries(receipt.cases) : [];
  if (!cases.length) return 'timing_receipt has no timed case';
  for (const [sig, c] of cases) {
    if (!c || !c.baseline || !timeOk(c.baseline.ms)) return `case ${sig} has no merged baseline time (ms)`;
    const cur = c.current || c.candidate;
    if (needCurrent && (!cur || !timeOk(cur.ms))) return `case ${sig} has no merged candidate time (ms)`;
  }
  return '';
}
function receiptNote(receipt, ranks, needCurrent) {
  if (!receipt || typeof receipt !== 'object') return '';
  const notes = alignmentOk(receipt.alignment) ? [] : ['no alignment method'];
  const cases = receipt.cases && typeof receipt.cases === 'object' ? Object.entries(receipt.cases) : [];
  for (const [sig, c] of cases) {
    if (legGap(c && c.baseline, ranks, 'baseline')) notes.push(`case ${sig}: no per-rank baseline times`);
    if (needCurrent && legGap(c && (c.current || c.candidate), ranks, 'candidate'))
      notes.push(`case ${sig}: no per-rank candidate times`);
  }
  return notes.join('; ');
}
// The Benchmark's complete baseline record in receipt shape: what the lane returns when the director's
// own receipt is missing or incomplete, so the baseline receipt survives a failed validation.
function benchReceipt(cases, ranks, source) {
  return {
    world_size: ranks,
    source: `Benchmark baseline_per_case${source ? ` (${source})` : ''}`,
    alignment: cases[0].alignment ?? null,
    cases: Object.fromEntries(cases.map((c) => [String(c.name),
      { baseline: { ms: c.latency_ms, rank_ms: c.rank_ms ?? null } }])),
  };
}

// Warm-start resolver output = experience_store.py `resolve` JSON, verbatim.
const WARMSTART_RESOLVE_SCHEMA = obj({
  read_reason: { type: 'string' }, slug: { type: 'string' },
  // What this run's name derived to vs the page actually served, so a surprising match is visible.
  requested_slug: { type: 'string' }, match_tier: { type: 'string' },
  // store mode only: the key the candidates came from.
  canonical_id: { type: 'string' },
  // Which of the keyed read's planes answered ("remote" | "local").
  read_plane: { type: 'string' },
  // Same kernel, another language: the wrong target_language was passed, not an empty store.
  other_language_pages: { type: 'array', items: { type: 'string' } },
  filtered: obj({
    total: { type: 'number' }, retired: { type: 'number' }, below_min_speedup: { type: 'number' },
    same_direction_collapsed: { type: 'number' }, demoted_by_hint: { type: 'number' },
    // The caveat on every number above: the service pages `--scan` rows, so a saturated read
    // describes a prefix of the page and reads identically to a thin page.
    scanned: { type: 'number' }, scan_saturated: { type: 'boolean' },
  }),
  candidates: {
    type: 'array',
    items: obj({
      rank: { type: 'number' }, slug: { type: 'string' }, speedup: { type: 'number' },
      exp_dir: { type: 'string' }, arch: { type: 'string' },
      // Declared for the same reason canonical_id is: without it the attest step below has no
      // address to write this run's verdict back to.
      session_id: { type: 'string' },
      patch_path: { type: 'string' }, prose_path: { type: 'string' },
      strategy: { type: 'string' }, status: { type: 'string' },
      // direction = the IDEA (one rank each).
      direction: { type: 'string' }, bench_key: { type: 'string' }, comparable: { type: 'boolean' },
    }, ['rank', 'patch_path']),
  },
}, ['read_reason']);
// Warm-start writer output = experience_store.py `write` JSON, verbatim.
const WARMSTART_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, reason: { type: 'string' }, slug: { type: 'string' },
  dir: { type: 'string' }, speedup: { type: 'number' },
  reproduced: { type: 'string' }, reproductions: { type: 'number' },
  // store mode only: where the same result landed on the KB plane.
  remote: obj({
    written: { type: 'boolean' }, reason: { type: 'string' }, canonical_id: { type: 'string' },
    session_id: { type: 'string' }, champion: { type: 'boolean' }, replaced: { type: 'boolean' },
  }),
}, ['written']);

// ---------------------------------------------------------------------------
// Prompt helpers.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// --- Hung-agent guard ------------------------------------------------------ An agent LLM call that
// HANGS (no response, no terminal error) blocks a parallel()/pipeline() round-barrier forever, so every
// agent() call is bounded: after AGENT_TIMEOUT_MS it resolves null and the round proceeds. The 3h default
// fires only on a hang: a MOE verify plus a GPU-lock wait takes ~95 min, and 60min killed such lanes
// (GEAK #606). Keep it under run_workflow.mjs's 4h spawn backstop. args.agent_timeout_ms=0 disables.
// A thrown API/agent error is retried AGENT_RETRIES times, then resolves null; a timeout is not retried.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 10800000, 10);
// The wait limit agentT applies: AGENT_TIMEOUT_MS, widened after Setup only for a UT-module multi-GPU
// task (UT_MR) so that an agent's up to 3 full UT runs + 2 fit (Benchmark runs an unweighted UT 3
// times, a weighted oracle once); 0 (off) stays off. For such a task the limit can exceed the
// standalone runtime's 4h spawn backstop named above; the backstop then ends the agent first, so
// start that runtime with --agent-timeout-ms at or above the wait limit the lane logs.
let AGENT_WAIT_MS = AGENT_TIMEOUT_MS;
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
// Labels whose agent the guard gave up on. The agent is still running; stopAfterTimeout() reads this.
const TIMED_OUT_LABELS = new Set();
// GEAK-ABLATION-ARMS-v1 (B6): registered mechanical dispatches are folded into the NEXT agent instead
// of getting an agent of their own; the command still runs, verbatim, as that agent's first action.
const ABLATION_ARM = String(A.ablation_arm || '').trim().toUpperCase();
const ABL_B6 = ABLATION_ARM === 'B6';
const ABL_PENDING = [];
const ABL_DRAINED = [];
function ablDrainPrelude() {
  if (!ABL_PENDING.length) return '';
  const cmds = ABL_PENDING.splice(0, ABL_PENDING.length);
  for (const c of cmds) ABL_DRAINED.push(c);
  return '## FIRST, run these registered maintenance commands EXACTLY as written, then continue with your task.\n' +
    cmds.map((c) => '```bash\n' + c + '\n```').join('\n') + '\n\n';
}

// LLM token+time accounting (PURELY ADDITIVE; args.llm_stats="false" makes it a no-op).
const LLM_STATS = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
const LLM_TL = { schema: 'geak.agent_timeline/1', workflow: 'kernel_lane', events: [], nested: [] };
const TL_ROLE_RE = /You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\./;
// Called AT DISPATCH (before the await) so LLM_TL.events is in dispatch order, not completion order.
function tlAgent(prompt, o, attempt, dispatchId) {
  if (!LLM_STATS) return null;
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
  // NIT (Astra re-review #6): thread the router's host dispatch id into the timeline event so the abort
  // receipt_ref (keyed on dispatch_id) is reachable END-TO-END — a post-mortem can match
  // commit_failure.receipt_ref.dispatch_id to THIS event's dispatch_id in the persisted agent_timeline/
  // ledger geak_report.py renders, closing "reachable by dispatch id".
  if (dispatchId != null) e.dispatch_id = dispatchId;
  LLM_TL.events.push(e);
  return e;
}

// --- Expt-3 cost ladder: route EVERY agent scope to the lowest sufficient model.
// <<ROUTING-INLINE-START>>
const ROUTE_LANES = ['claude-haiku-5-5', 'claude-sonnet-5-5', 'claude-opus-5-5'];
const ROUTE_LANE_OF = { small: 0, medium: 1, high: 2, escalate: 2 };
const ROUTE_TOP = ROUTE_LANES.length - 1;
const ROUTE_MODEL_BRAIN = ROUTE_LANES[ROUTE_TOP];
const ROUTE_MODEL_DECIDER = ROUTE_LANES[1];
const ROUTE_MODEL_CHEAP = ROUTE_LANES[0];
const ROUTE_BRAIN_ROLES = ['director', 'tech_lead'];
const ROUTE_SEP = String.fromCharCode(0);   // scope-key separator: textual in source, U+0000 at runtime
const ROUTE_TIER_MAP = (function () {
  const m = {};
  // Deterministic fixed-script helpers in the light knn lane (schema-forced, low reasoning):
  m['Optimize' + ROUTE_SEP + 'clock'] = ROUTE_MODEL_CHEAP;            // clock ${tag} (effort:'low', CLOCK_SCHEMA)
  m['Optimize' + ROUTE_SEP + 'storage:reclaim'] = ROUTE_MODEL_CHEAP; // storage:reclaim r${round}
  m['WarmStart' + ROUTE_SEP + 'warm_start:resolve'] = ROUTE_MODEL_CHEAP; // warm_start:resolve
  return m;
})();
const ROUTING_ON = (function () {
  // kernel_lane runs ONLY as a native-Workflow child lane (no `process`); A.routing is forwarded in
  // from the parent (e2e_workflow.js / kernel_workflow.js), so it is the single enable source here.
  const a = String(A.routing != null ? A.routing : '').trim().toLowerCase();
  return a === 'true' || a === '1' || a === 'on';   // anything else (incl. unset) stays OFF — default-OFF
})();
// Thresholds and caps.
function __routeNum(v, d) { const x = Number(v); return v != null && v !== '' && Number.isFinite(x) ? x : d; }
const ROUTE_CONF_ESCALATE = __routeNum(A.route_conf_escalate, 0.70);
const ROUTE_MAX_FAILS_PER_LANE = Math.max(1, __routeNum(A.route_max_retries_per_lane, 3));
// Worker dispatches on the top lane (Opus 5.5) per run: no cap unless route_max_top_escalations is set
// (null = none). With a cap, a capped dispatch runs one lane down. Never compare null numerically.
const ROUTE_MAX_TOP_DISPATCHES = (function () {
  const x = __routeNum(A.route_max_top_escalations, null);
  return x == null ? null : Math.max(0, x);
})();
const ROUTE_MAX_OUTPUT_TOKENS = Math.max(0, __routeNum(A.route_max_output_tokens, 1000000));
// Per-scope starting floors, {scope: lane 0..2} — the ONLY thing route_review.py proposes changing.
const ROUTE_FLOORS = (function () {
  let f = A.route_floors;
  if (typeof f === 'string') { try { f = JSON.parse(f); } catch (e) { f = null; } }
  return (f && typeof f === 'object') ? f : {};
})();
// Decider backend.
const ROUTE_DECIDER = String(A.route_decider || 'sonnet').trim().toLowerCase();
const ROUTE_DECISION_SCHEMA = {
  type: 'object',
  properties: {
    complexity: { type: 'string', enum: ['small', 'medium', 'high', 'escalate'] },
    action: { type: 'string', enum: ['continue', 'retry', 'verify', 'escalate', 'complete'] },
    scope_drift: { type: 'boolean' },
    confidence: { type: 'number', minimum: 0, maximum: 1 },
    risk_flag: { type: 'string', enum: ['none', 'security', 'data_loss'] },
  },
  required: ['complexity', 'action', 'scope_drift', 'confidence', 'risk_flag'],
  additionalProperties: false,
};
const __routeLadder = {};   // scope -> { lane, fails, pending, evidence, why, decision }
const __routeAudit = [];    // every decision, dispatch, failure and skip, in order
let __routeTopUsed = 0;
let __routeKilled = false;
let __routeJevWarned = false;

function __routeLabelPrefix(label) {
  const s = String(label == null ? '' : label);
  const sp = s.indexOf(' ');                // colon is part of static identity; a space starts dynamic text
  return sp >= 0 ? s.slice(0, sp) : s;
}
// Ladder identity.
function __routeScope(label) {
  const parts = String(label == null ? '' : label).split(' ');
  const rest = parts[1] || '';
  const c = rest.indexOf(':');
  return c >= 0 ? parts[0] + ':' + rest.slice(c + 1) : parts[0];
}
// Static part.
function __routeModel(opts) {
  if (!ROUTING_ON) return undefined;
  const pfx = __routeLabelPrefix(opts && opts.label);
  if (pfx.indexOf('route:') === 0) return ROUTE_MODEL_DECIDER;
  if (ROUTE_BRAIN_ROLES.indexOf(pfx.split(':')[0]) >= 0) return ROUTE_MODEL_BRAIN;
  return ROUTE_TIER_MAP[((opts && opts.phase) || '') + ROUTE_SEP + pfx];
}
function __routeValid(d) {
  return !!d && typeof d === 'object' && ROUTE_LANE_OF[d.complexity] != null &&
    typeof d.confidence === 'number' && d.confidence >= 0 && d.confidence <= 1 &&
    ['none', 'security', 'data_loss'].indexOf(d.risk_flag) >= 0 &&
    ['continue', 'retry', 'verify', 'escalate', 'complete'].indexOf(d.action) >= 0 &&
    typeof d.scope_drift === 'boolean';
}
// Pure: (previous ladder state, decider answer) -> next state.
function __routeStep(st, d, scope) {
  const ok = __routeValid(d);
  const floor = Math.max(0, Math.min(ROUTE_TOP, __routeNum(ROUTE_FLOORS[scope], 0)));
  if (!st) {
    const want = ok && d.confidence < ROUTE_CONF_ESCALATE ? ROUTE_LANE_OF[d.complexity] : 0;
    const lane = Math.max(floor, want);
    const why = !ok ? 'decider gave no valid answer -> default SMALL'
      : want ? `classified ${d.complexity}; small-lane confidence ${d.confidence} < ${ROUTE_CONF_ESCALATE}`
      : `classified ${d.complexity}; small-lane confidence ${d.confidence} >= ${ROUTE_CONF_ESCALATE} -> SMALL`;
    return { lane, fails: 0, pending: false, evidence: '', decision: ok ? d : null,
             why: floor > want ? `${why}; raised to floor ${floor}` : why };
  }
  const reasons = [];
  if (st.fails >= ROUTE_MAX_FAILS_PER_LANE) reasons.push(`${st.fails} failures on this lane`);
  if (ok && d.confidence < ROUTE_CONF_ESCALATE) reasons.push(`confidence ${d.confidence} < ${ROUTE_CONF_ESCALATE}`);
  if (ok && d.risk_flag !== 'none') reasons.push(`risk_flag=${d.risk_flag}`);
  const up = reasons.length > 0 && st.lane < ROUTE_TOP;
  return { lane: up ? st.lane + 1 : st.lane, fails: up ? 0 : st.fails, pending: false,
           evidence: st.evidence, decision: ok ? d : null,
           why: up ? 'escalate: ' + reasons.join('; ')
             : (ok ? 'retry on the same lane' : 'decider gave no valid answer -> retry on the same lane') };
}
function __routeSessionSpent() {
  try { return (typeof budget !== 'undefined' && budget && typeof budget.spent === 'function') ? budget.spent() : null; }
  catch (e) { return null; }
}
// budget.spent() is the whole SESSION's running output-token total, not this run's: a probe on
// 2026-09-28 read 4,727,204 before the lane began, and an absolute cap tripped on the first worker.
const ROUTE_SPENT_AT_START = __routeSessionSpent();
function __routeSpent() {
  const now = __routeSessionSpent();
  return now == null ? null : now - (ROUTE_SPENT_AT_START || 0);
}
// Kill switch.
function __routeOverBudget() {
  if (__routeKilled) return true;
  if (!ROUTE_MAX_OUTPUT_TOKENS) return false;
  const s = __routeSpent();
  if (s != null && s >= ROUTE_MAX_OUTPUT_TOKENS) {
    __routeKilled = true;
    __routeAudit.push({ event: 'kill_switch', spent_output_tokens: s, cap: ROUTE_MAX_OUTPUT_TOKENS });
    log(`  [route] KILL SWITCH: ${s} output tokens since this lane started >= cap ${ROUTE_MAX_OUTPUT_TOKENS} — no new rounds or worker dispatches.`);
  }
  return __routeKilled;
}
async function __routeClassify(p, o, scope, st) {
  if (ROUTE_DECIDER !== 'sonnet' && !__routeJevWarned) {
    __routeJevWarned = true;
    log(`  [route] decider '${ROUTE_DECIDER}' is not wired on this branch — using Sonnet 5.5.`);
  }
  const cur = st ? st.lane : 0;
  const ph = String((o && o.phase) || 'Route').replace(/[^A-Za-z0-9_.-]/g, '') || 'Route';
  const ev = st && st.evidence ? `\nLAST ATTEMPT on ${ROUTE_LANES[cur]} FAILED its deterministic check: ${st.evidence}\n` : '';
  const q = `You are the route_classifier. PHASE=${ph}.
Do not use any tools and do not open any file. Answer only from the text below, as StructuredOutput.
Decide how capable a model this task needs. Most tasks are smaller than they look.
- complexity: small = scripted, mechanical or schema-shaped; medium = bounded work with a clear check;
  high = open-ended engineering or debugging; escalate = only a frontier model can do it.
- confidence: 0..1, how likely ${ROUTE_LANES[cur]} is to finish this task correctly on its next attempt.
- action: what should happen next (continue | retry | verify | escalate | complete).
- scope_drift: true if the task, or the failed attempt, has wandered outside what was asked.
- risk_flag: security or data_loss if the work touches either, else none.

SCOPE: ${scope}
${ev}TASK (first 1500 characters):
${String(p == null ? '' : p).slice(0, 1500)}`;
  const run = typeof agentT === 'function' ? agentT : agent;
  try {
    return await run(q, { phase: (o && o.phase) || '', label: `route:classify ${scope}`, effort: 'low',
                          schema: ROUTE_DECISION_SCHEMA });
  } catch (e) { return null; }
}
// The whole decision for one dispatch.
async function __routePick(p, o) {
  const fixed = __routeModel(o);
  if (fixed) return { model: fixed };
  const scope = __routeScope(o && o.label);
  if (__routeOverBudget()) {
    __routeAudit.push({ event: 'skip', scope, label: (o && o.label) || '' });
    return { skip: true, scope };
  }
  if (ROUTE_EIKOS_ON) return __routeEikosPick(p, o, scope);
  if (ROUTE_P1_ON) return __p1Pick(p, o, scope);   // eikos v2 E1: gated, OFF by default (route_decider=p1 only)
  let st = __routeLadder[scope];
  if (!st || st.pending) {                   // classify once; RE-classify only after a failed cycle
    const d = await __routeClassify(p, o, scope, st);
    st = __routeStep(st, d, scope);
    __routeLadder[scope] = st;
    __routeAudit.push({ event: 'decide', scope, lane: st.lane, why: st.why, decision: st.decision });
  }
  let lane = st.lane, capped = false;
  // eikos v2 E1 (bug 4): snapshot the top-lane quota BEFORE the live dispatch consumes it, so the P1
  // shadow reduces on the SAME pre-dispatch state the live pick saw (reading it afterwards manufactures
  // a `live=Opus / shadow=defer` disagreement when the live pick took the last unit).
  const __p1TopUsedPre = __routeTopUsed;
  if (lane === ROUTE_TOP) {
    if (ROUTE_MAX_TOP_DISPATCHES != null && __routeTopUsed >= ROUTE_MAX_TOP_DISPATCHES) { lane = ROUTE_TOP - 1; capped = true; }
    else __routeTopUsed++;
  }
  __routeAudit.push({ event: 'dispatch', scope, label: (o && o.label) || '', phase: (o && o.phase) || '',
                      model: ROUTE_LANES[lane], capped });
  // eikos v2 E1 (bugs 3,4): pass the model ACTUALLY dispatched (ROUTE_LANES[lane], post
  // quota-downgrade) and the pre-dispatch quota snapshot; __p1Shadow records the dispatched model for
  // the outcome hook.
  const __p1Did = `d${++__routeDispatchSeq}`;
  const __p1Rec = { id: __p1Did, label: (o && o.label) || '', scope, seq: __routeDispatchSeq, model: ROUTE_LANES[lane],
                    outcome: null, p1Outcome: null, decider: 'sonnet' };
  __routeDispatchById[__p1Did] = __p1Rec;
  (__routeDispatches[(o && o.label) || ''] = __routeDispatches[(o && o.label) || ''] || []).push(__p1Rec);
  __p1Shadow('sonnet', scope, (o && o.label) || '', ROUTE_LANES[lane], __p1TopUsedPre);   // never changes the pick above
  return { model: ROUTE_LANES[lane], scope, dispatch_id: __p1Did };
}
// Deterministic outcome of one cycle for a worker scope.
function __routeOutcome(scope, ok, evidence, kind, source, label, dispatchId, p1Kind) {
  // When P1 is the LIVE decider there is no legacy decider to keep identical, so P1 gets the corrected kind.
  if (ROUTE_P1_ON) return __p1OutcomeLive(scope, ok, evidence, (p1Kind || kind) || 'work_product', source, label, dispatchId);  // gated; OFF by default
  const st = __routeLadder[scope];
  if (!st) return;
  if (st.decider === 'eikos') return __routeEikosOutcome(st, scope, ok, evidence, kind || 'work_product', source, label, dispatchId, p1Kind);
  // eikos v2 E1 shadow (bug 3 / R1): attribute to the model ACTUALLY dispatched.
  const __p1st = __p1Ladder[scope];
  const __p1d = dispatchId != null ? __routeDispatchById[dispatchId] : null;
  let __p1skip = false;
  // eikos v2 E1 (R1 residual / Astra re-review #3 finding 3): VALIDATE whenever an id was supplied
  // (dispatchId != null), not only when it RESOLVES (the pre-fix `if (__p1d)` let a NON-NULL but
  // UNKNOWN id bypass validation and silently bump the shadow floor via last_dispatch_model, with no
  // ignored-outcome event).
  if (dispatchId != null) {
    const __p1why = !__p1d ? 'unknown dispatch id'
      : (__p1d.scope !== scope || __p1d.label !== String(label || '')) ? 'dispatch id issued for another label or scope'
      : __p1d.p1Outcome != null ? `duplicate: p1 outcome already recorded (${__p1d.p1Outcome})` : '';
    if (__p1why) {
      __routeAudit.push({ event: 'p1_outcome_ignored', decider: 'sonnet', scope, label: label || '',
        dispatch_id: dispatchId, ok: !!ok, kind: kind || 'work_product', why: __p1why });
      __p1skip = true;
    }
  }
  if (!__p1skip) {
    const __p1Model = __p1d ? __p1d.model
      : (__p1st && __p1st.last_dispatch_model != null ? __p1st.last_dispatch_model : ROUTE_LANES[st.lane]);
    __p1Observe(scope, __p1Model, ok, (p1Kind || kind) || 'work_product');
    if (__p1d) __p1d.p1Outcome = ok ? 'ok' : ((p1Kind || kind) || 'work_product');
  }
  if (ok) { st.fails = 0; return; }
  st.fails++;
  st.pending = true;
  st.evidence = String(evidence == null ? '' : evidence).slice(0, 300);
  __routeAudit.push({ event: 'fail', scope, model: ROUTE_LANES[st.lane], fails: st.fails, evidence: st.evidence,
                      kind: kind || 'work_product' });
}
function __routeReport() {
  // eikos v2 E1 (bug 2): report the TRUE live decider.
  return { lanes: ROUTE_LANES, decider: ROUTE_P1_ON ? 'p1' : ROUTE_EIKOS_ON ? 'eikos' : ROUTE_MODEL_DECIDER,
           thresholds: { conf_escalate: ROUTE_CONF_ESCALATE, max_retries_per_lane: ROUTE_MAX_FAILS_PER_LANE,
                         max_top_escalations: ROUTE_MAX_TOP_DISPATCHES, max_output_tokens: ROUTE_MAX_OUTPUT_TOKENS,
                         floors: ROUTE_FLOORS },
           top_dispatches: __routeTopUsed, killed: __routeKilled, spent_output_tokens: __routeSpent(),
           session_output_tokens_at_start: ROUTE_SPENT_AT_START,
           ladder: __routeLadder, audit: __routeAudit,
           config_errors: __routeConfigErrors,   // arguments the lane could not use, and what it used instead
           // eikos v2 (E1): P1 measured-signal router. Shadow summary even when OFF; `live` is true only under route_decider=p1.
           p1: { reducer: P1_REDUCER_VERSION, live: ROUTE_P1_ON, shadow: __p1ShadowSummary(),
                 // eikos v2 E1 (bug 2): expose P1's OWN state separately from the legacy ladder, and the
                 // per-dispatch terminal outcomes (incl. successes) that otherwise lived only in memory.
                 ladder: __p1Ladder, dispatches: __routeDispatchById,
                 // eikos v2 E1 (R2): the measured P1 policy is SCOPED to the roles with a terminal
                 // success/work_product signal.
                 coverage: {
                   roles_with_terminal_feedback: ['eng/deep (engineer cycle outcome)', 'commit (committed=true/false)', 'verify (usable measurement = success; invalid/empty = infra)'],
                   roles_without_terminal_feedback: ['plan/replan (tech_lead)', 'profile/reprofile', 'tech_lead:memory', 'integrate', 'condense', 'route:classify', 'storage reclaim'],
                   note: 'roles_without_terminal_feedback still report no_result on a null return (moves nothing); their success/work_product is not yet distinguished at the host layer.' } },
           // route_decider=eikos only: its settings, all reported as unvalidated experimental values.
           eikos: ROUTE_EIKOS_ON ? { question: ROUTE_EIKOS_QUESTION_VERSION, policy: ROUTE_EIKOS_POLICY_VERSION,
             reducer: ROUTE_REDUCER_VERSION, tau: ROUTE_EIKOS_TAU, tau_status: 'unvalidated_experimental',
             policy_rank: ROUTE_LANES, price_order: ROUTE_PRICE_ORDER, price_order_is: 'list-price proxy, not predicted task cost',
             class_floor_rank: ROUTE_CLASS_FLOOR, class_max_fails: ROUTE_CLASS_MAX_FAILS,
             carrier_model: ROUTE_EIKOS_CARRIER_MODEL, carrier_timeout_ms: ROUTE_EIKOS_CARRIER_TIMEOUT_MS,
             consults: __routeConsults, max_consults: ROUTE_EIKOS_MAX_CONSULTS, choices: __routeChoiceCounts(),
             eligible_scopes: 'engineers with a registered direction; every other scope is routed by host policy',
             units: 'consultation = one carrier call; dispatch = one agentT call (its transport retries included); occurrence = one frozen card' } : undefined };
}
// --- Eikos decider (route_decider=eikos; OFF unless chosen) --------------------------------------
// Contract: research/eikos_ladder_routing_proposal_20261009.md rev 2, section 7, plus Astra's S0 plan
// review (2026-10-12).
//
// Units.
const ROUTE_EIKOS_ON = ROUTE_DECIDER === 'eikos';
// iraj465 (CHANGES_REQUESTED) Fix 1 (bug 1): run_e2e.py probes the Eikos carrier ONCE at startup; if it
// is down it sets the route_eikos_probe_failed workflow ARG (and, for the standalone top-level only,
// also exports GEAK_EIKOS_PROBE_FAILED=1).
const ROUTE_EIKOS_PROBE_DOWN = (function () {
  const a = String(A.route_eikos_probe_failed != null ? A.route_eikos_probe_failed : '').trim().toLowerCase();
  return a === '1' || a === 'true' || a === 'on';   // anything else (incl. unset) stays OFF
})();
// eikos v2 (E1): P1 is the DETERMINISTIC measured-signal host router (no model, no d3).
const ROUTE_P1_ON = ROUTE_DECIDER === 'p1';
const ROUTE_EIKOS_QUESTION = 'lane_sufficiency';
const ROUTE_EIKOS_QUESTION_VERSION = 'lane_sufficiency.v3';
const ROUTE_EIKOS_POLICY_VERSION = 'lane_sufficiency.policy.v1';
const ROUTE_REDUCER_VERSION = 'lane_reduce.v2';
const __routeConfigErrors = [];
if (['sonnet', 'eikos', 'jev', 'p1'].indexOf(ROUTE_DECIDER) < 0) {
  __routeConfigErrors.push(`route_decider=${JSON.stringify(ROUTE_DECIDER)} is unknown; the Sonnet decider is used`);
}
// The existing knobs keep their reviewed parsing; a value it would coerce is recorded, not hidden.
['route_max_top_escalations', 'route_max_retries_per_lane', 'route_max_output_tokens'].forEach((k) => {
  const v = A[k];
  if (v != null && v !== '' && !(typeof v === 'number' ? Number.isInteger(v) && v >= 0 : /^\s*\d+\s*$/.test(String(v)))) {
    __routeConfigErrors.push(`${k}=${JSON.stringify(v)} is not a nonnegative integer; it was coerced as before (see thresholds)`);
  }
});
if (A.route_conf_escalate != null && A.route_conf_escalate !== '' &&
    !(Number.isFinite(Number(A.route_conf_escalate)) && Number(A.route_conf_escalate) >= 0 && Number(A.route_conf_escalate) <= 1)) {
  __routeConfigErrors.push(`route_conf_escalate=${JSON.stringify(A.route_conf_escalate)} is not a number in [0, 1]`);
}
Object.keys(ROUTE_FLOORS).forEach((k) => {
  const v = ROUTE_FLOORS[k];
  if (!(Number.isInteger(Number(v)) && Number(v) >= 0 && Number(v) <= ROUTE_TOP)) {
    __routeConfigErrors.push(`route_floors.${k}=${JSON.stringify(v)} is not a lane number 0..${ROUTE_TOP}; it was clamped`);
  }
});
(function () {
  let f = A.route_floors;
  if (f == null || f === '') return;
  if (typeof f === 'string') { try { f = JSON.parse(f); } catch (e) { f = null; } }
  if (!f || typeof f !== 'object' || Array.isArray(f)) {
    __routeConfigErrors.push('route_floors is not a JSON object of {scope: lane}; no floors are used');
  }
})();
function __routeIntArg(name, dflt) {
  const v = A[name];
  if (v == null || v === '') return dflt;
  const x = typeof v === 'number' ? v : (/^\s*\d+\s*$/.test(String(v)) ? Number(v) : NaN);
  if (Number.isInteger(x) && x >= 0) return x;
  __routeConfigErrors.push(`${name}=${JSON.stringify(v)} is not a nonnegative integer; ${dflt} is used`);
  return dflt;
}
const ROUTE_EIKOS_TAU = (function () {
  const out = { paperwork: 0.70, evidence: 0.80, code_change: 0.90 };
  let t = A.route_eikos_tau;
  if (t == null || t === '') return out;
  if (typeof t === 'string') { try { t = JSON.parse(t); } catch (e) { t = null; } }
  if (!t || typeof t !== 'object' || Array.isArray(t)) {
    __routeConfigErrors.push('route_eikos_tau is not a JSON object; the built-in values are used');
    return out;
  }
  Object.keys(t).forEach((k) => {
    const v = t[k];
    if (!(k in out)) __routeConfigErrors.push(`route_eikos_tau has an unknown class ${JSON.stringify(k)}; ignored`);
    else if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > 1) {
      __routeConfigErrors.push(`route_eikos_tau.${k}=${JSON.stringify(v)} is not a number in [0, 1]; ${out[k]} is used`);
    } else out[k] = v;
  });
  return out;
})();
const ROUTE_EIKOS_MAX_CONSULTS = __routeIntArg('route_eikos_max_consults', 40);
const ROUTE_EIKOS_CARRIER_MODEL = (typeof A.route_eikos_carrier_model === 'string' && A.route_eikos_carrier_model.trim())
  ? A.route_eikos_carrier_model.trim() : 'claude-sonnet-5-5';
const ROUTE_EIKOS_CARRIER_TIMEOUT_MS = Math.max(10000, __routeIntArg('route_eikos_carrier_timeout_ms', 180000));
// Episode-vs-promotion identity (eikos v2).
const ROUTE_EPISODE_IDENTITY = (A.route_episode_identity === true
  || A.route_episode_identity === 1 || A.route_episode_identity === '1');
// Host POLICY rank for floors and escalation: the existing lane numbering (ROUTE_LANES), so a numeric
// operator floor keeps its meaning. A host policy, not a claim that capability rises with it.
function __routeRank(m) { return ROUTE_LANES.indexOf(m); }
// List-price proxy (output $ per million: 5, 10, 20), used ONLY to choose among lanes already
// allowed and feasible. Not predicted task cost: tokenizer, cache, retries and quota differ per lane.
const ROUTE_PRICE_ORDER = ['claude-haiku-5-5', 'claude-sonnet-5-5', 'claude-opus-5-5'];
const ROUTE_CLASS_FLOOR = { paperwork: 0, evidence: 1, code_change: 1 };        // policy rank
// Work-product failures a lane may take in one scope before every lane at or below its rank is out:
// a code change escalates at once; other classes get one retry. Counts never reset (finite policy).
const ROUTE_CLASS_MAX_FAILS = { paperwork: 2, evidence: 2, code_change: 1 };
const ROUTE_EIKOS_RUNTIME = 'claude-code workflow agent (Bash, Read, Write, Edit)';
const __routeTasks = {};       // engineer dispatch label -> its direction, registered at the dispatch site
const __routeSkipped = {};     // dispatch labels the router skipped (kill switch, deferral)
const __routeDispatches = {};  // dispatch label -> its dispatch records, for display
const __routeDispatchById = {}; // host-issued dispatch id -> { id, label, scope, task_key, seq, model, outcome }
const __routeLocks = {};       // scope -> tail of its decision chain (one decision at a time per scope)
let __routeClock = null;       // { left_s } as the round loop last measured it
let __routeConsults = 0;
let __routeDispatchSeq = 0;

function __routeScopeClass(scope) {
  const s = String(scope || '');
  const head = s.split(':')[0];
  if (['eng', 'deep', 'integrate', 'author'].indexOf(head) >= 0) return 'code_change';
  if (head === 'commit' || s === 'kb:cite') return 'paperwork';
  if (['benchmark_engineer', 'profile_engineer', 'reprofile', 'verify', 'warm_start', 'researcher', 'kb',
       'update_experience'].indexOf(head) >= 0) return 'evidence';
  return 'code_change';        // unknown: the strictest class
}
// A task is what the next attempt is asked to do: its direction (or label), the whole dispatch prompt
// and the requested effort.
function __routeTaskKey(label, p, o) {
  const t = __routeTasks[label];
  const who = t ? 'direction:' + t.id : 'label:' + String(label || '');
  return who + ':' + eikosSha256(JSON.stringify([t || null, String(p == null ? '' : p), (o && o.effort) || null])).slice(0, 16);
}
function __routeRoundOf(label) {
  const m = /\br(\d+)(?:_d\d+)?\b/.exec(String(label || ''));
  return m ? parseInt(m[1], 10) : 0;
}
function __routeOpFloorRank(scope) {
  const f = ROUTE_FLOORS[scope];
  return f == null ? 0 : Math.max(0, Math.min(ROUTE_TOP, __routeNum(f, 0)));
}
// Card size caps, mirrored by eikos_decide.py (MAX_DIRECTION_PROMPT_CHARS, MAX_STATE_CHARS).
const ROUTE_CARD_PROMPT_CHARS = 8000;
const ROUTE_CARD_STATE_CHARS = 12000;
const ROUTE_CONDENSE_TARGET_CHARS = 6000;
// The frozen card for one occurrence.
function __routeCard(p, o, scope, cls, occ, fit) {
  const label = String((o && o.label) || '');
  const d = (fit && fit.direction) || __routeTasks[label];
  if (!d) return { card: null, ineligible: 'no task projection for this scope (S0 asks only for engineers with a registered direction)' };
  const floor = typeof CANDIDATE_FLOOR !== 'undefined' ? CANDIDATE_FLOOR : 'configured';
  const prompt = String(p == null ? '' : p);
  const hasDeadline = typeof DEADLINE_EPOCH !== 'undefined' && !!DEADLINE_EPOCH;
  const clk = __routeClock && Number.isFinite(__routeClock.left_s) ? __routeClock : null;
  const card = {
    card_version: 1, scope, scope_class: cls, role: label.split(/[ :]/)[0] || 'agent',
    phase: String((o && o.phase) || ''), round: __routeRoundOf(label), occurrence: occ,
    task: {
      goal: 'Change the kernel source along this direction to make it faster, measure it, and leave best_patch.diff.',
      goal_source: 'direction',
      acceptance: `An independent verify step finds that the patch applies, the correctness check passes, and the measured speedup is above the candidate floor (${floor}). A correct patch that does not beat the floor does not meet this event.`,
      constraints: ['Work only in the private workspace the dispatch names.', 'Use the GPU only through gpu_lock.sh.',
                    'The engineer\'s own numbers are not accepted; verify re-measures.'].concat(fit
        ? [`The direction prompt here was condensed from ${fit.from_chars} to ${d.prompt.length} characters; the engineer gets the full text.`]
        : []),
      direction: d, dispatch_prompt_sha256: eikosSha256(prompt), dispatch_prompt_chars: prompt.length },
    budget: { directions_used: typeof dispatched !== 'undefined' ? dispatched : 0,
              directions_budget: typeof BUDGET !== 'undefined' ? BUDGET : 1,
              minutes_left: hasDeadline && clk ? Math.max(0, Math.round(clk.left_s / 60)) : 'unknown',
              minutes_left_source: !hasDeadline ? 'no_deadline' : (clk ? 'round_clock' : 'not_recorded') },
    config: { lanes: ROUTE_LANES.slice(), effort: (o && o.effort) || null, runtime: ROUTE_EIKOS_RUNTIME,
              attempt_timeout_s: (typeof AGENT_TIMEOUT_MS !== 'undefined' && AGENT_TIMEOUT_MS > 0) ? Math.round(AGENT_TIMEOUT_MS / 1000) : 'unknown',
              transport_retries: typeof AGENT_RETRIES !== 'undefined' ? AGENT_RETRIES : 0 },
  };
  const why = d.prompt.length > ROUTE_CARD_PROMPT_CHARS || d.title.length > 300 || String(d.id).length > 40 ||
    d.focus_files.length > 20 || d.focus_files.some((f) => f.length > 200 || !f.length) ? 'the direction does not fit the card'
    : (scope.length > 160 || card.phase.length > 40) ? 'scope or phase too long'
    : JSON.stringify(card).length > ROUTE_CARD_STATE_CHARS ? `card over ${ROUTE_CARD_STATE_CHARS} characters` : '';
  return { card, ineligible: why };
}
// The reduction's inputs, frozen before the consultation and persisted with it (--host-json).
function __routeHostInput(scope, cls) {
  const st = __routeLadder[scope];
  const fails = st ? st.fails_by_lane : {};
  return { reducer_version: ROUTE_REDUCER_VERSION, scope_class: cls, tau: ROUTE_EIKOS_TAU[cls],
           policy_rank: ROUTE_LANES.slice(), price_order: ROUTE_PRICE_ORDER.slice(),
           class_floor_rank: ROUTE_CLASS_FLOOR[cls], operator_floor_rank: __routeOpFloorRank(scope),
           scope_floor_rank: st ? st.floor_rank : ROUTE_CLASS_FLOOR[cls],
           effective_rank: st ? st.effective_rank : -1,
           episode_identity: ROUTE_EPISODE_IDENTITY,        // eikos v2 knob; default-off == current behavior
           excluded: ROUTE_LANES.filter((m) => (fails[m] || 0) >= ROUTE_CLASS_MAX_FAILS[cls]),
           top_quota_left: ROUTE_MAX_TOP_DISPATCHES == null ? null : Math.max(0, ROUTE_MAX_TOP_DISPATCHES - __routeTopUsed),
           available: ROUTE_LANES.slice() };
}
// Python's repr of a probability: '0.31', '1.0', '5e-05', '1.5e-07'. No sign, no prefix parsing.
const ROUTE_PROB_RE = /^(?:[01]\.[0-9]+|[0-9](?:\.[0-9]+)?e-[0-9]+)$/;
function __routeParseEstimates(s) {
  if (typeof s !== 'string') return null;
  const parts = s.split(';');
  if (parts.length !== ROUTE_LANES.length) return null;
  const out = {};
  for (let i = 0; i < parts.length; i++) {
    const kv = parts[i].split('=');
    if (kv.length !== 2 || kv[0] !== ROUTE_LANES[i] || !ROUTE_PROB_RE.test(kv[1])) return null;
    const v = Number(kv[1]);
    if (!Number.isFinite(v) || v < 0 || v > 1) return null;
    out[kv[0]] = v;
  }
  return out;
}
// One condensation: the carrier model rewrites a direction prompt that is over the card cap, so Eikos
// can still be asked about it.
const ROUTE_CONDENSE_SCHEMA = { type: 'object', properties: { prompt: { type: 'string' } },
                                required: ['prompt'], additionalProperties: false };
async function __routeCondense(d, scope, phase) {
  const prompt = `You are the route_condenser. PHASE=${String(phase || 'Route').replace(/[^A-Za-z0-9_.-]/g, '') || 'Route'}.
Below is the direction an engineer will receive (${d.prompt.length} characters). Rewrite it in at most ${ROUTE_CONDENSE_TARGET_CHARS} characters for a model that only estimates how hard the task is. Keep every concrete element: the bottleneck it targets, the code changes it asks for, the files and functions, the numbers, the expected gain and the risks. Drop repetition, background and step-by-step procedure. Add nothing that is not in the text. Do not run any tool and do not modify any file.
Return {"prompt": "<the condensed direction>"} as StructuredOutput.

DIRECTION TITLE: ${d.title}
DIRECTION:
${d.prompt}`;
  const opts = { phase, label: `eikos:condense ${scope}`, model: ROUTE_EIKOS_CARRIER_MODEL, schema: ROUTE_CONDENSE_SCHEMA };
  const ev = (typeof tlAgent === 'function') ? tlAgent(prompt, opts, 1) : null;
  let to;
  const guard = new Promise((resolve) => { to = setTimeout(() => resolve(null), ROUTE_EIKOS_CARRIER_TIMEOUT_MS); });
  const call = Promise.resolve().then(() => agent(prompt, opts)).then(
    (r) => { clearTimeout(to); return r; }, () => { clearTimeout(to); return null; });
  const r = await Promise.race([call, guard]);
  const s = r && typeof r.prompt === 'string' ? r.prompt.trim() : '';
  if (!s || s.length > ROUTE_CARD_PROMPT_CHARS) return null;
  if (ev) ev.ok = true;
  return Object.assign({}, d, { prompt: s });
}
// One consultation: the carrier runs ONE fixed command built here and relays its JSON.
async function __routeEikosConsult(card, host) {
  if (__routeConsults >= ROUTE_EIKOS_MAX_CONSULTS) return { status: 'consult_budget_exhausted' };
  __routeConsults++;
  const stateJson = JSON.stringify(card);
  const hostJson = JSON.stringify(host);
  const cmd = `python3 -B ${eikosShellQuote(WORKFLOW_DIR + '/../e2e_workflow/scripts/eikos_decide.py')} ` +
    `--decision ${ROUTE_EIKOS_QUESTION} --scope ${eikosShellQuote(EVAL_DIR)} --round ${card.round} ` +
    `--timeout-s ${EIKOS_CLI_TIMEOUT_S} --state-json ${eikosShellQuote(stateJson)} --host-json ${eikosShellQuote(hostJson)}`;
  const ph = String(card.phase || 'Route').replace(/[^A-Za-z0-9_.-]/g, '') || 'Route';
  const prompt = `You are the route_carrier. PHASE=${ph}.
Run EXACTLY this command and nothing else:
\`\`\`bash
${cmd}
\`\`\`
It prints one JSON object. Return that object unchanged as StructuredOutput. Do NOT edit the command, do NOT modify any file, and do NOT run anything else.`;
  const opts = { phase: card.phase, label: `eikos:route ${card.scope}`, model: ROUTE_EIKOS_CARRIER_MODEL,
                 schema: EIKOS_ENVELOPE_SCHEMA };
  const ev = (typeof tlAgent === 'function') ? tlAgent(prompt, opts, 1) : null;
  let to;
  const guard = new Promise((resolve) => { to = setTimeout(() => resolve({ __timeout: true }), ROUTE_EIKOS_CARRIER_TIMEOUT_MS); });
  const call = Promise.resolve().then(() => agent(prompt, opts)).then(
    (r) => { clearTimeout(to); return r; },
    (e) => { clearTimeout(to); return { __error: String((e && e.message) || e).slice(0, 200) }; });
  const env = await Promise.race([call, guard]);
  if (!env) return { status: 'no_result' };
  if (env.__timeout) return { status: 'timeout' };
  if (env.__error) return { status: 'error', error: env.__error };
  if (ev) ev.ok = true;
  const dec = {};
  EIKOS_DECISION_FIELDS.concat(['estimates', 'questions_version', 'policy_version', 'decision_host_sha256'])
    .forEach((k) => { dec[k] = env[k] === undefined ? null : env[k]; });
  const res = { envelope: env, relay_ok: env.state_sha256 === eikosSha256(stateJson),
                host_relay_ok: env.host_sha256 === eikosSha256(hostJson),
                decision_ok: env.decision_sha256 === eikosSha256(eikosCanonical(dec)), receipt_ok: env.receipt === 'written' };
  if (!res.relay_ok || !res.host_relay_ok) return Object.assign(res, { status: 'relay_mismatch' });
  if (!res.decision_ok) return Object.assign(res, { status: 'decision_altered' });
  if (env.status !== 'ok') return Object.assign(res, { status: String(env.status || 'malformed').slice(0, 40) });
  // Not replay-safe unless the decision is persisted AND its attempt receipted: otherwise a resume could
  // draw new estimates for the same occurrence. Unusable, like any other missing answer.
  if (env.persisted !== true || env.receipt !== 'written') return Object.assign(res, { status: 'not_persisted' });
  if (env.questions_version !== ROUTE_EIKOS_QUESTION_VERSION || env.policy_version !== ROUTE_EIKOS_POLICY_VERSION) {
    return Object.assign(res, { status: 'question_mismatch' });
  }
  const est = __routeParseEstimates(env.estimates);
  if (!est) return Object.assign(res, { status: 'malformed' });
  // Reused under a different host input (a resume after state moved on): the estimates stand, the
  // reduction runs on the CURRENT constraints, and the change is recorded as a host intervention.
  res.host_changed = env.decision_host_sha256 !== env.host_sha256;
  return Object.assign(res, { status: 'ok', estimates: est });
}
// Pure: every rule for one occurrence.
function __routeReduce(x) {
  const top = x.policy_rank[x.policy_rank.length - 1];
  const lo = Math.max(x.class_floor_rank, x.operator_floor_rank, x.scope_floor_rank,
                      x.episode_identity ? -1 : x.effective_rank);
  const allowed = x.policy_rank.filter((m, r) => r >= lo && x.excluded.indexOf(m) < 0);
  const filtered = [];
  const feasible = x.price_order.filter((m) => allowed.indexOf(m) >= 0).filter((m) => {
    const why = x.available.indexOf(m) < 0 ? 'unavailable'
      : (m === top && x.top_quota_left != null && !(x.top_quota_left > 0)) ? 'top-lane quota used' : null;   // null quota: no cap
    if (why) filtered.push({ model: m, why });
    return !why;
  });
  const base = { reducer_version: x.reducer_version, lower_rank: lo, allowed, feasible, filtered, excluded: x.excluded,
                 tau: x.tau, threshold_met: null, chosen_estimate: null, max_estimate: null, override: null };
  if (!feasible.length) {
    return Object.assign(base, { model: null, reason: `no feasible lane at policy rank >= ${lo}: the worker is deferred` });
  }
  if (x.backend !== 'ok' || !x.estimates) {
    return Object.assign(base, { model: feasible[0], reason: `no usable estimate (${x.backend}): cheapest feasible allowed lane` });
  }
  const est = (m) => x.estimates[m];
  const maxEst = Math.max.apply(null, feasible.map(est));
  const qualifying = feasible.filter((m) => est(m) >= x.tau);
  if (qualifying.length) {
    return Object.assign(base, { model: qualifying[0], threshold_met: true, chosen_estimate: est(qualifying[0]),
                                 max_estimate: maxEst, reason: `cheapest feasible lane with estimate >= ${x.tau}` });
  }
  let pick = feasible[0];
  feasible.forEach((m) => { if (est(m) > est(pick)) pick = m; });   // ties keep the cheaper lane
  return Object.assign(base, { model: pick, threshold_met: false, chosen_estimate: est(pick), max_estimate: maxEst,
    reason: `no feasible lane reached ${x.tau}: host policy override to the highest estimate`,
    override: { kind: 'no_lane_qualified', policy: 'highest_estimate', tau: x.tau, chosen_estimate: est(pick), max_estimate: maxEst } });
}
// How each decision was reached: an Eikos estimate met tau, a recorded host override, a host fallback with
// no usable estimate, or a deferral.
function __routeChoiceCounts() {
  const c = { eikos_qualified: 0, override_no_lane_qualified: 0, fallback_no_estimate: 0, deferred: 0 };
  __routeAudit.forEach((e) => {
    if (e.event !== 'decide' || e.decider !== 'eikos') return;
    if (!e.decided) c.deferred++;
    else if (e.threshold_met === true) c.eikos_qualified++;
    else if (e.threshold_met === false) c.override_no_lane_qualified++;
    else c.fallback_no_estimate++;
  });
  return c;
}
function __routeSerial(scope, fn) {
  const prev = __routeLocks[scope] || Promise.resolve();
  const next = prev.then(fn, fn);
  __routeLocks[scope] = next.then(() => {}, () => {});
  return next;
}
function __routeEikosPick(p, o, scope) {
  return __routeSerial(scope, async () => {
    const label = String((o && o.label) || '');
    const cls = __routeScopeClass(scope);
    const key = __routeTaskKey(label, p, o);
    const st = __routeLadder[scope] || (__routeLadder[scope] = {
      decider: 'eikos', scope_class: cls, occurrences: 0, floor_rank: ROUTE_CLASS_FLOOR[cls], effective_rank: -1, fails_by_lane: {},
      history: [], tasks: {}, model: null, lane: -1, why: '' });
    let tk = st.tasks[key];
    // eikos v2 E1 (bug 4): pre-dispatch top-lane quota snapshot for the P1 shadow (updated just before
    // this occurrence reserves a unit).
    let __p1TopUsedPre = __routeTopUsed;
    const fresh = !tk || tk.pending;
    if (fresh) {
      st.occurrences++;
      const occ = { seq: st.occurrences,
                    trigger: tk ? (tk.last_kind === 'work_product' ? 'work_product_failure' : 'non_improvement')
                      : (st.occurrences > 1 ? 'changed_task' : 'first_sight'),
                    prior: st.history.slice(-6) };
      const host = __routeHostInput(scope, cls);
      // Nothing to ask when no lane could be dispatched whatever the answer.
      const none = !__routeReduce(Object.assign({}, host, { backend: 'not_asked', estimates: null })).model;
      // A registered direction too long for the card is condensed first, and only when Eikos will be asked.
      const reg = __routeTasks[label];
      let fit = null;
      // iraj465 (CHANGES_REQUESTED) Fix 1 (bug 2): the failed-probe gate runs BEFORE every carrier --
      // both the eikos:condense condensation here AND the eikos:route consultation below.
      if (!none && !ROUTE_EIKOS_PROBE_DOWN && reg && reg.prompt.length > ROUTE_CARD_PROMPT_CHARS) {
        const short = await __routeCondense(reg, scope, (o && o.phase) || '');
        if (short) fit = { direction: short, from_chars: reg.prompt.length };
      }
      const c = __routeCard(p, o, scope, cls, occ, fit);
      // A failed startup probe is a no-answer: the pick falls back to the static control ladder
      // (__routeReduce estimates-null path; __routeReduce byte-identical), dispatching NO eikos:route
      // carrier.
      const res = ROUTE_EIKOS_PROBE_DOWN ? (none ? { status: 'not_asked_no_feasible_lane' } : { status: 'probe_down' })
        : c.ineligible ? { status: 'ineligible' } : none ? { status: 'not_asked_no_feasible_lane' }
        : await __routeEikosConsult(c.card, host);
      // Other scopes may have reserved the top lane while this one waited: reduce on the current inputs.
      const now = __routeHostInput(scope, cls);
      const red = __routeReduce(Object.assign({}, now, { backend: res.status, estimates: res.estimates || null }));
      // observability only (eikos v2 shadow): what the episode-identity policy WOULD choose; never changes live `red`.
      const __redShadow = __routeReduce(Object.assign({}, now, { backend: res.status, estimates: res.estimates || null, episode_identity: true }));
      __routeAudit.push({ event: 'shadow_identity', scope, label,
                          live_model: red.model, episode_model: __redShadow.model,
                          differ: red.model !== __redShadow.model,
                          effective_rank: now.effective_rank, lo_live: red.lower_rank, lo_episode: __redShadow.lower_rank });
      const interventions = [];
      if (res.host_changed) interventions.push('decision reused from a different host input; reduced on the current one');
      if (JSON.stringify(now) !== JSON.stringify(host)) interventions.push('host input changed during the consultation; reduced on the current one');
      // eikos v2 E1 (bug 4): the P1 shadow pick is emitted once per dispatch (below), on this same
      // pre-reservation quota the live `red` saw — not after the reservation consumes a top-lane unit.
      __p1TopUsedPre = __routeTopUsed;
      const reserved = red.model === ROUTE_MODEL_BRAIN;
      if (reserved) __routeTopUsed++;                       // reserved at decision time, released if not dispatched
      tk = st.tasks[key] = { key, seq: occ.seq, model: red.model, estimates: res.estimates || null,
                             backend: res.status, pending: false, last_kind: null, reserved };
      st.model = red.model; st.lane = red.model ? __routeRank(red.model) : -1; st.why = red.reason;
      __routeAudit.push({ event: 'decide', decider: 'eikos', scope, label, scope_class: cls, task_key: key, seq: occ.seq,
        trigger: occ.trigger, card_sha256: c.card ? eikosSha256(JSON.stringify(c.card)) : null, ineligible: c.ineligible || null,
        condensed: fit ? { from_chars: fit.from_chars, to_chars: fit.direction.prompt.length, model: ROUTE_EIKOS_CARRIER_MODEL } : null,
        backend: res.status, logical_key: res.envelope ? res.envelope.logical_key || null : null,
        relay_ok: res.relay_ok == null ? null : res.relay_ok, decision_ok: res.decision_ok == null ? null : res.decision_ok,
        receipt_ok: res.receipt_ok == null ? null : res.receipt_ok,
        persisted: res.envelope ? res.envelope.persisted === true : null, estimates: res.estimates || null,
        decided: red.model, threshold_met: red.threshold_met, chosen_estimate: red.chosen_estimate,
        max_estimate: red.max_estimate, override: red.override, interventions, reduce: red, why: red.reason });
    } else if (tk.model) {
      // A repeated dispatch of a decided occurrence passes the CURRENT hard constraints too: a failure
      // elsewhere in the scope may have excluded its lane, or the top-lane quota may be used.
      const now = __routeHostInput(scope, cls);
      const feasible = __routeReduce(Object.assign({}, now, { backend: 'check', estimates: null })).feasible;
      if (feasible.indexOf(tk.model) < 0) {
        now.scope_floor_rank = Math.max(now.scope_floor_rank, __routeRank(tk.model));
        const red = __routeReduce(Object.assign(now, { backend: tk.estimates ? 'ok' : tk.backend, estimates: tk.estimates }));
        __routeAudit.push({ event: 'override', scope, label, seq: tk.seq, kind: 'decided_lane_no_longer_feasible',
                            from: tk.model, to: red.model, threshold_met: red.threshold_met, why: red.reason });
        tk.model = red.model;
      }
      __p1TopUsedPre = __routeTopUsed;   // eikos v2 E1 (bug 4): pre-reservation quota for this repeat dispatch's shadow
      if (tk.model === ROUTE_MODEL_BRAIN) { tk.reserved = true; __routeTopUsed++; }   // feasible: a unit is left
    }
    const release = () => { if (tk.reserved) { tk.reserved = false; __routeTopUsed = Math.max(0, __routeTopUsed - 1); } };
    if (!tk.model) {
      __routeAudit.push({ event: 'skip', scope, label, seq: tk.seq, why: 'deferred: no feasible lane' });
      return { skip: true, deferred: true, scope, reason: 'no feasible lane (deferred)' };
    }
    if (__routeOverBudget()) {                               // the consultation itself may have spent the rest
      release();
      __routeAudit.push({ event: 'skip', scope, label, seq: tk.seq, why: 'kill switch tripped before the worker' });
      return { skip: true, scope, reason: 'output-token cap reached' };
    }
    tk.reserved = false;                                     // the reservation is now this dispatch
    const id = `d${++__routeDispatchSeq}`;
    const rec = { id, label, scope, task_key: key, seq: tk.seq, model: tk.model, outcome: null };
    __routeDispatchById[id] = rec;
    (__routeDispatches[label] = __routeDispatches[label] || []).push(rec);
    st.effective_rank = Math.max(st.effective_rank, __routeRank(tk.model));   // only a lane actually dispatched
    __routeAudit.push({ event: 'dispatch', scope, label, dispatch_id: id, seq: tk.seq, phase: (o && o.phase) || '',
                        model: tk.model, served_model: 'unverified', capped: false });
    // eikos v2 E1 (bug 4): ONE shadow event per actual dispatch (fresh or repeat), on the dispatched
    // model and the pre-reservation quota snapshot — a faithful dispatch-level disagreement rate.
    __p1Shadow('eikos', scope, label, tk.model, __p1TopUsedPre);   // never changes the live pick/dispatch above
    return { model: tk.model, scope, dispatch_id: id };
  });
}
// One dispatch's outcome, bound by the host-issued dispatch id that the dispatch site carried through
// engineer, verify and result (never by label order, never by a worker-supplied id).
function __routeEikosOutcome(st, scope, ok, evidence, kind, source, label, dispatchId, p1Kind) {
  const d = dispatchId != null ? __routeDispatchById[dispatchId] : null;
  const why = dispatchId == null ? 'no dispatch id' : !d ? 'unknown dispatch id'
    : (d.label !== String(label || '') || d.scope !== scope) ? 'dispatch id issued for another label or scope'
    : d.outcome != null ? `duplicate: outcome already recorded (${d.outcome})` : '';
  if (why) {
    __routeAudit.push({ event: 'outcome_ignored', scope, label: label || '', dispatch_id: dispatchId == null ? null : dispatchId,
                        ok, kind, why });
    return;
  }
  d.outcome = ok ? 'ok' : kind;
  __p1Observe(scope, d.model, ok, p1Kind || kind);   // eikos v2 E1 shadow (bug 5): corrected kind for P1 ONLY; live eikos below keeps `kind`
  if (ok) return;
  const ev = String(evidence == null ? '' : evidence).slice(0, 300);
  st.history = st.history.concat([{ lane: d.model, outcome: kind, evidence: ev, source: String(source || 'host').slice(0, 80) }]).slice(-6);
  if (kind === 'work_product') {
    st.fails_by_lane[d.model] = (st.fails_by_lane[d.model] || 0) + 1;
    if (st.fails_by_lane[d.model] >= ROUTE_CLASS_MAX_FAILS[st.scope_class]) st.floor_rank = Math.max(st.floor_rank, __routeRank(d.model) + 1);
  }
  const tk = st.tasks[d.task_key];
  const current = !!tk && tk.seq === d.seq;
  if (current && (kind === 'work_product' || kind === 'non_improvement')) { tk.pending = true; tk.last_kind = kind; }
  __routeAudit.push({ event: 'fail', decider: 'eikos', scope, label: label || '', dispatch_id: d.id, seq: d.seq, model: d.model,
    kind, evidence: ev, source: source || 'host', late: !current, fails_on_lane: st.fails_by_lane[d.model] || 0,
    scope_floor_rank: st.floor_rank, reconsult_next_attempt: current && (kind === 'work_product' || kind === 'non_improvement') });
}

// <<P1-ROUTER-START>>  (eikos v2 — E1: the P1 deterministic measured-signal host router)
// P1 reduces over MEASURED host signals ONLY — class/operator/scope floors, per-lane work_product
// failure counts and the corrected episode identity (effective_rank is a measured fact, NEVER a floor).
//
// FIRST LANDING = SHADOW ONLY.
const P1_REDUCER_VERSION = 'p1_measured.v1';
const __p1Ladder = {};   // scope -> P1's OWN measured-signal state, kept apart from __routeLadder
// A fresh P1 scope.
function __p1NewState(cls) {
  return { scope_class: cls, fails_by_lane: {}, floor_rank: ROUTE_CLASS_FLOOR[cls], effective_rank: -1,
           stagnation: 0, occurrences: 0,
           // eikos v2 E1 (bug 3): the model ACTUALLY dispatched for this scope's last pick, captured at
           // dispatch; the sonnet outcome hook attributes the failure to this, not the (possibly higher)
           // ladder lane, so a quota-downgraded Sonnet failure is not booked as an Opus failure.
           last_dispatch_model: null };
}
// The MEASURED reduction inputs for a scope, read from __p1Ladder.
function __p1HostInput(scope, cls, topUsedSnapshot) {
  const st = __p1Ladder[scope];
  const fails = st ? st.fails_by_lane : {};
  const used = topUsedSnapshot == null ? __routeTopUsed : topUsedSnapshot;
  return { reducer_version: P1_REDUCER_VERSION, scope_class: cls,
           policy_rank: ROUTE_LANES.slice(), price_order: ROUTE_PRICE_ORDER.slice(),
           class_floor_rank: ROUTE_CLASS_FLOOR[cls], operator_floor_rank: __routeOpFloorRank(scope),
           scope_floor_rank: st ? st.floor_rank : ROUTE_CLASS_FLOOR[cls],
           effective_rank: st ? st.effective_rank : -1,            // measured only; NOT a floor
           stagnation: st ? st.stagnation : 0,
           excluded: ROUTE_LANES.filter((m) => (fails[m] || 0) >= ROUTE_CLASS_MAX_FAILS[cls]),
           top_quota_left: ROUTE_MAX_TOP_DISPATCHES == null ? null : Math.max(0, ROUTE_MAX_TOP_DISPATCHES - used),
           available: ROUTE_LANES.slice() };
}
// Pure: the P1 signal->lane decision tree.
function __p1Reduce(x) {
  const top = x.policy_rank[x.policy_rank.length - 1];
  const lo = Math.max(x.class_floor_rank, x.operator_floor_rank, x.scope_floor_rank);  // effective_rank NOT a floor
  const allowed = x.policy_rank.filter((m, r) => r >= lo && x.excluded.indexOf(m) < 0);
  const filtered = [];
  const feasible = x.price_order.filter((m) => allowed.indexOf(m) >= 0).filter((m) => {
    const why = x.available.indexOf(m) < 0 ? 'unavailable'
      : (m === top && x.top_quota_left != null && !(x.top_quota_left > 0)) ? 'top-lane quota used' : null;   // null quota: no cap
    if (why) filtered.push({ model: m, why });
    return !why;
  });
  const base = { reducer_version: x.reducer_version, lower_rank: lo, allowed, feasible, filtered,
                 excluded: x.excluded, effective_rank: x.effective_rank, stagnation: x.stagnation,
                 scope_floor_rank: x.scope_floor_rank };
  if (!feasible.length) {
    return Object.assign(base, { model: null, reason: `no feasible lane at policy rank >= ${lo}: the worker is deferred` });
  }
  return Object.assign(base, { model: feasible[0],
    reason: `cheapest feasible lane at or above the earned floor ${lo} (deterministic; no model/d3 consult)` });
}
// Update P1's OWN measured state from a dispatch outcome.
function __p1Observe(scope, liveModel, ok, kind) {
  const st = __p1Ladder[scope];
  if (!st) return;                                   // only scopes P1 has shadowed/decided
  if (liveModel != null && __routeRank(liveModel) >= 0) {
    st.effective_rank = Math.max(st.effective_rank, __routeRank(liveModel));   // measured fact, not a floor
  }
  if (ok) { st.stagnation = 0; return; }
  if (kind === 'work_product' && liveModel != null && __routeRank(liveModel) >= 0) {
    st.fails_by_lane[liveModel] = (st.fails_by_lane[liveModel] || 0) + 1;
    if (st.fails_by_lane[liveModel] >= ROUTE_CLASS_MAX_FAILS[st.scope_class]) {
      st.floor_rank = Math.max(st.floor_rank, __routeRank(liveModel) + 1);   // earn the next lane up
    }
  } else if (kind === 'non_improvement') {
    st.stagnation += 1;                              // SEPARATE stagnation counter; never climbs the ladder
  }
  // invalid_measurement / no_result: not worker-attributable — neither excludes a lane nor stagnates.
}
// eikos v2 E1: the durable P1-shadow telemetry sink (a best-effort append-only mirror of each p1_shadow
// verdict to <EVAL_DIR>/reports/trace/p1_shadow.jsonl) was REMOVED.
function __p1Shadow(decider, scope, label, liveModel, topUsedSnapshot) {
  if (!__p1Ladder[scope]) __p1Ladder[scope] = __p1NewState(__routeScopeClass(scope));
  const st = __p1Ladder[scope];
  const cls = st.scope_class;
  // eikos v2 E1 (bug 3): remember the model actually dispatched so the outcome hook attributes the
  // failure to it (not the ladder lane). Only on an actual dispatch (a non-null live model).
  if (liveModel != null) st.last_dispatch_model = liveModel;
  // eikos v2 E1 (bug 4): reduce on the pre-dispatch quota the live pick saw, when supplied.
  const x = __p1HostInput(scope, cls, topUsedSnapshot);
  const red = __p1Reduce(x);
  const lp = liveModel == null ? null : liveModel;
  __routeAudit.push({ event: 'p1_shadow', decider, scope, label: label || '', scope_class: cls,
    live_pick: lp, p1_pick: red.model, differ: lp !== red.model,
    signals: { class_floor_rank: x.class_floor_rank, operator_floor_rank: x.operator_floor_rank,
               scope_floor_rank: x.scope_floor_rank, lower_rank: red.lower_rank, effective_rank: x.effective_rank,
               fails_by_lane: Object.assign({}, st.fails_by_lane), stagnation: x.stagnation,
               excluded: red.excluded, top_quota_left: x.top_quota_left, allowed: red.allowed, feasible: red.feasible },
    p1_reason: red.reason });
  return red;
}
// GATED LIVE path (route_decider=p1).
function __p1Pick(p, o, scope) {
  return __routeSerial(scope, async () => {
    const label = String((o && o.label) || '');
    const cls = __routeScopeClass(scope);
    if (!__p1Ladder[scope]) __p1Ladder[scope] = __p1NewState(cls);
    const st = __p1Ladder[scope];
    st.occurrences++;
    const x = __p1HostInput(scope, cls);
    const red = __p1Reduce(x);
    st.model = red.model; st.lane = red.model ? __routeRank(red.model) : -1; st.why = red.reason;
    __routeAudit.push({ event: 'decide', decider: 'p1', scope, label, scope_class: cls, seq: st.occurrences,
      decided: red.model, lower_rank: red.lower_rank, allowed: red.allowed, feasible: red.feasible,
      scope_floor_rank: x.scope_floor_rank, effective_rank: x.effective_rank, stagnation: x.stagnation,
      excluded: red.excluded, reduce: red, why: red.reason });
    if (!red.model) {
      __routeAudit.push({ event: 'skip', decider: 'p1', scope, label, seq: st.occurrences, why: 'deferred: no feasible lane' });
      return { skip: true, deferred: true, scope, reason: 'no feasible lane (deferred)' };
    }
    if (__routeOverBudget()) {
      __routeAudit.push({ event: 'skip', decider: 'p1', scope, label, seq: st.occurrences, why: 'kill switch tripped before the worker' });
      return { skip: true, scope, reason: 'output-token cap reached' };
    }
    if (red.model === ROUTE_MODEL_BRAIN) __routeTopUsed++;        // consume a top-lane unit, like the other deciders
    const id = `d${++__routeDispatchSeq}`;
    st.effective_rank = Math.max(st.effective_rank, __routeRank(red.model));
    const rec = { id, label, scope, seq: st.occurrences, model: red.model, outcome: null, decider: 'p1' };
    __routeDispatchById[id] = rec;
    (__routeDispatches[label] = __routeDispatches[label] || []).push(rec);
    __routeAudit.push({ event: 'dispatch', decider: 'p1', scope, label, dispatch_id: id, seq: st.occurrences,
                        phase: (o && o.phase) || '', model: red.model, capped: false });
    return { model: red.model, scope, dispatch_id: id };
  });
}
// GATED LIVE outcome handler (route_decider=p1).
function __p1OutcomeLive(scope, ok, evidence, kind, source, label, dispatchId) {
  const d = dispatchId != null ? __routeDispatchById[dispatchId] : null;
  const why = dispatchId == null ? 'no dispatch id' : !d ? 'unknown dispatch id'
    : (d.label !== String(label || '') || d.scope !== scope) ? 'dispatch id issued for another label or scope'
    : d.outcome != null ? `duplicate: outcome already recorded (${d.outcome})` : '';
  if (why) {
    __routeAudit.push({ event: 'outcome_ignored', decider: 'p1', scope, label: label || '',
                        dispatch_id: dispatchId == null ? null : dispatchId, ok, kind, why });
    return;
  }
  d.outcome = ok ? 'ok' : (kind || 'work_product');
  __p1Observe(scope, d.model, ok, kind || 'work_product');
  const ev = String(evidence == null ? '' : evidence).slice(0, 300);
  const st = __p1Ladder[scope];
  // eikos v2 E1 (bug 2): emit ONE terminal outcome per completed dispatch — INCLUDING successes, which
  // previously returned here without any audit event (their status lived only in __routeDispatchById).
  if (ok) {
    __routeAudit.push({ event: 'outcome', decider: 'p1', scope, label: label || '', dispatch_id: d.id, seq: d.seq,
      model: d.model, ok: true, kind: 'ok', evidence: ev, source: source || 'host',
      effective_rank: st ? st.effective_rank : null, scope_floor_rank: st ? st.floor_rank : null,
      stagnation: st ? st.stagnation : null });
    return;
  }
  __routeAudit.push({ event: 'fail', decider: 'p1', scope, label: label || '', dispatch_id: d.id, seq: d.seq,
    model: d.model, kind: kind || 'work_product', evidence: ev, source: source || 'host',
    fails_on_lane: st ? (st.fails_by_lane[d.model] || 0) : 0, scope_floor_rank: st ? st.floor_rank : null,
    stagnation: st ? st.stagnation : null });
}
// eikos v2 E1 (bug 1): route a NON-ENGINEER worker's attributable terminal outcome to P1 ONLY — the
// gated live-P1 handler under route_decider=p1, else P1's SHADOW state.
function __p1WorkerOutcome(scope, ok, evidence, kind, source, label, dispatchId) {
  if (!ROUTING_ON || !scope) return;
  if (ROUTE_P1_ON) { __p1OutcomeLive(scope, ok, evidence, kind || 'work_product', source, label, dispatchId); return; }
  // SHADOW path (sonnet/eikos live).
  const d = dispatchId != null ? __routeDispatchById[dispatchId] : null;
  if (dispatchId != null) {
    const why = !d ? 'unknown dispatch id'
      : (d.scope !== scope || d.label !== String(label || '')) ? 'dispatch id issued for another label or scope'
      : d.p1Outcome != null ? `duplicate: p1 outcome already recorded (${d.p1Outcome})` : '';
    if (why) {
      __routeAudit.push({ event: 'p1_worker_outcome_ignored', decider: ROUTE_EIKOS_ON ? 'eikos' : 'sonnet',
        scope, label: label || '', dispatch_id: dispatchId, ok: !!ok, kind: kind || 'work_product', why });
      return;
    }
  }
  const model = d ? d.model : (__p1Ladder[scope] ? __p1Ladder[scope].last_dispatch_model : null);
  __p1Observe(scope, model, ok, kind || 'work_product');   // shadow state only
  if (d) d.p1Outcome = ok ? 'ok' : (kind || 'work_product');
  const st = __p1Ladder[scope];
  __routeAudit.push({ event: 'p1_worker_outcome', decider: ROUTE_EIKOS_ON ? 'eikos' : 'sonnet', scope,
    label: label || '', ok: !!ok, kind: kind || 'work_product', model: model || null,
    evidence: String(evidence == null ? '' : evidence).slice(0, 300),
    fails_by_lane: st ? Object.assign({}, st.fails_by_lane) : {},
    scope_floor_rank: st ? st.floor_rank : null, stagnation: st ? st.stagnation : null });
}
// eikos v2 E1 (R8 / Astra re-review #9 BUG 3): the ONE verdict table for a dispatched commit attempt
// (initial / retry).
function __commitAttemptOutcome(receipt, route, scopeLabel) {
  if (!ROUTING_ON) return;
  const did = (route && route.dispatch_id) || null;
  const sc = __routeScope(scopeLabel);
  if (receipt != null && receipt.committed === true) {
    __p1WorkerOutcome(sc, true, 'committed=true', 'work_product', 'host', scopeLabel, did);
  } else if (receipt != null && receipt.committed === false) {
    __p1WorkerOutcome(sc, false, 'committed=false', 'work_product', 'host', scopeLabel, did);
  } else {
    __p1WorkerOutcome(sc, false,
      receipt == null ? 'no_result: no transcript/ambiguous (writer termination unconfirmed)'
                      : 'no_result: receipt without an explicit committed verdict (ambiguous, not a work_product failure)',
      'no_result', 'host', scopeLabel, did);
  }
}
// Report summary: how often P1 would have differed from the live pick, and the direction of each difference.
function __p1ShadowSummary() {
  const ev = __routeAudit.filter((e) => e.event === 'p1_shadow');
  const differ = ev.filter((e) => e.differ === true);
  const byDir = {};
  differ.forEach((e) => {
    const k = `${e.live_pick || 'defer'}->${e.p1_pick || 'defer'}`;
    byDir[k] = (byDir[k] || 0) + 1;
  });
  const cheaperOrEqual = differ.filter((e) => e.p1_pick && e.live_pick &&
    __routeRank(e.p1_pick) <= __routeRank(e.live_pick)).length;
  return { shadow_events: ev.length, differ: differ.length, p1_cheaper_or_equal: cheaperOrEqual, transitions: byDir };
}
// <<P1-ROUTER-END>>
// <<ROUTING-INLINE-END>>

// `route`, when given, is a host-owned object the caller keeps with its pipeline context: agentT writes
// the dispatch id the router issued (null if nothing was dispatched), and the caller passes it back with
// the outcome.
async function agentT(p, o, route) {
  if (!ROUTING_ON) return agentTCore(p, o);
  const pick = await __routePick(p, o);
  if (route) route.dispatch_id = pick.skip ? null : (pick.dispatch_id || null);
  if (pick.skip) {
    __routeSkipped[(o && o.label) || ''] = true;
    log(`  [route] ${(o && o.label) || 'agent'} skipped — ${pick.reason || 'output-token cap reached'}.`);
    return null;
  }
  if (pick.model) {                         // clone opts, don't mutate
    o = Object.assign({}, o, { model: pick.model });
    try { log(`  [route] ${(o && o.label) || 'agent'} @${(o && o.phase) || ''} -> ${pick.model}`); } catch (e) {}
  }
  // eikos v2 E1 (Astra re-review #6 — NIT): thread the host dispatch id the router just issued (the
  // SAME id written to `route.dispatch_id` above, and the one the abort `receipt_ref` keys on) into the
  // agent timeline event, so the commit's execution evidence is reachable BY DISPATCH ID end-to-end.
  const r = await agentTCore(p, o, pick.dispatch_id || null);
  // An engineer's cycle is judged by the oracle after verify (see the Optimize round), not here.
  if (pick.scope && !/^(eng|deep):/.test(pick.scope) && r == null) {
    __routeOutcome(pick.scope, false, 'no result (timeout, API fault or lost StructuredOutput)', 'no_result',
                   (o && o.label) || '', (o && o.label) || '', pick.dispatch_id);
  }
  return r;
}

// Per-call options (stripped before the agent call): timeout_ms overrides AGENT_TIMEOUT_MS, max_retries
// overrides AGENT_RETRIES, and timeout_marker makes a timeout return {__agent_timed_out, label} instead of
// null, for callers that must tell a hang from an empty answer (mode=mega).
async function agentTCore(p, o) {
  // eikos v2 E1 (Astra re-review #6 — NIT): the router's host dispatch id is threaded as an OPTIONAL
  // 3rd argument and read via `arguments` so this function's declared signature stays two-arg — the
  // in-repo routing test (scripts/test_eikos_route.js) slices the agentT region using this function's
  // two-arg declaration as its end sentinel, so a signature change would break that slice.
  const dispatchId = arguments.length > 2 ? arguments[2] : null;
  const label = (o && o.label) ? o.label : 'agent';
  const timeoutMs = Number(o && o.timeout_ms) > 0 ? Number(o.timeout_ms) : AGENT_WAIT_MS;
  const timeoutMarker = !!(o && o.timeout_marker);
  const retries = Number(o && o.max_retries) > 0 ? Math.max(1, Number(o.max_retries)) : AGENT_RETRIES;
  if (o && ('timeout_ms' in o || 'timeout_marker' in o || 'max_retries' in o)) {
    o = Object.fromEntries(Object.entries(o).filter(([k]) => !['timeout_ms', 'timeout_marker', 'max_retries'].includes(k)));
  }
  if (ABL_B6 && typeof p === 'string') { const pre = ablDrainPrelude(); if (pre) p = pre + p; }
  for (let attempt = 1; attempt <= retries; attempt++) {
    const ev = tlAgent(p, o, attempt, dispatchId);   // record AT DISPATCH; ok=false until it resolves (dispatch id threaded for receipt-by-id — Astra re-review #6)
    try {
      if (typeof setTimeout !== 'function' || !(timeoutMs > 0)) {
        const r0 = await agent(p, o); if (ev && r0) ev.ok = true; return r0;
      }
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(timeoutMs / 60000)}min with no return — ` +
            `${timeoutMarker ? 'returning a timeout marker' : 'resolving null so the round proceeds'}.`);
          if (!timeoutMarker) TIMED_OUT_LABELS.add(label);
          resolve(timeoutMarker ? { __agent_timed_out: true, label } : null);
        }, timeoutMs);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      const r = await Promise.race([
        agent(p, o).then((rr) => { clearTimeout(to); return rr; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
      if (ev && r) ev.ok = true;
      return r;
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < retries) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${retries} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${retries} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}

// eikos v2 E1 (R4 round-4 / Astra re-review #4): the round-3 `agentGuaranteed` helper was removed with
// the restore-and-continue recovery it served (its only two call sites — the pre-attempt snapshot and
// the recovery dispatch — are gone under ABORT-AND-QUARANTINE).

// LANE STOP. A null from agentT does not stop the agent: a timed-out benchmark_engineer kept its UT
// runs on the GPU for 2+ hours after its lane had failed, and the retried lane timed out waiting for
// that GPU (GEAK #606). So before the lane ENDS on an agent that returned nothing, it stops everything
// still running under its EVAL_DIR and blocks gpu_lock.sh there (scripts/stop_lane_work.py). Only on
// lane-ending paths: the stop file blocks every later GPU command under EVAL_DIR. Non-fatal.
// The standalone runtime exposes a host hook that bypasses the saturated agent queue. Native
// Workflow hosts without that capability retain the best-effort, one-command agent fallback.
const STOP_LANE_SCHEMA = obj({ ok: { type: 'boolean' }, blocked: { type: 'boolean' },
  stopped: { type: 'array', items: { type: 'object', additionalProperties: true } },
  survivors: { type: 'array', items: { type: 'object', additionalProperties: true } } },
  ['ok', 'blocked', 'stopped', 'survivors']);
async function stopLaneWork(evalDir, reason, phaseName) {
  if (!evalDir) return null;
  let r;
  if (typeof agent.stopLaneWork === 'function') {
    try {
      r = await agent.stopLaneWork(evalDir, reason);
    } catch (e) {
      log(`  [lane-stop] ${reason}: host cleanup failed (${String(e)}; NON-FATAL).`);
      return null; // Do not fall back into the queue we are trying to unblock.
    }
  } else {
    const q = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;
    const cmd = `python3 ${q(`${WORKFLOW_DIR}/scripts/stop_lane_work.py`)} --eval-dir ${q(evalDir)} --block --reason ${q(reason)}`;
    r = await agentT(
      'You are the kernel lane\'s GPU release step. Run EXACTLY this command, once, and return its JSON ' +
      'stdout verbatim as StructuredOutput. It stops only processes inside this lane\'s EVAL_DIR and never ' +
      'signals your own shell or its ancestors. Do not edit it, retry it, or act on what it reports; a ' +
      'non-zero exit is a valid answer, so return what it printed.\n```bash\n' + cmd + '\n```',
      { phase: phaseName, label: 'stop:lane', schema: STOP_LANE_SCHEMA });
  }
  if (!r) { log(`  [lane-stop] ${reason}: the stop step returned nothing (NON-FATAL).`); return null; }
  const n = Array.isArray(r.stopped) ? r.stopped.length : 0;
  const left = Array.isArray(r.survivors) ? r.survivors.length : 0;
  log(`  [lane-stop] ${reason}: stopped ${n} process(es) under ${evalDir}${left ? `, ${left} still up` : ''}; gpu_lock.sh ${r.blocked ? 'blocked' : 'NOT blocked'} there.`);
  return r;
}

// The same release when the lane carries on without a timed-out agent (engineer, verify, integrate):
// stop and block only that agent's own dir, so the next round does not queue behind its GPU work.
// `dir` must be where the agent runs gpu_lock.sh, and nothing the lane still needs may run under it.
async function stopAfterTimeout(r, label, dir, phaseName) {
  if (r == null && TIMED_OUT_LABELS.delete(label)) await stopLaneWork(dir, `${label} timed out`, phaseName);
  return r;
}

// ---------------------------------------------------------------------------
// WALL-CLOCK DEADLINE (opt-in; absent => byte-identical to a build without it).
const DEADLINE_EPOCH = (() => {
  const v = parseInt(A.deadline_epoch != null ? A.deadline_epoch : 0, 10);
  return Number.isFinite(v) && v > 0 ? v : 0;
})();
const NO_STOP_S = Math.max(0, parseInt(A.no_stop_s != null ? A.no_stop_s : 900, 10));
// Run-level cap on how many TechLead stops may be refused (forcedReplans is not reset per round).
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

// --- Eikos shadow pilot (OFF unless eikos_shadow=round_continue) -----------------------------
// Plumbing pilot for research/eikos_decision_audit_20261001.md sec 5.
// <<EIKOS-SHADOW-START>>
const EIKOS_SHADOW_ON = String(A.eikos_shadow || '').split(',').map((x) => x.trim()).indexOf('round_continue') >= 0;
const EIKOS_CARRIER_MODEL = String(A.eikos_carrier_model || 'claude-sonnet-5-5');
const EIKOS_CARRIER_TIMEOUT_MS = Math.max(10000, parseInt(A.eikos_carrier_timeout_ms != null ? A.eikos_carrier_timeout_ms : 180000, 10));
const EIKOS_CLI_TIMEOUT_S = 30;
const EIKOS_SHADOW_LOG = [];
const EIKOS_ENVELOPE_SCHEMA = { type: 'object', properties: { status: { type: 'string' } },
                                required: ['status'], additionalProperties: true };
const eikosSpecialties = new Set();
let eikosLastOutcome = 'none';     // set from measurements at the end of each round

function eikosShellQuote(x) { return "'" + String(x).replace(/'/g, "'\\''") + "'"; }

// SHA-256 of a string's UTF-8 bytes, hex.
function eikosSha256(str) {
  const b = unescape(encodeURIComponent(String(str)));
  const K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];
  const H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const bytes = [];
  for (let i = 0; i < b.length; i++) bytes.push(b.charCodeAt(i) & 0xff);
  const bitLen = bytes.length * 8;
  bytes.push(0x80);
  while (bytes.length % 64 !== 56) bytes.push(0);
  for (let i = 7; i >= 0; i--) bytes.push(i >= 4 ? 0 : (bitLen >>> (i * 8)) & 0xff);
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));
  const W = new Array(64);
  for (let off = 0; off < bytes.length; off += 64) {
    for (let t = 0; t < 16; t++) {
      W[t] = (bytes[off + 4 * t] << 24) | (bytes[off + 4 * t + 1] << 16) | (bytes[off + 4 * t + 2] << 8) | bytes[off + 4 * t + 3];
    }
    for (let t = 16; t < 64; t++) {
      const s0 = rotr(W[t - 15], 7) ^ rotr(W[t - 15], 18) ^ (W[t - 15] >>> 3);
      const s1 = rotr(W[t - 2], 17) ^ rotr(W[t - 2], 19) ^ (W[t - 2] >>> 10);
      W[t] = (W[t - 16] + s0 + W[t - 7] + s1) | 0;
    }
    let [a, bb, c, d, e, f, g, h] = H;
    for (let t = 0; t < 64; t++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const t1 = (h + S1 + ((e & f) ^ (~e & g)) + K[t] + W[t]) | 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const t2 = (S0 + ((a & bb) ^ (a & c) ^ (bb & c))) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = bb; bb = a; a = (t1 + t2) | 0;
    }
    H[0] = (H[0] + a) | 0; H[1] = (H[1] + bb) | 0; H[2] = (H[2] + c) | 0; H[3] = (H[3] + d) | 0;
    H[4] = (H[4] + e) | 0; H[5] = (H[5] + f) | 0; H[6] = (H[6] + g) | 0; H[7] = (H[7] + h) | 0;
  }
  return H.map((x) => ('00000000' + (x >>> 0).toString(16)).slice(-8)).join('');
}

// The frozen state. Every field names its source; see the audit's provenance table.
function eikosSnapshot(roundNo, leftS, leftSource) {
  return {
    round: roundNo,
    directions_used: dispatched,                 // direction engineers dispatched (not $ or tokens)
    directions_budget: BUDGET,
    tracked_incumbent_speedup: cumulative,       // the lane's tracked best; commit NOT independently confirmed
    best_seen_speedup: bestSeen,                 // best verified candidate, committed or not
    rounds_without_improvement: noImprove,
    rounds_without_improvement_limit: MAX_NO_IMPROVE,
    last_round_outcome: eikosLastOutcome,
    specialties_dispatched: Array.from(eikosSpecialties).sort(),
    minutes_left: (DEADLINE_EPOCH && Number.isFinite(leftS)) ? Math.round(leftS / 60) : 'unknown',
    minutes_left_source: DEADLINE_EPOCH ? leftSource : 'no_deadline',
    noise_band: 'unknown',                       // not tracked by the lane; never estimated
    min_improve: MIN_IMPROVE,                    // configured threshold, not a noise measurement
    progress_delta: PROGRESS_DELTA,              // configured threshold, not a noise measurement
  };
}

// The host's own refusal predicate (the forced re-plan loop guard), evaluated on the same values.
function eikosStopPermitted(leftS) {
  if (DEADLINE_EPOCH && !Number.isFinite(leftS)) return { permitted: false, reason: 'clock_unavailable' };
  if (DEADLINE_EPOCH && leftS > NO_STOP_S && forcedReplans < MAX_FORCED_REPLANS) {
    return { permitted: false, reason: 'fixed_window' };
  }
  return { permitted: true, reason: null };
}

// Python json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False), for the decision
// digest.
function eikosCanonical(v) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v === undefined ? null : v);
  if (Array.isArray(v)) return '[' + v.map(eikosCanonical).join(',') + ']';
  return '{' + Object.keys(v).sort().map((k) => JSON.stringify(k) + ':' + eikosCanonical(v[k])).join(',') + '}';
}
const EIKOS_DECISION_FIELDS = ['attempt_id', 'decision_attempt_id', 'logical_key', 'status', 'choice',
  'confidence', 'would_be_action', 'reused', 'persisted', 'receipt'];

function eikosRawChoice(p) {
  if (!p) return 'no_plan';
  if (p.stop) return 'stop';
  return (p.directions && p.directions.length) ? 'continue' : 'empty_directions';
}

function eikosCommand(stateJson, roundNo) {
  return `python3 -B ${eikosShellQuote(WORKFLOW_DIR + '/../e2e_workflow/scripts/eikos_decide.py')} ` +
    `--decision round_continue --scope ${eikosShellQuote(EVAL_DIR)} --round ${roundNo} ` +
    `--timeout-s ${EIKOS_CLI_TIMEOUT_S} --state-json ${eikosShellQuote(stateJson)}`;
}

// One attempt, own wait bound, no retries. Returns the envelope, or {__timeout} / {__error}.
async function eikosCarrier(stateJson, roundNo) {
  const prompt = `Run EXACTLY this command and nothing else:
\`\`\`bash
${eikosCommand(stateJson, roundNo)}
\`\`\`
It prints one JSON object. Return that object unchanged as StructuredOutput. Do NOT edit the command, do NOT modify any file, and do NOT run anything else.`;
  const opts = { phase: 'Optimize', label: `eikos:round_continue r${roundNo}`,
                 model: EIKOS_CARRIER_MODEL, schema: EIKOS_ENVELOPE_SCHEMA };
  const ev = (typeof tlAgent === 'function') ? tlAgent(prompt, opts, 1) : null;
  let to;
  const guard = new Promise((resolve) => { to = setTimeout(() => resolve({ __timeout: true }), EIKOS_CARRIER_TIMEOUT_MS); });
  const call = Promise.resolve().then(() => agent(prompt, opts)).then(
    (r) => { clearTimeout(to); return r; },
    (e) => { clearTimeout(to); return { __error: String((e && e.message) || e).slice(0, 200) }; });
  const r = await Promise.race([call, guard]);
  if (ev && r && !r.__timeout && !r.__error) ev.ok = true;
  return r;
}

// The shadow call at the effective decision point.
async function eikosShadowRound(roundNo, snap, firstChoice, finalPlan, forcedThisRound, leftS,
                                rawFinal, hostReason) {
  const finalChoice = eikosRawChoice(finalPlan);          // what the host acts on
  const effective = finalChoice === 'continue' ? 'continue' : 'stop';
  const exhausted = DEADLINE_EPOCH && Number.isFinite(leftS) && leftS > NO_STOP_S && forcedReplans >= MAX_FORCED_REPLANS;
  const rec = {
    decision: 'round_continue', round: roundNo,
    raw_model_choice: { first: firstChoice, final: rawFinal != null ? rawFinal : finalChoice },
    forced_replans_this_round: forcedThisRound,
    effective_baseline_action: effective,
    host_stop_reason: effective === 'stop'
      ? (hostReason || (exhausted ? 'forced_window_exhausted' : 'tech_lead_stop')) : 'none',
    state: snap, eligibility: eikosStopPermitted(leftS),
    carrier: null, envelope: null, relay_ok: null, decision_ok: null, receipt_ok: null,
    outcome: effective === 'stop' ? { next_round: 'not_observed' } : null,
  };
  EIKOS_SHADOW_LOG.push(rec);
  if (!rec.eligibility.permitted) { rec.carrier = { status: 'skipped', eikos_calls: 0 }; return rec; }
  const stateJson = JSON.stringify(snap);
  const env = await eikosCarrier(stateJson, roundNo);
  if (!env) {
    rec.carrier = { status: 'no_result' };
  } else if (env.__timeout) {
    rec.carrier = { status: 'timeout', wait_ms: EIKOS_CARRIER_TIMEOUT_MS,
                    note: 'wait ended; the carrier is not cancelled and may still finish; a late finish cannot change the first persisted decision' };
  } else if (env.__error) {
    rec.carrier = { status: 'error', error: env.__error };
  } else {
    // Did the script receive exactly the frozen state, and did its decision come back unaltered?
    rec.relay_ok = env.state_sha256 === eikosSha256(stateJson);
    const dec = {};
    EIKOS_DECISION_FIELDS.forEach((k) => { dec[k] = env[k] === undefined ? null : env[k]; });
    rec.decision_ok = env.decision_sha256 === eikosSha256(eikosCanonical(dec));
    // A decision whose attempt receipt was not written is a collection failure, even when valid.
    rec.receipt_ok = env.receipt === 'written';
    rec.envelope = env;
    rec.carrier = { status: 'returned' };
  }
  try { log(`  [eikos-shadow] r${roundNo}: baseline=${effective}, eikos=${rec.envelope ? rec.envelope.choice + '/' + rec.envelope.would_be_action + ' (' + rec.envelope.status + ')' : rec.carrier.status}` +
            `${rec.relay_ok === false ? ', STATE RELAY MISMATCH' : ''}${rec.decision_ok === false ? ', DECISION ALTERED IN RELAY' : ''}` +
            `${rec.receipt_ok === false ? ', RECEIPT NOT WRITTEN' : ''}`); } catch (e) {}
  return rec;
}
// <<EIKOS-SHADOW-END>>

// Expert-skills injection.
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills), whose \`scope\` is \`kernel\` (\`tuning\` entries ` +
    `belong to the e2e tuning phase and match every operator), and whose validation_status is ` +
    `\`validated\`, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.`;
}

// Appended to every role prompt only for multi-GPU jobs, so single-GPU prompts are unchanged. A
// UT-module multi-GPU task (UT_MR, known after Setup) gets the UT's own rules and its whole-UT limit
// as the lease run timeout; every other multi-GPU job, and Setup itself, gets the original text.
// The UT block does not repeat the original's "never background a lease": the roles start a long UT
// run in the background and poll it (one Bash call is killed after 10 min), so it says where the
// lease call goes in that recipe. A background command line that names a launcher (torchrun) matches
// gpu_lease.py's leaked-worker signature once its shell is gone, hence the script file for that case.
function multiGpuBlock(role) {
  if (!GPU_JOB) return '';
  const n = GPU_RESOURCE.gpusPerJob;
  if (!UT_MR) {
    return `\n\n## Multi-GPU job (GPUS_PER_JOB=${n})\n` +
      `Every GPU command of this op needs ${n} GPUs. \`GPU_ID\` is the group spec \`${GPU_JOB}\`: one ` +
      `\`gpu_lock.sh $GPU_ID <cmd>\` holds the whole group for the whole command and sets ` +
      `HIP_VISIBLE_DEVICES to the leased ids, so start every rank inside that one call (e.g. ` +
      `\`torchrun --nproc_per_node=${n} ...\`). Never call gpu_lock.sh from inside a leased command (it is ` +
      `rejected) and never background a lease (\`&\`, nohup). A leased command is killed after ` +
      `GEAK_GPU_RUN_TIMEOUT seconds (default 900); export a larger value for long runs such as a cold ` +
      `JIT build. Leases on the same GPUs serialize, so put the arms you compare into one run.`;
  }
  return `\n\n## Multi-GPU job (GPUS_PER_JOB=${n})\n` +
    `Every GPU command of this op needs ${n} GPUs. \`GPU_ID\` is the group spec \`${GPU_JOB}\`: one ` +
    `\`gpu_lock.sh $GPU_ID <cmd>\` holds the whole group for the whole command and sets ` +
    `HIP_VISIBLE_DEVICES to the leased ids, so start every rank inside that one call. This UT starts ` +
    `its own rank processes: run python3 unittest.py directly; do not wrap it in torchrun or another ` +
    `multi-process launcher. (Only a test whose own documented command is a launcher, e.g. ` +
    `\`torchrun --nproc_per_node=${n} ...\` in its config, keeps that command.) Never call gpu_lock.sh ` +
    `from inside a leased command (it is rejected) and never background anything inside one: the lease ends ` +
    `with its command. When a role step tells you to start a long UT run in the background and poll, put the ` +
    `whole \`gpu_lock.sh $GPU_ID <cmd>\` call inside that \`nohup bash -c '...'\` and wait for its \`UT_EXIT=\` ` +
    `line before the next GPU command and before you return. (For a launcher command, write the call into a ` +
    `script file and background that file: a background command line that names torchrun is reaped as a ` +
    `leaked worker.) A leased command is killed after GEAK_GPU_RUN_TIMEOUT seconds, and this job's limit is the ` +
    `UT's whole-run limit, ${UT_TOTAL_S} s: prefix every \`gpu_lock.sh\` and \`profile_kernel.sh\` ` +
    `call with \`GEAK_GPU_RUN_TIMEOUT=${UT_TOTAL_S}\` (the COMMANDMENT's GPU commands carry the same ` +
    `prefix). Leases on the same GPUs serialize, so put the arms you compare into one run.`;
}

// mode=mega reads the shared role file, then its mode=mega fragment, whose sections replace the
// same-named sections of the role file.
const UT_CONTRACT = A.region_ut ? `${String(A.region_ut).replace(/\/+$/, '')}.mega_ut.json` : '';
const megaRoleFiles = (file) => `${WORKFLOW_DIR}/roles/${file}, then ` +
  `${WORKFLOW_DIR}/roles/_fragments/${file.replace(/\.md$/, '')}_mega.md (mode=mega: each section of the ` +
  `second file replaces the section with the same heading in the first)` + (A.region_ut ? `; KERNEL_PATH is a ` +
  `region UT: then also ${WORKFLOW_DIR}/knowledge/mega_region_ut.md and its contract ${UT_CONTRACT}` : '');
function roleAgent(role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${MEGA ? megaRoleFiles(`${role}.md`) : `${WORKFLOW_DIR}/roles/${role}.md`} and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  if (MEGA) return base + megaExpertSkillsBlock(role, phase) + analysisSkillBlock(role, phase);
  return base + expertSkillsBlock(role) + multiGpuBlock(role);
}

// ===========================================================================
// mode=mega backend: a multi-rank operator searched as independent whole-kernel candidates. The lane's
// own phase and round calls run for every mode; this section holds only the mega inputs and the steps
// around them. Hooks that can end the run early return {exit: result}.
// ===========================================================================

const STR = { type: 'string' }, NUM = { type: 'number' }, BOOL = { type: 'boolean' };
const STR_N = { type: ['string', 'null'] }, NUM_N = { type: ['number', 'null'] };
const STRS = { type: 'array', items: STR }, ANYOBJ = { type: 'object', additionalProperties: true };
const ANYOBJS = { type: 'array', items: ANYOBJ };

// ---- mode=mega args (parsed for every mode; checked only under MEGA) ----
// A region UT (args.region_ut, a UT task dir in any layout): an adapter agent writes its unified contract
// (knowledge/mega_region_ut.md), whose mega args fill every arg the caller left unset.
if (MEGA && A.region_ut) {
  const u = (await agent(`You are the region UT adapter. Read ${WORKFLOW_DIR}/knowledge/mega_region_ut.md ` +
    `("Adapter"), then the UT at ${A.region_ut}; write its contract to ${UT_CONTRACT} and return that JSON.`,
    { phase: 'Setup', label: 'region_ut', schema: ANYOBJ })) || {};
  if (u.error || !u.benchmark_harness || !(u.target_guards || []).length) {
    throw new Error(`region_ut ${A.region_ut}: ${u.error || 'no contract'}`);
  }
  for (const [k, v] of Object.entries(u)) if (A[k] == null && k !== 'ut') A[k] = v;
}
// Modeled time (allocated phase slices, not wall time) so a replay follows the same budget path.
const WORKFLOW_STARTED_MS = 0;
let MEGA_CLOCK_MS = 0;
const MEGA_PREP_MODEL_MS = 90 * 1000;

// Positive control calibrates detection; without it 1.000x is unfalsifiable. Null calibrates noise.
const POSITIVE_CONTROL = (A.positive_control && typeof A.positive_control === 'object')
  ? A.positive_control : null;
const PC_ABORT = POSITIVE_CONTROL ? (A.positive_control.abort_on_fail !== false) : false;
const GPU_MIN_FREE_GIB = Number(A.gpu_min_free_gib || 0);
// working_kernel preserves one runnable candidate across rounds; speedup keeps the normal score gate.
const OBJECTIVE = (() => {
  const v = String(A.objective || 'speedup').trim();
  if (MEGA && v !== 'speedup' && v !== 'working_kernel') {
    throw new Error(`args.objective must be 'speedup' or 'working_kernel', got '${v}'`);
  }
  return v;
})();
const WORKING_KERNEL = OBJECTIVE === 'working_kernel';

const MEGA_DEFAULT_SPECIALTY = String(A.mega_default_specialty ||
  (GPU_RESOURCE.gpusPerJob > 1 ? 'distributed' : 'compute'));
const MEGA_PROFILE = String(A.mega_profile || 'production').trim();
if (MEGA && !['production', 'audit'].includes(MEGA_PROFILE)) {
  throw new Error(`args.mega_profile must be 'production' or 'audit', got '${MEGA_PROFILE}'`);
}
const MEGA_PRODUCTION = MEGA_PROFILE === 'production';
const MEGA_STOP_ON_DELIVERABLE = A.mega_stop_on_deliverable != null
  ? String(A.mega_stop_on_deliverable) === 'true' : MEGA_PRODUCTION;
const MEGA_MIN_SEARCH_ATTEMPTS = Math.max(0, Number(A.mega_min_search_attempts != null
  ? A.mega_min_search_attempts : 2));
const MEGA_CANDIDATE_TIMEOUT_S = Math.max(300, Number(A.mega_candidate_timeout_s ||
  3600));
const MEGA_STRUCTURAL_ONLY = String(A.mega_structural_only != null ? A.mega_structural_only : 'false') === 'true';
const MEGA_FINAL_TIMEOUT_S = Math.max(600, Number(A.mega_final_timeout_s ||
  (MEGA_PRODUCTION ? 7200 : 3600)));
const MEGA_TIME_BUDGET_S = Math.max(1800, Number(A.mega_time_budget_s ||
  28800));
const MEGA_FINAL_RESERVE_S = Math.max(600, Math.min(MEGA_TIME_BUDGET_S / 2,
  Number(A.mega_final_reserve_s || (MEGA_PRODUCTION ? 7200 : 3600))));
const MEGA_CLOSEOUT_RESERVE_S = Math.max(300, Number(A.mega_closeout_reserve_s || 900));
const EXPERT_SKILL_ID = String(A.expert_skill_id || '');
if (MEGA && EXPERT_SKILL_ID && !/^[a-z0-9][a-z0-9_-]*$/.test(EXPERT_SKILL_ID)) {
  throw new Error('expert_skill_id must be a safe slug');
}
const EXPERT_SKILL_REVISION = String(A.expert_skill_revision || '');
// A mega run that validates a named skill on-card gets that validation's gates unless args set them.
const SKILL_GATES = MEGA && !!EXPERT_SKILL_ID && !MEGA_STRUCTURAL_ONLY &&
  String(A.use_expert_skills != null ? A.use_expert_skills : 'true') === 'true' &&
  String(A.expert_skill_usage || 'candidate_validation') === 'candidate_validation';
const gateArg = (value) => String(value != null ? value : SKILL_GATES) === 'true';
const optionalNumber = (value) => {
  const number = Number(value);
  return value == null || value === '' || !Number.isFinite(number) ? null : number;
};
const EXPERT_SKILL_RECORDED_SCORE = optionalNumber(A.expert_skill_recorded_score);
const EXPERT_SKILL_RECORDED_LOW = optionalNumber(A.expert_skill_recorded_low);
const EXPERT_SKILL_RECORDED_HIGH = optionalNumber(A.expert_skill_recorded_high);
const MEGA_FINAL_TOP_K = Math.max(1, Number(A.mega_final_top_k ||
  (MEGA_PRODUCTION ? 1 : 2)));
const MEGA_FINAL_FALLBACK_K = Math.max(MEGA_FINAL_TOP_K,
  Number(A.mega_final_fallback_k || (MEGA_PRODUCTION ? 2 : MEGA_FINAL_TOP_K)));
const MEGA_TIE_NOISE_PCT = Math.max(0, Number(
  A.mega_tie_noise_pct != null ? A.mega_tie_noise_pct : 1.45));
// Graph-captured targets require an isolated capture/replay smoke.
const MEGA_REQUIRE_GRAPH_CAPTURE = A.require_graph_capture != null
  ? String(A.require_graph_capture) === 'true'
  : SKILL_GATES || !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
const DIRECT_GRAPH_ACCURACY = String(A.direct_graph_accuracy || SKILL_GATES) === 'true';
const argList = (value) => (Array.isArray(value) ? value : String(value || '').split(','))
  .map((s) => String(s).trim()).filter(Boolean);
// Guards that create credit are separate from guards that can only veto, so a win on a
// veto-only guard cannot be promoted as progress on the target.
const TARGET_GUARDS = argList(A.target_guards);
const REGRESSION_GUARDS = argList(A.regression_guards);
const BIMODAL_GUARDS = argList(A.bimodal_guards);
const BASELINE_ACTIVATION = A.baseline_activation &&
  typeof A.baseline_activation === 'object' ? A.baseline_activation : {};
// operator_e2e is the only metric comparable across a launch-structure change; per-stage timers are
// diagnostics.
const PROMOTION_METRIC = String(A.promotion_metric || 'legacy').trim();
if (MEGA && !['legacy', 'operator_e2e', 'changed_kernel'].includes(PROMOTION_METRIC)) {
  throw new Error(`args.promotion_metric must be 'legacy', 'operator_e2e' or 'changed_kernel', got ${PROMOTION_METRIC}`);
}
// Strict proof runs must state the metric; inheriting historical mixed behavior is not auditable.
const EXPERT_SKILL_DIR = EXPERT_SKILL_ID
  ? `${EXPERT_SKILLS_DIR}/skills/${EXPERT_SKILL_ID}` : '';
const PINNED_MEGA_SKILL = USE_EXPERT_SKILLS && !!EXPERT_SKILL_DIR;
const EXPERT_SKILL_PLANNER_GUIDE_FILE = `${EXPERT_SKILL_DIR}/planner_guide.md`;
const EXPERT_SKILL_AUTHOR_GUIDE_FILE = `${EXPERT_SKILL_DIR}/author_guide.md`;
const EXPERT_SKILL_BUNDLE_TOOL = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILLS_DIR}/_contribute/validate_skill.py`
  : String(A.expert_skill_bundle_tool || `${EXPERT_SKILLS_DIR}/_contribute/validate_skill.py`);
const EXPERT_SKILL_PLAYBOOK_FILE = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILL_DIR}/skill.md` : String(A.expert_skill_playbook || '');
const EXPERT_SKILL_PLANNER_EXTENSION_FILE =
  PINNED_MEGA_SKILL ? `${EXPERT_SKILL_DIR}/planner_extension.yaml`
    : String(A.expert_skill_planner_extension || '');
const EXPERT_SKILL_CONTRACT_FILE = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILL_DIR}/contract.yaml` : String(A.expert_skill_contract || '');
const EXPERT_SKILL_VALIDATION_FILE = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILL_DIR}/validation.yaml` : String(A.expert_skill_validation || '');
const EXPERT_SKILL_RUNTIME_FILE = PINNED_MEGA_SKILL
  ? `${EXPERT_SKILL_DIR}/runtime_validation.py` : String(A.expert_skill_runtime_file || '');
// expert_skill_id alone pins a mega skill: a missing status or digest is read once from the skill files
// here, before Setup, so the checks below fail as early as when the caller passes them.
const SKILL_PIN_KEYS = ['validation_status', 'bundle_sha256', 'planner_extension_sha256', 'contract_sha256'];
let SKILL_PIN = {};
if (MEGA && PINNED_MEGA_SKILL && SKILL_PIN_KEYS.some((k) => !A[`expert_skill_${k}`])) {
  phase('Setup')
  SKILL_PIN = (await agent(
    `Run exactly: python3 ${EXPERT_SKILL_BUNDLE_TOOL} ${EXPERT_SKILL_ID} --emit-bundle\n` +
    `It prints one JSON object. Return its ${SKILL_PIN_KEYS.join(', ')} values verbatim. If the ` +
    `command fails, return error with its last output lines. Do nothing else.`,
    { phase: 'Setup', label: 'skill:pin', schema: { type: 'object', additionalProperties: true,
      properties: Object.fromEntries([...SKILL_PIN_KEYS, 'error'].map((k) => [k, STR])),
      required: [] } })) || {};
  if (SKILL_PIN.error || !SKILL_PIN.bundle_sha256) {
    throw new Error(`expert skill ${EXPERT_SKILL_ID}: --emit-bundle failed: ${SKILL_PIN.error || 'no output'}`);
  }
  log(`expert skill ${EXPERT_SKILL_ID}: ${SKILL_PIN.validation_status}, bundle ${String(SKILL_PIN.bundle_sha256).slice(0, 8)}`);
}
const EXPERT_SKILL_BUNDLE_SHA256 = String(A.expert_skill_bundle_sha256 || SKILL_PIN.bundle_sha256 || '');
const EXPERT_SKILL_PLANNER_EXTENSION_SHA256 =
  String(A.expert_skill_planner_extension_sha256 || SKILL_PIN.planner_extension_sha256 || '');
const EXPERT_SKILL_CONTRACT_SHA256 =
  String(A.expert_skill_contract_sha256 || SKILL_PIN.contract_sha256 || '');
const EXPERT_SKILL_VALIDATION_STATUS = String(
  A.expert_skill_validation_status || SKILL_PIN.validation_status || ''
);
// A pinned mega skill runs on-card unless the run is structural-only.
const EXPERT_SKILL_USAGE = String(A.expert_skill_usage ||
  (MEGA && PINNED_MEGA_SKILL ? (MEGA_STRUCTURAL_ONLY ? 'authoring' : 'candidate_validation') : ''));
const REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY = USE_EXPERT_SKILLS &&
  (PINNED_MEGA_SKILL ||
   String(A.require_expert_skill_bundle_identity || 'false') === 'true');
const validSha256 = (value) => /^[a-f0-9]{64}$/.test(String(value || ''));
if (MEGA && (REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY &&
    (!EXPERT_SKILL_ID ||
     !EXPERT_SKILL_PLANNER_EXTENSION_FILE || !EXPERT_SKILL_CONTRACT_FILE ||
     !validSha256(EXPERT_SKILL_BUNDLE_SHA256) ||
     !validSha256(EXPERT_SKILL_PLANNER_EXTENSION_SHA256) ||
     !validSha256(EXPERT_SKILL_CONTRACT_SHA256)))) {
  throw new Error(
    'require_expert_skill_bundle_identity needs skill id/revision, Planner Extension/contract ' +
    'paths, and canonical bundle/extension/contract SHA-256 values'
  );
}
const EXPERT_SKILL_CONTRACT_TOOL = PINNED_MEGA_SKILL
  ? `${WORKFLOW_DIR}/tools/expert_skill_contract.py`
  : String(A.expert_skill_contract_tool || `${WORKFLOW_DIR}/tools/expert_skill_contract.py`);
const EXPERT_SKILL_REFERENCE_PATH = String(A.expert_skill_reference_path || '').replace(/\/+$/, '');
const CHECK_EXPERT_SKILL_CONTRACT = USE_EXPERT_SKILLS &&
  !!EXPERT_SKILL_ID && !!EXPERT_SKILL_CONTRACT_FILE &&
  String(A.verify_expert_skill_contract != null
    ? A.verify_expert_skill_contract : 'true') === 'true';
const REQUIRE_EXPERT_SKILL_CONTRACT = CHECK_EXPERT_SKILL_CONTRACT &&
  (PINNED_MEGA_SKILL || MEGA_STRUCTURAL_ONLY ||
    String(A.require_expert_skill_contract || 'false') === 'true');
// Skill source-shape checks are advisory diagnostics. Runtime admission is
// decided by the activated on-card behavior gates below.
const STRUCTURAL_CONTRACT_BLOCKING =
  String(A.block_on_expert_skill_contract || 'false') === 'true';
// Structure check + on-card Verify are reserved before the Author is sized (else Verify got <300s).
const MEGA_VERIFY_RESERVE_S = Math.max(300, Number(A.mega_verify_reserve_s || 1200));
const MEGA_POST_AUTHOR_RESERVE_S = MEGA_STRUCTURAL_ONLY ? 0
  : (CHECK_EXPERT_SKILL_CONTRACT ? 600 : 0) + MEGA_VERIFY_RESERVE_S;
const MEGA_STALL_LIMIT = Math.max(0, Number(A.mega_stall_limit != null ? A.mega_stall_limit : 3));
// Candidate lanes advanced per round. Their GPU work shares one lease pool and queues on it.
const MEGA_DIRECTIONS = Math.max(1, Math.floor(Number(A.mega_directions_per_round || 1)) || 1);
if (MEGA && MEGA_PRODUCTION &&
    MEGA_CANDIDATE_TIMEOUT_S - 150 - MEGA_POST_AUTHOR_RESERVE_S < 900) {
  throw new Error(`mega_candidate_timeout_s=${MEGA_CANDIDATE_TIMEOUT_S} cannot fit a 900s Author ` +
    `plus the ${MEGA_POST_AUTHOR_RESERVE_S}s structural+on-card Verify reserve`);
}
if (MEGA && MEGA_STRUCTURAL_ONLY && !CHECK_EXPERT_SKILL_CONTRACT) {
  throw new Error('mega_structural_only requires an enabled Expert Skill contract');
}
if (MEGA && USE_EXPERT_SKILLS && EXPERT_SKILL_VALIDATION_STATUS &&
    !['validated', 'experimental'].includes(EXPERT_SKILL_VALIDATION_STATUS)) {
  throw new Error('only validated or explicitly pinned experimental Expert Skills may run');
}
if (MEGA && USE_EXPERT_SKILLS && EXPERT_SKILL_ID && !EXPERT_SKILL_VALIDATION_STATUS) {
  throw new Error('an explicitly pinned Expert Skill requires validation status');
}
if (MEGA && USE_EXPERT_SKILLS && PINNED_MEGA_SKILL) {
  if (!['authoring', 'candidate_validation'].includes(EXPERT_SKILL_USAGE) ||
      !REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY || !EXPERT_SKILL_VALIDATION_FILE) {
    throw new Error(
      'pinned Mega Expert Skill requires exact bundle identity, validation file, and ' +
      'expert_skill_usage=authoring|candidate_validation'
    );
  }
  if (EXPERT_SKILL_USAGE === 'authoring' && !MEGA_STRUCTURAL_ONLY) {
    throw new Error('pinned authoring usage requires mega_structural_only=true');
  }
  if (EXPERT_SKILL_USAGE === 'candidate_validation' &&
      !CHECK_EXPERT_SKILL_CONTRACT) {
    throw new Error('pinned candidate_validation requires contract diagnostics');
  }
}
const GRAPH_CONTRACT_TOOL = PINNED_MEGA_SKILL
  ? `${WORKFLOW_DIR}/tools/expert_skill_runtime.py` : String(A.graph_contract_tool ||
    (!USE_EXPERT_SKILLS ? `${WORKFLOW_DIR}/tools/generic_graph_contract.py` : ''));
// Only the skill runtime injects arrival jitter into graph replays; generic_graph_contract.py does not,
// so a generic finalist cannot report jittered replays and is not asked to.
const ARRIVAL_JITTER_CONTRACT = PINNED_MEGA_SKILL;
const BENCH_HARNESS = String(A.benchmark_harness || '');
const CANDIDATE_IMPORT_MODULES = Object.freeze(argList(
  A.candidate_import_modules || [],
));
const EXPERT_SKILL_SOURCE_FILES = argList(A.expert_skill_source_files || []);
const EXPERT_SKILL_ACCURACY_CASES = Object.freeze(argList(A.expert_skill_accuracy_cases || []));
const REQUIRED_ACCURACY_CASES = Object.freeze(argList(A.required_accuracy_cases != null
  ? A.required_accuracy_cases : (USE_EXPERT_SKILLS ? EXPERT_SKILL_ACCURACY_CASES : [])));
const FORBIDDEN_CANDIDATE_TREE_DIGESTS = new Set(argList(
  A.forbidden_candidate_tree_digests || [],
).filter(validSha256));
const MEGA_EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);
const megaPlanner = (role, phase) => role === 'tech_lead' && (phase === 'analyze' || phase === 'plan_round');

const positiveIntArg = (name, value, fallback) => {
  const n = Number(value == null ? fallback : value);
  if (Number.isSafeInteger(n) && n > 0) return n;
  return fallback;
};
const REQUIRED_REPLAYS = positiveIntArg('required_replays', A.required_replays, 1000);
const REQUIRED_PAIRS = positiveIntArg('required_pairs', A.required_pairs, 5);
const REQUIRED_PAIRS_BY_GUARD = (A.required_pairs_by_guard &&
  typeof A.required_pairs_by_guard === 'object') ? A.required_pairs_by_guard : {};
const RF = Object(A.regression_floor_by_guard);
const RFI = { ...(A.regression_floor_by_guard ? { REGRESSION_FLOORS: RF } : {}),
  ...(A.mega_min_speedup ? { MIN_TARGET_SPEEDUP: +A.mega_min_speedup } : {}) };
const REQUIRE_OVERLAP = String(A.require_overlap != null ? A.require_overlap : 'false') === 'true';
const REQUIRE_ATTRIBUTION = gateArg(A.require_attribution);
const REQUIRE_ARTIFACT_DISTINCT = gateArg(A.require_artifact_distinct);
// Require an activated switched path to have actually launched on hardware.
const REQUIRE_HW_ACTIVATION = gateArg(A.require_hardware_activation);
const ACCURACY_METRIC = String(A.accuracy_metric || 'relL2');
const ACCURACY_THRESHOLD = Number(A.accuracy_threshold != null ? A.accuracy_threshold : 0.10);
const launchTargetArg = Number(A.launch_target != null ? A.launch_target : 2);
const LAUNCH_TARGET = Number.isSafeInteger(launchTargetArg) && launchTargetArg > 0
  ? launchTargetArg : 2;
const ALLOW_PARTIAL_FUSION =
  String(A.allow_partial_fusion != null ? A.allow_partial_fusion : !SKILL_GATES) === 'true';
if (MEGA && PINNED_MEGA_SKILL &&
    EXPERT_SKILL_USAGE === 'candidate_validation') {
  const missing = [
    [!MEGA_STRUCTURAL_ONLY, 'mega_structural_only=false'],
    [MEGA_REQUIRE_GRAPH_CAPTURE, 'require_graph_capture'],
    [DIRECT_GRAPH_ACCURACY, 'direct_graph_accuracy'],
    [REQUIRE_HW_ACTIVATION, 'require_hardware_activation'],
    [REQUIRE_ARTIFACT_DISTINCT, 'require_artifact_distinct'],
    [REQUIRE_ATTRIBUTION, 'require_attribution'],
    [REQUIRED_REPLAYS >= 256, 'required_replays>=256'],
    [REQUIRED_PAIRS >= 5, 'required_pairs>=5'],
    [LAUNCH_TARGET === 1 || LAUNCH_TARGET === 2, 'launch_target in {1,2}'],
    [!ALLOW_PARTIAL_FUSION, 'allow_partial_fusion=false'],
    [REQUIRED_ACCURACY_CASES.length > 0, 'required_accuracy_cases'],
  ].filter(([ok]) => !ok).map(([, label]) => label);
  if (missing.length) {
    throw new Error(`pinned candidate_validation missing gates: ${missing.join(', ')}`);
  }
}

// Optional profile-analysis skill (knowledge/analysis_skills/INDEX.md), run after Profile. Default none.
const ANALYSIS_SKILL = String(A.analysis_skill != null ? A.analysis_skill : 'none').trim();
const ANALYSIS_SKILL_ON = !!ANALYSIS_SKILL && ANALYSIS_SKILL !== 'none' && ANALYSIS_SKILL !== 'false';
if (MEGA && ANALYSIS_SKILL_ON && !/^[a-z0-9][a-z0-9_-]*$/.test(ANALYSIS_SKILL)) {
  throw new Error(`args.analysis_skill must be a safe skill slug, got ${JSON.stringify(ANALYSIS_SKILL)}`);
}
const ANALYSIS_SKILL_INPUTS = ANALYSIS_SKILL_ON ? {
  ANALYSIS_SKILL: ANALYSIS_SKILL,
  ANALYSIS_SKILL_DIR: `${WORKFLOW_DIR}/knowledge/analysis_skills/${ANALYSIS_SKILL}`,
} : {};
if (MEGA && ANALYSIS_SKILL_ON) log(`Profile-analysis skill: ${ANALYSIS_SKILL} (analysis only; Step-3 TechLead owns directions).`);
// Generic mega (Expert Skills off) only; skill-on and non-mega runs get {}.
const GMI = !USE_EXPERT_SKILLS ? {
  MEGA_MEASUREMENT_GUIDE: `${WORKFLOW_DIR}/knowledge/mega_measurement.md`,
  GENERIC_MEGA_NOTE: 'Read MEGA_MEASUREMENT_GUIDE\'s role table and MEGA_GENERIC_LESSONS before acting; they apply to your role here.',
  MEGA_GENERIC_LESSONS: `${WORKFLOW_DIR}/knowledge/mega_generic_lessons.md`,
  MEGA_AB_TOOL: `${WORKFLOW_DIR}/tools/mega_graph_ab.py`,
} : {};
// MoE regime guide + paired graph A/B for runs pinning a megamoe Skill (it names Skill designs, so a
// generic run never gets it by default); args.mega_regimes='true'|'false' overrides.
const MEGA_REGIMES_ON = A.mega_regimes != null
  ? String(A.mega_regimes) === 'true' : /^megamoe/.test(EXPERT_SKILL_ID);
const MRI = MEGA_REGIMES_ON ? {
  MEGA_REGIMES_GUIDE: `${WORKFLOW_DIR}/knowledge/mega_regimes.md`,
  MEGA_AB_TOOL: `${WORKFLOW_DIR}/tools/mega_graph_ab.py`,
  MEGA_REGIME_NOTE: 'Read MEGA_REGIMES_GUIDE for the target size\'s regime before choosing a direction; use MEGA_AB_TOOL for fast paired graph A/B while iterating.',
  ...(USE_EXPERT_SKILLS ? { MEGA_GENERIC_LESSONS: `${WORKFLOW_DIR}/knowledge/mega_generic_lessons.md` } : {}),
} : {};

const MEGA_RESUME_STATE = !!STATE_DIR &&
  String(A.mega_resume_state != null ? A.mega_resume_state : 'false') === 'true';
   // run-global cross-KERNEL technique blackboard (deep)

// Three-valued so that "could not read the pool" never collapses into "free".
const POOL_SCHEMA = obj({
  verdict: { type: 'string', enum: ['free', 'occupied', 'unknown'] },
  cards_total: NUM,
  cards_free: NUM,
  min_free_gib: NUM,
  foreign_pids: STRS,
  note: STR,
}, ['verdict']);

const MEGA_SETUP_SCHEMA = obj({
  eval_dir: STR, workspace: STR, baseline_dir: STR,
  kernel_name: STR, source_files: STRS, notes: STR,
  // Frozen baseline: timing and parity must use the real online kernel (baseline_src/ or
  // meta.baseline_callable), never kernel_src/. The run aborts if neither is confirmed.
  baseline_frozen: BOOL, baseline_callable: STR,
  mega_workspace_from_original: BOOL,
  mega_candidate_state_checked: BOOL,
  // Resume only: the director fills this when STATE_DIR holds a prior best, so a continued wave keeps
  // its cumulative speedup and history. Absent on a fresh run.
  resumed: BOOL,
  workflow_revision: STR,
  workflow_clean: BOOL,
  prior_state: obj({
    cumulative: NUM, insights: STRS,
    // Declared, because memoryMerge keys rows by `id` and dedupes on it, and the ladder is read back
  // out of `roadmap_rung`.
  ledger: { type: 'array', items: obj({
    id: STR, title: STR, specialty: STR,
    // confirmed | partial | unresolved | dead_end — see tech_lead.md "Verdict by SPEC, not by result".
    verdict: { type: 'string', enum: ['confirmed', 'partial', 'unresolved', 'dead_end'] },
    roadmap_rung: STR_N,
    expected: NUM, actual: NUM,
    lesson: STR,
    first_round: NUM, last_round: NUM,
  }, ['id', 'verdict']) },
    bottleneck_now: STR, best_per_case: perCase,
    // Shelf entries are keyed by id, aged by base_round and offered by files; an entry without files is
    // marked footprint=unknown and never offered.
    shelf: { type: 'array', items: obj({
      id: STR, title: STR, specialty: STR,
      geomean: NUM, patch: STR,
      files: STRS,
      // Fusion chain id (its terminal rung), carried across waves; empty for a standalone optimisation.
      chain_id: STR,
      footprint: { type: 'string', enum: ['known', 'unknown'] },
      base_round: NUM, shelved_round: NUM,
      absorbed: BOOL,
    }, ['id', 'geomean']) },
    candidate_registry: ANYOBJS,
    measurement_calibration: ANYOBJ,
    state_sequence: NUM,
    // round -> files CANONICAL took on that round. Keys are round numbers as strings.
    absorbed_files: { type: 'object' },
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name']);

const MEGA_CANDIDATE_SCHEMA = obj({
  candidate_id: STR,
  candidate_source: { type: 'string', enum: ['search', 'integrated'] },
  base_candidate_id: STR,
  candidate_status: {
    type: 'string',
    enum: ['authoring', 'runnable', 'scored', 'finalist', 'rejected'],
    description: 'A finalized GPU-free source checkpoint remains authoring until runtime verification.',
  },
  claim_complete: {
    type: 'boolean',
    description: 'True when this turn finalized its committed artifact, including a GPU-free static checkpoint; hardware gates are separate fields.',
  },
  checkpoint_complete: BOOL,
  structural_verified: BOOL,
  runtime_verified: BOOL,
  score_complete: BOOL,
  verification_status: STR,
  gpu_executed: BOOL,
  structural_report: STR,
  structural_skill_id: STR,
  structural_candidate_head: STR,
  structural_candidate_tree_digest: STR,
  structural_skill_bundle_sha256: STR,
  structural_planner_extension_sha256: STR,
  structural_contract_revision: STR,
  structural_contract_sha256: STR,
  contract_failures: { type: 'array', items: obj({
    id: STR, category: STR,
    severity: STR,
    messages: STRS,
  }, ['id', 'category', 'severity', 'messages']) },
  failed_required_checks: STRS,
  plan_consistency_errors: STRS,
  input_errors: STRS,
  provenance_attestation_valid: BOOL,
  attempt_id: STR,
  evidence_manifest: STR,
  target_guard: STR,
  correctness: STR,
  build: BOOL,
  tree: STR,
  head: STR,
  patch_file: STR,
  changed_files: STRS,
  topology: obj({}, []),
  absolute_score: NUM,
  per_case: perCase,
  activation_on_hardware: STR,
  path_marker: STR,
  launches: NUM,
  liveness_replays: NUM,
  graph_safe: STR,
  topology_sig: STR,
  provenance: STR,
  next_blocker: STR,
  notes: STR,
  activation: obj({
    mode: STR, switch_name: STR, switch_value: STR,
    path_marker: STR, marker_how: STR,
    switches: {
      type: 'array',
      items: obj({ switch_name: STR, switch_value: STR }, []),
    },
  }, []),
}, ['candidate_id', 'candidate_source', 'candidate_status', 'claim_complete',
  'contract_failures']);

const EXPERT_SKILL_CONTRACT_VERIFY_SCHEMA = obj({
  candidate_id: STR,
  candidate_head: STR,
  candidate_tree_digest: STR,
  skill_id: STR,
  skill_bundle_sha256: STR,
  planner_extension_sha256: STR,
  claim_complete: BOOL,
  report_path: STR,
  structural_compatible: BOOL,
  independent_structure_pass: BOOL,
  capability_eligible: BOOL,
  reference_copy_detected: BOOL,
  reference_copy_suspected: BOOL,
  provenance_status: STR,
  contract_revision: STR,
  contract_sha256: STR,
  contract_failures: { type: 'array', items: obj({
    id: STR, category: STR,
    severity: STR,
    messages: STRS,
  }, ['id', 'category', 'severity', 'messages']) },
  plan_consistent: BOOL,
  semantic_features_passed: NUM,
  semantic_features_total: NUM,
  hardware_verified: BOOL,
  accuracy_verified: BOOL,
  performance_verified: BOOL,
  next_blocker: STR,
  notes: STR,
}, [
  'candidate_id', 'candidate_head', 'claim_complete', 'structural_compatible',
  'candidate_tree_digest', 'skill_id', 'skill_bundle_sha256',
  'planner_extension_sha256',
  'independent_structure_pass', 'capability_eligible', 'hardware_verified',
  'accuracy_verified', 'performance_verified', 'plan_consistent',
  'contract_revision', 'contract_sha256', 'contract_failures',
  'failed_required_checks', 'plan_consistency_errors', 'input_errors',
  'provenance_attestation_valid',
]);

const MEGA_TOPOLOGY_SCHEMA = obj({
  launch_count: NUM,
  included_regions: STRS,
  included_queues: STRS,
  capabilities: STRS,
  require_overlap: BOOL,
  parameters: ANYOBJ,
  notes: STR,
});

const MEGA_ANALYZE_BASE_SCHEMA = obj({
  kernel_type: STR, kernel_file: STR, entry_point: STR,
  modifiable_files: STRS,
  bottleneck_guess: STR, roadmap_summary: STR,
  // The ladder: Analyze's ranked rungs and what each is conditional on.
  candidate_directions: { type: 'array', items: obj({
    id: STR,              // short, stable, quotable: D0, D1, ...
    title: STR,
    candidate_id: STR,
    candidate_source: { type: 'string', enum: ['search', 'integrated'] },
    base_candidate_id: STR,
    specialty: STR,
    focus_files: STRS,
    prompt: STR,
    rationale: STR,
    rung_deviation: STR,
    expected_speedup: NUM,
    gated_on: STRS,
    mandatory_arms: STRS,
    is_positive_control: BOOL,
    // How a rung closes: terminal on a measurement; enabling on functional acceptance, since a
    // prerequisite step has no standalone speedup. See stepRoleOf.
    step_role: { type: 'string', enum: ['enabling', 'terminal'] },
    enables: STR,
    cost_budget_pct: NUM,
    // A terminal rung must say what shape closes it.
    target_shape: {
      type: 'object',
      properties: {
        launches: NUM,
        stages_fused: STRS,
        require_overlap: BOOL,
      },
      required: ['launches'],
      additionalProperties: true,
    },
    target_topology: MEGA_TOPOLOGY_SCHEMA,
  }, ['id', 'title', 'gated_on']) },
  // perf_knowledge mapping (reference only): operator/language and the most relevant card paths.
  kk_operator: STR_N, kk_language: STR_N,
  kk_refs: STRS,
  baseline_operator_map: { type: 'array', items: obj({
    role: STR,
    file: STR,
    symbol: STR,
    kind: STR,
    semantics: STR,
    inputs: STRS,
    outputs: STRS,
    layout: STR,
    synchronization: STR,
    action: { type: 'string', enum: [
      'reuse', 'extract_shared', 'inline', 'rewrite_equivalent', 'rewire', 'extend',
    ] },
    target_role: STR,
    proof: STR,
  }, ['role', 'file', 'symbol', 'semantics', 'action']) },
  prior_art: ANYOBJS,
  // Fine-grained dependency graph; taskGraphGate owns semantic validation.
  task_graph: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      nodes: { type: 'array', items: obj({
        id: STR, stage: STR, duration_us: NUM,
        // measured | assumed.
        source: { type: 'string', enum: ['measured', 'assumed'] },
      }, ['id', 'source']) },
      edges: { type: 'array', items: obj({
        from: STR, to: STR,
        // data | launch_boundary | barrier | assumed.
        enforced_by: STR,
        scope: STR,
        fan_in: NUM,
        // Consumer operands the producer does not supply (weights, scales, descriptors).
        producer_independent_operands: STRS,
      }, ['from', 'to', 'enforced_by']) },
      unknowns: ANYOBJS,
      critical_path_us: NUM,
      measured_e2e_us: NUM,
    },
  },
  // Per-pipe resource timeline, required alongside task_graph: the graph says what MAY overlap, this
  // says which unit is idle and what could fill it.
  resource_timeline: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      pipes: { type: 'array', items: obj({
        stage: STR,
        pipe: STR,   // valu|mfma|vmem|lds|hbm|interconnect|scalar
        utilization_pct: NUM_N,
        source: STR,
      }, ['stage', 'pipe', 'utilization_pct']) },
      interkernel_gap_us: obj({
        median: NUM, max: NUM, n_boundaries: NUM,
      }, ['median']),
      // The gate derives the class from the numbers and reports a mismatch with this one.
      class: { type: 'string', enum: ['throughput_bound', 'latency_bound', 'launch_bound', 'mixed'] },
      // id, the size fields and rides_on are read by the gate, so they are declared; rides_on records that
      // another rung absorbs this hole.
      idle_pipe_opportunities: { type: 'array', items: obj({
        id: STR, window: STR, idle_pipe: STR,
        recoverable_us: NUM_N, pct_of_e2e: NUM_N,
        rides_on: STR,
      }, []) },
      closed_axes: ANYOBJS,
      unknowns: ANYOBJS,
    },
  },
  // Lowerable whole-kernel schedule contract.
  mega_plan_ir: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      plan_version: STR,
      expert_skill_id: STR_N,
      expert_skill_revision: STR_N,
      expert_skill_bundle_sha256: STR_N,
      expert_skill_planner_extension_sha256: STR_N,
      expert_skill_validation_status: STR_N,
      expert_skill_auto_apply: { type: ['boolean', 'null'] },
      expert_skill_explicit_pin_modes: {
        type: 'array', items: STR,
      },
      target: obj({
        launch_count: NUM,
        required_regions: STRS,
        required_queues: STRS,
        required_capabilities: STRS,
      }, ['launch_count', 'required_regions', 'required_queues']),
      work_domains: { type: 'array', items: obj({
        id: STR, unit: STR,
        extent: STR, index_type: STR,
      }, ['id', 'unit', 'extent', 'index_type']) },
      regions: { type: 'array', items: obj({
        id: STR, role: STR,
        work_domain: STR, engine: STR,
      }, ['id', 'role', 'work_domain']) },
      buffers: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: {
          id: STR, role: STR,
          element_type: STR, capacity: STR,
          address_space: STR, format: STR,
          producers: STRS,
          consumers: STRS,
          lifetime: STR, alias_group: STR,
        },
        required: ['id', 'role', 'element_type', 'capacity', 'address_space'],
      } },
      counters: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: {
          id: STR, element_type: STR,
          capacity: STR, scope: STR,
          lifecycle: STR, reset_owner: STR,
          generation: STR,
          producers: STRS,
          consumers: STRS,
          publish: STR, wait: STR,
        },
        required: [
          'id', 'element_type', 'capacity', 'scope', 'lifecycle',
          'producers', 'consumers',
        ],
      } },
      queues: { type: 'array', items: obj({
        id: STR, work_domain: STR,
        claim_unit: STR, owner: STR,
        shards: NUM, stride_bytes: NUM,
        chunk_policy: STR, ready_when: STR,
        priority: NUM,
      }, ['id', 'work_domain', 'claim_unit', 'owner']) },
      events: { type: 'array', items: obj({
        id: STR, producer: STR,
        consumer: STR, scope: STR,
        counter: STR, publish: STR,
        wait: STR,
      }, ['id', 'producer', 'consumer', 'scope']) },
      abi: {
        type: 'object',
        additionalProperties: true,
        properties: {
          entry_point: STR,
          arguments: { type: 'array', items: {
            type: 'object', additionalProperties: true,
            properties: {
              id: STR, type: STR,
              role: STR, source: STR,
              consumers: STRS,
              optional: BOOL,
            },
            required: ['id', 'type', 'role'],
          } },
          parameters: ANYOBJ,
        },
        required: ['entry_point', 'arguments'],
      },
      resources: {
        type: 'object',
        additionalProperties: true,
        properties: {
          arch: STR,
          workgroup: obj({
            wave_size: NUM, wave_count: NUM,
            thread_count: NUM,
          }, ['wave_size', 'wave_count', 'thread_count']),
          local_memory: obj({
            total_bytes: NUM,
            limit_bytes: NUM,
            allocation_rule: STR,
            allocations: { type: 'array', items: {
              type: 'object', additionalProperties: true,
            } },
          }, ['total_bytes', 'limit_bytes', 'allocation_rule', 'allocations']),
          registers: obj({
            vgpr_max: NUM_N,
            sgpr_max: NUM_N,
            scratch_bytes_max: NUM_N,
            source: STR,
          }, ['vgpr_max', 'sgpr_max', 'scratch_bytes_max', 'source']),
          occupancy: obj({
            compute_units: NUM,
            grid_blocks: NUM,
            min_workgroups_per_cu: NUM,
            requires_full_grid_residency: BOOL,
          }, [
            'compute_units', 'grid_blocks', 'min_workgroups_per_cu',
            'requires_full_grid_residency',
          ]),
          engines: { type: 'array', items: obj({
            region: STR, pipe: STR,
          }, ['region', 'pipe']) },
          parameters: ANYOBJ,
        },
        required: [
          'arch', 'workgroup', 'local_memory', 'registers', 'occupancy', 'engines',
        ],
      },
      schedule: {
        type: 'object',
        additionalProperties: true,
        properties: {
          primary_loop: obj({
            kind: STR,
            carried_state: STRS,
            queue_priority: STRS,
            queue_selection: STR,
            progress_invariants: STRS,
          }, ['kind', 'carried_state', 'progress_invariants']),
          policies: ANYOBJ,
          parameters: ANYOBJ,
        },
        required: ['primary_loop', 'policies'],
      },
      compiler_constraints: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: { id: STR, rule: STR },
        required: ['id', 'rule'],
      } },
      evidence_requirements: ANYOBJ,
      known_unknowns: ANYOBJS,
    },
    required: [
      'plan_version', 'target', 'work_domains', 'regions', 'buffers', 'counters',
      'queues', 'events', 'abi', 'resources', 'schedule',
      'compiler_constraints', 'evidence_requirements', 'known_unknowns',
    ],
  },
}, ['kernel_type', 'roadmap_summary']);
const MEGA_ANALYZE_SCHEMA = {
  ...MEGA_ANALYZE_BASE_SCHEMA,
  properties: {
    ...MEGA_ANALYZE_BASE_SCHEMA.properties,
    candidate_directions: {
      ...MEGA_ANALYZE_BASE_SCHEMA.properties.candidate_directions,
      items: {
        ...MEGA_ANALYZE_BASE_SCHEMA.properties.candidate_directions.items,
        required: [...new Set([
          ...(MEGA_ANALYZE_BASE_SCHEMA.properties.candidate_directions.items.required || []),
          'candidate_id', 'candidate_source', 'base_candidate_id', 'specialty',
          'focus_files', 'prompt', 'target_shape', 'target_topology',
        ])],
      },
    },
  },
  required: [...new Set([
    ...(MEGA_ANALYZE_BASE_SCHEMA.required || []),
    'modifiable_files', 'candidate_directions', 'prior_art',
    'baseline_operator_map', 'task_graph', 'resource_timeline', 'mega_plan_ir',
  ])],
};

const MEGA_BENCH_SCHEMA = obj({
  commandment_path: STR, correctness_cmd: STR,
  benchmark_cmd: STR, profile_cmd: STR, parse_hint: STR,
  baseline_per_case: ANYOBJS,
  baseline_geomean_ms: NUM, num_test_cases: NUM,
  // Workload-aligned outputs (present with a workload spec): metric = Σweight / Σ(weight/speedup);
  // baseline_weighted_total_ms is the baseline time the weights represent.
  workload_aligned: BOOL,
  baseline_weighted_total_ms: NUM,
  weights_provenance: STR, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  positive_control: obj({
    ran: BOOL,
    claim_complete: BOOL,
    attempt_id: STR,
    evidence_manifest: STR,
    switch_present: { type: ['boolean', 'null'] },
    switch_checked: STR,
    measured_pct: NUM_N,
    expected_lo: NUM, expected_hi: NUM,
    passed: BOOL, reps: NUM,
    control_pairs_pct: { type: 'array', items: NUM },
    null_pairs_pct: { type: 'array', items: NUM },
    null_arm_pct: NUM,
    note: STR,
    // Required only when the object exists at all: a run with no control omits it entirely.
  }, ['ran', 'claim_complete', 'switch_present', 'measured_pct']),
  reliable: BOOL, notes: STR,
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

const ANALYSIS_RESULT_SCHEMA = obj({
  status: { type: 'string', enum: ['ready', 'degraded'] },
  analysis_skill: STR,
  analysis_schema_version: STR,
  analysis_status: {
    type: 'string',
    enum: ['awaiting_measurement', 'evidence_complete', 'unavailable'],
  },
  analysis_json: STR,
  failure_note: STR,
}, [
  'status',
  'analysis_skill',
  'analysis_schema_version',
  'analysis_status',
  'analysis_json',
  'failure_note',
]);

const MEGA_PLAN_BASE_SCHEMA = obj({
  stop: BOOL, reasoning: STR,
  directions: {
    type: 'array',
    items: obj({
      id: STR, title: STR,
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'distributed', 'deep_explore'] },
      focus_files: STRS,
      expected_speedup: NUM, prompt: STR,
      kk_refs: STRS, // optional: perf_knowledge card paths for THIS direction (REFERENCE ONLY)
      // Binds the direction to Analyze's pipe table (knowledge/pipe_occupancy.md).
      fills_pipe: STR,
      pipe_util_pct: NUM,
      headroom_basis: STR,
      graph_refs: STRS, // task_graph node/edge ids this direction acts on
      // The idle_pipe_opportunities id this direction fills, so each hole can be checked as owned or not.
      fills_hole: STR,
      // Binds the direction to the ladder: roadmap_rung is the rung id or `off_ladder`.
      roadmap_rung: STR,
      rung_deviation: STR,
      // Step kind: terminal (default) is judged on speed; enabling is a prerequisite in a multi-step
      // fusion, judged on function (builds, path taken, correct, no deadlock).
      step_role: { type: 'string', enum: ['enabling', 'terminal'] },
      enables: STR,
      cost_budget_pct: NUM,
      gated_on: STRS,
      mandatory_arms: STRS,
      target_shape: {
        type: 'object',
        properties: {
          launches: NUM,
          stages_fused: STRS,
          require_overlap: BOOL,
        },
        required: ['launches'],
        additionalProperties: true,
      },
      // Operator-neutral multi-lever descriptor for a whole-kernel candidate.
      target_topology: MEGA_TOPOLOGY_SCHEMA,
      candidate_id: STR,
      candidate_source: STR,
      base_candidate_id: STR,
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

const MEGA_PLAN_SCHEMA = {
  ...MEGA_PLAN_BASE_SCHEMA,
  properties: {
    ...MEGA_PLAN_BASE_SCHEMA.properties,
    directions: {
      ...MEGA_PLAN_BASE_SCHEMA.properties.directions,
      items: {
        ...MEGA_PLAN_BASE_SCHEMA.properties.directions.items,
        properties: {
          ...MEGA_PLAN_BASE_SCHEMA.properties.directions.items.properties,
          candidate_source: {
            type: 'string', enum: ['search', 'integrated'],
          },
        },
        required: [...new Set([
          ...(MEGA_PLAN_BASE_SCHEMA.properties.directions.items.required || []),
          'candidate_id', 'candidate_source', 'base_candidate_id',
          'focus_files', 'roadmap_rung', 'step_role', 'gated_on',
          'target_shape', 'target_topology',
        ])],
      },
    },
  },
  required: [...new Set([...(MEGA_PLAN_BASE_SCHEMA.required || []), 'reasoning'])],
};

const MEGA_VERIFY_SCHEMA = obj({
  status: STR, correctness: STR,
  claim_complete: BOOL, attempt_id: STR,
  timeout_recovery_safe: BOOL,
  active_gpu_processes: NUM,
  lane_lock_free: BOOL,
  evidence_complete: BOOL,
  evidence_manifest: STR, candidate_id: STR,
  candidate_source: STR, candidate_tree: STR,
  candidate_head: STR,
  activation: obj({
    mode: STR, switch_name: STR, switch_value: STR,
    path_marker: STR, marker_how: STR,
    switches: {
      type: 'array',
      items: obj({ switch_name: STR, switch_value: STR }, []),
    },
  }, []),
  verified_geomean: NUM_N, verified_arithmetic: NUM_N,
  verified_weighted: NUM_N, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: STR, notes: STR,
  graph_safe: STR,
  liveness: STR, // pass|fail|n/a — deadlock/stale-read stress (distributed specialty)
  replay_count: NUM,
  replay_results: { type: 'array', items: obj({
    guard: STR, count: NUM,
    status: STR, graph_safe: STR,
  }, ['guard', 'count', 'status']) },
  arms_run: STRS,
  // Accuracy as a number (metric, value, threshold); missing fields fall back to `correctness`.
  accuracy: obj({
    metric: STR, value: { type: 'number', minimum: 0 },
    threshold: { type: 'number', exclusiveMinimum: 0 },
    guard: STR, method: STR,
  }, []),
  accuracy_results: { type: 'array', items: obj({
    metric: STR, value: { type: 'number', minimum: 0 },
    threshold: { type: 'number', exclusiveMinimum: 0 },
    guard: STR, method: STR,
  }, ['metric', 'value', 'threshold', 'guard']) },
  // Paired launches per call; how_counted is the evidence. Missing fields leave the launch criterion unjudged.
  launch_shape: obj({
    launches_base: NUM, launches_cand: NUM,
    per_rank: BOOL, target: NUM,
    stages_fused: STRS, how_counted: STR,
  }, []),
  // reps = interleaved A/B pairs; null_arm_pct = the delta of an arm doing identical work (the noise
  // floor).
  reps: NUM, null_arm_pct: NUM,
  // Raw per-pair readings tagged by guard/route (bimodal split, target-guard check).
  paired_readings: { type: 'array', items: obj({
    guard: STR, base: NUM, cand: NUM,
  }, ['guard', 'base', 'cand']) },
  // Did the patched path execute in the measured run?
  activation_confirmed: STR,  // yes|no|unknown
  activation_evidence: STR,   // the command + the marker output that proves it
  // Did the switched path run on the device this round?
  activation_on_hardware: STR, // yes|no|n/a|unknown
  hardware_evidence: STR,       // the gpu_lock cmd + the device-side observable it produced
  // A marker proves the host path ran, not that the arms compiled differently: a JIT disk cache can
  // serve one binary to both arms if the switch is not in the cache key.
  artifact_distinct: STR,     // yes|no|n/a|unknown
  artifact_hash_base: STR,
  artifact_hash_candidate: STR,
  touched_files: STRS,
  // Overlap (see knowledge/overlap_instrument.md).
  overlap: obj({
    measured: { type: 'string', enum: ['yes', 'no', 'unknown'] },
    fraction: NUM,        // wall-clock, >=2 distinct roles active
    cu_fraction: NUM,     // CU-weighted; wall-clock alone is manufacturable
    method: STR,
    // The meter's controls: base_reading on the unfused path must read ~0 (else the meter is broken);
    // forced_reading must read high (else the meter is dead).
    base_reading: NUM,
    forced_reading: NUM,
    clock_skew_ns: NUM,   // s_memrealtime coherence across XCDs — assumed at your peril
    meter_overhead_pct: NUM,
    note: STR,
  }, ['measured']),
  // Attribution (knowledge/fusion_preconditions.md) for a patch that changes the launch count.
  attribution: obj({
    changed_us: NUM,
    replaced_sum_us: NUM,
    guard: STR,
    // e2e minus the sum of all kernel times, both arms. Where a gap-win hides.
    residual_ms_base: NUM,
    residual_ms_cand: NUM,
    method: STR,
    absolute_to_frozen: BOOL,
    note: STR,
  }, []),
}, ['status', 'verified_geomean', 'touched_files']);

const MEGA_MEMORY_SCHEMA = obj({
  insights: STRS,
  candidate_registry: ANYOBJS,
  measurement_calibration: ANYOBJ,
  state_written: BOOL, state_round: NUM,
  state_path: STR, state_generation: STR,
  state_sequence: NUM,
  // memoryMerge keys and dedupes rows by id and reads the ladder from roadmap_rung; a row without
  // verdict is treated as untried.
  ledger: { type: 'array', items: obj({
    id: STR, title: STR, specialty: STR,
    // confirmed | partial | unresolved | dead_end — see tech_lead.md "Verdict by SPEC, not by result".
    verdict: { type: 'string', enum: ['confirmed', 'partial', 'unresolved', 'dead_end'] },
    roadmap_rung: STR_N,
    expected: NUM, actual: NUM,
    lesson: STR,
    first_round: NUM, last_round: NUM,
  }, ['id', 'verdict']) },
  // Durable hardware/toolchain facts (lowerings true for every kernel on this box), as opposed to
  // insights about this operator.
  knowledge_delta: { type: 'array', items: obj({
    fact: STR,
    // Which card it belongs in, so the merge is a placement decision and not a search.
    card: STR,
    // The artifact that establishes it: an ISA dump path, a counter, a sha256, a measured number.
    evidence: STR,
    // What a reader would have believed without this fact; a lowering earns a card only when the
    // source-level reading says otherwise.
    contradicts: STR,
    // true = holds for any kernel on this target; false = this operator's shape.
    generalizes: BOOL,
  }, ['fact', 'evidence', 'generalizes']) },
  bottleneck_now: STR, suggest_next: STR,
}, ['insights']);

const MEGA_SELECTION_SCHEMA = obj({
  claim_complete: BOOL,
  attempt_id: STR,
  evidence_manifest: STR,
  selected_candidate_id: STR,
  selected_source: STR,
  selected_tree: STR,
  selected_head: STR,
  materialized_from_head: STR,
  selected_score: NUM,
  selected_patch: STR,
  candidates: { type: 'array', items: obj({
    candidate_id: STR, source: STR,
    tree: STR, head: STR, score: NUM,
    status: STR,
    touched_files: STRS,
    activation: obj({
      mode: STR, switch_name: STR, switch_value: STR,
      path_marker: STR, marker_how: STR,
      switches: {
        type: 'array',
        items: obj({ switch_name: STR, switch_value: STR }, []),
      },
    }, []),
    correctness: STR, guards_pass: BOOL,
    path_marker: { type: 'string', description: 'The path marker line the candidate printed.' },
    path_marker_count: { type: 'number', description: 'Ranks that printed path_marker.' },
    world_size: NUM,
    launches: NUM, launches_base: NUM,
    graph_safe: { type: 'string', enum: ['pass', 'fail'] },
    artifact_distinct: { type: 'string', enum: ['yes', 'no'] },
    artifact_hash_base: STR, artifact_hash_candidate: STR,
    overlap_measured: STR, overlap_fraction: NUM,
    overlap_cu_fraction: NUM,
    overlap_base_reading: NUM, overlap_forced_reading: NUM,
    attribution_complete: BOOL,
    attribution: obj({
      changed_us: NUM, replaced_sum_us: NUM,
      residual_ms_base: NUM, residual_ms_cand: NUM,
      guard: STR, method: STR,
      absolute_to_frozen: BOOL,
    }, []),
    accuracy_metric: STR, accuracy_value: NUM,
    accuracy_results: { type: 'array', items: obj({
      metric: STR, value: NUM,
      threshold: NUM, guard: STR,
      method: STR,
    }, ['metric', 'value', 'guard']) },
    liveness_replays: NUM, claim_complete: BOOL,
    replay_results: { type: 'array',
      description: 'One row per guard in TARGET_GUARDS and REGRESSION_GUARDS, named exactly as given.',
      items: obj({
      guard: STR, count: NUM,
      status: { type: 'string', enum: ['pass', 'fail'] },
      graph_safe: { type: 'string', enum: ['pass', 'fail'] },
      arrival_jitter: { type: 'boolean',
        description: 'True only when the graph contract injected arrival jitter (Expert Skill runtime).' },
    }, ['guard', 'count', 'status', 'graph_safe', 'arrival_jitter']) },
    paired_readings: { type: 'array', items: obj({
      guard: STR, base: NUM, cand: NUM,
    }, ['guard', 'base', 'cand']) },
    null_arm_pct: NUM,
    per_case: perCase,
    reason: STR,
  }, ['candidate_id', 'claim_complete', 'path_marker', 'path_marker_count', 'graph_safe',
    'artifact_distinct', 'replay_results']) },
  notes: STR,
}, ['claim_complete', 'attempt_id', 'evidence_manifest', 'selected_candidate_id', 'selected_tree']);

const MEGA_IDENTITY_SCHEMA = obj({
  passed: BOOL,
  selected_candidate_id: STR,
  source_head: STR,
  materialized_head: STR,
  note: STR,
}, ['passed', 'selected_candidate_id', 'source_head', 'materialized_head']);

const MEGA_VALIDATE_SCHEMA = obj({
  kernel_name: STR,
  director_verified_speedup_geomean: NUM,
  director_verified_speedup_arithmetic: NUM,
  director_verified_speedup_weighted: NUM, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: NUM,
  validation_status: STR, correctness: STR,
  autonomy_acceptance_confirmed: BOOL,
  workflow_unchanged: BOOL,
  workflow_revision_end: STR,
  materialization_matches: BOOL,
  selected_source_head: STR,
  selected_materialized_head: STR,
  final_patch_verified: BOOL,
  attribution: obj({
    changed_us: NUM, replaced_sum_us: NUM,
    guard: STR, residual_ms_base: NUM,
    residual_ms_cand: NUM, method: STR,
    absolute_to_frozen: BOOL,
  }, []),
  per_case: perCase, applied_to_original: STR,
  // Count of GPU-lease jobs still alive for this EVAL_DIR when validation finished.
  orphan_leases_swept: NUM,
  arbitration_note: STR, final_patch: STR,
}, ['director_verified_speedup_geomean', 'validation_status', 'correctness']);

// ---- mode=mega helpers and run state ----
function megaNowMs() { return WORKFLOW_STARTED_MS + MEGA_CLOCK_MS; }
function megaAdvanceMs(ms) { MEGA_CLOCK_MS += Math.max(0, Number(ms) || 0); }
function megaChargeMs(allocatedMs) { megaAdvanceMs(allocatedMs); }

// <<REPLAY:structural_evidence_identity>>
function structuralEvidenceIdentityVerdict(report, expected) {
  const r = report || {};
  const e = expected || {};
  const reasons = [];
  const requireEqual = (field, wanted, label = field) => {
    if (String(r[field] || '') !== String(wanted || '')) {
      reasons.push(`${label} mismatch: got ${String(r[field] || '(missing)')}, ` +
        `expected ${String(wanted || '(missing)')}`);
    }
  };
  requireEqual('candidate_id', e.candidateId, 'candidate id');
  requireEqual('candidate_head', e.candidateHead, 'candidate HEAD');
  requireEqual('skill_id', e.skillId, 'Skill id');
  requireEqual('contract_revision', e.skillRevision, 'Skill revision');
  if (e.contractSha256) {
    requireEqual('contract_sha256', e.contractSha256, 'contract digest');
  }
  if (e.plannerExtensionSha256) {
    requireEqual(
      'planner_extension_sha256',
      e.plannerExtensionSha256,
      'Planner Extension digest'
    );
  }
  if (e.skillBundleSha256) {
    requireEqual('skill_bundle_sha256', e.skillBundleSha256, 'Skill bundle digest');
  }
  if (!String(r.candidate_tree_digest || '')) {
    reasons.push('candidate tree digest missing');
  }
  return { pass: reasons.length === 0, reasons };
}
// <</REPLAY:structural_evidence_identity>>

// <<REPLAY:mega_topology_contract>>
function topologyLaunchCount(topo) {
  if (!topo || typeof topo !== 'object') return NaN;
  const value = topo.launch_count != null ? topo.launch_count : topo.launches;
  return Number(value);
}

function candidateClaimsPlanTarget(direction, planIr) {
  const planned = Number(planIr && planIr.target && planIr.target.launch_count);
  const declared = topologyLaunchCount(direction && direction.target_topology);
  // launch_count is an upper bound.
  return Number.isFinite(planned) && Number.isFinite(declared) && declared >= 1 &&
    declared <= planned;
}

function megaEffectiveLaneCheckpoint(candidate) {
  const c = candidate || {};
  const w = c.working_snapshot && typeof c.working_snapshot === 'object'
    ? c.working_snapshot : {};
  return String(c.working_head || '') && String(w.head || '') === String(c.working_head)
    ? w : c;
}

function authoringContractNeedsPreflight(
  direction, planIr, candidate, contractEnabled, expected = {}
) {
  if (!contractEnabled || !candidateClaimsPlanTarget(direction, planIr)) return false;
  const c = megaEffectiveLaneCheckpoint(candidate);
  const identityMatches =
    (!expected.skillId || c.structural_skill_id === expected.skillId) &&
    (!expected.revision || c.structural_contract_revision === expected.revision) &&
    (!expected.bundle || c.structural_skill_bundle_sha256 === expected.bundle) &&
    (!expected.planner ||
      c.structural_planner_extension_sha256 === expected.planner) &&
    (!expected.contract || c.structural_contract_sha256 === expected.contract);
  return !(c.structural_verified === true &&
    String(c.structural_candidate_head || '') &&
    String(c.structural_candidate_head) === String(c.head || '') &&
    identityMatches);
}

// Convert the operator-neutral descriptor into Verify's historical TARGET_SHAPE vocabulary.
function megaShapeFromTopology(topo) {
  if (!topo || typeof topo !== 'object') {
    return { launches: LAUNCH_TARGET, stages_fused: [], require_overlap: false };
  }
  return {
    launches: Number.isFinite(topologyLaunchCount(topo))
      ? topologyLaunchCount(topo) : LAUNCH_TARGET,
    stages_fused: Array.isArray(topo.included_regions)
      ? topo.included_regions.map(String)
      : (Array.isArray(topo.fused_stages) ? topo.fused_stages.map(String) : []),
    queues: Array.isArray(topo.included_queues) ? topo.included_queues.map(String) : [],
    capabilities: Array.isArray(topo.capabilities) ? topo.capabilities.map(String) : [],
    require_overlap: topo.require_overlap === true,
    parameters: topo.parameters && typeof topo.parameters === 'object'
      ? { ...topo.parameters } : {},
  };
}

function megaTopologyVerdict(direction, planIr, allowPartialFusion = true) {
  const topo = direction && direction.target_topology;
  if (!topo || typeof topo !== 'object') {
    return { pass: false, reason: 'target_topology is missing' };
  }
  if (direction.topology_inferred === true) {
    return { pass: false, reason:
      'planner omitted target_topology; orchestrator inferred a compatibility fallback' };
  }
  const launches = topologyLaunchCount(topo);
  const regions = Array.isArray(topo.included_regions)
    ? topo.included_regions.filter(Boolean)
    : (Array.isArray(topo.fused_stages) ? topo.fused_stages.filter(Boolean) : []);
  if (!Number.isFinite(launches) || launches < 1 || !regions.length) {
    return { pass: false, reason:
      `target_topology must declare finite launch_count and non-empty included_regions; got ` +
      `${JSON.stringify(topo)}` };
  }
  const terminal = String(direction.step_role || 'terminal') === 'terminal';
  // A direction naming its bucket is held to that bucket's target (one per regime), else to `target`.
  const bt = planIr && planIr.target_by_bucket;
  const target = bt && bt[String(direction.bucket)] || planIr && planIr.target || {};
  const planned = Number(target.launch_count); // an upper bound
  if (terminal && Number.isFinite(planned) && launches > planned &&
      !allowPartialFusion) {
    return { pass: false, reason:
      `partial fusion is disabled: terminal declares ${launches} launches, target is <= ${planned}` };
  }
  if (terminal && Number.isFinite(planned) && launches > planned &&
      !String(direction.rung_deviation || '').trim()) {
    return { pass: false, reason:
      `terminal topology declares ${launches} launches but mega_plan_ir target is ${planned}; ` +
      'declare rung_deviation for an intentional partial-fusion terminal' };
  }
  const requireAll = (declared, required, label) => {
    const have = new Set(Array.isArray(declared) ? declared.map(String) : []);
    const missing = (Array.isArray(required) ? required : [])
      .map(String).filter((value) => !have.has(value));
    return missing.length ? `${label} missing ${missing.join(',')}` : '';
  };
  if (terminal && launches <= planned) {
    const regionError = requireAll(regions, target.required_regions, 'included_regions');
    if (regionError) return { pass: false, reason: regionError };
    const queueError = requireAll(
      topo.included_queues, target.required_queues, 'included_queues');
    if (queueError) return { pass: false, reason: queueError };
    const capabilityError = requireAll(
      topo.capabilities, target.required_capabilities, 'capabilities');
    if (capabilityError) return { pass: false, reason: capabilityError };
  }
  return { pass: true, reason: '' };
}
// <</REPLAY:mega_topology_contract>>

// Add Skill context only to roles that consume it.
function megaExpertSkillsBlock(role, phase) {
  if (!USE_EXPERT_SKILLS || !MEGA_EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  if (EXPERT_SKILL_DIR) {
    if (!megaPlanner(role, phase) && !['engineer', 'deep_engineer'].includes(role)) return '';
    const guide = megaPlanner(role, phase)
      ? (phase === 'analyze' ? EXPERT_SKILL_PLANNER_EXTENSION_FILE
        : EXPERT_SKILL_PLANNER_GUIDE_FILE)
      : EXPERT_SKILL_AUTHOR_GUIDE_FILE;
    return `\n\n## MEGA EXPERT SKILL — ${
      EXPERT_SKILL_VALIDATION_STATUS === 'experimental'
        ? 'EXPERIMENTAL TRANSFER, EXPLICIT PIN' : 'VALIDATED PRIOR'
    }\n` +
      `Read ${guide}. ${
        megaPlanner(role, phase) && phase === 'analyze'
          ? 'Use its canonical Planner Extension to derive MegaPlanIR.'
          : 'Do not read or restate the full Skill contract; independent Verify owns it.'
      } ` +
      `The Skill is one validated solution for its stated size range (its skill.md "When to use"): ` +
      `inside that range treat it as a seed recipe — reproduce it first, then search beyond it; ` +
      `outside the range use it only for mechanisms${MRI.MEGA_REGIMES_GUIDE ? ' and let MEGA_REGIMES_GUIDE decide what transfers' : ''}. ` +
      `validation_status=${
        EXPERT_SKILL_VALIDATION_STATUS || 'legacy_validated'}, usage=${
        EXPERT_SKILL_USAGE || 'legacy'}. Candidate source remains search/integrated.` +
      (megaPlanner(role, phase) && REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY
        ? ` Before using the Skill, run python ${EXPERT_SKILL_BUNDLE_TOOL} ${EXPERT_SKILL_ID} ` +
          `--emit-bundle and require bundle_sha256=${EXPERT_SKILL_BUNDLE_SHA256}, ` +
          `planner_extension_sha256=${EXPERT_SKILL_PLANNER_EXTENSION_SHA256}, and ` +
          `contract_sha256=${EXPERT_SKILL_CONTRACT_SHA256}. A mismatch is a RunContract failure, ` +
          `not a reason to reinterpret stale knowledge.`
        : '');
  }
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills) and whose validation_status is \`validated\` and ` +
    `auto_apply is true, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.`;
}

function analysisSkillBlock(role, phase) {
  if (!ANALYSIS_SKILL_ON) return '';
  if (role === 'profile_engineer' && phase === 'analyze_profile') {
    return `\n\n## Profile-analysis skill (ANALYSIS ONLY — explicitly enabled this run)\n` +
      `Read ${ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR}/SKILL.md, execute only its checked-in ` +
      `builder/runner/analyzer/validator chain, and return the validated artifact contract. ` +
      `Never improvise arithmetic or change the generic Profile result.`;
  }
  if (role === 'tech_lead' && phase === 'plan_round') {
    return `\n\n## Profile-analysis evidence (explicitly enabled this run)\n` +
      `Inspect PROFILE_SUMMARY.analysis_result. Only Read analysis_json when status=ready and ` +
      `analysis_schema_version is the Skill's current schema. If analysis_status=awaiting_measurement, ` +
      `use findings only at their declared confidence; do not treat hypotheses, bounds, or claims as ` +
      `root cause and do not rank a direction from them. If evidence_complete, use only resolved ` +
      `high/medium-confidence evidence alongside generic profile/per-case data. A degraded or unavailable ` +
      `analysis contributes nothing; continue from the generic Profile result.`;
  }
  return '';
}

// Pool sample per round: it does not abort the run; an occupied pool redirects the round to lease-free work.
async function samplePool(round, timeoutMs) {
  if (!(GPU_MIN_FREE_GIB > 0)) return null;
  const p = await agentT(
    `Sample the GPU pool and report what you see. Do NOT acquire a lease, do NOT run any workload, ` +
    `do NOT try to free anything. Reading is the whole job.\n\n` +
    `1. Per-card free VRAM: read /sys/class/drm/card*/device/mem_info_vram_{total,used} (or ` +
    `\`rocm-smi --showmeminfo vram --csv\`) and compute free = total - used in GiB for every card.\n` +
    `2. Per-card utilisation: /sys/class/drm/card*/device/gpu_busy_percent.\n` +
    `3. \`rocm-smi --showpids\`. For each KFD PID it lists, check whether that PID exists in your own ` +
    `/proc. A PID that does NOT is in a FOREIGN NAMESPACE — that is an external tenant, not a stale ` +
    `lease of ours, and the two have opposite remedies. List those PIDs in foreign_pids.\n\n` +
    `verdict:"free" only if EVERY card has at least ${GPU_MIN_FREE_GIB} GiB free. If any card is ` +
    `below that floor, or a foreign tenant holds memory, verdict:"occupied". If you cannot read the ` +
    `pool at all, verdict:"unknown" — do not report an unreadable pool as free.`,
    { phase: 'Optimize', label: `pool sample r${round}`, schema: POOL_SCHEMA,
      ...(Number(timeoutMs) > 0 ? { timeout_ms: Number(timeoutMs), max_retries: 1 } : {}) });
  if (!p) {
    log(`POOL SAMPLE r${round}: sampler returned nothing. Treating as UNKNOWN, not free.`);
    return { verdict: 'unknown', note: 'sampler agent returned nothing' };
  }
  const detail = `${p.cards_free != null ? `${p.cards_free}/${p.cards_total} cards free, ` : ''}` +
    `min ${p.min_free_gib != null ? p.min_free_gib : '?'} GiB vs a ${GPU_MIN_FREE_GIB} GiB floor` +
    `${(p.foreign_pids || []).length ? `, foreign PIDs ${p.foreign_pids.join(',')}` : ''}`;
  if (p.verdict === 'free') {
    log(`POOL SAMPLE r${round}: FREE — ${detail}.`);
  } else {
    log(`POOL ${p.verdict.toUpperCase()} r${round}: ${detail}. ${p.note || ''}\n` +
        `  Plan this round GPU-LESS.`);
  }
  return p;
}

async function runProfileAnalysis(profileSummary, round, label) {
  if (!ANALYSIS_SKILL_ON || !profileSummary) return null;
  const result = await agentT(
    roleAgent(
      'profile_engineer',
      'analyze_profile',
      'Execute and validate the optional Step-2 analysis pipeline.',
      {
        WORKSPACE: CANONICAL,
        EVAL_DIR,
        ROUND: round,
        COMMANDMENT,
        PROFILE_SUMMARY: profileSummary,
        SKILL_DIR: WORKFLOW_DIR,
        ...ANALYSIS_SKILL_INPUTS, ...GMI, ...MRI,
      },
    ),
    {
      phase: 'Profile',
      label,
      schema: ANALYSIS_RESULT_SCHEMA,
      ...(MEGA_PRODUCTION ? { timeout_ms: 300000, max_retries: 1 } : {}),
    },
  );
  return result || {
    status: 'degraded',
    analysis_skill: ANALYSIS_SKILL,
    analysis_schema_version: '',
    analysis_status: 'unavailable',
    analysis_json: '',
    failure_note: 'analysis agent failed or returned invalid structured output',
  };
}

// Mega candidate registry (no blocking pre-loop reproduction phase).

// One candidate portfolio and lifecycle, with or without Expert Skills.
const MEGA_CANDIDATE_STATES = ['authoring', 'runnable', 'scored', 'finalist', 'rejected', 'parked'];
let megaCandidateRegistry = [];
let megaStateSequenceBase = 0;
let megaUnsafeTimeout = null;
// Consecutive rounds without new runtime evidence; drives measure-first and the stall stop.
let megaStall = 0;
let megaLastBlocker = '';
// A refused plan re-plans with the reason instead of ending the wave; 3 in a row still stops.
let megaRefusal = null;
let megaPcRemeasured = 0;
let megaMeasurementCalibration = {
  ready: false, candidate_id: '', attempt_id: '', evidence_manifest: '', note: 'not measured',
};

// <<REPLAY:analyze_resume_fallback>>
// A resumed Analyze with no ladder did not resume anything; rerun it once without the fast path.
function megaPlanIRVerdict(
  plan,
  expectedSkillRevision = '',
  expectedSkillId = '',
  expectedPlannerExtensionSha256 = '',
  expectedSkillBundleSha256 = '',
  expectedValidationStatus = '',
  expectedSkillUsage = ''
) {
  const errors = [];
  if (!plan || typeof plan !== 'object') return { pass: false, errors: ['plan missing'] };
  if (plan.plan_version !== 'mega-plan-v2') errors.push('plan_version must be mega-plan-v2');
  if (expectedSkillRevision &&
      String(plan.expert_skill_revision || '') !== String(expectedSkillRevision)) {
    errors.push(`expert_skill_revision must be ${expectedSkillRevision}`);
  }
  if (expectedSkillId && String(plan.expert_skill_id || '') !== String(expectedSkillId)) {
    errors.push(`expert_skill_id must be ${expectedSkillId}`);
  }
  if (expectedPlannerExtensionSha256 &&
      String(plan.expert_skill_planner_extension_sha256 || '') !==
        String(expectedPlannerExtensionSha256)) {
    errors.push('expert_skill_planner_extension_sha256 does not match the RunContract');
  }
  if (expectedSkillBundleSha256 &&
      String(plan.expert_skill_bundle_sha256 || '') !== String(expectedSkillBundleSha256)) {
    errors.push('expert_skill_bundle_sha256 does not match the RunContract');
  }
  if (expectedValidationStatus &&
      String(plan.expert_skill_validation_status || '') !== expectedValidationStatus) {
    errors.push('expert_skill_validation_status does not match bundle metadata');
  }
  if (expectedValidationStatus === 'experimental') {
    if (plan.expert_skill_auto_apply !== false) {
      errors.push('experimental Skill must report auto_apply=false');
    }
    if (!Array.isArray(plan.expert_skill_explicit_pin_modes) ||
        !plan.expert_skill_explicit_pin_modes.includes(expectedSkillUsage)) {
      errors.push('experimental Skill bundle does not permit the requested usage');
    }
  }
  const collections = [
    'work_domains', 'regions', 'buffers', 'counters', 'queues', 'events',
    'compiler_constraints', 'known_unknowns',
  ];
  const mayBeEmpty = new Set(['counters', 'known_unknowns']);
  for (const name of collections) {
    if (!Array.isArray(plan[name]) || (!mayBeEmpty.has(name) && !plan[name].length)) {
      errors.push(`${name} must be a ${mayBeEmpty.has(name) ? '' : 'non-empty '}array`);
    }
  }
  const ids = (values) => new Set((Array.isArray(values) ? values : [])
    .map((value) => String(value && value.id || '')).filter(Boolean));
  const domains = ids(plan.work_domains);
  const regions = ids(plan.regions);
  const counters = ids(plan.counters);
  const queues = ids(plan.queues);
  const requireUnique = (values, label) => {
    const present = (Array.isArray(values) ? values : [])
      .map((value) => String(value && value.id || '')).filter(Boolean);
    if (present.length !== new Set(present).size) errors.push(`${label} ids must be unique`);
  };
  for (const name of ['work_domains', 'regions', 'buffers', 'counters', 'queues', 'events']) {
    requireUnique(plan[name], name);
  }
  for (const d of Array.isArray(plan.work_domains) ? plan.work_domains : []) {
    if (d.parameters && d.parameters.owned_subdomain)
      errors.push(`work_domain ${d.id || '?'} owned_subdomain must be direct`);
    const o = d.owned_subdomain;
    if (o && ['unit', 'extent_per_claim', 'linear_index'].some((k) => !o[k]))
      errors.push(`work_domain ${d.id || '?'} owned_subdomain incomplete`);
  }
  for (const region of Array.isArray(plan.regions) ? plan.regions : []) {
    if (!domains.has(String(region.work_domain || ''))) {
      errors.push(`region ${region.id || '?'} references unknown work_domain`);
    }
  }
  const checkEndpoints = (item, label) => {
    for (const producer of Array.isArray(item.producers) ? item.producers : []) {
      if (!regions.has(String(producer))) errors.push(`${label} has unknown producer ${producer}`);
    }
    for (const consumer of Array.isArray(item.consumers) ? item.consumers : []) {
      if (!regions.has(String(consumer))) errors.push(`${label} has unknown consumer ${consumer}`);
    }
  };
  for (const buffer of Array.isArray(plan.buffers) ? plan.buffers : []) {
    checkEndpoints(buffer, `buffer ${buffer.id || '?'}`);
  }
  for (const counter of Array.isArray(plan.counters) ? plan.counters : []) {
    checkEndpoints(counter, `counter ${counter.id || '?'}`);
  }
  for (const queue of Array.isArray(plan.queues) ? plan.queues : []) {
    if (!domains.has(String(queue.work_domain || ''))) {
      errors.push(`queue ${queue.id || '?'} references unknown work_domain`);
    }
  }
  for (const event of Array.isArray(plan.events) ? plan.events : []) {
    if (!regions.has(String(event.producer || ''))) {
      errors.push(`event ${event.id || '?'} has unknown producer`);
    }
    if (!regions.has(String(event.consumer || ''))) {
      errors.push(`event ${event.id || '?'} has unknown consumer`);
    }
    if (event.counter && !counters.has(String(event.counter))) {
      errors.push(`event ${event.id || '?'} references unknown counter`);
    }
  }
  const target = plan.target || {};
  if (!(Number(target.launch_count) > 0)) errors.push('target.launch_count must be positive');
  const requireKnown = (values, known, label) => {
    for (const value of Array.isArray(values) ? values : []) {
      if (!known.has(String(value))) errors.push(`target ${label} has unknown id ${value}`);
    }
  };
  requireKnown(target.required_regions, regions, 'required_regions');
  requireKnown(target.required_queues, queues, 'required_queues');
  const resources = plan.resources || {};
  const workgroup = resources.workgroup || {};
  const localMemory = resources.local_memory || {};
  if (!String(resources.arch || '')) errors.push('resources.arch missing');
  if (!(Number(workgroup.wave_size) > 0) || !(Number(workgroup.thread_count) > 0)) {
    errors.push('resources.workgroup is incomplete');
  }
  if (!Number.isFinite(Number(localMemory.total_bytes)) ||
      Number(localMemory.total_bytes) < 0 ||
      Number(localMemory.total_bytes) > Number(localMemory.limit_bytes)) {
    errors.push('resources.local_memory is invalid');
  }
  const primaryLoop = plan.schedule && plan.schedule.primary_loop || {};
  if (!String(primaryLoop.kind || '')) errors.push('schedule.primary_loop.kind missing');
  if (!String(primaryLoop.queue_selection || '') &&
      !Array.isArray(primaryLoop.queue_priority)) {
    errors.push('schedule.primary_loop requires queue_selection or queue_priority');
  }
  for (const queue of Array.isArray(primaryLoop.queue_priority) ? primaryLoop.queue_priority : []) {
    if (!queues.has(String(queue))) errors.push(`primary loop references unknown queue ${queue}`);
  }
  if (!(plan.abi && String(plan.abi.entry_point || '') &&
      Array.isArray(plan.abi.arguments) && plan.abi.arguments.length)) {
    errors.push('abi entry_point/arguments missing');
  }
  if (!(plan.evidence_requirements && Object.keys(plan.evidence_requirements).length)) {
    errors.push('evidence_requirements must be non-empty');
  }
  return { pass: errors.length === 0, errors };
}

function analyzeResumeDegenerate(
  incremental,
  ver,
  requireCompleteGraph,
  requireMegaPlan = false,
  expectedSkillRevision = '',
  expectedSkillId = '',
  expectedPlannerExtensionSha256 = '',
  expectedSkillBundleSha256 = '',
  expectedValidationStatus = '',
  expectedSkillUsage = ''
) {
  const rungs = (ver && Array.isArray(ver.candidate_directions) ? ver.candidate_directions : [])
    .filter((c) => c && (c.id || c.title));
  const graphMissing = !!requireCompleteGraph &&
    (!(ver && ver.task_graph && Array.isArray(ver.task_graph.nodes) &&
       ver.task_graph.nodes.length) ||
     !(ver && ver.resource_timeline && Array.isArray(ver.resource_timeline.pipes)));
  const plan = ver && ver.mega_plan_ir;
  const planVerdict = megaPlanIRVerdict(
    plan,
    expectedSkillRevision,
    expectedSkillId,
    expectedPlannerExtensionSha256,
    expectedSkillBundleSha256,
    expectedValidationStatus,
    expectedSkillUsage
  );
  const megaPlanMissing = !!requireMegaPlan && !planVerdict.pass;
  if (rungs.length && !graphMissing && !megaPlanMissing) return { retry: false, reason: '' };
  if (!incremental && !graphMissing && !megaPlanMissing) return { retry: false, reason: '' };
  return { retry: true, reason:
    `${incremental ? 'ANALYZE RESUME DEGENERATE' : 'MEGA ANALYZE CONTRACT INCOMPLETE'}: ` +
    `${!rungs.length ? 'no structured candidate_directions' : ''}` +
    `${!rungs.length && graphMissing ? ' and ' : ''}` +
    `${graphMissing ? 'required task_graph/resource_timeline missing' : ''}` +
    `${(graphMissing || !rungs.length) && megaPlanMissing ? ' and ' : ''}` +
    `${megaPlanMissing
      ? `required mega_plan_ir invalid (${planVerdict.errors.slice(0, 4).join('; ')})`
      : ''}.` };
}
// <</REPLAY:analyze_resume_fallback>>
function applyBaselineOperatorMap(value) {
  if (!value || !Array.isArray(value.baseline_operator_map)) return value;
  const editable = [...new Set(value.baseline_operator_map
    .filter((row) => row && row.file && row.action !== 'reuse')
    .map((row) => String(row.file)))];
  value.modifiable_files = [...new Set([...(value.modifiable_files || []), ...editable])];
  for (const direction of (value.candidate_directions || [])) {
    direction.focus_files = [...new Set([...(direction.focus_files || []), ...editable])];
  }
  if (editable.length) {
    log(`Baseline operator map added ${editable.length} transformation file(s) to author scope.`);
  }
  return value;
}

// Task-graph gate (args.require_task_graph): makes ordering claims checkable as an enumerated graph.
// Advisory: a partial graph with honest unknowns beats an invented one. Pure function, lifted verbatim
// by tests/test_task_graph_gate.js between these markers; keep it free of log() and run state.
// <<REPLAY:task_graph_gate>>
function taskGraphGate(tg) {
  const nodes = tg && Array.isArray(tg.nodes) ? tg.nodes.length : 0;
  const edges = tg && Array.isArray(tg.edges) ? tg.edges.length : 0;
  if (!tg || (!nodes && !edges)) {
    return { verdict: 'MISSING', summary: '', caveat:
      'TASK GRAPH MISSING: this run required a tile-level dependency graph from Analyze and none ' +
      'was returned. Every direction ranked below is therefore asserted, not derived — there is no ' +
      'record of which orderings are required by the data and which are artifacts of the current ' +
      'code, no critical path to bound what any change can win, and no slack to say which nodes ' +
      'are worth touching. Read the rankings as hypotheses.' };
  }
  // Print the launch_boundary edge count: zero is a finding, all edges is a kernel-granularity graph.
  const eArr = Array.isArray(tg.edges) ? tg.edges : [];
  const launchEdges = eArr.filter((e) => e && e.enforced_by === 'launch_boundary').length;
  const unknowns = Array.isArray(tg.unknowns) ? tg.unknowns.length : 0;
  const assumed = (Array.isArray(tg.nodes) ? tg.nodes : [])
    .filter((n) => n && n.source === 'assumed').length;
  const cp = Number(tg.critical_path_us);
  const e2e = Number(tg.measured_e2e_us);
  const quantified = Number.isFinite(cp) && Number.isFinite(e2e) && e2e > 0;
  const summary = `Task graph: ${nodes} nodes, ${edges} edges, ${launchEdges} enforced only by a ` +
    `launch boundary, ${unknowns} declared unknown, ${assumed} node duration(s) assumed. ` +
    (quantified
      // 1 - cp/e2e: the share of e2e the dependence chain does not explain (not a ceiling on gains).
      ? `critical_path=${cp.toFixed(1)}µs vs e2e=${e2e.toFixed(1)}µs ` +
        `(dependence explains ${(100 * (cp / e2e)).toFixed(1)}% of e2e; the remaining ` +
        `${(100 * (1 - cp / e2e)).toFixed(1)}% is resource-bound work plus bubbles, and only the ` +
        'bubble set is addressable by reordering)'
      : 'critical path NOT quantified');

  if (quantified && assumed > 0) {
    return { verdict: 'UNMEASURED_NUMERIC', summary, caveat:
      `TASK GRAPH USES NUMERIC PLACEHOLDERS: ${assumed} node duration(s) are marked assumed while ` +
      `critical_path_us/measured_e2e_us are numeric.` };
  }
  if (quantified && cp > e2e) {
    return { verdict: 'INCONSISTENT', summary, caveat:
      `TASK GRAPH INCONSISTENT: critical_path_us (${cp}) exceeds measured_e2e_us (${e2e}). The ` +
      'longest path through the graph cannot be longer than the thing it is a path through, so ' +
      'either a node duration is wrong or an edge is spurious. Do not use this graph\'s headroom ' +
      'number to size any proposal.' };
  }
  // Every edge a launch boundary means the scope column was not filled in. Advisory.
  const narrow = eArr.filter((e) => e && e.scope && e.scope !== 'cross_rank' && e.scope !== 'hbm').length;
  if (edges && launchEdges === edges && narrow === 0) {
    return { verdict: 'KERNEL_GRANULARITY', summary, caveat:
      `TASK GRAPH IS KERNEL-GRANULAR: all ${edges} edges came back enforced_by=launch_boundary with ` +
      'no edge at a narrower scope, which is what a graph whose nodes are KERNELS looks like.' };
  }
  // WHAT THE CONSUMER NEEDS that the producer does not supply (weights, scales); advisory, [] is valid.
  const fanInEdges = eArr.filter((e) => e && Number(e.fan_in) > 0);
  const unanswered = fanInEdges
    .filter((e) => !Array.isArray(e.producer_independent_operands))
    .map((e) => `${e.from || '?'}->${e.to || '?'}`);
  if (unanswered.length) {
    return { verdict: 'OK', summary, caveat:
      `EDGES THAT DID NOT SAY WHAT THE CONSUMER LOADS ANYWAY (${unanswered.length}): ` +
      `${unanswered.join(', ')}.` };
  }
  return { verdict: 'OK', summary, caveat: '' };
}
// <</REPLAY:task_graph_gate>>

// <<REPLAY:pipe_occupancy_gate>>
// Pipe table gate, separate from taskGraphGate: the graph says what may overlap, this says whether the
// pipe is even busy.
const PIPE_SATURATED_PCT = 80;
const PIPE_LOW_PCT = 50;
const GAP_NEGLIGIBLE_US = 0.5;

function pipeOccupancyGate(rt, directions) {
  const dirs = Array.isArray(directions) ? directions : [];
  const pipes = rt && Array.isArray(rt.pipes)
    ? rt.pipes.filter((p) => p && p.utilization_pct != null &&
      Number.isFinite(Number(p.utilization_pct))) : [];
  if (!pipes.length) {
    return { verdict: 'MISSING', summary: '', class_derived: null, caveat:
      'PIPE TABLE MISSING: this run required a per-pipe resource timeline from Analyze and none was ' +
      'returned. Nothing below is priced against what the machine was actually doing — there is no ' +
      'record of which functional unit was the constraint, so a direction targeting an already-' +
      'saturated pipe and one filling an idle pipe are indistinguishable here, and the classic ' +
      'reflexes (raise occupancy, fuse launches) cannot be ruled out before they cost a lease. ' +
      'Read every expected_speedup as unbounded above.' };
  }
  const maxUtil = Math.max(...pipes.map((p) => Number(p.utilization_pct)));
  const satur = pipes.filter((p) => Number(p.utilization_pct) >= PIPE_SATURATED_PCT);
  const gap = rt.interkernel_gap_us && Number.isFinite(Number(rt.interkernel_gap_us.median))
    ? Number(rt.interkernel_gap_us.median) : null;
  const opps = Array.isArray(rt.idle_pipe_opportunities) ? rt.idle_pipe_opportunities.length : 0;
  const closed = Array.isArray(rt.closed_axes) ? rt.closed_axes.length : 0;

  // Derive the class from the numbers; a stated class the table does not support is worse than none.
  let derived = 'mixed';
  // Throughput-bound only when every measured engine is saturated (or only one was measured);
  // otherwise mixed, since idle engines can still take cross-engine work.
  if (satur.length && satur.length === pipes.length) derived = 'throughput_bound';
  else if (maxUtil < PIPE_LOW_PCT && gap !== null && gap > GAP_NEGLIGIBLE_US) derived = 'launch_bound';
  else if (maxUtil < PIPE_LOW_PCT) derived = 'latency_bound';

  const table = pipes
    .map((p) => `${p.stage || '?'}.${p.pipe || '?'}=${Number(p.utilization_pct).toFixed(1)}%`)
    .join(' ');
  const summary = `Pipe table: ${table}; max=${maxUtil.toFixed(1)}% ` +
    `(${satur.length} pipe(s) at/above ${PIPE_SATURATED_PCT}%); ` +
    `inter-kernel gap median=${gap === null ? 'UNMEASURED' : gap.toFixed(3) + 'us'}; ` +
    `class stated=${rt.class || 'none'} derived=${derived}; ` +
    `${opps} idle-pipe opportunit(y/ies), ${closed} closed axis/axes.`;

  const notes = [];
  if (rt.class && rt.class !== derived) {
    notes.push(`CLASS CONTRADICTS THE TABLE: Analyze stated class="${rt.class}" but the numbers it ` +
      `reported derive "${derived}" (max pipe ${maxUtil.toFixed(1)}%, gap ` +
      `${gap === null ? 'unmeasured' : gap.toFixed(3) + 'us'}). The class decides which lever ` +
      'families are candidates at all, so take the derived one and re-read the rankings against it.');
  }
  if (derived === 'latency_bound') {
    notes.push('LATENCY-BOUND: every pipe is below ' + PIPE_LOW_PCT + '% and the launch gap is ' +
      'negligible, so the machine is stalled on dependences, not on any resource. Two standard ' +
      'levers are dead here and should not be given a lease: raising OCCUPANCY (adding waves adds ' +
      'stalled waves when each wave\'s instruction stream has no independent work, and it spends ' +
      'the registers and LDS that software pipelining needs), and fusing to reclaim LAUNCH ' +
      'OVERHEAD (the gap is already zero). What does pay is making dependency-free work of the ' +
      'NEXT stage issuable in this one\'s shadow — a kernel boundary is a grid-wide barrier plus a ' +
      'pipeline drain, so that work is not unscheduled, it is inexpressible.');
    if (!opps) {
      notes.push('NO IDLE-PIPE OPPORTUNITIES LISTED: the table says the pipes are idle and the ' +
        'artifact names nothing that could be issued into them.');
    }
  }
  // Directions priced above what their own pipe can pay.
  const overclaimed = [];
  const unpriced = [];
  for (const d of dirs) {
    if (!d || d.specialty === 'deep_explore') continue;
    if (!d.fills_pipe) { unpriced.push(d.id || d.title || '?'); continue; }
    const util = Number.isFinite(Number(d.pipe_util_pct)) ? Number(d.pipe_util_pct) : null;
    const exp = Number(d.expected_speedup);
    if (util === null || !Number.isFinite(exp)) continue;
    const ceiling = 1 / Math.max(util / 100, 1e-6); // filling an idle pipe completely, the optimistic bound
    if (exp > ceiling) {
      overclaimed.push(`${d.id || d.title}: claims ${exp.toFixed(3)}x on ${d.fills_pipe} at ` +
        `${util.toFixed(1)}% utilization, whose perfect-fill bound is ${ceiling.toFixed(3)}x`);
    }
  }
  if (unpriced.length) {
    notes.push(`DIRECTIONS WITH NO PIPE NAMED (${unpriced.length}): ${unpriced.join(', ')}.`);
  }
  if (overclaimed.length) {
    notes.push('DIRECTION OVERCLAIMS ITS PIPE: ' + overclaimed.join('; ') +
      '. These are arithmetically impossible against the reported utilization and can be rejected ' +
      'without a run; if the utilization is wrong, that is the finding.');
  }
  // A sized hole no direction claims.
  if (dirs.length) {
    const claimed = new Set();
    for (const d of dirs) {
      if (!d) continue;
      if (d.fills_hole) claimed.add(String(d.fills_hole));
      for (const g of Array.isArray(d.graph_refs) ? d.graph_refs : []) claimed.add(String(g));
    }
    const unowned = (Array.isArray(rt.idle_pipe_opportunities) ? rt.idle_pipe_opportunities : [])
      .filter((o) => o && (Number(o.recoverable_us) > 0 || Number(o.pct_of_e2e) > 0))
      .filter((o) => !o.rides_on && !claimed.has(String(o.id)))
      .map((o) => `${o.id || o.window || '?'} (${Number(o.recoverable_us) > 0 ? Number(o.recoverable_us).toFixed(0) + 'us' : ''}` +
        `${Number(o.pct_of_e2e) > 0 ? (Number(o.recoverable_us) > 0 ? ', ' : '') + Number(o.pct_of_e2e).toFixed(1) + '% of e2e' : ''})`);
    if (unowned.length) {
      notes.push(`SIZED HOLE WITH NO DIRECTION (${unowned.length}): ${unowned.join('; ')}.`);
    }
  }
  const verdict = notes.length ? (overclaimed.length || (rt.class && rt.class !== derived) ? 'INCONSISTENT' : 'ADVISORY') : 'OK';
  return { verdict, summary, class_derived: derived, caveat: notes.join(' ') };
}
// <</REPLAY:pipe_occupancy_gate>>

// <<REPLAY:roadmap_ladder_gate>>
// Did the round plan against Analyze's ladder?
//
// <<REPLAY:open_rungs>>
// A rung is done when it produced a number, not when it was planned.
const RUNG_OUTCOMES = ['never_planned', 'faulted', 'unmeasured', 'measured'];
function rungIdOf(c) {
  return String((c && c.id) || String((c && c.title) || '').trim().split(/\s+/)[0] || '').trim();
}
function openRungs(ladder, tally) {
  const t = tally instanceof Map ? tally : new Map(Object.entries(tally || {}));
  return (Array.isArray(ladder) ? ladder : [])
    .filter((c) => c && (c.id || c.title))
    .map((c) => {
      const id = rungIdOf(c);
      const e = t.get(id) || {};
      const outcome = RUNG_OUTCOMES.includes(e.last_outcome) ? e.last_outcome : 'never_planned';
      return { id, title: c.title || id, gated_on: Array.isArray(c.gated_on) ? c.gated_on : [],
        is_positive_control: !!c.is_positive_control,
        // Carried so the next wave knows what closing this rung would even mean.
        step_role: stepRoleOf(c), enables: String(c.enables || '') || undefined,
        mandatory_arms: Array.isArray(c.mandatory_arms) ? c.mandatory_arms : [],
        target_shape: c.target_shape || undefined,
        attempts: Number(e.attempts) || 0, last_outcome: outcome };
    })
    .filter((c) => c.last_outcome !== 'measured');
}
// <</REPLAY:open_rungs>>

// <<REPLAY:direction_contract>>
// Join the planner's direction back to Analyze's rung, so dropped step_role/mandatory_arms do not turn
// a prerequisite into a terminal speed test.
function enrichDirection(direction, ladder) {
  const d = direction || {};
  const idOf = (c) => String((c && (c.id || c.title)) || '').trim().split(/\s+/)[0];
  const rungId = String(d.roadmap_rung || '').trim();
  const rung = (Array.isArray(ladder) ? ladder : []).find((c) => idOf(c) === rungId) || {};
  const pickArray = (a, b) => Array.isArray(a) ? a : (Array.isArray(b) ? b : []);
  return {
    ...d,
    gated_on: pickArray(d.gated_on, rung.gated_on),
    mandatory_arms: pickArray(d.mandatory_arms, rung.mandatory_arms),
    step_role: d.step_role || rung.step_role || 'terminal',
    enables: d.enables || rung.enables || undefined,
    cost_budget_pct: d.cost_budget_pct != null ? d.cost_budget_pct : rung.cost_budget_pct,
    target_shape: d.target_shape || rung.target_shape || undefined,
  };
}

// <</REPLAY:direction_contract>>

// <<REPLAY:autonomy_transition>>
function finalStatusVerdict(directorStatus, scopedGuardFailure) {
  if (directorStatus === 'flagged' || scopedGuardFailure) return 'flagged';
  return directorStatus || 'unknown';
}

function finalMetricFailure(metric, targetGuards, primary, guards, attribution) {
  const scoped = Array.isArray(targetGuards) && targetGuards.length > 0;
  if ((scoped || metric === 'changed_kernel') && !(Number(primary) > 1.0)) return true;
  if (scoped && (!guards || !guards.complete || !guards.regression_pass)) return true;
  if (metric === 'changed_kernel') {
    const a = attribution || {};
    if (!(a.absolute_to_frozen === true && Number(a.changed_us) > 0 &&
          Number(a.replaced_sum_us) > 0)) return true;
  }
  return false;
}

// <</REPLAY:autonomy_transition>>

// Advisory: never blocks a direction, but a skipped rung must be stated via rung_deviation.
function roadmapLadderGate(ladder, directions, completedRungs) {
  const rungs = (Array.isArray(ladder) ? ladder : []).filter((c) => c && (c.id || c.title));
  const dirs = Array.isArray(directions) ? directions : [];
  const done = new Set(Array.isArray(completedRungs) ? completedRungs : completedRungs instanceof Set ? [...completedRungs] : []);
  if (!rungs.length) {
    return { verdict: 'MISSING', summary: '', planned: [], caveat:
      'LADDER MISSING: Analyze returned no candidate_directions, so there is no recorded ordering ' +
      'for this run and nothing below can be checked against a plan.' };
  }
  // `id` is what a plan can quote; fall back to the leading token of the title, which is how these
  // are written in practice ("D2 Per-token ... readiness").
  const idOf = (c) => String(c.id || String(c.title).trim().split(/\s+/)[0] || '').trim();
  const byId = new Map(rungs.map((c) => [idOf(c), c]));

  const planned = [];
  const undeclared = [];
  const offLadder = [];
  const gateViolations = [];
  for (const d of dirs) {
    if (!d) continue;
    const label = d.id || d.title || '?';
    const rung = (d.roadmap_rung || '').trim();
    if (!rung) { undeclared.push(label); continue; }
    if (rung === 'off_ladder') {
      if (!String(d.rung_deviation || '').trim()) offLadder.push(label);
      continue;
    }
    planned.push(rung);
    const c = byId.get(rung);
    if (!c) { offLadder.push(`${label} (names rung "${rung}", which is not on the ladder)`); continue; }
    const gates = Array.isArray(c.gated_on) ? c.gated_on.map(String) : [];
    const unmet = gates.filter((g) => !done.has(g));
    if (unmet.length && !String(d.rung_deviation || '').trim()) {
      gateViolations.push(`${label} takes ${rung}, whose gated_on [${unmet.join(', ')}] ` +
        'has not been dispatched, with no rung_deviation stated');
    }
  }

  const seen = new Set([...done, ...planned]);
  const never = rungs.filter((c) => !seen.has(idOf(c)));
  const controls = never.filter((c) => c.is_positive_control);
  const blocked = rungs.filter((c) => {
    const gates = Array.isArray(c.gated_on) ? c.gated_on.map(String) : [];
    return gates.length && !seen.has(idOf(c)) && gates.some((g) => !seen.has(g));
  });

  const summary = `Ladder: ${rungs.length} rung(s) [${rungs.map(idOf).join(', ')}]; ` +
    `completed to spec [${[...done].join(', ') || 'none'}]; ` +
    `this round [${planned.join(', ') || 'none on ladder'}]; ` +
    `${never.length} never reached.`;

  const notes = [];
  if (undeclared.length) {
    notes.push(`DIRECTIONS WITH NO RUNG NAMED (${undeclared.length}): ${undeclared.join(', ')}.`);
  }
  if (offLadder.length) {
    notes.push(`OFF-LADDER WITHOUT A STATED DEVIATION: ${offLadder.join('; ')}.`);
  }
  if (gateViolations.length) {
    notes.push('RUNG TAKEN OUT OF ORDER: ' + gateViolations.join('; ') + '. The prerequisite is ' +
      'usually what makes the result INTERPRETABLE, not merely what makes it likelier to win — a ' +
      'readiness rung measured before its bounding readout can read negative for a reason the ' +
      'bounding readout would have priced in advance.');
  }
  if (controls.length) {
    notes.push('THE LADDER\'S OWN POSITIVE CONTROL HAS NOT BEEN DISPATCHED: ' +
      controls.map(idOf).join(', ') + '. Analyze designated this rung as the change with a ' +
      'predicted direction that proves the harness can detect a win at all. Until it runs, every ' +
      'null below is ambiguous between "no effect" and "no instrument".');
  }
  if (blocked.length) {
    notes.push(`RUNGS STILL BLOCKED BY UNDISPATCHED PREREQUISITES (${blocked.length}): ` +
      blocked.map((c) => `${idOf(c)} gated_on [${(c.gated_on || []).join(', ')}]`).join('; ') +
      '. These cannot be reached by continuing to plan around the ladder. If a gate is being ' +
      'abandoned, say so; a gate that is simply never satisfied silently deletes everything above it.');
  }
  const verdict = notes.length
    ? (gateViolations.length || controls.length ? 'INCONSISTENT' : 'ADVISORY')
    : 'OK';
  return { verdict, summary, planned, caveat: notes.join(' ') };
}
// <</REPLAY:roadmap_ladder_gate>>

// <<REPLAY:mega_candidate_registry>>
// Mega whole-kernel candidates remain independent lineages until verified.
function validMegaCandidateId(value) {
  const id = String(value || '');
  return /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(id) && id !== 'frozen_baseline';
}
// Resumed waves restart round numbers, so `_rN` lineage grows without bound; hash-compact at the cap.
function megaChildCandidateId(parentId, round) {
  const id = `${parentId}_r${round}`;
  if (validMegaCandidateId(id)) return id;
  let h = 5381;
  for (const ch of id) h = (h * 33 + ch.charCodeAt(0)) >>> 0;
  const tag = `_${h.toString(36)}_r${round}`;
  return parentId.slice(0, 64 - tag.length).replace(/[._-]+$/, '') + tag;
}
function validMegaCandidateSourceForId(_id, source) {
  return source === 'search' || source === 'integrated';
}
function normalizeMegaActivation(raw) {
  const a = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const switches = Array.isArray(a.switches)
    ? a.switches.map((s) => ({
      switch_name: String(s && s.switch_name || ''),
      switch_value: String(s && s.switch_value || ''),
    })).filter((s) => s.switch_name) : [];
  return {
    mode: String(a.mode || ''),
    switch_name: String(a.switch_name || ''),
    switch_value: String(a.switch_value || ''),
    path_marker: String(a.path_marker || ''),
    marker_how: String(a.marker_how || ''),
    switches,
  };
}
function normalizeContractFailures(raw) {
  return (Array.isArray(raw) ? raw : []).map((failure) => ({
    id: String(failure && failure.id || ''),
    category: String(failure && failure.category || 'semantic'),
    severity: String(failure && failure.severity || 'required'),
    tier: failure && failure.tier === 'surface' ? 'surface' : 'semantic',
    messages: Array.isArray(failure && failure.messages)
      ? failure.messages.map(String) : [],
  })).filter((failure) => failure.id);
}
function normalizeMegaWorkingSnapshot(raw) {
  const w = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  const failures = normalizeContractFailures(w.contract_failures);
  const structuralConflict = w.structural_verified === true &&
    failures.some((failure) => failure.severity === 'required');
  return {
    head: String(w.head || ''),
    status: String(w.status || ''),
    patch: String(w.patch || ''),
    attempt_id: String(w.attempt_id || ''),
    checkpoint_complete: w.checkpoint_complete === true,
    structural_verified: w.structural_verified === true && !structuralConflict,
    structural_report: String(w.structural_report || ''),
    structural_skill_id: String(w.structural_skill_id || ''),
    structural_candidate_head: String(w.structural_candidate_head || ''),
    structural_candidate_tree_digest: String(w.structural_candidate_tree_digest || ''),
    structural_skill_bundle_sha256: String(w.structural_skill_bundle_sha256 || ''),
    structural_planner_extension_sha256:
      String(w.structural_planner_extension_sha256 || ''),
    structural_contract_revision: String(w.structural_contract_revision || ''),
    structural_contract_sha256: String(w.structural_contract_sha256 || ''),
    evidence_manifest: String(w.evidence_manifest || ''),
    activation: normalizeMegaActivation(w.activation),
    topology_sig: String(w.topology_sig || ''),
    topology: w.topology && typeof w.topology === 'object' && !Array.isArray(w.topology)
      ? { ...w.topology } : {},
    changed_files: Array.isArray(w.changed_files) ? w.changed_files.map(String) : [],
    contract_failures: structuralConflict ? [{
      id: 'structural_state_conflict', category: 'plan', severity: 'required',
      messages: ['working structural pass conflicts with required failures'],
    }] : failures,
    next_blocker: String(w.next_blocker || ''),
    notes: String(w.notes || ''),
  };
}
// Git commit ids from different agents: one may be abbreviated (>= 7 hex), so compare by prefix.
function sameCommit(a, b) {
  const x = String(a || '').trim().toLowerCase();
  const y = String(b || '').trim().toLowerCase();
  if (x && x === y) return true;
  if (!/^[0-9a-f]{7,40}$/.test(x) || !/^[0-9a-f]{7,40}$/.test(y)) return false;
  return x.length <= y.length ? y.startsWith(x) : x.startsWith(y);
}

function normalizeMegaCandidate(raw) {
  const c = raw || {};
  const sourceValid = ['search', 'integrated'].includes(c.source);
  const source = sourceValid ? c.source : 'search';
  const idValid = validMegaCandidateId(c.id);
  const sourceIdValid = validMegaCandidateSourceForId(c.id, source);
  const candidateStates = typeof MEGA_CANDIDATE_STATES === 'undefined'
    ? ['authoring', 'runnable', 'scored', 'finalist', 'rejected', 'parked'] : MEGA_CANDIDATE_STATES;
  const status = (!sourceValid || !idValid || !sourceIdValid) ? 'rejected'
    : (candidateStates.includes(c.status) ? c.status : 'authoring');
  const score = Number(c.absolute_score);
  const launches = Number(c.launches);
  const rawContractFailures = normalizeContractFailures(c.contract_failures);
  const structuralConflict = c.structural_verified === true &&
    rawContractFailures.some((failure) => failure.severity === 'required');
  const contractFailures = structuralConflict ? [{
    id: 'structural_state_conflict', category: 'plan', severity: 'required',
    messages: [`legacy structural pass conflicts with failures: ${
      rawContractFailures.map((failure) => failure.id).join(', ')}`],
  }] : rawContractFailures;
  return {
    id: String(c.id || ''),
    id_valid: idValid,
    source,
    source_valid: sourceValid,
    source_id_valid: sourceIdValid,
    base_id: String(c.base_id || 'frozen_baseline'),
    tree: String(c.tree || ''),
    head: String(c.head || ''),
    working_head: String(c.working_head || ''),
    working_status: String(c.working_status || ''),
    working_snapshot: normalizeMegaWorkingSnapshot(c.working_snapshot),
    patch: String(c.patch || ''),
    status,
    topology_sig: String(c.topology_sig || ''),
    topology: c.topology && typeof c.topology === 'object' && !Array.isArray(c.topology)
      ? { ...c.topology } : {},
    activation: normalizeMegaActivation(c.activation),
    provenance: String(c.provenance || source),
    claim_complete: c.claim_complete === true,
    checkpoint_complete: c.checkpoint_complete === true,
    structural_verified: c.structural_verified === true && !structuralConflict,
    runtime_verified: c.runtime_verified === true,
    score_complete: c.score_complete === true,
    verification_status: String(c.verification_status || ''),
    correctness: String(c.correctness || ''),
    gpu_executed: c.gpu_executed === true,
    activation_on_hardware: String(c.activation_on_hardware || ''),
    structural_report: String(c.structural_report || ''),
    structural_skill_id: String(c.structural_skill_id || ''),
    structural_candidate_head: String(c.structural_candidate_head || ''),
    structural_candidate_tree_digest: String(c.structural_candidate_tree_digest || ''),
    structural_skill_bundle_sha256: String(c.structural_skill_bundle_sha256 || ''),
    structural_planner_extension_sha256:
      String(c.structural_planner_extension_sha256 || ''),
    structural_contract_revision: String(c.structural_contract_revision || ''),
    structural_contract_sha256: String(c.structural_contract_sha256 || ''),
    contract_failures: contractFailures,
    attempt_id: String(c.attempt_id || ''),
    changed_files: Array.isArray(c.changed_files) ? c.changed_files.map(String) : [],
    evidence_manifest: String(c.evidence_manifest || ''),
    target_guard: String(c.target_guard || ''),
    final_tree: String(c.final_tree || ''),
    final_head: String(c.final_head || ''),
    final_attempt_id: String(c.final_attempt_id || ''),
    final_evidence_manifest: String(c.final_evidence_manifest || ''),
    absolute_score: Number.isFinite(score) && score > 0 ? score : null,
    per_case: Array.isArray(c.per_case) ? c.per_case : [],
    paired_readings: Array.isArray(c.paired_readings) ? c.paired_readings : [],
    denominator_mismatch: Array.isArray(c.denominator_mismatch) ? c.denominator_mismatch : [],
    component_breakdown: Array.isArray(c.component_breakdown) ? c.component_breakdown : [],
    hot_path_counters: c.hot_path_counters && typeof c.hot_path_counters === 'object'
      ? c.hot_path_counters : {},
    null_arm_pct: c.null_arm_pct == null ? null : Number(c.null_arm_pct),
    reps: Math.max(0, Number(c.reps || 0)),
    artifact_hash_base: String(c.artifact_hash_base || ''),
    artifact_hash_candidate: String(c.artifact_hash_candidate || ''),
    correctness_pass: c.correctness_pass === true,
    activation_pass: c.activation_pass === true,
    head_pass: c.head_pass === true,
    launch_pass: c.launch_pass === true,
    liveness_pass: c.liveness_pass === true,
    graph_pass: c.graph_pass === true,
    guards_pass: c.guards_pass !== false,
    artifact_distinct: c.artifact_distinct === true,
    measurement_pass: c.measurement_pass === true,
    launches: Number.isInteger(launches) && launches > 0 ? launches : null,
    attempts: Math.max(0, Number(c.attempts || 0)),
    next_blocker: String(c.next_blocker || ''),
    notes: String(c.notes || ''),
  };
}

function upsertMegaCandidate(registry, incoming) {
  const list = Array.isArray(registry) ? registry.map(normalizeMegaCandidate) : [];
  let next = normalizeMegaCandidate(incoming);
  if (!next.id) return list;
  const i = list.findIndex((c) => c.id === next.id);
  if (i < 0) return [...list, next];
  const prev = list[i];
  const priorStructural = megaEffectiveLaneCheckpoint(prev);
  const sameStructuralHead = priorStructural.structural_verified === true &&
    !!priorStructural.structural_candidate_head &&
    next.head === priorStructural.structural_candidate_head &&
    next.head === priorStructural.head;
  const runtimeOnlyExactHeadFailure = sameStructuralHead &&
    !next.structural_verified && next.contract_failures.length === 0;
  if (runtimeOnlyExactHeadFailure) {
    next = normalizeMegaCandidate({
      ...next,
      checkpoint_complete: priorStructural.checkpoint_complete,
      structural_verified: true,
      structural_report: priorStructural.structural_report,
      structural_skill_id: priorStructural.structural_skill_id,
      structural_candidate_head: priorStructural.structural_candidate_head,
      structural_candidate_tree_digest: priorStructural.structural_candidate_tree_digest,
      structural_skill_bundle_sha256: priorStructural.structural_skill_bundle_sha256,
      structural_planner_extension_sha256:
        priorStructural.structural_planner_extension_sha256,
      structural_contract_revision: priorStructural.structural_contract_revision,
      structural_contract_sha256: priorStructural.structural_contract_sha256,
      contract_failures: [],
    });
  }
  if (prev.claim_complete && !next.claim_complete) {
    list[i] = normalizeMegaCandidate({
      ...prev,
      attempts: Math.max(prev.attempts, next.attempts),
      working_head: next.head || next.working_head || prev.working_head,
      working_status: next.status || next.working_status || prev.working_status,
      working_snapshot: {
        head: next.head || next.working_head || prev.working_head,
        status: next.status || next.working_status || prev.working_status,
        patch: next.patch,
        attempt_id: next.attempt_id,
        checkpoint_complete: next.checkpoint_complete,
        structural_verified: next.structural_verified,
        structural_report: next.structural_report,
        structural_skill_id: next.structural_skill_id,
        structural_candidate_head: next.structural_candidate_head,
        structural_candidate_tree_digest: next.structural_candidate_tree_digest,
        structural_skill_bundle_sha256: next.structural_skill_bundle_sha256,
        structural_planner_extension_sha256:
          next.structural_planner_extension_sha256,
        structural_contract_revision: next.structural_contract_revision,
        structural_contract_sha256: next.structural_contract_sha256,
        evidence_manifest: next.evidence_manifest,
        activation: next.activation,
        topology_sig: next.topology_sig,
        topology: next.topology,
        changed_files: next.changed_files,
        contract_failures: next.contract_failures,
        next_blocker: next.next_blocker,
        notes: next.notes,
      },
      next_blocker: next.next_blocker || prev.next_blocker,
      notes: next.notes || prev.notes,
    });
  } else {
    list[i] = normalizeMegaCandidate({
      ...prev,
      ...next,
      claim_complete: next.claim_complete === true,
      attempts: Math.max(prev.attempts, next.attempts),
      working_head: '',
      working_status: '',
      working_snapshot: {},
    });
  }
  return list;
}

function megaRegistryForSearch(registry) {
  return (Array.isArray(registry) ? registry : []).map(normalizeMegaCandidate).map((c) => {
    const effective = megaEffectiveLaneCheckpoint(c);
    return {
      id: c.id,
      source: c.source,
      base_id: c.base_id,
      tree: c.tree,
      head: c.head,
      working_head: c.working_head,
      working_status: c.working_status,
      status: c.status,
      checkpoint_complete: effective.checkpoint_complete === true,
      structural_verified: effective.structural_verified === true,
      runtime_verified: c.runtime_verified,
      score_complete: c.score_complete,
      structural_report: effective.structural_report || '',
      structural_skill_id: effective.structural_skill_id || '',
      structural_candidate_head: effective.structural_candidate_head || '',
      structural_candidate_tree_digest: effective.structural_candidate_tree_digest || '',
      structural_skill_bundle_sha256: effective.structural_skill_bundle_sha256 || '',
      structural_planner_extension_sha256:
        effective.structural_planner_extension_sha256 || '',
      structural_contract_revision: effective.structural_contract_revision || '',
      structural_contract_sha256: effective.structural_contract_sha256 || '',
      contract_failures: effective.contract_failures || [],
      verification_status: c.verification_status,
      correctness: c.correctness,
      gpu_executed: c.gpu_executed,
      activation_on_hardware: c.activation_on_hardware,
      absolute_score: c.absolute_score,
      target_guard: c.target_guard,
      per_case: c.per_case,
      denominator_mismatch: c.denominator_mismatch,
      component_breakdown: c.component_breakdown,
      hot_path_counters: c.hot_path_counters,
      topology_sig: effective.topology_sig || c.topology_sig,
      next_blocker: effective.next_blocker || c.next_blocker,
      notes: effective.notes || c.notes,
      attempts: c.attempts,
    };
  });
}

function megaHistoryForSearch(sourceHistory, registry) {
  const h = sourceHistory || {};
  const searchRegistry = megaRegistryForSearch(registry)
    .filter((c) => c.source === 'search' || c.source === 'integrated');
  return {
    ...h,
    // Global insights are free-form and have no reliable source label.
    insights: searchRegistry.flatMap((c) =>
      [c.next_blocker, c.notes].filter((text) => String(text || '').trim())),
    rounds: (Array.isArray(h.rounds) ? h.rounds : []).map((round) => {
      const directions = (Array.isArray(round.directions) ? round.directions : [])
        .filter((d) => ['search', 'integrated'].includes(String(d && d.candidate_source || '')));
      const directionIds = new Set(directions.map((d) => String(d && d.id || '')));
      const results = (Array.isArray(round.results) ? round.results : [])
        .filter((r) => directionIds.has(String(r && r.id || '')));
      return { ...round, directions, results };
    }).filter((round) => round.directions.length),
    ledger: searchRegistry.map((c) => ({
      id: c.id,
      direction: c.id,
      candidate_id: c.id,
      candidate_source: c.source,
      expected: null,
      actual: c.absolute_score,
      verdict: c.status,
      lesson: c.next_blocker || c.notes || '',
    })),
  };
}

function megaCandidateHardPass(c) {
  const n = normalizeMegaCandidate(c);
  const configuredTarget = typeof TARGET_GUARDS === 'undefined'
    ? '' : TARGET_GUARDS.join(',');
  const targetReadout = megaTargetsReadout(n.paired_readings,
    String(configuredTarget || n.target_guard).split(','));
  const structuralPass = typeof STRUCTURAL_CONTRACT_BLOCKING === 'undefined' ||
    !STRUCTURAL_CONTRACT_BLOCKING || n.structural_verified;
  return structuralPass &&
    n.claim_complete && n.correctness_pass && n.activation_pass && n.head_pass &&
    n.launch_pass && n.liveness_pass && n.graph_pass && n.guards_pass &&
    n.artifact_distinct && n.id_valid && n.source_valid && n.source_id_valid &&
    n.measurement_pass && String(n.attempt_id) && String(n.evidence_manifest) &&
    n.artifact_hash_base && n.artifact_hash_candidate &&
    n.artifact_hash_base !== n.artifact_hash_candidate &&
    n.reps >= 5 && Number.isFinite(n.null_arm_pct) &&
    targetReadout.count >= 5 && Number(targetReadout.score) > 0 &&
    Math.abs(Number(targetReadout.score) - Number(n.absolute_score)) <= 0.002 &&
    targetReadout.sign_p <= 0.05 &&
    (Number(targetReadout.min) - 1) * 100 > Math.abs(n.null_arm_pct) &&
    Number.isFinite(n.absolute_score) && n.absolute_score > 0;
}

function selectMegaCandidate(registry, tieNoisePct) {
  const eligible = (Array.isArray(registry) ? registry : [])
    .map(normalizeMegaCandidate)
    .filter((c) => (c.status === 'scored' || c.status === 'finalist') &&
      megaCandidateHardPass(c) && c.absolute_score > 1.0)
    .sort((a, b) => b.absolute_score - a.absolute_score);
  if (!eligible.length) return { selected: null, eligible: [], tie_kept_skill: false };
  return { selected: eligible[0], eligible, tie_kept_skill: false };
}

function selectMegaSearchParent(registry) {
  return (Array.isArray(registry) ? registry : [])
    .map(normalizeMegaCandidate)
    .filter((c) => (c.source === 'search' || c.source === 'integrated') &&
      (c.status === 'scored' || c.status === 'finalist') &&
      megaCandidateHardPass(c) && c.absolute_score > 1.0)
    .sort((a, b) => b.absolute_score - a.absolute_score)[0] || null;
}

function megaCalibrationClaimPass(pc, configured) {
  const p = pc || {};
  const c = configured || {};
  const got = Number(p.measured_pct);
  const lo = Number(c.expected_pct_lo), hi = Number(c.expected_pct_hi);
  const sign = Math.sign(lo + hi) || 1;
  const ctrl = Array.isArray(p.control_pairs_pct)
    ? p.control_pairs_pct.map(Number).filter(Number.isFinite) : [];
  const nul = Array.isArray(p.null_pairs_pct)
    ? p.null_pairs_pct.map(Number).filter(Number.isFinite) : [];
  const requiredControlPairs = Math.max(1, Number(c.required_control_pairs || 5));
  const requiredNullPairs = Math.max(1, Number(c.required_null_pairs || 8));
  const nullWorst = nul.length ? Math.max(...nul.map(Math.abs)) : NaN;
  const minEffect = Math.min(Math.abs(lo), Math.abs(hi));
  const maxPlausible = Number.isFinite(Number(c.implausible_pct))
    ? Math.abs(Number(c.implausible_pct)) : Math.max(Math.abs(lo), Math.abs(hi)) * 2;
  return p.claim_complete === true && p.ran === true && p.switch_present === true &&
    String(p.attempt_id || '').length > 0 && String(p.evidence_manifest || '').length > 0 &&
    p.passed === true && Number.isFinite(got) && Math.sign(got) === sign &&
    Math.abs(got) >= minEffect * 0.5 && Math.abs(got) <= maxPlausible &&
    ctrl.length >= requiredControlPairs && ctrl.every((x) => Math.sign(x) === sign) &&
    nul.length >= requiredNullPairs && Number.isFinite(nullWorst) && Math.abs(got) >= 3 * nullWorst;
}

function pairedGuardReadout(rows, guard, bimodalGuards = []) {
  const rawPairs = (Array.isArray(rows) ? rows : []).filter((r) =>
    String(r.guard || '') === String(guard) &&
    Number(r.base) > 0 && Number(r.cand) > 0);
  let pairs = rawPairs;
  let bimodal = false, split = null;
  if ((Array.isArray(bimodalGuards) ? bimodalGuards : [])
      .map(String).includes(String(guard)) && rawPairs.length >= 8) {
    const values = rawPairs.flatMap((r, index) => [
      { value: Number(r.base), arm: 'base', index },
      { value: Number(r.cand), arm: 'cand', index },
    ]).sort((a, b) => a.value - b.value);
    let widest = 0, splitAt = -1;
    for (let i = 0; i + 1 < values.length; i++) {
      const gap = (values[i + 1].value - values[i].value) / values[i].value;
      if (gap > widest) { widest = gap; splitAt = i; }
    }
    if (widest >= 0.03 && splitAt >= 0) {
      const lowArms = new Set(values.slice(0, splitAt + 1).map((x) => x.arm));
      const highArms = new Set(values.slice(splitAt + 1).map((x) => x.arm));
      // Arm-blind state conditioning is valid only when BOTH arms draw both states.
      if (lowArms.size === 2 && highArms.size === 2) {
        split = (values[splitAt].value + values[splitAt + 1].value) / 2;
        pairs = rawPairs.filter((r) => Number(r.base) <= split && Number(r.cand) <= split);
        bimodal = true;
      }
    }
  }
  const ratios = pairs.map((r) => Number(r.base) / Number(r.cand)).sort((a, b) => a - b);
  const median = ratios.length
    ? (ratios.length % 2 ? ratios[(ratios.length - 1) / 2]
      : (ratios[ratios.length / 2 - 1] + ratios[ratios.length / 2]) / 2)
    : null;
  const favorable = ratios.filter((v) => v > 1.0).length;
  const choose = (n, k) => {
    let out = 1;
    for (let i = 1; i <= k; i++) out = out * (n - k + i) / i;
    return out;
  };
  let signP = 1;
  if (ratios.length) {
    let tail = 0;
    for (let k = favorable; k <= ratios.length; k++) tail += choose(ratios.length, k);
    signP = tail / Math.pow(2, ratios.length);
  }
  return {
    raw_count: rawPairs.length, count: ratios.length, score: median,
    favorable, sign_p: signP, bimodal, split,
  };
}

// Each target passes alone; headline = geomean.
function megaTargetsReadout(rows, guards) {
  const rs = guards.filter(Boolean).map((g) => pairedGuardReadout(rows, g,
    typeof BIMODAL_GUARDS === 'undefined' ? [] : BIMODAL_GUARDS));
  if (rs.length < 2) return { ...(rs[0] || { count: 0, score: null, sign_p: 1 }), min: rs[0] ? rs[0].score : null };
  const s = rs.map((r) => Number(r.score));
  return { count: Math.min(...rs.map((r) => r.count)), sign_p: Math.max(...rs.map((r) => r.sign_p)),
    score: s.every((v) => v > 0) ? Math.exp(s.reduce((a, v) => a + Math.log(v), 0) / s.length) : null,
    min: Math.min(...s) };
}

function megaFinalSelectionVerdict(selection, finalists, opts) {
  const s = selection || {};
  const o = opts || {};
  const list = Array.isArray(finalists) ? finalists.map(normalizeMegaCandidate) : [];
  const source = list.find((c) => c.id === String(s.selected_candidate_id || ''));
  const row = Array.isArray(s.candidates)
    ? s.candidates.find((c) => String(c.candidate_id || '') === String(s.selected_candidate_id || ''))
    : null;
  const reasons = [];
  const returnedIds = Array.isArray(s.candidates)
    ? s.candidates.map((c) => String(c.candidate_id || '')) : [];
  for (const candidate of list) {
    if (returnedIds.filter((id) => id === candidate.id).length !== 1) {
      reasons.push(`finalist ${candidate.id} does not have exactly one evidence row`);
    }
  }
  if (returnedIds.some((id) => !list.some((c) => c.id === id))) {
    reasons.push('selection contains an evidence row outside the supplied finalists');
  }
  const allGuards = [...new Set([...(o.targetGuards || []), ...(o.regressionGuards || [])])];
  const rowFailures = (r) => {
    const fail = [];
    if (!r || r.claim_complete !== true) return ['incomplete evidence row'];
    if (String(r.status || '').toLowerCase() !== 'verified') fail.push('verification status');
    const registered = list.find((c) => c.id === String(r.candidate_id || ''));
    if (!registered) fail.push('candidate is not registered');
    else {
      if (String(r.source || '') !== registered.source) fail.push('source mismatch');
      if (String(r.tree || '') !== registered.tree) fail.push('tree mismatch');
      if (!sameCommit(r.head, registered.head)) fail.push('head mismatch');
    }
    if (!(Number(r.score) > 1.0)) fail.push('score <= 1.0');
    if (typeof megaFrozenPerCase === 'function' &&
        megaFrozenPerCase(r.per_case, o.frozenBaseline, 0.05).mismatch.length) {
      fail.push('paired base drifted from the frozen baseline');
    }
    if (!String(r.correctness || '').toLowerCase().startsWith('pass')) fail.push('correctness');
    const perCase = new Map((Array.isArray(r.per_case) ? r.per_case : [])
      .map((x) => [String(x.name || ''), Number(x.speedup)]));
    const pairs = Array.isArray(r.paired_readings) ? r.paired_readings : [];
    const nullArm = r.null_arm_pct == null ? null : Number(r.null_arm_pct);
    for (const guard of o.targetGuards || []) {
      const readout = pairedGuardReadout(pairs, guard, BIMODAL_GUARDS);
      const needed = Number((o.requiredPairsByGuard || {})[guard] || o.requiredPairs || 1);
      if (readout.raw_count < needed || readout.count < Math.min(5, needed) ||
          !(Number(readout.score) > 1.0) ||
          readout.sign_p > 0.05 || !Number.isFinite(nullArm) ||
          (Number(readout.score) - 1) * 100 <= Math.abs(nullArm)) {
        fail.push(`target paired result ${guard}`);
      }
      if (readout.bimodal && readout.raw_count < 16) fail.push(`bimodal guard ${guard} needs 16 pairs`);
      if (!(perCase.get(String(guard)) > 1.0) ||
          Math.abs(Number(perCase.get(String(guard))) - Number(readout.score)) >
            Number(o.scoreTolerance || 0.002)) fail.push(`target guard ${guard}`);
    }
    for (const guard of o.regressionGuards || []) {
      const readout = pairedGuardReadout(pairs, guard, BIMODAL_GUARDS);
      const needed = Number((o.requiredPairsByGuard || {})[guard] || o.requiredPairs || 1);
      const fl = typeof RF > 'u' ? 1 : RF[guard] || 1;
      if (readout.raw_count < needed || readout.count < Math.min(5, needed) ||
          !(Number(readout.score) >= fl)) {
        fail.push(`regression paired result ${guard}`);
      }
      if (readout.bimodal && readout.raw_count < 16) fail.push(`bimodal guard ${guard} needs 16 pairs`);
      if (!(perCase.get(String(guard)) >= fl) ||
          Math.abs(Number(perCase.get(String(guard))) - Number(readout.score)) >
            Number(o.scoreTolerance || 0.002)) fail.push(`regression guard ${guard}`);
    }
    const targetScores = (o.targetGuards || []).map((guard) =>
      pairedGuardReadout(pairs, guard, BIMODAL_GUARDS).score)
      .filter((v) => Number.isFinite(Number(v)));
    const recomputedScore = targetScores.length === (o.targetGuards || []).length
      ? Math.exp(targetScores.reduce((sum, value) => sum + Math.log(Number(value)), 0) /
        targetScores.length) : null;
    if (recomputedScore == null || !Number.isFinite(Number(recomputedScore)) ||
        Math.abs(Number(r.score) - Number(recomputedScore)) > Number(o.scoreTolerance || 0.002)) {
      fail.push('headline score does not match paired target readings');
    }
    if (!Number.isFinite(nullArm)) fail.push('null arm');
    if (r.guards_pass !== true) fail.push('guard verdict');
    const worldSize = Number(o.worldSize || r.world_size);
    const marker = String(r.path_marker || '');
    if (!marker || !Number.isInteger(worldSize) || worldSize < 1 ||
        Number(r.path_marker_count) !== worldSize) {
      fail.push('candidate path marker');
    }
    const realizedLaunches = Number(r.launches);
    const baselineLaunches = Number(r.launches_base || o.baselineLaunches);
    const targetLaunches = Number(o.launchTarget || 2);
    // launch_target is an upper bound.
    const launchPass = Number.isInteger(realizedLaunches) && realizedLaunches >= 1 &&
      (o.allowPartialFusion === true || realizedLaunches <= targetLaunches) &&
      realizedLaunches < baselineLaunches;
    if (!launchPass) fail.push('no measured partial/full fusion');
    if (String(r.graph_safe || '').toLowerCase() !== 'pass') fail.push('graph safety');
    const hb = String(r.artifact_hash_base || ''), hc = String(r.artifact_hash_candidate || '');
    if (String(r.artifact_distinct || '').toLowerCase() !== 'yes' || !hb || !hc || hb === hc) {
      fail.push('artifact distinctness');
    }
    const replayRows = Array.isArray(r.replay_results) ? r.replay_results : [];
    for (const guard of allGuards) {
      const replay = replayRows.find((x) => String(x.guard || '') === String(guard));
      const needed = Number(o.requiredReplays || 0);
      if (!replay || Number(replay.count || 0) < needed ||
          String(replay.status || '').toLowerCase() !== 'pass' ||
          String(replay.graph_safe || '').toLowerCase() !== 'pass' ||
          (o.requireArrivalJitter !== false && replay.arrival_jitter !== true)) {
        fail.push(`replay ${guard}`);
      }
    }
    if (Number(r.liveness_replays || 0) < Number(o.requiredReplays || 0)) fail.push('liveness depth');
    if (String(r.accuracy_metric || '') !== String(o.accuracyMetric || '') ||
        r.accuracy_value == null ||
        !Number.isFinite(Number(r.accuracy_value)) ||
        Number(r.accuracy_value) < 0 ||
        Number(r.accuracy_value) >= Number(o.accuracyThreshold)) fail.push('accuracy');
    if (o.requireOverlap) {
      if (String(r.overlap_measured || '').toLowerCase() !== 'yes' ||
          !(Number(r.overlap_fraction) > 0) ||
          !(Number(r.overlap_cu_fraction) > 0) ||
          r.overlap_base_reading == null || r.overlap_forced_reading == null ||
          Math.abs(Number(r.overlap_base_reading)) > 0.05 ||
          !(Number(r.overlap_forced_reading) > 0.10)) fail.push('controlled overlap');
    }
    if (o.requireAttribution) {
      const a = r.attribution || {};
      if (r.attribution_complete !== true || !(Number(a.changed_us) > 0) ||
          !(Number(a.replaced_sum_us) > 0) ||
          a.residual_ms_base == null || a.residual_ms_cand == null ||
          !Number.isFinite(Number(a.residual_ms_base)) ||
          !Number.isFinite(Number(a.residual_ms_cand)) ||
          !(o.targetGuards || []).includes(String(a.guard || '')) ||
          !String(a.method || '') || a.absolute_to_frozen !== true) fail.push('attribution');
    }
    return fail;
  };
  for (const candidate of list) {
    const candidateRow = (Array.isArray(s.candidates) ? s.candidates : [])
      .find((r) => String(r.candidate_id || '') === candidate.id);
    if (!candidateRow || candidateRow.claim_complete !== true) {
      reasons.push(`finalist ${candidate.id} did not complete independent final measurement`);
    }
  }
  if (s.claim_complete !== true) reasons.push('selection claim is incomplete');
  if (!String(s.attempt_id || '') || !String(s.evidence_manifest || '')) {
    reasons.push('selection evidence manifest is missing');
  }
  if (!source) reasons.push('selected id is not a supplied finalist');
  if (!row || row.claim_complete !== true) reasons.push('selected finalist evidence row is incomplete');
  if (source && String(s.selected_source || '') !== source.source) reasons.push('selected source mismatches registry');
  if (source && row && String(row.source || '') !== source.source) reasons.push('finalist row source mismatches registry');
  if (source && row && String(row.tree || '') !== source.tree) reasons.push('finalist row tree mismatches registry');
  if (source && row && !sameCommit(row.head, source.head)) reasons.push('finalist row head mismatches registry');
  if (String(s.selected_tree || '') !== String(o.selectedWorkspace || '')) {
    reasons.push('selected tree is not the controlled materialized workspace');
  }
  if (!String(s.selected_head || '') || !source || !sameCommit(s.materialized_from_head, source.head)) {
    reasons.push('materialized workspace is not bound to the selected candidate head');
  }
  if (!(Number(s.selected_score) > 1.0)) reasons.push('selected score does not beat frozen baseline');
  if (row && Math.abs(Number(row.score) - Number(s.selected_score)) > 1e-9) {
    reasons.push('selected score mismatches finalist evidence row');
  }
  if (row) reasons.push(...rowFailures(row).map((x) => `selected finalist: ${x}`));
  const passedRows = (Array.isArray(s.candidates) ? s.candidates : [])
    .filter((r) => rowFailures(r).length === 0)
    .map((row) => {
      const scores = (o.targetGuards || []).map((guard) =>
        pairedGuardReadout(row.paired_readings, guard, BIMODAL_GUARDS).score);
      return {
        row,
        score: Math.exp(scores.reduce((sum, value) => sum + Math.log(Number(value)), 0) /
          scores.length),
      };
    })
    .sort((a, b) => b.score - a.score);
  if (passedRows.length) {
    const expected = passedRows[0];
    if (String(s.selected_candidate_id || '') !== String(expected.row.candidate_id || '')) {
      reasons.push('selection is not the fastest fully passing finalist under the tie policy');
    }
  } else {
    reasons.push('no finalist evidence row clears the full contract');
  }
  return { pass: reasons.length === 0, reasons, source, row };
}

// The absolute score is frozen-relative.
function megaFrozenPerCase(rows, frozen, tol) {
  const f = new Map((Array.isArray(frozen) ? frozen : []).map((r) =>
    [String(r && r.name || ''), Number(r && (r.latency_ms || r.baseline_ms))]));
  const mismatch = [];
  const perCase = (Array.isArray(rows) ? rows : []).map((r) => {
    const ms = f.get(String(r && r.name || '')), opt = Number(r && r.optimized_ms);
    const base = Number(r && r.baseline_ms);
    if (!(ms > 0 && opt > 0 && base > 0) || Math.abs(base - ms) / ms <= tol) return r;
    mismatch.push({ name: r.name, reported_base_ms: base, frozen_ms: ms,
      deviation_pct: Math.round(1000 * (base - ms) / ms) / 10 });
    return { ...r, baseline_ms: ms, speedup: ms / opt, switch_speedup: Number(r.speedup) };
  });
  return { perCase, mismatch };
}

function megaCandidateFromVerification(meta, ver, opts) {
  const o = opts || {};
  const frozenFix = megaFrozenPerCase((ver || {}).per_case, o.frozenBaseline,
    Number(o.denominatorTolerance || 0.05));
  const denomBad = frozenFix.mismatch.length > 0;
  const v = denomBad ? { ...ver, per_case: frozenFix.perCase } : (ver || {});
  const aggregate = denomBad ? NaN
    : Number(v.verified_weighted != null ? v.verified_weighted : v.verified_geomean);
  const score = promotionScore(v, o.promotionMetric || 'operator_e2e',
    o.targetGuards || [], o.regressionGuards || [], aggregate);
  const launches = Number(v.launch_shape && v.launch_shape.launches_cand);
  const baseLaunches = Number(v.launch_shape && v.launch_shape.launches_base);
  const targetLaunches = Number(o.launchTarget || 2);
  // launch_target is an upper bound; partial fusion lifts it.
  const launchPass = o.allowPartialFusion === true
    ? (Number.isInteger(launches) && Number.isInteger(baseLaunches) &&
       launches >= 1 && launches < baseLaunches)
    : (Number.isInteger(launches) && launches >= 1 && launches <= targetLaunches &&
       (!Number.isInteger(baseLaunches) || launches < baseLaunches));
  const accuracy = Array.isArray(v.accuracy_results) ? v.accuracy_results : [];
  const metricAccuracy = accuracy.filter((a) =>
    !o.accuracyMetric || String(a.metric) === String(o.accuracyMetric));
  const requiredAccuracyCases = Array.isArray(o.requiredAccuracyCases)
    ? o.requiredAccuracyCases.map(String) : [];
  const requiredAccuracy = requiredAccuracyCases.length
    ? requiredAccuracyCases.map((name) =>
      metricAccuracy.find((a) => {
        const guard = String(a.guard || '');
        return guard === name || guard.startsWith(`${name}_`);
      })).filter(Boolean)
    : metricAccuracy.filter((a) =>
      !(o.targetGuards || []).length || (o.targetGuards || []).includes(String(a.guard)));
  const accuracyPass = requiredAccuracy.length > 0 &&
    (!requiredAccuracyCases.length || requiredAccuracy.length === requiredAccuracyCases.length) &&
    requiredAccuracy.every((a) => Number.isFinite(Number(a.value)) &&
      Number(a.value) >= 0 && Number(a.value) < Number(o.accuracyThreshold || a.threshold));
  const livenessPass = String(v.liveness || '').toLowerCase() === 'pass';
  const guards = guardContract(v, o.targetGuards || [], o.regressionGuards || [], score);
  const targetGuard = (o.targetGuards || []).join(',');
  const targetReadout = megaTargetsReadout(v.paired_readings, o.targetGuards || []);
  const nullArm = v.null_arm_pct == null ? null : Number(v.null_arm_pct);
  const measurementPass = !!targetGuard &&
    targetReadout.count >= Number(o.requiredPairs || 1) &&
    Number.isFinite(Number(targetReadout.score)) &&
    Math.abs(Number(targetReadout.score) - Number(score)) <= Number(o.scoreTolerance || 0.002) &&
    targetReadout.sign_p <= 0.05 &&
    Number.isFinite(nullArm) &&
    (Number(targetReadout.min) - 1) * 100 > Math.abs(nullArm);
  const verificationStatusPass = String(v.status || '').toLowerCase() === 'verified';
  const verificationFailed = v.claim_complete === true && !verificationStatusPass;
  const record = normalizeMegaCandidate({
    ...(meta || {}),
    claim_complete: v.claim_complete === true,
    attempt_id: v.attempt_id || (meta && meta.attempt_id),
    evidence_manifest: v.evidence_manifest || '',
    target_guard: targetGuard,
    absolute_score: score,
    per_case: v.per_case || [],
    paired_readings: v.paired_readings || [],
    denominator_mismatch: frozenFix.mismatch,
    component_breakdown: v.component_breakdown || [],
    hot_path_counters: v.hot_path_counters || {},
    null_arm_pct: nullArm,
    reps: Number(v.reps || targetReadout.count),
    artifact_hash_base: v.artifact_hash_base || '',
    artifact_hash_candidate: v.artifact_hash_candidate || '',
    launches,
    correctness_pass: verificationStatusPass &&
      accuracyPass && String(v.correctness || '').toLowerCase().startsWith('pass'),
    activation_pass: verificationStatusPass &&
      String(v.activation_confirmed || '').toLowerCase() === 'yes' &&
      String(v.activation_on_hardware || '').toLowerCase() === 'yes',
    head_pass: !!String(meta && meta.head || '') &&
      sameCommit(v.candidate_head, meta && meta.head),
    launch_pass: launchPass,
    liveness_pass: livenessPass && Number(v.replay_count || 0) >= Number(o.requiredReplays || 30),
    graph_pass: String(v.graph_safe || '').toLowerCase() === 'pass',
    guards_pass: guards.regression_pass !== false,
    artifact_distinct: String(v.artifact_distinct || '').toLowerCase() === 'yes' ||
      (!!v.artifact_hash_base && !!v.artifact_hash_candidate &&
       v.artifact_hash_base !== v.artifact_hash_candidate),
    measurement_pass: verificationStatusPass && measurementPass,
    runtime_verified: verificationStatusPass && accuracyPass &&
      String(v.activation_on_hardware || '').toLowerCase() === 'yes' &&
      launchPass && livenessPass,
    score_complete: verificationStatusPass && measurementPass,
    verification_status: String(v.status || ''),
    correctness: String(v.correctness || ''),
    gpu_executed: String(v.activation_on_hardware || '').toLowerCase() === 'yes',
    activation_on_hardware: String(v.activation_on_hardware || ''),
    next_blocker: (denomBad ? `DENOMINATOR: paired base deviates from the frozen baseline (${
      frozenFix.mismatch.map((m) => `${m.name} ${m.reported_base_ms}ms vs frozen ${m.frozen_ms}ms, ${
        m.deviation_pct}%`).join('; ')}); score re-based on frozen latency. If the base arm was the frozen tree ` +
      'this is machine drift: re-check GPU state and re-measure, do not edit code for it; if it was the ' +
      'candidate switch-off path, shared code regressed: fix that first. ' : '') + (verificationFailed
      ? `independent Verify ${v.status || 'failed'} at HEAD ${
        v.candidate_head || meta && meta.head || '(missing)'}; correctness=${
        v.correctness || 'unknown'}, activation_on_hardware=${
        v.activation_on_hardware || 'unknown'}; evidence=${
        v.evidence_manifest || '(missing)'}; prior_handoff=${
        meta && meta.next_blocker || '(none)'}`
      : String(meta && meta.next_blocker || '')),
    notes: v.claim_complete === true && v.notes ? String(v.notes) :
      String(meta && meta.notes || ''),
  });
  record.status = megaCandidateHardPass(record) ? 'scored'
    : (record.correctness_pass && record.activation_pass && record.launch_pass ? 'runnable' : 'authoring');
  return record;
}
// <</REPLAY:mega_candidate_registry>>

// <<REPLAY:enabling_step>>
function stepRoleOf(d) {
  const s = String((d && d.step_role) || '').trim().toLowerCase();
  return s === 'enabling' ? 'enabling' : 'terminal';
}

// <</REPLAY:enabling_step>>

const REQUIRE_TASK_GRAPH = !!A.require_task_graph;
let TWO_LAUNCH_REACHED = false;
let AUTONOMY_ACCEPTANCE_REACHED = false;
const ACCEPTANCE_CAVEATS = [];
const OVERLAP_CAVEATS = [];
const LAUNCH_CAVEATS = [];
const GUARD_CAVEATS = [];
const OBJECTIVE_CAVEATS = [];
const ATTRIBUTION_CAVEATS = [];
const CHAIN_CAVEATS = [];
const LADDER_MEASURED = new Set();
// Hardware/toolchain facts this wave established, for a human to merge into knowledge/.
const KNOWLEDGE_DELTA = [];
let TASK_GRAPH_CAVEAT = '';
let PIPE_TABLE_CAVEAT = '';
const dispatchedRungs = new Set();
// "Did taking this rung produce a number" (the set above answers "was it taken").
const rungTally = new Map();
function recordRungOutcome(id, outcome) {
  const key = String(id || '').trim();
  if (!key) return;
  const e = rungTally.get(key) || { attempts: 0, last_outcome: 'never_planned' };
  if (e.last_outcome !== 'measured') e.last_outcome = outcome;
  rungTally.set(key, e);
}

const structuralGuardNames = [...new Set([...TARGET_GUARDS, ...REGRESSION_GUARDS])];
const noGpuFrontMatter = MEGA_STRUCTURAL_ONLY;

let PC_OVERSHOOT = '';
let PC_UNDERSHOOT = '';
// Whether a control actually ran in this wave (a missing switch means two identical arms).
let PC_RAN = false;
let PC_PASSED = false;
let noEvidence = 0;
// Rounds in which the switched path never reached a card, under every objective including
// working_kernel.
let noHardware = 0;
// max_no_hardware (default MAX_NO_IMPROVE): raise it for whole-fusion authoring, whose early rounds
// may crash before launch; a run truly stuck off-hardware still stops.
const MAX_NO_HARDWARE = Math.max(1, parseInt(A.max_no_hardware != null ? A.max_no_hardware : MAX_NO_IMPROVE, 10));
let stopReason = null;
let insightBook = [];
// Candidate shelf: verified non-winners kept as offers; absorbedByRound[r] is the file set CANONICAL
// took in round r, which ages shelved patches out.
let shelf = [];
let absorbedByRound = {};
const SHELF_OFFER_K = Number(A.shelf_offer_k) > 0 ? Number(A.shelf_offer_k) : 2;

// <<REPLAY:memory_merge>>
// Cross-round memory is append-and-age, never replaced: an insight the summariser stops repeating
// survives, every insight keeps its origin round, and evictions from a full board are reported.
function normInsight(s) {
  return String(s == null ? '' : s)
    .replace(/^\[r\d+[^\]]*\]\s*/, '')     // strip a provenance tag if the agent echoed one back
    .replace(/\s+/g, ' ').trim().toLowerCase();
}
function mergeInsights(book, incoming, round, roundVoid, max) {
  const MAX = Number.isFinite(max) ? max : 40;
  const out = book.map((e) => Object.assign({}, e));
  const index = new Map(out.map((e, i) => [normInsight(e.text), i]));
  for (const raw of (Array.isArray(incoming) ? incoming : [])) {
    const text = String(raw == null ? '' : raw).replace(/^\[r\d+[^\]]*\]\s*/, '').trim();
    if (!text) continue;
    const k = normInsight(text);
    if (index.has(k)) {
      // Restating an insight refreshes it, but cannot upgrade one from a round with no evidence.
      const e = out[index.get(k)];
      e.last_round = round; e.restated = (e.restated || 0) + 1;
      if (!roundVoid) e.void_round = e.void_round && false;
      continue;
    }
    out.push({ text, first_round: round, last_round: round, restated: 0, void_round: !!roundVoid });
    index.set(k, out.length - 1);
  }
  if (out.length <= MAX) return { book: out, evicted: [] };
  // Age out by last-seen round, then by how often the finding kept coming back.
  const ranked = out.map((e, i) => ({ e, i })).sort((a, b) =>
    (b.e.last_round - a.e.last_round) || ((b.e.restated || 0) - (a.e.restated || 0)) || (b.i - a.i));
  const keep = new Set(ranked.slice(0, MAX).map((x) => x.i));
  return { book: out.filter((_, i) => keep.has(i)), evicted: out.filter((_, i) => !keep.has(i)) };
}
function renderInsights(book) {
  return book.map((e) => `[r${e.first_round}` +
    (e.last_round !== e.first_round ? `-r${e.last_round}` : '') +
    (e.void_round ? ' FROM-VOID-ROUND' : '') + `] ${e.text}`);
}
// <</REPLAY:memory_merge>>

// <<REPLAY:candidate_shelf>>
// The shelf keeps verified non-winners as offers.
const shelfFile = (s) => String(s == null ? '' : s).replace(/^\.\//, '').replace(/^\/+/, '').trim();
const shelfFiles = (c) => [...new Set((Array.isArray(c.touched_files) ? c.touched_files : [])
  .map(shelfFile).filter(Boolean))];

// <</REPLAY:candidate_shelf>>

// <<REPLAY:bimodal_split>>
  // a gap smaller than this is a spread, not two states.

// Credit comes only from explicit target guards; regression guards are vetoes.
function guardContract(result, targetGuards, regressionGuards, fallbackScore) {
  const rows = Array.isArray(result && result.per_case) ? result.per_case : [];
  const byName = new Map(rows.map((r) => [String(r.name || '').trim(), r]));
  const targets = (Array.isArray(targetGuards) ? targetGuards : []).map(String).filter(Boolean);
  const regressions = (Array.isArray(regressionGuards) ? regressionGuards : []).map(String).filter(Boolean);
  const missingTargets = targets.filter((g) => !byName.has(g));
  const missingRegressions = regressions.filter((g) => !byName.has(g));
  const targetValues = targets.map((g) => Number((byName.get(g) || {}).speedup))
    .filter((v) => Number.isFinite(v) && v > 0);
  const score = targets.length
    ? (targetValues.length === targets.length
      ? Math.exp(targetValues.reduce((s, v) => s + Math.log(v), 0) / targetValues.length)
      : 0)
    : Number(fallbackScore || 0);
  const regressed = regressions.filter((g) => {
    const v = Number((byName.get(g) || {}).speedup);
    return Number.isFinite(v) && v < (typeof RF > 'u' ? 1 : RF[g] || 1);
  });
  return {
    score,
    complete: missingTargets.length === 0 && missingRegressions.length === 0,
    regression_pass: missingRegressions.length === 0 && regressed.length === 0,
    missing_targets: missingTargets,
    missing_regressions: missingRegressions,
    regressed,
  };
}

function promotionScore(result, metric, targetGuards, regressionGuards, fallbackScore) {
  if (!result) return 0;
  if (metric === 'changed_kernel') {
    const a = result.attribution || {};
    const changed = Number(a.changed_us), replaced = Number(a.replaced_sum_us);
    return a.absolute_to_frozen === true &&
      Number.isFinite(changed) && changed > 0 && Number.isFinite(replaced) && replaced > 0
      ? replaced / changed : 0;
  }
  return guardContract(result, targetGuards, regressionGuards, fallbackScore).score;
}
// <</REPLAY:bimodal_split>>

const ATTRIBUTION_METRIC = PROMOTION_METRIC === 'legacy' ? 'changed_kernel' : PROMOTION_METRIC;

// A COMPLETE control that failed the gate is re-measured fresh (bounded); the gate never changes.
async function remeasureMegaCalibration(label, prior) {
  if (megaMeasurementCalibration.ready || !POSITIVE_CONTROL ||
      MEGA_STRUCTURAL_ONLY || megaPcRemeasured >= 2 ||
      !prior || prior.claim_complete !== true) return;
  megaPcRemeasured += 1;
  const r = await agentT(
    roleAgent('benchmark_engineer', 'recalibrate',
      `RE-MEASURE the positive control fresh (attempt ${megaPcRemeasured}/2): the last complete ` +
      `attempt failed the gate. Same gate; never re-gate, drop or reuse pairs.`, {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE,
        ANALYSIS: analysis, POSITIVE_CONTROL, PREVIOUS_CONTROL: prior,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      }),
    { phase: 'Benchmark', label: `mega recalibration ${label}`, schema: MEGA_BENCH_SCHEMA,
      ...(MEGA_PRODUCTION ? { timeout_ms: 7200000, timeout_marker: true, max_retries: 1 } : {}) });
  const pc = r && r.positive_control;
  const ok = megaCalibrationClaimPass(pc, POSITIVE_CONTROL);
  log(`MEGA recalibration ${megaPcRemeasured}/2 (${label}): ` + (pc ? `${pc.measured_pct}%, ` +
    `worst null ${Math.max(...(pc.null_pairs_pct || [NaN]).map(Math.abs))}% -> ` : '') +
    (ok ? 'PASS' : 'FAIL'));
  if (ok) {
    megaMeasurementCalibration = {
      ready: true, candidate_id: 'synthetic_control',
      attempt_id: pc.attempt_id || `recalibrated:${label}`,
      evidence_manifest: pc.evidence_manifest || '',
      note: `re-measured synthetic control passed at ${Number(pc.measured_pct).toFixed(2)}% ` +
        `(attempt ${megaPcRemeasured}; earlier failed attempts kept on disk)`,
    };
  }
}

async function recoverMegaCalibration(label) {
  if (megaMeasurementCalibration.ready || !POSITIVE_CONTROL) return;
  const recovered = await agentT(
    roleAgent('benchmark_engineer', 'recover',
      `RECOVER ONLY the completed positive-control aggregate from ${EVAL_DIR}. Do not run a GPU ` +
      `command or take a lease. Ignore force-emitted/partial claims and return claim_complete:false ` +
      `unless the evidence manifest and every pair are final.`, {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
        ANALYSIS: analysis, POSITIVE_CONTROL,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      }),
    { phase: 'Benchmark', label: `mega calibration recovery ${label}`, schema: MEGA_BENCH_SCHEMA,
      ...(MEGA_PRODUCTION ? { timeout_ms: 300000, max_retries: 1 } : {}) });
  const pc = recovered && recovered.positive_control;
  if (megaCalibrationClaimPass(pc, POSITIVE_CONTROL)) {
    megaMeasurementCalibration = {
      ready: true, candidate_id: 'synthetic_control',
      attempt_id: pc.attempt_id || `recovered:${label}`,
      evidence_manifest: pc.evidence_manifest || '',
      note: `recovered completed synthetic control at ${Number(pc.measured_pct).toFixed(2)}%`,
    };
    log(`MEGA measurement calibration RECOVERED (${label}): ${megaMeasurementCalibration.note}.`);
  } else await remeasureMegaCalibration(label, pc);
}

async function persistMegaCandidateState(currentRound, finalizing) {
  if (!STATE_DIR) return true;
  const w = megaStateWrite(currentRound, finalizing);
  return megaCheckStateWrite(currentRound, w,
    await agentT(roleAgent('tech_lead', 'update_memory', w.task, w.inputs), w.opts));
}

// mode=mega memory step: the tech_lead update_memory call writes the candidate registry to STATE.json.
function megaStateWrite(currentRound, finalizing) {
  const sequence = megaStateSequenceBase + Number(currentRound) * 10 + (finalizing ? 9 : 0);
  const generation = `mega:${currentRound}:${megaCandidateRegistry.reduce(
    (m, c) => Math.max(m, Number(c.attempts || 0)), 0)}:${sequence}`;
  return {
    sequence, generation,
    task: 'Persist the mega candidate registry exactly; do not benchmark, plan, or modify candidate source.',
    inputs: {
        STATE_DIR, CANONICAL, ROUND: currentRound,
        CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase,
        CANDIDATE_REGISTRY: megaCandidateRegistry,
        MEASUREMENT_CALIBRATION: megaMeasurementCalibration,
        STATE_GENERATION: generation,
        STATE_SEQUENCE: sequence,
        PRIOR_HISTORY: history, OPEN_RUNGS: [], ROADMAP_LADDER: [],
        LADDER_DISPATCHED: [], LADDER_COMPLETED: [],
        SHELF: shelf, ABSORBED_FILES: absorbedByRound,
      },
    opts: { phase: 'Optimize', label: `mega candidate state r${currentRound}`, schema: MEGA_MEMORY_SCHEMA,
      // Large late-round registries need a real-time hung guard wider than the modeled turn clock.
      ...(MEGA_PRODUCTION ? { timeout_ms: 600000, max_retries: 1 } : {}) },
  };
}

// Checks the STATE write's echo; on a lost echo reads STATE.json back and retries the write once.
async function megaCheckStateWrite(currentRound, w, persisted) {
  const { sequence, generation } = w;
  const echoOk = (r) => Boolean(r && r.state_written === true &&
    Number(r.state_round) === Number(currentRound) &&
    Number(r.state_sequence) === sequence &&
    String(r.state_generation || '') === generation);
  if (!echoOk(persisted)) {
    // Distinguish a landed write with a lost echo from a genuinely missing write.
    const readState = (labelSuffix) => agentT(
      roleAgent('tech_lead', 'update_memory',
        'READ-ONLY verification: read STATE.json in STATE_DIR and report its state_written (true iff ' +
        'the file exists and parses as JSON), state_round (its last_round), state_sequence, and ' +
        'state_generation EXACTLY as stored. Do NOT write, benchmark, plan, or modify anything.', {
          STATE_DIR, ROUND: currentRound,
          EXPECT_SEQUENCE: sequence, EXPECT_GENERATION: generation,
        }),
      { phase: 'Optimize', label: `mega state verify r${currentRound}${labelSuffix || ''}`, schema: MEGA_MEMORY_SCHEMA,
        ...(MEGA_PRODUCTION ? { timeout_ms: 180000, max_retries: 1 } : {}) });
    let verify = await readState('');
    if (!echoOk(verify)) {
      // Genuinely-missing write. Retry the WRITE once with a larger cap (registry grown by round 4+).
      const retry = await agentT(
        roleAgent('tech_lead', 'update_memory',
          'RETRY persist: the prior attempt did not land STATE.json on disk. Persist the mega ' +
          'candidate registry exactly and WRITE STATE.json now; do not benchmark, plan, or modify ' +
          'candidate source.', {
            STATE_DIR, CANONICAL, ROUND: currentRound,
            CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase,
            CANDIDATE_REGISTRY: megaCandidateRegistry,
            MEASUREMENT_CALIBRATION: megaMeasurementCalibration,
            STATE_GENERATION: generation, STATE_SEQUENCE: sequence,
            PRIOR_HISTORY: history, OPEN_RUNGS: [], ROADMAP_LADDER: [],
            LADDER_DISPATCHED: [], LADDER_COMPLETED: [],
            SHELF: shelf, ABSORBED_FILES: absorbedByRound,
          }),
        { phase: 'Optimize', label: `mega candidate state r${currentRound} retry`, schema: MEGA_MEMORY_SCHEMA,
          ...(MEGA_PRODUCTION ? { timeout_ms: 600000, max_retries: 1 } : {}) });
      if (!echoOk(retry)) verify = await readState(' recheck');
      if (!echoOk(retry) && !echoOk(verify)) {
        // In-memory-only checkpoint is not resumable. Stop before a round advances from stale state (lane git commit stays intact).
        log(`MEGA STATE PERSIST FAILED r${currentRound}: STATE.json did not advance to seq=${sequence} ` +
          `after retry.`);
        return false;
      }
    }
    log(`MEGA state persist r${currentRound}: primary echo was incomplete, but STATE.json on disk ` +
      `verifies (seq=${sequence}, gen=${generation}); lineage intact, continuing.`);
  }
  return true;
}

function megaCandidateById(id) {
  return megaCandidateRegistry.find((c) => c.id === id) || null;
}

function megaLaneTree(candidateId) {
  if (!validMegaCandidateId(candidateId)) {
    throw new Error(`invalid mega candidate id '${candidateId}'`);
  }
  const prior = megaCandidateById(candidateId);
  if (prior && prior.tree) return prior.tree;
  return STATE_DIR
    ? `${STATE_DIR}/candidates/${candidateId}/tree`
    : `${EVAL_DIR}/candidate_lanes/${candidateId}`;
}

function megaBaseTree(baseId, requesterSource) {
  if (!baseId || baseId === 'frozen_baseline') return `${EVAL_DIR}/baseline`;
  const base = megaCandidateById(baseId);
  return base && base.tree && base.status !== 'rejected' &&
    (!(requesterSource === 'search' || requesterSource === 'integrated') ||
      base.source === 'search' || base.source === 'integrated') &&
    (base.status === 'scored' || base.status === 'finalist') &&
    base.id_valid && base.source_valid && base.source_id_valid
    ? base.tree : `${EVAL_DIR}/baseline`;
}

function megaBaseHead(baseId, requesterSource) {
  if (!baseId || baseId === 'frozen_baseline') return '';
  const base = megaCandidateById(baseId);
  return base &&
    (!(requesterSource === 'search' || requesterSource === 'integrated') ||
      base.source === 'search' || base.source === 'integrated') &&
    (base.status === 'scored' || base.status === 'finalist')
    ? String(base.head || '') : '';
}

function validMegaSearchDirection(direction) {
  const d = direction || {};
  const identity = `${d.candidate_id || d.id || ''} ${d.title || ''}`.toLowerCase();
  // Reject only explicit diagnostic-only work, never generic "control" substrings.
  if (/\b(?:no[-_ ]?payload|diagnostic[-_ ]only|instrument(?:ation)?[-_ ]only|meter(?:ing)?[-_ ]only|control[-_ ]only|profil(?:e|ing)[-_ ]only)\b/.test(identity)) {
    return false;
  }
  return true;
}

function freshMegaDirectionLeak(direction) {
  const d = direction || {};
  const text = [
    d.title, d.prompt, d.fills_hole, d.enables,
  ].map((value) => String(value || '')).join('\n');
  const externalArtifactPath = /(?:^|[\s"'`])\/[^\s"'`]*(?:geak_state|geak_runs)\//i.test(text);
  // A negation ("do not read X, prior lanes/state") covers its clause; a ", continue ..." ends it.
  const asserted = text.replace(/\b(?:no|not|never|without|forbid\w*|avoid\w*)\b[^.;\n]{0,240}/gi,
    (clause) => {
      const cut = clause.search(/,\s*(?:and\s+|then\s+|but\s+)?(?:continue|resume|reconcile|reuse|build on|start from)\b/i);
      return ' ' + (cut >= 0 ? clause.slice(cut) : '');
    });
  const priorArtifactClaim =
    /\b(?:continue|resume|reconcile|existing|prior|stale)\b[\s\S]{0,96}\b(?:candidate|lane|checkpoint|head|state|revision)\b/i
      .test(asserted);
  return externalArtifactPath || priorArtifactClaim;
}

function freshMegaAuthorLeak(result) {
  const r = result || {};
  const text = [r.notes, r.next_blocker].map((value) => String(value || '')).join('\n')
    .replace(/\bcop(?:y|ied|ying)\b[\s\S]{0,64}\bfrozen[_ -]?baseline\b/ig, '')
    .replace(/\b(?:no|not|never|without)\b[^.;\n]{0,160}/gi, ' ');
  // Reproducing the pinned Skill or the current plan is authoring; only prior-candidate reuse leaks.
  return /\b(?:reproduc(?:e|ed|ing)|cop(?:y|ied|ying)|reus(?:e|ed|ing))\b[\s\S]{0,128}\b(?:prior|previous|v\d+|candidate (?:source|lineage))\b/i
    .test(text);
}

function safeVerifyTimeoutRecovery(result) {
  const r = result || {};
  return r.timeout_recovery_safe === true &&
    Number(r.active_gpu_processes || 0) === 0 &&
    r.lane_lock_free === true;
}

function megaWipContinuationDirection(currentRound) {
  const lane = megaCandidateRegistry.find((c) =>
    (c.source === 'search' || c.source === 'integrated') &&
    (c.status === 'authoring' || c.status === 'runnable'));
  if (!lane) return null;
  // Resume from the working checkpoint; the top-level record can hold an older (empty) topology.
  const e = typeof megaEffectiveLaneCheckpoint === 'function' ? megaEffectiveLaneCheckpoint(lane) : lane;
  const wip = { ...lane,
    topology: e.topology && Object.keys(e.topology).length ? e.topology : (lane.topology || {}),
    contract_failures: Array.isArray(e.contract_failures) ? e.contract_failures : [],
    next_blocker: e.next_blocker || lane.next_blocker };
  const sem = wip.contract_failures.filter((f) => f && f.tier !== 'surface');
  const surf = wip.contract_failures.filter((f) => f && f.tier === 'surface');
  const target = analysis && analysis.mega_plan_ir && analysis.mega_plan_ir.target || {};
  const wipLaunches = topologyLaunchCount(wip.topology);
  const reachesTarget = Number.isFinite(wipLaunches) &&
    wipLaunches === Number(target.launch_count);
    return {
    id: `r${currentRound}_continue_${wip.id}`, candidate_id: wip.id,
    candidate_source: wip.source, base_candidate_id: wip.base_id,
    title: `continue candidate ${wip.id}`, specialty: MEGA_DEFAULT_SPECIALTY,
    tree: wip.tree,
    target_topology: {
      launch_count: wipLaunches,
      included_regions: Array.isArray(wip.topology.included_regions)
        ? wip.topology.included_regions
        : (Array.isArray(wip.topology.fused_stages) ? wip.topology.fused_stages : []),
      included_queues: Array.isArray(wip.topology.included_queues)
        ? wip.topology.included_queues : (reachesTarget ? target.required_queues || [] : []),
      capabilities: Array.isArray(wip.topology.capabilities)
        ? wip.topology.capabilities
        : (reachesTarget ? target.required_capabilities || [] : []),
      parameters: wip.topology.parameters || {},
      require_overlap: wip.topology.require_overlap === true,
    },
    contract_failures: wip.contract_failures,
    prompt: (sem.length
      ? `Resolve required semantic contract failures: ${JSON.stringify(sem)}`
      : (wip.next_blocker || 'Continue the first unresolved measured blocker.')) +
      (surf.length ? ` Surface-tier spelling hints (never outrank runtime evidence): ${
        surf.map((f) => f.id).join(', ')}.` : ''),
  };
}

function bindMegaDirectionToWip(currentRound, direction) {
  const continuation = megaWipContinuationDirection(currentRound);
  if (!continuation) return direction;
  const planned = direction || {};
  // WIP identity is controller-owned: a Planner may not rename the lane or rebase an authoring WIP (silent frozen_baseline fallback).
  const ask = String(planned.prompt || '');
  const other = String(planned.candidate_id || '');
  const keepAsk = ask && !(other && other !== continuation.candidate_id && ask.includes(other)) &&
    !/\b(?:restart|recreate|re-create|rebase|from scratch)\b/i.test(ask);
  return {
    ...planned,
    ...continuation,
    id: planned.id || continuation.id,
    title: planned.title || continuation.title,
    specialty: planned.specialty || continuation.specialty,
    ...(planned.roadmap_rung ? { roadmap_rung: planned.roadmap_rung } : {}),
    ...(planned.step_role ? { step_role: planned.step_role } : {}),
    ...(Array.isArray(planned.gated_on) ? { gated_on: planned.gated_on } : {}),
    ...(Array.isArray(planned.focus_files) ? { focus_files: planned.focus_files } : {}),
    ...(planned.target_shape ? { target_shape: planned.target_shape } : {}),
    ...(planned.rewind_head ? { rewind_head: planned.rewind_head,
      rewind_reason: planned.rewind_reason } : {}),
    // Keep the Planner's task (unless it renames the lane/base) and, for the same lane, its topology.
    prompt: keepAsk ? `${ask}\n\nLANE STATE (${continuation.candidate_id}): ${continuation.prompt}`
      : continuation.prompt,
    target_topology: other === continuation.candidate_id && planned.target_topology ||
      continuation.target_topology,
  };
}

// mode=mega inputs of the lane's tech_lead plan_round call.
function megaPlanPrompt(currentRound, remaining, pool) {
  const searchHistory = megaHistoryForSearch(history, megaCandidateRegistry);
  return roleAgent('tech_lead', 'plan_round',
      'Choose one whole-kernel candidate direction. Expert knowledge guides the common candidate lifecycle; it never creates a reserved lane.', {
        ...(pool ? { GPU_POOL: pool, GPU_MIN_FREE_GIB } : {}),
        EVAL_DIR, ROUND: currentRound, BUDGET_REMAINING: remaining,
        CUMULATIVE_SPEEDUP: cumulative, BASELINE_GEOMEAN_MS,
        SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary, ...GMI, ...MRI,
        CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: searchHistory,
        MEGA_CANDIDATE_REGISTRY: megaRegistryForSearch(megaCandidateRegistry),
        MEASUREMENT_CALIBRATION: megaMeasurementCalibration,
        MEGA_STALL: { rounds: megaStall, limit: MEGA_STALL_LIMIT },
        ...(MEGA_DIRECTIONS > 1 ? { MAX_DIRECTIONS: MEGA_DIRECTIONS } : {}),
        ...(megaRefusal ? { PREVIOUS_PLAN_REFUSED: megaRefusal.reason } : {}),
        STRUCTURAL_CONTRACT_POLICY: STRUCTURAL_CONTRACT_BLOCKING ? 'blocking' : 'advisory',
        MEGA_PROFILE, CANDIDATE_TIMEOUT_S: MEGA_CANDIDATE_TIMEOUT_S,
        ...(MEGA_STRUCTURAL_ONLY ? {
          STRUCTURAL_ONLY: '1',
          STRUCTURAL_TARGET: 'complete the matched Expert Skill contract and MegaPlanIR target',
        } : {}),
        DEFAULT_BASE_CANDIDATE: (selectMegaSearchParent(megaCandidateRegistry) || {}).id ||
          'frozen_baseline',
        ROADMAP: `${EVAL_DIR}/roadmap.md`, ROADMAP_LADDER: LADDER,
        LADDER_DISPATCHED: [...dispatchedRungs], LADDER_COMPLETED: [...LADDER_MEASURED],
        OPEN_RUNGS: openRungs(LADDER, rungTally),
        ...(analysis && analysis.task_graph
          ? { TASK_GRAPH: analysis.task_graph } : {}),
        ...(analysis && analysis.resource_timeline
          ? { RESOURCE_TIMELINE: analysis.resource_timeline } : {}),
        ...(analysis && analysis.mega_plan_ir
          ? { MEGA_PLAN_IR: analysis.mega_plan_ir } : {}),
        BASELINE_OPERATOR_MAP: analysis && analysis.baseline_operator_map || [],
        ...(USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE ? {
          EXPERT_SKILL_PLANNER_EXTENSION: EXPERT_SKILL_PLANNER_GUIDE_FILE,
          EXPERT_SKILL_ID,
          EXPERT_SKILL_REVISION,
          EXPERT_SKILL_BUNDLE_TOOL,
          EXPERT_SKILL_BUNDLE_SHA256,
          EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
          EXPERT_SKILL_CONTRACT_SHA256,
        } : {}),
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, LAUNCH_TARGET,
        REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
        REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        KERNEL_KNOWLEDGE_DIR: '', KK_OPERATOR: '', KK_LANGUAGE: '', KK_REFS: [], ...KB_INPUTS,
      });
}

function megaPlanOpts(currentRound) {
  return { phase: 'Optimize', label: `mega:plan r${currentRound}`, schema: MEGA_PLAN_SCHEMA,
    // Late-round plans read a large registry and ran 6-7 min; a 5 min guard nulled them.
    ...(MEGA_PRODUCTION ? { timeout_ms: 900000, max_retries: 1 } : {}) };
}

// The planned directions, each bound to a candidate lane. A plan with no direction continues the WIP lane.
function megaPlannedDirections(currentRound, remaining, plan) {
  const noPlan = !plan || plan.stop || !Array.isArray(plan.directions) || !plan.directions.length;
  const wipDir = noPlan ? megaWipContinuationDirection(currentRound) : null;
  // A planner that returned nothing never decided to stop: optimize the best scored lane.
  const best = noPlan && !wipDir && !plan ? selectMegaSearchParent(megaCandidateRegistry) : null;
  if (noPlan && !best) return wipDir ? [wipDir] : [];
  const raws = best ? [{ id: `r${currentRound}_child_${best.id}`, candidate_id: best.id,
    candidate_source: best.source, base_candidate_id: best.id, title: `optimize ${best.id}`,
    specialty: MEGA_DEFAULT_SPECIALTY, step_role: 'terminal', focus_files: [],
    roadmap_rung: 'off_ladder', target_topology: best.topology,
    prompt: best.next_blocker || 'Continue the first unresolved measured blocker.' }]
    : plan.directions.slice(0, Math.max(1, Math.min(MEGA_DIRECTIONS, remaining)));
  // The WIP lane takes the slot naming it, else the last one, so the planner's first choice still runs.
  let bindAt = 0;
  const wip = raws.length > 1 ? megaWipContinuationDirection(currentRound) : null;
  if (wip) {
    const named = raws.findIndex((p) => String(p && p.candidate_id || '') === wip.candidate_id);
    const parks = raws.some((p) => p && String(p.park_candidate_id || '') === wip.candidate_id &&
      String(p.park_reason || '').trim());
    bindAt = parks ? -1 : named >= 0 ? named : raws.length - 1;
  }
  const out = [];
  for (const [i, planned] of raws.entries()) {
    const d = normalizePlannedMegaDirection(currentRound, planned, i !== bindAt);
    if (!d) continue;
    if (out.some((o) => o.candidate_id === d.candidate_id)) {
      log(`Mega round ${currentRound}: dropped a second direction for lane ${d.candidate_id}; ` +
        'one writer per lane per round.');
      continue;
    }
    out.push(d);
  }
  return out;
}

// One planned direction -> a lane assignment (leak check, parking, WIP binding, id, base, topology), or
// null. `extra` marks a direction that is not rebound to the active WIP lane.
function normalizePlannedMegaDirection(currentRound, planned, extra) {
  let raw = planned;
  const freshLane = !MEGA_RESUME_STATE && megaCandidateRegistry.length === 0;
  if (freshLane && freshMegaDirectionLeak(raw)) {
    const requestedId = String(raw && (raw.candidate_id || raw.id) || '');
    const analyzed = LADDER.find((rung) =>
      String(rung && (rung.candidate_id || rung.id) || '') === requestedId
    ) || LADDER[0];
    if (!analyzed || freshMegaDirectionLeak(analyzed)) {
      log('Mega fresh-state plan rejected: Planner referenced prior/external candidate artifacts ' +
        'and no clean Analyze direction is available.');
      return null;
    }
    log('Mega fresh-state plan ignored a prior/external candidate reference and restored the ' +
      'clean Analyze direction from the frozen baseline.');
    raw = {
      ...analyzed,
      candidate_source: 'search',
      base_candidate_id: 'frozen_baseline',
    };
  }
  // Parking freezes (never deletes) the active lane so a structurally different lane can start.
  const parkId = String(raw.park_candidate_id || '');
  const parked = parkId && String(raw.park_reason || '').trim() && megaCandidateById(parkId);
  const parkedNow = !!parked && ['authoring', 'runnable'].includes(parked.status) &&
    String(raw.candidate_id || '') !== parkId;
  if (parkedNow) {
    megaCandidateRegistry = megaCandidateRegistry.map((c) => c.id !== parkId ? c
      : normalizeMegaCandidate({ ...c, status: 'parked',
        notes: `parked r${currentRound}: ${raw.park_reason}. ${c.notes || ''}` }));
    log(`Mega round ${currentRound}: Planner parked lane ${parkId} (${raw.park_reason}); resumable.`);
  }
  const plannerCandidateId = String(raw.candidate_id || '');
  const bound = parkedNow || extra ? raw : bindMegaDirectionToWip(currentRound, raw);
  if (bound !== raw) {
    if (plannerCandidateId && plannerCandidateId !== bound.candidate_id) {
      log(`Mega round ${currentRound}: Planner requested replacement lane ${plannerCandidateId}; ` +
        `continuing recoverable WIP ${bound.candidate_id} in place instead.`);
    }
    raw = bound;
  }
  if (!validMegaSearchDirection(raw)) {
    log(`Mega search plan rejected diagnostic-only direction ` +
      `${String(raw.candidate_id || raw.id || raw.title || '(unnamed)')}; a search budget unit must ` +
      `author or optimize a selectable whole-kernel candidate.`);
    const continuation = megaWipContinuationDirection(currentRound);
    if (continuation) {
      log(`Mega round ${currentRound}: continuing resumable WIP lane ${continuation.candidate_id} ` +
        `inside this invocation instead of terminating or opening another workflow.`);
      raw = continuation;
    } else {
      raw = {
        id: `search_whole_r${currentRound}`,
        candidate_id: `search_whole_r${currentRound}`,
        candidate_source: 'search',
        base_candidate_id: 'frozen_baseline',
        title: 'autonomous runnable Mega-kernel candidate',
        specialty: MEGA_DEFAULT_SPECIALTY,
        step_role: 'terminal',
        focus_files: [],
        prompt: 'Author a complete runnable operator candidate from the frozen source/task graph. Full ' +
          'or partial fusion is valid if paired rank-max performance beats the frozen baseline. ' +
          'Diagnostics may be temporary inside this turn but cannot be its output.',
      };
    }
  }
  const requestedId = String(raw.candidate_id || raw.id || `search_r${currentRound}`);
  const safeRequestedId = validMegaCandidateId(requestedId)
    ? requestedId : `search_r${currentRound}`;
  const requested = megaCandidateById(safeRequestedId);
  const candidateId = requested && (requested.status === 'scored' || requested.status === 'finalist')
    ? megaChildCandidateId(safeRequestedId, currentRound) : safeRequestedId;
  const requestedBaseId = String(raw.base_candidate_id ||
    (candidateId !== safeRequestedId ? safeRequestedId : '') ||
    ((selectMegaSearchParent(megaCandidateRegistry) || {}).id || 'frozen_baseline'));
  const requestedBase = megaCandidateById(requestedBaseId);
  const baseId = requestedBaseId === 'frozen_baseline' ||
    (requestedBase &&
      (requestedBase.source === 'search' || requestedBase.source === 'integrated') &&
      (requestedBase.status === 'scored' || requestedBase.status === 'finalist'))
    ? requestedBaseId : 'frozen_baseline';
  const targetShape = raw.target_shape || {};
  const planIr = analysis && analysis.mega_plan_ir || {};
  const targetTopology = raw.target_topology || {
    launch_count: Number.isFinite(Number(targetShape.launches))
      ? Number(targetShape.launches)
      : Number(planIr.target && planIr.target.launch_count || LAUNCH_TARGET),
    included_regions: Array.isArray(targetShape.stages_fused)
      ? targetShape.stages_fused.map(String)
      : (Array.isArray(planIr.regions) ? planIr.regions.map((r) => String(r.id || '')).filter(Boolean) : []),
    included_queues: Array.isArray(planIr.queues)
      ? planIr.queues.map((queue) => String(queue.id || '')).filter(Boolean) : [],
    capabilities: [],
    require_overlap: targetShape.require_overlap === true || REQUIRE_OVERLAP,
  };
  return {
    ...enrichDirection(raw, LADDER),
    id: raw.id || candidateId,
    candidate_id: candidateId,
    candidate_source: raw.candidate_source === 'integrated' ? 'integrated' : 'search',
    base_candidate_id: baseId,
    tree: megaLaneTree(candidateId),
    // The topology descriptor is the lowerable whole-kernel contract shared by Author and Verify.
    target_topology: targetTopology,
    topology_inferred: !raw.target_topology,
  };
}

// The direction contract of one round: up to MEGA_DIRECTIONS directions, each on its own candidate lane.
function megaPlanRound(currentRound, remaining, plan) {
  const planned = megaPlannedDirections(currentRound, remaining, plan);
  if (!planned.length) return { stop: true, reason: 'no mega candidate direction was planned' };
  const directions = [];
  let refusal = '';
  for (const d of planned) {
    if (analysis && analysis.resource_timeline) {
      const pg = pipeOccupancyGate(analysis.resource_timeline, [d]);
      if (pg.summary) log(`Mega round ${currentRound}: ${pg.summary}`);
      if (pg.caveat) log(`Mega round ${currentRound}: ${pg.caveat}`);
    }
    const lg = roadmapLadderGate(LADDER, [d], LADDER_MEASURED);
    log(`Mega round ${currentRound}: ${lg.summary}`);
    if (lg.caveat) log(`Mega round ${currentRound}: ${lg.caveat}`);
    const topologyVerdict = megaTopologyVerdict(
      d, analysis && analysis.mega_plan_ir, ALLOW_PARTIAL_FUSION);
    if (!topologyVerdict.pass) {
      log(`Mega round ${currentRound}: TOPOLOGY CONTRACT: ${topologyVerdict.reason}`);
      refusal = `Mega direction refused before authoring: ${topologyVerdict.reason}`;
      continue;
    }
    for (const rung of lg.planned) {
      dispatchedRungs.add(rung);
      const tally = rungTally.get(rung) || { attempts: 0, last_outcome: 'never_planned' };
      tally.attempts += 1;
      if (tally.last_outcome !== 'measured') tally.last_outcome = 'unmeasured';
      rungTally.set(rung, tally);
    }
    directions.push(d);
  }
  if (!directions.length) return { refused: true, reason: refusal };
  return { directions, refusal };
}

// mode=mega engineer step for one candidate lane: the engineer call's prompt/options and `after` (-> verifyStep).
// Lanes run concurrently, each on its own modeled clock (`clock.ms`); the round charges the longest.
let megaRoundStartMs = 0;
function megaEngineerTurn(currentRound, d) {
  const clock = { ms: megaRoundStartMs };
  const megaNowMs = () => WORKFLOW_STARTED_MS + clock.ms;
  const megaAdvanceMs = (ms) => { clock.ms += Math.max(0, Number(ms) || 0); };
  const megaChargeMs = megaAdvanceMs;
  const turnStartedMs = megaNowMs();
  const dispatchDeadlineMs = WORKFLOW_STARTED_MS +
    (MEGA_TIME_BUDGET_S - MEGA_FINAL_RESERVE_S) * 1000;
  const candidateId = d.candidate_id;
  const source = d.candidate_source;
  if (!['search', 'integrated'].includes(source)) {
    return { result: { d, clock, stop: true, reason: `invalid Mega candidate source '${String(source)}'; Expert Skills ` +
      'must use the common search/integrated lifecycle' } };
  }
  const existing = megaCandidateById(candidateId);
  if (existing && existing.status === 'parked') {
    megaCandidateRegistry = megaCandidateRegistry.map((c) => c.id !== candidateId ? c
      : normalizeMegaCandidate({ ...c, status: 'authoring' }));
    log(`Mega round ${currentRound}: resuming parked lane ${candidateId}.`);
  }
  megaAdvanceMs(MEGA_PREP_MODEL_MS);
  const dispatchRemainingS = (dispatchDeadlineMs - megaNowMs()) / 1000;
  if (MEGA_PRODUCTION && dispatchRemainingS < 900) {
    return { result: {
      d, clock, stop: true,
      reason: `only ${Math.max(0, Math.round(dispatchRemainingS))}s remain before the production ` +
        `final-validation reserve; no new candidate attempt was started`,
    } };
  }
  const requestedTurnS = MEGA_CANDIDATE_TIMEOUT_S;
  const turnDeadlineMs = MEGA_PRODUCTION
    ? Math.min(turnStartedMs + requestedTurnS * 1000, dispatchDeadlineMs)
    : turnStartedMs + requestedTurnS * 1000;
  const turnBudgetS = (turnDeadlineMs - turnStartedMs) / 1000;
  const prepElapsedS = (megaNowMs() - turnStartedMs) / 1000;
  const availableAfterPrepS = (turnDeadlineMs - megaNowMs()) / 1000;
  if (MEGA_PRODUCTION && availableAfterPrepS < 900) {
    return { result: {
      d, clock, stop: true,
      reason: `candidate preparation consumed ${Math.round(prepElapsedS)}s; only ` +
        `${Math.max(0, Math.round(availableAfterPrepS))}s remain in the shared turn budget, so no ` +
        `Engineer was started`,
    } };
  }
  const gpuCandidateDevelopment = !MEGA_STRUCTURAL_ONLY &&
    EXPERT_SKILL_USAGE === 'candidate_validation';
  const engineerBudgetS = MEGA_PRODUCTION
    ? Math.max(240, Math.min(
      Math.floor(turnBudgetS * (gpuCandidateDevelopment ? 0.85 : 0.60)),
      Math.floor(availableAfterPrepS - Math.max(360, 60 + MEGA_POST_AUTHOR_RESERVE_S))))
    : MEGA_CANDIDATE_TIMEOUT_S;
  const commandBudgetS = Math.max(180, engineerBudgetS - 60);

  const baseCandidateId = (existing && existing.base_id) || d.base_candidate_id || 'frozen_baseline';
  const attempts = (existing ? existing.attempts : 0) + 1;
  const attemptId = `${candidateId}:r${currentRound}:a${attempts}`;
  const tree = d.tree || megaLaneTree(candidateId);
  const laneManifest = tree.replace(/\/tree$/, '') + '/lane.json';
  const laneLock = tree.replace(/\/tree$/, '') + '/author.lock';
  const baseTree = megaBaseTree(baseCandidateId, source);
  const baseHead = megaBaseHead(baseCandidateId, source);
  const outDir = `${EVAL_DIR}/round_${currentRound}/candidate_${candidateId}`;
  const persistentJitCache = STATE_DIR
    ? `${STATE_DIR}/jit_cache/${candidateId}` : `${outDir}/jit_cache`;
  const effectivePrior = existing ? megaEffectiveLaneCheckpoint(existing) : null;
  const priorForAgent = existing ? {
    ...existing,
    ...effectivePrior,
    ...(effectivePrior.head !== existing.head ? {
      verification_status: '', correctness: 'pending', gpu_executed: false,
      activation_on_hardware: 'unknown', runtime_verified: false,
      score_complete: false, absolute_score: null,
    } : {}),
  } : null;
  const candidateClaimsSkillTarget =
    candidateClaimsPlanTarget(d, analysis && analysis.mega_plan_ir);
  const authorPreflightRequired = authoringContractNeedsPreflight(
    d, analysis && analysis.mega_plan_ir, existing, STRUCTURAL_CONTRACT_BLOCKING, {
      skillId: EXPERT_SKILL_ID,
      revision: EXPERT_SKILL_REVISION,
      bundle: EXPERT_SKILL_BUNDLE_SHA256,
      planner: EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
      contract: EXPERT_SKILL_CONTRACT_SHA256,
    }
  );
  const stagedGpuAuthoring = authorPreflightRequired && gpuCandidateDevelopment;
  if (authorPreflightRequired) {
    log(`Mega round ${currentRound}: full-target candidate ${candidateId} has no current exact-HEAD ` +
      `structural pass; ${stagedGpuAuthoring
        ? 'staged GPU construction/JIT smoke is allowed, but promotion remains blocked'
        : 'authoring is GPU-forbidden until independent contract verification'}.`);
  }
  let eng = null;
  let laneWriterTimedOut = false;
  let laneTimeoutMarker = null;
  const authorGpuProhibited = MEGA_STRUCTURAL_ONLY ||
    (authorPreflightRequired && !stagedGpuAuthoring);
  const role = 'engineer';
  return {
    prompt:
      roleAgent(role, 'optimize',
        `Advance candidate lane ${candidateId}; never edit or replace another candidate lane.`, {
          CANDIDATE_ID: candidateId, CANDIDATE_SOURCE: source,
          BASE_CANDIDATE_ID: baseCandidateId, BASE_TREE: baseTree, BASE_HEAD: baseHead,
          CANDIDATE_TREE: tree, OUTPUT_DIR: outDir, ATTEMPT_ID: attemptId,
          CANDIDATE_PYTHONPATH: tree,
          CANDIDATE_IMPORT_MODULES,
          LANE_MANIFEST: laneManifest, LANE_LOCK: laneLock,
          CANDIDATE_TIMEOUT_S: commandBudgetS, ENGINEER_BUDGET_S: engineerBudgetS,
          PERSISTENT_JIT_CACHE_DIR: persistentJitCache,
          SPECIALTY: d.specialty || MEGA_DEFAULT_SPECIALTY, DIRECTION: d,
          KERNEL_PATH: tree, OP_SPEC, TASK_DIR: KERNEL_PATH_ORIG, COMMANDMENT,
          GPU_ID: authorGpuProhibited
            ? (MEGA_STRUCTURAL_ONLY
              ? 'STRUCTURAL_ONLY_NO_GPU' : 'STRUCTURAL_PREFLIGHT_REQUIRED_NO_GPU')
            : GPU_SINGLE,
          GPUS_PER_JOB: authorGpuProhibited
            ? '0' : String(GPU_RESOURCE.gpusPerJob),
          AUTHOR_GPU_MODE: authorGpuProhibited ? 'forbidden' : 'bounded_smoke',
          ...(MEGA_STALL_LIMIT && megaStall >= MEGA_STALL_LIMIT ? { MEASURE_FIRST: String(megaStall) } : {}),
          ...(/^[0-9a-f]{7,40}$/.test(String(d.rewind_head || ''))
            ? { REWIND_TO: d.rewind_head, REWIND_REASON: String(d.rewind_reason || '') } : {}),
          AUTHORING_CONTRACT_PREFLIGHT_REQUIRED:
            authorPreflightRequired ? '1' : '0',
          STAGED_GPU_AUTHORING: stagedGpuAuthoring ? '1' : '0',
          AUTHORING_STRUCTURAL_EVIDENCE_HEAD:
            effectivePrior && effectivePrior.structural_candidate_head || '',
          FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
          TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, LAUNCH_TARGET,
          ...(MEGA_STRUCTURAL_ONLY ? {
            STRUCTURAL_ONLY: '1',
            STRUCTURAL_TARGET: 'complete the Expert Skill target encoded by MegaPlanIR; no GPU commands',
          } : {}),
          SKILL_DIR: WORKFLOW_DIR,
          KERNEL_KNOWLEDGE_DIR: '',
          ...(USE_EXPERT_SKILLS ? {
            EXPERT_SKILL_ID,
            EXPERT_SKILL_REVISION,
            EXPERT_SKILL_RECIPE: EXPERT_SKILL_AUTHOR_GUIDE_FILE,
          } : {}),
          codebase_context: `${EVAL_DIR}/codebase_context.md`,
          profiling_summary: profileSummary ? profileSummary.summary_path : '',
          baseline_per_case: BASELINE_PER_CASE,
          UT_PATH: KERNEL_PATH_ORIG,
          BASELINE_OPERATOR_MAP: analysis && analysis.baseline_operator_map || [],
          INSIGHTS: megaHistoryForSearch(history, megaCandidateRegistry).insights,
          PRIOR_CANDIDATE: priorForAgent, ...GMI, ...MRI,
        }) +
      '\n\nFollow "Candidate lane procedure" in the mode=mega engineer fragment.',
    opts: { phase: 'Optimize', label: `mega:${source}:${candidateId}`, schema: MEGA_CANDIDATE_SCHEMA,
      timeout_ms: engineerBudgetS * 1000, timeout_marker: true, max_retries: 1 },
    after: async (engReply) => {
      eng = engReply;
      if (eng && eng.__agent_timed_out) {
        megaUnsafeTimeout = eng;
        laneTimeoutMarker = eng;
        laneWriterTimedOut = true;
        log(`Mega round ${currentRound}: candidate author ${eng.label || candidateId} exceeded its ` +
          `deadline. Stopping this workflow invocation because the underlying agent cannot be ` +
          `cancelled and may still hold ${laneLock}.`);
        eng = null;
      }
      if (!eng || eng.claim_complete !== true) {
        // A timed-out writer usually exits within minutes; wait for it instead of ending the wave.
        const drainS = laneWriterTimedOut ? 1500 : 0;
        const laneDir = tree.replace(/\/tree$/, '');
        const recovered = await agentT(
          roleAgent('engineer', 'recover',
            `RECOVER ONLY candidate ${candidateId}: follow "PHASE=recover" in the mode=mega engineer fragment.`, {
              ...(drainS ? { DRAIN_S: drainS, LANE_LOCK: laneLock, LANE_DIR: laneDir } : {}),
              OUT_DIR: outDir, OUTPUT_DIR: outDir, CANDIDATE_ID: candidateId,
              CANDIDATE_SOURCE: source, BASE_CANDIDATE_ID: baseCandidateId,
              CANDIDATE_TREE: tree, KERNEL_PATH: tree, ATTEMPT_ID: attemptId,
              SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
            }),
          { phase: 'Optimize', label: `mega:recover:${candidateId}`, schema: MEGA_CANDIDATE_SCHEMA,
            ...(MEGA_PRODUCTION ? { timeout_ms: (180 + drainS) * 1000, max_retries: 1 } : {}) });
        megaAdvanceMs(drainS * 1000);
        // Before the claim branch, else megaUnsafeTimeout survives and skips the finalist.
        if (laneWriterTimedOut && recovered && !recovered.__agent_timed_out &&
            recovered.writer_quiescent === true) {
          laneWriterTimedOut = false;
          // Clear only this lane's marker; a parallel lane may have set its own.
          if (megaUnsafeTimeout === laneTimeoutMarker) megaUnsafeTimeout = null;
          log(`Mega round ${currentRound}: timed-out author of ${candidateId} is quiescent; its ` +
            `committed WIP stays in the lane and the wave continues.`);
        }
        if (recovered && recovered.claim_complete === true) {
          eng = recovered;
          laneWriterTimedOut = false;
        }
      }
      // A new id that builds on this run's own candidate is lineage, not reuse.
      if (!MEGA_RESUME_STATE && !existing && !megaCandidateById(baseCandidateId) &&
          freshMegaAuthorLeak(eng)) {
        return {
          d, clock, stop: true,
          reason: `fresh candidate Author disclosed reuse of a prior candidate/plan; rejecting ${
            candidateId} before structural Verify`,
        };
      }
      megaChargeMs(engineerBudgetS * 1000);

      const reportedChangedFiles = Array.isArray(eng && eng.changed_files)
        ? eng.changed_files.map(String) : [];
      const preVerifyStatus = 'authoring';
      const candidateActivation = normalizeMegaActivation(
        (eng && eng.activation) || (existing && existing.activation)
      );
      let meta = normalizeMegaCandidate({
        id: candidateId, source, base_id: baseCandidateId, tree,
        status: preVerifyStatus,
        head: (eng && eng.head) || (eng && eng.candidate_head) ||
          (existing && existing.working_head) || (existing && existing.head) || '',
        patch: (eng && eng.patch_file) || (existing && existing.patch) || '',
        topology_sig: (eng && eng.topology_sig) || (existing && existing.topology_sig) || '',
        topology: (eng && eng.topology) || d.target_topology ||
          (existing && existing.topology) || {},
        activation: candidateActivation,
        provenance: (eng && eng.provenance) || (existing && existing.provenance) || source,
        claim_complete: false,
        checkpoint_complete: !!(eng && eng.claim_complete === true),
        structural_verified: false,
        runtime_verified: false,
        score_complete: false,
        structural_report: String(eng && eng.structural_report || ''),
        contract_failures: normalizeContractFailures(eng && eng.contract_failures),
        attempt_id: (eng && eng.attempt_id) || attemptId,
        evidence_manifest: String(eng && eng.evidence_manifest || ''),
        attempts,
        changed_files: reportedChangedFiles,
        next_blocker: (eng && eng.next_blocker) || '',
        notes: (eng && eng.notes) || '',
      });
      megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, meta);
      if (laneWriterTimedOut) {
        await persistMegaCandidateState(currentRound);
        return {
          d, clock, stop: true,
          reason: `candidate author timeout for ${candidateId}; invocation stopped to preserve single-writer ` +
            `lineage safety`,
        };
      }

      // mode=mega verify step of the round pipeline: score this lane on the 8-GPU pool.
      const verifyStep = async () => {
        let ver = null;
        let verifierTimedOut = false;
        const turnRemainingS = turnBudgetS - (megaNowMs() - turnStartedMs) / 1000;
        let verifyBudgetS = MEGA_PRODUCTION
          ? Math.max(0, Math.min(Math.floor(turnRemainingS - 60),
            Math.floor((dispatchDeadlineMs - megaNowMs()) / 1000 - 60)))
          : MEGA_CANDIDATE_TIMEOUT_S;
        const expectedHead = String(eng && (eng.head || eng.candidate_head) || '');
        let structural = null;
        let structuralPassThisTurn = false;
        const shouldStructuralVerify = CHECK_EXPERT_SKILL_CONTRACT &&
          eng && eng.claim_complete === true && expectedHead;
        if (shouldStructuralVerify) {
          structural = await agentT(
            roleAgent('verify_engineer', 'verify_structure',
              'GPU-FREE independent post-authoring structure verification. Seal the candidate before ' +
              'reading an optional reference; never edit candidate source and never run a GPU command.', {
                CANDIDATE_ID: candidateId, CANDIDATE_SOURCE: source,
                CANDIDATE_TREE: tree, EXPECTED_HEAD: expectedHead,
                CANDIDATE_IMPORT_MODULES,
                FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
                EXPERT_SKILL_ID,
                EXPERT_SKILL_REVISION,
                EXPERT_SKILL_BUNDLE_TOOL,
                EXPERT_SKILL_BUNDLE_SHA256,
                EXPERT_SKILL_PLANNER_EXTENSION: EXPERT_SKILL_PLANNER_EXTENSION_FILE,
                EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
                EXPERT_SKILL_PLAYBOOK: EXPERT_SKILL_PLAYBOOK_FILE,
                EXPERT_SKILL_CONTRACT: EXPERT_SKILL_CONTRACT_FILE,
                EXPERT_SKILL_CONTRACT_SHA256,
                EXPERT_SKILL_VALIDATION: EXPERT_SKILL_VALIDATION_FILE,
                EXPERT_SKILL_REFERENCE_PATH,
                EXPERT_SKILL_CONTRACT_TOOL,
                STRUCTURAL_VERIFY_DIR: `${outDir}/structure`,
                LANE_LOCK: laneLock,
                REFERENCE_WAS_HIDDEN: EXPERT_SKILL_REFERENCE_PATH ? '1' : 'n/a',
                MEGA_PLAN_IR: analysis && analysis.mega_plan_ir || {},
                SKILL_DIR: WORKFLOW_DIR,
              }),
            { phase: 'Verify', label: `mega:structure:${candidateId}`,
              schema: EXPERT_SKILL_CONTRACT_VERIFY_SCHEMA,
              timeout_ms: 600000, max_retries: 1 });
          const structuralIdentity = structuralEvidenceIdentityVerdict(structural, {
            candidateId,
            candidateHead: expectedHead,
            skillId: EXPERT_SKILL_ID,
            skillRevision: EXPERT_SKILL_REVISION,
            contractSha256: EXPERT_SKILL_CONTRACT_SHA256,
            plannerExtensionSha256: EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
            skillBundleSha256: EXPERT_SKILL_BUNDLE_SHA256,
          });
          const requiredIds = (structural && structural.failed_required_checks || []).map(String);
          const planErrors = (structural && structural.plan_consistency_errors || []).map(String);
          const inputErrors = (structural && structural.input_errors || []).map(String);
          const forbiddenTreeReuse = !MEGA_RESUME_STATE && FORBIDDEN_CANDIDATE_TREE_DIGESTS.has(
            String(structural && structural.candidate_tree_digest || '')
          );
          const authoritativeIds = new Set([
            ...requiredIds, ...planErrors.map((error) => error.split(':', 1)[0]),
          ]);
          const structuralFailures = normalizeContractFailures(
            structural && structural.contract_failures
          ).filter((failure) => authoritativeIds.has(failure.id));
          const structuralPass = !!(structural && structural.claim_complete === true &&
            structural.structural_compatible === true &&
            structural.independent_structure_pass === true &&
            structural.capability_eligible === true &&
            structural.plan_consistent === true &&
            structural.provenance_attestation_valid === true &&
            requiredIds.length === 0 && planErrors.length === 0 && inputErrors.length === 0 &&
            !forbiddenTreeReuse &&
            structural.reference_copy_detected !== true &&
            structural.reference_copy_suspected !== true &&
            structuralIdentity.pass);
          structuralPassThisTurn = structuralPass;
          if (!structuralIdentity.pass) {
            structuralFailures.unshift({
              id: 'structural_evidence_identity',
              category: 'plan',
              severity: 'required',
              messages: structuralIdentity.reasons,
            });
          }
          if (forbiddenTreeReuse) {
            structuralFailures.unshift({
              id: 'fresh_candidate_tree_reuse',
              category: 'plan',
              severity: 'required',
              messages: ['candidate tree digest matches a forbidden prior process artifact'],
            });
          }
          meta = normalizeMegaCandidate({
            ...meta,
            structural_verified: structuralPass,
            structural_report: structural && structural.report_path || '',
            structural_skill_id: structuralPass
              ? String(structural && structural.skill_id || '') : '',
            structural_candidate_head: structuralPass
              ? String(structural && structural.candidate_head || '') : '',
            structural_candidate_tree_digest: structuralPass
              ? String(structural && structural.candidate_tree_digest || '') : '',
            structural_skill_bundle_sha256: structuralPass
              ? String(structural && structural.skill_bundle_sha256 || '') : '',
            structural_planner_extension_sha256: structuralPass
              ? String(structural && structural.planner_extension_sha256 || '') : '',
            structural_contract_revision: structuralPass
              ? String(structural && structural.contract_revision || EXPERT_SKILL_REVISION) : '',
            structural_contract_sha256: structuralPass
              ? String(structural && structural.contract_sha256 || '') : '',
            contract_failures: structuralPass
              ? [] : structuralFailures,
            next_blocker: structuralPass
              ? meta.next_blocker
              : (forbiddenTreeReuse
                ? 'fresh candidate tree exactly matches a forbidden prior process artifact'
                : !structuralIdentity.pass
                ? `structural evidence identity failed: ${structuralIdentity.reasons.join('; ')}`
              : (structural && structural.next_blocker) ||
                'independent static structure contract did not pass'),
          });
          megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, meta);
        }
        if (shouldStructuralVerify) megaChargeMs(600000);
        if (MEGA_PRODUCTION) {
          verifyBudgetS = Math.max(0, Math.min(
            verifyBudgetS,
            Math.floor((turnDeadlineMs - megaNowMs()) / 1000 - 60),
            Math.floor((dispatchDeadlineMs - megaNowMs()) / 1000 - 60),
          ));
        }
        const contractBlocksRuntime = STRUCTURAL_CONTRACT_BLOCKING;
        const priorLaneHead = existing
          ? String(existing.working_head || existing.head || '') : '';
        const sourceAdvancedThisTurn = existing
          ? expectedHead !== priorLaneHead : reportedChangedFiles.length > 0;
        const structurallyReadyAuthoring = !!(
          eng && String(eng.candidate_status || '') === 'authoring' &&
          eng.claim_complete === true && eng.build !== false &&
          meta.checkpoint_complete &&
          (!STRUCTURAL_CONTRACT_BLOCKING || (
            structuralPassThisTurn &&
            meta.structural_candidate_head === expectedHead &&
            !!meta.structural_candidate_tree_digest
          ))
        );
        const postAuthoringVerify = structurallyReadyAuthoring &&
          ((sourceAdvancedThisTurn && candidateClaimsSkillTarget) ||
            (!authorPreflightRequired && !sourceAdvancedThisTurn));
        const gpuWaitBudgetS = Math.max(30, Math.min(300 * MEGA_DIRECTIONS,
          Math.floor(verifyBudgetS / (MEGA_DIRECTIONS > 1 ? 2 : 4))));
        const gpuRunBudgetS = Math.max(60, verifyBudgetS - gpuWaitBudgetS - 60);
        const shouldVerify = eng && eng.claim_complete === true && expectedHead &&
          (!contractBlocksRuntime || meta.structural_verified) &&
          !MEGA_STRUCTURAL_ONLY &&
          (!MEGA_PRODUCTION || verifyBudgetS >= 300) &&
          (postAuthoringVerify ||
            ['runnable', 'scored', 'finalist'].includes(String(eng.candidate_status || '')));
        const verifyInputs = !shouldVerify ? null : {
              CANDIDATE_ID: candidateId, CANDIDATE_SOURCE: source,
              CANDIDATE_TREE: tree, EXPECTED_HEAD: expectedHead,
          PERSISTENT_JIT_CACHE_DIR: persistentJitCache,
          CANDIDATE_IMPORT_MODULES,
          EXPECTED_STRUCTURAL_TREE_DIGEST: meta.structural_candidate_tree_digest,
          EXPERT_SKILL_CONTRACT_TOOL,
          EXPERT_SKILL_CONTRACT: EXPERT_SKILL_CONTRACT_FILE,
          LANE_LOCK: laneLock,
          GPU_WAIT_TIMEOUT_S: gpuWaitBudgetS,
          GPU_RUN_TIMEOUT_S: gpuRunBudgetS,
              BASE_CANDIDATE_ID: baseCandidateId,
              CANONICAL: baseTree, PATCH: (eng && eng.patch_file) || '',
              VERIFY_DIR: `${outDir}/verify`, ATTEMPT_ID: attemptId,
              VERIFY_TIMEOUT_S: verifyBudgetS,
              GPU_ID: GPU_SINGLE, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
              BASELINE_PER_CASE, FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
          VERIFY_TIER: 'score',
          MODIFIABLE_FILES: 'WHOLE_CANDIDATE_TREE',
          SPECIALTY: d.specialty || MEGA_DEFAULT_SPECIALTY,
          TARGET_SHAPE: megaShapeFromTopology(d.target_topology),
              TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
              REQUIRE_ARTIFACT_DISTINCT: true, REQUIRE_OVERLAP: false,
              REQUIRE_ATTRIBUTION: false, REQUIRED_REPLAYS: 30,
          REQUIRE_GRAPH_CAPTURE: MEGA_REQUIRE_GRAPH_CAPTURE ? '1' : '0',
              REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD, LAUNCH_TARGET,
              ACCURACY_METRIC, ACCURACY_THRESHOLD,
          REQUIRED_ACCURACY_CASES,
          MEGA_PLAN_IR: analysis && analysis.mega_plan_ir || {},
          RESOURCE_TIMELINE: analysis && analysis.resource_timeline || {},
          REQUIRE_RESOURCE_VERIFY: '1',
          SEARCH_ACCEPTS_PARTIAL_FUSION: ALLOW_PARTIAL_FUSION ? '1' : '0',
          ...(USE_EXPERT_SKILLS ? {
            EXPERT_SKILL_ID,
            EXPERT_SKILL_REVISION,
            EXPERT_SKILL_FILE: `${EXPERT_SKILL_DIR}/skill.md`,
            EXPERT_SKILL_PLAYBOOK: EXPERT_SKILL_PLAYBOOK_FILE,
            EXPERT_SKILL_CONTRACT: EXPERT_SKILL_CONTRACT_FILE,
            EXPERT_SKILL_VALIDATION: EXPERT_SKILL_VALIDATION_FILE,
            EXPERT_SKILL_RUNTIME_FILE,
            EXPERT_SKILL_ACCURACY_CASES,
            EXPERT_SKILL_SOURCE_FILES,
          } : {}),
          GRAPH_CONTRACT_TOOL, ...RFI, ...GMI, ...MRI,
          GRAPH_CONTRACT_REPLAYS: 30,
          ACTIVATION: (eng && eng.activation) ? JSON.stringify(eng.activation) : 'UNDECLARED',
        };
        const verifyAfter = async (verReply) => {
          if (shouldVerify) {
            ver = verReply;
            if (ver && ver.__agent_timed_out) {
              verifierTimedOut = true;
              log(`Mega round ${currentRound}: verify for ${ver.label || candidateId} exceeded its ` +
                `deadline.`);
              ver = null;
            }
            if (!ver || ver.claim_complete !== true) {
              const recoveryBudgetS = MEGA_PRODUCTION
                ? Math.max(0, Math.floor(turnBudgetS - (megaNowMs() - turnStartedMs) / 1000 - 60))
                : 300;
              if (recoveryBudgetS >= 30) {
                const recoveredVer = await agentT(
                  roleAgent('verify_engineer', 'recover',
                    `RECOVER verification for candidate ${candidateId}, attempt ${attemptId}: follow "PHASE=recover" ` +
                    'in the mode=mega verify_engineer fragment.', {
                      VERIFY_DIR: `${outDir}/verify`, CANDIDATE_ID: candidateId,
                      CANDIDATE_SOURCE: source, CANDIDATE_TREE: tree,
                      EXPECTED_HEAD: expectedHead, ATTEMPT_ID: attemptId,
                      SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
                    }),
                  { phase: 'Verify', label: `mega:verify-recover:${candidateId}`, schema: MEGA_VERIFY_SCHEMA,
                    timeout_ms: Math.min(120, recoveryBudgetS) * 1000, max_retries: 1 });
                if (recoveredVer && recoveredVer.claim_complete === true) {
                  ver = recoveredVer;
                  verifierTimedOut = false;
                } else if (verifierTimedOut && safeVerifyTimeoutRecovery(recoveredVer)) {
                  const retryBudgetS = Math.max(300, Math.min(1800, MEGA_FINAL_RESERVE_S));
                  const retry = await agentT(
                    roleAgent('verify_engineer', 'verify',
                      'TIMEOUT RECOVERY: verify the same exact HEAD read-only. Run only the earliest unresolved ' +
                      'GPU stage (construction/JIT, then the smallest correctness smoke) and return its concrete ' +
                      'result; do not run the full score suite unless the smoke is already cached.', {
                        ...verifyInputs,
                        VERIFY_DIR: `${outDir}/verify_timeout_retry`,
                        VERIFY_TIER: 'timeout_recovery_smoke',
                        GPU_WAIT_TIMEOUT_S: Math.min(120, gpuWaitBudgetS),
                        GPU_RUN_TIMEOUT_S: Math.min(600, gpuRunBudgetS),
                        VERIFY_TIMEOUT_S: retryBudgetS,
                        REQUIRED_REPLAYS: 1,
                        REQUIRED_PAIRS: 1,
                        TIMEOUT_RECOVERY_ATTEMPT: '1',
                      }),
                    { phase: 'Verify', label: `mega:verify-timeout-retry:${candidateId}`,
                      schema: MEGA_VERIFY_SCHEMA, timeout_ms: retryBudgetS * 1000,
                      timeout_marker: true, max_retries: 1 });
                  if (retry && retry.claim_complete === true) {
                    ver = retry;
                    verifierTimedOut = false;
                  } else if (retry && !retry.__agent_timed_out) {
                    // Quiescent and the bounded retry returned: the round ends unscored, the wave continues.
                    verifierTimedOut = false;
                    meta.next_blocker = `independent Verify of HEAD ${expectedHead} timed out before ` +
                      `sealing (retry: ${retry.status || 'unscored'}); route a fresh full Verify of this ` +
                      `HEAD before new edits`;
                    megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, meta);
                  } else {
                    megaUnsafeTimeout = retry || { label: `mega:verify-timeout-retry:${candidateId}` };
                  }
                } else if (verifierTimedOut) {
                  megaUnsafeTimeout = { label: `mega:verify:${candidateId}` };
                }
              }
            }
          }
          if (shouldVerify) megaChargeMs(verifyBudgetS * 1000);

          let record = meta;
          if (ver) {
            record = megaCandidateFromVerification(meta, ver, {
              targetGuards: TARGET_GUARDS, regressionGuards: [],
              launchTarget: LAUNCH_TARGET, promotionMetric: PROMOTION_METRIC,
              accuracyMetric: ACCURACY_METRIC, accuracyThreshold: ACCURACY_THRESHOLD,
              requiredAccuracyCases: REQUIRED_ACCURACY_CASES,
              requiredReplays: Math.min(30, REQUIRED_REPLAYS), requiredPairs: REQUIRED_PAIRS,
              allowPartialFusion: ALLOW_PARTIAL_FUSION, frozenBaseline: BASELINE_PER_CASE,
            });
            record.attempts = attempts;
            record.tree = tree;
            if (record.status === 'scored' && record.absolute_score <= 1.0) {
              record.status = 'runnable';
              record.next_blocker = record.next_blocker ||
                `candidate is correct but ${record.absolute_score.toFixed(4)}x does not beat frozen ` +
                `baseline; keep optimizing this lane, never promote it to a finalist`;
            }
            megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, record);
          }
          if (verifierTimedOut) {
            await persistMegaCandidateState(currentRound);
            return {
              d, clock, stop: true,
              reason: `candidate score verification timeout for ${candidateId}; invocation stopped before any ` +
                `other EP8 work was dispatched`,
            };
          }

          const hasEvidence = !!(ver && ver.claim_complete === true);
          const reachedHardware = !!(ver &&
            String(ver.activation_on_hardware || '').toLowerCase() === 'yes');
          // Source-only checkpoints do not consume the hardware-failure budget; only a complete candidate that
          // reached Verify counts.
          const hardwareWasDue = !!shouldVerify;
          const blockerKey = String(record.next_blocker || '').slice(0, 160);
          // Analysis gets the identity-bound evidence of THIS candidate instead of scanning EVAL_DIR.
          if (ANALYSIS_SKILL_ON && ver) profileSummary = { ...profileSummary,
            analysis_result: await runProfileAnalysis({ ...profileSummary, candidate_evidence: {
              candidate_id: candidateId, head: record.head, attempt_id: record.attempt_id,
              target_guard: record.target_guard, frozen_per_case: BASELINE_PER_CASE,
              per_case: record.per_case, paired_readings: record.paired_readings,
              denominator_mismatch: record.denominator_mismatch,
              component_breakdown: record.component_breakdown, hot_path_counters: record.hot_path_counters,
              evidence_manifest: record.evidence_manifest } }, currentRound, 'mega:analyze') };
          const verificationFaulted = !!(ver && ver.claim_complete === true &&
            (String(ver.correctness || '').toLowerCase().startsWith('fail') ||
              /fail|fault|crash|error/.test(String(ver.status || '').toLowerCase())));
          const rungOutcome = ver && ver.claim_complete === true &&
              Array.isArray(ver.per_case) && ver.per_case.length
            ? 'measured'
            : (verificationFaulted || eng && (eng.build === false ||
              String(eng.correctness || '').toLowerCase().startsWith('fail')))
              ? 'faulted' : 'unmeasured';
          recordRungOutcome(d.roadmap_rung, rungOutcome);
          if (rungOutcome === 'measured' && d.roadmap_rung) {
            LADDER_MEASURED.add(String(d.roadmap_rung));
          }
          return { d, clock, stop: false, direction: d, record, ver, hasEvidence, reachedHardware, hardwareWasDue,
            blockerKey, gpuExecuted: !!(eng && eng.gpu_executed === true),
            engStatus: eng ? eng.candidate_status : 'pending' };
        };
        return !shouldVerify ? { after: verifyAfter } : {
          prompt: roleAgent('verify_engineer', 'verify',
            'Independently verify and score this whole-tree mega candidate against the frozen baseline.',
            verifyInputs),
          opts: { phase: 'Verify', label: `mega:verify:${candidateId}`, schema: MEGA_VERIFY_SCHEMA,
            timeout_ms: verifyBudgetS * 1000, timeout_marker: true, max_retries: 1 },
          after: verifyAfter,
        };
      };
      return { d, clock, verifyStep };
    },
  };
}

// mode=mega round bookkeeping, called from the lane round after Verify.
// The round's modeled clock is its longest lane; a lane that stopped (timeout) ends the run.
function megaCloseRound(turns) {
  MEGA_CLOCK_MS = Math.max(megaRoundStartMs, ...turns.filter((t) => t && t.clock).map((t) => t.clock.ms));
  return { done: turns.filter((t) => t && !t.stop), stopped: turns.find((t) => t && t.stop) || null };
}

// Candidates are never merged: the winner is the best eligible registry candidate, and only once the
// positive control has calibrated the harness.
function megaSelectCandidate() {
  return megaMeasurementCalibration.ready
    ? selectMegaCandidate(megaCandidateRegistry, MEGA_TIE_NOISE_PCT).selected : null;
}

function megaWinnerOf(selected) {
  return {
    id: selected.id, source: selected.source, geomean: selected.absolute_score,
    per_case: selected.per_case, patch: selected.patch, tree: selected.tree,
    topology_sig: selected.topology_sig, provenance: selected.provenance,
    activation: selected.activation,
  };
}

function megaRoundCounters(turns, improved) {
  const hasEvidence = turns.some((t) => t.hasEvidence);
  noEvidence = hasEvidence ? 0 : noEvidence + 1;
  noImprove = improved ? 0 : noImprove + 1;
  if (turns.some((t) => t.reachedHardware)) noHardware = 0;
  else if (turns.some((t) => t.hardwareWasDue)) noHardware += 1;
  const blockerMoved = turns.some((t) => t.gpuExecuted && t.blockerKey !== megaLastBlocker);
  megaStall = hasEvidence || improved || blockerMoved ? 0 : megaStall + 1;
  megaLastBlocker = turns.length ? turns[turns.length - 1].blockerKey : megaLastBlocker;
}

function megaRoundEntry(currentRound, turns, selected, improved) {
  return {
    round: currentRound,
    directions: turns.map(({ direction: d, record }) => ({ id: d.id, title: d.title,
      specialty: d.specialty, candidate_id: record.id, candidate_source: record.source })),
    results: turns.map(({ direction: d, record, ver, engStatus }) => ({ id: d.id,
      status: ver ? ver.status : engStatus,
      verified: record.absolute_score, candidate_status: record.status })),
    winner: selected ? { source: selected.id, geomean: selected.absolute_score } : null,
    improved, cumulative,
  };
}

let megaSelection = null;
let megaSelectionContractPassed = false;
const runFinalistBatch = async (batch, label) => {
  if (!batch.length || !megaMeasurementCalibration.ready) return null;
  const batchStartedMs = megaNowMs();
  const remainingS = MEGA_TIME_BUDGET_S -
    (megaNowMs() - WORKFLOW_STARTED_MS) / 1000 - MEGA_CLOSEOUT_RESERVE_S;
  if (remainingS < 600) {
    log(`MEGA finalist skipped: only ${Math.max(0, Math.round(remainingS))}s remain before the ` +
      `closeout reserve; candidate state is preserved for the next production invocation.`);
    return null;
  }
  const finalistTimeoutS = Math.min(MEGA_FINAL_TIMEOUT_S, remainingS);
  const batchDeadlineMs = batchStartedMs + finalistTimeoutS * 1000;
  // We are committed to running this finalist batch; charge its budget to the deterministic
  // clock so a subsequent fallback batch sees the closeout window shrink.
  megaAdvanceMs(finalistTimeoutS * 1000);
  const result = await agentT(
    roleAgent('director', 'select_mega',
      MEGA_PRODUCTION
        ? 'Validate this highest-ranked workflow-authored candidate; do not pay for lower-ranked fallbacks unless it fails.'
        : 'Re-measure the top workflow-authored Mega candidates together; select the fastest correct whole kernel.', {
        EVAL_DIR, FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
        BASELINE_TREE: `${EVAL_DIR}/baseline`, COMMANDMENT,
        GPU_ID: GPU_SINGLE, SKILL_DIR: WORKFLOW_DIR,
        CANDIDATES: batch, TOP_K: batch.length, CANDIDATE_IMPORT_MODULES,
        MEGA_PROFILE,
        TIE_NOISE_PCT: MEGA_TIE_NOISE_PCT,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        LAUNCH_TARGET, BASELINE_LAUNCH_COUNT: optionalNumber(A.baseline_launch_count),
        BIMODAL_GUARDS, BASELINE_ACTIVATION,
        REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        ACCURACY_METRIC, ACCURACY_THRESHOLD,
        REQUIRED_ACCURACY_CASES,
        REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
        REQUIRE_GRAPH_CAPTURE: MEGA_REQUIRE_GRAPH_CAPTURE ? '1' : '0',
        DIRECT_GRAPH_ACCURACY: DIRECT_GRAPH_ACCURACY ? '1' : '0',
        ...(USE_EXPERT_SKILLS ? {
          EXPERT_SKILL_ID,
          EXPERT_SKILL_REVISION,
          EXPERT_SKILL_FILE: `${EXPERT_SKILL_DIR}/skill.md`,
          EXPERT_SKILL_PLAYBOOK: EXPERT_SKILL_PLAYBOOK_FILE,
          EXPERT_SKILL_CONTRACT: EXPERT_SKILL_CONTRACT_FILE,
          EXPERT_SKILL_VALIDATION: EXPERT_SKILL_VALIDATION_FILE,
          EXPERT_SKILL_RUNTIME_FILE,
          EXPERT_SKILL_TARGET_SCORE: EXPERT_SKILL_RECORDED_SCORE,
          EXPERT_SKILL_TARGET_BAND: [
            EXPERT_SKILL_RECORDED_LOW, EXPERT_SKILL_RECORDED_HIGH,
          ],
          EXPERT_SKILL_ACCURACY_CASES,
          EXPERT_SKILL_SOURCE_FILES,
        } : {}),
        GRAPH_CONTRACT_TOOL, ...RFI, ...GMI, ...MRI,
        GRAPH_CONTRACT_REPLAYS: REQUIRED_REPLAYS,
        SELECTED_WORKSPACE: `${EVAL_DIR}/mega_selected`,
      }),
    { phase: 'Validate', label, schema: MEGA_SELECTION_SCHEMA,
      timeout_ms: finalistTimeoutS * 1000, timeout_marker: true, max_retries: 1 });
  if (result && (result.__agent_timed_out || result.claim_complete === true)) return result;
  // Recovery is a read-only reconciliation; model a fixed bounded window on the deterministic clock.
  const recoveryRemainingS = Math.min(300, Math.max(0, Math.floor(finalistTimeoutS)));
  if (recoveryRemainingS < 30) return result;
  const recovered = await agentT(
    roleAgent('director', 'recover_mega',
      `RECOVER ONLY a completed finalist selection from ${EVAL_DIR}. Read the atomic selection ` +
      `evidence and ${EVAL_DIR}/mega_selected; do not run a GPU command, take a lease, edit a ` +
      `candidate, or create a new selection.`, {
        EVAL_DIR, CANDIDATES: batch, SELECTED_WORKSPACE: `${EVAL_DIR}/mega_selected`,
        SKILL_DIR: WORKFLOW_DIR,
      }),
    { phase: 'Validate', label: `${label}:recover`, schema: MEGA_SELECTION_SCHEMA,
      timeout_ms: Math.min(300, recoveryRemainingS) * 1000, max_retries: 1 });
  return recovered && recovered.claim_complete === true ? recovered : result;
};
let finalists = [];

// Mega provenance/topology labels never change the quality bar.
let mega_deliverable = null, mega_note = '';
// Set by the phase hooks below, read across phases.
let FINAL_WORKSPACE = null;
let LADDER = null;
let PRIOR_ART = null;
let PRIOR_ART_RECORDED = null;
let SHELF_STATS = null;
let WORKFLOW_REVISION_START = null;
let benchR = null;
let ranked = null;
let reportReady = null;

// ---- mode=mega phase hooks ----
async function megaAfterSetup() {
  if (setup.mega_workspace_from_original !== true || setup.mega_candidate_state_checked !== true) {
    throw new Error(
      'MEGA SETUP REFUSED: ordinary workspace must be seeded from KERNEL_PATH_ORIG.');
  }
  // A grown registry (~120 KB after 18 rounds) does not fit in the director's setup reply: read it in chunks.
  if (MEGA_RESUME_STATE && setup.resumed && setup.prior_state) {
    const stateFile = `${STATE_DIR}/STATE.json`;
    const chunkTool = `python3 ${WORKFLOW_DIR}/tools/state_chunks.py ${stateFile}`;
    const runChunk = (cmd, label, props) => agent(
      `Run exactly: ${cmd}\nIt prints one JSON object. Return its fields verbatim. If the command ` +
      `fails, return error with its last output lines. Do nothing else.`,
      { phase: 'Setup', label, schema: { type: 'object', additionalProperties: true,
        properties: { ...props, error: STR }, required: [] } });
    for (const field of ['candidate_registry', 'insights', 'ledger']) {
      const replied = setup.prior_state[field];
      if (field !== 'candidate_registry' && Array.isArray(replied)) continue;
      const head = (await runChunk(`${chunkTool} ${field}`, `state:${field}`,
        { length: NUM, chunks: NUM })) || {};
      if (head.error || !Number.isInteger(head.chunks)) {
        throw new Error(`MEGA RESUME REFUSED: reading ${field} from ${stateFile} failed: ` +
          `${head.error || 'no output'}`);
      }
      const values = [];
      for (let i = 0; i < head.chunks; i++) {
        const part = (await runChunk(`${chunkTool} ${field} --chunk ${i}`, `state:${field}:${i}`,
          { chunk: NUM, items: { type: 'array' } })) || {};
        if (part.error || part.chunk !== i || !Array.isArray(part.items)) {
          throw new Error(`MEGA RESUME REFUSED: reading ${field} chunk ${i} from ${stateFile} failed: ` +
            `${part.error || 'bad reply'}`);
        }
        values.push(...part.items);
      }
      if (values.length !== head.length) {
        throw new Error(`MEGA RESUME REFUSED: ${field} in ${stateFile} has ${head.length} entries, ` +
          `${values.length} were read back.`);
      }
      if (field === 'candidate_registry' && Array.isArray(replied)) {
        const byId = new Map(values.map((r) => [r && r.id, r]));
        const merged = values.map((r) => ({ ...r }));
        for (const r of replied) {
          const i = merged.findIndex((m) => r && m.id === r.id);
          if (i >= 0) merged[i] = { ...merged[i], ...r };
          else merged.push(r);
        }
        setup.prior_state[field] = merged;
        log(`Resume: read ${values.length} candidate rows from STATE.json; setup returned ` +
          `${replied.length} (${replied.filter((r) => r && !byId.has(r.id)).length} not in STATE.json).`);
      } else {
        setup.prior_state[field] = values;
        log(`Resume: read ${values.length} ${field} entries from STATE.json.`);
      }
    }
  }
  WORKFLOW_REVISION_START = String(setup.workflow_revision || '');
  log(`MEGA unified mode: Expert Skills are ${USE_EXPERT_SKILLS ? 'enabled as advisory context' :
    'disabled'}; all candidates use the same autonomous lifecycle.`);
}

function megaAnalyzeInputs() {
  return {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR: '',
    GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
    ...(A.require_task_graph ? { REQUIRE_TASK_GRAPH: '1' } : {}),
    ...(USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE ? {
      EXPERT_SKILL_PLANNER_EXTENSION: EXPERT_SKILL_PLANNER_EXTENSION_FILE,
      EXPERT_SKILL_ID,
      EXPERT_SKILL_REVISION,
      EXPERT_SKILL_BUNDLE_TOOL,
      EXPERT_SKILL_BUNDLE_SHA256,
      EXPERT_SKILL_PLANNER_EXTENSION_SHA256,
      EXPERT_SKILL_CONTRACT_SHA256,
      EXPERT_SKILL_VALIDATION_STATUS,
      EXPERT_SKILL_USAGE,
    } : {}),
  };
}

// Whether an analysis misses its structured contract (planner extension pinned when a skill is used).
function megaAnalyzeContract(incremental, a) {
  const pin = USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE;
  return analyzeResumeDegenerate(incremental, a, !!A.require_task_graph, true,
    pin ? EXPERT_SKILL_REVISION : '', pin ? EXPERT_SKILL_ID : '',
    pin ? EXPERT_SKILL_PLANNER_EXTENSION_SHA256 : '', pin ? EXPERT_SKILL_BUNDLE_SHA256 : '',
    USE_EXPERT_SKILLS ? EXPERT_SKILL_VALIDATION_STATUS : '', USE_EXPERT_SKILLS ? EXPERT_SKILL_USAGE : '');
}

// mode=mega after the analyze call: contract check, one full re-run, task graph and ladder gates.
async function megaAfterAnalyze() {
  if (analysis && analysis.__agent_timed_out) {
    return { exit: {
      mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
      reason: 'Analyze exceeded the production timeout; no second Analyze or GPU phase was started',
    } };
  }

  {
    const d = megaAnalyzeContract(INCREMENTAL, analysis);
    if (d.retry) {
      log(d.reason);
      // The same call again, as a full (non-incremental) analysis.
      const full = await agentT(
        roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', megaAnalyzeInputs()),
        { phase: 'Analyze', label: 'tech_lead:analyze:full',
          schema: MEGA_ANALYZE_SCHEMA,
          ...(MEGA_PRODUCTION
            // 900s: same rationale as the primary analyze call site above.
            ? { timeout_ms: 900000, timeout_marker: true, max_retries: 1 } : {}) });
      if (full && full.__agent_timed_out) {
        return { exit: {
          mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
          final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
          reason: 'full Analyze fallback exceeded the production timeout; no GPU phase was started',
        } };
      }
      const got = (full && Array.isArray(full.candidate_directions) ? full.candidate_directions : [])
        .filter((c) => c && (c.id || c.title));
      const fullContract = megaAnalyzeContract(false, full);
      if (got.length && !fullContract.retry) {
        log(`ANALYZE RE-RUN recovered a ladder of ${got.length} rung(s). Using the full analysis.`);
        analysis = full;
      } else {
        log(`ANALYZE RE-RUN still missed its structured contract: ${fullContract.reason}`);
        if (full) analysis = full;
        return { exit: {
          mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
          final_speedup: 0, final_geomean: 0,
          validation_status: 'analyze_contract_incomplete',
          reason: fullContract.reason,
        } };
      
      }
    }
  }
  analysis = applyBaselineOperatorMap(analysis);
  if (REQUIRE_TASK_GRAPH) {
    const g = taskGraphGate(analysis && analysis.task_graph);
    if (g.summary) log(g.summary);
    TASK_GRAPH_CAVEAT = g.caveat;
    if (g.caveat) log(g.caveat);
    // Print both the graph and the pipe table and carry both to the report; they fail independently.
    const pg = pipeOccupancyGate(analysis && analysis.resource_timeline, []);
    if (pg.summary) log(pg.summary);
    PIPE_TABLE_CAVEAT = pg.caveat;
    if (pg.caveat) log(pg.caveat);
  }

  // Analyze's ladder and what has been taken off it, threaded into every plan_round call.
  LADDER = (analysis && Array.isArray(analysis.candidate_directions)) ? analysis.candidate_directions : [];
  {
    const lg0 = roadmapLadderGate(LADDER, [], new Set());
    if (lg0.summary) log(lg0.summary);
    if (lg0.caveat) log(lg0.caveat);
  }
  // Under working_kernel an empty ladder disables rung, chain-debt and forcing tracking.
  if (WORKING_KERNEL && !LADDER.length) {
    const msg = 'LADDER EMPTY under objective=working_kernel: Analyze returned no candidate_directions, ' +
      'so there is no staged path to a fused terminal rung.';
    log(msg);
    CHAIN_CAVEATS.push(`setup: ${msg}`);
  }

  // Prior art: a missing prior_art key (unknown) differs from [] (searched, none found).
  PRIOR_ART_RECORDED = !!(analysis && Array.isArray(analysis.prior_art));
  PRIOR_ART = PRIOR_ART_RECORDED ? analysis.prior_art : [];
  if (!PRIOR_ART_RECORDED) {
    log(`PRIOR ART UNRECORDED: analysis.json has no "prior_art" key, so the step-4d sweep is not on ` +
        `the record — this is NOT the same as finding none. Every later statement about what does or ` +
        `does not exist in the baseline is UNSOURCED and must be read as such.`);
  } else if (PRIOR_ART.length === 0) {
    log(`PRIOR ART: none — the sweep ran and reported an empty result.`);
  }
  for (const pa of PRIOR_ART) {
    const where = pa.implemented_at || '?';
    const eff = pa.measured_effect ? ` (measured ${pa.measured_effect})` : '';
    if (pa.in_baseline === false) {
      log(`PRIOR ART NOT IN BASELINE: "${pa.direction}" is already implemented at ${where}${eff} but ` +
          `is ABSENT from the tree being optimized. Port or measure it before re-deriving it; if the ` +
          `intent was to optimize the tree that HAS it, this run is pointed at the wrong target.`);
    } else {
      log(`PRIOR ART: "${pa.direction}" already implemented at ${where}${eff}; ` +
          `enable via ${pa.how_to_enable || '?'}. Measure it, do not re-derive it.`);
    }
    // in_baseline must carry evidence.
    if (pa.in_baseline != null && !pa.evidence) {
      log(`PRIOR ART UNEVIDENCED: "${pa.direction}" asserts in_baseline=${pa.in_baseline} with no ` +
          `"evidence" field naming the check that established it. Treat the flag as unverified.`);
    }
  }

}

// mode=mega structural-only runs: placeholder baseline, no GPU benchmark.
function megaStructuralBench() {
  return {
    baseline_per_case: structuralGuardNames.map((name) => ({
      name, latency_ms: 1, baseline_ms: 1, speedup: 0,
      role: TARGET_GUARDS.includes(name) ? 'target' : 'regression_guard',
      runs_ms: [], drift_pct: 0, params: 'structural-only placeholder',
      metric: PROMOTION_METRIC,
    })),
    baseline_geomean_ms: 1,
    num_test_cases: structuralGuardNames.length,
    reliable: false,
    positive_control: { claim_complete: false },
    notes: 'STRUCTURAL_ONLY: no GPU benchmark.',
  };
}

function megaBenchInputs() {
  return {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE,
    ANALYSIS: analysis,
    TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
    REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD, ...RFI, ...GMI, ...MRI,
    ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
    ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
    ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
    ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}),
  };
}

// mode=mega after the setup call: recover a baseline a dead or timed-out setup agent left on disk.
async function megaAfterBenchmark(benchFirst) {
  // A bench agent that dies usually failed to report, not to measure.
  benchR = benchFirst;
  // A timed-out setup agent usually left its baseline/control on disk: fall through to the GPU-free
  // RECOVERY below; only an empty disk makes the timeout terminal.
  const benchTimedOut = !!(benchR && benchR.__agent_timed_out);
  if (benchTimedOut) {
    log('Benchmark setup hit its full timeout window without returning.');
    benchR = null; // a timeout marker carries no baseline_per_case; force the recovery path below
  }
  if (!benchR || !benchR.baseline_per_case) {
    log('Benchmark setup returned nothing.');
    benchR = await agentT(
      roleAgent('benchmark_engineer', 'recover',
        'The previous benchmark_engineer for this EVAL_DIR did not return: follow "PHASE=recover" in the ' +
        'mode=mega benchmark_engineer fragment.',
        { WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, ANALYSIS: analysis,
          TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
          REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
          ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}) }),
      { phase: 'Benchmark', label: 'benchmark_recover', schema: MEGA_BENCH_SCHEMA,
        ...(MEGA_PRODUCTION ? { timeout_ms: 1200000, max_retries: 1 } : {}) });
    if (!benchR || !Array.isArray(benchR.baseline_per_case) || !benchR.baseline_per_case.length) {
      if (benchTimedOut) {
        // Nothing recoverable: return an explicit timeout status rather than throwing.
        return { exit: {
          mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
          final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
          reason: 'benchmark/calibration agent exceeded the production timeout; recovery from EVAL_DIR ' +
                  'found no baseline on disk',
        } };
      }
      throw new Error('Benchmark setup failed: no baseline recorded, and none recoverable from EVAL_DIR');
    }
    log(`Benchmark RECOVERED from disk: ${benchR.baseline_per_case.length} cases. ${benchR.notes || ''}`);
  }
  bench = benchR;
}

async function megaCalibrate() {
  if (POSITIVE_CONTROL) {
    const pc = benchR.positive_control || {};
    const pcComplete = pc.claim_complete === true;
    if (!pcComplete) {
      log(`Positive control PENDING: claim_complete is not true` +
        `${pc.attempt_id ? ` for attempt ${pc.attempt_id}` : ''}. A force-emitted or partial 0.00% ` +
        `is not a failed instrument. Candidate authoring may continue, but mega performance ranking ` +
        `remains disabled until a completed calibration manifest is recovered.`);
    } else {
    const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
    const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
    const got = Number(pc.measured_pct);
    // The bounds are asymmetric: below lo the instrument cannot see the effect (abort); above hi it
    // over-reads (pass with overshoot); absurd or wrong sign measures something else (abort).
    // <<REPLAY:pc_gate>>  scripts/replay_runs.js lifts this region verbatim; keep it pure (reads only
    // lo, hi, got, pc, POSITIVE_CONTROL; no log, throw or outside state).
    const mLo = Math.min(Math.abs(lo), Math.abs(hi));       // smallest effect the loop must resolve
    const mHi = Math.max(Math.abs(lo), Math.abs(hi));
    const wantSign = Math.sign(lo + hi) || 1;               // +1 = control is faster, -1 = slower
    const mGot = Math.abs(got);
    const ABSURD = Number.isFinite(Number(POSITIVE_CONTROL.implausible_pct))
      ? Math.abs(Number(POSITIVE_CONTROL.implausible_pct)) : (Number.isFinite(mHi) ? 2 * mHi : NaN);
    const nullArm = Number(pc.null_arm_pct);
    // An overshoot is only benign if the null arm is quiet.
    const nullQuiet = Number.isFinite(nullArm) && Math.abs(nullArm) <= mLo / 2;
    // switch_present=false: both arms ran the same code, so the control did not run.
    const switchAbsent = pc.switch_present === false;
    const ran = pc.ran !== false && !switchAbsent &&
      Number.isFinite(got) && Number.isFinite(lo) && Number.isFinite(hi);

    // magnitude=constructed marks a synthetic control whose band is a target.
    const constructed = String(POSITIVE_CONTROL.magnitude || 'recorded') === 'constructed';
    const UNDERSHOOT_FRAC = 0.5;   // below half the target, assume the injection did not take
    const RESOLVE_K = 3;           // effect must beat the worst null pair by this factor
    const nullPairs = Array.isArray(pc.null_pairs_pct) ? pc.null_pairs_pct.map(Number).filter(Number.isFinite) : [];
    const nullWorst = nullPairs.length ? Math.max(...nullPairs.map(Math.abs))
      : (Number.isFinite(nullArm) ? Math.abs(nullArm) : NaN);
    const ctrlPairs = Array.isArray(pc.control_pairs_pct) ? pc.control_pairs_pct.map(Number).filter(Number.isFinite) : [];
    const signUnanimous = ctrlPairs.length >= 3 && ctrlPairs.every((d) => Math.sign(d) === wantSign);
    const resolvedByScale = Number.isFinite(nullWorst) && mGot >= RESOLVE_K * nullWorst;

    // For a bimodal null, a second distribution-free route: effect and null pairs do not overlap by
    // magnitude. Both routes must still clear mLo * UNDERSHOOT_FRAC.
    const absCtrl = ctrlPairs.map(Math.abs);
    const absNull = nullPairs.map(Math.abs);
    const resolvedBySeparation = absCtrl.length >= 5 && absNull.length >= 5 &&
      Math.min(...absCtrl) > Math.max(...absNull);
    const resolved = resolvedByScale || resolvedBySeparation;
    const undershoot = ran && constructed && Math.sign(got) === wantSign &&
      mGot < mLo && mGot >= mLo * UNDERSHOOT_FRAC && resolved && signUnanimous;

    // Wrong sign is an insensitivity failure, not an overshoot: the loop did not see the change it was
    // handed, whatever else it saw.
    const tooSmall = ran && (Math.sign(got) !== wantSign || (mGot < mLo && !undershoot));
    const absurd = ran && Number.isFinite(ABSURD) && mGot > ABSURD;
    const overshoot = ran && mGot > mHi && !absurd;
    const ok = ran && !tooSmall && !absurd && (!overshoot || nullQuiet);
    // <</REPLAY:pc_gate>>
    PC_RAN = ran;
    PC_PASSED = ok && (megaCalibrationClaimPass({ ...pc, passed: ok }, POSITIVE_CONTROL));
    if (switchAbsent) {
      log(`POSITIVE CONTROL SWITCH ABSENT: "${POSITIVE_CONTROL.name || 'unnamed'}" names a knob that ` +
          `is not in the tree it was run against, so both arms executed identical code and the ` +
          `${Number.isFinite(got) ? got.toFixed(2) + '%' : 'reported'} reading is a base-vs-base null. ` +
          `This is a LAUNCH-ARGS defect, not an instrument failure — fix \`positive_control.how\` to ` +
          `name a switch that exists, or have the benchmark engineer construct a synthetic control ` +
          `(benchmark_engineer.md 5b) and declare magnitude:'constructed'.`);
    }
    log(`Positive control "${POSITIVE_CONTROL.name || 'unnamed'}": ` +
        `measured ${Number.isFinite(got) ? got.toFixed(2) + '%' : 'NOT RUN'}, ` +
        `expected ${lo}..${hi}% (absurd above ${Number.isFinite(ABSURD) ? ABSURD.toFixed(2) + '% in magnitude' : '?'}) ` +
        `-> ${ok ? (overshoot ? 'PASS (OVERSHOOT)' : undershoot ? 'PASS (UNDERSHOOT)' : 'PASS') : 'FAIL'}` +
        (Number.isFinite(nullArm) ? ` (null arm ${nullArm.toFixed(2)}%)` : ' (null arm UNREPORTED)'));
    if (ok && undershoot) {
      // An undershoot may proceed, but every reported number carries the caveat.
      PC_UNDERSHOOT = `the positive control read ${got.toFixed(2)}% for a CONSTRUCTED effect targeted at ` +
        `${lo}..${hi}%, with a ${Number.isFinite(nullArm) ? nullArm.toFixed(2) : '?'}% null arm ` +
        `(worst null pair ${Number.isFinite(nullWorst) ? nullWorst.toFixed(2) : '?'}pp) and ` +
        `${ctrlPairs.length}/${ctrlPairs.length} pairs agreeing in sign. The effect is ` +
        `${Number.isFinite(nullWorst) && nullWorst > 0 ? (mGot / nullWorst).toFixed(1) + 'x' : '>>'} the worst ` +
        `null pair, so the loop resolves it; the shortfall against the target is a sizing miss in the ` +
        `injected knob, which was never measured before this run. Treat every effect below as possibly ` +
        `reading LOW by roughly ${(mLo - mGot).toFixed(2)}pp at this size — a marginal win may be real, ` +
        `and a marginal loss may be worse than it looks.`;
      log(`POSITIVE CONTROL UNDERSHOOT: ${PC_UNDERSHOOT}`);
    }
    if (ok && overshoot) {
      // Not a failure, but it must not vanish either: it is a standing caveat on every number the run
      // goes on to report, and the report has to say so.
      PC_OVERSHOOT = `the positive control read ${got.toFixed(2)}% for a change recorded at ` +
        `${lo}..${hi}%, with a ${nullArm.toFixed(2)}% null arm. The harness resolves the known effect ` +
        `with the right sign and clean separation, so it is sensitive enough to trust; but it reads ` +
        `HIGH by roughly ${(mGot - mHi).toFixed(2)}pp at this effect size, so treat every speedup below ` +
        `as possibly carrying the same upward scale bias.`;
      log(`POSITIVE CONTROL OVERSHOOT: ${PC_OVERSHOOT}`);
    }
    if (!ok) {
      const why = `Positive control FAILED: a change with a known effect of ${lo}..${hi}% measured ` +
        `${Number.isFinite(got) ? got.toFixed(2) + '%' : 'nothing (control did not run)'}. ` +
        (absurd
          ? `That is beyond the ${ABSURD.toFixed(2)}% plausibility ceiling — at that size the harness ` +
            `is not measuring this change, it is measuring something else (wrong arm, wrong guard, ` +
            `wrong binary). `
          : overshoot
            ? `It overshot the expected ceiling AND the null arm is ` +
              `${Number.isFinite(nullArm) ? Math.abs(nullArm).toFixed(2) + '%, too large relative to the ' + mLo + '% effect' : 'UNREPORTED'}, ` +
              `so the excess cannot be told apart from interleave drift. `
            : (Math.sign(got) !== wantSign && Number.isFinite(got)
                ? `It came back with the WRONG SIGN, so the loop is not tracking the change it was ` +
                  `handed — a wiring error (arms swapped, wrong guard, wrong binary) before it is a ` +
                  `sensitivity one. `
                : ``) +
              `The measurement loop cannot resolve an effect of the size this run is looking for, so ` +
              `every number it reports — including 1.000x — is uninterpretable. Fix the harness, not ` +
              `the kernel. `) +
        // Say which undershoot condition failed, in the reader's numbers; the two causes need opposite fixes.
        (tooSmall && Math.sign(got) === wantSign && mGot < mLo
          ? `This is an UNDER-read, so before touching the harness, establish which of the two it is. ` +
            (!constructed
              ? `This control is declared \`magnitude: 'recorded'\`, meaning ${lo}..${hi}% is an effect ` +
                `someone has actually measured — so reading ${mGot.toFixed(2)}% IS the instrument's fault. ` +
                `If instead that band was a target you aimed a synthetic knob at and never measured, the ` +
                `control is \`magnitude: 'constructed'\` and should be declared as such; do NOT relabel a ` +
                `recorded effect to get past this gate, that falsifies the experiment. `
              : mGot < mLo * UNDERSHOOT_FRAC
                ? `It read under half the target (${(mLo * UNDERSHOOT_FRAC).toFixed(2)}%), which is the ` +
                  `signature of an injection that never took effect — arms sharing a JIT cache entry ` +
                  `(knowledge/jit_arm_isolation.md), the env var unread, the gated code not on the hot ` +
                  `path. Verify the two arms have distinct disk_keys before blaming the loop. `
                : !signUnanimous
                  ? `Its ${ctrlPairs.length} pair(s) do not all agree in sign, so the median is coming out ` +
                    `of a disagreement rather than an effect — that IS a resolution failure. `
                  : !resolved
                    ? `It is only ${Number.isFinite(nullWorst) && nullWorst > 0 ? (mGot / nullWorst).toFixed(1) + 'x' : '?'} ` +
                      `the worst null pair (${Number.isFinite(nullWorst) ? nullWorst.toFixed(2) : '?'}pp), under the ` +
                      `${RESOLVE_K}x an effect must clear to be told apart from the interleave, and its ` +
                      `pairs do not separate from the null's either ` +
                      `(smallest control pair ${absCtrl.length ? Math.min(...absCtrl).toFixed(2) : '?'}pp ` +
                      `vs largest null pair ${absNull.length ? Math.max(...absNull).toFixed(2) : '?'}pp` +
                      `${absCtrl.length >= 5 && absNull.length >= 5 ? '' : '; separation needs >=5 pairs on BOTH arms'}), ` +
                      `so neither route to resolution is open. Quiet the interleave or raise the injected ` +
                      `cost. If the null is fat-tailed rather than noisy -- a few readings well above an ` +
                      `otherwise tight cluster -- collect more null pairs and check whether the two arms ` +
                      `separate; that route does not care about the tail. `
                    : ``)
          : ``) +
        `Note from benchmark engineer: ${pc.note || '(none)'}`;
      if (PC_ABORT) throw new Error(why);
      log(`WARNING: ${why}`);
    }
    }
  }

  if (PC_PASSED) {
    const pc = benchR.positive_control || {};
    megaMeasurementCalibration = {
      ready: true,
      candidate_id: 'synthetic_control',
      attempt_id: pc.attempt_id || 'benchmark-control',
      evidence_manifest: pc.evidence_manifest || '',
      note: `independent synthetic control passed at ${Number(pc.measured_pct).toFixed(2)}%`,
    };
  }
  if (!PC_PASSED) {
    await remeasureMegaCalibration('after-setup', benchR.positive_control);
  }
}

// mode=mega baseline profile before any agent: only the measured baseline table and declared launch shape.
function megaProfileStub() {
  const targetBaseline = BASELINE_PER_CASE.find((row) =>
    TARGET_GUARDS.includes(String(row && row.name || ''))) || BASELINE_PER_CASE[0] || {};
  return {
    bottleneck: 'unknown',
    profiler_used: 'benchmark-only',
    device: String(A.device_description || 'task-declared accelerator'),
    dispatch_count: Number(A.baseline_launch_count ||
      targetBaseline.launch_count || targetBaseline.dispatch_count || 0),
    key_metrics: {},
    top_kernels: [],
    top_opportunities: [
      'derive search directions from the frozen-source task graph and rank-max baseline; ' +
      'do not infer fused overlap before a fused candidate exists',
    ],
    rank_max_ms: Number(targetBaseline.latency_ms || targetBaseline.baseline_ms || 0),
    path_marker: '',
    summary_path: '',
    shift_note: 'pre-candidate baseline only',
  };
}

// mode=mega merges the per-stage baseline measurement (if any) into the stub, then runs the analysis skill.
async function megaAfterProfile(mp) {
  // Failure or timeout keeps the stub; it never ends the wave.
  if (mp && mp.bottleneck) profileSummary = { ...profileSummary, ...mp };
  if (mp && pipeOccupancyGate(mp.resource_timeline, []).verdict !== 'MISSING') {
    analysis = { ...analysis, resource_timeline: { ...(analysis && analysis.resource_timeline), ...mp.resource_timeline } };
  }
  if (profileSummary && profileSummary.__agent_timed_out) {
    return { exit: {
      mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
      reason: 'baseline profile exceeded the production timeout; no candidate GPU work was started',
    } };
  }
  const baselineAnalysisResult = await runProfileAnalysis(
    profileSummary, 0, 'analyze_profile:baseline');
  if (profileSummary) {
    profileSummary = {
      ...profileSummary,
      analysis_result: baselineAnalysisResult,
    };
  }

}

async function megaLoopInit() {
  // The evidence K gets resized from.
  SHELF_STATS = { k: SHELF_OFFER_K, shelved: 0, withheld_unknown_footprint: 0,
    rounds_offered: 0, offers: 0, hits: 0, hit_ids: [] };

  // Resume: restore cumulative speedup and insight/ledger history from the prior wave.
  if (setup.resumed && setup.prior_state) {
    const ps = setup.prior_state;
    if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
    if (Array.isArray(ps.insights)) {
      insightBook = mergeInsights(insightBook, ps.insights, 0, false).book;
      history.insights = renderInsights(insightBook);
    }
    if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
    if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
    if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
    if (Array.isArray(ps.shelf)) {
      shelf = ps.shelf.filter((e) => e && e.id && !e.absorbed)
        .map((e) => ({ ...e, base_round: 0, files: shelfFiles(e),
          footprint: shelfFiles(e).length ? 'known' : 'unknown' }));
    }
    if (ps.absorbed_files && typeof ps.absorbed_files === 'object') {
      const prior = [];
      for (const k of Object.keys(ps.absorbed_files)) for (const f of (ps.absorbed_files[k] || [])) prior.push(f);
      if (prior.length) absorbedByRound[0.5] = [...new Set(prior.map(shelfFile))];
    }
    const megaRestoreRejected = [];
    if (Array.isArray(ps.candidate_registry)) {
      for (const c of ps.candidate_registry) {
        const n = normalizeMegaCandidate(c);
        if (!n.id_valid || !n.source_valid || !n.source_id_valid) {
          log(`MEGA candidate state rejected: id=${n.id || '(empty)'} source=${String(c && c.source || '')}.`);
          megaRestoreRejected.push(n.id || '(empty)');
          continue;
        }
        const staleIdentityReasons = [];
        if (CHECK_EXPERT_SKILL_CONTRACT && n.structural_verified) {
          if (n.structural_skill_id !== EXPERT_SKILL_ID) {
            staleIdentityReasons.push('Skill id changed or was not recorded');
          }
          if (n.structural_contract_revision !== EXPERT_SKILL_REVISION) {
            staleIdentityReasons.push('Skill revision changed');
          }
          if (EXPERT_SKILL_CONTRACT_SHA256 &&
              n.structural_contract_sha256 !== EXPERT_SKILL_CONTRACT_SHA256) {
            staleIdentityReasons.push('contract digest changed');
          }
          if (EXPERT_SKILL_PLANNER_EXTENSION_SHA256 &&
              n.structural_planner_extension_sha256 !==
                EXPERT_SKILL_PLANNER_EXTENSION_SHA256) {
            staleIdentityReasons.push('Planner Extension digest changed');
          }
          if (EXPERT_SKILL_BUNDLE_SHA256 &&
              n.structural_skill_bundle_sha256 !== EXPERT_SKILL_BUNDLE_SHA256) {
            staleIdentityReasons.push('Skill bundle digest changed');
          }
          if (!n.structural_candidate_head || n.structural_candidate_head !== n.head) {
            staleIdentityReasons.push('candidate HEAD changed or was not recorded');
          }
          if (!n.structural_candidate_tree_digest) {
            staleIdentityReasons.push('candidate tree digest was not recorded');
          }
        }
        const staleContract = staleIdentityReasons.length > 0;
        if (staleContract) {
          log(`MEGA candidate ${n.id}: invalidating structural evidence identity: ` +
            `${staleIdentityReasons.join('; ')}.`);
        }
        megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, {
          ...n,
          ...(staleContract ? {
            structural_verified: false,
            structural_report: '',
            structural_skill_id: '',
            structural_candidate_head: '',
            structural_candidate_tree_digest: '',
            structural_skill_bundle_sha256: '',
            structural_planner_extension_sha256: '',
            structural_contract_revision: '',
            structural_contract_sha256: '',
            contract_failures: [{
              id: 'structural_evidence_identity_changed',
              category: 'plan',
              severity: 'required',
              messages: staleIdentityReasons,
            }],
            next_blocker: `re-run Expert Skill bundle ${EXPERT_SKILL_REVISION} contract on exact ` +
              `HEAD ${n.head || n.working_head || '(missing)'} before device verification`,
          } : {}),
          // State may carry evidence and a head, never authority to redirect the lane outside its root.
          tree: megaLaneTree(n.id),
        });
      }
    }
    // A resume whose rows were compacted past validity (`source` dropped) silently ran fresh.
    if (megaRestoreRejected.length) {
      throw new Error(`MEGA RESUME REFUSED: setup returned invalid candidate_registry rows ` +
        `(${megaRestoreRejected.join(', ')}); rows must keep id/source/base_id/tree/head/status.`);
    }
    if (Number.isFinite(Number(ps.state_sequence))) {
      megaStateSequenceBase = Math.max(0, Number(ps.state_sequence));
    }
    // Calibration is intentionally NOT restored as authority.
    log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
    if (shelf.length) log(`  shelf: ${shelf.length} verified non-winner(s) carried forward, ` +
      `${shelf.filter((e) => e.footprint === 'known').length} with a known file footprint.`);
    log(`  mega candidates: ${megaCandidateRegistry.length}, calibration=${megaMeasurementCalibration.ready ? 'ready' : 'pending'}.`);
  }
}

// mode=mega before the lane's plan call: the modeled-clock deadline, calibration recovery and the GPU pool.
// Returns {stop} or {pool}.
async function megaBeforePlan(currentRound) {
  const elapsedS = (megaNowMs() - WORKFLOW_STARTED_MS) / 1000;
  if (elapsedS >= MEGA_TIME_BUDGET_S - MEGA_FINAL_RESERVE_S) {
    stopReason = `${MEGA_PROFILE} dispatch deadline reached after ${Math.round(elapsedS / 60)} minutes; ` +
      `preserving ${Math.round(MEGA_FINAL_RESERVE_S / 60)} minutes for finalist validation.`;
    return { stop: true };
  }
  if (!megaMeasurementCalibration.ready &&
      (currentRound === 2 || (currentRound > 2 && currentRound % 3 === 1))) {
    await recoverMegaCalibration(`before-round-${currentRound}`);
  }
  return { pool: MEGA_STRUCTURAL_ONLY
    ? null : await samplePool(currentRound, MEGA_PRODUCTION ? 120000 : 0) };
}

// mode=mega after the lane's plan call: the direction contract. Returns {directions}, {skip} for a refused
// plan (re-plan next round) or {stop}.
function megaPlanStep(currentRound, remaining, rawPlan) {
  const plan = megaPlanRound(currentRound, remaining, rawPlan);
  if (plan.refused && !plan.stop) {
    const count = megaRefusal ? megaRefusal.count + 1 : 1;
    megaRefusal = { reason: plan.reason, count };
    dispatched += 1;
    megaStall += 1;
    log(`Mega round ${currentRound}: ${plan.reason} (refusal ${count}/3; re-planning).`);
    if (count >= 3) {
      stopReason = `${count} consecutive refused directions; last: ${plan.reason}`;
      return { stop: true };
    }
    return { skip: true };
  }
  // A refused extra direction still reaches the next plan; it does not count toward the stop.
  megaRefusal = plan.refusal ? { reason: plan.refusal, count: 0 } : null;
  if (plan.stop) {
    stopReason = plan.reason;
    return { stop: true };
  }
  megaRoundStartMs = MEGA_CLOCK_MS;
  return { directions: plan.directions };
}

// mode=mega stop rules, checked at the end of the lane round.
function megaStopRule(currentRound, done, selected) {
  for (const t of done) {
    log(`Mega round ${round}: candidate=${t.record.id}[${t.record.source}:${t.record.status}], ` +
      `score=${Number.isFinite(t.record.absolute_score) ? t.record.absolute_score.toFixed(4) + 'x' : 'pending'}, ` +
      `selected=${selected ? selected.id : 'none'}, calibration=` +
      `${megaMeasurementCalibration.ready ? 'ready' : 'pending'}, budget=${dispatched}/${BUDGET}.`);
  }
  const structuralDone = done.find((t) => t.record.structural_verified);
  if (MEGA_STRUCTURAL_ONLY && structuralDone) {
    stopReason = `structural-only acceptance reached by ${structuralDone.record.id}: source checkpoint and ` +
      `MegaPlanIR consistency passed; hardware/accuracy/liveness/performance remain unverified.`;
    return true;
  }
  const searchAttempts = megaCandidateRegistry
    .filter((c) => c.source === 'search' || c.source === 'integrated')
    .reduce((sum, c) => sum + Number(c.attempts || 0), 0);
  const expertTargetLaunches = Number(
    analysis && analysis.mega_plan_ir && analysis.mega_plan_ir.target &&
      analysis.mega_plan_ir.target.launch_count
  );
  const expertTargetConfigured = USE_EXPERT_SKILLS &&
    Number.isFinite(EXPERT_SKILL_RECORDED_LOW) &&
    Number.isFinite(expertTargetLaunches);
  const expertTargetReached = !expertTargetConfigured || (
    selected &&
    Number(selected.absolute_score) >= Number(EXPERT_SKILL_RECORDED_LOW) &&
    Number(selected.launches) === expertTargetLaunches
  );
  if (MEGA_STOP_ON_DELIVERABLE && selected &&
      Number(selected.absolute_score) > 1.0 &&
      expertTargetReached &&
      searchAttempts >= MEGA_MIN_SEARCH_ATTEMPTS) {
    stopReason = `production early-stop after ${round} round(s): candidate ${selected.id} ` +
      `has calibrated ${Number(selected.absolute_score).toFixed(4)}x speedup over frozen ` +
      `baseline${expertTargetConfigured ? `, reached Expert Skill target ` +
        `${Number(EXPERT_SKILL_RECORDED_LOW).toFixed(4)}x at ${expertTargetLaunches} launches` : ''}, ` +
      `and ${searchAttempts} search attempt(s) have run; reserve the remaining ` +
      `${BUDGET - dispatched} direction(s) for finalist validation instead of exhaustive search.`;
    return true;
  }
  if (REQUIRE_HW_ACTIVATION && noHardware >= MAX_NO_HARDWARE) {
    stopReason = `${noHardware} consecutive portfolio attempts did not execute the candidate path ` +
      `on hardware; stopping before another unmeasured direction. Lane WIP remains resumable.`;
    log(`Mega hardware stop: ${stopReason}`);
    return true;
  }
  if (MEGA_STALL_LIMIT && megaStall >= 2 * MEGA_STALL_LIMIT) {
    stopReason = `${megaStall} consecutive rounds produced no new runtime evidence (independent ` +
      'Verify, better score, or a changed on-card blocker); lane WIP remains resumable.';
    log(`Mega stall stop: ${stopReason}`);
    return true;
  }
  return false;
}

async function megaAfterLoop() {

  // Record which stop fired, in loop order: the budget counts directions, not GPU leases, so the
  // artifacts alone cannot tell "budget spent" from "hardware left".
  if (!stopReason) {
    const left = BUDGET - dispatched;
    if (dispatched >= BUDGET) {
      stopReason = `the direction budget was exhausted (${dispatched}/${BUDGET} charged over ${round} round(s)).`;
    } else if (!WORKING_KERNEL && noEvidence >= MAX_NO_IMPROVE) {
      stopReason = `${noEvidence} consecutive round(s) produced no admissible measurement, which is the ` +
        `max_no_improve cap (${MAX_NO_IMPROVE}). ${left} of ${BUDGET} budget unit(s) were left unspent.`;
    } else if (!WORKING_KERNEL && noImprove >= MAX_NO_IMPROVE) {
      stopReason = `${noImprove} round(s) in a row measured a candidate and none beat the current best by ` +
        `more than ${(MIN_IMPROVE * 100).toFixed(0)}%, which is the max_no_improve cap (${MAX_NO_IMPROVE}). ` +
        `${left} of ${BUDGET} budget unit(s) were left unspent.`;
    } else {
      stopReason = `the loop ended after ${round} round(s) with ${left} of ${BUDGET} budget unit(s) unspent ` +
        'and no stop condition recorded.';
    }
  }
  log(`Rounds ended: ${stopReason}`);

  if (megaUnsafeTimeout) {
    log(`MEGA SAFE STOP: ${megaUnsafeTimeout.label || 'agent'} may still be running after the ` +
      `orchestration timeout.`);
    return { exit: {
      mode: MODE,
      mega_deliverable: false,
      mega_selected_candidate: '',
      mega_selected_source: '',
      mega_candidate_registry: megaCandidateRegistry,
      mega_note: 'agent timeout: candidate lane WIP is preserved, but no overlapping work was launched',
      eval_dir: EVAL_DIR,
      kernel_name: KERNEL_NAME,
      final_speedup: 0,
      final_geomean: 0,
      validation_status: 'agent_timeout',
      rounds: round,
      budget_used: dispatched,
      budget_total: BUDGET,
      stop_reason: stopReason,
      report_path: '',
      final_patch: '',
    } };
  }
  FINAL_WORKSPACE = CANONICAL;
  if (!megaMeasurementCalibration.ready) await recoverMegaCalibration('before-final-selection');
  const provisional = selectMegaCandidate(megaCandidateRegistry, MEGA_TIE_NOISE_PCT);
  const orderedEligible = provisional.selected
    ? [provisional.selected, ...provisional.eligible.filter((c) => c.id !== provisional.selected.id)]
    : provisional.eligible;
  ranked = orderedEligible.slice(0,
    MEGA_PRODUCTION ? MEGA_FINAL_FALLBACK_K : MEGA_FINAL_TOP_K);
  const finalistOrder = ranked;
  let selectionVerdict = megaFinalSelectionVerdict(null, [], {
    selectedWorkspace: `${EVAL_DIR}/mega_selected`,
  });
  if (MEGA_PRODUCTION) {
    for (let i = 0; i < finalistOrder.length; i++) {
      const batch = [finalistOrder[i]];
      const attempt = await runFinalistBatch(batch, `director:mega-finalist-${i + 1}`);
      if (attempt && attempt.__agent_timed_out) {
        megaUnsafeTimeout = attempt;
        selectionVerdict = megaFinalSelectionVerdict(null, batch, {
          selectedWorkspace: `${EVAL_DIR}/mega_selected`,
        });
        finalists = batch;
        break;
      }
      const verdict = megaFinalSelectionVerdict(attempt, batch, {
        selectedWorkspace: `${EVAL_DIR}/mega_selected`,
        launchTarget: LAUNCH_TARGET, worldSize: GPU_RESOURCE.gpusPerJob,
        allowPartialFusion: ALLOW_PARTIAL_FUSION,
        baselineLaunches: optionalNumber(A.baseline_launch_count),
        frozenBaseline: BASELINE_PER_CASE,
        targetGuards: TARGET_GUARDS, regressionGuards: REGRESSION_GUARDS,
        requiredReplays: REQUIRED_REPLAYS, requiredPairs: REQUIRED_PAIRS,
        requiredPairsByGuard: REQUIRED_PAIRS_BY_GUARD,
        accuracyMetric: ACCURACY_METRIC, accuracyThreshold: ACCURACY_THRESHOLD,
        requireOverlap: REQUIRE_OVERLAP, requireAttribution: REQUIRE_ATTRIBUTION,
        requireArrivalJitter: ARRIVAL_JITTER_CONTRACT,
        tieNoisePct: MEGA_TIE_NOISE_PCT,
      });
      finalists = batch;
      selectionVerdict = verdict;
      if (verdict.pass) {
        const identity = await agentT(
          roleAgent('director', 'check_mega_identity',
            'Independently compare the selected materialized tree with the exact source candidate head; do not run GPU work.', {
              EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
              SELECTED_CANDIDATE_ID: finalistOrder[i].id,
              SELECTED_SOURCE_TREE: finalistOrder[i].tree,
              SELECTED_SOURCE_HEAD: finalistOrder[i].head,
              SELECTED_WORKSPACE: `${EVAL_DIR}/mega_selected`,
              SELECTED_MATERIALIZED_HEAD: attempt.selected_head,
            }),
          { phase: 'Validate', label: `director:mega-identity-${i + 1}`,
            schema: MEGA_IDENTITY_SCHEMA, timeout_ms: 60000,
            timeout_marker: true, max_retries: 1 });
        if (identity && identity.__agent_timed_out) {
          megaUnsafeTimeout = identity;
          selectionVerdict = { pass: false, reasons: ['independent materialization identity timed out'] };
          break;
        }
        if (!(identity && identity.passed === true &&
            identity.selected_candidate_id === finalistOrder[i].id &&
            sameCommit(identity.source_head, finalistOrder[i].head) &&
            sameCommit(identity.materialized_head, attempt.selected_head))) {
          log(`MEGA production finalist ${finalistOrder[i].id} materialization identity failed. ` +
            `${i + 1 < finalistOrder.length ? 'Trying the next scored fallback.' : 'No fallback remains.'}`);
          selectionVerdict = { pass: false, reasons: ['independent materialization identity failed'] };
          continue;
        }
        megaSelection = attempt;
        break;
      }
      log(`MEGA production finalist ${finalistOrder[i].id} failed: ${verdict.reasons.join('; ')}. ` +
        `${i + 1 < finalistOrder.length ? 'Trying the next scored fallback.' : 'No fallback remains.'}`);
    }
  } else {
    finalists = finalistOrder;
    megaSelection = await runFinalistBatch(finalists, 'director:mega-finalist-selection');
    if (megaSelection && megaSelection.__agent_timed_out) {
      megaUnsafeTimeout = megaSelection;
      megaSelection = null;
    }
    selectionVerdict = megaFinalSelectionVerdict(megaSelection, finalists, {
      selectedWorkspace: `${EVAL_DIR}/mega_selected`,
      launchTarget: LAUNCH_TARGET, worldSize: GPU_RESOURCE.gpusPerJob,
      allowPartialFusion: ALLOW_PARTIAL_FUSION,
      baselineLaunches: optionalNumber(A.baseline_launch_count),
      frozenBaseline: BASELINE_PER_CASE,
      targetGuards: TARGET_GUARDS, regressionGuards: REGRESSION_GUARDS,
      requiredReplays: REQUIRED_REPLAYS, requiredPairs: REQUIRED_PAIRS,
      requiredPairsByGuard: REQUIRED_PAIRS_BY_GUARD,
      accuracyMetric: ACCURACY_METRIC, accuracyThreshold: ACCURACY_THRESHOLD,
      requireOverlap: REQUIRE_OVERLAP, requireAttribution: REQUIRE_ATTRIBUTION,
      requireArrivalJitter: ARRIVAL_JITTER_CONTRACT,
      tieNoisePct: MEGA_TIE_NOISE_PCT,
    });
  }
  if (selectionVerdict.pass) {
    megaSelectionContractPassed = true;
    FINAL_WORKSPACE = megaSelection.selected_tree;
    const selectedRecord = selectionVerdict.source;
    const finalRow = selectionVerdict.row;
    const finalTargetReadout = megaTargetsReadout(finalRow.paired_readings, TARGET_GUARDS);
    selectedRecord.status = 'finalist';
    selectedRecord.absolute_score = Number(finalTargetReadout.score);
    selectedRecord.per_case = Array.isArray(finalRow.per_case) ? finalRow.per_case : [];
    selectedRecord.claim_complete = true;
    selectedRecord.attempt_id = String(megaSelection.attempt_id || '');
    selectedRecord.evidence_manifest = String(megaSelection.evidence_manifest || '');
    selectedRecord.paired_readings = Array.isArray(finalRow.paired_readings)
      ? finalRow.paired_readings : [];
    selectedRecord.null_arm_pct = Number(finalRow.null_arm_pct);
    selectedRecord.reps = finalTargetReadout.count;
    selectedRecord.artifact_hash_base = String(finalRow.artifact_hash_base || '');
    selectedRecord.artifact_hash_candidate = String(finalRow.artifact_hash_candidate || '');
    selectedRecord.correctness_pass = true;
    selectedRecord.activation_pass = true;
    selectedRecord.head_pass = true;
    selectedRecord.launch_pass = true;
    selectedRecord.liveness_pass = true;
    selectedRecord.graph_pass = true;
    selectedRecord.guards_pass = true;
    selectedRecord.artifact_distinct = true;
    selectedRecord.measurement_pass = true;
    selectedRecord.final_tree = megaSelection.selected_tree;
    selectedRecord.final_head = String(megaSelection.selected_head || '');
    selectedRecord.final_attempt_id = String(megaSelection.attempt_id || '');
    selectedRecord.final_evidence_manifest = String(megaSelection.evidence_manifest || '');
    megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, selectedRecord);
    cumulative = Number(megaSelection.selected_score) || cumulative;
    finalWinner = {
      id: megaSelection.selected_candidate_id,
      source: megaSelection.selected_source,
      geomean: cumulative,
      tree: FINAL_WORKSPACE,
      source_tree: selectedRecord.tree,
      source_head: selectedRecord.head,
      materialized_head: megaSelection.selected_head,
      patch: megaSelection.selected_patch || '',
      provenance: selectedRecord ? selectedRecord.provenance : megaSelection.selected_source,
      topology_sig: selectedRecord ? selectedRecord.topology_sig : '',
      activation: selectedRecord ? selectedRecord.activation : {},
      per_case: selectedRecord.per_case,
    };
    log(`MEGA FINALIST SELECTED: ${finalWinner.id}[${finalWinner.source}] ${cumulative.toFixed(4)}x.`);
  } else {
    const why = !megaMeasurementCalibration.ready
      ? 'measurement calibration is incomplete'
      : (!finalistOrder.length
        ? 'no workflow-authored candidate passed the score-tier contract'
        : `final selection contract failed: ${selectionVerdict.reasons.join('; ')}`);
    ACCEPTANCE_CAVEATS.push(`Mega finalist selection unavailable: ${why}.`);
    log(`MEGA FINALIST SELECTION INCOMPLETE: ${why}; no performance result will be presented as a deliverable.`);
    finalWinner = null;
  }

  if (megaUnsafeTimeout) {
    log(`MEGA SAFE STOP: finalist agent ${megaUnsafeTimeout.label || ''} timed out; no fallback or ` +
      `second validation starts while that agent may still own resources.`);
    return { exit: {
      mode: MODE, mega_deliverable: false, mega_selected_candidate: '',
      mega_selected_source: '', mega_candidate_registry: megaCandidateRegistry,
      mega_note: 'finalist timeout: no overlapping fallback was launched',
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME, final_speedup: 0, final_geomean: 0,
      validation_status: 'agent_timeout', rounds: round, budget_used: dispatched,
      budget_total: BUDGET, stop_reason: 'finalist agent timeout', report_path: '', final_patch: '',
    } };
  }
}

function megaFinalStepOpts() {
  return MEGA_PRODUCTION ? { timeout_ms: 300000, timeout_marker: true, max_retries: 1 } : {};
}

function megaReportInputs() {
  return {
    EVAL_DIR, WORKSPACE: FINAL_WORKSPACE, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    ...({
      MEGA_CANDIDATE_REGISTRY: megaCandidateRegistry,
      MEGA_SELECTION: megaSelection || {},
    }),
    PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
    LAUNCH_TARGET, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
    REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    ACCURACY_METRIC, ACCURACY_THRESHOLD, AUTONOMY_ACCEPTANCE_REACHED,
    ...(ACCEPTANCE_CAVEATS.length ? { ACCEPTANCE_CAVEATS } : {}),
    STOP_REASON: stopReason,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
    PROFILE_SUMMARY: profileSummary,
    PRIOR_ART_SWEEP: PRIOR_ART_RECORDED ? JSON.stringify(PRIOR_ART) : 'UNRECORDED',
    ...(PC_OVERSHOOT ? { POSITIVE_CONTROL_OVERSHOOT: PC_OVERSHOOT } : {}),
    ...(PC_UNDERSHOOT ? { POSITIVE_CONTROL_UNDERSHOOT: PC_UNDERSHOOT } : {}),
    ...(SHELF_STATS.rounds_offered || SHELF_STATS.shelved ? { SHELF_ACTIVITY: SHELF_STATS } : {}),
    ...(OVERLAP_CAVEATS.length ? { OVERLAP_CAVEATS } : {}),
    ...(LAUNCH_CAVEATS.length ? { LAUNCH_CAVEATS, TWO_LAUNCH_REACHED } : {}),
    ...(GUARD_CAVEATS.length ? { GUARD_CAVEATS } : {}),
    ...(ATTRIBUTION_CAVEATS.length ? { ATTRIBUTION_CAVEATS } : {}),
    ...(CHAIN_CAVEATS.length ? { CHAIN_CAVEATS } : {}),
    ...(WORKING_KERNEL ? { OBJECTIVE, OBJECTIVE_NOTE:
      'This wave was run for a WORKING FUSED ARTIFACT, not for speed. One direction per round, the ' +
      'commit gate was "does it run", and the no-improve stop was disabled because a debug round is ' +
      'non-improving by construction. Do NOT read a speedup out of this wave' +
      (PC_RAN ? '.' : ' — it ran no positive control, so every timing reading in it is void.') } : {}),
    ...(OBJECTIVE_CAVEATS.length ? { OBJECTIVE_CAVEATS } : {}),
    ...(KNOWLEDGE_DELTA.length ? { KNOWLEDGE_DELTA, KNOWLEDGE_DELTA_NOTE:
      'Facts this wave established about the HARDWARE/TOOLCHAIN, not about this kernel. Give them ' +
      'their own section titled "Knowledge deltas — merge into knowledge/". List the generalizing ' +
      'ones first with their target card and their evidence verbatim; list the operator-specific ' +
      'ones under a separate heading and say plainly that they are not card material. This workflow ' +
      'does not edit knowledge/ itself.' } : {}),
    ...(TASK_GRAPH_CAVEAT ? { TASK_GRAPH_CAVEAT } : {}),
    ...(PIPE_TABLE_CAVEAT ? { PIPE_TABLE_CAVEAT } : {}),
    ...(LADDER.length ? { ROADMAP_LADDER_FINAL: JSON.stringify(
      roadmapLadderGate(LADDER, [], LADDER_MEASURED)).slice(0, 3000) } : {}),
    ...(REQUIRE_TASK_GRAPH && analysis && analysis.task_graph
      ? { TASK_GRAPH: JSON.stringify(analysis.task_graph).slice(0, 4000) } : {}),
    ...(REQUIRE_TASK_GRAPH && analysis && analysis.resource_timeline
      ? { RESOURCE_TIMELINE: JSON.stringify(analysis.resource_timeline).slice(0, 3000) } : {}),
  };
}

async function megaAfterReport() {
  reportReady = !!(report && !report.__agent_timed_out &&
    String(report.report_path || '') && String(report.final_patch || ''));
  if (report && report.__agent_timed_out) megaUnsafeTimeout = report;
  if (megaUnsafeTimeout) {
    return { exit: {
      mode: MODE, mega_deliverable: false, mega_selected_candidate: '',
      mega_selected_source: '', mega_candidate_registry: megaCandidateRegistry,
      mega_note: 'report timeout: no identity validation or state writer was started afterward',
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME, final_speedup: 0, final_geomean: 0,
      validation_status: 'agent_timeout', rounds: round, budget_used: dispatched,
      budget_total: BUDGET, stop_reason: 'report agent timeout', report_path: '', final_patch: '',
    } };
  }
}

// mode=mega: production runs validate only a reported, selected finalist.
function megaSkipsValidation() {
  return MEGA_PRODUCTION && !(reportReady && !!finalWinner && megaSelectionContractPassed);
}

function megaValidateInputs() {
  return {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: FINAL_WORKSPACE, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE,
    APPLY_TO_ORIGINAL, COMMANDMENT,
    FINAL_PATCH: reportReady ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
    LAUNCH_TARGET, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
    REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    ACCURACY_METRIC, ACCURACY_THRESHOLD, AUTONOMY_ACCEPTANCE_REACHED,
    WORKFLOW_REVISION_START,
    ...(finalWinner ? {
      VALIDATION_TIER: MEGA_PRODUCTION ? 'identity_only' : 'full',
      MEGA_SELECTION_EVIDENCE: megaSelection,
      SELECTED_SOURCE_TREE: finalWinner.source_tree,
      SELECTED_SOURCE_HEAD: finalWinner.source_head,
      SELECTED_MATERIALIZED_HEAD: finalWinner.materialized_head,
    } : {}),
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  };
}

async function megaFinish() {
  if (validation && validation.__agent_timed_out) {
    return {
      mode: MODE, mega_deliverable: false, mega_selected_candidate: '',
      mega_selected_source: '', mega_candidate_registry: megaCandidateRegistry,
      mega_note: 'identity validation timeout: no state writer was started afterward',
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME, final_speedup: 0, final_geomean: 0,
      validation_status: 'agent_timeout', rounds: round, budget_used: dispatched,
      budget_total: BUDGET, stop_reason: 'identity validation agent timeout',
      report_path: reportReady ? report.report_path : '', final_patch: reportReady ? report.final_patch : '',
    };
  }
  const aggregatePrimary = HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean;
  const finalPrimary = (TARGET_GUARDS.length || PROMOTION_METRIC === 'changed_kernel')
    ? promotionScore(validation || {}, PROMOTION_METRIC, TARGET_GUARDS,
      REGRESSION_GUARDS, aggregatePrimary)
    : aggregatePrimary;
  const finalGuards = guardContract(validation || {}, TARGET_GUARDS, REGRESSION_GUARDS, aggregatePrimary);
  const directorStatus = validation ? validation.validation_status : 'unknown';
  const finalAttribution = (validation && validation.attribution) || {};
  const scopedGuardFailure = finalMetricFailure(
    PROMOTION_METRIC, TARGET_GUARDS, finalPrimary, finalGuards, finalAttribution);
  // Scoped regression guards are vetoes in every mode.
  const finalValidationStatus = finalStatusVerdict(directorStatus, scopedGuardFailure);
  // Surface leaked lease jobs in the completion line: they keep running after the report and contend
  // with the measurements just accepted.
  if (validation && validation.orphan_leases_swept > 0) {
    log(`WARNING: ${validation.orphan_leases_swept} orphaned GPU lease(s) were still alive at closeout ` +
        `and have been killed.`);
  }
  const correctnessClean = directorStatus === 'accepted' &&
    reportReady &&
    String(validation && validation.correctness || '').toLowerCase().startsWith('pass') &&
    validation && validation.materialization_matches === true &&
    validation.final_patch_verified === true &&
    (APPLY_TO_ORIGINAL !== 'true' ||
      String(validation.applied_to_original || '').toLowerCase() === 'true') &&
    sameCommit(validation.selected_source_head, finalWinner && finalWinner.source_head) &&
    sameCommit(validation.selected_materialized_head, finalWinner && finalWinner.materialized_head) &&
    finalGuards && finalGuards.regression_pass !== false;
  mega_deliverable = !!(megaSelection && megaSelection.claim_complete === true &&
    megaSelectionContractPassed && finalWinner && correctnessClean && Number.isFinite(finalPrimary) &&
    finalPrimary > 1.0 && !scopedGuardFailure);
  mega_note = mega_deliverable
    ? `mega: selected ${finalWinner.id}[${finalWinner.source}] at ${finalPrimary.toFixed(3)}x vs ` +
      `the frozen baseline; all finalist gates passed.`
    : `mega: no fully verified finalist cleared the common correctness/guard/performance contract; ` +
      `workflow-authored candidate WIP remains recorded, but no result is presented as shipped.`;
  if (!mega_deliverable && finalWinner) {
    const rejectedFinal = megaCandidateById(finalWinner.id);
    if (rejectedFinal) {
      rejectedFinal.status = 'runnable';
      rejectedFinal.next_blocker =
        `independent final validation failed (${finalValidationStatus}); re-establish materialization, ` +
        `correctness and full guard evidence before finalist selection`;
      rejectedFinal.notes = `${rejectedFinal.notes || ''} FINAL VALIDATION: ${mega_note}`.trim();
      megaCandidateRegistry = upsertMegaCandidate(megaCandidateRegistry, rejectedFinal);
    }
    const fallback = selectMegaCandidate(megaCandidateRegistry, MEGA_TIE_NOISE_PCT).selected;
    cumulative = fallback ? fallback.absolute_score : 1.0;
    finalWinner = null;
  }
  const finalStateSaved = await persistMegaCandidateState(round, true);
  if (!finalStateSaved) {
    mega_deliverable = false;
    finalWinner = null;
    mega_note = 'mega: final candidate state could not be persisted; no result is presented as shipped.';
  }
  log(mega_note);

  log(`COMPLETE. ${KERNEL_NAME}: verified ${TARGET_GUARDS.length ? `target-guard ${PROMOTION_METRIC}` :
      (HAS_WORKLOAD ? 'time-weighted' : 'geomean')} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
      `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
      ` (status ${finalValidationStatus}${`, mega_deliverable=${mega_deliverable}, selected=${finalWinner ? finalWinner.id : 'none'}`}). Results in ${EVAL_DIR}`);

  return {
    mode: MODE,
    ...({
      mega_deliverable,
      mega_selected_candidate: finalWinner ? finalWinner.id : '',
      mega_selected_source: finalWinner ? finalWinner.source : '',
      mega_candidate_registry: megaCandidateRegistry,
      mega_note,
    }),
    target_language: undefined,
    authored: undefined,
    eval_dir: EVAL_DIR,
    kernel_name: KERNEL_NAME,
    workload_aligned: HAS_WORKLOAD,
    final_speedup: finalPrimary,                 // PRIMARY metric (weighted when workload-aligned)
    final_weighted: finalWeighted,
    final_geomean: finalGeomean,
    final_arithmetic: validation ? validation.director_verified_speedup_arithmetic : null,
    tech_lead_reported_geomean: report ? report.final_speedup_geomean : cumulative,
    validation_status: finalValidationStatus,
    autonomy_acceptance_reached: AUTONOMY_ACCEPTANCE_REACHED,
    autonomy_acceptance_confirmed: !!(validation && validation.autonomy_acceptance_confirmed === true &&
      validation.workflow_unchanged === true && directorStatus !== 'flagged' &&
      !scopedGuardFailure),
    acceptance_caveats: ACCEPTANCE_CAVEATS,
    rounds: report ? report.rounds : round,
    budget_used: dispatched,
    budget_total: BUDGET,
    stop_reason: stopReason,
    report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
    final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
  };
}
// ===========================================================================
// end of mode=mega backend
// ===========================================================================

// ===========================================================================
// PHASE: Setup
// ===========================================================================
phase('Setup');
const setup = await agentT(
  MEGA ? roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC,
    ...(STATE_DIR ? { STATE_DIR, MEGA_RESUME_STATE } : {}),
  }) : roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK,
    WORKFLOW_DIR, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC, EXPECTED_GFX, EXPECTED_TARGET,
    EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: MEGA ? MEGA_SETUP_SCHEMA : SETUP_SCHEMA,
    // 5min cut the mega director off while reproducing a grown prior_state; 15min cut an 18-row resume.
    ...(MEGA && MEGA_PRODUCTION ? { timeout_ms: 1800000, max_retries: 1 } : {}) });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const GFX = String(setup.device_gfx || '').trim().toLowerCase();
const DEVICE_TARGET = String(setup.device_target || '').trim().toLowerCase();
const SETUP_DEVICE_NAME = String(setup.device_name || '').trim();
const SETUP_PHYSICAL_CU_COUNT = Number(setup.physical_cu_count);
if (!MEGA && !GFX) {
  throw new Error('Setup failed: director did not return device_gfx (required; refuse CDNA knowledge without an ISA)');
}
if (!MEGA && !DEVICE_TARGET) {
  throw new Error('Setup failed: director did not return device_target (r9700 | r9600d | w7900 | unknown)');
}
if (!MEGA && (!Number.isFinite(SETUP_PHYSICAL_CU_COUNT) || SETUP_PHYSICAL_CU_COUNT <= 0)) {
  throw new Error('Setup failed: director did not return a positive physical_cu_count');
}
if (!MEGA && GFX === 'gfx1200') {
  throw new Error('Detected gfx1200, which is not validated by this workflow; refusing R9700 guidance');
}
if (!MEGA && GFX === 'gfx1100' && DEVICE_TARGET !== 'w7900') {
  throw new Error(
    `Detected gfx1100 with device_target=${DEVICE_TARGET}: only the exact Radeon Pro W7900 is ` +
    'validated on gfx1100; refusing RDNA3 guidance');
}
if (!MEGA && DEVICE_TARGET === 'w7900' && GFX !== 'gfx1100') {
  throw new Error(`Setup returned device_target=w7900 on ${GFX}: w7900 requires gfx1100`);
}
if (!MEGA && EXPECTED_GFX && GFX !== EXPECTED_GFX) {
  throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${GFX || 'unknown'}`);
}
if (!MEGA && EXPECTED_TARGET && DEVICE_TARGET !== EXPECTED_TARGET) {
  throw new Error(`GPU product mismatch: expected ${EXPECTED_TARGET}, detected ${DEVICE_TARGET || 'unknown'}`);
}
const normalizeDeviceName = (name) => String(name || '').trim().replace(/\s+/g, ' ').toLowerCase();
if (EXPECTED_DEVICE_NAME &&
    normalizeDeviceName(SETUP_DEVICE_NAME) !== normalizeDeviceName(EXPECTED_DEVICE_NAME)) {
  log(`Setup identity telemetry differs from the deterministic probe: device_name expected ` +
      `${EXPECTED_DEVICE_NAME}, Director returned ${SETUP_DEVICE_NAME || 'unknown'}; using expected value.`);
}
if (EXPECTED_PHYSICAL_CU_COUNT > 0 &&
    SETUP_PHYSICAL_CU_COUNT !== EXPECTED_PHYSICAL_CU_COUNT) {
  log(`Setup identity telemetry differs from the deterministic probe: physical_cu_count expected ` +
      `${EXPECTED_PHYSICAL_CU_COUNT}, Director returned ${SETUP_PHYSICAL_CU_COUNT}; using expected value.`);
}
const DEVICE_NAME = EXPECTED_DEVICE_NAME || SETUP_DEVICE_NAME;
const PHYSICAL_CU_COUNT = EXPECTED_PHYSICAL_CU_COUNT > 0
  ? EXPECTED_PHYSICAL_CU_COUNT : SETUP_PHYSICAL_CU_COUNT;
// Products with a `product:` section in e2e_workflow/knowledge/analysis_skills/roofline/peaks.md.
// The W7900's section holds AMD's published peaks, with no BF16, FP8 or INT4 entry; the roles say
// which axes that leaves.
const CALIBRATED_PRODUCTS = ['r9700', 'r9600d'];
// ISA rows of peaks.md that hold only at the CU count they were measured on: the 8050S / 8040S share
// gfx1151 with the 40-CU 8060S. Any other part of an isolated family is a hard unknown, as
// roofline_tools.resolve_peaks(gfx, cu=...) also returns.
const TABULATED_CU = { gfx1151: 40 };
const GPU_FAMILY = gpuFamily(GFX);
const ROOFLINE_STATUS = (CALIBRATED_PRODUCTS.includes(DEVICE_TARGET) && GFX === 'gfx1201')
  ? `calibrated-${DEVICE_TARGET}`
  : (DEVICE_TARGET === 'w7900' && GFX === 'gfx1100') ? 'calibrated-w7900'
  : /^gfx120/.test(GFX) ? 'unknown-device-not-r9700'
  : TABULATED_CU[GFX] === Number(PHYSICAL_CU_COUNT) ? `tabulated-${GFX}`
  : GPU_FAMILY.isolate ? `unknown-${GFX}-not-tabulated`
  : 'non-rdna';
// PROFILING is OFF by default; args.profile='on' restores it. Off: no baseline profile, no reprofile
// after a commit or a warm-start adopt, and neither the benchmark engineer nor the engineers run the
// profiler. On a prepared extract UT one profiled run costs 20-40 min, and on the AgentX stacks
// rocprofv3 aborts outright (LLVM option clash with FlyDSL), so every profile agent spent 10-30 min
// on the fault ladder before falling back (GEAK #606). Every consumer already tolerates a null summary.
const PROFILE_ON = String(A.profile != null ? A.profile : 'off').trim().toLowerCase() === 'on';
// gfx1100 counters are valid only at the profile_standard level, which the operator sets and declares
// in gpu_perf_level (Hyperloom records it as stable_std; amd-smi prints AMDSMI_DEV_PERF_LEVEL_STABLE_STD).
// With profiling off no agent collects counters, so the safe set also needs PROFILE_ON.
// The key is sent only on the W7900, so every other prompt stays byte-identical.
const GPU_PERF_LEVEL = String(A.gpu_perf_level || '').trim().toLowerCase().replace(/^amdsmi_dev_perf_level_/, '');
const AT_PROFILE_STANDARD = GPU_PERF_LEVEL === 'profile_standard' || GPU_PERF_LEVEL === 'stable_std';
const COUNTER_INPUTS = DEVICE_TARGET === 'w7900'
  ? { COUNTER_GUIDANCE: PROFILE_ON && AT_PROFILE_STANDARD ? 'gfx1100-safe-set' : 'unavailable' } : {};
if (GPU_FAMILY.isolate) {
  // Until the perf/learned stores implement metadata-enforced gens filtering, fail closed:
  // an isolating family (GPU_FAMILIES) gets the dedicated in-tree hardware guide but no CDNA-biased cards.
  USE_EXPERT_SKILLS = false;
  KERNEL_KNOWLEDGE_DIR = '';
  USE_LEARNED_READ = false;
  WARM_START_ON = false;
}
if (!MEGA && !GPU_RESOURCE.explicit && setup.gpu_requirement) {
  applyGpuResource(resolveGpuRequest(A, GPU_LIST, setup.gpu_requirement));
  if (GPU_RESOURCE.warning) log(`GPU requirement WARNING: ${GPU_RESOURCE.warning}`);
  if (GPU_JOB) {
    log(`GPU requirement: ${GPU_RESOURCE.gpusPerJob} GPUs per job (${setup.gpu_requirement.kind}: ` +
      `${setup.gpu_requirement.evidence || 'no evidence'}) -> ${GPU_JOB}`);
  }
}
// The GPU count is final from here. A UT-module multi-GPU task (UT_MR: more than one GPU, and the task
// itself declares more than one) owes each rank's time and the alignment method for every timed case
// (A01): Benchmark, Report and Validate switch to the *_MR schemas and get GPUS_PER_JOB. After Setup
// every GPU command carries the UT's whole-run limit as GEAK_GPU_RUN_TIMEOUT, and the agent wait limit
// covers up to 3 full UT runs + 2 (A22; see AGENT_WAIT_MS). Any other job adds nothing: prompts, schemas, limits
// and the return are unchanged.
const RANKS = GPU_RESOURCE.gpusPerJob;
const DECLARATION = setup.task_declaration && typeof setup.task_declaration === 'object' ? setup.task_declaration : {};
const DECLARED_GPUS = Number.isInteger(DECLARATION.required_gpu_count) ? DECLARATION.required_gpu_count : 0;
UT_MR = !MEGA && RANKS > 1 && DECLARED_GPUS > 1;
if (UT_MR) {
  const total = DECLARATION.ut_total_s;
  const usable = typeof total === 'number' && Number.isFinite(total) && total > 0;
  UT_TOTAL_S = usable ? total : 3600;
  if (AGENT_TIMEOUT_MS > 0) AGENT_WAIT_MS = Math.max(AGENT_TIMEOUT_MS, (3 + 2) * UT_TOTAL_S * 1000);
  const evidence = Array.isArray(DECLARATION.evidence) ? DECLARATION.evidence.join('; ') : '';
  const limitNote = usable ? ''
    : ` (task_declaration.ut_total_s ${JSON.stringify(total)} is not a positive number: multi-rank default)`;
  const waitNote = AGENT_WAIT_MS > 0
    ? `${AGENT_WAIT_MS} ms = max(agent_timeout_ms ${AGENT_TIMEOUT_MS}, (3 + 2) x ${UT_TOTAL_S} s)`
    : `off (agent_timeout_ms ${AGENT_TIMEOUT_MS})`;
  log(`UT-module multi-GPU task: ${RANKS} GPUs per job; the task declares ${DECLARED_GPUS} ` +
    `(${evidence || 'no evidence'}). Whole-UT limit ${UT_TOTAL_S} s${limitNote} -> ` +
    `GEAK_GPU_RUN_TIMEOUT=${UT_TOTAL_S} on every GPU command; agent wait limit ${waitNote}.`);
}
const MR_INPUTS = UT_MR ? { GPUS_PER_JOB: RANKS } : {};
// `_task` is the e2e head's DIRECTORY suffix, and the director returns the basename verbatim, so it
// would otherwise ride into the session id and the stored record. The canonical id is folded store-
// side (experience_store.remote_identity); this keeps the rest of the run calling it one name.
// A note the Director appends ("<name> (triton, author mode; ...)") is dropped the same way.
const KERNEL_NAME = MEGA ? setup.kernel_name : String(setup.kernel_name || '').trim().split(/[\s(;]/)[0]
  .replace(/_task$/, '') || setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);
if (MEGA) {
  const r = await megaAfterSetup();
  if (r) return r.exit;
}

// ---------------------------------------------------------------------------
// Enforce a FROZEN REAL-ONLINE BASELINE in BOTH modes (author AND same-language optimize).
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
// PHASE: Author (mode=author only) — write a fresh from-scratch impl as the optimize loop's CODE SEED.
// ===========================================================================
if (MODE === 'author') {
  phase('Author');
  const authored = await agentT(
    roleAgent('author_engineer', 'author', 'Write the simplest correct baseline in the target language.', {
      TARGET_LANGUAGE, OP_SPEC, WORKSPACE: CANONICAL, TASK_DIR: KERNEL_PATH_ORIG,
      GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_KNOWLEDGE_DIR,
      // The author engineer reads the learned index too, so the switch has to reach it.
      LEARNED_KB: USE_LEARNED_READ ? 'on' : 'off',
    }),
    { phase: 'Author', label: `author:${TARGET_LANGUAGE}`, schema: AUTHOR_SCHEMA });
  if (!authored || !authored.authored || !says(authored.correctness, 'pass')) {
    log(`Author mode FAILED for ${TARGET_LANGUAGE}: ${authored ? authored.notes || authored.correctness : 'no result'}. Aborting (no seed to optimize).`);
    if (!authored) await stopLaneWork(EVAL_DIR, `author:${TARGET_LANGUAGE} returned nothing`, 'Author');
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
let analysis = await agentT(
  roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', MEGA ? megaAnalyzeInputs() : {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR, GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...COUNTER_INPUTS,
    ...RESUME_INPUT,
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: MEGA ? MEGA_ANALYZE_SCHEMA : ANALYZE_SCHEMA,
    // mode=mega: 30 min; the task-graph analysis took 13.5 min fresh and was cut at 15 min on a resume.
    ...(MEGA && MEGA_PRODUCTION ? { timeout_ms: 1800000, timeout_marker: true, max_retries: 1 } : {}) });
if (MEGA) {
  const r = await megaAfterAnalyze();
  if (r) return r.exit;
}
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

// perf_knowledge pointers resolved by the TechLead in analyze (REFERENCE ONLY; threaded to the planner
// + engineers so they read focused op/language cards instead of the whole base).
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || '';
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];

// ===========================================================================
// PHASE: Benchmark setup (Benchmark Engineer)
// ===========================================================================
phase('Benchmark');
let bench = MEGA && noGpuFrontMatter ? megaStructuralBench() : await agentT(
  roleAgent('benchmark_engineer', 'setup', 'Build the COMMANDMENT and record a reliable baseline.', MEGA ? megaBenchInputs() : {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    ANALYSIS: analysis, PROFILING: PROFILE_ON ? 'on' : 'off',
    ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
    ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
    ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
    ...MR_INPUTS,
  }),
  { phase: 'Benchmark', label: 'benchmark_engineer', schema: MEGA ? MEGA_BENCH_SCHEMA : UT_MR ? BENCH_SCHEMA_MR : BENCH_SCHEMA,
    // Baseline/control setup has a separate production window and can recover artifacts on timeout.
    ...(MEGA && MEGA_PRODUCTION ? { timeout_ms: 10800000, timeout_marker: true, max_retries: 1 } : {}) });
if (MEGA) {
  const r = await megaAfterBenchmark(bench);
  if (r) return r.exit;
}
if (!bench || !bench.baseline_per_case) {
  await stopLaneWork(EVAL_DIR, 'benchmark_engineer returned no baseline', 'Benchmark');
  throw new Error('Benchmark setup failed: no baseline recorded');
}
const BASELINE_PER_CASE = bench.baseline_per_case;
const BASELINE_GEOMEAN_MS = bench.baseline_geomean_ms;
// A multi-rank baseline needs a merged time per case; each rank's time and the alignment method are
// kept when the UT reports them, and their absence is only noted (the merged time is the baseline).
const BENCH_RANK_GAP = UT_MR ? benchRankGap(BASELINE_PER_CASE) : '';
const BENCH_RANK_NOTE = UT_MR && !BENCH_RANK_GAP ? benchRankNote(BASELINE_PER_CASE, RANKS) : '';
if (BENCH_RANK_GAP) {
  bench.reliable = false;
  bench.notes = `${bench.notes ? `${bench.notes} ` : ''}[lane] multi-GPU baseline without a merged time: ` +
    `${BENCH_RANK_GAP}.`;
  log(`Benchmark WARNING: ${RANKS}-GPU job but ${BENCH_RANK_GAP}; baseline marked unreliable.`);
} else if (BENCH_RANK_NOTE) {
  bench.notes = `${bench.notes ? `${bench.notes} ` : ''}[lane] multi-GPU baseline uses the merged time; ` +
    `${BENCH_RANK_NOTE}.`;
  log(`Benchmark note: ${RANKS}-GPU job, ${BENCH_RANK_NOTE}; merged times used.`);
}
log(`Benchmark done. ${bench.num_test_cases || BASELINE_PER_CASE.length} cases, baseline geomean ${BASELINE_GEOMEAN_MS} ms, reliable=${bench.reliable}`);
if (MEGA) {
  const r = await megaCalibrate();
  if (r) return r.exit;
}

// ===========================================================================
// PHASE: Baseline profile (Profile Engineer) — only with args.profile='on' (see PROFILE_ON).
// ===========================================================================
phase('Profile');
// mode=mega profiles per stage (MEGA_MEASUREMENT_GUIDE §1) when the guide is given and GPUs are used.
let profileSummary = MEGA ? megaProfileStub() : null;
const baselineProfile = !(MEGA ? GMI.MEGA_MEASUREMENT_GUIDE && !noGpuFrontMatter : PROFILE_ON) ? null : await agentT(
  MEGA ? roleAgent('profile_engineer', 'mega_baseline',
    'Measure the frozen baseline per stage: follow MEGA_MEASUREMENT_GUIDE §1, not the role file.', {
      WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE,
      COMMANDMENT, BASELINE_PER_CASE, TARGET_GUARDS, REGRESSION_GUARDS, ...GMI, ...MRI,
    })
  : roleAgent('profile_engineer', 'baseline', 'Profile the baseline and classify the bottleneck.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: 0,
    COMMANDMENT, GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...COUNTER_INPUTS,
    ...RESUME_INPUT,
  }),
  MEGA ? { phase: 'Profile', label: 'profile_engineer:mega_baseline', schema: PROFILE_SCHEMA,
    timeout_ms: 3600000, max_retries: 1 }
  : { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
if (MEGA) {
  const r = await megaAfterProfile(baselineProfile);
  if (r) return r.exit;
} else profileSummary = baselineProfile;
log(PROFILE_ON
  ? `Baseline bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'} (dispatch_count=${profileSummary ? profileSummary.dispatch_count : '?'})`
  : 'Profile: skipped (args.profile is off).');

// ===========================================================================
// PHASE: Research (Deep Research Agent — OPT-IN via args.dra_enabled) Runs AFTER Profile / BEFORE the
// optimize loop: profile + COMMANDMENT + analysis exist by now, so the researcher has the facts it
// needs.
// ===========================================================================
let researchBriefPath = '';   // EVAL_DIR/deep_search_brief.md when the DRA produced one; '' otherwise
if (DRA_ENABLED && !MEGA) {
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
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source} — also set by a warm-start adopt
let roundsCommitted = 0;     // rounds this run actually landed; a warm-start adopt is NOT one of them
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };
// One row per KB-seeded direction, joined against what the verifier measured (see the push site in the
// round loop).
const citations = [];

// DEEP-MODE resume: restore cumulative speedup + insight/ledger history from the prior wave so this
// continuation builds ON the cumulative best (canonical was already seeded from STATE_DIR/best by the
// director) and does not re-explore dead directions.
if (!MEGA && setup.resumed && setup.prior_state) {
  const ps = setup.prior_state;
  if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
  if (Array.isArray(ps.insights)) history.insights = ps.insights;
  if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
  if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
  if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
}

// ===========================================================================
// PHASE: WarmStart.
// ===========================================================================
// One benched candidate -> one of kb/attest.py's four outcomes.
const kbVerdict = (ver, sp) => {
  if (!ver) return 'not_reproduced';
  if (says(ver.correctness, 'fail')) return 'failed';
  if (says(ver.status, 'apply_failed')) {
    return /outside_editable_set/i.test(String(ver.notes || '')) ? 'inapplicable' : 'not_reproduced';
  }
  if (!says(ver.status, 'verified')) return 'not_reproduced';
  if (!(sp > 1.0)) return 'failed';              // applied, ran, correct, just did not go faster
  return says(ver.correctness, 'pass') ? 'inapplicable' : 'not_reproduced';   // fast but never adopted
};
// mode=mega seeds no global score from authoring WIP: each candidate lane advances on its own and
// cumulative moves only when a fully scored candidate wins. megaLoopInit restores its registry.
if (MEGA) {
  const r = await megaLoopInit();
  if (r) return r.exit;
}
let warm_start = { adopted: false, read_reason: WARM_START_ON ? 'read' : 'disabled', candidates: [] };
let skipLoop = false;
if (!MEGA && WARM_START_ON && !setup.resumed && KB_ROOT_OK) {
  phase('WarmStart');
  if (!GFX) {
    warm_start.read_reason = 'missing_arch';
    log('[kb] warm-start skipped: no gfx detected from the baseline profile device string.');
  } else {
    // The two planes take a different root flag and a different name→page rule (the store addresses
    // by canonical id, so there is nothing to match fuzzily), and print the same JSON.
    const localResolveCmd = KB_MODE === 'store'
      ? `resolve-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : `resolve --root ${JSON.stringify(KB_ARTIFACTS_DIR)} --match ${WARM_START_MATCH}`;
    // Bring the local store level with the tree before reading it, scoped to this kernel and arch so it
    // costs a stat walk over one page.
    //
    // `|| true`: a sync that cannot run must degrade to a thinner page, never a failed warm start.
    const syncCmd = KB_MODE === 'store' && KB_ARTIFACTS_DIR
      ? `python3 ${JSON.stringify(EXPERIENCE_STORE)} sync-local --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --store ${JSON.stringify(KB_STORE_DIR)} --kernel-name ${JSON.stringify(KERNEL_NAME)} --gfx ${GFX} \\
  > ${JSON.stringify(EVAL_DIR + '/kb_sync.json')} 2>&1 || true\n`
      : '';
    const commonArgs =
      `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\\n` +
      `  --gfx ${GFX} --top-n 3 --min-speedup ${WARM_START_MIN_SPEEDUP} \\\n` +
      `  --refs-dir ${JSON.stringify(EVAL_DIR + '/kb_references')}`;
    // Deliberately NO `--precision` on the read, though the write below states it.
    //
    // The e2e lane pushed this branch into cmd_resolve via `read_planes`; this one stays in bash
    // because its fallback is a DIFFERENT SUBCOMMAND — off the store scheme `localResolveCmd` reads
    // the curated tree, not a second plane of the same store, and no plane list expresses that.
    const resolveScript = KB_REMOTE === 'off'
      ? `${syncCmd}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\\n  ${commonArgs}`
      : `${syncCmd}${KB_ENV_PRELUDE}
REMOTE_OUT=''
if [ -n "$KB_STORE_TOKEN" ]; then
  REMOTE_OUT=$(python3 ${JSON.stringify(EXPERIENCE_STORE)} resolve-remote --plane remote${KB_VERSION_FLAG} \\
  ${commonArgs} 2>/dev/null || true)
  if printf '%s' "$REMOTE_OUT" | python3 -c 'import json,sys; sys.exit(0 if (json.load(sys.stdin).get("candidates") or []) else 1)' 2>/dev/null; then
    printf '%s\\n' "$REMOTE_OUT"; exit 0
  fi
fi
python3 ${JSON.stringify(EXPERIENCE_STORE)} ${localResolveCmd} \\
  ${commonArgs}`;
    const resolved = await agentT(
      `You are the warm-start resolver. Run EXACTLY this ` +
      `${KB_REMOTE === 'off' && !syncCmd ? 'command' : 'script'} ` +
      `and return its single-line JSON stdout verbatim as StructuredOutput — do not add, drop, reorder, ` +
      `or reinterpret any field. ` +
      // Saying the sync line is not the answer keeps the agent from relaying its summary, or from
      // treating its `|| true` as an error worth retrying.
      (syncCmd ? `Its first line brings the on-disk knowledge base up to date and prints nothing ` +
        `(its summary is redirected to a file); the JSON you return is the LAST command's. ` : '') +
      (KB_REMOTE === 'off' ? '' :
        `It tries the shared KB Store service first and falls back to the on-disk knowledge base by ` +
        `itself; run it as one script, do not split it into separate commands. `) +
      `An empty candidate list is a VALID answer meaning "nothing recorded for this kernel yet". Do not ` +
      `retry with different flags and do not invent candidates:
\`\`\`bash
${resolveScript}
\`\`\``,
      { phase: 'WarmStart', label: 'warm_start:resolve', schema: WARMSTART_RESOLVE_SCHEMA }) || {};
    warm_start.read_reason = resolved.read_reason || 'read';
    warm_start.slug = resolved.slug || '';
    warm_start.match_tier = resolved.match_tier || '';
    warm_start.filtered = resolved.filtered || null;
    const cands = Array.isArray(resolved.candidates) ? resolved.candidates : [];
    warm_start.candidates = cands.map(c => ({
      rank: c.rank, slug: c.slug, speedup: c.speedup, direction: c.direction || '', status: 'read',
      session_id: c.session_id || '', exp_dir: c.exp_dir || '',
    }));
    const f = resolved.filtered || {};
    // Which plane answered — read off the answer, not inferred from the mode.
    warm_start.plane = resolved.read_plane === 'remote' ? 'remote'
      : resolved.canonical_id ? 'store'
        : KB_MODE === 'store' ? 'none' : 'local';
    if (resolved.canonical_id) log(`[kb] plane=${warm_start.plane} key=${resolved.canonical_id}`);
    else if (KB_REMOTE !== 'off') log(`[kb] plane=${warm_start.plane} (service had no candidates, ` +
      `or no credentials)`);
    log(`[kb] experience read: slug=${resolved.slug || '?'} (${resolved.match_tier || 'exact'} match of ` +
      `${resolved.requested_slug || KERNEL_NAME}) reason=${warm_start.read_reason} ` +
      `candidates=${cands.length}${f.total ? ` of ${f.total} recorded [${f.retired || 0} retired, ` +
      `${f.below_min_speedup || 0} below ${WARM_START_MIN_SPEEDUP}x, ` +
      `${f.same_direction_collapsed || 0} same-direction]` : ''}` +
      // Every count on this line is over the rows FETCHED; a saturated scan means the page held more.
      `${f.scan_saturated ? ` (SCAN SATURATED at ${f.scanned} — the page holds more)` : ''}`);
    const otherLangs = Array.isArray(resolved.other_language_pages) ? resolved.other_language_pages : [];
    if (!cands.length && otherLangs.length) {   // wrong target_language, not an empty store
      log(`[kb] NOTE: no ${TARGET_LANGUAGE} page, but the store holds ${otherLangs.join(', ')} — ` +
        `check the target_language this lane was invoked with.`);
    }
    // reference-only: prose is already mirrored to EVAL_DIR/kb_references by the resolver; do not apply.
    if (!WARM_START_REF_ONLY) {
      const editableCsv = ((analysis && analysis.modifiable_files) || []).join(',');
      for (const c of cands) {                              // already rank-ordered (fastest first)
        const rec = warm_start.candidates.find(x => x.rank === c.rank);
        const remapOut = `${EVAL_DIR}/warm_start/cand_${c.rank}/remapped.diff`;
        const ver = await agentT(
          roleAgent('verify_engineer', 'verify',
            'Validate a HISTORICAL warm-start patch — the SAME gate as any round candidate. The patch was ' +
            'won in the workspace that produced it, whose paths need not exist here, so FIRST run EXACTLY:\n' +
            '```bash\n' +
            `mkdir -p ${JSON.stringify(`${EVAL_DIR}/warm_start/cand_${c.rank}`)}\n` +
            `python3 ${JSON.stringify(EXPERIENCE_STORE)} remap --patch ${JSON.stringify(c.patch_path)} \\\n` +
            `  --out ${JSON.stringify(remapOut)} --editable ${JSON.stringify(editableCsv)} \\\n` +
            `  --workspace ${JSON.stringify(CANONICAL)}\n` +
            '```\n' +
            'It prints one line of JSON. remapped=true: apply the REMAPPED patch (its paths were rewritten ' +
            'onto this workspace). reason "no_change_needed": apply the original PATCH. Any other reason ' +
            '(notably "unmapped_paths" — the patch touches files this workspace does not have): return ' +
            'status:"apply_failed", notes "patch_outside_editable_set", and DO NOT apply or benchmark. ' +
            'Apply with `git apply "$P" || git apply --3way "$P"`. On ANY failure restore the working copy ' +
            'fully (git checkout -- . and delete untracked files) before returning so the next candidate ' +
            'starts from a clean tree.', {
            CANONICAL, PATCH: c.patch_path, REMAP_OUT: remapOut,
            VERIFY_DIR: `${EVAL_DIR}/warm_start/cand_${c.rank}`,
            EDITABLE_SET: (analysis && analysis.modifiable_files) || [],
            GPU_ID: GPU_SINGLE, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
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
# the remapped patch when verify produced one (its paths fit THIS workspace), else the stored one
P=${JSON.stringify(remapOut)}; [ -f "$P" ] || P=${JSON.stringify(c.patch_path)}
git apply "$P" || git apply --3way "$P"
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "warm-start adopt: ${c.slug} (${sp.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH applies fail, apply manually to match intent, then add -A + commit and RE-RUN the COMMANDMENT
correctness check; only report committed=true if it still passes. Return JSON {committed, current_best_diff, note}.`,
            { phase: 'WarmStart', label: `warm_start:adopt c${c.rank}`, schema: COMMIT_SCHEMA });
          if (adopt && adopt.committed) {
            // The optimize loop now builds ON this patch: cumulative starts at the adopted Nx, so this
            // run's own rounds only earn the delta above it (split out as incremental_speedup below).
            cumulative = sp;
            bestPerCase = (ver.per_case && ver.per_case.length) ? ver.per_case : bestPerCase;
            finalWinner = { source: `warm_start:${c.slug}`, geomean: sp,
              arithmetic: ver.verified_arithmetic || sp, per_case: bestPerCase, patch: c.patch_path };
            if (bestSeen < sp) bestSeen = sp;
            warm_start.adopted = true;
            warm_start.adopted_speedup = sp;
            warm_start.slug = c.slug;
            // Carried into the write-back: the idea this run started from, and the entry it
            // descends from — lineage a later curation pass cannot recover from the diff alone.
            warm_start.direction = c.direction || '';
            warm_start.exp_dir = c.exp_dir || '';
            if (rec) { rec.status = 'adopted'; rec.outcome = 'validated'; rec.verified_speedup = sp; }
            log(`[kb] warm-start ADOPTED ${c.slug} @ ${sp.toFixed(2)}x — optimizing from the patched state.`);
            if (PROFILE_ON) {
              profileSummary = await agentT(
                roleAgent('profile_engineer', 'reprofile',
                  'Re-profile the adopted warm-start state and classify the new bottleneck.', {
                  WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE, ROUND: 0,
                  COMMANDMENT, PREVIOUS_METRICS: profileSummary, ...COUNTER_INPUTS,
                }),
                { phase: 'WarmStart', label: 'warm_start:reprofile', schema: PROFILE_SCHEMA }) || profileSummary;
              if (history) history.bottleneck_now = profileSummary ? profileSummary.bottleneck : history.bottleneck_now;
            }
            if (WARM_START_RETURN_AFTER) { skipLoop = true; warm_start.returned_after_read_kb = true; }
            break;
          }
        }
        if (rec && rec.status !== 'adopted') {
          rec.status = (ver && ver.status) ? `rejected:${ver.status}` : 'rejected:apply_failed';
          rec.outcome = kbVerdict(ver, sp);
          rec.verified_speedup = sp || 0;
        }
        log(`[kb] warm-start candidate c${c.rank} ${rec ? rec.status : 'rejected'} (${sp ? sp.toFixed(2) + 'x' : 'no measure'}).`);
      }
      // Record what THIS box saw, so the next one reads a ledger and not just a speedup.
      const benched = warm_start.candidates.filter(x => x.outcome && (x.session_id || x.exp_dir));
      if (benched.length) {
        const planeFlags = warm_start.plane === 'remote' ? '--plane remote'
          : warm_start.plane === 'store' ? `--plane local --store ${JSON.stringify(KB_STORE_DIR)}` : '';
        const cmds = benched.map(x =>
          `python3 ${JSON.stringify(EXPERIENCE_STORE)} attest ${planeFlags} ` +
          (x.exp_dir ? `--exp-dir ${JSON.stringify(x.exp_dir)} ` : '') +
          (planeFlags && x.session_id ? `--session-id ${JSON.stringify(x.session_id)} ` : '') +
          `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} ` +
          `--gfx ${GFX}${KB_VERSION_FLAG} --outcome ${x.outcome} ` +
          `--measured-speedup ${Number(x.verified_speedup) || 0} ` +
          `--note ${JSON.stringify(`kernel_lane warm-start c${x.rank}: ${x.status}`.slice(0, 200))} ` +
          `--measured-by ${JSON.stringify('kernel_lane:' + GFX)} --apply || true`);
        await agentT(
          `You are the kernel knowledge-base attestor. Run EXACTLY these commands in order and ` +
          `return {"ran": <how many you ran>, "note": "<anything that failed>"}. Each records what ` +
          `this box saw when it benched a stored patch. Do NOT edit them, do NOT add or drop any, ` +
          `and do NOT retry a failure — a repeat would double-count the attempt.\n` +
          '```bash\n' + KB_ENV_PRELUDE + cmds.join('\n') + '\n```',
          { phase: 'WarmStart', label: 'kb:attest',
            schema: obj({ ran: { type: 'number' }, note: { type: 'string' } }, []) });
        log(`[kb] attested ${benched.length} benched candidate(s): ` +
          benched.map(x => `c${x.rank}=${x.outcome}`).join(' '));
      }
    }
  }
}

if (EVO && !(await EVO.initialize())) {
  EVO_FALLBACK = EVO.result();
  EVO = null;
  // The ordinary lane plans with the ordinary schema: drop the EVO-only lineage fields added above.
  delete PLAN_SCHEMA.properties.directions.items.properties.parent_id;
  delete PLAN_SCHEMA.properties.directions.items.properties.donor_ids;
}

let plannerStopReason = '';
// mode=mega ends on the budget, its modeled clock or a mega stop rule, not on no-improve rounds.
while (!skipLoop && dispatched < BUDGET && (MEGA || noImprove < MAX_NO_IMPROVE) && round < MAX_ROUNDS) {
  // --- (0) HARD STOP: no new round may START past the deadline ----------
  // Checked BEFORE round++ so an expired check does not inflate the round count. The in-flight round
  // is never interrupted — a killed round leaves a half-verified patch and no report.
  let left = await secondsLeft(`pre-r${round + 1}`);
  __routeClock = { left_s: left };
  if (left <= 0) {
    deadlineHit = true;
    log(`WALL-CLOCK DEADLINE reached after ${round} round(s) (budget used ${dispatched}/${BUDGET}) — ` +
        `starting no further rounds; going straight to report + validation.`);
    break;
  }
  if (DEADLINE_EPOCH) log(`[deadline] ${(left / 60).toFixed(0)} min remain before the hard stop.`);
  if (ROUTING_ON && __routeOverBudget()) {
    log(`ROUTING OUTPUT-TOKEN CAP reached after ${round} round(s) — starting no further rounds; going straight to report + validation.`);
    break;
  }

  round++;
  const remaining = BUDGET - dispatched;
  phase('Optimize');
  // mode=mega: the modeled-clock deadline, calibration recovery and the GPU pool come first.
  const megaPre = MEGA ? await megaBeforePlan(round) : null;
  if (megaPre && megaPre.stop) break;

  // --- (a) Plan the round (TechLead) ------------------------------------
  const planInputs = (forbidStop) => ({
    ...(EVO ? { EVOLUTION: EVO.plannerInput() } : {}),
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
    KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
    GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...COUNTER_INPUTS,
    ...KB_INPUTS,
    // DRA brief (REFERENCE), from main.
    ...(researchBriefPath ? { DEEP_SEARCH_BRIEF: researchBriefPath } : {}),
    // Stated UNCONDITIONALLY, in both directions.
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

  const evoPlannerRole = EVO
    ? `\n\nAlso Read ${WORKFLOW_DIR}/../evo/roles/planner.md for the opt-in archive tools and selection policy.` : '';
  // Eikos shadow: freeze the state the planning attempt sees, before that attempt.
  let eikosSnap = EIKOS_SHADOW_ON ? eikosSnapshot(round, left, 'pre_plan_clock') : null;
  const forcedBeforeRound = forcedReplans;
  let plan = await agentT(
    MEGA ? megaPlanPrompt(round, remaining, megaPre.pool)
      : roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).',
        planInputs(false)) + evoPlannerRole,
    MEGA ? megaPlanOpts(round)
      : { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });
  // mode=mega: bind the directions to candidate lanes and refuse those that break the topology contract.
  if (MEGA) {
    plan = megaPlanStep(round, remaining, plan);
    if (plan.stop) break;
    if (plan.skip) continue;
  }
  const eikosFirstChoice = EIKOS_SHADOW_ON ? eikosRawChoice(plan) : null;

  // NO EARLY STOP: keep refusing while the window is materially unspent.
  while (!MEGA && DEADLINE_EPOCH && left > NO_STOP_S && forcedReplans < MAX_FORCED_REPLANS &&
         (!plan || plan.stop || !plan.directions || plan.directions.length === 0)) {
    forcedReplans++;
    log(`Round ${round}: stop=true with ${(left / 60).toFixed(0)} min left — refusing (#${forcedReplans}), re-planning.`);
    if (EIKOS_SHADOW_ON) eikosSnap = eikosSnapshot(round, left, 'pre_replan_clock');
    plan = await agentT(
      roleAgent('tech_lead', 'plan_round', 'Your stop was refused: plan at least one direction.',
        planInputs(true)) + evoPlannerRole,
      { phase: 'Optimize', label: `tech_lead:replan r${round}#${forcedReplans}`, schema: PLAN_SCHEMA });
    left = await secondsLeft(`replan-r${round}`);
    __routeClock = { left_s: left };
  }

  // Eikos shadow: the model's own final choice, read before the harness filter below can change it.
  const eikosFinalRaw = EIKOS_SHADOW_ON ? eikosRawChoice(plan) : null;
  if (!MEGA && plan && Array.isArray(plan.directions) && ROOFLINE_STATUS.startsWith('unknown')) {
    const before = plan.directions.length;
    plan.directions = plan.directions.filter((d) =>
      !/roofline|peak\s*(?:flops|bandwidth)|%\s*of\s*peak/i.test(
        `${d && d.title || ''} ${d && d.prompt || ''} ${d && d.why || ''}`));
    if (plan.directions.length !== before) {
      log(`Round ${round}: removed ${before - plan.directions.length} direction(s) that relied on unknown peaks.`);
    }
    if (before > 0 && plan.directions.length === 0) {
      plannerStopReason = 'all_directions_filtered_unknown_peaks';
    }
  }
  const eikosRec = EIKOS_SHADOW_ON
    ? await eikosShadowRound(round, eikosSnap, eikosFirstChoice, plan, forcedReplans - forcedBeforeRound, left,
                             eikosFinalRaw, plannerStopReason)
    : null;

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    if (plannerStopReason === 'all_directions_filtered_unknown_peaks') {
      log(`Round ${round}: harness stopped after all directions were filtered because calibrated peaks are unknown.`);
    } else {
      log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    }
    break;
  }

  // A mega direction is already bound to its candidate lane.
  let directions = MEGA ? plan.directions : plan.directions.slice(0, remaining).map((d, i) => ({
    ...d,
    idx: i,
    id: d.id || `r${round}_d${i}`,
    gpu_id: GPU_MODE === 'pin' && !GPU_JOB ? onEngine(GPU_LIST[i % GPU_LIST.length]) : GPU_POOL,
    out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
  }));
  // deep_explore is a DEDICATED-ROUND, heavyweight mandate: if the plan includes one, run ONLY it this
  // round (its broad ground-up rewrite touches many files and can't be merged with specialist patches),
  // and charge DEEP_COST against the budget. Otherwise each specialist direction costs 1.
  const deepDir = !MEGA && directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  if (EIKOS_SHADOW_ON) directions.forEach((d) => eikosSpecialties.add(d.specialty));
  log(`Round ${round}: ${directions.length} direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    async (d) => {
      // mode=mega: the engineer authors one candidate in its own tree; mt.after checks and records it.
      const mt = MEGA ? megaEngineerTurn(round, d) : null;
      if (mt && mt.result) return mt.result;
      if (EVO && !MEGA) {
        try {
          d.evo_prepared = await EVO.prepare(d);
        } catch (error) {
          EVO.result().errors.push({ round, operation: 'prepare', id: d.id,
            message: String(error && error.message || error) });
          return { d, eng: null, evo_prepare_failed: true };
        }
      }
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
      const route = ROUTING_ON && !MEGA ? {} : null;   // receives this dispatch's host-issued id (agentT)
      if (ROUTING_ON && !MEGA) {                       // the Eikos decider's card reads the direction, never the prompt
        const xs = Number(d.expected_speedup);
        __routeTasks[`${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`] = {
          id: String(d.id), title: String(d.title || ''), specialty: d.specialty,
          expected_speedup: Number.isFinite(xs) && xs > 0 ? xs : null,
          focus_files: (d.focus_files || []).map(String), prompt: String(d.prompt || '') };
      }
      return agentT(MEGA ? mt.prompt :
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
${EVO ? 'Your selected-parent workspace is already prepared. Do not recreate it.\n' : `First create YOUR private workspace, then optimize.
\`\`\`bash
# Issue #429: ALWAYS use materialize_workspace.sh — do NOT inline tar/cp. Agents that omitted
# --exclude='*.so' previously copied multi-GiB aiter/jit/*.so into every engineer/verify clone.
# The script excludes nested *.so/*.o and aiter/jit, preserves symlinks (never -h), and may
# share an immutable aiter tree via --link-aiter under EVAL_DIR/_shared. Candidate builds still
# use per-workspace .torch_ext via gpu_lock.sh (isolation unchanged).
mkdir -p ${d.out_dir}
bash ${WORKFLOW_DIR}/scripts/materialize_workspace.sh \\
  --src ${CANONICAL} --dst ${d.out_dir}/workspace \\
  --shared-root ${EVAL_DIR}/_shared --link-aiter
\`\`\`
`}${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git diff > ${d.out_dir}/best_patch.diff\` when geomean>${CANDIDATE_FLOOR_TXT}.

## Inputs
${cfg({
        ...(EVO ? { EVOLUTION: EVO.engineerInput(d) } : {}),
        SPECIALTY: d.specialty,
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
        ...(isDeep ? { TARGET: d.expected_speedup ? `reach ${d.expected_speedup}x (or ~90% of the roofline ceiling), whichever is the harder bar` : 'reach ~90% of the roofline ceiling' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        PROFILING: PROFILE_ON ? 'on' : 'off',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.` +
      // Built inline, not via roleAgent(), so the injection has to be appended here too (and, for a
      // UT-module multi-GPU task only, the multi-GPU block).
      expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer') +
      (UT_MR ? multiGpuBlock(isDeep ? 'deep_engineer' : 'engineer') : '') +
      (EVO ? `\n\nAlso Read ${WORKFLOW_DIR}/../evo/roles/engineer.md for the selected-parent and donor contract.` : '') +
      fastModeEngineerBlock(d.out_dir),
      MEGA ? mt.opts : { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA },
      route
    ).then((eng) => MEGA ? mt.after(eng) : ({ d, eng, route }));
    },

    async (prev) => {
      // mode=mega: the candidate's verify (structure check first); mv.after scores and records it.
      let mv = null, d, eng, route, patch, vlabel, vroute = null;
      if (MEGA) {
        mv = prev.verifyStep ? await prev.verifyStep() : { result: prev };
        if (mv.result) return mv.result;
        if (!mv.prompt) return mv.after(null);
        ({ d } = prev);
        vlabel = mv.opts.label;
      } else {
        ({ d, eng, route } = prev);
        // Engineers run gpu_lock.sh from KERNEL_PATH; verify works in out_dir/verify, so it is not blocked.
        await stopAfterTimeout(eng, `${d.specialty === 'deep_explore' ? 'deep' : 'eng'} ${d.id}:${d.specialty}`,
          `${d.out_dir}/workspace`, 'Optimize');
        if (EVO && prev.evo_prepare_failed) return { d, eng, ver: null };
        patch = `${d.out_dir}/best_patch.diff`;
        // Harvest is MEASUREMENT-anchored, not return-value-anchored.
        const trustworthyBelowBaseline = eng && eng.status !== 'failed' && !(primSpeedup(eng) > CANDIDATE_FLOOR) &&
          eng.status !== 'invalid_measurement' && eng.measurement_valid === true;
        if (trustworthyBelowBaseline) {
          return { d, eng, ver: null, route };
        }
        if (EVO) {
          const custody = await EVO.checkPatch(d, patch);
          if (!custody) return { d, eng, ver: null };
          patch = custody.patch_object_path;
        }
        const recovered = !eng || eng.status === 'failed';
        if (recovered) log(`Round ${round} dir ${d.id}: engineer return ${eng ? 'failed' : 'missing'} — ` +
          `sending best_patch.diff to verify anyway (oracle decides).`);
        // eikos v2 E1 (R2): carry a host dispatch id for the verifier too, so its OWN terminal outcome can
        // reach P1's shadow (coverage beyond the engineer cycle + committer).
        vlabel = `verify ${d.id}${recovered ? ' (recovered)' : ''}`;
        vroute = ROUTING_ON ? {} : null;
      }
      return agentT(MEGA ? mv.prompt :
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          ENGINEER_WORKSPACE: `${d.out_dir}/workspace`,
          EVAL_DIR, WORKFLOW_DIR, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
        MEGA ? mv.opts : { phase: 'Verify', label: vlabel, schema: VERIFY_SCHEMA }, vroute
      ).then((ver) => stopAfterTimeout(ver, vlabel, `${d.out_dir}/verify`, 'Verify')).then((ver) => {
        if (MEGA) return mv.after(ver);
        // eikos v2 E1 (R2): feed the verifier's own terminal outcome to P1 (shadow by default).
        if (ROUTING_ON && ver != null) {
          // eikos v2 E1 (R2 residual / Astra re-review #3 finding 4): "not invalid_measurement" was too
          // broad — VERIFY_SCHEMA permits ANY status string, so a non-null status:'failed' (or any
          // unknown status) was booked as SUCCESS.
          const vUsable = says(ver.status, 'verified') || says(ver.status, 'correctness_failed');
          const vInvalid = says(ver.status, 'invalid_measurement');
          __p1WorkerOutcome(__routeScope(vlabel), vUsable,
            `verify status=${ver.status || ''} correctness=${ver.correctness || ''}`,
            vUsable ? 'work_product' : (vInvalid ? 'invalid_measurement' : 'no_result'),
            'host', vlabel, vroute ? vroute.dispatch_id : null);
        }
        return { d, eng, ver, patch, route };
      });
    }
  );

  const clean = results.filter(Boolean);
  // mode=mega: the round's modeled clock; a lane that stopped (timeout) ends the run after the bookkeeping.
  const megaTurns = MEGA ? megaCloseRound(results) : null;
  if (megaTurns && megaTurns.stopped && !megaTurns.done.length) {
    stopReason = megaTurns.stopped.reason;
    break;
  }
  if (EVO) await EVO.recordBatch(clean);
  // mode=mega selects from its candidate registry below; whole-kernel candidates are never merged.
  const verified = MEGA ? [] : clean.filter(r => r.ver && says(r.ver.status, 'verified') &&
    says(r.ver.correctness, 'pass') && primSpeedup(r.ver) > CANDIDATE_FLOOR);
  // Routing: a direction that did not produce a verified, correct, above-floor patch is a failed cycle
  // for its lane. Honest below-floor counts too — the task was the speedup, not a clean return.
  if (ROUTING_ON) for (const r of clean) {
    const kindTag = r.d.specialty === 'deep_explore' ? 'deep' : 'eng';
    const sc = `${kindTag}:${r.d.specialty}`;
    const ev = !r.eng ? 'engineer returned nothing'
      : !r.ver ? `engineer reported ${primSpeedup(r.eng)}x <= floor ${CANDIDATE_FLOOR}`
      : `verify status=${r.ver.status} correctness=${r.ver.correctness} speedup=${primSpeedup(r.ver)}x floor=${CANDIDATE_FLOOR}`;
    // What kind of failure, for the Eikos decider: only a wrong or unusable work product says anything
    // about the lane.
    const kind = __routeSkipped[`${kindTag} ${r.d.id}:${r.d.specialty}`] ? 'no_result'
      : !r.ver ? 'non_improvement'
      : says(r.ver.status, 'invalid_measurement') ? 'invalid_measurement'
      : says(r.ver.status, 'apply_failed') && !r.eng ? 'no_result'
      : (says(r.ver.status, 'apply_failed') || says(r.ver.status, 'correctness_failed') ||
         says(r.ver.correctness, 'fail')) ? 'work_product'
      : 'non_improvement';
    // eikos v2 E1 (bug 5): the `!r.ver -> non_improvement` branch conflates a DELIBERATELY skipped
    // verification after a valid below-floor measurement (`trustBelow`, a true non_improvement) with a
    // verifier that was asked and returned NOTHING (ver==null from the recovered verify path — a
    // no_result that must move nothing).
    const trustBelow = r.eng && r.eng.status !== 'failed' && !(primSpeedup(r.eng) > CANDIDATE_FLOOR) &&
      r.eng.status !== 'invalid_measurement' && r.eng.measurement_valid === true;
    const p1Kind = (kind === 'non_improvement' && !r.ver && !trustBelow) ? 'no_result' : kind;
    __routeOutcome(sc, verified.indexOf(r) >= 0, ev, kind, `${r.ver ? 'verify' : 'engineer'} ${r.d.id}`,
                   `${kindTag} ${r.d.id}:${r.d.specialty}`, r.route ? r.route.dispatch_id : null, p1Kind);
  }

  // --- (d) Build candidate list; integrate if >=2 verified -------------- `geomean` here is the
  // PRIMARY metric used for sorting/gating/cumulative: the time-weighted ratio-of-sums when a workload
  // spec is active, else the unweighted geomean (unchanged behavior).
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
        EVAL_DIR, WORKFLOW_DIR, GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          files: r.d.focus_files || [], patch: r.patch })),
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    await stopAfterTimeout(integrate, `integrate r${round}`, `${EVAL_DIR}/round_${round}/integrate`, 'Merge');
    const integrateCustody = EVO && integrate && integrate.best && integrate.best.patch_file
      ? await EVO.checkPatch({ id: `r${round}_integrated` }, integrate.best.patch_file) : null;
    const integPrim = integrate && integrate.best ? primSpeedup({
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
    }) : 0;
    if (integrate && (!EVO || integrateCustody) && integrate.conclusion === 'improved' && integrate.best &&
      integPrim > Math.max(...candidates.map(c => c.geomean))) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: EVO ? integrateCustody.patch_object_path : integrate.best.patch_file,
      });
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const megaSelected = MEGA ? megaSelectCandidate() : null;
  const winner = MEGA ? (megaSelected && megaWinnerOf(megaSelected)) : candidates[0] || null;
  const improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  // Separate question from `improved`: is the SEARCH advancing, not did it beat the incumbent.
  const madeProgress = !!(winner && bestSeen > 0 && winner.geomean > bestSeen * (1 + PROGRESS_DELTA));
  // eikos v2 E1 (R3): did the winner ACTUALLY get installed into the canonical workspace this round?
  let installed = false;

  // --- (e) Commit the winner into the canonical workspace ---------------
  // mode=mega: the candidate tree is the artifact, so the winner is recorded and nothing is applied.
  if (MEGA && winner && (improved || !finalWinner)) {
    cumulative = Math.max(cumulative, winner.geomean);
    bestPerCase = winner.per_case.length ? winner.per_case : bestPerCase;
    finalWinner = winner;
  } else if (!MEGA && improved) {
    // eikos v2 E1 (R4 round-4 / Astra re-review #4): ABORT-AND-QUARANTINE replaces the round-3
    // restore-and-continue recovery, so there is NO pre-attempt snapshot/capture op here.
    const commitRoute = {};   // eikos v2 E1 (bug 1): carries the host dispatch id so the commit's outcome binds to P1
    // eikos v2 E1 (R7 — commit-salvage): hoist the committer request to a const so the ONE-SHOT RETRY
    // below re-dispatches the BYTE-IDENTICAL committer prompt (there is no second copy to drift).
    const commitPrompt =
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
re-check is not required.) Return JSON {committed, current_best_diff, note}.`;
    const commitRes = await agentT(
      commitPrompt,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA }, commitRoute);
    // observability only (eikos v2): one terminal outcome per completed commit dispatch, before
    // `cumulative` advances.
    let committed = !!(commitRes && commitRes.committed);
    __routeAudit.push({ event: 'commit_outcome', round, improved: true,
                        committed,
                        geomean: winner.geomean, source: winner.source });
    // eikos v2 E1 (bug 1 / R8 Astra re-review #9 BUG 3): feed the committer's attributable terminal
    // outcome to P1 (shadow by default; the gated live-P1 handler under route_decider=p1) through the
    // ONE verdict table.
    __commitAttemptOutcome(commitRes, commitRoute, `commit r${round}`);

    // eikos v2 E1 (R7 round-7 / commit-salvage — user-APPROVED Option A; Astra re-review #7-commit): do
    // NOT let a TRANSIENT/AMBIGUOUS null commit vaporize a fully-verified run.
    //
    // CLASSIFY.
    const __commitExclusiveOwnership = (receipt, atRoute) =>
      // 'not_started' | 'terminated' | 'ambiguous'.
      (!(atRoute && atRoute.dispatch_id)) ? 'not_started' : (receipt != null ? 'terminated' : 'ambiguous');
    const __retainQuarantine = (stage, atRoute, atReceipt, opts) => {
      // QUARANTINE + RETAIN: persist a recovery reference to the verified integrated winner (its patch
      // lives on disk under round_${round}/integrate|engineer_*) so the verified work is RECOVERABLE,
      // not discarded.
      const o = opts || {};
      const reason = o.reason || 'commit_ambiguous_unconfirmed_termination';
      const why = o.why || 'timed out — CANNOT confirm the writer stopped';
      const note = o.note || 'verified winner RETAINED for recovery — ambiguous commit (writer termination UNCONFIRMED): no second writer launched';
      const recovery = {
        retained: true, stage, workspace: CANONICAL, round,
        verified_patch: winner.patch, source: winner.source, geomean: winner.geomean,
        integrate_dir: `${EVAL_DIR}/round_${round}/integrate`,
        note,
      };
      __routeAudit.push({ event: 'commit_retain', round, improved: true, quarantined: true,
                          reason, stage,
                          dispatch_id: (atRoute && atRoute.dispatch_id) || null, receipt: atReceipt || null,
                          recovery, source: winner.source, geomean: winner.geomean, workspace: CANONICAL });
      log(`Round ${round}: commit UNCONFIRMED at '${stage}' (${why}). ` +
          `NOT launching a second writer; QUARANTINE + RETAIN the verified winner (${winner.source} ` +
          `${winner.geomean.toFixed(2)}x, patch ${winner.patch}) as RECOVERABLE. Incumbent stays ` +
          `${cumulative.toFixed(2)}x; STOPPING THE RUN AS FAILED.`);
      return {
        mode: MODE,
        llm_timeline: LLM_STATS ? { ...LLM_TL, instance: EVAL_DIR } : undefined,
        eval_dir: EVAL_DIR,
        kernel_name: KERNEL_NAME,
        validation_status: 'failed',
        status: 'failed',
        stopped_by: 'commit_ambiguous_quarantine',
        commit_failure: {
          reason, quarantined: true, workspace: CANONICAL,
          round, source: winner.source, geomean: winner.geomean, committed: false, ambiguous: true,
          dispatch_id: (atRoute && atRoute.dispatch_id) || null, receipt: atReceipt || null,
          retained_candidate: recovery,
          receipt_ref: {
            dispatch_id: (atRoute && atRoute.dispatch_id) || null,
            scope: stage,
            timeline: `${EVAL_DIR}/reports/trace/agent_timeline.json`,
            ledger: `${EVAL_DIR}/reports/trace/llm_calls.jsonl`,
          },
        },
        retained_candidate: recovery,
        cumulative_at_failure: cumulative,
        rounds: round,
        budget_used: dispatched,
        budget_total: BUDGET,
        routing: ROUTING_ON ? __routeReport() : undefined,
      };
    };
    // eikos v2 E1 (R9 / Astra re-review #10, Finding 3 residual — terminal-failure provenance): track the
    // ACTUAL attempt that produces the terminal verdict so the terminal commit_failure reports the LAST failing
    // attempt (e.g. d2), NOT the initial dispatch (d1). Defaults to the initial committer; the EXPLICIT-false
    // RETRY below reassigns it. The per-attempt audit (commit_outcome/commit_retry) is left intact.
    let termRoute = commitRoute, termRes = commitRes, termScope = `commit r${round}`;
    if (ROUTING_ON && !committed && !(commitRes != null && commitRes.committed === false)) {
      // BUG 1 GATE: the FIRST commit dispatch must grant EXCLUSIVE ownership before we launch any salvage
      // writer. An AMBIGUOUS (timed-out) initial commit => retain + quarantine (never a second writer).
      if (__commitExclusiveOwnership(commitRes, commitRoute) === 'ambiguous') {
        return __retainQuarantine(`commit r${round}`, commitRoute, commitRes);
      }
      // (1) RETRY the commit dispatch ONCE — the initial dispatch is confirmed TERMINATED or
      // NOT-STARTED, so a second writer has EXCLUSIVE ownership.
      const retryRoute = {};
      const commitRetry = await agentT(
        commitPrompt,
        { phase: 'Merge', label: `commit r${round} retry`, schema: COMMIT_SCHEMA }, retryRoute);
      // BUG 3 verdict table: committed===true -> success; committed===false (EXPLICIT) -> terminal (NO host
      // salvage); null / no-verdict -> AMBIGUOUS (infra). Each attempt reaches __p1WorkerOutcome, correctly bound.
      const retryCommitted = commitRetry != null && commitRetry.committed === true;
      const retryExplicitFail = commitRetry != null && commitRetry.committed === false;
      __routeAudit.push({ event: 'commit_retry', round, improved: true, committed: retryCommitted,
                          explicit_fail: retryExplicitFail, ambiguous: commitRetry == null, source: winner.source,
                          geomean: winner.geomean, dispatch_id: (retryRoute && retryRoute.dispatch_id) || null,
                          receipt: commitRetry || null });
      __commitAttemptOutcome(commitRetry, retryRoute, `commit r${round} retry`);
      if (retryCommitted) {
        committed = true;
        log(`Round ${round}: commit was AMBIGUOUS/null; the ONE-SHOT RETRY confirmed committed=true — ` +
            `continuing to director:validate (no abort).`);
      } else if (retryExplicitFail) {
        // BUG 3: an EXPLICIT {committed:false} retry is a GENUINE failure.
        termRoute = retryRoute; termRes = commitRetry; termScope = `commit r${round} retry`;
        log(`Round ${round}: the commit RETRY returned an EXPLICIT {committed:false} — a genuine failure; ` +
            `TERMINAL (host-side salvage removed in R9). ABORT-AND-QUARANTINE below.`);
      } else if (__commitExclusiveOwnership(commitRetry, retryRoute) === 'ambiguous') {
        // BUG 1: the retry TIMED OUT (receipt-less) — its writer may be live; do NOT launch another writer.
        return __retainQuarantine(`commit r${round} retry`, retryRoute, commitRetry);
      } else {
        // eikos v2 E1 (R9 / Astra re-review #10, Finding 2 — SHORTEST ROUTE): the automatic HOST-SIDE
        // salvage commit has been REMOVED.
        return __retainQuarantine(`commit r${round} retry`, retryRoute, commitRetry,
          { reason: 'commit_unconfirmed_no_verdict',
            why: 'the retry resolved WITHOUT a committed verdict (still-unconfirmed); host-side salvage removed in R9',
            note: 'verified winner RETAINED for recovery — commit UNCONFIRMED (retry returned no committed verdict; host-side salvage removed): no second writer, no host-install' });
      }
    }
    // eikos v2 E1 (R3/R4 — SHARED HARNESS, BOTH ARMS): a FAILED commit did NOT install the winner, so
    // the tracked incumbent must not advance on it.
    if (!ROUTING_ON || committed) {
      cumulative = winner.geomean;
      bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
      finalWinner = winner;
      roundsCommitted += 1;
      installed = true;        // the winner is now the installed incumbent artifact

      // --- (f) Re-profile the new best (only with args.profile='on') -------
      if (PROFILE_ON) {
        if (FAST_MODE && (dispatched >= BUDGET || round >= MAX_ROUNDS)) {
          log(`  [fast_mode] last round (${round}) -- skipping the re-profile.`);
        } else {
          profileSummary = await agentT(
            roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
              WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
              COMMANDMENT, PREVIOUS_METRICS: profileSummary, ...COUNTER_INPUTS,
            }),
            { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
        }
      }
    } else {
      // eikos v2 E1 (R4 round-4 / Astra re-review #4): ABORT-AND-QUARANTINE (replaces round-3
      // recovery).
      const ambiguous = termRes == null;   // the TERMINAL attempt's verdict (R9 Finding 3 residual: not commitRes)
      __routeAudit.push({ event: 'commit_failed', round, improved: true, committed: false,
                          ambiguous, quarantined: true, reason: 'commit_failed',
                          // provenance FIX (R9 / Astra re-review #10, Finding 3 residual): the ACTUAL failing
                          // attempt (the LAST verdict that terminated the run, e.g. d2) — NOT the initial dispatch.
                          dispatch_id: (termRoute && termRoute.dispatch_id) || null,
                          receipt: termRes || null, workspace: CANONICAL,
                          // NIT (Astra re-review #5): `receipt` is the committer's TYPED return, not
                          // its full tool transcript.
                          receipt_ref: { dispatch_id: (termRoute && termRoute.dispatch_id) || null,
                                         scope: termScope,
                                         timeline: `${EVAL_DIR}/reports/trace/agent_timeline.json`,
                                         ledger: `${EVAL_DIR}/reports/trace/llm_calls.jsonl` },
                          // the INITIAL committer attempt retained for the per-attempt audit trail (differs from
                          // the terminal attempt only when a d1 no-verdict was followed by a d2 explicit-false).
                          initial_attempt: { dispatch_id: (commitRoute && commitRoute.dispatch_id) || null,
                                             receipt: commitRes || null },
                          source: winner.source, geomean: winner.geomean });
      log(`Round ${round}: commit ${ambiguous ? 'AMBIGUOUS (committer timed out/unverified — cannot confirm ' +
          'it stopped)' : 'FAILED'} under routing — ABORT-AND-QUARANTINE: the canonical workspace is left ` +
          `AS-IS (NOT restored, NOT continued) for post-mortem; incumbent stays ${cumulative.toFixed(2)}x; ` +
          `STOPPING THE RUN AS FAILED rather than recovering-and-continuing from a workspace we cannot prove ` +
          `is intact.`);
      return {
        mode: MODE,
        // eikos v2 E1 (R5 / Astra re-review #5 — failed-run ACCOUNTING): carry the TERMINAL agent
        // timeline so the parent dispatcher persists it and runs the token/time/cost ledger + report
        // for an ABORTED run too.
        llm_timeline: LLM_STATS ? { ...LLM_TL, instance: EVAL_DIR } : undefined,
        eval_dir: EVAL_DIR,
        kernel_name: KERNEL_NAME,
        // Failure/validation metadata so the scorer + report classify this as a TERMINAL FAILED run (not
        // 'unknown') while its spend is still counted; N stays not-completed / not-accepted downstream.
        validation_status: 'failed',
        // TERMINAL FAILURE (eikos v2 E1, R4): a routed commit could not be confirmed installed.
        status: 'failed',
        stopped_by: 'commit_failed',
        commit_failure: {
          reason: 'commit_failed', quarantined: true, workspace: CANONICAL,
          round, source: winner.source, geomean: winner.geomean,
          committed: false, ambiguous,
          // provenance FIX (R9 / Astra re-review #10, Finding 3 residual): the terminal failure now points to
          // the ACTUAL failing attempt (the LAST verdict that terminated the run, e.g. d2) — NOT the initial
          // dispatch (d1). With a d1 no-verdict + d2 explicit-false, this reports d2.
          dispatch_id: (termRoute && termRoute.dispatch_id) || null,
          receipt: termRes || null,             // the ACTUAL failing attempt's typed receipt/transcript
          // NIT (Astra re-review #5): a LINK to the committer's FULL execution evidence (its tool transcript),
          // not just the typed `receipt` above — reachable by dispatch id in the persisted timeline/ledger that
          // geak_report.py renders under the eval dir, so a post-mortem has evidence even when receipt is null.
          receipt_ref: {
            dispatch_id: (termRoute && termRoute.dispatch_id) || null,
            scope: termScope,
            timeline: `${EVAL_DIR}/reports/trace/agent_timeline.json`,
            ledger: `${EVAL_DIR}/reports/trace/llm_calls.jsonl`,
          },
          // the INITIAL committer attempt retained for the per-attempt audit trail (R5 NIT evidence link);
          // differs from the terminal attempt only when a d1 no-verdict was followed by a d2 explicit-false.
          initial_attempt: {
            dispatch_id: (commitRoute && commitRoute.dispatch_id) || null,
            receipt: commitRes || null,
          },
        },
        cumulative_at_failure: cumulative,      // the incumbent did NOT advance
        rounds: round,
        budget_used: dispatched,
        budget_total: BUDGET,
        routing: ROUTING_ON ? __routeReport() : undefined,   // carries the terminal commit_failed audit + identity
      };
    }
  }

  if (winner && winner.geomean > bestSeen) bestSeen = winner.geomean;
  // eikos v2 E1 (R3): reset the stagnation stop ONLY on genuine new search progress (a candidate better
  // than any seen before) or a REAL install (the winner actually committed into the canonical
  // workspace).
  if (MEGA) megaRoundCounters(megaTurns.done, improved);
  else if (madeProgress || installed) { noImprove = 0; } else { noImprove++; }
  if (EIKOS_SHADOW_ON) {
    // Measured facts only.
    const o = { verified_candidates: verified.length, winner_speedup: winner ? winner.geomean : null,
                improved: improved, made_progress: madeProgress, installed: installed, commit_reported: 'not_captured',
                tracked_incumbent_after: cumulative };
    if (eikosRec) eikosRec.outcome = o;
    eikosLastOutcome = o;
  }

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  // mode=mega: the round's history entry and the candidate registry are persisted to STATE.
  if (MEGA) history.rounds.push(megaRoundEntry(round, megaTurns.done, megaSelected, improved));
  const megaState = MEGA && STATE_DIR ? megaStateWrite(round) : null;
  let mem = MEGA && !STATE_DIR ? true
    : (!MEGA && FAST_MODE && (dispatched >= BUDGET || round >= MAX_ROUNDS)) ? null : await agentT(
    roleAgent('tech_lead', 'update_memory', megaState ? megaState.task
      : 'Distill durable insights + update the hypothesis ledger.', megaState ? megaState.inputs : {
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
    megaState ? megaState.opts : { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (MEGA && megaState) mem = await megaCheckStateWrite(round, megaState, mem);
  if (MEGA && megaTurns.stopped) {
    stopReason = megaTurns.stopped.reason;
    break;
  }
  if (MEGA && !mem) {
    stopReason = `candidate state for round ${round} could not be persisted; stopped before ` +
      `dispatching another lane`;
    break;
  }
  if (!MEGA && mem) {
    if (mem.insights) history.insights = mem.insights;
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  // Enforce the round's KB budget.
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

  // Join the planner's declared citations against what the verifier independently measured.
  for (const r of clean) {
    for (const cardRef of (r.d.learned_refs || [])) {
      citations.push({
        card: cardRef, round, direction: r.d.id, specialty: r.d.specialty,
        // null, NOT 0, when the verifier produced nothing.
        cited_then_verified: r.ver && Number.isFinite(r.ver.verified_geomean)
          ? r.ver.verified_geomean : null,
        // `winner` is only the best candidate OF THIS ROUND; `improved` is whether it beat the
        // incumbent by MIN_IMPROVE.
        became_winner: !!(winner && winner.id === r.d.id && improved),
      });
    }
  }
  if (!MEGA) history.rounds.push({
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

  if (MEGA && megaStopRule(round, megaTurns.done, megaSelected)) break;
  // mode=mega keeps every round's artifacts: they are its candidates' Verify evidence.
  if (MEGA) continue;

  // Issue #429: reclaim superseded heavy copies after each round, then CONTINUE optimizing.
  const reclaimCmd =
    `bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round}` +
    `; bytes=$(du -sb ${EVAL_DIR} 2>/dev/null | awk '{print $1}');` +
    ` soft=\${GEAK_EVAL_SOFT_BYTES:-34359738368}; hard=\${GEAK_EVAL_HARD_BYTES:-68719476736};` +
    ` if [ -n "$bytes" ] && [ "$bytes" -gt "$hard" ]; then` +
    `   bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round} --force-heavy;` +
    ` elif [ -n "$bytes" ] && [ "$bytes" -gt "$soft" ]; then` +
    `   bash ${WORKFLOW_DIR}/scripts/reclaim_eval_artifacts.sh --eval-dir ${EVAL_DIR} --keep-round ${round};` +
    ` fi` +
    `; echo STORAGE_RECLAIM_DONE round=${round}`;
  if (ABL_B6) {
    ABL_PENDING.push(reclaimCmd);
    log(`  [B6] storage reclaim r${round} folded into the next agent — no dedicated dispatch.`);
  } else
  await agentT(
    `Storage reclaim after round ${round} (issue #429). Run EXACTLY this bash, then return ` +
    `{ok:true, note:"reclaimed"}. Do NOT stop optimizing — reclaim frees disk so later rounds can run.\n` +
    `\`\`\`bash\n${reclaimCmd}\n\`\`\``,
    { phase: 'Optimize', label: `storage:reclaim r${round}`,
      schema: { type: 'object', additionalProperties: true,
        properties: { ok: { type: 'boolean' }, note: { type: 'string' } },
        required: ['ok'] } });
}

if (MEGA) {
  const r = await megaAfterLoop();
  if (r) return r.exit;
}

// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.' +
    (UT_MR ? ' Multi-GPU job: copy each case\'s per-rank baseline times and alignment from ' +
      'BASELINE_PER_CASE (rank_ms, alignment) into its per_case row (baseline_rank_ms, alignment) when ' +
      'they are there; a case with only the merged time keeps it.' : ''), MEGA ? megaReportInputs() : {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
    ...MR_INPUTS,
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: UT_MR ? REPORT_SCHEMA_MR : REPORT_SCHEMA, ...(MEGA ? megaFinalStepOpts() : {}) });
if (MEGA) {
  const r = await megaAfterReport();
  if (r) return r.exit;
}

// ===========================================================================
// PHASE: Director validation + arbitration
// ===========================================================================
phase('Validate');
const validation = MEGA && megaSkipsValidation() ? null : await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', MEGA ? megaValidateInputs() : {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    APPLY_TO_ORIGINAL, COMMANDMENT, FROZEN_ORACLE: FROZEN_ORACLE ? 'true' : 'false',
    FINAL_PATCH: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
    ...MR_INPUTS,
  }),
  { phase: 'Validate', label: 'director:validate', schema: MEGA ? MEGA_VALIDATE_SCHEMA : UT_MR ? VALIDATE_SCHEMA_MR : VALIDATE_SCHEMA,
    ...(MEGA ? megaFinalStepOpts() : {}) });

// Multi-rank receipt (A01). The director reports the merged time of every timed case (the baseline
// leg always, the candidate leg once a candidate was produced), plus each rank's time and the
// alignment method when the UT reports them. Neither an incomplete receipt nor missing per-rank detail
// changes validation_status: what is missing goes into arbitration_note and timing_receipt_gap. The
// lane returns the director's receipt when it has the merged times, else the Benchmark's baseline
// record, else null.
let TIMING_RECEIPT = null;
let RECEIPT_GAP = '';
if (UT_MR) {
  RECEIPT_GAP = validation ? receiptGap(validation.timing_receipt, RANKS, !!finalWinner)
    : 'the director returned no validation';
  const note = validation && !RECEIPT_GAP ? receiptNote(validation.timing_receipt, RANKS, !!finalWinner) : '';
  const missing = [RECEIPT_GAP, note].filter(Boolean).join('; ');
  if (validation && missing) {
    validation.arbitration_note = `${validation.arbitration_note ? `${validation.arbitration_note} ` : ''}` +
      `[lane] ${RANKS}-GPU job, per-rank receipt incomplete: ${missing}; merged times used.`;
    log(`Validate note: ${RANKS}-GPU job, ${missing}; validation_status unchanged.`);
  }
  if (!RECEIPT_GAP && note) RECEIPT_GAP = note;
  TIMING_RECEIPT = validation && !receiptGap(validation.timing_receipt, RANKS, !!finalWinner)
    ? validation.timing_receipt
    : (!BENCH_RANK_GAP ? benchReceipt(BASELINE_PER_CASE, RANKS, bench.rank_timing_source) : null);
}

const finalGeomean = validation ? validation.director_verified_speedup_geomean : cumulative;
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = validation && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
if (MEGA) return await megaFinish();
const finalPrimary = HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean;
log(`COMPLETE. ${KERNEL_NAME}: verified ${HAS_WORKLOAD ? 'time-weighted' : 'geomean'} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${validation ? validation.validation_status : '?'}). Results in ${EVAL_DIR}`);

// ===========================================================================
// PHASE: UpdateExperience — distill ONE reusable card into this workflow's knowledge/learned/ (see
// knowledge/learned/README.md).
// ===========================================================================
let learned_card = null;
// Two runs must never become a card, and neither is a failure.
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
          CURATION_ISOLATE: GPU_FAMILY.isolate ? 'true' : 'false',
          REQUIRED_PLATFORMS: GPU_FAMILY.isolate ? [GFX] : [],
          WINNER: {
            kernel: KERNEL_NAME, language: TARGET_LANGUAGE, mode: MODE, gfx: GFX,
            device_target: DEVICE_TARGET, physical_cu_count: PHYSICAL_CU_COUNT,
            kernel_class: (analysis && analysis.kernel_type) || '',
            speedup: finalPrimary, validation_status: validation ? validation.validation_status : '',
            bottleneck: profileSummary ? profileSummary.bottleneck : '',
          },
          // `rounds[]` (per-round directions + per-candidate claimed/verified/status + running
          // `cumulative`) is what makes the card's `stack:` attribution and `pitfall:` lines derivable.
          HISTORY: history,
          // What this run's plan cited and what the verifier then measured.
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
// Write this run's outcome back — the producer half of the loop.
// ===========================================================================
let kb_written = null;
if (KB_WRITE_OK && GFX && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  const kernelClass = (analysis && analysis.kernel_type) || 'unknown';
  const finalPatch = report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`;
  const reportPath = report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`;
  // What the store cannot infer from a diff but needs to curate this entry later.
  const winnerDirection = kebab(warm_start.direction || '');
  const metricKind = HAS_WORKLOAD ? 'time_weighted' : 'geomean';
  // Commas separate the list, so a case name may not contain one.
  const caseNames = (bestPerCase || []).map(c => c && String(c.name || '').replace(/,/g, ';'))
    .filter(Boolean).join(',');
  // write-remote runs the directory write first and files the same entry in the store, so the two
  // planes cannot drift; its extra `remote` field rides along under the schema's open object.
  const remoteWriteOn = KB_REMOTE !== 'off' && !!KB_STORE_DIR;
  const writeCmd = remoteWriteOn
    ? `write-remote --plane both --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
    : KB_MODE === 'store'
      ? `write-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : 'write';
  kb_written = await agentT(
    `You are the experience writer. Run EXACTLY this command (it applies its own gates and prints a ` +
    `single-line JSON) and return that JSON verbatim as StructuredOutput. If the command errors, return ` +
    `{"written": false, "reason": "io_error"}` +
    (remoteWriteOn
      ? ` — do NOT retry it and do NOT adjust its arguments. It may reach the shared KB Store service, ` +
        `and every write that service accepts is PERMANENT (it exposes no delete), so a second attempt ` +
        `creates a second permanent record instead of fixing the first.`
      : `.`) + `
\`\`\`bash
${remoteWriteOn ? KB_ENV_PRELUDE + '\n' : ''}python3 ${JSON.stringify(EXPERIENCE_STORE)} ${writeCmd} --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\
  --gfx ${GFX} --kernel-class ${JSON.stringify(kernelClass)} \\
  --speedup ${finalPrimary} --baseline-wall-ms ${BASELINE_GEOMEAN_MS} \\
  --patch ${JSON.stringify(finalPatch)} --eval-dir ${JSON.stringify(EVAL_DIR)} \\
  --report ${JSON.stringify(reportPath)} --metric-kind ${metricKind} \\
  --direction ${JSON.stringify(winnerDirection)} --case-names ${JSON.stringify(caseNames)}\
${OP_SPEC.dtype ? ` \\\n  --precision ${JSON.stringify(String(OP_SPEC.dtype))}` : ''}\
${warm_start.exp_dir ? ` \\\n  --parent ${JSON.stringify(warm_start.exp_dir)}` : ''}
\`\`\``,
    { phase: 'Validate', label: 'kb:write', schema: WARMSTART_WRITE_SCHEMA });
  const remoteWrite = (kb_written && kb_written.remote) || null;
  if (remoteWrite) {
    log(remoteWrite.written
      // `replaced` distinguishes the two write outcomes the store has: a NEW patch appends a session
      // under the same key, the SAME patch remeasured lands back on its own content-addressed one.
      ? `[kb] plane=store ${remoteWrite.replaced ? 'updated' : 'appended'} ` +
        `${remoteWrite.canonical_id}/${remoteWrite.session_id}${remoteWrite.champion ? ' (champion)' : ''}`
      : `[kb] plane=store not written: ${remoteWrite.reason || 'unknown'}`);
  }
  log(kb_written && kb_written.written
    ? `[kb] experience written: ${kb_written.slug} (speedup ${finalPrimary.toFixed(2)}, direction ` +
      `${winnerDirection || 'unlabeled'})`
    : kb_written && kb_written.reason === 'duplicate_impl'
      // Not a failure — the expected outcome of adopting a warm start and not beating it.
      ? `[kb] experience already known — counted as reproduction #${kb_written.reproductions || '?'} of ` +
        `${kb_written.reproduced || kb_written.dir}`
      : `[kb] experience not written: ${kb_written ? kb_written.reason : 'writer returned nothing'}`);
}

// finalPrimary is the total vs the pristine baseline; when a warm-start patch was adopted, split out
// the delta ABOVE it so a KB-derived gain is never reported as this run's own work.
//
// With no round committed after the adopt, the workspace still holds exactly the adopted code, and the
// two numbers are two measurements of the SAME thing taken at different times — the ratio is bench
// noise (~2% on this box; `_gemm_a16_w16_kernel` reported 0.98 and read as a regression).
const noRoundsAfterAdopt = !!(warm_start.adopted && roundsCommitted === 0);
const incrementalSpeedup = warm_start.adopted && warm_start.adopted_speedup
  ? (noRoundsAfterAdopt ? 1.0
     : Number.isFinite(finalPrimary) ? finalPrimary / warm_start.adopted_speedup : null)
  : finalPrimary;

return {
  ...(EVO ? { evolution: EVO.result() } : EVO_FALLBACK ? { evolution: EVO_FALLBACK } : {}),
  mode: MODE,
  // `instance` = this lane's eval dir: a stable identity so the parser dedupes a timeline reached
  // twice (parent-merge + glob) without collapsing two distinct lanes that share a shape.
  llm_timeline: LLM_STATS ? { ...LLM_TL, instance: EVAL_DIR } : undefined,
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
  // What KIND of number `final_speedup` is, so bake-off ranking and campaign summaries don't read a
  // host_bound/unprimed float as a clean device-time win.
  timing_basis: validation ? validation.timing_basis || 'unknown' : 'unknown',
  // UT-module multi-GPU tasks only (UT_MR; any other return has none of these keys): the per-rank
  // timing receipt (A01) and, when the director's was missing or incomplete, why ('' when complete).
  ...(UT_MR ? { gpus_per_job: RANKS, timing_receipt: TIMING_RECEIPT, timing_receipt_gap: RECEIPT_GAP } : {}),
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
  // null when the step was off, or the run earned nothing worth a card.
  stopped_by: deadlineHit ? 'deadline'
    : (ROUTING_ON && __routeKilled) ? 'route_output_cap'
    : (dispatched >= BUDGET) ? 'budget'
    : (noImprove >= MAX_NO_IMPROVE) ? 'no_improve'
    : (plannerStopReason || 'tech_lead_stop'),
  deadline_hit: deadlineHit,
  // Eikos shadow pilot: one record per reached decision point, plus why the loop ended. Absent when OFF.
  eikos_shadow: EIKOS_SHADOW_ON ? {
    decision: 'round_continue', carrier_model: EIKOS_CARRIER_MODEL, carrier_timeout_ms: EIKOS_CARRIER_TIMEOUT_MS,
    controls_branch: false, records: EIKOS_SHADOW_LOG,
    loop_end_reason: deadlineHit ? 'deadline' : (dispatched >= BUDGET) ? 'budget'
      : (noImprove >= MAX_NO_IMPROVE) ? 'no_improve' : 'tech_lead_stop',
  } : undefined,
  // Expt-3 cost ladder: lanes, thresholds, every decision/dispatch/failure. Absent when routing is OFF.
  routing: ROUTING_ON ? __routeReport() : undefined,
  forced_replans: forcedReplans,
  // What the plan cited and whether it carried its round.
  learned_citations: citations,
  // Rounds where the planner overran the KB budget.
  kb_cap_bound_rounds: kbCapBound,
  // The monoculture canary, and it costs no GPU time: how many DISTINCT (specialty, focus_files)
  // directions this run explored vs how many it issued.
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
    // How the store was reached and what it filtered out: tells a genuine cold start from a
    // name/language mismatch that only looks like an empty KB.
    match_tier: warm_start.match_tier || '',
    filtered: warm_start.filtered || null,
    adopted_speedup: warm_start.adopted ? warm_start.adopted_speedup : null,
    total_speedup: Number.isFinite(finalPrimary) ? finalPrimary : null,
    incremental_speedup: incrementalSpeedup,
    // true = the run adopted a stored patch and never improved on it; incremental is 1.0 by
    // definition, not by measurement.
    no_rounds_after_adopt: noRoundsAfterAdopt,
    rounds_committed: roundsCommitted,
    incremental_improved: !!(warm_start.adopted && Number.isFinite(incrementalSpeedup) && incrementalSpeedup > 1 + MIN_IMPROVE),
    returned_after_read_kb: !!warm_start.returned_after_read_kb,
    candidates: warm_start.candidates,
  },
  kb_written: kb_written && kb_written.written ? { slug: kb_written.slug, dir: kb_written.dir } : null,
};

function createEvolution() {
  let sourcePaths = A.evolution_source_paths == null ? [] : A.evolution_source_paths;
  if (A.kernel_lane_script) throw new Error('optimizer=evo cannot override kernel_lane_script');
  if (!['optimize', 'author'].includes(String(A.mode || 'optimize').trim().toLowerCase() || 'optimize')) {
    throw new Error('optimizer=evo currently supports optimize and author lanes, not bakeoff');
  }
  function validateSourcePaths(paths) {
    if (!Array.isArray(paths) || paths.length > 256 ||
      new Set(paths).size !== paths.length ||
      paths.some(sourcePath => typeof sourcePath !== 'string' || !sourcePath ||
        sourcePath.length > 1024 || sourcePath.includes('\\') || sourcePath.startsWith('/') ||
        sourcePath.split('/').some(part => !part || part === '.' || part === '..' || part.toLowerCase() === '.git') ||
        /[^\x20-\x7e]/.test(sourcePath) || paths.some(other => sourcePath.startsWith(other + '/')))) {
      throw new Error('EVO analyzed/supplemental evolution_source_paths must list safe relative source files, not directories');
    }
    return paths;
  }
  const helper = `${WORKFLOW_DIR}/../evo/archive.py`;
  const errors = [];
  let archive = '';
  const quote = value => "'" + String(value).replace(/'/g, "'\"'\"'") + "'";
  const command = (operation, values) => ['python3', helper, operation, '--archive', archive, ...values]
    .map(quote).join(' ');

  async function run(operation, values, prelude = '') {
    const result = await agentT(
      'Execute the following deterministic archive command with Bash exactly as supplied. ' +
      'Do not edit source, repair failures, rerun a successful command, or invent output. ' +
      'This operation is source bookkeeping, not correctness or performance verification. ' +
      'Return its actual exit_code and the complete stdout as a string; put diagnostics in stderr.\n' +
      '```bash\nset -euo pipefail\n' + prelude + command(operation, values) + '\n```',
      { phase: 'Optimize', label: `evo:${operation}`,
        schema: obj({ exit_code: { type: 'number' }, stdout: { type: 'string' }, stderr: { type: 'string' } },
          ['exit_code', 'stdout']) });
    if (!result || result.exit_code !== 0 || typeof result.stdout !== 'string') {
      let diagnostic = result ? result.stderr || 'command failed' : 'no return';
      if (result && typeof result.stdout === 'string') {
        let receipt;
        try { receipt = JSON.parse(result.stdout); } catch {}
        if (receipt && receipt.ok === false && receipt.error &&
            typeof receipt.error.code === 'string' && typeof receipt.error.message === 'string') {
          diagnostic = `${receipt.error.code}: ${receipt.error.message}`;
        }
      }
      throw new Error(`EVO ${operation} did not complete: ${diagnostic}`);
    }
    let receipt;
    try { receipt = JSON.parse(result.stdout); } catch { receipt = null; }
    if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt) ||
        receipt.ok !== true || receipt.command !== operation || receipt.archive !== archive) {
      throw new Error(`EVO ${operation} returned an invalid receipt`);
    }
    return receipt;
  }

  // Returns the init receipt, or null when EVO cannot start (unusable source list or a failed/invalid
  // init receipt). The caller then runs the ordinary lane; the reason is kept in result().errors.
  async function initialize() {
    try {
      const editable = analysis && analysis.modifiable_files != null ? analysis.modifiable_files : [];
      validateSourcePaths(editable);
      validateSourcePaths(sourcePaths);
      sourcePaths = validateSourcePaths([...new Set([...editable, ...sourcePaths])].sort());
      if (!sourcePaths.length) {
        throw new Error('optimizer=evo requires source files from Analyze modifiable_files or evolution_source_paths');
      }
      archive = `${String(EVAL_DIR).replace(/\/+$/, '')}/evolution_archive`;
      return await run('init', ['--canonical', CANONICAL,
        ...sourcePaths.flatMap(sourcePath => ['--source-path', sourcePath])]);
    } catch (error) {
      const message = String(error && error.message || error);
      errors.push({ operation: 'init', message, fallback: 'ordinary_lane' });
      log(`EVO could not start; continuing as the ordinary lane: ${message}`);
      return null;
    }
  }

  function plannerInput() {
    return { archive, query_command: command('query', []), source_paths: sourcePaths };
  }

  async function prepare(direction) {
    const workspace = `${direction.out_dir}/workspace`;
    const prelude = `mkdir -p ${quote(direction.out_dir)}\n` +
      ['bash', `${WORKFLOW_DIR}/scripts/materialize_workspace.sh`, '--src', CANONICAL,
        '--dst', workspace, '--shared-root', `${EVAL_DIR}/_shared`, '--link-aiter'].map(quote).join(' ') +
        ` > ${quote(`${direction.out_dir}/materialize.json`)}\n`;
    const donorIds = direction.donor_ids == null ? [] : direction.donor_ids;
    if (!Array.isArray(donorIds) || donorIds.some(donorId => typeof donorId !== 'string')) {
      throw new Error('donor_ids must be a list of archive entry IDs');
    }
    if (direction.parent_id != null && typeof direction.parent_id !== 'string') {
      throw new Error('parent_id must be an archive entry ID');
    }
    const parentId = direction.parent_id || 'INCUMBENT';
    const prepared = await run('prepare', ['--workspace', workspace, '--canonical', CANONICAL,
      '--initialize-index', '--parent', parentId,
      ...donorIds.flatMap(donorId => ['--donor', donorId])], prelude);
    if (!prepared.receipt || prepared.receipt.workspace !== workspace ||
        !/^[a-f0-9]{64}$/.test(prepared.receipt.materialized_canonical_source_id) ||
        prepared.receipt.requested_parent_id !== parentId || !Array.isArray(prepared.receipt.donors) ||
        JSON.stringify(prepared.receipt.donors.map(donor => donor.id)) !== JSON.stringify(donorIds)) {
      throw new Error('prepare receipt does not match the selected workspace, parent and donors');
    }
    return prepared;
  }

  function engineerInput(direction) {
    return { ...plannerInput(), parent_id: direction.parent_id || 'INCUMBENT',
      donor_ids: direction.donor_ids || [], prepared: direction.evo_prepared };
  }

  async function recordBatch(clean) {
    const reports = clean.map(result => ({
      id: `r${round}_d${result.d.idx}`,
      parent_id: result.d.parent_id || 'INCUMBENT', donor_ids: result.d.donor_ids || [],
      title: result.d.title, workspace: `${result.d.out_dir}/workspace`,
      patch: result.patch || `${result.d.out_dir}/best_patch.diff`, engineer: result.eng, verifier: result.ver,
    }));
    try {
      const reportJson = JSON.stringify(reports);
      const receipt = await run('record-batch', ['--canonical', CANONICAL, '--report-json', reportJson,
        '--report-sha256', sha256Hex(reportJson)]);
      if (receipt.has_failures) errors.push({ round, operation: 'record-batch', results: receipt.results });
      return receipt;
    } catch (error) {
      const message = String(error && error.message || error);
      errors.push({ round, operation: 'record-batch', message });
      log(message);
      return null;
    }
  }

  async function checkPatch(direction, patch) {
    try {
      const receipt = await run('check-patch', ['--canonical', CANONICAL, '--patch', patch]);
      if (typeof receipt.patch_object_path !== 'string' || !/^[a-f0-9]{64}$/.test(receipt.patch_sha256) ||
          receipt.patch_object_path !== `${archive}/objects/${receipt.patch_sha256}`) {
        throw new Error('check-patch did not return a bound immutable patch');
      }
      return receipt;
    } catch (error) {
      const message = String(error && error.message || error);
      errors.push({ round, operation: 'check-patch', id: direction.id, message });
      log(message);
      return null;
    }
  }

  return { initialize, plannerInput, prepare, engineerInput, recordBatch, checkPatch, run,
    result: () => ({ archive, errors }) };
}

// SHA-256 of a string's UTF-8 bytes (FIPS 180-4).
function sha256Hex(text) {
  const bytes = [];
  for (const character of unescape(encodeURIComponent(text))) bytes.push(character.charCodeAt(0));
  const K = [];
  for (let candidate = 2, found = 0; found < 64; candidate++) {
    let prime = true;
    for (let divisor = 2; divisor * divisor <= candidate; divisor++) if (candidate % divisor === 0) { prime = false; break; }
    if (prime) { K.push((Math.cbrt(candidate) % 1) * 2 ** 32 >>> 0); found++; }
  }
  const H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const bitLength = bytes.length * 8;
  bytes.push(0x80);
  while (bytes.length % 64 !== 56) bytes.push(0);
  for (let shift = 56; shift >= 0; shift -= 8) bytes.push(Math.floor(bitLength / 2 ** shift) & 0xff);
  const rotate = (value, count) => (value >>> count) | (value << (32 - count));
  const W = new Array(64);
  for (let offset = 0; offset < bytes.length; offset += 64) {
    for (let i = 0; i < 16; i++) {
      W[i] = (bytes[offset + 4 * i] << 24) | (bytes[offset + 4 * i + 1] << 16) |
        (bytes[offset + 4 * i + 2] << 8) | bytes[offset + 4 * i + 3];
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotate(W[i - 15], 7) ^ rotate(W[i - 15], 18) ^ (W[i - 15] >>> 3);
      const s1 = rotate(W[i - 2], 17) ^ rotate(W[i - 2], 19) ^ (W[i - 2] >>> 10);
      W[i] = (W[i - 16] + s0 + W[i - 7] + s1) | 0;
    }
    let [a, b, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      const t1 = (h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + W[i]) | 0;
      const t2 = ((rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) | 0;
      h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
    }
    [a, b, c, d, e, f, g, h].forEach((value, index) => { H[index] = (H[index] + value) | 0; });
  }
  return H.map(value => (value >>> 0).toString(16).padStart(8, '0')).join('');
}
