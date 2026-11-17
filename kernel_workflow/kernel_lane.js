export const meta = {
  name: 'kernel-lane',
  description: 'SINGLE-LANGUAGE kernel optimization worker (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, and integration. Optimizes ONE kernel in ONE language (mode=optimize) or authors a fresh seed then optimizes it (mode=author). This is the worker invoked per lane by the kernel-workflow dispatcher (kernel_workflow.js) and by e2e_workflow; prefer calling kernel-workflow directly unless you specifically want one unchanged lane. Target: AMD Instinct MI-series (CDNA gfx942/gfx950) and the validated RDNA4 product Radeon AI PRO R9700 (gfx1201) — auto-detected on-box; on R9700 use knowledge/amd_rdna4.md (wave32, WMMA). Generic gfx1201 without R9700 identity is not a calibrated target.',
  whenToUse: 'Internal single-language worker. Prefer the kernel-workflow dispatcher (kernel_workflow.js) as the entry point; invoke this directly only to run one unchanged lane. Pass args.kernel_path (required), args.mode, args.target_language, args.budget, args.gpu_ids, args.gpu_mode, args.gpus_per_job, args.job_gpu_ids, args.task.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies the bottleneck' },
    { title: 'Research', detail: 'OPT-IN (args.dra_enabled): researcher fans research questions out in parallel via native WebSearch/WebFetch, writes a ranked-directions brief the planner seeds from' },
    { title: 'WarmStart', detail: 'search the experience KB by canonical id (geak:kernel:*) — the shared service when credentialed, else the on-disk store, brought level with kb_artifacts/ first — for the best curated patch per optimization direction for this (kernel,language,gfx), validate each through the verify gate, adopt the first that passes [warm_start!=off]' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline, then the TechLead curates ONE distilled card into knowledge/learned/ on a measured win [update_experience!=off]' },
  ],
};

// Args + defaults. No filesystem access here: agents do all FS work and every path comes from args.
const A = args || {};
if (A.optimizer != null && typeof A.optimizer !== 'string') {
  throw new Error('optimizer must be greedy or evo (a string)');
}
const OPTIMIZER = String(A.optimizer || '').trim().toLowerCase();
if (OPTIMIZER && !['greedy', 'evo'].includes(OPTIMIZER)) {
  throw new Error('optimizer must be greedy or evo');
}
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/model directory)');

// WORKFLOW_DIR = dirname of the launching scriptPath (holds roles/, knowledge/, scripts/).
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js, roles/, knowledge/, scripts/ (i.e. the dirname of this script).');
}
// EXP_ROOT: run dirs; default a sibling exp/ beside kernel_workflow/, or args.exp_root.
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
];
const GPU_FAMILY_DEFAULT = { name: 'default', isa: null, isolate: false, matrixCore: 'MFMA' };
const gpuFamily = (gfx) => GPU_FAMILIES.find((f) => f.isa.test(String(gfx || ''))) || GPU_FAMILY_DEFAULT;
// ---- end GPU families ----
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// Min verified geomean gain over the cumulative best for a round winner to be COMMITTED (default
// 2%). Raise it on a noisy box; lower it to bank small compounding wins.
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
})();
// Min verified speedup to enter the round's candidate list (default 1.0). Lower it for a
// transcription (e.g. Triton -> Gluon/HIP) that starts below the baseline, or its recovery stays
// invisible. Committing still needs MIN_IMPROVE over `cumulative`, so such a candidate is tracked but
// never banked.
const CANDIDATE_FLOOR = (() => {
  const v = parseFloat(A.candidate_floor != null ? A.candidate_floor : 1.0);
  return Number.isFinite(v) && v > 0 ? v : 1.0;
})();
// For the Optimize prompt: `${1.0}` would print "1"; the default prompt keeps saying "geomean>1.0".
const CANDIDATE_FLOOR_TXT = Number.isInteger(CANDIDATE_FLOOR)
  ? CANDIDATE_FLOOR.toFixed(1) : String(CANDIDATE_FLOOR);
// A round made progress when its winner beats the best candidate seen by PROGRESS_DELTA (default
// MIN_IMPROVE) even without clearing `cumulative`, so a climb from below the baseline is not ended
// at MAX_NO_IMPROVE. Negative also counts a round that gives some ground.
const PROGRESS_DELTA = (() => {
  const v = parseFloat(A.progress_delta != null ? A.progress_delta : MIN_IMPROVE);
  return Number.isFinite(v) && v > -1 ? v : MIN_IMPROVE;
})();
// Direction-budget cost of one `deep_explore` direction (default 2): it runs its own
// measure->profile->rewrite loop, always in a dedicated round (enforced below).
const DEEP_COST = (() => {
  const v = parseInt(A.deep_cost != null ? A.deep_cost : 2, 10);
  return Number.isFinite(v) && v >= 1 ? v : 2;
})();
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map(s => s.trim()).filter(Boolean);
// Default 'pool': every GPU consumer gets the whole pool and gpu_lock.sh flocks whichever GPU is free
// and idle at acquire time, so placement follows task cost (a pinned round waits on its slowest
// lane). gpu_mode='pin' (direction i on GPU_LIST[i % n]) exists only to A/B the scheduler.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
// Multi-GPU jobs: gpus_per_job > 1 leases any N of gpu_ids per command (gpu_lock.sh pool:N:<ids>);
// job_gpu_ids pins one group (group:<ids>). Either may come from op_spec.resource, else Setup's
// `detected` count (scripts/detect_gpu_requirement.py). A detected count that does not fit gpu_ids
// stays single-GPU with a warning; only an explicit request throws.
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
  GPU_JOB = r.jobSpec;
  GPU_POOL = GPU_JOB || (GPU_MODE === 'pin' ? GPU_LIST[0] : GPU_LIST.join(','));
  GPU_SINGLE = GPU_JOB || GPU_LIST[0];
}
applyGpuResource(resolveGpuRequest(A, GPU_LIST));
const TASK = A.task || '';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();

// mode=author: seed the workspace from an op task dir (oracle + frozen online kernel in
// baseline_src/), author_engineer writes a passing seed, then the same optimize loop runs, timed
// against the frozen online kernel. KERNEL_KNOWLEDGE_DIR (default sibling perf_knowledge/) is
// reference only.
const MODE = String(A.mode != null ? A.mode : 'optimize').trim() || 'optimize';
const TARGET_LANGUAGE = String(A.target_language != null ? A.target_language : 'triton').trim() || 'triton';
const OP_SPEC = A.op_spec || {};
// e2e sets op_spec.cuda_graph_safe=true for ops on the graph-captured decode path; verify then adds
// a capture+replay smoke for kernels that pass isolated but host-sync or lazily compile under
// capture (cuda_graph_capture_unsafe / NO_BINARY_FOR_GPU).
const REQUIRE_GRAPH_CAPTURE = !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
// Optional workload alignment: benchmark exactly the workload's (shape, dtype) cases, weight each by
// count * baseline_latency, and optimize the time-weighted ratio-of-sums instead of the geomean.
//   workload_spec_path  workload-v1 json (parse_profile.py --workload-out); benchmark_engineer reads it
//   op_spec.workload    inline cases (workload-v1 "kernels[].cases" or the whole object); takes
//                       precedence, weight_source becomes "caller"
// Correctness always stays on the frozen oracle; this only shapes the performance measurement.
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const WORKLOAD_SPEC = (OP_SPEC && OP_SPEC.workload) || A.workload || null;
const HAS_WORKLOAD = !!(WORKLOAD_SPEC_PATH ||
  (Array.isArray(WORKLOAD_SPEC) && WORKLOAD_SPEC.length) ||
  (WORKLOAD_SPEC && Array.isArray(WORKLOAD_SPEC.kernels) && WORKLOAD_SPEC.kernels.length));
// Primary metric: the time-weighted speedup when a workload spec is in play and reported, else geomean.
const primSpeedup = (o) => {
  if (!o) return 0;
  const w = o.verified_weighted != null ? o.verified_weighted
          : (o.speedup_weighted != null ? o.speedup_weighted : null);
  if (HAS_WORKLOAD && Number.isFinite(w)) return w;
  const g = o.verified_geomean != null ? o.verified_geomean : o.speedup_geomean;
  return Number.isFinite(g) ? g : 0;
};
// Gate fields (`correctness`, `status`) are free strings ("PASS - 15/15 draws"): match the leading
// word, not `=== 'pass'`. "PASS ..."/"passed" open the gate; "FAIL"/"did not pass" do not.
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
// Expert skills (perf_knowledge/expert_skills/): human-authored recipes, advisory only. A matched
// `validated` skill is a high-prior candidate that planning/author roles reproduce and gate by the
// isolated A/B; it never overrides measurement. Opt-in (use_expert_skills="true"); off injects nothing.
const USE_EXPERT_SKILLS_REQUESTED = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
let USE_EXPERT_SKILLS = USE_EXPERT_SKILLS_REQUESTED;
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
// Only planning + authoring roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);
// Expected-identity isolation must apply before Setup (no CDNA skills/KB on a declared isolated run).
// Detected identity is OR'd in after director setup — do not read GFX here (TDZ).
let SKILLS_ISOLATE = gpuFamily(EXPECTED_GFX).isolate;

// Deep Research Agent (DRA), opt-in (dra_enabled="true"): after Profile the `researcher` derives
// ranked research questions, the script fans them out in parallel (one hang-guarded WebSearch/
// WebFetch agent each), and a synthesis pass writes deep_search.{md,json} + deep_search_brief.md
// into EVAL_DIR for plan_round to seed from.
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

// Learned-knowledge sink, derived from WORKFLOW_DIR only, so lanes spawned by e2e also curate into
// kernel_workflow/knowledge/learned/. update_experience = on (default) | off | false | none; the
// bake-off dispatcher passes `off` and curates once centrally.
const LEARNED_DIR = `${WORKFLOW_DIR}/knowledge/learned`;
// How much of one round the learned KB may steer (a bounded planner beat free picking from the whole
// index by 22.7% geomean on 16 kernels; the KB is "an accelerant, NOT a cage"):
//   KB_DIR_CAP        directions per round that may draw on cards (default 1)
//   KB_COLD_DIRECTION at least one direction per round must be profile-only (default on)
// USE_LEARNED_READ is declared in this file because it is used here (test_lane_refs.py). Default on;
// use_learned_kb=false also turns the budget block off.
let USE_LEARNED_READ = String(A.use_learned_kb != null ? A.use_learned_kb : 'true') === 'true';
const KB_DIR_CAP = Math.max(0, parseInt(A.kb_dir_cap != null ? A.kb_dir_cap : 1, 10));
const KB_COLD_DIRECTION = String(A.kb_cold_direction != null ? A.kb_cold_direction : 'true') === 'true';
let kbCapBound = 0;      // rounds where the cap actually had to strip something
const UPDATE_EXPERIENCE = String(A.update_experience != null ? A.update_experience : 'on').trim().toLowerCase() || 'on';
const UPDATE_EXPERIENCE_ON = UPDATE_EXPERIENCE !== 'off' && UPDATE_EXPERIENCE !== 'false' && UPDATE_EXPERIENCE !== 'none';
// Whether the caller froze the kernel via oracle_freezer (only the bake-off dispatcher does), so
// director knows if a GEAK_TIMING_RECEIPT can exist. Must be an arg: this lane never runs Freeze.
const FROZEN_ORACLE = String(A.frozen_oracle != null ? A.frozen_oracle : 'false') === 'true';

// Warm start from the experience KB (canonical id, both planes): before the optimize loop, run the
// top-3 patches for this (kernel, language, gfx) through the verify_engineer gate; write the win back
// after. on (default) adopts the first that passes; reference reads them as prose only;
// return_after_read adopts then returns; off/false/none, a STATE_DIR resume or no arch = cold start.
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
// How loosely the resolver maps this kernel name to a stored page: names are layout-derived
// (`fused_moe_kernel` vs e2e's `fused_moe_kernel_task`), so `exact` misses; `fuzzy` also takes op_kind.
const WARM_START_MATCH = ['exact', 'normalized', 'fuzzy'].includes(String(A.warm_start_match || '').trim())
  ? String(A.warm_start_match).trim() : 'fuzzy';
// Plane the experience is read from (only the command differs; both print the same JSON):
//   store (default)  on-disk KB Store keyed by canonical id, the service's addressing
//   local            the kb_artifacts/ tree keyed by slug; escape hatch for an unusable store
// Writes still go to both (`write-remote --plane both`) and `sync-local` runs before each read; the
// tree stays the buffer a curation pass edits.
const KB_MODE = String(A.kb_mode || 'store').trim().toLowerCase() === 'local' ? 'local' : 'store';
const KB_STORE_DIR = String(A.kb_store_dir ||
  (KB_ARTIFACTS_DIR ? KB_ARTIFACTS_DIR.replace(/\/[^/]*$/, '') + '/kb_store_local' : '')).replace(/\/+$/, '');
// The key carries rocm <major>.<minor>; without /opt/rocm records would file under `unspecified`.
const KB_FRAMEWORK_VERSION = String(A.kb_framework_version || '').trim();
const KB_VERSION_FLAG = KB_FRAMEWORK_VERSION ? ` --framework-version ${JSON.stringify(KB_FRAMEWORK_VERSION)}` : '';
const KB_ROOT_OK = KB_MODE === 'store' ? !!KB_STORE_DIR : !!KB_ARTIFACTS_DIR;
// KB Store service on top of KB_MODE's plane: `auto` (default) when credentials exist, `off` never.
const KB_REMOTE = String(A.kb_remote || 'auto').trim().toLowerCase() === 'off' ? 'off' : 'auto';
// Credentials live in the user's profile, invisible to this process (non-interactive shells), so
// every emitted command exports them itself and branches on what it got. The token is read from a
// 0600 file into a variable, never argv (`ps` is world-readable).
// If SSL_CERT_FILE is unset, point urllib/requests/curl/node at the first readable AMD-root bundle
// (path only; KB_CA_BUNDLE overrides) so the gateway's internal CA verifies inside containers.
// GEAK_KB_STORE_* wins over bare KB_STORE_* (a Hyperloom host has its own) and is re-exported bare.
const KB_ENV_PRELUDE =
  'export KB_STORE_URL="${GEAK_KB_STORE_URL:-${KB_STORE_URL:-https://global.primus-safe.amd.com/knowledge-base}}"; ' +
  'export KB_STORE_TOKEN="${GEAK_KB_STORE_TOKEN:-${KB_STORE_TOKEN:-$(cat ~/.geak_kb_token 2>/dev/null)}}"; ' +
  'if [ -z "${SSL_CERT_FILE:-}" ]; then for _ca in "${KB_CA_BUNDLE:-}" ' +
  '/shared_nfs/hyperloom/ca/amd-ca-combined.pem "$HOME/amd-extra-ca-bundle.pem"; do ' +
  '[ -n "$_ca" ] && [ -r "$_ca" ] && { export SSL_CERT_FILE="$_ca" REQUESTS_CA_BUNDLE="$_ca" ' +
  'CURL_CA_BUNDLE="$_ca" NODE_EXTRA_CA_CERTS="$_ca"; break; }; done; fi; ';
// A store-mode write records both planes in one call, so it needs both roots (the tree is the source).
const KB_WRITE_OK = KB_ROOT_OK && !!KB_ARTIFACTS_DIR;
const kebab = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);

// Deep-mode hooks, set only by e2e_workflow's deep_mode head scheduler; unset ones never reach a prompt.
//   STATE_DIR        stable dir per (kernel, backend) across deep waves: director resumes from
//                    STATE_DIR/best + prior_state, speedups stay vs the frozen baseline, and
//                    update_memory writes STATE.json and syncs best/ each round
//   SHARED_KB        cross-backend blackboard (read by plan + engineers, appended by update_memory)
//   GLOBAL_KB        run-global cross-kernel technique blackboard
//   E2E_FEEDBACK     latest e2e A/B result + problems (engaged?, cudagraph, memory, decode, delta)
//   HARNESS_ADDENDUM e2e-refined bench emphasis (timing weights, capture, hard gates); never the oracle
//   MAX_NO_IMPROVE   consecutive non-improving rounds before stopping (default 2)
const STATE_DIR = String(A.state_dir || '').replace(/\/+$/, '');
const SHARED_KB = String(A.shared_kb || '').trim();
const GLOBAL_KB = String(A.global_kb || '').trim();   // run-global cross-KERNEL technique blackboard (deep)
const E2E_FEEDBACK = String(A.e2e_feedback || '').trim();
const HARNESS_ADDENDUM = String(A.harness_addendum || '').trim();
// On a resumed deep wave, Analyze and the baseline Profile return the prior roadmap/profile with
// delta updates instead of re-deriving it. Benchmark is never incremental: it re-pins a fresh baseline.
const INCREMENTAL = !!STATE_DIR && String(A.incremental_analyze || '') === 'true';
const RESUME_INPUT = INCREMENTAL ? { INCREMENTAL_RESUME: '1' } : {};
const MAX_NO_IMPROVE = Math.max(1, parseInt(A.max_no_improve != null ? A.max_no_improve : 2, 10));
// Conditional inputs: an unset hook spreads {} and adds nothing to the prompt.
const KB_INPUTS = {
  ...(SHARED_KB ? { SHARED_KB } : {}),
  ...(GLOBAL_KB ? { GLOBAL_KB } : {}),
  ...(E2E_FEEDBACK ? { E2E_FEEDBACK } : {}),
  ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
};

// Reusable JSON-schema fragments.
const perCase = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      baseline_ms: { type: 'number' },
      optimized_ms: { type: 'number' },
      speedup: { type: 'number' },
      // Workload alignment only: weight = this case's baseline time share, the coefficient in
      // Σweight / Σ(weight/speedup); count is informational (regime cases have none).
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
  // Timing and parity baselines must be the real online kernel (baseline_src/ or an importable
  // meta.baseline_callable), never kernel_src/; the script aborts unless director confirms one.
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  // GPUs one run of the task's test/benchmark needs (roles/director.md "GPU requirement").
  gpu_requirement: obj({
    kind: { type: 'string', enum: ['single', 'fixed', 'scalable', 'unknown'] },
    gpus_per_job: { type: 'number' }, evidence: { type: 'string' },
  }, ['kind']),
  // Deep-mode resume only (STATE_DIR with a prior best): restores cumulative speedup + history.
  resumed: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name', 'device_gfx', 'device_target', 'physical_cu_count']);

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
  // perf_knowledge mapping (reference only): operator/language + most relevant card paths; empty when
  // no card applies.
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] },
  kk_refs: { type: 'array', items: { type: 'string' } },
}, ['kernel_type', 'roadmap_summary']);

const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  // Workload-aligned outputs: metric Σ weight_i / Σ (weight_i/speedup_i); baseline_weighted_total_ms
  // is Σ weight_i in ms; workload_aligned means the weights are real (not 1).
  workload_aligned: { type: 'boolean' },
  baseline_weighted_total_ms: { type: 'number' },
  weights_provenance: { type: 'string' }, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

const PROFILE_SCHEMA = obj({
  bottleneck: { type: 'string' }, profiler_used: { type: 'string' }, dispatch_count: { type: 'number' },
  // Detected accelerator (e.g. "MI300X / gfx942 / CDNA3, 304 CU, ~5.3 TB/s") for roofline + grid sizing.
  device: { type: 'string' },
  key_metrics: { type: 'object', additionalProperties: true },
  top_kernels: { type: 'array', items: { type: 'object', additionalProperties: true } },
  top_opportunities: { type: 'array', items: { type: 'string' } },
  summary_path: { type: 'string' }, shift_note: { type: 'string' },
}, ['bottleneck', 'top_opportunities']);

// DRA schemas: stages 0-2 facts + ranked questions; 3/4 one answer per question (parallel); 5 optional
// blindspot critique; 7 the ranked-directions portfolio + brief for the planner.
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

// Stage 7 portfolio; `directions` stays compact (the planner reads deep_search_brief.md on disk).
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
      // Learned cards (filenames) that seeded this direction, declared by the planner because its
      // reads are semantic; the script joins them with the verifier's result, the only way a card
      // can lose standing. A citation is not a causal claim.
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
  // Σ weight_i / Σ (weight_i / speedup_i) vs the true baseline; primary when workload_aligned.
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

// The failed-commit path needs no schema: it aborts and quarantines, with no snapshot or restore.

// No host-side commit fallback by design: an unconfirmed commit is quarantined and retained (the
// verified winner stays recoverable), never installed from agent-reported evidence.

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
  // clean | host_bound | unprimed | unknown | not_applicable; required, so no speedup goes unlabelled.
  timing_basis: { type: 'string' },
  per_case: perCase, applied_to_original: { type: 'string' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);

// Warm-start resolver output = experience_store.py `resolve` JSON, verbatim.
const WARMSTART_RESOLVE_SCHEMA = obj({
  read_reason: { type: 'string' }, slug: { type: 'string' },
  // Derived name vs the page served, so a surprising match is visible.
  requested_slug: { type: 'string' }, match_tier: { type: 'string' },
  // store mode only: the key the candidates came from (declared so the relaying agent keeps it).
  canonical_id: { type: 'string' },
  // Plane that answered ("remote" | "local"): the only field telling them apart; attest uses it.
  read_plane: { type: 'string' },
  // Same kernel, another language: the wrong target_language was passed, not an empty store.
  other_language_pages: { type: 'array', items: { type: 'string' } },
  filtered: obj({
    total: { type: 'number' }, retired: { type: 'number' }, below_min_speedup: { type: 'number' },
    same_direction_collapsed: { type: 'number' }, demoted_by_hint: { type: 'number' },
    // The service pages `--scan` rows: a saturated read covers only a prefix of the page.
    scanned: { type: 'number' }, scan_saturated: { type: 'boolean' },
  }),
  candidates: {
    type: 'array',
    items: obj({
      rank: { type: 'number' }, slug: { type: 'string' }, speedup: { type: 'number' },
      exp_dir: { type: 'string' }, arch: { type: 'string' },
      // Declared so attest has an address to write this run's verdict back to.
      session_id: { type: 'string' },
      patch_path: { type: 'string' }, prose_path: { type: 'string' },
      strategy: { type: 'string' }, status: { type: 'string' },
      // direction = the idea (one rank each); comparable=false: a different bench key than rank 1's.
      direction: { type: 'string' }, bench_key: { type: 'string' }, comparable: { type: 'boolean' },
    }, ['rank', 'patch_path']),
  },
}, ['read_reason']);
// Warm-start writer output (experience_store.py `write` JSON); "duplicate_impl" counts a reproduction.
const WARMSTART_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, reason: { type: 'string' }, slug: { type: 'string' },
  dir: { type: 'string' }, speedup: { type: 'number' },
  reproduced: { type: 'string' }, reproductions: { type: 'number' },
  // store mode only: the KB-plane write; `replaced` tells a new candidate from a re-measured one.
  remote: obj({
    written: { type: 'boolean' }, reason: { type: 'string' }, canonical_id: { type: 'string' },
    session_id: { type: 'string' }, champion: { type: 'boolean' }, replaced: { type: 'boolean' },
  }),
}, ['written']);

// Prompt helpers: agents read their role and knowledge files; the script passes paths + JSON inputs.
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// Hung-agent guard: an agent call that never returns would block a parallel()/pipeline() barrier
// forever, so agentT resolves it to null after AGENT_TIMEOUT_MS (default 60 min, far above any real
// build or bench; 0 disables; cache keys unchanged, so resume works). A thrown API error is retried
// AGENT_RETRIES times, then resolves null; a timeout is not retried. Downstream code tolerates null;
// after a persistent failure, resume with Workflow({resumeFromRunId}).
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
// GEAK-ABLATION-ARMS-v1 (B6): registered mechanical dispatches (e.g. storage reclaim) run as the next
// agent's first action instead of in an agent of their own, which pays the full bootstrap cost.
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

// LLM accounting (args.llm_stats="false" disables): records each agent call's role/phase/attempt.
// No timestamps (no Date.now() here); scripts/llm_ledger.py takes durations from the transcripts.
const LLM_STATS = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
const LLM_TL = { schema: 'geak.agent_timeline/1', workflow: 'kernel_lane', events: [], nested: [] };
const TL_ROLE_RE = /You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\./;
// Called AT DISPATCH (before the await) so LLM_TL.events is in dispatch order, not completion order.
// Returns the event so the caller flips `ok` once the attempt resolves (null when off).
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
  // Routed dispatches carry their dispatch id into the timeline, so commit_failure.receipt_ref can be
  // matched to this event; non-routed calls pass none and the event is unchanged.
  if (dispatchId != null) e.dispatch_id = dispatchId;
  LLM_TL.events.push(e);
  return e;
}

// Cost ladder: route every agent scope to the lowest sufficient model. Policy:
// e2e_workflow/routing/SKILL.md. Off unless A.routing is truthy; agentT() then never consults it.
//   brain    director:* / tech_lead:*       top lane, fixed
//   helper   the fixed-script scopes below  ROUTE_MODEL_CHEAP, fixed (their schema is the check)
//   decider  route:* (the classifier)       ROUTE_MODEL_DECIDER, fixed, never classified itself
//   worker   every other scope              classified once, then moved up one lane only on a
//                                           deterministic failure
// Scope key = phase + U+0000 + label prefix (text before the first space).
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
  // Fixed-script helpers (schema-forced, low reasoning):
  m['Optimize' + ROUTE_SEP + 'clock'] = ROUTE_MODEL_CHEAP;            // clock ${tag} (effort:'low', CLOCK_SCHEMA)
  m['Optimize' + ROUTE_SEP + 'storage:reclaim'] = ROUTE_MODEL_CHEAP; // storage:reclaim r${round}
  m['WarmStart' + ROUTE_SEP + 'warm_start:resolve'] = ROUTE_MODEL_CHEAP; // warm_start:resolve
  return m;
})();
const ROUTING_ON = (function () {
  // A.routing (forwarded by the parent) is the only enable source: this child lane has no `process`,
  // and test_dispatch.py::test_lane_script_uses_no_native_forbidden_constructs rejects it here.
  const a = String(A.routing != null ? A.routing : '').trim().toLowerCase();
  return a === 'true' || a === '1' || a === 'on';   // anything else (incl. unset) stays OFF — default-OFF
})();
// Thresholds and caps are owned by code, never the decider; each is a per-run arg, so an approved
// route_review.py proposal needs no code change.
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
// Decider backend: 'sonnet' (default) classifies; 'eikos' and 'p1' are separate paths below; any
// other value (e.g. the reserved 'jev') warns and uses Sonnet.
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
// Ladder identity. Coarser than the dispatch label, finer than its prefix: `eng d3:memory` and
// `eng d7:memory` share one ladder ('eng:memory'), `eng d3:compute` gets its own.
function __routeScope(label) {
  const parts = String(label == null ? '' : label).split(' ');
  const rest = parts[1] || '';
  const c = rest.indexOf(':');
  return c >= 0 ? parts[0] + ':' + rest.slice(c + 1) : parts[0];
}
// Static part. A model for brain / helper / decider scopes; undefined for a worker (which __routePick
// classifies). OFF -> undefined everywhere.
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
// Pure: (ladder state, decider answer) -> next state. `confidence` = the decider's estimate that the
// current lane (the smallest, for a new scope) succeeds. First sight: stay on the smallest lane unless
// confidence < gate (an invalid answer stays small). After a failure: move up one lane if confidence
// < gate, the lane failed MAX times or a risk_flag is set; otherwise retry on the same lane.
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
// budget.spent() is the whole session's output-token total, so the cap applies to growth since this
// lane started; other session spend still counts, so the switch can trip early, never late.
const ROUTE_SPENT_AT_START = __routeSessionSpent();
function __routeSpent() {
  const now = __routeSessionSpent();
  return now == null ? null : now - (ROUTE_SPENT_AT_START || 0);
}
// Kill switch on output tokens (budget.spent(); dollars are settled later by the ledger). Once
// tripped: no new round or worker dispatch; brain dispatches still run so report + validation survive.
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
// The whole decision for one dispatch. Returns { model } to override, {} to leave the pinned model,
// or { skip: true } when the kill switch has tripped.
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
  // Snapshot the top-lane quota before the live dispatch consumes it, so the P1 shadow reduces on the
  // same state the live pick saw (else a false `live=Opus / shadow=defer` disagreement).
  const __p1TopUsedPre = __routeTopUsed;
  if (lane === ROUTE_TOP) {
    if (ROUTE_MAX_TOP_DISPATCHES != null && __routeTopUsed >= ROUTE_MAX_TOP_DISPATCHES) { lane = ROUTE_TOP - 1; capped = true; }
    else __routeTopUsed++;
  }
  __routeAudit.push({ event: 'dispatch', scope, label: (o && o.label) || '', phase: (o && o.phase) || '',
                      model: ROUTE_LANES[lane], capped });
  // P1 shadow telemetry only: issue an immutable per-dispatch record keyed by a host dispatch id, with
  // the model actually dispatched (after quota downgrade) and the pre-dispatch quota snapshot, so the
  // outcome binds to this dispatch. Never changes the live pick, the reduction or __routeLadder.
  const __p1Did = `d${++__routeDispatchSeq}`;
  const __p1Rec = { id: __p1Did, label: (o && o.label) || '', scope, seq: __routeDispatchSeq, model: ROUTE_LANES[lane],
                    outcome: null, p1Outcome: null, decider: 'sonnet' };
  __routeDispatchById[__p1Did] = __p1Rec;
  (__routeDispatches[(o && o.label) || ''] = __routeDispatches[(o && o.label) || ''] || []).push(__p1Rec);
  __p1Shadow('sonnet', scope, (o && o.label) || '', ROUTE_LANES[lane], __p1TopUsedPre);   // never changes the pick above
  return { model: ROUTE_LANES[lane], scope, dispatch_id: __p1Did };
}
// Deterministic outcome of one worker-scope cycle; ok=false re-classifies before the next dispatch.
// `evidence` is compact (status words and numbers, never a log). `kind`: work_product,
// non_improvement, invalid_measurement or no_result; Sonnet counts all failures alike, Eikos tells
// them apart. `p1Kind` (optional) re-classifies `kind` for the P1 shadow only.
function __routeOutcome(scope, ok, evidence, kind, source, label, dispatchId, p1Kind) {
  // With P1 live there is no legacy state to keep identical, so P1 gets the corrected kind.
  if (ROUTE_P1_ON) return __p1OutcomeLive(scope, ok, evidence, (p1Kind || kind) || 'work_product', source, label, dispatchId);  // gated; OFF by default
  const st = __routeLadder[scope];
  if (!st) return;
  if (st.decider === 'eikos') return __routeEikosOutcome(st, scope, ok, evidence, kind || 'work_product', source, label, dispatchId, p1Kind);
  // P1 shadow: book the outcome on the model of its own dispatch record (keyed by dispatchId), so
  // overlapping same-scope dispatches each book their own model. The record is validated (scope +
  // label) and deduplicated (a replay must not promote the floor); without an id, fall back to
  // last_dispatch_model, then ROUTE_LANES[st.lane]. Writes only __p1Ladder.
  const __p1st = __p1Ladder[scope];
  const __p1d = dispatchId != null ? __routeDispatchById[dispatchId] : null;
  let __p1skip = false;
  // Validate whenever an id was supplied: an unknown non-null id is rejected (ignored + audited), as in
  // __p1WorkerOutcome and the Python DispatchBook.resolve replica. A null id takes the fallback.
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
  // Report the live decider ('p1', 'eikos' or the Sonnet model); P1's own ladder and outcomes are below.
  return { lanes: ROUTE_LANES, decider: ROUTE_P1_ON ? 'p1' : ROUTE_EIKOS_ON ? 'eikos' : ROUTE_MODEL_DECIDER,
           thresholds: { conf_escalate: ROUTE_CONF_ESCALATE, max_retries_per_lane: ROUTE_MAX_FAILS_PER_LANE,
                         max_top_escalations: ROUTE_MAX_TOP_DISPATCHES, max_output_tokens: ROUTE_MAX_OUTPUT_TOKENS,
                         floors: ROUTE_FLOORS },
           top_dispatches: __routeTopUsed, killed: __routeKilled, spent_output_tokens: __routeSpent(),
           session_output_tokens_at_start: ROUTE_SPENT_AT_START,
           ladder: __routeLadder, audit: __routeAudit,
           config_errors: __routeConfigErrors,   // arguments the lane could not use, and what it used instead
           // P1 measured-signal router: shadow summary even when off; `live` only under route_decider=p1.
           p1: { reducer: P1_REDUCER_VERSION, live: ROUTE_P1_ON, shadow: __p1ShadowSummary(),
                 // P1's own ladder (separate from the legacy one) and every per-dispatch terminal outcome.
                 ladder: __p1Ladder, dispatches: __routeDispatchById,
                 // P1 is measured only for roles with a terminal success/work_product signal; others only add
                 // `no_result` (agentT's null path), which moves nothing.
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
// Eikos decider (route_decider=eikos; contract: research/eikos_ladder_routing_proposal_20261009.md
// rev 2 §7). Eikos never picks a lane: via a transport-only carrier it estimates, per lane, that the
// next attempt meets the task's acceptance event; __routeReduce (pure host code) applies the rules and
// picks. Thresholds, classes and floors are unvalidated experimental settings (the report says so).
// Units: a consultation = one carrier call (no retries); a dispatch = one agentT() call (agentTCore's
// transport retries included); an occurrence = one frozen card (`seq`), reused on replay. Only a
// host-verified work-product failure counts against a lane.
const ROUTE_EIKOS_ON = ROUTE_DECIDER === 'eikos';
// run_e2e.py probes the Eikos carrier once at startup and sets the route_eikos_probe_failed arg when it
// is down (forwarded parent -> child like routing; this lane has no `process`). The eikos:route
// carrier is then skipped and __routeReduce falls back to the static ladder (estimates null).
const ROUTE_EIKOS_PROBE_DOWN = (function () {
  const a = String(A.route_eikos_probe_failed != null ? A.route_eikos_probe_failed : '').trim().toLowerCase();
  return a === '1' || a === 'true' || a === 'on';   // anything else (incl. unset) stays OFF
})();
// P1: deterministic measured-signal host router (no model). Unless route_decider=p1 it only runs in
// shadow (a 'p1_shadow' audit beside the live pick); route_decider=p1 makes __p1Pick live.
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
// Off (default): a scope's highest dispatched lane (effective_rank) stays a floor. On: only the
// failure-earned floor (scope_floor_rank / floor_rank) remains, so a scope can route back down.
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
// A task is what the next attempt is asked to do: its direction (or label), the whole dispatch prompt and
// the requested effort. A change in any of them is a different attempt with its own occurrence, even
// under the same direction ID.
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
// Card size caps, mirrored by eikos_decide.py (MAX_DIRECTION_PROMPT_CHARS, MAX_STATE_CHARS). A direction
// prompt over the cap is condensed by the carrier model (__routeCondense) rather than refused; the
// engineer itself always receives the full text.
const ROUTE_CARD_PROMPT_CHARS = 8000;
const ROUTE_CARD_STATE_CHARS = 12000;
const ROUTE_CONDENSE_TARGET_CHARS = 6000;
// The frozen card for one occurrence. Only engineers with a registered direction are eligible: the
// direction goes in verbatim (or condensed, `fit`, which the card says), the dispatch prompt only as
// digest + length. Other scopes are routed by host policy. A card never holds an answer or outcome.
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
// Condense a direction prompt over the card cap so Eikos can still be asked. Plain agent() (journaled,
// never routed), one attempt under the carrier's wait bound; null leaves the card ineligible.
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
// One consultation: the carrier runs ONE fixed command built here and relays its JSON. It is called with
// agent() directly, so it is never routed. One attempt, its own wait bound, no retries; the lane caps
// consultations. Everything the host acts on is checked against digests and the expected versions.
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
  // Unless persisted and receipted, a resume could redraw estimates for this occurrence: unusable.
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
// Pure: every rule for one occurrence (x = __routeHostInput() + { backend, estimates }). Floors use the
// policy rank, and the scope's effective lane (highest dispatched) is a floor too. Unavailable lanes and
// used quota go first; the list-price proxy orders the rest. A missing estimate is unknown (not zero,
// not a threshold miss) and passes every hard constraint. No I/O, clock or shared state.
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
    // Pre-reservation top-lane quota for the P1 shadow, emitted once per dispatch below (repeats
    // included, deferrals not).
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
      // The failed-probe gate precedes every carrier (eikos:condense here, eikos:route below): a
      // probe-down run makes zero carrier calls. Condense only when Eikos will be asked.
      if (!none && !ROUTE_EIKOS_PROBE_DOWN && reg && reg.prompt.length > ROUTE_CARD_PROMPT_CHARS) {
        const short = await __routeCondense(reg, scope, (o && o.phase) || '');
        if (short) fit = { direction: short, from_chars: reg.prompt.length };
      }
      const c = __routeCard(p, o, scope, cls, occ, fit);
      // Probe down = no answer: the static ladder (estimates null) picks and no carrier is dispatched;
      // recorded as 'probe_down' ('not_asked_no_feasible_lane' when nothing is feasible).
      const res = ROUTE_EIKOS_PROBE_DOWN ? (none ? { status: 'not_asked_no_feasible_lane' } : { status: 'probe_down' })
        : c.ineligible ? { status: 'ineligible' } : none ? { status: 'not_asked_no_feasible_lane' }
        : await __routeEikosConsult(c.card, host);
      // Other scopes may have reserved the top lane while this one waited: reduce on the current inputs.
      const now = __routeHostInput(scope, cls);
      const red = __routeReduce(Object.assign({}, now, { backend: res.status, estimates: res.estimates || null }));
      // shadow only: what the episode-identity policy would choose; never changes `red`.
      const __redShadow = __routeReduce(Object.assign({}, now, { backend: res.status, estimates: res.estimates || null, episode_identity: true }));
      __routeAudit.push({ event: 'shadow_identity', scope, label,
                          live_model: red.model, episode_model: __redShadow.model,
                          differ: red.model !== __redShadow.model,
                          effective_rank: now.effective_rank, lo_live: red.lower_rank, lo_episode: __redShadow.lower_rank });
      const interventions = [];
      if (res.host_changed) interventions.push('decision reused from a different host input; reduced on the current one');
      if (JSON.stringify(now) !== JSON.stringify(host)) interventions.push('host input changed during the consultation; reduced on the current one');
      // The P1 shadow (emitted per dispatch below) uses the pre-reservation quota `red` saw.
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
      // A repeat dispatch must pass the current hard constraints too; if its lane is no longer feasible,
      // re-reduce on its own estimates (never below its decided lane) or defer. Recorded either way.
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
    // One P1 shadow event per actual dispatch, on the dispatched model and pre-reservation quota.
    __p1Shadow('eikos', scope, label, tk.model, __p1TopUsedPre);   // never changes the live pick/dispatch above
    return { model: tk.model, scope, dispatch_id: id };
  });
}
// One dispatch's outcome, bound by the host-issued dispatch id carried through engineer, verify and
// result. No id, an unknown id, another label's or scope's id, or a repeat is recorded and changes
// nothing; a late outcome still counts against its own lane but schedules nothing.
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
// P1 reduces over measured host signals only (class/operator/scope floors, per-lane work_product
// failures); no model call. Only a worker-attributable work_product failure raises a scope's earned
// floor; non_improvement feeds a separate stagnation counter; infra kinds move nothing. It takes the
// cheapest feasible lane at or above the earned floor; effective_rank is reported, never a floor.
// Shadow by default: own state (__p1Ladder) and reducer (__p1Reduce), no legacy state touched;
// __p1Shadow records a 'p1_shadow' audit event per live pick. Live only under route_decider=p1.
const P1_REDUCER_VERSION = 'p1_measured.v1';
const __p1Ladder = {};   // scope -> P1's OWN measured-signal state, kept apart from __routeLadder
// A fresh P1 scope: floor_rank starts at the class floor; only work_product failures raise it.
function __p1NewState(cls) {
  return { scope_class: cls, fails_by_lane: {}, floor_rank: ROUTE_CLASS_FLOOR[cls], effective_rank: -1,
           stagnation: 0, occurrences: 0,
           // model actually dispatched for the last pick (a quota downgrade is not booked as Opus)
           last_dispatch_model: null };
}
// P1's reduction inputs from __p1Ladder. The top-lane quota is read (never consumed) from the live
// counter, or from `topUsedSnapshot`, the count before the live dispatch took its unit.
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
// Pure P1 reducer (x = __p1HostInput()): floor = max(class, operator, earned scope floor); drop excluded
// and unavailable lanes and a used-up top quota (nothing left = defer), then take the cheapest feasible
// lane. Same shape as __routeReduce without estimates. No I/O, clock or shared state.
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
// Update P1's state from an outcome; `liveModel` is the lane that actually ran. Only work_product
// earns the next lane up; non_improvement bumps stagnation; infra moves nothing. Writes only __p1Ladder.
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
// p1_shadow events reach run_result.json only via __routeAudit -> __routeReport() (no fs or clock here).
// Shadow emission: record P1's would-be pick as a 'p1_shadow' audit event (live_pick, p1_pick,
// differ, signals) without changing the live pick; returns the P1 reduction.
function __p1Shadow(decider, scope, label, liveModel, topUsedSnapshot) {
  if (!__p1Ladder[scope]) __p1Ladder[scope] = __p1NewState(__routeScopeClass(scope));
  const st = __p1Ladder[scope];
  const cls = st.scope_class;
  // On an actual dispatch, remember its model so the outcome hook books failures to it.
  if (liveModel != null) st.last_dispatch_model = liveModel;
  // Reduce on the pre-dispatch quota the live pick saw, when supplied.
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
// Live P1 path (route_decider=p1 only): shadow state + __p1Reduce, a host dispatch id, and the same
// kill switch and top-lane quota as the other deciders.
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
// Live P1 outcome handler (route_decider=p1 only): binds by host dispatch id, then __p1Observe.
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
  // One terminal audit event per completed dispatch, successes included.
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
// A non-engineer worker's attributable terminal outcome (e.g. the committer's {committed:false}) goes to
// P1 only: the live handler under route_decider=p1, else P1's shadow state; never the sonnet/eikos
// ladder. The model comes from the dispatch id when present, else the shadow last_dispatch_model.
// agentT already reports `no_result` for a null result; this adds the non-null success/failure.
function __p1WorkerOutcome(scope, ok, evidence, kind, source, label, dispatchId) {
  if (!ROUTING_ON || !scope) return;
  if (ROUTE_P1_ON) { __p1OutcomeLive(scope, ok, evidence, kind || 'work_product', source, label, dispatchId); return; }
  // Shadow path: with a dispatch id, bind to its record (validate scope + label, deduplicate so a
  // replay cannot promote the floor) and book d.model, the model actually dispatched.
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
// The verdict table for a dispatched commit attempt (initial or retry), sent to P1 via
// __p1WorkerOutcome: committed===true = success; explicit committed===false = work_product failure;
// a null receipt or no explicit verdict = no_result (moves nothing). Idempotent (deduplicated by
// dispatch id); routed runs only.
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
  // Thread the issued dispatch id (also in route.dispatch_id; receipt_ref keys on it) into the timeline.
  const r = await agentTCore(p, o, pick.dispatch_id || null);
  // An engineer's cycle is judged by the oracle after verify (see the Optimize round), not here. For
  // every other worker the one deterministic failure visible at this layer is no result at all.
  if (pick.scope && !/^(eng|deep):/.test(pick.scope) && r == null) {
    __routeOutcome(pick.scope, false, 'no result (timeout, API fault or lost StructuredOutput)', 'no_result',
                   (o && o.label) || '', (o && o.label) || '', pick.dispatch_id);
  }
  return r;
}

async function agentTCore(p, o) {
  // The dispatch id is an optional 3rd argument read via `arguments`: test_eikos_route.js slices
  // agentT up to the two-arg `agentTCore(p, o)` declaration. Absent when not routed.
  const dispatchId = arguments.length > 2 ? arguments[2] : null;
  const label = (o && o.label) ? o.label : 'agent';
  if (ABL_B6 && typeof p === 'string') { const pre = ablDrainPrelude(); if (pre) p = pre + p; }
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    const ev = tlAgent(p, o, attempt, dispatchId);   // record AT DISPATCH; ok=false until it resolves (dispatch id threaded for receipt-by-id — Astra re-review #6)
    try {
      if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) {
        const r0 = await agent(p, o); if (ev && r0) ev.ok = true; return r0;
      }
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — resolving null so the round proceeds.`);
          resolve(null);
        }, AGENT_TIMEOUT_MS);
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

// Commit recovery is abort-and-quarantine, so nothing here needs a non-routed guaranteed dispatch.


// Optional wall-clock deadline, enforced in code (prompt text is not a constraint):
//   DEADLINE_EPOCH  unix seconds after which no new optimization round may start
//   NO_STOP_S       while more than this many seconds remain, the TechLead may not stop the run on
//                   its own convergence gates; it is sent back to re-plan
// No clock here (Date.now() would break resume): a low-effort agent reads the time, and an unreadable
// clock returns Infinity (budget-only behaviour) rather than ending a healthy run.
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

// Eikos shadow pilot (eikos_shadow=round_continue; research/eikos_decision_audit_20261001.md §5): at
// the round's continue/stop point, ask local Eikos via one fixed command and record the answer; it
// never controls the branch.
//   - State is frozen before the planning attempt and never holds the TechLead's answer; untracked
//     fields (noise_band) are "unknown". Asked only where the host would permit a stop.
//   - The state travels in one shell-quoted command a carrier runs verbatim (no files or HTTP here);
//     eikos_decide.py persists the first decision per key and returns the state's SHA-256, which the
//     lane checks against its own digest.
//   - One carrier attempt with its own wait bound; a late finish cannot change the recorded decision.
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

// Hex SHA-256 of a string's UTF-8 bytes, in pure JS (the sandbox has no crypto or TextEncoder;
// encodeURIComponent/unescape yield the bytes).
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

// Python json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False) for the decision digest.
// Digested fields are strings, booleans and null only: numbers format differently (1.0 vs 1).
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

// The shadow call at the effective decision point (the TechLead's decision is final). rawFinal /
// hostReason: when the harness changes the plan after the model answered (e.g. the unknown-peaks
// filter empties it), the model's choice is read before that change and the host's reason recorded.
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
    // Relay check on digests computed here: a carrier that edits the state or the decision is caught.
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

// Expert-skills injection: '' when off or for a non-consumer role (both call sites: roleAgent and
// the inline Optimize prompt); when on, an advisory pointer to Read the fragment + skills index.
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

// Appended to every role prompt only for multi-GPU jobs, so single-GPU prompts are unchanged.
function multiGpuBlock(role) {
  if (!GPU_JOB) return '';
  const n = GPU_RESOURCE.gpusPerJob;
  return `\n\n## Multi-GPU job (GPUS_PER_JOB=${n})\n` +
    `Every GPU command of this op needs ${n} GPUs. \`GPU_ID\` is the group spec \`${GPU_JOB}\`: one ` +
    `\`gpu_lock.sh $GPU_ID <cmd>\` holds the whole group for the whole command and sets ` +
    `HIP_VISIBLE_DEVICES to the leased ids, so start every rank inside that one call (e.g. ` +
    `\`torchrun --nproc_per_node=${n} ...\`). Never call gpu_lock.sh from inside a leased command (it is ` +
    `rejected) and never background a lease (\`&\`, nohup). A leased command is killed after ` +
    `GEAK_GPU_RUN_TIMEOUT seconds (default 900); export a larger value for long runs such as a cold ` +
    `JIT build. Leases on the same GPUs serialize, so put the arms you compare into one run.`;
}

function roleAgent(role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role) + multiGpuBlock(role);
}

// PHASE: Setup
phase('Setup');
const setup = await agentT(
  roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK,
    WORKFLOW_DIR, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC, EXPECTED_GFX, EXPECTED_TARGET,
    EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const GFX = String(setup.device_gfx || '').trim().toLowerCase();
const DEVICE_TARGET = String(setup.device_target || '').trim().toLowerCase();
const SETUP_DEVICE_NAME = String(setup.device_name || '').trim();
const SETUP_PHYSICAL_CU_COUNT = Number(setup.physical_cu_count);
if (!GFX) {
  throw new Error('Setup failed: director did not return device_gfx (required; refuse CDNA knowledge without an ISA)');
}
if (!DEVICE_TARGET) {
  throw new Error('Setup failed: director did not return device_target (r9700 | r9600d | unknown)');
}
if (!Number.isFinite(SETUP_PHYSICAL_CU_COUNT) || SETUP_PHYSICAL_CU_COUNT <= 0) {
  throw new Error('Setup failed: director did not return a positive physical_cu_count');
}
if (GFX === 'gfx1200') {
  throw new Error('Detected gfx1200, which is not validated by this workflow; refusing R9700 guidance');
}
if (EXPECTED_GFX && GFX !== EXPECTED_GFX) {
  throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${GFX || 'unknown'}`);
}
if (EXPECTED_TARGET && DEVICE_TARGET !== EXPECTED_TARGET) {
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
const CALIBRATED_PRODUCTS = ['r9700', 'r9600d'];
// peaks.md ISA rows valid only at their measured CU count (8050S/8040S share gfx1151 with the 40-CU
// 8060S); any other part of an isolated family is a hard unknown, as in roofline_tools.resolve_peaks.
const TABULATED_CU = { gfx1151: 40 };
const GPU_FAMILY = gpuFamily(GFX);
const ROOFLINE_STATUS = (CALIBRATED_PRODUCTS.includes(DEVICE_TARGET) && GFX === 'gfx1201')
  ? `calibrated-${DEVICE_TARGET}`
  : /^gfx120/.test(GFX) ? 'unknown-device-not-r9700'
  : TABULATED_CU[GFX] === Number(PHYSICAL_CU_COUNT) ? `tabulated-${GFX}`
  : GPU_FAMILY.isolate ? `unknown-${GFX}-not-tabulated`
  : 'non-rdna';
if (GPU_FAMILY.isolate) {
  // Until the perf/learned stores implement metadata-enforced gens filtering, fail closed:
  // an isolating family (GPU_FAMILIES) gets the dedicated in-tree hardware guide but no CDNA-biased cards.
  USE_EXPERT_SKILLS = false;
  KERNEL_KNOWLEDGE_DIR = '';
  USE_LEARNED_READ = false;
  WARM_START_ON = false;
}
if (!GPU_RESOURCE.explicit && setup.gpu_requirement) {
  applyGpuResource(resolveGpuRequest(A, GPU_LIST, setup.gpu_requirement));
  if (GPU_RESOURCE.warning) log(`GPU requirement WARNING: ${GPU_RESOURCE.warning}`);
  if (GPU_JOB) {
    log(`GPU requirement: ${GPU_RESOURCE.gpusPerJob} GPUs per job (${setup.gpu_requirement.kind}: ` +
      `${setup.gpu_requirement.evidence || 'no evidence'}) -> ${GPU_JOB}`);
  }
}
// Strip the e2e head's `_task` directory suffix so the session id and record use one name (the store
// folds the canonical id itself: experience_store.remote_identity).
// A note the Director appends ("<name> (triton, author mode; ...)") is dropped the same way.
const KERNEL_NAME = String(setup.kernel_name || '').trim().split(/[\s(;]/)[0]
  .replace(/_task$/, '') || setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);

// Enforce a frozen real-online baseline in both modes: without baseline_src/ or meta.baseline_callable
// the unittest would time kernel_src/ against itself (a fake speedup). The script has no FS access,
// so it trusts the director's verdict; missing -> abort/re-extract.
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

// PHASE: Author (mode=author only): write a from-scratch seed that becomes CANONICAL's HEAD; the rest
// of the pipeline runs unchanged on it. The speedup denominator stays the frozen online kernel
// (baseline_src/), never the seed. No correct seed -> abort with a structured result (e2e drops the
// language).
if (MODE === 'author') {
  phase('Author');
  const authored = await agentT(
    roleAgent('author_engineer', 'author', 'Write the simplest correct baseline in the target language.', {
      TARGET_LANGUAGE, OP_SPEC, WORKSPACE: CANONICAL, TASK_DIR: KERNEL_PATH_ORIG,
      GPU_ID: GPU_POOL, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_KNOWLEDGE_DIR,
      // The author engineer also reads the learned index, so the KB switch must reach it.
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

// PHASE: Analyze + Roadmap (TechLead)
phase('Analyze');
const analysis = await agentT(
  roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR, GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...RESUME_INPUT,
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: ANALYZE_SCHEMA });
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

// perf_knowledge pointers from analyze (reference only), passed to the planner + engineers; empty when
// no operator card applies or KERNEL_KNOWLEDGE_DIR is unset.
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || '';
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];

// PHASE: Benchmark setup (Benchmark Engineer)
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

// PHASE: Baseline profile (Profile Engineer)
phase('Profile');
let profileSummary = await agentT(
  roleAgent('profile_engineer', 'baseline', 'Profile the baseline and classify the bottleneck.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: 0,
    COMMANDMENT, GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...RESUME_INPUT,
  }),
  { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
log(`Baseline bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'} (dispatch_count=${profileSummary ? profileSummary.dispatch_count : '?'})`);

// PHASE: Research (DRA, opt-in via args.dra_enabled), after Profile: writes
// EVAL_DIR/deep_search_brief.md for plan_round. Questions fan out with parallel(), each agent behind
// agentT()'s hang guard, so a hung researcher cannot wedge the barrier.
let researchBriefPath = '';   // EVAL_DIR/deep_search_brief.md when the DRA produced one; '' otherwise
if (DRA_ENABLED) {
  phase('Research');
  const RESEARCH_DIR = `${EVAL_DIR}/research`;

  // Stages 0-2: extract facts, generate + rank research questions (one agent).
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

  // Stages 3/4: research each question in parallel (zero-arg thunks, each hang-guarded by agentT()).
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

  // Stages 5/6 (optional): blindspot critique + a second parallel research wave.
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

  // Stage 7: synthesize the ranked-directions portfolio + the compact planner brief.
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

// PHASE: Optimization loop (budget-controlled)
let dispatched = 0;          // counts ONLY optimization-direction engineers (the budget)
let round = 0;
let cumulative = 1.0;        // best verified geomean speedup vs the TRUE baseline
let bestSeen = 0;            // best verified geomean of any candidate, committed or not
let noImprove = 0;
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source} — also set by a warm-start adopt
let roundsCommitted = 0;     // rounds this run actually landed; a warm-start adopt is NOT one of them
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };
// One row per KB-seeded direction, joined with the verifier's result; fed to update_experience and
// returned, so a driver can see whether cited cards ever carry a round.
const citations = [];

// Deep-mode resume: restore cumulative speedup + insight/ledger history from the prior wave (director
// already seeded canonical from STATE_DIR/best).
if (setup.resumed && setup.prior_state) {
  const ps = setup.prior_state;
  if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
  if (Array.isArray(ps.insights)) history.insights = ps.insights;
  if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
  if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
  if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
}

// PHASE: WarmStart. The recorded speedup only ranks; a fresh verify-gate measurement decides adoption.
// gfx comes from the baseline profile's `device` string.
// kbVerdict maps one benched candidate to a kb/attest.py outcome: `inapplicable` is about the pairing
// (files missing here, or won but not committable) and is excluded from retire arithmetic; a patch
// that applied and ran but answered wrong is `failed`.
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
let warm_start = { adopted: false, read_reason: WARM_START_ON ? 'read' : 'disabled', candidates: [] };
let skipLoop = false;
if (WARM_START_ON && !setup.resumed && KB_ROOT_OK) {
  phase('WarmStart');
  if (!GFX) {
    warm_start.read_reason = 'missing_arch';
    log('[kb] warm-start skipped: no gfx detected from the baseline profile device string.');
  } else {
    // The planes differ in root flag and name->page rule (the store keys by canonical id); same JSON out.
    const localResolveCmd = KB_MODE === 'store'
      ? `resolve-remote --plane local --store ${JSON.stringify(KB_STORE_DIR)}${KB_VERSION_FLAG}`
      : `resolve --root ${JSON.stringify(KB_ARTIFACTS_DIR)} --match ${WARM_START_MATCH}`;
    // Sync the local store with the tree (this kernel + arch only) before reading, to pick up the
    // backlog and hand-imported entries; idempotent and non-destructive (cmd_sync_local). `|| true`: a
    // failed sync means a thinner page, not a failed warm start. Its summary goes to a file because
    // its failures (e.g. `skipped.unreadable` on a root-written tree) otherwise look like an empty KB.
    const syncCmd = KB_MODE === 'store' && KB_ARTIFACTS_DIR
      ? `python3 ${JSON.stringify(EXPERIENCE_STORE)} sync-local --root ${JSON.stringify(KB_ARTIFACTS_DIR)} \\
  --store ${JSON.stringify(KB_STORE_DIR)} --kernel-name ${JSON.stringify(KERNEL_NAME)} --gfx ${GFX} \\
  > ${JSON.stringify(EVAL_DIR + '/kb_sync.json')} 2>&1 || true\n`
      : '';
    const commonArgs =
      `--kernel-name ${JSON.stringify(KERNEL_NAME)} --language ${JSON.stringify(TARGET_LANGUAGE)} \\\n` +
      `  --gfx ${GFX} --top-n 3 --min-speedup ${WARM_START_MIN_SPEEDUP} \\\n` +
      `  --refs-dir ${JSON.stringify(EVAL_DIR + '/kb_references')}`;
    // No `--precision` on the read (the write records it): a patch proven at bf16 often helps at fp8,
    // and filtering would turn thin pages into empty ones.
    // Remote first, then the hand-curated local tree, which a thin remote page must not shadow; "no
    // candidates" (not "no error") triggers the second read. This stays in bash rather than
    // cmd_resolve's `read_planes` because the fallback is a different subcommand.
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
      // Say the sync line is not the answer, so the agent neither relays it nor retries its `|| true`.
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
    // The plane that answered, from `read_plane` (kb/plane.py:read_planes), not the mode; attest picks
    // its flags from it. No canonical id = no keyed read answered (`none` in store mode).
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
      // Counts are over the rows fetched; a saturated scan means the page holds more.
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
            profileSummary = await agentT(
              roleAgent('profile_engineer', 'reprofile',
                'Re-profile the adopted warm-start state and classify the new bottleneck.', {
                WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_SINGLE, ROUND: 0,
                COMMANDMENT, PREVIOUS_METRICS: profileSummary,
              }),
              { phase: 'WarmStart', label: 'warm_start:reprofile', schema: PROFILE_SCHEMA }) || profileSummary;
            if (history) history.bottleneck_now = profileSummary ? profileSummary.bottleneck : history.bottleneck_now;
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
      // Record what this box saw, so the next reads a ledger, not just a speedup. Only candidates
      // with an `outcome` were benched (the loop stops after adopting).
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
while (!skipLoop && dispatched < BUDGET && noImprove < MAX_NO_IMPROVE) {
  // --- (0) Hard stop: no new round may start past the deadline (checked before round++; an in-flight
  // round is never interrupted, which would leave a half-verified patch and no report).
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

  // --- (a) Plan the round (TechLead)
  const planInputs = (forbidStop) => ({
    ...(EVO ? { EVOLUTION: EVO.plannerInput() } : {}),
    EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
    BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
    CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
    KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
    GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
    ...KB_INPUTS,
    // DRA brief (reference): plan_round seeds directions[] from it (tech_lead.md). Absent unless the
    // research ran; kept in planInputs so the forced re-plan gets it too.
    ...(researchBriefPath ? { DEEP_SEARCH_BRIEF: researchBriefPath } : {}),
    // Always stated, on or off: the role gates its learned-KB reads on it (test_kb_switch.py).
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
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).',
      planInputs(false)) + evoPlannerRole,
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });
  const eikosFirstChoice = EIKOS_SHADOW_ON ? eikosRawChoice(plan) : null;

  // No early stop: refuse a TechLead stop while the window is materially unspent (the caller's rule is
  // to optimize until the budget is spent), bounded by MAX_FORCED_REPLANS so a role with no direction
  // left eventually stops instead of planning the window away.
  while (DEADLINE_EPOCH && left > NO_STOP_S && forcedReplans < MAX_FORCED_REPLANS &&
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
  if (plan && Array.isArray(plan.directions) && ROOFLINE_STATUS.startsWith('unknown')) {
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

  let directions = plan.directions.slice(0, remaining).map((d, i) => ({
    ...d,
    idx: i,
    id: d.id || `r${round}_d${i}`,
    gpu_id: GPU_MODE === 'pin' && !GPU_JOB ? GPU_LIST[i % GPU_LIST.length] : GPU_POOL,
    out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
  }));
  // A deep_explore direction runs alone in its round (its rewrite cannot merge with specialist
  // patches) and costs DEEP_COST; each specialist direction costs 1.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  if (EIKOS_SHADOW_ON) directions.forEach((d) => eikosSpecialties.add(d.specialty));
  log(`Round ${round}: ${directions.length} direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction
  const results = await pipeline(
    directions,
    async (d) => {
      if (EVO) {
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
      const route = ROUTING_ON ? {} : null;   // receives this dispatch's host-issued id (agentT)
      if (ROUTING_ON) {                       // the Eikos decider's card reads the direction, never the prompt
        const xs = Number(d.expected_speedup);
        __routeTasks[`${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`] = {
          id: String(d.id), title: String(d.title || ''), specialty: d.specialty,
          expected_speedup: Number.isFinite(xs) && xs > 0 ? xs : null,
          focus_files: (d.focus_files || []).map(String), prompt: String(d.prompt || '') };
      }
      return agentT(
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
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        GFX, DEVICE_TARGET, PHYSICAL_CU_COUNT, ROOFLINE_STATUS,
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.` +
      // Built inline, not via roleAgent(), so the injection has to be appended here too.
      expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer') +
      (EVO ? `\n\nAlso Read ${WORKFLOW_DIR}/../evo/roles/engineer.md for the selected-parent and donor contract.` : ''),
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA }, route
    ).then((eng) => ({ d, eng, route }));
    },

    async (prev) => {
      const { d, eng, route } = prev;
      if (EVO && prev.evo_prepare_failed) return { d, eng, ver: null };
      let patch = `${d.out_dir}/best_patch.diff`;
      // Harvest is anchored on measurement, not the return value: an engineer that died or mis-returned
      // may still have left an above-floor best_patch.diff, so every case except a clean below-floor
      // return goes to verify, which reports apply_failed for a missing or empty patch.
      // An invalid/missing source verdict must reach Verify, even after an empty-patch return.
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
      // The verifier carries a host dispatch id too, so its own terminal outcome reaches P1's shadow.
      const vlabel = `verify ${d.id}${recovered ? ' (recovered)' : ''}`;
      const vroute = ROUTING_ON ? {} : null;
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          ENGINEER_WORKSPACE: `${d.out_dir}/workspace`,
          EVAL_DIR, WORKFLOW_DIR, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
        }),
        { phase: 'Verify', label: vlabel, schema: VERIFY_SCHEMA }, vroute
      ).then((ver) => {
        // Feed the verifier's own outcome to P1's shadow only (null is already `no_result` via agentT).
        // A usable measurement is a success even when it finds a bad kernel; no verifier fault is inferred.
        if (ROUTING_ON && ver != null) {
          // Success needs a recognized verdict: `verified` or `correctness_failed` (both usable
          // measurements). `invalid_measurement` is infra and any other status is `no_result`; neither
          // moves anything.
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
  if (EVO) await EVO.recordBatch(clean);
  const verified = clean.filter(r => r.ver && says(r.ver.status, 'verified') &&
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
    // about the lane. A skipped or silent engineer is no result; an invalid measurement is not the
    // patch's fault; a correct patch that did not beat the floor is a non-improvement.
    const kind = __routeSkipped[`${kindTag} ${r.d.id}:${r.d.specialty}`] ? 'no_result'
      : !r.ver ? 'non_improvement'
      : says(r.ver.status, 'invalid_measurement') ? 'invalid_measurement'
      : says(r.ver.status, 'apply_failed') && !r.eng ? 'no_result'
      : (says(r.ver.status, 'apply_failed') || says(r.ver.status, 'correctness_failed') ||
         says(r.ver.correctness, 'fail')) ? 'work_product'
      : 'non_improvement';
    // `!r.ver` is a true non_improvement only after a deliberately skipped verify (`trustBelow`); a
    // verifier that returned nothing is no_result. Only P1's shadow gets this corrected `p1Kind`.
    const trustBelow = r.eng && r.eng.status !== 'failed' && !(primSpeedup(r.eng) > CANDIDATE_FLOOR) &&
      r.eng.status !== 'invalid_measurement' && r.eng.measurement_valid === true;
    const p1Kind = (kind === 'non_improvement' && !r.ver && !trustBelow) ? 'no_result' : kind;
    __routeOutcome(sc, verified.indexOf(r) >= 0, ev, kind, `${r.ver ? 'verify' : 'engineer'} ${r.d.id}`,
                   `${kindTag} ${r.d.id}:${r.d.specialty}`, r.route ? r.route.dispatch_id : null, p1Kind);
  }

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the primary metric (time-weighted when a workload spec is active); the raw
  // unweighted geomean is kept for the report.
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
  const winner = candidates[0] || null;
  const improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  // Is the search advancing (not: did it beat the incumbent)? Round 1 decides on `improved` alone
  // (bestSeen > 0); at the defaults this implies `improved`. A round with no candidate is never progress.
  const madeProgress = !!(winner && bestSeen > 0 && winner.geomean > bestSeen * (1 + PROGRESS_DELTA));
  // Whether the winner was installed this round: besides new search progress, the only thing that
  // resets the stagnation stop (an improved candidate whose commit failed does not).
  let installed = false;

  // --- (e) Commit the winner into the canonical workspace
  if (improved) {
    // No pre-attempt snapshot or restore: a failed or ambiguous commit aborts the run and quarantines
    // the workspace (below), because a restore cannot be proven safe against a still-live committer.
    const commitRoute = {};   // eikos v2 E1 (bug 1): carries the host dispatch id so the commit's outcome binds to P1
    // One const so the one-shot retry below re-sends the identical committer prompt; parity checks
    // extract this literal.
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
    // One terminal outcome per completed commit dispatch, before `cumulative` advances. `let`: a
    // successful retry below may set it for the advance-vs-abort decision; P1 and the `commit_outcome`
    // audit keep the committer's first self-report.
    let committed = !!(commitRes && commitRes.committed);
    __routeAudit.push({ event: 'commit_outcome', round, improved: true,
                        committed,
                        geomean: winner.geomean, source: winner.source });
    // The committer's outcome goes to P1 through the one verdict table (__commitAttemptOutcome).
    __commitAttemptOutcome(commitRes, commitRoute, `commit r${round}`);

    // Commit salvage (routed runs only; routing=false never enters). An explicit {committed:false}
    // receipt is a genuine failure: abort and quarantine below. Anything else short of
    // {committed:true} (a null receipt or no explicit verdict) is ambiguous, and the verified winner is
    // salvaged by one retry, but only when the first dispatch is confirmed not started or terminated:
    // agentTCore cannot cancel a timed-out agent(), so a timed-out writer may still be live, and a
    // second one (its prompt runs `git checkout -- .`) must not start. Otherwise quarantine and retain
    // the verified winner as recoverable.
    const __commitExclusiveOwnership = (receipt, atRoute) =>
      // 'not_started' (no dispatch id issued) | 'terminated' (a receipt: agent() resolved, the writer
      // exited) | 'ambiguous' (dispatch id but no receipt: the writer may still be live).
      (!(atRoute && atRoute.dispatch_id)) ? 'not_started' : (receipt != null ? 'terminated' : 'ambiguous');
    const __retainQuarantine = (stage, atRoute, atReceipt, opts) => {
      // Quarantine + retain: record where the verified winner's patch lives (round_${round}/integrate or
      // engineer_*) so it stays recoverable. The only non-advance outcome besides an explicit
      // {committed:false}; `opts` carries the reason (ambiguous timeout vs no verdict).
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
    // The attempt behind the terminal verdict, so commit_failure reports the last failing attempt (an
    // explicit-false retry reassigns it); the per-attempt audit is separate.
    let termRoute = commitRoute, termRes = commitRes, termScope = `commit r${round}`;
    if (ROUTING_ON && !committed && !(commitRes != null && commitRes.committed === false)) {
      // A second writer needs exclusive ownership: an ambiguous (timed-out) first commit is quarantined.
      if (__commitExclusiveOwnership(commitRes, commitRoute) === 'ambiguous') {
        return __retainQuarantine(`commit r${round}`, commitRoute, commitRes);
      }
      // Retry once with the identical commitPrompt (the first dispatch is confirmed terminated or not
      // started, so this writer is exclusive).
      const retryRoute = {};
      const commitRetry = await agentT(
        commitPrompt,
        { phase: 'Merge', label: `commit r${round} retry`, schema: COMMIT_SCHEMA }, retryRoute);
      // Verdict table: true = success; explicit false = terminal; null / no verdict = ambiguous.
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
        // An explicit {committed:false} retry is terminal: record this attempt as the provenance and
        // fall through to abort-and-quarantine (committed stays false).
        termRoute = retryRoute; termRes = commitRetry; termScope = `commit r${round} retry`;
        log(`Round ${round}: the commit RETRY returned an EXPLICIT {committed:false} — a genuine failure; ` +
            `TERMINAL (host-side salvage removed in R9). ABORT-AND-QUARANTINE below.`);
      } else if (__commitExclusiveOwnership(commitRetry, retryRoute) === 'ambiguous') {
        // The retry timed out (no receipt): its writer may be live, so no further writer.
        return __retainQuarantine(`commit r${round} retry`, retryRoute, commitRetry);
      } else {
        // The retry ended without a committed verdict: still unconfirmed. Nothing is installed host-side
        // from agent-reported evidence; quarantine and retain the verified winner as recoverable.
        return __retainQuarantine(`commit r${round} retry`, retryRoute, commitRetry,
          { reason: 'commit_unconfirmed_no_verdict',
            why: 'the retry resolved WITHOUT a committed verdict (still-unconfirmed); host-side salvage removed in R9',
            note: 'verified winner RETAINED for recovery — commit UNCONFIRMED (retry returned no committed verdict; host-side salvage removed): no second writer, no host-install' });
      }
    }
    // With routing on (any decider), a failed commit installed nothing, so the incumbent must not
    // advance on it; routing=false keeps the legacy unconditional advance (`installed` = `improved`).
    if (!ROUTING_ON || committed) {
      cumulative = winner.geomean;
      bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
      finalWinner = winner;
      roundsCommitted += 1;
      installed = true;        // the winner is now the installed incumbent artifact

      // --- (f) Re-profile the new best
      profileSummary = await agentT(
        roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
          WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL, ROUND: round,
          COMMANDMENT, PREVIOUS_METRICS: profileSummary,
        }),
        { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
    } else {
      // Abort and quarantine (routed runs only; routing=false advanced above). The commit failed or is
      // ambiguous: an explicit {committed:false}, a null/unverified return, an unconfirmed timeout, a
      // failed correctness recheck or index.lock contention. The workspace is left exactly as the
      // attempt left it (no reset, clean or snapshot), flagged for post-mortem, and the run stops with
      // a terminal failed result, returning before the progress accounting below, so nothing advances.
      // The `commit_failed` audit carries the committer's receipt and the attempt identity.
      const ambiguous = termRes == null;   // the TERMINAL attempt's verdict (R9 Finding 3 residual: not commitRes)
      __routeAudit.push({ event: 'commit_failed', round, improved: true, committed: false,
                          ambiguous, quarantined: true, reason: 'commit_failed',
                          // the attempt whose verdict ended the run (e.g. d2), not the initial dispatch
                          dispatch_id: (termRoute && termRoute.dispatch_id) || null,
                          receipt: termRes || null, workspace: CANONICAL,
                          // `receipt` is the typed return; receipt_ref links the full transcript by
                          // dispatch id in the persisted timeline/ledger, even when receipt is null.
                          receipt_ref: { dispatch_id: (termRoute && termRoute.dispatch_id) || null,
                                         scope: termScope,
                                         timeline: `${EVAL_DIR}/reports/trace/agent_timeline.json`,
                                         ledger: `${EVAL_DIR}/reports/trace/llm_calls.jsonl` },
                          // the initial attempt; differs only when a d1 no-verdict preceded a d2 explicit-false
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
        // Carry the timeline so the dispatcher ledgers and reports an aborted run's spend too
        // (kernel_workflow.js only persists it and runs geak_report.py when r.eval_dir && r.llm_timeline).
        llm_timeline: LLM_STATS ? { ...LLM_TL, instance: EVAL_DIR } : undefined,
        eval_dir: EVAL_DIR,
        kernel_name: KERNEL_NAME,
        // Failure/validation metadata so the scorer + report classify this as a TERMINAL FAILED run (not
        // 'unknown') while its spend is still counted; N stays not-completed / not-accepted downstream.
        validation_status: 'failed',
        // A routed commit could not be confirmed installed; the workspace is quarantined as-is.
        status: 'failed',
        stopped_by: 'commit_failed',
        commit_failure: {
          reason: 'commit_failed', quarantined: true, workspace: CANONICAL,
          round, source: winner.source, geomean: winner.geomean,
          committed: false, ambiguous,
          // the attempt whose verdict ended the run (e.g. d2), not the initial dispatch
          dispatch_id: (termRoute && termRoute.dispatch_id) || null,
          receipt: termRes || null,             // the ACTUAL failing attempt's typed receipt/transcript
          // link to the full transcript by dispatch id (the timeline/ledger geak_report.py renders)
          receipt_ref: {
            dispatch_id: (termRoute && termRoute.dispatch_id) || null,
            scope: termScope,
            timeline: `${EVAL_DIR}/reports/trace/agent_timeline.json`,
            ledger: `${EVAL_DIR}/reports/trace/llm_calls.jsonl`,
          },
          // the initial attempt; differs only when a d1 no-verdict preceded a d2 explicit-false
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
  // Reset the stagnation stop only on new search progress or a real install, so re-finding an
  // uninstallable candidate still stagnates. With routing off `installed === improved` (legacy rule).
  if (madeProgress || installed) { noImprove = 0; } else { noImprove++; }
  if (EIKOS_SHADOW_ON) {
    // Measured facts only. The commit agent's own return is not captured by the lane, so a commit is
    // recorded as "not_captured", never inferred from the tracked incumbent value.
    const o = { verified_candidates: verified.length, winner_speedup: winner ? winner.geomean : null,
                improved: improved, made_progress: madeProgress, installed: installed, commit_reported: 'not_captured',
                tracked_incumbent_after: cumulative };
    if (eikosRec) eikosRec.outcome = o;
    eikosLastOutcome = o;
  }

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
  // Enforce the round's KB budget. Stripping a citation cannot un-influence a planned direction (only
  // the prompt can); it makes overruns observable: `kbCapBound` counts the rounds where it bound.
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

  // Join the planner's declared citations with the verifier's blind measurement. The standing test is
  // `became_winner`, not `verified > 1.0`: against the frozen baseline every non-regressing direction
  // clears 1.0 once the kernel is well past it.
  for (const r of clean) {
    for (const cardRef of (r.d.learned_refs || [])) {
      citations.push({
        card: cardRef, round, direction: r.d.id, specialty: r.d.specialty,
        // null, not 0, when the verifier produced nothing: `kb.py` scores <=1.0 as a real loss.
        cited_then_verified: r.ver && Number.isFinite(r.ver.verified_geomean)
          ? r.ver.verified_geomean : null,
        // Both: `winner` is only this round's best, `improved` means it beat the incumbent by
        // MIN_IMPROVE; a round winner below the committed best advanced nothing.
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

  // Reclaim superseded heavy copies after each round (--force-heavy under pressure) and continue:
  // disk pressure never aborts the loop. An agent runs the bash; STORAGE_NOTE mirrors it to the next
  // round's tech_lead.
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

// PHASE: Final report (TechLead)
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA });

// PHASE: Director validation + arbitration
phase('Validate');
const validation = await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    APPLY_TO_ORIGINAL, COMMANDMENT, FROZEN_ORACLE: FROZEN_ORACLE ? 'true' : 'false',
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

// PHASE: UpdateExperience: distill one reusable card into knowledge/learned/ (README.md there), once
// at the end, because only the final director-verified speedup may be cited (hence `validation` is
// required). Add-only: a skipped or failing step leaves the run unchanged.
let learned_card = null;
// Never a card, not a failure: a run on a shared GPU (it measured contention) or on a held-out kernel
// (the instrument for whether the KB helps). Defaults are permissive; the caller declares a busy box.
const BOX_QUIET = String(A.box_quiet != null ? A.box_quiet : 'true') === 'true';
const HELD_OUT = String(A.held_out != null ? A.held_out : 'false') === 'true';
const kbGate = !BOX_QUIET ? 'the box was contended — the number measured neighbours, not the kernel'
  : HELD_OUT ? 'this kernel is in the held-out split and is the measuring instrument'
  : '';
if (kbGate) log(`[kb] not distilling: ${kbGate}.`);
// Only an `accepted` validation, not any validation object: `flagged` means the director doubts the
// number (install no-op, correctness fail, contended box).
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
          // Context only: what the plan cited and the verifier measured; `kb.py drain` applies the
          // counters from the ledger filed below.
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

// File the citation ledger outside the curation gate: it is the loss half of the loop, and gating it
// on a win would keep losses out of the counters. Writing a new card still needs a verified win;
// held_out runs neither write nor vote. `kb.py cite` builds the JSON (one-line agent, no filesystem
// here), so the schema cannot drift in a model-written proposal.
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

// Write this run's outcome back (the producer half of the loop): one call files the kb_artifacts/
// tree, the on-disk store and, when credentialed, the service. The script gates itself and never
// fails the run; `finalPrimary > 1.0` only saves an agent on a run that cannot pass.
let kb_written = null;
if (KB_WRITE_OK && GFX && Number.isFinite(finalPrimary) && finalPrimary > 1.0) {
  const kernelClass = (analysis && analysis.kernel_type) || 'unknown';
  const finalPatch = report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`;
  const reportPath = report && report.report_path ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`;
  // Curation metadata the store cannot infer from a diff. `direction` (the idea) comes only from a
  // warm start; round titles are free-form, so an entry stays unlabeled until curation.
  const winnerDirection = kebab(warm_start.direction || '');
  const metricKind = HAS_WORKLOAD ? 'time_weighted' : 'geomean';
  // Commas separate the list, so a case name may not contain one.
  const caseNames = (bestPerCase || []).map(c => c && String(c.name || '').replace(/,/g, ';'))
    .filter(Boolean).join(',');
  // write-remote --plane both writes the tree, then the local store, then the service, so planes
  // cannot drift; without credentials `_open_plane` degrades to the local planes and reports
  // `remote_unavailable` (no credential branch needed here).
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
      // A new patch appends a session under the key; the same patch remeasured replaces its own.
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

// finalPrimary is vs the pristine baseline; after a warm-start adopt, split out the delta above it so
// a KB gain is not reported as this run's work. With no round committed after the adopt, the ratio is
// bench noise on the same code, so report 1.0.
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
  // What kind of number `final_speedup` is, so rankings do not read a host_bound/unprimed float as a
  // clean device-time win ('not_applicable' = own-baseline ratio, no frozen-oracle receipt).
  timing_basis: validation ? validation.timing_basis || 'unknown' : 'unknown',
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
  // null when the step was off, or the run earned nothing worth a card.
  // Which condition ended the loop, recorded here (not inferred later), so a run that used its budget
  // is distinguishable from one that gave up.
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
  // Cost ladder: lanes, thresholds, every decision/dispatch/failure; absent when routing is off.
  routing: ROUTING_ON ? __routeReport() : undefined,
  forced_replans: forcedReplans,
  // What the plan cited and whether it carried its round; always returned (empty is a finding too).
  learned_citations: citations,
  // Rounds where the planner overran the KB budget (binding every round = the KB took over planning).
  kb_cap_bound_rounds: kbCapBound,
  // Monoculture canary: distinct (specialty, focus_files) directions explored vs issued. A caging KB
  // lowers this without raising the speedup; KB-less runs report it too, as the baseline.
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

// SHA-256 of a string's UTF-8 bytes (FIPS 180-4). The workflow sandbox has no crypto module; record-batch
// passes this digest so archive.py can refuse report JSON that changed while an agent retyped the command.
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
