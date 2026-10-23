export const meta = {
  name: 'kernel-mega',
  description: 'mode=mega worker of the kernel-workflow dispatcher: optimizes a multi-rank operator (e.g. an EP MoE layer) as a portfolio of independent whole-kernel candidate lanes, each authored, verified on-card (correctness, CUDA-graph replay, paired A/B against the frozen baseline) and ranked; the fastest fully verified candidate is delivered.',
  whenToUse: 'Prefer the kernel-workflow dispatcher with args.mode=mega; launch this script directly only to see per-phase progress. Pass args.kernel_path (required), args.workflow_dir (required), args.mode=mega, args.gpu_ids, args.gpus_per_job, args.budget, args.task.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies; optional analysis_engineer validates Step-2 evidence' },
    { title: 'Optimize', detail: 'budget loop: normal modes advance one canonical tree; mega advances independent whole-kernel candidate lanes' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline' },
  ],
};

// Args/defaults; agents perform filesystem work through caller-supplied paths.
const A = args || {};
// Resume-safe deterministic modeled time. Charge allocated phase slices rather
// than wall time: replay/cached agents must follow the same budget path.
const WORKFLOW_STARTED_MS = 0;
let MEGA_CLOCK_MS = 0;
const MEGA_PREP_MODEL_MS = 90 * 1000;
function megaNowMs() { return WORKFLOW_STARTED_MS + MEGA_CLOCK_MS; }
function megaAdvanceMs(ms) { MEGA_CLOCK_MS += Math.max(0, Number(ms) || 0); }
function megaChargeMs(allocatedMs) { megaAdvanceMs(allocatedMs); }
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/model directory)');

// Caller-supplied directory holding roles, knowledge, tools and scripts.
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_mega.js, roles/mega/, knowledge/, scripts/, tools/ (i.e. the dirname of this script).');
}
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

const KERNEL_PATH_ORIG = A.kernel_path;
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// Positive control calibrates detection; without it 1.000x is unfalsifiable. Null calibrates noise.
const POSITIVE_CONTROL = (A.positive_control && typeof A.positive_control === 'object')
  ? A.positive_control : null;
const PC_ABORT = POSITIVE_CONTROL ? (A.positive_control.abort_on_fail !== false) : false;
const GPU_MIN_FREE_GIB = Number(A.gpu_min_free_gib || 0);
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
})();
// Budget cost of ONE `deep_explore` direction: broad rewrite authority + its own multi-iteration
// measure→profile→rewrite loop, so it is charged >1 (default 2) and always runs in a DEDICATED round.
const DEEP_COST = (() => {
  const v = parseInt(A.deep_cost != null ? A.deep_cost : 2, 10);
  return Number.isFinite(v) && v >= 1 ? v : 2;
})();
// working_kernel preserves one runnable candidate across rounds; speedup keeps the normal score gate.
const OBJECTIVE = (() => {
  const v = String(A.objective || 'speedup').trim();
  if (v !== 'speedup' && v !== 'working_kernel') {
    throw new Error(`args.objective must be 'speedup' or 'working_kernel', got '${v}'`);
  }
  return v;
})();
const WORKING_KERNEL = OBJECTIVE === 'working_kernel';
function resolveGpuRequest(A) {
  const taskResource = (A.op_spec && A.op_spec.resource) || {};
  const parseIds = (name, value) => {
    const raw = String(value);
    const parts = raw.split(',').map(s => s.trim()).filter(Boolean);
    if (!parts.length) throw new Error(`${name} must contain at least one GPU id`);
    if (parts.some(id => !/^\d+$/.test(id))) throw new Error(`${name} contains a non-numeric GPU id`);
    const ids = parts.map(id => String(Number(id)));
    if (new Set(ids).size !== ids.length) throw new Error(`${name} contains duplicate GPU ids`);
    return ids;
  };
  const gpuList = parseIds('gpu_ids', A.gpu_ids != null ? A.gpu_ids : '0');
  const directFixed = String(A.job_gpu_ids || '').trim();
  const fixedValue = directFixed || taskResource.job_gpu_ids;
  const fixedIds = String(fixedValue || '').trim()
    ? parseIds('job_gpu_ids', fixedValue)
    : [];
  const inferredCount = fixedIds.length || 1;
  const countValue = A.gpus_per_job != null
    ? A.gpus_per_job
    : (taskResource.gpus_per_job != null ? taskResource.gpus_per_job : inferredCount);
  const countText = String(countValue).trim();
  if (!/^[1-9]\d*$/.test(countText)) {
    throw new Error('gpus_per_job must be a positive integer');
  }
  const gpusPerJob = Number(countText);
  if (!Number.isSafeInteger(gpusPerJob)) throw new Error('gpus_per_job is too large');
  if (fixedIds.length && fixedIds.length !== gpusPerJob) {
    throw new Error(`gpus_per_job=${gpusPerJob} does not match job_gpu_ids size ${fixedIds.length}`);
  }
  if (!fixedIds.length && gpusPerJob > gpuList.length) {
    throw new Error(`gpus_per_job=${gpusPerJob} exceeds gpu_ids pool size ${gpuList.length}`);
  }
  const pool = new Set(gpuList);
  if (fixedIds.some(id => !pool.has(id))) {
    throw new Error('job_gpu_ids must be a subset of gpu_ids');
  }
  const sharedSpec = fixedIds.length
    ? `group:${fixedIds.join(',')}`
    : (gpusPerJob > 1 ? `pool:${gpusPerJob}:${gpuList.join(',')}` : '');
  return {
    gpuList,
    gpusPerJob,
    fixedIds,
    sharedSpec,
    specForIndex: (index) => sharedSpec || gpuList[index % gpuList.length],
  };
}
const GPU_RESOURCE = resolveGpuRequest(A);
const GPU_LIST = GPU_RESOURCE.gpuList;
const GPU_IDS = GPU_LIST.join(',');
const TASK = A.task || '';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();

// mode=mega keeps independent whole-kernel candidate lanes and selects the fastest fully verified
// speedup. The kernel-workflow dispatcher sends optimize/author runs to kernel_lane.js instead.
const MODE = String(A.mode != null ? A.mode : 'mega').trim().toLowerCase() || 'mega';
if (MODE !== 'mega') {
  throw new Error(`kernel_mega.js runs mode=mega only, got '${MODE}'; use kernel_workflow.js for other modes`);
}
const MEGA_DEFAULT_SPECIALTY = String(A.mega_default_specialty ||
  (GPU_RESOURCE.gpusPerJob > 1 ? 'distributed' : 'compute'));
const TARGET_LANGUAGE = String(A.target_language != null ? A.target_language : 'triton').trim() || 'triton';
const OP_SPEC = A.op_spec || {};
// Mega keeps independent workflow-authored whole-kernel candidates. A matched Expert Skill supplies
// knowledge and an optional contract; external reference trees are never candidate bases.
const MEGA_PROFILE = String(A.mega_profile || 'production').trim();
if (!['production', 'audit'].includes(MEGA_PROFILE)) {
  throw new Error(`args.mega_profile must be 'production' or 'audit', got '${MEGA_PROFILE}'`);
}
const MEGA_PRODUCTION = MEGA_PROFILE === 'production';
const MEGA_STOP_ON_DELIVERABLE = A.mega_stop_on_deliverable != null
  ? String(A.mega_stop_on_deliverable) === 'true' : MEGA_PRODUCTION;
const MEGA_MIN_SEARCH_ATTEMPTS = Math.max(0, Number(A.mega_min_search_attempts != null
  ? A.mega_min_search_attempts : 2));
const MEGA_CANDIDATE_TIMEOUT_S = Math.max(300, Number(A.mega_candidate_timeout_s ||
  3600));
const MEGA_STRUCTURAL_ONLY = MODE === 'mega' &&
  String(A.mega_structural_only != null ? A.mega_structural_only : 'false') === 'true';
const MEGA_ROUTE_ONLY = MODE === 'mega' &&
  String(A.mega_route_only != null ? A.mega_route_only : 'false') === 'true';
const MEGA_FINAL_TIMEOUT_S = Math.max(600, Number(A.mega_final_timeout_s ||
  (MEGA_PRODUCTION ? 7200 : 3600)));
const MEGA_TIME_BUDGET_S = Math.max(1800, Number(A.mega_time_budget_s ||
  28800));
const MEGA_FINAL_RESERVE_S = Math.max(600, Math.min(MEGA_TIME_BUDGET_S / 2,
  Number(A.mega_final_reserve_s || (MEGA_PRODUCTION ? 7200 : 3600))));
const MEGA_CLOSEOUT_RESERVE_S = Math.max(300, Number(A.mega_closeout_reserve_s || 900));
const EXPERT_SKILL_ID = String(A.expert_skill_id || '');
if (EXPERT_SKILL_ID && !/^[a-z0-9][a-z0-9_-]*$/.test(EXPERT_SKILL_ID)) {
  throw new Error('expert_skill_id must be a safe slug');
}
const EXPERT_SKILL_REVISION = String(A.expert_skill_revision || '');
const EXPERT_SKILL_BUNDLE_SHA256 = String(A.expert_skill_bundle_sha256 || '');
const EXPERT_SKILL_PLANNER_EXTENSION_SHA256 =
  String(A.expert_skill_planner_extension_sha256 || '');
const EXPERT_SKILL_CONTRACT_SHA256 = String(A.expert_skill_contract_sha256 || '');
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
const REQUIRE_GRAPH_CAPTURE = A.require_graph_capture != null
  ? String(A.require_graph_capture) === 'true'
  : !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
const DIRECT_GRAPH_ACCURACY = String(A.direct_graph_accuracy || 'false') === 'true';
// Workload alignment (optional): benchmark exactly the workload's (shape, dtype) cases, weight each
// by its time share, and score the time-weighted ratio-of-sums. workload_spec_path is a workload-v1
// json; op_spec.workload (inline) takes precedence. Unset = unweighted. Correctness is unaffected.
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const WORKLOAD_SPEC = (OP_SPEC && OP_SPEC.workload) || A.workload || null;
const HAS_WORKLOAD = !!(WORKLOAD_SPEC_PATH ||
  (Array.isArray(WORKLOAD_SPEC) && WORKLOAD_SPEC.length) ||
  (WORKLOAD_SPEC && Array.isArray(WORKLOAD_SPEC.kernels) && WORKLOAD_SPEC.kernels.length));
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
// diagnostics. changed_kernel needs a same-timeline changed_us/replaced_sum_us measurement; legacy
// keeps the old global score.
const PROMOTION_METRIC = String(A.promotion_metric || 'legacy').trim();
if (!['legacy', 'operator_e2e', 'changed_kernel'].includes(PROMOTION_METRIC)) {
  throw new Error(`args.promotion_metric must be 'legacy', 'operator_e2e' or 'changed_kernel', got ${PROMOTION_METRIC}`);
}
// Strict proof runs must state the metric; inheriting historical mixed behavior is not auditable.
if (String(A.strict_autonomy || 'false') === 'true' && PROMOTION_METRIC === 'legacy') {
  throw new Error('args.strict_autonomy requires an explicit promotion_metric');
}
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
const KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
const EXPERT_SKILL_DIR = EXPERT_SKILL_ID
  ? `${EXPERT_SKILLS_DIR}/skills/${EXPERT_SKILL_ID}` : '';
const PINNED_MEGA_SKILL = MODE === 'mega' && USE_EXPERT_SKILLS && !!EXPERT_SKILL_DIR;
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
const EXPERT_SKILL_VALIDATION_STATUS = String(
  A.expert_skill_validation_status || ''
);
const EXPERT_SKILL_USAGE = String(A.expert_skill_usage || '');
const REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY = USE_EXPERT_SKILLS &&
  (PINNED_MEGA_SKILL ||
   String(A.require_expert_skill_bundle_identity || 'false') === 'true');
const validSha256 = (value) => /^[a-f0-9]{64}$/.test(String(value || ''));
if (REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY &&
    (!EXPERT_SKILL_ID ||
     !EXPERT_SKILL_PLANNER_EXTENSION_FILE || !EXPERT_SKILL_CONTRACT_FILE ||
     !validSha256(EXPERT_SKILL_BUNDLE_SHA256) ||
     !validSha256(EXPERT_SKILL_PLANNER_EXTENSION_SHA256) ||
     !validSha256(EXPERT_SKILL_CONTRACT_SHA256))) {
  throw new Error(
    'require_expert_skill_bundle_identity needs skill id/revision, Planner Extension/contract ' +
    'paths, and canonical bundle/extension/contract SHA-256 values'
  );
}
const EXPERT_SKILL_CONTRACT_TOOL = PINNED_MEGA_SKILL
  ? `${WORKFLOW_DIR}/tools/expert_skill_contract.py`
  : String(A.expert_skill_contract_tool || `${WORKFLOW_DIR}/tools/expert_skill_contract.py`);
const EXPERT_SKILL_REFERENCE_PATH = String(A.expert_skill_reference_path || '').replace(/\/+$/, '');
const CHECK_EXPERT_SKILL_CONTRACT = MODE === 'mega' && USE_EXPERT_SKILLS &&
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
// Structural check + on-card Verify are reserved BEFORE the Author is sized; a leftover-based
// Verify budget was always <300s after a full Author turn, so no turn ever reached the card.
const MEGA_VERIFY_RESERVE_S = Math.max(300, Number(A.mega_verify_reserve_s || 1200));
const MEGA_POST_AUTHOR_RESERVE_S = MEGA_STRUCTURAL_ONLY ? 0
  : (CHECK_EXPERT_SKILL_CONTRACT ? 600 : 0) + MEGA_VERIFY_RESERVE_S;
const MEGA_STALL_LIMIT = Math.max(0, Number(A.mega_stall_limit != null ? A.mega_stall_limit : 3));
const MEASURE_FIRST_TEXT = (n) => MEGA_STALL_LIMIT && n >= MEGA_STALL_LIMIT
  ? `MEASURE_FIRST: ${n} rounds produced no new runtime evidence; before any source edit run a ` +
    'bounded on-card smoke of the current HEAD and report its concrete result. ' : '';
if (MODE === 'mega' && MEGA_PRODUCTION &&
    MEGA_CANDIDATE_TIMEOUT_S - 150 - MEGA_POST_AUTHOR_RESERVE_S < 900) {
  throw new Error(`mega_candidate_timeout_s=${MEGA_CANDIDATE_TIMEOUT_S} cannot fit a 900s Author ` +
    `plus the ${MEGA_POST_AUTHOR_RESERVE_S}s structural+on-card Verify reserve`);
}
if (MEGA_STRUCTURAL_ONLY && !CHECK_EXPERT_SKILL_CONTRACT) {
  throw new Error('mega_structural_only requires an enabled Expert Skill contract');
}
if (USE_EXPERT_SKILLS && EXPERT_SKILL_VALIDATION_STATUS &&
    !['validated', 'experimental'].includes(EXPERT_SKILL_VALIDATION_STATUS)) {
  throw new Error('only validated or explicitly pinned experimental Expert Skills may run');
}
if (USE_EXPERT_SKILLS && EXPERT_SKILL_ID && !EXPERT_SKILL_VALIDATION_STATUS) {
  throw new Error('an explicitly pinned Expert Skill requires validation status');
}
if (USE_EXPERT_SKILLS && PINNED_MEGA_SKILL) {
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
    (MODE === 'mega' && !USE_EXPERT_SKILLS ? `${WORKFLOW_DIR}/tools/generic_graph_contract.py` : ''));
const FAST_TEST_KEY_TOOL = `${WORKFLOW_DIR}/tools/fast_test_key.py`;
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
const EXPERT_SKILL_ROLES = new Set([
  'tech_lead', 'author_engineer', 'engineer', 'deep_engineer', 'mega_search_lead',
]);

const CAPABILITY_EVAL = String(A.capability_eval ?? 'false') === 'true';
const STRICT_AUTONOMY = String(A.strict_autonomy != null ? A.strict_autonomy : 'false') === 'true';
if (STRICT_AUTONOMY && !CAPABILITY_EVAL) {
  throw new Error('args.strict_autonomy requires capability_eval=true');
}
if (STRICT_AUTONOMY && TARGET_GUARDS.length === 0) {
  throw new Error('args.strict_autonomy requires at least one explicit target_guards entry');
}
const positiveIntArg = (name, value, fallback) => {
  const n = Number(value == null ? fallback : value);
  if (Number.isSafeInteger(n) && n > 0) return n;
  if (STRICT_AUTONOMY) throw new Error(`args.${name} must be a positive integer`);
  return fallback;
};
const REQUIRED_REPLAYS = positiveIntArg('required_replays', A.required_replays, 1000);
const REQUIRED_PAIRS = positiveIntArg('required_pairs', A.required_pairs, 5);
const REQUIRED_PAIRS_BY_GUARD = (A.required_pairs_by_guard &&
  typeof A.required_pairs_by_guard === 'object') ? A.required_pairs_by_guard : {};
if (STRICT_AUTONOMY) {
  for (const [guard, count] of Object.entries(REQUIRED_PAIRS_BY_GUARD)) {
    if (!(Number.isSafeInteger(Number(count)) && Number(count) > 0)) {
      throw new Error(`args.required_pairs_by_guard[${guard}] must be a positive integer`);
    }
  }
}
const RF = Object(A.regression_floor_by_guard);
const RFI = { ...(A.regression_floor_by_guard ? { REGRESSION_FLOORS: RF } : {}),
  ...(A.mega_min_speedup ? { MIN_TARGET_SPEEDUP: +A.mega_min_speedup } : {}) };
const REQUIRE_OVERLAP = String(A.require_overlap != null ? A.require_overlap : 'false') === 'true';
const REQUIRE_ATTRIBUTION = String(A.require_attribution != null ? A.require_attribution : 'false') === 'true';
const REQUIRE_ARTIFACT_DISTINCT = String(
  A.require_artifact_distinct != null ? A.require_artifact_distinct : 'false') === 'true';
// Require an activated switched path to have actually launched on hardware.
const REQUIRE_HW_ACTIVATION = String(
  A.require_hardware_activation != null ? A.require_hardware_activation : 'false') === 'true';
const ACCURACY_METRIC = String(A.accuracy_metric || 'relL2');
const ACCURACY_THRESHOLD = Number(A.accuracy_threshold != null ? A.accuracy_threshold : 0.10);
if (STRICT_AUTONOMY && (!(ACCURACY_THRESHOLD > 0) || !Number.isFinite(ACCURACY_THRESHOLD))) {
  throw new Error('args.accuracy_threshold must be a positive finite number');
}
const launchTargetArg = Number(A.launch_target != null ? A.launch_target : 2);
if (STRICT_AUTONOMY && !(Number.isSafeInteger(launchTargetArg) && launchTargetArg > 0)) {
  throw new Error('args.launch_target must be a positive integer');
}
const LAUNCH_TARGET = Number.isSafeInteger(launchTargetArg) && launchTargetArg > 0
  ? launchTargetArg : 2;
const ALLOW_PARTIAL_FUSION =
  String(A.allow_partial_fusion != null ? A.allow_partial_fusion : 'true') === 'true';
if (PINNED_MEGA_SKILL &&
    EXPERT_SKILL_USAGE === 'candidate_validation') {
  const missing = [
    [!MEGA_STRUCTURAL_ONLY, 'mega_structural_only=false'],
    [REQUIRE_GRAPH_CAPTURE, 'require_graph_capture'],
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
const CONTAINMENT_PREFLIGHT = (A.containment_preflight &&
  typeof A.containment_preflight === 'object') ? A.containment_preflight : {};
const REFERENCE_LEAK_MARKER_FILE = String(A.reference_leak_marker_file || '');
if (STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean !== true) {
  throw new Error(
    'args.strict_autonomy requires a clean containment_preflight supplied by the trusted launcher');
}
// Trees whose contents would constitute an imported answer. Given to VERIFY (to compare against) and
// deliberately NOT to engineers — handing them the list would be handing them the location.
const KNOWN_REFERENCE_PATHS = (Array.isArray(A.known_reference_paths) ? A.known_reference_paths
  : String(A.known_reference_paths || '').split(',')).map(s => String(s).trim()).filter(Boolean);
const KNOWN_REFERENCE_HASHES = Array.isArray(A.known_reference_hashes)
  ? A.known_reference_hashes.filter((r) => r && r.path_sha256 &&
    (r.sha256 || r.normalized_sha256)) : [];
if (STRICT_AUTONOMY && KNOWN_REFERENCE_PATHS.length) {
  throw new Error(
    'args.strict_autonomy refuses known_reference_paths in workflow args; supply ' +
    'known_reference_hashes generated out of band so no agent-readable address is published.');
}
if (CAPABILITY_EVAL) {
  log(`CAPABILITY EVAL mode: prior art is advisory CONCLUSIONS only; engineers get no reference ` +
      `paths, and verify rejects byte-identical files as status:"plagiarized". ` +
      `reference evidence=${KNOWN_REFERENCE_HASHES.length
        ? `${KNOWN_REFERENCE_HASHES.length} out-of-band file hash(es)`
        : (KNOWN_REFERENCE_PATHS.length ? `${KNOWN_REFERENCE_PATHS.length} hidden path(s)` :
          '(none given — provenance check DISABLED)')}`);
}

// Reference containment: under capability_eval a reference tree must live outside the ancestor that
// holds the task, workflow and run artifacts, because engineers can reach anything one `ls ..` away.
// Set here and enforced by the post-Setup gate, which sweeps the same ancestor.
let RUN_TREE_ANCESTOR = '';
if (CAPABILITY_EVAL) {
  // Normalized by hand: workflow scripts run without Node's `path`/`fs`, so require() would throw
  // here at runtime and turn a safety check into a crash.
  const abs = (p) => {
    const out = [];
    for (const seg of String(p).split('/')) {
      if (!seg || seg === '.') continue;
      if (seg === '..') out.pop(); else out.push(seg);
    }
    return '/' + out.join('/');
  };
  const parts = [KERNEL_PATH_ORIG, EXP_ROOT, WORKFLOW_DIR].filter(Boolean).map(abs);
  let ancestor = parts.length ? parts[0] : '';
  for (const p of parts.slice(1)) {                       // longest common directory prefix
    const a = ancestor.split('/'), b = p.split('/');
    let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++;
    ancestor = a.slice(0, i).join('/') || '/';
  }
  RUN_TREE_ANCESTOR = ancestor;
  const inside = KNOWN_REFERENCE_PATHS.filter(
    (p) => ancestor && (abs(p) === ancestor || abs(p).startsWith(ancestor + '/')));
  if (inside.length) {
    throw new Error(
      `REFERENCE NOT CONTAINED: capability_eval is on, but ${inside.length} reference tree(s) sit ` +
      `inside the run's own tree (${ancestor}): ${inside.join(' ')}. An engineer reaches these by ` +
      `walking up one directory, so the run would measure a copy and report it as a derivation — ` +
      `that has already happened twice here. Move each tree outside ${ancestor} (for a git worktree: ` +
      `git worktree move <old> <new>) and pass the new path in known_reference_paths, so verify can ` +
      `still compare against it. Set capability_eval=false if you WANT the reference ported.`);
  }
  log(`REFERENCE CONTAINMENT ok: ${KNOWN_REFERENCE_PATHS.length} reference tree(s) resolve outside ` +
      `the run tree ${ancestor}. NOTE this checks CONFIGURED PATHS ONLY, and that is a small fraction ` +
      `of the exposure. Sweeping this tree for reference-only markers afterwards found the answer in ` +
      `five more places, none of them a known_reference_path: the positive control's own applied ` +
      `workspace left behind under artifacts/, a previous wave's output tree, the SIBLING EVAL DIRS ` +
      `inside exp_root (one directory up from the dir every engineer is handed by name), the durable ` +
      `patches in state_dir, and a git bundle of the candidate branch. Run ` +
      `scripts/reference_leak_sweep.sh --tree ${ancestor} before trusting capability_eval, and note ` +
      `it still cannot see a reference reachable as a branch in another repository the run can read ` +
      `(JIT cache, sibling clone) — only verify's byte-identity check stands between that and a ` +
      `counted win.`);
}

// Optional profile-analysis skill (see knowledge/analysis_skills/INDEX.md): after Profile, an
// analysis_engineer adds operator-specific structure (e.g. MoE route imbalance, stage time-share).
// Analysis only, never directions. Default none: no inputs and no prompt text are added.
const ANALYSIS_SKILL = String(A.analysis_skill != null ? A.analysis_skill : 'none').trim();
const ANALYSIS_SKILL_ON = !!ANALYSIS_SKILL && ANALYSIS_SKILL !== 'none' && ANALYSIS_SKILL !== 'false';
if (ANALYSIS_SKILL_ON && !/^[a-z0-9][a-z0-9_-]*$/.test(ANALYSIS_SKILL)) {
  throw new Error(`args.analysis_skill must be a safe skill slug, got ${JSON.stringify(ANALYSIS_SKILL)}`);
}
const ANALYSIS_SKILL_INPUTS = ANALYSIS_SKILL_ON ? {
  ANALYSIS_SKILL: ANALYSIS_SKILL,
  ANALYSIS_SKILL_DIR: `${WORKFLOW_DIR}/knowledge/analysis_skills/${ANALYSIS_SKILL}`,
} : {};
if (ANALYSIS_SKILL_ON) log(`Profile-analysis skill: ${ANALYSIS_SKILL} (analysis only; Step-3 TechLead owns directions).`);
// Generic mega (Expert Skills off) only; skill-on and non-mega runs get {}.
const GMI = MODE === 'mega' && !USE_EXPERT_SKILLS ? {
  MEGA_MEASUREMENT_GUIDE: `${WORKFLOW_DIR}/knowledge/mega_measurement.md`,
  GENERIC_MEGA_NOTE: 'Read MEGA_MEASUREMENT_GUIDE\'s role table and MEGA_GENERIC_LESSONS before acting; they apply to your role here.',
  MEGA_GENERIC_LESSONS: `${WORKFLOW_DIR}/knowledge/mega_generic_lessons.md`,
  MEGA_AB_TOOL: `${WORKFLOW_DIR}/tools/mega_graph_ab.py`,
} : {};
// MoE regime guide + paired graph A/B for runs pinning a megamoe Skill (it names Skill designs, so a
// generic run never gets it by default); args.mega_regimes='true'|'false' overrides.
const MEGA_REGIMES_ON = MODE === 'mega' && (A.mega_regimes != null
  ? String(A.mega_regimes) === 'true' : /^megamoe/.test(EXPERT_SKILL_ID));
const MRI = MEGA_REGIMES_ON ? {
  MEGA_REGIMES_GUIDE: `${WORKFLOW_DIR}/knowledge/mega_regimes.md`,
  MEGA_AB_TOOL: `${WORKFLOW_DIR}/tools/mega_graph_ab.py`,
  MEGA_REGIME_NOTE: 'Read MEGA_REGIMES_GUIDE for the target size\'s regime before choosing a direction; use MEGA_AB_TOOL for fast paired graph A/B while iterating.',
  ...(USE_EXPERT_SKILLS ? { MEGA_GENERIC_LESSONS: `${WORKFLOW_DIR}/knowledge/mega_generic_lessons.md` } : {}),
} : {};

// Continuation and feedback hooks, all optional (unset = identical prompts): STATE_DIR resumes per-
// (kernel, backend) state across waves; SHARED_KB/GLOBAL_KB are blackboards; E2E_FEEDBACK and
// HARNESS_ADDENDUM steer planning and benchmarking.
const STATE_DIR = String(A.state_dir || '').replace(/\/+$/, '');
const MEGA_RESUME_STATE = MODE === 'mega' && !!STATE_DIR &&
  String(A.mega_resume_state != null ? A.mega_resume_state : 'false') === 'true';
// Fast-test cache (mega + STATE_DIR only, default off): reuse a prior wave's front-matter measurements
// (control calibration, profile replays) when the validity key matches. It never touches the scoring
// denominator: every candidate A/B re-measures the frozen baseline.
const MEGA_FAST_TEST = MODE === 'mega' && !!STATE_DIR &&
  String(A.mega_fast_test != null ? A.mega_fast_test : 'false') === 'true';
const SHARED_KB = String(A.shared_kb || '').trim();
const GLOBAL_KB = String(A.global_kb || '').trim();   // run-global cross-KERNEL technique blackboard (deep)
const E2E_FEEDBACK = String(A.e2e_feedback || '').trim();
const HARNESS_ADDENDUM = String(A.harness_addendum || '').trim();
// On a resumed wave Analyze and baseline Profile return the prior result plus deltas instead of
// re-deriving it. Benchmark is never incremental (it re-pins a fresh baseline). Unset = identical prompts.
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
      // Workload-alignment fields, present only with a workload spec: weight is the case's baseline time
      // share (metric = Σweight / Σ(weight/speedup)); count is informational.
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
// Verdict vocabulary keeps clean/leak/skipped/unknown distinct. "The scanner could not run" can
// never be reported as clean — that collapse is the exact failure this subsystem exists to prevent.
const CONTAINMENT_GATE_SCHEMA = obj({
  verdict: { type: 'string', enum: ['clean', 'leak', 'skipped', 'unknown'] },
  findings: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, ['verdict']);

// Three-valued so that "could not read the pool" never collapses into "free".
const POOL_SCHEMA = obj({
  verdict: { type: 'string', enum: ['free', 'occupied', 'unknown'] },
  cards_total: { type: 'number' },
  cards_free: { type: 'number' },
  min_free_gib: { type: 'number' },
  foreign_pids: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, ['verdict']);

const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, workspace: { type: 'string' }, baseline_dir: { type: 'string' },
  kernel_name: { type: 'string' }, source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // Frozen baseline: timing and parity must use the real online kernel (baseline_src/ or
  // meta.baseline_callable), never kernel_src/. The run aborts if neither is confirmed.
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  mega_workspace_from_original: { type: 'boolean' },
  mega_candidate_state_checked: { type: 'boolean' },
  // Resume only: the director fills this when STATE_DIR holds a prior best, so a continued wave keeps
  // its cumulative speedup and history. Absent on a fresh run.
  resumed: { type: 'boolean' },
  workflow_revision: { type: 'string' },
  workflow_clean: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    // Declared, because memoryMerge keys rows by `id` and dedupes on it, and the ladder is read back
  // out of `roadmap_rung`. A row missing `id` merges as a new row every round; a row missing
  // `verdict` is treated by the next plan_round as untried. Both read as normal ledger growth.
  ledger: { type: 'array', items: obj({
    id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
    // confirmed | partial | unresolved | dead_end — see tech_lead.md "Verdict by SPEC, not by
    // result". `unresolved` is what keeps a rung re-openable instead of closing everything above it.
    verdict: { type: 'string', enum: ['confirmed', 'partial', 'unresolved', 'dead_end'] },
    roadmap_rung: { type: ['string', 'null'] },
    expected: { type: 'number' }, actual: { type: 'number' },
    lesson: { type: 'string' },
    first_round: { type: 'number' }, last_round: { type: 'number' },
  }, ['id', 'verdict']) },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
    // Shelf entries are keyed by id, aged by base_round and offered by files; an entry without files is
    // marked footprint=unknown and never offered.
    shelf: { type: 'array', items: obj({
      id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
      geomean: { type: 'number' }, patch: { type: 'string' },
      files: { type: 'array', items: { type: 'string' } },
      // Fusion chain id (its terminal rung), carried across waves so a chain's producer commit does not
      // age its consumer half off the shelf. Empty for a standalone optimisation.
      chain_id: { type: 'string' },
      footprint: { type: 'string', enum: ['known', 'unknown'] },
      base_round: { type: 'number' }, shelved_round: { type: 'number' },
      absorbed: { type: 'boolean' },
    }, ['id', 'geomean']) },
    candidate_registry: { type: 'array', items: { type: 'object', additionalProperties: true } },
    measurement_calibration: { type: 'object', additionalProperties: true },
    state_sequence: { type: 'number' },
    // round -> files CANONICAL took on that round. Keys are round numbers as strings.
    absorbed_files: { type: 'object' },
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name']);

const AUTHOR_SCHEMA = obj({
  authored: { type: 'boolean' }, target_language: { type: 'string' }, correctness: { type: 'string' },
  baseline_ms: { type: 'number' }, kernel_src_path: { type: 'string' }, entry_point: { type: 'string' },
  build: { type: 'boolean' }, notes: { type: 'string' },
}, ['authored', 'correctness']);

// mode=mega only. Every whole-kernel lane — validated artifact, skill reproduction, or open search —
// uses one lifecycle. A WIP may advance its own lane without becoming the global incumbent.
const MEGA_CANDIDATE_SCHEMA = obj({
  candidate_id: { type: 'string' },
  candidate_source: { type: 'string', enum: ['search', 'integrated'] },
  base_candidate_id: { type: 'string' },
  candidate_status: {
    type: 'string',
    enum: ['authoring', 'runnable', 'scored', 'finalist', 'rejected'],
    description: 'A finalized GPU-free source checkpoint remains authoring until runtime verification.',
  },
  claim_complete: {
    type: 'boolean',
    description: 'True when this turn finalized its committed artifact, including a GPU-free static checkpoint; hardware gates are separate fields.',
  },
  checkpoint_complete: { type: 'boolean' },
  structural_verified: { type: 'boolean' },
  runtime_verified: { type: 'boolean' },
  score_complete: { type: 'boolean' },
  verification_status: { type: 'string' },
  gpu_executed: { type: 'boolean' },
  structural_report: { type: 'string' },
  structural_skill_id: { type: 'string' },
  structural_candidate_head: { type: 'string' },
  structural_candidate_tree_digest: { type: 'string' },
  structural_skill_bundle_sha256: { type: 'string' },
  structural_planner_extension_sha256: { type: 'string' },
  structural_contract_revision: { type: 'string' },
  structural_contract_sha256: { type: 'string' },
  contract_failures: { type: 'array', items: obj({
    id: { type: 'string' }, category: { type: 'string' },
    severity: { type: 'string' },
    messages: { type: 'array', items: { type: 'string' } },
  }, ['id', 'category', 'severity', 'messages']) },
  failed_required_checks: { type: 'array', items: { type: 'string' } },
  plan_consistency_errors: { type: 'array', items: { type: 'string' } },
  input_errors: { type: 'array', items: { type: 'string' } },
  provenance_attestation_valid: { type: 'boolean' },
  attempt_id: { type: 'string' },
  evidence_manifest: { type: 'string' },
  target_guard: { type: 'string' },
  correctness: { type: 'string' },
  build: { type: 'boolean' },
  tree: { type: 'string' },
  head: { type: 'string' },
  patch_file: { type: 'string' },
  changed_files: { type: 'array', items: { type: 'string' } },
  topology: obj({}, []),
  absolute_score: { type: 'number' },
  per_case: perCase,
  activation_on_hardware: { type: 'string' },
  path_marker: { type: 'string' },
  launches: { type: 'number' },
  liveness_replays: { type: 'number' },
  graph_safe: { type: 'string' },
  topology_sig: { type: 'string' },
  provenance: { type: 'string' },
  next_blocker: { type: 'string' },
  notes: { type: 'string' },
  activation: obj({
    mode: { type: 'string' }, switch_name: { type: 'string' }, switch_value: { type: 'string' },
    path_marker: { type: 'string' }, marker_how: { type: 'string' },
    switches: {
      type: 'array',
      items: obj({ switch_name: { type: 'string' }, switch_value: { type: 'string' } }, []),
    },
  }, []),
}, ['candidate_id', 'candidate_source', 'candidate_status', 'claim_complete',
  'contract_failures']);

const EXPERT_SKILL_CONTRACT_VERIFY_SCHEMA = obj({
  candidate_id: { type: 'string' },
  candidate_head: { type: 'string' },
  candidate_tree_digest: { type: 'string' },
  skill_id: { type: 'string' },
  skill_bundle_sha256: { type: 'string' },
  planner_extension_sha256: { type: 'string' },
  claim_complete: { type: 'boolean' },
  report_path: { type: 'string' },
  structural_compatible: { type: 'boolean' },
  independent_structure_pass: { type: 'boolean' },
  capability_eligible: { type: 'boolean' },
  reference_copy_detected: { type: 'boolean' },
  reference_copy_suspected: { type: 'boolean' },
  provenance_status: { type: 'string' },
  contract_revision: { type: 'string' },
  contract_sha256: { type: 'string' },
  contract_failures: { type: 'array', items: obj({
    id: { type: 'string' }, category: { type: 'string' },
    severity: { type: 'string' },
    messages: { type: 'array', items: { type: 'string' } },
  }, ['id', 'category', 'severity', 'messages']) },
  plan_consistent: { type: 'boolean' },
  semantic_features_passed: { type: 'number' },
  semantic_features_total: { type: 'number' },
  hardware_verified: { type: 'boolean' },
  accuracy_verified: { type: 'boolean' },
  performance_verified: { type: 'boolean' },
  next_blocker: { type: 'string' },
  notes: { type: 'string' },
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

// Operator-neutral whole-kernel topology.
const MEGA_TOPOLOGY_SCHEMA = obj({
  launch_count: { type: 'number' },
  included_regions: { type: 'array', items: { type: 'string' } },
  included_queues: { type: 'array', items: { type: 'string' } },
  capabilities: { type: 'array', items: { type: 'string' } },
  require_overlap: { type: 'boolean' },
  parameters: { type: 'object', additionalProperties: true },
  notes: { type: 'string' },
});

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

const ANALYZE_SCHEMA = obj({
  kernel_type: { type: 'string' }, kernel_file: { type: 'string' }, entry_point: { type: 'string' },
  modifiable_files: { type: 'array', items: { type: 'string' } },
  bottleneck_guess: { type: 'string' }, roadmap_summary: { type: 'string' },
  // The ladder: Analyze's ranked rungs and what each is conditional on. Each rung declares id,
  // gated_on (rungs that must run to spec first; [] is an explicit answer), mandatory_arms and
  // is_positive_control, so roadmapLadderGate can check dispatch order against them; prose cannot be
  // checked.
  candidate_directions: { type: 'array', items: obj({
    id: { type: 'string' },              // short, stable, quotable: D0, D1, ...
    title: { type: 'string' },
    candidate_id: { type: 'string' },
    candidate_source: { type: 'string', enum: ['search', 'integrated'] },
    base_candidate_id: { type: 'string' },
    specialty: { type: 'string' },
    focus_files: { type: 'array', items: { type: 'string' } },
    prompt: { type: 'string' },
    rationale: { type: 'string' },
    rung_deviation: { type: 'string' },
    expected_speedup: { type: 'number' },
    gated_on: { type: 'array', items: { type: 'string' } },
    mandatory_arms: { type: 'array', items: { type: 'string' } },
    is_positive_control: { type: 'boolean' },
    // How a rung closes: terminal on a measurement; enabling on functional acceptance, since a
    // prerequisite step has no standalone speedup. See stepRoleOf.
    step_role: { type: 'string', enum: ['enabling', 'terminal'] },
    enables: { type: 'string' },
    cost_budget_pct: { type: 'number' },
    // A terminal rung must say what shape closes it. Without this a three-launch partial fusion and
    // the requested two-launch terminal are indistinguishable to the rung tracker.
    target_shape: {
      type: 'object',
      properties: {
        launches: { type: 'number' },
        stages_fused: { type: 'array', items: { type: 'string' } },
        require_overlap: { type: 'boolean' },
      },
      required: ['launches'],
      additionalProperties: true,
    },
    target_topology: MEGA_TOPOLOGY_SCHEMA,
  }, ['id', 'title', 'gated_on']) },
  // perf_knowledge mapping (reference only): operator/language and the most relevant card paths.
  // Empty when no card applies.
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] },
  kk_refs: { type: 'array', items: { type: 'string' } },
  baseline_operator_map: { type: 'array', items: obj({
    role: { type: 'string' },
    file: { type: 'string' },
    symbol: { type: 'string' },
    kind: { type: 'string' },
    semantics: { type: 'string' },
    inputs: { type: 'array', items: { type: 'string' } },
    outputs: { type: 'array', items: { type: 'string' } },
    layout: { type: 'string' },
    synchronization: { type: 'string' },
    action: { type: 'string', enum: [
      'reuse', 'extract_shared', 'inline', 'rewrite_equivalent', 'rewire', 'extend',
    ] },
    target_role: { type: 'string' },
    proof: { type: 'string' },
  }, ['role', 'file', 'symbol', 'semantics', 'action']) },
  prior_art: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // Fine-grained dependency graph; taskGraphGate owns semantic validation.
  task_graph: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      nodes: { type: 'array', items: obj({
        id: { type: 'string' }, stage: { type: 'string' }, duration_us: { type: 'number' },
        // measured | assumed. The gate counts assumed durations, because a graph whose every
        // duration is assumed is a hypothesis wearing a measurement's formatting.
        source: { type: 'string', enum: ['measured', 'assumed'] },
      }, ['id', 'source']) },
      edges: { type: 'array', items: obj({
        from: { type: 'string' }, to: { type: 'string' },
        // data | launch_boundary | barrier | assumed. `launch_boundary` is the load-bearing value:
        // it marks an ordering the DATA does not require, which is the only kind fusion can delete.
        enforced_by: { type: 'string' },
        scope: { type: 'string' },
        fan_in: { type: 'number' },
        // Consumer operands the producer does not supply (weights, scales, descriptors). They can load while
        // the producer runs and have no node in the graph, so they are named here. [] is a real answer.
        producer_independent_operands: { type: 'array', items: { type: 'string' } },
      }, ['from', 'to', 'enforced_by']) },
      unknowns: { type: 'array', items: { type: 'object', additionalProperties: true } },
      critical_path_us: { type: 'number' },
      measured_e2e_us: { type: 'number' },
    },
  },
  // Per-pipe resource timeline, required alongside task_graph: the graph says what MAY overlap, this
  // says which unit is idle and what could fill it.
  //   {pipes: [{stage, pipe, utilization_pct, source}], interkernel_gap_us: {median, max, n_boundaries},
  //    class: throughput_bound|latency_bound|launch_bound|mixed, stall_reason: [...],
  //    idle_pipe_opportunities: [...], closed_axes: [...], unknowns: [...]}
  // `class` rules out lever families early (knowledge/pipe_occupancy.md). The gate drops pipe rows
  // without a finite utilization_pct, so the fields are declared.
  resource_timeline: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      pipes: { type: 'array', items: obj({
        stage: { type: 'string' },
        pipe: { type: 'string' },   // valu|mfma|vmem|lds|hbm|interconnect|scalar
        utilization_pct: { type: ['number', 'null'] },
        source: { type: 'string' },
      }, ['stage', 'pipe', 'utilization_pct']) },
      interkernel_gap_us: obj({
        median: { type: 'number' }, max: { type: 'number' }, n_boundaries: { type: 'number' },
      }, ['median']),
      // The gate DERIVES the class from the numbers and compares it against this one; a mismatch is
      // itself reportable, so the stated value must be present to be contradicted.
      class: { type: 'string', enum: ['throughput_bound', 'latency_bound', 'launch_bound', 'mixed'] },
      // id, the size fields and rides_on are read by the gate, so they are declared; rides_on records that
      // another rung absorbs this hole.
      idle_pipe_opportunities: { type: 'array', items: obj({
        id: { type: 'string' }, window: { type: 'string' }, idle_pipe: { type: 'string' },
        recoverable_us: { type: ['number', 'null'] }, pct_of_e2e: { type: ['number', 'null'] },
        rides_on: { type: 'string' },
      }, []) },
      closed_axes: { type: 'array', items: { type: 'object', additionalProperties: true } },
      unknowns: { type: 'array', items: { type: 'object', additionalProperties: true } },
    },
  },
  // Lowerable whole-kernel schedule contract. Unlike roadmap prose, every downstream Mega role
  // receives this object verbatim and static Verify checks the authored source against it.
  mega_plan_ir: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      plan_version: { type: 'string' },
      expert_skill_id: { type: ['string', 'null'] },
      expert_skill_revision: { type: ['string', 'null'] },
      expert_skill_bundle_sha256: { type: ['string', 'null'] },
      expert_skill_planner_extension_sha256: { type: ['string', 'null'] },
      expert_skill_validation_status: { type: ['string', 'null'] },
      expert_skill_auto_apply: { type: ['boolean', 'null'] },
      expert_skill_explicit_pin_modes: {
        type: 'array', items: { type: 'string' },
      },
      target: obj({
        launch_count: { type: 'number' },
        required_regions: { type: 'array', items: { type: 'string' } },
        required_queues: { type: 'array', items: { type: 'string' } },
        required_capabilities: { type: 'array', items: { type: 'string' } },
      }, ['launch_count', 'required_regions', 'required_queues']),
      work_domains: { type: 'array', items: obj({
        id: { type: 'string' }, unit: { type: 'string' },
        extent: { type: 'string' }, index_type: { type: 'string' },
      }, ['id', 'unit', 'extent', 'index_type']) },
      regions: { type: 'array', items: obj({
        id: { type: 'string' }, role: { type: 'string' },
        work_domain: { type: 'string' }, engine: { type: 'string' },
      }, ['id', 'role', 'work_domain']) },
      buffers: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: {
          id: { type: 'string' }, role: { type: 'string' },
          element_type: { type: 'string' }, capacity: { type: 'string' },
          address_space: { type: 'string' }, format: { type: 'string' },
          producers: { type: 'array', items: { type: 'string' } },
          consumers: { type: 'array', items: { type: 'string' } },
          lifetime: { type: 'string' }, alias_group: { type: 'string' },
        },
        required: ['id', 'role', 'element_type', 'capacity', 'address_space'],
      } },
      counters: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: {
          id: { type: 'string' }, element_type: { type: 'string' },
          capacity: { type: 'string' }, scope: { type: 'string' },
          lifecycle: { type: 'string' }, reset_owner: { type: 'string' },
          generation: { type: 'string' },
          producers: { type: 'array', items: { type: 'string' } },
          consumers: { type: 'array', items: { type: 'string' } },
          publish: { type: 'string' }, wait: { type: 'string' },
        },
        required: [
          'id', 'element_type', 'capacity', 'scope', 'lifecycle',
          'producers', 'consumers',
        ],
      } },
      queues: { type: 'array', items: obj({
        id: { type: 'string' }, work_domain: { type: 'string' },
        claim_unit: { type: 'string' }, owner: { type: 'string' },
        shards: { type: 'number' }, stride_bytes: { type: 'number' },
        chunk_policy: { type: 'string' }, ready_when: { type: 'string' },
        priority: { type: 'number' },
      }, ['id', 'work_domain', 'claim_unit', 'owner']) },
      events: { type: 'array', items: obj({
        id: { type: 'string' }, producer: { type: 'string' },
        consumer: { type: 'string' }, scope: { type: 'string' },
        counter: { type: 'string' }, publish: { type: 'string' },
        wait: { type: 'string' },
      }, ['id', 'producer', 'consumer', 'scope']) },
      abi: {
        type: 'object',
        additionalProperties: true,
        properties: {
          entry_point: { type: 'string' },
          arguments: { type: 'array', items: {
            type: 'object', additionalProperties: true,
            properties: {
              id: { type: 'string' }, type: { type: 'string' },
              role: { type: 'string' }, source: { type: 'string' },
              consumers: { type: 'array', items: { type: 'string' } },
              optional: { type: 'boolean' },
            },
            required: ['id', 'type', 'role'],
          } },
          parameters: { type: 'object', additionalProperties: true },
        },
        required: ['entry_point', 'arguments'],
      },
      resources: {
        type: 'object',
        additionalProperties: true,
        properties: {
          arch: { type: 'string' },
          workgroup: obj({
            wave_size: { type: 'number' }, wave_count: { type: 'number' },
            thread_count: { type: 'number' },
          }, ['wave_size', 'wave_count', 'thread_count']),
          local_memory: obj({
            total_bytes: { type: 'number' },
            limit_bytes: { type: 'number' },
            allocation_rule: { type: 'string' },
            allocations: { type: 'array', items: {
              type: 'object', additionalProperties: true,
            } },
          }, ['total_bytes', 'limit_bytes', 'allocation_rule', 'allocations']),
          registers: obj({
            vgpr_max: { type: ['number', 'null'] },
            sgpr_max: { type: ['number', 'null'] },
            scratch_bytes_max: { type: ['number', 'null'] },
            source: { type: 'string' },
          }, ['vgpr_max', 'sgpr_max', 'scratch_bytes_max', 'source']),
          occupancy: obj({
            compute_units: { type: 'number' },
            grid_blocks: { type: 'number' },
            min_workgroups_per_cu: { type: 'number' },
            requires_full_grid_residency: { type: 'boolean' },
          }, [
            'compute_units', 'grid_blocks', 'min_workgroups_per_cu',
            'requires_full_grid_residency',
          ]),
          engines: { type: 'array', items: obj({
            region: { type: 'string' }, pipe: { type: 'string' },
          }, ['region', 'pipe']) },
          parameters: { type: 'object', additionalProperties: true },
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
            kind: { type: 'string' },
            carried_state: { type: 'array', items: { type: 'string' } },
            queue_priority: { type: 'array', items: { type: 'string' } },
            queue_selection: { type: 'string' },
            progress_invariants: { type: 'array', items: { type: 'string' } },
          }, ['kind', 'carried_state', 'progress_invariants']),
          policies: { type: 'object', additionalProperties: true },
          parameters: { type: 'object', additionalProperties: true },
        },
        required: ['primary_loop', 'policies'],
      },
      compiler_constraints: { type: 'array', items: {
        type: 'object', additionalProperties: true,
        properties: { id: { type: 'string' }, rule: { type: 'string' } },
        required: ['id', 'rule'],
      } },
      evidence_requirements: { type: 'object', additionalProperties: true },
      known_unknowns: { type: 'array', items: { type: 'object', additionalProperties: true } },
    },
    required: [
      'plan_version', 'target', 'work_domains', 'regions', 'buffers', 'counters',
      'queues', 'events', 'abi', 'resources', 'schedule',
      'compiler_constraints', 'evidence_requirements', 'known_unknowns',
    ],
  },
}, ['kernel_type', 'roadmap_summary']);
const MEGA_ANALYZE_SCHEMA = {
  ...ANALYZE_SCHEMA,
  properties: {
    ...ANALYZE_SCHEMA.properties,
    candidate_directions: {
      ...ANALYZE_SCHEMA.properties.candidate_directions,
      items: {
        ...ANALYZE_SCHEMA.properties.candidate_directions.items,
        required: [...new Set([
          ...(ANALYZE_SCHEMA.properties.candidate_directions.items.required || []),
          'candidate_id', 'candidate_source', 'base_candidate_id', 'specialty',
          'focus_files', 'prompt', 'target_shape', 'target_topology',
        ])],
      },
    },
  },
  required: [...new Set([
    ...(ANALYZE_SCHEMA.required || []),
    'modifiable_files', 'candidate_directions', 'prior_art',
    'baseline_operator_map', 'task_graph', 'resource_timeline', 'mega_plan_ir',
  ])],
};

const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  // Workload-aligned outputs (present with a workload spec): metric = Σweight / Σ(weight/speedup);
  // baseline_weighted_total_ms is the baseline time the weights represent.
  workload_aligned: { type: 'boolean' },
  baseline_weighted_total_ms: { type: 'number' },
  weights_provenance: { type: 'string' }, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  // args.positive_control result. switch_present=false means both arms ran the same code. Fields are
  // declared so a dropped one cannot validate as a passing control.
  positive_control: obj({
    ran: { type: 'boolean' },
    claim_complete: { type: 'boolean' },
    attempt_id: { type: 'string' },
    evidence_manifest: { type: 'string' },
    switch_present: { type: ['boolean', 'null'] },
    switch_checked: { type: 'string' },
    measured_pct: { type: ['number', 'null'] },
    expected_lo: { type: 'number' }, expected_hi: { type: 'number' },
    passed: { type: 'boolean' }, reps: { type: 'number' },
    control_pairs_pct: { type: 'array', items: { type: 'number' } },
    null_pairs_pct: { type: 'array', items: { type: 'number' } },
    null_arm_pct: { type: 'number' },
    note: { type: 'string' },
    // Required only when the object exists at all: a run with no control omits it entirely. When it
    // IS reported, these three are what separate a control from a number, so none may be defaulted.
  }, ['ran', 'claim_complete', 'switch_present', 'measured_pct']),
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

// Operator-neutral native analysis for whole-kernel fusion. Operator-specific component names,
// transport counters and wait metrics are map/array values supplied by the caller's harness.
const MEGA_ANALYSIS_SCHEMA = obj({
  bottleneck: { type: 'string' },
  top_opportunities: { type: 'array', items: { type: 'string' } },
  dispatch_count: { type: 'number' },
  device: { type: 'string' },
  path_marker: { type: 'string' },
  path_marker_count: { type: 'number' }, world_size: { type: 'number' },
  rank_max_ms: { type: 'number' },
  component_timings: { type: 'array', items: {
    type: 'object', additionalProperties: true,
  } },
  speedup_pct: { type: ['number', 'null'] },
  perf_guard_floor: { type: ['number', 'null'] }, perf_guard_pass: { type: 'boolean' },
  transport_metrics: { type: 'object', additionalProperties: true },
  wait_metrics: { type: 'object', additionalProperties: true },
  per_rank_straggler: { type: 'array', items: { type: 'object', additionalProperties: true } },
  fusion_note: { type: 'string' },
  summary_path: { type: 'string' },
}, ['bottleneck', 'top_opportunities']);

// Fast-test cache load: the agent recomputes the validity key from disk (frozen base revision, harness
// hash, control spec, guards) and compares it with the stored key; only an exact match (key_valid)
// reuses prior measurements. bench/analysis have the same shape as the live Benchmark/Profile results.
const FAST_TEST_LOAD_SCHEMA = obj({
  cache_present: { type: 'boolean' },   // any cache artifact found at all
  key_valid: { type: 'boolean' },       // cache present AND its stored key matches the current key
  commandment_materialized: { type: 'boolean' },
  commandment_current_identity: { type: 'boolean' },
  current_key: { type: 'string' }, cached_key: { type: 'string' },
  note: { type: 'string' },
  bench: { type: 'object', additionalProperties: true },     // reconstructed BENCH_SCHEMA payload (CACHE_KIND=bench)
  analysis: { type: 'object', additionalProperties: true },  // reconstructed MEGA_ANALYSIS payload (CACHE_KIND=profile)
}, [
  'cache_present', 'key_valid', 'commandment_materialized',
  'commandment_current_identity',
]);

// Fast-test cache publish: copy this wave's fresh artifacts into STATE_DIR/fast_test_cache and write
// fast_test_key.json with the validity key computed from disk.
const FAST_TEST_PUBLISH_SCHEMA = obj({
  published: { type: 'boolean' }, key: { type: 'string' },
  cache_dir: { type: 'string' }, files: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, ['published']);

const ANALYSIS_RESULT_SCHEMA = obj({
  status: { type: 'string', enum: ['ready', 'degraded'] },
  analysis_skill: { type: 'string' },
  analysis_schema_version: { type: 'string' },
  analysis_status: {
    type: 'string',
    enum: ['awaiting_measurement', 'evidence_complete', 'unavailable'],
  },
  analysis_json: { type: 'string' },
  failure_note: { type: 'string' },
}, [
  'status',
  'analysis_skill',
  'analysis_schema_version',
  'analysis_status',
  'analysis_json',
  'failure_note',
]);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'distributed', 'deep_explore'] },
      focus_files: { type: 'array', items: { type: 'string' } },
      expected_speedup: { type: 'number' }, prompt: { type: 'string' },
      kk_refs: { type: 'array', items: { type: 'string' } }, // optional: perf_knowledge card paths for THIS direction (REFERENCE ONLY)
      // Binds the direction to Analyze's pipe table: fills_pipe is the unit it loads or relieves,
      // pipe_util_pct that unit's measured utilization, headroom_basis which bound caps the claim (idle
      // fraction or measured_e2e_us - critical_path_us). See knowledge/pipe_occupancy.md.
      fills_pipe: { type: 'string' },
      pipe_util_pct: { type: 'number' },
      headroom_basis: { type: 'string' },
      graph_refs: { type: 'array', items: { type: 'string' } }, // task_graph node/edge ids this direction acts on
      // The idle_pipe_opportunities id this direction fills, so each hole can be checked as owned or not.
      fills_hole: { type: 'string' },
      // Binds the direction to the ladder: roadmap_rung is the rung id or `off_ladder`. rung_deviation is
      // required when the rung is off the ladder, out of order, or has unmet gated_on: say what is skipped
      // and what happens to it. Skipping is allowed; skipping silently is not.
      roadmap_rung: { type: 'string' },
      rung_deviation: { type: 'string' },
      // Step kind: terminal (default) is judged on speed; enabling is a prerequisite in a multi-step fusion,
      // judged on function (builds, path taken, correct, no deadlock). enables names the terminal rung and is
      // required for enabling steps; cost_budget_pct is the slowdown this step is allowed.
      step_role: { type: 'string', enum: ['enabling', 'terminal'] },
      enables: { type: 'string' },
      cost_budget_pct: { type: 'number' },
      gated_on: { type: 'array', items: { type: 'string' } },
      mandatory_arms: { type: 'array', items: { type: 'string' } },
      target_shape: {
        type: 'object',
        properties: {
          launches: { type: 'number' },
          stages_fused: { type: 'array', items: { type: 'string' } },
          require_overlap: { type: 'boolean' },
        },
        required: ['launches'],
        additionalProperties: true,
      },
      // Operator-neutral multi-lever descriptor for a whole-kernel candidate.
      // Skill-specific region/queue/capability names remain data values.
      target_topology: MEGA_TOPOLOGY_SCHEMA,
      candidate_id: { type: 'string' },
      candidate_source: { type: 'string' },
      base_candidate_id: { type: 'string' },
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

const MEGA_PLAN_SCHEMA = {
  ...PLAN_SCHEMA,
  properties: {
    ...PLAN_SCHEMA.properties,
    directions: {
      ...PLAN_SCHEMA.properties.directions,
      items: {
        ...PLAN_SCHEMA.properties.directions.items,
        properties: {
          ...PLAN_SCHEMA.properties.directions.items.properties,
          candidate_source: {
            type: 'string', enum: ['search', 'integrated'],
          },
        },
        required: [...new Set([
          ...(PLAN_SCHEMA.properties.directions.items.required || []),
          'candidate_id', 'candidate_source', 'base_candidate_id',
          'focus_files', 'roadmap_rung', 'step_role', 'gated_on',
          'target_shape', 'target_topology',
        ])],
      },
    },
  },
  required: [...new Set([...(PLAN_SCHEMA.required || []), 'reasoning'])],
};

const ENG_SCHEMA = obj({
  engineer_id: { type: 'string' }, specialty: { type: 'string' }, task: { type: 'string' },
  strategy: { type: 'string' }, speedup_geomean: { type: 'number' }, speedup_arithmetic: { type: 'number' },
  // Time-weighted ratio-of-sums vs the TRUE baseline (PRIMARY metric when workload_aligned).
  // = Σ weight_i / Σ (weight_i / speedup_i). Omitted on unweighted runs.
  speedup_weighted: { type: 'number' },
  per_case: perCase, status: { type: 'string' }, patch_file: { type: 'string' },
  claim_complete: { type: 'boolean' }, attempt_id: { type: 'string' },
  evidence_manifest: { type: 'string' },
  candidate_id: { type: 'string' }, candidate_source: { type: 'string' },
  base_candidate_id: { type: 'string' }, candidate_status: { type: 'string' },
  candidate_tree: { type: 'string' }, candidate_head: { type: 'string' },
  topology_sig: { type: 'string' }, provenance: { type: 'string' },
  next_blocker: { type: 'string' },
  // Concrete arms completed by the engineer. Verify independently reports its own list; a rung
  // whose mandatory control arm was omitted stays open even if the candidate produced a number.
  arms_run: { type: 'array', items: { type: 'string' } },
  strategies_tried: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // How the new code is turned on and how Verify proves it ran. default_on changes behavior with no
  // switch; switch requires switch_name/switch_value and a path_marker to grep. An ungated-by-nobody
  // switch would benchmark the baseline and read 1.000x.
  activation: obj({
    mode: { type: 'string' }, switch_name: { type: 'string' }, switch_value: { type: 'string' },
    path_marker: { type: 'string' }, marker_how: { type: 'string' },
  }, []),
}, ['status', 'speedup_geomean']);

const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  claim_complete: { type: 'boolean' }, attempt_id: { type: 'string' },
  timeout_recovery_safe: { type: 'boolean' },
  active_gpu_processes: { type: 'number' },
  lane_lock_free: { type: 'boolean' },
  evidence_complete: { type: 'boolean' },
  evidence_manifest: { type: 'string' }, candidate_id: { type: 'string' },
  candidate_source: { type: 'string' }, candidate_tree: { type: 'string' },
  candidate_head: { type: 'string' },
  activation: obj({
    mode: { type: 'string' }, switch_name: { type: 'string' }, switch_value: { type: 'string' },
    path_marker: { type: 'string' }, marker_how: { type: 'string' },
    switches: {
      type: 'array',
      items: obj({ switch_name: { type: 'string' }, switch_value: { type: 'string' } }, []),
    },
  }, []),
  verified_geomean: { type: ['number', 'null'] }, verified_arithmetic: { type: ['number', 'null'] },
  verified_weighted: { type: ['number', 'null'] }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: { type: 'string' }, notes: { type: 'string' },
  graph_safe: { type: 'string' },
  liveness: { type: 'string' }, // pass|fail|n/a — deadlock/stale-read stress (distributed specialty)
  replay_count: { type: 'number' },
  replay_results: { type: 'array', items: obj({
    guard: { type: 'string' }, count: { type: 'number' },
    status: { type: 'string' }, graph_safe: { type: 'string' },
  }, ['guard', 'count', 'status']) },
  arms_run: { type: 'array', items: { type: 'string' } },
  // Accuracy as a number (metric, value, threshold, guard/route) so functionalAcceptance can check
  // value < threshold. Missing fields fall back to the `correctness` string check.
  accuracy: obj({
    metric: { type: 'string' }, value: { type: 'number', minimum: 0 },
    threshold: { type: 'number', exclusiveMinimum: 0 },
    guard: { type: 'string' }, method: { type: 'string' },
  }, []),
  accuracy_results: { type: 'array', items: obj({
    metric: { type: 'string' }, value: { type: 'number', minimum: 0 },
    threshold: { type: 'number', exclusiveMinimum: 0 },
    guard: { type: 'string' }, method: { type: 'string' },
  }, ['metric', 'value', 'threshold', 'guard']) },
  // Launches per call for base and candidate, read paired; target is caller-owned; how_counted is the
  // evidence (trace record count, marker tally). Missing fields leave the launch criterion unjudged.
  launch_shape: obj({
    launches_base: { type: 'number' }, launches_cand: { type: 'number' },
    per_rank: { type: 'boolean' }, target: { type: 'number' },
    stages_fused: { type: 'array', items: { type: 'string' } }, how_counted: { type: 'string' },
  }, []),
  // reps = interleaved A/B pairs; null_arm_pct = the delta of an arm doing identical work (the noise
  // floor). A win smaller than |null_arm_pct| is unreadable. Missing or low values make the result
  // provisional rather than void.
  reps: { type: 'number' }, null_arm_pct: { type: 'number' },
  // Raw per-pair readings tagged by guard/route, so the driver can split caller-declared bimodal guards
  // by state (modeSplit/pairedModeAware) and check the number came from a target guard. Absent:
  // verified_geomean is used as before.
  paired_readings: { type: 'array', items: obj({
    guard: { type: 'string' }, base: { type: 'number' }, cand: { type: 'number' },
  }, ['guard', 'base', 'cand']) },
  // Did the patched path execute in the measured run? yes requires the observed path_marker; no or
  // unknown voids the comparison rather than making it negative.
  activation_confirmed: { type: 'string' },  // yes|no|unknown
  activation_evidence: { type: 'string' },   // the command + the marker output that proves it
  // Did the switched path run on the device this round? yes requires a gpu_lock-wrapped run with the
  // switch set and a device-side observable (timing, trace record, on-device marker). Compile-only or
  // CPU checks are no; n/a means there is no switched path. With REQUIRE_HW_ACTIVATION, a committable or
  // enabling candidate must be yes.
  activation_on_hardware: { type: 'string' }, // yes|no|n/a|unknown
  hardware_evidence: { type: 'string' },       // the gpu_lock cmd + the device-side observable it produced
  // A marker proves the host path ran, not that the arms compiled differently: a JIT disk cache can
  // serve one binary to both arms if the switch is not in the cache key. For JIT candidates Verify
  // reports cache keys / IR-ISA hashes and they must differ; n/a is for non-JIT candidates.
  artifact_distinct: { type: 'string' },     // yes|no|n/a|unknown
  artifact_hash_base: { type: 'string' },
  artifact_hash_candidate: { type: 'string' },
  // Files the patch touches. Required: an empty list would read as "touches nothing" and make the
  // patch look combinable. The candidate shelf uses it to decide whether a patch still applies.
  touched_files: { type: 'array', items: { type: 'string' } },
  // Overlap (see knowledge/overlap_instrument.md). After fusion there is one trace record, so overlap
  // needs its own meter. measured is three-valued: no = measured and none found; unknown = not
  // measured.
  overlap: obj({
    measured: { type: 'string', enum: ['yes', 'no', 'unknown'] },
    fraction: { type: 'number' },        // wall-clock, >=2 distinct roles active
    cu_fraction: { type: 'number' },     // CU-weighted; wall-clock alone is manufacturable
    method: { type: 'string' },
    // The meter's controls: base_reading on the unfused path must read ~0 (else the meter is broken);
    // forced_reading must read high (else the meter is dead).
    base_reading: { type: 'number' },
    forced_reading: { type: 'number' },
    clock_skew_ns: { type: 'number' },   // s_memrealtime coherence across XCDs — assumed at your peril
    meter_overhead_pct: { type: 'number' },
    note: { type: 'string' },
  }, ['measured']),
  // Attribution (see knowledge/fusion_preconditions.md): for any patch that adds, removes or merges a
  // launch, report changed_us (patched kernel) and replaced_sum_us (the kernels it replaces), paired at
  // the same guard. An end-to-end win can come from inter-kernel gaps rather than the kernel. A sign
  // mismatch rejects under promotion_metric=changed_kernel; under operator_e2e it only withholds the
  // mechanism claim.
  attribution: obj({
    changed_us: { type: 'number' },
    replaced_sum_us: { type: 'number' },
    guard: { type: 'string' },
    // e2e minus the sum of all kernel times, both arms. Where a gap-win hides.
    residual_ms_base: { type: 'number' },
    residual_ms_cand: { type: 'number' },
    method: { type: 'string' },
    absolute_to_frozen: { type: 'boolean' },
    note: { type: 'string' },
  }, []),
}, ['status', 'verified_geomean', 'touched_files']);

const INTEGRATE_SCHEMA = obj({
  attempted: { type: 'boolean' },
  combos_tried: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // The script reads patch_file, geomean, weighted, arithmetic and per_case, so they are declared.
  // Without touched_files the union of all patches on the table is assumed, which over-ages the shelf
  // instead of offering a stale patch.
  best: obj({
    patch_file: { type: 'string' }, geomean: { type: 'number' },
    weighted: { type: ['number', 'null'] }, arithmetic: { type: 'number' },
    per_case: perCase, touched_files: { type: 'array', items: { type: 'string' } },
    // PATCHES / SHELF_PATCHES ids this combination contains; informational, shelf aging does not
    // depend on it.
    patches: { type: 'array', items: { type: 'string' } },
  }, []),
  improved_over_best_individual: { type: 'boolean' },
  conclusion: { type: 'string' }, notes: { type: 'string' },
}, ['attempted', 'conclusion']);

const MEMORY_SCHEMA = obj({
  insights: { type: 'array', items: { type: 'string' } },
  candidate_registry: { type: 'array', items: { type: 'object', additionalProperties: true } },
  measurement_calibration: { type: 'object', additionalProperties: true },
  state_written: { type: 'boolean' }, state_round: { type: 'number' },
  state_path: { type: 'string' }, state_generation: { type: 'string' },
  state_sequence: { type: 'number' },
  // memoryMerge keys and dedupes rows by id and reads the ladder from roadmap_rung; a row without
  // verdict is treated as untried.
  ledger: { type: 'array', items: obj({
    id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
    // confirmed | partial | unresolved | dead_end — see tech_lead.md "Verdict by SPEC, not by
    // result". `unresolved` is what keeps a rung re-openable instead of closing everything above it.
    verdict: { type: 'string', enum: ['confirmed', 'partial', 'unresolved', 'dead_end'] },
    roadmap_rung: { type: ['string', 'null'] },
    expected: { type: 'number' }, actual: { type: 'number' },
    lesson: { type: 'string' },
    first_round: { type: 'number' }, last_round: { type: 'number' },
  }, ['id', 'verdict']) },
  // Durable hardware/toolchain facts (lowerings true for every kernel on this box), as opposed to
  // insights about this operator. knowledge/ is read-only to the script, so candidates are surfaced in
  // the log and report for a human to merge. See tech_lead.md for the bar.
  knowledge_delta: { type: 'array', items: obj({
    fact: { type: 'string' },
    // Which card it belongs in, so the merge is a placement decision and not a search.
    card: { type: 'string' },
    // The artifact that establishes it: an ISA dump path, a counter, a sha256, a measured number.
    // A fact whose evidence is "the engineer said so" is an insight, not a knowledge delta.
    evidence: { type: 'string' },
    // What a reader would have believed without this fact; a lowering earns a card only when the
    // source-level reading says otherwise.
    contradicts: { type: 'string' },
    // true = holds for any kernel on this target; false = this operator's shape. Only the first
    // kind is a card edit; the second belongs in the report and nowhere else.
    generalizes: { type: 'boolean' },
  }, ['fact', 'evidence', 'generalizes']) },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const COMMIT_SCHEMA = obj({
  committed: { type: 'boolean' }, current_best_diff: { type: 'string' }, note: { type: 'string' },
  head_before: { type: 'string' }, head_after: { type: 'string' },
  exact_patch: { type: 'boolean' }, transaction_valid: { type: 'boolean' },
}, ['committed']);

const MEGA_SELECTION_SCHEMA = obj({
  claim_complete: { type: 'boolean' },
  attempt_id: { type: 'string' },
  evidence_manifest: { type: 'string' },
  selected_candidate_id: { type: 'string' },
  selected_source: { type: 'string' },
  selected_tree: { type: 'string' },
  selected_head: { type: 'string' },
  materialized_from_head: { type: 'string' },
  selected_score: { type: 'number' },
  selected_patch: { type: 'string' },
  candidates: { type: 'array', items: obj({
    candidate_id: { type: 'string' }, source: { type: 'string' },
    tree: { type: 'string' }, head: { type: 'string' }, score: { type: 'number' },
    status: { type: 'string' },
    touched_files: { type: 'array', items: { type: 'string' } },
    activation: obj({
      mode: { type: 'string' }, switch_name: { type: 'string' }, switch_value: { type: 'string' },
      path_marker: { type: 'string' }, marker_how: { type: 'string' },
      switches: {
        type: 'array',
        items: obj({ switch_name: { type: 'string' }, switch_value: { type: 'string' } }, []),
      },
    }, []),
    correctness: { type: 'string' }, guards_pass: { type: 'boolean' },
    path_marker: { type: 'string' }, path_marker_count: { type: 'number' },
    world_size: { type: 'number' },
    launches: { type: 'number' }, launches_base: { type: 'number' },
    graph_safe: { type: 'string' }, artifact_distinct: { type: 'string' },
    artifact_hash_base: { type: 'string' }, artifact_hash_candidate: { type: 'string' },
    overlap_measured: { type: 'string' }, overlap_fraction: { type: 'number' },
    overlap_cu_fraction: { type: 'number' },
    overlap_base_reading: { type: 'number' }, overlap_forced_reading: { type: 'number' },
    attribution_complete: { type: 'boolean' },
    attribution: obj({
      changed_us: { type: 'number' }, replaced_sum_us: { type: 'number' },
      residual_ms_base: { type: 'number' }, residual_ms_cand: { type: 'number' },
      guard: { type: 'string' }, method: { type: 'string' },
      absolute_to_frozen: { type: 'boolean' },
    }, []),
    accuracy_metric: { type: 'string' }, accuracy_value: { type: 'number' },
    accuracy_results: { type: 'array', items: obj({
      metric: { type: 'string' }, value: { type: 'number' },
      threshold: { type: 'number' }, guard: { type: 'string' },
      method: { type: 'string' },
    }, ['metric', 'value', 'guard']) },
    liveness_replays: { type: 'number' }, claim_complete: { type: 'boolean' },
    replay_results: { type: 'array', items: obj({
      guard: { type: 'string' }, count: { type: 'number' },
      status: { type: 'string' }, graph_safe: { type: 'string' },
      arrival_jitter: { type: 'boolean' },
    }, ['guard', 'count', 'status']) },
    paired_readings: { type: 'array', items: obj({
      guard: { type: 'string' }, base: { type: 'number' }, cand: { type: 'number' },
    }, ['guard', 'base', 'cand']) },
    null_arm_pct: { type: 'number' },
    per_case: perCase,
    reason: { type: 'string' },
  }, ['candidate_id', 'claim_complete']) },
  notes: { type: 'string' },
}, ['claim_complete', 'selected_candidate_id', 'selected_tree']);

const MEGA_IDENTITY_SCHEMA = obj({
  passed: { type: 'boolean' },
  selected_candidate_id: { type: 'string' },
  source_head: { type: 'string' },
  materialized_head: { type: 'string' },
  note: { type: 'string' },
}, ['passed', 'selected_candidate_id', 'source_head', 'materialized_head']);

const REPORT_SCHEMA = obj({
  final_speedup_geomean: { type: 'number' }, final_speedup_arithmetic: { type: 'number' },
  final_speedup_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  rounds: { type: 'number' }, budget_used: { type: 'number' },
  report_path: { type: 'string' }, final_patch: { type: 'string' }, per_case: perCase,
}, ['final_speedup_geomean', 'report_path', 'final_patch']);

const VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' }, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  autonomy_acceptance_confirmed: { type: 'boolean' },
  workflow_unchanged: { type: 'boolean' },
  workflow_revision_end: { type: 'string' },
  materialization_matches: { type: 'boolean' },
  selected_source_head: { type: 'string' },
  selected_materialized_head: { type: 'string' },
  final_patch_verified: { type: 'boolean' },
  attribution: obj({
    changed_us: { type: 'number' }, replaced_sum_us: { type: 'number' },
    guard: { type: 'string' }, residual_ms_base: { type: 'number' },
    residual_ms_cand: { type: 'number' }, method: { type: 'string' },
    absolute_to_frozen: { type: 'boolean' },
  }, []),
  per_case: perCase, applied_to_original: { type: 'string' },
  // Count of GPU-lease jobs still alive for this EVAL_DIR when validation finished. Non-zero means
  // some direction backgrounded a lease and produced a measurement that is NOT in the report.
  orphan_leases_swept: { type: 'number' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status', 'correctness']);

// Prompt helpers. Agents read their role file and knowledge themselves; the script passes paths and
// JSON inputs.
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// Bound hung calls and retry transient API failures. Timeout recovery is phase-owned.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
async function agentT(p, o) {
  const label = (o && o.label) ? o.label : 'agent';
  const timeoutMs = Number(o && o.timeout_ms) > 0 ? Number(o.timeout_ms) : AGENT_TIMEOUT_MS;
  const timeoutMarker = !!(o && o.timeout_marker);
  const callRetries = Number(o && o.max_retries) > 0
    ? Math.max(1, Number(o.max_retries)) : AGENT_RETRIES;
  const callOpts = o && (Object.prototype.hasOwnProperty.call(o, 'timeout_ms') ||
      Object.prototype.hasOwnProperty.call(o, 'timeout_marker') ||
      Object.prototype.hasOwnProperty.call(o, 'max_retries'))
    ? Object.fromEntries(Object.entries(o).filter(([key]) =>
      key !== 'timeout_ms' && key !== 'timeout_marker' && key !== 'max_retries')) : o;
  for (let attempt = 1; attempt <= callRetries; attempt++) {
    try {
      if (typeof setTimeout !== 'function' || !(timeoutMs > 0)) return await agent(p, callOpts);
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(timeoutMs / 60000)}min with no return — ` +
            `${timeoutMarker ? 'returning a timeout marker so Mega stops safely' : 'resolving null so the round proceeds'}.`);
          resolve(timeoutMarker ? { __agent_timed_out: true, label } : null);
        }, timeoutMs);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      return await Promise.race([
        agent(p, callOpts).then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < callRetries) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${callRetries} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${callRetries} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}

// Add Skill context only to roles that consume it.
function expertSkillsBlock(role, phase) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  if (MODE === 'mega' && EXPERT_SKILL_DIR) {
    if (!['mega_search_lead', 'engineer', 'deep_engineer'].includes(role)) return '';
    const guide = role === 'mega_search_lead'
      ? (phase === 'analyze' ? EXPERT_SKILL_PLANNER_EXTENSION_FILE
        : EXPERT_SKILL_PLANNER_GUIDE_FILE)
      : EXPERT_SKILL_AUTHOR_GUIDE_FILE;
    return `\n\n## MEGA EXPERT SKILL — ${
      EXPERT_SKILL_VALIDATION_STATUS === 'experimental'
        ? 'EXPERIMENTAL TRANSFER, EXPLICIT PIN' : 'VALIDATED PRIOR'
    }\n` +
      `Read ${guide}. ${
        role === 'mega_search_lead' && phase === 'analyze'
          ? 'Use its canonical Planner Extension to derive MegaPlanIR.'
          : 'Do not read or restate the full Skill contract; independent Verify owns it.'
      } ` +
      `The Skill is one validated solution for its stated size range (its skill.md "When to use"): ` +
      `inside that range treat it as a seed recipe — reproduce it first, then search beyond it; ` +
      `outside the range use it only for mechanisms${MRI.MEGA_REGIMES_GUIDE ? ' and let MEGA_REGIMES_GUIDE decide what transfers' : ''}. ` +
      `validation_status=${
        EXPERT_SKILL_VALIDATION_STATUS || 'legacy_validated'}, usage=${
        EXPERT_SKILL_USAGE || 'legacy'}. Candidate source remains search/integrated.` +
      (role === 'mega_search_lead' && REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY
        ? ` Before using the Skill, run python ${EXPERT_SKILL_BUNDLE_TOOL} ${EXPERT_SKILL_ID} ` +
          `--emit-bundle and require bundle_sha256=${EXPERT_SKILL_BUNDLE_SHA256}, ` +
          `planner_extension_sha256=${EXPERT_SKILL_PLANNER_EXTENSION_SHA256}, and ` +
          `contract_sha256=${EXPERT_SKILL_CONTRACT_SHA256}. A mismatch is a RunContract failure, ` +
          `not a reason to reinterpret stale knowledge.`
        : '');
  }
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/mega/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills) and whose validation_status is \`validated\` and ` +
    `auto_apply is true, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.` +
    // Capability evaluation may use mechanisms, never reference addresses.
    (CAPABILITY_EVAL
      ? ` **Capability eval: use a skill's MECHANISM and MEASUREMENTS, never its addresses.** Its ` +
        `Sources/Procedure sections may cite a branch, commit, or sibling checkout containing the ` +
        `finished implementation. Do not read, port, copy, or diff against any of them — write the ` +
        `code yourself from the described mechanism. A patch containing a file byte-identical to ` +
        `such a reference is rejected at verify as \`plagiarized\`, whatever it measures.`
      : '');
}

function analysisSkillBlock(role, phase) {
  if (!ANALYSIS_SKILL_ON) return '';
  if (role === 'analysis_engineer') {
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
  if (role === 'mega_search_lead' && phase === 'plan_round') return analysisSkillBlock('tech_lead', 'plan_round');
  return '';
}

function roleAgent(role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/mega/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role, phase) + analysisSkillBlock(role, phase);
}

// PHASE: Setup
phase('Setup');
const setup = await agentT(
  roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC,
    ...(STRICT_AUTONOMY ? { STRICT_AUTONOMY: '1' } : {}),
    ...(STATE_DIR ? { STATE_DIR, MEGA_RESUME_STATE } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA,
    // 5min cut the mega director off while reproducing a grown prior_state. Setup draws
    // no MEGA_CLOCK and timeout_ms is not in the agentT cache key; 15min cut an 18-row resume.
    ...(MODE === 'mega' && MEGA_PRODUCTION ? { timeout_ms: 1800000, max_retries: 1 } : {}) });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const KERNEL_NAME = setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);
if (MODE === 'mega' &&
    (setup.mega_workspace_from_original !== true || setup.mega_candidate_state_checked !== true)) {
  throw new Error(
    'MEGA SETUP REFUSED: ordinary workspace must be seeded from KERNEL_PATH_ORIG. Candidate state ' +
    'must be checked and resumes only inside STATE_DIR/candidates; a legacy STATE_DIR/best must never ' +
    'replace the frozen base.');
}
if (STRICT_AUTONOMY && setup.resumed) {
  throw new Error(
    'STRICT AUTONOMY REFUSED: director seeded this run from STATE_DIR/best. A proof run must start ' +
    'from the frozen input tree with an empty/fresh state directory; otherwise inherited code and ' +
    'manual state edits cannot be distinguished from work derived in this run.');
}
if (STRICT_AUTONOMY && (setup.workflow_clean !== true ||
    !String(setup.workflow_revision || '').trim())) {
  throw new Error(
    'STRICT AUTONOMY REFUSED: GEAK workflow/knowledge must be a clean, committed revision before ' +
    'launch. Mid-wave or uncommitted instructions make the run non-reproducible.');
}
const WORKFLOW_REVISION_START = String(setup.workflow_revision || '');
if (STRICT_AUTONOMY) {
  const roots = new Set(Array.isArray(CONTAINMENT_PREFLIGHT.roots)
    ? CONTAINMENT_PREFLIGHT.roots.map((r) => String(r).replace(/\/+$/, '')) : []);
  const expectedRoots = [KERNEL_PATH_ORIG, EXP_ROOT, STATE_DIR, WORKFLOW_DIR]
    .filter(Boolean).map((r) => String(r).replace(/\/+$/, ''));
  const missingRoots = expectedRoots.filter((r) => !roots.has(r));
  if (CONTAINMENT_PREFLIGHT.content_scan !== 'clean' ||
      CONTAINMENT_PREFLIGHT.skill_address_scan !== 'clean' ||
      String(CONTAINMENT_PREFLIGHT.workflow_revision || '') !== WORKFLOW_REVISION_START ||
      Number(CONTAINMENT_PREFLIGHT.reference_hash_count || 0) !== KNOWN_REFERENCE_HASHES.length ||
      missingRoots.length) {
    throw new Error(
      `STRICT AUTONOMY REFUSED: containment attestation is not bound to this workflow/roots ` +
      `(missing roots: ${missingRoots.join(', ') || 'none'}).`);
  }
}

// Containment enforcement after Setup (the startup check only filters paths): run the two content
// scanners over the materialised eval dir, baseline copy and inherited assets. It is an agent because
// the workflow script has no shell.
if (STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean === true) {
  log(`CONTAINMENT PREFLIGHT clean: trusted bootstrap scanned the run tree before agent startup ` +
      `(marker manifest ${String(CONTAINMENT_PREFLIGHT.marker_manifest_sha256 || '').slice(0, 12)}…, ` +
      `${Number(CONTAINMENT_PREFLIGHT.reference_hash_count || 0)} opaque reference hashes).`);
}
if (CAPABILITY_EVAL && RUN_TREE_ANCESTOR &&
    !(STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean === true)) {
  const gate = await agentT(
    `Run two containment scanners and report their exit codes verbatim. Do not fix anything, do not ` +
    `read any file they flag, do not summarise the flagged content — reporting the paths is the whole job.\n\n` +
    // Scan every ref tree (the default cap of 200 makes the hit count a lower bound). The full sweep
    // costs about five minutes once, before any lease.
    `1. REF_SCAN_MAX_TREES=100000 bash ${WORKFLOW_DIR}/scripts/reference_leak_sweep.sh ` +
    `--tree ${RUN_TREE_ANCESTOR} --markers ${REFERENCE_LEAK_MARKER_FILE || '(missing-marker-file)'}\n` +
    `   If its output still contains a REF_SCAN_MAX_TREES coverage NOTE, the scan is PARTIAL: say so\n` +
    `   in your note and return verdict "unknown" rather than clean, whatever the exit code was.\n` +
    `2. bash ${WORKFLOW_DIR}/scripts/skill_address_scan.sh --skills-dir ${EXPERT_SKILLS_DIR || '(skip: expert skills off)'} ` +
    `--scan-root ${RUN_TREE_ANCESTOR}\n\n` +
    `If a script is missing or its --skills-dir is empty, record that step as "skipped" with the reason ` +
    `and do NOT call it clean. "The scanner did not run" and "the scanner found nothing" are different ` +
    `results and this run has already been damaged twice by conflating them.`,
    { phase: 'Setup', label: 'containment gate', schema: CONTAINMENT_GATE_SCHEMA });
  if (!gate || gate.verdict !== 'clean') {
    const verdict = gate ? gate.verdict : 'unknown';
    if (STRICT_AUTONOMY || verdict === 'leak') {
      throw new Error(
        `CONTAINMENT GATE FAILED (${verdict}): strict/capability evidence requires an explicit clean ` +
        `scan of ${RUN_TREE_ANCESTOR} before budget is spent.\n` +
        `${gate ? (gate.findings || []).join('\n') : 'gate agent returned nothing'}\n` +
        `${gate ? gate.note || '' : ''}`);
    }
    log(`CONTAINMENT GATE ${verdict}: scanner did not establish clean. ${gate ? gate.note || '' : ''}`);
  } else log(`CONTAINMENT GATE clean: ${gate.note || ''}`);
}

// Pool sample: the script reads the GPU pool itself each round and hands the lead the result, so a
// round is never planned for idle cards that an external tenant holds. It does not abort the run; an
// occupied pool redirects the round to lease-free work. Sampled per round because tenants come and go.
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
        `  Plan this round GPU-LESS. A GPU direction dispatched now will spend its clock inside flock ` +
        `and return nothing; an instrument authored now, with its driver, is a partial the next round ` +
        `can run in one lease.`);
  }
  return p;
}

async function runProfileAnalysis(profileSummary, round, label) {
  if (!ANALYSIS_SKILL_ON || !profileSummary) return null;
  const result = await agentT(
    roleAgent(
      'analysis_engineer',
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
      ...(MODE === 'mega' && MEGA_PRODUCTION ? { timeout_ms: 300000, max_retries: 1 } : {}),
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

// Require the Director's frozen online baseline; self-timing kernel_src would create a fake win.
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

// PHASE: Author (mode=author only): write a fresh seed implementation; the rest of the pipeline runs
// unchanged on it. The speedup denominator is always the frozen online kernel, never the seed. No
// correct seed: abort with a structured result.
if (MODE === 'author') {
  phase('Author');
  const authored = await agentT(
    roleAgent('author_engineer', 'author', 'Write the simplest correct baseline in the target language.', {
      TARGET_LANGUAGE, OP_SPEC, WORKSPACE: CANONICAL, TASK_DIR: KERNEL_PATH_ORIG,
      GPU_ID: GPU_RESOURCE.specForIndex(0), SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_KNOWLEDGE_DIR,
      GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
    }),
    { phase: 'Author', label: `author:${TARGET_LANGUAGE}`, schema: AUTHOR_SCHEMA });
  if (!authored || !authored.authored || authored.correctness !== 'pass') {
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

// Mega candidate registry (no blocking pre-loop reproduction phase).

// One candidate portfolio and lifecycle, with or without Expert Skills.
const MEGA_CANDIDATE_STATES = ['authoring', 'runnable', 'scored', 'finalist', 'rejected', 'parked'];
let megaCandidateRegistry = [];
let megaStateSequenceBase = 0;
let megaUnsafeTimeout = null;
// Consecutive rounds without new runtime evidence (independent Verify, a better score, or an
// on-card Author result that changed the blocker). Drives measure-first and the stall stop.
let megaStall = 0;
let megaLastBlocker = '';
// A refused plan re-plans with the reason instead of ending the wave; 3 in a row still stops.
let megaRefusal = null;
let megaPcRemeasured = 0;
let megaMeasurementCalibration = {
  ready: false, candidate_id: '', attempt_id: '', evidence_manifest: '', note: 'not measured',
};
if (MODE === 'mega') {
  log(`MEGA unified mode: Expert Skills are ${USE_EXPERT_SKILLS ? 'enabled as advisory context' :
    'disabled'}; all candidates use the same autonomous lifecycle.`);
}

// PHASE: Analyze + Roadmap (TechLead)
phase('Analyze');
let analysis = await agentT(
  roleAgent(MODE === 'mega' ? 'mega_search_lead' : 'tech_lead',
    'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR: MODE === 'mega' ? '' : KERNEL_KNOWLEDGE_DIR,
    GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
    ...(A.require_task_graph ? { REQUIRE_TASK_GRAPH: '1' } : {}),
    ...(MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE ? {
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
    ...(CAPABILITY_EVAL ? { CAPABILITY_EVAL: '1' } : {}),
    ...(STRICT_AUTONOMY ? {
      STRICT_AUTONOMY: '1', TARGET_GUARDS, REGRESSION_GUARDS,
      PROMOTION_METRIC, LAUNCH_TARGET: Number(A.launch_target || 2),
      REQUIRE_OVERLAP, REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    } : {}),
    ...(STATE_DIR && MODE !== 'mega' ? { STATE_DIR } : {}),
    ...(MODE === 'mega' ? {} : RESUME_INPUT),
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze',
    schema: MODE === 'mega' ? MEGA_ANALYZE_SCHEMA : ANALYZE_SCHEMA,
    ...(MODE === 'mega' && MEGA_PRODUCTION
      // 30 min: the task-graph analysis (four docs + source + DAG) took 13.5 min fresh and was cut
      // AT StructuredOutput at 15 min on a resumed wave. One-time; not drawn from MEGA_CLOCK_MS.
      ? { timeout_ms: 1800000, timeout_marker: true, max_retries: 1 } : {}) });
if (MODE === 'mega' && analysis && analysis.__agent_timed_out) {
  return {
    mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
    final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
    reason: 'Analyze exceeded the production timeout; no second Analyze or GPU phase was started',
  };
}

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
      : ''}. ` +
    'Re-running one full frozen-source analysis before Benchmark.' };
}
// <</REPLAY:analyze_resume_fallback>>

{
  const d = analyzeResumeDegenerate(
    INCREMENTAL, analysis, MODE === 'mega' && !!A.require_task_graph, MODE === 'mega',
    MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
      ? EXPERT_SKILL_REVISION : '',
    MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
      ? EXPERT_SKILL_ID : '',
    MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
      ? EXPERT_SKILL_PLANNER_EXTENSION_SHA256 : '',
    MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
      ? EXPERT_SKILL_BUNDLE_SHA256 : '',
    MODE === 'mega' && USE_EXPERT_SKILLS ? EXPERT_SKILL_VALIDATION_STATUS : '',
    MODE === 'mega' && USE_EXPERT_SKILLS ? EXPERT_SKILL_USAGE : '');
  if (d.retry) {
    log(d.reason);
    // Identical to the call above except that RESUME_INPUT is absent — that omission IS the fix.
    const full = await agentT(
      roleAgent(MODE === 'mega' ? 'mega_search_lead' : 'tech_lead',
        'analyze', 'Analyze the kernel and write the roadmap.', {
        WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
        KERNEL_KNOWLEDGE_DIR: MODE === 'mega' ? '' : KERNEL_KNOWLEDGE_DIR,
        GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
        ...(A.require_task_graph ? { REQUIRE_TASK_GRAPH: '1' } : {}),
        ...(MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE ? {
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
        ...(CAPABILITY_EVAL ? { CAPABILITY_EVAL: '1' } : {}),
        ...(STRICT_AUTONOMY ? {
          STRICT_AUTONOMY: '1', TARGET_GUARDS, REGRESSION_GUARDS,
          PROMOTION_METRIC, LAUNCH_TARGET: Number(A.launch_target || 2),
          REQUIRE_OVERLAP, REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        } : {}),
        // Kept on the recovery call too: the full analysis re-derives the ladder from the source,
        // but the rungs a prior wave already spent, and the ones it left owed, exist only here.
        ...(STATE_DIR && MODE !== 'mega' ? { STATE_DIR } : {}),
      }),
      { phase: 'Analyze', label: 'tech_lead:analyze:full',
        schema: MODE === 'mega' ? MEGA_ANALYZE_SCHEMA : ANALYZE_SCHEMA,
        ...(MODE === 'mega' && MEGA_PRODUCTION
          // 900s: same rationale as the primary analyze call site above.
          ? { timeout_ms: 900000, timeout_marker: true, max_retries: 1 } : {}) });
    if (MODE === 'mega' && full && full.__agent_timed_out) {
      return {
        mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
        final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
        reason: 'full Analyze fallback exceeded the production timeout; no GPU phase was started',
      };
    }
    const got = (full && Array.isArray(full.candidate_directions) ? full.candidate_directions : [])
      .filter((c) => c && (c.id || c.title));
    const fullContract = analyzeResumeDegenerate(false, full,
      MODE === 'mega' && !!A.require_task_graph, MODE === 'mega',
      MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
        ? EXPERT_SKILL_REVISION : '',
      MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
        ? EXPERT_SKILL_ID : '',
      MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
        ? EXPERT_SKILL_PLANNER_EXTENSION_SHA256 : '',
      MODE === 'mega' && USE_EXPERT_SKILLS && EXPERT_SKILL_PLANNER_EXTENSION_FILE
        ? EXPERT_SKILL_BUNDLE_SHA256 : '',
      MODE === 'mega' && USE_EXPERT_SKILLS ? EXPERT_SKILL_VALIDATION_STATUS : '',
      MODE === 'mega' && USE_EXPERT_SKILLS ? EXPERT_SKILL_USAGE : '');
    if (got.length && !fullContract.retry) {
      log(`ANALYZE RE-RUN recovered a ladder of ${got.length} rung(s). Using the full analysis.`);
      analysis = full;
    } else {
      log(`ANALYZE RE-RUN still missed its structured contract: ${fullContract.reason}`);
      if (full) analysis = full;
      if (MODE === 'mega') {
        return {
          mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
          final_speedup: 0, final_geomean: 0,
          validation_status: 'analyze_contract_incomplete',
          reason: fullContract.reason,
        };
      }
    }
  }
}
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
if (MODE === 'mega') analysis = applyBaselineOperatorMap(analysis);
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

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
      // 1 - cp/e2e is the share of e2e the dependence chain does not explain, not a ceiling on gains: on a
      // throughput-bound operator the gap is arithmetic, which overlap cannot remove.
      ? `critical_path=${cp.toFixed(1)}µs vs e2e=${e2e.toFixed(1)}µs ` +
        `(dependence explains ${(100 * (cp / e2e)).toFixed(1)}% of e2e; the remaining ` +
        `${(100 * (1 - cp / e2e)).toFixed(1)}% is resource-bound work plus bubbles, and only the ` +
        'bubble set is addressable by reordering)'
      : 'critical path NOT quantified');

  if (quantified && assumed > 0) {
    return { verdict: 'UNMEASURED_NUMERIC', summary, caveat:
      `TASK GRAPH USES NUMERIC PLACEHOLDERS: ${assumed} node duration(s) are marked assumed while ` +
      `critical_path_us/measured_e2e_us are numeric. Before Benchmark these numbers are not ` +
      'measurements and must not rank directions; preserve them as unknown/null until current-run data exists.' };
  }
  if (quantified && cp > e2e) {
    return { verdict: 'INCONSISTENT', summary, caveat:
      `TASK GRAPH INCONSISTENT: critical_path_us (${cp}) exceeds measured_e2e_us (${e2e}). The ` +
      'longest path through the graph cannot be longer than the thing it is a path through, so ' +
      'either a node duration is wrong or an edge is spurious. Do not use this graph\'s headroom ' +
      'number to size any proposal.' };
  }
  // Every edge a launch boundary and none narrower means the scope column was not filled in; the graph
  // cannot then tell a required barrier from an incidental one. Advisory.
  const narrow = eArr.filter((e) => e && e.scope && e.scope !== 'cross_rank' && e.scope !== 'hbm').length;
  if (edges && launchEdges === edges && narrow === 0) {
    return { verdict: 'KERNEL_GRANULARITY', summary, caveat:
      `TASK GRAPH IS KERNEL-GRANULAR: all ${edges} edges came back enforced_by=launch_boundary with ` +
      'no edge at a narrower scope, which is what a graph whose nodes are KERNELS looks like. Such ' +
      'a graph restates the launch order and cannot separate an ordering the data requires from one ' +
      'the current code imposes. Treat any fusion direction ranked from it as unsupported.' };
  }
  // WHAT THE CONSUMER NEEDS that the producer does not supply (weights, scales, descriptors): graph nodes
  // are output tiles, so these never appear as edges. Ask per edge; [] is a valid answer. Advisory:
  // missing answers make prefetch unrankable, not the graph invalid.
  const fanInEdges = eArr.filter((e) => e && Number(e.fan_in) > 0);
  const unanswered = fanInEdges
    .filter((e) => !Array.isArray(e.producer_independent_operands))
    .map((e) => `${e.from || '?'}->${e.to || '?'}`);
  if (unanswered.length) {
    return { verdict: 'OK', summary, caveat:
      `EDGES THAT DID NOT SAY WHAT THE CONSUMER LOADS ANYWAY (${unanswered.length}): ` +
      `${unanswered.join(', ')}. Each names an operand the consumer waits for; none names the ` +
      'operands it needs that the producer does not supply — weights, scales, descriptors, index ' +
      'arrays. Those depend on nothing and can be loaded during the producer\'s own execution, ' +
      'which is a different and usually cheaper lever than moving the dependent operand earlier. ' +
      'It cannot be ranked from this graph because it has no node here.' };
  }
  return { verdict: 'OK', summary, caveat: '' };
}
// <</REPLAY:task_graph_gate>>

// <<REPLAY:pipe_occupancy_gate>>
// Pipe table gate, separate from taskGraphGate: the graph says what may overlap, this says whether the
// pipe is even busy. It prints the table, derives the class from the numbers, and names directions
// that claim more than their pipe's idle fraction.
const PIPE_SATURATED_PCT = 80;   // at or above this, the pipe is the constraint
const PIPE_LOW_PCT = 50;         // below this for EVERY pipe, nothing is throughput-bound
const GAP_NEGLIGIBLE_US = 0.5;   // inter-kernel gap below this cannot fund a launch-count direction

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
        'artifact names nothing that could be issued into them. That is the one question this ' +
        'analysis existed to answer; without it the directions below cannot have been derived from it.');
    }
  }
  // Directions priced above what their own pipe can pay. Checked here rather than trusted, because
  // expected_speedup is the field most likely to be a wish.
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
    notes.push(`DIRECTIONS WITH NO PIPE NAMED (${unpriced.length}): ${unpriced.join(', ')}. A ` +
      'direction that cannot say which functional unit it fills has not been derived from the ' +
      'table and its expected_speedup is unbounded by anything. Treat as hypotheses.');
  }
  if (overclaimed.length) {
    notes.push('DIRECTION OVERCLAIMS ITS PIPE: ' + overclaimed.join('; ') +
      '. These are arithmetically impossible against the reported utilization and can be rejected ' +
      'without a run; if the utilization is wrong, that is the finding.');
  }
  // A sized hole no direction claims. Checked only once directions exist; rides_on (another rung
  // absorbs it) is a valid answer, silence is not.
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
      notes.push(`SIZED HOLE WITH NO DIRECTION (${unowned.length}): ${unowned.join('; ')}. Analyze ` +
        'located these, measured them and put a number on them, and no direction in this round ' +
        'names one as the thing it fills. A hole nobody claims is not a hole nobody can reach — it ' +
        'is a hole nobody looked at. Either give it a direction, or set `rides_on` to the rung that ' +
        'will absorb it so the claim is recorded and can be checked when that rung reports.');
    }
  }
  const verdict = notes.length ? (overclaimed.length || (rt.class && rt.class !== derived) ? 'INCONSISTENT' : 'ADVISORY') : 'OK';
  return { verdict, summary, class_derived: derived, caveat: notes.join(' ') };
}
// <</REPLAY:pipe_occupancy_gate>>

// <<REPLAY:roadmap_ladder_gate>>
// Did the round plan against Analyze's ladder? Diffs dispatched rungs against it, exposing skipped
// controls and never-proposed fusion rungs.
//
// <<REPLAY:open_rungs>>
// A rung is done when it produced a number, not when it was planned. Outcomes per rung:
//   never_planned  no round has taken it
//   faulted        a direction took it and the code did not run
//   unmeasured     it ran but returned no verified number
//   measured       a verified number exists; only this closes the rung
const RUNG_OUTCOMES = ['never_planned', 'faulted', 'unmeasured', 'measured'];
function rungIdOf(c) {
  return String((c && c.id) || String((c && c.title) || '').trim().split(/\s+/)[0] || '').trim();
}
// Grade one round's result for its rung. Enabling rungs close on functional acceptance; terminal
// rungs close on a measurement, including a losing one.
function rungOutcomeOf(res, role) {
  if (!res) return 'unmeasured';
  const engFailed = !res.eng || /fail|error|crash/i.test(String(res.eng.status || ''));
  if (engFailed) return 'faulted';
  if (res.inactive || res.unbacked || res.same_artifact) return 'unmeasured';
  if (role === 'enabling') {
    return functionalAcceptance(res.ver).pass ? 'measured' : 'unmeasured';
  }
  return verifyCompleted(res.ver) && Number(res.ver && res.ver.verified_geomean) > 0
    ? 'measured' : 'unmeasured';
}
// tally: Map<rungId, {attempts, last_outcome}>. Returns the entries the next wave still owes,
// strongest-evidence-last so `measured` can never be reintroduced by a later weaker grade.
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
        // Carried so the next wave knows what closing this rung would even mean. A prerequisite
        // re-planned as a speed experiment is a prerequisite that gets rejected again.
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

function mandatoryArmsVerdict(direction, ver, strict) {
  const expected = Array.isArray(direction && direction.mandatory_arms)
    ? [...new Set(direction.mandatory_arms.map(String).filter(Boolean))] : [];
  if (!expected.length) return { pass: true, expected, actual: [], missing: [] };
  const actual = Array.isArray(ver && ver.arms_run)
    ? [...new Set(ver.arms_run.map(String).filter(Boolean))] : [];
  const have = new Set(actual);
  const missing = expected.filter((a) => !have.has(a));
  return {
    pass: missing.length === 0 && (!strict || actual.length > 0),
    expected, actual, missing,
    caveat: missing.length
      ? `mandatory arm(s) not independently verified: ${missing.join(', ')}`
      : '',
  };
}

function shouldVerifyDirection(direction, eng, forceVerify, score) {
  if (!eng || /fail|error|crash/i.test(String(eng.status || ''))) return false;
  if (forceVerify || String((direction && direction.step_role) || '').toLowerCase() === 'enabling') {
    return true;
  }
  return Number(score) > 1.0;
}

function strictDirectionVerdict(direction, ladder, completedRungs) {
  const d = direction || {};
  const rungId = String(d.roadmap_rung || '').trim();
  if (!rungId || rungId === 'off_ladder') {
    return { pass: false, reason: 'strict autonomy direction must name a ladder rung' };
  }
  const idOf = (c) => String((c && (c.id || c.title)) || '').trim().split(/\s+/)[0];
  const rung = (Array.isArray(ladder) ? ladder : []).find((c) => idOf(c) === rungId);
  if (!rung) return { pass: false, reason: `unknown roadmap rung ${rungId}` };
  const done = new Set(Array.isArray(completedRungs) ? completedRungs
    : (completedRungs instanceof Set ? [...completedRungs] : []));
  const unmet = (Array.isArray(rung.gated_on) ? rung.gated_on : []).filter((g) => !done.has(String(g)));
  if (unmet.length) {
    return { pass: false, reason: `${rungId} prerequisite(s) not completed to spec: ${unmet.join(', ')}` };
  }
  if (String(d.step_role || '') === 'enabling' && !String(d.enables || '').trim()) {
    return { pass: false, reason: `${rungId} is enabling but names no terminal rung` };
  }
  const rungRole = String(rung.step_role || 'terminal');
  if (String(d.step_role || 'terminal') !== rungRole ||
      (rungRole === 'enabling' && String(d.enables || '') !== String(rung.enables || ''))) {
    return { pass: false, reason: `${rungId} changed its Analyze step_role/enables contract` };
  }
  const expected = Array.isArray(rung.mandatory_arms) ? rung.mandatory_arms : [];
  const supplied = new Set(Array.isArray(d.mandatory_arms) ? d.mandatory_arms.map(String) : []);
  const missing = expected.filter((a) => !supplied.has(String(a)));
  if (missing.length) {
    return { pass: false, reason: `${rungId} dropped mandatory arm(s): ${missing.join(', ')}` };
  }
  if (rung.target_shape) {
    const planned = d.target_shape || {};
    const expectedStages = [...new Set((rung.target_shape.stages_fused || []).map(String))].sort();
    const plannedStages = [...new Set((planned.stages_fused || []).map(String))].sort();
    if (Number(planned.launches) !== Number(rung.target_shape.launches) ||
        (!!rung.target_shape.require_overlap && planned.require_overlap !== true) ||
        expectedStages.join('\0') !== plannedStages.join('\0')) {
      return { pass: false, reason: `${rungId} changed its Analyze target_shape` };
    }
  }
  return { pass: true, reason: '' };
}
// <</REPLAY:direction_contract>>

// <<REPLAY:autonomy_transition>>
function winnerCommitTransition(requested, commitResult, winner, winnerResult, strict) {
  const strictProof = !strict || (commitResult && commitResult.transaction_valid === true &&
    commitResult.exact_patch === true && String(commitResult.head_before || '').trim() &&
    String(commitResult.head_after || '').trim() &&
    commitResult.head_before !== commitResult.head_after);
  const committed = !!(requested && commitResult && commitResult.committed === true && winner && strictProof);
  const acceptanceReached = !!(committed && winnerResult && winnerResult.autonomy_contract &&
    winnerResult.autonomy_contract.accepted === true);
  return { committed, acceptanceReached };
}

function landedRungOutcome(baseOutcome, role, hasTargetShape, evidencePass,
                           winnerCommitted, winnerMatches, enablingLanded) {
  if (role === 'enabling') return enablingLanded ? baseOutcome : 'unmeasured';
  if (hasTargetShape) {
    return evidencePass && winnerCommitted && winnerMatches ? baseOutcome : 'unmeasured';
  }
  return baseOutcome;
}

function finalStatusVerdict(directorStatus, scopedGuardFailure, strictProofFailure) {
  if (directorStatus === 'flagged' || scopedGuardFailure) return 'flagged';
  if (strictProofFailure) return 'autonomy_incomplete';
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

function integrationEligible(integrate, score, bestIndividual, guards) {
  return !!(integrate && integrate.conclusion === 'improved' && integrate.best &&
    guards && guards.complete && guards.regression_pass && Number(score) > Number(bestIndividual));
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
      'for this run and nothing below can be checked against a plan. Each round is then free to ' +
      'start from the profile, which is exactly the state in which the most expensive direction — ' +
      'the one that is conditional on two cheaper ones — is never reached, because nothing is ' +
      'tracking that it was owed.' };
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
    notes.push(`DIRECTIONS WITH NO RUNG NAMED (${undeclared.length}): ${undeclared.join(', ')}. ` +
      'A direction that cannot say which rung it is — including "off_ladder" — has not been ' +
      'planned against the ladder, and the ladder cannot then be used to tell what is still owed.');
  }
  if (offLadder.length) {
    notes.push(`OFF-LADDER WITHOUT A STATED DEVIATION: ${offLadder.join('; ')}. Leaving the ladder ` +
      'is allowed and often right; leaving it silently means the rung it displaced is not recorded ' +
      'anywhere as still open.');
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

// <<REPLAY:evidence_stop>>
// What the no-improve counter may count: only rounds that produced admissible evidence. Not counted:
//   inactive       the patched path did not run, or was not proven to run
//   unbacked       a claim whose declared patch does not exist
//   same_artifact  both arms resolved to one compiled binary
//   outcome        not `measured` per rungOutcomeOf
const evidenceGap = (r, roleOf, outcomeOf) => {
  if (!r) return 'no result was returned';
  if (r.inactive) return `the patched path was ${r.inactive === 'no' ? 'not executed' : 'not proven to execute'}`;
  if (r.unbacked) return 'the claim has no patch behind it';
  if (r.same_artifact) return 'both arms were the same compiled binary';
  const outcome = outcomeOf(r, roleOf(r.d));
  return outcome === 'measured' ? null : `the direction ${outcome === 'faulted' ? 'faulted' : 'returned no verified number'}`;
};
function roundEvidence(clean, roleOf, outcomeOf) {
  const arr = Array.isArray(clean) ? clean : [];
  let measured = 0;
  const gaps = [];
  for (const r of arr) {
    const gap = evidenceGap(r, roleOf, outcomeOf);
    if (gap === null) measured++;
    else gaps.push(`${(r && r.d && r.d.id) || '?'}: ${gap}`);
  }
  return { measured, total: arr.length, gaps };
}
// <</REPLAY:evidence_stop>>

// <<REPLAY:claim_boundary>>
function claimBoundary(speedupOf, requireComplete) {
  const usable = (c) => !!c && (!requireComplete || c.claim_complete === true) &&
    Array.isArray(c.per_case) && c.per_case.length > 0;
  return {
    needsRecovery: (eng) => !usable(eng),
    recovered: (onDisk) => usable(onDisk),
    unbacked: (r) => !!r && !!r.ver && r.ver.status === 'apply_failed' && speedupOf(r.eng) > 1.0,
  };
}
// <</REPLAY:claim_boundary>>

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
    // Global insights are free-form and have no reliable source label. Rebuild the search board from
    // candidate records that belong to the common lifecycle.
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
      // Arm-blind state conditioning is valid only when BOTH arms draw both states. Otherwise the
      // gap may simply be a large real effect separating base from candidate.
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
      if (String(r.head || '') !== registered.head) fail.push('head mismatch');
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
          replay.arrival_jitter !== true) fail.push(`replay ${guard}`);
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
  if (source && row && String(row.head || '') !== source.head) reasons.push('finalist row head mismatches registry');
  if (String(s.selected_tree || '') !== String(o.selectedWorkspace || '')) {
    reasons.push('selected tree is not the controlled materialized workspace');
  }
  if (!String(s.selected_head || '') ||
      !source || String(s.materialized_from_head || '') !== source.head) {
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

// The absolute score is frozen-relative. A paired base arm that drifted from the frozen latency (e.g. a
// candidate-tree switch-off path that regressed shared code) is re-scored against the frozen row.
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
      String(v.candidate_head || '') === String(meta && meta.head || ''),
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

// <<REPLAY:objective_gate>>
// objective=working_kernel may skip the positive control, which makes every timing from that wave
// void (1.000x included). pcRan says whether the control ran in this wave.
function objectiveVerdict(objective, pcRan, geomean) {
  if (objective !== 'working_kernel') return { state: 'scored', caveat: '' };
  if (pcRan) return { state: 'scored', caveat: '' };
  if (!Number.isFinite(Number(geomean))) return { state: 'no_number', caveat: '' };
  return {
    state: 'void_no_control',
    caveat: 'VOID AS A TIMING RESULT: this wave ran with objective=working_kernel and no positive ' +
      'control, so nothing established that its harness can resolve an effect at all. The reading ' +
      'is withdrawn rather than reported with a warning, 1.000x included — an uncontrolled 1.000x ' +
      'is what a blind harness emits. The artifact is still judged on whether it RUNS. To obtain a ' +
      'timing result, re-measure it in a wave that pays for the control.',
  };
}
// A round's candidate is committed on running, not on being faster. liveness n/a is accepted for
// non-distributed candidates; fail (timeout) is a failure.
function runsCleanly(ver, requirements) {
  return functionalAcceptance(ver, requirements).pass;
}
// <</REPLAY:objective_gate>>

// <<REPLAY:enabling_step>>
// Enabling vs terminal steps. terminal: judged on speed with the full paired protocol. enabling: a
// prerequisite judged on function, its timing recorded as cost and committed so the next step can
// build on it. It must name the terminal rung it enables and stay within cost_budget_pct; the cost is
// debt until the terminal step pays it, against a baseline pinned at the chain's first enabling commit.
const ENABLING_DEFAULT_BUDGET_PCT = 5.0;   // a prerequisite that costs more than this is a design error
const CHAIN_DEBT_MAX_ROUNDS = 2;           // rounds an unpaid chain may stay open before it is called out

const round2 = (x) => Math.round(Number(x) * 100) / 100;

function stepRoleOf(d) {
  const s = String((d && d.step_role) || '').trim().toLowerCase();
  return s === 'enabling' ? 'enabling' : 'terminal';
}

// Four functional conditions, reported by name. verifyCompleted is shared by the enabling and
// terminal paths so they agree: a correct-but-slower terminal fusion (status regression) counts as
// completed. Excluded: apply_failed, plagiarized, harness_modified.
function verifyCompleted(ver) {
  if (!ver) return false;
  const s = String(ver.status == null ? '' : ver.status).toLowerCase();
  return ['verified', 'regression', 'slower'].includes(s) || s.startsWith('verif');
}

function functionalAcceptance(ver, requirements) {
  const req = requirements || {};
  const s = (v) => String(v == null ? '' : v).toLowerCase();
  const missing = [];
  if (!ver) return { pass: false, missing: ['no verify result at all'] };
  if (!verifyCompleted(ver)) {
    missing.push(`verify did not complete (status=${ver.status || 'none'})`);
  }
  if (!s(ver.correctness).startsWith('pass')) missing.push('correctness did not pass');
  // When accuracy is reported, hold it to the threshold numerically. Without it, the correctness
  // string is used, except under strict autonomy, where absence fails.
  const acc = (ver && ver.accuracy) || {};
  const accRows = Array.isArray(ver && ver.accuracy_results) && ver.accuracy_results.length
    ? ver.accuracy_results : (Object.keys(acc).length ? [acc] : []);
  const accuracyGuards = Array.isArray(req.accuracyGuards) ? req.accuracyGuards.map(String) : [];
  const validateAccuracy = (row, guard) => {
    const val = Number(row && row.value), threshold = Number(row && row.threshold);
    if (!row || String(row.metric || '') !== String(req.accuracyMetric || 'relL2')) {
      return `${guard || 'accuracy'} missing metric ${req.accuracyMetric || 'relL2'}`;
    }
    if (!(Number.isFinite(val) && val >= 0 && Number.isFinite(threshold) && threshold > 0 &&
          threshold <= Number(req.accuracyThreshold || threshold) &&
          val < threshold && val < Number(req.accuracyThreshold || threshold) &&
          String(row.method || '').trim())) {
      return `${guard || row.guard || 'accuracy'} has invalid value/threshold/method`;
    }
    return '';
  };
  if (req.requireAccuracy) {
    const guards = accuracyGuards.length ? accuracyGuards : [''];
    for (const guard of guards) {
      const row = guard ? accRows.find((r) => String(r.guard || '') === guard) : accRows[0];
      const gap = validateAccuracy(row, guard);
      if (gap) missing.push(gap);
    }
  } else {
    const accVal = Number(acc.value), accThr = Number(acc.threshold);
    if (Number.isFinite(accVal) && Number.isFinite(accThr) && !(accVal < accThr)) {
      missing.push(`accuracy ${acc.metric || 'error'}=${accVal} is not < threshold ${accThr}`);
    }
  }
  if (s(ver.activation_confirmed) !== 'yes') missing.push('the new path was not confirmed to run');
  // With requireHardwareActivation, the switched path must be shown to execute on the device this
  // round; absent/no/unknown is void, not negative.
  if (req.requireHardwareActivation && s(ver.activation_on_hardware) !== 'yes') {
    missing.push('the switched path was not confirmed to run ON HARDWARE this round (a compile-only / ' +
      'static-ISA / CPU-dry-run proof does not exercise it; require a gpu_lock device run, with the ' +
      `switch set, that reached the candidate — got activation_on_hardware=${ver.activation_on_hardware || 'none'})`);
  }
  if (req.requireLiveness) {
    if (s(ver.liveness) !== 'pass') missing.push('liveness pass is required and was not reported');
    const requiredReplays = Number(req.requiredReplays || 0);
    const replayGuards = Array.isArray(req.replayGuards) ? req.replayGuards.map(String) : [];
    const replayRows = Array.isArray(ver.replay_results) ? ver.replay_results : [];
    if (replayGuards.length) {
      for (const guard of replayGuards) {
        const row = replayRows.find((r) => String(r.guard || '') === guard);
        if (!row || String(row.status || '').toLowerCase() !== 'pass' ||
            Number(row.count) < requiredReplays ||
            (req.requireGraphSafe && String(row.graph_safe || '').toLowerCase() !== 'pass')) {
          missing.push(`${guard} replay evidence is missing or below ${requiredReplays}`);
        }
      }
    } else {
      const replayCount = Number(ver.replay_count);
      if (requiredReplays > 0 && (!Number.isFinite(replayCount) || replayCount < requiredReplays)) {
        missing.push(`replay_count ${Number.isFinite(replayCount) ? replayCount : 'missing'} < required ${requiredReplays}`);
      }
    }
  } else if (!['pass', 'n/a', ''].includes(s(ver.liveness))) {
    missing.push('liveness failed (hang or timeout)');
  }
  // Graph capture/replay safety is a functional requirement; reported-and-bad fails, n/a passes.
  if (req.requireGraphSafe) {
    if (s(ver.graph_safe) !== 'pass') missing.push('graph capture/replay pass is required and was not reported');
  } else if (!['pass', 'n/a', ''].includes(s(ver.graph_safe))) {
    missing.push('graph capture/replay is not safe');
  }
  if (req.requireArtifactDistinct && s(ver.artifact_distinct) !== 'yes') {
    missing.push('JIT artifact distinctness proof is required and was not reported');
  } else if (req.requireArtifactDistinct &&
      (!String(ver.artifact_hash_base || '').trim() ||
       !String(ver.artifact_hash_candidate || '').trim() ||
       String(ver.artifact_hash_base) === String(ver.artifact_hash_candidate))) {
    missing.push('JIT artifact hashes are missing or identical');
  }
  const requiredArms = Array.isArray(req.requiredArms) ? req.requiredArms.map(String).filter(Boolean) : [];
  if (requiredArms.length) {
    const actualArms = new Set(Array.isArray(ver.arms_run) ? ver.arms_run.map(String) : []);
    const missingArms = requiredArms.filter((a) => !actualArms.has(a));
    if (missingArms.length) missing.push(`mandatory arm(s) not independently verified: ${missingArms.join(', ')}`);
  }
  return { pass: missing.length === 0, missing };
}

// The verdict for ONE enabling direction. `geomean` is its measured speedup, which is expected to be
// below 1 and is not a reason to reject.
function enablingVerdict(d, ver, geomean, requirements) {
  const fa = functionalAcceptance(ver, requirements);
  const enables = String((d && d.enables) || '').trim();
  const budget = Number((d && d.cost_budget_pct) != null ? d.cost_budget_pct : ENABLING_DEFAULT_BUDGET_PCT);
  const g = Number(geomean);
  const costPct = Number.isFinite(g) && g > 0 ? round2((1 / g - 1) * 100) : null;
  if (!enables) {
    return { commit: false, cost_pct: costPct, reason:
      'declared step_role=enabling but named no terminal rung in `enables`. A step that is a ' +
      'prerequisite to nothing is not a prerequisite, it is a change that made the operator slower.' };
  }
  if (!fa.pass) {
    return { commit: false, cost_pct: costPct, reason:
      `failed functional acceptance: ${fa.missing.join('; ')}. An enabling step is exempt from the ` +
      'speed bar and from nothing else — these four conditions are the whole bar it is held to.' };
  }
  if (costPct != null && Number.isFinite(budget) && costPct > budget) {
    return { commit: false, cost_pct: costPct, reason:
      `costs ${costPct}%, over its own declared budget of ${budget}%. A prerequisite is allowed to ` +
      'be slower than what it replaces; it is not allowed to cost more than the fusion it enables ' +
      'can return. Over budget is a design error, not a temporary slowdown.' };
  }
  return { commit: true, cost_pct: costPct, enables, reason:
    `functional acceptance passed (builds, path taken, correct, no hang) and it enables ${enables}. ` +
    `Committed on FUNCTION at ${costPct == null ? 'an unread cost' : costPct + '% cost'}, carried as ` +
    'debt against that rung. Not scored as a win and cumulative is unchanged.' };
}

// The outstanding balance. `debt` is a list of {round, id, enables, cost_pct}.
function chainDebtReport(debt, roundNow, ladderMeasured) {
  const open = (Array.isArray(debt) ? debt : []).filter((e) => e && !ladderMeasured.has(e.enables));
  if (!open.length) return { open: [], overdue: [], caveat: '' };
  const byRung = new Map();
  for (const e of open) {
    const k = e.enables;
    const cur = byRung.get(k) || { enables: k, steps: [], cost_pct: 0, oldest_round: e.round };
    cur.steps.push(e.id);
    cur.cost_pct = round2(cur.cost_pct + (Number(e.cost_pct) || 0));
    cur.oldest_round = Math.min(cur.oldest_round, e.round);
    byRung.set(k, cur);
  }
  const rows = [...byRung.values()];
  const overdue = rows.filter((r) => roundNow - r.oldest_round >= CHAIN_DEBT_MAX_ROUNDS);
  const caveat = 'CHAIN DEBT OUTSTANDING: ' + rows.map((r) =>
    `${r.enables} owes ${r.cost_pct}% from [${r.steps.join(', ')}] since round ${r.oldest_round}`).join('; ') +
    '. These steps were committed on function, not on speed, so the canonical tree is currently ' +
    'that much slower and none of it has been paid back yet. The terminal rung is the only thing ' +
    'that can pay it.' +
    (overdue.length ? ` OVERDUE (${CHAIN_DEBT_MAX_ROUNDS}+ rounds unpaid): ${overdue.map((r) => r.enables).join(', ')}. ` +
      'Plan the terminal rung this round or state what happens to the debt — a chain abandoned ' +
      'half-built leaves the tree slower than it was found, which is the worst of the three outcomes.' : '');
  return { open: rows, overdue, caveat };
}
// <</REPLAY:enabling_step>>

// <<REPLAY:terminal_forcing>>
// Near the end of the budget, force the round onto an owed terminal rung when a chain is overdue
// (CHAIN_DEBT_MAX_ROUNDS) or remaining <= owed + 1. Returns null when nothing is owed or the planner
// already takes an owed rung.
function terminalRungToForce(debtReport, ladder, plannedRungs, remaining) {
  const open = (debtReport && Array.isArray(debtReport.open)) ? debtReport.open : [];
  if (!open.length) return null;
  const idOf = (c) => String((c && c.id) || String((c && c.title) || '').trim().split(/\s+/)[0] || '').trim();
  const planned = new Set((Array.isArray(plannedRungs) ? plannedRungs : []).map((r) => String(r).trim()));
  const owed = open
    .map((r) => ({ rung: String(r.enables || '').trim(), oldest: Number(r.oldest_round) }))
    .filter((r) => r.rung && !planned.has(r.rung));
  if (!owed.length) return null;
  const overdue = new Set(((debtReport && Array.isArray(debtReport.overdue)) ? debtReport.overdue : [])
    .map((r) => String(r.enables || '').trim()));
  const budgetTight = Number(remaining) <= owed.length + 1;
  const anyOverdue = owed.some((r) => overdue.has(r.rung));
  if (!budgetTight && !anyOverdue) return null;
  // Oldest-owed (and overdue-first) is the chain closest to being abandoned, so close it first.
  owed.sort((a, b) => {
    const ao = overdue.has(a.rung) ? 0 : 1, bo = overdue.has(b.rung) ? 0 : 1;
    if (ao !== bo) return ao - bo;
    return (a.oldest || Infinity) - (b.oldest || Infinity);
  });
  const rungId = owed[0].rung;
  const rung = (Array.isArray(ladder) ? ladder : []).find((c) => idOf(c) === rungId) || null;
  return { rungId, rung, reason: overdue.has(rungId) ? 'overdue' : 'budget_tight' };
}
// <</REPLAY:terminal_forcing>>

const CLAIM = claimBoundary(primSpeedup, MODE === 'mega');

const REQUIRE_TASK_GRAPH = !!A.require_task_graph;
// Target launch count per EP rank (default 2: the fused kernel plus the pre-dispatch quant).
// See launchVerdict.
function functionalRequirementsFor(direction, purpose) {
  const distributed = String((direction && direction.specialty) || '') === 'distributed';
  const shapeChanging = !!(direction && direction.target_shape);
  const strictPath = STRICT_AUTONOMY && (distributed || shapeChanging);
  const runtimeCommit = purpose === 'commit' && stepRoleOf(direction) === 'terminal';
  return {
    requireAccuracy: strictPath,
    requireLiveness: strictPath,
    requireGraphSafe: strictPath,
    requireArtifactDistinct: strictPath && REQUIRE_ARTIFACT_DISTINCT,
    // Kept even for a still-slow terminal commit and not waived at commit: "the switched path ran on a
    // card" is what commit-on-running means.
    requireHardwareActivation: REQUIRE_HW_ACTIVATION,
    requiredReplays: strictPath ? REQUIRED_REPLAYS : 0,
    replayGuards: strictPath ? [...new Set([...TARGET_GUARDS, ...REGRESSION_GUARDS])] : [],
    accuracyMetric: ACCURACY_METRIC,
    accuracyThreshold: ACCURACY_THRESHOLD,
    accuracyGuards: strictPath
      ? [...new Set([...REQUIRED_ACCURACY_CASES, ...TARGET_GUARDS])]
      : [],
    // Null/attribution/overlap arms decide final acceptance, not whether a correct running terminal
    // artifact is preserved for the next round. Enabling arms remain part of functional acceptance.
    requiredArms: runtimeCommit ? [] : ((direction && direction.mandatory_arms) || []),
    strict: STRICT_AUTONOMY,
  };
}
// Overlap caveats, carried to the report. TWO_LAUNCH_REACHED ratchets true once any verified
// candidate meets the launch target.
let TWO_LAUNCH_REACHED = false;
let AUTONOMY_ACCEPTANCE_REACHED = false;
const ACCEPTANCE_CAVEATS = [];
const OVERLAP_CAVEATS = [];
// Launch-shape caveats: launches per candidate and whether the target shape was met.
const LAUNCH_CAVEATS = [];
// Guard caveats: under-sampled bimodal target readings, or waves with only off-target readings.
const GUARD_CAVEATS = [];
// Withdrawn timing readings, so a voided number is still visible in the report.
const OBJECTIVE_CAVEATS = [];
// Attribution caveats and rejections, so "refused for this reason" is distinguishable from "no
// candidate" in the report.
const ATTRIBUTION_CAVEATS = [];
// Chain caveats: enabling steps kept or refused and the outstanding debt, so a half-built fusion is
// not reported as a null result.
const CHAIN_CAVEATS = [];
// Enabling steps committed on function, each {round, id, enables, cost_pct}. Cleared per rung when
// that rung is finally measured.
const CHAIN_DEBT = [];
// Rung ids that have produced a measurement. Kept alongside rungTally because the debt report needs
// the same fact and must not depend on the tally's grading order.
const LADDER_MEASURED = new Set();
// Cumulative speedup and canonical commit pinned at a chain's first enabling step; the terminal
// step is measured against this pin so the chain's own overhead is not counted as a win.
let CHAIN_BASELINE = null;
// Hardware/toolchain facts this wave established, for a human to merge into knowledge/. See
// MEMORY_SCHEMA.knowledge_delta for why this is surfaced rather than written.
const KNOWLEDGE_DELTA = [];
let TASK_GRAPH_CAVEAT = '';
let PIPE_TABLE_CAVEAT = '';
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
const LADDER = (analysis && Array.isArray(analysis.candidate_directions)) ? analysis.candidate_directions : [];
const dispatchedRungs = new Set();
// Separate from the set above and it has to be: that one answers "was this taken", this one answers
// "did taking it produce a number". Only the second can say what the next wave still owes.
const rungTally = new Map();   // rungId -> { attempts, last_outcome }
function recordRungOutcome(id, outcome) {
  const key = String(id || '').trim();
  if (!key) return;
  const e = rungTally.get(key) || { attempts: 0, last_outcome: 'never_planned' };
  // `measured` is absorbing. A rung that produced a number once is closed; a later round that takes
  // it again and faults must not reopen it, or the ladder never terminates.
  if (e.last_outcome !== 'measured') e.last_outcome = outcome;
  rungTally.set(key, e);
}
if (STRICT_AUTONOMY) {
  if (!LADDER.length) {
    throw new Error(
      'STRICT AUTONOMY REFUSED: Analyze returned no candidate_directions. A proof run with no ' +
      'machine-readable ladder cannot show that the two-launch terminal was derived.');
  }
  const terminalTargets = LADDER.filter((c) => c && c.target_shape &&
    Number(c.target_shape.launches) <= LAUNCH_TARGET);
  if (!terminalTargets.length) {
    throw new Error(
      `STRICT AUTONOMY REFUSED: no ladder rung declares target_shape.launches <= ${LAUNCH_TARGET}. ` +
      'A prose promise to fuse the operator cannot close the machine-tracked acceptance shape.');
  }
  const incompleteTargets = terminalTargets.filter((c) =>
    !Array.isArray(c.mandatory_arms) || c.mandatory_arms.length === 0 ||
    !Array.isArray(c.target_shape.stages_fused) || c.target_shape.stages_fused.length === 0 ||
    (REQUIRE_OVERLAP && c.target_shape.require_overlap !== true));
  if (incompleteTargets.length) {
    throw new Error(
      'STRICT AUTONOMY REFUSED: terminal target rung(s) must declare mandatory_arms and the required ' +
      `overlap shape: ${incompleteTargets.map((c) => c.id || c.title).join(', ')}`);
  }
}
{
  const lg0 = roadmapLadderGate(LADDER, [], new Set());
  if (lg0.summary) log(lg0.summary);
  if (lg0.caveat) log(lg0.caveat);
}
// Under working_kernel an empty ladder disables rung, chain-debt and forcing tracking. Not fatal
// (analyze_resume_fallback may re-request directions), but escalated to the report.
if (WORKING_KERNEL && !LADDER.length) {
  const msg = 'LADDER EMPTY under objective=working_kernel: Analyze returned no candidate_directions, ' +
    'so there is no staged path to a fused terminal rung. The chain-debt and terminal-forcing ' +
    'machinery has nothing to act on and every round will start from the profile. Re-run Analyze so ' +
    'it emits the fusion ladder, or this wave cannot converge on the two-launch shape by construction.';
  log(msg);
  CHAIN_CAVEATS.push(`setup: ${msg}`);
}

// Prior art: an existing implementation is a measurement, not a round; one missing from the tree is a
// ceiling on the run. A missing prior_art key (unknown) is different from [] (searched, none found).
const PRIOR_ART_RECORDED = !!(analysis && Array.isArray(analysis.prior_art));
const PRIOR_ART = PRIOR_ART_RECORDED ? analysis.prior_art : [];
if (!PRIOR_ART_RECORDED) {
  log(`PRIOR ART UNRECORDED: analysis.json has no "prior_art" key, so the step-4d sweep is not on ` +
      `the record — this is NOT the same as finding none. Every later statement about what does or ` +
      `does not exist in the baseline is UNSOURCED and must be read as such.`);
} else if (PRIOR_ART.length === 0) {
  log(`PRIOR ART: none — the sweep ran and reported an empty result.`);
}
for (const pa of PRIOR_ART) {
  const where = CAPABILITY_EVAL ? 'HIDDEN_REFERENCE' : (pa.implemented_at || '?');
  const eff = pa.measured_effect ? ` (measured ${pa.measured_effect})` : '';
  if (pa.in_baseline === false && CAPABILITY_EVAL) {
    // Same finding, opposite instruction. Here the absent implementation is the ANSWER KEY: it
    // bounds what a successful run should reach, and copying it makes the run unreadable.
    log(`PRIOR ART NOT IN BASELINE (capability eval): "${pa.direction}" exists at ${where}${eff} but ` +
        `is ABSENT from the tree being optimized. Treat it as the answer key, NOT as a source: it is ` +
        `the bar this run must reach on its own. Do not port it, and do not surface its location to ` +
        `engineers.`);
  } else if (pa.in_baseline === false) {
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

// Leak scrub under capability_eval: flag engineer-visible artifacts that mention a known reference
// path. It warns instead of aborting; the provenance gate in Verify is the enforcing half.
if (CAPABILITY_EVAL && KNOWN_REFERENCE_PATHS.length) {
  const needles = KNOWN_REFERENCE_PATHS.flatMap((p) => {
    const base = p.replace(/\/+$/, '').split('/').pop();
    return [p.replace(/\/+$/, ''), ...(base && base.length > 6 ? [base] : [])];
  });
  // Workflow scripts have no fs, so this file-content check cannot run here; say so and point to the
  // sweep script that does walk the tree.
  if (typeof fs === 'undefined') {
    log('CONTAINMENT CHECK INERT: the in-workflow reference-path scan of EVAL_DIR cannot run — ' +
        'workflow scripts have no fs. This is not a clean result, it is no result. Run ' +
        'scripts/reference_leak_sweep.sh --tree <run tree> and scripts/skill_address_scan.sh ' +
        'out of band; only those actually read the tree.');
  }
  for (const f of (typeof fs === 'undefined' ? [] : ['roadmap.md', 'codebase_context.md', 'analysis.json'])) {
    let txt = '';
    try { txt = fs.readFileSync(`${EVAL_DIR}/${f}`, 'utf8'); } catch { continue; }
    // analysis.json legitimately carries locations in prior_art[].implemented_at (that field is for
    // the orchestrator and the human, and is not forwarded to engineers). Scan the rest of it only.
    if (f === 'analysis.json') {
      try {
        const j = JSON.parse(txt); delete j.prior_art; txt = JSON.stringify(j);
      } catch { /* unparseable -> scan verbatim, a false positive here is cheap */ }
    }
    const hit = needles.find((n) => txt.includes(n));
    if (hit) {
      log(`LEAK WARNING: ${f} mentions the reference location "${hit}" while capability_eval is on. ` +
          `Engineers read this file. Prior art must reach them as MECHANISM, never as an address — ` +
          `a path in the roadmap is how the last run ended up with a byte-identical copy of the ` +
          `answer. Verify's provenance check will reject any patch that imports from it.`);
    }
  }
}

// perf_knowledge pointers from Analyze (reference only), passed to the planner and engineers.
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || '';
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];

// Fast-test cache (mega + STATE_DIR): reuse a prior wave's front-matter measurements when the
// validity key matches. Disk I/O goes through an agent. When off, no agent() call is made, so the
// call sequence and resume cache keys are unchanged.
const FAST_TEST_CACHE_DIR = STATE_DIR ? `${STATE_DIR}/fast_test_cache` : '';
async function fastTestCacheLoad(kind /* 'bench' | 'profile' */) {
  if (!MEGA_FAST_TEST) return null;
  const res = await agentT(
    roleAgent('benchmark_engineer', 'fast_test_load',
      `FAST-TEST CACHE LOAD (${kind}). NO GPU, NO lease: read the cache and validate its key, measure ` +
      `nothing. Reconstruct the ${kind === 'bench' ? 'benchmark' : 'mega_analysis'} payload from the ` +
      `cached artifacts ONLY if the stored key matches the key you recompute from disk now.`, {
        CACHE_DIR: FAST_TEST_CACHE_DIR, CACHE_KIND: kind, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR,
        FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
        BENCH_HARNESS,
        FAST_TEST_KEY_TOOL: FAST_TEST_KEY_TOOL, CONTROL_JSON: JSON.stringify(POSITIVE_CONTROL),
        GUARDS_JSON: JSON.stringify([...TARGET_GUARDS, ...REGRESSION_GUARDS]),
        TARGET_GUARDS, REGRESSION_GUARDS,
        ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}),
      }),
    { phase: 'Benchmark', label: `fast_test_load:${kind}`, schema: FAST_TEST_LOAD_SCHEMA,
      timeout_ms: 300000, max_retries: 1 });
  const commandmentReady = kind !== 'bench' ||
    (res && res.commandment_materialized === true &&
     res.commandment_current_identity === true &&
     String(res.bench && res.bench.commandment_path || '') === COMMANDMENT);
  if (!res || res.key_valid !== true || !commandmentReady) {
    log(`Fast-test cache MISS (${kind}): ` +
        `${res ? (!commandmentReady ? 'current EVAL_DIR COMMANDMENT was not materialized' :
          (res.note || (res.cache_present ? 'stored key mismatches current key' : 'no cache on disk'))) :
          'load agent returned nothing'}` +
        ` — running the real ${kind} phase fresh (and re-publishing).`);
    return null;
  }
  log(`Fast-test cache HIT (${kind}): reusing prior on-card artifacts (key ${String(res.cached_key || '').slice(0, 12)}…). ${res.note || ''}`);
  return res;
}
async function fastTestCachePublish() {
  if (!MEGA_FAST_TEST) return;
  const res = await agentT(
    roleAgent('benchmark_engineer', 'fast_test_publish',
      'FAST-TEST CACHE PUBLISH. NO GPU, NO lease: copy THIS wave\'s freshly-measured artifacts into the ' +
      'cache and write the validity key, so a later fast-test wave can reuse them. Copy files, measure ' +
      'nothing.', {
        CACHE_DIR: FAST_TEST_CACHE_DIR, EVAL_DIR, MEGA_ANALYSIS_DIR: `${EVAL_DIR}/mega_analysis`,
        SKILL_DIR: WORKFLOW_DIR, FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
        BENCH_HARNESS,
        FAST_TEST_KEY_TOOL: FAST_TEST_KEY_TOOL, CONTROL_JSON: JSON.stringify(POSITIVE_CONTROL),
        GUARDS_JSON: JSON.stringify([...TARGET_GUARDS, ...REGRESSION_GUARDS]),
        TARGET_GUARDS, REGRESSION_GUARDS,
        ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}),
      }),
    { phase: 'Profile', label: 'fast_test_publish', schema: FAST_TEST_PUBLISH_SCHEMA,
      timeout_ms: 300000, max_retries: 1 });
  if (res && res.published) log(`Fast-test cache PUBLISHED to ${res.cache_dir || FAST_TEST_CACHE_DIR} (key ${String(res.key || '').slice(0, 12)}…, ${(res.files || []).length} files).`);
  else log(`Fast-test cache publish did not complete: ${res ? (res.note || 'no confirmation') : 'agent returned nothing'}.`);
}

// PHASE: Benchmark setup (Benchmark Engineer)
phase('Benchmark');
const structuralGuardNames = [...new Set([...TARGET_GUARDS, ...REGRESSION_GUARDS])];
const structuralBench = {
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
const noGpuFrontMatter = MEGA_STRUCTURAL_ONLY || MEGA_ROUTE_ONLY;
const benchCache = noGpuFrontMatter ? null : await fastTestCacheLoad('bench');
const bench = noGpuFrontMatter ? structuralBench : benchCache ? benchCache.bench : await agentT(
  roleAgent('benchmark_engineer', 'setup', 'Build the COMMANDMENT and record a reliable baseline.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0),
    ANALYSIS: analysis,
    TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
    STRICT_AUTONOMY, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD, ...RFI, ...GMI, ...MRI,
    ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
    ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
    ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
    ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}),
  }),
  { phase: 'Benchmark', label: 'benchmark_engineer', schema: BENCH_SCHEMA,
    ...(MODE === 'mega' && MEGA_PRODUCTION
      // Baseline/control setup has a separate production window and can recover artifacts on timeout.
      ? { timeout_ms: 10800000, timeout_marker: true, max_retries: 1 } : {}) });
// A bench agent that dies usually failed to report, not to measure. Before giving up, ask one agent
// to recover the results already on disk (recover only, never re-measure).
let benchR = bench;
// A timed-out setup agent usually left its baseline/control on disk: fall through to the GPU-free
// RECOVERY below; only an empty disk makes the timeout terminal.
const benchTimedOut = !!(MODE === 'mega' && benchR && benchR.__agent_timed_out);
if (benchTimedOut) {
  log('Benchmark setup hit its full timeout window without returning. Its measurements are usually ' +
      'already on disk — attempting RECOVERY from EVAL_DIR before treating this as a terminal timeout.');
  benchR = null; // a timeout marker carries no baseline_per_case; force the recovery path below
}
if (!benchR || !benchR.baseline_per_case) {
  log('Benchmark setup returned nothing. Attempting RECOVERY from EVAL_DIR before aborting — an ' +
      'agent that died mid-phase usually left its measurements on disk.');
  benchR = await agentT(
    roleAgent('benchmark_engineer', 'recover',
      'The previous benchmark_engineer for this EVAL_DIR did not return. RECOVER ONLY: read ' +
      `${EVAL_DIR} — baseline_timing.json, COMMANDMENT.md, setup_ab_*.json, ab_logs/ — and ` +
      'reconstruct the return JSON from what is already there. Do NOT run any GPU command and do ' +
      'NOT take a lease: this step exists to avoid re-measuring, so a fresh measurement here is a ' +
      'failure, not a fallback. Write baseline_timing.json / COMMANDMENT.md if measurements exist ' +
      'but the file does not. A setup_ab_control*.json with claim_complete:true IS the positive ' +
      'control result — report its measured median and null arm rather than re-running it. If the ' +
      'measurements genuinely are not on disk, return baseline_per_case: [] and say so in notes. ' +
      'The timed-out agent may still be writing: if setup_ab_* files changed in the last 10 min, ' +
      'poll up to 12 min for baseline_timing.json before concluding.',
      { WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, ANALYSIS: analysis,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        STRICT_AUTONOMY, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}) }),
    { phase: 'Benchmark', label: 'benchmark_recover', schema: BENCH_SCHEMA,
      ...(MODE === 'mega' && MEGA_PRODUCTION ? { timeout_ms: 1200000, max_retries: 1 } : {}) });
  if (!benchR || !Array.isArray(benchR.baseline_per_case) || !benchR.baseline_per_case.length) {
    if (benchTimedOut) {
      // Nothing recoverable: return an explicit timeout status rather than throwing.
      return {
        mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
        final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
        reason: 'benchmark/calibration agent exceeded the production timeout; recovery from EVAL_DIR ' +
                'found no baseline on disk',
      };
    }
    throw new Error('Benchmark setup failed: no baseline recorded, and none recoverable from EVAL_DIR');
  }
  log(`Benchmark RECOVERED from disk: ${benchR.baseline_per_case.length} cases. ${benchR.notes || ''}`);
}
const BASELINE_PER_CASE = benchR.baseline_per_case;
const BASELINE_GEOMEAN_MS = benchR.baseline_geomean_ms;
log(`Benchmark done. ${benchR.num_test_cases || BASELINE_PER_CASE.length} cases, baseline geomean ${BASELINE_GEOMEAN_MS} ms, reliable=${benchR.reliable}`);

// Positive-control gate. Runs BEFORE any direction budget is spent, because the thing it can prove
// false — "this harness can detect a real win" — invalidates every measurement taken after it.
let PC_OVERSHOOT = '';   // set when the control passed but read HIGH; travels into the report
let PC_UNDERSHOOT = '';  // set when a CONSTRUCTED control passed but read LOW; ditto
// Whether a control actually ran in this wave (a missing switch means two identical arms).
let PC_RAN = false;
let PC_PASSED = false;
if (POSITIVE_CONTROL) {
  const pc = benchR.positive_control || {};
  const pcComplete = MODE !== 'mega' || pc.claim_complete === true;
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
  // over-reads (pass with overshoot); absurd or wrong sign measures something else (abort). Computed on
  // magnitude plus expected sign, so a control may be a deliberate slowdown (benchmark_engineer.md 5b).
  // <<REPLAY:pc_gate>>  scripts/replay_runs.js lifts this region verbatim; keep it pure (reads only
  // lo, hi, got, pc, POSITIVE_CONTROL; no log, throw or outside state).
  const mLo = Math.min(Math.abs(lo), Math.abs(hi));       // smallest effect the loop must resolve
  const mHi = Math.max(Math.abs(lo), Math.abs(hi));
  const wantSign = Math.sign(lo + hi) || 1;               // +1 = control is faster, -1 = slower
  const mGot = Math.abs(got);
  const ABSURD = Number.isFinite(Number(POSITIVE_CONTROL.implausible_pct))
    ? Math.abs(Number(POSITIVE_CONTROL.implausible_pct)) : (Number.isFinite(mHi) ? 2 * mHi : NaN);
  const nullArm = Number(pc.null_arm_pct);
  // An overshoot is only benign if the null arm is quiet. If the byte-identical arm is ALSO reading
  // high, the interleave is drifting and the overshoot is the drift, not the effect.
  const nullQuiet = Number.isFinite(nullArm) && Math.abs(nullArm) <= mLo / 2;
  // Pre-flight: switch_present=false means both arms ran the same code, so the control did not run
  // whatever it reported. Undefined keeps the old behaviour (historical replays unchanged).
  const switchAbsent = pc.switch_present === false;
  const ran = pc.ran !== false && !switchAbsent &&
    Number.isFinite(got) && Number.isFinite(lo) && Number.isFinite(hi);

  // magnitude=constructed marks a synthetic control whose band is a target. An undershoot is admissible
  // only if the reading is an effect: above a floor, at least RESOLVE_K x the worst null pair, and sign-
  // unanimous. A recorded control gets no such latitude.
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
  PC_PASSED = ok && (MODE !== 'mega' ||
    megaCalibrationClaimPass({ ...pc, passed: ok }, POSITIVE_CONTROL));
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

if (MODE === 'mega' && PC_PASSED) {
  const pc = benchR.positive_control || {};
  megaMeasurementCalibration = {
    ready: true,
    candidate_id: 'synthetic_control',
    attempt_id: pc.attempt_id || 'benchmark-control',
    evidence_manifest: pc.evidence_manifest || '',
    note: `independent synthetic control passed at ${Number(pc.measured_pct).toFixed(2)}%`,
  };
}
if (MODE === 'mega' && !PC_PASSED) {
  await remeasureMegaCalibration('after-setup', benchR.positive_control);
}

// PHASE: Baseline profile (Profile Engineer)
phase('Profile');
let profileSummary;
const profileCache = MODE === 'mega' ? null : await fastTestCacheLoad('profile');
if (MODE === 'mega') {
  // Before a candidate exists, use only the caller's measured baseline table and declared launch
  // shape. Candidate-only overlap/path metrics are established by score/finalist verification.
  const targetBaseline = BASELINE_PER_CASE.find((row) =>
    TARGET_GUARDS.includes(String(row && row.name || ''))) || BASELINE_PER_CASE[0] || {};
  profileSummary = {
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
  if (GMI.MEGA_MEASUREMENT_GUIDE && !noGpuFrontMatter) {
    const mp = await agentT(roleAgent('profile_engineer', 'mega_baseline',
      'Measure the frozen baseline per stage: follow MEGA_MEASUREMENT_GUIDE §1, not the role file.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0),
        COMMANDMENT, BASELINE_PER_CASE, TARGET_GUARDS, REGRESSION_GUARDS, ...GMI, ...MRI,
      }), { phase: 'Profile', label: 'profile_engineer:mega_baseline', schema: PROFILE_SCHEMA,
      timeout_ms: 3600000, max_retries: 1 });
    // Failure or timeout keeps the stub; it never ends the wave.
    if (mp && mp.bottleneck) profileSummary = { ...profileSummary, ...mp };
    if (mp && pipeOccupancyGate(mp.resource_timeline, []).verdict !== 'MISSING') {
      analysis = { ...analysis, resource_timeline: { ...(analysis && analysis.resource_timeline), ...mp.resource_timeline } };
    }
  }
} else if (profileCache) {
  profileSummary = profileCache.analysis;
} else {
  profileSummary = await agentT(
    roleAgent('profile_engineer', 'baseline', 'Profile the baseline and classify the bottleneck.', {
      WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0), ROUND: 0,
      COMMANDMENT,
      ...RESUME_INPUT,
    }),
    { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
}
if (MODE === 'mega' && profileSummary && profileSummary.__agent_timed_out) {
  return {
    mode: MODE, mega_deliverable: false, eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
    final_speedup: 0, final_geomean: 0, validation_status: 'agent_timeout',
    reason: 'baseline profile exceeded the production timeout; no candidate GPU work was started',
  };
}
const baselineAnalysisResult = await runProfileAnalysis(
  profileSummary, 0, 'analysis_engineer:baseline');
if (profileSummary) {
  profileSummary = {
    ...profileSummary,
    analysis_result: baselineAnalysisResult,
  };
}
log(`Baseline bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'} (dispatch_count=${profileSummary ? profileSummary.dispatch_count : '?'})`);

// Publish this wave's front-matter measurements to the fast-test cache when any stage ran fresh.
// No agent() call when MEGA_FAST_TEST is off.
if (!noGpuFrontMatter && MEGA_FAST_TEST && !benchCache) await fastTestCachePublish();

// PHASE: Optimization loop (budget-controlled)
let dispatched = 0;          // counts ONLY optimization-direction engineers (the budget)
let round = 0;
let cumulative = 1.0;        // best verified geomean speedup vs the TRUE baseline
let noImprove = 0;
// Rounds with no admissible measurement. Separate from noImprove (no wins vs no readings); both stop
// at MAX_NO_IMPROVE and are never summed. See roundEvidence.
let noEvidence = 0;
// Rounds in which the switched path never reached a card, under every objective including
// working_kernel. Any round that reaches the card (even to crash) resets it. Inert unless
// REQUIRE_HW_ACTIVATION.
let noHardware = 0;
// max_no_hardware (default MAX_NO_IMPROVE): raise it for whole-fusion authoring, whose early rounds
// may crash before launch; a run truly stuck off-hardware still stops.
const MAX_NO_HARDWARE = Math.max(1, parseInt(A.max_no_hardware != null ? A.max_no_hardware : MAX_NO_IMPROVE, 10));
// Why the round loop ended, recorded for the report.
let stopReason = null;
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source}
// Mega does not seed a global score from authoring WIP: each lane advances independently and
// cumulative changes only when a fully scored candidate wins.
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };
// The blackboard behind `history.insights`. Kept as records rather than strings so an insight can
// carry the round it came from and whether that round produced evidence at all.
let insightBook = [];
// Candidate shelf: verified non-winners kept as offers; absorbedByRound[r] is the file set CANONICAL
// took in round r, which ages shelved patches out.
let shelf = [];
let absorbedByRound = {};
const SHELF_MAX = 24;
// K historical candidates offered to the integrator per round. Kept small because every offer costs
// verification time on the shared lease; size it from the measured hit rate (0 if offers never combine).
const SHELF_OFFER_K = Number(A.shelf_offer_k) > 0 ? Number(A.shelf_offer_k) : 2;
// The evidence K gets resized from. See the report dispatch for why it is reported even when it is
// all zeros.
const SHELF_STATS = { k: SHELF_OFFER_K, shelved: 0, withheld_unknown_footprint: 0,
  rounds_offered: 0, offers: 0, hits: 0, hit_ids: [] };

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
      // Restating an insight refreshes it and is what keeps a still-relevant finding alive under
      // ageing — but it cannot upgrade a finding that came out of a round with no evidence.
      const e = out[index.get(k)];
      e.last_round = round; e.restated = (e.restated || 0) + 1;
      if (!roundVoid) e.void_round = e.void_round && false;
      continue;
    }
    out.push({ text, first_round: round, last_round: round, restated: 0, void_round: !!roundVoid });
    index.set(k, out.length - 1);
  }
  if (out.length <= MAX) return { book: out, evicted: [] };
  // Age out by last-seen round, then by how often the finding kept coming back. Ties break toward
  // the more recent entry.
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
// What was already tried, derived from the round log. Void (inactive) directions are listed
// separately: they were never tried, so they are not dead ends.
function deadEnds(rounds) {
  const tried = [], untried = [];
  for (const r of (Array.isArray(rounds) ? rounds : [])) {
    const titles = new Map((r.directions || []).map((d) => [d.id, d]));
    for (const res of (r.results || [])) {
      const d = titles.get(res.id) || {};
      const label = `r${r.round} "${d.title || res.id}" (${d.specialty || '?'})`;
      if (res.inactive) {
        untried.push(`${label}: VOID — the patched path was ${res.inactive === 'no' ? 'not executed' : 'not proven to execute'}. ` +
          `NOT a dead end: this direction has not actually been tested. Re-run it with activation proven.`);
      } else if (res.status === 'apply_failed') {
        untried.push(`${label}: patch did not apply — untested, not disproven.`);
      } else if (Number.isFinite(res.verified) && res.verified > 0) {
        tried.push(`${label}: verified ${res.verified.toFixed(3)}x${res.verified < 1.005 ? ' — no gain' : ''}`);
      } else {
        tried.push(`${label}: ${res.status || 'no result'}`);
      }
    }
  }
  return { tried, untried };
}
// <</REPLAY:memory_merge>>

// <<REPLAY:candidate_shelf>>
// The shelf keeps verified non-winners as offers. A patch is a diff against the CANONICAL it was cut
// from, so eligibility is decided by file sets: if nothing absorbed since then touches its files, it
// still applies. An empty file set means an unknown footprint; such entries are held back and named.
const shelfFile = (s) => String(s == null ? '' : s).replace(/^\.\//, '').replace(/^\/+/, '').trim();
const shelfFiles = (c) => [...new Set((Array.isArray(c.touched_files) ? c.touched_files : [])
  .map(shelfFile).filter(Boolean))];

function shelfAdd(shelf, entries, round, max) {
  const out = shelf.slice();
  for (const e of entries) {
    const files = shelfFiles(e);
    const row = {
      id: e.id, title: e.title || '', specialty: e.specialty || '',
      geomean: Number(e.geomean) || 0, patch: e.patch || '', files,
      // Fusion chain id (the terminal rung both halves name), so shelfEligible ignores overlap with the
      // chain's own earlier half. Empty for a standalone optimisation.
      chain_id: String(e.chain_id || '').trim(),
      // Three-valued on purpose, same reason `artifact_distinct` is: "we could not tell" has to stay
      // sayable, or the only expressible answer is the one that flatters the candidate.
      footprint: files.length ? 'known' : 'unknown',
      base_round: round, shelved_round: round, absorbed: false,
    };
    const at = out.findIndex((x) => x.id === e.id);
    if (at >= 0) out[at] = { ...out[at], ...row }; else out.push(row);
  }
  out.sort((a, b) => b.geomean - a.geomean);
  return { shelf: out.slice(0, max), evicted: out.slice(max) };
}

// absorbedByRound[round] is a file array, or {files, chain_id} for a commit inside a fusion chain.
function shelfEligible(shelf, absorbedByRound, k) {
  const norm = (v) => Array.isArray(v)
    ? { files: v, chain_id: '' }
    : { files: (v && v.files) || [], chain_id: String((v && v.chain_id) || '').trim() };
  const eligible = [], stale = [], unknown = [];
  for (const e of shelf) {
    if (e.absorbed) continue;
    if (e.footprint !== 'known') { unknown.push(e); continue; }
    const moved = new Set();
    for (const r of Object.keys(absorbedByRound || {})) {
      if (Number(r) <= e.base_round) continue;
      const a = norm(absorbedByRound[r]);
      // A commit on the same chain does not age the entry: both halves of a fusion edit the same files.
      if (a.chain_id && e.chain_id && a.chain_id === e.chain_id) continue;
      for (const f of a.files) moved.add(shelfFile(f));
    }
    const clash = e.files.filter((f) => moved.has(f));
    if (clash.length) stale.push({ ...e, clash }); else eligible.push(e);
  }
  eligible.sort((a, b) => b.geomean - a.geomean);
  return { offer: eligible.slice(0, Math.max(0, k)), eligible, stale, unknown };
}
// <</REPLAY:candidate_shelf>>

// <<REPLAY:overlap_gate>>
// Is an overlap claim backed by an instrument (knowledge/overlap_instrument.md)? A fused kernel has one
// trace record, so a faster fused kernel is not by itself evidence of overlap. Never fails a candidate;
// it makes "not measured" visible.
const OVERLAP_BASE_MAX = 0.05;  // the unfused base path should read zero; 5% is instrument slop.
function overlapVerdict(ver, opts) {
  const o = (ver && ver.overlap) || {};
  const won = Number((opts && opts.geomean) || 0) > 1.0;
  const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : null);
  const frac = num(o.fraction), base = num(o.base_reading), forced = num(o.forced_reading);
  const m = o.measured === 'yes' || o.measured === 'no' ? o.measured : 'unknown';

  // The negative control first, because it invalidates everything downstream. A meter that finds
  // overlap on the unfused base path is measuring its own artefacts.
  if (m !== 'unknown' && base != null && base > OVERLAP_BASE_MAX) {
    return { state: 'meter_broken', caveat:
      `The overlap meter reads ${(base * 100).toFixed(1)}% on the unfused base path, whose true overlap ` +
      `is zero by construction (separate launches, disjoint intervals). The meter is measuring its own ` +
      `artefacts, so its ${frac != null ? `${(frac * 100).toFixed(1)}% ` : ''}reading on the fused ` +
      `path carries no information. Overlap is UNMEASURED for this candidate.` };
  }
  // A meter that has never read a known non-zero is indistinguishable from a dead one.
  if (m !== 'unknown' && base == null) {
    return { state: 'meter_unvalidated', caveat:
      `Overlap was reported as "${m}" but the meter's negative control is missing: nobody ran it ` +
      `against the unfused path, where the answer is known to be zero. An uncontrolled meter reading ` +
      `is not evidence — the same rule the workflow applies to a benchmark with no positive control.` };
  }
  if (m === 'no' && forced == null) {
    return { state: 'meter_unvalidated', caveat:
      `The meter reports NO overlap and has never read a known non-zero (no forced-concurrency ` +
      `control). A dead meter and a correct "there is no overlap here" produce the same output, and ` +
      `this run cannot tell them apart. Treat the negative finding as unconfirmed.` };
  }
  if (m === 'unknown') {
    return { state: 'unmeasured', caveat: won
      ? `This candidate improves latency but overlap was NOT measured` +
        `${o.note ? ` (${String(o.note).slice(0, 200)})` : ''}. In the fused shape latency cannot ` +
        `distinguish overlap from removed launch overhead or better L2 residency, so this result ` +
        `must not be reported as demonstrating overlap. Name the collection experiment that is missing.`
      : `Overlap was not measured for this candidate; no overlap claim may be made from it either way.` };
  }
  // Measured, controlled. What is left is whether the number and the latency agree.
  if (m === 'yes' && frac != null && frac > 0.05 && !won) {
    return { state: 'contention', caveat:
      `The meter reports ${(frac * 100).toFixed(1)}% overlap but the operator did not get faster. ` +
      `Concurrency without a latency win is CONTENTION — two roles co-resident and competing for the ` +
      `same CUs, issue slots or LDS. That is a finding about the partition, not a partial success.` };
  }
  if (m === 'yes' && (frac == null || frac <= 0.05) && won) {
    return { state: 'win_without_overlap', caveat:
      `A latency win with essentially no measured overlap (${frac == null ? 'fraction unreported' : `${(frac * 100).toFixed(1)}%`}). ` +
      `The win is probably real but it is NOT an overlap result — look for the mechanism (launch ` +
      `overhead, L2 residency, grid shape) and attribute it, or the next round will build on the ` +
      `wrong cause.` };
  }
  if (m === 'no') {
    return { state: 'measured_none', caveat:
      `The meter was built, controlled, and reports no overlap. That is a result about this edge, ` +
      `not a missing measurement.` };
  }
  return { state: 'measured', caveat: '' };
}
// <</REPLAY:overlap_gate>>

// <<REPLAY:attribution_gate>>
// Is the win inside the kernel that changed (knowledge/fusion_preconditions.md)? Under operator_e2e,
// attribution explains the operator result and a mismatch only withholds the mechanism claim; under
// changed_kernel the task owns a comparable same-timeline kernel measure and a mismatch rejects.
const ATTRIBUTION_EPS_PCT = 0.5;  // below this the two readings are the same number

function attributionVerdict(ver, opts) {
  const a = (ver && ver.attribution) || {};
  const won = Number((opts && opts.geomean) || 0) > 1.0;
  const metric = String((opts && opts.metric) || 'changed_kernel');
  const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : null);
  const chg = num(a.changed_us), rep = num(a.replaced_sum_us);
  const e2ePct = ((Number((opts && opts.geomean) || 1)) - 1) * 100;

  if (chg == null || rep == null || rep <= 0) {
    if (!won) return { state: 'not_applicable', reject: false, caveat: '' };
    return { state: 'unattributed', reject: false, caveat:
      `This candidate claims ${e2ePct >= 0 ? '+' : ''}${e2ePct.toFixed(2)}% end-to-end but did not ` +
      `report the changed kernel's own time against the kernels it replaced. End-to-end moves for ` +
      `reasons that live BETWEEN launches, so on its own it cannot say whether the kernel got ` +
      `faster. Report attribution.changed_us / replaced_sum_us, read paired in one collection, or ` +
      `the number stays unattributed.` };
  }

  // Positive delta = the changed kernel is FASTER than what it replaced.
  const kPct = ((rep - chg) / rep) * 100;
  const resB = num(a.residual_ms_base), resC = num(a.residual_ms_cand);
  const gapNote = (resB != null && resC != null)
    ? ` The residual (end-to-end minus the sum of kernel times) moved ${resB >= 0 ? '+' : ''}` +
      `${resB.toFixed(4)} -> ${resC >= 0 ? '+' : ''}${resC.toFixed(4)} ms, i.e. ` +
      `${Math.abs((resB - resC) * 1000).toFixed(0)}us of the claim is in the gaps.`
    : '';

  if (won && kPct < -ATTRIBUTION_EPS_PCT) {
    const reject = metric === 'changed_kernel';
    return { state: reject ? 'gap_win' : 'mechanism_mismatch', reject, caveat:
      `${reject ? 'REJECTED as a changed-kernel win.' : 'Operator win retained, mechanism claim withheld.'} ` +
      `End-to-end reads ${e2ePct >= 0 ? '+' : ''}${e2ePct.toFixed(2)}% but the ` +
      `kernel that was changed is SLOWER than the kernels it replaced: ${chg.toFixed(1)}us against ` +
      `${rep.toFixed(1)}us, ${kPct.toFixed(2)}%.${gapNote} The win is not inside the code under ` +
      `test. Find and measure the mechanism before calling it overlap. Under operator_e2e that ` +
      `cross-launch/cross-rank effect remains part of the product metric; under changed_kernel it is ` +
      `outside the declared score.` };
  }
  if (!won && kPct > ATTRIBUTION_EPS_PCT) {
    return { state: 'kernel_win_e2e_loss', reject: false, caveat:
      `The changed kernel is ${kPct.toFixed(2)}% faster than what it replaced, but end-to-end did ` +
      `not improve.${gapNote} The cost moved rather than disappeared -- into launch structure, ` +
      `arrival skew, or a downstream kernel's wait. This is a real kernel result with a live ` +
      `regression attached; locate it before shipping.` };
  }
  if (won && Math.abs(kPct) <= ATTRIBUTION_EPS_PCT) {
    const reject = metric === 'changed_kernel';
    return { state: reject ? 'gap_win' : 'mechanism_mismatch', reject, caveat:
      `${reject ? 'REJECTED as a changed-kernel win.' : 'Operator win retained, mechanism claim withheld.'} ` +
      `End-to-end reads ${e2ePct >= 0 ? '+' : ''}${e2ePct.toFixed(2)}% while the ` +
      `changed kernel is flat against the kernels it replaced (${chg.toFixed(1)}us vs ` +
      `${rep.toFixed(1)}us, ${kPct >= 0 ? '+' : ''}${kPct.toFixed(2)}%).${gapNote} A flat kernel ` +
      `cannot explain the end-to-end win; report the operator result and withhold the claimed mechanism.` };
  }
  return { state: 'attributed', reject: false, caveat: won
    ? '' : `Kernel time ${kPct >= 0 ? '+' : ''}${kPct.toFixed(2)}%, consistent with the end-to-end reading.` };
}
// <</REPLAY:attribution_gate>>

// <<REPLAY:launch_gate>>
// Is the fusion at the caller's target launch count and regions? Never rejects (a faster partial
// fusion is a partial rung); it records the count and tells the ladder whether shape_met. unjudged
// (no count reported) is a hole, not a pass.
function launchVerdict(ver, target, expectedStages) {
  const ls = (ver && ver.launch_shape) || {};
  const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : null);
  const cand = num(ls.launches_cand), base = num(ls.launches_base);
  // The orchestrator/ladder owns the target. A verifier reports what ran; it cannot move the goal.
  const tgt = num(target) != null ? num(target) : num(ls.target);
  const stages = Array.isArray(ls.stages_fused) ? ls.stages_fused.map(String) : [];
  const expected = Array.isArray(expectedStages) ? expectedStages.map(String) : [];
  const missingStages = expected.filter((s) => !stages.includes(s));
  const proofMissing = ls.per_rank !== true || !String(ls.how_counted || '').trim() ||
    missingStages.length > 0;
  if (tgt == null || !Number.isInteger(tgt) || tgt <= 0) {
    return { state: 'unjudged', shape_met: false, caveat:
      'Launch target is missing or invalid; the caller must supply a positive integer.' };
  }
  if (cand == null || !Number.isInteger(cand) || cand <= 0) {
    return { state: 'unjudged', shape_met: false, caveat:
      `Launch count NOT reported, so the caller's target (${tgt}) is UNJUDGED. Report ` +
      `launch_shape.launches_cand for one operator call with how_counted.` };
  }
  if (expected.length && proofMissing) {
    return { state: 'unjudged', shape_met: false, caveat:
      `Launch count ${cand} was reported without a complete proof: per_rank must be true, ` +
      `how_counted must be non-empty, and stages_fused is missing [${missingStages.join(', ') || 'none'}].` };
  }
  if (cand <= tgt) {
    return { state: 'met', shape_met: true, caveat: base != null && base !== cand
      ? `Launch count ${base} -> ${cand} (target ${tgt}); the two-launch fused shape is reached.` : '' };
  }
  return { state: 'above_target', shape_met: false, caveat:
    `Launch count is ${cand}${base != null ? ` (from ${base})` : ''}, target ${tgt}. The regions are ` +
    `not all fused yet -- this is a partial rung, not a wrong result, but the terminal launch ` +
    `shape has NOT been reached and the fusion rung must not be reported as closed on it.` };
}
// <</REPLAY:launch_gate>>

// <<REPLAY:bimodal_split>>
// Some guards are bimodal (two discrete latency states) rather than noisy, and separation is then
// unreachable by deep sampling. Condition on the state: classify readings, compare like with like and
// report state occupancy separately. Two rules: classification is arm-blind (pooled readings, no
// per-arm threshold), and declining to classify (no clean gap) falls back to all pairs.
const BIMODAL_MIN_GAP_PCT = 3.0;  // a gap smaller than this is a spread, not two states.
function modeSplit(readings, minGapPct) {
  const gapMin = Number.isFinite(minGapPct) ? minGapPct : BIMODAL_MIN_GAP_PCT;
  const v = readings.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length < 6) return { classified: false, reason: `only ${v.length} readings; need >=6 to see a gap` };
  // The widest relative gap between consecutive sorted readings. Anything else (k-means, a fixed
  // threshold) needs a parameter that is not in the data.
  let at = -1, best = 0;
  for (let i = 0; i < v.length - 1; i++) {
    const g = v[i] > 0 ? ((v[i + 1] - v[i]) / v[i]) * 100 : 0;
    if (g > best) { best = g; at = i; }
  }
  if (best < gapMin) {
    return { classified: false, reason: `widest gap ${best.toFixed(2)}% < ${gapMin}% — one state, not two`,
             gap_pct: best };
  }
  const cut = (v[at] + v[at + 1]) / 2;
  // The slow state must be the minority, or "slow" is just the operating point and the few fast
  // readings are the anomaly. Naming the wrong cluster inverts every conclusion below it.
  const nSlow = v.length - at - 1;
  if (nSlow > v.length / 2) {
    return { classified: false, reason: `the upper cluster holds ${nSlow}/${v.length} readings — that is ` +
             `the operating point, not a tail`, gap_pct: best };
  }
  return { classified: true, cut, gap_pct: best, n_fast: at + 1, n_slow: nSlow };
}

// `pairs` = [{base, cand}, ...] collected interleaved. Returns the mode-aware paired analysis.
function pairedModeAware(pairs, minGapPct) {
  const ok = pairs.filter((p) => Number.isFinite(Number(p.base)) && Number.isFinite(Number(p.cand)));
  const split = modeSplit([...ok.map((p) => p.base), ...ok.map((p) => p.cand)], minGapPct);
  const pct = (p) => ((Number(p.base) - Number(p.cand)) / Number(p.base)) * 100;  // >0 = candidate faster
  const all = ok.map(pct);
  const signTest = (xs) => {
    if (!xs.length) return { n: 0, agree: 0, p: null };
    const pos = xs.filter((x) => x > 0).length, neg = xs.filter((x) => x < 0).length;
    const agree = Math.max(pos, neg);
    // Two-sided sign test at unanimity is what the existing doctrine leans on; report it for any
    // level of agreement so a 9-of-10 is not silently rounded to unanimous.
    return { n: xs.length, agree, p: agree === xs.length ? Math.pow(2, -xs.length + 1) : null };
  };
  const med = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null);

  if (!split.classified) {
    return { conditioned: false, reason: split.reason, gap_pct: split.gap_pct,
             all: { deltas: all, median: med(all), ...signTest(all) } };
  }
  const isSlow = (x) => Number(x) > split.cut;
  const fast = ok.filter((p) => !isSlow(p.base) && !isSlow(p.cand));
  const mixed = ok.filter((p) => isSlow(p.base) !== isSlow(p.cand));
  const bothSlow = ok.filter((p) => isSlow(p.base) && isSlow(p.cand));
  const fastD = fast.map(pct);
  // Occupancy is an effect, not noise: a candidate that lands in the slow state less often is faster.
  const occ = (rows, pick) => {
    const n = rows.length; const k = rows.filter((p) => isSlow(pick(p))).length;
    return { n, slow: k, pct: n ? (k / n) * 100 : null };
  };
  return {
    conditioned: true, cut: split.cut, gap_pct: split.gap_pct,
    fast: { deltas: fastD, median: med(fastD), ...signTest(fastD) },
    all: { deltas: all, median: med(all), ...signTest(all) },
    dropped: { mixed_mode_pairs: mixed.length, both_slow_pairs: bothSlow.length },
    occupancy: { base: occ(ok, (p) => p.base), cand: occ(ok, (p) => p.cand) },
  };
}

// Pairs to collect so enough land with both arms fast: at ~20% slow rate a pair is usable with
// probability 0.64, so 10 fast pairs need ~16.
function pairsNeeded(wantFastPairs, slowRatePct) {
  const s = Math.min(0.9, Math.max(0, (Number(slowRatePct) || 0) / 100));
  const usable = (1 - s) * (1 - s);
  return usable <= 0 ? null : Math.ceil(wantFastPairs / usable);
}

// Groups Verify's raw readings by guard, classifies each arm-blind, and reports the conditioned median
// and whether enough both-fast pairs were collected. Only target-route readings may carry a win.
// Returns {available, guards, target_median, off_target, caveats}; without readings the aggregate
// geomean is used.
function analyzeGuards(pairedReadings, targetGuards, wantFastPairs) {
  const rows = Array.isArray(pairedReadings) ? pairedReadings : [];
  if (!rows.length) return { available: false, guards: [], target_median: null, off_target: [], caveats: [] };
  const byGuard = new Map();
  for (const r of rows) {
    const g = String((r && r.guard) || '').trim() || '(unnamed)';
    if (!byGuard.has(g)) byGuard.set(g, []);
    byGuard.get(g).push({ base: Number(r.base), cand: Number(r.cand) });
  }
  // Target scope: the explicit set if configured; else any guard whose name says "uniform" (this
  // campaign's target route); else, if none match, every guard is target and nothing is filtered.
  let targets = new Set((Array.isArray(targetGuards) ? targetGuards : []).map((g) => String(g).trim()).filter(Boolean));
  if (!targets.size) {
    const uni = [...byGuard.keys()].filter((g) => /uniform/i.test(g));
    if (uni.length) targets = new Set(uni);
  }
  const allTarget = targets.size === 0;
  const want = Number.isFinite(Number(wantFastPairs)) && Number(wantFastPairs) > 0 ? Number(wantFastPairs) : 10;
  const guards = [], caveats = [];
  let targetMedian = null;
  for (const [g, pairs] of byGuard) {
    const analysis = pairedModeAware(pairs, undefined);
    const isTarget = allTarget || targets.has(g);
    const slow = analysis.conditioned && analysis.occupancy
      ? Math.max(analysis.occupancy.base.pct || 0, analysis.occupancy.cand.pct || 0)
      : 0;
    const needed = pairsNeeded(want, slow);
    const usableFast = analysis.conditioned ? (analysis.fast.n || 0) : (analysis.all.n || 0);
    const median = analysis.conditioned ? analysis.fast.median : analysis.all.median;
    guards.push({ guard: g, is_target: isTarget, conditioned: analysis.conditioned,
      median, fast_pairs: usableFast, pairs_needed: needed, occupancy: analysis.occupancy || null });
    if (isTarget && targetMedian == null && Number.isFinite(Number(median))) targetMedian = Number(median);
    if (isTarget && needed != null && usableFast < want && Number(analysis.all.n) < needed) {
      caveats.push(`${g}: ${usableFast} usable fast-state pair(s) of the ~${needed} needed at a ` +
        `${Math.round(slow)}% slow-state rate; the mode-aware median is UNDER-SAMPLED, so any win on ` +
        `this guard is PROVISIONAL until more pairs are collected.`);
    }
  }
  const offTarget = guards.filter((r) => !r.is_target).map((r) => r.guard);
  if (!allTarget && targetMedian == null && offTarget.length) {
    caveats.push(`No reading on a TARGET route [${[...targets].join(', ')}]; the only readings are on ` +
      `off-target guard(s) [${offTarget.join(', ')}], which are do-no-harm rails and NOT the metric. ` +
      `This measured nothing it is allowed to claim a win on -- the exact state a wave once shipped a ` +
      `number from a skew guard in.`);
  }
  return { available: true, guards, target_median: targetMedian, off_target: offTarget, caveats };
}

// Credit comes only from explicit target guards; regression guards are vetoes. Uses per_case (the
// verifier's noise-aware result).
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

const metricScore = (result) =>
  promotionScore(result, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS, primSpeedup(result));
const ATTRIBUTION_METRIC = PROMOTION_METRIC === 'legacy' ? 'changed_kernel' : PROMOTION_METRIC;

// <<REPLAY:autonomy_acceptance>>
// Completion and acceptance are separate: a correct but slow two-launch artifact can be committed under
// working_kernel without counting as a successful optimization.
function autonomyAcceptanceVerdict(direction, ver, opts) {
  const o = opts || {};
  const shape = (direction && direction.target_shape) || null;
  if (!shape) return { applicable: false, complete: true, accepted: false, reasons: [] };
  const reasons = [];
  const scoreReasons = [];
  const fa = functionalAcceptance(ver, o.functionalRequirements || {});
  if (!fa.pass) reasons.push(...fa.missing);

  const lv = launchVerdict(ver, Number(shape.launches || o.launchTarget || 2), shape.stages_fused || []);
  if (!lv.shape_met) reasons.push(lv.caveat || 'launch target was not met');

  const gc = guardContract(ver, o.targetGuards || [], o.regressionGuards || [], o.fallbackScore);
  const declaredScore = promotionScore(ver, o.promotionMetric || 'operator_e2e',
    o.targetGuards || [], o.regressionGuards || [], gc.score);
  if (!gc.complete) {
    if (gc.missing_targets.length) reasons.push(`missing target guard(s): ${gc.missing_targets.join(', ')}`);
    if (gc.missing_regressions.length) reasons.push(`missing regression guard(s): ${gc.missing_regressions.join(', ')}`);
  }
  if (!gc.regression_pass && gc.regressed.length) {
    reasons.push(`regression guard(s) below baseline: ${gc.regressed.join(', ')}`);
  }

  const allGuards = [...new Set([...(o.targetGuards || []), ...(o.regressionGuards || [])])];
  if (o.requirePaired && allGuards.length) {
    const pairs = Array.isArray(ver && ver.paired_readings) ? ver.paired_readings : [];
    if (!Number.isFinite(Number(ver && ver.null_arm_pct))) {
      reasons.push('same-session byte-identical null_arm_pct is required');
    } else if ((declaredScore - 1) * 100 <= Math.abs(Number(ver.null_arm_pct))) {
      scoreReasons.push(`target claim ${((declaredScore - 1) * 100).toFixed(3)}% does not clear null ` +
        `${Math.abs(Number(ver.null_arm_pct)).toFixed(3)}%`);
    }
    for (const guard of allGuards) {
      const n = pairs.filter((p) => String(p.guard || '') === String(guard)).length;
      const needed = Number((o.requiredPairsByGuard || {})[guard] || o.requiredPairs || 1);
      if (n < needed) {
        reasons.push(`${guard} has ${n} paired reading(s), requires ${needed}`);
      }
    }
    const targetNeed = Math.max(Number(o.requiredPairs || 1),
      ...(o.targetGuards || []).map((g) => Number((o.requiredPairsByGuard || {})[g] || 0)));
    const ga = analyzeGuards(pairs, o.targetGuards || [], targetNeed);
    if (!ga.available || !Number.isFinite(Number(ga.target_median))) {
      reasons.push('target guard has no readable paired result');
    }
    reasons.push(...ga.caveats);
    const choose = (n, k) => {
      let out = 1;
      for (let i = 1; i <= k; i++) out = out * (n - k + i) / i;
      return out;
    };
    for (const guard of (o.targetGuards || [])) {
      const gp = pairs.filter((p) => String(p.guard || '') === String(guard));
      const wins = gp.filter((p) => Number(p.base) > Number(p.cand)).length;
      let tail = 0;
      for (let k = wins; k <= gp.length; k++) tail += choose(gp.length, k);
      const p = gp.length ? tail / Math.pow(2, gp.length) : 1;
      if (p > 0.05) scoreReasons.push(`${guard} paired sign test p=${p.toFixed(4)} > 0.05`);
    }
  }

  const needsOverlap = !!(shape.require_overlap || o.requireOverlap);
  if (needsOverlap) {
    const ov = overlapVerdict(ver, { geomean: gc.score });
    const frac = Number(ver && ver.overlap && ver.overlap.fraction);
    const cuFrac = Number(ver && ver.overlap && ver.overlap.cu_fraction);
    const baseControl = Number(ver && ver.overlap && ver.overlap.base_reading);
    const forced = Number(ver && ver.overlap && ver.overlap.forced_reading);
    if (ver && ver.overlap && ver.overlap.measured !== 'yes') {
      reasons.push('controlled overlap evidence measured=yes is required');
    } else if (!Number.isFinite(baseControl) || baseControl > 0.05 ||
               !Number.isFinite(forced) || forced <= 0.05) {
      reasons.push('overlap meter requires passing base and forced-concurrency controls');
    } else if (!['measured', 'contention'].includes(ov.state)) {
      reasons.push(ov.caveat || `overlap meter state is ${ov.state}`);
    } else if (!(frac > 0.05) || !(cuFrac > 0.05)) {
      reasons.push('on-edge wall-clock and CU-weighted overlap must both exceed instrument slop');
    }
  }

  if (o.requireAttribution) {
    const a = (ver && ver.attribution) || {};
    if (!(Number(a.changed_us) > 0 && Number(a.replaced_sum_us) > 0 &&
          (o.targetGuards || []).includes(String(a.guard || '')) &&
          String(a.method || '').trim() &&
          Number.isFinite(Number(a.residual_ms_base)) &&
          Number.isFinite(Number(a.residual_ms_cand)) &&
          (o.promotionMetric !== 'changed_kernel' || a.absolute_to_frozen === true))) {
      reasons.push('launch-changing terminal has invalid attribution values/guard/method/residuals');
    } else {
      const at = attributionVerdict(ver, {
        geomean: declaredScore, metric: o.promotionMetric || 'operator_e2e',
      });
      if (at.reject) reasons.push(at.caveat);
    }
  }

  const complete = reasons.length === 0;
  const accepted = complete && scoreReasons.length === 0 && declaredScore > 1.0 && !!o.positiveControlRan;
  const acceptanceReasons = [...reasons, ...scoreReasons];
  if (complete && !(declaredScore > 1.0)) {
    acceptanceReasons.push(`declared target score ${declaredScore.toFixed(4)}x is not > 1.0`);
  }
  if (complete && !o.positiveControlRan) acceptanceReasons.push('positive control did not pass in this wave');
  return { applicable: true, complete, accepted, score: declaredScore, launch: lv, guards: gc,
    reasons: acceptanceReasons };
}
// <</REPLAY:autonomy_acceptance>>

// Resume: restore cumulative speedup and insight/ledger history from the prior wave. No-op on a fresh
// run.
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
  if (MODE === 'mega' && Array.isArray(ps.candidate_registry)) {
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
  if (MODE === 'mega' && Number.isFinite(Number(ps.state_sequence))) {
    megaStateSequenceBase = Math.max(0, Number(ps.state_sequence));
  }
  // Calibration is intentionally NOT restored as authority. Every wave runs/recover-checks its own
  // control; prior calibration remains in STATE for audit only and cannot license fresh scores.
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
  if (shelf.length) log(`  shelf: ${shelf.length} verified non-winner(s) carried forward, ` +
    `${shelf.filter((e) => e.footprint === 'known').length} with a known file footprint.`);
  if (MODE === 'mega') {
    log(`  mega candidates: ${megaCandidateRegistry.length}, calibration=${megaMeasurementCalibration.ready ? 'ready' : 'pending'}.`);
  }
}

// A COMPLETE control that failed the gate is re-measured fresh (bounded); the gate never changes.
async function remeasureMegaCalibration(label, prior) {
  if (MODE !== 'mega' || megaMeasurementCalibration.ready || !POSITIVE_CONTROL ||
      MEGA_STRUCTURAL_ONLY || MEGA_ROUTE_ONLY || megaPcRemeasured >= 2 ||
      !prior || prior.claim_complete !== true) return;
  megaPcRemeasured += 1;
  const r = await agentT(
    roleAgent('benchmark_engineer', 'recalibrate',
      `RE-MEASURE the positive control fresh (attempt ${megaPcRemeasured}/2): the last complete ` +
      `attempt failed the gate. Same gate; never re-gate, drop or reuse pairs.`, {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0),
        ANALYSIS: analysis, POSITIVE_CONTROL, PREVIOUS_CONTROL: prior,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      }),
    { phase: 'Benchmark', label: `mega recalibration ${label}`, schema: BENCH_SCHEMA,
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
  if (MODE !== 'mega' || megaMeasurementCalibration.ready || !POSITIVE_CONTROL) return;
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
    { phase: 'Benchmark', label: `mega calibration recovery ${label}`, schema: BENCH_SCHEMA,
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
  if (MODE !== 'mega' || !STATE_DIR) return true;
  const sequence = megaStateSequenceBase + Number(currentRound) * 10 + (finalizing ? 9 : 0);
  const generation = `mega:${currentRound}:${megaCandidateRegistry.reduce(
    (m, c) => Math.max(m, Number(c.attempts || 0)), 0)}:${sequence}`;
  const persisted = await agentT(
    roleAgent('tech_lead', 'update_memory',
      'Persist the mega candidate registry exactly; do not benchmark, plan, or modify candidate source.', {
        STATE_DIR, CANONICAL, ROUND: currentRound,
        CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase,
        CANDIDATE_REGISTRY: megaCandidateRegistry,
        MEASUREMENT_CALIBRATION: megaMeasurementCalibration,
        STATE_GENERATION: generation,
        STATE_SEQUENCE: sequence,
        PRIOR_HISTORY: history, OPEN_RUNGS: [], ROADMAP_LADDER: [],
        LADDER_DISPATCHED: [], LADDER_COMPLETED: [],
        SHELF: shelf, ABSORBED_FILES: absorbedByRound,
      }),
    { phase: 'Optimize', label: `mega candidate state r${currentRound}`, schema: MEMORY_SCHEMA,
      // Large late-round registries need a real-time hung guard wider than the modeled turn clock.
      ...(MEGA_PRODUCTION ? { timeout_ms: 600000, max_retries: 1 } : {}) });
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
      { phase: 'Optimize', label: `mega state verify r${currentRound}${labelSuffix || ''}`, schema: MEMORY_SCHEMA,
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
        { phase: 'Optimize', label: `mega candidate state r${currentRound} retry`, schema: MEMORY_SCHEMA,
          ...(MEGA_PRODUCTION ? { timeout_ms: 600000, max_retries: 1 } : {}) });
      if (!echoOk(retry)) verify = await readState(' recheck');
      if (!echoOk(retry) && !echoOk(verify)) {
        // In-memory-only checkpoint is not resumable. Stop before a round advances from stale state (lane git commit stays intact).
        log(`MEGA STATE PERSIST FAILED r${currentRound}: STATE.json did not advance to seq=${sequence} ` +
          `after retry. The invocation must stop; continuing would lose structured candidate progress.`);
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

async function planMegaCandidateTurn(currentRound, remaining, pool) {
  const searchHistory = megaHistoryForSearch(history, megaCandidateRegistry);
  const plan = await agentT(
    roleAgent('mega_search_lead', 'plan_round',
      'Choose one whole-kernel candidate direction. Expert knowledge guides the common candidate lifecycle; it never creates a reserved lane.', {
        ...(pool ? { GPU_POOL: pool, GPU_MIN_FREE_GIB } : {}),
        EVAL_DIR, ROUND: currentRound, BUDGET_REMAINING: remaining,
        CUMULATIVE_SPEEDUP: cumulative, BASELINE_GEOMEAN_MS,
        SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary, ...GMI, ...MRI,
        CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: searchHistory,
        MEGA_CANDIDATE_REGISTRY: megaRegistryForSearch(megaCandidateRegistry),
        MEASUREMENT_CALIBRATION: megaMeasurementCalibration,
        MEGA_STALL: { rounds: megaStall, limit: MEGA_STALL_LIMIT },
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
        ...(CHAIN_DEBT.length ? {
          CHAIN_DEBT: chainDebtReport(CHAIN_DEBT, currentRound, LADDER_MEASURED).open,
          CHAIN_BASELINE,
        } : {}),
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, LAUNCH_TARGET,
        REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
        REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        KERNEL_KNOWLEDGE_DIR: '', KK_OPERATOR: '', KK_LANGUAGE: '', KK_REFS: [], ...KB_INPUTS,
      }),
    { phase: 'Optimize', label: `mega:plan r${currentRound}`, schema: MEGA_PLAN_SCHEMA,
      // Late-round plans read a large registry and ran 6-7 min; a 5 min guard nulled them.
      ...(MEGA_PRODUCTION ? { timeout_ms: 900000, max_retries: 1 } : {}) });
  const noPlan = !plan || plan.stop || !Array.isArray(plan.directions) || !plan.directions.length;
  const wipDir = noPlan ? megaWipContinuationDirection(currentRound) : null;
  // A planner that returned nothing never decided to stop: optimize the best scored lane.
  const best = noPlan && !wipDir && !plan ? selectMegaSearchParent(megaCandidateRegistry) : null;
  if (noPlan && !best) return wipDir;
  let raw = best ? { id: `r${currentRound}_child_${best.id}`, candidate_id: best.id,
    candidate_source: best.source, base_candidate_id: best.id, title: `optimize ${best.id}`,
    specialty: MEGA_DEFAULT_SPECIALTY, step_role: 'terminal', focus_files: [],
    roadmap_rung: 'off_ladder', target_topology: best.topology,
    prompt: best.next_blocker || 'Continue the first unresolved measured blocker.' }
    : plan.directions[0];
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
  const bound = parkedNow ? raw : bindMegaDirectionToWip(currentRound, raw);
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

async function runMegaCandidateTurn(currentRound, remaining) {
  const turnStartedMs = megaNowMs();
  const dispatchDeadlineMs = WORKFLOW_STARTED_MS +
    (MEGA_TIME_BUDGET_S - MEGA_FINAL_RESERVE_S) * 1000;
  if (!megaMeasurementCalibration.ready &&
      (currentRound === 2 || (currentRound > 2 && currentRound % 3 === 1))) {
    await recoverMegaCalibration(`before-round-${currentRound}`);
  }
  const pool = (MEGA_STRUCTURAL_ONLY || MEGA_ROUTE_ONLY)
    ? null : await samplePool(currentRound, MEGA_PRODUCTION ? 120000 : 0);
  const d = await planMegaCandidateTurn(currentRound, remaining, pool);
  if (!d) return { stop: true, reason: 'no mega candidate direction was planned' };
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
    return {
      refused: true, stop: MEGA_ROUTE_ONLY,
      reason: `Mega direction refused before authoring: ${topologyVerdict.reason}`,
    };
  }
  if (MEGA_ROUTE_ONLY) {
    return {
      stop: true,
      direction: d,
      reason: `route-only PASS: candidate=${d.candidate_id}, rung=${
        d.roadmap_rung || d.id}, launch_count=${topologyLaunchCount(d.target_topology)}, ` +
        `target=${Number(analysis && analysis.mega_plan_ir &&
          analysis.mega_plan_ir.target &&
          analysis.mega_plan_ir.target.launch_count)}; no candidate source or GPU was touched`,
    };
  }
  if (STRICT_AUTONOMY) {
    const strict = strictDirectionVerdict(d, LADDER, LADDER_MEASURED);
    if (!strict.pass) {
      return { refused: true, stop: MEGA_ROUTE_ONLY, reason:
        `strict Mega direction refused before authoring: ${strict.reason}` };
    }
  }
  for (const rung of lg.planned) {
    dispatchedRungs.add(rung);
    const tally = rungTally.get(rung) || { attempts: 0, last_outcome: 'never_planned' };
    tally.attempts += 1;
    if (tally.last_outcome !== 'measured') tally.last_outcome = 'unmeasured';
    rungTally.set(rung, tally);
  }
  const candidateId = d.candidate_id;
  const source = d.candidate_source;
  if (!['search', 'integrated'].includes(source)) {
    return { stop: true, reason: `invalid Mega candidate source '${String(source)}'; Expert Skills ` +
      'must use the common search/integrated lifecycle' };
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
    return {
      stop: true,
      reason: `only ${Math.max(0, Math.round(dispatchRemainingS))}s remain before the production ` +
        `final-validation reserve; no new candidate attempt was started`,
    };
  }
  const requestedTurnS = MEGA_CANDIDATE_TIMEOUT_S;
  const turnDeadlineMs = MEGA_PRODUCTION
    ? Math.min(turnStartedMs + requestedTurnS * 1000, dispatchDeadlineMs)
    : turnStartedMs + requestedTurnS * 1000;
  const turnBudgetS = (turnDeadlineMs - turnStartedMs) / 1000;
  const prepElapsedS = (megaNowMs() - turnStartedMs) / 1000;
  const availableAfterPrepS = (turnDeadlineMs - megaNowMs()) / 1000;
  if (MEGA_PRODUCTION && availableAfterPrepS < 900) {
    return {
      stop: true,
      reason: `candidate preparation consumed ${Math.round(prepElapsedS)}s; only ` +
        `${Math.max(0, Math.round(availableAfterPrepS))}s remain in the shared turn budget, so no ` +
        `Engineer was started`,
    };
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
  const authorGpuProhibited = MEGA_STRUCTURAL_ONLY ||
    (authorPreflightRequired && !stagedGpuAuthoring);
  const role = 'engineer';
  const roleFile = 'engineer.md';
    eng = await agentT(
      roleAgent(role, 'optimize',
        `Advance candidate lane ${candidateId}; never edit or replace another candidate lane.`, {
          CANDIDATE_ID: candidateId, CANDIDATE_SOURCE: source,
          BASE_CANDIDATE_ID: baseCandidateId, BASE_TREE: baseTree, BASE_HEAD: baseHead,
          CANDIDATE_TREE: tree, OUTPUT_DIR: outDir, ATTEMPT_ID: attemptId,
          CANDIDATE_PYTHONPATH: tree,
          CANDIDATE_IMPORT_MODULES,
          LANE_MANIFEST: laneManifest, LANE_LOCK: laneLock,
          CANDIDATE_TIMEOUT_S: commandBudgetS,
          PERSISTENT_JIT_CACHE_DIR: persistentJitCache,
          SPECIALTY: d.specialty || MEGA_DEFAULT_SPECIALTY, DIRECTION: d,
          KERNEL_PATH: tree, OP_SPEC, TASK_DIR: KERNEL_PATH_ORIG, COMMANDMENT,
          GPU_ID: authorGpuProhibited
            ? (MEGA_STRUCTURAL_ONLY
              ? 'STRUCTURAL_ONLY_NO_GPU' : 'STRUCTURAL_PREFLIGHT_REQUIRED_NO_GPU')
            : GPU_RESOURCE.specForIndex(0),
          GPUS_PER_JOB: authorGpuProhibited
            ? '0' : String(GPU_RESOURCE.gpusPerJob),
          AUTHOR_GPU_MODE: authorGpuProhibited ? 'forbidden' : 'bounded_smoke',
          ...(MEASURE_FIRST_TEXT(megaStall) ? { MEASURE_FIRST: String(megaStall) } : {}),
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
      `\n\n${authorGpuProhibited
        ? `AUTHOR_GPU_MODE=forbidden (${MEGA_STRUCTURAL_ONLY ? 'STRUCTURAL-ONLY RUN'
          : 'AUTHORING CONTRACT PREFLIGHT IS REQUIRED'}): no rocm-smi, torchrun, benchmark, ` +
          `correctness command or GPU runtime import. Repair required failures in category order, ` +
          `commit a coherent checkpoint, and return claim_complete=true with ` +
          `candidate_status=authoring; runtime_verified, gpu_executed, and score_complete remain ` +
          `false separately. `
        : `AUTHOR_GPU_MODE=bounded_smoke: use the supplied EP8 device for bounded construction/JIT/` +
          `correctness/short-perf smokes of the HEAD you are editing, before AND after source edits, ` +
          `repeatedly in this turn. Author smokes are development evidence; independent Verify owns ` +
          `scoring. Revert diagnostic-only probes before committing. ${MEASURE_FIRST_TEXT(megaStall)}`}` +
      (/^[0-9a-f]{7,40}$/.test(String(d.rewind_head || ''))
        ? `REWIND_TO=${d.rewind_head}: after taking the lane lock and before other edits, run ` +
          `\`git tag -f parked_r${currentRound} HEAD && git reset --hard ${d.rewind_head}\` in ` +
          `${tree}; the old HEAD stays recoverable through that tag. ` : '') +
      `Before any source edit, acquire the single-writer lane lock in your persistent shell: ` +
      `mkdir -p "$(dirname "${laneLock}")"; exec 9>"${laneLock}"; flock -n 9, and keep fd 9 open ` +
      `through the final commit/manifest write, then release it with exec 9>&-. Start every background ` +
      `command or timer with 9>&- so no orphan inherits the lock. If the lock is held, return incomplete without editing. ` +
      `Candidate-lane setup is part of this task. If ${tree}/.git does not exist, create ${tree}. ` +
      (baseHead
        ? `Export exact parent commit ${baseHead} from ${baseTree} with git archive into the lane. `
        : `Copy ${baseTree} into it with a tar pipe excluding .git/build/__pycache__/.torch_ext/*.so/*.o. `) +
      `then initialize a fresh git repository and commit "candidate base ${baseCandidateId}". ` +
      `Before any long-running command, atomically write ${laneManifest} with candidate_id/source/` +
      `base_candidate_id/tree/attempt_id so a ` +
      `timed-out first attempt is discoverable on resume. ` +
      `If it exists, continue from its HEAD; never recreate it. Read ${WORKFLOW_DIR}/roles/mega/${roleFile}. ` +
      `Before every candidate Python or torchrun command, export ` +
      `PYTHONPATH="${tree}\${PYTHONPATH:+:\$PYTHONPATH}" and fail closed unless both ` +
      `every module in ${JSON.stringify(CANDIDATE_IMPORT_MODULES)} resolves under ${tree}. ` +
      `Commit WIP to THIS lane after every real advance. Write ${outDir}/candidate_result.json ` +
      `atomically only after its evidence files are final. Return candidate_id/source/base, tree, head, ` +
      `candidate_status, claim_complete, attempt_id, evidence_manifest, patch_file (cumulative from the ` +
      `lane root), correctness, absolute_score, per_case, topology_sig, next_blocker and notes. ` +
      `This Engineer gets ` +
      `${engineerBudgetS}s and every GPU command is bounded by ${commandBudgetS}s. A timeout or ` +
      `force-emit returns claim_complete:false and status authoring; it is not score 0.`,
      { phase: 'Optimize', label: `mega:${source}:${candidateId}`, schema: MEGA_CANDIDATE_SCHEMA,
        timeout_ms: engineerBudgetS * 1000, timeout_marker: true, max_retries: 1 });

    if (eng && eng.__agent_timed_out) {
    megaUnsafeTimeout = eng;
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
          `RECOVER ONLY candidate ${candidateId}. First confirm the timed-out writer is quiescent and ` +
          `the lane lock is free. ` + (drainS ? `If not, wait up to ${drainS}s total, in order: ` +
          `\`flock -w <left> ${laneLock} true\`; the same on every ` +
          `\${KERNEL_EXTRACT_LOCK_DIR:-/tmp/team_gpu_locks}/gpu_*.lock; then poll every 30s until no ` +
          `non-zombie process (ps state not Z) has a cmdline or cwd under ${laneDir} or ${outDir}, ` +
          `excluding your own shell. Zombies and stale lease_*.json files are not holders; never ` +
          `use kill -0. Return writer_quiescent=true only if all three held. ` : '') +
          `Read ${outDir}/candidate_result.json and completed ` +
          `measurement aggregates. Accept only a manifest with claim_complete:true and attempt_id. ` +
          `Do not use a partial/older worker result, do not run a GPU command, and do not edit ${tree}. ` +
          `If the writer is still active or the lock is held, return claim_complete:false.`, {
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
      megaUnsafeTimeout = null;
      log(`Mega round ${currentRound}: timed-out author of ${candidateId} is quiescent; its ` +
        `committed WIP stays in the lane and the wave continues.`);
    }
    if (recovered && recovered.claim_complete === true) {
      eng = recovered;
      laneWriterTimedOut = false;
    }
  }
  if (!MEGA_RESUME_STATE && !existing && freshMegaAuthorLeak(eng)) {
    return {
      stop: true,
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
      stop: true,
      reason: `candidate author timeout for ${candidateId}; invocation stopped to preserve single-writer ` +
        `lineage safety`,
    };
  }

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
  const gpuWaitBudgetS = Math.max(30, Math.min(300, Math.floor(verifyBudgetS / 4)));
  const gpuRunBudgetS = Math.max(60, verifyBudgetS - gpuWaitBudgetS - 60);
  const shouldVerify = eng && eng.claim_complete === true && expectedHead &&
    (!contractBlocksRuntime || meta.structural_verified) &&
    !MEGA_STRUCTURAL_ONLY &&
    (!MEGA_PRODUCTION || verifyBudgetS >= 300) &&
    (postAuthoringVerify ||
      ['runnable', 'scored', 'finalist'].includes(String(eng.candidate_status || '')));
  if (shouldVerify) {
    const verifyInputs = {
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
          GPU_ID: GPU_RESOURCE.specForIndex(0), SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
          BASELINE_PER_CASE, FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
      VERIFY_TIER: 'score',
      MODIFIABLE_FILES: 'WHOLE_CANDIDATE_TREE',
      SPECIALTY: d.specialty || MEGA_DEFAULT_SPECIALTY,
      TARGET_SHAPE: megaShapeFromTopology(d.target_topology),
          TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
          REQUIRE_ARTIFACT_DISTINCT: true, REQUIRE_OVERLAP: false,
          REQUIRE_ATTRIBUTION: false, REQUIRED_REPLAYS: 30,
      REQUIRE_GRAPH_CAPTURE: REQUIRE_GRAPH_CAPTURE ? '1' : '0',
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
    ver = await agentT(
      roleAgent('verify_engineer', 'verify',
        'Independently verify and score this whole-tree mega candidate against the frozen baseline.',
        verifyInputs),
      { phase: 'Verify', label: `mega:verify:${candidateId}`, schema: VERIFY_SCHEMA,
        timeout_ms: verifyBudgetS * 1000, timeout_marker: true, max_retries: 1 });
    if (ver && ver.__agent_timed_out) {
      verifierTimedOut = true;
      log(`Mega round ${currentRound}: verify for ${ver.label || candidateId} exceeded its ` +
        `deadline. Checking process quiescence and on-disk evidence before one safe retry.`);
      ver = null;
    }
    if (!ver || ver.claim_complete !== true) {
      const recoveryBudgetS = MEGA_PRODUCTION
        ? Math.max(0, Math.floor(turnBudgetS - (megaNowMs() - turnStartedMs) / 1000 - 60))
        : 300;
      if (recoveryBudgetS >= 30) {
        const recoveredVer = await agentT(
          roleAgent('verify_engineer', 'recover',
            `RECOVER verification for candidate ${candidateId}, attempt ${attemptId}. Do not run a GPU ` +
            `command or edit source. If the prior Verify timed out, check twice at least 10s apart that ` +
            `no matching GPU/torchrun/lease process remains and the lane lock is free. Set ` +
            `timeout_recovery_safe=true only then; report active_gpu_processes and evidence_complete.`, {
              VERIFY_DIR: `${outDir}/verify`, CANDIDATE_ID: candidateId,
              CANDIDATE_SOURCE: source, CANDIDATE_TREE: tree,
              EXPECTED_HEAD: expectedHead, ATTEMPT_ID: attemptId,
              SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
            }),
          { phase: 'Verify', label: `mega:verify-recover:${candidateId}`, schema: VERIFY_SCHEMA,
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
              schema: VERIFY_SCHEMA, timeout_ms: retryBudgetS * 1000,
              timeout_marker: true, max_retries: 1 });
          if (retry && retry.claim_complete === true) {
            ver = retry;
            verifierTimedOut = false;
          } else if (retry && !retry.__agent_timed_out) {
            // Quiescence was confirmed twice and the bounded retry returned: nothing is left running, so the
            // round ends unscored and the wave continues.
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
      requiredReplays: 30, requiredPairs: REQUIRED_PAIRS,
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
      stop: true,
      reason: `candidate score verification timeout for ${candidateId}; invocation stopped before any ` +
        `other EP8 work was dispatched`,
    };
  }

  // A score may be recorded before calibration completes, but it cannot update the incumbent or be
  // published until a synthetic control proves the harness sees an effect.
  const selection = megaMeasurementCalibration.ready
    ? selectMegaCandidate(megaCandidateRegistry, MEGA_TIE_NOISE_PCT)
    : { selected: null, eligible: [], tie_kept_skill: false };
  const selected = selection.selected;
  const improved = !!(selected && selected.absolute_score > cumulative * (1 + MIN_IMPROVE));
  if (selected && (improved || !finalWinner)) {
    cumulative = Math.max(cumulative, selected.absolute_score);
    bestPerCase = selected.per_case.length ? selected.per_case : bestPerCase;
    finalWinner = {
      id: selected.id, source: selected.source, geomean: selected.absolute_score,
      per_case: selected.per_case, patch: selected.patch, tree: selected.tree,
      topology_sig: selected.topology_sig, provenance: selected.provenance,
      activation: selected.activation,
    };
  }

  const hasEvidence = !!(ver && ver.claim_complete === true);
  noEvidence = hasEvidence ? 0 : noEvidence + 1;
  noImprove = improved ? 0 : noImprove + 1;
  const reachedHardware = !!(ver &&
    String(ver.activation_on_hardware || '').toLowerCase() === 'yes');
  // Source-only checkpoints do not consume the hardware-failure budget; only a complete candidate that
  // reached Verify counts.
  const hardwareWasDue = !!shouldVerify;
  if (reachedHardware) noHardware = 0;
  else if (hardwareWasDue) noHardware += 1;
  const blockerKey = String(record.next_blocker || '').slice(0, 160);
  megaStall = hasEvidence || improved ||
    (eng && eng.gpu_executed === true && blockerKey !== megaLastBlocker) ? 0 : megaStall + 1;
  megaLastBlocker = blockerKey;
  // Analysis gets the identity-bound evidence of THIS candidate instead of scanning EVAL_DIR.
  if (ANALYSIS_SKILL_ON && ver) profileSummary = { ...profileSummary,
    analysis_result: await runProfileAnalysis({ ...profileSummary, candidate_evidence: {
      candidate_id: candidateId, head: record.head, attempt_id: record.attempt_id,
      target_guard: record.target_guard, frozen_per_case: BASELINE_PER_CASE,
      per_case: record.per_case, paired_readings: record.paired_readings,
      denominator_mismatch: record.denominator_mismatch,
      component_breakdown: record.component_breakdown, hot_path_counters: record.hot_path_counters,
      evidence_manifest: record.evidence_manifest } }, currentRound, 'mega:analyze') };
  history.rounds.push({
    round: currentRound,
    directions: [{ id: d.id, title: d.title, specialty: d.specialty,
      candidate_id: candidateId, candidate_source: source }],
    results: [{ id: d.id, status: ver ? ver.status : (eng ? eng.candidate_status : 'pending'),
      verified: record.absolute_score, candidate_status: record.status }],
    winner: selected ? { source: selected.id, geomean: selected.absolute_score } : null,
    improved, cumulative,
  });
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
  const stateSaved = await persistMegaCandidateState(currentRound);
  if (!stateSaved) {
    return {
      stop: true,
      reason: `candidate state for round ${currentRound} could not be persisted; stopped before ` +
        `dispatching another lane`,
    };
  }
  return { stop: false, direction: d, record, ver, improved, selected };
}

// Working-kernel mode is budget/decision bounded; no-evidence separately catches a blind harness.
while (dispatched < BUDGET &&
    (MODE === 'mega' || WORKING_KERNEL ||
     (noImprove < MAX_NO_IMPROVE && noEvidence < MAX_NO_IMPROVE))) {
  round++;
  const remaining = BUDGET - dispatched;
  phase('Optimize');

  if (MODE === 'mega') {
    const elapsedS = (megaNowMs() - WORKFLOW_STARTED_MS) / 1000;
    if (elapsedS >= MEGA_TIME_BUDGET_S - MEGA_FINAL_RESERVE_S) {
      stopReason = `${MEGA_PROFILE} dispatch deadline reached after ${Math.round(elapsedS / 60)} minutes; ` +
        `preserving ${Math.round(MEGA_FINAL_RESERVE_S / 60)} minutes for finalist validation.`;
      break;
    }
    const turn = await runMegaCandidateTurn(round, remaining);
    if (turn.refused && !turn.stop) {
      const count = megaRefusal ? megaRefusal.count + 1 : 1;
      megaRefusal = { reason: turn.reason, count };
      dispatched += 1;
      megaStall += 1;
      log(`Mega round ${round}: ${turn.reason} (refusal ${count}/3; re-planning).`);
      if (count >= 3) {
        stopReason = `${count} consecutive refused directions; last: ${turn.reason}`;
        break;
      }
      continue;
    }
    megaRefusal = null;
    if (turn.stop) {
      if (MEGA_ROUTE_ONLY) {
        const d = turn.direction || {};
        return {
          mode: MODE, mega_route_only: true,
          route_pass: String(turn.reason || '').startsWith('route-only PASS:'),
          eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
          candidate_id: d.candidate_id || '', roadmap_rung: d.roadmap_rung || d.id || '',
          launch_count: topologyLaunchCount(d.target_topology),
          target_launch_count: Number(analysis && analysis.mega_plan_ir &&
            analysis.mega_plan_ir.target && analysis.mega_plan_ir.target.launch_count),
          validation_status: 'route_only', stop_reason: turn.reason,
        };
      }
      stopReason = turn.reason;
      break;
    }
    dispatched += 1;
    log(`Mega round ${round}: candidate=${turn.record.id}[${turn.record.source}:${turn.record.status}], ` +
      `score=${Number.isFinite(turn.record.absolute_score) ? turn.record.absolute_score.toFixed(4) + 'x' : 'pending'}, ` +
      `selected=${turn.selected ? turn.selected.id : 'none'}, calibration=` +
      `${megaMeasurementCalibration.ready ? 'ready' : 'pending'}, budget=${dispatched}/${BUDGET}.`);
    if (MEGA_STRUCTURAL_ONLY && turn.record.structural_verified) {
      stopReason = `structural-only acceptance reached by ${turn.record.id}: source checkpoint and ` +
        `MegaPlanIR consistency passed; hardware/accuracy/liveness/performance remain unverified.`;
      break;
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
      turn.selected &&
      Number(turn.selected.absolute_score) >= Number(EXPERT_SKILL_RECORDED_LOW) &&
      Number(turn.selected.launches) === expertTargetLaunches
    );
    if (MEGA_STOP_ON_DELIVERABLE && turn.selected &&
        Number(turn.selected.absolute_score) > 1.0 &&
        expertTargetReached &&
        searchAttempts >= MEGA_MIN_SEARCH_ATTEMPTS) {
      stopReason = `production early-stop after ${round} round(s): candidate ${turn.selected.id} ` +
        `has calibrated ${Number(turn.selected.absolute_score).toFixed(4)}x speedup over frozen ` +
        `baseline${expertTargetConfigured ? `, reached Expert Skill target ` +
          `${Number(EXPERT_SKILL_RECORDED_LOW).toFixed(4)}x at ${expertTargetLaunches} launches` : ''}, ` +
        `and ${searchAttempts} search attempt(s) have run; reserve the remaining ` +
        `${BUDGET - dispatched} direction(s) for finalist validation instead of exhaustive search.`;
      break;
    }
    if (REQUIRE_HW_ACTIVATION && noHardware >= MAX_NO_HARDWARE) {
      stopReason = `${noHardware} consecutive portfolio attempts did not execute the candidate path ` +
        `on hardware; stopping before another unmeasured direction. Lane WIP remains resumable.`;
      log(`Mega hardware stop: ${stopReason}`);
      break;
    }
    if (MEGA_STALL_LIMIT && megaStall >= 2 * MEGA_STALL_LIMIT) {
      stopReason = `${megaStall} consecutive rounds produced no new runtime evidence (independent ` +
        'Verify, better score, or a changed on-card blocker); lane WIP remains resumable.';
      log(`Mega stall stop: ${stopReason}`);
      break;
    }
    continue;
  }

  // (a) Plan the round. Sample the pool first; GPU_POOL is omitted when sampling is disabled, keeping
  // the prompt unchanged.
  const pool = await samplePool(round);
  const plan = await agentT(
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).', {
      ...(pool ? { GPU_POOL: pool, GPU_MIN_FREE_GIB } : {}),
      EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
      BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
      CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
      // The ladder Analyze built, and what has been taken off it. Passed as DATA, not as a path to
      // maybe-read: a plan phase that is only told where roadmap.md lives plans from the profile.
      ROADMAP: `${EVAL_DIR}/roadmap.md`,
      ROADMAP_LADDER: LADDER,
      LADDER_DISPATCHED: [...dispatchedRungs],
      LADDER_COMPLETED: [...LADDER_MEASURED],
      OPEN_RUNGS: openRungs(LADDER, rungTally),
      TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, LAUNCH_TARGET,
      STRICT_AUTONOMY, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
      REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      // Outstanding chain debt: committed enabling steps that only their terminal rung can pay back.
      ...(CHAIN_DEBT.length ? {
        CHAIN_DEBT: chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED).open,
        CHAIN_BASELINE,
      } : {}),
      // Pass the dependency graph and pipe table to the planner, since its directions are judged against
      // them.
      ...(REQUIRE_TASK_GRAPH && analysis && analysis.task_graph
        ? { TASK_GRAPH: JSON.stringify(analysis.task_graph).slice(0, 4000) } : {}),
      ...(REQUIRE_TASK_GRAPH && analysis && analysis.resource_timeline
        ? { RESOURCE_TIMELINE: JSON.stringify(analysis.resource_timeline).slice(0, 3000) } : {}),
      KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
      ...KB_INPUTS,
    }),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    stopReason = `the TechLead planned no directions for round ${round}` +
      `${plan && plan.reasoning ? `: ${plan.reasoning}` : '.'} ` +
      `${BUDGET - dispatched} of ${BUDGET} budget unit(s) were left unspent.`;
    break;
  }

  let directions = plan.directions.slice(0, remaining).map((d, i) => {
    const enriched = enrichDirection(d, LADDER);
    return {
      ...enriched,
      idx: i,
      id: enriched.id || `r${round}_d${i}`,
      gpu_id: GPU_RESOURCE.specForIndex(i),
      out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
    };
  });
  // Near the end of the budget, force the round onto an open terminal rung of a committed chain. Runs
  // before the deep/working clamps. See terminalRungToForce.
  {
    const debtReport = chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED);
    const plannedRungs = directions.map((d) => String(d.roadmap_rung || '').trim()).filter(Boolean);
    const force = terminalRungToForce(debtReport, LADDER, plannedRungs, remaining);
    if (force) {
      const c = force.rung || {};
      const forced = {
        idx: 0,
        id: `r${round}_close_${force.rungId}`,
        title: c.title || `close fusion rung ${force.rungId}`,
        specialty: MEGA_DEFAULT_SPECIALTY,
        focus_files: Array.isArray(c.focus_files) ? c.focus_files : [],
        expected_speedup: Number(c.expected_speedup) || undefined,
        roadmap_rung: force.rungId,
        step_role: 'terminal',
        gated_on: Array.isArray(c.gated_on) ? c.gated_on : [],
        mandatory_arms: Array.isArray(c.mandatory_arms) ? c.mandatory_arms : [],
        target_shape: c.target_shape || { launches: LAUNCH_TARGET, require_overlap: REQUIRE_OVERLAP },
        rung_deviation: `forced onto terminal rung ${force.rungId} (${force.reason}): its enabling ` +
          `steps are committed and this rung is the only thing that pays their debt, so the remaining ` +
          `budget is spent closing the fusion rather than exploring.`,
        prompt: `Close the fusion. Enabling/producer steps for terminal rung "${force.rungId}"` +
          `${c.title ? ` (${c.title})` : ''} are already committed to the canonical tree: their ` +
          `completion signalling and buffers exist but nothing consumes them yet. Write the CONSUMER ` +
          `half that collapses the launches into the two-launch fused shape, and report launch_shape, ` +
          `accuracy (relL2), overlap and attribution. This is step_role=terminal: it is judged on ` +
          `FUNCTION and MEASUREMENT -- it may be slower than the tree the first time it lands and is ` +
          `still committed on RUNNING -- not on beating the current best. ${c.rationale || c.prompt || ''}`,
        gpu_id: GPU_RESOURCE.specForIndex(0),
        out_dir: `${EVAL_DIR}/round_${round}/engineer_0`,
      };
      log(`Round ${round}: FORCING terminal rung ${force.rungId} (${force.reason}); ` +
          `${directions.length} planned direction(s) deferred. The budget no longer has slack to both ` +
          `explore and close every open chain, and an unclosed chain ships a tree slower than it started.`);
      directions = [forced];
    }
  }
  // deep_explore runs alone in its round (broad rewrites cannot merge with specialist patches) and
  // costs DEEP_COST; other directions cost 1.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  // One direction per round under working_kernel: with one lease, extra directions would only return
  // static analysis.
  if (WORKING_KERNEL && directions.length > 1) {
    const dropped = directions.slice(1).map(d => d.id);
    directions = directions.slice(0, 1);
    log(`Round ${round}: objective=working_kernel — one direction gets the lease. Kept ${directions[0].id}, ` +
        `deferred ${dropped.join(', ')}. They are not dead ends; they were not run.`);
  }
  // Price the directions against the pipe table before the lease is spent.
  if (REQUIRE_TASK_GRAPH) {
    const pg = pipeOccupancyGate(analysis && analysis.resource_timeline, directions);
    if (pg.caveat) log(`Round ${round}: ${pg.caveat}`);
  }
  // And against the ladder (every run, not gated on require_task_graph), before the budget is charged.
  {
    const lg = roadmapLadderGate(LADDER, directions, LADDER_MEASURED);
    log(`Round ${round}: ${lg.summary}`);
    if (lg.caveat) log(`Round ${round}: ${lg.caveat}`);
    if (STRICT_AUTONOMY) {
      const invalid = directions.map((d) => ({ d, v: strictDirectionVerdict(d, LADDER, LADDER_MEASURED) }))
        .filter((x) => !x.v.pass);
      if (invalid.length) {
        throw new Error(
          'STRICT AUTONOMY PLAN REFUSED before budget charge: ' +
          invalid.map((x) => `${x.d.id}: ${x.v.reason}`).join('; '));
      }
    }
    for (const r of lg.planned) {
      dispatchedRungs.add(r);
      const e = rungTally.get(r) || { attempts: 0, last_outcome: 'never_planned' };
      e.attempts += 1;
      // Provisional: a rung that is taken counts as unmeasured until a verified number arrives.
      // If the round dies before the grading below runs, that is exactly the state to carry forward.
      if (e.last_outcome !== 'measured') e.last_outcome = 'unmeasured';
      rungTally.set(r, e);
    }
  }
  // Constrain each direction's focus_files to Analyze's modifiable whitelist; if that empties it, steer
  // focus onto the whitelist and log it, instead of a round that dies in Verify as out-of-whitelist.
  // No-op without a whitelist or when focus is already a subset.
  {
    const mod = (analysis && Array.isArray(analysis.modifiable_files)) ? analysis.modifiable_files : [];
    if (mod.length) {
      const inMod = (f) => mod.some((m) => f === m || f.endsWith('/' + m) || m.endsWith('/' + f));
      for (const d of directions) {
        const focus = Array.isArray(d.focus_files) ? d.focus_files : [];
        if (!focus.length) continue;
        const keep = focus.filter(inMod);
        if (keep.length === focus.length) continue;
        if (keep.length) {
          log(`Round ${round}: ${d.id} focus_files narrowed to the MODIFIABLE whitelist ` +
              `(${focus.length}->${keep.length}); dropped ${focus.filter((f) => !inMod(f)).join(', ')}.`);
          d.focus_files = keep;
        } else {
          log(`Round ${round}: ${d.id} focus_files are DISJOINT from the MODIFIABLE whitelist ` +
              `(${focus.join(', ')} vs allowed ${mod.join(', ')}) -- the lane-gap deadlock. Steering ` +
              `focus onto the modifiable set so the engineer edits files it is allowed to.`);
          d.focus_files = mod.slice();
        }
      }
    }
  }
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
        ? `Then Read ${WORKFLOW_DIR}/roles/mega/deep_engineer.md and ALL knowledge files under ${WORKFLOW_DIR}/knowledge/ ` +
          `(you have broad authority — combine algorithm + memory + compute + host_runtime levers in one ` +
          `coherent rewrite), and follow them. You MAY edit ANY modifiable source (kernel + Python wrapper ` +
          `+ C++ binding), not just focus_files. Run your OWN multi-iteration measure→(self-)profile→rewrite ` +
          `loop and push to the TARGET; keep the best correct version.`
        : `Then Read ${WORKFLOW_DIR}/roles/mega/engineer.md and ${WORKFLOW_DIR}/knowledge/mega_self_monitoring.md and the ` +
          `knowledge files for your specialty, and follow them.`;
      return agentT(
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
First create YOUR private workspace, then optimize.
\`\`\`bash
# Fresh, ISOLATED workspace via tar-copy that EXCLUDES build artifacts (.git/build/__pycache__/.torch_ext/
# *.so/*.o) — no 'rm' anywhere. Each engineer's out_dir is unique per (round,engineer), so the workspace
# is clean on creation; the tar excludes mean no stale build cache is ever inherited (torch .torch_ext
# stores ABSOLUTE paths, so excluding it forces each workspace to build its own fresh).
mkdir -p ${d.out_dir}/workspace
( cd ${CANONICAL} && tar --exclude=./.git --exclude='*/.git' --exclude=./build --exclude='*/build' \\
    --exclude=./__pycache__ --exclude='*/__pycache__' --exclude=./.torch_ext --exclude='*/.torch_ext' \\
    --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd ${d.out_dir}/workspace && tar -xf - )
# The tar excluded .git ON PURPOSE (the source history must never be readable from a workspace), so
# make a FRESH one-commit repo here instead. Without this \`git diff\` has no HEAD to diff against and
# best_patch.diff has to be hand-maintained — which is how a wave shipped patches nobody could
# regenerate. A fresh repo carries no history, so nothing leaks and nothing extra is copied.
cd ${d.out_dir}/workspace
printf '%s\\n' 'build/' '__pycache__/' '*.so' '.torch_ext/' '.rocprofv3/' '*.o' > .gitignore
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "workspace baseline"
\`\`\`
${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git add -A && git diff HEAD > ${d.out_dir}/best_patch.diff\`
whenever your implementation BUILDS and passes correctness -- NOT only when geomean>1.0. A step that
is a prerequisite for a later fusion (step_role=enabling) and the final consumer half of a fusion are
both SLOWER than the tree on their own, by construction, and the workflow decides their fate from the
saved patch: an enabling step that leaves no patch cannot be committed on function, and a terminal
fusion that leaves no patch cannot be committed on RUNNING. Withholding the diff because it is not yet
faster is how this project shipped half a fusion nobody could carry forward. (Stage first: a plain
\`git diff\` omits files you CREATED, and a patch missing a new file applies cleanly and then fails at
import.) Report your measured geomean honestly whatever it is; do not suppress a patch to avoid a
number below 1.0.

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: {
          id: d.id, title: d.title, focus_files: d.focus_files || [],
          expected_speedup: d.expected_speedup, prompt: d.prompt,
          roadmap_rung: d.roadmap_rung, gated_on: d.gated_on || [],
          mandatory_arms: d.mandatory_arms || [], step_role: d.step_role,
          enables: d.enables, cost_budget_pct: d.cost_budget_pct,
          target_shape: d.target_shape,
          target_guards: TARGET_GUARDS, regression_guards: REGRESSION_GUARDS,
          promotion_metric: PROMOTION_METRIC, required_pairs: REQUIRED_PAIRS,
          required_pairs_by_guard: REQUIRED_PAIRS_BY_GUARD,
        },
        ...(isDeep ? { TARGET: d.expected_speedup ? `reach ${d.expected_speedup}x (or ~90% of the roofline ceiling), whichever is the harder bar` : 'reach ~90% of the roofline ceiling' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        // What earlier rounds already spent budget on. Without this an engineer sees only what was
        // learned, never what was attempted, and re-proposes a direction the loop already bought.
        ...(function () { const de = deadEnds(history.rounds); return {
          ...(de.tried.length ? { ALREADY_TRIED: de.tried } : {}),
          ...(de.untried.length ? { NOT_YET_ACTUALLY_TESTED: de.untried } : {}) }; })(),
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.`,
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA }
    ).then((eng) => ({ d, eng }));
    },

    // Recover an engineer's claim from disk before dropping it: a complete worker_result and patch may
    // exist even if the agent returned no StructuredOutput. Parse only; no GPU, no re-measurement.
    async (prev) => {
      const { d } = prev;
      let { eng } = prev;
      const patch = `${d.out_dir}/best_patch.diff`;
      if (CLAIM.needsRecovery(eng)) {
        // No fs in workflow scripts, so recovery is a cheap read-only agent (recover only).
        const onDisk = await agentT(
          // The role is `engineer` (roles/engineer.md); a wrong name sends the agent to a missing file.
          roleAgent('engineer', 'recover',
            `The engineer for ${d.out_dir} did not return a usable claim. RECOVER ONLY: read ` +
            `${d.out_dir}/worker_result.json AND compare it with the mtimes/claim_complete fields of ` +
            'ab_*.json aggregates beside it. If a completed aggregate is newer, the declaration is stale: ' +
            'recover from that aggregate and name it. Use raw logs only if both are absent/truncated. ' +
            'Return the claim already on disk. Do NOT run any GPU ' +
            'command, do NOT take a lease, do NOT re-measure and do NOT improve the result: a fresh ' +
            'measurement here is a failure, not a fallback. Report per_case EXACTLY as recorded, ' +
            'including guards the engineer marked UNRESOLVED. If no claim is on disk, return ' +
            'per_case: [] and say so in notes.',
            { OUT_DIR: d.out_dir, SKILL_DIR: WORKFLOW_DIR, BASELINE_PER_CASE, COMMANDMENT }),
          { phase: 'Optimize', label: `recover ${d.id}`, schema: ENG_SCHEMA });
        if (CLAIM.recovered(onDisk)) {
          log(`${d.id}: no usable StructuredOutput, but a claim was on disk — ` +
              `${onDisk.per_case.length} case(s), geomean ${onDisk.speedup_geomean}. RECOVERED. ` +
              'The claim boundary is not allowed to delete a measurement.');
          eng = onDisk;
        } else {
          log(`${d.id}: no StructuredOutput and nothing recoverable from ${d.out_dir}.`);
        }
      }
      // Working-kernel rounds and enabling prerequisites must reach Verify even when the engineer reports a
      // regression.
      if (!shouldVerifyDirection(
        d, eng, WORKING_KERNEL || PROMOTION_METRIC === 'changed_kernel', metricScore(eng))) {
        return { d, eng, ver: null };
      }
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
          // Analyze's file whitelist for verify_engineer step 5; sent as UNDECLARED when absent, never omitted.
          MODIFIABLE_FILES: (analysis && Array.isArray(analysis.modifiable_files)
            && analysis.modifiable_files.length)
            ? analysis.modifiable_files : 'UNDECLARED',
          // Verify applies specialty-specific gates (see verify_engineer.md step 4c: a
          // `distributed` patch can be numerically correct and still deadlock).
          ...(d.specialty ? { SPECIALTY: d.specialty } : {}),
          MANDATORY_ARMS: d.mandatory_arms || [],
          ENGINEER_ARMS_RUN: (eng && eng.arms_run) || [],
          TARGET_SHAPE: d.target_shape || null,
          TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
          STRICT_AUTONOMY, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
          REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD, LAUNCH_TARGET,
          ACCURACY_METRIC, ACCURACY_THRESHOLD,
          // The engineer's declared activation, verbatim; UNDECLARED when absent, so Verify does not assume
          // default-on.
          ACTIVATION: (eng && eng.activation) ? JSON.stringify(eng.activation) : 'UNDECLARED',
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
          // Strict Verify sees only out-of-band file hashes, never a reference path, which still catches exact
          // or comment-only copies. Non-strict capability mode keeps the path input.
          ...(CAPABILITY_EVAL && KNOWN_REFERENCE_HASHES.length
            ? { KNOWN_REFERENCE_HASHES } : {}),
          ...(CAPABILITY_EVAL && !STRICT_AUTONOMY && KNOWN_REFERENCE_PATHS.length
            ? { KNOWN_REFERENCE_PATHS: KNOWN_REFERENCE_PATHS.join(' ') } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}`, schema: VERIFY_SCHEMA }
      ).then((ver) => ({ d, eng, ver, patch }));
    }
  );

  const clean = results.filter(Boolean);

  // Unbacked claim: a measured win whose declared patch does not exist (e.g. the effect lived only in
  // bench flags). Verify reports apply_failed; name it as a reporting defect, not a failed idea.
  for (const r of clean) {
    if (!CLAIM.unbacked(r)) continue;
    r.unbacked = true;
    log(`${r.d.id}: CLAIM NOT BACKED BY A PATCH — the engineer claims ` +
        `${primSpeedup(r.eng).toFixed(4)}x but ${r.patch} does not apply. This is a REPORTING ` +
        'failure, not a null result: if the effect was real it came from flags or an unsaved edit ' +
        'and has been lost. Do NOT record this direction as "tried, did not work" — record it as ' +
        'unmeasured, and re-dispatch it with the patch written first.');
  }

  // --- ACTIVATION: did the candidate's code actually RUN? ---
  // An unexercised patch gets its own outcome and is never filed as a null result; unknown counts as no.
  for (const r of clean) {
    if (!r.ver || r.ver.status === 'apply_failed') continue;
    const act = String(r.ver.activation_confirmed || 'unknown').toLowerCase();
    if (act === 'yes') continue;
    r.inactive = act;
    const pct = ((metricScore(r.ver) - 1) * 100);
    log(`INACTIVE ${r.d.id}: the patched code path was ${act === 'no' ? 'NOT executed' : 'NOT PROVEN to execute'} ` +
        `during the measured run, so its ${Number.isFinite(pct) ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : 'result'} ` +
        `is VOID — not a negative result. Do NOT record this direction as tried-and-failed; it has ` +
        `not been tried. Evidence: ${r.ver.activation_evidence || '(none supplied)'}`);
  }

  // --- ARTIFACT DISTINCTNESS: were the two arms the same binary? ---
  // E.g. a JIT switch outside the cache key. Identical artifacts are a void experiment, never a null.
  // Only equality is fatal; `n/a` (not a JIT candidate) and `unknown` (proof not runnable) are reported.
  for (const r of clean) {
    if (!r.ver || r.ver.status === 'apply_failed' || r.inactive) continue;
    const ad = String(r.ver.artifact_distinct || '').toLowerCase();
    const hb = String(r.ver.artifact_hash_base || '').trim();
    const hc = String(r.ver.artifact_hash_candidate || '').trim();
    const sameHash = hb && hc && hb === hc;
    if (ad !== 'no' && !sameHash) {
      if (ad === 'unknown' || (!ad && (hb || hc))) {
        log(`ARTIFACT DISTINCTNESS UNPROVEN ${r.d.id}: base=${hb || '(none)'} candidate=${hc || '(none)'}. ` +
            `The result stands, but a JIT candidate whose arms were never shown to differ cannot ` +
            `close an axis — do not record a 1.000 here as a null result.`);
      }
      continue;
    }
    r.inactive = 'no';
    r.same_artifact = true;
    const pct = ((metricScore(r.ver) - 1) * 100);
    log(`SAME BINARY ${r.d.id}: base and candidate resolved to the SAME compiled artifact` +
        `${sameHash ? ` (${hb})` : ''}, so its ` +
        `${Number.isFinite(pct) ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : 'result'} measures ` +
        `nothing about the idea — it is VOID, not a null. The switch did not enter the compiler's ` +
        `cache key. Re-run with the switch anchored in the OUTER traced scope before this direction ` +
        `is called tried.`);
  }

  // Attribution runs before the verified filter because it can reject; decided for every verified
  // direction, not only self-described fusions.
  for (const r of clean) {
    if (!(r.ver && verifyCompleted(r.ver) &&
      String(r.ver.correctness || '').toLowerCase().startsWith('pass') && !r.inactive)) continue;
    // An enabling step makes no win claim, so there is no win to attribute to the wrong kernel. Its
    // bar is functional acceptance plus its cost budget, both applied below.
    if (stepRoleOf(r.d) === 'enabling') continue;
    const at = attributionVerdict(r.ver, { geomean: metricScore(r.ver), metric: ATTRIBUTION_METRIC });
    r.attribution_state = at.state;
    if (at.caveat) r.attribution_caveat = at.caveat;
    if (at.reject) r.attribution_rejected = at.state;
  }

  // Evaluate the full terminal contract after activation/artifact/attribution and before promotion.
  // Under working_kernel an incomplete-but-functional artifact may be committed, but cannot close its
  // rung or satisfy autonomy.
  for (const r of clean) {
    if (stepRoleOf(r.d) !== 'terminal' || !r.d.target_shape) continue;
    const av = autonomyAcceptanceVerdict(r.d, r.ver, {
      functionalRequirements: functionalRequirementsFor(r.d),
      launchTarget: LAUNCH_TARGET,
      targetGuards: TARGET_GUARDS,
      regressionGuards: REGRESSION_GUARDS,
      promotionMetric: PROMOTION_METRIC,
      fallbackScore: metricScore(r.ver),
      requirePaired: STRICT_AUTONOMY,
      requiredPairs: REQUIRED_PAIRS, requiredPairsByGuard: REQUIRED_PAIRS_BY_GUARD,
      requireOverlap: REQUIRE_OVERLAP,
      requireAttribution: REQUIRE_ATTRIBUTION,
      positiveControlRan: PC_RAN,
    });
    r.autonomy_contract = av;
    if (av.accepted) {
      log(`AUTONOMY CANDIDATE ${r.d.id}: evidence clears the contract at ${av.score.toFixed(4)}x; ` +
          `acceptance remains pending until this exact patch is committed to canonical.`);
    } else if (av.applicable) {
      const msg = `${r.d.id}: ${av.reasons.join('; ') || 'terminal did not clear the acceptance score'}`;
      ACCEPTANCE_CAVEATS.push(msg);
      log(`AUTONOMY INCOMPLETE ${msg}`);
    }
  }

  // Enabling steps are judged before the speed filter, which they cannot pass by design: judged on
  // function, committed on function, cost carried as debt (see <<REPLAY:enabling_step>>).
  const enablingKeeps = [];
  for (const r of clean) {
    if (stepRoleOf(r.d) !== 'enabling' || r.inactive) continue;
    const ev = enablingVerdict(r.d, r.ver, primSpeedup(r.ver), functionalRequirementsFor(r.d));
    r.enabling_verdict = ev;
    if (ev.commit && r.patch) {
      enablingKeeps.push(r);
      log(`ENABLING KEEP ${r.d.id}: ${ev.reason}`);
    } else {
      log(`ENABLING REJECT ${r.d.id}: ${ev.reason}${ev.commit && !r.patch ? ' (and it produced no patch to keep)' : ''}`);
      CHAIN_CAVEATS.push(`${r.d.id} [enabling, not kept]: ${ev.reason}`);
    }
  }

  const verified = clean.filter(r => r.ver && verifyCompleted(r.ver) &&
    functionalAcceptance(r.ver, functionalRequirementsFor(
      r.d, WORKING_KERNEL ? 'commit' : 'acceptance')).pass &&
    // Under working_kernel the gate is "does it run": a terminal fusion is slower the first time it
    // lands, and a >1.0 filter would delete it before it can be committed. Speedup mode keeps >1.0.
    (WORKING_KERNEL || metricScore(r.ver) > 1.0) && !r.inactive &&
    !r.attribution_rejected &&
    (WORKING_KERNEL || (guardContract(r.ver, TARGET_GUARDS, REGRESSION_GUARDS, primSpeedup(r.ver)).complete &&
      guardContract(r.ver, TARGET_GUARDS, REGRESSION_GUARDS, primSpeedup(r.ver)).regression_pass)) &&
    // A strict speedup wave may not promote an acceptance-terminal whose launch/overlap/guard
    // contract is incomplete. A working-kernel wave still keeps a functional partial artifact.
    (WORKING_KERNEL || !STRICT_AUTONOMY || !r.d.target_shape ||
      (r.autonomy_contract && r.autonomy_contract.complete)) &&
    // Enabling steps never compete for the round win; they have their own commit path.
    stepRoleOf(r.d) !== 'enabling');

  // Rejection by exclusion is invisible, so print rejected directions with their number.
  for (const r of clean) {
    if (!r.attribution_rejected) continue;
    log(`ATTRIBUTION ${r.attribution_rejected.toUpperCase()} ${r.d.id}: ${r.attribution_caveat}`);
    ATTRIBUTION_CAVEATS.push(`${r.d.id} [${r.attribution_rejected}]: ${r.attribution_caveat}`);
  }
  for (const r of clean) {
    if (r.attribution_rejected || !r.attribution_caveat) continue;
    log(`ATTRIBUTION ${String(r.attribution_state).toUpperCase()} ${r.d.id}: ${r.attribution_caveat}`);
    if (r.attribution_state !== 'attributed') {
      ATTRIBUTION_CAVEATS.push(`${r.d.id} [${r.attribution_state}]: ${r.attribution_caveat}`);
    }
  }

  // plagiarized and harness_modified are already excluded, but exclusion is invisible; print them with
  // their speedup, since they are findings about the run.
  for (const r of clean) {
    const st = r.ver && r.ver.status;
    if (st !== 'plagiarized' && st !== 'harness_modified') continue;
    const pct = ((metricScore(r.ver) - 1) * 100);
    log(`${st.toUpperCase()} ${r.d.id}: REJECTED as a win despite ` +
        `${Number.isFinite(pct) ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : 'an unreadable number'}. ` +
        `${st === 'plagiarized'
          ? 'The patch contains a file byte-identical to a known reference — the result was imported, not derived.'
          : 'The patch edited the measurement harness, so subject and instrument changed together.'} ` +
        `${r.ver.notes || ''}`);
  }

  // Mark wins with weak provenance as provisional. This does NOT reject the result; the conditions
  // travel with the number into the report.
  for (const r of verified) {
    const claimPct = (metricScore(r.ver) - 1) * 100;
    const reps = Number(r.ver.reps);
    const nullArm = Number(r.ver.null_arm_pct);
    const reasons = [];
    if (!Number.isFinite(reps) || reps < 5) reasons.push(`reps=${Number.isFinite(reps) ? reps : 'unreported'} (<5)`);
    if (!Number.isFinite(nullArm)) reasons.push('no null arm');
    else if (claimPct <= Math.abs(nullArm)) reasons.push(`claim ${claimPct.toFixed(2)}% <= null arm ${Math.abs(nullArm).toFixed(2)}%`);
    // Mode-aware, target-scoped conditioning of the raw readings: an under-sampled bimodal target guard,
    // or a win read only off a skew guard, becomes a provisional reason. Never rejects.
    const ga = analyzeGuards(r.ver.paired_readings, TARGET_GUARDS, 10);
    if (ga.available) {
      r.guard_analysis = ga;
      for (const c of ga.caveats) { reasons.push(c); GUARD_CAVEATS.push(`${r.d.id}: ${c}`); }
    }
    if (reasons.length) {
      r.provisional = reasons.join('; ');
      log(`PROVISIONAL ${r.d.id}: +${claimPct.toFixed(2)}% claimed but ${r.provisional}. ` +
          `Carried forward, but it must be re-measured with >=5 interleaved reps and a null arm ` +
          `before it is reported as a result.`);
    }
    // Overlap, judged for every verified direction. Never rejects; see overlapVerdict.
    const ov = overlapVerdict(r.ver, { geomean: metricScore(r.ver) });
    r.overlap_state = ov.state;
    if (ov.caveat) {
      r.overlap_caveat = ov.caveat;
      log(`OVERLAP ${ov.state.toUpperCase()} ${r.d.id}: ${ov.caveat}`);
      if (ov.state !== 'measured_none') OVERLAP_CAVEATS.push(`${r.d.id} [${ov.state}]: ${ov.caveat}`);
    }
    // And the objective. In a controlled wave this is a no-op on every result.
    const oj = objectiveVerdict(OBJECTIVE, PC_RAN, metricScore(r.ver));
    r.objective_state = oj.state;
    if (oj.caveat) {
      r.objective_void = true;
      r.objective_caveat = oj.caveat;
      log(`TIMING VOID ${r.d.id}: ${oj.caveat}`);
      OBJECTIVE_CAVEATS.push(`${r.d.id}: ${oj.caveat}`);
    }
    // Launch shape, judged for every verified direction. Never rejects; records the count and whether the
    // target shape was reached. See launchVerdict.
    const directionLaunchTarget = Number(r.d && r.d.target_shape && r.d.target_shape.launches);
    const lv = launchVerdict(r.ver,
      Number.isFinite(directionLaunchTarget) ? directionLaunchTarget : LAUNCH_TARGET,
      (r.d && r.d.target_shape && r.d.target_shape.stages_fused) || []);
    r.launch_state = lv.state;
    r.launch_shape_met = lv.shape_met;
    if (Number(r.ver && r.ver.launch_shape && r.ver.launch_shape.launches_cand) <= LAUNCH_TARGET) {
      TWO_LAUNCH_REACHED = true;
    }
    if (lv.caveat) {
      r.launch_caveat = lv.caveat;
      log(`LAUNCH ${lv.state.toUpperCase()} ${r.d.id}: ${lv.caveat}`);
      if (lv.state !== 'met') LAUNCH_CAVEATS.push(`${r.d.id} [${lv.state}]: ${lv.caveat}`);
    }
  }

  // (d) Build the candidate list; integrate if >= 2 verified. geomean is the primary metric (time-
  // weighted with a workload spec, else unweighted); the raw unweighted geomean is kept for the report.
  let candidates = verified.map(r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: metricScore(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch,
    touched_files: Array.isArray(r.ver.touched_files) ? r.ver.touched_files : [],
    // Chain id (terminal rung), carried onto the shelf so a later commit on the same chain does not age
    // this candidate off. Empty for a standalone optimisation.
    chain_id: String((r.d && (r.d.enables || r.d.roadmap_rung)) || '').trim(),
  }));

  // What the shelf offers this round; stale and unknown entries are logged, not dropped silently.
  const shelfPick = shelfEligible(shelf, absorbedByRound, SHELF_OFFER_K);
  if (shelfPick.offer.length) { SHELF_STATS.rounds_offered++; SHELF_STATS.offers += shelfPick.offer.length; }
  if (shelf.length) {
    log(`SHELF r${round}: ${shelf.length} shelved, ${shelfPick.eligible.length} still apply, ` +
        `offering ${shelfPick.offer.length} (K=${SHELF_OFFER_K})` +
        (shelfPick.offer.length ? `: ${shelfPick.offer.map((e) => `${e.id}@${e.geomean.toFixed(3)}x`).join(', ')}` : '') +
        (shelfPick.stale.length ? `; ${shelfPick.stale.length} aged out (a later winner touched their files: ` +
          `${shelfPick.stale.map((e) => `${e.id}[${e.clash.join(',')}]`).join('; ')})` : '') +
        (shelfPick.unknown.length ? `; ${shelfPick.unknown.length} withheld for UNKNOWN footprint ` +
          `(${shelfPick.unknown.map((e) => e.id).join(', ')}) — their verify reported no touched_files, ` +
          `so whether they conflict is unknown, and unknown is not orthogonal` : ''));
  }

  let integrate = null;
  // One verified direction plus a shelved offer is a real merge opportunity — historically this
  // round would have skipped the merge phase entirely and the shelved work would age out unused.
  if (!STRICT_AUTONOMY &&
      (verified.length >= 2 || (verified.length >= 1 && shelfPick.offer.length > 0))) {
    phase('Merge');
    integrate = await agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        GPU_ID: GPU_RESOURCE.specForIndex(0), SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, STRICT_AUTONOMY,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          files: r.d.focus_files || [], patch: r.patch })),
        // Earlier verified losers whose files nothing has since touched. Offers, not obligations; the
        // integrator says when one does not combine.
        SHELF_PATCHES: shelfPick.offer.map((e) => ({
          id: e.id, title: e.title, specialty: e.specialty, patch: e.patch,
          files: e.files, verified_geomean_when_cut: e.geomean,
          cut_against_round: e.base_round, shelved_round: e.shelved_round,
        })),
        SHELF_NOTE: shelfPick.offer.length
          ? 'SHELF_PATCHES come from earlier rounds. Their `verified_geomean_when_cut` was measured ' +
            'against the CANONICAL of round `cut_against_round`, NOT against the CANONICAL you have ' +
            'now, so it is a reason to try one, never a number you may carry into your result. The ' +
            'file sets were checked mechanically: nothing absorbed since each patch was cut touches ' +
            'a file it touches, so it should still apply. Take one only if you can show it composes; ' +
            'reporting "offered N, none composed, here is why" is a real answer and is how the size ' +
            'of this offer gets tuned.'
          : '',
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    const integPrim = integrate && integrate.best ? promotionScore({
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
      per_case: integrate.best.per_case || [],
    }, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
    primSpeedup({ verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean })) : 0;
    const integGuards = integrate && integrate.best
      ? guardContract(integrate.best, TARGET_GUARDS, REGRESSION_GUARDS, integPrim)
      : { complete: false, regression_pass: false };
    if (integrationEligible(integrate, integPrim,
      Math.max(...candidates.map(c => c.geomean)), integGuards)) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
        // An integration's footprint, if it reported one; otherwise everything that was on the
        // table. See INTEGRATE_SCHEMA.best — the fallback deliberately over-states.
        touched_files: Array.isArray(integrate.best.touched_files) && integrate.best.touched_files.length
          ? integrate.best.touched_files
          : [...new Set([...candidates.flatMap((c) => c.touched_files || []),
                         ...shelfPick.offer.flatMap((e) => e.files)])],
        integrated_from: Array.isArray(integrate.best.patches) ? integrate.best.patches : [],
      });
      const took = shelfPick.offer.filter((e) => (integrate.best.patches || []).includes(e.id));
      SHELF_STATS.hits += took.length;
      SHELF_STATS.hit_ids.push(...took.map((e) => e.id));
      if (took.length) log(`SHELF HIT r${round}: the integration took ${took.map((e) => e.id).join(', ')} ` +
        `from the shelf — cross-round combination, not a re-derivation.`);
      else if (shelfPick.offer.length) log(`SHELF MISS r${round}: ${shelfPick.offer.length} offered, ` +
        `none reported as included. ${integrate.notes ? `Integrator: ${String(integrate.notes).slice(0, 300)}` : ''}`);
    } else if (shelfPick.offer.length) {
      log(`SHELF MISS r${round}: ${shelfPick.offer.length} offered, the integration did not improve ` +
          `on the best individual, so nothing from the shelf was taken.`);
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const winner = candidates[0] || null;
  // Under working_kernel the commit gate is the objective, not speed: commit what runs, so the debug
  // loop is cumulative instead of independent first attempts.
  const winnerResult = winner ? (verified.find(r => r.d.id === winner.id) || null) : null;
  const winnerVer = winnerResult && winnerResult.ver;
  const winnerRuns = !!(winner && runsCleanly(
    winnerVer, functionalRequirementsFor(winnerResult && winnerResult.d, 'commit')));
  let improved = WORKING_KERNEL
    ? winnerRuns
    : !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  let winnerCommitted = false;
  if (WORKING_KERNEL) {
    log(`Round ${round}: objective=working_kernel — commit gate is "does it run", not speed. ` +
        `${winner ? (winnerRuns ? `${winner.id} RUNS (correctness pass, activation confirmed, liveness not failing) ` +
          `-> committed.` : `${winner.id} does not yet run -> not committed; next round continues from the ` +
          `current canonical tree.`) : 'no verified candidate this round.'}`);
  }

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    let commitResult = await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
git rev-parse HEAD > ${EVAL_DIR}/round_${round}/winner_precommit_head.txt
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
${STRICT_AUTONOMY
  ? 'If the patch does not apply exactly, return committed=false. Do NOT hand-merge in strict autonomy.'
  : 'If BOTH `git apply` and `git apply --3way` fail, inspect the patch and apply it manually (edit the'}
${STRICT_AUTONOMY ? '' : `
files to match the patch's intent), then \`add -A\` + commit. The applied source is NOT guaranteed to
match the patch verbatim after a hand-merge, so after committing, RE-RUN the COMMANDMENT correctness
check (cd ${CANONICAL} && the COMMANDMENT CORRECTNESS cmd via gpu_lock); only report committed=true if
it still passes. (When a clean \`git apply\`/\`--3way\` succeeds, correctness was already verified and a
re-check is not required.)`}
Return JSON {committed, current_best_diff, head_before, head_after, exact_patch,
 transaction_valid, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });
    if (STRICT_AUTONOMY) {
      commitResult = await agentT(
        `Independently verify the canonical commit transaction for round ${round}. Do not edit files.
Read ${EVAL_DIR}/round_${round}/winner_precommit_head.txt as head_before and run
\`git -C ${CANONICAL} rev-parse HEAD\` as head_after. Require exactly one commit in
\`head_before..head_after\`. Compute stable patch ids for ${winner.patch} and for
\`git -C ${CANONICAL} diff head_before..head_after\`; they must match. Also require the commit
subject to start with "round ${round} winner:". Return committed=true and exact_patch=true only if
all checks pass. If HEAD is unchanged, return transaction_valid=true, committed=false and
exact_patch=false only when \`git status --porcelain\` is also empty. A missing pre-head, dirty
working tree, multi-commit HEAD, or patch mismatch is
transaction_valid=false. Return JSON
{committed, head_before, head_after, exact_patch, transaction_valid, current_best_diff, note}.`,
        { phase: 'Merge', label: `commit transaction check r${round}`, schema: COMMIT_SCHEMA });
      if (!(commitResult && commitResult.transaction_valid === true &&
            commitResult.committed === true && commitResult.exact_patch === true &&
            String(commitResult.head_before || '').trim() &&
            String(commitResult.head_after || '').trim() &&
            commitResult.head_before !== commitResult.head_after)) {
        throw new Error(
          `STRICT AUTONOMY ABORT: winner commit transaction for round ${round} could not be ` +
          'independently matched to the verified patch. Canonical may have changed; do not continue.');
      }
    }
    const transition = winnerCommitTransition(
      improved, commitResult, winner, winnerResult, STRICT_AUTONOMY);
    winnerCommitted = transition.committed;
    if (!winnerCommitted) {
      improved = false;
      const msg = `WINNER NOT COMMITTED r${round}: ${winner.id} passed selection but the canonical commit ` +
        `did not report committed=true. Cumulative, rung completion and autonomy acceptance stay unchanged.`;
      log(msg);
      if (winnerResult && winnerResult.autonomy_contract && winnerResult.autonomy_contract.accepted) {
        ACCEPTANCE_CAVEATS.push(msg);
      }
    }
    if (winnerCommitted) {
    // A voided reading never becomes the next round's reference: under working_kernel with no control,
    // cumulative and bestPerCase stay put and only the artifact advances.
    const winnerVoid = WORKING_KERNEL && !!(verified.find(r => r.d.id === winner.id) || {}).objective_void;
    // Under working_kernel a committed winner can be slower than the tree; the artifact advances but
    // cumulative only moves up. In speedup mode this guard is a no-op.
    if (!winnerVoid && (!WORKING_KERNEL || winner.geomean > cumulative)) {
      cumulative = winner.geomean;
      bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
    }
    finalWinner = winner;
    if (transition.acceptanceReached) {
      AUTONOMY_ACCEPTANCE_REACHED = true;
      log(`AUTONOMY ACCEPTANCE COMMITTED ${winner.id}: the exact contract-clearing patch now ` +
          `exists in canonical at target score ${winner.geomean.toFixed(4)}x.`);
    }
    noImprove = 0;

    // --- (f) Re-profile the new best ------------------------------------
    profileSummary = await agentT(
      roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0), ROUND: round,
        COMMANDMENT, PREVIOUS_METRICS: profileSummary,
      }),
      { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
    const reprofileAnalysisResult = await runProfileAnalysis(
      profileSummary,
      round,
      `analysis_engineer:reprofile r${round}`,
    );
    if (profileSummary) {
      profileSummary = {
        ...profileSummary,
        analysis_result: reprofileAnalysisResult,
      };
    }
    }
  }

  // (e2) Commit the enabling steps: committed because they work and the next step builds on them.
  // Applied after the winner so its patch still applies; a conflicting one is logged and shelved.
  const enablingUnlanded = [];
  const enablingLandedIds = new Set();
  if (enablingKeeps.length) {
    let res = await agentT(
      `You are the TechLead committing round ${round}'s ENABLING steps into the canonical workspace.

These are NOT round winners. Each one passed functional acceptance — it builds, its path is
confirmed to run, correctness passes, and it does not hang — and each is a prerequisite for a
fusion step that has not been written yet. They are expected to be SLOWER on their own. Commit
them so the next round has something to build the consumer half against.

\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
git rev-parse HEAD > ${EVAL_DIR}/round_${round}/enabling_precommit_head.txt
\`\`\`
Then, for EACH patch below IN ORDER, run \`git apply <patch> || git apply --3way <patch>\` and commit
it on its own with the message shown. If a patch fails BOTH plain and --3way apply, do NOT hand-merge
it and do NOT force it: skip it, leave the tree as it was before that patch, and report it in
\`skipped\`. A prerequisite that no longer applies is a prerequisite whose assumptions changed, and
silently reconciling it is how a chain ends up half-applied in a way nobody can see.

${enablingKeeps.map((r) => `- patch: ${r.patch}\n  message: "round ${round} enabling: ${r.d.id} ` +
  `(enables ${r.enabling_verdict.enables}, cost ${r.enabling_verdict.cost_pct}%)"`).join('\n')}

Finally: \`git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff\`
and re-run the COMMANDMENT CORRECTNESS check on ${CANONICAL} via gpu_lock. Report JSON
{committed: [<patch paths that landed>], skipped: [{patch, reason}], correctness_after,
 head_before, head_after, exact_patch, transaction_valid, note}.`,
      { phase: 'Merge', label: `commit enabling r${round}`, schema: obj({
        committed: { type: 'array', items: { type: 'string' } },
        skipped: { type: 'array', items: { type: 'object', additionalProperties: true } },
        correctness_after: { type: 'string' }, note: { type: 'string' },
        head_before: { type: 'string' }, head_after: { type: 'string' },
        exact_patch: { type: 'boolean' }, transaction_valid: { type: 'boolean' },
      }, ['committed']) });
    if (STRICT_AUTONOMY) {
      res = await agentT(
        `Independently verify the enabling commit transaction for round ${round}. Do not edit files.
Read ${EVAL_DIR}/round_${round}/enabling_precommit_head.txt and compare it with
\`git -C ${CANONICAL} rev-parse HEAD\`. If unchanged, report committed:[]; the orchestrator stops so
the patch can be retried. If changed, strict mode has exactly one direction: require exactly one
commit, subject starting "round ${round} enabling:", and a stable patch-id equal to
${enablingKeeps[0].patch}. Then run the COMMANDMENT correctness entry on ${CANONICAL}; only a pass
may return transaction_valid=true, exact_patch=true and
committed:["${enablingKeeps[0].patch}"]. Any missing pre-head,
dirty working tree, unexpected/multiple commit, patch mismatch or correctness failure is transaction_valid=false.
Return {committed, skipped:[], correctness_after, head_before, head_after,
exact_patch, transaction_valid, note}.`,
        { phase: 'Merge', label: `enabling transaction check r${round}`, schema: obj({
          committed: { type: 'array', items: { type: 'string' } },
          skipped: { type: 'array', items: { type: 'object', additionalProperties: true } },
          correctness_after: { type: 'string' }, note: { type: 'string' },
          head_before: { type: 'string' }, head_after: { type: 'string' },
          exact_patch: { type: 'boolean' }, transaction_valid: { type: 'boolean' },
        }, ['committed', 'transaction_valid']) });
      if (!(res && res.transaction_valid === true && res.exact_patch === true &&
            Array.isArray(res.committed) && res.committed.includes(enablingKeeps[0].patch) &&
            String(res.head_before || '').trim() && String(res.head_after || '').trim() &&
            res.head_before !== res.head_after)) {
        throw new Error(
          `STRICT AUTONOMY ABORT: enabling commit transaction for round ${round} is not ` +
          'independently accounted. Canonical may have changed; do not continue.');
      }
    }
    const reportedLanded = new Set(Array.isArray(res && res.committed) ? res.committed : []);
    const enablingCommitCorrect = !!(res && String(res.correctness_after || '').toLowerCase().startsWith('pass'));
    const landed = enablingCommitCorrect ? reportedLanded : new Set();
    if (reportedLanded.size && !enablingCommitCorrect) {
      const msg = `r${round} enabling commits changed canonical but correctness_after did not pass`;
      CHAIN_CAVEATS.push(msg);
      if (STRICT_AUTONOMY) throw new Error(`STRICT AUTONOMY ABORT: ${msg}`);
    }
    if (landed.size && CHAIN_BASELINE == null) {
      CHAIN_BASELINE = { round, cumulative, note:
        'Pinned at the first enabling commit. Every enabling step below makes the canonical tree ' +
        'slower, so the terminal step must state its claim against THIS number, not against the ' +
        'canonical it will actually be measured on.' };
      log(`CHAIN BASELINE PINNED at round ${round}, cumulative=${cumulative.toFixed(4)}x. ` +
          CHAIN_BASELINE.note);
    }
    const enablingAbsorbed = [];
    let enablingChain = '';
    for (const r of enablingKeeps) {
      if (!landed.has(r.patch)) {
        log(`ENABLING NOT LANDED ${r.d.id}: the patch did not apply to the canonical tree. It stays ` +
            'on the shelf; the rung it enables remains owed.');
        CHAIN_CAVEATS.push(`${r.d.id} [enabling, did not apply]: ${r.enabling_verdict.enables} still owed.`);
        enablingUnlanded.push(r);
        continue;
      }
      enablingLandedIds.add(r.d.id);
      CHAIN_DEBT.push({ round, id: r.d.id, enables: r.enabling_verdict.enables,
        cost_pct: r.enabling_verdict.cost_pct });
      CHAIN_CAVEATS.push(`${r.d.id} [enabling, committed r${round}]: ${r.enabling_verdict.reason}`);
      for (const f of (Array.isArray(r.ver && r.ver.touched_files) ? r.ver.touched_files : [])) {
        const nf = shelfFile(f); if (nf) enablingAbsorbed.push(nf);
      }
      if (!enablingChain) enablingChain = String(r.enabling_verdict.enables || '').trim();
    }
    // Record the files the enabling commits took, tagged with their chain, so unrelated shelf entries age
    // while the same chain's consumer half does not.
    if (enablingAbsorbed.length) {
      const prev = absorbedByRound[round];
      const prevFiles = Array.isArray(prev) ? prev : (prev && prev.files) || [];
      const prevChain = (prev && !Array.isArray(prev) && prev.chain_id) || enablingChain;
      absorbedByRound[round] = { files: [...new Set([...prevFiles, ...enablingAbsorbed])], chain_id: prevChain };
    }
    // Committing these does NOT advance cumulative. They are known to be slower; recording them as
    // progress would make the run's headline number a measure of how much overhead it has installed.
    log(`Round ${round}: ${landed.size} enabling step(s) committed on function. cumulative unchanged ` +
        `at ${cumulative.toFixed(4)}x — an enabling step is not a win and must not move the headline.`);
    if (res && res.correctness_after && !String(res.correctness_after).toLowerCase().startsWith('pass')) {
      log(`ENABLING COMMIT WARNING r${round}: correctness after the enabling commits reads ` +
          `"${res.correctness_after}". The canonical tree is the input to every later round; a chain ` +
          'that lands broken is worse than a chain that never lands.');
      CHAIN_CAVEATS.push(`r${round} enabling commit: correctness_after=${res.correctness_after}`);
    }
  }

  // Only landed state can close a staged rung; the commit agent can fail or skip the patch.
  for (const r of clean) {
    if (!(r.d && r.d.roadmap_rung) || r.d.roadmap_rung === 'off_ladder') continue;
    const role = stepRoleOf(r.d);
    let outcome = rungOutcomeOf(r, role);
    const contract = r.autonomy_contract;
    const evidencePass = !r.d.target_shape ? true
      : (STRICT_AUTONOMY ? !!(contract && contract.accepted) : !!(contract && contract.complete));
    outcome = landedRungOutcome(outcome, role, !!r.d.target_shape, evidencePass,
      winnerCommitted, !!(winner && winner.id === r.d.id), enablingLandedIds.has(r.d.id));
    r.rung_outcome = outcome;
    recordRungOutcome(r.d.roadmap_rung, outcome);
    if (outcome === 'measured') LADDER_MEASURED.add(r.d.roadmap_rung);
  }
  {
    const owed = openRungs(LADDER, rungTally);
    if (owed.length) {
      log(`Round ${round}: rungs still owed after evidence + commit gates: ` +
        owed.map((c) => `${c.id}(${c.last_outcome}, ${c.attempts} attempt(s))`).join(', ') +
        '. A staged rung closes only when its exact verified patch exists in canonical.');
    }
  }

  // The outstanding balance, recomputed every round against what has actually been measured.
  {
    const dr = chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED);
    if (dr.caveat) {
      log(`Round ${round}: ${dr.caveat}`);
      if (dr.overdue.length) CHAIN_CAVEATS.push(`r${round} OVERDUE: ${dr.caveat}`);
    }
  }

  const enablingLanded = CHAIN_DEBT.filter((e) => e.round === round).length;
  // Did this round measure anything? Same admissibility standard as promotion. See roundEvidence.
  const ev = roundEvidence(clean, stepRoleOf, rungOutcomeOf);
  if (ev.measured) noEvidence = 0;
  else noEvidence++;
  // On-hardware accounting: reset when any clean direction's switched path executed on the device,
  // else increment. Live under every objective when REQUIRE_HW_ACTIVATION is set.
  if (REQUIRE_HW_ACTIVATION) {
    const touched = clean.some((r) => r && r.ver &&
      String(r.ver.activation_on_hardware || '').toLowerCase() === 'yes');
    if (touched) noHardware = 0;
    else noHardware++;
    if (!touched) {
      log(`Round ${round}: NO candidate reached the device this round (none reported ` +
        `activation_on_hardware=yes). noHardware=${noHardware}/${MAX_NO_HARDWARE} consecutive such rounds. ` +
        'A round banked entirely on compile/static/CPU screens is not on-device progress — it is exactly ' +
        'how a fused ON-path deferred its JIT-trace crash across rounds. At the cap the wave hard-stops so ' +
        'a human fixes the ON-path or the lease, rather than spending another lease on OFF scaffolding.');
    }
    if (noHardware >= MAX_NO_HARDWARE) {
      stopReason = `${noHardware} consecutive round(s) never executed the switched path on hardware ` +
        `(the no-hardware cap, ${MAX_NO_HARDWARE}). ${BUDGET - dispatched} of ${BUDGET} budget unit(s) ` +
        'were left unspent. This is not an exhausted search and not a slow kernel: the ON path never ' +
        'reached a card — it may not even trace/launch there. Fix the activation or the lease before ' +
        'planning another direction; do not bank more compile-only scaffolding.';
      log(`Round ${round}: hard-stop — ${stopReason}`);
      break;
    }
  }
  if (improved) {
    // noImprove was already reset at the commit above.
  } else if (enablingLanded) {
    // A round that landed a prerequisite made progress the headline number cannot show; not charged to
    // noImprove.
    log(`Round ${round}: NOT counted toward noImprove — ${enablingLanded} enabling step(s) landed. ` +
        'A prerequisite that passed functional acceptance and entered the canonical tree is progress ' +
        'the cumulative speedup cannot show.');
  } else if (!ev.measured) {
    // No admissible reading this round: a harness, lease or reporting fault, not charged to noImprove.
    log(`Round ${round}: NOT counted toward noImprove — none of the ${ev.total} direction(s) ` +
        `produced an admissible measurement, so the round produced no evidence about the kernel ` +
        `at all. ${ev.gaps.join('; ')}. noEvidence=${noEvidence}/${MAX_NO_IMPROVE} consecutive ` +
        `such rounds; at the cap the run stops, because the thing to fix is the instrument, not ` +
        `the direction. Fix the lease or the activation before spending more budget.`);
  } else {
    noImprove++;
  }

  // (e3) Shelve this round's verified non-winners with their patches. The integrated candidate is not
  // shelved (it combines shelf entries).
  {
    // An enabling step that passed on function but did not apply is shelved for the next round.
    const unlandedEntries = enablingUnlanded.map((r) => ({
      source: `enabling ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
      geomean: metricScore(r.ver) || 1.0, per_case: (r.ver && r.ver.per_case) || [], patch: r.patch,
      touched_files: Array.isArray(r.ver && r.ver.touched_files) ? r.ver.touched_files : [],
      // Chain tag: the terminal rung this producer half enables. On the shelf, the consumer half that
      // eventually lands must not age this entry off -- they are two halves of the same fusion.
      chain_id: String((r.enabling_verdict && r.enabling_verdict.enables) || (r.d && r.d.enables) || '').trim(),
    }));
    // Correct, activated patches kept out only by the speed gate are shelved too. Exclusions for cause
    // (attribution-rejected, inactive, enabling) stay excluded.
    const inVerified = new Set(verified.map((r) => r.d && r.d.id));
    const slowLosers = clean.filter((r) => r.patch && !r.inactive && !r.attribution_rejected &&
      stepRoleOf(r.d) !== 'enabling' && verifyCompleted(r.ver) &&
      String(r.ver.correctness || '').toLowerCase().startsWith('pass') &&
      !inVerified.has(r.d && r.d.id) && metricScore(r.ver) <= 1.0)
      .map((r) => ({
        source: `slow ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        geomean: metricScore(r.ver) || 1.0, per_case: (r.ver && r.ver.per_case) || [], patch: r.patch,
        touched_files: Array.isArray(r.ver && r.ver.touched_files) ? r.ver.touched_files : [],
        chain_id: String((r.d && (r.d.enables || r.d.roadmap_rung)) || '').trim(),
      }));
    if (slowLosers.length) log(`SHELF r${round}: keeping ${slowLosers.length} correct-but-slower ` +
      `patch(es) [${slowLosers.map((c) => `${c.id}@${c.geomean.toFixed(3)}x`).join(', ')}] that lost only ` +
      `on speed -- they may compose with a later patch, and re-deriving them costs a lease.`);
    const losers = candidates.filter((c) => c !== winner && c.source !== 'integrated' && c.patch)
      .concat(unlandedEntries).concat(slowLosers);
    if (losers.length) {
      const added = shelfAdd(shelf, losers, round, SHELF_MAX);
      shelf = added.shelf;
      const noFiles = losers.filter((c) => !(c.touched_files || []).length);
      SHELF_STATS.shelved += losers.length;
      SHELF_STATS.withheld_unknown_footprint += noFiles.length;
      log(`SHELVED r${round}: ${losers.map((c) => `${c.id}@${c.geomean.toFixed(3)}x`).join(', ')} ` +
          `(${shelf.length}/${SHELF_MAX} on the shelf)` +
          (noFiles.length ? `. ${noFiles.map((c) => c.id).join(', ')} reported no touched_files, so ` +
            `their footprint is UNKNOWN and they will never be offered — unknown is not orthogonal.` : '') +
          (added.evicted.length ? ` Evicted the weakest: ${added.evicted.map((c) => c.id).join(', ')}.` : ''));
    }
    // What CANONICAL took on. Recorded only when the winner was actually committed — a winner that
    // did not clear MIN_IMPROVE changed nothing, so nothing aged.
    if (improved && winner) {
      const took = (winner.touched_files || []).map(shelfFile).filter(Boolean);
      // The object form carries the winner's chain, so the same chain's shelved consumer half is not aged
      // off; merged with this round's enabling absorption, dropping the tag if chains differ.
      if (took.length) {
        const prev = absorbedByRound[round];
        const prevFiles = Array.isArray(prev) ? prev : (prev && prev.files) || [];
        const prevChain = (prev && !Array.isArray(prev) && prev.chain_id) || '';
        const wChain = String(winner.chain_id || '').trim();
        const chain_id = !prevChain ? wChain : (prevChain === wChain ? wChain : '');
        absorbedByRound[round] = { files: [...new Set([...prevFiles, ...took])], chain_id };
      } else log(`SHELF WARN r${round}: the committed winner ${winner.id} reported no touched_files, so ` +
        `nothing was recorded as absorbed and shelved patches will NOT be aged against it. Every ` +
        `offer from here on may be stale in a way this check cannot see.`);
    }
  }

  // A round in which every direction was INACTIVE produced no evidence about the kernel. Its
  // distilled insights are tagged accordingly rather than being trusted as measurements.
  const allInactive = clean.length > 0 && clean.every(r => r.inactive);

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', 'update_memory', 'Distill durable insights + update the hypothesis ledger.', {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ROUND_RESULTS: clean.map(r => ({ id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        expected: r.d.expected_speedup, claimed: r.eng ? r.eng.speedup_geomean : 0,
        verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
        // The rung this direction ran under, so the ledger row settles the right rung.
        roadmap_rung: r.d.roadmap_rung || null, rung_deviation: r.d.rung_deviation || null,
        notes: r.eng ? r.eng.notes : '' })),
      INTEGRATE: integrate, WINNER: winner ? { source: winner.source, geomean: winner.geomean } : null,
      IMPROVED: improved, REPROFILE_SHIFT: profileSummary ? profileSummary.shift_note : '',
      PRIOR_HISTORY: history,
      // The ladder and what is unspent on it, written into memory for the next wave.
      ROADMAP_LADDER: LADDER, LADDER_DISPATCHED: [...dispatchedRungs],
      LADDER_COMPLETED: [...LADDER_MEASURED],
      // Ladder minus rungs that produced a number, with attempt counts and last outcome; written as
      // open_rungs in STATE.json for the next wave's analyze fast path.
      OPEN_RUNGS: openRungs(LADDER, rungTally),
      // Outstanding chain debt, carried to the next wave.
      ...(CHAIN_DEBT.length ? {
        CHAIN_DEBT: chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED).open,
        CHAIN_BASELINE,
      } : {}),
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase,
        // The shelf travels to the next wave via STATE.json (setup.prior_state.shelf) and must be copied
        // verbatim; an entry without its patch path or files cannot be offered or conflict-checked.
        SHELF: shelf, ABSORBED_FILES: absorbedByRound } : {}),
      ...(SHARED_KB ? { SHARED_KB, TARGET_LANGUAGE } : {}),
    }),
    { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (mem.insights) {
      const before = insightBook.length;
      const merged = mergeInsights(insightBook, mem.insights, round, allInactive, 40);
      insightBook = merged.book;
      history.insights = renderInsights(insightBook);
      const added = insightBook.filter((e) => e.first_round === round).length;
      log(`Round ${round} memory: ${added} new insight(s), ${before} carried forward, ` +
          `${insightBook.length} on the board` + (allInactive ? ' (this round produced NO evidence — its entries are tagged FROM-VOID-ROUND)' : ''));
      // Never silent. A board that drops findings without saying so is the bug this replaced.
      for (const e of merged.evicted) log(`INSIGHT EVICTED (board full, last seen r${e.last_round}): ${e.text}`);
    }
    // Log knowledge deltas one line each; non-generalizing ones are kept but marked.
    if (Array.isArray(mem.knowledge_delta)) {
      for (const k of mem.knowledge_delta) {
        if (!k || !k.fact || !k.evidence) continue;
        KNOWLEDGE_DELTA.push({ round, ...k });
        log(`KNOWLEDGE DELTA r${round} [${k.generalizes ? `card: ${k.card || 'UNPLACED'}` : 'this operator only'}]: ` +
            `${k.fact} | evidence: ${k.evidence}` +
            (k.contradicts ? ` | would otherwise have been read as: ${k.contradicts}` : ''));
      }
    }
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  history.rounds.push({
    round,
    directions: directions.map(d => ({ id: d.id, title: d.title, specialty: d.specialty })),
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
      // Carried so the next round can tell "tried and failed" from "never actually ran".
      inactive: r.inactive || null })),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0 } : null,
    winner: winner ? { source: winner.source, geomean: winner.geomean } : null,
    improved, cumulative,
  });
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}, ` +
      `cumulative=${cumulative.toFixed(2)}x, measured ${ev.measured}/${ev.total} direction(s), ` +
      `noImprove=${noImprove}/${MAX_NO_IMPROVE}, noEvidence=${noEvidence}/${MAX_NO_IMPROVE}, ` +
      `budget=${dispatched}/${BUDGET}`);
  if (STRICT_AUTONOMY && AUTONOMY_ACCEPTANCE_REACHED) {
    stopReason = `strict autonomy acceptance was reached in round ${round}; the accepted terminal ` +
      `artifact is committed and ${BUDGET - dispatched} of ${BUDGET} budget unit(s) remain unspent.`;
    break;
  }
}

// Record which stop fired, in loop order: the budget counts directions, not GPU leases, so the
// artifacts alone cannot tell "budget spent" from "hardware left".
if (!stopReason) {
  const left = BUDGET - dispatched;
  if (dispatched >= BUDGET) {
    stopReason = `the direction budget was exhausted (${dispatched}/${BUDGET} charged over ${round} round(s)). ` +
      'The budget counts directions, not GPU leases; a round of directions that never reached the ' +
      'hardware costs the same as a round that did.';
  } else if (!WORKING_KERNEL && noEvidence >= MAX_NO_IMPROVE) {
    stopReason = `${noEvidence} consecutive round(s) produced no admissible measurement, which is the ` +
      `max_no_improve cap (${MAX_NO_IMPROVE}). ${left} of ${BUDGET} budget unit(s) were left unspent. ` +
      'This is an instrument failure, not an exhausted search: the next wave should fix the lease ' +
      'or the activation path before it plans another direction.';
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

if (MODE === 'mega' && megaUnsafeTimeout) {
  log(`MEGA SAFE STOP: ${megaUnsafeTimeout.label || 'agent'} may still be running after the ` +
    `orchestration timeout. No recovery, fallback, finalist GPU work, report agent or validation ` +
    `agent will be started in this invocation.`);
  return {
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
  };
}

let megaSelection = null;
let megaSelectionContractPassed = false;
let FINAL_WORKSPACE = CANONICAL;
if (MODE === 'mega') {
  phase('Validate');
  if (!megaMeasurementCalibration.ready) await recoverMegaCalibration('before-final-selection');
  const provisional = selectMegaCandidate(megaCandidateRegistry, MEGA_TIE_NOISE_PCT);
  const orderedEligible = provisional.selected
    ? [provisional.selected, ...provisional.eligible.filter((c) => c.id !== provisional.selected.id)]
    : provisional.eligible;
  const ranked = orderedEligible.slice(0,
    MEGA_PRODUCTION ? MEGA_FINAL_FALLBACK_K : MEGA_FINAL_TOP_K);
  const finalistOrder = ranked;
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
          GPU_ID: GPU_RESOURCE.specForIndex(0), SKILL_DIR: WORKFLOW_DIR,
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
          REQUIRE_GRAPH_CAPTURE: REQUIRE_GRAPH_CAPTURE ? '1' : '0',
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
            identity.source_head === finalistOrder[i].head &&
            identity.materialized_head === attempt.selected_head)) {
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
}

if (MODE === 'mega' && megaUnsafeTimeout) {
  log(`MEGA SAFE STOP: finalist agent ${megaUnsafeTimeout.label || ''} timed out; no fallback or ` +
    `second validation starts while that agent may still own resources.`);
  return {
    mode: MODE, mega_deliverable: false, mega_selected_candidate: '',
    mega_selected_source: '', mega_candidate_registry: megaCandidateRegistry,
    mega_note: 'finalist timeout: no overlapping fallback was launched',
    eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME, final_speedup: 0, final_geomean: 0,
    validation_status: 'agent_timeout', rounds: round, budget_used: dispatched,
    budget_total: BUDGET, stop_reason: 'finalist agent timeout', report_path: '', final_patch: '',
  };
}

// PHASE: Final report (TechLead)
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    EVAL_DIR, WORKSPACE: FINAL_WORKSPACE, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    ...(MODE === 'mega' ? {
      MEGA_CANDIDATE_REGISTRY: megaCandidateRegistry,
      MEGA_SELECTION: megaSelection || {},
    } : {}),
    STRICT_AUTONOMY, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
    LAUNCH_TARGET, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
    REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    ACCURACY_METRIC, ACCURACY_THRESHOLD, AUTONOMY_ACCEPTANCE_REACHED,
    ...(ACCEPTANCE_CAVEATS.length ? { ACCEPTANCE_CAVEATS } : {}),
    // Why the loop ended, for the next wave's starting conditions.
    STOP_REASON: stopReason,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
    PROFILE_SUMMARY: profileSummary,
    // The report's provenance paragraph must cite this, not recollection. UNRECORDED means the
    // report may not assert anything about what is or is not in the baseline.
    PRIOR_ART_SWEEP: PRIOR_ART_RECORDED ? JSON.stringify(PRIOR_ART) : 'UNRECORDED',
    // A control that passed but read HIGH is a standing caveat on every number below it. It is
    // supplied here so the Measurement-confidence section states it instead of quietly dropping it.
    ...(PC_OVERSHOOT ? { POSITIVE_CONTROL_OVERSHOOT: PC_OVERSHOOT } : {}),
    ...(PC_UNDERSHOOT ? { POSITIVE_CONTROL_UNDERSHOOT: PC_UNDERSHOOT } : {}),
    // Shelf activity this wave (offered vs composed), reported once anything was shelved; K is sized from it.
    ...(SHELF_STATS.rounds_offered || SHELF_STATS.shelved ? { SHELF_ACTIVITY: SHELF_STATS } : {}),
    ...(OVERLAP_CAVEATS.length ? { OVERLAP_CAVEATS } : {}),
    // Launch-shape caveats, present only when some candidate missed the target or did not report;
    // TWO_LAUNCH_REACHED says whether any verified candidate hit it.
    ...(LAUNCH_CAVEATS.length ? { LAUNCH_CAVEATS, TWO_LAUNCH_REACHED } : {}),
    // Guard-scope caveats: an under-sampled bimodal target route, or a win read only off a skew rail.
    // Present only when something was off, so a wave measured cleanly on target is unchanged.
    ...(GUARD_CAVEATS.length ? { GUARD_CAVEATS } : {}),
    ...(ATTRIBUTION_CAVEATS.length ? { ATTRIBUTION_CAVEATS } : {}),
    // A half-built fusion: committed prerequisites whose chain never closed, so the cumulative number is
    // below the start for a reason the number does not show.
    ...(CHAIN_CAVEATS.length ? { CHAIN_CAVEATS } : {}),
    ...(CHAIN_DEBT.length ? {
      CHAIN_DEBT_FINAL: chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED).open,
      CHAIN_BASELINE,
      CHAIN_NOTE:
        'Enabling steps are committed on FUNCTION (builds, path taken, correct, no deadlock), not ' +
        'on speed, because the producer half of a fusion cannot be faster on its own. Any rung ' +
        'still listed in CHAIN_DEBT_FINAL is a fusion that was started and not closed: state the ' +
        'cost it left in the tree, and state that the speedup for that chain is NOT YET MEASURED ' +
        'rather than reporting the degraded number as the result. CHAIN_BASELINE is what the ' +
        'terminal step must be compared against — the canonical tree already contains the ' +
        'overhead, so measuring against the canonical would credit the chain with removing its own ' +
        'cost.',
    } : {}),
    // Only on a non-default objective: under working_kernel the wave's timings are withdrawn and the
    // deliverable is whether an artifact ran.
    ...(WORKING_KERNEL ? { OBJECTIVE, OBJECTIVE_NOTE:
      'This wave was run for a WORKING FUSED ARTIFACT, not for speed. One direction per round, the ' +
      'commit gate was "does it run", and the no-improve stop was disabled because a debug round is ' +
      'non-improving by construction. Do NOT read a speedup out of this wave' +
      (PC_RAN ? '.' : ' — it ran no positive control, so every timing reading in it is void.') } : {}),
    ...(OBJECTIVE_CAVEATS.length ? { OBJECTIVE_CAVEATS } : {}),
    // Knowledge deltas in the report, split into generalizing (card edit owed) and not, each with its
    // evidence.
    ...(KNOWLEDGE_DELTA.length ? { KNOWLEDGE_DELTA, KNOWLEDGE_DELTA_NOTE:
      'Facts this wave established about the HARDWARE/TOOLCHAIN, not about this kernel. Give them ' +
      'their own section titled "Knowledge deltas — merge into knowledge/". List the generalizing ' +
      'ones first with their target card and their evidence verbatim; list the operator-specific ' +
      'ones under a separate heading and say plainly that they are not card material. This workflow ' +
      'does not edit knowledge/ itself.' } : {}),
    ...(TASK_GRAPH_CAVEAT ? { TASK_GRAPH_CAVEAT } : {}),
    ...(PIPE_TABLE_CAVEAT ? { PIPE_TABLE_CAVEAT } : {}),
    // Recomputed against the final dispatch record: which rungs the wave never reached.
    ...(LADDER.length ? { ROADMAP_LADDER_FINAL: JSON.stringify(
      roadmapLadderGate(LADDER, [], LADDER_MEASURED)).slice(0, 3000) } : {}),
    ...(REQUIRE_TASK_GRAPH && analysis && analysis.task_graph
      ? { TASK_GRAPH: JSON.stringify(analysis.task_graph).slice(0, 4000) } : {}),
    ...(REQUIRE_TASK_GRAPH && analysis && analysis.resource_timeline
      ? { RESOURCE_TIMELINE: JSON.stringify(analysis.resource_timeline).slice(0, 3000) } : {}),
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA,
    ...(MODE === 'mega' && MEGA_PRODUCTION
      ? { timeout_ms: 300000, timeout_marker: true, max_retries: 1 } : {}) });
const reportReady = !!(report && !report.__agent_timed_out &&
  String(report.report_path || '') && String(report.final_patch || ''));
if (MODE === 'mega' && report && report.__agent_timed_out) megaUnsafeTimeout = report;
if (MODE === 'mega' && megaUnsafeTimeout) {
  return {
    mode: MODE, mega_deliverable: false, mega_selected_candidate: '',
    mega_selected_source: '', mega_candidate_registry: megaCandidateRegistry,
    mega_note: 'report timeout: no identity validation or state writer was started afterward',
    eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME, final_speedup: 0, final_geomean: 0,
    validation_status: 'agent_timeout', rounds: round, budget_used: dispatched,
    budget_total: BUDGET, stop_reason: 'report agent timeout', report_path: '', final_patch: '',
  };
}

// PHASE: Director validation + arbitration
phase('Validate');
const shouldRunFinalValidation = !(MODE === 'mega' && MEGA_PRODUCTION) ||
  (reportReady && !!finalWinner && megaSelectionContractPassed);
const validation = !shouldRunFinalValidation ? null : await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: FINAL_WORKSPACE, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0),
    APPLY_TO_ORIGINAL, COMMANDMENT,
    FINAL_PATCH: reportReady ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    STRICT_AUTONOMY, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
    LAUNCH_TARGET, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
    REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    ACCURACY_METRIC, ACCURACY_THRESHOLD, AUTONOMY_ACCEPTANCE_REACHED,
    WORKFLOW_REVISION_START,
    ...(MODE === 'mega' && finalWinner ? {
      VALIDATION_TIER: MEGA_PRODUCTION ? 'identity_only' : 'full',
      MEGA_SELECTION_EVIDENCE: megaSelection,
      SELECTED_SOURCE_TREE: finalWinner.source_tree,
      SELECTED_SOURCE_HEAD: finalWinner.source_head,
      SELECTED_MATERIALIZED_HEAD: finalWinner.materialized_head,
    } : {}),
    ...(STRICT_AUTONOMY && KNOWN_REFERENCE_HASHES.length
      ? { KNOWN_REFERENCE_HASHES } : {}),
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA,
    ...(MODE === 'mega' && MEGA_PRODUCTION
      ? { timeout_ms: 300000, timeout_marker: true, max_retries: 1 } : {}) });
if (MODE === 'mega' && validation && validation.__agent_timed_out) {
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

const finalGeomean = validation ? validation.director_verified_speedup_geomean : cumulative;
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = validation && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
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
const strictProofFailure = STRICT_AUTONOMY &&
  (!AUTONOMY_ACCEPTANCE_REACHED || !(validation && validation.autonomy_acceptance_confirmed === true) ||
   !(validation && validation.workflow_unchanged === true));
// Correctness/apply failures outrank the more specific autonomy label; never soften `flagged` into
// `autonomy_incomplete`. Scoped regression guards are vetoes in every mode, not only strict mode.
const finalValidationStatus = finalStatusVerdict(directorStatus, scopedGuardFailure, strictProofFailure);
// Surface leaked lease jobs in the completion line: they keep running after the report and contend
// with the measurements just accepted.
if (validation && validation.orphan_leases_swept > 0) {
  log(`WARNING: ${validation.orphan_leases_swept} orphaned GPU lease(s) were still alive at closeout ` +
      `and have been killed. Some direction backgrounded a lease job — its measurement is NOT in ` +
      `the report, and it may have contended with the validated numbers.`);
}

// Mega provenance/topology labels never change the quality bar. Every source must pass the same final
// correctness/guard contract, and the same frozen-baseline score chooses what ships.
let mega_deliverable = null, mega_note = '';
if (MODE === 'mega') {
  const correctnessClean = directorStatus === 'accepted' &&
    reportReady &&
    String(validation && validation.correctness || '').toLowerCase().startsWith('pass') &&
    validation && validation.materialization_matches === true &&
    validation.final_patch_verified === true &&
    (APPLY_TO_ORIGINAL !== 'true' ||
      String(validation.applied_to_original || '').toLowerCase() === 'true') &&
    String(validation.selected_source_head || '') === String(finalWinner && finalWinner.source_head || '') &&
    String(validation.selected_materialized_head || '') ===
      String(finalWinner && finalWinner.materialized_head || '') &&
    !strictProofFailure &&
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
}

log(`COMPLETE. ${KERNEL_NAME}: verified ${TARGET_GUARDS.length ? `target-guard ${PROMOTION_METRIC}` :
    (HAS_WORKLOAD ? 'time-weighted' : 'geomean')} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${finalValidationStatus}${MODE === 'mega' ? `, mega_deliverable=${mega_deliverable}, selected=${finalWinner ? finalWinner.id : 'none'}` : ''}). Results in ${EVAL_DIR}`);

return {
  mode: MODE,
  ...(MODE === 'mega' ? {
    mega_deliverable,
    mega_selected_candidate: finalWinner ? finalWinner.id : '',
    mega_selected_source: finalWinner ? finalWinner.source : '',
    mega_candidate_registry: megaCandidateRegistry,
    mega_note,
  } : {}),
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
  validation_status: finalValidationStatus,
  autonomy_acceptance_reached: AUTONOMY_ACCEPTANCE_REACHED,
  autonomy_acceptance_confirmed: !!(validation && validation.autonomy_acceptance_confirmed === true &&
    validation.workflow_unchanged === true && directorStatus !== 'flagged' &&
    !scopedGuardFailure && !strictProofFailure),
  acceptance_caveats: ACCEPTANCE_CAVEATS,
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  stop_reason: stopReason,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
};
