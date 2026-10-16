// ===========================================================================
// From here on: mode=bakeoff — the multi-language bake-off orchestrator. Pass
// args.backends to pick the languages, or leave it empty to auto-discover.
// ===========================================================================
if (MODE !== 'bakeoff') {
  throw new Error(`unknown mode='${MODE}'. Use optimize | author | bakeoff.`);
}

// Sibling e2e_workflow dir — we REFERENCE its op_benchmarker role + bench scripts in place (no copy).
const E2E_WF_DIR = String(A.e2e_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/e2e_workflow')).replace(/\/+$/, '');
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');
const EXPECTED_GFX = String(A.expected_gfx || '').trim().toLowerCase();
const EXPECTED_TARGET = String(A.expected_target || '').trim().toLowerCase();
const EXPECTED_DEVICE_NAME = String(A.expected_device_name || '').trim();
const EXPECTED_PHYSICAL_CU_COUNT = Number(A.expected_physical_cu_count || 0);
if (!!EXPECTED_GFX !== !!EXPECTED_TARGET) {
  throw new Error('expected_gfx and expected_target must be supplied together');
}
if ((EXPECTED_GFX && !/^gfx[0-9a-f]+$/.test(EXPECTED_GFX)) ||
    (EXPECTED_TARGET && !['r9700', 'mi350x', 'mi355x', 'unknown'].includes(EXPECTED_TARGET))) {
  throw new Error('invalid bakeoff identity: expected_gfx must be a gfx token and expected_target r9700|mi350x|mi355x|unknown');
}
if (EXPECTED_TARGET === 'r9700' && EXPECTED_GFX !== 'gfx1201') {
  throw new Error('expected_target=r9700 requires expected_gfx=gfx1201');
}
if (['mi350x', 'mi355x'].includes(EXPECTED_TARGET) && EXPECTED_GFX !== 'gfx950') {
  throw new Error(`expected_target=${EXPECTED_TARGET} requires expected_gfx=gfx950`);
}
if (EXPECTED_GFX === 'gfx1200' || EXPECTED_TARGET === 'gfx1200') {
  throw new Error('gfx1200 is not supported: this workflow is hardware-validated only on R9700 / gfx1201');
}
let RDNA4_ISOLATE = EXPECTED_GFX === 'gfx1201';
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge')).replace(/\/+$/, '');
if (RDNA4_ISOLATE) KERNEL_KNOWLEDGE_DIR = '';
const KERNEL_PATH_ORIG = A.kernel_path;
const KERNEL_NAME_HINT = String(KERNEL_PATH_ORIG).replace(/\/+$/, '').split('/').pop();
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
const TASK = A.task || '';
const OP_SPEC = A.op_spec || {};
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const ENABLE_FP8 = String(A.enable_fp8 != null ? A.enable_fp8 : 'false');
const GPU_LIST = String(A.gpu_ids != null ? A.gpu_ids : '0').split(',').map(s => s.trim()).filter(Boolean);
// Forwarded verbatim to each lane worker; the dispatcher itself does not schedule. Without the
// passthrough gpu_mode would be settable only on a direct kernel_lane.js call, so the pinned "before"
// arm would be unreachable through the normal entry point and the A/B could not be run at all.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
// Explicit backend list (empty => auto-discover from the freeze + op_benchmarker probe).
const BACKENDS = (Array.isArray(A.backends) ? A.backends
  : (typeof A.backends === 'string' ? A.backends.split(',') : []))
  .map(s => String(s == null ? '' : s).trim().toLowerCase()).filter(Boolean);
// Expert-skills passthrough (advisory; OFF by default -> nothing injected).
let USE_EXPERT_SKILLS = !RDNA4_ISOLATE && String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
const EXPERT_SKILL_ROLES = new Set(['op_benchmarker']);

// Warm-start experience KB. Passed to each lane explicitly (the bakeoff lane invocation spreads
// specific keys, not ...A) so every language lane reads/writes its own <kernel>__<lang>__<gfx> slug.
let WARM_START = RDNA4_ISOLATE ? 'off'
  : (String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on');
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts')).replace(/\/+$/, '');
// Plane selection, forwarded the same way and for the same reason: every bakeoff lane must read and
// write the plane the run was launched with, not each its own default.
const KB_PLANE_ARGS = {
  ...(A.kb_mode != null ? { kb_mode: String(A.kb_mode) } : {}),
  ...(A.kb_store_dir != null ? { kb_store_dir: String(A.kb_store_dir) } : {}),
  ...(A.kb_framework_version != null ? { kb_framework_version: String(A.kb_framework_version) } : {}),
};

