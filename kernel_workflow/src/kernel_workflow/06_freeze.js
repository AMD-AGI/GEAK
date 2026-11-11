// ===========================================================================
// PHASE: Freeze — establish the ONE immutable oracle + frozen baseline (denominator)
// ===========================================================================
phase('Freeze');
const oracle = await agentT(
  roleAgent('oracle_freezer', 'freeze',
    'Freeze the input kernel into an immutable op task dir (no server). Create the run dir too.', {
      KERNEL_PATH: KERNEL_PATH_ORIG, EXP_ROOT, KERNEL_NAME_HINT, GPU_ID: GPU_LIST[0],
      OP_SPEC, WORKLOAD_SPEC_PATH, SKILL_DIR: WORKFLOW_DIR,
      EXPECTED_GFX, EXPECTED_TARGET, EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
      // harness_lib.py (the shared timing/correctness lib) ships with e2e_workflow; gpu_lock.sh ships here.
      HARNESS_LIB: `${E2E_WF_DIR}/scripts/harness_lib.py`,
      GPU_LOCK: `${WORKFLOW_DIR}/scripts/gpu_lock.sh`,
    }),
  { phase: 'Freeze', label: 'oracle_freezer', schema: FREEZE_SCHEMA });
if (!oracle || !says(oracle.smoke, 'pass') || !oracle.task_dir || oracle.baseline_frozen === false) {
  log(`Freeze FAILED (${oracle ? oracle.notes || oracle.smoke : 'no result'}); aborting — no comparable baseline.`);
  return { mode: MODE, validation_status: 'freeze_failed', winner: null,
    reason: oracle ? oracle.notes || 'freeze smoke did not pass' : 'oracle_freezer returned nothing' };
}
const DETECTED_GFX = String(oracle.device_gfx || '').trim().toLowerCase();
const DETECTED_TARGET = String(oracle.device_target || '').trim().toLowerCase();
const DETECTED_DEVICE_NAME = String(oracle.device_name || '').trim();
const DETECTED_PHYSICAL_CU_COUNT = Number(oracle.physical_cu_count);
if (!DETECTED_GFX || !DETECTED_TARGET) {
  throw new Error('Freeze failed: oracle_freezer did not return structured device_gfx and device_target');
}
if (DETECTED_GFX === 'gfx1200') {
  throw new Error('Detected gfx1200, which is not validated by this workflow; refusing R9700 guidance');
}
if (EXPECTED_GFX && DETECTED_GFX !== EXPECTED_GFX) {
  throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${DETECTED_GFX}`);
}
if (EXPECTED_TARGET && EXPECTED_TARGET !== 'unknown' && DETECTED_TARGET !== EXPECTED_TARGET) {
  throw new Error(`GPU product mismatch: expected ${EXPECTED_TARGET}, detected ${DETECTED_TARGET}`);
}
if (!Number.isFinite(DETECTED_PHYSICAL_CU_COUNT) || DETECTED_PHYSICAL_CU_COUNT <= 0) {
  throw new Error('Freeze failed: oracle_freezer did not return a positive physical_cu_count');
}
if (EXPECTED_DEVICE_NAME && DETECTED_DEVICE_NAME !== EXPECTED_DEVICE_NAME) {
  throw new Error(
    `GPU product name mismatch: expected ${EXPECTED_DEVICE_NAME}, detected ${DETECTED_DEVICE_NAME || 'unknown'}`);
}
if (EXPECTED_PHYSICAL_CU_COUNT > 0 &&
    DETECTED_PHYSICAL_CU_COUNT !== EXPECTED_PHYSICAL_CU_COUNT) {
  throw new Error(
    `GPU physical CU mismatch: expected ${EXPECTED_PHYSICAL_CU_COUNT}, ` +
    `detected ${DETECTED_PHYSICAL_CU_COUNT}`);
}
const LANE_GFX = EXPECTED_GFX || DETECTED_GFX;
const LANE_TARGET = EXPECTED_TARGET && EXPECTED_TARGET !== 'unknown' ? EXPECTED_TARGET : DETECTED_TARGET;
const LANE_DEVICE_NAME = EXPECTED_DEVICE_NAME || DETECTED_DEVICE_NAME;
const LANE_PHYSICAL_CU_COUNT =
  EXPECTED_PHYSICAL_CU_COUNT || DETECTED_PHYSICAL_CU_COUNT;
if (DETECTED_GFX === 'gfx1201') {
  RDNA4_ISOLATE = true;
  KERNEL_KNOWLEDGE_DIR = '';
  USE_EXPERT_SKILLS = false;
  WARM_START = 'off';
}
const EVAL_DIR = oracle.eval_dir || `${EXP_ROOT}/bakeoff_${KERNEL_NAME_HINT}`;
log(`Freeze done. op_kind=${oracle.op_kind}, task_dir=${oracle.task_dir}, live_backend=${oracle.live_backend || '?'}`);

