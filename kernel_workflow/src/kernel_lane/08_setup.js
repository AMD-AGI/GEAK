// ===========================================================================
// PHASE: Setup
// ===========================================================================
// An operational owner may supply a captured identity when this process has no
// direct GPU access. The owner authenticates its observation/device provenance;
// the Director verifies these exact bytes with the shared offline parser.
const GPU_IDENTITY_CAPTURE_REF = A.gpu_identity_capture_ref;
if (GPU_IDENTITY_CAPTURE_REF !== undefined) {
  const ref = GPU_IDENTITY_CAPTURE_REF;
  if (!ref || typeof ref !== 'object' || Array.isArray(ref)
      || Object.keys(ref).sort().join(',') !== 'path,sha256'
      || typeof ref.path !== 'string' || !ref.path.startsWith('/')
      || /[\0\r\n]/.test(ref.path)
      || !ref.path.slice(1).split('/').every(part => part && part !== '.' && part !== '..')
      || typeof ref.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(ref.sha256)) {
    throw new Error('invalid GPU identity capture reference');
  }
  if (typeof A.expected_gfx !== 'string' || !/^gfx[0-9a-f]+$/.test(EXPECTED_GFX)
      || typeof A.expected_target !== 'string'
      || !['r9700', 'mi350x', 'mi355x', 'unknown'].includes(EXPECTED_TARGET)
      || typeof A.expected_device_name !== 'string' || !EXPECTED_DEVICE_NAME
      || !Number.isSafeInteger(A.expected_physical_cu_count) || A.expected_physical_cu_count <= 0) {
    throw new Error('captured GPU identity requires complete expected identity');
  }
}
phase('Setup');
const setup = await agentT(
  roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK,
    WORKFLOW_DIR, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC, EXPECTED_GFX, EXPECTED_TARGET,
    EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
    ...(GPU_IDENTITY_CAPTURE_REF !== undefined ? { GPU_IDENTITY_CAPTURE_REF } : {}),
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
if (A.population_lineage) {
  if (EVAL_DIR !== A.eval_dir || CANONICAL !== `${A.eval_dir}/workspace`) {
    throw new Error('population Setup must use its assigned private eval_dir/workspace');
  }
  const isolation = await runProcess({cmd: 'python3', args: ['-c',
    'import pathlib,subprocess,sys\ne,w,o=map(lambda p:pathlib.Path(p).resolve(strict=True),sys.argv[1:])\n' +
    'assert w.parent == e and not (w == o or w in o.parents or o in w.parents), "population workspace is not isolated"\n' +
    'assert pathlib.Path(subprocess.check_output(["git","-C",str(w),"rev-parse","--show-toplevel"],text=True).strip()).resolve() == w, "population workspace needs its own repository"',
    EVAL_DIR, CANONICAL, KERNEL_PATH_ORIG], timeoutMs: 30000, graceMs: 2000});
  if (isolation.status !== 'completed' || isolation.code !== 0 || isolation.strays_killed
      || !isolation.group_cleared || isolation.stdio_held) throw new Error('population Setup failed source isolation checks');
}
const GFX = String(setup.device_gfx || '').trim().toLowerCase();
const DEVICE_TARGET = String(setup.device_target || '').trim().toLowerCase();
const SETUP_DEVICE_NAME = String(setup.device_name || '').trim();
const SETUP_PHYSICAL_CU_COUNT = Number(setup.physical_cu_count);
if (!GFX) {
  throw new Error('Setup failed: director did not return device_gfx (required; refuse CDNA knowledge without an ISA)');
}
if (!DEVICE_TARGET) {
  throw new Error('Setup failed: director did not return device_target');
}
if (!Number.isFinite(SETUP_PHYSICAL_CU_COUNT) || SETUP_PHYSICAL_CU_COUNT <= 0) {
  throw new Error('Setup failed: director did not return a positive physical_cu_count');
}
const normalizeDeviceName = (name) => String(name || '').trim().replace(/\s+/g, ' ').toLowerCase();
if (GPU_IDENTITY_CAPTURE_REF !== undefined
    && (GFX !== EXPECTED_GFX || DEVICE_TARGET !== EXPECTED_TARGET
      || normalizeDeviceName(SETUP_DEVICE_NAME) !== normalizeDeviceName(EXPECTED_DEVICE_NAME)
      || !Number.isSafeInteger(setup.physical_cu_count)
      || SETUP_PHYSICAL_CU_COUNT !== EXPECTED_PHYSICAL_CU_COUNT)) {
  throw new Error('captured GPU identity differs from expected identity');
}
if (GFX === 'gfx1200') {
  throw new Error('Detected gfx1200, which is not validated by this workflow; refusing R9700 guidance');
}
if (EXPECTED_GFX && GFX !== EXPECTED_GFX) {
  throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${GFX || 'unknown'}`);
}
if (EXPECTED_TARGET && EXPECTED_TARGET !== 'unknown' && DEVICE_TARGET !== EXPECTED_TARGET) {
  throw new Error(`GPU product mismatch: expected ${EXPECTED_TARGET}, detected ${DEVICE_TARGET || 'unknown'}`);
}
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
const PHYSICAL_CU_COUNT = EXPECTED_PHYSICAL_CU_COUNT > 0
  ? EXPECTED_PHYSICAL_CU_COUNT : SETUP_PHYSICAL_CU_COUNT;
let ROOFLINE_STATUS = 'non-rdna';
if (DEVICE_TARGET === 'r9700' && GFX === 'gfx1201') {
  ROOFLINE_STATUS = 'calibrated-r9700';
} else if (/^gfx120/.test(GFX)) {
  ROOFLINE_STATUS = 'unknown-device-not-r9700';
} else if (GFX === 'gfx950' && !['mi350x', 'mi355x'].includes(DEVICE_TARGET)) {
  ROOFLINE_STATUS = 'unknown_peaks';
}
const RDNA4_ISOLATE = GFX === 'gfx1201';
if (RDNA4_ISOLATE) {
  // Until the perf/learned stores implement metadata-enforced gens filtering, fail closed:
  // R9700 / gfx1201 receive the dedicated in-tree hardware guide but no CDNA-biased external cards.
  USE_EXPERT_SKILLS = false;
  KERNEL_KNOWLEDGE_DIR = '';
  USE_LEARNED_READ = false;
  WARM_START_ON = false;
}
// `_task` is the e2e head's DIRECTORY suffix, and the director returns the basename verbatim, so it
// would otherwise ride into the session id and the stored record. The canonical id is folded store-
// side (experience_store.remote_identity); this keeps the rest of the run calling it one name.
const KERNEL_NAME = String(setup.kernel_name || '').replace(/_task$/, '') || setup.kernel_name;
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
