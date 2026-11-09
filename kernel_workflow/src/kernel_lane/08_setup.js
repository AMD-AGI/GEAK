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
// Fresh assigned workspaces use the same materializer and initial Git commit as
// the Director recipe, but their existence no longer depends on its response.
// Author and saved-state setup have different contracts and retain their route.
let MATERIALIZED_SETUP = null;
if (EVAL_DIR_OVERRIDE && MODE === 'optimize' && !STATE_DIR
    && A.prepared_original_ref === undefined && A.prepare_only !== true
    && A.canonical_selection_resume !== true && !A.population_lineage?.imports?.length) {
  if (typeof runProcess !== 'function') {
    throw new Error('assigned fresh Setup requires the native runProcess capability');
  }
  const materialized = await runProcess({cmd: 'python3', args: ['-c', String.raw`
import json, os, pathlib, subprocess, sys
stage = 'preflight'
class SetupPreflightError(Exception):
    pass
def require(condition, reason):
    if not condition:
        raise SetupPreflightError(reason)
try:
    source_arg, eval_arg, workflow_arg = sys.argv[1:]
    require(pathlib.Path(source_arg).is_dir(), 'source_unavailable')
    source = pathlib.Path(source_arg).resolve(strict=True)
    evaluation = pathlib.Path(eval_arg)
    require(evaluation.is_absolute() and str(evaluation) == eval_arg, 'eval_path_not_canonical_absolute')
    require(evaluation.parent.is_dir(), 'eval_parent_unavailable')
    require(evaluation.parent.resolve(strict=True) == evaluation.parent, 'eval_parent_not_canonical')
    require(not evaluation.exists() and not evaluation.is_symlink(), 'eval_path_already_exists')
    require(not (evaluation == source or evaluation in source.parents or source in evaluation.parents), 'eval_source_overlap')
    materializer = pathlib.Path(workflow_arg) / 'scripts/materialize_workspace.sh'
    require(materializer.is_file(), 'materializer_unavailable')
    evaluation.mkdir()
    (evaluation / 'original_kernel_path.txt').write_text(source_arg + '\n')
    for name in ('baseline', 'workspace'):
        stage = 'materialize_' + name
        subprocess.run(['bash', str(materializer), '--src', source_arg,
                        '--dst', str(evaluation / name), '--shared-root', str(evaluation / '_shared'),
                        '--link-aiter'], check=True, stdout=sys.stderr)
    workspace = evaluation / 'workspace'
    # Preserve the Director recipe's source-path link text before Git records it.
    reference_arg = source_arg + '/reference_io.pt'
    if pathlib.Path(reference_arg).exists():
        stage = 'link_reference_io'
        subprocess.run(['ln', '-sfn', reference_arg, str(workspace / 'reference_io.pt')],
                       check=True, stdout=sys.stderr)
    stage = 'initialize_repository'
    ignore = workspace / '.gitignore'
    require(ignore.resolve(strict=False).is_relative_to(workspace), 'gitignore_outside_workspace')
    with ignore.open('a') as stream:
        stream.write('\nbuild/\n__pycache__/\n*.pyc\nresults.*\n*.so\n.torch_ext/\n.rocprofv3/\n*.o\n/.geak/\n')
    git_env = dict(os.environ, GIT_PAGER='cat', GIT_TERMINAL_PROMPT='0', GIT_EDITOR='true')
    for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY'):
        git_env.pop(key, None)
    git = ['git', '-C', str(workspace), '-c', 'user.email=team@workflow', '-c', 'user.name=team']
    for command in (['init', '-q'], ['add', '-A'], ['commit', '-q', '-m', 'baseline']):
        subprocess.run(git + command, env=git_env, check=True, stdout=sys.stderr)
    top = subprocess.check_output(git + ['rev-parse', '--show-toplevel'], env=git_env, text=True).strip()
    require(pathlib.Path(top).resolve(strict=True) == workspace, 'git_root_differs_from_workspace')
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], env=git_env, text=True).strip()
    print(json.dumps({'ok': True, 'eval_dir': eval_arg, 'workspace': str(workspace),
                      'baseline_dir': str(evaluation / 'baseline'), 'initial_commit': commit}))
except Exception as error:
    # Only fixed codes raised above are exposed; never OS, Git or environment text.
    reason = str(error) if type(error) is SetupPreflightError else 'unknown'
    print(json.dumps({'ok': False, 'stage': stage, 'error_type': type(error).__name__, 'reason': reason}))
    sys.exit(1)
`, KERNEL_PATH_ORIG, EVAL_DIR_OVERRIDE, WORKFLOW_DIR], timeoutMs: AGENT_TIMEOUT_MS, graceMs: 2000});
  let result = null;
  try { result = JSON.parse(materialized.stdout.trim()); } catch { /* retain typed failure below */ }
  if (materialized.status !== 'completed' || materialized.code !== 0 || materialized.strays_killed
      || !materialized.group_cleared || materialized.stdio_held || result?.ok !== true
      || result.eval_dir !== EVAL_DIR_OVERRIDE || result.workspace !== `${EVAL_DIR_OVERRIDE}/workspace`
      || result.baseline_dir !== `${EVAL_DIR_OVERRIDE}/baseline`
      || !/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(result.initial_commit || '')) {
    throw new Error('assigned Setup materialization failed; partial preparation retained: ' + JSON.stringify({
      status: materialized.status, code: materialized.code, strays_killed: materialized.strays_killed,
      group_cleared: materialized.group_cleared, stdio_held: materialized.stdio_held,
      stage: result?.stage || 'unknown', error_type: result?.error_type || 'unknown',
      reason: result?.reason || 'unknown' }));
  }
  MATERIALIZED_SETUP = result;
}
// Captured identity is deterministic runtime input. Never make a model
// retranscribe it before the workflow may use the verified measurements.
const normalizeDeviceName = (name) => String(name || '').trim().replace(/\s+/g, ' ').toLowerCase();
let VERIFIED_GPU_IDENTITY = null;
if (GPU_IDENTITY_CAPTURE_REF !== undefined) {
  if (typeof runProcess !== 'function') {
    throw new Error('captured GPU identity requires the native runProcess capability');
  }
  const measured = await runProcess({cmd: 'python3', args: [
    `${WORKFLOW_DIR}/../scripts/gpu_identity.py`, '--input', GPU_IDENTITY_CAPTURE_REF.path,
    '--input-sha256', GPU_IDENTITY_CAPTURE_REF.sha256,
  ], timeoutMs: AGENT_TIMEOUT_MS > 0 ? Math.min(30000, AGENT_TIMEOUT_MS) : 30000, graceMs: 2000});
  let identity = null;
  try { identity = JSON.parse(measured.stdout.trim()); } catch { /* fail closed below */ }
  if (measured.status !== 'completed' || measured.code !== 0 || measured.strays_killed
      || !measured.group_cleared || measured.stdio_held
      || !identity || typeof identity !== 'object' || Array.isArray(identity)
      || typeof identity.gfx !== 'string' || !/^gfx[0-9a-f]+$/.test(identity.gfx)
      || typeof identity.target !== 'string'
      || !['r9700', 'mi350x', 'mi355x', 'unknown'].includes(identity.target)
      || typeof identity.marketing_name !== 'string' || !identity.marketing_name.trim()
      || !Number.isSafeInteger(identity.physical_cu_count) || identity.physical_cu_count <= 0
      || !Number.isSafeInteger(identity.visible_gpu_agents) || identity.visible_gpu_agents <= 0) {
    throw new Error('native captured GPU identity verification failed; no live-probe or expected-value fallback');
  }
  if (identity.gfx !== EXPECTED_GFX || identity.target !== EXPECTED_TARGET
      || normalizeDeviceName(identity.marketing_name) !== normalizeDeviceName(EXPECTED_DEVICE_NAME)
      || identity.physical_cu_count !== EXPECTED_PHYSICAL_CU_COUNT) {
    throw new Error('native captured GPU identity differs from expected identity');
  }
  VERIFIED_GPU_IDENTITY = {
    device_gfx: identity.gfx, device_target: identity.target,
    device_name: identity.marketing_name, physical_cu_count: identity.physical_cu_count,
  };
}
const setup = await agentT(
  roleAgent('director', 'setup', MATERIALIZED_SETUP
    ? 'Inspect the materialized evaluation environment and return verified metadata; do not rematerialize or reset it.'
    : 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK,
    WORKFLOW_DIR, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC, EXPECTED_GFX, EXPECTED_TARGET,
    EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
    ...(MATERIALIZED_SETUP ? { MATERIALIZED_SETUP } : {}),
    ...(GPU_IDENTITY_CAPTURE_REF !== undefined ? { GPU_IDENTITY_CAPTURE_REF } : {}),
    ...(A.prepared_original_ref !== undefined ? { PREPARED_ORIGINAL_REF: A.prepared_original_ref } : {}),
    ...(VERIFIED_GPU_IDENTITY ? { VERIFIED_GPU_IDENTITY } : {}),
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
if (A.population_lineage || MATERIALIZED_SETUP) {
  if (EVAL_DIR !== A.eval_dir || CANONICAL !== `${A.eval_dir}/workspace`) {
    throw new Error('population Setup must use its assigned private eval_dir/workspace; ' +
      JSON.stringify({ expected_eval_dir: A.eval_dir, expected_workspace: `${A.eval_dir}/workspace`,
        actual_eval_dir: EVAL_DIR, actual_workspace: CANONICAL }));
  }
  const isolation = await runProcess({cmd: 'python3', args: ['-c',
    'import pathlib,subprocess,sys\ne,w,o=map(lambda p:pathlib.Path(p).resolve(strict=True),sys.argv[1:4])\n' +
    'assert w.parent == e and not (w == o or w in o.parents or o in w.parents), "population workspace is not isolated"\n' +
    'assert pathlib.Path(subprocess.check_output(["git","-C",str(w),"rev-parse","--show-toplevel"],text=True).strip()).resolve() == w, "population workspace needs its own repository"\n' +
    'if len(sys.argv)>4:\n' +
    ' assert subprocess.check_output(["git","-C",str(w),"rev-parse","HEAD"],text=True).strip() == sys.argv[4], "materialized initial commit changed"\n' +
    ' subprocess.check_call(["git","-C",str(w),"diff","--quiet","HEAD","--"])',
    EVAL_DIR, CANONICAL, KERNEL_PATH_ORIG,
    ...(MATERIALIZED_SETUP ? [MATERIALIZED_SETUP.initial_commit] : [])], timeoutMs: 30000, graceMs: 2000});
  if (isolation.status !== 'completed' || isolation.code !== 0 || isolation.strays_killed
      || !isolation.group_cleared || isolation.stdio_held) throw new Error('population Setup failed source isolation checks');
}
const deviceSetup = VERIFIED_GPU_IDENTITY || setup;
const GFX = String(deviceSetup.device_gfx || '').trim().toLowerCase();
const DEVICE_TARGET = String(deviceSetup.device_target || '').trim().toLowerCase();
const SETUP_DEVICE_NAME = String(deviceSetup.device_name || '').trim();
const SETUP_PHYSICAL_CU_COUNT = Number(deviceSetup.physical_cu_count);
if (!GFX) {
  throw new Error('Setup failed: director did not return device_gfx (required; refuse CDNA knowledge without an ISA)');
}
if (!DEVICE_TARGET) {
  throw new Error('Setup failed: director did not return device_target');
}
if (!Number.isFinite(SETUP_PHYSICAL_CU_COUNT) || SETUP_PHYSICAL_CU_COUNT <= 0) {
  throw new Error('Setup failed: director did not return a positive physical_cu_count');
}
if (GPU_IDENTITY_CAPTURE_REF !== undefined
    && (GFX !== EXPECTED_GFX || DEVICE_TARGET !== EXPECTED_TARGET
      || normalizeDeviceName(SETUP_DEVICE_NAME) !== normalizeDeviceName(EXPECTED_DEVICE_NAME)
      || !Number.isSafeInteger(deviceSetup.physical_cu_count)
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
