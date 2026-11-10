// ---------------------------------------------------------------------------
// Args + defaults. (The script cannot touch the filesystem or read its own
// path; agents do all FS work, and every path is supplied/derived from args —
// nothing about the install location is hard-coded.)
// ---------------------------------------------------------------------------
const A = args || {};
// Experimental source-bound planner entry. Refuse other lifecycle paths before any host/model call.
const NATIVE_PLANNER_SEED = A.native_planner_seed_ref !== undefined;
if (NATIVE_PLANNER_SEED) {
  const ref = A.native_planner_seed_ref;
  if (!ref || typeof ref !== 'object' || Array.isArray(ref)
      || Object.keys(ref).sort().join(',') !== 'bytes,path,sha256'
      || typeof ref.path !== 'string' || !ref.path.startsWith('/')
      || typeof ref.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(ref.sha256)
      || !Number.isSafeInteger(ref.bytes) || ref.bytes < 1 || ref.bytes > 4194304)
    throw new Error('invalid native_planner_seed_ref');
  const conflict = ['first_plan_ref', 'prepared_original_ref', 'state_dir', 'population_lineage',
    'prepare_only', 'canonical_selection_resume', 'gpu_identity_capture_ref', 'deadline_epoch',
    'search_reserve_s', 'search_duration_s', 'preparation_allowance_s', 'engineer_reserve_s',
    'delivery_reserve_s', 'final_validation_reserve_s', 'final_cleanup_reserve_s', 'delivery_facts',
    'settlement_deadline_epoch', 'wait_for_process_settlement', 'shared_kb', 'global_kb',
    'e2e_feedback', 'harness_addendum', 'incremental_analyze', 'apply_to_original',
    'external_measurement_command', 'external_result_command', 'external_final_measurement_command',
    'external_final_result_command', 'objective', 'op_spec', 'budget', 'eval_dir',
    'dra_enabled', 'dra_blindspot', 'dra_max_blindspots', 'dra_max_questions']
    .find(key => Object.hasOwn(A, key));
  if (conflict || (A.mode != null && A.mode !== 'optimize')
      || (A.optimizer != null && A.optimizer !== 'evo')
      || (A.warm_start != null && !['off', 'false', 'none'].includes(String(A.warm_start))))
    throw new Error(`native planner seed conflicts with lifecycle configuration: ${conflict || 'mode'}`);
  if (!Number.isSafeInteger(A.agent_timeout_ms) || A.agent_timeout_ms <= 0
      || !Number.isSafeInteger(A.agent_retries) || A.agent_retries < 1
      || !Number.isSafeInteger(A.native_planner_deadline_epoch) || A.native_planner_deadline_epoch <= 0
      || !agent.processTimeoutSupported || typeof runProcess !== 'function')
    throw new Error('native planner seed requires finite native process deadline, timeout and retry count');
}
if (A.evolution_planner_case_evidence_view !== undefined
    && typeof A.evolution_planner_case_evidence_view !== 'boolean')
  throw new Error('evolution_planner_case_evidence_view must be a boolean');
if (A.record_planner_inputs != null && typeof A.record_planner_inputs !== 'boolean')
  throw new Error('record_planner_inputs must be a boolean');
if (A.prepare_only != null && typeof A.prepare_only !== 'boolean')
  throw new Error('prepare_only must be a boolean');
if (A.prepare_only === true && ((String(A.mode ?? 'optimize').trim() || 'optimize') !== 'optimize'
    || A.state_dir || A.prepared_original_ref !== undefined || A.population_lineage
    || A.objective?.kind === 'external_primary'
    || A.search_duration_s != null || A.preparation_allowance_s != null))
  throw new Error('prepare_only needs a fresh ORIGINAL lane, without author/resume/import/population or relative search controls');
if (Object.keys(A).some((key) => key.startsWith('ablation_'))) {
  throw new Error('ablation_* study controls are retired; use the original revision to replay those studies.');
}
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
