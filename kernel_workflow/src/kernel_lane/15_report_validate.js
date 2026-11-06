const {report, finalPatch, validation, externalPromotion, finalGeomean, finalWeighted, finalPrimary} =
  await nativeDeliveryBoundary(async () => {
// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
if (PROCESS_BOUNDED) setProcessDeadline(DEADLINE_EPOCH - FINAL_VALIDATION_RESERVE_S);
await captureCanonicalSelection(history, 'terminal', round, finalWinner, stopReason);
try {
  await captureFinalEvaluation(CANONICAL);
} catch (cause) {
  const error = new Error(`final_capture_failed: ${String(cause)}`);
  error.code = 'final_capture_failed';
  throw error;
}
if (EXTERNAL_OBJECTIVE) history.final_selection = frozenFinalEvaluation;
const finalPatch = `${EVAL_DIR}/host_final_patch.diff`;
// The first export must succeed. Later role/host cutoffs must preserve its pointer and
// explicit uncertainty rather than erase this lane's already retained source evidence.
async function refreshFinalPatch(stage) {
  try { await writeFrozenFinalPatch(); return true; }
  catch (error) {
    const delivery = history.delivery || (history.delivery = {candidates: [], delivered: null, incomplete: []});
    (delivery.incomplete || (delivery.incomplete = [])).push({status: 'export_unverified',
      stage, path: finalPatch, error: String(error)});
    log(`[delivery] final export unverified after ${stage}; retained pointer ${finalPatch}.`);
    return false;
  }
}
if (!EXTERNAL_OBJECTIVE) {  // Retained delivery: export committed sources within the task boundary.
  // Resolve only now: CANONICAL and the host helpers exist after Setup/Optimize.
  // Scope is caller-declared, or the extracted task's existing kernel_src boundary;
  // model analysis and generated evaluation files do not define deliverable sources.
  const requestedPaths = A.editable_source_paths === undefined
    ? await hostRun('test', ['-d', `${CANONICAL}/kernel_src`]) !== null ? ['kernel_src'] : null
    : A.editable_source_paths;
  if (!Array.isArray(requestedPaths) || requestedPaths.length === 0 || requestedPaths.some(p =>
      typeof p !== 'string' || p.startsWith('/') || p.includes('\0')
      || p.split('/').includes('..') || !p.split('/').some(part => part && part !== '.'))) {
    throw nativeDeliveryError('native final export requires declared relative editable_source_paths or an extracted kernel_src directory');
  }
  const sourcePaths = [...new Set(requestedPaths.map(p => p.split('/').filter(part => part && part !== '.').join('/')))];
  const root = ((await git(CANONICAL, 'rev-list', '--max-parents=0', 'HEAD')) || '').trim();
  const head = ((await git(CANONICAL, 'rev-parse', '--verify', 'HEAD')) || '').trim();
  if (![root, head].every(rev => /^[0-9a-f]{40,64}$/.test(rev))) {
    throw nativeDeliveryError('native final export could not resolve the committed ORIGINAL and selected HEAD');
  }
  for (const path of sourcePaths) {
    // Either side permits newly added sources and retained source deletions.
    const sourceIn = async rev => /^(blob|tree)$/.test(((await git(CANONICAL, 'cat-file', '-t', `${rev}:${path}`)) || '').trim());
    if (!await sourceIn(root) && !await sourceIn(head)) {
      throw nativeDeliveryError('native final export source scope is absent from ORIGINAL and selected HEAD');
    }
  }
  // current_best.diff remains the separate, unchanged broad root-to-HEAD evidence.
  frozenFinalPatch = await git(CANONICAL, '--literal-pathspecs', 'diff', '--binary', '--no-color', '--no-ext-diff',
    '--src-prefix=a/', '--dst-prefix=b/', root, head, '--', ...sourcePaths);
  await writeFrozenFinalPatch();
}
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
if (PROCESS_BOUNDED) setProcessDeadline(DEADLINE_EPOCH - FINAL_CLEANUP_RESERVE_S);
// Report may consume its own window. Restore the captured bytes using the existing
// validation allowance before a Director can measure them, never an expired Report window.
let exportReady = await refreshFinalPatch('report');
phase('Validate');
const finalMeasurementFailure = await measureFinalEvaluation(CANONICAL);
let validation = exportReady ? await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    APPLY_TO_ORIGINAL, COMMANDMENT, FROZEN_ORACLE: FROZEN_ORACLE ? 'true' : 'false',
    FINAL_PATCH: finalPatch,
    // Delivery's primary claim belongs to the verified canonical winner. Report is explanatory.
    TECH_LEAD_REPORTED_GEOMEAN: !EXTERNAL_OBJECTIVE && !HAS_WORKLOAD ? cumulative
      : report ? report.final_speedup_geomean : cumulative,
    ...(HAS_WORKLOAD && !EXTERNAL_OBJECTIVE ? { TECH_LEAD_REPORTED_WEIGHTED: cumulative }
        : HAS_WORKLOAD && report && report.final_speedup_weighted != null
          ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA }) : {validation_status: 'unavailable', final_failure: 'final_patch_export_unverified'};
const directorValidationStatus = validation && validation.validation_status;
if (!EXTERNAL_OBJECTIVE) {
  let measuredPatch = null;
  try { measuredPatch = await hostRun('cat', ['--', finalPatch]); } catch (_) {}
  if (measuredPatch !== frozenFinalPatch) {
    validation = {...validation, director_review_status: directorValidationStatus || 'unavailable',
      validation_status: measuredPatch === null ? 'unavailable' : 'source_changed',
      final_failure: measuredPatch === null ? 'director_patch_identity_unavailable' : 'director_patch_changed'};
  }
  if (!await refreshFinalPatch('director')) {
    // A role could have changed this same path. Previous readback is not current
    // attestation: retain it for recovery, but withhold validation and promotion.
    validation = {...validation, director_review_status: directorValidationStatus || 'unavailable',
      validation_status: 'unavailable', final_failure: 'final_patch_export_unverified'};
  }
}
if (EXTERNAL_OBJECTIVE) {
  // A report or review may write files, but cannot replace the captured patch.
  exportReady = await refreshFinalPatch('director') && exportReady;
  validation = {...validation,
    director_review_status: validation ? 'returned' : 'unavailable',
    validation_status: validation ? validation.validation_status : 'unavailable',
    director_reported_evaluation: validation && validation.external_evaluation || null,
    external_evaluation: frozenFinalEvaluation};
}
validation = EXTERNAL_OBJECTIVE && !exportReady ? {...validation,
  primary_speedup: null, external_measurement_valid: false, validation_status: 'unavailable',
  native_reproducible_gain: false, native_reproducible_regression: false,
  development_geomean_speedup: null, final_failure: 'final_patch_export_unverified'}
  : finalMeasurementFailure ? {...validation,
  primary_speedup: null, external_measurement_valid: false, validation_status: 'invalid_measurement',
  native_reproducible_gain: false, native_reproducible_regression: false,
  development_geomean_speedup: null, final_failure: finalMeasurementFailure}
  : await checkedExternalMeasurement(validation, KERNEL_PATH_ORIG, finalPatch, true);
if (EXTERNAL_OBJECTIVE && validation && validation.external_measurement_valid) {
  if (!says(directorValidationStatus, 'accepted')) {
    validation.validation_status = directorValidationStatus || 'unavailable';
  } else if (!nativeGainEligible(validation)) {
    validation.validation_status = 'no_improvement';
  }
}
const externalPromotion = !EXTERNAL_OBJECTIVE || !!(nativeGainEligible(validation)
  && validation && says(validation.validation_status, 'accepted'));

const finalGeomean = EXTERNAL_OBJECTIVE ? null : (validation ? validation.director_verified_speedup_geomean : null);
// PRIMARY headline uses the declared metric; missing weighted measurement stays unavailable.
const finalWeighted = !EXTERNAL_OBJECTIVE && validation
  && Number.isFinite(validation.director_verified_speedup_weighted)
  && (!HAS_WORKLOAD || validation.director_verified_speedup_weighted > 0)
  ? validation.director_verified_speedup_weighted : null;
const finalPrimary = EXTERNAL_OBJECTIVE ? (validation ? nativePrimary(validation) : null)
  : (HAS_WORKLOAD ? (Number.isFinite(finalWeighted) ? finalWeighted : null) : finalGeomean);
log(`COMPLETE. ${KERNEL_NAME}: verified ${EXTERNAL_OBJECTIVE ? 'external primary' : HAS_WORKLOAD ? 'time-weighted' : 'geomean'} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${validation ? validation.validation_status : '?'}). Results in ${EVAL_DIR}`);

// Validation outcomes stay local to this lane. The return gate withholds promotion without
// erasing the host-delivered source or another lane's independently accepted result.

return {report, finalPatch, validation, externalPromotion, finalGeomean, finalWeighted, finalPrimary};
});
