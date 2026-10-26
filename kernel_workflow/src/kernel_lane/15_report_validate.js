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
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.'
    + (OPTIMIZER === 'evo' && DESIGN_GENOME_MODE !== 'off'
      ? ` Read ${WORKFLOW_DIR}/roles/design_genome.md when summarizing retained author claims and implementation gaps.` : ''), {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA });
const finalPatch = EXTERNAL_OBJECTIVE ? await writeFrozenFinalPatch()
  : (report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`);

// ===========================================================================
// PHASE: Director validation + arbitration
// ===========================================================================
if (PROCESS_BOUNDED) setProcessDeadline(DEADLINE_EPOCH - FINAL_CLEANUP_RESERVE_S);
phase('Validate');
const finalMeasurementFailure = await measureFinalEvaluation(CANONICAL);
let validation = await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_POOL,
    APPLY_TO_ORIGINAL, COMMANDMENT, FROZEN_ORACLE: FROZEN_ORACLE ? 'true' : 'false',
    FINAL_PATCH: finalPatch,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });
const directorValidationStatus = validation && validation.validation_status;
if (EXTERNAL_OBJECTIVE) {
  // A report or review may write files, but cannot replace the captured patch.
  await writeFrozenFinalPatch();
  validation = {...validation,
    director_review_status: validation ? 'returned' : 'unavailable',
    validation_status: validation ? validation.validation_status : 'accepted',
    director_reported_evaluation: validation && validation.external_evaluation || null,
    external_evaluation: frozenFinalEvaluation};
}
validation = finalMeasurementFailure ? {...validation,
  primary_speedup: null, external_measurement_valid: false, validation_status: 'invalid_measurement',
  native_reproducible_gain: false, native_reproducible_regression: false,
  development_geomean_speedup: null, final_failure: finalMeasurementFailure}
  : await checkedExternalMeasurement(validation, KERNEL_PATH_ORIG, finalPatch, true);
if (EXTERNAL_OBJECTIVE && validation && validation.external_measurement_valid) {
  if (directorValidationStatus && !says(directorValidationStatus, 'accepted')) {
    validation.validation_status = directorValidationStatus;
  } else if (!nativeGainEligible(validation)) {
    validation.validation_status = 'no_improvement';
  }
}
const externalPromotion = !EXTERNAL_OBJECTIVE || !!(nativeGainEligible(validation)
  && validation && says(validation.validation_status, 'accepted'));

const finalGeomean = EXTERNAL_OBJECTIVE ? null : (validation ? validation.director_verified_speedup_geomean : cumulative);
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = !EXTERNAL_OBJECTIVE && validation && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
const finalPrimary = EXTERNAL_OBJECTIVE ? (validation ? nativePrimary(validation) : null)
  : (HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean);
log(`COMPLETE. ${KERNEL_NAME}: verified ${EXTERNAL_OBJECTIVE ? 'external primary' : HAS_WORKLOAD ? 'time-weighted' : 'geomean'} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${validation ? validation.validation_status : '?'}). Results in ${EVAL_DIR}`);
