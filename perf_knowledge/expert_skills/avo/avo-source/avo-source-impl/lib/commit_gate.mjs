// Pure, testable AVO commit-gate logic (no I/O, no runtime globals).
//
// This is the single source of truth for the deterministic commit decision used
// by avo_workflow.js. Kept in its own module so it can be unit-tested with node
// (see tests/commit_gate.test.mjs) independently of the Workflow runtime.
//
// Port of GEAK LineageStore.maybe_commit's gate (paper §3.2 + anti-lazy floor +
// B1 significance margin).

/** Geometric mean of an array of per-case speedups (>0). 0 if none valid. */
export function geomean(speedups) {
  const xs = (speedups || []).map(Number).filter((s) => Number.isFinite(s) && s > 0);
  if (xs.length === 0) return 0;
  return Math.exp(xs.reduce((a, s) => a + Math.log(s), 0) / xs.length);
}

/** Geomean of a per-case array of {speedup}. */
export function geomeanPerCase(perCase) {
  return geomean((perCase || []).map((c) => (c ? c.speedup : NaN)));
}

/** Normalize Director structured output into the commit-gate contract. */
export function normalizeDirectorValidation(validation) {
  if (!validation) return null;
  const correctness = String(validation.correctness || '').toLowerCase();
  const neutralStatuses = new Set([
    'n/a', 'na', 'not_applicable', 'not applicable',
  ]);
  const statuses = [validation.formal_status, validation.validation_status]
    .map((value) => String(value || '').trim().toLowerCase())
    // Non-ASM repeated validation has no formal protocol. Its Director report
    // may spell the neutral formal status in several conventional ways. ASM
    // still fails closed later because requireFormalEvidence checks the
    // preserved formal_status and protocol.
    .filter((status) => status && !neutralStatuses.has(status));
  const statusesAccepted = statuses.length > 0
    && statuses.every((status) => status === 'accepted');
  const normalizationRejectionReason = correctness !== 'pass'
    ? `director correctness is ${correctness || 'missing'}`
    : (!statuses.length
      ? 'director supplied no actionable validation status'
      : (statusesAccepted
        ? ''
        : `director status conflict: ${statuses.join(',')}`));
  const speedup = Number(validation.director_verified_speedup_geomean);
  const weighted = Number(validation.director_verified_speedup_weighted);
  return {
    status: correctness === 'pass' && statusesAccepted ? 'ok' : 'failed',
    correctness: correctness === 'pass' ? 'pass' : 'fail',
    verified_geomean: speedup,
    verified_weighted: Number.isFinite(weighted) && weighted > 0 ? weighted : undefined,
    verified_arithmetic: Number(
      validation.director_verified_speedup_arithmetic || speedup),
    per_case: validation.per_case || [],
    cv: Number(validation.cv),
    protocol: validation.protocol || '',
    formal_status: validation.formal_status || validation.validation_status || '',
    infrastructure_status: validation.infrastructure_status || '',
    screening_only: validation.screening_only === true,
    ci_lower: Number(validation.ci_lower),
    ci_upper: Number(validation.ci_upper),
    positive_block_fraction: Number(validation.positive_block_fraction),
    calibrated_margin: Number(validation.calibrated_margin),
    aa_calibration_passed: validation.aa_calibration_passed === true,
    bimodal: validation.bimodal === true,
    benchmark_report: validation.benchmark_report || '',
    notes: validation.notes || '',
    director_validation_status: validation.validation_status || 'unknown',
    normalization_rejection_reason: normalizationRejectionReason,
  };
}

export function formalPrecommitEnabled(asmMode, requested) {
  return !!asmMode || !!requested;
}

/** Build gate input from the authoritative harness report, never LLM-transcribed numbers. */
export function authoritativeFormalValidation(validation, report, reportPath = '',
                                              expectedSessionId = '') {
  if (!validation || !report || Number(report.schema_version) !== 2) {
    throw new Error('missing or unsupported formal benchmark report');
  }
  const protocol = String(report.protocol || '').toLowerCase().replace(/-/g, '_');
  const formalStatus = String(report.status || '').toLowerCase();
  const allowedStatus = new Set(['accepted', 'no_significant_gain', 'unstable']);
  const correctnessEvidence = report.correctness || {};
  const correctness = String(correctnessEvidence.status || '').toLowerCase();
  const health = String(report.health && report.health.status || '').toLowerCase();
  const stats = report.statistics || {};
  const blocks = Array.isArray(report.blocks) ? report.blocks : [];
  const aaBlocks = Array.isArray(report.aa_blocks) ? report.aa_blocks : [];
  const blockHealth = Array.isArray(report.block_health) ? report.block_health : [];
  const measurementContract = report.measurement_contract || {};
  const healthScope = String(measurementContract.health_scope || '')
    .toLowerCase().replace(/_/g, ' ');
  const phaseHealth = healthScope === 'phase boundaries';
  const perBlockHealth = healthScope === 'per block';
  const backendProtocol = String(measurementContract.backend_protocol || '');
  const performanceBackend = String(measurementContract.performance_backend || '');
  const device = report.device || {};
  const hashes = report.artifact || {};
  const hashValues = [
    hashes.baseline_sha256, hashes.candidate_sha256, hashes.capture_identity_sha256,
  ];
  if (protocol !== 'paired_abba' || !allowedStatus.has(formalStatus)) {
    throw new Error('invalid formal protocol/status');
  }
  if (!expectedSessionId || String(report.session_id || '') !== String(expectedSessionId)) {
    throw new Error('formal report session identity mismatch');
  }
  if (correctness !== 'pass' || health !== 'healthy') {
    throw new Error('formal correctness/health evidence is not healthy');
  }
  if (!device.physical_gpu_uuid || !device.physical_gpu_bdf) {
    throw new Error('formal report device identity is incomplete');
  }
  if (backendProtocol !== 'hip_abi_persistent_reciprocal_role_swap'
      || healthScope !== 'per block'
      || performanceBackend !== 'generated-persistent') {
    throw new Error('formal report persistent backend contract is invalid');
  }
  if (stats.contaminated === true) {
    throw new Error('formal report contains contaminated timing evidence');
  }
  if (Number(report.transient_failures || 0) !== 0) {
    throw new Error('formal report contains transient failures');
  }
  if (phaseHealth) {
    throw new Error('formal report phase-boundary health is insufficient');
  }
  const requiredHealth = blocks.length + aaBlocks.length;
  if (blocks.length < 5 || aaBlocks.length < 3
      || blockHealth.length < requiredHealth) {
    throw new Error('formal report has insufficient blocks/health evidence');
  }
  if (perBlockHealth) {
    const phaseCount = (phase) => blockHealth.filter(
      (item) => String(item && item.phase || '') === phase).length;
    if (phaseCount('aa_slot_calibration') < aaBlocks.length
        || phaseCount('ab_reciprocal') < blocks.length) {
      throw new Error('formal report missing per-block health evidence');
    }
  }
  if (!blockHealth.every((item) => item && item.before && item.after
      && item.before.formal_checks && item.after.formal_checks)) {
    throw new Error('formal report block health snapshots are incomplete');
  }
  if (!hashValues.every((value) => /^[0-9a-f]{64}$/i.test(String(value || '')))) {
    throw new Error('formal report artifact identity is incomplete');
  }
  if (hashes.disk_verified !== true) {
    throw new Error('formal report artifact identity was not verified on disk');
  }
  if (correctnessEvidence.backend !== 'generated-persistent'
      || correctnessEvidence.method
        !== 'private_capture_view_baseline_candidate'
      || Number(correctnessEvidence.regions_total) <= 0
      || Number(correctnessEvidence.regions_equivalent)
        !== Number(correctnessEvidence.regions_total)
      || correctnessEvidence.capture_identity_sha256
        !== hashes.capture_identity_sha256
      || !/^[0-9a-f]{64}$/i.test(
        String(correctnessEvidence.reference_output_sha256 || ''))
      || !/^[0-9a-f]{64}$/i.test(
        String(correctnessEvidence.candidate_output_sha256 || ''))
      || (correctnessEvidence.validation_policy_sha256
        && !/^[0-9a-f]{64}$/i.test(
          String(correctnessEvidence.validation_policy_sha256)))
      || String(correctnessEvidence.validation_policy_sha256 || '')
        !== String(hashes.validation_policy_sha256 || '')
      || measurementContract.correctness_backend !== 'generated-persistent'
      || measurementContract.correctness_method
        !== 'private_capture_view_baseline_candidate') {
    throw new Error('formal correctness backend/evidence contract is invalid');
  }
  const finite = (value) => Number.isFinite(Number(value));
  if (!finite(stats.speedup_estimate)
      || !Array.isArray(stats.speedup_ci95) || stats.speedup_ci95.length !== 2
      || !stats.speedup_ci95.every(finite)
      || !finite(stats.speedup_cv) || !finite(stats.positive_block_fraction)
      || !finite(stats.calibrated_margin)
      || !finite(stats.baseline_us_median) || !finite(stats.candidate_us_median)
      || typeof stats.aa_calibration_passed !== 'boolean'
      || typeof stats.bimodal !== 'boolean') {
    throw new Error('formal report statistics are incomplete');
  }
  return {
    ...validation,
    validation_status: formalStatus,
    formal_status: formalStatus,
    correctness,
    director_verified_speedup_geomean: Number(stats.speedup_estimate),
    director_verified_speedup_arithmetic: Number(stats.speedup_estimate),
    per_case: [{
      name: validation.per_case && validation.per_case[0]
        ? validation.per_case[0].name : 'realcap_capture',
      baseline_ms: Number(stats.baseline_us_median) / 1000.0,
      optimized_ms: Number(stats.candidate_us_median) / 1000.0,
      speedup: Number(stats.speedup_estimate),
    }],
    cv: Number(stats.speedup_cv),
    protocol,
    infrastructure_status: health,
    screening_only: false,
    ci_lower: Number(stats.speedup_ci95[0]),
    ci_upper: Number(stats.speedup_ci95[1]),
    positive_block_fraction: Number(stats.positive_block_fraction),
    calibrated_margin: Number(stats.calibrated_margin),
    aa_calibration_passed: stats.aa_calibration_passed,
    bimodal: stats.bimodal,
    benchmark_report: reportPath,
    artifact: hashes,
    device: report.device || {},
  };
}

export function classifyFinalValidation(validation, asmMode, opts = {},
                                        expectedSpeedup = 1.0) {
  const ver = normalizeDirectorValidation(validation);
  let accepted = !!ver && (asmMode
    ? passesCommitGate(ver, 1.0, { ...opts, requireFormalEvidence: true })
    : ver.status === 'ok');
  if (accepted && Number(expectedSpeedup) > 1.0) {
    const defaultTol = asmMode ? 0.03 : 0.10;
    const tolerance = Number(opts.finalReproductionTolerance != null
      ? opts.finalReproductionTolerance : defaultTol);
    const requiredUpper = Number(expectedSpeedup) * (1.0 - Math.max(0, tolerance));
    // Formal evidence may use the upper confidence bound; repeated/source
    // validation has no CI and must reproduce with its measured geomean.
    const reproduced = asmMode ? ver.ci_upper : ver.verified_geomean;
    if (!Number.isFinite(reproduced) || reproduced < requiredUpper) {
      return { accepted: false, status: 'reproduction_mismatch', verify: ver };
    }
  }
  if (accepted) return { accepted: true, status: 'accepted', verify: ver };
  const infra = String(validation && validation.infrastructure_status || '').toLowerCase();
  const formal = String(validation && validation.formal_status || '').toLowerCase();
  const status = infra && infra !== 'healthy'
    ? 'infrastructure_unhealthy'
    : (formal === 'unstable'
      || (validation && (validation.bimodal === true
        || validation.aa_calibration_passed === false))
      ? 'unstable' : 'no_significant_gain');
  return { accepted: false, status, verify: ver };
}

/**
 * Decide whether a verified candidate should be committed to the lineage.
 *
 * The significance margin is **variance-aware** (C1): if the verifier reports the
 * candidate's measured noise level `ver.cv` (coefficient of variation of the repeated
 * latencies), the required improvement scales with it — `max(significanceMargin,
 * noiseK * cv)`. This stops a within-noise "win" from committing and seeding a false
 * best that the whole rest of the lineage then branches from. With no `cv` reported
 * the behavior is exactly the old fixed-margin gate (backward compatible).
 *
 * Under `requireFormalEvidence` the candidate is judged on its CI lower bound, which
 * already discounts sampling noise, so the cv band is dropped in favour of the
 * A/A-calibrated `ver.calibrated_margin` when the paired run measured one. Stacking
 * both would charge the same noise twice and reject real sub-1.35*cv wins.
 *
 * @param {object} ver        verify result: {status, correctness, verified_geomean, cv?}
 * @param {number} cumulative current best verified geomean speedup vs the TRUE baseline
 * @param {object} opts       {epsilon, minCommitSpeedup, significanceMargin, noiseK}
 * @returns {{accepted: boolean, reason: string, candidate?: number,
 *            threshold?: number, ci_lower?: number}}
 */
export function evaluateCommitGate(ver, cumulative, opts = {}) {
  const epsilon = opts.epsilon != null ? opts.epsilon : 0.001;
  const minCommitSpeedup = opts.minCommitSpeedup != null ? opts.minCommitSpeedup : 1.0;
  const significanceMargin = opts.significanceMargin != null ? opts.significanceMargin : 0.01;
  const noiseK = opts.noiseK != null ? opts.noiseK : 1.0;        // measured-noise multiplier
  const reject = (reason, details = {}) => ({
    accepted: false, reason, ...details,
  });

  if (!ver) return reject('missing verification result');
  if (ver.status === 'failed') {
    return reject(ver.normalization_rejection_reason
      || 'verification status normalized to failed');
  }
  if (String(ver.correctness || '').toLowerCase() !== 'pass') {
    return reject('correctness did not pass');
  }
  if (opts.requireFormalEvidence) {
    const protocol = String(ver.protocol || '').toLowerCase().replace(/-/g, '_');
    const infra = String(ver.infrastructure_status || ver.infra_status || '').toLowerCase();
    const formalStatus = String(ver.formal_status || ver.validation_status || '').toLowerCase();
    const screeningOnly = ver.screening_only === true || ver.screening_only === 1
      || String(ver.screening_only || '').toLowerCase() === 'true';
    const positiveFraction = Number(ver.positive_block_fraction);
    const ciLower = Number(ver.ci_lower);
    if (protocol !== 'paired_abba') return reject(`formal protocol is ${protocol || 'missing'}`);
    if (screeningOnly) return reject('formal evidence is screening-only');
    if (infra !== 'healthy') return reject(`infrastructure is ${infra || 'missing'}`);
    if (formalStatus !== 'accepted') return reject(`formal status is ${formalStatus || 'missing'}`);
    if (ver.aa_calibration_passed !== true) return reject('A/A calibration did not pass');
    if (ver.bimodal === true || String(ver.bimodal || '').toLowerCase() === 'true') {
      return reject('formal measurements are bimodal');
    }
    if (!Number.isFinite(positiveFraction) || positiveFraction < 0.8) {
      return reject('positive block fraction is below 0.8');
    }
    if (!Number.isFinite(ciLower) || ciLower <= 0) {
      return reject('formal CI lower bound is missing or non-positive');
    }
  }
  const cand = Number(ver.verified_geomean);
  if (!(cand > 0)) return reject('verified geomean is missing or non-positive');
  if (cand <= minCommitSpeedup) {
    return reject('candidate did not clear the anti-lazy floor', {
      candidate: cand, threshold: minCommitSpeedup,
    });
  }
  const base = Number.isFinite(cumulative) && cumulative > 0 ? cumulative : 1.0;
  const cv = Number(ver.cv != null ? ver.cv : ver.variance_cv);  // measured noise (optional)
  const hasCv = Number.isFinite(cv) && cv >= 0;
  const requireCvForMicroGain = !!opts.requireCvForMicroGain;
  const microGainCeiling = opts.microGainCeiling != null ? Number(opts.microGainCeiling) : 1.02;
  if (requireCvForMicroGain && !hasCv && cand < base * microGainCeiling) {
    return reject('micro-gain has no CV evidence');               // no COV => no micro-gain commit
  }
  const calibratedMargin = Number(ver.calibrated_margin);
  const hasCalibratedMargin = Number.isFinite(calibratedMargin) && calibratedMargin > 0;
  // Formal evidence is judged on the CI lower bound below, which already discounts
  // sampling noise. Widening the margin by noiseK*cv on top of that charges the same
  // noise twice (effective bar ~1.35*cv), so the A/A-calibrated rig-bias floor — an
  // empirical measurement of the same quantity — replaces the cv band whenever the
  // paired run produced one. Without a calibrated floor we fail closed to cv.
  const cvBandApplies = hasCv && cv > 0
    && !(opts.requireFormalEvidence && hasCalibratedMargin);
  const margin = cvBandApplies
    ? Math.max(significanceMargin, noiseK * cv)                   // require beating the noise band
    : significanceMargin;
  const effectiveMargin = hasCalibratedMargin
    ? Math.max(margin, calibratedMargin) : margin;
  const threshold = base * Math.max(1 - epsilon, 1 + effectiveMargin);
  if (opts.requireFormalEvidence) {
    const ciLower = Number(ver.ci_lower);
    if (ciLower < threshold) {
      return reject('formal CI lower bound did not clear the gate', {
        candidate: cand, threshold, ci_lower: ciLower,
      });
    }
  }
  if (cand < threshold) {
    return reject('candidate did not clear the significance/noise threshold', {
      candidate: cand, threshold,
    });
  }
  return { accepted: true, reason: 'accepted', candidate: cand, threshold };
}

/** Backward-compatible boolean facade over the explainable gate decision. */
export function passesCommitGate(ver, cumulative, opts = {}) {
  return evaluateCommitGate(ver, cumulative, opts).accepted;
}
