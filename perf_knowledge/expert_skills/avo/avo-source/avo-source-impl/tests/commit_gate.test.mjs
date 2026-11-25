// Unit tests for the deterministic AVO commit gate.
// Run on the rig (node not required locally):  node tests/commit_gate.test.mjs
// Exit code 0 = all pass; non-zero = failure (CI-friendly).

import assert from 'node:assert/strict';
import {
  authoritativeFormalValidation, classifyFinalValidation, formalPrecommitEnabled,
  evaluateCommitGate, geomean, geomeanPerCase,
  normalizeDirectorValidation, passesCommitGate,
} from '../lib/commit_gate.mjs';

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`ok   - ${name}`); }
  catch (e) { console.error(`FAIL - ${name}\n      ${e.message}`); process.exitCode = 1; }
}
const approx = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

// --- geomean -------------------------------------------------------------
test('geomean of [1.6,1.4] ~= 1.4967', () => {
  assert.ok(approx(geomean([1.6, 1.4]), Math.sqrt(1.6 * 1.4)));
});
test('geomean ignores non-positive / non-finite', () => {
  assert.ok(approx(geomean([2.0, 0, -1, NaN, 2.0]), 2.0));
});
test('geomean of empty -> 0', () => assert.equal(geomean([]), 0));
test('geomeanPerCase reads .speedup', () => {
  assert.ok(approx(geomeanPerCase([{ speedup: 4 }, { speedup: 1 }]), 2.0));
});

// --- gate: defaults (epsilon=0.001, floor=1.0, margin=0.01) ---------------
const ok = (g) => ({ status: 'ok', correctness: 'pass', verified_geomean: g });

test('commits when clearly above best + margin', () => {
  assert.equal(passesCommitGate(ok(1.20), 1.0), true);
});
test('rejects when correctness fails', () => {
  assert.equal(passesCommitGate({ status: 'ok', correctness: 'fail', verified_geomean: 2.0 }, 1.0), false);
});
test('rejects failed status', () => {
  assert.equal(passesCommitGate({ status: 'failed', correctness: 'pass', verified_geomean: 2.0 }, 1.0), false);
});
test('rejects lazy ~1.0x rewrite (anti-lazy floor)', () => {
  assert.equal(passesCommitGate(ok(1.0), 1.0), false);
  assert.equal(passesCommitGate(ok(1.0005), 1.0), false); // below 1+margin over baseline
});
test('rejects within-noise tie vs best (significance margin)', () => {
  // best=1.50, candidate=1.505 (<1%) must NOT commit with margin=0.01
  assert.equal(passesCommitGate(ok(1.505), 1.50), false);
});
test('commits when it clears best by > margin', () => {
  assert.equal(passesCommitGate(ok(1.52), 1.50), true); // 1.52 >= 1.50*1.01=1.515
});
test('rejects non-positive / missing geomean', () => {
  assert.equal(passesCommitGate(ok(0), 1.0), false);
  assert.equal(passesCommitGate({ status: 'ok', correctness: 'pass' }, 1.0), false);
  assert.equal(passesCommitGate(null, 1.0), false);
});

// --- gate: custom opts ---------------------------------------------------
test('higher anti-lazy floor blocks small gains', () => {
  assert.equal(passesCommitGate(ok(1.03), 1.0, { minCommitSpeedup: 1.05 }), false);
  assert.equal(passesCommitGate(ok(1.06), 1.0, { minCommitSpeedup: 1.05 }), true);
});
test('zero margin reverts to epsilon-tolerant tie', () => {
  assert.equal(passesCommitGate(ok(1.50), 1.50, { significanceMargin: 0 }), true);
});
test('ASM-tuned margin (0.3%) commits a clean sub-0.5% micro-gain the default (0.5%) floors out', () => {
  // ASM instruction-level wins are sub-1%; with a clean measurement (cv~0) a +0.4%
  // gain should commit under the ASM default but NOT under the HIP/Triton default.
  const v = (g, cv) => ({ status: 'ok', correctness: 'pass', verified_geomean: g, cv });
  const asm = { significanceMargin: 0.003, noiseK: 0.85 };
  const def = { significanceMargin: 0.005, noiseK: 0.85 };
  assert.equal(passesCommitGate(v(1.004, 0.0), 1.0, asm), true);
  assert.equal(passesCommitGate(v(1.004, 0.0), 1.0, def), false);
  // Still variance-aware: a +0.4% "win" with 1% noise is rejected even for ASM.
  assert.equal(passesCommitGate(v(1.004, 0.01), 1.0, asm), false);
});

// --- gate: variance-aware margin (C1) ------------------------------------
const okv = (g, cv) => ({ status: 'ok', correctness: 'pass', verified_geomean: g, cv });

test('high measured noise widens the required margin', () => {
  // best=1.50, cand=1.55 (+3.3%). cv=5% -> need >5% over best (>=1.575) -> reject.
  assert.equal(passesCommitGate(okv(1.55, 0.05), 1.50), false);
  // a clearly-above-noise gain commits.
  assert.equal(passesCommitGate(okv(1.60, 0.05), 1.50), true);   // 1.60 >= 1.50*1.05=1.575
});
test('low noise falls back to the fixed significance margin', () => {
  // cv=0.2% < margin 1% -> margin stays 1% -> 1.52 vs 1.50 commits as before.
  assert.equal(passesCommitGate(okv(1.52, 0.002), 1.50), true);
});
test('noiseK scales the noise band', () => {
  // cv=2%, noiseK=2 -> margin 4% -> 1.55 (+3.3%) rejected; default noiseK=1 would pass.
  assert.equal(passesCommitGate(okv(1.55, 0.02), 1.50, { noiseK: 2 }), false);
  assert.equal(passesCommitGate(okv(1.55, 0.02), 1.50), true);
});
test('absent cv -> identical to legacy fixed-margin gate', () => {
  assert.equal(passesCommitGate(ok(1.52), 1.50), true);
  assert.equal(passesCommitGate(ok(1.505), 1.50), false);
});

test('requires CV for micro-gain when configured', () => {
  const opts = { significanceMargin: 0.002, noiseK: 0.2, minCommitSpeedup: 1.0, requireCvForMicroGain: true, microGainCeiling: 1.02 };
  assert.equal(passesCommitGate(ok(1.01), 1.0, opts), false);
  assert.equal(passesCommitGate(okv(1.01, 0.0), 1.0, opts), true);
  assert.equal(passesCommitGate(ok(1.03), 1.0, opts), true);
});

// --- formal paired evidence ------------------------------------------------
const formal = (overrides = {}) => ({
  status: 'ok',
  correctness: 'pass',
  verified_geomean: 1.10,
  cv: 0.005,
  protocol: 'paired_abba',
  formal_status: 'accepted',
  infrastructure_status: 'healthy',
  screening_only: false,
  ci_lower: 1.08,
  ci_upper: 1.12,
  positive_block_fraction: 0.9,
  calibrated_margin: 0.01,
  aa_calibration_passed: true,
  bimodal: false,
  ...overrides,
});
const formalOpts = { requireFormalEvidence: true, significanceMargin: 0.002, noiseK: 0.85 };
const harnessReport = (overrides = {}) => ({
  schema_version: 2,
  session_id: 'test-session',
  protocol: 'paired_abba',
  status: 'accepted',
  transient_failures: 0,
  correctness: {
    status: 'pass',
    backend: 'generated-persistent',
    method: 'private_capture_view_baseline_candidate',
    regions_identical: 12,
    regions_equivalent: 12,
    regions_total: 12,
    capture_identity_sha256: 'c'.repeat(64),
    reference_output_sha256: 'd'.repeat(64),
    candidate_output_sha256: 'd'.repeat(64),
    validation_policy_sha256: '',
  },
  health: { status: 'healthy' },
  artifact: {
    baseline_sha256: 'a'.repeat(64),
    candidate_sha256: 'b'.repeat(64),
    capture_identity_sha256: 'c'.repeat(64),
    disk_verified: true,
  },
  device: {
    physical_gpu_uuid: '0x1234',
    physical_gpu_bdf: '0000:01:00.0',
  },
  measurement_contract: {
    health_scope: 'per block',
    backend_protocol: 'hip_abi_persistent_reciprocal_role_swap',
    performance_backend: 'generated-persistent',
    correctness_backend: 'generated-persistent',
    correctness_method: 'private_capture_view_baseline_candidate',
  },
  blocks: Array.from({ length: 5 }, () => ({})),
  aa_blocks: Array.from({ length: 3 }, () => ({})),
  block_health: [
    ...Array.from({ length: 3 }, (_, index) => ({
      phase: 'aa_slot_calibration', index,
      before: { formal_checks: {} }, after: { formal_checks: {} },
    })),
    ...Array.from({ length: 5 }, (_, index) => ({
      phase: 'ab_reciprocal', index,
      before: { formal_checks: {} }, after: { formal_checks: {} },
    })),
  ],
  statistics: {
    speedup_estimate: 1.10,
    speedup_ci95: [1.08, 1.12],
    speedup_cv: 0.005,
    positive_block_fraction: 1.0,
    calibrated_margin: 0.01,
    baseline_us_median: 100.0,
    candidate_us_median: 90.909,
    aa_calibration_passed: true,
    bimodal: false,
  },
  ...overrides,
});

test('authoritative report overrides Director-transcribed performance fields', () => {
  const value = authoritativeFormalValidation({
    correctness: 'fail',
    validation_status: 'unstable',
    director_verified_speedup_geomean: 99,
  }, harnessReport(), '/expected/realcap_paired_result.json', 'test-session');
  assert.equal(value.correctness, 'pass');
  assert.equal(value.validation_status, 'accepted');
  assert.equal(value.director_verified_speedup_geomean, 1.10);
  assert.equal(value.ci_lower, 1.08);
  assert.equal(value.benchmark_report, '/expected/realcap_paired_result.json');
});

test('authoritative report rejects incomplete identity or health evidence', () => {
  assert.throws(() => authoritativeFormalValidation(
    {}, harnessReport({ artifact: {} }), '', 'test-session'), /artifact identity/);
  assert.throws(() => authoritativeFormalValidation(
    {}, harnessReport({
      artifact: {
        baseline_sha256: 'a'.repeat(64),
        candidate_sha256: 'b'.repeat(64),
        capture_identity_sha256: 'c'.repeat(64),
      },
    }), '', 'test-session'), /not verified on disk/);
  assert.throws(() => authoritativeFormalValidation(
    {}, harnessReport({ block_health: [] }), '', 'test-session'), /insufficient blocks/);
  assert.throws(() => authoritativeFormalValidation(
    {}, harnessReport({ measurement_contract: {} }), '', 'test-session'),
  /persistent backend contract/);
  assert.throws(() => authoritativeFormalValidation(
    {}, harnessReport({
      measurement_contract: {
        health_scope: 'per block',
        backend_protocol: 'hip_abi_persistent_reciprocal_role_swap',
        performance_backend: 'hip-abi',
        correctness_backend: 'generated-persistent',
        correctness_method: 'private_capture_view_baseline_candidate',
      },
    }), '', 'test-session'), /persistent backend contract/);
  assert.throws(() => authoritativeFormalValidation(
    {}, harnessReport(), '', 'other-session'), /session identity/);
  assert.throws(() => authoritativeFormalValidation(
    {}, harnessReport({
      correctness: {
        status: 'pass', backend: 'kerncap-validate',
        method: 'shared_capture_output_legacy',
        regions_equivalent: 12, regions_total: 12,
        capture_identity_sha256: 'c'.repeat(64),
        reference_output_sha256: 'd'.repeat(64),
        candidate_output_sha256: 'd'.repeat(64),
      },
    }), '', 'test-session'), /correctness backend/);
});

test('authoritative report rejects phase-boundary-only health evidence', () => {
  const phases = ['aa_slot_calibration', 'ab_forward', 'ab_reverse'];
  const report = harnessReport({
    measurement_contract: {
      health_scope: 'phase boundaries',
      backend_protocol: 'hip_abi_persistent_reciprocal_role_swap',
      performance_backend: 'generated-persistent',
      correctness_backend: 'generated-persistent',
      correctness_method: 'private_capture_view_baseline_candidate',
    },
    block_health: phases.map((phase) => ({
      phase,
      before: { formal_checks: {} },
      after: { formal_checks: {} },
    })),
  });
  assert.throws(() => authoritativeFormalValidation(
    {}, report, '', 'test-session'), /persistent backend contract|phase-boundary health/);
});

test('authoritative report enforces persistent per-block health and contamination', () => {
  const report = harnessReport({
    measurement_contract: {
      health_scope: 'per block',
      backend_protocol: 'hip_abi_persistent_reciprocal_role_swap',
      performance_backend: 'generated-persistent',
      correctness_backend: 'generated-persistent',
      correctness_method: 'private_capture_view_baseline_candidate',
    },
    block_health: [
      ...Array.from({ length: 3 }, (_, index) => ({
        phase: 'aa_slot_calibration', index,
        before: { formal_checks: {} }, after: { formal_checks: {} },
      })),
      ...Array.from({ length: 5 }, (_, index) => ({
        phase: 'ab_reciprocal', index,
        before: { formal_checks: {} }, after: { formal_checks: {} },
      })),
    ],
  });
  assert.equal(authoritativeFormalValidation(
    {}, report, '', 'test-session').validation_status, 'accepted');
  assert.throws(() => authoritativeFormalValidation(
    {}, { ...report, statistics: { ...report.statistics, contaminated: true } },
    '', 'test-session'), /contaminated/);
  assert.throws(() => authoritativeFormalValidation(
    {}, { ...report, block_health: report.block_health.slice(0, -1) },
    '', 'test-session'), /insufficient blocks|per-block/);
});

test('formal gate accepts healthy paired evidence', () => {
  assert.equal(passesCommitGate(formal(), 1.0, formalOpts), true);
});
test('formal gate rejects screening or missing paired protocol', () => {
  assert.equal(passesCommitGate(formal({ screening_only: true }), 1.0, formalOpts), false);
  assert.equal(passesCommitGate(formal({ protocol: 'screening' }), 1.0, formalOpts), false);
});
test('formal gate rejects unhealthy, unstable, or bimodal evidence', () => {
  assert.equal(passesCommitGate(formal({ infrastructure_status: 'infrastructure_unhealthy' }), 1.0, formalOpts), false);
  assert.equal(passesCommitGate(formal({ formal_status: 'unstable' }), 1.0, formalOpts), false);
  assert.equal(passesCommitGate(formal({ aa_calibration_passed: false }), 1.0, formalOpts), false);
  assert.equal(passesCommitGate(formal({ bimodal: true }), 1.0, formalOpts), false);
});
test('formal gate requires CI lower bound and positive blocks to clear current best', () => {
  assert.equal(passesCommitGate(formal({ ci_lower: 1.005 }), 1.0, formalOpts), false);
  assert.equal(passesCommitGate(formal({ positive_block_fraction: 0.7 }), 1.0, formalOpts), false);
  assert.equal(passesCommitGate(formal({ ci_lower: 1.08, verified_geomean: 1.10 }), 1.07, formalOpts), false);
});

// Regression: aiter_fmoe_fp8_blockscale_subgu256 avo-v2 precommit (15 A/B blocks, all
// positive) was rejected because 0.85*cv demanded +0.88% while the CI lower bound —
// which already discounts that same noise — offered +0.71%.
const v2Base = 1.312152;
const v2 = formal({
  verified_geomean: 1.3282021420350438,
  cv: 0.0103684815166065,
  ci_lower: 1.3214186356451958,
  ci_upper: 1.3312790573316955,
  calibrated_margin: 0.0034681536320075868,
  positive_block_fraction: 1.0,
});
test('formal gate charges measurement noise once, via the A/A-calibrated floor', () => {
  assert.equal(passesCommitGate(v2, v2Base, formalOpts), true);
  // The calibrated floor still binds: base*1.00347 = 1.31670.
  assert.equal(passesCommitGate({ ...v2, ci_lower: 1.315 }, v2Base, formalOpts), false);
});
test('formal gate falls back to the cv band with no A/A-calibrated floor', () => {
  assert.equal(passesCommitGate({ ...v2, calibrated_margin: 0 }, v2Base, formalOpts), false);
});
test('screening gate keeps the cv band, having no CI to lean on', () => {
  assert.equal(passesCommitGate({ ...v2, verified_geomean: 1.320025 }, v2Base,
    { significanceMargin: 0.002, noiseK: 0.85 }), false);
});

test('director normalization preserves canonical formal evidence', () => {
  const normalized = normalizeDirectorValidation({
    correctness: 'pass',
    validation_status: 'accepted',
    formal_status: 'accepted',
    director_verified_speedup_geomean: 1.10,
    protocol: 'paired_abba',
    infrastructure_status: 'healthy',
    screening_only: false,
    ci_lower: 1.08,
    ci_upper: 1.12,
    positive_block_fraction: 0.9,
    calibrated_margin: 0.01,
    aa_calibration_passed: true,
    bimodal: false,
  });
  assert.equal(normalized.status, 'ok');
  assert.equal(normalized.ci_lower, 1.08);
  assert.equal(passesCommitGate(normalized, 1.0, formalOpts), true);
});

test('director normalization accepts conventional non-formal status spellings', () => {
  for (const formalStatus of [
    'not_applicable', 'not applicable', 'n/a', 'na', ' N/A ',
  ]) {
    const normalized = normalizeDirectorValidation({
      correctness: 'pass',
      validation_status: 'accepted',
      formal_status: formalStatus,
      director_verified_speedup_geomean: 1.08434,
      cv: 0.00272,
      protocol: 'repeated',
    });
    assert.equal(normalized.status, 'ok', formalStatus);
    assert.equal(passesCommitGate(normalized, 1.0, {
      minCommitSpeedup: 1.002,
      significanceMargin: 0.002,
      noiseK: 0.2,
    }), true, formalStatus);
    assert.equal(passesCommitGate(normalized, 1.0, formalOpts), false, formalStatus);
  }
});

test('director normalization preserves independently remeasured weighted score', () => {
  const normalized = normalizeDirectorValidation({
    correctness: 'pass',
    validation_status: 'accepted',
    formal_status: 'not_applicable',
    director_verified_speedup_geomean: 1.079643,
    director_verified_speedup_weighted: 1.111881,
  });
  assert.equal(normalized.verified_geomean, 1.079643);
  assert.equal(normalized.verified_weighted, 1.111881);
});

test('gate evaluation exposes the normalized rejection reason', () => {
  const normalized = normalizeDirectorValidation({
    correctness: 'pass',
    validation_status: 'accepted',
    formal_status: 'unstable',
    director_verified_speedup_geomean: 1.20,
  });
  const decision = evaluateCommitGate(normalized, 1.0);
  assert.equal(decision.accepted, false);
  assert.match(decision.reason, /status conflict: unstable,accepted/);
});

test('source precommit accepts the observed n/a Director report', () => {
  const normalized = normalizeDirectorValidation({
    correctness: 'pass',
    validation_status: 'accepted',
    formal_status: 'n/a',
    director_verified_speedup_geomean: 1.118382,
    cv: 0.00442,
    protocol: 'repeated',
  });
  const decision = evaluateCommitGate(normalized, 1.092932, {
    minCommitSpeedup: 1.002,
    significanceMargin: 0.002,
    noiseK: 0.2,
    requireCvForMicroGain: true,
    microGainCeiling: 1.02,
  });
  assert.equal(decision.accepted, true);
});

test('director normalization makes non-accepted status fail closed', () => {
  for (const status of ['unstable', 'no_significant_gain', 'infrastructure_unhealthy']) {
    const normalized = normalizeDirectorValidation({
      correctness: 'pass',
      validation_status: status,
      director_verified_speedup_geomean: 2.0,
    });
    assert.equal(normalized.status, 'failed');
    assert.equal(passesCommitGate(normalized, 1.0), false);
  }
});

test('director normalization rejects conflicting formal and validation statuses', () => {
  const normalized = normalizeDirectorValidation({
    correctness: 'pass',
    formal_status: 'accepted',
    validation_status: 'unstable',
    director_verified_speedup_geomean: 2.0,
  });
  assert.equal(normalized.status, 'failed');
});

test('ASM formal precommit cannot be disabled by configuration', () => {
  assert.equal(formalPrecommitEnabled(true, false), true);
  assert.equal(formalPrecommitEnabled(false, false), false);
});

test('final ASM decision rejects missing evidence and unhealthy results', () => {
  assert.equal(classifyFinalValidation({
    correctness: 'pass', validation_status: 'accepted',
    director_verified_speedup_geomean: 2.0,
  }, true, formalOpts).status, 'no_significant_gain');
  assert.equal(classifyFinalValidation({
    correctness: 'pass', validation_status: 'accepted',
    formal_status: 'accepted', infrastructure_status: 'infrastructure_unhealthy',
    director_verified_speedup_geomean: 2.0,
  }, true, formalOpts).status, 'infrastructure_unhealthy');
  assert.equal(classifyFinalValidation({
    correctness: 'pass', validation_status: 'accepted',
    formal_status: 'accepted', infrastructure_status: 'healthy',
    aa_calibration_passed: false,
    director_verified_speedup_geomean: 2.0,
  }, true, formalOpts).status, 'unstable');
});

test('final ASM decision accepts complete paired evidence', () => {
  const raw = {
    correctness: 'pass',
    validation_status: 'accepted',
    formal_status: 'accepted',
    director_verified_speedup_geomean: 1.10,
    cv: 0.005,
    protocol: 'paired_abba',
    infrastructure_status: 'healthy',
    screening_only: false,
    ci_lower: 1.08,
    ci_upper: 1.12,
    positive_block_fraction: 0.9,
    calibrated_margin: 0.01,
    aa_calibration_passed: true,
    bimodal: false,
  };
  assert.deepEqual(
    classifyFinalValidation(raw, true, formalOpts).status,
    'accepted');
  assert.deepEqual(
    classifyFinalValidation(raw, true, formalOpts, 1.20).status,
    'reproduction_mismatch');
});

test('final source decision must reproduce the committed lineage best', () => {
  const raw = {
    correctness: 'pass',
    validation_status: 'accepted',
    formal_status: 'not_applicable',
    infrastructure_status: 'healthy',
    director_verified_speedup_geomean: 1.10,
  };
  assert.equal(
    classifyFinalValidation(raw, false, formalOpts, 1.12).status,
    'accepted');
  assert.equal(
    classifyFinalValidation(raw, false, {}, 1.20).status,
    'accepted');
  assert.equal(
    classifyFinalValidation(raw, false, {}, 1.25).status,
    'reproduction_mismatch');
  // Tight 3% tolerance still flags the historical Kimi director/lineage gap.
  assert.equal(
    classifyFinalValidation(raw, false, { finalReproductionTolerance: 0.03 }, 1.20).status,
    'reproduction_mismatch');
});

console.log(`\n${passed} passed${process.exitCode ? ' (with failures)' : ''}`);
