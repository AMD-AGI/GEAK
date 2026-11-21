// Unit tests for the roofline model (pure; no GPU/LLM).
// Run on the rig (node not required locally):  node tests/roofline.test.mjs
// Exit code 0 = all pass; non-zero = failure (CI-friendly).

import assert from 'node:assert/strict';
import {
  analyzeCase, classifyRegime, ridgePoint, rooflineSummary, recommendLevers,
  assignSpecialists, PEAKS,
} from '../lib/roofline.mjs';

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`ok   - ${name}`); }
  catch (e) { console.error(`FAIL - ${name}\n      ${e.message}`); process.exitCode = 1; }
}
const approx = (a, b, eps = 1e-3) => Math.abs(a - b) <= eps;

// --- ridge point ---------------------------------------------------------
test('fp16 ridge point ~= 245.5 FLOP/byte', () => {
  assert.ok(approx(ridgePoint('fp16'), PEAKS.gfx942.compute.fp16 / PEAKS.gfx942.hbm_bw, 1e-6));
  assert.ok(ridgePoint('fp16') > 240 && ridgePoint('fp16') < 250);
});
test('fp8 ridge is ~2x fp16 (2x peak compute, same BW)', () => {
  assert.ok(approx(ridgePoint('fp8') / ridgePoint('fp16'), 2.0, 1e-3));
});

// --- regime classification ----------------------------------------------
test('basement: far below BOTH ceilings', () => {
  assert.equal(classifyRegime({ ai: 10, ridge_ai: 245, mem_frac: 0.1, cmp_frac: 0.02 }), 'basement');
});
test('memory-bound: near BW ceiling, AI < ridge', () => {
  assert.equal(classifyRegime({ ai: 10, ridge_ai: 245, mem_frac: 0.5, cmp_frac: 0.02 }), 'memory-bound');
});
test('compute-bound: near FLOP ceiling, AI >= ridge', () => {
  assert.equal(classifyRegime({ ai: 1000, ridge_ai: 245, mem_frac: 0.12, cmp_frac: 0.5 }), 'compute-bound');
});
test('unknown when fractions missing', () => {
  assert.equal(classifyRegime({ ai: 10, ridge_ai: 245, mem_frac: NaN, cmp_frac: 0.5 }), 'unknown');
});

// --- analyzeCase: memory-bound synthetic (AI=10, 50% of BW) --------------
test('analyzeCase: memory-bound at ~0.5 of roofline', () => {
  const bytes = 1e9, flops = 1e10;                 // AI = 10
  const latency_ms = (bytes / (0.5 * PEAKS.gfx942.hbm_bw)) * 1000; // 50% of peak BW
  const a = analyzeCase({ name: 'm', latency_ms, flops, bytes, dtype: 'fp16' });
  assert.equal(a.regime, 'memory-bound');
  assert.ok(approx(a.ai, 10, 1e-6));
  assert.ok(approx(a.fraction_of_roofline, 0.5, 1e-3));
  assert.ok(approx(a.mem_frac, 0.5, 1e-3));
  assert.ok(approx(a.potential_speedup, 2.0, 1e-2));
});

// --- analyzeCase: compute-bound synthetic (AI=1000, 50% of FLOPs) --------
test('analyzeCase: compute-bound at ~0.5 of roofline', () => {
  const bytes = 1e9, flops = 1e12;                 // AI = 1000 (>> ridge)
  const latency_ms = (flops / (0.5 * PEAKS.gfx942.compute.fp16)) * 1000;
  const a = analyzeCase({ name: 'c', latency_ms, flops, bytes, dtype: 'fp16' });
  assert.equal(a.regime, 'compute-bound');
  assert.ok(approx(a.fraction_of_roofline, 0.5, 1e-3));
  assert.ok(approx(a.cmp_frac, 0.5, 1e-3));
});

// --- analyzeCase: near-roofline + basement flags ------------------------
test('near_roofline flag fires above default 0.80', () => {
  const bytes = 1e9, flops = 1e12;
  const latency_ms = (flops / (0.9 * PEAKS.gfx942.compute.fp16)) * 1000;
  const a = analyzeCase({ latency_ms, flops, bytes });
  assert.equal(a.near_roofline, true);
});
test('basement regime when far below both ceilings', () => {
  const bytes = 1e9, flops = 1e10;                 // AI = 10
  const latency_ms = (bytes / (0.1 * PEAKS.gfx942.hbm_bw)) * 1000; // 10% of BW
  const a = analyzeCase({ latency_ms, flops, bytes });
  assert.equal(a.regime, 'basement');
});

// --- graceful degradation -----------------------------------------------
test('missing flops/bytes -> unknown, null fields', () => {
  const a = analyzeCase({ name: 'x', latency_ms: 1.0 });
  assert.equal(a.regime, 'unknown');
  assert.equal(a.fraction_of_roofline, null);
  assert.equal(a.potential_speedup, null);
});

// --- telemetry sanity: fraction_of_roofline > 1 is impossible -----------
test('fraction > 1 (impossible) -> flagged invalid, null fields like missing telemetry', () => {
  // tiny latency vs flops => achieved >> peak => fraction >> 1
  const a = analyzeCase({ name: 'bad', flops: 1e12, bytes: 1e9, latency_ms: 0.001, dtype: 'fp16' });
  assert.equal(a.fraction_of_roofline, null);
  assert.equal(a.regime, 'unknown');
  assert.equal(a.potential_speedup, null);
  assert.equal(a.telemetry_invalid, true);
});
test('within-tolerance fraction slightly above 1 is kept as noise', () => {
  const flops = 1e12, bytes = 1e9;                  // AI=1000 (compute), 1.03x of ceiling
  const latency_ms = (flops / (1.03 * PEAKS.gfx942.compute.fp16)) * 1000;
  const a = analyzeCase({ name: 'noise', flops, bytes, latency_ms });
  assert.notEqual(a.fraction_of_roofline, null);
  assert.equal(a.telemetry_invalid, false);
});
test('summary: all-invalid telemetry -> ok=false (never a false roofline stop)', () => {
  const bad = { name: 'bad', flops: 1e12, bytes: 1e9, latency_ms: 0.001 };
  const s = rooflineSummary([bad]);
  assert.equal(s.ok, false);
  assert.equal(s.n_valid, 0);
  assert.equal(s.n_invalid, 1);
  assert.equal(s.geomean_fraction, null);
});
test('summary: invalid case excluded, valid case still drives steering', () => {
  const bad = { name: 'bad', flops: 1e12, bytes: 1e9, latency_ms: 0.001 };
  const good = { name: 'mem', flops: 1e10, bytes: 1e9,            // AI=10, 50% BW
    latency_ms: (1e9 / (0.5 * PEAKS.gfx942.hbm_bw)) * 1000 };
  const s = rooflineSummary([good, bad]);
  assert.equal(s.ok, true);
  assert.equal(s.n_valid, 1);
  assert.equal(s.n_invalid, 1);
  assert.equal(s.worst_case.name, 'mem');
  assert.ok(approx(s.geomean_fraction, 0.5, 1e-3));
});

// --- rooflineSummary -----------------------------------------------------
test('summary picks worst (most potential) case + its regime', () => {
  const peak = PEAKS.gfx942;
  const memCase = { name: 'mem', flops: 1e10, bytes: 1e9,            // AI=10, 50% BW -> 2x left
    latency_ms: (1e9 / (0.5 * peak.hbm_bw)) * 1000 };
  const cmpCase = { name: 'cmp', flops: 1e12, bytes: 1e9,            // AI=1000, 90% FLOPs -> ~1.1x left
    latency_ms: (1e12 / (0.9 * peak.compute.fp16)) * 1000 };
  const s = rooflineSummary([memCase, cmpCase]);
  assert.equal(s.ok, true);
  assert.equal(s.n_valid, 2);
  assert.equal(s.worst_case.name, 'mem');           // 2x potential beats 1.1x
  assert.equal(s.dominant_regime, 'memory-bound');
});
test('summary ok=false when no case has telemetry', () => {
  const s = rooflineSummary([{ name: 'a', latency_ms: 1 }, { name: 'b', latency_ms: 2 }]);
  assert.equal(s.ok, false);
  assert.equal(s.geomean_fraction, null);
});

// --- recommendLevers -----------------------------------------------------
test('lever specialties map to the team taxonomy', () => {
  assert.equal(recommendLevers('memory-bound').specialty, 'memory');
  assert.equal(recommendLevers('compute-bound').specialty, 'compute');
  assert.equal(recommendLevers('basement').specialty, 'host_runtime');
  assert.equal(recommendLevers('unknown').specialty, 'algorithm');
  assert.ok(recommendLevers('memory-bound').levers.length > 0);
});

// --- A2: measured bytes preferred over analytic --------------------------
test('analyzeCase prefers measured_bytes over analytic bytes', () => {
  const peak = PEAKS.gfx942;
  // analytic says AI=10 (memory-bound); measured HBM bytes are 10x smaller (cache
  // reuse) -> AI=100, a different regime. The measured value must win.
  const latency_ms = (1e10 / (0.5 * peak.compute.fp16)) * 1000; // arbitrary positive
  const a = analyzeCase({ name: 'g', latency_ms, flops: 1e11, bytes: 1e10, measured_bytes: 1e9 });
  assert.equal(a.bytes_source, 'measured');
  assert.ok(Math.abs(a.ai - 100) < 1e-6);             // 1e11 / 1e9 (measured), not /1e10
});
test('analyzeCase falls back to analytic bytes + tags source', () => {
  const a = analyzeCase({ name: 'g', latency_ms: 1.0, flops: 1e11, bytes: 1e10 });
  assert.equal(a.bytes_source, 'analytic');
  assert.ok(Math.abs(a.ai - 10) < 1e-6);
});
test('analyzeCase carries rocprof counters through', () => {
  const a = analyzeCase({ name: 'g', latency_ms: 1.0, flops: 1e11, bytes: 1e10,
    counters: { occupancy_pct: 62, mfma_active_pct: 71 } });
  assert.equal(a.counters.occupancy_pct, 62);
});

// --- assignSpecialists (operator=team regime->specialist dispatch) -------
test('assignSpecialists: dominant regime first, then orthogonal cover', () => {
  assert.deepEqual(assignSpecialists('memory-bound', 3), ['memory', 'compute', 'host_runtime']);
  assert.deepEqual(assignSpecialists('compute-bound', 2), ['compute', 'memory']);
  assert.deepEqual(assignSpecialists('basement', 1), ['host_runtime']);
});
test('assignSpecialists: unknown -> algorithm-first generic order', () => {
  assert.deepEqual(assignSpecialists('unknown', 4), ['algorithm', 'memory', 'compute', 'host_runtime']);
  assert.deepEqual(assignSpecialists(undefined, 1), ['algorithm']);
});
test('assignSpecialists: cycles when k > 4', () => {
  assert.deepEqual(assignSpecialists('memory-bound', 5),
    ['memory', 'compute', 'host_runtime', 'algorithm', 'memory']);
});

test('gfx950 MI355X peaks and fp16 ridge ~312 FLOP/byte', () => {
  assert.equal(PEAKS.gfx950.label, 'MI355X');
  assert.equal(PEAKS.gfx950.compute.fp16, 2.5e15);
  assert.equal(PEAKS.gfx950.compute.fp4, 10.0e15);
  assert.equal(PEAKS.gfx950.hbm_bw, 8.0e12);
  assert.ok(approx(ridgePoint('fp16', 'gfx950'), 2.5e15 / 8.0e12, 1e-9));
  assert.equal(PEAKS.gfx950_mi355.compute.fp8, PEAKS.gfx950.compute.fp8);
});

console.log(`\n${passed} passed${process.exitCode ? ' (with failures)' : ''}`);
