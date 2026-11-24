// Unit tests for roofline-driven stop + attention allocation (pure).
// Run on the rig (node not required locally):  node tests/headroom.test.mjs

import assert from 'node:assert/strict';
import { shouldStopRoofline, allocateAttention } from '../lib/headroom.mjs';

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`ok   - ${name}`); }
  catch (e) { console.error(`FAIL - ${name}\n      ${e.message}`); process.exitCode = 1; }
}

const summary = (frac, ok = true) => ({ ok, geomean_fraction: frac });

// --- shouldStopRoofline --------------------------------------------------
test('stops when at/above target and min commits met', () => {
  assert.equal(shouldStopRoofline(summary(0.90), { target: 0.85, commits: 3, minCommits: 1 }).stop, true);
});
test('does NOT stop below target', () => {
  assert.equal(shouldStopRoofline(summary(0.70), { target: 0.85, commits: 3 }).stop, false);
});
test('does NOT stop before min commits even if near roofline', () => {
  assert.equal(shouldStopRoofline(summary(0.95), { target: 0.85, commits: 0, minCommits: 2 }).stop, false);
});
test('no telemetry -> never stops (relative-only mode)', () => {
  assert.equal(shouldStopRoofline(summary(null, false), { target: 0.85, commits: 5 }).stop, false);
  assert.equal(shouldStopRoofline(null, { commits: 5 }).stop, false);
});
test('exactly at target stops', () => {
  assert.equal(shouldStopRoofline(summary(0.85), { target: 0.85, commits: 1 }).stop, true);
});

// --- allocateAttention ---------------------------------------------------
const A = [
  { name: 'big', regime: 'memory-bound', fraction_of_roofline: 0.5, potential_speedup: 2.0 },
  { name: 'mid', regime: 'compute-bound', fraction_of_roofline: 0.8, potential_speedup: 1.25 },
  { name: 'done', regime: 'compute-bound', fraction_of_roofline: 0.98, potential_speedup: 1.0 }, // skipped
];

test('uniform weights: focus = most potential speedup', () => {
  const r = allocateAttention(A);
  assert.equal(r.focus.name, 'big');
  assert.equal(r.ranked.length, 2);            // saturated 'done' filtered out
});
test('time weights flip the focus (Amdahl)', () => {
  // 'mid' is small headroom but dominates runtime -> bigger opportunity.
  const r = allocateAttention(A, { weights: { big: 1, mid: 100, done: 1 } });
  assert.equal(r.focus.name, 'mid');
});
test('no telemetry -> no focus', () => {
  const r = allocateAttention([{ name: 'x', potential_speedup: null }]);
  assert.equal(r.focus, null);
  assert.deepEqual(r.ranked, []);
});

console.log(`\n${passed} passed${process.exitCode ? ' (with failures)' : ''}`);
