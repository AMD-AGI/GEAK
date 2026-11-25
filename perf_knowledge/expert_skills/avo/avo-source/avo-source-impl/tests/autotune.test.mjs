// Unit tests for the bounded autotune sweep (pure).
// Run on the rig (node not required locally):  node tests/autotune.test.mjs

import assert from 'node:assert/strict';
import {
  configKey, expandConfigSpace, selectBestConfig, excludeFailed,
} from '../lib/autotune.mjs';

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`ok   - ${name}`); }
  catch (e) { console.error(`FAIL - ${name}\n      ${e.message}`); process.exitCode = 1; }
}

// --- configKey -----------------------------------------------------------
test('configKey is order-independent + stable', () => {
  assert.equal(configKey({ b: 2, a: 1 }), configKey({ a: 1, b: 2 }));
  assert.equal(configKey({ a: 1, b: 2 }), 'a=1,b=2');
  assert.equal(configKey(null), '');
});

// --- expandConfigSpace ---------------------------------------------------
test('cartesian product of the grid', () => {
  const cfgs = expandConfigSpace({ BLOCK_M: [64, 128], num_warps: [4, 8] });
  assert.equal(cfgs.length, 4);
  assert.ok(cfgs.some(c => c.BLOCK_M === 64 && c.num_warps === 4));
  assert.ok(cfgs.some(c => c.BLOCK_M === 128 && c.num_warps === 8));
});
test('caps deterministically + keeps the first combination', () => {
  const cfgs = expandConfigSpace({ a: [1, 2, 3, 4], b: [1, 2, 3, 4] }, { max: 5 }); // 16 -> 5
  assert.equal(cfgs.length, 5);
  assert.deepEqual(cfgs[0], { a: 1, b: 1 });            // first (default) always kept
  // deterministic: same call yields the same subsample
  assert.deepEqual(cfgs, expandConfigSpace({ a: [1, 2, 3, 4], b: [1, 2, 3, 4] }, { max: 5 }));
});
test('scalar values and empty space', () => {
  assert.deepEqual(expandConfigSpace({ x: 7 }), [{ x: 7 }]);
  assert.deepEqual(expandConfigSpace({}), []);
});

// --- selectBestConfig ----------------------------------------------------
test('picks best verified geomean among correct results', () => {
  const r = selectBestConfig([
    { config: { a: 1 }, verified_geomean: 1.4, correct: true },
    { config: { a: 2 }, verified_geomean: 1.9, correct: true },
    { config: { a: 3 }, verified_geomean: 2.5, correct: false },  // incorrect -> ineligible
  ]);
  assert.deepEqual(r.config, { a: 2 });
});
test('tie on geomean -> closer to roofline wins', () => {
  const r = selectBestConfig([
    { config: { a: 1 }, verified_geomean: 1.90, fraction_of_roofline: 0.6 },
    { config: { a: 2 }, verified_geomean: 1.90, fraction_of_roofline: 0.8 },
  ]);
  assert.deepEqual(r.config, { a: 2 });
});
test('no eligible result -> null', () => {
  assert.equal(selectBestConfig([{ config: {}, verified_geomean: 0 }]), null);
  assert.equal(selectBestConfig([]), null);
});

// --- excludeFailed -------------------------------------------------------
test('excludeFailed drops known-bad config points', () => {
  const cfgs = expandConfigSpace({ a: [1, 2], b: [1] });   // a=1,b=1 ; a=2,b=1
  const kept = excludeFailed(cfgs, [configKey({ a: 1, b: 1 })]);
  assert.equal(kept.length, 1);
  assert.deepEqual(kept[0], { a: 2, b: 1 });
});

console.log(`\n${passed} passed${process.exitCode ? ' (with failures)' : ''}`);
