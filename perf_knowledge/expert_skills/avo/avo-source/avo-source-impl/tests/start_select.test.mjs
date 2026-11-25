// Unit tests for Diverse Fan-out start-point selection.
// Run on the rig:  node tests/start_select.test.mjs   (exit 0 = all pass)

import assert from 'node:assert/strict';
import { selectStartPoints } from '../lib/start_select.mjs';

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`ok   - ${name}`); }
  catch (e) { console.error(`FAIL - ${name}\n      ${e.message}`); process.exitCode = 1; }
}

const lineage = [
  { id: 'v0', ref: 'avo-v0', speedup: 1.0, strategy: 'baseline' },
  { id: 'v1', ref: 'avo-v1', speedup: 1.4, strategy: 'tiling' },
  { id: 'v2', ref: 'avo-v2', speedup: 1.9, strategy: 'tiling' },     // best, same bucket as v1
  { id: 'v3', ref: 'avo-v3', speedup: 1.6, strategy: 'vectorize' },
];

test('policy=best returns only the global best', () => {
  assert.deepEqual(selectStartPoints(lineage, { policy: 'best', k: 3 }), ['avo-v2']);
});
test('k=1 collapses to best regardless of policy', () => {
  assert.deepEqual(selectStartPoints(lineage, { policy: 'diverse', k: 1 }), ['avo-v2']);
});
test('topk: best first, then by speedup, baseline excluded', () => {
  assert.deepEqual(selectStartPoints(lineage, { policy: 'topk', k: 3 }), ['avo-v2', 'avo-v3', 'avo-v1']);
});
test('diverse: one per strategy bucket (best of tiling=v2, vectorize=v3), best first', () => {
  // buckets: tiling->v2(1.9), vectorize->v3(1.6); reps ranked => [v2, v3]; best first.
  assert.deepEqual(selectStartPoints(lineage, { policy: 'diverse', k: 3 }), ['avo-v2', 'avo-v3', 'avo-v1']);
});
test('diverse caps at k', () => {
  assert.deepEqual(selectStartPoints(lineage, { policy: 'diverse', k: 2 }), ['avo-v2', 'avo-v3']);
});
test('only baseline -> [best] (nothing to diversify)', () => {
  assert.deepEqual(selectStartPoints([{ id: 'v0', ref: 'avo-v0', speedup: 1.0, strategy: 'baseline' }],
    { policy: 'diverse', k: 3 }), ['avo-v0']);
});
test('empty lineage -> []', () => {
  assert.deepEqual(selectStartPoints([], { policy: 'topk', k: 3 }), []);
});

console.log(`\n${passed} passed${process.exitCode ? ' (with failures)' : ''}`);
