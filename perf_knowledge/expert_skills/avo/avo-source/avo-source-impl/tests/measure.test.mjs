// Unit tests for robust measurement helpers (pure).
// Run on the rig (node not required locally):  node tests/measure.test.mjs

import assert from 'node:assert/strict';
import {
  median, mean, stddev, mad, cv, dropOutliers, summarizeRepeats, aggregateCV,
} from '../lib/measure.mjs';

let passed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log(`ok   - ${name}`); }
  catch (e) { console.error(`FAIL - ${name}\n      ${e.message}`); process.exitCode = 1; }
}
const approx = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

test('median odd/even/empty', () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([1, 2, 3, 4]), 2.5);
  assert.equal(median([]), null);
});
test('mean / stddev', () => {
  assert.ok(approx(mean([2, 4, 6]), 4));
  assert.ok(approx(stddev([2, 4, 6]), Math.sqrt(((2 - 4) ** 2 + 0 + (6 - 4) ** 2) / 3)));
  assert.equal(stddev([5]), 0);
});
test('mad robust spread', () => {
  assert.equal(mad([1, 1, 1, 1]), 0);
  assert.equal(mad([1, 2, 3, 4, 5]), 1);     // median 3; |dev|=2,1,0,1,2 -> median 1
});
test('cv = stddev/mean, null when undefined', () => {
  assert.ok(approx(cv([10, 10, 10]), 0));
  assert.equal(cv([]), null);
});
test('ignores non-finite values', () => {
  assert.equal(median([1, NaN, 2, Infinity, 3]), 2);
});

test('dropOutliers removes a spike, keeps the cluster', () => {
  const xs = [10, 10.1, 9.9, 10.2, 100];      // 100 is a throttled outlier
  const kept = dropOutliers(xs, 3);
  assert.ok(!kept.includes(100));
  assert.equal(kept.length, 4);
});
test('dropOutliers no-op for <4 points', () => {
  assert.deepEqual(dropOutliers([1, 100, 2]), [1, 100, 2]);
});
test('dropOutliers falls back when MAD=0', () => {
  assert.deepEqual(dropOutliers([5, 5, 5, 5]), [5, 5, 5, 5]);
});

test('summarizeRepeats: outlier-filtered median + cv + counts', () => {
  const s = summarizeRepeats([10, 10.1, 9.9, 10.2, 100]);
  assert.ok(s.median >= 9.9 && s.median <= 10.2);
  assert.equal(s.n_raw, 5);
  assert.equal(s.n_used, 4);                  // spike dropped
  assert.ok(s.cv < 0.05);                     // tight cluster -> low noise
});
test('summarizeRepeats empty', () => {
  const s = summarizeRepeats([]);
  assert.equal(s.median, null);
  assert.equal(s.n_used, 0);
});

test('aggregateCV is the worst (max) shape noise', () => {
  assert.ok(approx(aggregateCV([0.01, 0.05, 0.02]), 0.05));
  assert.equal(aggregateCV([]), null);
});

console.log(`\n${passed} passed${process.exitCode ? ' (with failures)' : ''}`);
