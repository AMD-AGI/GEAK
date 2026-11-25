import test from 'node:test';
import assert from 'node:assert/strict';

import {
  dedupeVerifyItems, selectScreeningFinalist,
} from '../lib/verify_plan.mjs';

test('identical candidate bytes on the same base are verified once', () => {
  const items = [
    { i: 0, startRef: 'avo-v0', candidateIdentity: 'sha256:a' },
    { i: 3, startRef: 'avo-v0', candidateIdentity: 'sha256:a' },
    { i: 1, startRef: 'avo-v0', candidateIdentity: 'sha256:b' },
  ];
  const plan = dedupeVerifyItems(items);
  assert.deepEqual(plan.unique.map((item) => item.i), [0, 1]);
  assert.equal(plan.duplicates.length, 1);
  assert.equal(plan.duplicates[0].duplicateOf, 0);
});

test('identical bytes on different base refs are not deduplicated', () => {
  const plan = dedupeVerifyItems([
    { i: 0, startRef: 'avo-v0', candidateIdentity: 'sha256:a' },
    { i: 1, startRef: 'avo-v1', candidateIdentity: 'sha256:a' },
  ]);
  assert.equal(plan.unique.length, 2);
  assert.equal(plan.duplicates.length, 0);
});

test('items without an identity remain independently verifiable', () => {
  const plan = dedupeVerifyItems([{ i: 0 }, { i: 1 }]);
  assert.equal(plan.unique.length, 2);
  assert.equal(plan.duplicates.length, 0);
});

test('screening finalist is the fastest clean candidate with stable ties', () => {
  const result = selectScreeningFinalist([
    { i: 2, ver: { status: 'ok', correctness: 'pass', verified_geomean: 1.1 } },
    { i: 0, ver: { status: 'ok', correctness: 'pass', verified_geomean: 1.1 } },
    { i: 1, ver: { status: 'failed', correctness: 'pass', verified_geomean: 2 } },
  ]);
  assert.equal(result.i, 0);
});
