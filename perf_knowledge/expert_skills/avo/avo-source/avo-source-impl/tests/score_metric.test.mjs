import assert from 'node:assert/strict';
import {
  normalizeCommitScoreField,
  verifiedCommitScore,
  withCommitScore,
} from '../lib/score_metric.mjs';

assert.equal(normalizeCommitScoreField(), 'verified_geomean');
assert.equal(normalizeCommitScoreField('verified_weighted'), 'verified_weighted');
assert.throws(() => normalizeCommitScoreField('arithmetic'), /unsupported/);

const ver = { verified_geomean: 1.079643, verified_weighted: 1.111881 };
assert.equal(verifiedCommitScore(ver, 'verified_geomean'), 1.079643);
assert.equal(verifiedCommitScore(ver, 'verified_weighted'), 1.111881);
assert.equal(verifiedCommitScore(
  { verified_geomean: 1.02 }, 'verified_weighted'), 1.02);

const selected = withCommitScore(ver, 'verified_weighted');
assert.equal(selected.verified_geomean, 1.111881);
assert.equal(selected.raw_verified_geomean, 1.079643);
assert.equal(ver.verified_geomean, 1.079643);

console.log('9 passed');
