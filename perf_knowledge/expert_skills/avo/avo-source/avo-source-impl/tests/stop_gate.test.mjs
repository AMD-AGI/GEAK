// Run on the rig (node not required locally):  node tests/stop_gate.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  affordableRounds, canAffordAnotherRound, roundCostS, shouldHonorTeamStop,
} from '../lib/stop_gate.mjs';

// The run that prompted this gate: a 12h budget that exited at round 7, ~2.3h in, with
// ~10h unspent. Seven rounds in 8280s average ~1183s each.
const REPORTED = { remainingS: 34920, estStepS: 1183, overheadS: 150 };

test('a round costs the round plus the supervisor overhead ahead of it', () => {
  assert.equal(roundCostS(REPORTED), 1333);
});

test('the reported early exit is refused because the budget plainly fits more rounds', () => {
  assert.equal(affordableRounds(REPORTED), 26);
  const decision = shouldHonorTeamStop({
    stop: true, earlyStop: false, streak: 3, confirmations: 3, ...REPORTED,
  });
  assert.equal(decision.honored, false);
  assert.equal(decision.reason, 'budget_remains');
});

test('confirmations alone cannot buy a stop while budget remains', () => {
  // This is the loophole that lost the 10h: TechLead re-issued the stop until the streak
  // was met, so any streak-only rule was satisfiable by repetition.
  for (const streak of [3, 4, 10, 99]) {
    const decision = shouldHonorTeamStop({
      stop: true, earlyStop: false, streak, confirmations: 3, ...REPORTED,
    });
    assert.equal(decision.honored, false, `streak=${streak}`);
  }
});

test('early_stop is the operator opting in, so it honors a stop immediately', () => {
  const decision = shouldHonorTeamStop({
    stop: true, earlyStop: true, streak: 1, confirmations: 3, ...REPORTED,
  });
  assert.equal(decision.honored, true);
  assert.equal(decision.reason, 'early_stop_enabled');
});

test('a spent budget honors a confirmed stop instead of starting a round that cannot finish', () => {
  const decision = shouldHonorTeamStop({
    stop: true, earlyStop: false, streak: 3, confirmations: 3,
    remainingS: 900, estStepS: 1183, overheadS: 150,
  });
  assert.equal(decision.honored, true);
  assert.equal(decision.reason, 'budget_spent');
});

test('a spent budget still requires the confirmations', () => {
  const decision = shouldHonorTeamStop({
    stop: true, earlyStop: false, streak: 1, confirmations: 3,
    remainingS: 900, estStepS: 1183, overheadS: 150,
  });
  assert.equal(decision.honored, false);
  assert.equal(decision.reason, 'awaiting_confirmations');
});

test('the boundary is whether one whole round still fits', () => {
  const spent = { estStepS: 1000, overheadS: 200 };   // one round costs 1200s
  assert.equal(canAffordAnotherRound({ ...spent, remainingS: 1200 }), true);
  assert.equal(canAffordAnotherRound({ ...spent, remainingS: 1199 }), false);
});

test('an unmeasurable budget counts as affordable so the gate never fires on missing data', () => {
  // Before any round completes there is no duration sample; refusing the stop is the
  // recoverable direction (a wasted round), honoring it is not (a truncated run).
  assert.equal(affordableRounds({ remainingS: 34920, estStepS: 0, overheadS: 150 }), null);
  assert.equal(canAffordAnotherRound({ remainingS: 34920, estStepS: 0, overheadS: 150 }), true);
  assert.equal(shouldHonorTeamStop({
    stop: true, earlyStop: false, streak: 5, confirmations: 3,
    remainingS: 0, estStepS: 0, overheadS: 150,
  }).honored, false);
});

test('no stop request is not a stop decision', () => {
  const decision = shouldHonorTeamStop({
    stop: false, earlyStop: true, streak: 0, confirmations: 3, ...REPORTED,
  });
  assert.equal(decision.honored, false);
  assert.equal(decision.reason, 'no_stop_requested');
});
