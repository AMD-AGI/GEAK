#!/usr/bin/env node
// A round that measured nothing must not be counted as a round that found nothing.
//
// `max_no_improve` exists to stop a search that has run out of ideas. To conclude that, the round
// has to have looked. One run launched with budget 8 and max_no_improve 3, ran three rounds, spent
// seven of its eight budget units and two of its GPU leases, and stopped at the top of round 4 with
// a lease still in the budget and round 4 named in its own plan as the round that had to spend it.
// Of the three rounds the counter charged, exactly one had measured a candidate. Round 1's three
// directions returned static analysis and copies of the baseline latencies; round 2's two returned
// the same. The run ended because a counter meant for "no more wins" had been fed two rounds of
// "no readings".
//
// The rule this file pins: a round produces evidence only when at least one direction returns a
// number the round is allowed to believe, judged by the SAME admissibility the promotion path uses.
// The failure modes are all ways the exemption could be written too narrowly (and one earlier
// version was: it required EVERY direction to be `inactive`, so one direction returning nothing at
// all broke it, and "never got the lease" was never `inactive` to begin with) or too widely (a VOID
// reading counted as evidence, which puts the stopping criterion back where it started).
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const region = (name) => {
  const m = src.match(new RegExp(`// <<REPLAY:${name}>>([\\s\\S]*?)// <</REPLAY:${name}>>`));
  if (!m) { console.error(`  FAIL: no <<REPLAY:${name}>> region — nothing to test`); process.exit(1); }
  return m[1];
};
// The real predicates, not re-implementations: `rungOutcomeOf` decides what counts as measured and
// it asks `stepRoleOf` how the step closes, so all three regions are loaded together.
const { roundEvidence, rungOutcomeOf, stepRoleOf } =
  // eslint-disable-next-line no-new-func
  new Function(`${region('enabling_step')}\n${region('open_rungs')}\n${region('evidence_stop')}\n` +
    'return { roundEvidence, rungOutcomeOf, stepRoleOf };')();

const EV = (clean) => roundEvidence(clean, stepRoleOf, rungOutcomeOf);

// A direction that ran, was verified, and produced a number — win or loss.
const MEASURED = (id, geo) => ({
  d: { id, roadmap_rung: 'D1' },
  eng: { status: 'success' },
  ver: { status: 'verified', correctness: 'pass', activation_confirmed: 'yes', verified_geomean: geo },
});

console.log('\n# a round that measured something');
{
  const ev = EV([MEASURED('r3_d0', 0.9914)]);
  ok(ev.measured === 1, 'a candidate that RAN AND LOST is evidence — it is exactly what max_no_improve is for');
  ok(ev.gaps.length === 0, 'and nothing is reported as missing');
}
{
  const ev = EV([MEASURED('r3_d0', 0.9914), { d: { id: 'r3_d1' }, eng: { status: 'partial' }, ver: null }]);
  ok(ev.measured === 1,
     'one real reading is enough: a round is not disqualified because a SECOND direction returned nothing');
  ok(ev.total === 2 && ev.gaps.length === 1 && /r3_d1/.test(ev.gaps[0]),
     'and the direction that returned nothing is still named, so the round log says what was lost');
}

console.log('\n# the wave-14 rounds, one at a time');
{
  // Round 1: two engineers planned for the GPU never got the lease and returned static analysis;
  // the third copied the baseline latencies into `optimized_ms` and its path was never observed.
  const ev = EV([
    { d: { id: 'r1_d0' }, eng: { status: 'success' }, ver: { status: 'verified', verified_geomean: 1.0 }, inactive: 'unknown' },
    { d: { id: 'r1_d1' }, eng: { status: 'partial' }, ver: { status: 'verified', verified_geomean: 1.0 }, inactive: 'no' },
    { d: { id: 'r1_d2' }, eng: { status: 'partial' }, ver: null },
  ]);
  ok(ev.measured === 0, 'round 1 measured NOTHING, so it must not be charged to the no-improve counter');
  ok(ev.gaps.length === 3, 'and every one of the three directions says why it is not evidence');
  ok(/not proven to execute/.test(ev.gaps[0]) && /not executed/.test(ev.gaps[1]),
     'an unproven activation and a disproven one are distinguished, because they are fixed differently');
}
{
  // The narrow version of this rule was `clean.every(r => r.inactive)`. This is the shape that broke
  // it: two directions unexecuted, one that returned no result object at all.
  const ev = EV([
    { d: { id: 'r2_d0' }, eng: { status: 'success' }, ver: { status: 'verified', verified_geomean: 1.0 }, inactive: 'no' },
    { d: { id: 'r2_d1' }, eng: { status: 'partial' }, ver: null },
  ]);
  ok(ev.measured === 0,
     'a round is exempt even when its directions failed in DIFFERENT ways — "every direction was inactive" was too narrow');
}
{
  const ev = EV([]);
  ok(ev.measured === 0 && ev.total === 0,
     'a round in which every engineer died measured nothing; the old rule required a non-empty list and charged it');
}

console.log('\n# a VOID reading is not evidence either');
{
  const unbacked = EV([Object.assign(MEASURED('r5_d0', 1.21), { unbacked: true })]);
  ok(unbacked.measured === 0 && /no patch behind it/.test(unbacked.gaps[0]),
     'a claim whose patch does not exist cannot be handed over, so it cannot close a round');
  const same = EV([Object.assign(MEASURED('r5_d1', 1.0), { same_artifact: true })]);
  ok(same.measured === 0 && /same compiled binary/.test(same.gaps[0]),
     'two arms that resolved to one binary measured the cache, not the idea');
}
{
  const ev = EV([{ d: { id: 'r6_d0' }, eng: { status: 'crashed' }, ver: null }]);
  ok(ev.measured === 0 && /faulted/.test(ev.gaps[0]),
     'a direction that faulted is named as a fault, which is the state that gets a retry rather than a redesign');
}

console.log('\n# an enabling step closes on function, here as everywhere else');
{
  const producer = {
    d: { id: 'r4_d0', step_role: 'enabling', enables: 'D2', cost_budget_pct: 3.0 },
    eng: { status: 'success' },
    ver: { status: 'verified', correctness: 'pass', activation_confirmed: 'yes', liveness: 'n/a', verified_geomean: 0.97 },
  };
  ok(EV([producer]).measured === 1,
     'a prerequisite that compiled, ran, was correct and did not deadlock IS evidence, even at 0.97x');
  const broken = JSON.parse(JSON.stringify(producer));
  broken.ver.correctness = 'fail';
  ok(EV([broken]).measured === 0,
     'and one that failed correctness is not — `enabling` is a different bar, not the absence of one');
}

console.log('\n# the loop still terminates when nothing ever works');
{
  const m = src.match(
    /while \(dispatched < BUDGET &&\s*\(MODE === 'mega' \|\| WORKING_KERNEL \|\|\s*\(([^)]*)\)\)\) \{/);
  ok(!!m, 'the round loop guards on more than the budget');
  ok(!!m && /noImprove < MAX_NO_IMPROVE/.test(m[1]) && /noEvidence < MAX_NO_IMPROVE/.test(m[1]),
     'both counters remain armed for ordinary optimize/author runs; Mega has lane-local WIP and its own budget stop');
  ok(/if \(ev\.measured\) noEvidence = 0;/.test(src),
     'and one real reading resets the no-evidence counter, so the cap only fires on CONSECUTIVE blind rounds');
}

console.log('\n# the run says why it stopped');
{
  ok(/stop_reason: stopReason/.test(src), 'the returned result carries the stop reason');
  ok(/STOP_REASON: stopReason/.test(src), 'and the final report is handed it, since that is what the next wave reads');
  ok(/budget unit\(s\) were left unspent/.test(src),
     'the reason states what was left on the table — a stop with budget remaining is a different finding from a stop that ran out');
  ok(/The budget counts directions, not GPU leases/.test(src),
     'and it says the budget is denominated in directions, which is why "budget left" and "hardware left" are not the same sentence');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
