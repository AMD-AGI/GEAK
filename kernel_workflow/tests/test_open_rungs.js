#!/usr/bin/env node
// A rung is closed by a verified number, not by having been planned.
//
// The wave-14/15 chain this exists to prevent: analyze produced a four-rung ladder ending in the
// fusion rung the program existed to reach. The rung below it was planned, its device arm hit an
// illegal memory access, and it was retired -- but `dispatchedRungs` is filled from what a round
// PLANNED, so it counted as taken and nothing was recorded as owed. The fusion rung's producer side
// was written in a later round and measured by nobody. The wave ended with the ladder nominally
// clear, the handover note inside one engineer's round directory, and no measurement. The next wave
// started from an empty ladder and never saw any of it.
//
// So: openRungs() must keep a rung that faulted, keep a rung that ran and produced nothing, keep a
// rung nobody ever took, and drop only a rung with a verified number -- including one that lost.
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
// open_rungs asks each rung HOW it closes, which lives in the enabling_step region: an enabling rung
// closes on functional acceptance, a terminal one on a measurement. Both are loaded together.
const { openRungs, rungOutcomeOf, rungIdOf } =
  // eslint-disable-next-line no-new-func
  new Function(`${region('enabling_step')}\n${region('open_rungs')}\nreturn { openRungs, rungOutcomeOf, rungIdOf };`)();

const LADDER = [
  { id: 'D0', title: 'D0 Instrumentation', gated_on: [], is_positive_control: true },
  { id: 'D1', title: 'D1 Per-m-tile readiness', gated_on: ['D0'] },
  { id: 'D2', title: 'D2 Fold combine — the two-launch shape', gated_on: ['D0', 'D1'] },
  { id: 'D3', title: 'D3 Static screens', gated_on: [] },
];

console.log('\n# what closes a rung');
{
  const owed = openRungs(LADDER, new Map([['D0', { attempts: 1, last_outcome: 'measured' }]]));
  ok(owed.map((c) => c.id).join(',') === 'D1,D2,D3', 'a measured rung is the only kind that drops off');
}
{
  const owed = openRungs(LADDER, new Map([
    ['D0', { attempts: 1, last_outcome: 'measured' }],
    ['D1', { attempts: 1, last_outcome: 'faulted' }],
  ]));
  const d1 = owed.find((c) => c.id === 'D1');
  ok(!!d1, 'a rung whose arm faulted is STILL OWED — this is the exact rung the chain turned on');
  ok(d1.attempts === 1 && d1.last_outcome === 'faulted',
     'and it carries how many times it was taken and how it ended, so the next wave can decide to retry or retire it');
}
{
  const owed = openRungs(LADDER, new Map([['D2', { attempts: 1, last_outcome: 'unmeasured' }]]));
  ok(owed.some((c) => c.id === 'D2'),
     'a rung that was authored and produced no number is owed — "we wrote the producer side" is not a measurement');
}
{
  const owed = openRungs(LADDER, new Map());
  ok(owed.length === 4 && owed.every((c) => c.last_outcome === 'never_planned' && c.attempts === 0),
     'a rung nobody took reports never_planned rather than being absent');
  ok(owed[0].is_positive_control === true && owed[2].gated_on.join(',') === 'D0,D1',
     'the control flag and the gates travel with the rung — a carried-forward rung with no gates is re-planned in the wrong order');
}

console.log('\n# grading a round result');
{
  ok(rungOutcomeOf({ eng: { status: 'ok' }, ver: {
    status: 'regression', correctness: 'pass', verified_geomean: 0.97,
  } }) === 'measured',
     'a rung that ran and LOST is measured and closed — otherwise the ladder never terminates');
  ok(rungOutcomeOf({ eng: { status: 'ok' }, ver: {
    status: 'verified', correctness: 'pass', verified_geomean: 1.04,
  } }) === 'measured',
     'and so is a rung that won');
  ok(rungOutcomeOf({ eng: { status: 'failed' }, ver: null }) === 'faulted',
     'an engineer failure is faulted, not measured');
  ok(rungOutcomeOf({ eng: { status: 'ok' }, ver: null }) === 'unmeasured',
     'reaching verify is not the same as returning a number');
  ok(rungOutcomeOf({ eng: null, ver: null }) === 'faulted' && rungOutcomeOf(null) === 'unmeasured',
     'the empty shapes never grade as measured');
}

console.log('\n# an enabling rung closes on function, not on a number');
{
  const passing = { eng: { status: 'ok' }, ver: { status: 'verified', correctness: 'pass',
    activation_confirmed: 'yes', liveness: 'n/a' } };
  ok(rungOutcomeOf(passing, 'enabling') === 'measured',
     'a prerequisite that builds, runs its path, is correct and does not hang is CLOSED even with no speedup number');
  ok(rungOutcomeOf(passing, 'terminal') === 'unmeasured',
     'the same result on a terminal rung is not closed — a terminal rung owes a number');
  const noPath = { eng: { status: 'ok' }, ver: { status: 'verified', correctness: 'pass',
    activation_confirmed: 'no', liveness: 'n/a' } };
  ok(rungOutcomeOf(noPath, 'enabling') === 'unmeasured',
     'a prerequisite whose path never ran is not closed — exempt from the speed bar, not from activation');
}

console.log('\n# ids');
{
  ok(rungIdOf({ title: 'D2 Per-token arrival counters' }) === 'D2',
     'a rung with no id is identified by the leading token of its title, which is how they are written');
  ok(openRungs([{ title: 'D9 x' }], new Map([['D9', { attempts: 2, last_outcome: 'measured' }]])).length === 0,
     'and that id is the one the tally is keyed by, or nothing ever matches and every rung stays owed forever');
}

console.log('\n# wired in');
{
  ok(/OPEN_RUNGS: openRungs\(LADDER, rungTally\)/.test(src),
     'the owed list is threaded to update_memory — the only phase whose output the next wave reads');
  ok(/let outcome = rungOutcomeOf\(r, role\);[\s\S]{0,700}recordRungOutcome\(r\.d\.roadmap_rung, outcome\);/.test(src),
     'and it is fed from post-gate round evidence, not from what was merely planned');
  ok(/if \(e\.last_outcome !== 'measured'\) e\.last_outcome = outcome;/.test(src),
     "`measured` is absorbing: a later round that retakes a closed rung and faults must not reopen it");
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
