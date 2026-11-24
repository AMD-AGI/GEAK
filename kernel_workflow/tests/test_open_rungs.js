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
const src = require('./mega_source')();

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
// open_rungs uses the step-role helpers of the enabling_step region; both are loaded together.
const { openRungs, rungIdOf } =
  // eslint-disable-next-line no-new-func
  new Function(`${region('enabling_step')}\n${region('open_rungs')}\nreturn { openRungs, rungIdOf };`)();

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
  ok(/if \(e\.last_outcome !== 'measured'\) e\.last_outcome = outcome;/.test(src),
     "`measured` is absorbing: a later round that retakes a closed rung and faults must not reopen it");
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
