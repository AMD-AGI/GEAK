#!/usr/bin/env node
// roadmapLadderGate only ADVISES that the terminal fusion rung was left unplanned. That is safe while
// there is budget to explore and still come back to close the chain, and fatal once there is not: a
// wave commits enabling steps, defers their terminal rung "one more round" every round, runs out of
// budget, and ships a tree slower than it started with the fusion never closed. terminalRungToForce
// converts the advisory into an action at the point deferral stops being safe. Each assertion is a
// way that conversion could go wrong.
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

const m = src.match(/\/\/ <<REPLAY:terminal_forcing>>([\s\S]*?)\/\/ <<\/REPLAY:terminal_forcing>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:terminal_forcing>> region'); process.exit(1); }
const { terminalRungToForce } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { terminalRungToForce };`)();

const LADDER = [
  { id: 'D2', title: 'D2 combine fold', rationale: 'fold combine into gemm2' },
  { id: 'D3', title: 'D3 two-launch megakernel' },
];
const debt = (rungs) => ({
  open: rungs.map((r) => ({ enables: r.rung, steps: r.steps || ['e1'], cost_pct: 2, oldest_round: r.since })),
  overdue: rungs.filter((r) => r.overdue).map((r) => ({ enables: r.rung, oldest_round: r.since })),
});

console.log('\n# nothing owed, nothing forced');
{
  ok(terminalRungToForce({ open: [], overdue: [] }, LADDER, [], 1) === null,
     'no open chain debt -> no forcing, whatever the budget');
}

console.log('\n# with slack, the planner is left alone');
{
  const f = terminalRungToForce(debt([{ rung: 'D3', since: 2 }]), LADDER, [], 5);
  ok(f === null, 'one owed rung with remaining=5 is not forced — there is room to explore and still close it');
}

console.log('\n# when the budget can no longer both explore and pay every chain, force');
{
  const f = terminalRungToForce(debt([{ rung: 'D3', since: 2 }]), LADDER, [], 2);
  ok(f && f.rungId === 'D3' && f.reason === 'budget_tight',
     'remaining=2 with one owed rung (<= owed+1) forces D3 as budget_tight');
  ok(f.rung && f.rung.title === 'D3 two-launch megakernel',
     'and the ladder entry is resolved so the forced direction can be built from its title/rationale');
}

console.log('\n# an overdue chain forces regardless of remaining budget');
{
  const f = terminalRungToForce(debt([{ rung: 'D3', since: 1, overdue: true }]), LADDER, [], 9);
  ok(f && f.rungId === 'D3' && f.reason === 'overdue',
     'a chain unpaid for CHAIN_DEBT_MAX_ROUNDS forces even with remaining=9 — deferring it further abandons it');
}

console.log('\n# the planner already closing the chain is not overridden');
{
  const f = terminalRungToForce(debt([{ rung: 'D3', since: 2 }]), LADDER, ['D3'], 1);
  ok(f === null, 'D3 already in this round\'s planned rungs -> nothing to force, the planner chose it');
}

console.log('\n# overdue-first, then oldest-first, so the chain closest to abandonment closes first');
{
  const f = terminalRungToForce(
    debt([{ rung: 'D2', since: 3 }, { rung: 'D3', since: 1, overdue: true }]), LADDER, [], 2);
  ok(f && f.rungId === 'D3', 'the overdue rung is picked ahead of a merely-owed newer one');
  const f2 = terminalRungToForce(
    debt([{ rung: 'D2', since: 1 }, { rung: 'D3', since: 3 }]), LADDER, [], 3);
  ok(f2 && f2.rungId === 'D2',
     'with neither overdue, the chain open longest (smallest oldest_round) closes first');
}

if (failures) { console.error(`\nFAIL: ${failures} assertion(s) failed.`); process.exit(1); }
console.log('\nall passed');
