#!/usr/bin/env node
// A step that ENABLES a fusion is not a step that speeds one up.
//
// The round filter is `primSpeedup(ver) > 1.0`. For a fusion built in stages that is not a quality
// bar, it is a structural block: the producer half adds completion signalling and a second buffer,
// has no consumer yet, and can only measure its own overhead. It is supposed to be slower. Under a
// speed-only filter it is discarded, never enters the canonical tree, and the consumer half is then
// written against a tree where the producer half does not exist -- so it is never written at all.
// That is how this project reached a half-fused kernel and stopped there.
//
// Each assertion below is a way the fix could be wrong in a way nobody would notice:
//   - `enabling` becomes a way to commit anything, including a step that does not work
//   - a prerequisite is committed but its cost vanishes, so the tree silently gets slower
//   - the debt is tracked but never comes due, so a chain can be abandoned half-built in silence
//   - the terminal step is measured against the degraded tree and credited with removing the
//     overhead its own prerequisites installed
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

const m = src.match(/\/\/ <<REPLAY:enabling_step>>([\s\S]*?)\/\/ <<\/REPLAY:enabling_step>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:enabling_step>> region — nothing to test'); process.exit(1); }
const { stepRoleOf, functionalAcceptance, enablingVerdict, chainDebtReport, CHAIN_DEBT_MAX_ROUNDS, verifyCompleted } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { stepRoleOf, functionalAcceptance, enablingVerdict, chainDebtReport, CHAIN_DEBT_MAX_ROUNDS, verifyCompleted };`)();

console.log('\n# verifyCompleted is shared by the enabling and terminal paths so they cannot disagree');
{
  ok(verifyCompleted({ status: 'verified' }) && verifyCompleted({ status: 'regression' }) &&
     verifyCompleted({ status: 'slower' }),
     'a completed verification includes regression/slower — a terminal fusion is CORRECT but slower ' +
     'the first time it lands, and the candidate filter must not drop it for reporting exactly that');
  ok(!verifyCompleted({ status: 'apply_failed' }) && !verifyCompleted({ status: 'plagiarized' }) &&
     !verifyCompleted({ status: 'harness_modified' }) && !verifyCompleted(null),
     'the states where the comparison never happened or was tainted are not completions');
}

// A prerequisite that passed everything functional and, as expected, made the operator slower.
const GOOD_VER = { status: 'verified', correctness: 'pass', activation_confirmed: 'yes', liveness: 'n/a' };
const PRODUCER = { id: 'r3_d0', step_role: 'enabling', enables: 'D2', cost_budget_pct: 3.0 };

console.log('\n# the default does not change');
{
  ok(stepRoleOf({}) === 'terminal' && stepRoleOf(null) === 'terminal',
     'a direction that says nothing is terminal — the speed bar stays the default for everything else');
  ok(stepRoleOf({ step_role: 'ENABLING' }) === 'enabling',
     'the role is read case-insensitively; a direction is not silently demoted by capitalisation');
  ok(stepRoleOf({ step_role: 'prerequisite' }) === 'terminal',
     'an unrecognised role falls back to the STRICTER bar, not the looser one');
}

console.log('\n# the case the whole change exists for');
{
  const v = enablingVerdict(PRODUCER, GOOD_VER, 0.98);
  ok(v.commit === true,
     'a prerequisite that works and is 2% SLOWER is kept — under the old filter this is exactly what was thrown away');
  ok(v.cost_pct === 2.04, `and its cost is recorded rather than discarded (got ${v.cost_pct}%)`);
  ok(v.enables === 'D2', 'and it is filed against the rung it is a prerequisite for');
}

console.log('\n# exempt from the speed bar, and from nothing else');
{
  const cases = [
    ['correctness did not pass', { ...GOOD_VER, correctness: 'fail' }],
    ['the new path was not confirmed to run', { ...GOOD_VER, activation_confirmed: 'no' }],
    ['liveness failed (hang or timeout)', { ...GOOD_VER, liveness: 'fail' }],
    // Acceptance criterion 4 carries a NUMBER; a "pass" string over a relL2 above threshold is not a pass.
    ['is not < threshold', { ...GOOD_VER, accuracy: { metric: 'relL2', value: 0.4, threshold: 0.10 } }],
    // Criterion 4 also wants 1000 CUDA-Graph replays with no deadlock; a bad graph replay is a
    // function failure, not a speed one.
    ['graph capture/replay is not safe', { ...GOOD_VER, graph_safe: 'fail' }],
  ];
  for (const [needle, ver] of cases) {
    const v = enablingVerdict(PRODUCER, ver, 0.98);
    ok(v.commit === false && v.reason.includes(needle),
       `refused when ${needle} — and the reason names which condition failed`);
  }
  ok(enablingVerdict(PRODUCER, null, 0.98).commit === false,
     'and a step with no verify result at all is refused, not waved through for lack of evidence');
  // The new fields DEGRADE rather than crash: a run that never wired accuracy/graph_safe is unchanged.
  ok(enablingVerdict(PRODUCER, { ...GOOD_VER }, 0.98).commit === true,
     'a producer with no accuracy and no graph_safe reported still commits on function — absence is not a failure');
  ok(enablingVerdict(PRODUCER, { ...GOOD_VER, accuracy: { metric: 'relL2', value: 0.05, threshold: 0.10 } }, 0.98).commit === true,
     'and a reported relL2 comfortably under threshold passes the accuracy check');
}

console.log('\n# the two ways `enabling` could become a blank cheque');
{
  const v = enablingVerdict({ id: 'x', step_role: 'enabling' }, GOOD_VER, 0.90);
  ok(v.commit === false && /prerequisite to nothing/.test(v.reason),
     'a prerequisite that names no terminal rung is refused — otherwise the label commits any regression');
}
{
  const v = enablingVerdict(PRODUCER, GOOD_VER, 0.80);
  ok(v.commit === false && /over its own declared budget/.test(v.reason),
     'a 25% cost against a 3% budget is refused: slower is expected, this much slower is a design error');
  const w = enablingVerdict({ ...PRODUCER, cost_budget_pct: 30 }, GOOD_VER, 0.80);
  ok(w.commit === true, 'and the budget is the direction\'s own declared number, not a fixed constant');
}
{
  ok(enablingVerdict({ ...PRODUCER, cost_budget_pct: undefined }, GOOD_VER, 0.98).commit === true,
     'a missing budget falls back to a default rather than refusing everything');
  const v = enablingVerdict({ ...PRODUCER, cost_budget_pct: undefined }, GOOD_VER, 0.80);
  ok(v.commit === false,
     'and that default is a real bound — 25% still does not land when no budget was declared');
}
{
  const v = enablingVerdict(PRODUCER, GOOD_VER, 1.03);
  ok(v.commit === true && v.cost_pct < 0,
     'a prerequisite that happens to be FASTER is kept too, with a negative cost — the role is about how it is judged, not about which way the number went');
}

console.log('\n# the debt has to come due');
{
  const debt = [{ round: 1, id: 'r1_d0', enables: 'D2', cost_pct: 2.0 },
                { round: 2, id: 'r2_d1', enables: 'D2', cost_pct: 1.5 }];
  const open = chainDebtReport(debt, 2, new Set());
  ok(open.open.length === 1 && open.open[0].cost_pct === 3.5,
     'two prerequisites for the same rung accumulate into one balance of 3.5%');
  ok(/D2 owes 3.5%/.test(open.caveat) && /round 1/.test(open.caveat),
     'and the caveat names the rung, the amount, and how long it has been owed');
  ok(chainDebtReport(debt, 2, new Set(['D2'])).caveat === '',
     'once the terminal rung is MEASURED the balance clears — measured, not won: a chain that closed and lost still closed');
}
{
  const debt = [{ round: 1, id: 'r1_d0', enables: 'D2', cost_pct: 2.0 }];
  ok(chainDebtReport(debt, 1, new Set()).overdue.length === 0,
     'a chain opened this round is not yet overdue');
  ok(chainDebtReport(debt, 1 + CHAIN_DEBT_MAX_ROUNDS, new Set()).overdue.length === 1,
     `and it is overdue after ${CHAIN_DEBT_MAX_ROUNDS} rounds — a chain abandoned half-built leaves the tree slower with nothing to show`);
  ok(/OVERDUE/.test(chainDebtReport(debt, 9, new Set()).caveat),
     'which is said in words, not only in a field nobody reads');
}
{
  ok(chainDebtReport([], 3, new Set()).caveat === '' && chainDebtReport(null, 3, new Set()).open.length === 0,
     'no debt produces no noise, so the caveat means something when it appears');
}

console.log('\n# wired into the round');
{
  ok(/stepRoleOf\(r\.d\) !== 'enabling'\)/.test(src),
     'the speed filter explicitly excludes enabling steps rather than relying on them scoring above 1.0');
  ok(/if \(stepRoleOf\(r\.d\) === 'enabling'\) continue;/.test(src),
     'the attribution gate skips them: a step making no win claim has no win to attribute to the wrong kernel');
  ok(/commit enabling r\$\{round\}/.test(src),
     'they are committed to the canonical tree — being kept but not committed is the same as being discarded');
  const winnerCommit = src.indexOf('You are the TechLead committing round ${round}\'s winning patch');
  const enablingCommit = src.indexOf("You are the TechLead committing round ${round}'s ENABLING steps");
  ok(winnerCommit > 0 && enablingCommit > winnerCommit,
     'and AFTER the winner, so the winner\'s patch still applies against the tree it was cut from');
  ok(/cumulative unchanged/.test(src),
     'committing one does not move cumulative — a headline that counts installed overhead as progress is worse than no headline');
  ok(/NOT counted toward noImprove — \$\{enablingLanded\} enabling step\(s\) landed/.test(src),
     'and a round that landed one does not count toward the no-improve stop, or the run gives up before the step that pays for it');
  ok(/CHAIN_BASELINE = \{ round, cumulative/.test(src),
     'the baseline is pinned at the first enabling commit, so the terminal step is not credited with removing its own chain\'s overhead');
  ok(/CHAIN_DEBT: chainDebtReport\(CHAIN_DEBT, round, LADDER_MEASURED\)\.open/.test(src),
     'the balance reaches the planner, which is the only phase that can decide to close the chain');
  ok(/CHAIN_CAVEATS\.length \? \{ CHAIN_CAVEATS \}/.test(src),
     'and the report, so a wave that stopped half-fused does not read as a failed search');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
