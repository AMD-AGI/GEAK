#!/usr/bin/env node
// `objective: 'working_kernel'` is a budget change, and a budget change that only edits prose does
// nothing. Fourteen waves have been planned under text that already told the planner leases are
// scarce; one run still planned four directions into a pool of one lease, and two of its three round-1
// engineers never got a card. So every part of this mode is enforced in code, and this suite exists
// to check that the enforcement is real and — more importantly — that the mode did not become a way
// to smuggle an uncontrolled number into a report.
//
// The four things that must hold together (missing any one makes the others pointless):
//
//   1. one direction per round, clamped, not requested
//   2. the no-improve stop cannot apply — a debug round is non-improving BY CONSTRUCTION, so the
//      default stop-after-2 would end a 15-lease wave on round 3
//   3. the commit gate is "does it run", or nothing is ever committed and round N+1 restarts from the
//      unfused tree (this is exactly how an earlier run's fused kernel was lost)
//   4. skipping the positive control makes timings INADMISSIBLE, not cheap
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

const m = src.match(/\/\/ <<REPLAY:objective_gate>>([\s\S]*?)\/\/ <<\/REPLAY:objective_gate>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:objective_gate>> region — nothing to test'); process.exit(1); }
const e = src.match(/\/\/ <<REPLAY:enabling_step>>([\s\S]*?)\/\/ <<\/REPLAY:enabling_step>>/);
if (!e) { console.error('  FAIL: no <<REPLAY:enabling_step>> region'); process.exit(1); }
const { objectiveVerdict, runsCleanly } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\n${e[1]}\nreturn { objectiveVerdict, runsCleanly };`)();

console.log('\n# a speedup wave is untouched');
{
  ok(objectiveVerdict('speedup', false, 1.06).state === 'scored',
     'the default objective scores its numbers whatever this gate thinks — the mode must not change ' +
     'a run that did not ask for it');
  ok(objectiveVerdict('speedup', false, 1.0).state === 'scored' &&
     objectiveVerdict('speedup', true, 0.94).state === 'scored',
     'and that holds for a null and for a loss, so no speedup wave changes behaviour');
}

console.log('\n# an uncontrolled working_kernel wave may not ship ANY number');
{
  const v = objectiveVerdict('working_kernel', false, 1.09);
  ok(v.state === 'void_no_control', 'a claimed 1.09x from a wave that ran no control is VOID');
  ok(/withdrawn/.test(v.caveat) && /RUNS|runs/.test(v.caveat),
     'and the caveat says the reading is withdrawn while the artifact is still judged on running');
}
{
  ok(objectiveVerdict('working_kernel', false, 1.0).state === 'void_no_control',
     'a 1.000x is voided TOO — an uncontrolled 1.000x is exactly what a blind harness emits, and ' +
     'exempting it would re-open the hole the positive control was built to close');
  ok(objectiveVerdict('working_kernel', false, 0.91).state === 'void_no_control',
     'a loss is voided as well: an uncontrolled instrument is no more trustworthy pointing down');
  ok(objectiveVerdict('working_kernel', false, undefined).state === 'no_number' &&
     objectiveVerdict('working_kernel', false, NaN).state === 'no_number',
     'no reading at all is NOT a void reading — "nothing was measured" and "we refuse to read what ' +
     'was measured" are different report lines');
}
{
  ok(objectiveVerdict('working_kernel', true, 1.06).state === 'scored',
     'a working_kernel wave that DID pay for the control keeps its numbers — the mode buys leases ' +
     'by dropping the control, and a wave that keeps the control keeps the scoring');
}

console.log('\n# the commit gate is "does it run"');
{
  const good = { status: 'verified', correctness: 'pass', activation_confirmed: 'yes', liveness: 'pass' };
  ok(runsCleanly(good), 'correctness pass + activation confirmed + liveness pass runs');
  ok(runsCleanly({ ...good, liveness: 'n/a' }),
     "liveness n/a is accepted — it is the honest answer for a non-distributed candidate and forcing " +
     'a claim there would manufacture one');
  ok(!runsCleanly({ ...good, liveness: 'fail' }),
     'a liveness FAIL is a deadlock or a timeout, which this workflow has never counted as a skip');
  ok(!runsCleanly({ ...good, activation_confirmed: 'unknown' }) &&
     !runsCleanly({ ...good, activation_confirmed: 'no' }),
     'an unproven activation cannot pass: a patch that never executed passes every other field ' +
     'cleanly, which is the state that once produced a 1.000x on an unexercised patch');
  ok(!runsCleanly({ ...good, correctness: 'fail' }) && !runsCleanly(null) && !runsCleanly({}),
     'failed correctness, a missing verification, and an empty one all fail — absence is not a pass');
}

console.log('\n# the three enforcement points are in the script, not in prose');
{
  // WORKING_KERNEL short-circuits BOTH speedup-side stops. The second one (`noEvidence`) was added
  // later and is armed the same way; if it were left outside the short-circuit, a debug wave that
  // legitimately produces no speed reading for three rounds would stop for the same wrong reason.
  ok(/while \(dispatched < BUDGET &&\s*\(MODE === 'mega' \|\| WORKING_KERNEL \|\|/.test(src),
     'the no-improve stop is disabled in this mode — otherwise a 15-lease budget ends on round 3 ' +
     'and every other part of the mode is decoration');
  ok(/if \(WORKING_KERNEL && directions\.length > 1\)/.test(src),
     'the one-direction rule is a clamp on the planner output, not an instruction to the planner');
  ok(/let improved = WORKING_KERNEL\s*\n\s*\? winnerRuns/.test(src),
     'the commit gate switches to winnerRuns, so a still-crashing candidate is still committed once ' +
     'it runs, and the next round starts from it rather than from the unfused tree');
  ok(/\(WORKING_KERNEL \|\| metricScore\(r\.ver\) > 1\.0\) && !r\.inactive/.test(src),
     'the >1.0 bar is lifted from the `verified` filter in this mode: the terminal half of a staged ' +
     'fusion is slower than the tree the first time it lands, and gating verified on speed here ' +
     'deletes the fused megakernel before winnerRuns can ever commit it — the exact line the consumer ' +
     'half hit while the producer half (enabling) sailed past. In speedup mode the bar stays.');
  ok(/WORKING_KERNEL \|\| PROMOTION_METRIC === 'changed_kernel'/.test(src),
     'the earlier engineer-to-verify boundary is lifted for working and verifier-scored candidates — fixing only the later verified filter ' +
     'leaves ver=null and makes commit-on-running unreachable');
  ok(/const winnerVoid = WORKING_KERNEL && /.test(src) &&
     /if \(!winnerVoid && \(!WORKING_KERNEL \|\| winner\.geomean > cumulative\)\) \{/.test(src),
     'and a voided reading never becomes `cumulative`: the artifact advances, the number does not. ' +
     'Under working_kernel the commit gate is RUNNING, so a committed winner may be SLOWER than the ' +
     'tree (a terminal fusion is, the first time it lands); the guard makes cumulative ratchet UP ' +
     'only, so a committed regression cannot become the number the next round beats by reverting');
}
{
  ok(/args\.objective must be 'speedup' or 'working_kernel'/.test(src),
     'an unrecognised objective throws at startup rather than silently falling back to speedup — a ' +
     'typo that reads as the default is how a debug wave gets scored on speed');
  ok(/PC_RAN = ran;/.test(src) && /let PC_RAN = false;/.test(src),
     'PC_RAN is the gate\'s own verdict on whether a control RAN, not whether one was configured: a ' +
     'control whose switch was absent ran two identical arms and must not license a number');
  ok(/\.\.\.\(WORKING_KERNEL \? \{ OBJECTIVE, OBJECTIVE_NOTE:/.test(src),
     'the report says what the wave was for, and only on a non-default objective, so a speedup ' +
     "wave's report is byte-identical to before");
}

console.log(failures === 0
  ? '\nPASS: the debug objective changes the budget in code, and buys its leases by making numbers ' +
    'inadmissible rather than by making them cheap.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
