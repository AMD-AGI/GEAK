#!/usr/bin/env node
// Behavioral test for the roadmap-ladder gate, plus the plumbing that makes it reachable.
//
// Same house rule as test_pipe_occupancy_gate.js: LIFT the shipped gate verbatim out of
// kernel_mega.js between its `<<REPLAY:roadmap_ladder_gate>>` markers and run it. A grep test
// would go green the moment somebody writes the words.
//
// WHAT THE GATE IS FOR — and why the other two gates cannot cover it. taskGraphGate asks whether
// the dependency graph exists. pipeOccupancyGate asks whether each direction was priced against a
// busy or an idle pipe. Both examine the directions that WERE issued, so neither can see the
// direction that was never issued.
//
// A wave's Analyze produced a correct four-rung ladder ending in exactly the fused two-launch shape
// the program existed to reach, wrote it to roadmap.md and to analysis.json:candidate_directions,
// and then plan_round was handed neither artifact. Six directions were dispatched over three
// rounds. The bounding readout never ran; the rung the ladder itself designated as the positive
// control never ran, so the wave had no control and improvised a substitute mid-flight; the
// readiness rung ran first, without its own mandatory publish-only arm, and read negative; and the
// fusion rung, gated on it, was never proposed. The ladder was right and structurally an orphan.
// Every one of those losses is invisible per-direction and obvious the moment dispatched rungs are
// diffed against the ladder. Hence: one section per loss.
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

// --- lift ------------------------------------------------------------------------------------
const m = src.match(
  /\/\/ <<REPLAY:roadmap_ladder_gate>>[\s\S]*?\n([\s\S]*?)\n\s*\/\/ <<\/REPLAY:roadmap_ladder_gate>>/);
if (!m) {
  console.error('FAILED: the <<REPLAY:roadmap_ladder_gate>> markers are missing from kernel_mega.js.');
  console.error('Without them this test cannot reach the shipped gate and would silently test nothing.');
  process.exit(1);
}
// eslint-disable-next-line no-new-func
const roadmapLadderGate = new Function(`${m[1]}\nreturn roadmapLadderGate;`)();
ok(typeof roadmapLadderGate === 'function', 'gate lifted from kernel_mega.js and is callable');
ok((() => { try { roadmapLadderGate(null, null, null); return true; } catch { return false; } })(),
   'the lifted region is pure (no closure over run state)');

// The wave-13 ladder, in the shape Analyze is now asked to return it.
const LADDER = [
  { id: 'D0', title: 'D0 Readout: no-payload control + instrumented wait', gated_on: [] },
  { id: 'D1', title: 'D1 Positive control: delete the vacuous join', gated_on: [], is_positive_control: true },
  { id: 'D2', title: 'D2 Per-token cross-rank readiness', gated_on: ['D0'],
    mandatory_arms: ['publish-only (signals without waits)'] },
  { id: 'D3', title: 'D3 Persistent fusion — the two-launch acceptance shape', gated_on: ['D2'] },
];

console.log('\n# 1. a run with no ladder at all is reported, not silently accepted');
{
  const r = roadmapLadderGate([], [{ id: 'r1_d0' }], []);
  ok(r.verdict === 'MISSING', 'an absent ladder is MISSING, distinct from OK');
  ok(/no recorded ordering/i.test(r.caveat), 'and the caveat says what is lost: the ordering');
}

console.log('\n# 2. the direction that never names a rung');
{
  const r = roadmapLadderGate(LADDER, [{ id: 'r1_d0', title: 'LDS bank conflicts' }], []);
  ok(/NO RUNG NAMED/.test(r.caveat), 'a direction with no roadmap_rung is named in the caveat');
  ok(/r1_d0/.test(r.caveat), 'and it is named by id, so it is actionable');
  // Declaring off_ladder WITH a reason is the supported way to leave the ladder, and must be quiet
  // about the leaving itself — otherwise the gate trains planners to stop declaring.
  const r2 = roadmapLadderGate(LADDER,
    [{ id: 'r1_d0', roadmap_rung: 'off_ladder', rung_deviation: 'profile shows a bigger lever; D0 still owed' }],
    []);
  ok(!/OFF-LADDER WITHOUT/.test(r2.caveat), 'a declared deviation does not itself raise a flag');
  const r3 = roadmapLadderGate(LADDER, [{ id: 'r1_d0', roadmap_rung: 'off_ladder' }], []);
  ok(/OFF-LADDER WITHOUT A STATED DEVIATION/.test(r3.caveat),
     'but off_ladder with no reason does — the displaced rung would otherwise vanish');
}

console.log('\n# 3. the rung taken before its prerequisite (the wave-13 D2-before-D0)');
{
  const r = roadmapLadderGate(LADDER, [{ id: 'r1_d1', roadmap_rung: 'D2' }], []);
  ok(/RUNG TAKEN OUT OF ORDER/.test(r.caveat), 'D2 dispatched with D0 undone is flagged');
  ok(/D0/.test(r.caveat), 'the unmet prerequisite is named');
  ok(r.verdict === 'INCONSISTENT', 'and it is INCONSISTENT, not merely advisory');
  ok(/interpretable/i.test(r.caveat),
     'the reason given is interpretability, not lower odds — that is the whole point of a gate');
  const okOrder = roadmapLadderGate(LADDER, [{ id: 'r2_d0', roadmap_rung: 'D2' }], ['D0', 'D1']);
  ok(!/OUT OF ORDER/.test(okOrder.caveat), 'with D0 dispatched, the same rung passes');
}

console.log('\n# 4. the designated positive control that never ran');
{
  const r = roadmapLadderGate(LADDER, [{ id: 'r1_d0', roadmap_rung: 'D2' }], ['D0']);
  ok(/POSITIVE CONTROL HAS NOT BEEN DISPATCHED/.test(r.caveat), 'D1 being unspent is called out');
  ok(/no effect.*no instrument/i.test(r.caveat),
     'with the consequence stated: every null is ambiguous until it runs');
  const r2 = roadmapLadderGate(LADDER, [{ id: 'r2_d0', roadmap_rung: 'D2' }], ['D0', 'D1']);
  ok(!/POSITIVE CONTROL HAS NOT/.test(r2.caveat), 'and it goes quiet once the control is dispatched');
}

console.log('\n# 5. the rung at the top that nothing will ever reach');
{
  // The failure that cost the wave: D3 is gated on D2, D2 is never taken, so D3 is unreachable —
  // and nothing in a per-direction check can see it, because D3 is not among the directions.
  const r = roadmapLadderGate(LADDER, [{ id: 'r1_d0', roadmap_rung: 'off_ladder', rung_deviation: 'x' }], []);
  ok(/RUNGS STILL BLOCKED BY UNDISPATCHED PREREQUISITES/.test(r.caveat),
     'a rung whose gate is unsatisfied is reported even though no direction mentions it');
  ok(/D3/.test(r.caveat) && /D2/.test(r.caveat), 'both the blocked rung and its gate are named');
  ok(/never reached/.test(r.summary), 'the summary counts what the wave has not reached');
  const done = roadmapLadderGate(LADDER, [{ id: 'r3_d0', roadmap_rung: 'D3' }], ['D0', 'D1', 'D2']);
  ok(done.verdict === 'OK' && !done.caveat, 'a wave that climbs the whole ladder is clean');
}

console.log('\n# 6. rung ids survive when Analyze only wrote them into the title');
{
  // These are written "D2 Per-token ..." in practice. Falling back to the leading token means an
  // older-format ladder still gets checked instead of silently degrading to MISSING.
  const titleOnly = LADDER.map(({ id, ...rest }) => rest);
  const r = roadmapLadderGate(titleOnly, [{ id: 'r1_d0', roadmap_rung: 'D2' }], []);
  ok(r.verdict !== 'MISSING', 'a ladder with titles but no id fields is still checkable');
  ok(/D2/.test(r.summary), 'and the derived ids appear in the summary');
}

console.log('\n# 7. the gate is wired in, and the ladder actually reaches plan_round');
{
  ok(/roleAgent\('tech_lead', 'plan_round'[\s\S]{0,900}?ROADMAP_LADDER: LADDER/.test(src),
     'plan_round is handed the ladder as DATA, not just a path it might read');
  ok(/roleAgent\('tech_lead', 'plan_round'[\s\S]{0,900}?LADDER_DISPATCHED/.test(src),
     'and the cross-round record of what has already been taken off it');
  ok(/roadmapLadderGate\(LADDER, directions, LADDER_MEASURED\)/.test(src),
     'the gate runs against each round\'s directions and completed-to-spec prerequisites');
  // Before the budget is charged: a skip flagged after the spend is a post-mortem, not a gate.
  const gateAt = src.indexOf('roadmapLadderGate(LADDER, directions, LADDER_MEASURED)');
  const chargeAt = src.indexOf('dispatched += roundCost');
  ok(gateAt > 0 && chargeAt > gateAt, 'and it runs BEFORE the round is charged to the budget');
  ok(/ROADMAP_LADDER_FINAL/.test(src),
     'the end-of-wave ladder state travels to the report, where the unreached rung belongs');

  const plan = src.slice(src.indexOf('const PLAN_SCHEMA'), src.indexOf('const PLAN_SCHEMA') + 3000);
  ok(/roadmap_rung:/.test(plan) && /rung_deviation:/.test(plan),
     'PLAN_SCHEMA carries both fields, so a planner is asked rather than trusted');
}

console.log('\n# 8. the role contract says the three things the gate cannot enforce');
{
  const tl = fs.readFileSync(path.join(WF, 'roles', 'mega', 'tech_lead.md'), 'utf8').replace(/\s+/g, ' ');
  ok(/is_positive_control/.test(tl) && /gated_on/.test(tl) && /mandatory_arms/.test(tl),
     'analyze is asked for all three structured ladder fields');
  ok(/mandatory_arms` are part of the direction's `prompt`, verbatim/i.test(tl),
     'a rung\'s mandatory arms must reach the engineer who runs that rung');
  ok(/interpretability, not about odds|interpretability, not odds/i.test(tl),
     'and the reason a prerequisite matters is stated as interpretability');
  // The verdict rule is the half the gate structurally cannot check: it fires after the numbers
  // are in, and the numbers can be impeccable.
  ok(/Verdict by SPEC, not by result/.test(tl),
     'the ledger grades by spec — a rung run without a mandatory arm is unresolved, not dead_end');
  ok(/confirmed\|partial\|unresolved\|dead_end/.test(tl),
     'and `unresolved` is actually in the verdict enum, not only in the prose');
}

console.log(
  failures === 0
    ? '\nPASS: the ladder is threaded, checked each round, and graded by spec.'
    : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
