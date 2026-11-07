#!/usr/bin/env node
// Behavioral test for how the positive-control gate decides an effect is RESOLVED -- told apart from
// the interleave rather than merely large.
//
// WHY THIS EXISTS. The gate had one resolution test: the effect must be RESOLVE_K times the worst
// null pair. That is correct when the null scatters around a centre and wrong when it is a mixture.
// Once the 512_rank-mixed-skew guard was measured 10 times against itself: nine pairs inside
// 3pp and one at 9.02pp, because that guard drops into a discrete slow state on roughly one run in
// five. A candidate measured over the same interleave read +17.34% median, 10/10 pairs positive,
// range +14.30..+18.65, with the two arms' raw readings not overlapping at all -- and scored 1.9x,
// a FAIL. One draw from a tail set the whole criterion.
//
// So a second route was added: the effect pairs and the null pairs, by magnitude, do not overlap.
// The risk in adding ANY second route is that it is a way to pass, and the thing it must not do is
// rescue a control that genuinely sits inside its own null. The load-bearing assertions here are the
// negative ones -- most of all that earlier run's real control, the one that motivated looking at this
// code at all, STILL FAILS.
//
// Like the other gate tests, this LIFTS the shipped block from between the <<REPLAY:pc_gate>>
// markers and runs it, so it cannot drift away from what ships.
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

const m = src.match(/\/\/ <<REPLAY:pc_gate>>[\s\S]*?\n([\s\S]*?)\n\s*\/\/ <<\/REPLAY:pc_gate>>/);
if (!m) {
  console.error('FAILED: the <<REPLAY:pc_gate>> markers are missing from kernel_mega.js.');
  console.error('Without them this test cannot reach the shipped gate and would silently test nothing.');
  process.exit(1);
}
// eslint-disable-next-line no-new-func
const gate = new Function('pc', 'POSITIVE_CONTROL', `
  const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
  const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
  const got = Number(pc.measured_pct);
${m[1]}
  return { ok, ran, tooSmall, undershoot, resolved, resolvedByScale, resolvedBySeparation, signUnanimous };
`);

// A constructed slowdown aimed at -9..-4, which is the shipped band for this operator.
const PC = { expected_pct_lo: -9.0, expected_pct_hi: -4.0, magnitude: 'constructed', implausible_pct: 25.0 };
const run = (ctrl, nul, measured) => gate(
  { ran: true, measured_pct: measured, control_pairs_pct: ctrl, null_pairs_pct: nul,
    null_arm_pct: nul.reduce((a, b) => a + b, 0) / nul.length },
  PC);

// --- 1. the scale route still works, and is still the primary --------------------------------
console.log('\n# the original route: many times the worst null pair');
let r = run([-5.1, -5.4, -4.9, -5.6, -5.2], [0.3, -0.4, 0.2, 0.5, -0.3], -5.2);
ok(r.resolvedByScale === true, 'a quiet unimodal null resolves on scale alone (5.2 vs 3x0.5)');
ok(r.ok === true, 'and the control passes');

// --- 2. the new route: a fat-tailed null the scale test cannot survive -------------------------
// The wave-9 candidate's own numbers, as pair percentages. Nine tight null pairs and one excursion.
console.log('\n# a fat-tailed null: separation resolves what scale cannot');
const NULL_BIMODAL = [-3.20, -2.78, 2.45, -9.02, -1.51, 1.35, 2.72, -2.06, -5.00, 1.92];
const EFFECT_10 = [16.41, 16.66, 18.40, 18.57, 17.72, 14.44, 18.56, 16.96, 14.30, 18.65];
r = gate({ ran: true, measured_pct: -17.34, control_pairs_pct: EFFECT_10.map((x) => -x),
           null_pairs_pct: NULL_BIMODAL, null_arm_pct: -1.78 }, PC);
ok(r.resolvedByScale === false, '17.34 is only 1.9x the 9.02pp tail draw, so the scale route fails');
ok(r.resolvedBySeparation === true,
   'but the smallest effect pair (14.30) is above the largest null pair (9.02) -- no overlap');
ok(r.resolved === true, 'so the effect is resolved, by the route that a single tail draw cannot break');

// --- 3. THE NEGATIVE THAT MATTERS -------------------------------------------------------------
// An earlier run's actual control. It is sign-unanimous, monotone on a dose ladder, and its smallest pair
// is well inside the null. Adding the separation route must NOT have rescued it.
console.log('\n# the control that started all this must still fail');
const W10_CTRL = [-0.344, -1.567, -3.734, -1.518, -3.319, -3.254];
const W10_NULL = [0.126, -0.9, 1.881, -1.2, 0.4, -0.7, 1.1, -0.3];
r = gate({ ran: true, measured_pct: -2.41, control_pairs_pct: W10_CTRL,
           null_pairs_pct: W10_NULL, null_arm_pct: 0.126 },
         { expected_pct_lo: -5.0, expected_pct_hi: -2.5, magnitude: 'constructed', implausible_pct: 15.0 });
ok(r.signUnanimous === true, 'one run\'s control was 6/6 the same sign -- unanimity alone is not enough');
ok(r.resolvedByScale === false, 'it is 1.3x the worst null pair, failing the scale route');
ok(r.resolvedBySeparation === false,
   'and its smallest pair (0.344pp) is INSIDE the null (worst 1.881pp), so separation fails too');
ok(r.ok === false, 'one run still aborts -- the new route did not launder the case it came from');

// --- 4. separation must not be reachable on thin evidence --------------------------------------
console.log('\n# separation needs enough pairs on both arms to mean anything');
// 3 vs 3 separates trivially by chance: 2/C(6,3) = 0.1. The floor is 5 and 5 (2/C(10,5) = 7.9e-3).
r = run([-6, -7, -8], [0.5, -0.4, 0.3], -7);
ok(r.resolvedBySeparation === false, '3 control pairs vs 3 null pairs is not separation evidence');
r = run([-6, -7, -8, -6.5, -7.5], [0.5, -0.4, 0.3], -7);
ok(r.resolvedBySeparation === false, '5 control pairs against only 3 null pairs is still not enough');
r = run([-6, -7, -8, -6.5, -7.5], [0.5, -0.4, 0.3, 0.6, -0.5], -7);
ok(r.resolvedBySeparation === true, '5 and 5 is the floor, and it is met here');

// --- 5. one overlapping pair kills separation ---------------------------------------------------
// The test is complete separation, not "mostly separated". A single effect pair that falls inside
// the null range means the two could have come from one distribution.
console.log('\n# separation is complete or it is nothing');
r = run([-6, -7, -8, -6.5, -0.4], [0.5, -0.4, 0.3, 0.6, -0.5], -6.5);
ok(r.resolvedBySeparation === false,
   'one effect pair at 0.4pp, inside a null spanning 0.6pp, breaks separation');

// --- 6. resolution is necessary but never sufficient --------------------------------------------
// Both routes feed the same `resolved`, and `resolved` only ever unlocks the constructed-undershoot
// allowance. It must not let through an injection that never took effect.
console.log('\n# resolution does not override the half-target floor');
r = run([-1.9, -2.0, -1.95, -1.85, -1.92], [0.05, -0.04, 0.03, 0.06, -0.05], -1.93);
ok(r.resolvedBySeparation === true, 'a tiny effect against a tinier null does separate');
ok(r.undershoot === false && r.ok === false,
   'but 1.93% is under half the 4.0% target, the signature of an injection that never took ' +
   'effect, so it still aborts -- separation is not a way past that floor');

// --- 7. the absent switch: a well-formed reading with nothing behind it -------------------------
// The pre-flight, and it outranks everything above. If the knob `how` names is not in the tree, the
// two arms ran identical code and `measured_pct` is the session's drift. A wave shipped a prescribed
// control whose env var existed in no tree in the run; it read -1.00% and nothing in the numbers
// could have told anyone. So the gate is given a fact the numbers cannot supply, and honours it.
console.log('\n# an absent switch is not a reading');
{
  const textbook = { ran: true, measured_pct: -5.2, control_pairs_pct: [-5.1, -5.4, -4.9, -5.6, -5.2],
                     null_pairs_pct: [0.3, -0.4, 0.2, 0.5, -0.3], null_arm_pct: 0.06 };
  ok(gate(textbook, PC).ok === true, 'a textbook control passes when the switch question is not raised');
  ok(gate({ ...textbook, switch_present: true }, PC).ok === true,
     'and still passes when the switch is confirmed present');
  const absent = gate({ ...textbook, switch_present: false }, PC);
  ok(absent.ran === false,
     'switch_present:false means the control DID NOT RUN, however good the numbers look');
  ok(absent.ok === false,
     'so it fails the gate — an unproven instrument must not be certified by its own drift');
  // Historical controls never reported the field. If undefined changed the verdict, every recorded
  // control in every prior run would re-decide on replay, which is exactly what replay is for
  // catching. Keep the default inert.
  ok(gate({ ...textbook, switch_present: undefined }, PC).ok === true,
     'an unreported switch_present is inert, so historical controls replay unchanged');
}

console.log(failures
  ? `\nFAILED: ${failures} assertion(s).`
  : '\nPASS: resolution has two routes, and the second one rejects the case it was written for.');
process.exit(failures ? 1 : 0);
