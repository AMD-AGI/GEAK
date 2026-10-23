#!/usr/bin/env node
// Acceptance criterion 1 is "fully fused, TWO launches" — one megakernel per EP rank plus the one
// separate pre-dispatch quant launch. Until this gate existed nothing in the workflow counted
// launches, so a candidate that fused dispatch+gemm1 but still launched combine on its own ran three
// launches and read exactly like a real two-launch fusion. This gate makes the count a stated number
// and reports whether the two-launch shape was actually reached. It never rejects; each assertion
// below is a way that reporting could silently go wrong.
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

const m = src.match(/\/\/ <<REPLAY:launch_gate>>([\s\S]*?)\/\/ <<\/REPLAY:launch_gate>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:launch_gate>> region — nothing to test'); process.exit(1); }
const { launchVerdict } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { launchVerdict };`)();

console.log('\n# an unreported launch count is a HOLE, not a pass');
{
  const v = launchVerdict({}, 2);
  ok(v.state === 'unjudged' && v.shape_met === false,
     'no launch_shape reported leaves criterion 1 UNJUDGED and shape_met false — a three-launch ' +
     'candidate must not read as accepted just because nobody counted');
  const v2 = launchVerdict({ launch_shape: { launches_base: 4 } }, 2);
  ok(v2.state === 'unjudged',
     'launches_base without launches_cand is still unjudged: the candidate side is the one criterion 1 asks about');
}

console.log('\n# a three-launch fusion is a partial rung, reported, never rejected');
{
  const v = launchVerdict({ launch_shape: { launches_base: 4, launches_cand: 3, target: 2 } }, 2);
  ok(v.state === 'above_target' && v.shape_met === false,
     'fused 4 -> 3 launches with target 2 is above_target and shape_met false — the fusion rung is NOT closed on it');
  ok(/caveat/i.test('caveat') && v.caveat && !/reject/i.test(v.caveat),
     'it carries a caveat and does not claim a rejection — partial, not wrong');
}

console.log('\n# the two-launch shape is met when the count reaches the target');
{
  const v = launchVerdict({ launch_shape: { launches_base: 4, launches_cand: 2, target: 2 } }, 2);
  ok(v.state === 'met' && v.shape_met === true, 'fused 4 -> 2 launches reaches the acceptance shape');
  const v1 = launchVerdict({ launch_shape: { launches_cand: 1, target: 2 } }, 2);
  ok(v1.shape_met === true, 'below target also counts as met — the target is a ceiling, not an equality');
}

console.log('\n# the caller owns the target');
{
  const v = launchVerdict({ launch_shape: { launches_cand: 2 } }, undefined);
  ok(v.state === 'unjudged',
    'with no target reported and none passed, launch shape remains unjudged');
  const v3 = launchVerdict({ launch_shape: { launches_cand: 3, target: 2 } }, 5);
  ok(v3.state === 'met',
     'an explicit orchestrator target wins over the verifier field; the verifier reports evidence, not policy');
}

console.log('\n# the verifier cannot move the target or omit the proof');
{
  const moved = launchVerdict({ launch_shape: {
    launches_cand: 3, target: 3, per_rank: true, how_counted: 'trace',
    stages_fused: ['dispatch', 'gemm1', 'gemm2', 'combine'],
  } }, 2, ['dispatch', 'gemm1', 'gemm2', 'combine']);
  ok(moved.state === 'above_target' && !moved.shape_met,
     'an orchestrator target of 2 outranks a verifier-reported target of 3');
  const guessed = launchVerdict({ launch_shape: {
    launches_cand: 2, target: 2,
  } }, 2, ['dispatch', 'gemm1', 'gemm2', 'combine']);
  ok(guessed.state === 'unjudged' && !guessed.shape_met,
     'a bare number without per-rank method and fused-stage evidence is unjudged');
}

if (failures) { console.error(`\nFAIL: ${failures} assertion(s) failed.`); process.exit(1); }
console.log('\nall passed');
