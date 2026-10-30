#!/usr/bin/env node
// The 512 guards are bimodal, and the statistic the doctrine prescribed for them cannot be computed.
//
// "Deep sample, judge by complete separation" is sound on a unimodal guard and unreachable on a
// bimodal one: both arms draw slow runs, the ranges interleave, and no depth of sampling separates
// them. So the two 512 guards sat demoted to regression-only while the effect being hunted (5-6%)
// hid under a 9.30% worst pair. Conditioning on the state is the way out — and conditioning is
// exactly the kind of tool that turns into a knob for manufacturing wins, so this suite spends most
// of its assertions on the ways it must REFUSE:
//
//   1. The split must be arm-blind. If which-arm-is-which can move the boundary, the boundary is a
//      free parameter and the engineer picks the one that flatters the candidate.
//   2. Refusing to classify must stay available. A classifier that always finds two clusters finds
//      them in pure noise, which is worse than the honest all-pairs analysis it replaced.
//   3. Occupancy must survive as a reported number, not be divided out. A candidate that enters the
//      slow state less often has the largest effect available on this guard; a fast-mode-only
//      report throws that win away as though it were noise.
//
// It lifts the pure region out of kernel_mega.js the way replay_runs.js does.
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

const m = src.match(/\/\/ <<REPLAY:bimodal_split>>([\s\S]*?)\/\/ <<\/REPLAY:bimodal_split>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:bimodal_split>> region — nothing to test'); process.exit(1); }
const { modeSplit, pairedModeAware, pairsNeeded, analyzeGuards } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { modeSplit, pairedModeAware, pairsNeeded, analyzeGuards };`)();

// Shapes taken from the measured 512_rank-mixed-skew guard: a tight cluster near 0.760 ms and a
// reproducible slow state ~9% above it.
const FAST = [0.760, 0.7605, 0.7598, 0.761, 0.7602, 0.7607, 0.7595, 0.7612];
const SLOW = [0.830, 0.8305, 0.8298, 0.831];

console.log('\n# the split finds two states only when they are there');
{
  const s = modeSplit([...FAST, ...SLOW]);
  ok(s.classified === true, 'a tight cluster plus a reproducible ~9% tail is classified as two states');
  ok(s.cut > Math.max(...FAST) && s.cut < Math.min(...SLOW),
     'the cut lands strictly between the clusters, so no reading is assigned to the wrong state');
  ok(s.n_fast === FAST.length && s.n_slow === SLOW.length, 'every reading is accounted for on one side');
}
{
  // The 8192 guard: unimodal, worst pair 1.09%. Nothing here is two states.
  const uni = [1.000, 1.003, 0.998, 1.006, 1.002, 0.997, 1.005, 1.001];
  const s = modeSplit(uni);
  ok(s.classified === false, 'a unimodal spread is REFUSED rather than split at its widest accident');
  ok(/gap/.test(s.reason || ''), 'the refusal says why, so the caller reports all pairs and knows it did');
}
{
  ok(modeSplit([0.76, 0.83, 0.76]).classified === false,
     'fewer than 6 readings cannot show a gap and are refused, not split on 2 points');
  ok(modeSplit([...FAST.slice(0, 4), ...SLOW, 0.8301, 0.8299]).classified === false,
     'if the upper cluster is the majority it IS the operating point — naming it "slow" would invert ' +
     'every conclusion, so the split refuses');
}

console.log('\n# classification is arm-blind');
{
  const pairs = [
    { base: 0.830, cand: 0.760 }, { base: 0.7602, cand: 0.7551 }, { base: 0.7605, cand: 0.7549 },
    { base: 0.7598, cand: 0.8301 }, { base: 0.7610, cand: 0.7552 }, { base: 0.7601, cand: 0.7548 },
    { base: 0.7606, cand: 0.7553 }, { base: 0.8305, cand: 0.8298 },
  ];
  const a = pairedModeAware(pairs);
  const swapped = pairedModeAware(pairs.map((p) => ({ base: p.cand, cand: p.base })));
  ok(a.conditioned && swapped.conditioned && a.cut === swapped.cut,
     'swapping which arm is baseline does not move the cut — the boundary is not a free parameter');
  ok(a.fast.n === swapped.fast.n && a.fast.agree === swapped.fast.agree &&
     a.fast.median > 0 && swapped.fast.median < 0,
     'the same pairs survive and the fast-mode delta changes sign, as a paired statistic must ' +
     '(magnitude shifts by O(delta^2): the percentage is against whichever arm is the denominator)');
}

console.log('\n# like is compared with like, and what was dropped is disclosed');
{
  const pairs = [
    { base: 0.7600, cand: 0.7550 }, { base: 0.7605, cand: 0.7548 }, { base: 0.7598, cand: 0.7552 },
    { base: 0.7610, cand: 0.7551 }, { base: 0.7602, cand: 0.7549 }, { base: 0.7607, cand: 0.7553 },
    { base: 0.8300, cand: 0.7550 },   // mixed: baseline slow
    { base: 0.7601, cand: 0.8298 },   // mixed: candidate slow
    { base: 0.8305, cand: 0.8301 },   // both slow
  ];
  const a = pairedModeAware(pairs);
  ok(a.conditioned === true, 'the pooled readings classify');
  ok(a.fast.n === 6, 'only the both-fast pairs enter the primary statistic');
  ok(a.dropped.mixed_mode_pairs === 2 && a.dropped.both_slow_pairs === 1,
     'the excluded pairs are counted and reported — a conditioned analysis that hides its exclusions ' +
     'is indistinguishable from a cherry-picked one');
  ok(a.fast.agree === 6 && Math.abs(a.fast.p - Math.pow(2, -5)) < 1e-12,
     'unanimity over 6 fast pairs is reported with its sign-test p, not as a ratio of medians');
  ok(a.all.deltas.length === 9 && a.all.median !== null,
     'the unconditioned all-pairs number is still computed, so conditioning is visible as a choice');
  ok(a.fast.median > a.all.median,
     'and on this data conditioning changes the answer — which is the whole reason it must be audited');
}
{
  const near = [
    { base: 0.760, cand: 0.759 }, { base: 0.761, cand: 0.758 }, { base: 0.759, cand: 0.760 },
    { base: 0.762, cand: 0.757 }, { base: 0.760, cand: 0.761 }, { base: 0.758, cand: 0.759 },
  ];
  const a = pairedModeAware(near);
  ok(a.conditioned === false && a.all.deltas.length === 6,
     'with no gap the caller still gets a complete all-pairs analysis — refusing to classify is not ' +
     'refusing to answer');
  ok(a.all.p === null, 'and a split 3-3 sign test reports no p rather than a flattering one');
}

console.log('\n# occupancy is a result, not a nuisance');
{
  // The candidate is NOT faster when both arms are fast. Its entire effect is entering the slow
  // state half as often. A fast-mode-only report would score this a null.
  const pairs = [
    { base: 0.8300, cand: 0.7600 }, { base: 0.8305, cand: 0.7602 }, { base: 0.7601, cand: 0.7600 },
    { base: 0.7603, cand: 0.7604 }, { base: 0.7600, cand: 0.7601 }, { base: 0.7605, cand: 0.8300 },
    { base: 0.8298, cand: 0.7603 }, { base: 0.7602, cand: 0.7599 },
  ];
  const a = pairedModeAware(pairs);
  ok(a.conditioned === true && Math.abs(a.fast.median) < 0.5,
     'the fast-mode delta is ~0 — by that number alone the candidate did nothing');
  ok(a.occupancy.base.slow === 3 && a.occupancy.cand.slow === 1,
     'but occupancy shows the baseline entered the slow state 3 times and the candidate once');
  ok(a.occupancy.base.pct > a.occupancy.cand.pct,
     'reported per arm as a percentage, so the real effect is impossible to miss in the report');
}

console.log('\n# the sample is sized for the usable pairs, not the collected ones');
{
  ok(pairsNeeded(10, 20) === 16,
     'at the measured ~20% slow rate, 10 usable pairs needs 16 collected — "10 pairs" was delivering ~6');
  ok(pairsNeeded(3, 0) === 3, 'on a guard with no slow state the ask is unchanged');
  ok(pairsNeeded(10, 50) === 40, 'and a guard that is half slow costs 4x, which is worth knowing up front');
}

console.log('\n# analyzeGuards is the call site the three functions never had');
{
  // No readings -> not available, nothing asserted, the aggregate geomean stays in charge.
  ok(analyzeGuards([], ['8192_uniform'], 10).available === false,
     'with no paired_readings the analysis is unavailable and changes nothing');
  // A clean, well-sampled unimodal target guard (arms overlap, small effect, no bimodal state) yields
  // a target_median and no caveat. base/cand interleave so the pooled readings show no 3% gap.
  const clean = [];
  for (let i = 0; i < 12; i++) clean.push({ guard: '8192_uniform', base: 5.42 + (i % 4) * 2e-3, cand: 5.41 + (i % 4) * 2e-3 });
  const cg = analyzeGuards(clean, ['8192_uniform'], 10);
  ok(cg.available && cg.target_median != null && cg.caveats.length === 0,
     'a well-sampled unimodal target route produces a conditioned median and no caveat');
}
{
  // A win read ONLY off a skew rail: target set is uniform, but the only readings are skew. That is
  // the exact w15 state — a number from 8192_rank-mixed-skew, out of scope.
  const skewOnly = [];
  for (let i = 0; i < 10; i++) skewOnly.push({ guard: '8192_rank-mixed-skew', base: 0.76, cand: 0.70 });
  const sg = analyzeGuards(skewOnly, ['8192_uniform'], 10);
  ok(sg.target_median === null && sg.off_target.join() === '8192_rank-mixed-skew' &&
     sg.caveats.some((c) => /No reading on a TARGET route/.test(c)),
     'a win read only off a skew rail has no target median and is called out — the out-of-scope state w15 shipped');
}
{
  // With no target configured, a guard named *uniform* is treated as the target automatically.
  const rows = [];
  for (let i = 0; i < 12; i++) rows.push({ guard: '512_uniform', base: 0.76 + i * 1e-3, cand: 0.73 + i * 1e-3 });
  const g = analyzeGuards(rows, [], 10);
  ok(g.guards[0].is_target === true,
     'with no explicit target set, a *uniform* guard is the target by default — the campaign\'s route');
}

console.log(failures === 0
  ? '\nPASS: the 512 statistic conditions on the state, refuses when it cannot, never hides occupancy, and is now WIRED.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
