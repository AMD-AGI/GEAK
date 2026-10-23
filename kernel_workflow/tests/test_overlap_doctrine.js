#!/usr/bin/env node
// Wiring guard for the overlap doctrine added from the megakernel/PDL literature review.
//
// These are not documentation assertions for their own sake. Each one guards a specific way the
// workflow can produce a CONFIDENTLY WRONG result, and each is placed on the surface the agent that
// could make that mistake actually reads:
//
//   1. Sub-timers under concurrency. Per-stage timers are only additive while stages are
//      serialized. A successful overlap inflates them. If the benchmark contract does not say so,
//      the first working fusion in this workflow gets reported as a regression and reverted. This
//      is the highest-cost failure in the set because it destroys a correct result silently.
//   2. Attribution. A fused win that was really launch-overhead removal means the megakernel was
//      the wrong build; without the ablation ladder nobody can tell.
//   3. The reduction test. Condition 1 of the precondition test is checkable statically in minutes
//      and kills most candidate edges. Skipping it buys a fusion designed around an impossible
//      overlap.
//   4. Compiler hoisting. `const __restrict__` on cross-block state defeats the readiness wait at
//      compile time and produces plausible stale data that passes tolerance checks.
//   5. Event pairing. `whole` as the default is what makes a fusion bisectable.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
// Prose in these files is hard-wrapped, so a phrase spanning a line break would not match a
// literal regex. Match against a whitespace-normalized copy for anything sentence-shaped.
const flat = (s) => s.replace(/\s+/g, ' ');
const read = (...p) => flat(fs.readFileSync(path.join(WF, ...p), 'utf8'));

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const df = read('knowledge', 'distributed_fusion.md');
const fp = read('knowledge', 'fusion_preconditions.md');
const tg = read('knowledge', 'tile_task_graph.md');
const po = read('knowledge', 'pipe_occupancy.md');
const amd = read('knowledge', 'amd_instinct.md');
const bench = read('roles', 'mega', 'benchmark_engineer.md');

console.log('\n# 1. per-stage timers are not summed once anything overlaps');
{
  // The knowledge and the role that writes the measurement contract must both carry it.
  // The fixed task prompt deliberately contains no mechanism doctrine.
  ok(/becomes an artifact/.test(df) && /62\.4 → 102\.9/.test(df),
     'distributed_fusion carries the mechanism and the measured instance');
  ok(/never `Σ stage_i`|never sum them/.test(df),
     'and states the prohibition, not merely the observation');
  ok(/inflates while the span shrinks|rises while e2e falls/.test(df + bench),
     'the inverted signal is named as EVIDENCE OF OVERLAP, so the timers are kept rather than dropped');
  ok(/diagnostic and are never summed/.test(bench),
     'benchmark_engineer must put it in METRIC — that is the text the optimize loop reads');
}

console.log('\n# 2. the win is attributed, or the megakernel may have been the wrong build');
{
  ok(/Attribute the win/.test(fp), 'fusion_preconditions has the attribution section');
  ok(/T4 → T5/.test(fp) && /graph capture/.test(fp),
     'the ablation ladder separates launch-cost removal from fine-grained overlap');
  ok(/fraction of each producer was actually overlapped/.test(fp),
     'and demands the overlapped FRACTION rather than the word "overlap"');
}

console.log('\n# 3. condition 1 is checked statically, first, from the source');
{
  ok(/first operation the consumer performs on the producer's output/.test(fp),
     'the reduction test is stated as a concrete thing to look at');
  ok(/\*\*28% of the producer/.test(fp) && /exactly one\*\*/.test(fp),
     'with the calibration that says how rare exploitable edges actually are');
}

console.log('\n# 4. a wait the compiler can move is not a wait');
{
  ok(/__restrict__/.test(df) && /hoisted the load/.test(df),
     'the aliasing-qualifier hazard is recorded with its mechanism');
  ok(/emitted ISA/.test(df) && /poisoned inputs|poison/.test(df),
     'and both verifications — read the ISA, test with poison, not with real inputs');
  ok(/passes a tolerance check/.test(df),
     'the reason it is dangerous: it is silent under the correctness gate we run');
}

console.log('\n# 5. event pairing is a decision with a default');
{
  ok(/tile_cover/.test(tg) && /tile_reduce/.test(tg), 'the four pairing strategies are enumerated');
  ok(/`whole` is the correct default/.test(tg),
     'with the conservative default, which is what makes a partial fusion bisectable');
  ok(/pairing/.test(tg.split('## The artifact')[1] || ''),
     'and the artifact carries it, so the choice survives into the implementation');
}

console.log('\n# 6. the cheap checks that come before a counter, and the costs that come after');
{
  ok(/min\(grid_blocks × blocks_per_CU, CUs\)/.test(po),
     'pipe_occupancy has the counter-free active-CU upper bound');
  ok(/a low upper bound is conclusive/.test(po),
     'and says why an optimistic bound is still usable');
  ok(/split loader/.test(po) && /WLoader|weight loader/.test(po),
     'the split-loader recipe is the concrete form of prologue overlap');
  ok(/dependency is on the first READ/.test(po),
     'framed by the reframing that makes it derivable rather than a trick to memorize');
  ok(/framework/i.test(po) && /4\.671/.test(po),
     'the fused form is charged its fixed framework cost');
  ok(/one register allocation|one resource shape/i.test(po),
     'and the one-resource-shape cost that bounds what should be swallowed');
}

console.log('\n# 7. AMD-specific: synchronize at the narrowest correct level');
{
  ok(/own L2/.test(amd) && /hierarchically/.test(amd),
     'amd_instinct records the multi-die coherence hierarchy and hierarchical sync');
  ok(/tile_task_graph\.md/.test(amd),
     'and connects it to the edge-scope column that is supposed to act on it');
  ok(/showcomputepartition/.test(amd),
     'partition mode is flagged, since it changes what "the device" means');
}

console.log('\n# 8. the negative results are kept, not filtered out');
{
  ok(/0\.82–0\.89×/.test(fp),
     'dynamic scheduling losing on regular dense work is recorded');
  ok(/lose to a tuned library kernel/.test(fp),
     'so is the compiler-tile-vs-library regression a fusion has to pay for');
  ok(/decays as the regime becomes compute-bound/.test(fp),
     'and the regime dependence, so a small-shape win is not generalized');
}

console.log(
  failures === 0
    ? '\nPASS: the overlap doctrine is on the surfaces that can act on it.'
    : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
