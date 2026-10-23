#!/usr/bin/env node
// Wiring guard for the ISA-level evidence doctrine distilled from an earlier run.
//
// One run produced three findings that live entirely BELOW the source diff, and each guards a
// distinct way this workflow spends a lease to learn nothing:
//
//   1. LICM → spill. The compiler hoisting constant-folded addresses out of a peeled loop tail is
//      invisible in the source and shows up only as scratch in the kernel descriptor. Without the
//      detection recipe an engineer tunes the loop body forever; without the fragility warning the
//      inline-asm anchor becomes an unrecorded dependency on one compiler version.
//   2. Identical ISA as a RESULT. A guard whose two arms compile to the same binary cannot have
//      moved. Not knowing this converts a settled guard into ten more paired reps.
//   3. Per-guard compiled config. A change gated on one tile config is structurally unable to
//      touch a guard that compiles another; attributing a delta there is attributing noise.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
// Prose is hard-wrapped; match against a whitespace-normalized copy.
const flat = (s) => s.replace(/\s+/g, ' ');
const read = (...p) => flat(fs.readFileSync(path.join(WF, ...p), 'utf8'));

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const hip = read('knowledge', 'hip_optimization.md');
const jit = read('knowledge', 'jit_arm_isolation.md');

console.log('\n# 1. the hoist that costs more than the recompute');
{
  ok(/LICM/.test(hip),
     'the compute card names LICM, so the failure is searchable at all');
  ok(/peeled/.test(hip) && /constant-fold/.test(hip),
     'the triggering shape — a peeled tail whose addresses constant-fold — is stated');
  ok(/scratch_load/.test(hip),
     'detection is anchored on scratch traffic inside the hot loop');
  ok(/kernel descriptor|private-segment/.test(hip),
     'and on the spill counts in the ISA metadata, not on reading the source');
  ok(/identity inline-asm|v_mov_b32/.test(hip),
     'the remedy (an identity inline-asm anchor) is written down, not just the diagnosis');
  ok(/bit-identical/.test(hip),
     'the correctness argument — the anchor changes eligibility, not the value — is explicit');
}

console.log('\n# 2. the anchor is recorded as fragile, not as a trick');
{
  ok(/Re-check after any toolchain bump/.test(hip),
     'a toolchain bump is named as the thing that silently undoes it');
  ok(/pure overhead/.test(hip),
     'both failure directions are given: LICM sees through it, or never hoisted at all');
  ok(/compiler version/.test(hip),
     'the compiler version must be recorded next to the ISA deltas');
  ok(/Lever 6b/.test(hip),
     'and it is linked to the other source-says-one-thing-ISA-says-another lever');
}

console.log('\n# 3. identical ISA is a result, not only a failure detector');
{
  ok(/byte-identical\s*\*\*?\s*ISA|byte-identical ISA/.test(jit),
     'the arm-isolation card states the inverted use of the same instrument');
  ok(/cannot be slower than itself/.test(jit),
     'the argument is given in a form an engineer can quote in a report');
  ok(/could not have moved/.test(jit),
     'and it is contrasted with what a measurement can establish (a noise floor only)');
  ok(/9b3da4e5b1181d5a/.test(jit),
     'with the reproduced hash, so the claim is auditable rather than anecdotal');
  ok(/zero GPU cost|zero lease/.test(jit),
     'the cost argument that makes anyone actually reach for it is stated');
}

console.log('\n# 4. a guard that compiles a different config is a different experiment');
{
  ok(/different compiled configuration|different kernel config/.test(jit),
     'per-guard config divergence is named');
  ok(/structurally incapable|cannot reach/.test(jit),
     'and the consequence — the change cannot touch that guard — is spelled out');
  ok(/enumerate the compiled config per guard|Enumerate the compiled config/i.test(jit),
     'the required action is an enumeration, not a caution');
  ok(/cache key/.test(jit),
     'with the mechanism for getting it (the resolved kernel name / cache key)');
  ok(/BM=64|non-persistent/.test(jit),
     'and a concrete instance, so the shape of the divergence is recognizable');
}

console.log('\n# 5. the null-arm obligation is still where it was');
{
  // Item (iv) from the wave-12 list — an escape hatch proved inert by a matching hash — was
  // already covered here before an earlier run. Guard it so a future edit does not drop it while
  // rewriting the section above.
  ok(/byte-identical work, its hash shown to match canonical/.test(jit),
     'a null arm must still be shown byte-identical to canonical, not merely asserted inert');
}

console.log(
  failures === 0
    ? '\nPASS: the ISA-evidence doctrine is on the surfaces that can act on it.'
    : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
