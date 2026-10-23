#!/usr/bin/env node
// Guard for the knowledge that was being carried in a per-WAVE task string instead of in the
// workflow itself.
//
// Five waves of a multi-rank fusion program produced three facts that decide how a round should be
// planned and how an A/B should be authored:
//   1. the fused kernel could not be compile-screened AT ALL (it calls the comm runtime at trace
//      time), so every hypothesis cost a full multi-rank run;
//   2. the device pool was ONE lease with a hard 900 s cap, so a round's GPU capacity is one
//      direction and a plan that only reports at the end reports nothing;
//   3. an env-gated arm on a caching JIT can dispatch the SAME binary on both arms, which reads
//      1.000 with every gate satisfied.
// All three were written into the wave's task text. A task string is consumed once and thrown away;
// the next environment starts without any of it, and a human has to remember to re-type it. That is
// the failure this file exists to prevent: durable knowledge belongs in the workflow, and the
// assertions below are what stop it from being summarised back out.

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

const lead = read('roles/mega/tech_lead.md');
const eng = read('roles/mega/engineer.md');
const deep = read('roles/mega/deep_engineer.md');
const fusion = read('knowledge/distributed_fusion.md');
const jit = read('knowledge/jit_arm_isolation.md');

console.log('\n# lease economics bind where the round is planned');
const iPlan = lead.indexOf('## PHASE=plan_round');
const iLease = lead.indexOf('LEASE ECONOMICS');
const iUpdate = lead.indexOf('## PHASE=update_memory');
ok(iLease > iPlan && iLease < iUpdate,
   'the lease rules sit inside plan_round, where GPU capacity is actually allocated');
ok(/\*\*One GPU direction per round\.\*\*/.test(lead),
   'one GPU direction per round when the pool is a single lease');
ok(/destroyed a complete four-guard table twice/.test(lead),
   'the rule keeps the incident that produced it, so it is not read as mere tidiness');
ok(/GEAK_GPU_RUN_TIMEOUT/.test(lead) && /900 s/.test(lead),
   'the lease cap is named concretely enough to plan against');
ok(/re-emit its claim after\n\s*every chunk/.test(lead),
   'chunked reporting is required, not suggested — an expiring lease must not lose the work');
ok(/\*\*Coverage before depth\.\*\*/.test(lead),
   'all guards at low reps before deepening one');
ok(/Four guards at 5\s*\n?\s*\s*pairs is a reportable geomean/i.test(lead),
   'coverage-first carries its arithmetic, not just its name');
// Extra parallelism in a lease-bound round has to be lease-FREE, or the rule just reduces throughput.
ok(/must be \*lease-free\* work \(rule 3c\)/.test(lead),
   'the one-direction rule redirects the spare capacity instead of idling it');
ok(/Engineers cannot message each other/.test(lead) && /STATE_DIR/.test(lead),
   'the harness has no inter-engineer channel, and the substitute is named');

// The rules above decide HOW MANY GPU directions a round gets. They say nothing about WHICH one gets
// the slot, and that turned out to be the binding question. One run obeyed every rule on this page --
// one lease per round, coverage before depth, chunked reporting, 40/40 invocations OK -- and still
// finished at 1.000x, because all three of its leases went to fresh local resource-tuning arms while
// the structural arm the task actually asked for sat authored and unmeasured in the bank. Banking is
// locally rational in every round; without a rule that forces the bank to be spent, it never is.
console.log('\n# WHICH direction gets the single lease, not just how many');
ok(/A BANKED ARM IS A DEBT, NOT AN ASSET/.test(lead),
   'a banked-but-unmeasured arm is framed as a debt, so it cannot accumulate unnoticed');
ok(/\*\*Zero of the four were ever measured\.\*\*/.test(lead),
   'the incident keeps the number that makes it a failure rather than a style note');
ok(/authored and compile-verified in an earlier round outranks a new arm/.test(lead),
   'an older banked arm outranks a fresh one of comparable size when the lease is allocated');
ok(/say in `reasoning`\s*\n?\s*which round is going to spend it/.test(lead),
   'banking a second artifact while the first is unmeasured has to be justified out loud');
// Rule 3c actively rewards banking ("a partial worth having"). Without this the two rules conflict
// and the one that costs nothing this round wins.
ok(/Rule 3c tells you/.test(lead),
   'the debt rule names the rule it is in tension with, instead of quietly contradicting it');
ok(/Two falsifications on one axis close the axis/.test(lead),
   'a twice-falsified axis does not get a third lease');
ok(/tests what\s*\n?\s*the TASK names as its objective/.test(lead),
   'the tie-break favours the task objective over the tidier local lever');
ok(/cannot be compile-screened at\s*\n?\s*all/.test(lead),
   'and it says why — a structural arm has no cheap substitute, a tile-shape arm does');

console.log('\n# a kernel that cannot be compile-screened changes the whole schedule');
ok(/Lever 10 — Some fused kernels cannot be compile-screened AT ALL/.test(fusion),
   'distributed_fusion carries the lease-only kernel class as a lever');
ok(/at trace\/codegen time/.test(fusion),
   'the mechanism is named — the comm runtime is called during codegen, not just at run time');
ok(/There is no partial screen/.test(fusion),
   'the card forecloses the natural workaround of screening it partially');
ok(/\*\*Test it once, at the start\*\*/.test(fusion),
   'the verdict is established once rather than re-discovered every round');
ok(/record-and-continue\*\*, never halt-on-first-failure/.test(fusion),
   'a lease-bound bisection must not spend the whole lease learning one bit');
// The rule must not be over-read into "static screening is useless".
ok(/compile screens are still valuable for every other kernel/i.test(fusion) &&
   /dead on hardware/.test(fusion),
   'compile screens keep their real use, with the counterexample that bounds it');
// It has to be the FIRST thing checked, since it prices every later iteration.
ok(/0\. \*\*Establish whether the kernel is compile-screenable\*\*/.test(fusion),
   'the screenability check is priority 0 — it sets the cost of every iteration after it');
ok(/Lever 10/.test(lead),
   'the tech lead is pointed at the lever when planning a lease-bound round');

console.log('\n# the JIT one-binary trap is taught at AUTHORING time, not only at verification');
ok(/what enters the key/i.test(jit) && /_key_anchor/.test(jit),
   'the card states what enters the compile-cache key and gives the anchor remedy');
ok(/per-machine, not per-tree/.test(jit),
   'the card kills the "separate checkouts isolate the arms" intuition explicitly');
ok(/only the outer function's `co_names`/.test(jit),
   'the worked mechanism is concrete, not a general warning');
// The half that is easy to forget: the null arm must MATCH canonical.
ok(/must \*\*match\*\* canonical as surely as the candidate must \*\*differ\*\*/.test(jit),
   'both directions of the proof are stated — differ from base AND match canonical');
ok(/name-normalis/i.test(jit),
   'the ISA hash is name-normalised so a symbol rename cannot fake a difference');
// Distinct failure from inactivation, or engineers will apply the wrong remedy.
ok(/Do not confuse this with activation/.test(jit) && /anchor the key/.test(jit),
   'inactive and one-binary are separated, with different remedies');
ok(/Triton|torch\.compile/.test(jit),
   'the card generalises past the one framework it was learned on');

console.log('\n# the roles actually route engineers to it');
ok(/jit_arm_isolation\.md/.test(eng),
   'engineer.md points at the card');
ok(/BEFORE you author the switch/.test(eng),
   'the pointer fires at authoring time, which is the only time the fix is cheap');
ok(/costs one\nhash; discovering it at verification costs the round/.test(eng),
   'the cost asymmetry is stated, so the step is not skipped under pressure');
ok(/jit_arm_isolation\.md/.test(deep) && /REQUIRED before authoring any env\/config-gated arm/.test(deep),
   'deep_engineer lists the card as required rather than optional reading');

console.log(
  failures === 0
    ? '\nPASS: what five waves learned lives in the workflow, not in a task string.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
