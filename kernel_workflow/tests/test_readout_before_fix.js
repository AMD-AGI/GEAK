#!/usr/bin/env node
// Guard for plan_round rule 3b: when the blocker is an opaque RUNTIME failure, the round's first
// direction is an instrument, not a fix.
//
// This rule was written after a program spent three consecutive rounds dispatching fix arms at one
// megakernel illegal access. Both rounds that actually moved the blocker were readout rounds (a
// compile-time phase ladder; a debug dump of the disputed bound). The four-fix round in between was
// marker-proved to have executed and changed nothing.
//
// The rule is prose in a role file, which is exactly the kind of thing that gets summarised away
// under a re-flow. The assertions below are the ratchet. They deliberately pin the INCIDENT NUMBERS
// as well as the rule: without them the paragraph reads as methodology advice and gets skipped when
// a round is under budget pressure, which is precisely when it is load-bearing.

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

console.log('\n# the rule exists and is a plan_round rule');
ok(/must be a READOUT,\s*\n?\s*not a fix/.test(lead),
   'plan_round carries the readout-before-fix rule');
// It has to bind where directions are chosen. In analyze there is no budget to protect yet, and in
// report it is a postmortem.
const iPlan = lead.indexOf('## PHASE=plan_round');
const iRule = lead.indexOf('must be a READOUT');
const iUpdate = lead.indexOf('## PHASE=update_memory');
ok(iPlan > 0 && iRule > iPlan && iRule < iUpdate,
   'the rule sits inside plan_round, where directions are actually assigned');

console.log('\n# it says WHY, not just what');
// The non-obvious half: a CORRECT inferred fix is also unreadable. Without this, the rule looks like
// "guess better" and a confident tech lead will believe it is exempt.
ok(/a right one is indistinguishable from a\s*\n?\s*wrong one/.test(lead),
   'the rule states that even a correct inference-aimed fix measures nothing');
ok(/missing instrument/.test(lead),
   'the failure is framed as a missing instrument rather than as a bug to be guessed');

console.log('\n# the incident keeps its numbers');
ok(/three consecutive rounds/i.test(lead) || /\*\*three consecutive rounds\*\*/.test(lead),
   'the cost is recorded: three rounds of fix arms on one fault');
ok(/16\/16 ranks/.test(lead) && /two-source-line bracket/.test(lead),
   'the payoff is recorded: an opaque 16/16 fault became a two-line bracket');
ok(/7\.5× under its clamp|7\.5x under its clamp/.test(lead),
   'the falsifying datum is recorded — a readout, not a fix, is what closed the hypothesis');

console.log('\n# a readout direction is specified concretely enough to assign');
ok(/caps an unbounded wait/.test(lead),
   'bounded-spin instrumentation is named as a qualifying readout');
ok(/truncates the kernel at N points/.test(lead),
   'phase-ladder bisection is named as a qualifying readout');
ok(/dumps the disputed quantity/.test(lead),
   'dumping the contested value is named as a qualifying readout');
// One-shot instruments waste the scarce resource: a lease, not an agent.
ok(/do not stop at the first failure/.test(lead) && /in one lease/.test(lead),
   'record-and-continue is preferred over halt-on-first, and the reason is the lease');

console.log('\n# compile screens are fenced off from standing in for a run');
ok(/compile-clean is not runnability evidence/.test(lead),
   'the compile-screen loophole is closed explicitly');
ok(/use them to \*rejec?t\*|use them to \*reject\*/.test(lead),
   'compile screens keep their real use (rejection), so the rule is not read as "never screen"');
ok(/dead on hardware/.test(lead),
   'the compile-screen claim carries its own counterexample');


console.log('\n# a shared box is not a given: plan against the pool you actually have');
ok(/SAMPLE THE POOL BEFORE YOU BUDGET A GPU DIRECTION/.test(lead),
   'plan_round checks hardware availability before spending a direction on it');
ok(/gpu_busy_percent/.test(lead) && /mem_info_vram_used/.test(lead),
   'the check names the sysfs files, so it is executable rather than aspirational');
ok(/foreign namespace/.test(lead),
   'an external tenant is distinguished from a stale lease of our own — the two have opposite fixes');
ok(/two consecutive rounds/.test(lead) && /95 minutes/.test(lead),
   'the incident keeps its cost: two rounds lost, one direction 95 minutes inside flock');
ok(/banks a runnable instrument plus its driver is a \*partial\*/.test(lead),
   'the GPU-less plan is specified as bankable work, not as "skip the round"');
ok(/A direction is not finished when its round ends/.test(lead),
   'a live prior-round engineer is not re-dispatched into a duplicate');
ok(/live parent; that is not the orphan case/.test(lead),
   'the rule stops short of killing an agent that is still owned');


// Rule 3c was prose for one wave, and prose is advisory: a whole wave planned three rounds against
// hardware an external tenant was holding, and measured nothing. The script now takes the sample
// itself and hands the lead a fact. These assertions pin the wiring, not just the paragraph.
console.log('\n# the script takes the pool sample itself, so the rule is not merely advisory');
const wf = read('kernel_mega.js');
ok(/async function samplePool\(round, timeoutMs\)/.test(wf),
   'kernel_mega.js samples the pool rather than trusting the lead to remember');
// It must run BEFORE the plan, or it is a postmortem of a round already budgeted.
const iSample = wf.indexOf('const pool = await samplePool(round)');
const iPlanCall = wf.indexOf("roleAgent('tech_lead', 'plan_round'", iSample);
const iMegaSample = wf.indexOf('const pool = (MEGA_STRUCTURAL_ONLY');
const iMegaPlanCall = wf.indexOf('const d = await planMegaCandidateTurn', iMegaSample);
ok(iSample > 0 && iPlanCall > iSample &&
   iMegaSample > 0 && iMegaPlanCall > iMegaSample,
   'both canonical and Mega candidate paths sample before plan_round, while directions remain changeable');
ok(/\.\.\.\(pool \? \{ GPU_POOL: pool, GPU_MIN_FREE_GIB \} : \{\}\)/.test(wf),
   'the verdict is threaded into the lead\'s inputs, not just logged');
// Three-valued for the same reason the containment verdict is.
ok(/enum: \['free', 'occupied', 'unknown'\]/.test(wf),
   'the pool verdict is three-valued: unreadable is not free');
ok(/Treating as UNKNOWN, not free/.test(wf),
   'a sampler that returns nothing degrades to unknown rather than to a green light');
ok(/does not abort the run; an\s*(\/\/\s*)?occupied pool redirects the round to lease-free work/.test(wf),
   'an occupied pool redirects the round; it does not abort the run');
// Off by default, so no existing run's prompt changes.
ok(/if \(!\(GPU_MIN_FREE_GIB > 0\)\) return null;/.test(wf),
   'the sample is opt-in via gpu_min_free_gib, so runs that never set it are unaffected');
ok(/`occupied` \*\*or\*\* `unknown`/.test(lead),
   'the lead is told unknown binds the same way occupied does');

console.log(
  failures === 0
    ? '\nPASS: an opaque runtime blocker buys an instrument before it buys a fix.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
