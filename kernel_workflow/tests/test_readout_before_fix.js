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
const megaRole = require('./mega_roles');

const ROOT = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

const lead = megaRole('tech_lead');

console.log('\n# the rule exists and is a plan_round rule');
// It has to bind where directions are chosen. In analyze there is no budget to protect yet, and in
// report it is a postmortem.
const iPlan = lead.indexOf('## PHASE=plan_round');
const iRule = lead.indexOf('must be a READOUT');
const iUpdate = lead.indexOf('## PHASE=update_memory');

console.log('\n# it says WHY, not just what');
// The non-obvious half: a CORRECT inferred fix is also unreadable. Without this, the rule looks like
// "guess better" and a confident tech lead will believe it is exempt.

console.log('\n# the incident keeps its numbers');

console.log('\n# a readout direction is specified concretely enough to assign');
// One-shot instruments waste the scarce resource: a lease, not an agent.

console.log('\n# compile screens are fenced off from standing in for a run');


console.log('\n# a shared box is not a given: plan against the pool you actually have');


// Rule 3c was prose for one wave, and prose is advisory: a whole wave planned three rounds against
// hardware an external tenant was holding, and measured nothing. The script now takes the sample
// itself and hands the lead a fact. These assertions pin the wiring, not just the paragraph.
console.log('\n# the script takes the pool sample itself, so the rule is not merely advisory');
const wf = require('./mega_source')();
ok(/async function samplePool\(round, timeoutMs\)/.test(wf),
   'kernel_lane.js samples the pool rather than trusting the lead to remember');
// It must run BEFORE the plan, or it is a postmortem of a round already budgeted.
const lane = require('./mega_source').full();
const iMegaSample = lane.indexOf('const megaPre = MEGA ? await megaBeforePlan(round) : null;');
const iMegaPlanCall = lane.indexOf('MEGA ? megaPlanPrompt(round, remaining, megaPre.pool)', iMegaSample);
ok(iMegaSample > 0 && iMegaPlanCall > iMegaSample,
   'the candidate path samples before planning, while directions remain changeable');
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

console.log(
  failures === 0
    ? '\nPASS: an opaque runtime blocker buys an instrument before it buys a fix.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
