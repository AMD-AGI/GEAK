#!/usr/bin/env node
// Guards on ACTIVATION, added after a round was spent measuring code that never executed.
//
// What happened: the engineer shipped its fast path behind an env-gated opt-in that defaulted OFF.
// Nobody set it. The candidate arm therefore ran the same instructions as the base arm, the harness
// dutifully reported 1.000x, and the round recorded the direction as tried-and-failed. It had not
// been tried. The three readings that must never again collapse into one number:
//
//   A. the code ran and was no faster        -> a real negative result, charge it to noImprove
//   B. the code never ran                    -> VOID, charge nothing, fix activation
//   C. nobody checked which of A or B it was -> must be treated as B, or B is always cheaper to reach
//
//   1. DECLARED   — the engineer states how its code turns on, and default_on is the required norm.
//   2. THREADED   — that declaration reaches verify, or verify is told it is UNDECLARED.
//   3. OBSERVED   — verify reports a marker it actually saw, not a restatement of the claim.
//   4. VOID != NULL — an unconfirmed activation is excluded from wins AND from the stopping rule.

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

const wf = read('kernel_mega.js');
const eng = read('roles/mega/engineer.md');
const verify = read('roles/mega/verify_engineer.md');

// --- 1. the engineer declares how its code turns on -------------------------
console.log('\n# declared');
ok(/activation: obj\(\{/.test(wf), 'ENG_SCHEMA carries an activation object');
ok(/mode: \{ type: 'string' \}, switch_name/.test(wf),
   'the declaration has a mode plus the switch identity');
ok(/path_marker/.test(wf) && /path_marker/.test(eng),
   'a path marker is part of the contract in both the schema and the role');
ok(/Your patch must be ON by default/.test(eng),
   'the engineer is told default-ON is the requirement, not a preference');
ok(/will not set it/.test(eng),
   'the role names the mechanism: the measurer does not know the flag exists');
ok(/reads \*\*1\.000x\*\*/.test(eng),
   'the role states the exact wrong reading a gated-off patch produces');
ok(/"It should be on" is not a marker/.test(eng),
   'the role rules out a restated intention standing in for an observable');

// --- 2. the declaration reaches verify --------------------------------------
console.log('\n# threaded to verify');
ok(/ACTIVATION: \(eng && eng\.activation\) \? JSON\.stringify\(eng\.activation\) : 'UNDECLARED'/.test(wf),
   'verify receives the declaration verbatim, or the literal UNDECLARED');
ok(/UNDECLARED/.test(verify), 'verify_engineer handles the UNDECLARED case explicitly');
ok(/do not assume default-ON/.test(verify),
   'UNDECLARED does not silently become "assume it is on" — that assumption IS the bug');
ok(/for the CANDIDATE arm only/.test(verify),
   'a switch is set for one arm; setting it for both is the same bug relocated');

// --- 3. verify must OBSERVE, not restate ------------------------------------
console.log('\n# observed');
ok(/activation_confirmed: \{ type: 'string' \}/.test(wf),
   'VERIFY_SCHEMA carries activation_confirmed');
ok(/activation_evidence/.test(wf) && /activation_evidence/.test(verify),
   'the evidence field exists in the schema and in the role contract');
ok(/is not\s*\n?\s*evidence/.test(verify),
   'the role says a restatement of the claim does not count as evidence');
ok(/An empty grep is `no`/.test(verify),
   'a marker that fails to appear is a negative, not an inconclusive');
// The gate must run BEFORE the numbers are trusted; after step 4 it would be certifying a
// measurement whose arms were never distinct.
ok(verify.indexOf('4d.') > 0 && verify.indexOf('4d.') < verify.indexOf('5. Reject if a patch'),
   'the activation step sits inside verification, ahead of the rejection/scoring steps');
ok(/BEFORE trusting any number/.test(verify),
   'the role orders activation ahead of trusting the measurement');

// --- 4. void is not the same as null ----------------------------------------
console.log('\n# void != null');
ok(/const act = String\(r\.ver\.activation_confirmed \|\| 'unknown'\)\.toLowerCase\(\)/.test(wf),
   'a missing activation_confirmed defaults to unknown rather than to pass');
ok(/if \(act === 'yes'\) continue/.test(wf),
   'only an explicit yes clears the gate');
ok(/INACTIVE \$\{r\.d\.id\}/.test(wf), 'inactive directions are logged per direction');
ok(/is VOID — not a negative result/.test(wf),
   'the log states the distinction the incident collapsed');
ok(/it has \` \+\s*\n?\s*`not been tried/.test(wf) || /not been tried/.test(wf),
   'the log forbids recording the direction as tried-and-failed');
ok(/&& !r\.inactive\)/.test(wf),
   'inactive directions cannot become verified wins either');
// The all-inactive exemption this file used to check has been widened and moved into
// `roundEvidence` (tests/test_evidence_stop.js) — "every direction was inactive" missed a round
// where the directions failed in different ways. What must remain true here is the consequence:
// an inactive direction is not evidence, so a round with nothing but inactive directions cannot
// advance the stopping criterion.
ok(/if \(r\.inactive\) return `the patched path was/.test(wf),
   'an inactive direction is excluded from a round\'s evidence by name');
ok(/NOT counted toward noImprove/.test(wf),
   'a round with no admissible measurement does not advance the stopping criterion');
ok(/produced no evidence about the kernel/.test(wf),
   'the code says why: the round measured nothing about the subject');

console.log(
  failures === 0
    ? '\nPASS: activation is declared, threaded, observed, and an unexercised patch is void rather than null.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
