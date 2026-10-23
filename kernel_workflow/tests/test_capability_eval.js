#!/usr/bin/env node
// Guards for capability-evaluation mode, added after a run imported its own answer.
//
// What happened: the prior-art sweep (test_measurement_gates.js #2) worked exactly as designed. It
// found a reference implementation beside the workspace and told everyone about it — path, branch,
// HEAD — in roadmap.md, which engineers read. An engineer's patch came back containing a 511-line
// file BYTE-IDENTICAL to that reference, measured +4.3%, and demonstrated nothing about the
// workflow's ability to optimize anything.
//
// The doctrine was not wrong. "Do not re-derive working code, port it" is correct when the goal is a
// fast kernel. It is destructive when the goal is measuring whether the workflow can PRODUCE a fast
// kernel. Two contexts, opposite instructions, and previously only one of them was expressible.
//
//   1. MODE EXISTS      — capability_eval is a real, default-OFF switch (OFF must be byte-identical).
//   2. ASYMMETRY        — locations reach VERIFY (which needs them to compare) and never engineers.
//   3. PROVENANCE GATE  — byte-identical files are status:"plagiarized", regardless of speedup.
//   4. NOT SILENT       — a rejected-but-fast direction is logged, not just filtered away.

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
const lead = read('roles/mega/tech_lead.md');
const verify = read('roles/mega/verify_engineer.md');

// --- 1. the mode exists and is off by default ------------------------------
console.log('\n# capability_eval mode');
ok(/const CAPABILITY_EVAL = String\(A\.capability_eval/.test(wf),
   'kernel_mega.js reads args.capability_eval');
ok(/String\(A\.capability_eval \?\? 'false'\) === 'true'/.test(wf),
   "the mode defaults to OFF (an existing run's behaviour must not change under it)");
ok(/const KNOWN_REFERENCE_PATHS = /.test(wf),
   'known_reference_paths is read as the set of trees that would constitute an imported answer');
ok(/provenance check DISABLED/.test(wf),
   'turning the mode on WITHOUT reference paths says so, rather than implying protection it lacks');
ok(/byte-identical/i.test(wf), 'the source records what the failure actually looked like');

// --- 2. the asymmetry: verify learns locations, engineers never do ---------
console.log('\n# location asymmetry');
ok(/KNOWN_REFERENCE_PATHS: KNOWN_REFERENCE_PATHS\.join\(' '\)/.test(wf),
   'verify_engineer receives the reference paths');
// The whole point. If engineers ever get this list, the mode is worse than useless: it documents
// where the answer lives while claiming the run derived it.
// Engineers are not spawned through roleAgent() — they get a hand-built prompt. Anchor on the
// round out_dir assignment (where a direction becomes an engineer) through to the verify call.
const iEng = wf.indexOf('You are Engineer ${d.id}');
const iVer = wf.indexOf("roleAgent('verify_engineer'", iEng);
ok(iEng > 0 && iVer > iEng, 'the engineer dispatch block can be located ahead of verify');
const engineerBlock = wf.slice(iEng, iVer);
ok(engineerBlock.length > 0 && !engineerBlock.includes('KNOWN_REFERENCE_PATHS'),
   'the engineer role is NOT given the reference paths');
ok(/CAPABILITY_EVAL \? \{ CAPABILITY_EVAL: '1' \} : \{\}/.test(wf),
   'tech_lead is told which doctrine applies before it writes the roadmap');
ok(/CONCLUSIONS ONLY — never implementations/.test(lead),
   'tech_lead.md states the rule in one unmissable line');
ok(/MUST NOT/.test(lead) && /port, copy, cherry-pick/.test(lead),
   'porting/copying/cherry-picking are named explicitly, not left to inference');
ok(/Quote \*mechanism\* instead/.test(lead),
   'the rule says what to do INSTEAD, so the finding is not simply lost');
ok(/getting this backwards silently converts a capability run into a copy\s+exercise/.test(lead),
   'tech_lead.md carries the consequence of choosing the wrong doctrine');

// --- 3. the leak scrub -----------------------------------------------------
// Prose instructions are precisely what failed the first time, so the roadmap gets checked.
console.log('\n# leak scrub');
ok(/LEAK WARNING/.test(wf), 'engineer-visible files are scanned for the reference location');
ok(/roadmap\.md', 'codebase_context\.md', 'analysis\.json'/.test(wf),
   'the scrub covers every file an engineer reads, not just the roadmap');
ok(/delete j\.prior_art/.test(wf),
   'prior_art[].implemented_at is exempt — it is the one field allowed to hold a location');
ok(/warns instead of aborting/.test(wf),
   'the scrub is documented as advisory, with verify named as the enforcing half');

// --- 4. the provenance gate ------------------------------------------------
console.log('\n# provenance gate');
ok(/status:"plagiarized"/.test(verify), 'verify_engineer can return plagiarized');
// Matched loosely on purpose: the enum is open to further statuses (`inactive` was added by
// tests/test_activation_gate.js), and pinning the whole string made an unrelated addition fail here.
ok(/"status": "verified\|correctness_failed\|apply_failed\|regression\|harness_modified\|plagiarized/.test(verify),
   'both new statuses are in the return schema');
ok(/report the measured numbers anyway/.test(verify),
   'a rejected patch still reports its numbers (the direction may be real and re-derivable)');
ok(/an imported win is a\s+\*negative\* result/.test(verify),
   'verify is told a big speedup is not mitigating — that is the trap');
ok(/Partial overlap is\s+not plagiarism/.test(verify),
   'convergent derivation is explicitly NOT plagiarism, so the gate cannot eat honest work');
ok(/whole-file identity/.test(verify), 'the trigger is stated precisely enough to apply');
ok(/harness_modified/.test(verify) && /independent of the speedup/.test(verify),
   'editing the harness is its own rejection, decoupled from the result');

// --- 5. rejection must be audible ------------------------------------------
console.log('\n# rejection is logged, not silent');
// `verified` already filters on status === 'verified', so both new statuses were excluded the moment
// they were defined. That is the bug: a direction that vanishes reads as "it didn't work", when what
// actually happened is far more interesting and needs a human.
ok(/st !== 'plagiarized' && st !== 'harness_modified'/.test(wf),
   'both statuses are picked out for explicit logging');
ok(/REJECTED as a win despite/.test(wf),
   'the log leads with the rejection and carries the speedup that made it tempting');
ok(/imported, not derived/.test(wf), 'the log says what the finding MEANS, not just its status');
ok(/exclusion is\s*\n\/\/ invisible/.test(wf) || /exclusion is[\s\S]{0,20}invisible/.test(wf),
   'the code explains why filtering alone was insufficient');

console.log(
  failures === 0
    ? '\nPASS: capability_eval separates "port it" from "derive it", and imported wins cannot be counted.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
