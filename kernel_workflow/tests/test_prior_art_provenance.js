#!/usr/bin/env node
// Guards on prior-art PROVENANCE, added after a report cleared a run it had never checked.
//
// What happened: the tech_lead's return schema had a `prior_art` array from the start, and the
// orchestrator had a loud `PRIOR ART NOT IN BASELINE:` log ready to fire on it. Neither fired,
// because the model never emitted the key at all — it wrote its prior-art findings as prose in
// roadmap.md, which nothing downstream can read. `Array.isArray(...) ? ... : []` then silently
// turned "no record" into "found none", and the final report went on to state:
//
//   "all prior art identified in analyze was `in_baseline: true` … the fused-megakernel path exists
//    in the tree as an env-gated opt-in … this run was pointed at the right tree."
//
// All three clauses were false — the baseline directory contained no such file and no such env var
// — and the sentence was unfalsifiable from inside the run, because there was nothing to check it
// against. It read as a provenance clearance. What it concealed was that 8 of the winning
// candidate's 11 files had been copied in from a reference tree.
//
//   1. RECORDED vs EMPTY — an omitted key and `[]` are different findings and must not collapse.
//   2. EVIDENCE          — `in_baseline` must name the check that settled it, or be flagged.
//   3. STRUCTURED        — the contract says JSON array, and says prose is not a substitute.
//   4. REPORT CITES      — the report gets the sweep verbatim and may not assert beyond it.

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

// --- 1. an omitted key is not an empty result -------------------------------
console.log('\n# omitted != empty');
ok(/const PRIOR_ART_RECORDED = !!\(analysis && Array\.isArray\(analysis\.prior_art\)\)/.test(wf),
   'the orchestrator tracks whether the key was PRESENT, separately from its contents');
ok(/PRIOR ART UNRECORDED/.test(wf),
   'a missing prior_art key produces its own log line');
ok(/UNSOURCED/.test(wf),
   'that line says what the omission costs: every later provenance statement is unsourced');
ok(/NOT the same as finding none/.test(wf),
   'the log distinguishes "did not look" from "looked and found nothing"');
ok(/PRIOR ART: none — the sweep ran and reported an empty result/.test(wf),
   'a genuinely empty sweep is also logged, so silence never means either one');

// --- 2. in_baseline must carry its evidence ---------------------------------
console.log('\n# in_baseline is evidenced');
ok(/PRIOR ART UNEVIDENCED/.test(wf),
   'an in_baseline flag with no evidence field is logged as unverified');
ok(/pa\.in_baseline != null && !pa\.evidence/.test(wf),
   'the check fires on the flag being ASSERTED, not on it being true');
ok(/"evidence":/.test(lead) || /"evidence"/.test(lead),
   'the return-JSON contract carries an evidence field');
ok(/Write the check, not the conclusion/.test(lead),
   'the contract asks for the command and its output, not a restatement of in_baseline');
ok(/guess wearing a schema field/.test(lead),
   'the contract says why an unevidenced boolean is worse than an absent one');

// --- 3. structured output, prose is not a substitute ------------------------
console.log('\n# structured, not prose');
ok(/`prior_art` is a REQUIRED key/.test(lead),
   'prior_art is stated to be required rather than optional');
ok(/`\[\]` is a real answer that is not the same as omitting it/.test(lead),
   'the empty-vs-omitted distinction is stated to the role, not only enforced in code');
ok(/a paragraph in `roadmap\.md` is not a\s*\n?substitute/.test(lead)
   || /paragraph in `roadmap\.md` is not a[\s\S]{0,20}substitute/.test(lead),
   'the contract explicitly rejects the failure mode that actually occurred (roadmap prose)');
ok(/fires nothing and is quoted by no one/.test(lead),
   'the contract explains the mechanism: nothing downstream can read prose');

// --- 4. the report may not out-run its source -------------------------------
console.log('\n# the report cites the sweep');
ok(/PRIOR_ART_SWEEP: PRIOR_ART_RECORDED \? JSON\.stringify\(PRIOR_ART\) : 'UNRECORDED'/.test(wf),
   'the report phase receives the sweep verbatim, or the string UNRECORDED');
ok(/PRIOR_ART_SWEEP/.test(lead),
   'the report contract names the input it must quote');
ok(/Do not write one from memory/.test(lead),
   'the report is forbidden from reconstructing provenance from recollection');
ok(/the only admissible sentence is that it is not\s*\n?\s*on the record/.test(lead)
   || /only admissible sentence[\s\S]{0,60}on the record/.test(lead),
   'UNRECORDED constrains the report to saying so, rather than asserting either way');
ok(/pointed at the right tree/.test(lead),
   'the actual false sentence is quoted, so the rule is anchored to a real incident');
ok(/8 of the winning/.test(lead),
   'the contract states the consequence the false clearance concealed');

console.log(
  failures === 0
    ? '\nPASS: prior art is structured, evidenced, and the report cannot claim more than the sweep recorded.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
