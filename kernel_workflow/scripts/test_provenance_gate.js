#!/usr/bin/env node
// Regression guard for the MERIT/PROVENANCE split (no GPU, no model needed).
//
// THE BUG THIS LOCKS OUT. `director.md`'s timing-receipt gate had only one field to speak through,
// `validation_status`, so "we cannot prove this ratio is device time" was written as `flagged` — the
// same token used for "the patch did not install" and "correctness failed". The bake-off's eligibility
// filter reads that field as MERIT and dropped the lane. On DeepSeek-V4-Pro's `dsa_sparse_mla_attn`
// that discarded a reproduced, correctness-passing 3.73x weighted win whose ONLY defect was a task
// frozen before the receipt contract existed: `winner` fell to null, the original kernel was never
// patched, and the e2e bench never fired. Benched by hand on the same harness afterwards, the kernel
// was worth 1.89x end-to-end serving throughput (636.5 -> 1201.3 tok/s, both legs spread <0.1%).
//
// Under test, against the expressions pulled from the real sources:
//   1. `provenanceOk` / `timingBasis` classify all four bases, and legacy records (no `timing_basis`)
//      degrade to `unknown` / not-ok rather than being assumed primed.
//   2. A merit-accepted lane with unproven provenance IS eligible to win the bake-off, and carries
//      `requires_e2e_confirmation` — the dsv4 case, which is the whole point.
//   3. A merit-FLAGGED lane is still ineligible. The #411 guard must survive this change.
//   4. The KB gate stays strict on BOTH axes: merit accepted AND provenance proven.
//
// Run:  node kernel_workflow/scripts/test_provenance_gate.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const wfSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js'), 'utf8');
const laneSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), 'utf8');

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail != null ? `-- ${detail}` : ''); failures++; }
  else console.log('  ok:', msg);
};

// Pull the real expressions out of the sources rather than restating them here — a test that restates
// the predicate passes forever while the shipped one rots. Each `grab` matches a WHOLE statement.
function grab(src, re, name, where) {
  const m = src.match(re);
  if (!m) throw new Error(`could not extract ${name} from ${where}`);
  return m[0];
}

const PROV_RE = /const timingBasis = [\s\S]*?const provenanceOk = \(v\) => \{[\s\S]*?\n\};\n/;
const wfProv = grab(wfSrc, PROV_RE, 'provenance helpers', 'kernel_workflow.js');
const laneProv = grab(laneSrc, PROV_RE, 'provenance helpers', 'kernel_lane.js');

// The two copies exist only because these workflow scripts have no module system. If they drift, one
// layer starts believing a number the other rejects, which is a subtler version of the original bug.
ok(wfProv === laneProv, 'provenance helpers are byte-identical in kernel_workflow.js and kernel_lane.js');

const helpers = new Function(`${wfProv}; return { timingBasis, provenanceOk };`)();
const { timingBasis, provenanceOk } = helpers;

console.log('\n1. basis classification');
ok(timingBasis({ timing_basis: 'device_verified' }) === 'device_verified', 'device_verified passes through');
ok(timingBasis({ timing_basis: 'HOST_BOUND' }) === 'host_bound', 'basis is case-normalised');
ok(timingBasis({}) === 'unknown', 'missing basis reads as unknown');
ok(timingBasis(null) === 'unknown', 'null validation reads as unknown');
ok(provenanceOk({ timing_basis: 'device_verified' }) === true, 'device_verified => provenance ok');
for (const b of ['host_bound', 'unprimed', 'unknown']) {
  ok(provenanceOk({ timing_basis: b }) === false, `${b} => provenance NOT ok`);
}
// The dsv4 records predate the fields entirely. Absence is not evidence of priming — the same
// sentence director.md uses — so a legacy record must never be treated as device-verified.
ok(provenanceOk({ validation_status: 'accepted' }) === false,
  'legacy record with no timing fields => provenance NOT ok (absence is not evidence of priming)');
ok(provenanceOk(null) === false, 'null validation => provenance NOT ok');
// An explicit boolean from the director wins over the derived label.
ok(provenanceOk({ timing_basis: 'unknown', timing_provenance_ok: true }) === true,
  'explicit timing_provenance_ok overrides the derived label');

console.log('\n2. bake-off eligibility is MERIT, not provenance');
const eligibility = grab(wfSrc,
  /const ACCEPTED = \(c\) => [\s\S]*?const winner = ranked\[0\] \|\| null;\n/,
  'eligibility block', 'kernel_workflow.js');
// `log` is the workflow runtime's; stub it and capture what the gate says out loud.
function rank(cands) {
  const lines = [];
  const fn = new Function('cands', 'log',
    `${eligibility}\nconst winnerRequiresE2E = !!(winner && winner.kind === 'lane' && !winner.timing_provenance_ok);\n` +
    `return { winner, ranked, rejectedByGate, winnerRequiresE2E };`);
  const out = fn(cands, (m) => lines.push(m));
  return { ...out, lines };
}

// The dsv4 lane, as the run actually recorded it: weighted 3.7348x, correctness pass, director's own
// arbitration ACCEPT, receipt absent because the task predates the contract.
const dsv4 = {
  lang: 'triton', mode: 'optimize', kind: 'lane', speedup: 3.7348,
  validation_status: 'accepted', timing_basis: 'unknown', timing_provenance_ok: false,
};
let r = rank([dsv4]);
ok(r.winner === dsv4, 'merit-accepted lane with UNPROVEN provenance wins the bake-off (the dsv4 case)');
ok(r.winnerRequiresE2E === true, 'that winner is marked requires_e2e_confirmation');
ok(r.rejectedByGate.length === 0, 'and is not counted as rejected by the merit gate');

const proven = { ...dsv4, lang: 'hip', speedup: 1.4, timing_basis: 'device_verified', timing_provenance_ok: true };
r = rank([dsv4, proven]);
ok(r.winner === dsv4, 'ranking is still on speedup — provenance does not reorder candidates');
ok(r.winnerRequiresE2E === true, 'the faster unproven lane still requires e2e confirmation');
r = rank([proven]);
ok(r.winnerRequiresE2E === false, 'a device_verified winner does NOT require e2e confirmation');

console.log('\n3. the #411 merit guard survives');
const meritFlagged = { ...dsv4, validation_status: 'flagged', timing_basis: 'device_verified',
  timing_provenance_ok: true };
r = rank([meritFlagged]);
ok(r.winner === null, 'merit-FLAGGED lane is still ineligible even with perfect provenance');
ok(r.rejectedByGate.length === 1, 'and is still reported as rejected by the gate');
ok(r.lines.some((l) => /NOT eligible to win/.test(l)), 'the rejection is still logged');
// A slower-than-baseline lane must never win by default (the guard that predates #411).
r = rank([{ ...dsv4, speedup: 0.17 }]);
ok(r.winner === null, 'a lane below the frozen baseline still cannot win');
// An env-tune candidate carries no director validation and is exempt from the lane merit gate.
r = rank([{ lang: 'aiter', mode: 'env-tune', kind: 'env', speedup: 1.2, validation_status: 'env' }]);
ok(r.winner && r.winner.kind === 'env', 'env-tune candidates remain eligible');

console.log('\n4. KB curation stays strict on BOTH axes');
// The bake-off's own card gate.
ok(/if \(winner && winner\.speedup > 1\.0 && !winnerRequiresE2E\) \{/.test(wfSrc),
  'bake-off distils a learned card only when the winner needs no e2e confirmation');
// The lane's card gate.
ok(/const kbProvenanceOk = provenanceOk\(validation\);/.test(laneSrc),
  'kernel_lane computes kbProvenanceOk');
ok(/UPDATE_EXPERIENCE_ON && kbAccepted && kbProvenanceOk/.test(laneSrc),
  'kernel_lane requires merit AND provenance before curating a card');
// And the lane hands both axes upward.
for (const f of ['timing_basis:', 'timing_provenance_ok:', 'requires_e2e_confirmation:']) {
  ok(laneSrc.includes(f), `kernel_lane returns ${f.slice(0, -1)}`);
}
// The schema must carry them or the director's answer is dropped before any of this runs.
const vs = grab(laneSrc, /const VALIDATE_SCHEMA = obj\(\{[\s\S]*?\n\}, \[[^\]]*\]\);\n/,
  'VALIDATE_SCHEMA', 'kernel_lane.js');
for (const f of ['timing_basis', 'timing_provenance_ok', 'requires_e2e_confirmation', 'timing_receipt']) {
  ok(vs.includes(f), `VALIDATE_SCHEMA declares ${f}`);
}

console.log('\n5. the director prompt no longer lets provenance write the merit field');
const dm = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'roles', 'director.md'), 'utf8');
const gate = grab(dm, /\*\*TIMING RECEIPT GATE[\s\S]*?\n7\. If `APPLY_TO_ORIGINAL=true`/, 'receipt gate', 'director.md');
ok(/NEVER sets `validation_status`/.test(gate), 'the gate states it never sets validation_status');
ok(!/`status: "flagged"`/.test(gate), 'no branch of the receipt gate assigns status: "flagged"');
for (const f of ['timing_provenance_ok', 'requires_e2e_confirmation', 'device_verified']) {
  ok(gate.includes(f), `the gate defines ${f}`);
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
