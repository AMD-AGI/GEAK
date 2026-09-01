#!/usr/bin/env node
// Replay of the REAL discarded win, through the real bake-off code (no GPU, no model, no network).
//
// `test_provenance_gate.js` proves the new predicates behave. This proves the specific incident does
// not recur, using the actual artefact the incident produced:
//
//   tests/fixtures/dsv4_dsa_sparse_mla_attn_director_validation.json
//
// which is a byte copy of `director_validation.json` from DeepSeek-V4-Pro's `dsa_sparse_mla_attn`
// task (arena run 20260830T005045Z-433821d2, e2e_cycle0, team_main_kernel_task_20260831_010623).
// The director reproduced the win twice under its own gpu_lock, agreed with the tech lead's number
// to +0.02%, confirmed correctness, confirmed the frozen baseline denominators had not drifted —
// and then wrote `validation_status: "flagged"`, because the task's harness predates the timing
// receipt and `validation_status` was the only field the receipt gate had to speak through.
//
// That one token cost the run everything downstream: the lane was dropped from `ranked`, `winner`
// fell to null, `applied_to_original` stayed false, the e2e bench never fired, and the campaign
// closed at `recovered_no_gain` speedup 1.0. Benched by hand afterwards on GEAK's own
// `bench_e2e.sh` — same server flags, workload, warmups and seed on both legs, the overlay the only
// variable — that kernel was worth 636.492 -> 1201.335 tok/s, i.e. 1.887x end-to-end serving
// throughput (both legs spread < 0.1% over 3 runs).
//
// The replay runs the recorded facts through the SHIPPED eligibility block, extracted from
// `kernel_workflow.js` rather than restated here, on two legs:
//
//   LEG A  the record exactly as it was written  -> reproduces the loss (winner === null)
//   LEG B  the same facts re-labelled the way the FIXED `director.md` mandates
//          -> the lane wins, carrying requires_e2e_confirmation, and is still barred from the KB
//
// Leg B's relabelling is not a guess: `relabel()` implements the four rules of the fixed gate, and
// each rule is asserted to be present in `roles/director.md` before it is applied. The prompt is the
// thing that changed for this record; the code change is what makes the prompt's answer survive.
//
// Run:  node kernel_workflow/scripts/test_dsv4_receipt_replay.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const wfSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js'), 'utf8');
const laneSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'), 'utf8');
const directorMd = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'roles', 'director.md'), 'utf8');
const REC = JSON.parse(fs.readFileSync(
  path.join(__dirname, 'tests', 'fixtures', 'dsv4_dsa_sparse_mla_attn_director_validation.json'), 'utf8'));

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail != null ? `-- ${detail}` : ''); failures++; }
  else console.log('  ok:', msg);
};

function grab(src, re, name, where) {
  const m = src.match(re);
  if (!m) throw new Error(`could not extract ${name} from ${where}`);
  return m[0];
}

// ---------------------------------------------------------------------------
// 0. The recorded facts. If the fixture ever stops saying these things it is no longer the incident,
//    and the rest of this file is testing a story instead of a record.
// ---------------------------------------------------------------------------
console.log('\n0. the recorded artefact still describes the incident');
ok(REC.correctness === 'pass', 'correctness passed', REC.correctness);
ok(Math.abs(REC.director_verified_speedup_weighted - 3.7348) < 1e-4,
  'director re-measured 3.7348x weighted', REC.director_verified_speedup_weighted);
ok((REC.director_verified_speedup_weighted_own_baselines || []).length === 2,
  'reproduced twice against the director\'s own baselines',
  JSON.stringify(REC.director_verified_speedup_weighted_own_baselines));
ok(Math.abs(REC.director_verified_speedup_weighted - REC.tech_lead_reported_speedup_weighted)
   / REC.tech_lead_reported_speedup_weighted < 0.001,
  'director and tech lead agree on the primary metric to within 0.1%');
ok(REC.timing_receipt === null || REC.timing_receipt === undefined,
  'the task emitted no timing receipt', String(REC.timing_receipt));
ok(String(REC.timing_basis || 'unknown').toLowerCase() === 'unknown',
  'so the basis is unknown, not device-verified', REC.timing_basis);
ok(REC.validation_status === 'flagged',
  'and the merit field was nevertheless written as flagged', REC.validation_status);
ok(/SOLELY BY THE TIMING-RECEIPT GATE/i.test(REC.arbitration_note || ''),
  'the director says in its own words that the flag is the receipt gate and nothing else');
ok(/NO corrective round is warranted/i.test(REC.arbitration_note || ''),
  'and that no merit condition failed');

// ---------------------------------------------------------------------------
// 1. The fixed gate's four rules, read out of the prompt. Leg B applies exactly these.
// ---------------------------------------------------------------------------
console.log('\n1. the fixed director prompt states the rules leg B will apply');
const gate = grab(directorMd, /\*\*TIMING RECEIPT GATE[\s\S]*?\n7\. If `APPLY_TO_ORIGINAL/, 'receipt gate', 'director.md');
ok(/This gate sets `timing_basis`\. It NEVER sets `validation_status`/.test(gate),
  'rule 1: the gate writes provenance, never merit');
ok(/Receipt ABSENT entirely[^\n]*`timing_basis: "unknown"`/.test(gate),
  'rule 2: an absent receipt means basis unknown');
ok(/`timing_provenance_ok`: `true` only when `timing_basis == "device_verified"`/.test(gate),
  'rule 3: provenance_ok iff device_verified');
ok(/`requires_e2e_confirmation`: `true` whenever `timing_provenance_ok` is `false`/.test(gate),
  'rule 4: unproven provenance requires an e2e confirmation');
ok(/eligible to win the bake-off and MUST be carried to the e2e/.test(gate),
  'and the prompt states the consequence the code must honour');
ok(/\*\*not\*\* eligible for KB curation/.test(gate),
  'while keeping the strict requirement on KB curation');

// Rule 1 means the merit verdict is whatever the merit arbitration said. The record's own
// arbitration_note settles that: reproduced, correctness pass, no corrective round warranted.
const relabel = (rec) => {
  const basis = (rec.timing_receipt == null) ? 'unknown' : String(rec.timing_basis || 'unknown');
  const provOk = basis === 'device_verified';
  return {
    ...rec,
    validation_status: 'accepted',   // rule 1: merit, decided by the arbitration, not by the receipt
    timing_basis: basis,             // rule 2
    timing_provenance_ok: provOk,    // rule 3
    requires_e2e_confirmation: !provOk, // rule 4
  };
};

// ---------------------------------------------------------------------------
// 2. The shipped eligibility block, lifted verbatim. A test that restates the filter passes forever
//    while the real one rots.
// ---------------------------------------------------------------------------
const provHelpers = grab(wfSrc, /const timingBasis = [\s\S]*?const provenanceOk = \(v\) => \{[\s\S]*?\n\};\n/,
  'provenance helpers', 'kernel_workflow.js');
const eligibility = grab(wfSrc,
  /const ACCEPTED = \(c\) =>[\s\S]*?const winnerRequiresE2E = [^\n]*\n/, 'eligibility block', 'kernel_workflow.js');
ok(/validation_status \|\| ''\)\.toLowerCase\(\) === 'accepted'/.test(eligibility),
  'the extracted block is still the merit filter (#411 guard intact)');
ok(/!winner\.timing_provenance_ok/.test(eligibility),
  'and still derives requires_e2e_confirmation from provenance alone');

// `cands` is built one line earlier in the real source; mirror only the field mapping, which is the
// part under test's control, and let the extracted filter do the deciding.
const bakeoff = new Function('rec', 'log', `
  ${provHelpers}
  const x = { r: rec, speedup: rec.director_verified_speedup_weighted };
  const cands = [{
    lang: 'triton', mode: 'author', kind: 'lane', speedup: x.speedup,
    validation_status: x.r.validation_status,
    timing_basis: timingBasis(x.r),
    timing_provenance_ok: provenanceOk(x.r),
    eval_dir: '', patch: x.r.final_patch, apply_env: '', tuning_artifact: '',
  }];
  ${eligibility}
  return { winner, winnerRequiresE2E, rejectedByGate };
`);

// ---------------------------------------------------------------------------
// 3. LEG A — the record as written. This must still reproduce the loss; if it does not, the fixture
//    or the extraction has drifted and leg B proves nothing.
// ---------------------------------------------------------------------------
console.log('\n3. LEG A: the record exactly as the old director wrote it');
const legA = bakeoff(REC, () => {});
ok(legA.winner === null, 'winner is null — the 3.73x lane is dropped', JSON.stringify(legA.winner));
ok(legA.rejectedByGate.length === 1 && legA.rejectedByGate[0].validation_status === 'flagged',
  'and it is dropped by the merit filter, reading a provenance verdict');

// ---------------------------------------------------------------------------
// 4. LEG B — the same facts, labelled by the fixed gate.
// ---------------------------------------------------------------------------
console.log('\n4. LEG B: the same facts, relabelled per the fixed gate');
const fixed = relabel(REC);
ok(fixed.validation_status === 'accepted', 'merit: accepted (nothing about the measurement changed)');
ok(fixed.timing_basis === 'unknown', 'provenance: still unknown — the fix does not pretend otherwise');
ok(fixed.timing_provenance_ok === false, 'provenance is NOT claimed ok');
const legB = bakeoff(fixed, () => {});
ok(legB.winner !== null, 'winner is the lane', JSON.stringify(legB.winner));
ok(legB.winner && Math.abs(legB.winner.speedup - 3.7348) < 1e-4,
  'carrying the 3.7348x it actually measured', legB.winner && legB.winner.speedup);
ok(legB.winnerRequiresE2E === true,
  'and flagged requires_e2e_confirmation — it wins a measurement, not a conclusion');
ok(legB.rejectedByGate.length === 0, 'nothing is dropped by the merit filter');

// The #411 guard has to survive all of this: a lane flagged for a REAL merit failure stays out even
// with perfect provenance. Otherwise the fix has simply removed the gate.
console.log('\n5. the merit guard still bites when merit is what failed');
const meritFail = { ...REC, correctness: 'fail', validation_status: 'flagged',
  timing_basis: 'device_verified', timing_provenance_ok: true, requires_e2e_confirmation: false };
const legC = bakeoff(meritFail, () => {});
ok(legC.winner === null, 'a genuinely flagged lane is still ineligible even when device_verified');

// ---------------------------------------------------------------------------
// 6. Winning must not be laundered into KB fact. The kernel's e2e number was never confirmed by the
//    pipeline itself (the hand A/B is evidence for this fix, not a run artefact), so the card gate
//    must stay shut on exactly this record.
// ---------------------------------------------------------------------------
console.log('\n6. the KB stays shut on an unconfirmed number');
ok(/if \(winner && winnerRequiresE2E\) \{/.test(wfSrc),
  'bake-off refuses to distil a card when the winner needs e2e confirmation');
ok(/kbAccepted && kbProvenanceOk/.test(laneSrc),
  'and the lane requires merit AND provenance before curating');
const laneProvOk = new Function(
  grab(laneSrc, /const timingBasis = [\s\S]*?const provenanceOk = \(v\) => \{[\s\S]*?\n\};\n/,
    'provenance helpers', 'kernel_lane.js') + '; return provenanceOk;')();
ok(laneProvOk(fixed) === false, 'so this record, even relabelled, earns no card');

console.log(failures
  ? `\nFAIL: ${failures} check(s) failed`
  : '\nPASS: the discarded dsa_sparse_mla_attn win is recovered by the fix, and only by the fix — '
    + 'it wins the bake-off owing an e2e confirmation, and owing the KB nothing.');
process.exit(failures ? 1 : 0);
