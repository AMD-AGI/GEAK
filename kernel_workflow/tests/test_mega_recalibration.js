#!/usr/bin/env node
'use strict';

// Regression tests for a run where a complete control that failed the gate on one null outlier
// kept calibration pending for the whole wave (recover never re-measured), and a slow planner nulled
// by a 5 min guard ended the rounds with 10 of 24 directions unspent.

const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_mega.js'), 'utf8');
const role = fs.readFileSync(path.resolve(__dirname, '..', 'roles', 'mega', 'benchmark_engineer.md'), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const fnBody = (name) => {
  const start = src.indexOf(`async function ${name}(`);
  if (start < 0) throw new Error(`missing ${name}`);
  return src.slice(start, src.indexOf('\n}\n', start));
};

const remeasure = fnBody('remeasureMegaCalibration');
ok(/megaPcRemeasured >= 2/.test(remeasure), 're-measurement is bounded to 2 attempts per wave');
ok(/prior\.claim_complete !== true/.test(remeasure),
  'only a COMPLETE failed control is re-measured (partial claims stay recover-only)');
ok(/megaCalibrationClaimPass\(pc, POSITIVE_CONTROL\)/.test(remeasure),
  'the re-measured control is judged by the unchanged calibration gate');
ok(/'benchmark_engineer', 'recalibrate'/.test(remeasure), 'uses the recalibrate role phase');
ok(/remeasureMegaCalibration\(label, pc\)/.test(fnBody('recoverMegaCalibration')),
  'recovery falls through to re-measurement when the recovered control fails');
ok(/remeasureMegaCalibration\('after-setup', benchR\.positive_control\)/.test(src),
  'a failed setup control is re-measured before rounds start');
ok(/## PHASE=recalibrate/.test(role) && /never reuse, drop, or cherry-pick/i.test(role) &&
  /may NOT change the gate/.test(role), 'role forbids re-gating or pair selection');

// The gate itself is not loosened: one outlier null still fails a 7.65% control.
const gate = new Function(`${src.match(/function megaCalibrationClaimPass[\s\S]*?\n}\n/)[0]}
  return megaCalibrationClaimPass;`)();
const cfg = { expected_pct_lo: -9, expected_pct_hi: -4, required_control_pairs: 5,
  required_null_pairs: 8, implausible_pct: 25 };
const base = { claim_complete: true, ran: true, switch_present: true, attempt_id: 'a',
  evidence_manifest: 'm', passed: true, measured_pct: -7.65,
  control_pairs_pct: [-7.06, -7.5, -8.06, -10.3, -7.77, -7.97, -5.95, -7.54] };
ok(!gate({ ...base, null_pairs_pct: [-1.38, 1.53, 0.81, -2.23, -5.81, 0.17, 0.39, -0.39] }, cfg),
  'the recorded control (worst null 5.81%) still fails the 3x-worst-null gate');
ok(gate({ ...base, null_pairs_pct: [-1.38, 1.53, 0.81, -2.23, -1.1, 0.17, 0.39, -0.39] }, cfg),
  'the same control passes on a quiet null set');

const plan = fnBody('planMegaCandidateTurn');
ok(/timeout_ms: 900000/.test(plan), 'planner guard is 15 min (late plans ran 6-7 min)');
ok(/noPlan && !wipDir && !plan \? selectMegaSearchParent/.test(plan),
  'a null (hung) plan continues the best scored lane; a deliberate stop does not');

// Resume: compacted rows that lost `source` must refuse the resume, not silently run fresh.
ok(/megaRestoreRejected\.push\(/.test(src) && /MEGA RESUME REFUSED: setup returned invalid candidate_registry rows/.test(src),
  'a resume with rejected registry rows is refused');
const director = fs.readFileSync(path.join(__dirname, '..', 'roles', 'mega', 'director.md'), 'utf8');
ok(/every row MUST keep, copied exactly:[\s\S]{0,80}`id`, `source`, `base_id`, `tree`, `head`/.test(director),
  'director keeps lane identity fields when compacting registry rows');
ok(/Analyze'[\s\S]{0,600}timeout_ms: 1800000/.test(src), 'Analyze guard is 30 min');
ok(/label: 'director:setup'[\s\S]{0,400}timeout_ms: 1800000/.test(src), 'mega setup guard is 30 min');
ok(!/mega_knowledge|MEGA_KNOWLEDGE/.test(src),
  'the generic line derives its plan from the mechanism; no argument injects knowledge files');
ok(/writer_quiescent === true\) \{\s*laneWriterTimedOut = false;\s*megaUnsafeTimeout = null;[\s\S]{0,300}\}\s*if \(recovered && recovered\.claim_complete === true\)/.test(src),
  'a quiescent recovered author clears megaUnsafeTimeout even when its result is complete');

if (failures) { console.error(`FAILED: ${failures} assertion(s).`); process.exit(1); }
console.log('all mega recalibration/plan-fallback checks passed');
