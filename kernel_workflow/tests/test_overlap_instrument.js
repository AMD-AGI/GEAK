#!/usr/bin/env node
// "Genuine compute/communication overlap, not serialization disguised by kernel boundaries" is an
// acceptance criterion that had no instrument behind it for several waves. The task text said a
// latency win with no measured overlap is "suspicious, not accepted", but there was no way to
// produce the measurement it demanded, so the honest answer was always "not measured" and the rule
// never bit once.
//
// The reason it is hard is that the fused shape blinds every instrument that would judge it: four
// launches give four trace records and settle overlap for free, one fused kernel gives one record,
// and the two per-stage timers both RISE when a stage is allowed to start early. So the only thing
// left pointing the right way is latency — and latency is precisely the quantity that cannot tell
// overlap apart from removed launch overhead or better L2 residency.
//
// This gate cannot manufacture the measurement. What it must do is stop "nobody measured it" from
// arriving in the report looking like "measured and fine", and stop an uncontrolled meter from being
// quoted as evidence. Every assertion below is a way that could silently go wrong:
//
//   - a missing field defaulting to the benign outcome (defect class B — the schema is
//     additionalProperties:true, so an omitted or renamed `overlap` validates)
//   - a meter nobody sanity-checked against the known-zero path
//   - a meter that reads 0 because it is dead, which looks identical to a conservative one
//   - concurrency with no latency win reported as partial success rather than as contention
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const m = src.match(/\/\/ <<REPLAY:overlap_gate>>([\s\S]*?)\/\/ <<\/REPLAY:overlap_gate>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:overlap_gate>> region — nothing to test'); process.exit(1); }
const { overlapVerdict } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { overlapVerdict };`)();

const WON = { geomean: 1.06 };
const LOST = { geomean: 0.99 };
// A fully controlled, believable measurement: meter reads ~0 on the unfused path and high on
// deliberately constructed concurrency.
const good = (extra) => ({ overlap: { measured: 'yes', fraction: 0.42, cu_fraction: 0.31,
  base_reading: 0.004, forced_reading: 0.88, ...extra } });

console.log('\n# an absent measurement never reads as a passed one');
{
  const v = overlapVerdict({}, WON);
  ok(v.state === 'unmeasured', 'a candidate with no overlap block at all is UNMEASURED, not measured-fine');
  ok(/NOT measured/.test(v.caveat) && /launch overhead|L2/.test(v.caveat),
     'and the caveat says why the latency win cannot stand in for the measurement');
}
{
  ok(overlapVerdict({ overlap: { measured: 'maybe', fraction: 0.9 } }, WON).state === 'unmeasured',
     'a value outside the enum falls to unmeasured rather than being read as a yes');
  ok(overlapVerdict({ overlap: { fraction: 0.9, cu_fraction: 0.8 } }, WON).state === 'unmeasured',
     'numbers with no `measured` verdict are not a measurement — an omitted field cannot be the ' +
     'benign answer');
  ok(overlapVerdict(null, WON).state === 'unmeasured' && overlapVerdict(undefined, LOST).state === 'unmeasured',
     'a missing verification object does not throw and does not pass');
}
{
  const v = overlapVerdict({}, LOST);
  ok(v.state === 'unmeasured' && !/improves latency/.test(v.caveat),
     'an unmeasured candidate that also lost gets the same state but is not accused of an overlap claim');
}

console.log('\n# the meter has to have read a known value before anyone believes it');
{
  const v = overlapVerdict({ overlap: { measured: 'yes', fraction: 0.42, base_reading: 0.37,
                                        forced_reading: 0.9 } }, WON);
  ok(v.state === 'meter_broken',
     'a meter that finds 37% overlap on the SCATTERED path, whose true overlap is zero by ' +
     'construction, is measuring its own artefacts');
  ok(/UNMEASURED/.test(v.caveat),
     'and the fused reading it produced is withdrawn, not reported alongside a warning');
}
{
  ok(overlapVerdict({ overlap: { measured: 'yes', fraction: 0.42, forced_reading: 0.9 } }, WON).state
       === 'meter_unvalidated',
     'a fraction with no negative control is an untested instrument — the same rule the workflow ' +
     'applies to a benchmark with no positive control');
  ok(overlapVerdict({ overlap: { measured: 'no', fraction: 0.0, base_reading: 0.002 } }, WON).state
       === 'meter_unvalidated',
     'a meter reporting NO overlap that has never read a known non-zero is indistinguishable from a ' +
     'dead one, so the negative finding is unconfirmed');
  ok(overlapVerdict({ overlap: { measured: 'yes', fraction: 0.42, base_reading: 0.05,
                                 forced_reading: 0.9 } }, WON).state !== 'meter_broken',
     'slop right at the threshold is tolerated — the control is a sanity check, not a precision claim');
}

console.log('\n# a controlled meter is allowed to deliver a clean result');
{
  const v = overlapVerdict(good(), WON);
  ok(v.state === 'measured' && v.caveat === '',
     'overlap measured, both controls read correctly, and the operator got faster: no caveat, which ' +
     'is what makes the caveats elsewhere mean something');
}
{
  const v = overlapVerdict({ overlap: { measured: 'no', fraction: 0.0, base_reading: 0.003,
                                        forced_reading: 0.85 } }, LOST);
  ok(v.state === 'measured_none' && /not a missing measurement/.test(v.caveat),
     '"the meter ran and there is no overlap on this edge" is a RESULT about the edge and must not ' +
     'collapse into "unknown" — they are the two answers the report most needs to tell apart');
}

console.log('\n# the number and the latency have to agree, and disagreement is a finding');
{
  const v = overlapVerdict(good(), LOST);
  ok(v.state === 'contention' && /CONTENTION/.test(v.caveat),
     '42% overlap with no latency win is two roles fighting for the same CUs — a finding about the ' +
     'partition, not a partial success to build on');
}
{
  const v = overlapVerdict(good({ fraction: 0.01 }), WON);
  ok(v.state === 'win_without_overlap' && /mechanism|launch overhead/.test(v.caveat),
     'a win with ~no overlap is probably real and is NOT an overlap result — unattributed, the next ' +
     'round builds on the wrong cause');
  ok(overlapVerdict(good({ fraction: undefined }), WON).state === 'win_without_overlap',
     'and "measured: yes" with the fraction left out is treated the same way, not as a high reading');
}

console.log('\n# the gate never fails a candidate');
{
  const states = [overlapVerdict({}, WON), overlapVerdict(good(), LOST),
                  overlapVerdict(good({ base_reading: 0.9 }), WON)];
  ok(states.every((s) => s.state && !/^(fail|reject)/.test(s.state)),
     'every outcome is a caveat, never a rejection — downgrading a real win with a named hole would ' +
     'teach the loop to stop naming the hole');
  ok(states.every((s) => typeof s.caveat === 'string' && s.caveat.length > 40),
     'and every non-clean outcome carries prose the report can print verbatim, so the hole reaches ' +
     'the human rather than a state string nobody expands');
}

console.log('\n# the instrument spec exists and says what the gate assumes');
{
  const doc = path.join(WF, 'knowledge', 'overlap_instrument.md');
  ok(fs.existsSync(doc), 'knowledge/overlap_instrument.md is present — the gate judges against it');
  const t = fs.readFileSync(doc, 'utf8');
  ok(/scattered/i.test(t) && /positive control|forced/i.test(t),
     'and it prescribes both controls the gate checks for');
  ok(/cu_fraction|CU-weighted|CU-time/i.test(t),
     'including the CU-weighted number, since wall-clock overlap is manufacturable by one lingering ' +
     'workgroup');
}

console.log(failures === 0
  ? '\nPASS: unmeasured overlap stays visibly unmeasured and an uncontrolled meter is not evidence.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
