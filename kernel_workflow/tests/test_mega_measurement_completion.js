#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_mega.js'), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# force-emitted control claims are pending, never zero-valued failures');
ok(/const pcComplete = MODE !== 'mega' \|\| pc\.claim_complete === true/.test(src),
  'Mega requires an atomic completed positive-control claim');
ok(/Positive control PENDING: claim_complete is not true/.test(src),
  'the incomplete state is surfaced explicitly');
ok(/candidate authoring may continue, but mega performance ranking/i.test(src),
  'an incomplete control blocks ranking rather than candidate authoring');

console.log('\n# the null-only bypass is gone');
ok(!/measure_null_only/.test(src),
  'the orchestrator no longer has a null-only positive-control bypass');
ok(/const POSITIVE_CONTROL = \(A\.positive_control/.test(src),
  'the caller supplies the control instead of a packaged operator task');

console.log('\n# scores remain disabled until calibration passes');
ok(/megaMeasurementCalibration\.ready\s*\?\s*selectMegaCandidate/.test(src),
  'candidate ranking is conditional on completed calibration');
ok(/if \(!batch\.length \|\| !megaMeasurementCalibration\.ready\) return null/.test(src),
  'finalist selection cannot run on an uncalibrated harness');

console.log('\n# verification is tiered');
const verifyRole = fs.readFileSync(path.resolve(__dirname, '..', 'roles', 'mega', 'verify_engineer.md'), 'utf8');
ok(/`score`:[\s\S]*supplied target guard[\s\S]*short liveness/.test(verifyRole),
  'score tier is the cheap target-only admission measurement');
ok(/`finalist`:[\s\S]*all target\/regression guards[\s\S]*overlap\/attribution/.test(verifyRole),
  'expensive full evidence is reserved for finalists');

console.log(failures === 0
  ? '\nPASS: incomplete measurements preserve work but cannot rank or ship a candidate.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
