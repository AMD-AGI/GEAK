#!/usr/bin/env node
// Regression guard for extractor PASS-status compatibility (no GPU or model required).
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const WORKFLOW = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');
const src = fs.readFileSync(WORKFLOW, 'utf8');
const start = src.indexOf('function isPassStatus');
const end = src.indexOf('// A FROZEN baseline', start);

if (start < 0 || end <= start) {
  console.error('FAIL: isPassStatus block not found in e2e_workflow.js');
  process.exit(1);
}

const isPassStatus = new Function(
  `${src.slice(start, end)}\nreturn isPassStatus;`,
)();

const accepted = [
  'pass', 'PASS', ' Pass ',
  { result: 'PASS', exit_code: 0 }, { status: 'pass' },
  { result: 'PASS', status: 'pass' },
];
const rejected = [
  null, undefined, false, true, 0, 1, '', 'fail', 'failed', 'passed',
  'RESULT: PASS', '1 failed, 9 passed', 'PASS with warnings',
  { result: 'FAIL' }, { result: '1 failed, 9 passed' },
  { result: false }, { status: 'unknown' }, { pass: true },
  { result: { status: 'PASS' } }, { result: 'FAIL', status: 'PASS' },
  { result: 'PASS', status: 'FAIL' }, {}, [],
];
let failures = 0;

const requiredGatePatterns = [
  /isPassStatus\(e\.smoke\) \|\| isPassStatus\(e\.unittest_smoke\)/,
  /!isPassStatus\(ext\.smoke\)/g,
  /!isPassStatus\(ext\.unittest_smoke\)/,
];

if (!requiredGatePatterns[0].test(src)) {
  console.error('FAIL: extractWithBaseline does not normalize both smoke fields');
  failures++;
}
if ((src.match(requiredGatePatterns[1]) || []).length !== 3) {
  console.error('FAIL: all three HeadKernel gates must use isPassStatus(ext.smoke)');
  failures++;
}
if (!requiredGatePatterns[2].test(src)) {
  console.error('FAIL: Milestone gate does not normalize unittest_smoke');
  failures++;
}
if (/\b(?:ext|e)\.(?:smoke|unittest_smoke)\s*[!=]==?\s*['"]pass['"]/.test(src)) {
  console.error('FAIL: legacy direct PASS comparison remains in e2e_workflow.js');
  failures++;
}

for (const value of accepted) {
  if (!isPassStatus(value)) {
    console.error('FAIL: should accept', JSON.stringify(value));
    failures++;
  }
}
for (const value of rejected) {
  if (isPassStatus(value)) {
    console.error('FAIL: should reject', JSON.stringify(value));
    failures++;
  }
}

if (failures) process.exit(1);
console.log(`PASS: ${accepted.length} accepted and ${rejected.length} rejected status forms`);
