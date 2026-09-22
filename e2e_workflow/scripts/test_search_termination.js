// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Verify the actual workflow snapshot, including timers firing during finalization.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const src = fs.readFileSync(path.join(__dirname, '..', 'e2e_workflow.js'), 'utf8');
const snapshot = src.match(/const searchTermination = \{[\s\S]*?\n\};/)[0];
const capture = new Function('cutoff', 'fast', 'deep', `
  let TIME_DEADLINE_HIT = cutoff;
  const FAST_MODE = fast, FAST_DEADLINE_HIT = fast;
  const DEEP_MODE = deep, DEEP_DEADLINE_HIT = deep;
  const TIME_BUDGET_MS = 14282000, TIME_HEAD_DEADLINE_MS = 6855360;
  const FAST_HEAD_DEADLINE_MS = 6000000, DEEP_HEAD_BUDGET_MS = 6500000;
  const ELAPSED_MS = 8880000;
  const remainingMs = () => TIME_BUDGET_MS - ELAPSED_MS;
  ${snapshot}
  TIME_DEADLINE_HIT = true;
  return searchTermination;
`);
assert.equal(capture(false, false, false).reason, 'completed');
for (const flags of [[true, false, false], [false, true, false], [false, false, true]]) {
  const outcome = capture(...flags);
  assert.equal(outcome.reason, 'dispatch_cutoff');
  assert.equal(outcome.remaining_s, 5402);
  assert.equal(outcome.budget_s, 14282);
  assert.equal(outcome.dispatch_cutoff_s, flags[1] ? 6000 : flags[2] ? 6500 : 6855.36);
}
assert(src.indexOf('const searchTermination =') < src.indexOf("phase('Finalize')", src.indexOf('const searchTermination =')));
assert(src.includes('search_termination: searchTermination'));
console.log('Search termination contract passed');
