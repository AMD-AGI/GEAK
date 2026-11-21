#!/usr/bin/env node
// Regression guard for the AgentX milestone loop retrying a kernel lane that errored (GEAK #606).
//
// In #606 every lane threw before it measured anything, the prepared-UT queue emptied, and the run
// entered Finalize with ~1315 of its 2160 min left. Pinned here, structurally on the real source:
//
//   1. A lane that threw (kl.ran === false) is re-queued once, flagged lane_retried, and skips the
//      "no speedup" dead_end; a second error on the retry falls through to the old path.
//   2. Retries go behind the untried prepared UTs, so they only use time left over.
//   3. A milestone whose every lane errored does not count toward noImprove.
//   4. The retried lane's task differs, so it never replays the failed attempt's cached agents.
//
// Run:  node e2e_workflow/scripts/test_milestone_lane_retry.js
'use strict';
const fs = require('fs');
const path = require('path');

const SRC = fs.readFileSync(path.resolve(__dirname, '..', 'e2e_workflow.js'), 'utf8');
let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const start = SRC.indexOf('// PHASE: Milestone loop');
const end = SRC.indexOf('// PHASE: Finalize + Report + Validate', start);
ok(start !== -1 && end > start, 'located the milestone loop');
if (failures) process.exit(1);
const loop = SRC.slice(start, end);

const retryAt = loop.indexOf('if (KERNEL_EXTRACT_ON && kl && kl.ran === false && !c.lane_retried) {');
const deadEndAt = loop.indexOf('kernel layer produced no speedup');
ok(retryAt !== -1 && retryAt < deadEndAt, 'an errored lane is caught before the no-speedup dead_end');
ok(/laneRetry\.push\(\{ \.\.\.c, lane_retried: true \}\);[\s\S]{0,400}?continue;/.test(loop.slice(retryAt)),
  'it is re-queued once, flagged lane_retried, and skips integrate');

const concatAt = loop.indexOf('if (laneRetry.length) kernelQueue = kernelQueue.concat(laneRetry);');
const integrateEnd = loop.indexOf('// --- (c) If improved: re-profile');
ok(concatAt !== -1 && concatAt < integrateEnd && concatAt > retryAt,
  'retries are appended behind the untried prepared UTs after the integrate loop');
ok(/const laneRetry = \[\];/.test(loop), 'laneRetry is per milestone');

ok(/const allLanesErrored = laneRetry\.length > 0 && laneRetry\.length === optimized\.filter\(Boolean\)\.length;/.test(loop) &&
  /\} else if \(!allLanesErrored\) \{\s*noImprove\+\+;/.test(loop),
  'a milestone in which every lane errored does not count toward noImprove');

ok(/\(c\.lane_retried \? ' \(Retry after a kernel-lane error\.\)' : ''\)/.test(loop),
  'the retried lane gets a distinct task, so cached agents of the failed attempt are not replayed');

if (failures) { console.error(`\n${failures} check(s) FAILED`); process.exit(1); }
console.log('\nall green');
