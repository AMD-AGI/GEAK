#!/usr/bin/env node
// Copyright (c) [2026] Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Regression guard for the fast_mode / max_rounds args of kernel_lane.js (no GPU, no model needed).
//
// Invariant under test: with fast_mode off and max_rounds unset, nothing changes -- the engineer
// prompt gets no extra text, the round loop has no extra bound, and the final re-profile / memory
// update still run. With fast_mode on, the engineer time box is rendered with the requested minutes.
// The real definitions are extracted from kernel_lane.js and evaluated, the same way
// e2e_workflow/scripts/test_expert_skills_off_identical.js checks expertSkillsBlock.
//
// Run:  node kernel_workflow/scripts/test_fast_mode_off_identical.js
'use strict';
const fs = require('fs');
const path = require('path');

const FILE = path.resolve(__dirname, '..', 'kernel_lane.js');
const src = fs.readFileSync(FILE, 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

// 1) Extract the arg parsing + fastModeEngineerBlock (contiguous, MAX_ROUNDS first, function last).
const start = src.indexOf('const MAX_ROUNDS = (() => {');
const fn = src.match(/function fastModeEngineerBlock\(outDir\) \{[\s\S]*?\n\}/);
ok(start >= 0 && fn && fn.index > start, 'found MAX_ROUNDS ... fastModeEngineerBlock in kernel_lane.js');
if (failures) process.exit(1);
const defs = src.slice(start, fn.index + fn[0].length);
// eslint-disable-next-line no-new-func
const load = (A) => new Function('A', `${defs}\nreturn { MAX_ROUNDS, FAST_MODE, ENGINEER_MINUTES, fastModeEngineerBlock };`)(A);

console.log('\n# defaults / off');
for (const A of [{}, { fast_mode: 'false' }, { fast_mode: false, max_rounds: 0 }, { max_rounds: 'junk' }]) {
  const m = load(A);
  ok(m.FAST_MODE === false, `FAST_MODE is false for ${JSON.stringify(A)}`);
  ok(m.fastModeEngineerBlock('/E/round_1/engineer_0') === '', `engineer block is '' for ${JSON.stringify(A)}`);
  ok(m.MAX_ROUNDS === Infinity, `MAX_ROUNDS is unbounded for ${JSON.stringify(A)}`);
}

console.log('\n# on');
const on = load({ fast_mode: 'TRUE', engineer_minutes: '4', max_rounds: '1' });
const block = on.fastModeEngineerBlock('/E/round_1/engineer_0');
ok(on.FAST_MODE === true && on.MAX_ROUNDS === 1 && on.ENGINEER_MINUTES === 4, 'args parsed (fast_mode, max_rounds=1, engineer_minutes=4)');
ok(block.includes('HARD TIME BOX'), 'engineer block rendered');
ok(block.includes('>= 240 (4 min)'), 'time box uses engineer_minutes');
ok(block.includes('/E/round_1/engineer_0/.t0') && block.includes('/E/round_1/engineer_0/best_patch.diff'),
  'block points at the engineer output dir');
ok(load({ fast_mode: 'true' }).fastModeEngineerBlock('/x').includes('>= 360 (6 min)'), 'engineer_minutes defaults to 6');

console.log('\n# wiring');
ok(src.includes('while (!skipLoop && dispatched < BUDGET && noImprove < MAX_NO_IMPROVE && round < MAX_ROUNDS) {'),
  'round loop is bounded by MAX_ROUNDS');
ok((src.match(/FAST_MODE && \(dispatched >= BUDGET \|\| round >= MAX_ROUNDS\)/g) || []).length === 2,
  'final re-profile and memory update are skipped only under FAST_MODE on the last round');
ok(src.includes("expertSkillsBlock(isDeep ? 'deep_engineer' : 'engineer') + fastModeEngineerBlock(d.out_dir),"),
  'engineer prompt appends fastModeEngineerBlock (empty when off)');

console.log(failures ? `\n${failures} failure(s)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
