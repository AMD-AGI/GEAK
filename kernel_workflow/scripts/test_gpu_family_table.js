#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

// The GPU family table is copied into kernel_workflow.js, kernel_lane.js and e2e_workflow.js (the
// Workflow sandbox has no import). This test keeps the copies identical and pins how each ISA is
// classified, so a new family is one table row plus its ISAs here, and an unlisted ISA stays on the
// Instinct (CDNA) default. Plain node, no GPU.

const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..', '..');
const FILES = ['kernel_workflow/kernel_workflow.js', 'kernel_workflow/kernel_lane.js', 'e2e_workflow/e2e_workflow.js'];
const BEGIN = '// ---- GPU families: keep this block identical';
const END = '// ---- end GPU families ----';

const blocks = FILES.map((rel) => {
  const s = fs.readFileSync(path.join(repo, rel), 'utf8');
  const i = s.indexOf(BEGIN);
  const j = s.indexOf(END, i);
  if (i < 0 || j < 0 || s.indexOf(BEGIN, i + 1) >= 0) throw new Error(`${rel}: expected one GPU family block`);
  return s.slice(i, j + END.length);
});
blocks.forEach((b, n) => {
  if (b !== blocks[0]) throw new Error(`${FILES[n]}: GPU family block differs from ${FILES[0]}`);
});

// eslint-disable-next-line no-new-func
const { gpuFamily, GPU_FAMILIES, GPU_FAMILY_DEFAULT } =
  new Function(`${blocks[0]}\nreturn { gpuFamily, GPU_FAMILIES, GPU_FAMILY_DEFAULT };`)();

// Every row has the default's fields, and an anchored ISA pattern that cannot match a longer name.
const fields = Object.keys(GPU_FAMILY_DEFAULT).sort().join(',');
for (const row of GPU_FAMILIES) {
  if (Object.keys(row).sort().join(',') !== fields) throw new Error(`row ${row.name} has fields ${Object.keys(row)}`);
  if (!(row.isa instanceof RegExp) || !/^\^.*\$$/.test(row.isa.source)) throw new Error(`row ${row.name}: isa must be anchored`);
}
// Family, isolation and matrix core, per ISA.
const policy = (f) => `${f.name} ${f.isolate ? 'isolated' : 'open'} ${f.matrixCore}`;
const CDNA = 'default open MFMA';
const R9700 = 'R9700 isolated WMMA';
const EXPECT = {
  gfx90a: CDNA, gfx942: CDNA, gfx950: CDNA, gfx1250: CDNA,
  gfx1100: CDNA, gfx1101: CDNA, gfx1102: CDNA, gfx1103: CDNA, // RDNA3 has no row: the default
  gfx1150: CDNA, gfx1151: CDNA, gfx1152: CDNA, gfx1153: CDNA,
  gfx1200: CDNA, gfx1201: R9700, gfx12010: CDNA, gfx1201x: CDNA,
  '': CDNA, unknown: CDNA,
};
for (const [gfx, want] of Object.entries(EXPECT)) {
  const got = policy(gpuFamily(gfx));
  if (got !== want) throw new Error(`${gfx || '(empty)'} gets "${got}", expected "${want}"`);
}
if (gpuFamily(undefined) !== GPU_FAMILY_DEFAULT) throw new Error('an undefined ISA must take the default');

console.log(`PASS: GPU family table identical in ${FILES.length} workflows; ${Object.keys(EXPECT).length} ISAs classified as pinned.`);
