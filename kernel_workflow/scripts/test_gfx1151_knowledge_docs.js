#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

// Doc contract for gfx1151 (RDNA 3.5 / Strix Halo). The roles route gfx11* to
// knowledge/amd_ryzen.md; that routing is only worth anything if the file still
// states the facts it is read for, and if its pointer to the long form resolves.
// Text only: no GPU, no npm install.

const fs = require('fs');
const path = require('path');

const wfDir = path.resolve(__dirname, '..');
const repoRoot = path.resolve(wfDir, '..');

function mustInclude(hay, needle, msg) {
  if (!hay.includes(needle)) throw new Error(msg || `missing ${needle}`);
}

const ryzen = fs.readFileSync(path.join(wfDir, 'knowledge', 'amd_ryzen.md'), 'utf8');
for (const fact of ['gfx1151', 'fp8 matrix path', 'none', 'WMMA', '8060S (40 CU)']) {
  mustInclude(ryzen, fact, `amd_ryzen.md no longer states: ${fact}`);
}
// The MFMA-only Triton knobs do not raise and do not change codegen here, so the
// only safe way to name them is as a trap. Naming them without that qualifier
// would send an agent tuning a knob that cannot move the number.
if (/matrix_instr_nonkdim/.test(ryzen)) {
  mustInclude(ryzen, 'accepted and silently ignored',
    'amd_ryzen.md names matrix_instr_nonkdim without saying it is silently ignored');
}

// The long form is a separate directory so amd_ryzen.md can stay a summary. A
// pointer to a directory that does not exist is worse than no pointer.
const longFormRel = 'perf_knowledge/hardware/rdna35_gfx1151';
// Agents read amd_ryzen.md as SKILL_DIR/knowledge/amd_ryzen.md, so the pointer is spelled from SKILL_DIR.
mustInclude(ryzen, `$SKILL_DIR/../${longFormRel}`, 'amd_ryzen.md lost the SKILL_DIR pointer to the long form');
const longFormDir = path.join(repoRoot, longFormRel);
const pages = fs.readdirSync(longFormDir).filter((f) => f.endsWith('.md'));
if (!pages.includes('arch.md')) {
  throw new Error(`${longFormRel} has no arch.md entry point`);
}
const reproDir = path.join(longFormDir, 'repro');
const docs = pages.map((f) => path.join(longFormDir, f)).concat([path.join(reproDir, 'README.md')]);

// Every relative link and every relative `sources:` entry must resolve. A renamed file
// otherwise leaves a dangling reference with no error anywhere.
for (const doc of docs) {
  const rel = path.relative(repoRoot, doc);
  const text = fs.readFileSync(doc, 'utf8');
  const targets = [...text.matchAll(/\]\((?!\w+:)([^)#\s]+)(?:#[^)]*)?\)/g)].map((m) => m[1]);
  const front = text.startsWith('---\n') ? text.slice(4, text.indexOf('\n---', 4)) : '';
  for (const m of front.matchAll(/^\s+- (\S+)\s*$/gm)) {
    if (!/^\w+:\/\//.test(m[1])) targets.push(m[1]);  // URLs are not checked here
  }
  for (const target of targets) {
    if (!fs.existsSync(path.resolve(path.dirname(doc), target))) {
      throw new Error(`${rel} references a missing file: ${target}`);
    }
  }
}

// The pages defer to the canonical sources instead of restating them, so arch.md must link each
// one with a markdown link that resolves to it (naming the file is not enough).
const archPath = path.join(longFormDir, 'arch.md');
const archLinks = [...fs.readFileSync(archPath, 'utf8').matchAll(/\]\((?!\w+:)([^)#\s]+)\)/g)]
  .map((m) => path.relative(repoRoot, path.resolve(longFormDir, m[1])));
for (const canonical of [
  'kernel_workflow/knowledge/amd_ryzen.md',
  'perf_knowledge/hardware/data/hw_constants.json',
  'e2e_workflow/knowledge/analysis_skills/roofline/peaks.md',
]) {
  if (!archLinks.includes(canonical)) throw new Error(`arch.md does not link the canonical ${canonical}`);
}

// Every [measured] item names where it comes from under repro/: a script, or repro/README.md
// (its one-line checks or its list of campaign-only claims). Every script must exist and be described.
const reproReadme = fs.readFileSync(path.join(reproDir, 'README.md'), 'utf8');
mustInclude(reproReadme, '## Claims with no script here', 'repro/README.md lost its list of claims with no script');
for (const page of pages) {
  const items = fs.readFileSync(path.join(longFormDir, page), 'utf8')
    .split(/\n\s*\n|\n(?=\s{0,3}(?:[-*]|\d+\.)\s)/);
  for (const item of items) {
    if (item.includes('[measured]') && !/\brepro\//.test(item)) {
      throw new Error(`${page}: a [measured] item cites nothing under repro/: ${item.trim().slice(0, 80)}`);
    }
  }
}
for (const script of fs.readdirSync(reproDir).filter((f) => f !== 'README.md' && !f.startsWith('.'))) {
  mustInclude(reproReadme, `\`${script}\``, `repro/README.md does not describe ${script}`);
}
for (const doc of docs) {
  for (const m of fs.readFileSync(doc, 'utf8').matchAll(/repro\/([\w.]+\.(?:py|sh))/g)) {
    if (!fs.existsSync(path.join(reproDir, m[1]))) {
      throw new Error(`${path.relative(repoRoot, doc)} cites repro/${m[1]}, which does not exist`);
    }
  }
}

// The shared kernel guides point gfx115x at the long form: every page they name must exist, and the
// occupancy sweep must not be pinned to gfx1151 (its VGPR steps differ from gfx1150/1152/1153's).
for (const guide of ['profiling_guide.md', 'triton_optimization.md']) {
  const text = fs.readFileSync(path.join(wfDir, 'knowledge', guide), 'utf8');
  for (const m of text.matchAll(/rdna35_gfx1151\/(?:` `)?(\w+\.md)/g)) {
    if (!pages.includes(m[1])) throw new Error(`${guide} points at missing ${longFormRel}/${m[1]}`);
  }
  if (/--arch gfx1151/.test(text)) throw new Error(`${guide} pins the occupancy sweep to gfx1151`);
}

mustInclude(
  fs.readFileSync(path.join(repoRoot, 'perf_knowledge', 'index', 'taxonomy.md'), 'utf8'),
  '| `gfx1151` |', 'taxonomy.md does not list gfx1151 as a hardware generation');

console.log('PASS: amd_ryzen.md states the gfx1151 facts; the long form resolves, links its canonical sources, and every cited script and guide pointer exists.');
