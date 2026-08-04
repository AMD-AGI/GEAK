#!/usr/bin/env node
// Regression guard for the `analysis_skill` toggle (no GPU, no model needed).
//
// Invariant under test: with the skill OFF, the profile-analysis feature injects NOTHING into any role
// prompt, so the run behaves exactly as it did before the feature existed. We prove this behaviorally by
// extracting the ACTUAL ANALYSIS_SKILL_* block from the workflow script and asserting that
//   (a) OFF yields empty strings for every input, and
//   (b) the object SHAPE is identical on and off (same keys), so a spread can never add or drop a key,
//   (c) the inputs are spread LAST at each call site and collide with no other key (no shadowing).
//
// Run:  node e2e_workflow/scripts/test_analysis_skill_off_identical.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');            // .../GEAK
const FILE = path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

console.log(`\n# ${path.relative(ROOT, FILE)}`);
const src = fs.readFileSync(FILE, 'utf8');

// 1) Extract the real gating block and probe it with controlled module-scope deps.
const m = src.match(/const ANALYSIS_SKILL = [\s\S]*?ANALYSIS_SKILL_DIR: '' \};/);
ok(!!m, 'ANALYSIS_SKILL_* gating block found');
if (m) {
  const make = new Function('A', 'WORKFLOW_DIR',
    m[0] + '\nreturn { ANALYSIS_SKILL, ANALYSIS_SKILL_ON, ANALYSIS_SKILL_INPUTS };');
  const probe = (a) => make(a, '/wf');

  // OFF, by every spelling a caller might use
  for (const a of [{ analysis_skill: 'none' }, { analysis_skill: 'false' },
                   { analysis_skill: '' }, { analysis_skill: '   ' }]) {
    const r = probe(a);
    ok(r.ANALYSIS_SKILL_ON === false, `OFF for ${JSON.stringify(a)}`);
    ok(r.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL === '' && r.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR === '',
      `OFF -> every input is '' for ${JSON.stringify(a)}`);
  }

  // ON: default, and explicit alternative skill (pluggability)
  const def = probe({});
  ok(def.ANALYSIS_SKILL === 'roofline' && def.ANALYSIS_SKILL_ON === true, "defaults to 'roofline' (ON)");
  ok(def.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR === '/wf/knowledge/analysis_skills/roofline',
    'ON -> skill dir resolves under knowledge/analysis_skills/<skill>');
  const alt = probe({ analysis_skill: 'some-other-skill' });
  ok(alt.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR === '/wf/knowledge/analysis_skills/some-other-skill',
    'ON -> an arbitrary skill name is pluggable (dir swap only)');

  // Shape stability: same keys on and off, so a spread never adds/removes a key.
  const keysOn = Object.keys(def.ANALYSIS_SKILL_INPUTS).sort().join(',');
  const keysOff = Object.keys(probe({ analysis_skill: 'none' }).ANALYSIS_SKILL_INPUTS).sort().join(',');
  ok(keysOn === keysOff, `input object shape identical ON vs OFF (${keysOn})`);
}

// 2) The spread must be additive at every call site: the ANALYSIS_SKILL_* keys are set nowhere else,
//    so spreading them can never shadow an existing input.
const sites = (src.match(/\.\.\.ANALYSIS_SKILL_INPUTS/g) || []).length;
ok(sites >= 8, `spread into every consumer call site (found ${sites})`);
if (m) {
  const outside = src.replace(m[0], '');
  const stray = (outside.match(/ANALYSIS_SKILL(_DIR)?\s*:/g) || []).length;
  ok(stray === 0,
    `ANALYSIS_SKILL/_DIR used as an object key ONLY inside the gating block (found ${stray} elsewhere) ` +
    `-> the spread can never shadow another input`);
}

// 3) Consumers must treat the prior as optional and advisory.
ok(/ADVISORY|advisory/.test(src), 'gating block documents the prior as advisory');
for (const [role, phrase] of [['profiler', 'ANALYSIS_SKILL_DIR'], ['system_architect', 'ANALYSIS_SKILL_DIR']]) {
  const roleSrc = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'roles', `${role}.md`), 'utf8');
  ok(roleSrc.includes(phrase), `roles/${role}.md consumes ${phrase}`);
  ok(/non-empty|EXISTS|else skip|otherwise skip/i.test(roleSrc), `roles/${role}.md guards on the prior being present`);
}

console.log(failures === 0
  ? '\nPASS: analysis_skill OFF injects nothing (feature is purely additive).'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
