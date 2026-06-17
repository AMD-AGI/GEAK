#!/usr/bin/env node
// Regression guard for the `use_expert_skills` toggle (no GPU, no model needed).
//
// Invariant under test: when use_expert_skills is OFF, the expert-skills feature injects NOTHING into
// any role prompt, so roleAgent's output is byte-identical to a build without the feature. We prove this
// behaviorally by extracting the ACTUAL `expertSkillsBlock` function source from each workflow script and
// asserting it returns '' (a) whenever the flag is off, and (b) for any non-consumer role even when on;
// and that roleAgent's return is exactly `base + expertSkillsBlock(role)` (purely additive).
//
// Run:  node e2e_workflow/scripts/test_expert_skills_off_identical.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const TARGETS = [
  { file: path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'),
    consumer: 'system_architect', nonConsumer: 'director' },
  { file: path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js'),
    consumer: 'tech_lead', nonConsumer: 'director' },
];

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

for (const t of TARGETS) {
  const rel = path.relative(ROOT, t.file);
  console.log(`\n# ${rel}`);
  const src = fs.readFileSync(t.file, 'utf8');

  // 1) Extract the real expertSkillsBlock function (body closes with a brace at column 0).
  const m = src.match(/function expertSkillsBlock\(role\) \{[\s\S]*?\n\}/);
  ok(!!m, 'expertSkillsBlock(role) defined');
  if (!m) continue;

  // 2) Rebuild it with controlled "module-scope" deps and probe its behavior.
  const make = new Function(
    'USE_EXPERT_SKILLS', 'EXPERT_SKILL_ROLES', 'EXPERT_SKILLS_DIR', 'WORKFLOW_DIR',
    m[0] + '\nreturn expertSkillsBlock;');
  const roles = new Set([t.consumer]);

  const off = make(false, roles, '/x/expert_skills', '/wf');
  ok(off(t.consumer) === '', `OFF -> '' for consumer role (${t.consumer})`);
  ok(off(t.nonConsumer) === '', `OFF -> '' for non-consumer role (${t.nonConsumer})`);

  const on = make(true, roles, '/x/expert_skills', '/wf');
  ok(on(t.consumer) !== '', `ON -> non-empty for consumer role (${t.consumer})`);
  ok(on(t.nonConsumer) === '', `ON -> '' for NON-consumer role (${t.nonConsumer}) (no pollution)`);
  ok(on(t.consumer).includes('/x/expert_skills/index.yaml'), 'ON block points at the skills index');
  ok(/ADVISORY/.test(on(t.consumer)), 'ON block is labelled ADVISORY');

  // 3) The default must be OFF (opt-in): use_expert_skills defaults to 'false'.
  ok(/A\.use_expert_skills != null \? A\.use_expert_skills : 'false'/.test(src),
    "use_expert_skills defaults to 'false' (opt-in)");

  // 4) roleAgent must be purely additive: returns `base + expertSkillsBlock(role)`.
  ok(/return base \+ expertSkillsBlock\(role\);/.test(src),
    'roleAgent returns base + expertSkillsBlock(role) (additive)');
  ok(/const base = `You are the \$\{role\}\. PHASE=\$\{phase\}\./.test(src),
    'roleAgent base template preserved (original anchor intact)');
}

console.log(failures === 0
  ? '\nPASS: use_expert_skills OFF is byte-identical (injection is purely additive).'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
