#!/usr/bin/env node
// Regression guard for the `use_expert_skills` toggle (no GPU, no model needed).
//
// Invariant under test: when use_expert_skills is OFF, the expert-skills feature injects NOTHING into
// any role prompt, so roleAgent's output is byte-identical to a build without the feature. We prove this
// behaviorally by extracting the ACTUAL `expertSkillsBlock` function source from each workflow script and
// asserting it returns '' (a) whenever the flag is off, and (b) for any non-consumer role even when on;
// and that roleAgent adds the skill block without changing its other prompt content.
//
// The E2E prompt has additional gated blocks, which are each checked for off-is-empty behavior.
// The shared kernel helper also carries phase-dependent native measurement instructions; its
// rendered prompts prove that switching expert skills off preserves those instructions.
//
// Run:  node e2e_workflow/scripts/test_expert_skills_off_identical.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const TARGETS = [
  { file: path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'),
    consumer: 'system_architect', nonConsumer: 'director' },
  // kernel_workflow.js is now a thin dispatcher; the expert-skills consumer roles (tech_lead etc.)
  // live in the single-language WORKER, kernel_lane.js — that is where the byte-identity invariant holds.
  { file: path.join(ROOT, 'kernel_workflow', 'kernel_lane.js'),
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

  // The shared kernel helper also carries the native measurement instructions, whose
  // final-validation text depends on phase. Exercise its actual output with skills off
  // and on, including that independent instruction block.
  const roleFrom = src.match(/function roleAgentFrom\(dir, role, phase, intro, inputs\) \{[\s\S]*?\n\}/);
  if (roleFrom) {
    const render = (skills, metric) => new Function(
      'expertSkillsBlock', 'externalMetricInstructions', 'TASK', 'cfg',
      roleFrom[0] + '\nreturn roleAgentFrom;')(
        skills, metric, 'fixture task', JSON.stringify);
    for (const [role, phase] of [[t.consumer, 'analyze'], [t.nonConsumer, 'validate']]) {
      const args = ['/wf', role, phase, 'fixture intro', { fixture: 'input' }];
      const base = render(() => '', () => '')(...args);
      ok(render(off, () => '')(...args) === base,
        `OFF role prompt is byte-identical without the feature (${role})`);
      ok(render(on, () => '')(...args) === base + on(role),
        `ON role prompt adds only its expert skill block (${role})`);
      const metric = final => final ? '\nfixture final measurement' : '\nfixture search measurement';
      ok(render(off, metric)(...args) === base + metric(role === 'director' && phase === 'validate'),
        `OFF preserves native measurement instructions (${role}/${phase})`);
    }
    continue;
  }

  // 4) roleAgent must be purely additive: `base + expertSkillsBlock(role)` optionally followed by
  //    further `+ <name>Block(role)` terms, and nothing else.
  const ret = src.match(/return base \+ expertSkillsBlock\(role\)((?:\s*\+\s*\w+\(role\))*);/);
  ok(!!ret, 'roleAgent returns base + expertSkillsBlock(role) [+ more blocks] (additive)');
  ok(/const base = `You are the \$\{role\}\. PHASE=\$\{phase\}\./.test(src),
    'roleAgent base template preserved (original anchor intact)');

  // 5) Every EXTRA injector on that return is held to the same bar: gate off -> ''. Each is rebuilt
  //    with its module-scope deps neutralised (SCREAMING_CASE is this codebase's module-constant
  //    convention), so an injector that forgets its early return fails here instead of silently
  //    polluting a feature-off prompt.
  for (const extra of (ret ? ret[1].match(/\w+(?=\(role\))/g) || [] : [])) {
    const fm = src.match(new RegExp(`function ${extra}\\(role\\) \\{[\\s\\S]*?\\n\\}`));
    ok(!!fm, `${extra}(role) defined`);
    if (!fm) continue;
    const deps = [...new Set(fm[0].match(/\b[A-Z][A-Z0-9_]{2,}\b/g) || [])];
    const off = new Function(...deps, `${fm[0]}\nreturn ${extra};`)(
      ...deps.map((d) => (/_ROLES$/.test(d) ? new Set() : '')));
    let out;
    try { out = off(t.consumer); } catch (e) { out = `threw: ${e.message}`; }
    ok(out === '', `${extra} -> '' when its gate is off (no pollution)`);
  }
}

console.log(failures === 0
  ? '\nPASS: use_expert_skills OFF is byte-identical (injection is purely additive).'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
