#!/usr/bin/env node
// Static/behavioral regression guard for the optional analysis_skill prompt injection.
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const FILE = path.join(ROOT, 'kernel_workflow', 'kernel_mega.js');
const src = fs.readFileSync(FILE, 'utf8');
let failures = 0;
const ok = (condition, message) => {
  if (condition) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log(`\n# ${path.relative(ROOT, FILE)}`);

const gate = src.match(
  /const ANALYSIS_SKILL = [\s\S]*?if \(ANALYSIS_SKILL_ON\) log\([^\n]+\);/,
);
ok(!!gate, 'analysis-skill gating block found');
if (gate) {
  const make = new Function(
    'A',
    'WORKFLOW_DIR',
    'log',
    `${gate[0]}\nreturn { ANALYSIS_SKILL, ANALYSIS_SKILL_ON, ANALYSIS_SKILL_INPUTS };`,
  );
  const probe = (args) => make(args, '/wf', () => {});
  for (const args of [
    {},
    { analysis_skill: 'none' },
    { analysis_skill: 'false' },
    { analysis_skill: '' },
    { analysis_skill: '   ' },
  ]) {
    const result = probe(args);
    ok(result.ANALYSIS_SKILL_ON === false, `OFF for ${JSON.stringify(args)}`);
    ok(
      Object.keys(result.ANALYSIS_SKILL_INPUTS).length === 0,
      `OFF injects zero input keys for ${JSON.stringify(args)}`,
    );
  }
  const on = probe({ analysis_skill: 'moe_bottleneck' });
  ok(on.ANALYSIS_SKILL_ON === true, 'explicit safe skill enables analysis');
  ok(
    on.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR ===
      '/wf/knowledge/analysis_skills/moe_bottleneck',
    'ON resolves the expected skill directory',
  );
  for (const unsafe of ['../roles', 'a/b', '/tmp/x', 'UPPER']) {
    let threw = false;
    try { probe({ analysis_skill: unsafe }); } catch (_) { threw = true; }
    ok(threw, `unsafe skill slug rejected: ${unsafe}`);
  }
}

const blockSource = src.match(
  /function analysisSkillBlock[\s\S]*?\n}\n\nfunction roleAgent/,
);
ok(!!blockSource, 'dynamic analysisSkillBlock found');
if (blockSource) {
  const functionText = blockSource[0].replace(/\n\nfunction roleAgent$/, '');
  const makeBlock = new Function(
    'ANALYSIS_SKILL_ON',
    'ANALYSIS_SKILL_INPUTS',
    `${functionText}\nreturn analysisSkillBlock;`,
  );
  const off = makeBlock(false, {});
  ok(off('profile_engineer', 'baseline') === '', 'OFF adds no profile role text');
  ok(off('tech_lead', 'plan_round') === '', 'OFF adds no TechLead role text');
  const on = makeBlock(true, {
    ANALYSIS_SKILL: 'moe_bottleneck',
    ANALYSIS_SKILL_DIR: '/wf/knowledge/analysis_skills/moe_bottleneck',
  });
  ok(on('profile_engineer', 'baseline') === '', 'generic Profile remains analysis-free');
  ok(on('analysis_engineer', 'analyze_profile').includes('checked-in'), 'ON injects deterministic analysis instructions');
  // The injected text is deliberately operator-agnostic (the skill system is generic, not MoE-
  // specific), so anchor on the contract it states — read analysis_result, gate on status/version —
  // not on any wording that names a particular workload.
  const planText = on('tech_lead', 'plan_round');
  ok(planText.includes('PROFILE_SUMMARY.analysis_result'), 'ON injects analysis-to-planning boundary');
  ok(/analysis_schema_version|awaiting_measurement/.test(planText), 'ON states the evidence gating rule');
  ok(on('engineer', 'optimize') === '', 'unrelated roles receive no analysis text');
}

const sites = (src.match(/\.\.\.ANALYSIS_SKILL_INPUTS/g) || []).length;
ok(sites === 1, `inputs spread only into dedicated analysis call (${sites})`);
ok(
  /base \+ expertSkillsBlock\(role, phase\) \+ analysisSkillBlock\(role, phase\)/.test(src),
  'roleAgent appends analysis text dynamically',
);
ok(
  /const ANALYSIS_RESULT_SCHEMA[\s\S]*?analysis_json:\s*\{\s*type:\s*'string'\s*\}/.test(src),
  'dedicated analysis-result schema carries generic analysis_json',
);
ok(
  /async function runProfileAnalysis[\s\S]*?analysis_engineer[\s\S]*?ANALYSIS_RESULT_SCHEMA/.test(src),
  'workflow has a dedicated analysis-agent call and schema',
);
ok(
  (src.match(/analysis_result:/g) || []).length >= 2,
  'baseline and reprofile attach validated analysis_result separately',
);
ok(
  !/moe_analysis_json:\s*\{/.test(src),
  'core workflow schema no longer hardcodes a MoE output field',
);

for (const roleFile of ['profile_engineer.md', 'tech_lead.md']) {
  const text = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'roles', 'mega', roleFile), 'utf8');
  ok(
    !text.includes('ANALYSIS_SKILL') && !text.includes('moe_analysis_json'),
    `${roleFile} stays free of always-on analysis-skill prompt text`,
  );
}

console.log(
  failures === 0
    ? '\nPASS: OFF injects no inputs or role text; ON is safe-slugged and analysis-only.'
    : `\nFAILED: ${failures} assertion(s).`,
);
process.exit(failures === 0 ? 0 : 1);
