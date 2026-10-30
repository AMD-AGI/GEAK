#!/usr/bin/env node
// Generic-mega measurement inputs must be empty for Expert-Skill runs and every non-mega mode:
// a skill-on mega wave has to replay with byte-identical prompts after this line was added.
// The MoE regime inputs (MRI) are the deliberate exception: runs pinning a megamoe Skill get them.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_mega.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const def = src.match(/const GMI = ([\s\S]*?);\n/);
ok(!!def, 'GMI is defined');
const gmi = (MODE, USE_EXPERT_SKILLS, A = {}) =>
  // eslint-disable-next-line no-new-func
  new Function('MODE', 'USE_EXPERT_SKILLS', 'WORKFLOW_DIR', 'A', `return ${def[1]};`)(
    MODE, USE_EXPERT_SKILLS, '/wf', A);
ok(Object.keys(gmi('mega', true)).length === 0, 'skill-on mega gets no measurement inputs');
ok(Object.keys(gmi('optimize', false)).length === 0, 'non-mega gets no measurement inputs');
ok(gmi('mega', false).MEGA_MEASUREMENT_GUIDE === '/wf/knowledge/mega_measurement.md',
  'generic mega gets the guide');
ok(gmi('mega', false).MEGA_GENERIC_LESSONS === '/wf/knowledge/mega_generic_lessons.md',
  'generic mega gets the lessons');
ok(gmi('mega', false).MEGA_AB_TOOL === '/wf/tools/mega_graph_ab.py' &&
   !('MEGA_REGIMES_GUIDE' in gmi('mega', false)),
  'generic mega gets the paired graph A/B tool, not the regime guide');
ok(!('MEGA_KNOWLEDGE' in gmi('mega', false, { mega_knowledge: 'read /k.md' })),
  'no launch argument can hand the generic mega roles extra knowledge files');

// MoE regime guide + A/B tool: every MoE mega run (skill on or off), nothing elsewhere.
const mriOn = src.match(/const MEGA_REGIMES_ON = ([\s\S]*?);\n/);
const mriDef = src.match(/const MRI = ([\s\S]*?\n\} : \{\});\n/);
ok(!!mriOn && !!mriDef, 'MEGA_REGIMES_ON and MRI are defined');
const mri = (MODE, USE_EXPERT_SKILLS, ANALYSIS_SKILL, EXPERT_SKILL_ID, A = {}) => {
  // eslint-disable-next-line no-new-func
  const on = new Function('MODE', 'A', 'ANALYSIS_SKILL', 'EXPERT_SKILL_ID', `return ${mriOn[1]};`)(
    MODE, A, ANALYSIS_SKILL, EXPERT_SKILL_ID);
  // eslint-disable-next-line no-new-func
  return new Function('MEGA_REGIMES_ON', 'USE_EXPERT_SKILLS', 'WORKFLOW_DIR', `return ${mriDef[1]};`)(
    on, USE_EXPERT_SKILLS, '/wf');
};
ok(Object.keys(mri('optimize', false, 'moe_bottleneck', '')).length === 0, 'non-mega gets no regime inputs');
ok(Object.keys(mri('mega', false, 'none', '')).length === 0, 'non-MoE mega gets no regime inputs');
ok(Object.keys(mri('mega', false, 'moe_bottleneck', '')).length === 0,
  'a generic MoE mega run gets no regime guide: it names Skill designs');
const skillOn = mri('mega', true, 'none', 'megamoe_small_mtpr_fusion');
ok(skillOn.MEGA_AB_TOOL === '/wf/tools/mega_graph_ab.py' &&
  skillOn.MEGA_GENERIC_LESSONS === '/wf/knowledge/mega_generic_lessons.md' &&
  skillOn.MEGA_MEASUREMENT_GUIDE === undefined, 'skill-on MoE mega gets regime inputs + lessons, no measurement guide');
ok(Object.keys(mri('mega', true, 'moe_bottleneck', 'x', { mega_regimes: 'false' })).length === 0,
  'mega_regimes=false turns them off');
ok(mri('mega', false, 'none', '', { mega_regimes: 'true' }).MEGA_REGIMES_GUIDE &&
   mri('mega', false, 'none', '', { mega_regimes: 'true' }).MEGA_GENERIC_LESSONS === undefined,
  'mega_regimes=true turns them on explicitly; a generic run keeps its lessons from GMI only');
const mriUses = src.split('\n').filter((l) => /\bMRI\b/.test(l) && !/const MRI =/.test(l));
ok(mriUses.filter((l) => /\.\.\.MRI\b/.test(l)).length === 7, 'MRI spread at the 7 GMI spread sites');
ok(mriUses.every((l) => /\.\.\.GMI, \.\.\.MRI\b/.test(l) || /MRI\.MEGA_REGIMES_GUIDE \?/.test(l)),
  'every MRI use is a spread next to GMI or the Skill block\'s guide reference');

// The generic graph-contract runner is the default only for generic mega; a pinned skill keeps its
// runtime, and an explicit graph_contract_tool arg still wins.
const gct = src.match(/const GRAPH_CONTRACT_TOOL = ([\s\S]*?);\n/);
ok(!!gct, 'GRAPH_CONTRACT_TOOL is defined');
const tool = (MODE, USE_EXPERT_SKILLS, PINNED_MEGA_SKILL, A = {}) =>
  // eslint-disable-next-line no-new-func
  new Function('MODE', 'USE_EXPERT_SKILLS', 'PINNED_MEGA_SKILL', 'A', 'WORKFLOW_DIR',
    `return ${gct[1]};`)(MODE, USE_EXPERT_SKILLS, PINNED_MEGA_SKILL, A, '/wf');
ok(tool('mega', true, true) === '/wf/tools/expert_skill_runtime.py', 'pinned skill keeps its runtime');
ok(tool('mega', true, false) === '', 'unpinned skill-on mega keeps an empty tool');
ok(tool('optimize', false, false) === '', 'non-mega keeps an empty tool');
ok(tool('mega', false, false) === '/wf/tools/generic_graph_contract.py', 'generic mega gets the runner');
ok(tool('mega', false, false, { graph_contract_tool: '/x.py' }) === '/x.py', 'explicit arg still wins');

// Every use is either a spread (adds nothing when {}) or the guard of the generic-only Profile call.
const uses = src.split('\n').filter((l) => /\bGMI\b/.test(l) && !/const GMI =/.test(l));
// profile guard + profile, analysis, benchmark setup, planner, engineer, verify, finalist spreads.
ok(uses.length === 8, `GMI used at 8 sites (got ${uses.length})`);
ok(uses.every((l) => /\.\.\.GMI\b/.test(l) || /if \(GMI\.MEGA_MEASUREMENT_GUIDE && /.test(l)),
  'every GMI use is a spread or the generic-only guard');
ok(/profileSummary = \{ \.\.\.profileSummary, \.\.\.mp \}/.test(src) &&
  !/mp\.__agent_timed_out/.test(src), 'a failed baseline measurement keeps the stub, never ends the wave');

for (const f of ['knowledge/mega_measurement.md', 'knowledge/mega_generic_lessons.md',
  'knowledge/mega_regimes.md', 'tools/mega_graph_ab.py',
  'scripts/multi_rank_analysis/reduce_stage_meter.py', 'tools/generic_graph_contract.py']) {
  ok(fs.existsSync(path.join(WF, f)), `${f} exists`);
}
const guide = fs.readFileSync(path.join(WF, 'knowledge/mega_measurement.md'), 'utf8');
ok(/GEAK_MEGA_STAGE_METER=1/.test(guide) && /never produces a scored latency/.test(guide),
  'guide fixes the meter switch and keeps the meter out of scoring');
ok(/graph_contract_adapter\.py/.test(guide) && /replaces\*\* the role's `GRAPH_CONTRACT_TOOL` command/.test(guide),
  'guide defines the adapter and the generic claim command');

if (failures) { console.error(`FAIL: ${failures} assertion(s) failed.`); process.exit(1); }
console.log('PASS: generic-mega inputs and graph-contract runner are gated off for skill-on and non-mega runs.');
