#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const megaRole = require('./mega_roles');

const wf = path.resolve(__dirname, '..');
const src = require('./mega_source')();
const verifier = megaRole('verify_engineer');
const director = megaRole('director');
let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# Expert-off Mega remains a complete generic lifecycle');
ok(/const PINNED_MEGA_SKILL = USE_EXPERT_SKILLS &&/.test(src),
  'disabled Expert Skills cannot select pinned Skill tools or files');
ok(/const CHECK_EXPERT_SKILL_CONTRACT = USE_EXPERT_SKILLS &&/.test(src),
  'disabled Expert Skills cannot schedule Skill structural verification');
ok(/const REQUIRED_ACCURACY_CASES = Object\.freeze\(argList\([\s\S]{0,240}A\.required_accuracy_cases/.test(src),
  'accuracy coverage has a run-owned input independent of Expert Skills');
ok(/requiredAccuracyCases: REQUIRED_ACCURACY_CASES/.test(src) &&
   /REQUIRED_ACCURACY_CASES,\s*\n\s*MEGA_PLAN_IR/.test(src) &&
   /REQUIRED_ACCURACY_CASES,\s*\n\s*REQUIRE_OVERLAP/.test(src),
  'score-tier and finalist verification receive the run-owned accuracy cases');
ok(/phase\('Analyze'\)/.test(require('./mega_source').full()) && /function megaPlanPrompt/.test(src) &&
   /function megaEngineerTurn/.test(src) && /'director', 'select_mega'/.test(src),
  'generic Analyze, Plan, Author, Verify and final selection remain present');
ok(verifier.includes('When `GRAPH_CONTRACT_TOOL` is empty') &&
   director.includes('When the optional\n     tool is empty') &&
   verifier.includes('never\n   waives direct graph accuracy or liveness evidence'),
  'an absent Skill runtime tool cannot waive generic graph accuracy or liveness');

if (failures) process.exitCode = 1;
else console.log('\nPASS: Expert-off Mega keeps the generic lifecycle.');
