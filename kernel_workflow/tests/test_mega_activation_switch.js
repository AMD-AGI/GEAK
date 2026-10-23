#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const wfDir = path.resolve(__dirname, '..');
const wf = fs.readFileSync(path.join(wfDir, 'kernel_mega.js'), 'utf8');
const engineer = fs.readFileSync(path.join(wfDir, 'roles', 'mega', 'engineer.md'), 'utf8');
const verify = fs.readFileSync(path.join(wfDir, 'roles', 'mega', 'verify_engineer.md'), 'utf8');
const playbook = fs.readFileSync(path.join(
  wfDir, '..', 'perf_knowledge', 'expert_skills', 'skills',
  'megamoe_ep_mega_fusion', 'skill.md'), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# activation belongs to each ordinary candidate');
ok(/activation: normalizeMegaActivation\(c\.activation\)/.test(wf),
  'candidate normalization preserves its activation');
ok(/activation: next\.activation/.test(wf) && /working_snapshot/.test(wf),
  'unverified working snapshots preserve activation');
ok(/ACTIVATION: \(eng && eng\.activation\) \? JSON\.stringify\(eng\.activation\) : 'UNDECLARED'/.test(wf),
  'score Verify receives every common candidate activation');
ok(/patch must be ON by default/i.test(engineer),
  'the common Engineer must make the authored candidate path measurable');

console.log('\n# Expert Skill constraints are normative knowledge, not a special lane');
ok(/AITER_MEGAMOE_FUSE_ALL=1/.test(playbook) &&
   /normative: true/.test(playbook) &&
   /normative_scope: explicitly_pinned_authoring/.test(playbook),
  'the explicitly pinned playbook makes transfer constraints normative');
ok(/SEARCH_ACCEPTS_PARTIAL_FUSION/.test(verify),
  'Verify can measure a different full or partial fusion activation');
ok(!/roles\/mega_engineer\.md/.test(wf) &&
   !/role = source === 'validated_skill' \? 'mega_engineer'/.test(wf),
  'no dedicated reproduction activation role remains');

console.log(failures === 0
  ? '\nPASS: activation is candidate-owned in the unified Mega lifecycle.'
  : `\nFAIL: ${failures} assertion(s)`);
process.exit(failures === 0 ? 0 : 1);
