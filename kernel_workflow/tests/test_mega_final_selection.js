#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const wf = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(wf, 'kernel_mega.js'), 'utf8');
const director = fs.readFileSync(path.join(wf, 'roles', 'mega', 'director.md'), 'utf8');
const skill = fs.readFileSync(
  path.join(wf, '..', 'perf_knowledge', 'expert_skills', 'skills',
    'megamoe_ep_mega_fusion', 'skill.md'), 'utf8',
);
const validation = fs.readFileSync(
  path.join(wf, '..', 'perf_knowledge', 'expert_skills', 'skills',
    'megamoe_ep_mega_fusion', 'validation.yaml'), 'utf8',
);

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# final selection receives only workflow-authored, above-baseline candidates');
ok(/c\.absolute_score > 1\.0/.test(src),
  'portfolio filter rejects every candidate at or below frozen MegaMoE V2');
ok(/candidate_source: raw\.candidate_source/.test(src) &&
   /Candidate source remains search\/integrated/.test(src) &&
   !/id: MEGA_SKILL_CANDIDATE_ID, source: 'validated_skill'/.test(src),
  'Expert Skills alter knowledge, not candidate source or registry lifecycle');
ok(!/source: 'validated_artifact'/.test(src),
  'no hand-authored implementation is registered as a selectable source');

console.log('\n# Director remeasures rather than trusting round scores');
ok(/roleAgent\('director', 'select_mega'/.test(src),
  'a dedicated final arbitration call exists inside the shared workflow');
ok(/CANDIDATES: batch/.test(src) && /FINAL_WORKSPACE = megaSelection\.selected_tree/.test(src),
  'the selected whole tree, not the last round, becomes the final validation workspace');
ok(/claim_complete === true/.test(src) && /selected_candidate_id/.test(src),
  'an incomplete final selection cannot materialize a result');

console.log('\n# the final gate is common and expensive only once');
ok(/A source label[\s\S]*never relaxes a[\s\S]*quality gate/.test(director) &&
   /speedup is `<=1\.0`/.test(director),
  'Director applies one source-blind correctness and above-baseline gate');
ok(/run every supplied regression guard/.test(director) &&
   /required liveness/.test(director),
  'finalists pay the complete guard and liveness contract');
ok(/isolated:\s*1\.0448/.test(validation) &&
   /do not copy[\s\S]*external implementation tree/i.test(skill) &&
   !/megamoe_ep_mega_fusion|1\.0448/.test(src),
  'the validated target belongs to Skill data, never workflow core or a reference arm');

console.log('\n# final output independently checks the selected score');
ok(/finalPrimary > 1\.0/.test(src) && /mega_deliverable = !!\(megaSelection/.test(src),
  'the final return cannot report a sub-baseline megakernel as deliverable');
ok(/validation\.materialization_matches === true/.test(src) &&
   /selected_source_head/.test(src) && /selected_materialized_head/.test(src),
  'a second Director call binds the final workspace to the selected candidate commit');
ok(src.lastIndexOf('persistMegaCandidateState(round, true)') >
   src.indexOf("roleAgent('director', 'validate'"),
  'finalist state is persisted only after independent final validation');
ok(/o\.allowPartialFusion === true \|\| realizedLaunches <= targetLaunches[\s\S]*realizedLaunches < baselineLaunches/.test(src),
  'a faster complete partial fusion may pass final search arbitration when the caller allows it');
ok(/realizedLaunches >= 1 &&/.test(src),
  'launch_target is an upper bound in final arbitration: fewer launches than the target also pass');

console.log(failures === 0
  ? '\nPASS: final Mega output is the fastest fully verified workflow-authored speedup.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
