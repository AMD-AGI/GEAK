#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const wf = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(wf, 'kernel_mega.js'), 'utf8');
const director = fs.readFileSync(path.join(wf, 'roles', 'mega', 'director.md'), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# production is the bounded default');
ok(/const BUDGET = parseInt\(A\.budget != null \? A\.budget : 6/.test(src) &&
   /A\.mega_profile \|\| 'production'/.test(src),
  'default profile uses six candidate turns');
ok(/A\.mega_time_budget_s \|\|\s*\n\s*28800/.test(src) &&
   /MEGA_PRODUCTION \? 7200 : 3600/.test(src) &&
   /A\.mega_closeout_reserve_s \|\| 900/.test(src),
  'modeled target is eight hours with two hours reserved for finalist validation');
ok(/A\.mega_candidate_timeout_s \|\|\s*\n\s*3600/.test(src) &&
   /MEGA_PRODUCTION \? 7200 : 3600/.test(src),
  'candidate and finalist calls have separate bounded timeouts');
ok(/A\.expert_skill_id/.test(src) &&
   /A\.candidate_import_modules/.test(src) &&
   /A\.expert_skill_playbook/.test(src) &&
   /A\.expert_skill_planner_extension/.test(src) &&
   /A\.expert_skill_bundle_sha256/.test(src) &&
   /A\.expert_skill_planner_extension_sha256/.test(src) &&
   /A\.expert_skill_contract_sha256/.test(src) &&
   /A\.expert_skill_contract/.test(src) &&
   /A\.expert_skill_validation/.test(src),
  'Expert Skill configuration is supplied by the caller, not a packaged task');

console.log('\n# valid output triggers early convergence');
ok(/MEGA_STOP_ON_DELIVERABLE && turn\.selected/.test(src) &&
   /expertTargetReached/.test(src) &&
   /EXPERT_SKILL_RECORDED_LOW/.test(src) &&
   /searchAttempts >= MEGA_MIN_SEARCH_ATTEMPTS/.test(src),
  'Skill-on search stops early only after reaching its full-fusion target');
ok(/dispatch deadline reached/.test(src) && /MEGA_TIME_BUDGET_S - MEGA_FINAL_RESERVE_S/.test(src),
  'the workflow preserves final-validation time instead of spending the full wall budget on search');
ok(/availableAfterPrepS < 900/.test(src) &&
   /if \(shouldStructuralVerify\) megaChargeMs\(600000/.test(src),
  'production reserves Author, structural, and GPU handoff time and charges structural verification');
ok(/GPU_WAIT_TIMEOUT_S: gpuWaitBudgetS/.test(src) &&
   /GPU_RUN_TIMEOUT_S: gpuRunBudgetS/.test(src),
  'inner GPU lease wait and run deadlines fit inside the verifier timeout');

console.log('\n# production validates best-first and pays fallback only on failure');
ok(/for \(let i = 0; i < finalistOrder\.length; i\+\+\)/.test(src) &&
   /Trying the next scored fallback/.test(src),
  'final candidates are validated sequentially in score order');
ok(/roleAgent\('director', 'recover_mega'/.test(src),
  'a completed finalist manifest is recovered without repaying GPU validation');
ok(/MEGA_PRODUCTION \? 1 : 2/.test(src) &&
   /MEGA_PRODUCTION \? 2 : MEGA_FINAL_TOP_K/.test(src),
  'one primary finalist is normal; a second is only a failure fallback');
ok(/A\.mega_tie_noise_pct != null \? A\.mega_tie_noise_pct : 1\.45/.test(src),
  'the caller may explicitly set a zero tie band');
ok(/VALIDATION_TIER: MEGA_PRODUCTION \? 'identity_only' : 'full'/.test(src),
  'production second Director call checks identity instead of rerunning the GPU suite');
ok(/Production identity-only tier/.test(director),
  'Director documents the no-duplicate-GPU validation contract');
ok(/timeout_marker: true/.test(src) && /MEGA SAFE STOP/.test(src),
  'a timed-out candidate/finalist stops safely instead of overlapping recovery or fallback GPU work');
ok(/function megaChargeMs\(allocatedMs\) \{ megaAdvanceMs\(allocatedMs\); \}/.test(src) &&
   !/Date\.now\(\)/.test(src),
  'production candidate budgets use deterministic allocated-time accounting');
ok(/if \(!eng \|\| eng\.claim_complete !== true\)/.test(src) &&
   /timed-out writer is quiescent/.test(src) &&
   /lane lock is free/.test(src),
  'author timeout recovery resumes only after single-writer quiescence');
ok(/const role = 'engineer';\s*\n\s*const roleFile = 'engineer\.md';/.test(src) &&
   /MEGA EXPERT SKILL/.test(src) &&
   /EXPERIMENTAL TRANSFER, EXPLICIT PIN/.test(src),
  'skill-enabled and skill-disabled Mega candidates use the same Engineer lifecycle');
ok(/max_retries: 1/.test(src),
  'bounded production calls cannot multiply their wall budget through API retries');
ok(/LANE_MANIFEST: laneManifest/.test(src) && /lane\.json/.test(director),
  'a first-attempt timeout leaves a discoverable lane for the next invocation');
ok(/reportReady && !!finalWinner && megaSelectionContractPassed/.test(src),
  'no finalist means no redundant full validation call');
ok(/validation\.final_patch_verified === true/.test(src),
  'identity-only validation still proves the final patch reproduces the selected tree');
ok(/APPLY_TO_ORIGINAL !== 'true'[\s\S]{0,120}validation\.applied_to_original/.test(src),
  'a requested apply-back must be confirmed before production reports delivery');

console.log(failures === 0
  ? '\nPASS: production is bounded for delivery while audit remains opt-in.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
