#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..', '..');
const wf = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_mega.js'), 'utf8');
const skill = fs.readFileSync(path.join(
  ROOT, 'perf_knowledge', 'expert_skills', 'skills',
  'megamoe_ep_mega_fusion', 'skill.md',
), 'utf8');
const skillDir = path.dirname(path.join(
  ROOT, 'perf_knowledge', 'expert_skills', 'skills',
    'megamoe_ep_mega_fusion', 'skill.md',
));
const playbook = skill;
const planner = fs.readFileSync(path.join(skillDir, 'planner_extension.yaml'), 'utf8');
const contract = fs.readFileSync(path.join(skillDir, 'contract.yaml'), 'utf8');
const validation = fs.readFileSync(path.join(skillDir, 'validation.yaml'), 'utf8');
const engineer = fs.readFileSync(path.join(
  ROOT, 'kernel_workflow', 'roles', 'mega', 'engineer.md',
), 'utf8');
const verifier = fs.readFileSync(path.join(
  ROOT, 'kernel_workflow', 'roles', 'mega', 'verify_engineer.md',
), 'utf8');
const director = fs.readFileSync(path.join(
  ROOT, 'kernel_workflow', 'roles', 'mega', 'director.md',
), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# the experimental transfer is explicit and workflow-owned');
ok(/role: experimental_authoring_prior/.test(skill) &&
   /does not create a reproduction mode/.test(skill),
  'skill declares explicitly pinned constraints inside the common Mega lifecycle');
ok(/planner_extension_file: planner_extension\.yaml/.test(skill) &&
   /contract_file: contract\.yaml/.test(skill) &&
   /runtime_validation_file: runtime_validation\.py/.test(skill) &&
   !/embedded_components:/.test(skill) &&
   fs.existsSync(path.join(skillDir, 'runtime_validation.py')),
  'skill points to independent planner, contract and runtime components');
ok(/normative: true/.test(playbook) &&
   /normative_scope: explicitly_pinned_authoring/.test(playbook) &&
   /does not create a reproduction mode,[\s\S]{0,120}reserved candidate source/.test(playbook),
  'normative implementation constraints do not create a reproduction sub-mode');
ok(/schema_version: expert-skill-contract-v1/.test(contract) &&
   /unified_semantic_loop/.test(contract) &&
   /direct_runtime_combine_item/.test(contract) &&
   /names_are_normative: false/.test(contract) &&
   /severity: advisory/.test(contract),
  'the generic checker receives semantic claims plus non-normative adapter hints');
ok(/schema_version: expert-skill-validation-v1/.test(validation) &&
   /status: experimental/.test(validation) &&
   /auto_apply: false/.test(validation),
  'reference evidence, transfer status, and auto-application are separate');
ok(/G1 and G2 are chosen inside the same loop[\s\S]*no grid barrier/i.test(skill),
  'expert knowledge carries the tile-pipeline mechanism, not merely launch fusion');

console.log('\n# runtime uses the same roles and candidate source');
ok(/EXPERIMENTAL TRANSFER, EXPLICIT PIN/.test(wf),
  'enabled skill is appended to the common planner/engineer prompt');
ok(/pinned authoring usage requires mega_structural_only=true/.test(wf) &&
   /pinned candidate_validation requires contract diagnostics/.test(wf),
  'experimental usage is mechanically limited to explicit authoring or validation');
ok(/expert_skill_id must be a safe slug/.test(wf) &&
   /PINNED_MEGA_SKILL/.test(wf) &&
   /explicitly pinned Expert Skill requires validation status/.test(wf),
  'explicit Skill status and component paths cannot float independently');
ok(/pinned candidate_validation missing gates/.test(wf) &&
   /required_replays>=256/.test(wf) &&
   /direct_graph_accuracy/.test(wf),
  'experimental candidate validation requires the full runtime evidence profile');
ok(!/\[REQUIRE_OVERLAP[^\]]*'require_overlap'\]/.test(wf) &&
   /\[LAUNCH_TARGET === 1 \|\| LAUNCH_TARGET === 2, 'launch_target in \{1,2\}'\]/.test(wf),
  'pinned candidate validation takes a one- or two-launch target and does not require overlap');
ok(/Candidate source remains search\/integrated/.test(wf),
  'expert guidance does not introduce a validated-skill candidate');
ok(!/role = source === 'validated_skill' \? 'mega_engineer'/.test(wf) &&
   /const role = 'engineer'/.test(wf),
  'a special Skill executor is absent from dispatch');
ok(!/id: MEGA_SKILL_CANDIDATE_ID, source: 'validated_skill'/.test(wf),
  'no reserved reproduction lane is registered');
ok(/authoringContractNeedsPreflight/.test(wf) &&
   /const stagedGpuAuthoring/.test(wf) &&
   /gpuCandidateDevelopment \? 0\.85 : 0\.60/.test(wf) &&
   /STAGED_GPU_AUTHORING/.test(wf) &&
   /STRUCTURAL_PREFLIGHT_REQUIRED_NO_GPU/.test(wf) &&
   /AUTHORING_CONTRACT_PREFLIGHT_REQUIRED/.test(wf),
  'candidate validation permits staged GPU smoke while structural-only remains GPU-forbidden');
ok(/fail-closed GPU prohibition unless/.test(engineer) &&
   /earliest construction\/JIT\/small-correctness smoke/.test(engineer) &&
   /Do not scaffold-and-bail/.test(
     fs.readFileSync(path.resolve(__dirname, '..', '..',
       'perf_knowledge/expert_skills/skills/megamoe_ep_mega_fusion/author_guide.md'), 'utf8')) &&
   /temporary compile-time[\s\S]*earns no correctness\/performance credit/.test(engineer) &&
   /ONE input, `AUTHOR_GPU_MODE`/.test(engineer) &&
   /A production fix does NOT revoke GPU access/.test(engineer) &&
   !/GPU authorization is revoked/.test(engineer) &&
   /AUTHOR_GPU_MODE: authorGpuProhibited \? 'forbidden' : 'bounded_smoke'/.test(wf),
  'one AUTHOR_GPU_MODE decides Author GPU use; a production fix reruns the smoke instead of revoking it');
ok(/preserve every failing contract check ID as its own entry/.test(engineer) &&
   /never collapse them into a summary/.test(engineer) &&
   /GPU absence is not a reason to defer source authoring/.test(engineer),
  'structural-only authoring keeps exact failures and commits coherent GPU-free source progress');
ok(/postAuthoringVerify/.test(wf) &&
   /priorLaneHead/.test(wf) &&
   /structuralPassThisTurn/.test(wf) &&
   /meta\.structural_candidate_head === expectedHead/.test(wf) &&
   /postAuthoringVerify \|\|/.test(wf),
  'a new structurally verified authoring HEAD reaches independent on-card Verify in the same turn');
ok(/required_changed_files_missing/.test(verifier) &&
   /Never synthesize failures from informational/.test(verifier) &&
   /resources\.parameters/.test(verifier),
  'structural verifier preserves exact PlanIR and does not promote reference parity advisories');
ok(/CANDIDATE_IMPORT_MODULES/.test(wf) &&
   /CANDIDATE_PYTHONPATH/.test(wf) &&
   /every module in \$\{JSON\.stringify\(CANDIDATE_IMPORT_MODULES\)\}/.test(wf),
  'the caller supplies a generic candidate import-identity contract');
ok(/IMPORT_IDENTITY_VOID/.test(engineer) &&
   /IMPORT_IDENTITY_VOID/.test(verifier) &&
   /IMPORT_IDENTITY_VOID/.test(director),
  'author, verifier and final arbiter reject installed/global module resolution');

console.log(failures === 0
  ? '\nPASS: experimental transfer knowledge is explicit and workflow-owned.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
