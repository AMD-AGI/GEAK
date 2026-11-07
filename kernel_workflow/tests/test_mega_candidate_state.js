#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const wf = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(wf, 'kernel_mega.js'), 'utf8');
const director = fs.readFileSync(path.join(wf, 'roles', 'mega', 'director.md'), 'utf8');
const lead = fs.readFileSync(path.join(wf, 'roles', 'mega', 'tech_lead.md'), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const identityBlock = src.match(
  /\/\/ <<REPLAY:structural_evidence_identity>>([\s\S]*?)\/\/ <<\/REPLAY:structural_evidence_identity>>/,
);
if (!identityBlock) throw new Error('structuralEvidenceIdentityVerdict not found');
// eslint-disable-next-line no-new-func
const structuralIdentity = new Function(
  `${identityBlock[1]}\nreturn structuralEvidenceIdentityVerdict;`,
)();
const EXPECTED_IDENTITY = {
  candidateId: 'candidate',
  candidateHead: 'abc123',
  skillId: 'skill',
  skillRevision: 'v1',
  contractSha256: 'c'.repeat(64),
  plannerExtensionSha256: 'p'.repeat(64),
  skillBundleSha256: 'b'.repeat(64),
};
const VALID_REPORT = {
  candidate_id: 'candidate',
  candidate_head: 'abc123',
  candidate_tree_digest: 't'.repeat(64),
  skill_id: 'skill',
  contract_revision: 'v1',
  contract_sha256: 'c'.repeat(64),
  planner_extension_sha256: 'p'.repeat(64),
  skill_bundle_sha256: 'b'.repeat(64),
};

console.log('\n# candidate registry is a first-class resume artifact');
ok(/candidate_registry: \{ type: 'array'/.test(src) &&
   /measurement_calibration: \{ type: 'object'/.test(src),
  'prior-state schema declares candidates and calibration');
ok(/Array\.isArray\(ps\.candidate_registry\)/.test(src) &&
   /for \(const c of ps\.candidate_registry\)/.test(src) &&
   /upsertMegaCandidate\(megaCandidateRegistry, \{/.test(src),
  'resume merges every prior candidate rather than replacing the registry');
ok(/const MEGA_RESUME_STATE = MODE === 'mega'/.test(src) &&
   /STATE_DIR, MEGA_RESUME_STATE/.test(src) &&
   /MEGA_RESUME_STATE=true[\s\S]{0,320}MUST[\s\S]{0,180}prior_state/.test(director),
  'a machine-readable continuation flag outranks stale fresh-run task prose');
ok(/Calibration is intentionally NOT restored as authority/.test(src) &&
   !/megaMeasurementCalibration = \{ \.\.\.megaMeasurementCalibration, \.\.\.ps\.measurement_calibration \}/.test(src),
  'prior calibration is audit data and cannot authorize a new wave');
ok(/megaStateSequenceBase = Math\.max\(0, Number\(ps\.state_sequence\)\)/.test(src) &&
   /megaStateSequenceBase \+ Number\(currentRound\) \* 10/.test(src),
  'state sequence advances from the prior wave instead of resetting with round 1');
ok(/const staleContract = staleIdentityReasons\.length > 0/.test(src) &&
   /n\.structural_contract_revision !== EXPERT_SKILL_REVISION/.test(src) &&
   /n\.structural_contract_sha256 !== EXPERT_SKILL_CONTRACT_SHA256/.test(src) &&
   /n\.structural_planner_extension_sha256 !==/.test(src) &&
   /n\.structural_skill_bundle_sha256 !== EXPERT_SKILL_BUNDLE_SHA256/.test(src) &&
   /n\.structural_candidate_head !== n\.head/.test(src) &&
   /invalidating structural evidence/.test(src),
  'revision, content digests and candidate HEAD invalidate old structural evidence');
ok(structuralIdentity(VALID_REPORT, EXPECTED_IDENTITY).pass,
  'a structural result bound to the full evidence identity passes');
ok(!structuralIdentity(
  { ...VALID_REPORT, candidate_head: 'wrong' }, EXPECTED_IDENTITY,
).pass,
  'a verifier result for another candidate HEAD is rejected');
ok(!structuralIdentity(
  { ...VALID_REPORT, skill_bundle_sha256: 'x'.repeat(64) }, EXPECTED_IDENTITY,
).pass,
  'same-revision Skill content drift is rejected by bundle digest');

console.log('\n# each lane owns persistent source state');
ok(/STATE_DIR}\/candidates\/\$\{candidateId\}\/tree/.test(src),
  'candidate workspaces live under stable per-id state paths');
ok(/If .*\/\.git does not exist[\s\S]*If it exists, continue from its HEAD/.test(src),
  'an existing lane is continued and never recreated');
ok(/candidate_registry: <CANDIDATE_REGISTRY>/.test(lead) &&
   /never collapse them into `\$STATE_DIR\/best`/.test(lead),
  'state writer preserves independent lineages instead of one mutable best tree');
ok(/Mega candidate-registry exception/.test(director) &&
   /resumed:true` plus `prior_state`/.test(director) &&
   /even when[\s\S]*`STATE_DIR\/best` is absent/.test(director),
  'Director restores candidate state without requiring a legacy global best');
ok(/MEGA STATE PERSIST FAILED/.test(src) &&
   /return false;/.test(src) &&
   /candidate state for round \$\{currentRound\} could not be persisted/.test(src),
  'a round cannot advance without a confirmed monotonic state write');
ok(/working_snapshot: \{/.test(src) &&
   /activation: next\.activation/.test(src) &&
   /topology: next\.topology/.test(src) &&
   /structural_skill_bundle_sha256: next\.structural_skill_bundle_sha256/.test(src) &&
   /structural_contract_sha256: next\.structural_contract_sha256/.test(src),
  'working progress persists its complete structural certificate');
ok(/megaEffectiveLaneCheckpoint/.test(src) &&
   /const effective = megaEffectiveLaneCheckpoint\(c\)/.test(src),
  'author preflight and Planner consume the effective working checkpoint');
ok(/structural_contract_revision: structuralPass/.test(src) &&
   /structural_contract_sha256: structuralPass/.test(src) &&
   /structural_candidate_tree_digest: structuralPass/.test(src) &&
   /structural_skill_bundle_sha256: structuralPass/.test(src) &&
   /structural_planner_extension_sha256: structuralPass/.test(src) &&
   /contract_failures: structuralPass/.test(src),
  'new structural evidence stores exact candidate, bundle, Planner and contract identity');
ok(/contract_failures: effective\.contract_failures \|\| \[\]/.test(src),
  'structured contract failures reach the planner through the candidate registry');
ok(/contract_failures: normalizeContractFailures\(eng && eng\.contract_failures\)/.test(src) &&
   /evidence_manifest: String\(eng && eng\.evidence_manifest/.test(src),
  'incomplete Engineer feedback enters the registry instead of being dropped before persistence');
ok(/runtimeOnlyExactHeadFailure/.test(src) &&
   /next\.contract_failures\.length === 0/.test(src) &&
   /contract_failures: \[\]/.test(src),
  'runtime-only exact-head preservation cannot retain stale structural failures');
ok(/structural_state_conflict/.test(src) &&
   /structural_verified: c\.structural_verified === true && !structuralConflict/.test(src),
  'legacy structural pass plus required failures is normalized fail-closed');
ok(/failed_required_checks/.test(src) && /authoritativeIds/.test(src) &&
   /authoritativeIds\.has\(failure\.id\)/.test(src),
  'reference parity prose cannot become an authoritative contract failure');
ok(/verification_status: String\(v\.status/.test(src) &&
   /gpu_executed: String\(v\.activation_on_hardware/.test(src) &&
   /next_blocker: \(denomBad[\s\S]{0,900}\(verificationFailed/.test(src) &&
   /verificationFaulted \|\| eng/.test(src),
  'completed Verify failures persist as hardware evidence and fault the roadmap rung');
ok(/gpu_executed=true[\s\S]{0,180}runtime_verified=false/.test(lead) &&
   /candidates\/<id>\/candidate_result\.json/.test(lead),
  'state writer distinguishes on-card failure and mirrors the latest lane checkpoint');
ok(/const effectivePrior = existing \? megaEffectiveLaneCheckpoint\(existing\)/.test(src) &&
   /AUTHORING_STRUCTURAL_EVIDENCE_HEAD:\s*effectivePrior &&/.test(src) &&
   /PRIOR_CANDIDATE: priorForAgent/.test(src),
  'Engineer handoff uses the effective working HEAD and structural certificate');
ok(/structurallyReadyAuthoring/.test(src) &&
   /!authorPreflightRequired && !sourceAdvancedThisTurn/.test(src) &&
   /\(postAuthoringVerify \|\|/.test(src),
  'an unchanged exact-head structural checkpoint reaches independent runtime Verify');

console.log(failures === 0
  ? '\nPASS: Mega candidate lanes and calibration resume independently.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
