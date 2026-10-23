#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_mega.js'), 'utf8');

const match = src.match(
  /\/\/ <<REPLAY:mega_topology_contract>>([\s\S]*?)\/\/ <<\/REPLAY:mega_topology_contract>>/,
);
if (!match) throw new Error('megaTopologyVerdict not found');
// eslint-disable-next-line no-new-func
const { verdict, claimsTarget, needsPreflight } = new Function(
  `${match[1]}\nreturn { verdict: megaTopologyVerdict, ` +
  `claimsTarget: candidateClaimsPlanTarget, needsPreflight: authoringContractNeedsPreflight };`,
)();

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const plan = {
  target: {
    launch_count: 2,
    required_regions: ['producer', 'consumer'],
    required_queues: ['upstream', 'downstream'],
    required_capabilities: ['item_overlap'],
  },
};

ok(claimsTarget({ target_topology: { launch_count: 2 } }, plan),
  'v2 launch_count marks an exact full-target candidate');
ok(claimsTarget({ target_topology: { launches: 2 } }, plan),
  'the historical launches spelling remains a compatibility input');
ok(!claimsTarget({ target_topology: { launch_count: 3 } }, plan),
  'a partial target does not bypass the full-target structural contract');
ok(claimsTarget({ target_topology: { launch_count: 1 } }, plan),
  'fusing further than the planned launch count still claims the target');
{
  const fewer = verdict({ target_topology: { launch_count: 1,
    included_regions: ['producer', 'consumer'], included_queues: ['upstream', 'downstream'],
    capabilities: ['item_overlap'], require_overlap: true } }, plan, false);
  ok(fewer.pass, `a 1-launch terminal passes a 2-launch plan without partial fusion: ${fewer.reason || ''}`);
  const more = verdict({ target_topology: { launch_count: 3,
    included_regions: ['producer', 'consumer'], included_queues: ['upstream', 'downstream'],
    capabilities: ['item_overlap'], require_overlap: true } }, plan, false);
  ok(!more.pass && /partial fusion is disabled/.test(more.reason),
    'a 3-launch terminal is refused when partial fusion is disabled');
}
ok(needsPreflight(
  { target_topology: { launch_count: 2 } }, plan, null, true,
), 'a new full-target candidate is GPU-forbidden before structural Verify');
ok(needsPreflight(
  { target_topology: { launch_count: 2 } },
  plan,
  { head: 'new', structural_candidate_head: 'old', structural_verified: true },
  true,
), 'a structural pass for another HEAD cannot authorize authoring GPU use');
ok(!needsPreflight(
  { target_topology: { launch_count: 2 } },
  plan,
  { head: 'same', structural_candidate_head: 'same', structural_verified: true },
  true,
), 'the exact independently verified full-target HEAD may use a GPU');
const identity = {
  skillId: 'skill', revision: 'v6', bundle: 'bundle',
  planner: 'planner', contract: 'contract',
};
ok(!needsPreflight(
  { target_topology: { launch_count: 2 } },
  plan,
  {
    head: 'scored-old',
    working_head: 'working-new',
    working_snapshot: {
      head: 'working-new',
      structural_verified: true,
      structural_skill_id: 'skill',
      structural_candidate_head: 'working-new',
      structural_skill_bundle_sha256: 'bundle',
      structural_planner_extension_sha256: 'planner',
      structural_contract_revision: 'v6',
      structural_contract_sha256: 'contract',
    },
  },
  true,
  identity,
), 'an exact current-identity working certificate authorizes the next GPU turn');
ok(needsPreflight(
  { target_topology: { launch_count: 2 } },
  plan,
  {
    working_head: 'working-new',
    working_snapshot: {
      head: 'working-new', structural_verified: true,
      structural_candidate_head: 'working-new',
      structural_skill_id: 'skill', structural_contract_revision: 'v6',
      structural_skill_bundle_sha256: 'stale',
      structural_planner_extension_sha256: 'planner',
      structural_contract_sha256: 'contract',
    },
  },
  true,
  identity,
), 'a stale working bundle cannot authorize GPU use');
ok(!needsPreflight(
  { target_topology: { launch_count: 3 } }, plan, null, true,
), 'a complete partial fallback is not forced through the full-target Skill contract');

ok(!verdict({}, plan).pass,
  'a Mega direction cannot omit target_topology');
ok(!verdict({
  topology_inferred: true,
  target_topology: {
    launch_count: 2,
    included_regions: ['producer', 'consumer'],
  },
}, plan).pass,
'a compatibility fallback does not satisfy a strict planner contract');
ok(verdict({
  step_role: 'terminal',
  target_topology: {
    launch_count: 2,
    included_regions: ['producer', 'consumer'],
    included_queues: ['upstream', 'downstream'],
    capabilities: ['item_overlap'],
  },
}, plan).pass,
'an operator-neutral terminal topology matching MegaPlanIR passes');
ok(!verdict({
  step_role: 'terminal',
  target_topology: {
    launch_count: 2,
    included_regions: ['producer'],
    included_queues: ['upstream', 'downstream'],
    capabilities: ['item_overlap'],
  },
}, plan).pass,
'a terminal cannot omit a required region');
ok(!verdict({
  step_role: 'terminal',
  target_topology: {
    launch_count: 3,
    included_regions: ['producer', 'consumer'],
  },
}, plan).pass,
'an undeclared partial terminal cannot silently replace the full target');
ok(verdict({
  step_role: 'terminal',
  rung_deviation: 'measured partial fusion is intentionally evaluated first',
  target_topology: {
    launch_count: 3,
    included_regions: ['producer', 'consumer'],
  },
}, plan).pass,
'an explicit partial-terminal deviation remains legal');
ok(!verdict({
  step_role: 'terminal',
  rung_deviation: 'diagnostic fallback only',
  target_topology: {
    launch_count: 3,
    included_regions: ['producer', 'consumer'],
  },
}, plan, false).pass,
'exact-target campaigns reject partial terminals even when a deviation is declared');
ok(verdict({
  step_role: 'terminal',
  target_topology: {
    launch_count: 1,
    included_regions: ['producer', 'consumer'],
    included_queues: ['upstream', 'downstream'],
    capabilities: ['item_overlap'],
  },
}, plan, false).pass,
'the launch target is a ceiling: fewer launches than planned is not partial fusion');
ok(!verdict({
  step_role: 'terminal',
  target_topology: {
    launch_count: 1,
    included_regions: ['producer'],
    included_queues: ['upstream', 'downstream'],
    capabilities: ['item_overlap'],
  },
}, plan, false).pass,
'a terminal below the launch target must still include every required region');
ok(/const MEGA_PLAN_SCHEMA = \{[\s\S]*'candidate_id'[\s\S]*'candidate_source'[\s\S]*'target_topology'/.test(src) &&
   /label: `mega:plan r\$\{currentRound\}`, schema: MEGA_PLAN_SCHEMA/.test(src),
  'Mega plan StructuredOutput requires complete candidate identity and topology');
ok(/if \(!topologyVerdict\.pass\) \{[\s\S]{0,280}return \{[\s\S]{0,120}Mega direction refused before authoring/.test(src),
  'every Mega run rejects an invalid topology before Engineer dispatch');
ok(/refused: true, stop: MEGA_ROUTE_ONLY,\s*reason: `Mega direction refused/.test(src) &&
   /refused: true, stop: MEGA_ROUTE_ONLY, reason:\s*`strict Mega direction refused/.test(src),
  'a refused direction ends only route-only runs; production re-plans');
ok(/if \(turn\.refused && !turn\.stop\) \{[\s\S]{0,200}dispatched \+= 1;[\s\S]{0,200}if \(count >= 3\) \{[\s\S]{0,160}break;[\s\S]{0,40}continue;/.test(src),
  'refusals are charged to the budget and three consecutive refusals still stop');
ok(/megaRefusal = null;\s*if \(turn\.stop\)/.test(src) &&
   /\.\.\.\(megaRefusal \? \{ PREVIOUS_PLAN_REFUSED: megaRefusal\.reason \} : \{\}\)/.test(src),
  'the refusal reason reaches the next plan and resets after an accepted direction');
ok(/const drainS = laneWriterTimedOut \? 1500 : 0;/.test(src) &&
   /timeout_ms: \(180 \+ drainS\) \* 1000/.test(src) &&
   /flock -w <left> \$\{laneLock\} true/.test(src) && /team_gpu_locks\}\/gpu_\*\.lock/.test(src) &&
   /non-zombie process/.test(src) && /under \$\{laneDir\} or \$\{outDir\}/.test(src) &&
   /max_retries: 1 \} : \{\}\) \}\);\s*megaAdvanceMs\(drainS \* 1000\);/.test(src),
  'a timed-out author gets a bounded, charged drain on the lane lock, GPU leases and lane processes');
ok(/if \(laneWriterTimedOut && recovered && !recovered\.__agent_timed_out &&\s*recovered\.writer_quiescent === true\) \{\s*laneWriterTimedOut = false;\s*megaUnsafeTimeout = null;/.test(src) &&
   /if \(laneWriterTimedOut\) \{\s*await persistMegaCandidateState\(currentRound\);\s*return \{\s*stop: true,/.test(src),
  'only a quiescent timed-out writer continues the wave; a live one still stops it');
ok(/megaTopologyVerdict\(\s*d, analysis && analysis\.mega_plan_ir, ALLOW_PARTIAL_FUSION\)/.test(src),
  'the caller-owned exact-target policy reaches the pre-author topology gate');
const byBucket = {
  target: plan.target,
  target_by_bucket: {
    64: { launch_count: 1, required_regions: ['merged'], required_queues: ['merged_q'],
      required_capabilities: ['single_launch'] },
  },
};
const small = {
  step_role: 'terminal', bucket: 64,
  target_topology: { launch_count: 1, included_regions: ['merged'], included_queues: ['merged_q'],
    capabilities: ['single_launch'] },
};
ok(verdict(small, byBucket, false).pass,
  'a direction naming its bucket is held to that bucket\'s target');
ok(!verdict({ ...small, bucket: 8192 }, byBucket, false).pass,
  'a bucket without its own target falls back to the shared target');
ok(!verdict({ ...small, bucket: undefined }, byBucket, false).pass &&
   verdict({ ...small, bucket: undefined, target_topology: {
     launch_count: 2, included_regions: ['producer', 'consumer'],
     included_queues: ['upstream', 'downstream'], capabilities: ['item_overlap'] } }, byBucket, false).pass,
  'a direction without a bucket keeps the shared-target behaviour');

ok(/allow_partial_fusion=false/.test(src) &&
   /SEARCH_ACCEPTS_PARTIAL_FUSION: ALLOW_PARTIAL_FUSION \? '1' : '0'/.test(src) &&
   /allowPartialFusion: ALLOW_PARTIAL_FUSION/.test(src),
  'pinned exact-target policy also reaches score verification and promotion');

console.log(failures ? `\nFAILED: ${failures}` :
  '\nPASS: Mega topology is an operator-neutral typed plan contract.');
process.exit(failures ? 1 : 0);
