#!/usr/bin/env node
// A resumed wave whose analyze fast path finds nothing must not proceed with an empty ladder.
//
// roles/mega/tech_lead.md lets analyze skip the cold re-read when INCREMENTAL_RESUME is set, on the
// condition that it does the full analysis if no prior roadmap exists. kernel_mega.js cannot
// check that condition -- it has no filesystem -- so the only observable is the returned schema.
//
// One run: the caller supplied a fresh EVAL_DIR, the prior roadmap was not in it, and the fast
// path returned a valid schema with no candidate_directions, and three rounds ran with no ladder.
// The rung ids were carried forward by hand from an earlier run by individual engineers, and D2 went
// unspent for three waves. The LADDER MISSING caveat fired every round and changed nothing.
//
// Each assertion below is a way this fallback could be wrong in a way nobody would notice:
//   - it fires on a NON-resumed run, turning every genuinely-ladderless first wave into a double
//     analyze call (cost, and it would mask the real finding)
//   - it fails to fire on the exact wave-15 shape
//   - it fires when a ladder IS present, i.e. it is reading the wrong field
//   - it is defined but never called, or called after the ladder gate has already read `analysis`
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

const m = src.match(/\/\/ <<REPLAY:analyze_resume_fallback>>([\s\S]*?)\/\/ <<\/REPLAY:analyze_resume_fallback>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:analyze_resume_fallback>> region — nothing to test'); process.exit(1); }
const { analyzeResumeDegenerate, megaPlanIRVerdict } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { analyzeResumeDegenerate, megaPlanIRVerdict };`)();

const LADDER = { candidate_directions: [
  { id: 'D0', title: 'D0 Instrumentation' },
  { id: 'D2', title: 'D2 Per-token readiness' },
] };
const MEGA_COMPLETE = {
  ...LADDER,
  task_graph: { nodes: [{ id: 'g1' }], edges: [{ from: 'g1', to: 'g2' }] },
  resource_timeline: { pipes: [{ stage: 'g1', pipe: 'mfma', utilization_pct: null }] },
  mega_plan_ir: {
    plan_version: 'mega-plan-v2',
    expert_skill_id: 'skill',
    expert_skill_revision: 'skill-v1',
    expert_skill_bundle_sha256: 'b'.repeat(64),
    expert_skill_planner_extension_sha256: 'p'.repeat(64),
    expert_skill_validation_status: 'experimental',
    expert_skill_auto_apply: false,
    expert_skill_explicit_pin_modes: ['authoring', 'candidate_validation'],
    target: {
      launch_count: 1, required_regions: ['producer', 'consumer'],
      required_queues: ['work'],
    },
    work_domains: [{ id: 'items' }],
    regions: [
      { id: 'producer', work_domain: 'items' },
      { id: 'consumer', work_domain: 'items' },
    ],
    buffers: [{
      id: 'payload', producers: ['producer'], consumers: ['consumer'],
    }],
    counters: [{
      id: 'ready', producers: ['producer'], consumers: ['consumer'],
    }],
    queues: [{ id: 'work', work_domain: 'items' }],
    events: [{
      id: 'ready_event', producer: 'producer', consumer: 'consumer', counter: 'ready',
    }],
    resources: {
      arch: 'test-arch',
      workgroup: { wave_size: 64, thread_count: 256 },
      local_memory: { total_bytes: 1024, limit_bytes: 2048 },
    },
    schedule: {
      primary_loop: {
        kind: 'unified', carried_state: ['active'], queue_priority: ['work'],
      },
    },
    abi: { entry_point: 'kernel', arguments: [{ id: 'payload' }] },
    compiler_constraints: [{ id: 'shape' }],
    evidence_requirements: { accuracy: 'required' },
    known_unknowns: [],
  },
};

console.log('\n# the wave-15 shape');
{
  const d = analyzeResumeDegenerate(true, { kernel_type: 'flydsl', candidate_directions: [] }, false);
  ok(d.retry === true, 'a resumed analyze that returns an empty ladder triggers the full re-run');
  ok(/ANALYZE RESUME DEGENERATE/.test(d.reason) && /candidate_directions/.test(d.reason),
     'and the reason names the flag and the artifact, so the log says what happened rather than that something happened');
}
{
  ok(analyzeResumeDegenerate(true, {}, false).retry === true,
     'a missing candidate_directions key is the same failure as an empty one');
  ok(analyzeResumeDegenerate(true, null, false).retry === true,
     'and so is no analysis object at all');
  ok(analyzeResumeDegenerate(true, { candidate_directions: [{}, { title: '' }] }, false).retry === true,
     'rungs with neither id nor title are not a ladder — this is the same filter the ladder gate uses');
}
{
  const incomplete = analyzeResumeDegenerate(false, LADDER, true);
  ok(incomplete.retry === true && /task_graph\/resource_timeline/.test(incomplete.reason),
    'a fresh Mega analysis missing its required structured graph is repaired before Benchmark');
}
{
  const incomplete = analyzeResumeDegenerate(
    false,
    { ...MEGA_COMPLETE, mega_plan_ir: null },
    true,
    true,
  );
  ok(incomplete.retry === true && /mega_plan_ir/.test(incomplete.reason),
    'a Mega analysis without a lowerable typed plan is repaired before Benchmark');
  ok(analyzeResumeDegenerate(false, MEGA_COMPLETE, true, true).retry === false,
    'a complete graph/resource/typed-plan contract proceeds without another Analyze');
  ok(analyzeResumeDegenerate(false, MEGA_COMPLETE, true, true, 'skill-v1').retry === false,
    'a PlanIR bound to the active Planner Extension revision proceeds');
  ok(analyzeResumeDegenerate(
    false, MEGA_COMPLETE, true, true, 'skill-v1', 'skill', 'p'.repeat(64), 'b'.repeat(64),
    'experimental', 'authoring',
  ).retry === false,
  'a PlanIR bound to complete experimental bundle metadata proceeds');
  const forgedStatus = JSON.parse(JSON.stringify(MEGA_COMPLETE));
  forgedStatus.mega_plan_ir.expert_skill_validation_status = 'validated';
  const forged = analyzeResumeDegenerate(
    false, forgedStatus, true, true,
    'skill-v1', 'skill', 'p'.repeat(64), 'b'.repeat(64),
    'experimental', 'authoring',
  );
  ok(forged.retry === true && /validation_status/.test(forged.reason),
    'a caller cannot relabel an experimental bundle as validated');
  const staleSkillPlan = JSON.parse(JSON.stringify(MEGA_COMPLETE));
  staleSkillPlan.mega_plan_ir.expert_skill_revision = 'stale-skill';
  const stale = analyzeResumeDegenerate(false, staleSkillPlan, true, true, 'skill-v1');
  ok(stale.retry === true && /expert_skill_revision/.test(stale.reason),
    'a PlanIR produced from a stale Planner Extension revision is rejected');
  const staleBundlePlan = JSON.parse(JSON.stringify(MEGA_COMPLETE));
  staleBundlePlan.mega_plan_ir.expert_skill_bundle_sha256 = 'x'.repeat(64);
  const staleBundle = analyzeResumeDegenerate(
    false, staleBundlePlan, true, true,
    'skill-v1', 'skill', 'p'.repeat(64), 'b'.repeat(64),
  );
  ok(staleBundle.retry === true && /bundle_sha256/.test(staleBundle.reason),
    'same-revision Planner knowledge drift is rejected by bundle digest');
  const dangling = JSON.parse(JSON.stringify(MEGA_COMPLETE));
  dangling.mega_plan_ir.events[0].counter = 'missing_counter';
  const invalid = analyzeResumeDegenerate(false, dangling, true, true);
  ok(invalid.retry === true && /unknown counter/.test(invalid.reason),
    'relationally invalid PlanIR is rejected even when every collection is populated');
  const misplacedOwned = JSON.parse(JSON.stringify(MEGA_COMPLETE.mega_plan_ir));
  misplacedOwned.work_domains[0].parameters = {
    owned_subdomain: {
      unit: 'subtile', extent_per_claim: 'n', linear_index: 'item*n+subtile',
    },
  };
  const misplacedVerdict = megaPlanIRVerdict(misplacedOwned);
  ok(!misplacedVerdict.pass &&
     misplacedVerdict.errors.some((e) => /owned_subdomain must be direct/.test(e)),
  'owned subdomains nested under parameters fail before contract verification');
  delete misplacedOwned.work_domains[0].parameters;
  misplacedOwned.work_domains[0].owned_subdomain = {
    unit: 'subtile', extent_per_claim: 'n', linear_index: 'item*n+subtile',
  };
  ok(megaPlanIRVerdict(misplacedOwned).pass,
    'a complete direct owned_subdomain passes generic PlanIR relations');
}

console.log('\n# it must not fire anywhere else');
{
  ok(analyzeResumeDegenerate(false, { candidate_directions: [] }, false).retry === false,
     'a FIRST wave with no ladder is a real finding and must not be retried into silence');
  ok(analyzeResumeDegenerate(true, LADDER, false).retry === false,
     'a resumed wave that did inherit its ladder proceeds untouched — the fast path is the point');
  ok(!analyzeResumeDegenerate(true, LADDER, false).reason,
     'and says nothing, so the log is not noisy on the normal path');
}

console.log('\n# it is wired in, once, and before anything reads the ladder');
{
  ok(/const d = analyzeResumeDegenerate\(\s*\n?\s*INCREMENTAL, analysis, MODE === 'mega'/.test(src),
     'the check is called with the resume flag and the analyze result');
  ok(/let analysis = await agentT\(/.test(src),
     '`analysis` is a let, so the recovered full analysis can replace the degenerate one');
  ok(/tech_lead:analyze:full/.test(src),
     'the re-run gets its own label — two analyze calls that look identical in the progress tree are unreadable');
  const reRun = src.indexOf("'tech_lead:analyze:full'");
  const ladderRead = src.indexOf('const LADDER = (analysis');
  ok(reRun > 0 && ladderRead > reRun,
     'the re-run happens BEFORE LADDER is computed — recovering a ladder nothing reads is not a fix');
  ok((src.match(/tech_lead:analyze:full/g) || []).length === 1,
     'and it happens once: a retry loop against a phase this expensive is how a wave spends its budget on nothing');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
