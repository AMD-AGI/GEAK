#!/usr/bin/env node
// A direction is a search and its evidence follows the candidate, not the claim: a climb `negative` owes
// `ceiling` + `lesson` (not the gates, not the audit); a `negative` that verify puts above the kept line is
// promoted and owes every gate of a win; the ceiling verdict reaches the next plan. Both modes.
'use strict';

const fs = require('fs');
const path = require('path');
const { launch, ok, done, engineerReturn, planReturn, SKILL } = require('./harness');

const inc = { entry: 'incumbent' };
const strict = { entry_mode: 'incumbent', budget: 1 };
const ceiling = { command: 'python3 run.py --bench (stubbed scale loads)', exit_code: 0, removed: 'the per-tile scale loads',
  headroom_pct: 6.5, output_path: '/x/ceiling.txt', verdict: 'headroom_remains' };
// A light negative: no profile, no G3/G4/ATT, only the search record.
const negative = (prompt, over) => {
  const r = engineerReturn(prompt, { claim: 'negative', ceiling, lesson: 'the mechanism lost while registers bind', steps_tried: 9, ...over });
  for (const k of ['profile_dir', 'profile_command', 'g3_occupancy', 'g4_screen', 'att']) delete r[k];
  return r;
};
const below = { status: 'verified', correctness: 'pass', verified_geomean: 0.98 };

(async () => {
  console.log('\n# 1. a negative owes ceiling + lesson, not the gates, and is not audited');
  for (const mode of ['strict', 'advisory']) {
    const x = await launch({ ...strict, method_strictness: mode }, (label, prompt) => {
      if (label === 'gluon_engineer:climb r1_d0') return negative(prompt);
      if (label === 'verify r1_d0') return below;
      return undefined;
    }, inc);
    const row = x.result.rounds[0];
    ok(row.status === 'verified' && !x.labels.some(l => /reask|evidence_audit/.test(l)), `${mode}: complete as returned, no reask, no audit`, x.labels);
    ok(row.ceiling && row.ceiling.verdict === 'headroom_remains' && row.steps_tried === 9, `${mode}: the ceiling and steps ride the history row`, row);
  }
  {
    const x = await launch(strict, (label, prompt) => {
      if (/^gluon_engineer:/.test(label)) return negative(prompt, { ceiling: undefined });
      if (label === 'verify r1_d0') return below;
      return undefined;
    }, inc);
    const reask = x.calls.find(c => c.label === 'gluon_engineer:evidence r1_d0 (reask 1)');
    ok(reask && /ceiling \(the targeted work removed or stubbed/.test(reask.prompt), 'no ceiling: asked for it by name');
    ok(x.result.rounds[0].status === 'rejected_missing_evidence', 'still none: rejected', x.result.rounds[0]);
  }
  {
    const short = await launch(strict, (label, prompt) => /^gluon_engineer:/.test(label)
      ? negative(prompt, { ceiling: { verdict: 'not_applicable', reason: 'n/a' } }) : label === 'verify r1_d0' ? below : undefined, inc);
    ok(short.result.rounds[0].missing.some(m => /ceiling\.reason/.test(m)), 'not_applicable needs a real reason', short.result.rounds[0]);
    const long = await launch(strict, (label, prompt) => /^gluon_engineer:/.test(label)
      ? negative(prompt, { ceiling: { verdict: 'not_applicable', reason: 'the target is the only work in the loop body; removing it removes the loop' } })
      : label === 'verify r1_d0' ? below : undefined, inc);
    ok(long.result.rounds[0].status === 'verified', 'not_applicable with a reason is complete', long.result.rounds[0]);
    const noLesson = await launch(strict, (label, prompt) => /^gluon_engineer:/.test(label)
      ? negative(prompt, { lesson: '' }) : label === 'verify r1_d0' ? below : undefined, inc);
    ok(noLesson.result.rounds[0].missing.some(m => /^lesson/.test(m)), 'a negative owes its lesson', noLesson.result.rounds[0]);
  }
  {
    const x = await launch(strict, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, [{ specialty: 'memory', layer: '4', paired: true }]);
      if (label === 'gluon_engineer:climb r1_d0') return negative(prompt, { layer: '4' });
      if (label === 'verify r1_d0') return below;
      return undefined;
    }, inc);
    ok(x.result.rounds[0].status === 'verified', 'a paired negative owes no pair screen and no ATT', x.result.rounds[0]);
  }

  console.log('\n# 2. a negative that verify puts above the kept line is promoted');
  {
    const x = await launch(strict, (label, prompt) => {
      if (label === 'gluon_engineer:climb r1_d0') return negative(prompt);
      if (label === 'gluon_engineer:evidence r1_d0 (promoted) (reask 1)') return engineerReturn(prompt, { claim: 'win' });
      return undefined;   // verify 1.1 > the kept line 1.0
    }, inc);
    const ask = x.calls.find(c => c.label === 'gluon_engineer:evidence r1_d0 (promoted) (reask 1)');
    ok(ask && /it is a candidate after all/.test(ask.prompt) && /g4_screen/.test(ask.prompt) && /profile_dir/.test(ask.prompt), 'asked for the gates of a win by name');
    ok(x.labels.includes('evidence_audit r1_d0 (promoted)'), 'and audited');
    ok(x.result.accepted && x.result.accepted.id === 'r1_d0' && x.result.rounds[0].claim === 'win', 'then accepted on verify\'s number', x.result.accepted);
  }
  {
    const x = await launch(strict, (label, prompt) => /^gluon_engineer:/.test(label) ? negative(prompt) : undefined, inc);
    const row = x.result.rounds[0];
    ok(row.status === 'rejected_missing_evidence' && !x.result.accepted, 'gates still missing after the ask: rejected, not accepted', row);
    ok(x.result.outcome === 'negative_keep_baseline', 'nothing delivered', x.result.outcome);
  }
  {
    const x = await launch({ ...strict, evidence_reasks: 0 }, (label, prompt) => {
      if (label === 'gluon_engineer:climb r1_d0') return negative(prompt);
      if (/promoted/.test(label)) return engineerReturn(prompt, { claim: 'win' });
      return undefined;
    }, inc);
    ok(x.labels.includes('gluon_engineer:evidence r1_d0 (promoted) (reask 1)') && x.result.accepted, 'promotion asks once even with evidence_reasks=0', x.labels);
  }
  {
    const x = await launch(strict, (label, prompt) => /^gluon_engineer:/.test(label)
      ? engineerReturn(prompt, { g4_screen: undefined }) : undefined, inc);
    ok(x.result.rounds[0].missing.some(m => /g4_screen/.test(m)), 'a claimed win still owes G4', x.result.rounds[0]);
  }

  console.log('\n# 3. the planner and the engineer are told');
  {
    const x = await launch({ ...strict, budget: 2 }, (label, prompt) => {
      if (label === 'gluon_engineer:climb r1_d0') return negative(prompt);
      if (label === 'verify r1_d0') return below;
      return undefined;
    }, inc);
    const p1 = x.calls.find(c => c.label === 'tech_lead:plan r1').prompt;
    ok(/Each direction is a search/.test(p1) && /headroom_remains` means the MECHANISM failed/.test(p1) && /shape-regime specialization/.test(p1),
      'strict plan addendum: search, ceiling semantics, cross-layer levers');
    const p2 = x.calls.find(c => c.label === 'tech_lead:plan r2');
    ok(p2 && /"verdict":"headroom_remains"/.test(p2.prompt), 'the next plan sees the ceiling in HISTORY');
    const mem = x.calls.find(c => c.label === 'tech_lead:memory r1').prompt;
    ok(/"ceiling":\{"verdict":"headroom_remains"/.test(mem), 'update_memory sees it too');
    const eng = x.calls.find(c => c.label === 'gluon_engineer:climb r1_d0').prompt;
    ok(/claim negative \(no step beat TRACK_START\): layer, `ceiling`/.test(eng) && /## Before a negative: the ceiling probe/.test(eng),
      'the engineer prompt states what a negative owes and where the probe is described');
  }
  {
    const climb = fs.readFileSync(path.join(SKILL, 'references', 'method', 'climb.md'), 'utf8');
    ok(/^## Before a negative: the ceiling probe$/m.test(climb) && /^## Levers that are not a layer's headline$/m.test(climb), 'the cited climb.md sections exist');
    const role = fs.readFileSync(path.join(__dirname, '..', 'roles', 'gluon_engineer.md'), 'utf8');
    ok(/^## Searching inside a direction/m.test(role) && !/make one attributable change, and land/.test(role), 'the role searches; one change is per step');
    const pages = [role, climb, fs.readFileSync(path.join(__dirname, '..', 'workflow.js'), 'utf8')].join('\n');
    ok(!/fused_moe|a_scale|w8a8|DeepSeek|1\.3\d\dx/i.test(pages.replace(/fused_moe: 'moe'/, '')), 'no kernel-specific data in the method text');
  }
  done('direction-search');
})();
