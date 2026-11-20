#!/usr/bin/env node
// method_strictness=advisory relaxes the METHOD in recover/climb/integrate (reading list, ATT for the
// pipeline layers, both pair screens, one-change / no-branch rules) and never the measurement (verify,
// edge_shapes, G3, G4, the re-profile of a win). strict is the default. What a climb `negative` owes
// (ceiling + lesson, no audit) is the same in both modes: test_direction_search.js.
'use strict';

const { launch, ok, done, engineerReturn, planReturn, goodEdge } = require('./harness');

const inc = { entry: 'incumbent' };
const adv = { entry_mode: 'incumbent', budget: 1, method_strictness: 'advisory' };
const strict = { entry_mode: 'incumbent', budget: 1 };

(async () => {
  console.log('\n# 1. what advisory relaxes');
  {
    const noReads = (label, prompt) => /^gluon_engineer:/.test(label)
      ? engineerReturn(prompt, { layer: '4', references_read: [] }) : undefined;   // no reads, no ATT
    const a = await launch(adv, noReads, inc);
    ok(a.result.rounds[0].status === 'verified' && !a.labels.some(l => /reask/.test(l)), 'no reading list and no ATT on layer 4: accepted', a.labels);
    const s = await launch(strict, noReads, inc);
    ok(s.result.rounds[0].status === 'rejected_missing_evidence', 'the same return is rejected under strict', s.result.rounds[0]);
    const eng = a.calls.find(c => c.label === 'gluon_engineer:climb r1_d0').prompt;
    ok(/## Recommended reading/.test(eng) && /METHOD_STRICTNESS: advisory/.test(eng) && /att \(recommended/.test(eng), 'the prompt says recommended');
    const plan = a.calls.find(c => c.label === 'tech_lead:plan r1').prompt;
    ok(/METHOD_STRICTNESS=advisory: a direction may bundle related changes/.test(plan) && /shape-regime specialization/.test(plan), 'the planner is told what is open');
  }
  {
    const paired = (over) => (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, [{ specialty: 'memory', layer: '4', paired: true }]);
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { layer: '4', ...over });
      return undefined;
    };
    const a = await launch(adv, paired({ pair_screen: { outcome: 'prerequisite' } }), inc);
    ok(a.result.rounds[0].status === 'verified', 'a pair owes only its outcome', a.result.rounds[0]);
    const b = await launch(adv, paired({ pair_screen: {} }), inc);
    ok(b.result.rounds[0].missing.some(m => /pair_screen\.outcome/.test(m)), 'but the outcome is still owed', b.result.rounds[0]);
  }
  {
    const b = await launch(adv, undefined, inc);
    ok(b.labels.includes('evidence_audit r1_d0'), 'a claimed win is still audited');
  }

  console.log('\n# 2. what advisory never relaxes');
  {
    const run = (over) => launch(adv, (label, prompt) => /^gluon_engineer:/.test(label) ? engineerReturn(prompt, over(prompt)) : undefined, inc);
    const g4 = await run(() => ({ g4_screen: undefined }));
    ok(g4.result.rounds[0].missing.some(m => /g4_screen/.test(m)), 'G4 still owed');
    const g3 = await run(() => ({ g3_occupancy: undefined }));
    ok(g3.result.rounds[0].missing.some(m => /g3_occupancy/.test(m)), 'G3 still owed');
    const prof = await run(() => ({ profile_command: 'bash profile_kernel.sh' }));
    ok(prof.result.rounds[0].missing.some(m => /profile_command/.test(m)), 'the gluon_profile.py re-profile still owed');
    const edge = await run(() => ({ edge_shapes: { ...goodEdge, exit_code: 1 } }));
    ok(edge.result.rounds[0].status === 'rejected_missing_evidence', 'a boundary-shape regression still rejects');
    const wrong = await launch(adv, (label) => label === 'verify r1_d0' ? { status: 'correctness_failed', correctness: 'fail', verified_geomean: 0 } : undefined, inc);
    ok(!wrong.result.accepted, 'verify still decides');
    const tr = await launch({ budget: 1, method_strictness: 'advisory' }, (label, prompt) =>
      /^gluon_engineer:transcribe/.test(label) ? engineerReturn(prompt, { references_read: [] }) : undefined, { entry: 'port' });
    ok(tr.labels.includes('gluon_engineer:evidence r1 (reask 1)'), 'transcribe keeps its reading list (a method stage only for recover/climb/integrate)', tr.labels);
  }

  console.log('\n# 3. default and validation');
  {
    const x = await launch(strict, undefined, inc);
    ok(x.result.method_strictness === 'strict' && /## Required reading/.test(x.calls.find(c => c.label === 'gluon_engineer:climb r1_d0').prompt), 'strict by default');
    const bad = await launch({ ...strict, method_strictness: 'loose' });
    ok(/method_strictness must be/.test(bad.error || '') && bad.calls.length === 0, 'unknown value refused up front', bad.error);
  }
  done('method-strictness');
})();
