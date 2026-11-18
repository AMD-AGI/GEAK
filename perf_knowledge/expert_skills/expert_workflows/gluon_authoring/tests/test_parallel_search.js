#!/usr/bin/env node
// Recover and climb run kernel_lane.js's search: tech_lead plans the round's directions, gluon_engineers
// carry them in parallel (each with its evidence gates and its own verify), an integrate step stacks the
// winners, the budget counts directions, and tech_lead's memory feeds the next plan.
'use strict';

const { launch, ok, done, inOrder, engineerReturn, planReturn, integrateReturn, gate, goodAtt, WF_DIR } = require('./harness');

const climb = { entry_mode: 'incumbent' };
const inc = { entry: 'incumbent' };
const ver = (v) => ({ status: 'verified', correctness: 'pass', verified_geomean: v });
const three = [{ specialty: 'compute', layer: '7' }, { specialty: 'memory', layer: '2' }, { specialty: 'host_runtime', layer: '6' }];

(async () => {
  console.log('\n# 1. a round: plan -> directions in parallel -> integrate -> verify -> keep/accept -> memory');
  {
    const x = await launch({ ...climb, budget: 6, max_no_improve: 9 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, three);
      if (label === 'verify r1_d0') return ver(1.12);
      if (label === 'verify r1_d1') return ver(1.10);
      if (label === 'verify r1_d2') return ver(1.05);
      if (label === 'verify r1_int') return ver(1.30);
      if (/^gluon_engineer:climb r1_d(\d)$/.test(label)) {
        const i = label.slice(-1);
        return engineerReturn(prompt, { layer: ['7', '2', '6'][i], profile_dir: `/x/prof_d${i}` });
      }
      return undefined;
    }, inc);
    ok(!x.error, 'runs', x.error);
    ok(inOrder(x.labels, ['tech_lead:plan r1', 'gluon_engineer:climb r1_d0', 'verify r1_d0', 'integrate r1', 'verify r1_int',
      'gluon_profiler:reprofile r1', 'tech_lead:memory r1', 'tech_lead:plan r2']), 'round order', x.labels);
    for (const i of [0, 1, 2]) ok(x.labels.includes(`gluon_engineer:climb r1_d${i}`) && x.labels.includes(`verify r1_d${i}`), `direction d${i} carried and verified`);
    const integ = x.calls.find(c => c.label === 'integrate r1').prompt;
    ok(/PHASE=integrate/.test(integ) && integ.includes(`INTEGRATOR_ROLE: ${WF_DIR}/roles/integrator.md`), 'integrate follows kernel_workflow integrator.md');
    ok(/"tag":"r1_d0"/.test(integ) && /"tag":"r1_d1"/.test(integ) && /"tag":"r1_d2"/.test(integ), 'all three improving directions are integration candidates');
    ok(/BEST_INDIVIDUAL: 1\.12/.test(integ), 'the bar is the best individual');
    const r1 = x.result.rounds.filter(r => r.round === 1);
    ok(r1.length === 4 && r1.find(r => r.id === 'r1_int').kept && r1.find(r => r.id === 'r1_int').accepted, 'the integrated candidate is kept and accepted', r1);
    const p2 = x.calls.find(c => c.label === 'gluon_engineer:climb r2_d0').prompt;
    ok(/TRACK_START: r1_int/.test(p2) && /LAST_PROFILE: \/x\/reprofile/.test(p2), 'round 2 starts from the integrated tag and its profile');
    const plan2 = x.calls.find(c => c.label === 'tech_lead:plan r2').prompt;
    ok(/an insight/.test(plan2) && /"id":"r1_d2"/.test(plan2), 'the next plan reads the memory and every direction of round 1');
    ok(x.result.budget_used === 6 && x.result.rounds_used === 4, 'budget counts directions, not rounds (3 + 1 + 1 + 1)', [x.result.budget_used, x.result.rounds_used]);
  }

  console.log('\n# 2. integration that does not beat the best individual: the single direction wins');
  {
    const x = await launch({ ...climb, budget: 2 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, three.slice(0, 2));
      if (label === 'verify r1_d0') return ver(1.12);
      if (label === 'verify r1_d1') return ver(1.10);
      if (label === 'verify r1_int') return ver(1.11);
      if (label === 'gluon_engineer:climb r1_d0') return engineerReturn(prompt, { layer: '7', profile_dir: '/x/prof_d0' });
      if (label === 'gluon_engineer:climb r1_d1') return engineerReturn(prompt, { layer: '2' });
      return undefined;
    }, inc);
    ok(x.result.accepted.id === 'r1_d0' && !x.labels.some(l => /reprofile/.test(l)), 'r1_d0 accepted; its own profile is the kept line', x.result.accepted);
    const y = await launch({ ...climb, budget: 2 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, three.slice(0, 2));
      if (label === 'verify r1_d0') return ver(1.12);
      if (label === 'verify r1_d1') return ver(0.97);
      if (label === 'gluon_engineer:climb r1_d1') return engineerReturn(prompt, { layer: '2' });
      return undefined;
    }, inc);
    ok(!y.labels.includes('integrate r1'), 'one improving direction: nothing to integrate', y.labels);
    const z = await launch({ ...climb, budget: 2 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, three.slice(0, 2));
      if (label === 'gluon_engineer:climb r1_d1') return engineerReturn(prompt, { layer: '2' });
      if (/^integrate r1/.test(label)) return integrateReturn(prompt, { g4_screen: undefined });
      return undefined;
    }, inc);
    ok(z.labels.includes('integrate r1 (reask 1)') && !z.labels.includes('verify r1_int'), 'an integration without its permuted screen is re-asked, then dropped', z.labels);
    ok(z.result.rounds.find(r => r.id === 'r1_int').status === 'rejected_missing_evidence', 'and recorded', z.result.rounds);
  }

  console.log('\n# 3. budget: the plan is cut to what is left; max_directions caps a round');
  {
    const x = await launch({ ...climb, budget: 5, max_no_improve: 9 }, (label, prompt) => {
      if (/^tech_lead:plan/.test(label)) return planReturn(prompt, three);
      return undefined;
    }, inc);
    ok(x.labels.includes('gluon_engineer:climb r2_d1') && !x.labels.includes('gluon_engineer:climb r2_d2'), 'round 2 gets the 2 directions left', x.labels);
    ok(x.result.budget_used === 5 && x.result.stopped_by === 'budget', 'budget spent exactly', x.result);
    const plan2 = x.calls.find(c => c.label === 'tech_lead:plan r2').prompt;
    ok(/BUDGET_REMAINING: 2\b/.test(plan2), 'tech_lead sees the remaining budget');
    const y = await launch({ ...climb, budget: 3, max_directions: 1, max_no_improve: 9 }, (label, prompt) => {
      if (/^tech_lead:plan/.test(label)) return planReturn(prompt, three);
      return undefined;
    }, inc);
    ok(!y.labels.some(l => /_d1$/.test(l)) && y.result.rounds_used === 3, 'max_directions 1: one direction per round', y.labels);
  }

  console.log('\n# 4. deep_explore runs alone and costs DEEP_COST');
  {
    const x = await launch({ ...climb, budget: 3 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, [{ specialty: 'memory', layer: '2' }, { specialty: 'deep_explore', layer: '4' }]);
      if (/^gluon_engineer:climb r1_d0/.test(label)) return engineerReturn(prompt, { layer: '4', att: goodAtt });
      return undefined;
    }, inc);
    const eng = x.calls.find(c => c.label === 'gluon_engineer:climb r1_d0').prompt;
    ok(/deep_explore/.test(eng) && !x.labels.includes('gluon_engineer:climb r1_d1'), 'only the deep direction runs', x.labels);
    ok(x.logs.some(l => /Round 1: 1 direction\(s\) \[deep_explore L4\], cost 2, budget 2\/3/.test(l)), 'charged 2', x.logs);
  }

  console.log('\n# 5. the layer gating order holds inside a round');
  {
    const x = await launch({ ...climb, budget: 6, max_no_improve: 1 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, [{ specialty: 'compute', layer: '1.5' }, { specialty: 'memory', layer: '4' },
        { specialty: 'memory', layer: '3' }, { specialty: 'compute', layer: '1.5' }]);
      if (/^gluon_engineer:climb r1_d0/.test(label)) return engineerReturn(prompt, { layer: '1.5', att: goodAtt });
      return undefined;
    }, inc);
    const r1 = x.result.rounds.filter(r => r.round === 1 && r.id !== 'r1_int');
    ok(r1.length === 2 && x.logs.filter(l => /deferred .*gate each other/.test(l)).length === 2, 'layer 4 and a second 1.5 are deferred', x.logs);
    ok(x.logs.some(l => /cost 2, budget 2\/6/.test(l)), 'deferred directions are not charged', x.logs);
    const y = await launch({ ...climb, budget: 2 }, (label, prompt) =>
      label === 'tech_lead:plan r1' ? planReturn(prompt, [{ specialty: 'memory', layer: '3' }, { specialty: 'memory', layer: '4' }]) : undefined, inc);
    ok(!y.labels.includes('gluon_engineer:climb r1_d1') && y.logs.some(l => /\(layer 4\): it and layer 3/.test(l)), '3 before 4: layer 4 waits', y.logs);
  }

  console.log('\n# 6. a paired direction owes the prerequisite-alone and pair screens');
  {
    const paired = [{ specialty: 'memory', layer: '4', paired: true }];
    const x = await launch({ ...climb, budget: 1 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, paired);
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { layer: '4', att: goodAtt });
      return undefined;
    }, inc);
    ok(x.labels.includes('gluon_engineer:evidence r1_d0 (reask 1)') && x.result.rounds[0].status === 'rejected_missing_evidence' &&
      x.result.rounds[0].missing.some(m => /pair_screen/.test(m)), 'without pair_screen: re-asked, then rejected', x.result.rounds[0]);
    const y = await launch({ ...climb, budget: 1 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, paired);
      if (/^gluon_engineer:/.test(label)) return engineerReturn(prompt, { layer: '4', att: goodAtt,
        pair_screen: { prerequisite_json: '/x/pre.json', prerequisite_delta_pct: -1.7, pair_json: '/x/pair.json', pair_delta_pct: 39.3 } });
      return undefined;
    }, inc);
    ok(y.result.rounds[0].status === 'verified', 'with both screens: verified', y.result.rounds[0]);
    ok(/pair_screen/.test(y.calls.find(c => c.label === 'gluon_engineer:climb r1_d0').prompt), 'the engineer is told what a pair owes');
  }

  console.log('\n# 7. recover: one direction per suspect, G2 on the round winner (integrated or single)');
  {
    const suspects = [{ specialty: 'compute', layer: '5', suspect: 'lost_RA' }, { specialty: 'memory', layer: '3', suspect: 'lost_layout' }];
    const run = (integ) => launch({ budget: 6 }, (label, prompt) => {
      if (label === 'verify r1') return { status: 'regression', correctness: 'pass', verified_geomean: 0.7 };
      if (label === 'tech_lead:plan r2') return planReturn(prompt, suspects);
      if (label === 'gluon_engineer:recover r2_d0') return engineerReturn(prompt, { suspect: 'lost_RA', layer: '5', parity_gate: gate(2) });
      if (label === 'gluon_engineer:recover r2_d1') return engineerReturn(prompt, { parity_gate: gate(2) });
      if (label === 'verify r2_d0') return ver(0.85);
      if (label === 'verify r2_d1') return ver(0.80);
      if (/^integrate r2/.test(label)) return integrateReturn(prompt, integ);
      if (label === 'verify r2_int') return ver(0.98);
      return undefined;
    }, { entry: 'port' });
    const x = await run({});
    const plan = x.calls.find(c => c.label === 'tech_lead:plan r2').prompt;
    ok(/STAGE=recover/.test(plan) && /lost_pipeline -> layer 4/.test(plan), 'the recover plan asks for one direction per suspect');
    ok(!x.logs.some(l => /deferred/.test(l)), 'layers 5 and 3 share a round');
    ok(x.result.parity === 'cleared' && x.labels.includes('gluon_engineer:climb r3_d0'), 'integrated 0.98 with its gate at 0 clears G2', x.result.parity);
    const y = await run({ parity_gate: undefined });
    const r2 = y.result.rounds.filter(r => r.round === 2);
    ok(y.labels.includes('integrate r2 (reask 1)') && r2.find(r => r.id === 'r2_int').status === 'rejected_missing_evidence' &&
      r2.find(r => r.id === 'r2_d0').kept, 'an integrated recover candidate without its gate is dropped; the single winner is kept', r2);
    ok(y.labels.includes('gluon_engineer:recover r3_d0'), 'and its gate (exit 2) keeps the run in recover', y.labels);
  }

  console.log('\n# 8. tech_lead may stop; with a deadline and time left the stop is refused (kernel_lane.js)');
  {
    const x = await launch({ ...climb, budget: 6 }, (label) => /^tech_lead:plan/.test(label) ? { stop: true, reasoning: 'done', directions: [] } : undefined, inc);
    ok(x.result.stopped_by === 'tech_lead_stop' && x.result.budget_used === 0, 'stop without a deadline', x.result.stopped_by);
    let plans = 0;
    const y = await launch({ ...climb, budget: 1, deadline_epoch: 1000 + 7200 }, (label, prompt) => {
      if (label.startsWith('clock')) return { epoch: 1000 };
      if (/^tech_lead:(re)?plan/.test(label)) return ++plans === 1 ? { stop: true, reasoning: 'done', directions: [] } : planReturn(prompt);
      return undefined;
    }, inc);
    const re = y.calls.find(c => c.label === 'tech_lead:replan r1#1');
    ok(re && /STOP_NOT_PERMITTED/.test(re.prompt) && y.labels.includes('gluon_engineer:climb r1_d0'), 'refused, re-planned, dispatched', y.labels);
  }

  console.log('\n# 9. the stall rule is kernel_lane.js\'s: progress means MIN_IMPROVE over the best candidate seen');
  {
    const vals = { 'verify r1_d0': 1.10, 'verify r2_d0': 1.11, 'verify r3_d0': 1.115 };
    const x = await launch({ ...climb, budget: 9 }, (label) => (label in vals ? ver(vals[label]) : undefined), inc);
    ok(x.result.stopped_by === 'max_no_improve' && x.result.rounds_used === 3, 'two rounds under +2% stop the climb', x.result);
    ok(x.result.rounds[1].kept && !x.result.rounds[1].accepted, 'a +0.9% direction moves the track but is not accepted', x.result.rounds[1]);
  }

  console.log('\n# 10. planning is kernel_workflow\'s tech_lead with the Gluon addendum');
  {
    const x = await launch({ ...climb, budget: 1 }, undefined, inc);
    const plan = x.calls.find(c => c.label === 'tech_lead:plan r1').prompt;
    ok(plan.includes(`Then Read ${WF_DIR}/roles/tech_lead.md`) && /PHASE=plan_round/.test(plan), 'tech_lead.md PHASE=plan_round');
    ok(/gluon_authoring expert workflow/.test(plan) && /Layer Backbone/.test(plan) && /paired: true/.test(plan), 'the Gluon addendum is appended');
    const mem = x.calls.find(c => c.label === 'tech_lead:memory r1').prompt;
    ok(mem.includes(`Then Read ${WF_DIR}/roles/tech_lead.md`) && /PHASE=update_memory/.test(mem), 'tech_lead.md PHASE=update_memory');
  }
  done('parallel-search');
})();
