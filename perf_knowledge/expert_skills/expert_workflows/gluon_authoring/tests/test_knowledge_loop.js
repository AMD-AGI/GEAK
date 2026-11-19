#!/usr/bin/env node
// kernel_lane.js's knowledge loop around the Gluon search: tech_lead analyze (perf_knowledge cards for
// the planner and every engineer), the experience KB read at the start (warm start, language gluon) and
// written at the end (update_experience, citation ledger, experience store), storage reclaim per round.
'use strict';

const { launch, ok, done, inOrder, planReturn, engineerReturn, WF_DIR, SKILL } = require('./harness');

const inc = { entry: 'incumbent' };
const climb = { entry_mode: 'incumbent' };
const ver = (v) => ({ status: 'verified', correctness: 'pass', verified_geomean: v });
const cand = { rank: 1, slug: 'k__gluon__gfx942', speedup: 1.3, patch_path: '/kb/p1.diff', session_id: 's1', direction: 'scale-feed' };

(async () => {
  console.log('\n# 1. tech_lead analyze runs before Benchmark; its cards reach the planner and every engineer');
  {
    const x = await launch({ ...climb, budget: 1 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, [{ specialty: 'memory', layer: '3', kk_refs: ['operators/gemm_op/backends/gluon.md'] }]);
      return undefined;
    }, inc);
    ok(inOrder(x.labels, ['director:setup', 'tech_lead:analyze', 'benchmark_engineer', 'gluon_profiler:baseline']), 'Setup -> Analyze -> Benchmark', x.labels);
    const an = x.calls.find(c => c.label === 'tech_lead:analyze').prompt;
    ok(an.includes(`Then Read ${WF_DIR}/roles/tech_lead.md`) && /PHASE=analyze/.test(an) && /kk_language gluon/.test(an), 'kernel_workflow tech_lead.md PHASE=analyze, told to stay Gluon');
    ok(/ANALYSIS: \{"kernel_type":"gemm"/.test(x.calls.find(c => c.label === 'benchmark_engineer').prompt), 'Benchmark gets the analysis');
    const plan = x.calls.find(c => c.label === 'tech_lead:plan r1').prompt;
    ok(/KK_OPERATOR: gemm_op/.test(plan) && /KK_REFS: \["operators\/gemm_op\/tuning\.md"\]/.test(plan) && /CODEBASE_CONTEXT: /.test(plan), 'planner gets the KK cards and codebase context');
    const eng = x.calls.find(c => c.label === 'gluon_engineer:climb r1_d0').prompt;
    ok(/KK_REFS: \["operators\/gemm_op\/backends\/gluon\.md"\]/.test(eng), 'a direction\'s own kk_refs replace the run\'s');
    ok(/SELF_MONITORING: \S+self_monitoring\.md/.test(eng) && /KERNEL_KNOWLEDGE_DIR: \S+perf_knowledge/.test(eng), 'engineer gets self_monitoring and the KK dir');
  }

  console.log('\n# 2. warm start (incumbent): a stored Gluon patch that passes verify becomes the kept line');
  {
    const x = await launch({ ...climb, budget: 1, max_no_improve: 9 }, (label) => {
      if (label === 'warm_start:resolve') return { read_reason: 'read', read_plane: 'local', canonical_id: 'geak:kernel:gfx942:k:gluon', candidates: [cand] };
      if (label === 'warm_start:verify c1') return ver(1.3);
      if (label === 'warm_start:adopt c1') return { committed: true, patch_file: '/x/ws.diff' };
      if (label === 'verify r1_d0') return ver(1.32);
      return undefined;
    }, inc);
    const res = x.calls.find(c => c.label === 'warm_start:resolve').prompt;
    ok(/--language gluon/.test(res) && /resolve-remote --plane local/.test(res), 'resolve is keyed by language gluon, store plane');
    ok(inOrder(x.labels, ['gluon_lead:budget', 'warm_start:resolve', 'warm_start:verify c1', 'warm_start:adopt c1', 'warm_start:reprofile', 'tech_lead:plan r1']), 'resolve -> verify -> adopt -> reprofile -> plan', x.labels);
    ok(/TRACK_START: warm_start/.test(x.calls.find(c => c.label === 'gluon_engineer:climb r1_d0').prompt), 'round 1 starts from the warm-start tag');
    ok(x.result.warm_start.adopted && x.result.warm_start.adopted_speedup === 1.3, 'adoption recorded', x.result.warm_start);
    const att = x.calls.find(c => c.label === 'kb:attest');
    ok(att && /--outcome validated/.test(att.prompt) && /--session-id "s1"/.test(att.prompt), 'the adopted candidate is attested validated (kernel_lane.js)', x.labels);
    ok(!x.result.accepted || x.result.accepted.id !== 'r1_d0', 'a +1.5% round over the adopted 1.30x is not accepted', x.result.accepted);
  }
  {
    const x = await launch({ ...climb, budget: 1 }, (label) => {
      if (label === 'warm_start:resolve') return { read_reason: 'read', read_plane: 'local', canonical_id: 'k', candidates: [cand] };
      if (label === 'warm_start:verify c1') return { status: 'correctness_failed', correctness: 'fail', verified_geomean: 0 };
      return undefined;
    }, inc);
    const att = x.calls.find(c => c.label === 'kb:attest');
    ok(att && /--outcome failed/.test(att.prompt) && /--language gluon/.test(att.prompt), 'a wrong stored patch is attested failed', x.labels);
    ok(!x.labels.includes('warm_start:adopt c1') && !x.result.warm_start.adopted, 'and not adopted');
  }

  console.log('\n# 3. a port reads the KB as reference only; warm_start=off reads nothing');
  {
    const x = await launch({ budget: 1 }, (label) => {
      if (label === 'warm_start:resolve') return { read_reason: 'read', candidates: [cand] };
      return undefined;
    }, { entry: 'port' });
    ok(x.labels.includes('warm_start:resolve') && !x.labels.some(l => /warm_start:(verify|adopt)/.test(l)), 'port: no adoption', x.labels);
    ok(/KB_REFERENCES: \S+kb_references/.test(x.calls.find(c => c.label === 'gluon_engineer:transcribe r1').prompt), 'the references reach the engineer');
    const y = await launch({ ...climb, budget: 1, warm_start: 'off' }, undefined, inc);
    ok(!y.labels.some(l => /^warm_start/.test(l)) && y.result.warm_start.read_reason === 'disabled', 'off: nothing read');
  }

  console.log('\n# 4. write-back after Validate, with kernel_lane.js\'s gates');
  {
    const x = await launch({ ...climb, budget: 1 }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, [{ specialty: 'memory', layer: '3', learned_refs: ['card-a.md'] }]);
      if (label === 'verify r1_d0') return ver(1.2);
      if (label === 'director:validate') return { validation_status: 'accepted', director_verified_speedup_geomean: 1.19 };
      return undefined;
    }, inc);
    ok(inOrder(x.labels, ['director:validate', 'update_experience', 'kb:cite', 'kb:write']), 'card, ledger, store, in that order', x.labels);
    const upd = x.calls.find(c => c.label === 'update_experience').prompt;
    ok(upd.includes(`Then Read ${WF_DIR}/roles/update_experience.md`) && /"language":"gluon"/.test(upd) && /"speedup":1\.19/.test(upd), 'kernel_workflow update_experience.md with the director\'s number, language gluon');
    ok(/kb\.py --kb-dir \S+knowledge\/learned cite/.test(x.calls.find(c => c.label === 'kb:cite').prompt) && x.result.citations[0].card === 'card-a.md' && x.result.citations[0].became_winner,
      'the ledger joins the cited card to what verify measured', x.result.citations);
    ok(/write-remote --plane both/.test(x.calls.find(c => c.label === 'kb:write').prompt) && /--language gluon/.test(x.calls.find(c => c.label === 'kb:write').prompt), 'store write keyed by gluon');
    const off = await launch({ ...climb, budget: 1, update_experience: 'off' }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, [{ specialty: 'memory', layer: '3', learned_refs: ['card-a.md'] }]);
      if (label === 'director:validate') return { validation_status: 'accepted', director_verified_speedup_geomean: 1.19 };
      return undefined;
    }, inc);
    ok(!off.labels.includes('update_experience') && off.labels.includes('kb:cite'), 'update_experience=off: no card; the ledger is still owed (kernel_lane.js)', off.labels);
    const held = await launch({ ...climb, budget: 1, held_out: 'true' }, (label, prompt) => {
      if (label === 'tech_lead:plan r1') return planReturn(prompt, [{ specialty: 'memory', layer: '3', learned_refs: ['card-a.md'] }]);
      if (label === 'director:validate') return { validation_status: 'accepted', director_verified_speedup_geomean: 1.19 };
      return undefined;
    }, inc);
    ok(!held.labels.includes('update_experience') && !held.labels.includes('kb:cite'), 'held_out: neither card nor ledger', held.labels);
    const flagged = await launch({ ...climb, budget: 1 }, (label) =>
      label === 'director:validate' ? { validation_status: 'flagged', director_verified_speedup_geomean: 1.19 } : undefined, inc);
    ok(!flagged.labels.includes('update_experience'), 'a flagged validation teaches nothing');
    const lost = await launch({ ...climb, budget: 1 }, (label) =>
      /^verify/.test(label) ? ver(0.95) : undefined, inc);
    ok(!lost.labels.some(l => /update_experience|kb:write/.test(l)), 'no win: no card, no store write');
  }

  console.log('\n# 5. storage reclaim after every round, on the Gluon round dirs');
  {
    const x = await launch({ ...climb, budget: 2, max_no_improve: 9 }, undefined, inc);
    const r = x.calls.filter(c => /^storage:reclaim r/.test(c.label));
    ok(r.length === 2 && /reclaim_eval_artifacts\.sh --eval-dir \/tmp\/eval\/gluon --keep-round 1/.test(r[0].prompt), 'one per round, scoped to EVAL_DIR/gluon', x.labels);
    const y = await launch({ ...climb, budget: 1, storage_reclaim: 'off' }, undefined, inc);
    ok(!y.labels.some(l => /storage:reclaim/.test(l)), 'storage_reclaim=off');
  }
  console.log('\n# 6. the archetype routes the workload reading (workloads/index.md keys)');
  {
    const x = await launch({ ...climb, budget: 1 }, (label, prompt) => {
      if (label.startsWith('gluon_lead:budget')) return { command: 'python3 kt/hw_budget.py --workload fused_moe --json', exit_code: 0,
        budget_json: '/x/b.json', workload: 'fused_moe', references_read: [...prompt.matchAll(/^\d+\. (\S+) — /gm)].map(m => m[1]) };
      if (label === 'gluon_engineer:climb r1_d0') { const r = engineerReturn(prompt); r.references_read = r.references_read.filter(p => !p.endsWith('moe.md')); return r; }
      return undefined;
    }, inc);
    const eng = x.calls.find(c => c.label === 'gluon_engineer:climb r1_d0').prompt;
    ok(eng.includes(`${SKILL}/references/workloads/moe.md — the moe archetype page`), 'fused_moe -> moe.md in the required reading');
    ok(x.labels.includes('gluon_engineer:evidence r1_d0 (reask 1)') && /moe\.md/.test(x.calls.find(c => c.label === 'gluon_engineer:evidence r1_d0 (reask 1)').prompt),
      'skipping it is sent back by name', x.labels);
    ok(/The moe archetype page\(s\)/.test(x.calls.find(c => c.label === 'tech_lead:plan r1').prompt) && x.result.workload === 'moe', 'the planner is pointed at it too');
    const y = await launch({ ...climb, budget: 1 }, undefined, inc);
    ok(y.result.workload === 'gemm', 'no budget archetype: analyze\'s kernel_type routes (gemm)', y.result.workload);
  }

  console.log('\n# 7. a guided run asks the planner to exercise named layers; gates unchanged');
  {
    const x = await launch({ ...climb, budget: 1, focus_layers: '4+,1.5' }, undefined, inc);
    const plan = x.calls.find(c => c.label === 'tech_lead:plan r1').prompt;
    ok(/GUIDED RUN: this run exercises layer\(s\) 4\+, 1\.5/.test(plan) && /inline asm needs its disassembly/.test(plan), 'addendum names the layers and keeps the gates');
    ok(!/GUIDED RUN/.test((await launch({ ...climb, budget: 1 }, undefined, inc)).calls.find(c => c.label === 'tech_lead:plan r1').prompt), 'off by default');
    const bad = await launch({ ...climb, focus_layers: '9' });
    ok(/focus_layers must be/.test(bad.error || '') && bad.calls.length === 0, 'an unknown layer is refused before any agent', bad.error);
  }
  done('knowledge-loop');
})();
