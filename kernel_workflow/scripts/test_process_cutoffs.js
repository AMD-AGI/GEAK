#!/usr/bin/env node
'use strict';
// Exercise the actual lane with inert agents; no subprocesses, model calls or GPU work.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'scripts/build_workflows.json'), 'utf8'));
const body = manifest['kernel_workflow/kernel_lane.js']
  .map(file => fs.readFileSync(path.join(root, file), 'utf8')).join('\n')
  .replace(/^export const meta/m, 'const meta');
const total = 2000000000;
const cutoffs = {search: total - 5100, report: total - 1800, validate: total - 300};
const perCase = [{name: 'c1', baseline_ms: 8, optimized_ms: 4, speedup: 2}];

async function lane({bounded, commitResult = {committed: true}, planExpired = false, externalExpired = false}) {
  const deadlines = [], calls = [], reports = [];
  let activeDeadline = null;
  let currentEpoch = total - 7200, reportedSpeedup;
  const setProcessDeadline = epoch => {activeDeadline = epoch; deadlines.push(epoch);};
  setProcessDeadline.secondsLeft = () => activeDeadline - currentEpoch;
  const g = {
    args: {kernel_path: '/original', workflow_dir: '/workflow', budget: 1,
      agent_timeout_ms: 45000, agent_retries: 1,
      warm_start: 'off', update_experience: 'off', llm_stats: 'false',
      ...(bounded ? {deadline_epoch: total, search_reserve_s: 5100,
        final_validation_reserve_s: 1800, final_cleanup_reserve_s: 300} : {}),
      ...(externalExpired ? {optimizer: 'evo', optimizer_command: ['/fixture'], optimizer_reserve_s: 2000} : {})},
    phase: () => {}, log: () => {}, setProcessDeadline,
    workflow: async () => {throw new Error('unexpected nested workflow');},
    budget: {total: null, spent: () => 0, remaining: () => Infinity},
    parallel: async ts => Promise.all(ts.map(t => t())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async item => {
      for (const stage of stages) item = await stage(item);
      return item;
    })),
    runProcess: async () => {
      if (!externalExpired) throw new Error('unexpected subprocess');
      assert.equal(activeDeadline, total - 2000, 'external bound includes its TERM grace and settlement');
      throw Object.assign(new Error('expired before read-only git command'), {code: 'GEAK_PROCESS_DEADLINE'});
    },
    agent: async (prompt, options) => {
      const label = options.label;
      calls.push({label, cutoff: activeDeadline, timeout: options.processTimeoutMs});
      if (label === 'director:setup') return {eval_dir: '/eval', workspace: '/canonical',
        kernel_name: 'fixture', device_gfx: 'gfx950', device_target: 'unknown',
        physical_cu_count: 256, baseline_frozen: true, source_files: ['k.py']};
      if (label === 'tech_lead:analyze') return {kernel_type: 'gemm', modifiable_files: ['k.py']};
      if (label === 'benchmark_engineer') return {baseline_per_case: perCase,
        baseline_geomean_ms: 8, commandment_path: '/eval/COMMANDMENT.md',
        correctness_cmd: 'fixture-correctness', benchmark_cmd: 'fixture-benchmark'};
      if (label.includes('profile')) return {bottleneck: 'memory', top_opportunities: []};
      if (label.startsWith('tech_lead:plan')) {
        if (planExpired) {currentEpoch = activeDeadline - 6; return null;}
        return {stop: false, directions: [{id: 'a', title: 'a', specialty: 'memory', prompt: 'a'}]};
      }
      if (label.startsWith('eng ')) return {status: 'ok', measurement_valid: true, speedup_geomean: 2};
      if (label.startsWith('verify ')) return {status: 'verified', correctness: 'pass',
        verified_geomean: 2, per_case: perCase};
      if (label.startsWith('commit ')) return commitResult;
      if (label.startsWith('tech_lead:memory')) return {insights: [], ledger: []};
      if (label.startsWith('storage:') || label.startsWith('kb:')) return {};
      if (label === 'tech_lead:report') {
        reports.push(prompt);
        reportedSpeedup = Number(prompt.match(/- CUMULATIVE_SPEEDUP: (\S+)/)[1]);
        return {final_speedup_geomean: reportedSpeedup, final_patch: '/final.diff', report_path: '/report', rounds: 1};
      }
      if (label === 'director:validate') return {validation_status: 'accepted',
        director_verified_speedup_geomean: reportedSpeedup};
      throw new Error('unhandled agent ' + label);
    },
  };
  g.agent.processDeadlineActive = bounded;
  const run = () => new Function(...Object.keys(g), `return (async () => {${body}})();`)(...Object.values(g));
  if (bounded && (!commitResult || commitResult.committed !== true)) {
    await assert.rejects(run, /bounded commit did not finish; retain workspace for owner reconciliation/);
    assert.equal(calls.filter(c => c.label.startsWith('commit ')).length, 1);
    assert.equal(calls.some(c => c.label === 'tech_lead:report' || c.label === 'director:validate'), false);
    assert.deepEqual(deadlines, [cutoffs.search]);
    return;
  }
  const value = await run();
  if (bounded) assert.equal(g.args.deadline_epoch, total, 'the original deadline remains unchanged');
  const expired = planExpired || externalExpired;
  assert.equal(value.final_speedup, expired ? 1 : 2);
  assert.equal(value.warm_start.rounds_committed, expired ? 0 : 1);
  assert.equal(value.stopped_by, expired ? 'deadline' : 'budget');
  assert.equal(reports.length, 1);
  assert.match(reports[0], expired ? /- CUMULATIVE_SPEEDUP: 1(?:\n|$)/ : /- CUMULATIVE_SPEEDUP: 2(?:\n|$)/);
  assert.equal(calls.filter(c => c.label === 'director:validate').length, 1);
  assert.deepEqual(deadlines, bounded
    ? [cutoffs.search, ...(externalExpired ? [total - 2000] : []), cutoffs.report, cutoffs.validate] : []);
  let expected = cutoffs.search;
  for (const call of calls) {
    if (call.label === 'tech_lead:report') expected = cutoffs.report;
    if (call.label === 'director:validate') expected = cutoffs.validate;
    assert.equal(call.cutoff, bounded ? expected : null, call.label);
    assert.equal(call.timeout, bounded ? 45000 : undefined, call.label);
  }
}

(async () => {
  await lane({bounded: false});
  await lane({bounded: true});
  await lane({bounded: true, commitResult: null});
  await lane({bounded: true, commitResult: {committed: false}});
  await lane({bounded: true, planExpired: true});
  await lane({bounded: true, externalExpired: true});
  console.log('process cutoffs: actual lane reserves final stages, bounds roles, and refuses unfinished commits');
})().catch(error => {console.error(error); process.exitCode = 1;});
