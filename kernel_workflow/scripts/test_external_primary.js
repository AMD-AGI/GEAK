#!/usr/bin/env node
'use strict';
// The native fold deliberately differs from every legacy scalar in this fixture.
// Development decisions use GEAK's geomean of checked per-case pairs; only export uses the native fold.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const text = fs.readFileSync(path.join(__dirname, '../src/kernel_lane/01b_args.js'), 'utf8');
const source = text.slice(text.indexOf('const EXTERNAL_OBJECTIVE'), text.indexOf('// Gate fields like'));
const objective = {kind: 'external_primary', primary_regime: 'prefill_extend'};
const ref = {candidate_id: 'candidate', evaluation_id: 'evaluation'};
const native = {objective, primary_regime: 'prefill_extend', primary_speedup: 1.4,
  native_reproducible_gain: true, native_reproducible_regression: false,
  development_geomean_speedup: 1.2, external_measurement_valid: true, external_evaluation: ref};
const calls = [];
let output = native;
const runProcess = async (options) => {
  calls.push(options);
  return {status: 'completed', code: 0, group_cleared: true, stdio_held: false,
    strays_killed: false, stdout: JSON.stringify(output)};
};
const make = (A) => new Function('A', 'HAS_WORKLOAD', 'runProcess', 'log', source
  + '\nreturn {primSpeedup,nativePrimary,nativeGainEligible,checkedExternalMeasurement};')(
    A, true, runProcess, () => {});
(async () => {
  const legacy = make({});
  assert.equal(legacy.primSpeedup({verified_weighted: 2, verified_geomean: 3}), 2);
  assert.equal(legacy.nativePrimary({verified_weighted: 2, verified_geomean: 3}), 2);
  const mode = make({objective, external_measurement_command: ['/owner/evaluate'],
    external_result_command: ['/owner/read']});
  assert.equal(mode.primSpeedup({verified_geomean: 3, verified_weighted: 2}), null);
  assert.equal(mode.primSpeedup(null), null);
  assert.equal(mode.nativeGainEligible({...native, native_reproducible_gain: false}), false);
  const read = await mode.checkedExternalMeasurement({external_evaluation: ref,
    primary_speedup: 999, verified_geomean: 3}, '/candidate', '/candidate.diff');
  assert.equal(mode.primSpeedup(read), 1.2);
  assert.equal(mode.nativePrimary(read), 1.4);
  assert.equal(mode.nativeGainEligible(read), true);
  assert.deepEqual(calls[0].args, ['--candidate-id','candidate','--evaluation-id','evaluation',
    '--workspace','/candidate','--patch','/candidate.diff']);
  output = {...native, native_reproducible_gain: false};
  const noGain = await mode.checkedExternalMeasurement({external_evaluation: ref}, '/candidate');
  assert.equal(mode.primSpeedup(noGain), 1.2); // the gate does not enter development scoring
  assert.equal(mode.nativePrimary(noGain), 1.4);
  assert.equal(mode.nativeGainEligible(noGain), false);
  output = {...native, development_geomean_speedup: undefined};
  const noDev = await mode.checkedExternalMeasurement({external_evaluation: ref}, '/candidate');
  assert.equal(noDev.external_measurement_valid, false);
  assert.equal(mode.primSpeedup(noDev), null);
  assert.equal(mode.nativePrimary(noDev), null);
  output = {...native, external_evaluation: {...ref, evaluation_id: 'different'}};
  const bad = await mode.checkedExternalMeasurement({external_evaluation: ref,
    validation_status: 'accepted', primary_speedup: 999}, '/candidate');
  assert.equal(mode.primSpeedup(bad), null);
  assert.equal(mode.nativePrimary(bad), null);
  assert.equal(bad.validation_status, 'invalid_measurement');
  await fullLane(false);
  await fullLane(true);
  await fullLane(false, true);
  console.log('external primary: checked development geomean, native export score and gain, no legacy fallback, source/patch arguments passed');
})().catch(error => {console.error(error); process.exitCode = 1;});

async function fullLane(finalNoGain, warm = false) {
  const body = fs.readFileSync(path.join(__dirname, '../kernel_lane.js'), 'utf8')
    .replace(/^export const meta/m, 'const meta');
  const ids = id => ({candidate_id: id, evaluation_id: id});
  // [native primary, native gain, development geomean]
  const records = {original: [1, false, 1], a: [1.4, true, 1.2], b: [1.6, true, 1.1],
    c: [9, false, 1.5], integrated: [1.5, true, 1.3], final: [1.55, !finalNoGain, 1.45],
    warm: [1.4, true, 1.2]};
  let cumulative, commits = 0;
  const g = {
    args: {kernel_path: '/original', workflow_dir: '/workflow', budget: 3,
      agent_timeout_ms: 0, agent_retries: 1, objective,
      external_measurement_command: ['/owner/evaluate'], external_result_command: ['/owner/read'],
      warm_start: warm ? 'on' : 'off', update_experience: 'off', llm_stats: 'false'},
    phase: () => {}, log: () => {},
    workflow: async () => {throw new Error('unexpected nested workflow');},
    budget: {total: null, spent: () => 0, remaining: () => Infinity},
    parallel: async ts => Promise.all(ts.map(t => t())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async item => {
      for (const stage of stages) item = await stage(item);
      return item;
    })),
    runProcess: async opts => {
      if (warm && opts.cmd === '/usr/bin/test') return {status: 'completed', code: 1, group_cleared: true};
      assert.equal(opts.cmd, '/owner/read');
      const id = opts.args[1], [score, gain, dev] = records[id];
      return {status: 'completed', code: 0, group_cleared: true, stdout: JSON.stringify({
        ...native, external_evaluation: ids(id), primary_speedup: score,
        native_reproducible_gain: gain, timing_basis: 'external_whole_call_ut',
        development_geomean_speedup: dev,
        per_case: [{name: 'c1', baseline_ms: 8, optimized_ms: 8 / dev, speedup: dev}]
      })};
    },
    agent: async (prompt, options) => {
      const label = options.label;
      if (label === 'director:setup') return {eval_dir: '/eval', workspace: '/canonical',
        kernel_name: 'fixture', device_gfx: 'gfx950', device_target: 'unknown',
        physical_cu_count: 256, baseline_frozen: true, source_files: ['k.py']};
      if (label === 'tech_lead:analyze') return {kernel_type: 'gemm', modifiable_files: ['k.py']};
      if (label === 'benchmark_engineer') return {external_evaluation: ids('original'),
        commandment_path: '/eval/COMMANDMENT.md', correctness_cmd: 'owner', benchmark_cmd: 'owner'};
      if (label === 'warm_start:resolve') return {read_reason: 'read', slug: 'fixture',
        candidates: [{rank: 1, slug: 'fixture', speedup: 2, patch_path: '/kb/warm.diff', exp_dir: '/kb/exp'}]};
      if (label === 'warm_start:verify c1') return {status: 'verified', correctness: 'pass',
        verified_geomean: 999, external_evaluation: ids('warm')};
      if (label === 'warm_start:adopt c1') return {committed: true};
      if (label.includes('profile')) return {bottleneck: 'memory', top_opportunities: []};
      if (label.startsWith('tech_lead:plan')) return {stop: false, directions: ['a','b','c'].map(id =>
        ({id, title: id, specialty: 'memory', prompt: id}))};
      if (label.startsWith('eng ')) return {status: 'ok', measurement_valid: true,
        speedup_geomean: 999, external_evaluation: ids(label.split(' ')[1].split(':')[0])};
      if (label.startsWith('verify ')) return {status: 'verified', correctness: 'pass',
        verified_geomean: 999, external_evaluation: ids(label.split(' ')[1])};
      if (label.startsWith('integrate ')) return {attempted: true, conclusion: 'improved',
        best: {geomean: 999, patch_file: '/integrated.diff', external_evaluation: ids('integrated')}};
      if (label.startsWith('commit ')) {commits++; return {committed: true};}
      if (label.startsWith('tech_lead:memory')) return {insights: [], ledger: []};
      if (label.startsWith('storage:') || label.startsWith('kb:')) return {};
      if (label === 'tech_lead:report') {
        cumulative = Number(prompt.split('\n').find(x => x.startsWith('- CUMULATIVE_SPEEDUP: ')).split(': ')[1]);
        return {final_speedup_geomean: 999, final_patch: '/final.diff', report_path: '/report'};
      }
      if (label === 'director:validate') return {validation_status: 'accepted',
        director_verified_speedup_geomean: 999, external_evaluation: ids('final')};
      throw new Error('unhandled agent ' + label);
    }
  };
  const value = await new Function(...Object.keys(g), `return (async () => {${body}})();`)(...Object.values(g));
  // GEAK ranks by the development geomean: c (1.5) wins despite its native no-gain gate;
  // integrated (1.3) does not beat it; native 9x/1.6x never enter development ranking.
  assert.equal(cumulative, 1.5);
  assert.equal(commits, 1);
  assert.equal(value.final_speedup, 1.55); // native primary of the fresh director result, never 999 or 1.45.
  assert.equal(value.final_geomean, null);
  assert.equal(value.final_arithmetic, null);
  assert.equal(value.workload_aligned, false);
  assert.equal(value.budget_used, 3);
  assert.equal(value.stopped_by, 'budget');
  assert.equal(value.outcome, finalNoGain ? 'no_improvement' : 'improved');
  assert.equal(value.final_patch, finalNoGain ? '' : '/final.diff');
  if (warm) {
    // Adopted on the development geomean (1.2 > 1); reported and divided on the native primary only.
    assert.equal(value.warm_start.adopted, true);
    assert.equal(value.warm_start.adopted_speedup, 1.4);
    assert.ok(Math.abs(value.warm_start.incremental_speedup - 1.55 / 1.4) < 1e-12);
    assert.equal(value.warm_start.incremental_improved, true);
  } else assert.equal(value.warm_start.adopted, false);
}
