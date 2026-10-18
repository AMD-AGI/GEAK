#!/usr/bin/env node
'use strict';
// Exercise the actual lane with inert agents and temporary Git repositories; no model/GPU work.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {execFileSync, spawnSync} = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'scripts/build_workflows.json'), 'utf8'));
const body = manifest['kernel_workflow/kernel_lane.js']
  .map(file => fs.readFileSync(path.join(root, file), 'utf8')).join('\n')
  .replace(/^export const meta/m, 'const meta');
const total = 2000000000;
const cutoffs = {search: total - 5100, report: total - 1800, validate: total - 300};
const perCase = [{name: 'c1', baseline_ms: 8, optimized_ms: 4, speedup: 2}];

async function lane({bounded, commitResult = {committed: true}, planExpired = false, externalExpired = false,
    successfulRounds = 0, commitAdvanced = false, pinExpired = false}) {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-cutoff-'));
  try {
  const canonical = path.join(temp, 'canonical'), evalDir = path.join(temp, 'eval');
  fs.mkdirSync(canonical); fs.mkdirSync(evalDir);
  const git = (...args) => execFileSync('git', ['-C', canonical, ...args], {encoding: 'utf8'}).trim();
  git('init', '-q');
  fs.writeFileSync(path.join(canonical, 'k.py'), 'baseline\n');
  fs.writeFileSync(path.join(canonical, '.gitignore'), 'cache/\n');
  git('add', '.');
  git('-c', 'user.name=fixture', '-c', 'user.email=fixture@localhost', 'commit', '-qm', 'baseline');
  fs.writeFileSync(path.join(canonical, 'retained.txt'), 'pre-existing untracked input\n');
  fs.mkdirSync(path.join(canonical, 'cache'));
  fs.writeFileSync(path.join(canonical, 'cache/result'), 'retained ignored cache\n');
  let previousHead = git('rev-parse', 'HEAD'), unfinishedHead = previousHead;
  const deadlines = [], calls = [], reports = [];
  let activeDeadline = null;
  let currentEpoch = total - 7200, reportedSpeedup, roundNumber = 0;
  const setProcessDeadline = epoch => {activeDeadline = epoch; deadlines.push(epoch);};
  setProcessDeadline.secondsLeft = () => activeDeadline - currentEpoch;
  const g = {
    args: {kernel_path: '/original', workflow_dir: '/workflow', budget: successfulRounds + 1,
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
    runProcess: async options => {
      if (externalExpired || (pinExpired && activeDeadline === cutoffs.search)) {
        if (externalExpired) assert.equal(activeDeadline, total - 2000,
          'external bound includes its TERM grace and settlement');
        currentEpoch = activeDeadline;
        throw Object.assign(new Error('expired before read-only git command'), {code: 'GEAK_PROCESS_DEADLINE'});
      }
      assert.ok(currentEpoch < activeDeadline, 'host recovery must run inside its reserved process window');
      assert.ok(options.timeoutMs > 0, 'host operations remain finite');
      const r = spawnSync(options.cmd, options.args, {input: options.input, encoding: 'utf8', timeout: 5000});
      if (r.error) throw r.error;
      return {status: 'completed', code: r.status, stdout: r.stdout, stderr: r.stderr,
        strays_killed: false, group_cleared: true, stdio_held: false};
    },
    agent: async (prompt, options) => {
      const label = options.label;
      calls.push({label, cutoff: activeDeadline, timeout: options.processTimeoutMs});
      if (label === 'director:setup') return {eval_dir: evalDir, workspace: canonical,
        kernel_name: 'fixture', device_gfx: 'gfx950', device_target: 'unknown',
        physical_cu_count: 256, baseline_frozen: true, source_files: ['k.py']};
      if (label === 'tech_lead:analyze') return {kernel_type: 'gemm', modifiable_files: ['k.py']};
      if (label === 'benchmark_engineer') return {baseline_per_case: perCase,
        baseline_geomean_ms: 8, commandment_path: '/eval/COMMANDMENT.md',
        correctness_cmd: 'fixture-correctness', benchmark_cmd: 'fixture-benchmark'};
      if (label.includes('profile')) return {bottleneck: 'memory', top_opportunities: []};
      if (label.startsWith('tech_lead:plan')) {
        if (planExpired) {currentEpoch = activeDeadline - 6; return null;}
        roundNumber++;
        return {stop: false, directions: [{id: 'a', title: 'a', specialty: 'memory', prompt: 'a'}]};
      }
      if (label.startsWith('eng ')) return {status: 'ok', measurement_valid: true, speedup_geomean: 2};
      if (label.startsWith('verify ')) return {status: 'verified', correctness: 'pass',
        verified_geomean: roundNumber + 1, per_case: perCase};
      if (label.startsWith('commit ')) {
        const result = roundNumber <= successfulRounds ? {committed: true} : commitResult;
        const completed = result && result.committed === true;
        if (bounded) {
          previousHead = git('rev-parse', 'HEAD');
          fs.writeFileSync(path.join(canonical, 'k.py'), `candidate ${roundNumber}\n`);
          fs.writeFileSync(path.join(evalDir, 'current_best.diff'), 'unfinished diff\n');
          if (!completed) {
            fs.writeFileSync(path.join(canonical, 'new.py'), 'unfinished new source\n');
            fs.writeFileSync(path.join(canonical, '*'), 'unfinished wildcard source\n');
          }
          if (completed || commitAdvanced) {
            git('add', '-A');
            git('-c', 'user.name=fixture', '-c', 'user.email=fixture@localhost', 'commit', '-qm', `round ${roundNumber}`);
            unfinishedHead = git('rev-parse', 'HEAD');
          } else {
            git('add', 'k.py'); // Interrupted after staging, with a new untracked source still present.
          }
          if (!completed) currentEpoch = cutoffs.search;
        }
        return result;
      }
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
  const value = await run();
  if (bounded) assert.equal(g.args.deadline_epoch, total, 'the original deadline remains unchanged');
  const expired = planExpired || externalExpired;
  const uncommitted = bounded && (pinExpired || !commitResult || commitResult.committed !== true);
  const expectedSpeedup = expired || (uncommitted && !successfulRounds) ? 1 : 2;
  assert.equal(value.final_speedup, expectedSpeedup);
  assert.equal(value.warm_start.rounds_committed, expired ? 0 : uncommitted ? successfulRounds : 1);
  assert.equal(value.stopped_by, uncommitted ? 'deadline_uncommitted' : expired ? 'deadline' : 'budget');
  assert.equal(value.deadline_hit, uncommitted || expired);
  assert.equal(reports.length, 1);
  assert.match(reports[0], new RegExp(`- CUMULATIVE_SPEEDUP: ${expectedSpeedup}(?:\\n|$)`));
  assert.equal(calls.filter(c => c.label === 'director:validate').length, 1);
  assert.deepEqual(deadlines, bounded
    ? [cutoffs.search, ...(externalExpired ? [total - 2000] : []),
      ...(uncommitted ? [cutoffs.report] : []), cutoffs.report, cutoffs.validate] : []);
  if (uncommitted) {
    assert.equal(git('rev-parse', 'HEAD'), previousHead, 'Report receives the exact pre-commit winner');
    assert.equal(fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'),
      successfulRounds ? 'candidate 1\n' : 'baseline\n');
    assert.equal(fs.existsSync(path.join(canonical, 'new.py')), false);
    assert.equal(fs.existsSync(path.join(canonical, '*')), false);
    assert.equal(fs.readFileSync(path.join(canonical, 'retained.txt'), 'utf8'), 'pre-existing untracked input\n');
    assert.equal(fs.readFileSync(path.join(canonical, 'cache/result'), 'utf8'), 'retained ignored cache\n');
    assert.equal(git('status', '--porcelain', '--untracked-files=no'), '');
    assert.match(reports[0], /deadline_uncommitted/);
    if (!pinExpired) {
      assert.equal(git('rev-parse', `refs/geak/incomplete-round-${roundNumber}`), unfinishedHead,
        'an advanced unfinished commit remains reachable');
      assert.match(fs.readFileSync(path.join(evalDir, `uncommitted_round_${roundNumber}.diff`), 'utf8'),
        new RegExp(`candidate ${roundNumber}`));
      assert.match(fs.readFileSync(path.join(evalDir, `uncommitted_round_${roundNumber}.diff`), 'utf8'),
        /unfinished new source/, 'new source bytes survive cleanup in the retained partial diff');
      assert.match(fs.readFileSync(path.join(evalDir, `uncommitted_round_${roundNumber}.diff`), 'utf8'),
        /unfinished wildcard source/, 'enumerated new paths are literal');
      const baseline = git('rev-list', '--max-parents=0', previousHead);
      assert.equal(fs.readFileSync(path.join(evalDir, 'current_best.diff'), 'utf8').trim(),
        git('diff', '--binary', baseline, previousHead, '--', '.'));
    }
    assert.equal(calls.some(c => c.label === `reprofile r${successfulRounds + 1}`), false);
  }
  let expected = cutoffs.search;
  for (const call of calls) {
    if (call.label === 'tech_lead:report') expected = cutoffs.report;
    if (call.label === 'director:validate') expected = cutoffs.validate;
    assert.equal(call.cutoff, bounded ? expected : null, call.label);
    assert.equal(call.timeout, bounded ? 45000 : undefined, call.label);
  }
  } finally {
    fs.rmSync(temp, {recursive: true, force: true});
  }
}

(async () => {
  await lane({bounded: false});
  await lane({bounded: true});
  await lane({bounded: true, commitResult: null});
  await lane({bounded: true, commitResult: {committed: false}});
  await lane({bounded: true, successfulRounds: 1, commitResult: null});
  await lane({bounded: true, successfulRounds: 1, commitResult: {committed: false}, commitAdvanced: true});
  await lane({bounded: true, pinExpired: true});
  await lane({bounded: true, planExpired: true});
  await lane({bounded: true, externalExpired: true});
  const deadline = fs.readFileSync(path.join(root, 'kernel_workflow/src/kernel_lane/07_deadline.js'), 'utf8');
  const guard = new Function('A', 'agent', 'setProcessDeadline', deadline);
  const args = {deadline_epoch: total, search_reserve_s: 5700, final_validation_reserve_s: 2100,
    final_cleanup_reserve_s: 300, optimizer: 'evo'};
  for (const optimizer_reserve_s of [undefined, 2100, 2160, 'bad']) {
    assert.throws(() => guard({...args, optimizer_reserve_s}, {processDeadlineActive: true}, () => {}),
      /optimizer reserve must exceed/);
  }
  guard({...args, optimizer_reserve_s: 2400}, {processDeadlineActive: true}, () => {});
  console.log('process cutoffs: prior winners survive unfinished commits; final/import reserves remain ordered');
})().catch(error => {console.error(error); process.exitCode = 1;});
