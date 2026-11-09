#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

async function lane({bounded, commitResult = {committed: true}, planExpired = false,
    successfulRounds = 0, commitAdvanced = false, pinExpired = false, commitAtDeadline = true,
    failThenSucceed = false, dirtyBeforeCommit = false, repeatedFailures = false, retained = false,
    directorNull = false, directorFlagged = false, directorChanged = false}) {
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
  const deadlines = [], calls = [], reports = [], cleanSnapshots = [], failedCommits = [];
  let activeDeadline = null;
  let currentEpoch = total - 7200, reportedSpeedup, roundNumber = 0, retainedPatch = null;
  const setProcessDeadline = epoch => {activeDeadline = epoch; deadlines.push(epoch);};
  setProcessDeadline.secondsLeft = () => activeDeadline - currentEpoch;
  const g = {
    args: {kernel_path: '/original', workflow_dir: '/workflow', budget: repeatedFailures ? 5 : failThenSucceed ? 2 : successfulRounds + 1,
      max_no_improve: 2, editable_source_paths: ['k.py'],
      agent_timeout_ms: 45000, agent_retries: 1,
      warm_start: 'off', update_experience: 'off', llm_stats: 'false',
      ...(bounded ? {deadline_epoch: total, search_reserve_s: 5100,
        final_validation_reserve_s: 1800, final_cleanup_reserve_s: 300} : {}),
      ...(retained ? {delivery_reserve_s: 1000} : {})},
    phase: () => {}, log: () => {}, setProcessDeadline,
    workflow: async () => {throw new Error('unexpected nested workflow');},
    budget: {total: null, spent: () => 0, remaining: () => Infinity},
    parallel: async ts => Promise.all(ts.map(t => t())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async item => {
      for (const stage of stages) item = await stage(item);
      return item;
    })),
    runProcess: async options => {
      if (pinExpired && activeDeadline === cutoffs.search && options.cmd === 'git'
          && options.args.includes('--show-toplevel')) {
        currentEpoch = activeDeadline;
        throw Object.assign(new Error('expired before read-only git command'), {code: 'GEAK_PROCESS_DEADLINE'});
      }
      if (bounded) assert.ok(currentEpoch < activeDeadline, 'host recovery must run inside its reserved process window');
      assert.ok(options.timeoutMs > 0, 'host operations remain finite');
      const r = spawnSync(options.cmd, options.args, {input: options.input, cwd: options.cwd,
        env: options.env ? {...process.env, ...options.env} : process.env, encoding: 'utf8', timeout: 5000});
      if (r.error) throw r.error;
      if (options.cmd === 'git' && options.args.includes('status') && r.status === 0 && !r.stdout.trim()) {
        cleanSnapshots.push({head: git('rev-parse', 'HEAD'), source: fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'),
          afterCommit: calls.filter(c => c.label.startsWith('commit ')).length});
      }
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
      if (label === 'benchmark_engineer') return {baseline_per_case: perCase, reliable: true,
        baseline_geomean_ms: 8, commandment_path: '/eval/COMMANDMENT.md',
        correctness_cmd: 'fixture-correctness', benchmark_cmd: 'fixture-benchmark'};
      if (label.includes('profile')) return {bottleneck: 'memory', top_opportunities: []};
      if (label.startsWith('tech_lead:plan')) {
        if (planExpired) {currentEpoch = activeDeadline - 6; return null;}
        roundNumber++;
        if (failThenSucceed && roundNumber === 2) {
          assert.equal(activeDeadline, cutoffs.search, 'ordinary commit failure restores the search cutoff');
          assert.equal(git('rev-parse', 'HEAD'), previousHead, 'next round starts from the retained commit');
          assert.equal(fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'), 'baseline\n');
          assert.match(prompt, /- CUMULATIVE_SPEEDUP: 1(?:\n|$)/, 'failed commit does not promote its score');
          assert.equal(git('status', '--porcelain', '--untracked-files=no'), '');
          assert.equal(fs.existsSync(path.join(canonical, 'new.py')), false);
          assert.match(fs.readFileSync(path.join(evalDir, 'uncommitted_round_1.diff'), 'utf8'), /candidate 1/);
        }
        return {stop: false, directions: [{id: 'a', title: 'a', specialty: 'memory', prompt: 'a'}]};
      }
      if (label.startsWith('eng ')) {
        // Every claimed candidate has real bytes against this round's canonical base.
        const out = prompt.match(/- OUTPUT_DIR: (\S+)/)[1], before = fs.readFileSync(path.join(canonical, 'k.py'), 'utf8');
        fs.mkdirSync(out, {recursive: true});
        const patch = path.join(out, 'best_patch.diff');
        const cutSource = retained && roundNumber > successfulRounds;
        fs.writeFileSync(patch, `diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n@@ -1 +1 @@\n-${before}+${cutSource ? 'retained' : `candidate ${roundNumber}`}\n`);
        if (cutSource) retainedPatch = patch;
        return {status: 'ok', measurement_valid: true, speedup_geomean: retained === 'reuse' ? 5 : 2};
      }
      if (label.startsWith('verify ')) {
        if (retained && label.endsWith('(delivery)')) {
          const measured = prompt.match(/- PATCH: (\S+)/)[1];
          assert.match(fs.readFileSync(measured, 'utf8'), /^\+retained$/m, 'the pinned bytes are qualified');
          if (retained === 'changed') fs.writeFileSync(measured, 'changed during qualification\n');
          return retained === 'rejected' ? {status: 'correctness_failed', correctness: 'fail', verified_geomean: 0}
            : {status: 'verified', correctness: 'pass', verified_geomean: retained === 'reuse' ? 5 : 1.9, per_case: perCase};
        }
        if (retained && roundNumber > successfulRounds) return null; // Cut before it finished.
        if (dirtyBeforeCommit) {
          fs.writeFileSync(path.join(canonical, 'k.py'), 'dirty tracked bytes\n');
          fs.writeFileSync(path.join(canonical, 'dirty_added.py'), 'dirty staged new source\n');
          git('add', 'k.py', 'dirty_added.py');
        }
        return {status: 'verified', correctness: 'pass', verified_geomean: repeatedFailures ? 2 : roundNumber + 1, per_case: perCase};
      }
      if (label.startsWith('commit ')) {
        const result = roundNumber <= successfulRounds || (failThenSucceed && roundNumber === 2)
          ? {committed: true} : commitResult;
        const completed = result && result.committed === true;
        {
          if (dirtyBeforeCommit) {
            assert.equal(fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'), 'baseline\n');
            assert.equal(fs.existsSync(path.join(canonical, 'dirty_added.py')), false);
            assert.equal(git('status', '--porcelain', '--untracked-files=no'), '');
            const dirty = fs.readFileSync(path.join(evalDir, `precommit_round_${roundNumber}.diff`), 'utf8');
            assert.match(dirty, /dirty tracked bytes/);
            assert.match(dirty, /dirty staged new source/);
          }
          previousHead = git('rev-parse', 'HEAD');
          fs.writeFileSync(path.join(canonical, 'k.py'), `candidate ${roundNumber}\n`);
          fs.writeFileSync(path.join(evalDir, 'current_best.diff'), 'unfinished diff\n');
          if (!completed) {
            fs.writeFileSync(path.join(canonical, 'new.py'), 'unfinished new source\n');
            fs.writeFileSync(path.join(canonical, '*'), 'unfinished wildcard source\n');
          }
          if (completed || commitAdvanced) {
            git('--literal-pathspecs', 'add', '--', 'k.py', ...(!completed ? ['new.py', '*'] : []));
            git('-c', 'user.name=fixture', '-c', 'user.email=fixture@localhost', 'commit', '-qm', `round ${roundNumber}`);
            unfinishedHead = git('rev-parse', 'HEAD');
          } else {
            git('add', 'k.py'); // Interrupted after staging, with a new untracked source still present.
          }
          if (!completed) failedCommits.push({round: roundNumber, previousHead, unfinishedHead,
            previousSource: git('show', `${previousHead}:k.py`) + '\n',
            callNumber: calls.filter(c => c.label.startsWith('commit ')).length});
          if (!completed && commitAtDeadline) currentEpoch = cutoffs.search;
        }
        return result;
      }
      if (label.startsWith('tech_lead:memory')) {
        if (retained === 'mutated') fs.writeFileSync(retainedPatch, 'mutated after capture\n');
        if (repeatedFailures) assert.match(prompt, /- IMPROVED: false(?:\n|$)/);
        return {insights: [], ledger: []};
      }
      if (label.startsWith('storage:') || label.startsWith('kb:')) return {};
      if (label === 'tech_lead:report') {
        reports.push(prompt);
        reportedSpeedup = Number(prompt.match(/- CUMULATIVE_SPEEDUP: (\S+)/)[1]);
        return {final_speedup_geomean: reportedSpeedup, final_patch: '/final.diff', report_path: '/report', rounds: 1};
      }
      if (label === 'director:validate' && directorNull) return null; // Final validation unavailable.
      if (label === 'director:validate' && directorChanged) {
        fs.writeFileSync(prompt.match(/- FINAL_PATCH: (\S+)/)[1], 'changed by final validator\n');
      }
      if (label === 'director:validate') return {validation_status: directorFlagged ? 'flagged' : 'accepted',
        director_verified_speedup_geomean: reportedSpeedup};
      throw new Error('unhandled agent ' + label);
    },
  };
  g.agent.processDeadlineActive = bounded;
  const run = () => new Function(...Object.keys(g), `return (async () => {${body}})();`)(...Object.values(g));
  const value = await run();
  if (bounded) assert.equal(g.args.deadline_epoch, total, 'the original deadline remains unchanged');
  const expired = planExpired;
  const failedCommit = bounded && (pinExpired || !commitResult || commitResult.committed !== true);
  const uncommitted = failedCommit && !failThenSucceed;
  const deadlineFailure = uncommitted && (pinExpired || commitAtDeadline);
  const delivered = retained === true || retained === 'reuse' || retained === 'mutated';
  const expectedSpeedup = retained === 'reuse' ? 5 : delivered ? 1.9 : retained ? 1
    : failThenSucceed ? 3 : expired ? 1 : successfulRounds + 2;
  const unvalidated = directorNull || directorFlagged || directorChanged;
  assert.equal(value.final_speedup, unvalidated ? null : expectedSpeedup);
  assert.equal(value.process_validated, !unvalidated);
  assert.equal(value.validation_status, directorNull ? 'unknown' : directorChanged ? 'source_changed'
    : directorFlagged ? 'flagged' : 'accepted');
  assert.equal(value.final_patch, unvalidated ? '' : value.delivered_patch);
  assert.equal(fs.readFileSync(value.delivered_patch, 'utf8').trim(),
    git('diff', '--binary', git('rev-list', '--max-parents=0', 'HEAD'), 'HEAD', '--', '.'),
    'host bytes remain exported even when final validation is unavailable or flagged');
  assert.equal(value.warm_start.rounds_committed, retained ? successfulRounds + (delivered ? 1 : 0)
    : expired ? 0 : uncommitted ? successfulRounds + 1 : 1);
  // A bounded run learns only an accepted final validation.
  if (bounded && expectedSpeedup > 1) {
    assert.equal(calls.some(c => c.label === 'kb:write'), !unvalidated, 'KB write requires accepted final validation');
  }
  if (retained) {
    assert.equal(fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'), delivered ? 'retained\n'
      : 'baseline\n', 'Report and validation see the delivered canonical source, else the incumbent bytes');
    assert.equal(calls.filter(c => c.label === 'verify a (delivery)').length, 1, 'qualified once');
    assert.match(reports[0], delivered ? /"delivered":"round_\d+_engineer_0"/
      : new RegExp(`"status":"${retained === 'rejected' ? 'correctness_failed' : 'source_changed'}"`));
  }
  assert.equal(value.stopped_by, deadlineFailure ? 'deadline_uncommitted' : expired ? 'deadline' : repeatedFailures ? 'no_improve' : 'budget');
  assert.equal(value.deadline_hit, deadlineFailure || expired);
  assert.equal(reports.length, 1);
  assert.match(reports[0], new RegExp(`- CUMULATIVE_SPEEDUP: ${expectedSpeedup}(?:\\n|$)`));
  assert.equal(calls.filter(c => c.label === 'director:validate').length, 1);
  assert.deepEqual(deadlines, bounded
    ? [cutoffs.search,
      ...(failedCommit ? Array.from({length: repeatedFailures ? 2 : 1}, () =>
        [cutoffs.report, ...(!deadlineFailure ? [cutoffs.search] : [])]).flat() : []),
      ...(!expired ? [cutoffs.report] : []),
      ...(retained ? [total - 4100, cutoffs.report] : []),
      ...(!expired ? [cutoffs.report] : []),
      cutoffs.report, cutoffs.validate] : []);
  if (repeatedFailures) {
    assert.equal(roundNumber, 2, 'repeated failed commits count toward plateau stopping');
    assert.equal(calls.filter(c => c.label.startsWith('tech_lead:memory')).length, 2);
    assert.equal((reports[0].match(/"status":"commit_failed"/g) || []).length, 2,
      'one complete history record is retained per failed round');
  }
  if (failThenSucceed) {
    assert.equal(calls.filter(c => c.label.startsWith('commit ')).length, 2);
    assert.equal(fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'), 'candidate 2\n');
    assert.match(reports[0], /commit_failed/);
    assert.doesNotMatch(reports[0], /deadline_uncommitted/);
  }
  if (uncommitted) {
    // Recovery must restore the incumbent first; the always-on host delivery then exports
    // the qualified source by its pin instead of dropping a winner after a commit interruption.
    for (const failed of failedCommits) {
      assert.ok(cleanSnapshots.some(row => row.afterCommit === failed.callNumber
        && row.head === failed.previousHead && row.source === failed.previousSource),
      `round ${failed.round} restores the exact previous committed tree before delivery`);
    }
    assert.equal(fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'),
      `candidate ${repeatedFailures ? 1 : successfulRounds + 1}\n`);
    assert.equal(fs.existsSync(path.join(canonical, 'new.py')), false);
    assert.equal(fs.existsSync(path.join(canonical, '*')), false);
    assert.equal(fs.readFileSync(path.join(canonical, 'retained.txt'), 'utf8'), 'pre-existing untracked input\n');
    assert.equal(fs.readFileSync(path.join(canonical, 'cache/result'), 'utf8'), 'retained ignored cache\n');
    assert.equal(git('status', '--porcelain', '--untracked-files=no'), '');
    assert.match(reports[0], deadlineFailure ? /deadline_uncommitted/ : /commit_failed/);
    if (!pinExpired) {
      assert.equal(git('rev-parse', `refs/geak/incomplete-round-${roundNumber}`), unfinishedHead,
        'an advanced unfinished commit remains reachable');
      assert.match(fs.readFileSync(path.join(evalDir, `uncommitted_round_${roundNumber}.diff`), 'utf8'),
        new RegExp(`candidate ${roundNumber}`));
      assert.match(fs.readFileSync(path.join(evalDir, `uncommitted_round_${roundNumber}.diff`), 'utf8'),
        /unfinished new source/, 'new source bytes survive cleanup in the retained partial diff');
      assert.match(fs.readFileSync(path.join(evalDir, `uncommitted_round_${roundNumber}.diff`), 'utf8'),
        /unfinished wildcard source/, 'enumerated new paths are literal');
      assert.equal(fs.readFileSync(path.join(evalDir, 'current_best.diff'), 'utf8').trim(),
        git('diff', '--binary', git('rev-list', '--max-parents=0', 'HEAD'), 'HEAD', '--', '.'));
    }
    assert.equal(calls.some(c => c.label === `reprofile r${successfulRounds + 1}`), false);
    assert.ok(value.delivery.delivered, 'a verified interrupted winner is delivered');
    assert.deepEqual(value.delivery_incomplete, []);
  }
  let expected = cutoffs.search;
  for (const call of calls) {
    if (call.label.endsWith('(delivery)')) { assert.equal(call.cutoff, total - 4100); continue; }
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
  const scenarios = [
    {bounded: false},
    {bounded: true},
    {bounded: true, commitResult: null},
    {bounded: true, commitResult: {committed: false}},
    {bounded: true, successfulRounds: 1, commitResult: null},
    {bounded: true, successfulRounds: 1, commitResult: {committed: false}, commitAdvanced: true},
    {bounded: true, commitResult: null, commitAtDeadline: false},
    {bounded: true, commitResult: {committed: false}, commitAtDeadline: false,
      failThenSucceed: true, commitAdvanced: true},
    {bounded: true, commitResult: null, commitAtDeadline: false, failThenSucceed: true},
    {bounded: true, commitResult: {committed: false}, commitAtDeadline: false, repeatedFailures: true},
    {bounded: true, dirtyBeforeCommit: true},
    {bounded: true, pinExpired: true},
    {bounded: true, planExpired: true},
    ...[true, 'rejected', 'changed', 'mutated'].map(retained => ({bounded: true, retained})),
    {bounded: true, retained: 'reuse', successfulRounds: 1},
    {bounded: true, directorNull: true},
    {bounded: true, directorFlagged: true},
    {bounded: true, directorChanged: true},
  ];
  let failed = 0;
  for (const scenario of scenarios) {
    try {
      await lane(scenario);
      console.log(`  ok: ${JSON.stringify(scenario)}`);
    } catch (error) {
      failed++;
      console.error(`  FAIL: ${JSON.stringify(scenario)}: ${error.stack || error}`);
    }
  }
  const deadline = fs.readFileSync(path.join(root, 'kernel_workflow/src/kernel_lane/07_deadline.js'), 'utf8');
  const guard = new Function('A', 'agent', 'setProcessDeadline', 'EXTERNAL_OBJECTIVE', 'says', deadline);
  const args = {deadline_epoch: total, search_reserve_s: 5700, final_validation_reserve_s: 2100,
    final_cleanup_reserve_s: 300};
  // Both policies share the ordinary loop and its process reserves.
  for (const optimizer of ['greedy', 'evo']) {
    guard({...args, optimizer}, {processDeadlineActive: true}, () => {});
  }
  // Zero disables additional qualification, while always-on custody and delivery remain active.
  guard({...args, delivery_reserve_s: 0}, {processDeadlineActive: true}, () => {});
  // A positive reserve must fit inside the Report allowance (3600 s here).
  guard({...args, delivery_reserve_s: 600}, {processDeadlineActive: true}, () => {});
  for (const bad of ['NaN', 'abc', '', '600', -5, 1.5, 3600, Infinity, true, false]) {
    assert.throws(() => guard({...args, delivery_reserve_s: bad}, {processDeadlineActive: true}, () => {}),
      /delivery_reserve_s/, `malformed delivery_reserve_s ${JSON.stringify(bad)} is refused`);
  }
  assert.throws(() => guard({delivery_reserve_s: 600}, {processDeadlineActive: false}, () => {}), /delivery_reserve_s/);
  assert.equal(failed, 0, `${failed}/${scenarios.length} process cutoff scenarios failed`);
  console.log(`PASS: ${scenarios.length} process cutoff cases and reserve guards; prior trees recover, verified sources export, final reserves stay ordered. CPU fixture only.`);
})().catch(error => {console.error(error); process.exitCode = 1;});
