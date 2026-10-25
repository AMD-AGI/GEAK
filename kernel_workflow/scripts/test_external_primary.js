#!/usr/bin/env node
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';
// The native fold deliberately differs from every legacy scalar in this fixture.
// Development and export use the checked native fold; all-case geomean remains diagnostic.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const text = fs.readFileSync(path.join(__dirname, '../src/kernel_lane/01b_args.js'), 'utf8');
const source = text.slice(text.indexOf('const EXTERNAL_OBJECTIVE'), text.indexOf('// Gate fields like'));
const laneSource = fs.readFileSync(path.join(__dirname, '../kernel_lane.js'), 'utf8');
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
const make = (A, hasWorkload = true) => new Function('A', 'HAS_WORKLOAD', 'runProcess', 'log', source
  + '\nreturn {primSpeedup,nativePrimary,nativeGainEligible,checkedExternalMeasurement};')(
    A, hasWorkload, runProcess, () => {});
(async () => {
  const legacy = make({});
  assert.equal(legacy.primSpeedup({verified_weighted: 2, verified_geomean: 3}), 2);
  assert.equal(legacy.nativePrimary({verified_weighted: 2, verified_geomean: 3}), 2);
  assert.equal(make({}, false).primSpeedup({verified_weighted: 2, verified_geomean: 3}), 3);
  const mode = make({objective, external_measurement_command: ['/owner/evaluate'],
    external_result_command: ['/owner/read']});
  assert.throws(() => make({objective, external_measurement_command: ['/owner/evaluate'],
    external_result_command: ['/owner/read'], external_final_measurement_command: ['/owner/final']}),
    /both admitted command prefixes/);
  assert.equal(mode.primSpeedup({verified_geomean: 3, verified_weighted: 2}), null);
  assert.equal(mode.primSpeedup(null), null);
  assert.equal(mode.nativeGainEligible({...native, native_reproducible_gain: false}), false);
  const read = await mode.checkedExternalMeasurement({external_evaluation: ref,
    primary_speedup: 999, verified_geomean: 3}, '/candidate', '/candidate.diff');
  assert.equal(mode.primSpeedup(read), 1.4);
  assert.equal(mode.nativePrimary(read), 1.4);
  assert.equal(read.development_geomean_speedup, 1.2, 'retain the checked all-case secondary');
  assert.equal(mode.primSpeedup({...read, primary_regime: 'foreign'}), null);
  assert.equal(mode.primSpeedup({...read, external_measurement_valid: false}), null);
  assert.equal(mode.nativeGainEligible(read), true);
  assert.deepEqual(calls[0].args, ['--candidate-id','candidate','--evaluation-id','evaluation',
    '--workspace','/candidate','--patch','/candidate.diff']);
  output = {...native, native_reproducible_gain: false};
  const noGain = await mode.checkedExternalMeasurement({external_evaluation: ref}, '/candidate');
  assert.equal(mode.primSpeedup(noGain), 1.4); // the gate does not enter development scoring
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
  await fullLane();
  await fullLane({retainedIntegration: true});
  await fullLane({finalNoGain: true});
  await fullLane({warm: true});
  await fullLane({staged: true});
  await fullLane({staged: true, invalidFinal: 'changed-candidate'});
  await fullLane({staged: true, invalidFinal: 'stale-evaluation'});
  await fullLane({staged: true, invalidFinal: 'capture-failed'});
  await fullLane({staged: true, invalidFinal: 'patch-capture-failed'});
  await fullLane({report: 'null'});
  await fullLane({report: 'missing', emptyPatch: true});
  await fullLane({bounded: true, report: 'null', emptyPatch: true, finalNoGain: true, originalFinal: true});
  await fullLane({report: 'stale'});
  await fullLane({report: 'wrong', overwriteHostPatch: true});
  await fullLane({staged: true, finalReaderReject: true});
  await fullLane({captureSelections: true});
  await fullLane({captureSelections: true, failBefore: true});
  await fullLane({staged: true, bounded: true, director: 'missing-ids'});
  await fullLane({director: 'unavailable'});
  await fullLane({finalHostFailure: true});
  await fullLane({director: 'veto'});
  await fullLane({director: 'veto', finalNoGain: true});
  console.log('external primary: native development ranking, retained geomean secondary, stock workload parity, host final patch and measurement, Director review, native export gate and identity checks passed');
})().catch(error => {console.error(error); process.exitCode = 1;});

async function fullLane({finalNoGain = false, warm = false, staged = false, invalidFinal = null,
  captureSelections = false, failBefore = false, bounded = false, director = 'accepted',
  finalHostFailure = false, finalReaderReject = false, report = 'normal', emptyPatch = false,
  overwriteHostPatch = false, originalFinal = false, retainedIntegration = false} = {}) {
  const body = laneSource.replace(/^export const meta/m, 'const meta');
  const selectedCandidate = 'b'.repeat(64);
  const rootCommit = 'a'.repeat(40), hostPatchPath = '/eval/host_final_patch.diff';
  const capturedPatch = emptyPatch ? '' : 'diff --git a/k.py b/k.py\n--- a/k.py\n+++ b/k.py\n@@ -1 +1 @@\n-original\n+selected\n';
  const patchHash = require('node:crypto').createHash('sha256').update(capturedPatch).digest('hex');
  const ids = id => ({candidate_id: id === 'final' ? selectedCandidate : id, evaluation_id: id});
  // [native primary, native gain, development geomean]
  const records = {original: [1, false, 1], a: [1.4, true, 1.2], b: [1.6, false, 1.1],
    c: [1.25, true, 1.5], integrated: [1.5, true, 1.3],
    final: originalFinal ? [1, false, 1] : [1.55, !finalNoGain, 1.45],
    warm: [1.4, true, 1.2]};
  let cumulative, commits = 0, captured = false, finalReads = 0, canonicalCaptures = 0;
  let finalMeasurements = 0, directorCalls = 0, activePhase = '', deadline;
  let rootReads = 0, patchReads = 0, hostPatchWrites = 0, hostPatch = null, kbWrites = 0;
  let integrationReads = 0;
  const retainedPatch = '/eval/round_1/engineer_1/best_patch.diff';
  const finalEvents = [], deadlines = [], fixtureErrors = [];
  const setProcessDeadline = epoch => { deadline = epoch; deadlines.push(epoch); };
  setProcessDeadline.secondsLeft = () => deadline - 1000;
  const g = {
    args: {kernel_path: '/original', workflow_dir: '/workflow', budget: bounded ? 0 : 3,
      agent_timeout_ms: 0, agent_retries: 1, objective,
      external_measurement_command: ['/owner/evaluate'], external_result_command: ['/owner/read'],
      capture_canonical_selections: captureSelections,
      ...(staged ? {external_final_measurement_command: ['/owner/final-evaluate'],
        external_final_result_command: ['/owner/final-read']} : {}),
      ...(bounded ? {deadline_epoch: 10000, search_reserve_s: 300,
        final_validation_reserve_s: 100, final_cleanup_reserve_s: 10} : {}),
      warm_start: warm ? 'on' : 'off', update_experience: 'off', llm_stats: 'false'},
    ...(bounded ? {setProcessDeadline} : {}),
    phase: title => { activePhase = title; }, log: () => {},
    workflow: async () => {throw new Error('unexpected nested workflow');},
    budget: {total: null, spent: () => 0, remaining: () => Infinity},
    parallel: async ts => Promise.all(ts.map(t => t())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async item => {
      for (const stage of stages) item = await stage(item);
      return item;
    })),
    runProcess: async opts => {
      if (opts.cmd === 'git') {
        assert.equal(captured, true, 'freeze the patch from the host-captured final source');
        assert.equal(cumulative, undefined, 'freeze patch bytes before Report');
        if (opts.args[2] === 'rev-list') {
          assert.deepEqual(opts.args, ['-C', '/canonical', 'rev-list', '--max-parents=0', 'HEAD']);
          rootReads++;
          return {status: 'completed', code: 0, group_cleared: true, stdout: rootCommit + '\n'};
        }
        assert.deepEqual(opts.args, ['-C', '/canonical', '--literal-pathspecs', 'diff', '--binary',
          '--no-color', '--no-ext-diff', '--src-prefix=a/', '--dst-prefix=b/', rootCommit, '--', 'k.py']);
        assert.equal(rootReads, 1);
        patchReads++;
        return {status: 'completed', code: invalidFinal === 'patch-capture-failed' ? 1 : 0,
          group_cleared: true, stdout: capturedPatch};
      }
      if (opts.cmd === 'sh') {
        assert.deepEqual(opts.args, ['-c', 'cat > "$1"', 'sh', hostPatchPath]);
        assert.equal(opts.input, capturedPatch, 'every host write restores the frozen bytes, including an empty patch');
        assert.equal(patchReads, 1, 'capture bytes once before restoring them');
        hostPatch = opts.input;
        hostPatchWrites++;
        return {status: 'completed', code: 0, group_cleared: true, stdout: ''};
      }
      if (opts.args.includes('--capture-selection')) {
        canonicalCaptures++;
        assert.equal(opts.cmd, '/owner/evaluate');
        assert.equal(opts.args.includes('--selected-evaluation-id'), commits > 0);
        const field = key => opts.args[opts.args.indexOf(key) + 1];
        const sequence = Number(field('--selection-sequence'));
        assert.equal(sequence, canonicalCaptures);
        if (failBefore && canonicalCaptures === 2) {
          return {status: 'spawn_error', code: null, group_cleared: true, stdout: ''};
        }
        if (failBefore && canonicalCaptures === 3) {
          assert.equal(opts.args.includes('--previous-selection'), true, 'later bytes retain the last successful reference');
          assert.equal(sequence, 3, 'failed attempted sequence is never renumbered');
        }
        const previous = opts.args.includes('--previous-selection') ? {
          evidence_id: field('--previous-selection'), artifact: 'canonical-selection.json'} : null;
        return {status: 'completed', code: 0, group_cleared: true, stdout: JSON.stringify({
          schema: 'geak-canonical-source-capture-v2', candidate_id: selectedCandidate,
          transition: {sequence, previous_capture_ref: previous, round: Number(field('--selection-round')),
            kind: field('--selection-kind'), stopped_by: opts.args.includes('--selection-stop-reason')
              ? field('--selection-stop-reason') : null},
          source_id: (commits ? 'd' : 'e').repeat(64),
          capture_ref: {evidence_id: 'f'.repeat(64), artifact: 'canonical-selection.json'},
          clock_ref: {path: '/owner/clock', sha256: 'a'.repeat(64)},
          canonical_commit: {status: 'committed', git_commit: 'b'.repeat(40)},
          development_evidence: {status: commits ? 'unresolved' : 'missing'}})};
      }
      if (warm && opts.cmd === '/usr/bin/test') return {status: 'completed', code: 1, group_cleared: true};
      if (opts.args.includes('--capture-only')) {
        assert.equal(opts.cmd, staged ? '/owner/final-evaluate' : '/owner/evaluate');
        assert.deepEqual(opts.args, ['--workspace', '/canonical', '--capture-only']);
        if (invalidFinal === 'capture-failed') return {status: 'completed', code: 1, group_cleared: true};
        captured = true;
        return {status: 'completed', code: 0, group_cleared: true, stdout: JSON.stringify({
          schema: 'geak-native-measurement-identity-v1', capture_only: true,
          candidate_id: selectedCandidate, source_id: 'd'.repeat(64), source_paths: ['k.py'], evaluation_id: 'final'})};
      }
      if (opts.cmd === (staged ? '/owner/final-evaluate' : '/owner/evaluate')) {
        assert.equal(captured, true, 'host evaluation uses the captured final source');
        assert.equal(hostPatch, capturedPatch, 'restore the host patch after Report');
        assert.equal(hostPatchWrites, 2);
        assert.equal(activePhase, 'Validate');
        assert.equal(directorCalls, 0, 'host measurement runs before Director review');
        assert.deepEqual(opts.args, ['--candidate-id', selectedCandidate, '--evaluation-id', 'final',
          '--workspace', '/canonical']);
        if (bounded) assert.equal(deadline, 9990, 'host measurement uses the Validate allowance');
        finalMeasurements++;
        finalEvents.push('measure');
        return {status: 'completed', code: finalHostFailure ? 1 : 0, group_cleared: true,
          strays_killed: false, stdio_held: false,
          stdout: 'Final evaluator completed; the trusted reader supplies the result.'};
      }
      const id = opts.args[3], [score, gain, dev] = records[id];
      if (retainedIntegration && activePhase === 'Merge') {
        integrationReads++;
        assert.deepEqual(opts.args, ['--candidate-id', 'b', '--evaluation-id', 'b',
          '--workspace', '/canonical', '--patch', retainedPatch],
        'a retained best is checked against the supplied integration base with its patch');
      }
      if (id === 'final') {
        finalReads++;
        finalEvents.push('read');
        assert.equal(finalMeasurements, 1);
        assert.equal(directorCalls, 1);
        assert.equal(hostPatch, capturedPatch, 'restore the host patch after Director review');
        assert.equal(hostPatchWrites, 3);
        assert.deepEqual(opts.args, ['--candidate-id', selectedCandidate, '--evaluation-id', 'final',
          '--workspace', '/original', '--patch', hostPatchPath]);
      }
      assert.equal(opts.cmd, staged && id === 'final' ? '/owner/final-read' : '/owner/read');
      if (id === 'final' && finalReaderReject) {
        return {status: 'completed', code: 1, group_cleared: true, stdout: ''};
      }
      return {status: 'completed', code: 0, group_cleared: true, stdout: JSON.stringify({
        ...native, external_evaluation: ids(id), primary_speedup: score,
        ...(id === 'final' ? {validation_status: 'accepted', correctness: 'pass', patch_sha256: patchHash} : {}),
        native_reproducible_gain: gain, timing_basis: 'external_whole_call_ut',
        development_geomean_speedup: dev,
        per_case: [{name: 'c1', baseline_ms: 8, optimized_ms: 8 / dev, speedup: dev}]
      })};
    },
    agent: async (prompt, options) => {
      const label = options.label;
      if (label === 'director:validate') {
        directorCalls++;
        finalEvents.push('director');
        assert.equal(finalMeasurements, 1, 'Director reviews a host-run evaluation');
        assert.ok(prompt.includes('do not launch another evaluation'));
        assert.ok(prompt.includes('- FINAL_PATCH: ' + hostPatchPath));
        assert.equal(hostPatch, capturedPatch, 'Director receives the frozen host patch');
        if (overwriteHostPatch) hostPatch = 'Director overwrote the host patch';
      }
      if (staged && label === 'director:validate') {
        assert.ok(prompt.includes(JSON.stringify(['/owner/final-evaluate', '--candidate-id',
          selectedCandidate, '--evaluation-id', 'final'])));
        assert.ok(!prompt.includes('["/owner/evaluate"]'));
      } else if (staged) {
        assert.ok(!prompt.includes('["/owner/final-evaluate"]'));
      }
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
      if (label.startsWith('integrate ')) {
        if (!retainedIntegration) return {attempted: true, conclusion: 'improved',
          best: {geomean: 999, patch_file: '/integrated.diff', external_evaluation: ids('integrated')}};
        const result = {attempted: true, conclusion: 'no_improvement',
          best: {geomean: 999, patch_file: retainedPatch, external_evaluation: ids('b')}};
        const {validate} = await import('../../interface/runtime/engine/schema.mjs');
        assert.equal(validate(result, options.schema).ok, true);
        assert.equal(validate({attempted: true, conclusion: 'no_improvement'}, options.schema).ok, true);
        const {patch_file, ...unbound} = result.best;
        for (const conclusion of ['no_improvement', 'improved']) {
          assert.deepEqual(validate({...result, conclusion, best: unbound}, options.schema).errors,
            ['$.best.patch_file: missing required field']);
        }
        assert.deepEqual(validate({...result, best: {patch_file}}, options.schema).errors,
          ['$.best.external_evaluation: missing required field']);
        return result;
      }
      if (label.startsWith('commit ')) {
        assert.ok(prompt.includes('winner: engineer b (1.60x)'),
          'commit the native-primary winner even when another candidate has higher geomean');
        if (captureSelections) assert.equal(canonicalCaptures, 2, 'source is captured before canonical mutation');
        commits++; return {committed: true};
      }
      if (label.startsWith('tech_lead:memory')) {
        if (retainedIntegration) {
          const integrate = JSON.parse(prompt.split('\n').find(x => x.startsWith('- INTEGRATE: ')).slice(13));
          assert.equal(integrate.conclusion, 'no_improvement');
          assert.equal(integrate.best.external_measurement_valid, true);
          assert.equal(integrate.best.primary_speedup, 1.6, 'retain the checked score, not the model scalar');
        }
        return {insights: [], ledger: []};
      }
      if (label === 'kb:write') {
        kbWrites++;
        assert.ok(prompt.includes('--patch ' + JSON.stringify(hostPatchPath)));
        assert.equal(hostPatch, capturedPatch, 'KB export uses the restored frozen patch');
        return {};
      }
      if (label.startsWith('storage:') || label.startsWith('kb:')) return {};
      if (label === 'tech_lead:report') {
        if (captureSelections) assert.equal(canonicalCaptures, 4, 'terminal capture precedes Report');
        assert.equal(captured, true, 'source is fixed before Report can edit it');
        assert.equal(hostPatch, capturedPatch, 'host patch exists before Report');
        assert.equal(hostPatchWrites, 1);
        cumulative = Number(prompt.split('\n').find(x => x.startsWith('- CUMULATIVE_SPEEDUP: ')).split(': ')[1]);
        if (overwriteHostPatch) hostPatch = 'Report overwrote the host patch';
        if (report === 'null') return null;
        return {final_speedup_geomean: 999, report_path: '/report',
          ...(report === 'missing' ? {} : {final_patch: report === 'stale' ? '/stale.diff'
            : report === 'wrong' ? '/wrong.diff' : '/final.diff'})};
      }
      if (label === 'director:validate') {
        // This model stub never executes an evaluator. It only reviews evidence.
        if (director === 'unavailable') return null;
        if (director === 'missing-ids') return {validation_status: 'accepted',
          director_verified_speedup_geomean: 999};
        return {validation_status: director === 'veto' ? 'flagged' : 'accepted',
          arbitration_note: director === 'veto' ? 'Explicit fixture Director objection' : '',
          director_verified_speedup_geomean: 999, external_evaluation: invalidFinal === 'changed-candidate'
            ? {candidate_id: 'e'.repeat(64), evaluation_id: 'final'} : invalidFinal === 'stale-evaluation'
            ? {candidate_id: selectedCandidate, evaluation_id: 'c'} : ids('final')};
      }
      throw new Error('unhandled agent ' + label);
    }
  };
  // Production catches role/process failures; fixture assertions must still fail the test.
  for (const name of ['agent', 'runProcess']) {
    const implementation = g[name];
    g[name] = async (...args) => {
      try { return await implementation(...args); }
      catch (error) { fixtureErrors.push(error); throw error; }
    };
  }
  if (bounded) g.agent.processDeadlineActive = true;
  const execute = async () => {
    try { return await new Function(...Object.keys(g), `return (async () => {${body}})();`)(...Object.values(g)); }
    finally { if (fixtureErrors.length) throw fixtureErrors[0]; }
  };
  if (invalidFinal === 'capture-failed' || invalidFinal === 'patch-capture-failed') {
    await assert.rejects(execute, /final_capture_failed/);
    assert.equal(captured, invalidFinal === 'patch-capture-failed');
    assert.equal(hostPatchWrites, 0, 'missing capture bytes cannot become a final patch');
    assert.equal(cumulative, undefined, 'Report did not run after capture failed');
    assert.equal(finalReads, 0);
    assert.equal(finalMeasurements, 0);
    assert.equal(directorCalls, 0);
    return;
  }
  const value = await execute();
  // b wins on native primary (1.6) despite c's higher geomean (1.5 vs 1.1).
  // Integrated 1.5 does not beat b; native no-gain does not prevent development.
  assert.equal(cumulative, bounded ? 1 : 1.6);
  assert.equal(commits, bounded ? 0 : 1);
  assert.equal(integrationReads, retainedIntegration ? 1 : 0);
  assert.equal(finalMeasurements, 1, 'one host final evaluation, independent of model claims');
  assert.equal(directorCalls, 1);
  assert.equal(rootReads, 1);
  assert.equal(patchReads, 1, 'Report and Director never trigger a recapture');
  assert.equal(hostPatchWrites, 3);
  assert.equal(hostPatch, capturedPatch);
  assert.equal(kbWrites, finalHostFailure || finalReaderReject || finalNoGain || director === 'veto' ? 0 : 1);
  assert.equal(value.director_review_status, director === 'unavailable' ? 'unavailable' : 'returned');
  if (bounded) assert.deepEqual(deadlines, [9700, 9900, 9990]);
  assert.equal(value.final_selection.candidate_id, selectedCandidate);
  assert.equal(value.final_selection.evaluation_id, 'final');
  if (captureSelections) {
    assert.deepEqual(value.canonical_selections.map(row => row.kind),
      ['initial', 'before_commit', 'after_commit', 'terminal']);
    assert.deepEqual(value.canonical_selections.map(row => row.source_id),
      ['e', failBefore ? null : 'e', 'd', 'd'].map(value => value ? value.repeat(64) : undefined));
    if (failBefore) assert.equal(value.canonical_selections[1].status, 'unresolved');
    assert.equal(value.canonical_selections[2].development_status, 'unresolved');
    assert.equal(value.canonical_selections[3].stopped_by, value.stopped_by);
  }
  if (finalHostFailure) {
    assert.equal(finalReads, 0, 'a failed host evaluation never reaches the score reader');
    assert.deepEqual(finalEvents, ['measure', 'director']);
    assert.equal(value.final_speedup, null);
    assert.equal(value.outcome, 'invalid_measurement');
    assert.equal(value.final_patch, '');
    assert.equal(value.final_patch_sha256, null);
    assert.equal(value.final_failure.class, 'final_measurement_failed');
    return;
  }
  if (invalidFinal) {
    assert.deepEqual(value.director_reported_evaluation, invalidFinal === 'changed-candidate'
      ? {candidate_id: 'e'.repeat(64), evaluation_id: 'final'}
      : {candidate_id: selectedCandidate, evaluation_id: 'c'},
    'retain mismatched Director IDs descriptively without replacing the host-selected source');
    assert.deepEqual(value.external_evaluation, ids('final'));
  }
  assert.deepEqual(finalEvents, ['measure', 'director', 'read']);
  assert.equal(finalReads, 1);
  if (finalReaderReject) {
    assert.equal(value.final_speedup, null);
    assert.equal(value.external_measurement_valid, false);
    assert.equal(value.validation_status, 'invalid_measurement');
    assert.equal(value.outcome, 'invalid_measurement');
    assert.equal(value.final_patch, '', 'a frozen patch does not override a rejecting final reader');
    assert.equal(value.final_patch_sha256, null);
    return;
  }
  assert.equal(value.final_speedup, originalFinal ? 1 : 1.55); // checked native result, never Director's 999.
  assert.equal(value.development_geomean_speedup, originalFinal ? 1 : 1.45, 'final all-case secondary is retained');
  if (originalFinal) {
    assert.equal(value.external_measurement_valid, true, 'ORIGINAL with an empty patch is a valid final measurement');
    assert.equal(value.native_reproducible_gain, false);
  }
  assert.equal(value.final_geomean, null);
  assert.equal(value.final_arithmetic, null);
  assert.equal(value.workload_aligned, false);
  assert.equal(value.budget_used, bounded ? 0 : 3);
  assert.equal(value.stopped_by, 'budget');
  assert.equal(value.validation_status, director === 'veto' ? 'flagged' : finalNoGain ? 'no_improvement' : 'accepted');
  assert.equal(value.outcome, finalNoGain || director === 'veto' ? 'no_improvement' : 'improved');
  assert.equal(value.final_patch, finalNoGain || director === 'veto' ? '' : hostPatchPath);
  assert.equal(value.patch_sha256, patchHash, 'retain the measured patch identity even on no-gain or veto');
  assert.equal(value.final_patch_sha256, finalNoGain || director === 'veto' ? null : patchHash);
  if (warm) {
    // Adopted, reported and divided on native primary (1.4 > 1); geomean stays diagnostic.
    assert.equal(value.warm_start.adopted, true);
    assert.equal(value.warm_start.adopted_speedup, 1.4);
    assert.ok(Math.abs(value.warm_start.incremental_speedup - 1.55 / 1.4) < 1e-12);
    assert.equal(value.warm_start.incremental_improved, true);
  } else assert.equal(value.warm_start.adopted, false);
}
