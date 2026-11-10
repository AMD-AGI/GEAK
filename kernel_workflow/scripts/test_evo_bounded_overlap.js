#!/usr/bin/env node
// CPU lifecycle witnesses, not performance measurements. Runs the real built lane,
// archive/custody/promotion code and (where marked native) createRuntime with its
// semaphore and process-bounded host commands. Scripted backend answers edit real
// temporary Git sources; explicit barriers test ordering, never speedup efficacy.
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {runMode, TEMPLATE, SCENARIOS, d, OUT, field} = require('./test_evo_first_stall_parity.js');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const latch = () => { let resolve; const promise = new Promise(r => {resolve = r;}); return {promise, resolve}; };
const within = promise => {
  let timer;
  return Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('fixture barrier did not settle')), 15000);
  })]).finally(() => clearTimeout(timer));
};
const template = {...TEMPLATE, preparation_allowance_s: 60, search_duration_s: 120,
  search_reserve_s: 60, final_validation_reserve_s: 20, final_cleanup_reserve_s: 1,
  no_stop_s: 10000};
const plan = (...directions) => ({stop: false, reasoning: 'fixture direction', directions});
const args = extra => ({evolution_start: 'search_start', evolution_bounded_overlap: true,
  evolution_evidence_handoff: true, budget: 3, ...extra});
const events = rec => rec.bounded_overlap.events;
const labels = rec => rec.calls.map(c => c.label);
const index = (rec, name) => labels(rec).indexOf(name);
const rows = rec => Object.fromEntries(rec.archive.entries.map(row => [row.id, row]));
const check = rec => { assert.equal(rec.error, null, rec.error); assert.equal(rec.runtime_state?.schemaFails || 0, 0); };
const reports = [];

async function optionalClosureFallbacks() {
  for (const failedRole of ['integrate', 'reprofile', 'tech_lead:memory']) {
    const outcomes = [];
    for (const overlap of [false, true]) {
      const successorEdit = latch();
      const rec = await within(runMode('evo', {
        name: `native_optional_${failedRole.replace(/:/g, '_')}_${overlap}`, nativeRuntime: true,
        args: args({budget: 4, evolution_bounded_overlap: overlap}),
        plans: {r1: plan(d('a', 'memory'), d('b', 'compute')),
          r2: plan(d('c', 'algorithm')), r3: plan(d('d', 'memory'))},
        eng: {r1_d0: {speed: 2}, r1_d1: {speed: 1.8}, r2_d0: {speed: 1.5}, r3_d0: {speed: 1.6}},
        verify: {r1_d0: 2, r1_d1: 1.8, r2_d0: 1.5, r3_d0: 1.6},
        async onCall(label) {
          if (overlap && label === 'verify r1_d0') await successorEdit.promise;
          if (label === `${failedRole} r1`) throw new Error(`fixture ${failedRole} backend failed`);
        },
        async beforeEngineerReturn(id, {rec}) {
          if (overlap && id === 'r2_d0') {
            assert.equal(index(rec, `${failedRole} r1`), -1, 'successor edits before optional closure call');
            successorEdit.resolve();
          }
        }
      }, template));
      check(rec);
      assert.ok(index(rec, `${failedRole} r1`) >= 0, 'failed optional role was invoked');
      assert.ok(index(rec, 'verify r2_d0') > index(rec, `${failedRole} r1`),
        'missing optional output must not deny successor qualification');
      assert.ok(index(rec, 'tech_lead:plan r3') >= 0, 'ordinary search continues after optional failure');
      assert.equal(rows(rec).r2_d0.verifier.verified_geomean, 1.5);
      assert.equal(rows(rec).r3_d0.verifier.verified_geomean, 1.6);
      assert.equal(rec.result.final_speedup, 2, 'verified individual remains the incumbent');
      assert.equal(rec.result.budget_used, 4, 'existing direction credits are unchanged');
      assert.match(fs.readFileSync(path.join(rec.fixture_root, 'eval/workspace/k.py'), 'utf8'), /r1_d0/);
      if (overlap) assert.ok(events(rec).filter(e => e.kind === 'closed').every(e => e.status === 'closed'));
      outcomes.push(rec);
    }
    const [serial, overlap] = outcomes;
    for (const key of ['final_speedup', 'budget_used', 'stopped_by'])
      assert.equal(overlap.result[key], serial.result[key], `${failedRole}: serial fallback outcome ${key}`);
    const nextMemory = rec => field(rec.calls.find(c => c.label === 'tech_lead:memory r2').role_prompt, 'PRIOR_HISTORY');
    if (failedRole === 'tech_lead:memory') {
      assert.deepEqual(nextMemory(overlap).insights, nextMemory(serial).insights);
      assert.deepEqual(nextMemory(overlap).ledger, nextMemory(serial).ledger);
    }
    if (failedRole === 'reprofile') {
      for (const rec of outcomes)
        assert.equal(field(rec.calls.find(c => c.label === 'tech_lead:plan r3').role_prompt, 'PROFILE_SUMMARY'), null);
    }
    reports.push(`native optional ${failedRole} failure: serial fallback, qualified incumbent, successor verification and further issuance preserved`);
  }
}

async function noPredecessorInputs() {
  for (const deep of [false, true]) {
    const outcomes = [];
    for (const overlap of [false, true]) {
      const rec = await runMode('evo', {name: `no_predecessor_${deep}_${overlap}`, stepFor: () => 0,
        args: args({budget: deep ? 3 : 1, evolution_bounded_overlap: overlap}),
        plans: {r1: plan(d('a', deep ? 'deep_explore' : 'memory')), r2: plan(d('b', 'compute'))},
        eng: {r1_d0: {speed: 1.2}, r2_d0: {speed: 1.3}}, verify: {r1_d0: 1.2, r2_d0: 1.3}}, template);
      check(rec);
      outcomes.push(rec.calls.filter(c => /^(tech_lead:plan |eng |deep )/.test(c.label))
        .map(({label, prompt, role_prompt}) => ({label, prompt, role_prompt})));
    }
    assert.deepEqual(outcomes[1], outcomes[0], 'no predecessor: planner and engineer inputs must match OFF');
  }
  reports.push('no predecessor: exact deterministic ON/OFF planner and engineer inputs, round1 and after deep');
}

async function recoveredCommitClosure() {
  const outcomes = [];
  for (const overlap of [false, true]) {
    const successorEdit = latch(), commitFailed = latch();
    const rec = await within(runMode('evo', {name: `native_recovered_commit_${overlap}`, nativeRuntime: true,
      args: args({budget: 3, evolution_bounded_overlap: overlap}), commitFail: 'r1',
      plans: {r1: plan(d('a', 'memory')), r2: plan(d('b', 'compute')), r3: plan(d('c', 'algorithm'))},
      eng: {r1_d0: {speed: 1.3}, r2_d0: {speed: 1.5}, r3_d0: {speed: 1.6}},
      verify: {r1_d0: 1.3, r2_d0: 1.5, r3_d0: 1.6},
      async onCall(label, prompt, {canonical}) {
        if (overlap && label === 'verify r1_d0') await successorEdit.promise;
        if (label === 'verify r2_d0') assert.equal(fs.readFileSync(path.join(canonical, 'k.py'), 'utf8'), 'A = 0\nB = 0\n');
        if (overlap && /r[23](?:_|\b)/.test(label)) {
          const timing = JSON.parse(prompt.match(/WORKFLOW_TIMING[^\n]*\n([^\n]+)/)[1]);
          assert.ok(timing.current_process_cutoff_epoch <= timing.search_cutoff_epoch);
        }
      },
      async onResult(label) { if (label === 'commit r1') commitFailed.resolve(); },
      async beforeEngineerReturn(id) {
        if (overlap && id === 'r2_d0') {successorEdit.resolve(); await commitFailed.promise; await delay(25);}
      }}, template));
    check(rec);
    assert.ok(index(rec, 'verify r2_d0') >= 0);
    assert.equal(rows(rec).r2_d0.verifier.verified_geomean, 1.5);
    assert.ok(index(rec, 'tech_lead:plan r3') >= 0);
    const history = field(rec.calls.find(c => c.label === 'tech_lead:report').role_prompt, 'HISTORY');
    assert.equal(history.rounds[0].status, 'commit_failed');
    assert.ok(history.rounds[0].recovery.previous_commit);
    assert.equal(rec.result.final_speedup, 1.6);
    assert.equal(rec.result.budget_used, 3);
    if (overlap) {
      assert.ok(events(rec).some(e => e.round === 1 && e.kind === 'closed' && e.status === 'closed'));
      assert.ok(events(rec).every(e => Number.isFinite(e.epoch) && e.epoch > 0));
    }
    outcomes.push(rec.result);
  }
  for (const key of ['final_speedup', 'budget_used', 'stopped_by']) assert.equal(outcomes[0][key], outcomes[1][key]);
  reports.push('native recovered commit: restored incumbent, successor qualification and r3 preserved with serial outcomes');
}

async function recoveryCallCaps() {
  for (const expireDuringRestore of [false, true]) {
    const planStarted = latch(), recoveryStarted = latch(), engineerRetried = latch();
    let cutoff = null, recovering = false, held = false, planAttempts = 0, engineerAttempts = 0;
    const callsInRecovery = [];
    const rec = await within(runMode('evo', {name: `recovery_caps_${expireDuringRestore}`, stepFor: () => 0,
      args: args({budget: 3, agent_retries: 2, engineer_reserve_s: 10, no_stop_s: 0}), commitFail: 'r1',
      plans: {r1: plan(d('a', 'memory')), r2: {stop: true, reasoning: 'fixture replan', directions: []},
        'r2#1': plan(d('b', 'compute')), r3: plan(d('c', 'algorithm'))},
      eng: {r1_d0: {speed: 1.3}, r2_d0: {speed: 1.5}, r3_d0: {speed: 1.6}},
      verify: {r1_d0: 1.3, r2_d0: 1.5, r3_d0: 1.6},
      onDeadline(epoch) {
        if (cutoff !== null) {recovering = epoch > cutoff; if (recovering) recoveryStarted.resolve();}
      },
      onResult(label, result, state) {if (label === 'tech_lead:plan r1') cutoff = state.processDeadline;},
      async onCall(label, prompt, state) {
        if (label === 'verify r1_d0') await planStarted.promise;
        if (label === 'tech_lead:plan r2' && ++planAttempts === 1) {
          planStarted.resolve(); await recoveryStarted.promise; throw new Error('fixture planner retry during recovery');
        }
        if (recovering && /r2(?:_|\b)/.test(label)) {
          const cap = state.options.processDeadlineEpoch;
          const expected = /^(eng |deep |tech_lead:(?:plan|replan) )/.test(label) ? cutoff - 10 : cutoff;
          assert.equal(cap, expected);
          callsInRecovery.push({label, global: state.processDeadline, explicit: cap,
            effective: Math.min(state.processDeadline, cap)});
          const facts = field(prompt, 'DELIVERY_FACTS');
          if (facts) {
            assert.equal(facts.as_of_epoch, state.now);
            assert.equal(facts.seconds_to_search_cutoff, cutoff - state.now);
            assert.equal(facts.seconds_to_engineer_cutoff, cutoff - 10 - state.now);
          }
          if (label.startsWith('tech_lead:replan '))
            assert.equal(field(prompt, 'MINUTES_REMAINING'), Math.round((cutoff - state.now) / 60));
        }
        if (label.startsWith('eng r2_d0:')) {
          if (++engineerAttempts === 1) throw new Error('fixture engineer retry during recovery');
          engineerRetried.resolve();
        }
      },
      async onProcess(req, state) {
        if (recovering && !held && req.cmd === 'git' && req.args[1] === state.canonical
            && req.args.slice(2).join(' ') === 'rev-parse HEAD') {
          held = true; await engineerRetried.promise;
          if (expireDuringRestore) state.setNow(cutoff - 6);
        }
      }}, template));
    check(rec);
    assert.ok(held);
    assert.ok(callsInRecovery.some(c => c.label === 'tech_lead:plan r2'));
    assert.ok(callsInRecovery.some(c => c.label.startsWith('tech_lead:replan r2')));
    assert.equal(callsInRecovery.filter(c => c.label.startsWith('eng r2_d0:')).length, 2);
    assert.ok(callsInRecovery.every(c => c.global > cutoff && c.effective <= cutoff - 10));
    fs.writeFileSync(path.join(rec.fixture_root, 'recovery_call_caps.json'), JSON.stringify({
      authority: 'CPU fixture call options; not GPU efficacy', original_search_cutoff_epoch: cutoff,
      engineer_cutoff_epoch: cutoff - 10, expireDuringRestore, calls: callsInRecovery}, null, 2));
    if (expireDuringRestore) {
      assert.equal(index(rec, 'verify r2_d0'), -1);
      assert.equal(index(rec, 'tech_lead:plan r3'), -1);
      assert.equal(rec.result.stopped_by, 'deadline_uncommitted');
    } else {
      assert.ok(index(rec, 'verify r2_d0') >= 0);
      assert.equal(rec.result.final_speedup, 1.6);
    }
  }
  reports.push('recovery interval: planner/replan/engineer retries retain fixed caps and truthful time facts; postrestore cutoff denies successor');
}

(async () => {
  if (process.argv.includes('--closure-recovery-only')) {
    await noPredecessorInputs(); await recoveredCommitClosure(); await recoveryCallCaps();
    console.log(reports.map(row => 'PASS: ' + row).join('\n'));
    console.log('Witness artifacts: ' + OUT);
    return;
  }
  if (process.argv.includes('--optional-closure-only')) {
    await optionalClosureFallbacks();
    console.log(reports.map(row => 'PASS: ' + row).join('\n'));
    console.log('Witness artifacts: ' + OUT);
    return;
  }
  // Check the refactor against the actual base, not just two modes of new code.
  const baseline = execFileSync('git', ['show', '1a84579b:kernel_workflow/kernel_lane.js'],
    {cwd: path.resolve(__dirname, '../..'), encoding: 'utf8', maxBuffer: 1024 * 1024});
  const offEdges = [
    {...SCENARIOS[0], name: 'profile_without_path', profile: {bottleneck: 'memory'}},
    {name: 'stall_then_stop', args: {evolution_import_stall_sources: true, no_stop_s: 100000},
      plans: {r1: plan(d('a', 'memory'))}, eng: {r1_d0: {speed: 1.01}}, verify: {r1_d0: 1.01}},
  ];
  for (const fixture of [...SCENARIOS, ...offEdges]) {
    for (const mode of fixture.name === 'stall_then_stop' ? ['evo'] : ['greedy', 'evo']) {
      const before = await runMode(mode, {...fixture, laneSource: baseline}, TEMPLATE);
      const after = await runMode(mode, fixture, TEMPLATE);
      check(before); check(after);
      assert.deepEqual(after.calls, before.calls, `${fixture.name}/${mode}: OFF prompts/order`);
      assert.deepEqual(after.host, before.host, `${fixture.name}/${mode}: OFF host operations`);
      assert.deepEqual(after.result, before.result, `${fixture.name}/${mode}: OFF outcomes`);
    }
  }
  reports.push('OFF parity: exact base prompts, schema-bearing calls, host operations and outcomes in 11 trajectories');

  {
    const successorEdit = latch();
    let slowSettled = false, fastVerify = false;
    const rec = await within(runMode('evo', {name: 'native_success', nativeRuntime: true,
      args: args({budget: 3}), plans: {r1: plan(d('a', 'memory'), d('b', 'compute')),
        r2: plan(d('c', 'algorithm'))},
      eng: {r1_d0: {speed: 2}, r1_d1: {speed: 1.8}, r2_d0: {speed: 1.5}},
      verify: {r1_d0: 2, r1_d1: 1.8, r2_d0: 1.5},
      async onCall(label, prompt, state) {
        if (label === 'verify r1_d0') {fastVerify = true; await successorEdit.promise;}
        if (label === 'tech_lead:plan r2') assert.equal(slowSettled, true, 'all engineers must settle');
        if (label === 'verify r2_d0') assert.ok(index(state.rec, 'tech_lead:memory r1') >= 0);
        if (label.startsWith('storage:reclaim')) {
          assert.ok(index(state.rec, 'tech_lead:memory r2') >= 0, 'no reclaim during active successor');
          state.command('bash', ['-e', '-c', prompt.match(/```bash\n([\s\S]*?)\n```/)[1]]);
        }
      },
      async beforeEngineerReturn(id, {ws, rec}) {
        if (id === 'r1_d1') {
          await delay(75);
          assert.equal(fastVerify, true, 'verification remains per-item, not all-engineers gated');
          assert.equal(index(rec, 'tech_lead:plan r2'), -1);
          slowSettled = true;
        }
        if (id === 'r2_d0') {
          assert.match(fs.readFileSync(path.join(ws, 'k.py'), 'utf8'), /r2_d0/);
          assert.equal(index(rec, 'verify r2_d0'), -1);
          assert.equal(index(rec, 'tech_lead:plan r3'), -1);
          successorEdit.resolve();
        }
      }}, template));
    check(rec);
    assert.equal(rec.result.final_speedup, 2, '1.5 beats old ORIGINAL but must not overwrite current2');
    assert.equal(index(rec, 'commit r2'), -1);
    assert.equal(rows(rec).r2_d0.parent_id, 'ORIGINAL', 'no silent default-parent drift');
    assert.equal(rows(rec).r2_d0.verifier.verified_geomean, 1.5);
    assert.equal(rec.result.budget_used, 3);
    assert.ok(events(rec).some(e => e.kind === 'reclaim_deferred'));
    const view = JSON.parse(fs.readFileSync(path.join(rec.fixture_root, 'eval/evolution/batch_2.json')));
    assert.equal(view.best_id, 'ORIGINAL');
    assert.equal(view.entries.find(r => r.id === 'r1_d0').verifier, null, 'source snapshot keeps pre-closure evidence');
    reports.push('native success: genuine next-round edit during held predecessor verification; current incumbent retained; immutable source context; safe reclaim');
  }

  for (const firstDeep of [false, true]) {
    const planned = latch();
    const rec = await within(runMode('evo', {name: `native_deep_${firstDeep}`, nativeRuntime: true,
      args: args({budget: 3}), plans: {r1: plan(d('a', firstDeep ? 'deep_explore' : 'memory')),
        r2: plan(d('b', firstDeep ? 'memory' : 'deep_explore'))},
      eng: {r1_d0: {speed: 1.2}, r2_d0: {speed: 1.3}}, verify: {r1_d0: 1.2, r2_d0: 1.3},
      async onCall(label, prompt, {rec}) {
        if (label === 'tech_lead:plan r2') {
          if (firstDeep) assert.ok(index(rec, 'tech_lead:memory r1') >= 0);
          planned.resolve();
        }
        if (!firstDeep && label === 'verify r1_d0') await planned.promise;
        if (label.startsWith(firstDeep ? 'eng r2_d0:' : 'deep r2_d0:'))
          assert.ok(index(rec, 'tech_lead:memory r1') >= 0, 'deep and integrator development must not overlap');
      }}, template));
    check(rec); assert.equal(rec.result.budget_used, 3);
  }
  reports.push('native deep exclusivity in both predecessor/successor orientations, preserving deep credits');

  {
    const successorEdit = latch();
    const rec = await within(runMode('evo', {name: 'native_stall_drain', nativeRuntime: true,
      args: args({budget: 5, max_no_improve: 1}), plans: {r1: plan(d('a', 'memory')),
        r2: plan(d('b', 'compute')), r3: plan(d('c', 'memory'))},
      eng: {r1_d0: {speed: 1.1}, r2_d0: {speed: 1.4}}, verify: {r2_d0: 1.4},
      async onCall(label) { if (label === 'verify r1_d0') await successorEdit.promise; },
      async beforeEngineerReturn(id) { if (id === 'r2_d0') successorEdit.resolve(); }}, template));
    check(rec); assert.equal(rec.result.budget_used, 2);
    assert.equal(index(rec, 'tech_lead:plan r3'), -1);
    assert.equal(rec.result.final_speedup, 1.4, 'issued successor still qualifies within cutoff');
    assert.equal(rec.result.stopped_by, 'no_improve');
    reports.push('native stall: issued successor fully accounted and qualified; no r+2 after stop');
  }

  await noPredecessorInputs();
  await recoveredCommitClosure();
  await recoveryCallCaps();

  for (const concurrency of [1, 2]) {
    const rec = await within(runMode('evo', {name: `native_slots_${concurrency}`, nativeRuntime: true, concurrency,
      args: args({budget: 2}), plans: {r1: plan(d('a', 'memory')), r2: plan(d('b', 'compute'))},
      eng: {r1_d0: {speed: 1.2}, r2_d0: {speed: 1.3}}, verify: {r1_d0: 1.2, r2_d0: 1.3}}, template));
    check(rec); assert.equal(rec.result.budget_used, 2);
  }
  reports.push('native semaphore: ordinary1-slot and2-slot queues complete without bypass/deadlock');

  {
    const plannedStop = latch();
    const rec = await within(runMode('evo', {name: 'native_successor_stop', nativeRuntime: true,
      args: args({budget: 5}), plans: {r1: plan(d('a', 'memory')),
        r2: {stop: true, reasoning: 'fixture stop', directions: []}},
      eng: {r1_d0: {speed: 1.2}}, verify: {r1_d0: 1.2},
      async onCall(label) {if (label === 'verify r1_d0') await plannedStop.promise;
        if (label === 'tech_lead:plan r2') plannedStop.resolve();}
    }, template));
    check(rec); assert.equal(rec.result.budget_used, 1);
    assert.equal(index(rec, 'tech_lead:plan r3'), -1, 'null successor must not restart planning after stop');
    assert.equal(labels(rec).filter(label => label === 'tech_lead:plan r2').length, 1);
    assert.equal(rec.result.stopped_by, 'tech_lead_stop');
    reports.push('native successor stop: a pending planner stop is terminal, never silently retried as a new round');
  }

  await optionalClosureFallbacks();

  {
    const rec = await within(runMode('evo', {name: 'native_missing_profile', nativeRuntime: true,
      profile: {bottleneck: 'reported only', summary_path: '/fixture/nonexistent/profile.md'},
      args: args({budget: 2}), plans: {r1: plan(d('a', 'memory')), r2: plan(d('b', 'compute'))},
      eng: {r1_d0: {speed: 1}, r2_d0: {speed: 1.2}}, verify: {r1_d0: 1, r2_d0: 1.2}}, template));
    check(rec);
    const input = field(rec.calls.find(c => c.label === 'tech_lead:plan r2').role_prompt, 'PROFILE_SUMMARY');
    assert.equal(input.summary_path, undefined);
    assert.equal(input.summary_snapshot_unavailable, true);
    assert.equal(input.bottleneck, 'reported only');
    reports.push('native optional profile absence: reported JSON preserved with explicit unknown file; no live path or extra gate');
  }

  for (const hook of ['shared_kb', 'global_kb', 'e2e_feedback', 'harness_addendum', 'state_dir']) {
    const rec = await runMode('evo', {name: `unsupported_${hook}`, args: args({[hook]: '/fixture/mutable'}),
      plans: {}, eng: {}}, template);
    assert.match(rec.error, /does not yet support mutable deep-mode/);
    assert.equal(rec.calls.some(c => c.label.startsWith('tech_lead:plan ')), false);
  }
  reports.push('unsupported mutable deep-mode hooks fail explicitly before optimizer planning');

  {
    // Explicit clock/cutoff witness in the real lane. Native elapsed-time queue
    // and process-settlement behavior is covered by test_process_deadlines.mjs.
    const successorEdit = latch(), release = latch();
    let active = false, searchCutoff = null;
    const rec = await within(runMode('evo', {name: 'cutoff_drain',
      args: args({budget: 2}), stepFor: () => 0,
      plans: {r1: plan(d('a', 'memory')), r2: plan(d('b', 'compute'))},
      eng: {r1_d0: {speed: 2}, r2_d0: {speed: 1.5}}, verify: {r1_d0: 2, r2_d0: 1.5},
      async onCall(label) {if (label === 'verify r1_d0') await successorEdit.promise;},
      async onResult(label, result, state) {
        if (label === 'tech_lead:plan r1') searchCutoff = state.processDeadline;
        if (label === 'verify r1_d0') {state.setNow(searchCutoff - 6); release.resolve();}
      },
      onDeadline(epoch) {if (active && searchCutoff !== null) assert.ok(epoch <= searchCutoff,
        'pending engineer/custody must settle before shared cutoff moves');},
      async beforeEngineerReturn(id) {
        if (id === 'r2_d0') {active = true; successorEdit.resolve(); await release.promise;
          await delay(40); active = false;}
      }}, template));
    check(rec); assert.equal(rec.result.budget_used, 2);
    assert.equal(index(rec, 'verify r2_d0'), -1);
    assert.equal(index(rec, 'commit r1'), -1);
    assert.equal(rec.result.stopped_by, 'deadline');
    assert.ok(rows(rec).r2_d0.patch_ref);
    reports.push('cutoff drain: no shared reserve extension during active successor; no late round qualification/commit; credits/source retained');
  }

  {
    let slowSettled = false;
    const rec = await within(runMode('evo', {name: 'native_capture_failure', nativeRuntime: true,
      args: args({budget: 3}), plans: {r1: plan(d('a', 'memory'), d('b', 'compute')),
        r2: plan(d('c', 'algorithm'))},
      eng: {r1_d0: {speed: 1.3}, r1_d1: {speed: 1.2}, r2_d0: {speed: 1.4}},
      verify: {r1_d1: 1.2, r2_d0: 1.4},
      async beforeEngineerReturn(id, {out}) {
        if (id === 'r1_d0') fs.unlinkSync(path.join(out, 'best_patch.diff'));
        if (id === 'r1_d1') {await delay(75); slowSettled = true;}
      },
      async onCall(label) {if (label === 'tech_lead:plan r2') assert.equal(slowSettled, true);}
    }, template));
    check(rec); assert.equal(rec.result.budget_used, 3);
    assert.equal(rows(rec).r1_d0.patch_ref, null);
    assert.ok(rows(rec).r1_d0.retention_error);
    reports.push('native capture failure: failed retention stays explicit and all remaining engineers settle before admission');
  }

  {
    // Real existing device-lock wrapper with CPU payloads and a private lock
    // directory; hardware probing/reaping are explicitly disabled for this test.
    const {runProcess} = await import('../../interface/runtime/engine/backends/base.mjs');
    const lockRoot = path.join(OUT, 'cpu_lock'); fs.mkdirSync(lockRoot);
    const acquired = path.join(lockRoot, 'successor_acquired');
    const ready = path.join(lockRoot, 'predecessor_ready'), release = path.join(lockRoot, 'release');
    const wrapper = path.resolve(__dirname, 'gpu_lock.sh');
    const env = {...process.env, KERNEL_EXTRACT_LOCK_DIR: lockRoot, GEAK_GPU_ALLOWED: '0',
      GEAK_GPU_REQUIRE_IDLE: '0', KERNEL_ENV_SKIP_ENUM_REAP: '1', KERNEL_ENV_KEEP_ARCH: '1',
      GEAK_GPU_LOCK_WAIT: '5', PYTORCH_ROCM_ARCH: 'gfx950', GPU_ARCHS: 'gfx950'};
    const first = runProcess({cmd: 'bash', args: [wrapper, '0', process.execPath, '-e',
      `const fs=require('fs');fs.writeFileSync(process.argv[1],'ready');const t=setInterval(()=>{if(fs.existsSync(process.argv[2]))clearInterval(t)},10)`, ready, release],
      cwd: lockRoot, env, timeoutMs: 5000});
    while (!fs.existsSync(ready)) await delay(10);
    const started = Date.now();
    const second = runProcess({cmd: 'bash', args: [wrapper, '0', process.execPath, '-e',
      `require('fs').writeFileSync(process.argv[1],'acquired')`, acquired], cwd: lockRoot, env, timeoutMs: 5000});
    await delay(100); assert.equal(fs.existsSync(acquired), false);
    fs.writeFileSync(release, 'release');
    const results = await Promise.all([first, second]);
    assert.ok(results.every(r => r.code === 0 && r.group_cleared), JSON.stringify(results));
    assert.ok(Date.now() - started >= 100);
    assert.equal(fs.existsSync(acquired), true);
    reports.push('native gpu_lock CPU payloads: successor waits behind predecessor ownership, no hardware activity or bypass');
  }

  {
    const dir = path.join(OUT, 'reclaim_guard');
    const active = path.join(dir, 'round_2/engineer_0/workspace'); fs.mkdirSync(active, {recursive: true});
    const artifact = path.join(active, 'active.so'); fs.writeFileSync(artifact, 'required dependency');
    fs.writeFileSync(path.join(dir, '.geak_overlap_active'), 'active');
    for (const extra of [[], ['--force-heavy']]) {
      const value = execFileSync('bash', [path.join(__dirname, 'reclaim_eval_artifacts.sh'),
        '--eval-dir', dir, '--keep-round', '0', ...extra], {encoding: 'utf8'});
      assert.match(value, /deferred/); assert.equal(fs.readFileSync(artifact, 'utf8'), 'required dependency');
    }
    reports.push('native reclaim: ordinary and force-heavy embedded callers preserve all active dependencies');
  }

  fs.writeFileSync(path.join(OUT, 'BOUNDED_OVERLAP_CPU_WITNESS.json'), JSON.stringify({
    authority: 'CPU lifecycle only; synthetic fitness is not kernel performance or policy efficacy', reports}, null, 2));
  console.log(reports.map(row => 'PASS: ' + row).join('\n'));
  console.log('Witness artifacts: ' + OUT);
})().catch(error => {console.error(error); process.exitCode = 1;});
