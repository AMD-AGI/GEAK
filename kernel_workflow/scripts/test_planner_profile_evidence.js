#!/usr/bin/env node
'use strict';
// CPU-only native fixture reuse: failed profiles must preserve planner evidence without
// changing any engineer/memory/profile role request or the default lane result.
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');

const harnessPath = path.join(__dirname, 'test_greedy_parity.js');
const fullHarness = fs.readFileSync(harnessPath, 'utf8').replace(/^#![^\n]*\n/, '');
const harness = fullHarness.slice(0, fullHarness.lastIndexOf('\n(async () => {'));
const load = failure => {
  const source = harness.replace('const response = actual(fallback(label, normalize(prompt)));',
    `if (${JSON.stringify(failure)} && label === ${JSON.stringify(failure)}) {\n` +
    '      fs.writeFileSync(path.join(evalDir, "p0.md"), "overwritten during failed reprofile"); return null; }\n' +
    '    const response = actual(fallback(label, normalize(prompt)));\n' +
    '    if (label === "profile_engineer:baseline") fs.writeFileSync(response.summary_path, "original profile");');
  return new Function('require', '__dirname', source + '\nreturn {runScenario, instrument, SCENARIOS, field};')
    (require, __dirname);
};

async function parity() {
  for (const [scenario, failed] of [
    ['a_improve_then_budget', 'reprofile r1'],
    ['b_warmstart_then_no_improve', 'warm_start:reprofile'],
  ]) {
    const h = load(failed), sc = h.SCENARIOS.find(row => row.name === scenario);
    assert.ok(sc, scenario);
    const body = h.instrument(fs.readFileSync(path.join(__dirname, '..', 'kernel_lane.js'), 'utf8'));
    const calls = {off: [], on: []}, results = {};
    for (const mode of ['off', 'on']) results[mode] = await h.runScenario(body,
      {...sc, args: {...sc.args, planner_profile_evidence: mode === 'on'}},
      (label, prompt, opts) => calls[mode].push({label, prompt, opts}));
    const planner = row => row.label.startsWith('tech_lead:plan ') || row.label.startsWith('tech_lead:replan ');
    assert.deepEqual(calls.on.filter(row => !planner(row)), calls.off.filter(row => !planner(row)),
      'all non-planner role requests/options, including engineers and memory, stay identical');
    assert.deepEqual(results.on.after_loop, results.off.after_loop, 'legacy optimization state stays identical');
    assert.deepEqual(results.on.result, results.off.result, 'scripted native results stay identical');
    const plans = calls.on.filter(planner);
    assert.ok(plans.length);
    for (const row of plans) {
      const status = h.field(row.prompt, 'EVIDENCE_STATUS');
      assert.match(status.current_source_sha256, /^[a-f0-9]{64}$/);
      assert.equal(status.latest_attempt.outcome, 'returned');
      assert.ok(status.items.every(item => item.applicability === 'UNKNOWN'
        && item.evidence_source_sha256 === null && item.currency === 'UNKNOWN'));
    }
    const afterFailure = plans.find(row => {
      const s = h.field(row.prompt, 'EVIDENCE_STATUS');
      return s.retained_profile_return?.currency === 'HISTORICAL';
    });
    assert.ok(afterFailure, 'a later planner sees explicitly historical prior return');
    assert.equal(h.field(afterFailure.prompt, 'PROFILE_SUMMARY').summary_path, '/cpu-fixture/eval/p0.md');
    const artifact = h.field(afterFailure.prompt, 'EVIDENCE_STATUS').items.find(row => row.path.endsWith('/p0.md'));
    assert.ok(artifact.available);
    assert.equal(artifact.artifact_sha256, require('crypto').createHash('sha256')
      .update('overwritten during failed reprofile').digest('hex'));
    assert.equal(artifact.currency, 'UNKNOWN', 'current path bytes must not be called retained historical bytes');
    assert.ok(calls.off.filter(planner).every(row => h.field(row.prompt, 'EVIDENCE_STATUS') === undefined));
  }
}

async function attempts() {
  const guard = fs.readFileSync(path.join(__dirname, '..', 'src', 'shared', 'agent_guard.js'), 'utf8');
  for (const [kind, expected] of [
    ['null', 'returned'], ['throw', 'threw'], ['deadline', 'no_end'], ['hang', 'no_end'],
  ]) {
    const seen = [];
    const agent = async () => {
      if (kind === 'throw') throw Object.assign(new Error('fixture'), {code: 'FIXTURE_ERROR'});
      if (kind === 'deadline') throw Object.assign(new Error('fixture'), {code: 'GEAK_PROCESS_DEADLINE'});
      if (kind === 'hang') return new Promise(() => {});
      return null;
    };
    agent.processDeadlineActive = kind === 'deadline';
    // Close over a local sink without adding any option to the actual role call.
    const call = new Function('agent', 'setTimeout', 'clearTimeout', 'seen',
      'const log=()=>{},tlAgent=()=>null,AGENT_RETRIES=1,AGENT_TIMEOUT_MS=2;\n' + guard +
      '\nreturn agentT("p", {}, row => seen.push(row));');
    assert.equal(await call(agent, setTimeout, clearTimeout, seen), null);
    assert.equal(seen.at(-1).outcome, expected, kind);
    assert.equal(seen.at(-1).attempt, 1);
  }
}

async function handoff() {
  const source = fs.readFileSync(path.join(__dirname, '..', 'src', 'kernel_lane', '09b_profile_evidence.js'), 'utf8');
  for (const imports of [[], ['peer2', 'peer7']]) {
    const globals = {A: {population_lineage: {root: '/population', imports}},
      PLANNER_PROFILE_EVIDENCE: true, USE_LEARNED_READ: true, LEARNED_DIR: '/learned',
      COMMANDMENT: '/eval/COMMANDMENT.md', WORKFLOW_DIR: '/workflow', CANONICAL: '/workspace',
      KK_OPERATOR: 'gemm', GFX: 'gfx950', KERNEL_NAME: 'fixture_kernel',
      runProcess: async request => ({status: 'completed', code: 0, group_cleared: true,
        stdout: JSON.stringify({paths: JSON.parse(request.args.at(-1))
          .map(path => ({path, kind: 'file', available: true}))})})};
    const value = await new Function(...Object.keys(globals), source + '\nreturn plannerProfileEvidenceInputs();')
      (...Object.values(globals));
    const commands = value.EVIDENCE_HANDOFF.commands.map(row => row.argv);
    const query = commands.find(argv => argv.includes('/workflow/scripts/query_population.py'));
    if (imports.length) assert.deepEqual(query.slice(2),
      ['--root', '/population', '--checkpoints-only', '--lineage', 'peer2', '--lineage', 'peer7']);
    else {
      assert.equal(query, undefined);
      assert.ok(value.EVIDENCE_HANDOFF.paths.every(row => row.path !== '/population'));
    }
    assert.deepEqual(commands.find(argv => argv.includes('/workflow/scripts/kb.py')).slice(2),
      ['--kb-dir', '/learned', 'match', '--operator', 'gemm', '--device', 'gfx950',
        '--kernel-name', 'fixture_kernel', '--max', '3']);
  }
}

(async () => {
  await attempts();
  await handoff();
  await parity();
  console.log('PASS: attempt outcomes, failed-profile planner retention, current-source identity and non-planner parity.');
})().catch(error => {console.error(error.stack || error); process.exitCode = 1;});
