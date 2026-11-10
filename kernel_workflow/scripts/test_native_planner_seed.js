#!/usr/bin/env node
// CPU integration of the generated lane, native planner/runtime, and fixed seed reader.
// No provider, GPU, network, task imports, archive materialization, or engineering.
// Run: node kernel_workflow/scripts/test_native_planner_seed.js
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {pathToFileURL} = require('node:url');

const ROOT = path.resolve(__dirname, '../..');
const WORKFLOW_DIR = path.join(ROOT, 'kernel_workflow');
const LANE = path.join(WORKFLOW_DIR, 'kernel_lane.js');
const READER = path.join(WORKFLOW_DIR, 'scripts/read_native_planner_seed.py');
const clone = value => JSON.parse(JSON.stringify(value));
const ref = file => {
  const bytes = fs.readFileSync(file);
  return {path: file, sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length};
};
const writeJSON = (file, value) => {
  fs.writeFileSync(file, JSON.stringify(value));
  return ref(file);
};

function fixture(root) {
  const dir = fs.mkdtempSync(path.join(root, 'case-'));
  const task = path.join(dir, 'task');
  fs.mkdirSync(path.join(task, 'kernel_src'), {recursive: true});
  const source = path.join(task, 'kernel_src/kernel.py');
  fs.writeFileSync(source, 'raise RuntimeError("CPU fixture must never import task code")\n');
  writeJSON(path.join(task, 'meta.json'), {schema: 'fixture.task/1'});
  writeJSON(path.join(task, 'case_manifest.json'), {cases: [{name: 'case0'}]});
  const ids = ['r1_d0', 'r1_d1', 'r1_d3'];
  const sources = ids.map((entry_id, i) => {
    const patch = path.join(dir, `${entry_id}.diff`);
    fs.writeFileSync(patch, `# frozen fixture patch ${i}; never apply\n`);
    return {...ref(patch), entry_id};
  });
  const patch = i => ({path: `/historical/${ids[i]}.diff`, sha256: sources[i].sha256,
    bytes: sources[i].bytes, base_commit: 'historical-unbound'});
  const original = {
    schema: 'geak.evolution/1', base: '/historical/base', base_commit: 'historical-unbound',
    baseline: {id: 'ORIGINAL', source: '/historical/base', speedup: 1},
    best_id: 'r1_d0', entries: [
      {id: 'r1_d0', status: 'verified', verified: true, speedup: 1.2, patch_ref: patch(0),
        base_commit: 'historical-unbound', engineer: {status: 'verified', primary_speedup: 1.2,
          per_case: [{name: 'case0', baseline_ms: 2, optimized_ms: 1, speedup: 2}],
          evidence_context: {case_measurement: {basis: 'native', cohort_ref: 'historical-cohort',
            reduction: 'ratio_of_sums', weight_source: 'historical-weights'},
            frozen_reference_per_case: [{name: 'case0', baseline_ms: 3, optimized_ms: 1, speedup: 3}],
            evidence_refs: ['historical-report']}}},
      {id: 'r1_d1', status: 'partial', verified: false, patch_ref: patch(1)},
      {id: 'r1_d2', status: 'failed', patch_ref: null, failure: 'source unavailable'},
      {id: 'r1_d3', status: 'failed', verified: false, patch_ref: patch(2), retention_error: 'rejected'},
    ],
  };
  const archive = clone(original);
  archive.base = task;
  archive.base_commit = null;
  archive.baseline.source = task;
  for (const row of archive.entries) {
    if (Object.hasOwn(row, 'base_commit')) row.base_commit = null;
    if (row.patch_ref) {
      row.patch_ref.path = sources.find(source => source.entry_id === row.id).path;
      row.patch_ref.base_commit = null;
    }
  }
  const seed = {
    schema: 'geak.native_planner_seed/1', id: 'J0_CPU_fixture', task_root: task,
    original_archive: writeJSON(path.join(dir, 'original_archive.json'), original),
    archive: writeJSON(path.join(dir, 'archive.json'), archive), source_refs: sources,
    task_files: ['meta.json', 'case_manifest.json', 'kernel_src/kernel.py'].map(p => ref(path.join(task, p))),
    context: {
      eval_dir: path.join(dir, 'uncreated_eval'), round: 5, budget_remaining: null,
      cumulative_speedup: null, baseline_geomean_ms: null, baseline_per_case: null,
      current_best_per_case: null, current_best_ref: null, final_winner: null,
      profile_summary: {unavailable: 'profile not bound'}, history: {unavailable: 'history not bound'},
      gfx: null, device_target: null, physical_cu_count: null, roofline_status: null,
      kk_operator: '', kk_language: '', kk_refs: [], research_brief_path: null,
      facts: {scope: 'CPU test, no current measurement or source qualification'},
    },
  };
  return {dir, task, source, sources, seed, archive,
    save: () => writeJSON(path.join(dir, 'seed.json'), seed)};
}

const direction = (id = 'caller-ID', more = {}) => ({id, title: 'fixture direction',
  specialty: 'memory', prompt: 'fixture only', parent_id: 'r1_d0', donor_ids: ['r1_d1'], ...more});
const plan = directions => ({stop: false, directions});

async function main() {
  const runtimeURL = p => pathToFileURL(path.join(ROOT, 'interface/runtime/engine', p)).href;
  const [{compileWorkflowBody, createRuntime}, {runProcess}, {schemaInstruction}, {neutralizeForBackend}] =
    await Promise.all(['run_workflow.mjs', 'backends/base.mjs', 'schema.mjs', 'config.mjs'].map(p => import(runtimeURL(p))));
  const compiled = compileWorkflowBody(fs.readFileSync(LANE, 'utf8'));
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-native-planner-seed-'));
  const passed = [];
  const test = async (name, run) => {
    try { await run(); passed.push(name); }
    catch (error) { error.message = `${name}: ${error.message}`; throw error; }
  };
  function harness(f, {args = {}, respond = () => plan([direction()]), schemaRetries = 1} = {}) {
    const hostCalls = [], agentCalls = [], backendCalls = [], phases = [];
    const rt = createRuntime({schemaRetries, concurrency: 1, agentTimeoutMs: 5000,
      log: () => {}, backend: {name: 'test', async runAgent(options) {
        backendCalls.push(options);
        const value = await respond(backendCalls.length, options);
        return {text: typeof value === 'string' ? value : JSON.stringify(value)};
      }}});
    // Match runScript's capability-bearing wrapper while retaining native rt.agent dispatch.
    const agent = (prompt, options) => {
      agentCalls.push({prompt, options});
      return rt.agent(prompt, options);
    };
    Object.defineProperty(agent, 'processTimeoutSupported', {value: true});
    Object.defineProperty(agent, 'processDeadlineActive', {value: false});
    const scriptArgs = {kernel_path: f.task, workflow_dir: WORKFLOW_DIR,
      native_planner_seed_ref: f.save(), native_planner_deadline_epoch: Math.floor(Date.now() / 1000) + 90,
      agent_timeout_ms: 2000, agent_retries: 1, warm_start: 'off',
      workload: [{name: 'case0', count: 1}], case_measurement_view: 'native_consistent',
      evolution_planner_case_evidence_view: true, ...args};
    const host = async options => {
      hostCalls.push(options);
      assert.equal(options.cmd, 'python3', 'only the read-only seed reader may execute');
      assert.deepEqual(options.args.slice(0, 3), ['-I', '-B', READER]);
      assert.equal(options.args.length, 6);
      assert(options.timeoutMs > 0 && options.timeoutMs <= 30000);
      return runProcess(options);
    };
    return {rt, scriptArgs, hostCalls, agentCalls, backendCalls, phases,
      run: () => compiled(agent, rt.parallel, rt.pipeline, title => phases.push(title), () => {},
        () => { throw new Error('nested workflow forbidden'); }, scriptArgs, rt.budget, host,
        setTimeout, clearTimeout, () => { throw new Error('deadline mutation forbidden'); })};
  }
  try {
    await test('native dispatch, schema retry, null context, and raw/resolved IDs', async () => {
      const f = fixture(root);
      const requested = plan([direction('caller-distinct', {donor_ids: ['r1_d1', 'r1_d1', 'ORIGINAL']}),
        direction('original-request', {parent_id: 'ORIGINAL', donor_ids: []})]);
      const h = harness(f, {respond: attempt => attempt === 1 ? {directions: []} : requested});
      const result = await h.run();
      assert.equal(result.status, 'resolved');
      assert.equal(result.continuation_materialization_qualified, false);
      assert.equal(result.source_closure_rechecked, true);
      assert.deepEqual(result.requested_plan, requested);
      assert.deepEqual(result.requested_selections[0], {index: 0, requested_id: 'caller-distinct',
        requested_parent_id: 'r1_d0', requested_donor_ids: ['r1_d1', 'r1_d1', 'ORIGINAL']});
      const [selected, original] = result.resolved_selections;
      assert.equal(selected.id, 'r5_d0');
      assert.equal(selected.requested_id, 'caller-distinct');
      assert.deepEqual(selected.requested_donor_ids, ['r1_d1', 'r1_d1', 'ORIGINAL']);
      assert.deepEqual(selected.donor_ids, ['r1_d1', 'ORIGINAL']);
      assert.equal(selected.parent_ref.sha256, f.sources[0].sha256);
      assert.equal(selected.donor_refs[0].sha256, f.sources[1].sha256);
      assert.equal(original.parent_ref.task_root, f.task);
      assert.deepEqual(original.parent_ref.task_files, f.seed.task_files);
      assert.equal(selected.native_resolution.parent_resolution, 'explicit');
      assert.deepEqual(selected.native_resolution.reference_errors, []);
      assert.equal(h.agentCalls.length, 1, 'schema retry stays within one rt.agent dispatch');
      assert.equal(h.backendCalls.length, 2);
      assert.equal(h.rt.state.schemaFails, 1);
      assert.equal(h.hostCalls.length, 2);
      assert.deepEqual(h.phases, [], 'no Setup or execution phase is entered');
      assert.equal(fs.existsSync(f.seed.context.eval_dir), false);
      const call = result.constructed_planner_call;
      assert.equal(call.boundary, 'before_agentT');
      assert.equal(call.prompt, h.agentCalls[0].prompt);
      assert.match(call.prompt, /^You are the tech_lead\. PHASE=plan_round\./);
      assert(call.prompt.includes(`First Read ${WORKFLOW_DIR}/roles/tech_lead.md`));
      assert(call.prompt.includes(`Decide this round's orthogonal directions (or stop). Read ${WORKFLOW_DIR}/roles/evolution.md`));
      assert.equal(call.options.phase, 'Optimize');
      assert.equal(call.options.label, 'tech_lead:plan r5');
      assert.equal(call.options.processDeadlineEpoch, h.scriptArgs.native_planner_deadline_epoch);
      assert.equal(h.agentCalls[0].options.processTimeoutMs, 2000);
      assert.deepEqual(call.options.schema.properties.directions.items.properties.parent_id, {type: 'string'});
      assert.deepEqual(call.options.schema.properties.directions.items.properties.donor_ids,
        {type: 'array', items: {type: 'string'}});
      const expected = neutralizeForBackend(call.prompt + schemaInstruction(call.options.schema), 'test');
      for (const backend of h.backendCalls) {
        assert(backend.prompt.startsWith(expected));
        assert.equal(backend.prompt.split('## OUTPUT CONTRACT (STRICT)').length - 1, 1);
        assert.match(backend.prompt.slice(expected.length), /Backend call allowance at dispatch:/);
        assert(backend.timeoutMs > 0 && backend.timeoutMs <= 2000);
        assert.equal(backend.label, 'tech_lead:plan r5');
      }
      const inputs = call.inputs;
      for (const key of ['BUDGET_REMAINING', 'CUMULATIVE_SPEEDUP', 'BASELINE_GEOMEAN_MS', 'CURRENT_BEST_PER_CASE'])
        assert.equal(inputs[key], null, `${key} must remain unknown`);
      assert.equal(inputs.EVOLUTION_BASE_COMMIT, null);
      assert.deepEqual(inputs.HISTORY, f.seed.context.history);
      assert.deepEqual(inputs.PROFILE_SUMMARY, f.seed.context.profile_summary);
      assert.equal(inputs.CURRENT_BEST_CASE_MEASUREMENT.unavailable, 'no bound current measurement');
      assert.equal(Object.hasOwn(inputs, 'MINUTES_REMAINING'), false);
      assert.equal(result.case_measurement_observation.raw_current_best.initial_original, false);
      assert.equal(result.case_measurement_observation.raw_current_best.native_primary, null);
      assert.equal(JSON.stringify(result).includes('baseline_timing.json'), false);
      assert.deepEqual(inputs.EVOLUTION_ENTRIES.map(row => [row.id, row.status]),
        f.archive.entries.map(row => [row.id, row.status]));
      assert.equal(inputs.EVOLUTION_ENTRIES[0].case_evidence.engineer.case_rows, 1);
      assert.deepEqual(inputs.PRIMARY_METRIC, {kind: 'time_weighted'});
      assert.equal(inputs.EVOLUTION_ENTRIES[0].case_evidence.metric, null,
        'a prospective objective cannot relabel unknown historical aggregate metrics');
      assert.equal(inputs.NATIVE_PLANNER_SEED_CONTEXT.historical_archive_metric, null);
      assert(result.direction_allocation.unavailable);
    });

    await test('native agentT retries backend failure', async () => {
      const h = harness(fixture(root), {args: {agent_retries: 2}, respond: attempt => {
        if (attempt === 1) throw new Error('fixture transient failure');
        return plan([direction()]);
      }});
      assert.equal((await h.run()).status, 'resolved');
      assert.equal(h.agentCalls.length, 2);
      assert.equal(h.backendCalls.length, 2);
      assert.equal(h.rt.state.schemaFails, 0);
    });

    await test('early conflicts and invalid references do not dispatch', async () => {
      for (const args of [{first_plan_ref: {}}, {prepared_original_ref: {}}, {state_dir: '/resume'},
        {population_lineage: {}}, {prepare_only: false}, {mode: 'author'}, {optimizer: 'greedy'},
        {search_duration_s: 10}, {preparation_allowance_s: 10}, {deadline_epoch: 1},
        {objective: {}}, {warm_start: 'on'}, {budget: 3}, {eval_dir: '/override'},
        {agent_timeout_ms: 0}, {agent_retries: 0}, {native_planner_seed_ref: {path: '/bad'}}]) {
        const h = harness(fixture(root), {args});
        await assert.rejects(h.run(), /native.planner.seed/);
        assert.equal(h.agentCalls.length, 0);
        assert.equal(h.hostCalls.length, 0);
      }
      for (const mutation of ['hash', 'schema', 'source']) {
        const f = fixture(root);
        if (mutation === 'schema') f.seed.schema = 'invalid/1';
        const h = harness(f);
        if (mutation === 'hash') h.scriptArgs.native_planner_seed_ref.sha256 = '0'.repeat(64);
        if (mutation === 'source') fs.appendFileSync(f.sources[0].path, '# modified\n');
        await assert.rejects(h.run(), /source closure rejected/);
        assert.equal(h.agentCalls.length, 0);
        assert.equal(h.hostCalls.length, 1);
      }
    });

    await test('missing, unknown, and rejected requested references never fall back', async () => {
      for (const change of [d => { delete d.parent_id; }, d => { delete d.donor_ids; },
        d => { d.parent_id = 'missing'; }, d => { d.parent_id = 'r1_d2'; },
        d => { d.parent_id = 'r1_d3'; }, d => { d.donor_ids = ['missing']; },
        d => { d.donor_ids = ['r1_d2']; }, d => { d.donor_ids = ['r1_d3']; }]) {
        const d = direction(); change(d);
        const h = harness(fixture(root), {respond: () => plan([d])});
        const result = await h.run();
        assert.equal(result.status, 'selection_rejected');
        assert(result.reference_errors.length > 0);
        assert.deepEqual(result.resolved_selections, []);
        assert.deepEqual(result.requested_plan, plan([d]));
        assert.equal(result.source_closure_rechecked, true);
        assert.equal(h.backendCalls.length, 1);
      }
    });

    await test('source mutation during planner call rejects useful resolution', async () => {
      for (const target of ['patch', 'task']) {
        const f = fixture(root);
        const h = harness(f, {respond: () => {
          fs.appendFileSync(target === 'patch' ? f.sources[0].path : f.source, '# modified by fake backend\n');
          return plan([direction()]);
        }});
        const result = await h.run();
        assert.equal(result.status, 'source_closure_changed');
        assert.equal(result.source_closure_rechecked, false);
        assert.equal(result.continuation_materialization_qualified, false);
        assert.deepEqual(result.resolved_selections, []);
        assert.equal(h.hostCalls.length, 2);
      }
    });

    await test('unknown budget preserves proposals; bound budget uses native deep sizing', async () => {
      const requested = plan([direction('normal'), direction('deep', {specialty: 'deep_explore'}), direction('last')]);
      for (const budget of [null, 2]) {
        const f = fixture(root); f.seed.context.budget_remaining = budget;
        const h = harness(f, {respond: () => requested});
        const result = await h.run();
        assert.equal(result.status, 'resolved');
        assert.deepEqual(result.requested_plan, requested);
        assert.deepEqual(result.resolved_selections.map(d => d.requested_id),
          budget === null ? ['normal', 'deep', 'last'] : ['deep']);
        if (budget === 2) {
          assert.equal(result.resolved_selections[0].direction_index, 1);
          assert.equal(result.resolved_selections[0].id, 'r5_d0');
          assert.deepEqual(result.direction_allocation, {budget_remaining: 2});
        }
      }
    });

    await test('native stop and exhausted schema errors return no resolution', async () => {
      for (const [response, status] of [[{stop: true, directions: []}, 'stopped'],
        [{directions: []}, 'planner_unavailable']]) {
        const h = harness(fixture(root), {respond: () => response});
        const result = await h.run();
        assert.equal(result.status, status);
        assert.deepEqual(result.resolved_selections, []);
        assert.equal(result.continuation_materialization_qualified, false);
        assert.equal(result.source_closure_rechecked, true);
      }
    });
    process.stdout.write(JSON.stringify({ok: true, tests: passed.length, passed}, null, 2) + '\n');
  } finally { fs.rmSync(root, {recursive: true, force: true}); }
}
main().catch(error => { process.stderr.write(String(error.stack || error) + '\n'); process.exitCode = 1; });
