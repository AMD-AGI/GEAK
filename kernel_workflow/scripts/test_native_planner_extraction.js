#!/usr/bin/env node
// CPU differential check of the shared native planner against the available approved J2 Git source.
// No model, GPU, network, archive creation, source materialization or task execution.
// Run: node kernel_workflow/scripts/test_native_planner_extraction.js
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');

const ROOT = path.resolve(__dirname, '../..');
const APPROVED = 'c7ef9b17803a25d858e191a8a1ca396b83fc7c79';
const read = p => fs.readFileSync(path.join(ROOT, p), 'utf8');
const approved = p => execFileSync('git', ['show', `${APPROVED}:${p}`], {cwd: ROOT, encoding: 'utf8'});
const slice = (source, first, after) => {
  const start = source.indexOf(first), end = source.indexOf(after, start + first.length);
  assert(start >= 0 && end > start, `source markers missing: ${first} / ${after}`);
  return source.slice(start, end);
};
const oldEvo = approved('kernel_workflow/src/kernel_lane/14a_evolution.js');
const newEvo = read('kernel_workflow/src/kernel_lane/14a_evolution.js');
const oldLoop = approved('kernel_workflow/src/kernel_lane/14_optimize_loop.js');
const newPlanner = read('kernel_workflow/src/kernel_lane/07a_native_planner.js');
const oldHelpers = oldEvo.slice(0, oldEvo.indexOf('// Retained population'));
const oldCase = slice(oldEvo, '  const caseEvidence = (row) =>', '  const selectedCases');
const oldPlan = slice(oldEvo, '    planInputs() {', '    memoryInputs(directions)');
const oldPrepare = slice(oldEvo, '    prepare(directions, round) {', '    async prepareWorkspace(direction)');
const oldSchema = slice(oldLoop, 'const evoPlanSchema =', 'let { cumulative');
const oldInputs = slice(oldLoop, '  let caseMeasurementObservation = null;', '  // Preserve parsed proposals');
const oldCall = slice(oldLoop, '  const callPlanner = async', '  let plan = await callPlanner');
const oldAllocation = slice(oldLoop, '  let directions = plan.directions.slice(', '    if (!directions.length)');
const helpers = approved('kernel_workflow/src/shared/helpers.js');
const role = approved('kernel_workflow/src/shared/role_agent.js');
const schema = slice(approved('kernel_workflow/src/kernel_lane/05_schemas.js'),
  'const PLAN_SCHEMA =', '// Optional author interpretation;');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const counts = {archive: 0, planner: 0, direction_selection: 0, seeded_unavailable: 0};

const cases = speedup => [{name: 'c', baseline_ms: 2, optimized_ms: 2 / speedup, speedup}];
const report = speedup => ({status: 'verified', primary_speedup: speedup, per_case: cases(speedup),
  strategy: 'fixture strategy', evidence_context: {
    case_measurement: {basis: 'native', cohort_ref: '/cohort', reduction: 'ratio_of_sums', weight_source: '/weights'},
    frozen_reference_per_case: cases(speedup), evidence_refs: ['/measurement']}});
const globals = (mode, controls = {}) => ({
  A: controls, CASE_MEASUREMENT_VIEW: mode, EXTERNAL_OBJECTIVE: null, HAS_WORKLOAD: false,
  log: () => {}, WORKFLOW_DIR: '/workflow', TASK: 'fixture task', DEEP_COST: 2,
  KERNEL_KNOWLEDGE_DIR: '/knowledge', KB_INPUTS: {SHARED_KB: '/shared_kb'},
  USE_LEARNED_READ: controls.learned !== false, KB_DIR_CAP: 1, KB_COLD_DIRECTION: true,
  primSpeedup: value => value?.primary_speedup ?? value?.speedup_geomean ?? null,
  expertSkillsBlock: () => '', multiGpuBlock: () => '', externalMetricInstructions: () => '',
  deliveryFactsInput: async () => ({EVALUATION_SERVICE: {configured_pool: ['0'], observation: false}}),
  agentT: async (prompt, options) => ({prompt, options}),
});

function archiveView(current, entries, holder, g) {
  const body = current ? `${newEvo}
    return evolutionDecisionView({base, baseCommit, archive, entries, getBestId: () => holder.value, metric});`
    : `${oldHelpers}
    let bestId = holder.value;
    const retained = id => entries.find(row => row.id === id && row.patch_ref && !row.retention_error);
    const defaultParent = () => retained(bestId)?.verified ? bestId : 'ORIGINAL';
    ${oldCase}
    const view = {${oldPlan}${oldPrepare}};
    return {entries, path: archive,
      planInputs() { bestId = holder.value; return view.planInputs(); },
      prepare(d, r) { bestId = holder.value; return view.prepare(d, r); }};`;
  return new Function(...Object.keys(g), 'entries', 'holder',
    `const base = '/base', baseCommit = 'fixture_commit', archive = '/archive', metric = {kind: 'geomean'};
    ${body}`)(...Object.values(g), entries, holder);
}

function archiveRows() {
  return Array.from({length: 15}, (_, i) => ({id: `r${i + 1}`, title: `title ${i}`, parent_id: 'ORIGINAL',
    donor_ids: [], status: i % 2 ? 'verified' : 'rejected', speedup: 1 + i / 10,
    patch_ref: i === 3 ? null : {path: `/patch/${i}`, sha256: String(i).padStart(64, '0')},
    retention_error: i === 4 ? 'fixture retention error' : undefined, verified: i === 0 || i === 14,
    engineer: report(1 + i / 10), verifier: report(1 + i / 10), reported_context: {notes: 'fixture notes'}}));
}

function compareArchives() {
  for (const mode of [undefined, 'native_consistent', 'current_mixed']) {
    for (const legacy of [undefined, false, true]) for (const planner of [undefined, false, true]) {
      const g = globals(mode, {evolution_case_evidence_view: legacy, evolution_planner_case_evidence_view: planner});
      const entries = archiveRows(), holder = {value: 'r1'};
      const before = JSON.stringify(entries);
      const oldView = archiveView(false, entries, holder, g), newView = archiveView(true, entries, holder, g);
      for (const best of ['r1', 'r15', 'missing', 'r4']) {
        holder.value = best;
        assert.deepStrictEqual(newView.planInputs(), oldView.planInputs()); counts.archive++;
        const directions = [
          {id: 'requested', parent_id: 'r2', donor_ids: ['ORIGINAL', 'r3', 'r3']},
          {parent_id: 'missing', donor_ids: ['r5', 'unknown']},
          {parent_id: 'r4', donor_ids: 'invalid'}, {}, {parent_id: 'ORIGINAL', donor_ids: []},
        ];
        assert.deepStrictEqual(newView.prepare(directions, 2), oldView.prepare(directions, 2)); counts.archive++;
      }
      assert.equal(JSON.stringify(entries), before, 'projection/resolution mutated archive entries');
      entries.push({id: 'r16', patch_ref: {path: '/patch/16'}, verified: true}); holder.value = 'r16';
      assert.deepStrictEqual(newView.planInputs(), oldView.planInputs()); counts.archive++;
      for (const invalidRound of [0, -1, 1.5, NaN]) {
        assert.throws(() => newView.prepare([], invalidRound), /positive evolution round/);
      }
    }
  }
}

async function plannerView(current, context, g) {
  const body = current ? `${newEvo}\n${newPlanner}
    context.evoPlanSchema = nativeEvolutionPlannerSchema(!!context.evo);
    context.getLeft = () => context.left;
    const view = nativePlannerCall(context);
    return {...view, setLeft: value => { context.left = value; }};`
    : `let {${Object.keys(context).join(', ')}} = context;
    const evolution = evo, startEvolution = null;
    ${oldHelpers}\n${oldSchema}\n${oldInputs}\n${oldCall}
    return {callPlanner, planInputs, caseMeasurementObservation: () => caseMeasurementObservation,
      setLeft: value => { left = value; }};`;
  return new AsyncFunction(...Object.keys(g), 'context', `${helpers}\n${schema}\n${role}\n${body}`)(
    ...Object.values(g), context);
}

async function comparePlanners() {
  for (const mode of [undefined, 'native_consistent', 'current_mixed']) for (const archive of ['none', 'empty', 'source']) {
    for (const record of [false, true]) for (const learned of [false, true]) {
      const g = globals(mode, {record_planner_inputs: record, learned, evolution_planner_case_evidence_view: true});
      const baseline = cases(1), winner = archive === 'source' ? report(2) : null;
      const holder = {value: 'r1'}, rows = archive === 'source' ? archiveRows() : [];
      const context = current => {
        const evo = archive === 'none' ? null : archiveView(current, rows, holder, g);
        return {EVAL_DIR: '/eval', round: 2, remaining: 4, cumulative: winner ? 2 : 1,
          handoffMetric: {kind: 'geomean'}, BASELINE_GEOMEAN_MS: 2, profileSummary: {bottleneck: 'memory'},
          bestPerCase: winner ? winner.per_case : baseline, finalWinner: winner,
          measurementRef: value => value ? {path: '/winner'} : null, history: {rounds: []},
          archiveHasSource: () => !!evo && evo.entries.some(row => row.patch_ref && !row.retention_error), evo,
          GFX: 'gfx950', DEVICE_TARGET: 'mi350x', PHYSICAL_CU_COUNT: 256, ROOFLINE_STATUS: 'known',
          researchBriefPath: learned ? '/research' : null, BASELINE_PER_CASE: baseline, left: 3600,
          KK_OPERATOR: 'gemm', KK_LANGUAGE: 'triton', KK_REFS: [], ENGINEER_CALL: {processDeadlineEpoch: 9000},
          DEADLINE_EPOCH: 10000, ENGINEER_RESERVE_S: 1000, ENGINEER_CUTOFF_EPOCH: 9000};
      };
      const oldView = await plannerView(false, context(false), g);
      const newView = await plannerView(true, context(true), g);
      for (const [attempt, left, forbidStop] of [['initial', 3600, false], ['replan1', 1800, true], ['replan2', 900, true]]) {
        oldView.setLeft(left); newView.setLeft(left);
        const intro = forbidStop ? 'Your stop was refused: plan at least one direction.' : "Decide this round's orthogonal directions (or stop).";
        const label = `tech_lead:${attempt} r2`;
        assert.deepStrictEqual(await newView.callPlanner(attempt, intro, forbidStop, label),
          await oldView.callPlanner(attempt, intro, forbidStop, label), `${mode}/${archive}/${record}/${learned}/${attempt}`);
        assert.deepStrictEqual(newView.caseMeasurementObservation(), oldView.caseMeasurementObservation());
        counts.planner++;
      }
    }
  }
}

function checkUnavailableCurrentMeasurement() {
  for (const mode of ['native_consistent', 'current_mixed']) {
    const g = globals(mode);
    const result = new Function(...Object.keys(g), `${newEvo}
      return currentBestCaseInputs(null, null, {unavailable: 'no bound current measurement'},
        {baselinePerCase: null, evalDir: '/prospective', allowInitialBenchmark: false});`)(...Object.values(g));
    assert.equal(result.CURRENT_BEST_PER_CASE, null);
    assert.equal(result.CURRENT_BEST_CASE_MEASUREMENT.unavailable, 'no bound current measurement');
    assert.equal(JSON.stringify(result).includes('initial_benchmark'), false);
    assert.equal(JSON.stringify(result).includes('baseline_timing.json'), false);
    counts.seeded_unavailable++;
  }
}

function compareDirectionSelection() {
  const plans = [[], [{id: 'a', specialty: 'memory'}, {specialty: 'compute'}],
    [{id: 'a', specialty: 'memory'}, {id: 'deep', specialty: 'deep_explore'}, {id: 'b', specialty: 'compute'}],
    [{idx: 99, id: '', specialty: 'deep_explore'}, {id: 'deep2', specialty: 'deep_explore'}]];
  for (const DEEP_COST of [1, 2, 4]) for (const evo of [false, true]) {
    for (const GPU_MODE of ['pin', 'pool', 'set']) for (const remaining of [0, 1, 2, 4]) for (const directions of plans) {
      const g = {DEEP_COST, GPU_MODE, GPU_JOB: '', GPU_LIST: ['0', '1'], GPU_POOL: '0,1', EVAL_DIR: '/eval'};
      const oldSelect = new Function(...Object.keys(g), 'plan', 'remaining', 'round', 'evo',
        `${oldAllocation}\n}\nreturn directions;`);
      const newSelect = new Function(...Object.keys(g), 'plan', 'remaining', 'round', 'evo',
        `${newPlanner}\nreturn nativeDirectionSelection(plan, remaining, round, evo).map(d => ({...d,
          gpu_id: GPU_MODE === 'pin' && !GPU_JOB ? GPU_LIST[d.idx % GPU_LIST.length] : GPU_POOL,
          out_dir: EVAL_DIR + '/round_' + round + '/engineer_' + d.idx}));`);
      const before = JSON.stringify(directions);
      assert.deepStrictEqual(newSelect(...Object.values(g), {directions}, remaining, 3, evo),
        oldSelect(...Object.values(g), {directions}, remaining, 3, evo));
      assert.equal(JSON.stringify(directions), before, 'direction sizing mutated the requested plan');
      counts.direction_selection++;
    }
  }
}

(async () => {
  compareArchives();
  await comparePlanners();
  compareDirectionSelection();
  checkUnavailableCurrentMeasurement();
  console.log(JSON.stringify({test: 'native_planner_extraction', approved_source_commit: APPROVED, status: 'pass', comparisons: counts}));
})().catch(error => { console.error(error); process.exitCode = 1; });
