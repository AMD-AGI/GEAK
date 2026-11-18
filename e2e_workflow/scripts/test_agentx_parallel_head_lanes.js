#!/usr/bin/env node
// Regression guard for the default-mode head track's scheduling (no GPU, no model needed).
//
// On a trace replay every head's bake-off and author lanes run at once, one leased card each, on
// the per-lane budgets of the serial loop, and the A/Bs follow serially in head order. Fixed
// ISL/OSL, a named ablation arm and a one-card run keep the serial loop.
//
// The dispatch and preparation half of the head track is EXTRACTED from the real source and run
// against mocked agents and lanes that record which card each job holds and when; the A/B half is
// replaced by a recorder.
//
// Run:  node e2e_workflow/scripts/test_agentx_parallel_head_lanes.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

const semStart = src.indexOf('function makeSem(ids) {');
const makeSemSrc = src.slice(semStart, src.indexOf('\n}\n', semStart) + 2);
const start = src.indexOf('  const headPool = ');
const end = src.indexOf('const { ext, isDominant, headCands } = p;', start);
ok(semStart !== -1 && start !== -1 && end > start, 'the default head track and makeSem located in e2e_workflow.js');

const FREE = [
  'KERNEL_EXTRACT_ON', 'ABLATION_ARM', 'GPU_LIST', 'heads', 'FAST_MODE', 'FAST_DEADLINE_HIT',
  'TIME_DEADLINE_HIT', 'log', 'headQueue', 'ELAPSED_MS', 'extractWithBaseline', 'EVAL_DIR', 'MODEL_PATH',
  'WORKLOAD', 'gemmSynthFor', 'profile', 'curOverlay', 'curFlags', 'curEnv', 'WORKFLOW_DIR', 'CONC',
  'EXTRACT_OP_SCHEMA', 'HEAD_PROTECT_PCT', 'extractFailWhy', 'flaggedHeads', 'history', 'safeAgent',
  'roleAgent', 'ENABLE_FP8', 'KERNEL_WF_DIR', 'KERNEL_BUDGET', 'OPBENCH_SCHEMA', 'HEAD_AUTHOR_MAX',
  'ablCandId', 'ABL_IMPL_MS', 'ABL', 'ablHeadReady', 'ABL_PREP_MS', 'ablDraw', 'ABL_AUDIT_P', 'ablEvent',
  'ABL_AUDIT_MS', 'laneWindowShort', 'A', 'fastBoundedWorkflow', 'KERNEL_WF_SCRIPT', 'KERNEL_KNOWLEDGE_DIR',
  'USE_EXPERT_SKILLS', 'EXPERT_SKILLS_DIR', 'KB_ARGS', 'FP8_MMA', 'GRAPH_REQ', 'TASK', 'parallel', '__integrate',
];
const body = `${makeSemSrc}
let headDispatched = 0, ABL_NESTED_CAP_MS = null;
${src.slice(start, end)}
    __integrate(h, p.headCands);
  }
  return { headDispatched, headQueue };`;
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const track = new AsyncFunction(...FREE, body);

const EIGHT = ['0', '1', '2', '3', '4', '5', '6', '7'];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function run({ extractOn = true, arm = '', gpus = EIGHT, nHeads = 3, langs = ['triton', 'hip'],
  deadline = false, refuse = () => false, transient = () => false } = {}) {
  const events = [], logs = [], lanes = [], bakes = [], integrated = [], clash = [];
  const busy = new Map();
  let inFlight = 0, maxInFlight = 0;
  const hold = async (gpu, label, ms) => {
    if (busy.has(gpu)) clash.push(`${label} on GPU ${gpu} while ${busy.get(gpu)} holds it`);
    busy.set(gpu, label);
    maxInFlight = Math.max(maxInFlight, ++inFlight);
    events.push(`start ${label}`);
    await sleep(ms);
    events.push(`end ${label}`);
    inFlight--;
    busy.delete(gpu);
  };
  const heads = Array.from({ length: nHeads }, (_, i) => ({
    short_name: `h${i}`, idx: i, gpu_id: gpus[i % gpus.length], pct_gpu_time: 10, prepared_ut: `ut${i}` }));
  const history = { ledger: [] };
  const env = {
    KERNEL_EXTRACT_ON: extractOn, ABLATION_ARM: arm, GPU_LIST: gpus, heads,
    FAST_MODE: false, FAST_DEADLINE_HIT: false, TIME_DEADLINE_HIT: deadline,
    log: (m) => logs.push(m),
    headQueue: heads.map((h) => ({ prepared_ut: h.prepared_ut })).concat([{ prepared_ut: 'rest' }]),
    ELAPSED_MS: 0,
    extractWithBaseline: async (role, task, desc, inputs) => {
      events.push(`extract ${inputs.KERNEL.short_name}`);
      return { smoke: 'pass', task_dir: `/ut/${inputs.KERNEL.short_name}`, op_kind: 'gemm' };
    },
    EVAL_DIR: '/eval', MODEL_PATH: '/model', WORKLOAD: {}, gemmSynthFor: () => null, profile: null,
    curOverlay: '', curFlags: '', curEnv: '', WORKFLOW_DIR: '/wf', CONC: 32, EXTRACT_OP_SCHEMA: {},
    HEAD_PROTECT_PCT: 30, extractFailWhy: () => 'x', flaggedHeads: [], history,
    roleAgent: (role, task, desc, inputs) => ({ inputs }),
    safeAgent: async (p, opts) => {
      const head = opts.label.split(' ')[1];
      bakes.push({ head, gpu: p.inputs.GPU_ID });
      await hold(p.inputs.GPU_ID, `bakeoff ${head}`, 5);
      return { gate: 'author_recommended', best_known_ms: 1, author_plan: langs.map((l) => ({ language: l, route: 'author' })) };
    },
    ENABLE_FP8: false, KERNEL_WF_DIR: '/kwf', KERNEL_BUDGET: 6, OPBENCH_SCHEMA: {}, HEAD_AUTHOR_MAX: 2,
    ablCandId: () => 'incumbent', ABL_IMPL_MS: 0, ABL: () => false, ablHeadReady: () => false, ABL_PREP_MS: 0,
    ablDraw: () => 1, ABL_AUDIT_P: 0, ablEvent: () => {}, ABL_AUDIT_MS: 0,
    laneWindowShort: (what) => refuse(what),
    A: {},
    fastBoundedWorkflow: async (ref, args, label) => {
      const attempt = lanes.filter((l) => l.label === label).length + 1;
      lanes.push({ label, gpu: args.gpu_ids, args, attempt });
      await hold(args.gpu_ids, `lane ${label}#${attempt}`, 20);
      return transient(label, attempt) ? { authored: false, validation_status: 'error', reason: 'died' }
        : { authored: true, final_weighted: 1.25, final_patch: `patch ${label}`, eval_dir: `/k/${label}` };
    },
    KERNEL_WF_SCRIPT: '/kwf/kernel_lane.js', KERNEL_KNOWLEDGE_DIR: '/kk', USE_EXPERT_SKILLS: false,
    EXPERT_SKILLS_DIR: '', KB_ARGS: {}, FP8_MMA: 'mfma', GRAPH_REQ: '', TASK: '',
    parallel: (ts) => Promise.all((ts || []).map((t) => Promise.resolve().then(t).catch(() => null))),
    __integrate: (h, cands) => {
      events.push(`integrate ${h.short_name}`);
      integrated.push({ head: h.short_name, sources: cands.map((c) => c.source) });
    },
  };
  const out = await track(...FREE.map((k) => env[k]));
  return { ...out, events, logs, lanes, bakes, integrated, clash, maxInFlight, history, heads };
}

const at = (r, e) => r.events.indexOf(e);
const lastLaneEnd = (r) => Math.max(...r.events.map((e, i) => (e.startsWith('end lane ') ? i : -1)));
const serialOrder = (r) => r.heads.every((h, i) => i === 0
  || at(r, `integrate ${r.heads[i - 1].short_name}`) < at(r, `extract ${h.short_name}`));
const laneArgs = (r, label) => {
  const l = r.lanes.find((x) => x.label === label);
  return l && JSON.stringify({ ...l.args, gpu_ids: null });
};

(async () => {
  console.log('\n# a trace replay on 8 cards: every head prepares at once, one card per job');
  const par = await run();
  ok(par.headDispatched === 3 && par.headQueue.map((c) => c.prepared_ut).join() === 'rest',
    'all three heads are dispatched and leave the queue; the rest stays for Milestone');
  ok(['h0', 'h1', 'h2'].every((h) => at(par, `start bakeoff ${h}`) < at(par, 'end bakeoff h0')),
    'the three bake-offs run at once');
  ok(par.lanes.length === 6 && par.maxInFlight === 6, 'the six author lanes run at once');
  ok(par.clash.length === 0, `no card is held by two jobs at once${par.clash.length ? `: ${par.clash[0]}` : ''}`);
  ok(par.bakes.every((b) => EIGHT.includes(b.gpu)) && par.lanes.every((l) => EIGHT.includes(l.gpu)),
    'every job runs on a card of the run');
  ok(par.lanes.every((l) => l.args.budget === 6 && l.args.gpu_ids === l.gpu && l.args.apply_to_original === 'false'),
    'every lane gets KERNEL_BUDGET and its own card');
  ok(par.lanes.every((l) => par.logs.some((m) => m.includes(`${l.label.split(':')[0]}: author ${l.label.split(':')[1]} lane on GPU ${l.gpu}.`))),
    'the log names the card of every lane');
  ok(par.integrated.map((x) => x.head).join() === 'h0,h1,h2'
    && par.integrated.every((x) => x.sources.join() === 'triton,hip'),
  'the A/Bs follow in head order, candidates in plan order');
  ok(at(par, 'integrate h0') > lastLaneEnd(par), 'the first A/B starts only after the last lane ends');

  console.log('\n# fewer cards than jobs: lanes wait for a card');
  const two = await run({ gpus: ['0', '1'] });
  ok(two.clash.length === 0 && two.maxInFlight === 2, 'never more jobs than cards, and both cards busy');
  ok(two.lanes.length === 6 && two.integrated.map((x) => x.head).join() === 'h0,h1,h2', 'every lane still runs, A/Bs in head order');

  console.log('\n# the serial loop is kept where heads depend on each other or on their own clock');
  for (const [what, opts] of [['fixed ISL/OSL', { extractOn: false }], ['a named ablation arm (A1)', { arm: 'A1' }],
    ['a one-card run', { gpus: ['0'] }]]) {
    const s = await run(opts);
    ok(s.maxInFlight === 1 && serialOrder(s), `${what}: one job at a time, each head integrated before the next is extracted`);
    ok(s.lanes.every((l) => l.gpu === s.heads[Number(l.label.slice(1, l.label.indexOf(':')))].gpu_id)
      && s.bakes.every((b) => b.gpu === s.heads[Number(b.head.slice(1))].gpu_id), `${what}: every job runs on its head's card`);
    ok(!s.logs.some((m) => m.includes('at once on GPUs')), `${what}: no parallel dispatch`);
  }
  const ser = await run({ arm: 'A1' });
  ok(['h0:triton', 'h0:hip', 'h2:hip'].every((l) => laneArgs(par, l) && laneArgs(par, l) === laneArgs(ser, l)),
    'a parallel lane gets the serial lane\'s arguments, the card aside');

  console.log('\n# per-lane guards are unchanged');
  const refused = await run({ refuse: (what) => what === 'author lane h1:hip' });
  ok(!refused.lanes.some((l) => l.label === 'h1:hip')
    && refused.history.ledger.some((r) => r.direction === 'h1:hip' && r.verdict === 'paused'),
  'a lane the start check refuses never runs, and is ledgered as paused');
  ok(refused.integrated.find((x) => x.head === 'h1').sources.join() === 'triton' && refused.lanes.length === 5,
    'the other lanes are unaffected');
  const retry = await run({ transient: (label, n) => label === 'h2:triton' && n === 1 });
  const tries = retry.lanes.filter((l) => l.label === 'h2:triton');
  ok(tries.length === 2 && tries[0].gpu === tries[1].gpu, 'a lane that dies transiently is retried once, on the same card');
  ok(retry.integrated.find((x) => x.head === 'h2').sources.join() === 'triton,hip', 'and its retry is a candidate');
  const late = await run({ deadline: true });
  ok(late.headDispatched === 0 && late.lanes.length === 0 && late.bakes.length === 0 && late.integrated.length === 0
    && late.logs.some((m) => m.includes('dispatch deadline reached')), 'past the dispatch deadline no head starts');

  if (failures) { console.error(`\nFAILED (${failures})`); process.exit(1); }
  console.log('\nall checks passed');
})().catch((e) => { console.error(e); process.exit(1); });
