#!/usr/bin/env node
// Regression guard for multi-GPU (multi-rank) dispatch in e2e_workflow.js (no GPU, no model needed).
//
// A validated multi-rank UT (required_gpu_count N > 1) used to be deferred by packing because every
// e2e lane leased one GPU. Now packing queues it and e2e places it. Pinned here:
//
//   1. The AgentX extractor's prompt and inputs are main's (A23): the arguments e2e reads, the
//      kernel_extractor:prepare inputs and every rendered role prompt equal those of main's
//      e2e_workflow.js (7681c9f5) under the same arguments. A multi-rank UT leases SERVING_GPU, which
//      the UT module's own documents say; the workflow adds nothing for it.
//   2. makeSem refuses a request its pool can never hold instead of waiting forever.
//   3. After the envelope / carried state is read, a multi-GPU candidate whose path in this run cannot
//      hold it (pool too small, or fast/deep head lanes, which lease one card) moves to deferred_tasks
//      with the reason. Corrective re-authoring is excluded the same way.
//   4. Milestone and serial HeadKernel lease a whole group from GPU_LIST and call kernel_lane.js with
//      main's arguments plus only the lane's own gpu_ids=<group>, gpus_per_job=N and op_spec (A22): the
//      lane reads the UT's time limit from the task. Single-GPU candidates keep byte-identical call objects.
//
// The code under test is EXTRACTED from the real source and run against stubs. main's e2e_workflow.js
// is read from $GEAK_MAIN_TREE, else `git show 7681c9f5:`, else /src/commits/7681c9f5; without any of
// them section 1 compares with pinned digests of main's renderings.
//
// Run:  node e2e_workflow/scripts/test_multi_gpu_dispatch.js
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow.js'), 'utf8');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const MAIN_COMMIT = '7681c9f5';
const E2E_REL = 'e2e_workflow/e2e_workflow.js';
function readMain() {
  const sources = [];
  if (process.env.GEAK_MAIN_TREE) {
    sources.push(['$GEAK_MAIN_TREE', () => fs.readFileSync(path.join(process.env.GEAK_MAIN_TREE, E2E_REL), 'utf8')]);
  }
  sources.push([`git show ${MAIN_COMMIT}`, () => execFileSync('git', ['-C', path.resolve(ROOT, '..'), 'show',
    `${MAIN_COMMIT}:${E2E_REL}`], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })]);
  sources.push([`/src/commits/${MAIN_COMMIT}`, () => fs.readFileSync(path.join('/src/commits', MAIN_COMMIT, E2E_REL), 'utf8')]);
  for (const [from, read] of sources) {
    try { return { from, src: read() }; } catch (_e) { /* try the next source */ }
  }
  return null;
}
const MAIN = readMain();
const sha = (v) => crypto.createHash('sha256').update(typeof v === 'string' ? v : JSON.stringify(v)).digest('hex');

let failures = 0;
// A lease that never resolves drains the event loop and would exit 0 mid-test: fail unless the end is reached.
process.exitCode = 1;
let finished = false;
process.on('exit', () => { if (!finished) console.error('  FAIL: stopped before the end (a GPU lease never resolved?)'); });
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail !== undefined ? `\n    ${detail}` : ''); failures++; }
  else console.log('  ok:', msg);
};
function between(start, end, from = 0, text = src) {
  const a = text.indexOf(start, from), b = text.indexOf(end, a);
  if (a < 0 || b <= a) { console.error(`  FAIL: source block not found: ${start}`); process.exit(1); }
  return text.slice(a, b);
}
const tick = () => new Promise((resolve) => setImmediate(resolve));

const makeSemSrc = between('function makeSem(', '\n// ---- Multi-GPU candidates');
const helpers = between('// ---- Multi-GPU candidates', '// ---- SINGLE-KERNEL PASS-THROUGH');
const phaseSelector = between('const PHASES =', 'const ST = A.state');
const reuse = between('function preparedUtDefect(', '\n// abDone ==');
// main's PD/multinode helpers the lanes call; the harnesses run them single-node (SERVING_TOPOLOGY '').
const pdConsts = between('const IS_PD =', 'const MN_PY');
const pdHelpers = between('// The one PD_KINDS x PD_ROLES walk', 'function roleAgent(');
const engineHelpers = ['function engineWorkload(', 'function captureShard(', 'function engineRegimes(']
  .map((start) => between(start, '\n}\n') + '\n}\n').join('');
const singleNode = `${pdConsts}${pdHelpers}\n${engineHelpers}`;
const SINGLE_NODE_ENV = { SERVING_TOPOLOGY: '', SERVING_TP: 4, SERVING_GPU: '4,5,6,7', curPd: { prefill: {}, decode: {} } };

// A validated 4-rank task as agentx_kernel_preparation.mjs packs it (cf. /ci wf_neg_mimo_tp4).
function multiRank(name, phase = 'extract', extra = {}) {
  const result = { editable: true, task_dir: `/eval/tasks/${name}`, target_callable: 'pkg.mono.runner:MonoRunner.launch',
    baseline_frozen: true, op_kind: phase === 'extract' ? 'decoder_layer' : 'attn', required_gpu_count: 4,
    regime: 'graph', shapes: { tp: 4, rows: 1, hidden: 4096 }, dtype: 'bf16',
    workload_path: `/eval/tasks/${name}/run/workload.json`, candidate_backends: ['hip'],
    ...(phase === 'extract' ? { unittest_smoke: 'pass' } : { smoke: 'pass' }) };
  return { short_name: name, entity_kind: 'gpu_kernel', device_kernel: `${name}_k2`, pct_gpu_time: 24.86,
    op_kind: result.op_kind, regime: 'graph', required_gpu_count: 4, candidate_backends: ['hip'],
    prepared_ut: { phase, result_path: `/eval/tasks/${name}/run/RETURN.json`, sha256: 'h', result }, ...extra };
}
function singleGpu(name, phase = 'extract') {
  const c = multiRank(name, phase);
  delete c.required_gpu_count;
  delete c.prepared_ut.result.required_gpu_count;
  return c;
}

(async () => {
  // ------------------------------------------------------------------ 1. extractor inputs + prompt
  console.log('\n# the AgentX extractor gets main\'s prompt and inputs');
  console.log(`  main's e2e_workflow.js: ${MAIN ? MAIN.from : `not readable -- pinned digests of ${MAIN_COMMIT}`}`);
  const argsRead = (text) => [...new Set([...text.matchAll(/(?<![\w$.])A\s*(?:\?\.|\.)\s*([A-Za-z_$][\w$]*)/g)].map((m) => m[1])
    .concat([...text.matchAll(/(?<![\w$.])A\s*(?:\?\.)?\[\s*(['"])([^'"]+)\1\s*\]/g)].map((m) => m[2])))].sort();
  // sha256 of the sorted args.* names main's e2e_workflow.js reads.
  const MAIN_ARGS_SHA256 = '8bb90bb6f82a13e020fd4bdf651360ed262dbd57fa76cfba95653f5f1d921809';
  if (process.env.PRINT_DIGESTS && MAIN) console.log(`    main args: ${sha(argsRead(MAIN.src))}`);
  ok(sha(argsRead(src)) === MAIN_ARGS_SHA256 && (!MAIN || sha(argsRead(MAIN.src)) === MAIN_ARGS_SHA256),
     'e2e reads exactly main\'s args.* set', MAIN && [argsRead(src).filter((k) => !argsRead(MAIN.src).includes(k)),
       argsRead(MAIN.src).filter((k) => !argsRead(src).includes(k))].map((d) => d.join(',')).join(' / '));
  // The pieces each rendering runs, cut from one source the same way.
  function renderers(text) {
    const cut = (a, b) => between(a, b, 0, text);
    const prepare = cut('async function prepareAgentXKernels(', "\nif (want('setup'))");
    const settings = cut('const ISL_DECLARED =', '// Seed config:');
    const captureSettings = cut('const CAPTURE_BYTE_BUDGET =', 'const TASK =');
    const measurementSettings = cut('// One lifecycle for EVERY throughput number', '// CUDA/HIP-graph deployment requirement');
    const hook = text.match(/function kernelExtractBlock\(role\) \{[\s\S]*?\n\}/)[0];
    const role = text.match(/function roleAgent\(role, phase, intro, inputs\) \{[\s\S]*?\n\}/)[0];
    const prepareInputs = async () => {
      const calls = [];
      const env = { phase: () => {}, log: () => {}, EVAL_DIR: '/eval', MODEL_PATH: '/m', GPU_LIST: ['0', '1'],
        WORKLOAD: {}, TASK: 'Review extraction coverage.', curFlags: '--tp 4', curEnv: '', curOverlay: '/ov', profile: null,
        BUDGET: 1, HEAD_BUDGET: 1, remainingMin: () => 60, WORKFLOW_DIR: '/wf', TRACELENS_INPUTS: {},
        ANALYSIS_SKILL_INPUTS: {}, KB_REF_INPUTS: {}, BASELINE_TPUT: 1, AGENTX_PREPARATION_SCHEMA: {},
        MN: null, IS_PD: false, PD_ROLES: ['prefill', 'decode'], captureShard: () => ({}),
        roleAgent: (r, ph, intro, inputs) => ({ role: r, ph, intro, inputs }),
        safeAgent: async (p, o) => { calls.push({ ...p, label: o && o.label }); return { delivery_path: '/eval/d.json', prepared: {} }; },
        readAgentXPreparation: () => ({ head_candidates: [], kernel_candidates: [], failed_tasks: [], deferred_tasks: [] }) };
      await new AsyncFunction('env', `let {${Object.keys(env).join(',')}} = env;\n${prepare}\nreturn prepareAgentXKernels('baseline', 1);`)(env);
      return calls;
    };
    const roleAgentFor = (A) => new Function('A', 'process', `
      const WORKFLOW_DIR = '/geak/e2e_workflow', MODEL_PATH = '/models/m';
      const log = () => {}, BACKEND = 'vllm', SERVING_TP = 4, SERVING_GPU = '4,5,6,7';
      const GPU_IDS = '0,1,2,3,4,5,6,7', EFFECTIVE_CONFIG_DIGEST = '';
      const E2E_LEARNED_KB_ENABLED = false, ARCH_CONSTRAINTS = '', cfg = (o) => JSON.stringify(o);
      const expertSkillsBlock = () => '', warmStartBlock = () => '';
      const correctnessEnvBlock = () => '', workloadIdentityBlock = () => '';
      const SERVING_TOPOLOGY = '', MN = null, MN_SPEC = null, MN_SERVING_ENV = '';
      const pdOut = () => ({}), pdFragmentBlock = () => '', mnRoleBlock = () => '';
      ${settings}\n${measurementSettings}\n${captureSettings}\n${hook}\n${role}
      return roleAgent;`)(A, undefined);
    return { prepareInputs, roleAgentFor };
  }
  // Everything section 1 compares: the prepare call, and the prompts of the extractor and of every
  // other role, AgentX and fixed-load.
  async function renderings(text) {
    const r = renderers(text);
    const [call] = await r.prepareInputs();
    const out = { prepare: call };
    for (const [name, A] of [['agentx', { workload_kind: 'agentx_trace_replay' }], ['fixed', {}]]) {
      const ra = r.roleAgentFor(A);
      out[`${name}:kernel_extractor:prepare`] = ra('kernel_extractor', 'prepare', call.intro, call.inputs);
      out[`${name}:kernel_extractor:repair_delivery`] = ra('kernel_extractor', 'repair_delivery', 'repair', { GPU_ID: '0' });
      for (const role of ['director', 'system_architect', 'op_benchmarker', 'e2e_integrator', 'kernel_surgeon', 'config_tuner']) {
        out[`${name}:${role}`] = ra(role, 'x', 'intro', { GPU_ID: '0' });
      }
    }
    return out;
  }
  const mine = await renderings(src);
  ok(mine.prepare && mine.prepare.inputs.GPU_ID === '0', 'kernel_extractor:prepare inputs carry the single-GPU GPU_ID');
  // main's renderings (sha256 of each), pinned so CI without main's source still compares.
  const PINNED = {
    'prepare': '818e2b743eea72cbfd00bd0785b44c73c0510ccb7cc806d3547f318cb0756934',
    'agentx:kernel_extractor:prepare': '712703b511e93341dec74f1cd36f835b99c261b6587057ba7620cee067d73907',
    'agentx:kernel_extractor:repair_delivery': 'a50d7c645ae2e6f9b6e20ce0880abbbf5e851911b42ea3e95a231cbd86cab388',
    'agentx:director': '66ca811520a2d12af5a91fb6363d1c576c707bb5bd112644a5975f49dc137019',
    'agentx:system_architect': '87e4bcb83f2fcddd86b85cd27147e2b93c1f311d845a16bbbbf5dc7bb2941025',
    'agentx:op_benchmarker': '22ab9f4fb8354adbc00b9f5613d5ded341495a83f322680ca8097d9a8f7e70db',
    'agentx:e2e_integrator': '7ca9c16be27502631455f5424484b71cdbc4590bc16be213d781eb4756212279',
    'agentx:kernel_surgeon': '8fe38b50ce3710d67ce7f51a5d72836ffeaa8ec673821e9a1c8ea8ae6139c096',
    'agentx:config_tuner': 'e29446b4c32700ae49758a79de71a9788ff7d74ef69a11a2b55e9c5a5d3ccae3',
    'fixed:kernel_extractor:prepare': '6ada0021fb3ec520f875699dea3c0367ace51f797a2c8eacc18f32ef3f66d11d',
    'fixed:kernel_extractor:repair_delivery': '6f3fde8309e6ea3e83801969f760fed59ee7f3fd9fa9f51a85a6fbefc5923f61',
    'fixed:director': '7f57df6a1e55c8b5c46e411869a7fba22e5225abaf63a262a67bb3bfe31a2291',
    'fixed:system_architect': 'a5ae72f716ea47d0301ae8c3b768433930ac30b898d4c823bc5198f3f0f133ce',
    'fixed:op_benchmarker': '170faf222a098902938959ee1b4c8a6e17f1ed1d32855710ae1db812a673ce73',
    'fixed:e2e_integrator': '4ac13cf03bdaddd061292578486c793aa2add4b851d43d9f62ceb3a33d259a83',
    'fixed:kernel_surgeon': '339360713ec00a11dd8364dbdc576d24f679aa370b404556116a223ce8d6d66c',
    'fixed:config_tuner': 'b4b350dff34808c712a95c330249ccddc0d9b07c8a27fb344e088383e9fa7980',
  };
  const theirs = MAIN ? await renderings(MAIN.src) : null;
  if (theirs && process.env.PRINT_DIGESTS) {
    console.log(`    digests: ${JSON.stringify(Object.fromEntries(Object.entries(theirs).map(([k, v]) => [k, sha(v)])))}`);
  }
  const differ = Object.keys(PINNED).filter((k) => (theirs ? JSON.stringify(theirs[k]) !== JSON.stringify(mine[k])
    : sha(mine[k]) !== PINNED[k]));
  ok(Object.keys(mine).length === Object.keys(PINNED).length && !differ.length,
     `the kernel_extractor:prepare call (role, phase, intro, inputs) and ${Object.keys(PINNED).length - 1} rendered ` +
     'prompts (the extractor\'s, every other role\'s; AgentX and fixed-load) are byte-identical to main\'s', differ.join(', '));
  if (theirs) {
    const stale = Object.keys(PINNED).filter((k) => sha(theirs[k]) !== PINNED[k]);
    ok(!stale.length, 'and main\'s renderings still have their pinned digests', stale.join(', '));
  }

  // ------------------------------------------------------------------ 2. makeSem capacity
  console.log('\n# makeSem refuses what its pool can never hold');
  const makeSem = new Function(`${makeSemSrc}\nreturn makeSem;`)();
  const sem = makeSem(['0', '1']);
  const throws = (fn) => { try { fn(); return ''; } catch (e) { return e.message; } };
  ok(/GPU request for 4 card\(s\) cannot be held by the 2-card pool \{0,1\}/.test(throws(() => sem.acquire(4))),
    'a 4-card request on a 2-card pool throws at once');
  ok(throws(() => sem.acquire(NaN)) && throws(() => sem.acquire(0)) && throws(() => sem.acquire(1.5)),
    'a malformed count throws too (never waits)');
  ok(throws(() => sem.acquire(['7'])) && throws(() => sem.acquire(['0', '0'])),
    'an id list naming a card outside the pool, or one card twice, throws');
  let rejected = '';
  await sem.with(3, async () => {}).catch((e) => { rejected = e.message; });
  ok(/3 card\(s\) cannot be held/.test(rejected), 'with() rejects instead of hanging');
  const both = await sem.acquire(2);
  let got = null;
  const waiting = sem.acquire(1).then((g) => { got = g; });
  await tick();
  ok(got === null, 'a request the pool CAN hold still waits while the cards are busy');
  sem.release(both);
  await waiting;
  ok(got && got.length === 1, 'and is served on release');
  sem.release(got);
  const big = makeSem(['4', '5', '6', '7']);
  const one = await big.acquire(['6']);
  let group = null;
  const g4 = big.acquire(4).then((g) => { group = g; });
  await tick();
  ok(group === null, 'an id lease (a single-GPU task holding its own card) blocks a whole-pool group');
  big.release(one);
  await g4;
  ok(group.join(',') === '4,5,6,7', 'the group comes back in pool order', group.join(','));

  // ------------------------------------------------------------------ 3. envelope / carried-state check
  console.log('\n# multi-GPU candidates the run cannot place are deferred with the reason');
  function placement({ A = {}, FAST_MODE = false, DEEP_MODE = false, GPU_LIST, SERVING_GPU = GPU_LIST.join(','),
    HEAD_BUDGET = 1, heads = [], kernels = [], profile = { deferred_tasks: [] } }) {
    const logs = [];
    const e = { A, FAST_MODE, DEEP_MODE, DEEP: DEEP_MODE, KERNEL_EXTRACT_ON: true, GPU_LIST, SERVING_GPU, HEAD_BUDGET,
      profile, log: (m) => logs.push(String(m)), headQueue: heads, kernelQueue: kernels };
    return { ...new Function('env', `let {${Object.keys(e).join(',')}} = env;\n${phaseSelector}\n${helpers}
      headQueue = deferUnplaceable(headQueue, 'head');
      kernelQueue = deferUnplaceable(kernelQueue, 'kernel');
      return { headQueue, kernelQueue, profile };`)(e), logs };
  }
  const four = ['4', '5', '6', '7'];
  const fits = placement({ GPU_LIST: four, heads: [multiRank('h4', 'extract_op')], kernels: [multiRank('k4')] });
  ok(fits.headQueue.length === 1 && fits.kernelQueue.length === 1 && fits.profile.deferred_tasks.length === 0
    && fits.logs.length === 0, 'GPU_LIST of 4 holds a 4-GPU head (serial HeadKernel) and kernel (Milestone): both stay queued');
  const small = placement({ GPU_LIST: ['0', '1'], heads: [multiRank('h4', 'extract_op'), singleGpu('h1', 'extract_op')],
    kernels: [multiRank('k4'), singleGpu('k1')] });
  ok(small.headQueue.map((c) => c.short_name).join() === 'h1' && small.kernelQueue.map((c) => c.short_name).join() === 'k1',
    'a 2-GPU pool keeps the single-GPU candidates and moves both 4-GPU ones out');
  const d = small.profile.deferred_tasks;
  ok(d.length === 2 && d.every((t) => t.required_gpu_count === 4 && t.task_dir && t.result_path && t.prepared_ut),
    'deferred entries keep required_gpu_count, task_dir, result_path and the validated prepared_ut');
  ok(/needs 4 GPUs: the serial HeadKernel pool GPU_LIST \{0,1\} holds only 2/.test(d[0].reason)
    && /needs 4 GPUs: the Milestone pool GPU_LIST \{0,1\} holds only 2/.test(d[1].reason), 'the reason names the pool', d.map((t) => t.reason).join(' | '));
  ok(small.logs.length === 2 && small.logs.every((m) => /\[multi-gpu\] \w+ moved to deferred_tasks \(needs 4 GPUs/.test(m)),
    'each move is logged');
  const carried = [{ short_name: 'later', reason: 'budget' }];
  const resumed = placement({ GPU_LIST: ['0', '1'], kernels: [multiRank('k4')], profile: { deferred_tasks: carried } });
  ok(carried.length === 1 && resumed.profile.deferred_tasks.map((t) => t.short_name).join() === 'later,k4',
    'a resumed run appends to its carried deferred_tasks without mutating args.state');
  const fast = placement({ FAST_MODE: true, GPU_LIST: four, heads: [multiRank('h4', 'extract_op')], kernels: [multiRank('k4')] });
  ok(fast.headQueue.length === 0 && /fast-mode HeadKernel lanes lease one GPU each; that path is not wired for multi-GPU jobs/
    .test(fast.profile.deferred_tasks[0].reason), 'fast mode: a 4-GPU head is deferred (its lanes lease one card)');
  ok(fast.kernelQueue.length === 1, 'fast mode runs no Milestone, so a kernel candidate stays queued for the run that does');
  const deep = placement({ DEEP_MODE: true, GPU_LIST: ['0', '1', '2', '3', '4', '5', '6', '7'], SERVING_GPU: '0,1,2,3',
    heads: [multiRank('h4', 'extract_op')] });
  ok(deep.headQueue.length === 0 && /deep-mode HeadKernel lanes lease one GPU each.*dedicated pool \{4,5,6,7\} holds 4, serving pool \{0,1,2,3\} holds 4/
    .test(deep.profile.deferred_tasks[0].reason), 'deep mode: deferred, with both pools sized separately',
  deep.profile.deferred_tasks[0] && deep.profile.deferred_tasks[0].reason);
  const kernelOnly = placement({ A: { phases: 'kernel' }, GPU_LIST: ['0'], heads: [multiRank('h4', 'extract_op')] });
  ok(kernelOnly.headQueue.length === 1, 'a queue this invocation does not dispatch is left for the one that does');
  const bad = placement({ GPU_LIST: four, kernels: [multiRank('kx', 'extract', { required_gpu_count: 'four' })] });
  ok(bad.kernelQueue.length === 0 && /not a GPU count/.test(bad.profile.deferred_tasks[0].reason),
    'a malformed required_gpu_count is deferred, never read as one GPU');
  // Placement: once the envelope or carried state is read, before any phase dispatches a lane.
  const at = (s) => src.indexOf(s);
  const check = at("headQueue = deferUnplaceable(headQueue, 'head');");
  ok(check > at('  log(`Loaded carried state:') && check > at("strategy = await prepareAgentXKernels('baseline'")
    && check < at("phase('ConfigSweep');") && check < at("phase('HeadKernel');") && check < at("phase('Milestone');")
    && at("kernelQueue = deferUnplaceable(kernelQueue, 'kernel');") > check,
  'the check runs after the envelope/carried state is read and before every dispatching phase');

  // ------------------------------------------------------------------ 4. Milestone
  console.log('\n# Milestone leases a whole group for a multi-GPU candidate');
  const schedule = between('  const cands = planCands.slice(0, remaining)', '  // --- (b) PARALLEL optimize');
  const optimize = between('  const milestoneGpus = ', '  // --- serial integrate');
  async function milestone(planCands, GPU_LIST, laneFor = async () => ({ final_patch: '', final_weighted: 1 })) {
    const lanes = [], busy = new Set(), overlaps = [];
    const e = { planCands, GPU_LIST, remaining: planCands.length, milestone: 1, kernelQueue: [], dispatched: 0,
      BUDGET: 9, MIN_KERNEL_TASKS: 1, KERNEL_EXTRACT_ON: true, IS_AGENTX: true, log: () => {},
      makeSem, parallel: (jobs) => Promise.all(jobs.map((f) => f())),
      EVAL_DIR: '/eval', MODEL_PATH: '/m', WORKLOAD: {}, curOverlay: '', curFlags: '', curEnv: '', WORKFLOW_DIR: '/wf',
      profile: {}, EXTRACT_SCHEMA: {}, KERNEL_WF_SCRIPT: '/kwf/kernel_lane.js', KERNEL_WF_DIR: '/kwf',
      USE_EXPERT_SKILLS: false, EXPERT_SKILLS_DIR: '', KB_ARGS: {}, KERNEL_BUDGET: 1, GRAPH_REQ: '', TASK: '',
      ...SINGLE_NODE_ENV,
      reserveBoundedWorkflow: async (ref, args, label) => {
        const ids = String(args.gpu_ids).split(',');
        if (ids.some((id) => busy.has(id))) overlaps.push(`${label}:${args.gpu_ids}`);
        ids.forEach((id) => busy.add(id));
        lanes.push({ label, args });
        try { return await laneFor(label); } finally { ids.forEach((id) => busy.delete(id)); }
      } };
    const out = await new AsyncFunction('env', `let {${Object.keys(e).join(',')}} = env;\n${singleNode}\n${reuse}\n${helpers}
      ${schedule}\n${optimize}\nreturn { cands, optimized };`)(e);
    return { ...out, lanes, overlaps };
  }
  // The legacy single-GPU call object, built exactly as Milestone has always built it.
  const legacyCall = (c, gpu) => ({ kernel_path: c.prepared_ut.result.task_dir, workflow_dir: '/kwf',
    use_expert_skills: 'false', expert_skills_dir: '', budget: 1, gpu_ids: gpu, exp_root: '/eval/kernels/_exp',
    task: 'Compare candidate backends ' + JSON.stringify(c.candidate_backends || []) +
      ' for this kernel; pick the fastest that passes the immutable unittest. ', apply_to_original: 'false' });

  const solo = await milestone([multiRank('mono_tp4')], four);
  const call = solo.lanes[0] && solo.lanes[0].args;
  ok(call && call.gpu_ids === '4,5,6,7' && call.gpus_per_job === 4,
    'gpu_ids is the 4-card group of GPU_LIST (= SERVING_GPU here) and gpus_per_job=4', JSON.stringify(call));
  ok(call && JSON.stringify(call.op_spec) === JSON.stringify({ op_kind: 'decoder_layer', shapes: { tp: 4, rows: 1, hidden: 4096 },
    dtype: 'bf16', regime: 'graph', cuda_graph_safe: true, workload_path: '/eval/tasks/mono_tp4/run/workload.json' }),
  'op_spec is built like HeadKernel\'s, from prepared_ut.result, with cuda_graph_safe=true', call && JSON.stringify(call.op_spec));
  const mainCall = legacyCall(multiRank('mono_tp4'), '4');
  ok(call && JSON.stringify(Object.keys(call)) === JSON.stringify([...Object.keys(mainCall), 'gpus_per_job', 'op_spec'])
    && JSON.stringify({ ...call, gpu_ids: '4', gpus_per_job: undefined, op_spec: undefined }) === JSON.stringify(mainCall),
  'the Milestone call is main\'s plus only gpu_ids=<group>, gpus_per_job and op_spec (no ut_timeout_s, no agent_timeout_ms)',
  call && Object.keys(call).join(','));

  const singles = [singleGpu('a'), singleGpu('b'), singleGpu('c')];
  const plain = await milestone(singles, ['0', '1']);
  ok(plain.lanes.length === 3 && plain.lanes.every((l, i) =>
    JSON.stringify(l.args) === JSON.stringify(legacyCall(singles[i], ['0', '1', '0'][i]))),
  'single-GPU candidates: byte-identical call objects, GPU_LIST[i % n] as before', JSON.stringify(plain.lanes.map((l) => l.args.gpu_ids)));
  ok(plain.overlaps.length === 1, 'and no allocator is involved (two lanes still share card 0 as before)');

  const mixed = [singleGpu('a'), multiRank('mono_tp4'), singleGpu('c')];
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const pending = milestone(mixed, four, async (label) => { if (label === 'a') await gate; return { final_patch: '', final_weighted: 1 }; });
  await tick(); await tick();
  release();
  const mix = await pending;
  const byLabel = Object.fromEntries(mix.lanes.map((l) => [l.label, l.args]));
  ok(JSON.stringify(byLabel.a) === JSON.stringify(legacyCall(mixed[0], '4'))
    && JSON.stringify(byLabel.c) === JSON.stringify(legacyCall(mixed[2], '6')),
  'beside a multi-GPU candidate, single-GPU calls stay byte-identical (their own GPU_LIST[i % n] card)');
  ok(byLabel.mono_tp4 && byLabel.mono_tp4.gpu_ids === '4,5,6,7' && byLabel.mono_tp4.gpus_per_job === 4,
    'the multi-GPU candidate gets the whole group');
  ok(mix.overlaps.length === 0 && mix.lanes.map((l) => l.label).indexOf('mono_tp4') > mix.lanes.map((l) => l.label).indexOf('a'),
    'one allocator for the milestone: the group waits for the single-GPU task holding one of its cards', mix.overlaps.join());
  const eight = ['0', '1', '2', '3', '4', '5', '6', '7'];
  const wide = await milestone([singleGpu('a'), multiRank('mono_tp4')], eight);
  ok(wide.lanes.find((l) => l.label === 'mono_tp4').args.gpu_ids === '1,2,3,4' && wide.overlaps.length === 0,
    'with spare cards the group is taken beside the busy card, not on it');
  let open;
  const hold = new Promise((resolve) => { open = resolve; });
  const inside = milestone([multiRank('mono_tp4'), singleGpu('b')], eight,
    async (label) => { if (label === 'mono_tp4') await hold; return { final_patch: '', final_weighted: 1 }; });
  await tick(); await tick();
  open();
  const ins = await inside;
  ok(ins.overlaps.length === 0 && ins.lanes.map((l) => l.label).join() === 'mono_tp4,b'
    && ins.lanes[1].args.gpu_ids === '1',
  'a single-GPU task whose own card is inside the group waits for it, then runs on that card', ins.overlaps.join());
  let threw = '';
  await milestone([multiRank('too_big')], ['0', '1']).catch((e) => { threw = e.message; });
  ok(/cannot be held by the 2-card pool/.test(threw), 'a group larger than GPU_LIST fails loudly, never on one GPU');

  // ------------------------------------------------------------------ 5. serial HeadKernel
  console.log('\n# serial HeadKernel leases a whole group for a multi-GPU head');
  const headsMap = between('  const heads = headQueue.slice(0, HEAD_BUDGET)', '  if (DEEP) {');
  const serial = between('  const headGpus = heads.some(', '  } // end serial head track');
  async function headTrack(headQueue, GPU_LIST) {
    const agents = [], lanes = [];
    const e = { headQueue, GPU_LIST, HEAD_BUDGET: headQueue.length, KERNEL_EXTRACT_ON: true, FAST_MODE: false,
      FAST_DEADLINE_HIT: false, TIME_DEADLINE_HIT: false, headDispatched: 0, ELAPSED_MS: 0, makeSem, log: () => {},
      EVAL_DIR: '/eval', MODEL_PATH: '/m', WORKLOAD: {}, profile: {}, curOverlay: '', curFlags: '', curEnv: '', curTput: 1,
      WORKFLOW_DIR: '/wf', CONC: 4, EXTRACT_OP_SCHEMA: {}, OPBENCH_SCHEMA: {}, HEAD_PROTECT_PCT: 99, HEAD_AUTHOR_MAX: 2,
      ENABLE_FP8: false, KERNEL_WF_DIR: '/kwf', KERNEL_WF_SCRIPT: '/kwf/kernel_lane.js', KERNEL_BUDGET: 1,
      KERNEL_KNOWLEDGE_DIR: '/pk', USE_EXPERT_SKILLS: false, EXPERT_SKILLS_DIR: '', KB_ARGS: {}, MATRIX_CORE_NAME: 'MFMA',
      // Serial HeadKernel's author-lane task string interpolates FP8_MMA (main since gfx115x).
      FP8_MMA: 'fp8 MFMA',
      GRAPH_REQ: '', TASK: '', A: {}, ABL_IMPL_MS: 0, ABL: () => false, ablCandId: () => 'incumbent',
      laneWindowShort: () => false, gemmSynthFor: () => 'false', flaggedHeads: [], history: { ledger: [] },
      ...SINGLE_NODE_ENV,
      roleAgent: (role, ph, intro, inputs) => ({ role, ph, intro, inputs }),
      safeAgent: async (p) => { agents.push(p); return { gate: 'author_recommended', author_plan: [{ language: 'hip', route: 'rewrite' }] }; },
      fastBoundedWorkflow: async (ref, args, label) => { lanes.push({ label, args }); return { final_weighted: 0.9, final_patch: '' }; } };
    await new AsyncFunction('env', `let {${Object.keys(e).join(',')}} = env;\n${singleNode}\n${reuse}\n${helpers}
      ${headsMap}\n${serial}`)(e);
    return { agents, lanes };
  }
  const h4 = multiRank('mono_head', 'extract_op');
  const h1 = singleGpu('gemm_head', 'extract_op');
  const ht = await headTrack([h4, h1], ['0', '1', '2', '3', '4', '5', '6', '7']);
  const lane4 = ht.lanes.find((l) => l.label === 'mono_head:hip');
  const lane1 = ht.lanes.find((l) => l.label === 'gemm_head:hip');
  ok(lane4 && lane4.args.gpu_ids === '0,1,2,3' && lane4.args.gpus_per_job === 4 && lane4.args.mode === 'optimize',
    'the 4-GPU head\'s lane gets gpu_ids=<4-card group>, gpus_per_job=4', lane4 && JSON.stringify(lane4.args));
  ok(lane4 && lane4.args.op_spec.regime === 'graph' && lane4.args.op_spec.cuda_graph_safe === true
    && lane4.args.op_spec.workload_path === '/eval/tasks/mono_head/run/workload.json',
  'with the HeadKernel op_spec from prepared_ut.result');
  ok(lane4 && lane1 && JSON.stringify(Object.keys(lane4.args)) === JSON.stringify([...Object.keys(lane1.args), 'gpus_per_job']),
    'and otherwise the single-GPU head lane\'s keys: only gpus_per_job is added (no ut_timeout_s, no agent_timeout_ms)',
    lane4 && Object.keys(lane4.args).join(','));
  const bake4 = ht.agents.find((a) => a.role === 'op_benchmarker' && /mono_head/.test(JSON.stringify(a.inputs.OP_TASK_DIR)));
  ok(bake4 && bake4.inputs.UT_GPU_SET === '0,1,2,3' && bake4.inputs.GPU_ID === '0'
    && /runs 4 ranks: run it only under ONE lease of the whole UT_GPU_SET/.test(bake4.intro),
  'its bake-off holds the same group (UT_GPU_SET + a whole-set lease note)');
  const bake1 = ht.agents.find((a) => a.role === 'op_benchmarker' && /gemm_head/.test(JSON.stringify(a.inputs.OP_TASK_DIR)));
  ok(bake1 && bake1.intro === 'DISCOVER existing impls, tune cheap levers, DECIDE author_plan.'
    && bake1.inputs.GPU_ID === '1' && !('UT_GPU_SET' in bake1.inputs), 'a single-GPU head\'s bake-off is unchanged');
  const keys1 = lane1 && Object.keys(lane1.args).join(',');
  ok(lane1 && lane1.args.gpu_ids === '1' && keys1 === 'kernel_path,workflow_dir,mode,target_language,op_spec,' +
    'perf_knowledge_dir,use_expert_skills,expert_skills_dir,budget,gpu_ids,exp_root,task,apply_to_original',
  'a single-GPU head\'s lane call keeps h.gpu_id and exactly its old keys', keys1);
  ok(JSON.stringify(lane1.args.op_spec) === JSON.stringify({ op_kind: 'attn', shapes: { tp: 4, rows: 1, hidden: 4096 },
    dtype: 'bf16', regime: 'graph', cuda_graph_safe: true, workload_path: '/eval/tasks/gemm_head/run/workload.json' }),
  'and its old op_spec');

  // ------------------------------------------------------------------ 6. corrective
  console.log('\n# corrective re-authoring is excluded for a multi-GPU candidate');
  const corrective = between('async function tryCorrectiveReauthor(', '\n// --- FAST-MODE wall-clock control');
  async function correct(spec) {
    const started = [], logs = [], profile = { deferred_tasks: [] };
    const e = { ...SINGLE_NODE_ENV, rejectClass: () => 'integration', HEAD_CORRECTIVE_MAX: 1, FAST_MODE: false, FAST_DEADLINE_HIT: false,
      TIME_DEADLINE_HIT: false, GPU_LIST: ['0', '1', '2', '3'], SERVING_GPU: '0,1,2,3', profile, log: (m) => logs.push(String(m)),
      trySurgicalFix: async () => { started.push('surgeon'); return null; },
      fastBoundedWorkflow: async (ref, args) => { started.push(`lane:${args.gpu_ids}`); return { authored: false }; },
      KERNEL_WF_SCRIPT: '/k', KERNEL_WF_DIR: '/k', KERNEL_KNOWLEDGE_DIR: '', USE_EXPERT_SKILLS: false, EXPERT_SKILLS_DIR: '',
      KB_ARGS: {}, WARM_START_OFF: true, KERNEL_BUDGET: 1, EVAL_DIR: '/eval', GRAPH_REQ: '', TASK: '', theSpec: spec };
    const r = await new AsyncFunction('env', `let {${Object.keys(e).join(',')}} = env;\n${singleNode}\n${helpers}\n${corrective}
      return tryCorrectiveReauthor(theSpec);`)(e);
    return { r, started, logs, deferred: profile.deferred_tasks };
  }
  const base = { short_name: 'k', task_dir: '/eval/tasks/k', kernel_eval_dir: '/eval/k', isolated: 1.3, reason: 'NO_BINARY_FOR_GPU', gpu_id: '0' };
  const c4 = await correct({ ...base, required_gpu_count: 4 });
  ok(c4.r.banked === false && c4.started.length === 0, 'no surgeon and no one-GPU re-author lane for a 4-GPU candidate');
  ok(c4.deferred.length === 1 && c4.deferred[0].required_gpu_count === 4 && c4.deferred[0].task_dir === '/eval/tasks/k'
    && /corrective re-author lanes lease one GPU each; that path is not wired for multi-GPU jobs/.test(c4.deferred[0].reason),
  'it is recorded in deferred_tasks with the reason', c4.deferred[0] && c4.deferred[0].reason);
  const c1 = await correct({ ...base, required_gpu_count: 1 });
  ok(c1.started.join() === 'surgeon,lane:0' && c1.deferred.length === 0, 'a single-GPU candidate is still corrected on its card');
  ok((src.match(/required_gpu_count: gpuNeed\((?:h|c)\)/g) || []).length === 3,
    'every corrective call site passes the candidate\'s GPU count');

  finished = true;
  console.log(failures === 0 ? '\nPASS: multi-GPU candidates are placed whole or deferred with a reason; single-GPU dispatch is unchanged.'
    : `\nFAILED: ${failures} assertion(s).`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(1); });
