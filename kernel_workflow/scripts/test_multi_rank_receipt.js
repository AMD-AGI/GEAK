#!/usr/bin/env node
// Regression guard for kernel_lane.js on multi-GPU jobs (no GPU, no agent, no network).
//
// The lane's multi-GPU additions -- the per-rank timing receipt (A01), the UT's prompt block with its
// whole-UT limit as the lease run timeout, and a wider agent wait limit -- apply only to a multi-GPU
// task delivered by the UT module (A22, "UT_MR"): the GPU count is above 1 once Setup has run, AND the
// task's own declaration, which Setup returns verbatim as task_declaration
// (detect_gpu_requirement.py --declaration), says more than one GPU. Every other job is main's lane
// byte for byte. Each lane run below is repeated with the same stub agents on main's kernel_lane.js:
//   A. single-GPU: every prompt, schema, wait limit, log line and the return equal main's, except the
//      Setup schema, which is main's SETUP_SCHEMA plus exactly the required task_declaration field.
//   B. a multi-GPU job that is not UT_MR (a torchrun test with no declaration, explicit or found by
//      Setup; a task that declares 1 GPU): the same.
//   C. UT_MR (the wf_neg_mimo_tp4 CI case, and a generated task as e2e dispatches it): Setup's prompt
//      is main's; every later prompt carries the UT block with GEAK_GPU_RUN_TIMEOUT=<the UT's limit>,
//      Benchmark/Report/Validate get GPUS_PER_JOB and the *_MR schemas, the return gets gpus_per_job,
//      timing_receipt, timing_receipt_gap, the agent wait limit widens -- and nothing else differs.
//   D. wf_neg fast tier: the per-rank receipt gate in the structured results.
//   E. a produced candidate needs per-rank candidate times too.
//   F. a GPU count Setup finds switches to UT_MR only with the task's declaration.
//   G. the multi-GPU schemas and the receipt checks on their own.
//   H. the Benchmark role: step 5a (per-rank baseline) under main's single-run setup of a weighted oracle.
// main's lane is read from $GEAK_MAIN_TREE/kernel_workflow/kernel_lane.js, else `git show d8bb12c8:`,
// else /src/commits/d8bb12c8. Without any of them A-C compare with pinned digests of main's runs
// (recompute them with main available and PRINT_DIGESTS=1 when a stub here changes).
// The workflow runtime globals are stubbed as in test_rdna4_policy.js; setTimeout only records the
// agent wait limit of each call. One `__expose(...)` call is injected after the schema block so the
// test compares the very objects the lane passes to agent().
//
// Run:  node kernel_workflow/scripts/test_multi_rank_receipt.js
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');

const WF_DIR = path.resolve(__dirname, '..');
const GEAK_ROOT = path.resolve(WF_DIR, '..');
const MAIN_COMMIT = 'd8bb12c8';
const LANE_REL = 'kernel_workflow/kernel_lane.js';
const asBody = (src) => src.replace(/^export const meta/m, 'const meta');
const SRC = fs.readFileSync(path.join(WF_DIR, 'kernel_lane.js'), 'utf8');
const EXPOSE_AT = 'const WARMSTART_RESOLVE_SCHEMA = ';
if (!SRC.includes(EXPOSE_AT)) {
  console.error(`FAIL: marker "${EXPOSE_AT}" not found in kernel_lane.js`);
  process.exit(1);
}
const EXPOSED = ['perCase', 'SETUP_SCHEMA', 'BENCH_SCHEMA', 'REPORT_SCHEMA', 'VALIDATE_SCHEMA', 'BENCH_SCHEMA_MR',
  'REPORT_SCHEMA_MR', 'VALIDATE_SCHEMA_MR', 'rankTimesOk', 'alignmentOk', 'benchRankGap', 'receiptGap',
  'benchReceipt'];
const BODY = asBody(SRC).replace(EXPOSE_AT, `__expose({ ${EXPOSED.join(', ')} });\n${EXPOSE_AT}`);

function readMain() {
  const sources = [];
  if (process.env.GEAK_MAIN_TREE) {
    sources.push(['$GEAK_MAIN_TREE', () => fs.readFileSync(path.join(process.env.GEAK_MAIN_TREE, LANE_REL), 'utf8')]);
  }
  sources.push([`git show ${MAIN_COMMIT}`, () => execFileSync('git', ['-C', GEAK_ROOT, 'show', `${MAIN_COMMIT}:${LANE_REL}`],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })]);
  sources.push([`/src/commits/${MAIN_COMMIT}`, () => fs.readFileSync(path.join('/src/commits', MAIN_COMMIT, LANE_REL), 'utf8')]);
  for (const [from, read] of sources) {
    try { return { from, body: asBody(read()) }; } catch (_e) { /* try the next source */ }
  }
  return null;
}
const MAIN = readMain();

let failures = 0;
const ok = (cond, msg, detail) => {
  if (!cond) { console.error('  FAIL:', msg, detail != null ? `-- ${detail}` : ''); failures++; }
  else console.log('  ok:', msg);
};
const clone = (v) => JSON.parse(JSON.stringify(v));

// --- fixtures: the wf_neg_mimo_tp4 numbers (one timed case, 4 ranks) --------------------------------
const SIG = 'swa_moe_r1_ctx128k';
const ALIGN = {
  host: 'torch.cuda.synchronize() then a gloo dist.barrier() over the 4 ranks before every timed window',
  counts: 'identical warmup / aligning / timed replay counts on every rank',
  aggregate: 'per window step time = max over ranks; case time = median over windows and rounds',
};
const BASE_RANKS = [0.054764, 0.054764, 0.054764, 0.054764];
const CAND_RANKS = [0.054881, 0.05488, 0.054881, 0.054881];
const benchMR = () => ({
  commandment_path: '/tmp/eval/COMMANDMENT.md', reliable: true, num_test_cases: 1,
  baseline_per_case: [{ name: SIG, latency_ms: 0.054764, rank_ms: BASE_RANKS, alignment: ALIGN }],
  baseline_geomean_ms: 0.054764, rank_timing_source: 'GEAK_PER_RANK_TIMING= / GEAK_TIMING_ALIGNMENT=',
});
const receipt = (withCurrent) => ({
  world_size: 4, source: 'run/ut_runs/<stamp>/summary.json timing', alignment: ALIGN,
  cases: { [SIG]: { baseline: { ms: 0.054764, rank_ms: BASE_RANKS },
                    ...(withCurrent ? { current: { ms: 0.054881, rank_ms: CAND_RANKS } } : {}) } },
});
// What `detect_gpu_requirement.py --declaration <task>` prints (test_detect_gpu_requirement.py pins
// the script): the wf_neg task declares 4 GPUs in run/requested_kernel.json and run/RETURN.json and
// has no meta.timeouts; a generated TP4 task declares world_size 4 and meta.timeouts.total_s 2400; a
// task without a declaration.
const WF_NEG_TASK = '/tmp/wf_neg_mimo_tp4/task';
const DECL_WF_NEG = { required_gpu_count: 4, ut_total_s: 3600, evidence: [
  `${WF_NEG_TASK}/run/requested_kernel.json: required_gpu_count = 4 (declared)`,
  `${WF_NEG_TASK}/run/RETURN.json: required_gpu_count = 4 (declared)`,
  `ut_total_s = 3600: ${WF_NEG_TASK}/meta.json: meta.timeouts.total_s absent: ut_runner's default for world_size 4 is 3600`] };
const GEN_TASK = '/tmp/eval/tasks/mono_tp4';
const DECL_GEN = { required_gpu_count: 4, ut_total_s: 2400, evidence: [
  `${GEN_TASK}/run/requested_kernel.json: required_gpu_count = 4 (declared)`,
  `${GEN_TASK}/run/RETURN.json: required_gpu_count = 4 (declared)`,
  `${GEN_TASK}/meta.json: capabilities.multi_rank.world_size = 4 (declared)`,
  `ut_total_s = 2400: ${GEN_TASK}/meta.json: meta.timeouts.total_s = 2400 (ut_runner timeouts, world_size 4)`] };
const declNone = (task) => ({ required_gpu_count: null, ut_total_s: 3600, evidence: [
  `${task}: no GPU count declared (run/requested_kernel.json or run/RETURN.json required_gpu_count, ` +
    'meta.json capabilities.multi_rank.world_size)',
  `ut_total_s = 3600: ${task}/meta.json: meta.timeouts.total_s absent: ut_runner's default for world_size 1 is 0, ` +
    'not a positive number of seconds; the multi-rank default 3600 is used'] });
const SETUP = {
  eval_dir: '/tmp/eval', workspace: '/tmp/eval/workspace', kernel_name: 'mimo_task', device_gfx: 'gfx942',
  device_target: 'unknown', physical_cu_count: 304, baseline_frozen: true, task_declaration: DECL_WF_NEG,
};
const setupWith = (extra) => ({ 'director:setup': { ...SETUP, ...extra } });
const DEFAULTS = {
  'director:setup': SETUP,
  'tech_lead:analyze': { kernel_type: 'hip', roadmap_summary: 'r', modifiable_files: ['kernel_src/x.hip'] },
  'benchmark_engineer': benchMR(),
  'profile_engineer': { bottleneck: 'latency', top_opportunities: [] },
  'reprofile': { bottleneck: 'latency', top_opportunities: [] },
  'clock': { epoch: 2000000000 },
  'tech_lead:plan': { stop: false, directions: [{ id: 'd1', title: 't', specialty: 'memory', prompt: 'p' }] },
  'eng ': { status: 'ok', speedup_geomean: 1.2, measurement_valid: true },
  'verify ': { status: 'verified', correctness: 'pass', verified_geomean: 1.2, per_case: [{ name: SIG, speedup: 1.2 }] },
  'commit ': { committed: true },
  'tech_lead:memory': { insights: [] },
  'tech_lead:report': { final_speedup_geomean: 1.0, report_path: '/tmp/eval/r.md', final_patch: '/tmp/eval/final_patch.diff' },
  'director:validate': { director_verified_speedup_geomean: 1.0, validation_status: 'accepted',
    timing_basis: 'not_applicable', timing_receipt: receipt(false) },
};
// /ci/kernel_workflow/wf_neg_mimo_tp4/lane_args.json with CASE_DIR=/tmp/wf_neg_mimo_tp4, GPUS=0,1,2,3
// (section C checks this copy against the file when the CI case is present).
const WF_NEG_TEXT = 'Author+optimize a hip implementation of this op vs the immutable oracle (beat ? ms). This kernel will be overlaid onto the LIVE decode path (CUDA-graph captured): its STEADY-STATE hot path MUST be host-sync-free (NO .item()/.cpu()/.tolist()/.sum().item()/torch.cuda.synchronize(), no Python branch on a GPU scalar). Cache any weight prep (transpose/requant/preshuffle) by weight.data_ptr() done ONCE, not per call. MEMORY FOOTPRINT IS A HARD CONSTRAINT: use the FUSED fp8 path (fold the block-scale into the operand scale, one fp8 MFMA GEMM) and cache only COMPACT fp8/preshuffled weights (never a bf16 expansion); the integrated kernel MUST fit at the accepted config\'s mem-fraction.  DEPLOYMENT REQUIREMENT — the server captures the steady-state decode path into a FULL CUDA/HIP graph, so this kernel runs INSIDE that captured graph. Your speedup MUST be INTRINSIC: better tiles/algorithm, fused quant (one fp8 MFMA, kill the dequant), or fewer ops/launches that reduce work INSIDE the captured region. Do NOT rely on a per-call CUDA/HIP-graph capture+replay WRAPPER for the speedup — inside the server\'s graph that wrapper falls back to eager and the win vanishes, and the e2e integrate gate WILL reject a wrapper-only win (this already happened: a 1.22x isolated MoE GEMM gave 0% e2e because only its static tile change survived the graph). The steady-state decode call must be graph-capture-safe: host-sync-free (no .item()/.cpu()/.tolist()/.synchronize(), no Python branch on a GPU scalar), shape-stable, and prep/compile ONCE (cache by data_ptr) so the captured region only LAUNCHES the kernel. VERIFY your speedup holds when the op is replayed under a CUDA graph, not just in eager timing.';
const WF_NEG = {
  kernel_path: WF_NEG_TASK, workflow_dir: WF_DIR, mode: 'optimize', target_language: 'hip',
  op_spec: { op_kind: 'decoder_layer', shapes: { tp: 4, rows: 1, context: 131072, hidden: 4096, q_heads_per_rank: 16,
    kv_heads_per_rank: 2, head_dim_qk: 192, head_dim_v: 128, rotary_dim: 64, window: 128, experts: 256, top_k: 8,
    expert_intermediate_per_rank: 512, kv_block_size: 64, kv_cache_dtype: 'fp8_e4m3fnuz',
    weights: 'qkv fp8 128x128 block scales; o_proj/router bf16; experts mxfp4 + e8m0' },
  dtype: 'bf16', regime: 'graph', cuda_graph_safe: true, workload_path: `${WF_NEG_TASK}/run/workload.json` },
  budget: 1, deadline_epoch: 1, warm_start: 'off', update_experience: 'off',
  gpu_ids: '0,1,2,3', gpus_per_job: 4, exp_root: '/tmp/wf_neg_mimo_tp4/kernels/_exp', task: WF_NEG_TEXT,
  apply_to_original: 'false',
};
const SINGLE = { ...WF_NEG, gpu_ids: '0', gpus_per_job: undefined };
const SINGLE_DEFAULTS = {
  'benchmark_engineer': { commandment_path: '/tmp/eval/COMMANDMENT.md', reliable: true,
    baseline_per_case: [{ name: SIG, latency_ms: 0.05 }], baseline_geomean_ms: 0.05 },
  'director:validate': { director_verified_speedup_geomean: 1.0, validation_status: 'accepted', timing_basis: 'not_applicable' },
};
// A main-style multi-GPU job: an all-reduce kernel whose test is a torchrun launch, no declaration.
const TORCHRUN_TASK = '/tmp/allreduce_task';
const TORCHRUN = { kernel_path: TORCHRUN_TASK, workflow_dir: WF_DIR, mode: 'optimize', target_language: 'triton',
  budget: 1, gpu_ids: '0,1', gpus_per_job: 2, exp_root: '/tmp/exp', apply_to_original: 'false',
  task: 'Optimize the all-reduce kernel; its correctness test runs torchrun --nproc_per_node=2 test_allreduce.py.' };
const TORCHRUN_SETUP = setupWith({ kernel_name: 'allreduce', task_declaration: declNone(TORCHRUN_TASK),
  gpu_requirement: { kind: 'fixed', gpus_per_job: 2, evidence: 'run.sh:1: torchrun --nproc_per_node=2 test_allreduce.py' } });
// A generated TP4 task as e2e's Milestone dispatches it (multiGpuLaneArgs: gpu_ids, gpus_per_job, op_spec).
const GEN = { kernel_path: GEN_TASK, workflow_dir: WF_DIR, use_expert_skills: 'false', expert_skills_dir: '', budget: 1,
  gpu_ids: '4,5,6,7', exp_root: '/tmp/eval/kernels/_exp',
  task: 'Compare candidate backends ["hip"] for this kernel; pick the fastest that passes the immutable unittest. ',
  apply_to_original: 'false', gpus_per_job: 4,
  op_spec: { op_kind: 'decoder_layer', shapes: { tp: 4, rows: 1, hidden: 4096 }, dtype: 'bf16', regime: 'graph',
    cuda_graph_safe: true, workload_path: `${GEN_TASK}/run/workload.json` } };

async function runLane(body, args, answers = {}) {
  const calls = [];
  const logs = [];
  const phases = [];
  let exposed = null;
  let wait = null;
  const table = { ...DEFAULTS, ...answers };
  const g = {
    args,
    phase: (t) => phases.push(t),
    log: (m) => logs.push(String(m)),
    workflow: async () => null,
    parallel: async (thunks) => Promise.all(thunks.map((t) => t())),
    pipeline: async (items, ...stages) => Promise.all(items.map(async (it, i) => {
      let v = it;
      for (const st of stages) v = await st(v, it, i);
      return v;
    })),
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
    __expose: (o) => { exposed = o; },
    // agentT's hung-agent guard: record each call's wait limit; never fire.
    setTimeout: (_fn, ms) => { wait = ms; return 0; },
    clearTimeout: () => {},
    agent: async (prompt, o) => {
      const key = Object.keys(table).find((k) => o.label === k || o.label.startsWith(k));
      const v = key && table[key] != null ? clone(table[key]) : null;
      calls.push({ label: o.label, prompt, schema: o.schema, wait, returned: v });
      wait = null;
      return v;
    },
  };
  const fn = new Function(...Object.keys(g), `return (async () => { ${body} })();`);
  let result = null;
  let error = null;
  try { result = await fn(...Object.values(g)); } catch (e) { error = e; }
  const call = (label) => calls.find((c) => c.label === label);
  return { result, error, calls, logs, phases, exposed, call };
}

// --- comparison with main ------------------------------------------------------------------------
// A run as compared: phases, every call's label, prompt, schema and wait limit, the logs and return.
const view = (run) => ({
  phases: run.phases, logs: run.logs, error: run.error ? String(run.error.message) : null,
  calls: run.calls.map((c) => ({ label: c.label, prompt: c.prompt, schema: JSON.stringify(c.schema), wait: c.wait })),
  result: run.result,
});
const digest = (v) => crypto.createHash('sha256').update(JSON.stringify(v).split(GEAK_ROOT).join('<GEAK>')).digest('hex');
// sha256 of view(main's run) per case; checked against main whenever main's lane is readable.
const PINNED = {
  single: 'baffae59bf67544b1aaf10de348e2c7ba6e51e5ae3e229776b838fede1415949',
  single_declared4: 'baffae59bf67544b1aaf10de348e2c7ba6e51e5ae3e229776b838fede1415949',
  torchrun: 'e032039f97320d245addd5c98e3f2f9cde289dca2f5576bab183c1aba82cbbc4',
  torchrun_found: '36a839b3c2bae51d55e1df2643ff2b6151b68e42702e2b24cdd57c55f672f6f4',
  torchrun_declared1: 'e032039f97320d245addd5c98e3f2f9cde289dca2f5576bab183c1aba82cbbc4',
  wf_neg: 'a30ec7e55a26c1c78a4f7a779b25bb33476f2eaf13ad5588d713c54a62ff3e81',
  wf_neg_rounds: '3b1be9e98c286f455512406a42ddbbb48c37888bc768a568d3904d6c84e9a430',
  wf_neg_profile_on: '8e2ab446fa482e18b6836fd71be125d4cc44dfb2faf69b13594769210a426b37',
  gen: '70d27a758583f821639d40a53943f4a4e3635acd3fabffe5ec0748a20924bba0',
};
const TASK_DECLARATION_SCHEMA = {
  type: 'object',
  properties: { required_gpu_count: { type: ['number', 'null'] }, ut_total_s: { type: 'number' },
    evidence: { type: 'array', items: { type: 'string' } } },
  required: ['required_gpu_count', 'ut_total_s', 'evidence'],
  additionalProperties: true,
};
// main's SETUP_SCHEMA from this lane's: drop exactly the task_declaration property and required entry.
function mainSetupSchema(schema, where) {
  const s = clone(schema);
  ok(JSON.stringify(s.properties.task_declaration) === JSON.stringify(TASK_DECLARATION_SCHEMA),
     `${where}: SETUP_SCHEMA declares task_declaration {required_gpu_count: number|null, ut_total_s, evidence[]}`);
  ok(s.required.filter((k) => k === 'task_declaration').length === 1 && s.required.at(-1) === 'task_declaration',
     `${where}: task_declaration is a required Setup field (appended to main's required list)`);
  delete s.properties.task_declaration;
  s.required = s.required.filter((k) => k !== 'task_declaration');
  return s;
}
const SENTENCE = 'This UT starts its own rank processes: run python3 unittest.py directly; do not wrap it in ' +
  'torchrun or another multi-process launcher.';
const REPORT_SENTENCE = ' Multi-GPU job: copy each case\'s per-rank baseline times and alignment from ' +
  'BASELINE_PER_CASE (rank_ms, alignment) into its per_case row (baseline_rank_ms, alignment); one merged ' +
  'time is not the whole result.';
const MR_RETURN_KEYS = ['gpus_per_job', 'timing_receipt', 'timing_receipt_gap'];
const blockAt = (p) => p.indexOf('\n\n## Multi-GPU job (');
// The role prompts of a run; on a multi-GPU job each ends with a multi-GPU block (main's too).
// (profile_engineer:baseline is one more, but only with args.profile=on: section C runs it once.)
const ROLE_CALLS = ['tech_lead:analyze', 'benchmark_engineer', 'tech_lead:report', 'director:validate'];
// The view main's lane would give for a UT_MR run of this lane: main's multi-GPU block in place of the
// UT block (none on the Optimize engineers, which main never gave one), no GPUS_PER_JOB input, no
// Report sentence, the base schemas, main's Setup schema, no UT_MR log line, none of the 3 keys, and
// every wait limit Setup's (agent_timeout_ms: Setup runs before the UT_MR widening).
function asMain(run, mainBlock) {
  const X = run.exposed;
  const base = new Map([[X.BENCH_SCHEMA_MR, X.BENCH_SCHEMA], [X.REPORT_SCHEMA_MR, X.REPORT_SCHEMA],
    [X.VALIDATE_SCHEMA_MR, X.VALIDATE_SCHEMA]]);
  const v = view(run);
  v.calls = run.calls.map((c, i) => {
    let p = c.prompt;
    if (i > 0 && blockAt(p) >= 0) p = p.slice(0, blockAt(p)) + (/^(eng|deep) /.test(c.label) ? '' : mainBlock);
    p = p.split('\n- GPUS_PER_JOB: 4\n').join('\n').split(REPORT_SENTENCE).join('');
    const schema = c.label === 'director:setup' ? mainSetupSchema(c.schema, 'UT_MR') : (base.get(c.schema) || c.schema);
    return { label: c.label, prompt: p, schema: JSON.stringify(schema), wait: run.calls[0].wait };
  });
  v.logs = run.logs.filter((l) => !l.startsWith('UT-module multi-GPU task: '));
  v.result = Object.fromEntries(Object.entries(run.result || {}).filter(([k]) => !MR_RETURN_KEYS.includes(k)));
  return v;
}
// Compares `expected` (this lane's run brought to main's form) with main's run of the same args and
// stubs, or with the pinned digest of it when main's lane is not readable.
async function sameAsMain(name, args, answers, expected, what) {
  if (!MAIN) {
    ok(PINNED[name] && digest(expected) === PINNED[name], `${what} (pinned digest of main's run)`, digest(expected));
    return null;
  }
  const m = await runLane(MAIN.body, args, answers);
  const mv = view(m);
  if (process.env.PRINT_DIGESTS) console.log(`    digest ${name}: ${digest(view(m))}`);
  ok(!PINNED[name] || digest(view(m)) === PINNED[name], `${name}: main's run still has its pinned digest`, digest(view(m)));
  const diff = [];
  if (JSON.stringify(mv.phases) !== JSON.stringify(expected.phases)) diff.push(`phases ${mv.phases} / ${expected.phases}`);
  if (mv.calls.length !== expected.calls.length) diff.push(`${mv.calls.length} / ${expected.calls.length} calls`);
  mv.calls.forEach((c, i) => {
    const e = expected.calls[i] || {};
    for (const k of ['label', 'prompt', 'schema', 'wait']) if (c[k] !== e[k]) diff.push(`call ${i} (${c.label}) ${k}`);
  });
  if (JSON.stringify(mv.logs) !== JSON.stringify(expected.logs)) diff.push('logs');
  if (JSON.stringify(mv.result) !== JSON.stringify(expected.result)) diff.push('return');
  if (mv.error !== expected.error) diff.push(`error ${mv.error} / ${expected.error}`);
  ok(!diff.length, what, diff.slice(0, 6).join('; '));
  return m;
}
// For a run that must be main's: its own view with main's Setup schema.
function plainAsMain(run, where) {
  const v = view(run);
  const setup = run.calls[0];
  if (setup && setup.label === 'director:setup') v.calls[0].schema = JSON.stringify(mainSetupSchema(setup.schema, where));
  return v;
}

const required = (schema) => schema.required || [];
const itemsOf = (schema, key) => schema.properties[key].items;

(async () => {
  console.log(`main's lane: ${MAIN ? MAIN.from : `not readable -- pinned digests of ${MAIN_COMMIT}`}`);
  // -------------------------------------------------------------------------
  console.log('\n# A. single-GPU: main\'s lane, except the Setup schema\'s task_declaration');
  const single = await runLane(BODY, SINGLE, { ...SINGLE_DEFAULTS, ...setupWith({ task_declaration: declNone(WF_NEG_TASK) }) });
  const X = single.exposed;
  ok(!single.error && X, 'the single-GPU fast-tier lane runs to its return', single.error && single.error.message);
  await sameAsMain('single', SINGLE, { ...SINGLE_DEFAULTS, ...setupWith({ task_declaration: declNone(WF_NEG_TASK) }) },
    plainAsMain(single, 'single-GPU'), 'single-GPU: every prompt, schema, wait limit, log line and the return are main\'s');
  for (const [label, name] of [['benchmark_engineer', 'BENCH_SCHEMA'], ['tech_lead:report', 'REPORT_SCHEMA'],
    ['director:validate', 'VALIDATE_SCHEMA']]) {
    ok(single.call(label).schema === X[name], `${label} gets the unchanged ${name} object`);
  }
  ok(single.call('director:setup').schema === X.SETUP_SCHEMA, 'director:setup gets SETUP_SCHEMA');
  ok(MR_RETURN_KEYS.every((k) => !(k in single.result)), 'the single-GPU return has no multi-rank key');
  ok(single.calls.every((c) => c.wait === 10800000), 'every agent waits the default 3 h');
  {
    // A task that declares 4 GPUs but runs on one is not UT_MR either.
    const answers = { ...SINGLE_DEFAULTS, ...setupWith({ task_declaration: DECL_WF_NEG }) };
    const r = await runLane(BODY, SINGLE, answers);
    await sameAsMain('single_declared4', SINGLE, answers, plainAsMain(r, 'single-GPU, declared 4'),
      'a task declaring 4 GPUs run on one GPU: main\'s lane');
  }

  // -------------------------------------------------------------------------
  console.log('\n# B. a multi-GPU job that is not UT_MR: main\'s lane, except the Setup schema\'s task_declaration');
  {
    const answers = { ...SINGLE_DEFAULTS, ...TORCHRUN_SETUP };
    const tr = await runLane(BODY, TORCHRUN, answers);
    ok(!tr.error && tr.phases.includes('Optimize'), 'the 2-GPU torchrun lane runs an optimization round to its return',
       tr.error && tr.error.message);
    await sameAsMain('torchrun', TORCHRUN, answers, plainAsMain(tr, 'torchrun'),
      'gpus_per_job=2, no declaration: every prompt (main\'s multi-GPU block, torchrun kept), schema, wait limit, log line and the return are main\'s');
    const bench = tr.call('benchmark_engineer');
    ok(bench.prompt.includes('(e.g. `torchrun --nproc_per_node=2 ...`)') && !bench.prompt.includes(SENTENCE) &&
       !bench.prompt.includes('GEAK_GPU_RUN_TIMEOUT=') && !bench.prompt.includes('GPUS_PER_JOB: 2'),
       'its block is the original one: torchrun example, lease default, no GPUS_PER_JOB input');
    ok(bench.schema === tr.exposed.BENCH_SCHEMA && MR_RETURN_KEYS.every((k) => !(k in tr.result)),
       'base schemas, no multi-rank return key');
    const found = await runLane(BODY, { ...TORCHRUN, gpus_per_job: undefined }, answers);
    ok(found.call('benchmark_engineer').prompt.includes('## Multi-GPU job (GPUS_PER_JOB=2)'),
       'without gpus_per_job, Setup\'s torchrun count makes it a 2-GPU job');
    await sameAsMain('torchrun_found', { ...TORCHRUN, gpus_per_job: undefined }, answers, plainAsMain(found, 'torchrun found'),
      'the same job sized by Setup\'s scan: main\'s lane');
    const one = { ...answers, ...setupWith({ ...TORCHRUN_SETUP['director:setup'], task_declaration: {
      required_gpu_count: 1, ut_total_s: 3600, evidence: [`${TORCHRUN_TASK}/meta.json: capabilities.multi_rank.world_size = 1 (declared)`] } }) };
    const d1 = await runLane(BODY, TORCHRUN, one);
    await sameAsMain('torchrun_declared1', TORCHRUN, one, plainAsMain(d1, 'declared 1'),
      'a 2-GPU job of a task that declares 1 GPU: main\'s lane');
  }

  // -------------------------------------------------------------------------
  console.log('\n# C. UT_MR: Setup as main, then the UT block, its lease limit and the receipts');
  const neg = await runLane(BODY, WF_NEG);
  let mainBlock = '';
  {
    const ciCase = process.env.GEAK_WF_NEG_CASE || '/ci/kernel_workflow/wf_neg_mimo_tp4';
    if (fs.existsSync(path.join(ciCase, 'lane_args.json'))) {
      const text = fs.readFileSync(path.join(ciCase, 'lane_args.json'), 'utf8')
        .split('${CASE_DIR}').join('/tmp/wf_neg_mimo_tp4').split('${GEAK}').join(GEAK_ROOT).split('${GPUS}').join('0,1,2,3');
      ok(JSON.stringify(JSON.parse(text)) === JSON.stringify(WF_NEG), `WF_NEG is ${ciCase}/lane_args.json`);
      let printed = null;
      try {
        printed = JSON.parse(execFileSync('python3', [path.join(WF_DIR, 'scripts/detect_gpu_requirement.py'), '--declaration',
          path.join(ciCase, 'task')], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }));
      } catch (_e) { /* no python3: the copy below stands */ }
      if (printed) {
        ok(JSON.stringify(JSON.parse(JSON.stringify(printed).split(path.join(ciCase, 'task')).join(WF_NEG_TASK))) ===
           JSON.stringify(DECL_WF_NEG), 'DECL_WF_NEG is what detect_gpu_requirement.py --declaration prints for its task');
      }
    } else console.log(`  (no ${ciCase}: the embedded copy of its lane_args.json and declaration are used)`);
    // DECL_GEN is what the script prints for a generated TP4 task laid out like GEN_TASK.
    const tmp = fs.mkdtempSync(path.join(require('os').tmpdir(), 'mr-decl-'));
    try {
      const put = (name, value) => {
        fs.mkdirSync(path.dirname(path.join(tmp, name)), { recursive: true });
        fs.writeFileSync(path.join(tmp, name), JSON.stringify(value));
      };
      put('meta.json', { capabilities: { multi_rank: { world_size: 4 } }, timeouts: { total_s: 2400 } });
      put('run/requested_kernel.json', { required_gpu_count: 4 });
      put('run/RETURN.json', { required_gpu_count: 4 });
      const printed = execFileSync('python3', [path.join(WF_DIR, 'scripts/detect_gpu_requirement.py'), '--declaration', tmp],
        { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
      ok(JSON.stringify(JSON.parse(printed.split(tmp).join(GEN_TASK))) === JSON.stringify(DECL_GEN),
         'DECL_GEN is what detect_gpu_requirement.py --declaration prints for a generated TP4 task');
    } catch (e) {
      if (e && e.code !== 'ENOENT') throw e;   // no python3: the copy stands
    } finally { fs.rmSync(tmp, { recursive: true, force: true }); }

    ok(!neg.error, 'the 4-GPU wf_neg fast-tier lane runs to its return', neg.error && neg.error.message);
    const setupPrompt = neg.call('director:setup').prompt;
    mainBlock = setupPrompt.slice(blockAt(setupPrompt));
    ok(blockAt(setupPrompt) > 0 && mainBlock.includes('(e.g. `torchrun --nproc_per_node=4 ...`)') &&
       mainBlock.includes('(default 900)') && !setupPrompt.includes(SENTENCE),
       'Setup (the GPU count is explicit, UT_MR not known yet) gets main\'s multi-GPU block');
    const UT_BLOCK = '\n\n## Multi-GPU job (GPUS_PER_JOB=4)\n' +
      'Every GPU command of this op needs 4 GPUs. `GPU_ID` is the group spec `pool:4:0,1,2,3`: one ' +
      '`gpu_lock.sh $GPU_ID <cmd>` holds the whole group for the whole command and sets ' +
      'HIP_VISIBLE_DEVICES to the leased ids, so start every rank inside that one call. ' + SENTENCE + ' ' +
      '(Only a test whose own documented command is a launcher, e.g. `torchrun --nproc_per_node=4 ...` in ' +
      'its config, keeps that command.) Never call gpu_lock.sh from inside a leased command (it is rejected) ' +
      'and never leave a lease running behind you: when a role step tells you to start a long UT run in the ' +
      'background and poll (`nohup ... &`), that run is one whole `gpu_lock.sh` call; wait for its `UT_EXIT=` ' +
      'line before the next GPU command and before you return. A leased command is killed after GEAK_GPU_RUN_TIMEOUT ' +
      'seconds, and this job\'s limit is the UT\'s whole-run limit, 3600 s: prefix every `gpu_lock.sh` and ' +
      '`profile_kernel.sh` call with `GEAK_GPU_RUN_TIMEOUT=3600` (the COMMANDMENT\'s GPU commands carry the ' +
      'same prefix). Leases on the same GPUs serialize, so put the arms you compare into one run.';
    const later = neg.calls.slice(1);
    const blocked = later.filter((c) => blockAt(c.prompt) >= 0);
    ok(ROLE_CALLS.every((l) => neg.call(l).prompt.endsWith(UT_BLOCK)) && blocked.every((c) => c.prompt.endsWith(UT_BLOCK)),
       'every prompt after Setup that has a multi-GPU block ends with the UT block: python3 unittest.py, ' +
       'GEAK_GPU_RUN_TIMEOUT=3600 (no meta.timeouts)', blocked.filter((c) => !c.prompt.endsWith(UT_BLOCK)).map((c) => c.label).join(', '));
    for (const label of ['benchmark_engineer', 'tech_lead:report', 'director:validate']) {
      ok(neg.call(label).prompt.includes('\n- GPUS_PER_JOB: 4\n'), `${label} gets GPUS_PER_JOB=4 as an input`);
    }
    ok(neg.call('benchmark_engineer').schema === neg.exposed.BENCH_SCHEMA_MR &&
       neg.call('tech_lead:report').schema === neg.exposed.REPORT_SCHEMA_MR &&
       neg.call('director:validate').schema === neg.exposed.VALIDATE_SCHEMA_MR,
       'Benchmark / Report / Validate get the multi-rank schemas');
    ok(MR_RETURN_KEYS.every((k) => k in neg.result), 'the return has gpus_per_job, timing_receipt, timing_receipt_gap');
    ok(neg.calls[0].wait === 10800000 && later.every((c) => c.wait === 5 * 3600 * 1000),
       'agent wait limit: Setup 3 h, then max(3 h, (3 + 2) x 3600 s) = 300 min',
       neg.calls.map((c) => c.wait).join(','));
    ok(neg.logs.some((l) => l.startsWith('UT-module multi-GPU task: 4 GPUs per job; the task declares 4 (') &&
       l.includes('Whole-UT limit 3600 s -> GEAK_GPU_RUN_TIMEOUT=3600 on every GPU command; agent wait limit ' +
       '18000000 ms = max(agent_timeout_ms 10800000, (3 + 2) x 3600 s).')), 'the lane logs the values it chose');
    await sameAsMain('wf_neg', WF_NEG, {}, asMain(neg, mainBlock),
      'wf_neg: Setup\'s prompt is main\'s, and later calls differ from main\'s only by the UT block, ' +
      'GPUS_PER_JOB, the *_MR schemas, the Report sentence and the 3 return keys');

    const rounds = { ...WF_NEG, deadline_epoch: undefined };
    const r = await runLane(BODY, rounds, { 'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: receipt(true) } });
    const eng = r.calls.find((c) => c.label.startsWith('eng '));
    ok(eng && eng.prompt.endsWith(UT_BLOCK), 'the inline Optimize engineer prompt ends with the UT block too');
    await sameAsMain('wf_neg_rounds', rounds, { 'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: receipt(true) } },
      asMain(r, mainBlock), 'with an optimization round: the same differences only (the engineer prompt gains the block)');

    const prof = await runLane(BODY, { ...WF_NEG, profile: 'on' });
    ok(!neg.call('profile_engineer:baseline') && prof.call('profile_engineer:baseline') &&
       prof.call('profile_engineer:baseline').prompt.endsWith(UT_BLOCK),
       'profiling is off by default (no profile call); with profile=on the profile engineer\'s prompt ends with the UT block too');
    await sameAsMain('wf_neg_profile_on', { ...WF_NEG, profile: 'on' }, {}, asMain(prof, mainBlock),
      'wf_neg with profile=on: the same differences from main only');
    const short = await runLane(BODY, WF_NEG, setupWith({ task_declaration: { ...DECL_WF_NEG, ut_total_s: 1200 } }));
    ok(short.calls.every((c) => c.wait === 10800000) && short.logs.some((l) => l.endsWith(
       'agent wait limit 10800000 ms = max(agent_timeout_ms 10800000, (3 + 2) x 1200 s).')),
       'a UT limit of 1200 s keeps the default 3 h: the wait limit is never below agent_timeout_ms');
    const off = await runLane(BODY, { ...WF_NEG, agent_timeout_ms: 0 });
    ok(off.calls.every((c) => c.wait === null) && off.logs.some((l) => l.endsWith('agent wait limit off (agent_timeout_ms 0).')),
       'agent_timeout_ms=0 keeps the wait limit off');
    const big = await runLane(BODY, { ...WF_NEG, agent_timeout_ms: 20000000 });
    ok(big.calls.every((c) => c.wait === 20000000), 'a larger agent_timeout_ms is kept');
    const bad = await runLane(BODY, WF_NEG, setupWith({ task_declaration: { ...DECL_WF_NEG, ut_total_s: 0 } }));
    ok(bad.call('benchmark_engineer').prompt.includes('`GEAK_GPU_RUN_TIMEOUT=3600`') &&
       bad.logs.some((l) => l.includes('Whole-UT limit 3600 s (task_declaration.ut_total_s 0 is not a positive number: multi-rank default)')),
       'an unusable ut_total_s from Setup falls back to 3600 s, never 0, and says so');
  }
  {
    const answers = { ...setupWith({ kernel_name: 'mono_tp4', task_declaration: DECL_GEN }),
      'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: receipt(true) } };
    const gen = await runLane(BODY, GEN, answers);
    ok(!gen.error && gen.phases.includes('Optimize'), 'a generated TP4 task dispatched like e2e\'s Milestone runs to its return',
       gen.error && gen.error.message);
    const later = gen.calls.slice(1);
    const blocked = later.filter((c) => blockAt(c.prompt) >= 0);
    ok(ROLE_CALLS.every((l) => blockAt(gen.call(l).prompt) > 0) && later.some((c) => /^eng /.test(c.label) && blockAt(c.prompt) > 0) &&
       blocked.every((c) => c.prompt.includes(SENTENCE) && c.prompt.includes('`GEAK_GPU_RUN_TIMEOUT=2400`') &&
       !c.prompt.includes('GEAK_GPU_RUN_TIMEOUT=3600')), 'its meta.timeouts.total_s=2400 is the lease limit of every later block');
    ok(gen.calls[0].wait === 10800000 && later.every((c) => c.wait === 12000000),
       'agent wait limit: Setup 3 h, then max(3 h, (3 + 2) x 2400 s) = 200 min');
    ok(MR_RETURN_KEYS.every((k) => k in gen.result) && gen.result.gpus_per_job === 4 && gen.result.timing_receipt_gap === '',
       'the return has the 3 keys');
    const genBlock = gen.calls[0].prompt.slice(blockAt(gen.calls[0].prompt));
    await sameAsMain('gen', GEN, answers, asMain(gen, genBlock),
      'generated task: differs from main\'s run only by the UT_MR additions');
  }

  // -------------------------------------------------------------------------
  console.log('\n# D. wf_neg_mimo_tp4 fast tier: per-rank receipt in the structured results');
  {
    ok(JSON.stringify(neg.phases) === JSON.stringify(['Setup', 'Analyze', 'Benchmark', 'Profile', 'Report', 'Validate']),
       'phases are Setup, Analyze, Benchmark, Profile, Report, Validate (no optimization round)', neg.phases.join(','));
    const r = neg.result;
    ok(r.validation_status === 'accepted' && r.timing_receipt_gap === '' && r.gpus_per_job === 4,
       'a complete director receipt keeps the verdict', `${r.validation_status} / ${r.timing_receipt_gap}`);
    ok(JSON.stringify(r.timing_receipt) === JSON.stringify(receipt(false)),
       'the lane returns the director\'s receipt: 4 per-rank baseline times + alignment per case');
    ok(neg.call('benchmark_engineer').returned.reliable === true, 'a per-rank baseline stays reliable');

    const noReceipt = await runLane(BODY, WF_NEG, { 'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: null } });
    const r2 = noReceipt.result;
    ok(r2.validation_status === 'flagged' && /no timing_receipt/.test(r2.timing_receipt_gap),
       'no director receipt -> flagged with the reason', `${r2.validation_status} / ${r2.timing_receipt_gap}`);
    ok(/no timing_receipt; one merged time is not a multi-rank result/.test(noReceipt.call('director:validate').returned.arbitration_note),
       'and the reason is in the director\'s arbitration_note');
    ok(r2.timing_receipt && r2.timing_receipt.world_size === 4 &&
       JSON.stringify(r2.timing_receipt.cases[SIG].baseline.rank_ms) === JSON.stringify(BASE_RANKS) &&
       JSON.stringify(r2.timing_receipt.alignment) === JSON.stringify(ALIGN),
       'the lane still returns the Benchmark\'s per-rank baseline record');

    const merged = { world_size: 4, alignment: ALIGN, cases: { [SIG]: { baseline: { ms: 0.0548, rank_ms: [0.0548] } } } };
    const r3 = (await runLane(BODY, WF_NEG, { 'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: merged } })).result;
    ok(r3.validation_status === 'flagged' && /no per-rank baseline times/.test(r3.timing_receipt_gap),
       'one merged time in the receipt -> flagged', r3.timing_receipt_gap);
    const noAlign = { ...receipt(false), alignment: null };
    const r4 = (await runLane(BODY, WF_NEG, { 'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: noAlign } })).result;
    ok(r4.validation_status === 'flagged' && /no alignment method/.test(r4.timing_receipt_gap),
       'per-rank times without the alignment method -> flagged', r4.timing_receipt_gap);

    const single1 = { ...benchMR(), baseline_per_case: [{ name: SIG, latency_ms: 0.054764, rank_ms: null, alignment: null }] };
    const nb = await runLane(BODY, WF_NEG, { 'benchmark_engineer': single1,
      'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: null } });
    const b = nb.call('benchmark_engineer').returned;
    ok(b.reliable === false && /without per-rank timing: baseline case swa_moe_r1_ctx128k has no per-rank times/.test(b.notes),
       'a Benchmark without per-rank times is marked reliable=false with the reason', b.notes);
    ok(nb.logs.some((l) => /Benchmark WARNING: 4-GPU job/.test(l)), 'and the lane logs it');
    ok(nb.result.validation_status === 'flagged' && nb.result.timing_receipt === null,
       'with no per-rank data anywhere the lane returns flagged and no receipt');
    // The same stubs on a job that is not UT_MR change nothing: no gate, no flag.
    const plain = await runLane(BODY, TORCHRUN, { ...TORCHRUN_SETUP, 'benchmark_engineer': single1,
      'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: null } });
    ok(plain.result.validation_status === 'accepted' && plain.call('benchmark_engineer').returned.reliable === true &&
       !plain.logs.some((l) => /WARNING: \d-GPU job/.test(l)), 'a job that is not UT_MR gets no receipt gate');
  }

  // -------------------------------------------------------------------------
  console.log('\n# E. a produced candidate needs per-rank candidate times too');
  {
    const rounds = { ...WF_NEG, deadline_epoch: undefined };
    const noCur = await runLane(BODY, rounds);
    ok(noCur.phases.includes('Optimize') && noCur.calls.some((c) => c.label.startsWith('commit ')),
       'one optimization round commits a candidate');
    ok(noCur.result.validation_status === 'flagged' && /no per-rank candidate times/.test(noCur.result.timing_receipt_gap),
       'a receipt without the candidate leg -> flagged', noCur.result.timing_receipt_gap);
    const withCur = await runLane(BODY, rounds, { 'director:validate': { ...DEFAULTS['director:validate'], timing_receipt: receipt(true) } });
    ok(withCur.result.validation_status === 'accepted' && withCur.result.timing_receipt_gap === '' &&
       JSON.stringify(withCur.result.timing_receipt.cases[SIG].current.rank_ms) === JSON.stringify(CAND_RANKS),
       'baseline + candidate per-rank legs -> accepted, both returned');
  }

  // -------------------------------------------------------------------------
  console.log('\n# F. a GPU count Setup finds: UT_MR only with the task\'s declaration');
  {
    const found = { kind: 'fixed', gpus_per_job: 4, evidence: 'run/RETURN.json: required_gpu_count = 4 (declared)' };
    const detected = await runLane(BODY, { ...WF_NEG, gpus_per_job: undefined }, setupWith({ gpu_requirement: found }));
    ok(!detected.call('director:setup').prompt.includes('Multi-GPU job'), 'Setup itself ran before the count was known');
    ok(detected.call('benchmark_engineer').schema === detected.exposed.BENCH_SCHEMA_MR &&
       detected.call('benchmark_engineer').prompt.includes('\n- GPUS_PER_JOB: 4\n') &&
       detected.call('benchmark_engineer').prompt.includes(SENTENCE),
       'with the declaration, the Benchmark after it gets the 4-GPU group, its schema and inputs');
    ok(detected.result.gpus_per_job === 4 && detected.result.validation_status === 'accepted' &&
       JSON.stringify(detected.result.timing_receipt) === JSON.stringify(receipt(false)),
       'and the lane returns the per-rank receipt');
    const undeclared = await runLane(BODY, { ...WF_NEG, gpus_per_job: undefined },
      setupWith({ gpu_requirement: found, task_declaration: declNone(WF_NEG_TASK) }));
    ok(undeclared.call('benchmark_engineer').prompt.includes('## Multi-GPU job (GPUS_PER_JOB=4)') &&
       undeclared.call('benchmark_engineer').schema === undeclared.exposed.BENCH_SCHEMA &&
       !undeclared.call('benchmark_engineer').prompt.includes(SENTENCE) &&
       MR_RETURN_KEYS.every((k) => !(k in undeclared.result)),
       'without it, the same 4-GPU job keeps main\'s block, schemas and return');
  }

  // -------------------------------------------------------------------------
  console.log('\n# G. the multi-GPU schemas and the receipt checks');
  {
    ok(JSON.stringify(X.BENCH_SCHEMA.required) === JSON.stringify(['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']) &&
       JSON.stringify(X.VALIDATE_SCHEMA.required) === JSON.stringify(['director_verified_speedup_geomean', 'validation_status']),
       'the base required lists are the historical ones');
    for (const name of ['perCase', 'BENCH_SCHEMA', 'REPORT_SCHEMA', 'VALIDATE_SCHEMA']) {
      ok(!/rank_ms|alignment|timing_receipt|rank_timing_source/.test(JSON.stringify(X[name])), `${name} declares no multi-rank field`);
    }
    const bItem = itemsOf(X.BENCH_SCHEMA_MR, 'baseline_per_case');
    for (const f of ['name', 'latency_ms', 'rank_ms', 'alignment']) {
      ok(required(bItem).includes(f), `Benchmark baseline_per_case items require ${f}`);
    }
    ok(JSON.stringify(bItem.properties.rank_ms.type) === '["array","null"]' &&
       JSON.stringify(bItem.properties.alignment.type) === '["object","null"]',
       'rank_ms / alignment are nullable: a missing value is stated, never filled with a merged time');
    ok(required(X.BENCH_SCHEMA_MR).includes('reliable'), 'the multi-GPU Benchmark must state reliable');
    ok(required(X.VALIDATE_SCHEMA_MR).includes('timing_receipt'), 'the multi-GPU director must return timing_receipt');
    for (const name of ['VALIDATE_SCHEMA_MR', 'REPORT_SCHEMA_MR']) {
      const s = X[name];
      const item = itemsOf(s, 'per_case');
      ok(['baseline_rank_ms', 'optimized_rank_ms', 'alignment'].every((f) => f in item.properties),
         `${name} per_case rows carry baseline_rank_ms / optimized_rank_ms / alignment`);
      const r = s.properties.timing_receipt;
      ok(r && ['world_size', 'alignment', 'cases'].every((f) => required(r).includes(f)),
         `${name} timing_receipt requires world_size, alignment, cases`);
      const caseSchema = r.properties.cases.additionalProperties;
      ok(required(caseSchema).includes('baseline') &&
         ['baseline', 'current'].every((leg) => ['ms', 'rank_ms'].every((f) => required(caseSchema.properties[leg]).includes(f))),
         `${name} receipt cases: baseline leg required, every leg carries the merged ms and rank_ms`);
    }
    for (const [mr, base] of [['BENCH_SCHEMA_MR', 'BENCH_SCHEMA'], ['REPORT_SCHEMA_MR', 'REPORT_SCHEMA'],
      ['VALIDATE_SCHEMA_MR', 'VALIDATE_SCHEMA']]) {
      ok(Object.keys(X[base].properties).every((k) => k in X[mr].properties) &&
         required(X[base]).every((k) => required(X[mr]).includes(k)),
         `${mr} keeps every field and requirement of ${base}`);
      ok(required(X[mr]).every((k) => k in X[mr].properties), `${mr}: required is a subset of properties`);
    }
    const { rankTimesOk, alignmentOk, benchRankGap, receiptGap, benchReceipt } = X;
    ok(rankTimesOk(BASE_RANKS, 4) && !rankTimesOk(BASE_RANKS, 2) && !rankTimesOk([0.05], 4),
       'rank_ms must hold exactly one value per rank');
    for (const v of [[0.05, 0.05, 0.05, 0], [0.05, 0.05, 0.05, NaN], [0.05, 0.05, 0.05, '0.05'], null, 0.05]) {
      ok(!rankTimesOk(v, 4), `rejects rank_ms ${JSON.stringify(v)}`);
    }
    ok(alignmentOk(ALIGN) && !alignmentOk({}) && !alignmentOk(null) && !alignmentOk('barrier') && !alignmentOk(['a']),
       'alignment must be a non-empty object');
    ok(benchRankGap(benchMR().baseline_per_case, 4) === '' && /no baseline case/.test(benchRankGap([], 4)),
       'a complete Benchmark record passes; an empty one does not');
    ok(/no alignment method/.test(benchRankGap([{ name: 'a', latency_ms: 0.05, rank_ms: BASE_RANKS, alignment: null }], 4)),
       'a Benchmark case without alignment is a gap');
    ok(/no merged time \(latency_ms\)/.test(benchRankGap([{ name: 'a', rank_ms: BASE_RANKS, alignment: ALIGN }], 4)),
       'a Benchmark case without its merged time is a gap');
    ok(receiptGap(receipt(false), 4, false) === '' && receiptGap(receipt(true), 4, true) === '',
       'complete receipts pass');
    ok(/world_size is 2, not 4/.test(receiptGap({ ...receipt(false), world_size: 2 }, 4, false)),
       'a receipt for another world size is a gap');
    ok(/no timed case/.test(receiptGap({ ...receipt(false), cases: {} }, 4, false)), 'an empty receipt is a gap');
    ok(/case swa_moe_r1_ctx128k has no merged baseline time/.test(receiptGap(
      { ...receipt(false), cases: { [SIG]: { baseline: { rank_ms: BASE_RANKS } } } }, 4, false)),
       'a leg without its merged (step) time is a gap');
    const old = { ...receipt(false), cases: { [SIG]: { baseline: { ms: 0.054764, rank_ms: BASE_RANKS },
                                                         candidate: { ms: 0.054881, rank_ms: CAND_RANKS } } } };
    ok(receiptGap(old, 4, true) === '', 'an old UT\'s `candidate` leg counts as the candidate leg');
    const built = benchReceipt(benchMR().baseline_per_case, 4, 'summary.json');
    ok(receiptGap(built, 4, false) === '' && built.source === 'Benchmark baseline_per_case (summary.json)',
       'the receipt built from the Benchmark record passes the same check');
  }

  // -------------------------------------------------------------------------
  console.log('\n# H. the Benchmark role under main\'s single-run setup of a weighted oracle');
  {
    const role = fs.readFileSync(path.join(WF_DIR, 'roles/benchmark_engineer.md'), 'utf8');
    const weighted = role.indexOf('**Weighted oracle:** skip the 3 runs.');
    const step5a = role.indexOf('### 5a. Multi-rank UT tasks: per-rank baseline');
    const ret = role.indexOf('## Return JSON');
    ok(weighted > 0 && step5a > weighted && ret > step5a,
       'step 5 ends with main\'s weighted-oracle paragraph; step 5a follows it, before Return JSON');
    const s5a = role.slice(step5a, ret);
    ok(s5a.includes('over your timed runs (3, or the single run of a weighted oracle)') && !s5a.includes('over your 3 runs'),
       'step 5a reads the per-rank baseline from the timed runs there are: 3, or a weighted oracle\'s single run');
    ok(s5a.includes('**Weighted oracle (single-run setup):**') &&
       s5a.includes('take the `next_run: full` path even when it says `usable: true`') &&
       s5a.includes('read `latency_ms`, `rank_ms` and `alignment` from that run'),
       'a multi-rank task times one full UT run of its own, also when the saved extraction baseline is usable');
    ok(role.includes('top-level `timing.per_case` of a generated `unittest.py`'),
       'the result layouts the role names for the baseline reader include the generated unittest.py\'s');
  }

  console.log(failures
    ? `\nFAIL: ${failures} check(s) failed.`
    : '\nPASS: UT-module multi-GPU tasks carry per-rank receipts and the UT\'s limits; every other job is main\'s lane.');
  process.exit(failures ? 1 : 0);
})();
