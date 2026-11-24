// Runs kernel_lane.js against scripted agents (no GPU, no LLM) and prints one JSON line:
// {result, calls: [{label, phase, prompt}], workflows: [...]}.
//   node lane_mockrun.mjs <scenario> [lane script]
// Scenarios: optimize, author, multigpu (Setup finds 2 GPUs per run), stop (planner stops at round 1),
// deadline (a wall-clock window: the clock agent and the no-early-stop re-plan).
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const [scenario = 'optimize', script = resolve(HERE, '..', '..', 'kernel_lane.js')] = process.argv.slice(2);
const E = '/x/eval';
const calls = [];
const workflows = [];
let epoch = 1000000;

const perCase = (s) => [{ name: 'c1', baseline_ms: 1, optimized_ms: 1 / s, speedup: s },
  { name: 'c2', baseline_ms: 2, optimized_ms: 2 / s, speedup: s }];
const SPEED = { d1: 1.3, d2: 1.1, d3: 1.05 };
const profile = { bottleneck: 'memory', top_opportunities: ['coalesce loads'], dispatch_count: 2 };

const answers = [
  [/^director:setup$/, () => ({ eval_dir: E, workspace: `${E}/workspace`, baseline_dir: `${E}/baseline`,
    kernel_name: 'k', device_gfx: 'gfx950', device_target: 'mi355x', device_name: 'MI355X',
    physical_cu_count: 256, baseline_frozen: true, baseline_callable: 'm:f',
    gpu_requirement: scenario === 'multigpu' ? { kind: 'fixed', gpus_per_job: 2, evidence: 'torchrun 2' }
      : { kind: 'single' } })],
  [/^author:/, () => ({ authored: true, target_language: 'triton', correctness: 'pass', baseline_ms: 1,
    kernel_src_path: `${E}/workspace/kernel.py`, entry_point: 'f', build: true })],
  [/^tech_lead:analyze$/, () => ({ kernel_type: 'gemm', roadmap_summary: 'tile + fuse', candidate_directions: [],
    kk_operator: null, kk_language: null, kk_refs: [] })],
  [/^benchmark_engineer$/, () => ({ commandment_path: `${E}/COMMANDMENT.md`, correctness_cmd: 'c', benchmark_cmd: 'b',
    baseline_per_case: [{ name: 'c1', baseline_ms: 1 }, { name: 'c2', baseline_ms: 2 }],
    baseline_geomean_ms: 1.4142, num_test_cases: 2, reliable: true })],
  [/^(profile_engineer:baseline|reprofile r\d+|warm_start:reprofile)$/, () => profile],
  [/^clock /, () => ({ epoch: (epoch += 60) })],
  [/^tech_lead:(re)?plan r(\d+)/, (p, m) => (scenario === 'stop' || Number(m[2]) > 2
    ? { stop: true, reasoning: 'done', directions: [] }
    : { stop: false, reasoning: 'two levers', directions: Number(m[2]) === 1
      ? [{ id: 'd1', title: 'coalesce', specialty: 'memory', prompt: 'p1' },
        { id: 'd2', title: 'unroll', specialty: 'compute', prompt: 'p2' }]
      : [{ id: 'd3', title: 'split', specialty: 'algorithm', prompt: 'p3' }] })],
  [/^(eng|deep) (d\d):/, (p, m) => ({ engineer_id: m[2], status: 'ok', speedup_geomean: SPEED[m[2]],
    speedup_arithmetic: SPEED[m[2]], per_case: perCase(SPEED[m[2]]), patch_file: `${E}/patches/${m[2]}.diff`,
    measurement_valid: true })],
  [/^verify (d\d)/, (p, m) => ({ status: 'pass', correctness: 'pass', verified_geomean: SPEED[m[1]],
    verified_arithmetic: SPEED[m[1]], per_case: perCase(SPEED[m[1]]), graph_safe: 'yes' })],
  [/^integrate r\d+$/, () => ({ attempted: false, conclusion: 'one winner', combos_tried: [] })],
  [/^commit r\d+$/, () => ({ committed: true, current_best_diff: `${E}/best.diff` })],
  [/^tech_lead:memory r\d+$/, () => ({ insights: ['coalescing pays'], ledger: [], bottleneck_now: 'compute' })],
  [/^storage:reclaim/, () => ({ ok: true })],
  [/^tech_lead:report$/, () => ({ final_speedup_geomean: 1.3, final_speedup_arithmetic: 1.3, rounds: 2,
    budget_used: 3, report_path: `${E}/report.md`, final_patch: `${E}/final_patch.diff`, per_case: perCase(1.3) })],
  [/^director:validate$/, () => ({ kernel_name: 'k', director_verified_speedup_geomean: 1.29,
    director_verified_speedup_arithmetic: 1.29, validation_status: 'accepted', correctness: 'pass',
    timing_basis: 'clean', per_case: perCase(1.29), final_patch: `${E}/final_patch.diff` })],
  [/^update_experience$/, () => ({ action: 'skipped', card_path: '' })],
  [/^warm_start:resolve$/, () => ({ read_reason: 'empty', candidates: [] })],
  [/^kb:/, () => ({ written: false, reason: 'off' })],
];

async function agent(prompt, opts = {}) {
  calls.push({ label: opts.label, phase: opts.phase, prompt });
  for (const [re, f] of answers) {
    const m = String(opts.label || '').match(re);
    if (m) return f(prompt, m);
  }
  throw new Error(`unscripted agent ${opts.label}`);
}
async function workflow(ref, a) { workflows.push({ scriptPath: ref && ref.scriptPath, args: a }); return null; }
const parallel = (ts) => Promise.all(ts.map((t) => Promise.resolve().then(t).catch(() => null)));
const pipeline = async (items, ...stages) => Promise.all((items || []).map(async (it, i) => {
  let prev = it;
  for (const st of stages) prev = await st(prev, it, i);
  return prev;
}));
const phase = () => {};
const log = () => {};

const args = { kernel_path: '/k/task', workflow_dir: '/wf', exp_root: '/x/exp', budget: 3, gpu_ids: '0,1',
  agent_timeout_ms: 0, update_experience: 'off', use_learned_kb: 'false', warm_start: 'off',
  ...(scenario === 'author' ? { mode: 'author', target_language: 'triton' } : { mode: 'optimize' }),
  ...(scenario === 'multigpu' ? { gpu_ids: '0,1,2,3' } : {}),
  ...(scenario === 'deadline' ? { deadline_epoch: 1000000 + 7200 } : {}) };

const src = (await readFile(script, 'utf8')).replace(/^[ \t]*export\s+(?=(const|let|var|function|class|async|default)\b)/gm, '');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const fn = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'workflow', 'args', 'budget', src);
const result = await fn(agent, parallel, pipeline, phase, log, workflow, args, {});
process.stdout.write(JSON.stringify({ result, calls, workflows }) + '\n');
