#!/usr/bin/env node
// The external-optimizer path of kernel_lane.js (args.optimizer != 'greedy'), end to end on the CPU.
//
// Agents are stubbed as in test_greedy_parity.js; runProcess is the runtime's real one, so the
// optimizer is a real process (a small shell script) working in a real git repository. Checked:
//   - a clean winner inside the editable set is fast-forwarded into the lane workspace, also when the
//     lane workspace is a subdirectory of its repository;
//   - a winner outside the editable set (also by moving another file into an editable directory), a
//     best.speedup that is not the lane's objective over its per-case pairs (geomean, or time-weighted
//     on a workload-aligned run), a best that does not beat the incumbent, a lane workspace that moved during
//     the run, a non-zero exit (its result's notes are only logged), a process left behind and a deadline timeout all leave the workspace
//     at the baseline commit;
//   - nothing the optimizer started is alive afterwards;
//   - without runProcess (the Workflow tool) the lane keeps the baseline;
//   - the real GEAK-EVO dispatch adapter (geak_evo/dispatch.py, over the CPU owner fixture of
//     tests/geak_evo/test_command.py) as the optimizer: its result is accepted and fast-forwarded.
//
// Run: node kernel_workflow/scripts/test_optimizer_dispatch.js
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..', '..');
const LANE = path.join(ROOT, 'kernel_workflow', 'kernel_lane.js');

// The optimizer stub. MODE selects the behaviour; it writes a pid file for anything it leaves behind.
const OPTIMIZER = `#!/bin/sh
set -e
task="$2"; out="$4"
ws=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workspace"])' "$task")
cd "$ws"
commit() {
  git add -A; git -c user.name=t -c user.email=t@t commit -qm win
  printf '{"schema_version":1,"status":"improved","stop_reason":"budget","rounds":2,"best":{"speedup":%s,"commit":"%s","per_case":{"c1":{"baseline_ms":2.2,"optimized_ms":1.1},"c2":{"baseline_ms":3.8,"optimized_ms":1.9}}}}' "\${1:-2.0}" "$(git rev-parse HEAD)" > "$out"
}
win() { echo "$1" >> "$2"; commit "$3"; }
case "$MODE" in
  win) win x k.py ;;
  outside) win x other.py ;;
  rename) mkdir -p kdir; git mv other.py kdir/moved.py; commit ;;
  mismatch) win x k.py 1.5 ;;
  below) echo x >> k.py; git add -A; git -c user.name=t -c user.email=t@t commit -qm same
    printf '{"schema_version":1,"status":"improved","stop_reason":"budget","rounds":1,"best":{"speedup":1.0,"commit":"%s","per_case":{"c1":{"baseline_ms":2,"optimized_ms":2},"c2":{"baseline_ms":4,"optimized_ms":4}}}}' "$(git rev-parse HEAD)" > "$out" ;;
  weighted) echo x >> k.py; git add -A; git -c user.name=t -c user.email=t@t commit -qm w
    printf '{"schema_version":1,"status":"improved","stop_reason":"budget","rounds":1,"best":{"speedup":%s,"commit":"%s","per_case":{"c1":{"baseline_ms":2,"optimized_ms":1},"c2":{"baseline_ms":4,"optimized_ms":1}}}}' "$SPEEDUP" "$(git rev-parse HEAD)" > "$out" ;;
  moved) win x k.py; git -C "$LANE_WS" -c user.name=t -c user.email=t@t commit -q --allow-empty -m other ;;
  fail) exit 1 ;;
  failnote) win x k.py; python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["notes"]="adapter says no"; json.dump(d,open(sys.argv[1],"w"))' "$out"; exit 1 ;;
  stray) win x k.py; sleep 30 >/dev/null 2>&1 & echo $! > "$PIDS" ;;
  slow) python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["budget"]["deadline_epoch"])' "$task" > "$PIDS.deadline"
    trap 'date +%s.%N > "$PIDS.term"; exit 143' TERM
    sleep 30 & echo $! > "$PIDS"; wait ;;
esac
`;

function answer(evalDir, ws, weights, names = ['c1', 'c2'], editable = ['k.py', 'kdir']) {
  const perCase = [{ name: names[0], baseline_ms: 2, optimized_ms: 2, speedup: 1 },
                   { name: names[1], baseline_ms: 4, optimized_ms: 4, speedup: 1 }]
    .map((c, i) => (weights ? { ...c, weight: weights[i] } : c));
  const field = (prompt, key) => {
    const line = prompt.split('\n').find((l) => l.startsWith(`- ${key}: `));
    return line ? JSON.parse(line.slice(key.length + 4)) : undefined;
  };
  return (label, prompt) => {
    if (label === 'director:setup') return { eval_dir: evalDir, workspace: ws, kernel_name: 'fix_kernel',
      device_gfx: 'gfx950', device_target: 'unknown', physical_cu_count: 256, baseline_frozen: true, source_files: ['k.py'] };
    if (label === 'tech_lead:analyze') return { kernel_type: 'gemm', roadmap_summary: 'r', modifiable_files: editable.map((p) => `${ws}/${p}`) };
    if (label === 'benchmark_engineer') return { commandment_path: `${evalDir}/COMMANDMENT.md`, baseline_per_case: perCase,
      baseline_geomean_ms: 2.83, correctness_cmd: 'true', benchmark_cmd: 'true' };
    if (label === 'profile_engineer:baseline') return { bottleneck: 'memory', top_opportunities: [], summary_path: `${evalDir}/p0.md` };
    if (label === 'warm_start:resolve') return { read_reason: 'empty', candidates: [] };
    if (label === 'kb:attest') return { ran: 1 };
    if (label === 'tech_lead:report') return { final_speedup_geomean: field(prompt, 'CUMULATIVE_SPEEDUP'),
      report_path: `${evalDir}/report.md`, final_patch: `${evalDir}/final_patch.diff`, rounds: 0 };
    if (label === 'director:validate') return { director_verified_speedup_geomean: field(prompt, 'TECH_LEAD_REPORTED_GEOMEAN'),
      validation_status: 'accepted', timing_basis: 'clean' };
    if (label === 'update_experience') return { action: 'created', card_path: '/learned/c.md' };
    if (label === 'kb:write') return { written: true, slug: 's', dir: '/d' };
    throw new Error(`unanswered agent label: ${label}`);
  };
}

// sub: the lane workspace is this subdirectory of the repository. o.weights: per-case workload
// weights the benchmark stage reports; o.env: extra environment for the optimizer stub; o.files
// ({name: Buffer}), o.cases and o.editable replace the default workspace files, case names and
// editable files.
async function runLane(tmp, name, mode, extraArgs, withRunProcess = true, sub = '', o = {}) {
  const { runProcess } = await import(path.join(ROOT, 'interface/runtime/engine/backends/base.mjs'));
  const dir = path.join(tmp, name);
  const repo = path.join(dir, 'repo');
  const ws = sub ? path.join(repo, sub) : repo;
  const evalDir = path.join(dir, 'eval');
  fs.mkdirSync(ws, { recursive: true });
  fs.mkdirSync(evalDir);
  const g = (...a) => execFileSync('git', ['-C', ws, '-c', 'user.name=t', '-c', 'user.email=t@t', ...a], { encoding: 'utf8' }).trim();
  execFileSync('git', ['-C', repo, 'init', '-q']);
  if (sub) fs.writeFileSync(path.join(repo, 'top.txt'), 't\n');
  for (const [file, bytes] of Object.entries(o.files || { 'k.py': 'k\n', 'other.py': 'o\n' })) {
    fs.writeFileSync(path.join(ws, file), bytes);
  }
  g('add', '-A');
  g('commit', '-qm', 'baseline');
  const base = g('rev-parse', 'HEAD');
  const pids = path.join(dir, 'pids');
  const body = fs.readFileSync(LANE, 'utf8').replace(/^export const meta/m, 'const meta');
  const logs = [];
  const globals = {
    args: { kernel_path: '/cpu-fixture/task', workflow_dir: '/cpu-fixture/kernel_workflow', agent_timeout_ms: 0,
      agent_retries: 1, budget: 3, optimizer: 'evo', optimizer_command: [path.join(tmp, 'optimizer.sh')],
      optimizer_reserve_s: 0, ...extraArgs },
    phase: () => {}, log: (m) => logs.push(String(m)),
    workflow: async () => { throw new Error('no nesting'); },
    agent: async (p, opts) => answer(evalDir, ws, o.weights, o.cases, o.editable)((opts && opts.label) || '', String(p)),
    parallel: async (ts) => Promise.all(ts.map((t) => t().catch(() => null))),
    pipeline: async () => { throw new Error('no pipeline'); },
    budget: { total: null, spent: () => 0, remaining: () => Infinity },
    runProcess: withRunProcess ? runProcess : undefined,
  };
  const env = { MODE: mode, PIDS: pids, LANE_WS: ws, ...(o.env || {}) };
  const saved = { ...process.env };
  Object.assign(process.env, env);
  try {
    const fn = new Function(...Object.keys(globals), `return (async () => { ${body} })();`);
    const result = await fn(...Object.values(globals));
    return { result, logs, base, head: g('rev-parse', 'HEAD'), pids: fs.existsSync(pids) ? fs.readFileSync(pids, 'utf8').trim().split('\n') : [] };
  } finally {
    for (const k of Object.keys(env)) { if (k in saved) process.env[k] = saved[k]; else delete process.env[k]; }
  }
}

const alive = (pid) => { try { process.kill(+pid, 0); return true; } catch { return false; } };
let failed = 0;
function check(name, cond, detail) {
  console.log(`  ${cond ? 'ok' : 'FAIL'}: ${name}${cond ? '' : ` -- ${detail}`}`);
  if (!cond) failed++;
}

(async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'geak-optimizer-dispatch-'));
  try {
    fs.writeFileSync(path.join(tmp, 'optimizer.sh'), OPTIMIZER, { mode: 0o755 });
    const winLog = (r) => r.logs.find((l) => l.includes('no winner')) || '';

    let r;
    const taskRef = { path: '/admitted/task.json', sha256: 'a'.repeat(64) };
    for (const [name, sub] of [['win', ''], ['win-subdir', 'pkg/op']]) {
      r = await runLane(tmp, name, 'win', { optimizer_task_ref: taskRef }, true, sub);
      const runDir = path.join(tmp, name, 'eval', 'optimizer', fs.readdirSync(path.join(tmp, name, 'eval', 'optimizer'))[0]);
      const winner = JSON.parse(fs.readFileSync(path.join(runDir, 'result.json'), 'utf8')).best.commit;
      const task = JSON.parse(fs.readFileSync(path.join(runDir, 'task.json'), 'utf8'));
      check(`${name}: the task names the same subdirectory of the optimizer's clone and carries task_ref`,
        task.workspace === path.join(runDir, 'repo', sub) && (!sub || fs.existsSync(path.join(runDir, 'repo', 'top.txt')))
        && JSON.stringify(task.task_ref) === JSON.stringify(taskRef), `workspace=${task.workspace} task_ref=${JSON.stringify(task.task_ref)}`);
      const edited = fs.readFileSync(path.join(tmp, name, 'repo', sub, 'k.py'), 'utf8') === 'k\nx\n';
      check(`${name}: a clean winner in the editable set is fast-forwarded`, r.head === winner && edited && !winLog(r)
        && Math.abs(r.result.final_speedup - 2) < 1e-9 && r.result.stopped_by === 'budget',
        `head=${r.head} winner=${winner} edited=${edited} speedup=${r.result.final_speedup} ${winLog(r)}`);
    }

    const kept = [
      ['outside', {}, 'outside the editable set'],
      ['rename', {}, 'outside the editable set'],
      ['mismatch', {}, 'not the geomean'],
      ['below', {}, 'does not beat the incumbent'],
      ['moved', {}, 'HEAD moved'],
      ['fail', {}, 'did not exit 0'],
      ['failnote', {}, 'did not exit 0 (1): adapter says no'],   // its committed winner is not adopted
      ['stray', {}, 'left processes running'],
      // 30 s of SIGTERM grace and 5 s of settling come off the time left, so ~5 s to run.
      ['slow', { deadline_epoch: 'now+40' }, 'past the lane deadline'],
    ];
    for (const [mode, extra, why] of kept) {
      if (extra.deadline_epoch === 'now+40') extra.deadline_epoch = Math.floor(Date.now() / 1000) + 40;
      r = await runLane(tmp, mode, mode, extra);
      const head = mode === 'moved' ? execFileSync('git', ['-C', path.join(tmp, mode, 'repo'), 'rev-parse', 'HEAD~1'], { encoding: 'utf8' }).trim() : r.head;
      check(`${mode}: the workspace keeps the baseline (${why})`, head === r.base && winLog(r).includes(why), winLog(r) || r.logs.slice(-3).join(' | '));
      if (r.pids.length) check(`${mode}: nothing the optimizer started is alive`, !r.pids.some(alive), r.pids.join(','));
      if (mode === 'slow') {
        const read = (ext) => parseFloat(fs.readFileSync(path.join(tmp, mode, `pids.${ext}`), 'utf8'));
        const [deadline, term] = [read('deadline'), read('term')];
        check('slow: SIGTERM arrives at the task\'s budget.deadline_epoch',
          Math.abs(term - deadline) < 1.5 && Math.abs(deadline - (extra.deadline_epoch - 35)) < 1e-6, `${term} vs ${deadline}`);
      }
    }

    // Workload-aligned: weights 3:1 and pair speedups 2x/4x give time-weighted 4/(3/2+1/4) = 16/7,
    // not the geomean sqrt(8). The geomean is refused; so is a workload run without weights.
    const workload = { workload_spec_path: '/cpu-fixture/workload.json' };
    r = await runLane(tmp, 'weighted', 'weighted', workload, true, '', { weights: [3, 1], env: { SPEEDUP: String(16 / 7) } });
    check('weighted: the time-weighted winner is applied and reported as primary',
      r.head !== r.base && Math.abs(r.result.final_speedup - 16 / 7) < 1e-9, `${r.result.final_speedup} ${winLog(r)}`);
    r = await runLane(tmp, 'weighted-geomean', 'weighted', workload, true, '', { weights: [3, 1], env: { SPEEDUP: String(Math.sqrt(8)) } });
    check('weighted-geomean: a geomean reported for a weighted run is refused',
      r.head === r.base && winLog(r).includes('not the time-weighted'), winLog(r));
    r = await runLane(tmp, 'weighted-none', 'weighted', workload, true, '', { env: { SPEEDUP: String(16 / 7) } });
    check('weighted-none: a workload run whose cases carry no weights is refused before running',
      r.head === r.base && winLog(r).includes('no workload weights') && !fs.existsSync(path.join(tmp, 'weighted-none', 'eval', 'optimizer')), winLog(r));

    // The real adapter. The probe builds the owner fixture once to learn its admitted task; the run
    // rebuilds it at the same path, so the task ref the lane copies into task.json matches.
    const entry = path.join(ROOT, 'tests', 'geak_evo', 'dispatch_lane_entry.py');
    const fixtureRoot = path.join(tmp, 'evo-fixture');
    const pyEnv = { ...process.env, EVO_LANE_FIXTURE_ROOT: fixtureRoot, PYTHONDONTWRITEBYTECODE: '1' };
    const probe = JSON.parse(execFileSync('python3', ['-B', entry, '--probe'], { env: pyEnv, encoding: 'utf8' }));
    const files = Object.fromEntries(Object.entries(probe.files).map(([f, b]) => [f, Buffer.from(b, 'base64')]));
    const deadline = Math.floor(Date.now() / 1000) + 4000;   // the owner's 3600 s budget + 30 s export fit
    r = await runLane(tmp, 'evo', 'evo', { optimizer_command: ['python3', '-B', entry], optimizer_task_ref: probe.task_ref,
      deadline_epoch: deadline }, true, 'pkg/op', { files, cases: Object.keys(probe.cases), editable: probe.editable,
      env: { EVO_LANE_FIXTURE_ROOT: fixtureRoot, EVO_LANE_DISPATCH_ROOT: path.join(tmp, 'evo', 'eval', 'optimizer'),
        PYTHONDONTWRITEBYTECODE: '1' } });
    const evoDir = path.join(tmp, 'evo', 'eval', 'optimizer');
    const evoRun = fs.existsSync(evoDir) ? path.join(evoDir, fs.readdirSync(evoDir)[0]) : null;
    const evoResult = evoRun && fs.existsSync(path.join(evoRun, 'result.json'))
      ? JSON.parse(fs.readFileSync(path.join(evoRun, 'result.json'), 'utf8')) : {};
    const evoTask = evoRun ? JSON.parse(fs.readFileSync(path.join(evoRun, 'task.json'), 'utf8')) : {};
    const pair = { baseline_ms: 1, optimized_ms: 0.5 };
    check('evo: the adapter\'s winner is applied with the lane objective over its measured pairs',
      evoResult.status === 'improved' && r.head === (evoResult.best || {}).commit && r.head !== r.base && !winLog(r)
      && Math.abs(r.result.final_speedup - 2) < 1e-9
      && JSON.stringify(evoResult.best.per_case) === JSON.stringify({ 'lane-a': pair, 'lane-b': pair }),
      `${JSON.stringify(evoResult).slice(0, 400)} ${winLog(r)} ${r.logs.slice(-3).join(' | ')}`);
    check('evo: the adapter got the SIGTERM instant as its deadline and changed only the editable file',
      evoTask.budget && evoTask.budget.deadline_epoch === deadline - 35
      && execFileSync('git', ['-C', path.join(tmp, 'evo', 'repo'), 'diff', '--name-only', r.base, r.head], { encoding: 'utf8' }).trim()
        === probe.editable.map((p) => `pkg/op/${p}`).join('\n'),
      JSON.stringify(evoTask.budget));

    r = await runLane(tmp, 'native', 'win', {}, false);
    check('without runProcess the lane keeps the baseline', r.head === r.base && winLog(r).includes('no runProcess'), winLog(r));
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  console.log(failed ? `\nFAIL: ${failed} check(s)` : '\nPASS: external optimizer dispatch');
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
