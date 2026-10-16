// An external optimizer (args.optimizer other than 'greedy', e.g. GEAK-EVO) replaces the greedy
// loop above and nothing else: Setup, the cases, the commands, the baseline and Report/Validate stay
// the same. It is one supplied command, run by the runtime's runProcess() -- no agent in the path:
//
//   <args.optimizer_command...> --task <dir>/task.json --out <dir>/result.json
//
// It works in its own clone of the workspace (task.workspace, at base_commit, at the same path inside
// the repository) and leaves its winner committed there. The lane's workspace changes only after a
// clean finish, by a fast-forward to best.commit; on every other outcome it keeps the baseline.
//
// PROVISIONAL: task.json/result.json keep the v2 field shapes until the GEAK-EVO entry command and
// its result exporter are bound. The objective is the greedy loop's: the time-weighted ratio of sums
// Σw/Σ(w/speedup) over the cases' workload weights when a workload spec is active, else the geomean
// of the per-case speedups. task.objective states which (with the weights and the workload spec as
// given), best.speedup must be that number, and a winner must beat the incumbent by MIN_IMPROVE as
// in the greedy loop. A workload-aligned run whose cases carry no weights is refused. Each speedup comes
// from the matching pair the optimizer reports for that case ({baseline_ms, optimized_ms}), never
// from the lane's baseline divided by a time taken in another session. How the pair was timed
// (interleaved or not) and its provenance are the optimizer's to check; Validate re-measures.
//
// args.optimizer_task_ref ({path, sha256}, optional) names an already admitted task. The lane copies
// it into task.json as task_ref without reading it; the command checks it against what it runs.
const OPTIMIZER = String(A.optimizer || 'greedy');
// The optimizer must finish this long before the lane deadline, leaving time for Report/Validate.
const OPTIMIZER_RESERVE_S = Math.max(0, parseInt(A.optimizer_reserve_s != null ? A.optimizer_reserve_s : 600, 10));
// How long the optimizer's process group gets between SIGTERM and SIGKILL, and how long after that
// runProcess may still wait for pipes and group to clear.
const OPTIMIZER_GRACE_MS = 30000;
const OPTIMIZER_SETTLE_MS = 5000;
// When the optimizer gets SIGTERM (epoch seconds, null without a deadline): early enough that it is
// stopped and settled before the reserve begins. task.budget.deadline_epoch is this instant.
const OPTIMIZER_STOP_EPOCH = DEADLINE_EPOCH
  ? DEADLINE_EPOCH - OPTIMIZER_RESERVE_S - (OPTIMIZER_GRACE_MS + OPTIMIZER_SETTLE_MS) / 1000 : null;
// Milliseconds left until then (Infinity without a deadline). Read again right before the run.
const optimizerMsLeft = () => (OPTIMIZER_STOP_EPOCH ? OPTIMIZER_STOP_EPOCH * 1000 - Date.now() : Infinity);

// One host command; null when it did not run cleanly to exit 0.
async function hostRun(cmd, args, opts = {}) {
  const r = await runProcess({ cmd, args, timeoutMs: 300000, graceMs: 2000, ...opts });
  return processClean(r) && r.code === 0 ? r.stdout : null;
}
// Finished by itself, and nothing it started can still be running.
const processClean = (r) => r.status === 'completed' && !r.strays_killed && r.group_cleared && !r.stdio_held;
const git = (dir, ...args) => hostRun('git', ['-C', dir, ...args]);
const writeFile = (path, text) => hostRun('sh', ['-c', 'cat > "$1"', 'sh', path], { input: text });

async function externalOptimize(name, start) {
  if (skipLoop || !(BUDGET > 0)) return greedyOptimize(start);   // nothing to optimize: same no-op result
  const noWinner = (why, stopReason = 'optimizer_error') => {
    log(`[${name}] no winner, the workspace keeps the baseline: ${why}`);
    return {
      rounds: 0, dispatched: 0, cumulativeSpeedup: start.cumulative, bestPerCase: start.bestPerCase,
      finalWinner: start.finalWinner, roundsCommitted: 0, history: start.history, citations: [],
      profileSummary: start.profileSummary, stopReason, forcedReplans: 0, kbCapBound: 0,
    };
  };
  if (typeof runProcess !== 'function') return noWinner('this runtime provides no runProcess()');
  const weighted = HAS_WORKLOAD;
  if (weighted && !BASELINE_PER_CASE.every(c => Number.isFinite(c.weight) && c.weight > 0)) {
    return noWinner('the run is workload-aligned but its cases carry no workload weights');
  }
  const taskRef = A.optimizer_task_ref;
  if (taskRef != null && !(taskRef && typeof taskRef.path === 'string' && taskRef.path
      && /^[0-9a-f]{64}$/.test(String(taskRef.sha256)))) {
    return noWinner('args.optimizer_task_ref must be {path, sha256}');
  }
  let argv = A.optimizer_command;
  try { if (typeof argv === 'string') argv = JSON.parse(argv); } catch (_) { argv = null; }
  if (!Array.isArray(argv) || !argv.length || !argv.every(a => typeof a === 'string' && a)) {
    return noWinner('args.optimizer_command must be a non-empty argv array of strings');
  }
  const editable = (analysis && analysis.modifiable_files) || [];
  if (!editable.length || !bench.correctness_cmd || !bench.benchmark_cmd) {
    return noWinner('the analysis has no editable files or the benchmark has no correctness/benchmark command');
  }
  // A command naming the lane workspace would act on it, not on the optimizer's clone.
  const commands = [bench.correctness_cmd, bench.benchmark_cmd, bench.profile_cmd].filter(Boolean);
  if (commands.some(c => c.includes(CANONICAL))) {
    return noWinner(`a benchmark command names the lane workspace ${CANONICAL}; it would not run in the optimizer's copy`);
  }
  const top = ((await git(CANONICAL, 'rev-parse', '--show-toplevel')) || '').trim();
  const prefix = ((await git(CANONICAL, 'rev-parse', '--show-prefix')) || '').trim();   // '' or 'sub/dir/'
  const base = ((await git(CANONICAL, 'rev-parse', '--verify', 'HEAD^{commit}')) || '').trim();
  if (!top || !base) return noWinner('could not read the workspace HEAD');
  // Editable paths relative to the lane workspace (the task's view) and to the repository (git's).
  const editableRel = editable.map(p => wsRelPath(p));
  if (editableRel.some(p => p === null)) return noWinner('an editable path lies outside the lane workspace');
  if (optimizerMsLeft() <= 0) return noWinner('the lane deadline leaves no time to optimize', 'deadline');

  const dir = `${EVAL_DIR}/optimizer/${name}-${base.slice(0, 12)}-${Date.now()}`;
  const clone = `${dir}/repo`;
  const owned = prefix ? `${clone}/${prefix.replace(/\/+$/, '')}` : clone;
  if ((await hostRun('mkdir', ['-p', dir])) === null
      || (await hostRun('git', ['clone', '-q', '--no-checkout', top, clone])) === null
      || (await git(clone, 'checkout', '-q', '--detach', base)) === null) {
    return noWinner(`could not create the optimizer's workspace ${owned}`);
  }
  const taskPath = `${dir}/task.json`;
  const resultPath = `${dir}/result.json`;
  const task = {
    schema_version: 1, eval_dir: dir, workspace: owned, base_commit: base,
    ...(taskRef != null ? { task_ref: taskRef } : {}),
    editable_paths: editableRel,
    kernel_name: KERNEL_NAME, language: TARGET_LANGUAGE,
    commands: { correctness: bench.correctness_cmd, benchmark: bench.benchmark_cmd,
      ...(bench.profile_cmd ? { profile: bench.profile_cmd } : {}) },
    baseline: { per_case: Object.fromEntries(BASELINE_PER_CASE.map(c => [c.name, c.baseline_ms])),
      geomean_ms: BASELINE_GEOMEAN_MS },
    objective: weighted
      ? { kind: 'time_weighted', weights: Object.fromEntries(BASELINE_PER_CASE.map(c => [c.name, c.weight])),
          ...(WORKLOAD_SPEC_PATH ? { workload_spec_path: WORKLOAD_SPEC_PATH } : {}),
          ...(WORKLOAD_SPEC ? { workload: WORKLOAD_SPEC } : {}) }
      : { kind: 'geomean' },
    incumbent_speedup: start.cumulative, min_improvement: MIN_IMPROVE,
    ...(start.profileSummary ? { profile_summary: start.profileSummary } : {}),
    device: { arch: GFX, gpus: GPU_LIST },
    // null = no limit. The command gets SIGTERM at deadline_epoch and SIGKILL 30 s later.
    budget: { max_rounds: BUDGET, deadline_epoch: OPTIMIZER_STOP_EPOCH,
      agent_timeout_s: AGENT_TIMEOUT_MS > 0 ? AGENT_TIMEOUT_MS / 1000 : null, min_improvement: MIN_IMPROVE },
  };
  if ((await writeFile(taskPath, JSON.stringify(task, null, 2) + '\n')) === null) {
    return noWinner(`could not write ${taskPath}`);
  }

  // Never retried. The run is bounded by the deadline, and runProcess reports whether anything it
  // started could still be running when it returned.
  const runMs = optimizerMsLeft();
  if (runMs <= 0) return noWinner('setup used up the time left to optimize', 'deadline');
  const ran = await runProcess({
    cmd: argv[0], args: [...argv.slice(1), '--task', taskPath, '--out', resultPath], cwd: owned,
    timeoutMs: Number.isFinite(runMs) ? Math.round(runMs) : 0, graceMs: OPTIMIZER_GRACE_MS,
    settleMs: OPTIMIZER_SETTLE_MS,
  });
  await writeFile(`${dir}/stdout.txt`, ran.stdout);
  await writeFile(`${dir}/stderr.txt`, ran.stderr);
  log(`[${name}] ${ran.status} exit=${ran.code} signal=${ran.signal} after ${Math.round(ran.duration_ms / 1000)}s` +
    `${ran.strays_killed ? ', stray processes killed' : ''}${ran.group_cleared ? '' : ', PROCESS GROUP NOT CLEARED'}` +
    `${ran.stdio_held ? ', AN ESCAPED PROCESS HELD ITS OUTPUT' : ''}`);
  // A failed or timed-out run's own notes are logged as diagnostics; nothing else of its result is used.
  const notes = async () => {
    try {
      const n = (JSON.parse((await hostRun('cat', [resultPath])) || 'null') || {}).notes;
      return typeof n === 'string' && n ? `: ${n.slice(0, 500)}` : '';
    } catch (_) { return ''; }
  };
  if (ran.status === 'timeout') return noWinner(`the optimizer ran past the lane deadline${await notes()}`, 'deadline');
  if (ran.status !== 'completed' || ran.code !== 0) {
    return noWinner(`the optimizer did not exit 0 (${ran.error || ran.code})${await notes()}`);
  }
  // A process left behind may have been writing; its result is not trusted.
  if (!processClean(ran)) return noWinner('the optimizer left processes running');

  let r = null;
  try { r = JSON.parse((await hostRun('cat', [resultPath])) || 'null'); } catch (_) { r = null; }
  const STOP = { deadline: 'deadline', budget: 'budget', no_improvement: 'no_improve', planner_stop: 'tech_lead_stop' };
  if (!r || r.schema_version !== 1) return noWinner('missing result or unknown schema_version');
  if (r.status === 'error') return noWinner(`the optimizer reported an error: ${r.notes || ''}`);
  if (!['improved', 'no_improvement'].includes(r.status) || !STOP[r.stop_reason]
      || !(Number.isInteger(r.rounds) && r.rounds >= 0)) {
    return noWinner(`invalid result (status=${r.status}, stop_reason=${r.stop_reason})`);
  }
  const best = r.best;
  if ((r.status === 'improved') !== !!best) return noWinner('status and best disagree');
  const positive = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0;
  if (best && !(positive(best.speedup) && /^[0-9a-f]{40,64}$/.test(String(best.commit || ''))
      && best.per_case && typeof best.per_case === 'object')) {
    return noWinner('invalid best');
  }
  // best.per_case is {case: {baseline_ms, optimized_ms}}, one pair per case of the lane.
  const perCase = best && BASELINE_PER_CASE.map(c => {
    const m = best.per_case[c.name] || {};
    return { name: c.name, baseline_ms: m.baseline_ms, optimized_ms: m.optimized_ms,
      speedup: m.baseline_ms / m.optimized_ms, ...(weighted ? { weight: c.weight } : {}) };
  });
  if (perCase && !perCase.every(c => positive(c.baseline_ms) && positive(c.optimized_ms) && positive(c.speedup))) {
    return noWinner('best.per_case does not give a positive baseline/optimized pair for every case');
  }
  // best.speedup must be the lane's objective computed from those pairs, and beat the incumbent.
  const geomean = perCase && Math.exp(perCase.reduce((a, c) => a + Math.log(c.speedup), 0) / perCase.length);
  const arithmetic = perCase && perCase.reduce((a, c) => a + c.speedup, 0) / perCase.length;
  const timeWeighted = perCase && weighted
    ? perCase.reduce((a, c) => a + c.weight, 0) / perCase.reduce((a, c) => a + c.weight / c.speedup, 0) : null;
  const primary = weighted ? timeWeighted : geomean;
  if (perCase && Math.abs(primary / best.speedup - 1) > 1e-6) {
    return noWinner(`best.speedup ${best.speedup} is not the ${weighted ? 'time-weighted' : 'geomean'} ` +
      `speedup of its per-case pairs (${primary})`);
  }
  if (perCase && !(primary > start.cumulative * (1 + MIN_IMPROVE))) {
    return noWinner(`best ${primary.toFixed(3)}x does not beat the incumbent ${start.cumulative}x by ${MIN_IMPROVE}`,
      STOP[r.stop_reason]);
  }
  if (best) {
    const refused = await fastForward(clone, base, best.commit, editableRel.map(p => prefix + p));
    if (refused) return noWinner(`winner not applied: ${refused}`);
    const root = ((await git(CANONICAL, 'rev-list', '--max-parents=0', 'HEAD')) || '').trim().split('\n')[0];
    const diff = root && await git(CANONICAL, '--no-pager', 'diff', `${root}..HEAD`);
    if (diff === null || (await writeFile(`${EVAL_DIR}/current_best.diff`, diff)) === null) {
      log(`[${name}] WARNING: could not write ${EVAL_DIR}/current_best.diff`);
    }
  }
  return {
    rounds: r.rounds, dispatched: r.rounds,
    cumulativeSpeedup: best ? primary : start.cumulative,
    bestPerCase: perCase || start.bestPerCase,
    // geomean is the PRIMARY metric, as in the greedy loop's candidates; the raw one is kept apart.
    finalWinner: best ? { source: name, geomean: primary, geomean_unweighted: geomean,
      weighted: timeWeighted, arithmetic, per_case: perCase,
      patch: `${EVAL_DIR}/current_best.diff`, commit: best.commit } : start.finalWinner,
    roundsCommitted: best ? 1 : 0,
    history: { ...start.history, ...(r.history_path ? { history_path: r.history_path } : {}) },
    citations: [], profileSummary: start.profileSummary,
    stopReason: STOP[r.stop_reason], forcedReplans: 0, kbCapBound: 0,
  };
}

// An editable path from the analysis, relative to the lane workspace; null if it lies outside it.
function wsRelPath(p) {
  const s = String(p).replace(/\/+$/, '');
  const ws = CANONICAL.replace(/\/+$/, '') + '/';
  if (s.startsWith(ws)) return s.slice(ws.length) || null;
  if (!s || s.startsWith('/') || s.split('/').includes('..')) return null;
  return s.replace(/^(\.\/)+/, '');
}

// Move the lane workspace from `base` to the optimizer's `commit` by a fast-forward. Returns null on
// success, else why it refused; a refusal leaves the workspace as it was.
async function fastForward(owned, base, commit, editable) {
  const head = ((await git(CANONICAL, 'rev-parse', 'HEAD')) || '').trim();
  if (head !== base) return `the workspace HEAD moved during the run (${head || '?'} != ${base})`;
  if ((await git(CANONICAL, 'status', '--porcelain', '--untracked-files=no')) !== '') {
    return 'the workspace has uncommitted changes';
  }
  if ((await git(owned, 'update-ref', 'refs/geak/winner', commit)) === null) {
    return `${commit} is not a commit in the optimizer's workspace`;
  }
  if ((await git(CANONICAL, 'fetch', '-q', '--no-tags', owned, 'refs/geak/winner')) === null) {
    return 'could not fetch the winner';
  }
  const fetched = ((await git(CANONICAL, 'rev-parse', 'FETCH_HEAD')) || '').trim();
  if (fetched !== commit) return `fetched ${fetched || '?'} instead of ${commit}`;
  if ((await git(CANONICAL, 'merge-base', '--is-ancestor', base, commit)) === null) {
    return `${commit} does not descend from ${base}`;
  }
  // --no-renames: a rename lists both its source and its destination.
  const changed = await git(CANONICAL, 'diff', '--no-renames', '--name-only', '-z', base, commit);
  if (changed === null) return 'could not list the changed files';
  const outside = changed.split('\0').filter(Boolean)
    .filter(f => !editable.some(e => f === e || f.startsWith(e + '/')));
  if (outside.length) return `it changes files outside the editable set: ${outside.slice(0, 5).join(', ')}`;
  if ((await git(CANONICAL, 'merge', '-q', '--ff-only', commit)) === null) return 'the fast-forward failed';
  const now = ((await git(CANONICAL, 'rev-parse', 'HEAD')) || '').trim();
  return now === commit ? null : `HEAD is ${now || '?'} after the fast-forward, not ${commit}`;
}

const optimizeStart = { cumulative, bestSeen, bestPerCase, finalWinner, profileSummary, history };
applyOptimizeResult(OPTIMIZER === 'greedy'
  ? await greedyOptimize(optimizeStart)
  : await externalOptimize(OPTIMIZER, optimizeStart));
