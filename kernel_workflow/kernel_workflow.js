export const meta = {
  name: 'kernel-workflow',
  description: 'Single ENTRY POINT for kernel optimization on AMD GPUs: Instinct MI-series (CDNA gfx942/gfx950) and the validated RDNA4 product Radeon AI PRO R9700 (gfx1201). The worker validates on-box identity via structured rocminfo data; generic gfx1201 is not a product identity. Dispatches on args.mode: optimize/author -> delegate one unchanged single-language lane to the kernel_lane worker (backward compatible); bakeoff -> freeze the input kernel into ONE immutable oracle + frozen baseline, discover per-language existing impls + offline-tune env backends (aiter/CK), then run one worker lane per backend language (HIP/Triton/FlyDSL/CK/...) in parallel over the GPU pool and pick the fastest verified result across ALL candidates (author/optimize lanes AND the tuned env backend) — every one scored against the SAME frozen original baseline (anti-cheating). Wraps the unchanged kernel_lane worker (one workflow() nesting level; the dispatcher is the bake-off orchestrator).',
  whenToUse: 'Optimize a kernel. Three modes, all via args.mode (there is NO natural-language mode detection — the caller picks): mode=optimize (DEFAULT) speeds up an EXISTING kernel and behaves exactly like the old single-language workflow; mode=author writes a fresh implementation from scratch, then optimizes it — use it when there is no source to edit yet, or to port the op to another language (pass args.target_language); mode=bakeoff tries several backend languages in parallel and keeps the fastest (pass args.backends, or leave empty to auto-discover). Bakeoff Freeze establishes structured product/ISA identity before Discover; optional expected_gfx/expected_target pins are validated against it. Anything else throws. Pass args.kernel_path (required), args.workflow_dir (required), args.mode, args.target_language, args.backends, args.budget, args.gpu_ids, args.gpu_mode (pool|pin, default pool).',
  phases: [
    { title: 'Freeze',   detail: 'oracle_freezer: freeze the input kernel -> immutable oracle + baseline_src/ (the ONE denominator) [bakeoff only]' },
    { title: 'Discover', detail: 'op_benchmarker: per-language existing-impl probe + measure + OFFLINE env tune (aiter/CK, shapes from the frozen oracle) -> author_plan, best_known_ms [bakeoff only]' },
    { title: 'Bakeoff',  detail: 'one unchanged kernel_lane worker per language (optimize if an impl exists, else author), parallel over the GPU pool [bakeoff only]' },
    { title: 'Report',   detail: 'rank ALL candidates (lanes + tuned env backend) on the SAME frozen baseline; write the comparison table + winner (+ optional apply_to_original), then curate ONE distilled card into knowledge/learned/ on a measured win [bakeoff only]' },
  ],
};

// GENERATED from kernel_workflow/src/kernel_workflow/ by scripts/build_workflows.mjs. Edit the parts there, then rebuild.
// ---------------------------------------------------------------------------
// Args. This is a DISPATCHER: on mode=optimize/author it forwards straight to the
// single-language worker (kernel_lane.js) unchanged; on mode=bakeoff it acts
// as the multi-language bake-off orchestrator. The worker script is a sibling in
// this same dir, so it is called via workflow() at exactly ONE nesting level
// (dispatcher=level0 -> worker=level1). The dispatcher never nests a second level.
// ---------------------------------------------------------------------------
const A = args || {};
if (Object.keys(A).some((key) => key.startsWith('ablation_'))) {
  throw new Error('ablation_* study controls are retired; use the original revision to replay those studies.');
}
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/op dir)');
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js + kernel_lane.js + roles/ + knowledge/ (i.e. the dirname of this script).');
}
// The single-language worker lane (unchanged behavior). Same dir as this dispatcher.
const WORKER = String(A.kernel_lane_script || `${WORKFLOW_DIR}/kernel_lane.js`);
const MODE = String(A.mode != null ? A.mode : 'optimize').trim().toLowerCase() || 'optimize';

// ---- Live execution tracker (read-only observer; see interface/geak_trace_collector.py) ----
// Started HERE so a direct kernel run is tracked from its first agent, not only
// at the end. Both dispatcher lanes (pass-through and bakeoff) pass through this
// point, so one call covers them. The observer resolves its own workflow run from
// the runtime's workflow record (no eval-dir substring matching) — needed because
// neither EXP_ROOT nor eval_dir exists yet on the pass-through branch.
//
// Detached, read-only and strictly non-fatal: it never influences model choice,
// prompts, budgets or optimization decisions. Set GEAK_LIVE_TRACE=0 to disable.
(function startLiveTracker() {
  try {
    if (String(process.env.GEAK_LIVE_TRACE || '1') === '0') return;
    // Same default as kernel_lane.js computes for EXP_ROOT: args.exp_root is NOT
    // required, so bailing out when it is absent silently disabled tracking for
    // every ordinary invocation.
    const expRoot = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp'))
      .replace(/\/+$/, '');
    const { spawn } = require('child_process');
    const child = spawn('python3', [
      '-B', `${WORKFLOW_DIR}/../interface/geak_trace_collector.py`,
      '--exp-root', expRoot,
      // Runtime-supplied invocation identity: the wf_*.json record carries this
      // same args object, so matching it identifies THIS launch deterministically
      // instead of guessing from timing or from being the only run around.
      '--identity-args', JSON.stringify(A),
      '--script-dir', WORKFLOW_DIR,
      // Per-run filename: two runs under one exp_root must not overwrite each other.
      '--out-dir', expRoot,
      '--watch',
      '--interval', String(process.env.GEAK_LIVE_TRACE_INTERVAL_S || '30'),
      '--resolve-timeout', '600',
      '--max-seconds', String(process.env.GEAK_LIVE_TRACE_MAX_S || '172800'),
    ], { detached: true, stdio: 'ignore' });
    // spawn reports a missing executable ASYNCHRONOUSLY: try/catch cannot see it,
    // and without this listener the ENOENT is an unhandled error event.
    child.on('error', (err) => {
      try { log(`Live execution tracker failed to start (non-fatal): ${err && err.message}`); }
      catch (_) {}
    });
    child.unref();
    log(`Live execution tracker started -> ${expRoot}/geak_trace_<runId>.json`);
  } catch (e) {
    try { log(`Live execution tracker not started (non-fatal): ${e && e.message}`); } catch (_) {}
  }
})();

// Prompt for the file_writer agent that persists the agent timeline and renders the run report.
// A function DECLARATION so it is hoisted above every caller (kernel_workflow's pass-through branch
// runs before most definitions).
function persistLlmStatsPrompt(evalDir, tlJson) {
  const tlPath = `${evalDir}/reports/trace/agent_timeline.json`;
  return `You are the file_writer. PHASE=persist_llm_stats.\n` +
    `Do exactly two things, in order:\n` +
    `1. Use the Write tool to create "${tlPath}" with EXACTLY the JSON below, verbatim ` +
    `(create parent directories if needed; do NOT reformat, truncate or summarize it):\n\n` +
    '```json\n' + tlJson + '\n```\n\n' +
    `2. Then run this Bash command (best-effort; if it fails, carry on and report ok=false).\n` +
    `   It runs the token/time/cost ledger AND renders the role-execution-tree report (HTML + MD):\n` +
    `   python3 -B "${WORKFLOW_DIR}/../interface/geak_report.py" --eval-dir "${evalDir}"\n\n` +
    `Then return {"written": true, "path": "${tlPath}", "ok": <true if the command exited 0 else false>}.`;
}
// Optional population of ordinary kernel_lane workflows. Each member owns its
// development state; shared publication only makes deliberate peer imports and
// a read-only best-so-far query possible. There is no whole-workflow GPU lease.
if (Object.prototype.hasOwnProperty.call(A, 'lineages')) {
  const need = (condition, message) => { if (!condition) throw new Error(message); };
  const absolute = value => typeof value === 'string' && value.startsWith('/')
    && !/[\0\r\n]/.test(value) && value !== '/'
    && value.slice(1).split('/').every(part => part && part !== '.' && part !== '..');
  need(MODE === 'optimize' && A.optimizer === 'evo', 'lineages requires mode=optimize and optimizer=evo');
  need(String(A.apply_to_original || 'false') !== 'true',
    'population returns an isolated selected patch; apply_to_original=true is not supported');
  need(typeof runProcess === 'function', 'lineages requires the process-capable workflow runtime');
  need(absolute(A.eval_dir), 'lineages requires an explicit absolute eval_dir without dot components');
  need(absolute(A.kernel_path), 'lineages requires an absolute kernel_path');
  need(Array.isArray(A.lineages) && A.lineages.length > 0, 'lineages must be a nonempty array');
  const members = A.lineages.map(value => {
    need(value && typeof value === 'object' && !Array.isArray(value)
      && Object.keys(value).every(key => ['id', 'optimizer', 'imports', 'focus'].includes(key)),
    'each lineage supports only id, optimizer, imports and focus');
    need(typeof value.id === 'string' && /^[a-z][a-z0-9_-]{0,63}$/.test(value.id), 'lineage IDs must be safe lowercase names');
    need(['greedy', 'evo'].includes(value.optimizer), 'each lineage requires optimizer=greedy or evo');
    const hasFocus = Object.prototype.hasOwnProperty.call(value, 'focus');
    need(!hasFocus || (typeof value.focus === 'string' && value.focus.trim()),
      'lineage focus must be a nonempty string when provided');
    const imports = value.imports === undefined ? [] : value.imports;
    need(Array.isArray(imports) && imports.every(id => typeof id === 'string')
      && new Set(imports).size === imports.length, 'lineage imports must be distinct peer IDs');
    need(value.optimizer === 'evo' || imports.length === 0, 'greedy lineages require imports=[]');
    return { id: value.id, optimizer: value.optimizer, imports: [...imports],
      ...(hasFocus ? { focus: value.focus } : {}) };
  });
  const ids = new Set(members.map(member => member.id));
  need(ids.size === members.length, 'lineage IDs must be distinct');
  need(members.every(member => member.imports.every(id => id !== member.id && ids.has(id))),
    'lineage imports must name other configured members');
  const gpuIds = String(A.gpu_ids == null ? '0' : A.gpu_ids).split(',').map(id => id.trim());
  need(gpuIds.every(id => /^(0|[1-9][0-9]*)$/.test(id)) && new Set(gpuIds).size === gpuIds.length,
    'lineages requires a comma-separated pool of distinct numeric GPU IDs');
  const gpuPool = gpuIds.join(',');
  const root = `${A.eval_dir}/population`;
  const commandKeys = ['external_measurement_command', 'external_result_command',
    'external_final_measurement_command', 'external_final_result_command'];
  const stateRoots = [];
  function option(command, name) {
    need(!command.some(part => part.startsWith(`${name}=`)), `use the separate ${name} argument form`);
    const indices = command.flatMap((part, index) => part === name ? [index] : []);
    need(indices.length === 1 && indices[0] + 1 < command.length
      && !command[indices[0] + 1].startsWith('--'), `native command requires exactly one separate ${name} value`);
    return indices[0];
  }
  const commands = {};
  for (const key of commandKeys) if (A[key] != null) {
    const command = A[key];
    need(Array.isArray(command) && command.length > 0
      && command.every(part => typeof part === 'string' && part && !/[\0\r\n]/.test(part)),
    `${key} must be a nonempty native argv array`);
    const stateIndex = option(command, '--state-root');
    const originalRoot = command[stateIndex + 1];
    need(absolute(originalRoot), 'native --state-root must be absolute');
    if (!stateRoots.includes(originalRoot)) stateRoots.push(originalRoot);
    const hasLock = command.some(part => part === '--gpu-lock-file' || part.startsWith('--gpu-lock-file='));
    const hasPool = command.some(part => part === '--gpu-pool' || part.startsWith('--gpu-pool='));
    need(hasLock !== hasPool, 'native commands require exactly one GPU lock or pool option');
    const gpuIndex = option(command, hasPool ? '--gpu-pool' : '--gpu-lock-file');
    commands[key] = { command, stateIndex, stateOrdinal: stateRoots.indexOf(originalRoot), gpuIndex };
  }
  const external = A.objective && A.objective.kind === 'external_primary';
  if (external) {
    need(Object.keys(A.objective).sort().join(',') === 'kind,primary_regime'
      && typeof A.objective.primary_regime === 'string' && A.objective.primary_regime.trim(),
    'external_primary requires exactly kind and primary_regime');
    need(commands.external_measurement_command && commands.external_result_command,
      'external population requires measurement and result-reader argv');
    need(!!commands.external_final_measurement_command === !!commands.external_final_result_command,
      'final external measurements require both command prefixes');
  }
  // Keep the caller's original deadline for the public native handoff without
  // shortening any member's declared search window. Reserve only an explicit
  // part of the existing post-search allowance; reject infeasible allocations.
  const finalReserve = external && A.search_reserve_s != null
    ? Number(A.population_final_reserve_s) : 0;
  need(Number.isFinite(finalReserve) && finalReserve >= 0, 'invalid population final reserve');
  if (external && A.search_reserve_s != null) {
    need(finalReserve > 0 && Number(A.search_reserve_s) - finalReserve > 0
      && Number(A.search_reserve_s) - finalReserve >= Number(A.final_validation_reserve_s || 0),
    'bounded native population requires positive population_final_reserve_s within the existing post-search allowance');
  }
  function pooledCommand(entry, pool = gpuPool) {
    const command = [...entry.command];
    command[entry.gpuIndex] = '--gpu-pool';
    command[entry.gpuIndex + 1] = pool;
    return command;
  }
  const opSpec = A.op_spec || {};
  const workload = opSpec.workload || A.workload || null;
  const hasWorkload = !!(String(A.workload_spec_path || opSpec.workload_path || '').trim()
    || (Array.isArray(workload) && workload.length)
    || (workload && Array.isArray(workload.kernels) && workload.kernels.length));
  const objective = external ? A.objective : { kind: hasWorkload ? 'time_weighted' : 'geomean' };
  const sourceKnowledge = {
    learned_dir: String(A.learned_dir || `${WORKFLOW_DIR}/knowledge/learned`),
    kb_artifacts_dir: String(A.kb_artifacts_dir || `${WORKFLOW_DIR.replace(/\/[^/]*$/, '')}/kb_artifacts`),
  };
  sourceKnowledge.kb_store_dir = String(A.kb_store_dir
    || `${sourceKnowledge.kb_artifacts_dir.replace(/\/[^/]*$/, '')}/kb_store_local`);
  for (const key of ['state_dir', 'shared_kb', 'global_kb']) if (A[key]) sourceKnowledge[key] = String(A[key]);
  need(Object.values(sourceKnowledge).every(absolute), 'population knowledge and state seed paths must be absolute');
  const prepared = members.map(member => {
    const directory = `${A.eval_dir}/lineages/${member.id}`;
    const child = { ...A, workflow_dir: WORKFLOW_DIR, mode: 'optimize', optimizer: member.optimizer,
      eval_dir: `${directory}/eval`, exp_root: `${directory}/exp`, state_dir: `${directory}/state`,
      learned_dir: `${directory}/knowledge/learned`, kb_artifacts_dir: `${directory}/knowledge/artifacts`,
      kb_store_dir: `${directory}/knowledge/store`, kb_remote_write: 'off',
      gpu_ids: gpuPool, gpu_mode: 'pool', apply_to_original: 'false',
      population_lineage: { id: member.id, imports: member.imports, root },
    };
    delete child.lineages;
    delete child.continuing_lineage;
    // TASK already reaches initial analysis, round planning and both engineer
    // policies. Leave the inherited task untouched when no focus was supplied.
    if (member.focus !== undefined) child.task = `${A.task == null ? '' : A.task}\n\nInitial exploration focus for this lineage:\n${member.focus}\n\n`
      + 'Use this focus in the initial Analysis and subsequent round directions to develop a distinct computational organization beyond tile-size tuning. '
      + 'Treat it as a revisable hypothesis: change direction when analysis or measurements invalidate its premise. '
      + 'Preserve the full original task contract, allowed source boundaries, correctness requirements, evaluator, and primary objective across every required case and regime. '
      + "Scenario priorities guide exploration only; selection and export remain governed by the full task's unchanged objective.";
    if (finalReserve) {
      child.deadline_epoch = Number(A.deadline_epoch) - finalReserve;
      child.search_reserve_s = Number(A.search_reserve_s) - finalReserve;
    }
    if (member.optimizer === 'evo') child.continuing_lineage = true;
    for (const key of ['shared_kb', 'global_kb']) if (sourceKnowledge[key]) child[key] = `${directory}/knowledge/${key}.md`;
    for (const [key, entry] of Object.entries(commands)) {
      const command = pooledCommand(entry);
      command[entry.stateIndex + 1] = `${directory}/native_state/${entry.stateOrdinal}`;
      child[key] = command;
    }
    return { ...member, directory, args: child };
  });
  const manifest = { schema: 'geak.population/1', root, kernel_path: A.kernel_path, objective,
    gpu_pool: gpuPool, common_final_reserve_s: finalReserve,
    ...(A.deadline_epoch != null ? {deadline_epoch: Number(A.deadline_epoch)} : {}),
    ...(A.search_reserve_s != null ? {search_cutoff_epoch: Number(A.deadline_epoch) - Number(A.search_reserve_s)} : {}),
    lineages: prepared.map(member => ({ id: member.id, optimizer: member.optimizer,
      imports: member.imports, eval_dir: member.args.eval_dir,
      ...(member.focus !== undefined ? { focus: member.focus } : {}),
      ...Object.fromEntries(['deadline_epoch', 'search_reserve_s', 'final_validation_reserve_s',
        'final_cleanup_reserve_s', 'budget'].filter(key => member.args[key] != null).map(key => [key, member.args[key]])),
      ...Object.fromEntries(commandKeys.filter(key => member.args[key]).map(key => [key, member.args[key]])),
    })) };
  let finalProcessCosts = null;
  async function processOutput(command, processArgs, timeoutMs = 30000, operation = null, input = undefined) {
    let result;
    try {
      result = await runProcess({ cmd: command, args: processArgs, input, timeoutMs, graceMs: 2000, settleMs: 5000 });
    } catch (error) {
      if (finalProcessCosts) finalProcessCosts.push({operation, status: 'not_started', error: String(error)});
      throw error;
    }
    if (finalProcessCosts) finalProcessCosts.push({operation, status: result && result.status, code: result && result.code,
      duration_ms: result && result.duration_ms != null ? result.duration_ms : null});
    need(result && result.status === 'completed' && result.code === 0 && result.group_cleared
      && !result.strays_killed && !result.stdio_held,
    `population host command failed: ${String(result && (result.stderr || result.status) || 'no result')}`);
    return result.stdout;
  }
  async function processJson(command, processArgs, timeoutMs = 30000, operation = null, input = undefined) {
    return JSON.parse(await processOutput(command, processArgs, timeoutMs, operation, input));
  }
  // Seed each private write sink before launching any agents. The publication
  // and lineage roots must be fresh, so a reused output path cannot adopt stale
  // checkpoints. Dereference KB copies to avoid writing through seed symlinks.
  const initialize = `import hashlib,json,shutil,sys
from pathlib import Path
c=json.loads(sys.stdin.read()); m=c['manifest']; root=Path(m['root']); out=root.parent
task=Path(m['kernel_path']).resolve(); resolved=out.resolve()
if resolved == task or task in resolved.parents or resolved in task.parents:
 raise ValueError('population output and original task must not overlap')
for source in c['seeds'].values():
 p=Path(source).resolve()
 if p == resolved or p in resolved.parents or resolved in p.parents:
  raise ValueError('population output and seed paths must not overlap')
if root.exists() or (out/'lineages').exists():
 raise ValueError('population and lineage output roots must be fresh')
out.mkdir(parents=True,exist_ok=True); root.mkdir(); (out/'lineages').mkdir()
for member in c['members']:
 a=member['args']; Path(member['directory']).mkdir()
 for key in ('learned_dir','kb_artifacts_dir','kb_store_dir','state_dir','shared_kb','global_kb'):
  if key not in a: continue
  destination=Path(a[key]); source=Path(c['seeds'][key]) if key in c['seeds'] else None
  destination.parent.mkdir(parents=True,exist_ok=True)
  if source is not None and source.exists():
   if source.is_dir(): shutil.copytree(source,destination,symlinks=False)
   else: shutil.copy2(source,destination)
  elif key in ('shared_kb','global_kb'): destination.write_text('')
  else: destination.mkdir()
 task_json=task/'task.json'
 if task_json.is_file(): m['task_sha256']=hashlib.sha256(task_json.read_bytes()).hexdigest()
with (root/'manifest.json').open('x') as stream: json.dump(m,stream,indent=2)
print(json.dumps(m))`;
  await processJson('python3', ['-c', initialize], 300000, null,
    JSON.stringify({ manifest, seeds: sourceKnowledge, members: prepared }));
  phase('Bakeoff');
  log(`population: ${prepared.length} lineages share GPU pool ${gpuPool}; checkpoints -> ${root}`);
  const persist = `import json,sys
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as stream: stream.write(sys.stdin.read()+'\\n')
print(json.dumps({'written':str(p)}))`;
  const settled = await parallel(prepared.map(member => async () => {
    const record = { id: member.id, optimizer: member.optimizer, imports: member.imports,
      ...(member.focus !== undefined ? { focus: member.focus } : {}),
      eval_dir: member.args.eval_dir, status: 'failed', result: null };
    try {
      const result = await workflow({ scriptPath: WORKER }, member.args);
      need(result && result.eval_dir === member.args.eval_dir, 'lineage returned a different private eval_dir');
      record.result = result;
      record.status = 'completed';
      await processJson('python3', ['-c', persist, `${member.args.eval_dir}/population_final_result.json`], 30000, null, JSON.stringify(result));
    } catch (error) {
      record.status = 'failed';
      record.error = String(error && error.message || error);
      log(`population lineage ${member.id} failed: ${record.error}`);
    }
    return record;
  }));
  const lineages = prepared.map((member, index) => settled[index] || {
    id: member.id, optimizer: member.optimizer, imports: member.imports,
    ...(member.focus !== undefined ? { focus: member.focus } : {}),
    eval_dir: member.args.eval_dir, status: 'failed', result: null, error: 'workflow returned no disposition',
  });
  const query = await processJson('python3', [`${WORKFLOW_DIR}/scripts/query_population.py`, '--root', root], 300000);
  need(query.schema === 'geak.population.query/1', 'population query returned an invalid schema');
  let selected = query.selected;
  let result = selected && selected.result ? { ...selected.result } : {
    mode: 'optimize', eval_dir: A.eval_dir, final_speedup: null, final_patch: '',
    validation_status: 'invalid_measurement', outcome: 'comparison_unavailable',
  };
  result.eval_dir = A.eval_dir;
  let commonFinal = null;
  if (external) {
    need(Array.isArray(query.finalists), 'population query did not return per-lineage finalists');
    const finalists = new Map(query.finalists.map(candidate => [candidate.lineage_id, candidate]));
    need(finalists.size === query.finalists.length && [...finalists.keys()].every(id => ids.has(id)),
      'population query returned duplicate or unknown finalists');
    // Every retained member is confirmed serially on this same allocated GPU.
    // Search ranking never filters the common comparison or gives an optimizer priority.
    const finalGpu = gpuIds[0];
    const measure = pooledCommand(commands.external_final_measurement_command || commands.external_measurement_command, finalGpu);
    const reader = pooledCommand(commands.external_final_result_command || commands.external_result_command, finalGpu);
    const confirmations = [];
    const copy = `import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=sys.argv[3]: raise ValueError('selected delivery patch changed')
out=Path(sys.argv[2]); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('xb') as stream: stream.write(raw)
out.chmod(0o444)
print(json.dumps({'path':str(out),'sha256':sys.argv[3]}))`;
    phase('Report');
    for (const member of [...prepared].sort((a, b) => a.id.localeCompare(b.id))) {
      const candidate = finalists.get(member.id);
      const confirmation = {id: member.id, source: candidate || null, status: 'no_retained_candidate',
        selection: null, patch_ref: null, evidence: null, process_costs: []};
      confirmations.push(confirmation);
      if (!candidate) {
        try {
          await processJson('python3', ['-c', persist, `${root}/${member.id}/common_final_result.json`],
            30000, null, JSON.stringify(confirmation));
        } catch (error) { confirmation.persistence_error = String(error); }
        continue;
      }
      let captured = null;
      const patch = `${root}/${member.id}/common_final_patch.diff`;
      finalProcessCosts = confirmation.process_costs;
      try {
      const ref = candidate.delivery_patch_ref;
      need(ref && ref.base === 'ORIGINAL' && absolute(ref.path)
        && /^[a-f0-9]{64}$/.test(ref.sha256), 'selected native source has no ORIGINAL delivery patch');
      await processJson('python3', ['-c', copy, ref.path, patch, ref.sha256], 30000, 'retain_patch');
      confirmation.patch_ref = {path: patch, sha256: ref.sha256};
      const workspace = `${root}/${member.id}/common_final_workspace`;
      await processOutput('mkdir', [workspace], 30000, 'create_workspace');
      await processOutput('bash', [`${WORKFLOW_DIR}/scripts/materialize_workspace.sh`,
        '--src', A.kernel_path, '--dst', workspace, '--shared-root', `${A.eval_dir}/_shared`, '--link-aiter'], 300000, 'materialize');
      const apply = `import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from geak.native.source import _snapshot,_patched
base,patch,workspace,manifest=map(Path,sys.argv[2:])
raw=(base/'task.json').read_bytes(); expected=json.loads(manifest.read_text())['task_sha256']
if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('population task changed')
source=_patched(_snapshot(base,json.loads(raw)['source_paths']),patch.read_bytes())
for name,(contents,mode) in source.items():
 p=workspace/name
 if p.is_symlink() or not p.resolve().is_relative_to(workspace.resolve()): raise ValueError('source is not private')
 p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(contents); p.chmod(0o644|mode)
print(json.dumps({'workspace':str(workspace)}))`;
      await processJson('python3', ['-c', apply, WORKFLOW_DIR.replace(/\/[^/]*$/, ''),
        A.kernel_path, patch, workspace, `${root}/manifest.json`], 300000, 'apply_source');
      captured = await processJson(measure[0], [...measure.slice(1), '--workspace', workspace, '--capture-only'], 30000, 'capture');
      confirmation.selection = captured;
      const hex = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
      need(captured.schema === 'geak-native-measurement-identity-v1' && captured.capture_only === true
        && hex(captured.candidate_id) && hex(captured.source_id)
        && captured.source_id === candidate.result.source_id
        && typeof captured.evaluation_id === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$/.test(captured.evaluation_id),
      'common final capture differs from the selected source');
      await processOutput(measure[0], [...measure.slice(1), '--captured',
        '--candidate-id', captured.candidate_id, '--evaluation-id', captured.evaluation_id],
      Number(A.agent_timeout_ms) > 0 ? Number(A.agent_timeout_ms) : 3600000, 'measure');
      const observed = await processJson(reader[0], [...reader.slice(1), '--workspace', A.kernel_path,
        '--patch', patch, '--candidate-id', captured.candidate_id, '--evaluation-id', captured.evaluation_id], 30000, 'read');
      need(observed.external_measurement_valid === true && observed.measurement_valid === true
        && observed.correctness === 'pass' && observed.objective && observed.objective.kind === 'external_primary'
        && observed.objective.primary_regime === objective.primary_regime && observed.primary_regime === objective.primary_regime
        && Number.isFinite(observed.primary_speedup) && observed.primary_speedup > 0
        && Number.isFinite(observed.development_geomean_speedup) && observed.development_geomean_speedup > 0
        && observed.source_id === captured.source_id && observed.patch_sha256 === ref.sha256
        && observed.external_evaluation && observed.external_evaluation.candidate_id === captured.candidate_id
        && observed.external_evaluation.evaluation_id === captured.evaluation_id
        && typeof observed.native_reproducible_gain === 'boolean' && typeof observed.native_reproducible_regression === 'boolean',
      'common final evidence differs from captured source, patch or objective');
      const gain = observed.native_reproducible_gain && !observed.native_reproducible_regression
        && observed.primary_speedup > 1;
      confirmation.status = gain ? 'qualified' : 'no_improvement';
      confirmation.evidence = observed;
      } catch (error) {
        confirmation.status = error && error.code === 'GEAK_PROCESS_DEADLINE' ? 'deadline' : 'failed';
        confirmation.reason = String(error && error.message || error);
        log(`Population final ${member.id} ${confirmation.status}; retained source remains available: ${confirmation.reason}`);
      } finally {
        finalProcessCosts = null;
        confirmation.duration_ms = confirmation.process_costs.every(cost => Number.isFinite(cost.duration_ms))
          ? confirmation.process_costs.reduce((total, cost) => total + cost.duration_ms, 0) : null;
      }
      // Save each settled confirmation before another member can consume the
      // remaining deadline. A host cutoff still returns the accumulated JS result.
      try {
        await processJson('python3', ['-c', persist, `${root}/${member.id}/common_final_result.json`],
          30000, null, JSON.stringify(confirmation));
      } catch (error) { confirmation.persistence_error = String(error); }
    }
    const qualified = confirmations.filter(row => row.status === 'qualified');
    const available = qualified.length ? qualified : confirmations.filter(row => row.status === 'no_improvement');
    available.sort((a, b) => b.evidence.primary_speedup - a.evidence.primary_speedup || a.id.localeCompare(b.id));
    const winner = available[0] || null;
    selected = winner ? {...winner.source, speedup: winner.evidence.primary_speedup} : null;
    commonFinal = {status: winner ? 'completed' : 'failed', gpu_id: finalGpu,
      complete: confirmations.every(row => ['qualified', 'no_improvement'].includes(row.status)),
      selected_id: winner ? winner.id : null, measurement_command: measure, result_command: reader, confirmations};
    if (winner) {
      const gain = winner.status === 'qualified';
      const patch = `${A.eval_dir}/host_final_patch.diff`;
      let delivery = winner.patch_ref;
      try {
        await processJson('python3', ['-c', copy, winner.patch_ref.path, patch, winner.patch_ref.sha256]);
        delivery = {path: patch, sha256: winner.patch_ref.sha256};
      } catch (error) {
        commonFinal.delivery_alias_error = String(error);
        log(`Population winner remains available at its immutable member patch: ${String(error)}`);
      }
      result = {...winner.source.result, ...winner.evidence, eval_dir: A.eval_dir,
        final_speedup: winner.evidence.primary_speedup, final_selection: winner.selection,
        final_patch: gain ? delivery.path : '', final_patch_sha256: gain ? delivery.sha256 : null,
        validation_status: gain ? 'accepted' : 'no_improvement', outcome: gain ? 'improved' : 'no_improvement'};
      commonFinal.patch_ref = delivery;
    } else {
      result = {mode: 'optimize', eval_dir: A.eval_dir, objective, final_speedup: null, primary_speedup: null,
        development_geomean_speedup: null, external_measurement_valid: false, measurement_valid: false,
        native_reproducible_gain: false, native_reproducible_regression: false,
        correctness: 'not_checked', status: 'invalid_measurement', phase_speedups: null,
        receipt_ref: null, patch_sha256: null,
        final_patch: '', final_patch_sha256: null, final_selection: null, external_evaluation: null,
        validation_status: 'invalid_measurement', outcome: 'invalid_measurement',
        final_failure: {class: 'population_final_measurement_failed', reason: 'no member completed a valid common final'}};
    }
  }
  const llmStatsOn = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
  if (llmStatsOn) result.llm_timeline = { schema: 'geak.agent_timeline/1', workflow: 'kernel_workflow',
    instance: A.eval_dir, events: [], nested: lineages.map(member => member.result && member.result.llm_timeline).filter(Boolean) };
  result.population = { schema: 'geak.population.result/1', root,
    search_selected: query.selected,
    selected: selected ? { id: selected.lineage_id, kind: selected.kind, round: selected.round,
      qualification: selected.qualification, speedup: selected.speedup, patch_ref: selected.patch_ref,
      delivery_patch_ref: selected.delivery_patch_ref } : null,
    common_final_reserve_s: finalReserve, common_final: commonFinal, lineages, query };
  try {
    await processJson('python3', ['-c', persist, `${root}/result.json`], 30000, null, JSON.stringify(result));
  } catch (error) {
    result.population.persistence_error = String(error);
    log(`Population aggregate could not be persisted; returning retained result: ${String(error)}`);
  }
  if (llmStatsOn && result.llm_timeline) {
    try {
      await agent(persistLlmStatsPrompt(A.eval_dir, JSON.stringify(result.llm_timeline)), {
        phase: 'Bakeoff', label: 'file_writer:persist_llm_stats', schema: { type: 'object',
          properties: { written: { type: 'boolean' }, path: { type: 'string' }, ok: { type: 'boolean' } },
          required: [], additionalProperties: true },
      });
    } catch (error) { log(`Population LLM report emit failed (NON-FATAL): ${String(error)}`); }
  }
  return result;
}
// ===========================================================================
// SINGLE-LANGUAGE PASS-THROUGH (mode=optimize | author) — byte-compatible with
// the pre-dispatcher behavior. Forward EVERY arg to the worker unchanged (the
// worker ignores dispatcher-only args like `backends`). One nesting level.
// ===========================================================================
if (MODE === 'optimize' || MODE === 'author') {
  phase('Bakeoff');            // reuse a declared phase slot for the single passthrough lane
  log(`mode=${MODE}: single-language pass-through -> ${WORKER}`);
  const r = await workflow({ scriptPath: WORKER }, { ...A, workflow_dir: WORKFLOW_DIR });
  // Final step: render the run report (token/time/cost ledger -> clickable role-execution-tree
  // HTML + MD twin) from the lane's OWN timeline. In passthrough the dispatcher makes no agent calls
  // itself, so the lane result carries the timeline (`llm_timeline`) + its `eval_dir`. This is the
  // top-level entry for optimize/author runs, so it owns the report (E2E nests kernel_lane.js directly
  // and reports at its own top level, so there is no double-emit). NOTE: this branch runs BEFORE the
  // obj()/agentT()/LLM_STATS definitions below, so it is self-contained: raw agent() + inline schema +
  // an inline llm_stats check. Entirely best-effort — never fail a real result over accounting.
  const llmStatsOn = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
  if (llmStatsOn && r && r.eval_dir && r.llm_timeline) {
    try {
      const tlJson = JSON.stringify({ ...r.llm_timeline, instance: r.eval_dir });
      await agent(persistLlmStatsPrompt(r.eval_dir, tlJson),
        { phase: 'Bakeoff', label: 'file_writer:persist_llm_stats',
          schema: { type: 'object',
            properties: { written: { type: 'boolean' }, path: { type: 'string' }, ok: { type: 'boolean' } },
            required: [], additionalProperties: true } });
      log(`LLM token+time+cost ledger + run report (HTML+MD) -> ${r.eval_dir}/reports/trace/ and ${r.eval_dir}/report/.`);
    } catch (e) {
      log(`LLM stats/report emit failed (NON-FATAL — the run is unaffected): ${String(e)}`);
    }
  }
  return r;
}

// ===========================================================================
// From here on: mode=bakeoff — the multi-language bake-off orchestrator. Pass
// args.backends to pick the languages, or leave it empty to auto-discover.
// ===========================================================================
if (MODE !== 'bakeoff') {
  throw new Error(`unknown mode='${MODE}'. Use optimize | author | bakeoff.`);
}

// Sibling e2e_workflow dir — we REFERENCE its op_benchmarker role + bench scripts in place (no copy).
const E2E_WF_DIR = String(A.e2e_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/e2e_workflow')).replace(/\/+$/, '');
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');
const EXPECTED_GFX = String(A.expected_gfx || '').trim().toLowerCase();
const EXPECTED_TARGET = String(A.expected_target || '').trim().toLowerCase();
const EXPECTED_DEVICE_NAME = String(A.expected_device_name || '').trim();
const EXPECTED_PHYSICAL_CU_COUNT = Number(A.expected_physical_cu_count || 0);
if (!!EXPECTED_GFX !== !!EXPECTED_TARGET) {
  throw new Error('expected_gfx and expected_target must be supplied together');
}
if ((EXPECTED_GFX && !/^gfx[0-9a-f]+$/.test(EXPECTED_GFX)) ||
    (EXPECTED_TARGET && !['r9700', 'mi350x', 'mi355x', 'unknown'].includes(EXPECTED_TARGET))) {
  throw new Error('invalid bakeoff identity: expected_gfx must be a gfx token and expected_target r9700|mi350x|mi355x|unknown');
}
if (EXPECTED_TARGET === 'r9700' && EXPECTED_GFX !== 'gfx1201') {
  throw new Error('expected_target=r9700 requires expected_gfx=gfx1201');
}
if (['mi350x', 'mi355x'].includes(EXPECTED_TARGET) && EXPECTED_GFX !== 'gfx950') {
  throw new Error(`expected_target=${EXPECTED_TARGET} requires expected_gfx=gfx950`);
}
if (EXPECTED_GFX === 'gfx1200' || EXPECTED_TARGET === 'gfx1200') {
  throw new Error('gfx1200 is not supported: this workflow is hardware-validated only on R9700 / gfx1201');
}
let RDNA4_ISOLATE = EXPECTED_GFX === 'gfx1201';
let KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge')).replace(/\/+$/, '');
if (RDNA4_ISOLATE) KERNEL_KNOWLEDGE_DIR = '';
const KERNEL_PATH_ORIG = A.kernel_path;
const KERNEL_NAME_HINT = String(KERNEL_PATH_ORIG).replace(/\/+$/, '').split('/').pop();
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
const TASK = A.task || '';
const OP_SPEC = A.op_spec || {};
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const ENABLE_FP8 = String(A.enable_fp8 != null ? A.enable_fp8 : 'false');
const GPU_LIST = String(A.gpu_ids != null ? A.gpu_ids : '0').split(',').map(s => s.trim()).filter(Boolean);
// Forwarded verbatim to each lane worker; the dispatcher itself does not schedule. Without the
// passthrough gpu_mode would be settable only on a direct kernel_lane.js call, so the pinned "before"
// arm would be unreachable through the normal entry point and the A/B could not be run at all.
const GPU_MODE = String(A.gpu_mode || 'pool') === 'pin' ? 'pin' : 'pool';
// Explicit backend list (empty => auto-discover from the freeze + op_benchmarker probe).
const BACKENDS = (Array.isArray(A.backends) ? A.backends
  : (typeof A.backends === 'string' ? A.backends.split(',') : []))
  .map(s => String(s == null ? '' : s).trim().toLowerCase()).filter(Boolean);
// Expert-skills passthrough (advisory; OFF by default -> nothing injected).
let USE_EXPERT_SKILLS = !RDNA4_ISOLATE && String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
const EXPERT_SKILL_ROLES = new Set(['op_benchmarker']);

// Warm-start experience KB. Passed to each lane explicitly (the bakeoff lane invocation spreads
// specific keys, not ...A) so every language lane reads/writes its own <kernel>__<lang>__<gfx> slug.
let WARM_START = RDNA4_ISOLATE ? 'off'
  : (String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on');
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts')).replace(/\/+$/, '');
// Plane selection, forwarded the same way and for the same reason: every bakeoff lane must read and
// write the plane the run was launched with, not each its own default.
const KB_PLANE_ARGS = {
  ...(A.kb_mode != null ? { kb_mode: String(A.kb_mode) } : {}),
  ...(A.kb_store_dir != null ? { kb_store_dir: String(A.kb_store_dir) } : {}),
  ...(A.kb_framework_version != null ? { kb_framework_version: String(A.kb_framework_version) } : {}),
};

// ---------------------------------------------------------------------------
// Small helpers every workflow uses: the "## Inputs" list in role prompts and
// JSON-schema builders for structured agent output.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrStr = { type: 'array', items: { type: 'string' } };
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
// ---------------------------------------------------------------------------
// Schemas (obj/arrStr/arrObj: src/shared/helpers.js).
// ---------------------------------------------------------------------------
// oracle_freezer output — the standalone counterpart of e2e's kernel_extractor.extract_op, built from a
// kernel dir instead of a live server. Same op-task-dir contract; also creates the run dir (EVAL_DIR).
const FREEZE_SCHEMA = obj({
  eval_dir: { type: 'string' },            // the isolated run dir this freeze created under EXP_ROOT
  op_kind: { type: 'string' },             // gemm|attn|elementwise|moe|other
  task_dir: { type: 'string' },            // the immutable op task dir (unittest.py + meta.json + baseline_src/)
  live_backend: { type: 'string' },        // the input kernel's language, e.g. "hip"
  candidate_backends: arrStr,
  baseline_frozen: { type: 'boolean' },
  baseline_callable: { type: 'string' },
  device_gfx: { type: 'string' },
  device_target: { type: 'string' },
  device_name: { type: 'string' },
  physical_cu_count: { type: 'number' },
  // Always "" from oracle_freezer — a freezer-built task dir records NO golden tensors (correctness is
  // live parity vs baseline_src/). Kept in the schema because an e2e kernel_extractor task dir, which
  // captures unsynthesizable real routing / paged-KV metadata, does ship a reference_io.pt and fills it.
  reference_io_sha256: { type: 'string' },
  op_spec: { type: 'object', additionalProperties: true },
  workload_path: { type: 'string' },
  smoke: { type: 'string' },               // "pass" required to proceed
  notes: { type: 'string' },
}, ['op_kind', 'task_dir', 'smoke']);

// op_benchmarker output — the role file is reused VERBATIM from e2e (referenced in place, never edited);
// the standalone-tune behavior is steered entirely by the discover prompt below. `baseline_ms` +
// `tuned_speedup` are extra fields this dispatcher asks the agent to fill (in the prompt) so a tuned env
// backend can be ranked against the SAME frozen input-kernel baseline as the author/optimize lanes.
const OPBENCH_SCHEMA = obj({
  short_name: { type: 'string' }, op_kind: { type: 'string' }, provenance_ok: { type: 'boolean' },
  winner_backend: { type: 'string' }, winner_kind: { type: 'string' },
  // null when nothing was timed -- a speedup is a measurement, and 0.0 read as
  // 'benched, not faster'. `measured` is the discriminator; gate on it, not on the number.
  isolated_speedup: { type: ['number', 'null'] }, measured: { type: 'boolean' },
  winner_editable: { type: 'boolean' },
  best_known_ms: { type: 'number' },
  baseline_ms: { type: 'number' },     // ms of the FROZEN input kernel (baseline_src/) on the same oracle
  tuned_speedup: { type: 'number' },   // best tuned env backend's speedup vs the FROZEN baseline (baseline_ms/tuned_ms)
  recommend_tier_c: { type: 'boolean' }, author_plan: arrObj, tuning_artifact: { type: 'string' },
  apply_env: { type: 'string' }, apply_flags: { type: 'string' }, code_patch: { type: 'string' },
  per_backend: arrObj, parity_note: { type: 'string' },
  gate: { type: 'string' }, harness_suspect: { type: 'boolean' }, reason: { type: 'string' },
}, ['gate', 'isolated_speedup']);

const REPORT_SCHEMA = obj({
  report_path: { type: 'string' }, applied_to_original: { type: 'string' }, note: { type: 'string' },
}, ['report_path']);

const UPDATE_EXPERIENCE_SCHEMA = obj({
  action: { type: 'string' },      // created | merged | skipped
  card_path: { type: 'string' },   // path under knowledge/learned/, or "" if nothing distilled
  key: { type: 'string' }, note: { type: 'string' },
}, []);

// ---------------------------------------------------------------------------
// Agent-call limits + timeline (cfg, tlAgent and roleAgent are in src/shared/).
// ---------------------------------------------------------------------------
// Hung-agent + API-fault guard (same contract as kernel_lane.js:agentT).
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
const LLM_TL = { schema: 'geak.agent_timeline/1', workflow: 'kernel_workflow', events: [], nested: [] };
// LLM token+time accounting (PURELY ADDITIVE; args.llm_stats="false" makes it a no-op).
// DELIBERATELY NO TIMESTAMPS: Date.now()/new Date() are unavailable in workflow scripts. Every
// duration in the report comes from the transcripts (scripts/llm_ledger.py); this records only the
// role/phase/attempt identity of each agent call so the ledger attributes tokens to the right agent.
// Each workflow declares its own LLM_TL (it names the workflow).
const LLM_STATS = String(A.llm_stats != null ? A.llm_stats : 'true').trim().toLowerCase() !== 'false';
const TL_ROLE_RE = /You are the ([A-Za-z0-9_.\-]+)\.\s*PHASE=([A-Za-z0-9_.\-]+)\./;
// Called AT DISPATCH (before the await), so LLM_TL.events is in dispatch order, not completion
// order. Two same-key agents that finish out of order no longer swap identities under the parser's
// positional join. Returns the event so the caller flips `ok` once the attempt resolves (null when
// off). Even an attempt that hangs or throws is recorded, because the record predates the await.
function tlAgent(prompt, o, attempt) {
  if (!LLM_STATS) return null;
  // Identity comes from the PROMPT, not opts.label: labels are free-form display strings. The
  // prompt's opening line always carries `You are the <role>. PHASE=<sub_phase>.`, the same line the
  // transcript records, so the ledger folds this call under its own agent instead of its predecessor.
  const m = TL_ROLE_RE.exec(String(prompt || ''));
  const e = {
    seq: LLM_TL.events.length,
    phase: (o && o.phase) || '',
    label: (o && o.label) || 'agent',
    role: m ? m[1] : '',
    sub_phase: m ? m[2] : '',
    attempt: attempt,
    ok: false,
  };
  LLM_TL.events.push(e);
  return e;
}

// Role prompt: the agent reads its role file and knowledge itself; the script passes only inputs.
// roleAgentFrom reads the role from any roles/ dir (kernel_workflow uses it for e2e's op_benchmarker,
// referenced in place). expertSkillsBlock is declared by each workflow (its text differs per workflow).
function roleAgentFrom(dir, role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${dir}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${dir}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg({ ...(typeof TASK === 'string' && TASK ? { TASK } : {}), ...inputs })}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role)
    + (typeof externalMetricInstructions === 'function'
      ? externalMetricInstructions(role === 'director' && phase === 'validate')
      : typeof EXTERNAL_METRIC_INSTRUCTIONS === 'string' ? EXTERNAL_METRIC_INSTRUCTIONS : '');
}
// This workflow's own roles live under WORKFLOW_DIR/roles/.
function roleAgent(role, phase, intro, inputs) {
  return roleAgentFrom(WORKFLOW_DIR, role, phase, intro, inputs);
}
// Hung-agent + API-fault guard around agent(): retries API errors, resolves null on timeout.
async function agentT(p, o) {
  // Parallel members share installed dependencies, not mutable compiler caches.
  if (A.population_lineage) o = { ...o, env: { ...(o && o.env), GEAK_ISOLATE_WORKSPACE_CACHES: '1' } };
  const label = (o && o.label) ? o.label : 'agent';
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    const ev = tlAgent(p, o, attempt);   // record AT DISPATCH; ok=false until it resolves
    try {
      // The standalone runtime's bounded mode terminates and settles the actual
      // subprocess. Never race it with a Promise that abandons a live role.
      if (agent.processDeadlineActive || typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) {
        const opts = agent.processDeadlineActive ? { ...o, processTimeoutMs: AGENT_TIMEOUT_MS } : o;
        const r0 = await agent(p, opts); if (ev && r0) ev.ok = true; return r0;
      }
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — resolving null so the round proceeds.`);
          resolve(null);
        }, AGENT_TIMEOUT_MS);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      const r = await Promise.race([
        agent(p, o).then((rr) => { clearTimeout(to); return rr; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
      if (ev && r) ev.ok = true;
      return r;
    } catch (e) {
      if (agent.processDeadlineActive && e
          && (e.code === 'GEAK_PROCESS_DEADLINE' || e.code === 'GEAK_AGENT_TIMEOUT')) {
        log(`  [deadline] ${label} stopped (${e.code}); no retry after its process cutoff.`);
        return null;
      }
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < AGENT_RETRIES) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${AGENT_RETRIES} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${AGENT_RETRIES} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also query ${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op, whose \`scope\` ` +
    `is \`kernel\` (\`tuning\` entries belong to the e2e tuning phase and match every operator), and whose ` +
    `validation_status is \`validated\`, and treat each as a HIGH-PRIOR candidate — advisory only, never ` +
    `overriding your isolated A/B vs the oracle, never reducing a result below the measured baseline.`;
}

// GPU semaphore: 1 GPU per lane, EXCLUSIVE. size==1 serializes lanes. (Lifted from e2e_workflow.js.)
function makeSem(ids) {
  const free = ids.slice(); const waiters = [];
  const pump = () => { while (waiters.length && waiters[0].n <= free.length) {
    const w = waiters.shift(); w.resolve(free.splice(0, w.n)); } };
  return {
    size: ids.length,
    acquire(n = 1) { if (n <= free.length) return Promise.resolve(free.splice(0, n));
      return new Promise((resolve) => { waiters.push({ n, resolve }); }); },
    release(got) { free.push(...got); pump(); },
    async with(n, fn) { const g = await this.acquire(n); try { return await fn(g); } finally { this.release(g); } },
  };
}

// PRIMARY metric of a worker (kernel_lane.js) result: its final_speedup is already the weighted number
// when the run was workload-aligned, else the geomean. Fall back defensively.
const primSpeedup = (r) => {
  if (!r) return 0;
  const g = r.final_speedup != null ? r.final_speedup
          : (r.final_weighted != null ? r.final_weighted
             : (r.final_geomean != null ? r.final_geomean : 0));
  return Number.isFinite(g) ? g : 0;
};

// Gate fields like `smoke` are free strings in the schemas, so an agent legitimately answers
// "PASS - 15/15 parity draws, geomean 0.994" and a `=== 'pass'` test throws the whole run away.
// Match the leading word instead: "PASS - ..." / "passed" gate open, "FAIL"/"did not pass" stay shut.
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);

// ===========================================================================
// PHASE: Freeze — establish the ONE immutable oracle + frozen baseline (denominator)
// ===========================================================================
phase('Freeze');
const oracle = await agentT(
  roleAgent('oracle_freezer', 'freeze',
    'Freeze the input kernel into an immutable op task dir (no server). Create the run dir too.', {
      KERNEL_PATH: KERNEL_PATH_ORIG, EXP_ROOT, KERNEL_NAME_HINT, GPU_ID: GPU_LIST[0],
      OP_SPEC, WORKLOAD_SPEC_PATH, SKILL_DIR: WORKFLOW_DIR,
      EXPECTED_GFX, EXPECTED_TARGET, EXPECTED_DEVICE_NAME, EXPECTED_PHYSICAL_CU_COUNT,
      // harness_lib.py (the shared timing/correctness lib) ships with e2e_workflow; gpu_lock.sh ships here.
      HARNESS_LIB: `${E2E_WF_DIR}/scripts/harness_lib.py`,
      GPU_LOCK: `${WORKFLOW_DIR}/scripts/gpu_lock.sh`,
    }),
  { phase: 'Freeze', label: 'oracle_freezer', schema: FREEZE_SCHEMA });
if (!oracle || !says(oracle.smoke, 'pass') || !oracle.task_dir || oracle.baseline_frozen === false) {
  log(`Freeze FAILED (${oracle ? oracle.notes || oracle.smoke : 'no result'}); aborting — no comparable baseline.`);
  return { mode: MODE, validation_status: 'freeze_failed', winner: null,
    reason: oracle ? oracle.notes || 'freeze smoke did not pass' : 'oracle_freezer returned nothing' };
}
const DETECTED_GFX = String(oracle.device_gfx || '').trim().toLowerCase();
const DETECTED_TARGET = String(oracle.device_target || '').trim().toLowerCase();
const DETECTED_DEVICE_NAME = String(oracle.device_name || '').trim();
const DETECTED_PHYSICAL_CU_COUNT = Number(oracle.physical_cu_count);
if (!DETECTED_GFX || !DETECTED_TARGET) {
  throw new Error('Freeze failed: oracle_freezer did not return structured device_gfx and device_target');
}
if (DETECTED_GFX === 'gfx1200') {
  throw new Error('Detected gfx1200, which is not validated by this workflow; refusing R9700 guidance');
}
if (EXPECTED_GFX && DETECTED_GFX !== EXPECTED_GFX) {
  throw new Error(`GPU architecture mismatch: expected ${EXPECTED_GFX}, detected ${DETECTED_GFX}`);
}
if (EXPECTED_TARGET && EXPECTED_TARGET !== 'unknown' && DETECTED_TARGET !== EXPECTED_TARGET) {
  throw new Error(`GPU product mismatch: expected ${EXPECTED_TARGET}, detected ${DETECTED_TARGET}`);
}
if (!Number.isFinite(DETECTED_PHYSICAL_CU_COUNT) || DETECTED_PHYSICAL_CU_COUNT <= 0) {
  throw new Error('Freeze failed: oracle_freezer did not return a positive physical_cu_count');
}
if (EXPECTED_DEVICE_NAME && DETECTED_DEVICE_NAME !== EXPECTED_DEVICE_NAME) {
  throw new Error(
    `GPU product name mismatch: expected ${EXPECTED_DEVICE_NAME}, detected ${DETECTED_DEVICE_NAME || 'unknown'}`);
}
if (EXPECTED_PHYSICAL_CU_COUNT > 0 &&
    DETECTED_PHYSICAL_CU_COUNT !== EXPECTED_PHYSICAL_CU_COUNT) {
  throw new Error(
    `GPU physical CU mismatch: expected ${EXPECTED_PHYSICAL_CU_COUNT}, ` +
    `detected ${DETECTED_PHYSICAL_CU_COUNT}`);
}
const LANE_GFX = EXPECTED_GFX || DETECTED_GFX;
const LANE_TARGET = EXPECTED_TARGET && EXPECTED_TARGET !== 'unknown' ? EXPECTED_TARGET : DETECTED_TARGET;
const LANE_DEVICE_NAME = EXPECTED_DEVICE_NAME || DETECTED_DEVICE_NAME;
const LANE_PHYSICAL_CU_COUNT =
  EXPECTED_PHYSICAL_CU_COUNT || DETECTED_PHYSICAL_CU_COUNT;
if (DETECTED_GFX === 'gfx1201') {
  RDNA4_ISOLATE = true;
  KERNEL_KNOWLEDGE_DIR = '';
  USE_EXPERT_SKILLS = false;
  WARM_START = 'off';
}
const EVAL_DIR = oracle.eval_dir || `${EXP_ROOT}/bakeoff_${KERNEL_NAME_HINT}`;
log(`Freeze done. op_kind=${oracle.op_kind}, task_dir=${oracle.task_dir}, live_backend=${oracle.live_backend || '?'}`);

// ===========================================================================
// PHASE: Discover — per-language existing-impl probe + measure + author_plan +
// OFFLINE per-backend tune. e2e's op_benchmarker role is reused IN PLACE and
// UNCHANGED (SKILL_DIR=E2E_WF_DIR so its knowledge/scripts resolve). Its Tier-B
// step is written for a live server; since there is none here, the prompt below
// REDIRECTS the tune to run OFFLINE on the frozen oracle's shapes (which carry the
// real M/N/K/bias/dtype) and to verify engagement in the ISOLATED unittest — so a
// standalone bake-off STILL gets the aiter/CK GEMM tune, without touching the role
// file. Only pure server-flag levers (--attention-backend swap, serving-only
// engagement probes) are skipped.
// ===========================================================================
phase('Discover');
const DISCOVER_INTRO = RDNA4_ISOLATE
  ? (
  'STANDALONE kernel bake-off — there is NO live server (do not try to launch or capture from one).\n' +
  'R9700 / gfx1201 ISOLATION: do NOT consult learned KB, expert skills, AITER CDNA win cards, or CK/AITER offline tune.\n' +
  'Do NOT inject MFMA / FNUZ / Instinct guidance. Discover existing impls by measuring the frozen oracle only.\n' +
  '(1) Tier-A DISCOVER: bench candidate backends on the immutable oracle in OP_TASK_DIR (HIP/Triton/hipBLASLt).\n' +
  '(2) SKIP Tier-B AITER/CK env tune — those inputs are CDNA-calibrated.\n' +
  '(3) Fill `baseline_ms` = the FROZEN input kernel`s ms on the oracle.\n' +
  '(4) DECIDE the author_plan as usual without CDNA SOTA cards.\n' +
  '(5) DO NOT curate e2e_workflow/knowledge/learned/. WRITE NOTHING under e2e_workflow/.'
  )
  : (
  'STANDALONE kernel bake-off — there is NO live server (do not try to launch or capture from one).\n' +
  '(1) Tier-A DISCOVER: bench every candidate backend on the immutable oracle in OP_TASK_DIR.\n' +
  '(2) Tier-B TUNE is STILL IN SCOPE — run it OFFLINE, not from a server capture. The step`s "capture ' +
  'shapes from a warm server" instruction is replaced here: take the tune shapes DIRECTLY from ' +
  'OP_TASK_DIR/meta.json (the frozen oracle carries the real M/N/K, the `bias` flag, and dtype — so you ' +
  'do NOT need a server and do NOT have to guess bias). Write them into the untuned CSV, run the gradlib ' +
  '(GEMM) / CK tuner as usual, deploy the tuned CSV (AITER_CONFIG_GEMM_BF16 / the CK env), then VERIFY ' +
  'engagement AND measure speed in the ISOLATED unittest/op_bench with AITER_LOG_TUNED_CONFIG=1 (look for ' +
  '"is tuned on cu_num" in THIS process`s output — NOT server.log). Return the best tuned backend as a ' +
  'winner_kind=env candidate with apply_env + tuning_artifact.\n' +
  '(3) Fill `baseline_ms` = the FROZEN input kernel`s ms on the oracle (baseline_src/ / meta.baseline_callable), ' +
  'and `tuned_speedup` = baseline_ms / (best tuned backend ms) — i.e. the tuned env win measured against ' +
  'the SAME frozen baseline the author lanes use, so it is directly comparable. If no tuned backend beats ' +
  'the baseline, set tuned_speedup=0.\n' +
  '(4) SKIP only pure server-flag levers (--attention-backend swap; serving-only fp8/MoE playbook probes ' +
  'that need a running server). (5) DECIDE the author_plan as usual.\n' +
  '(6) DO NOT run your step 6 (`CURATE SKILL_DIR/knowledge/learned/`). Your SKILL_DIR points at ' +
  'e2e_workflow, but this is a kernel_workflow run: e2e`s learned/ is e2e`s own memory (owned by its ' +
  'system_architect after an e2e A/B), and a kernel bake-off has no e2e-transfer evidence to put in it. ' +
  'This run`s learned sink is kernel_workflow/knowledge/learned/, curated by the TechLead`s ' +
  'update_experience step after Report. WRITE NOTHING under e2e_workflow/ — read it freely, but the ' +
  'only files you create or modify live under EVAL_DIR (plus the tuning artifacts you were asked for).'
  );
const bake = await agentT(
  roleAgentFrom(E2E_WF_DIR, 'op_benchmarker', 'bakeoff', DISCOVER_INTRO, {
    EVAL_DIR, OP_TASK_DIR: oracle.task_dir, OP_KIND: oracle.op_kind,
    CANDIDATE_BACKENDS: (BACKENDS.length ? BACKENDS : (oracle.candidate_backends || [])),
    GPU_ID: GPU_LIST[0], ENABLE_FP8, LIVE_SERVER: 'false',
    KERNEL_WF_DIR: WORKFLOW_DIR, KERNEL_BUDGET: BUDGET, SKILL_DIR: E2E_WF_DIR,
  }),
  { phase: 'Discover', label: 'op_benchmarker:discover', schema: OPBENCH_SCHEMA }) || {};

// Resolve lanes. The INCUMBENT (original input) language ALWAYS competes as an in-place `optimize` lane:
// it is the floor every rewrite must beat, and the entire premise of a bake-off is "keep the fastest,
// INCLUDING simply optimizing what we already have". args.backends only ADDS rewrite candidates on top —
// it can never drop the incumbent (otherwise a rewrite could "win" while being slower than the un-tried
// in-place optimization of the original). Other languages author (or rewrite->optimize if an editable impl
// already exists on the box).
// `author_plan` is a free-form array of objects in the schema, so the agent may name the language under
// any of a few obvious keys. Naive `String(a.language)` turned a missing key into the *string* "undefined",
// which is truthy — it survived .filter(Boolean) and became a real lane literally named `undefined` (lane
// dir `bakeoff/undefined/`, winner row `lang: "undefined"`). Read the aliases; an entry that still has no
// name is an entry about THIS op, so attribute it to the incumbent language rather than inventing one.
const liveLang = String(oracle.live_backend || '').toLowerCase();
const rawLang = (a) => String((a && (a.language || a.lang || a.backend || a.target_language)) || '').trim();
const langOf = (a) => rawLang(a).toLowerCase() || liveLang;
const unnamed = (bake.author_plan || []).filter(a => !rawLang(a)).length;
if (unnamed) log(`WARNING: ${unnamed} author_plan entr(y|ies) carried no language field; ` +
                 `attributed to the incumbent language '${liveLang || '(unknown)'}'. Check Discover output.`);
const modeOf = (a) => (a && a.route === 'rewrite') ? 'optimize' : 'author';
// route per language, for the explicit-`backends` path (an existing editable impl => optimize, not author).
const planByLang = Object.fromEntries((bake.author_plan || []).map(a => [langOf(a), modeOf(a)]));
if (!liveLang) {
  // The freezer could not identify the input language, so the incumbent optimize lane cannot be guaranteed.
  // Do NOT silently proceed with rewrites only — that would let a slower rewrite "win" over a baseline we
  // never tried to optimize in-place. Surface it loudly so it is caught, then continue with what we have.
  log('WARNING: oracle.live_backend is empty — cannot guarantee the incumbent-language optimize lane; ' +
      'rewrites will still run but the original language is under-represented. Check oracle_freezer.');
}
// Incumbent FIRST (force-included regardless of args.backends / discovery), then the requested/discovered
// rewrites. Deduped on lang+mode, NOT on lang alone: "optimize the existing Triton in place" and "author a
// fresh Triton from scratch" are two genuinely different candidates and both deserve a lane.
const wanted = [];
const seen = new Set();
const want = (lang, mode) => {
  if (!lang) return;
  const k = `${lang}:${mode}`;
  if (seen.has(k)) return;
  seen.add(k);
  wanted.push({ lang, mode });
};
want(liveLang, 'optimize');
if (BACKENDS.length) BACKENDS.forEach(l => want(l, l === liveLang ? 'optimize' : (planByLang[l] || 'author')));
else (bake.author_plan || []).forEach(a => want(langOf(a), modeOf(a)));
// Lane dir/log key: plain language when that language has a single lane, `<lang>_<mode>` when it has two.
const laneCount = {};
wanted.forEach(w => { laneCount[w.lang] = (laneCount[w.lang] || 0) + 1; });
const lanes = wanted.map(w => ({
  lang: w.lang, mode: w.mode,
  key: laneCount[w.lang] > 1 ? `${w.lang}_${w.mode}` : w.lang,
}));
if (!lanes.length) {
  log('Discover produced no viable lanes; aborting.');
  return { mode: MODE, task_dir: oracle.task_dir, validation_status: 'no_lanes', winner: null };
}
// a single discovered lane just runs that one lane below (equivalent to a plain single-language optimize).
log(`baseline ${bake.best_known_ms != null ? bake.best_known_ms + ' ms' : '(unknown)'}; ` +
    `${lanes.length} lane(s) = ${lanes.map(l => `${l.lang}:${l.mode}`).join(', ')}; ` +
    `GPUs=${GPU_LIST.length} => concurrency ${Math.min(lanes.length, GPU_LIST.length)}; ` +
    `est cost ~${lanes.length}x a single ${BUDGET}-budget run.`);

// ===========================================================================
// PHASE: Bakeoff — one UNCHANGED kernel_lane worker per language, parallel over the
// GPU pool. dispatcher(level0) -> worker(level1): exactly one nesting level per lane.
// Every lane runs against oracle.task_dir's immutable unittest => the SAME frozen
// baseline denominator => directly comparable speedups (anti-cheating invariant).
// ===========================================================================
phase('Bakeoff');
const sem = makeSem(GPU_LIST);
const results = await Promise.all(lanes.map(l => sem.with(1, async ([gpu]) => {
  try {
    const r = await workflow({ scriptPath: WORKER }, {
      kernel_path: oracle.task_dir, workflow_dir: WORKFLOW_DIR,
      // This lane runs on an oracle_freezer-frozen task dir, so a GEAK_TIMING_RECEIPT is EXPECTED and
      // its absence is a real fault. The optimize/author path spreads {...A} and never sets this, so a
      // pass-through lane (and e2e, which calls this worker directly) correctly defaults to false —
      // there is no Freeze on those routes, so director must not demand a receipt they cannot produce.
      frozen_oracle: 'true',
      expected_gfx: LANE_GFX, expected_target: LANE_TARGET,
      expected_device_name: LANE_DEVICE_NAME,
      expected_physical_cu_count: LANE_PHYSICAL_CU_COUNT,
      mode: l.mode, target_language: l.lang,
      op_spec: oracle.op_spec || OP_SPEC, workload_spec_path: oracle.workload_path || WORKLOAD_SPEC_PATH || '',
      budget: BUDGET, gpu_ids: gpu, gpu_mode: GPU_MODE, task: TASK, apply_to_original: 'false',
      exp_root: `${EVAL_DIR}/bakeoff/${l.key}`,
      use_expert_skills: USE_EXPERT_SKILLS ? 'true' : 'false', expert_skills_dir: EXPERT_SKILLS_DIR,
      perf_knowledge_dir: KERNEL_KNOWLEDGE_DIR,
      // Forward the KB switch. This arg object is explicit (the optimize/author path spreads {...A},
      // this one does not), so anything omitted here silently reverts to the lane's default — a
      // caller asking for a KB-off bakeoff would have got eight KB-on lanes and no error.
      use_learned_kb: A.use_learned_kb != null ? String(A.use_learned_kb) : 'true',
      // This explicit arg object drops nothing load-bearing that the parent set: forward llm_stats
      // when supplied so a parent opt-out reaches each lane (unset stays absent -> lane default on).
      ...(A.llm_stats != null ? { llm_stats: String(A.llm_stats) } : {}),
      // Curation is central in bake-off mode (see the UpdateExperience step below). In optimize/author
      // mode this dispatcher is a passthrough, so the lane keeps its default `on` and curates itself.
      update_experience: 'off',
      warm_start: WARM_START, kb_artifacts_dir: KB_ARTIFACTS_DIR, ...KB_PLANE_ARGS,
    });
    // Absorb this lane's own agent timeline so the dispatcher's returned timeline accounts for every
    // lane's attempts (kernel lanes never persist their own trace file, so this is the only path).
    if (LLM_STATS && r && r.llm_timeline) LLM_TL.nested.push(r.llm_timeline);
    const speedup = primSpeedup(r);
    log(`lane ${l.key}:${l.mode} -> ${speedup ? speedup.toFixed(2) + 'x' : 'no result'} (${r ? r.validation_status : 'null'})`);
    return { lane: l, r, speedup };
  } catch (e) {
    log(`lane ${l.key} failed: ${e && e.message ? e.message : e}`);
    return { lane: l, r: null, speedup: 0 };
  }
})));

// ===========================================================================
// PHASE: Report — rank ALL candidates on the SAME frozen baseline; write table + winner.
// Candidates are THREE classes, all scored against the one frozen input-kernel baseline:
//   (a) each kernel_lane lane (input-language optimize + per-language author) — a source patch;
//   (b) the tuned env backend from Discover (aiter/CK offline tune) — an env/config winner (no patch).
// ===========================================================================
phase('Report');
// (a) lane candidates
const cands = results.map(x => ({
  lang: x.lane.lang, mode: x.lane.mode, kind: 'lane', speedup: x.speedup,
  validation_status: x.r ? x.r.validation_status : 'failed',
  eval_dir: x.r ? x.r.eval_dir : '', patch: x.r ? x.r.final_patch : '', apply_env: '', tuning_artifact: '',
}));
// (b) tuned env backend candidate (only if it beat the frozen baseline)
const tunedSpeedup = Number(bake.tuned_speedup);
if (Number.isFinite(tunedSpeedup) && tunedSpeedup > 1.0 && bake.winner_backend && bake.winner_backend !== 'none') {
  cands.push({
    lang: bake.winner_backend, mode: 'env-tune', kind: 'env', speedup: tunedSpeedup,
    validation_status: 'env', eval_dir: EVAL_DIR, patch: '',
    apply_env: bake.apply_env || '', tuning_artifact: bake.tuning_artifact || '',
  });
  log(`tuned env backend ${bake.winner_backend} -> ${tunedSpeedup.toFixed(2)}x (vs frozen baseline)`);
}
// A candidate only WINS if it actually BEAT the frozen baseline (speedup > 1.0) — the SAME guard the
// tuned env candidate uses above. Slower-than-baseline lanes remain in the table (laneRows) for full
// transparency but must never win "by default"; winner=null => validation_status 'no_winner' => keep
// the original kernel. Without this, a lane that is SLOWER than baseline (e.g. the only non-failed lane
// at 0.17x) would be mislabeled the winner and mislead downstream automation reading .winner.
// ...and only if the director ACCEPTED it. Ranking on speed alone let a lane whose validation came
// back `flagged` (patch did not install, correctness failed, contended box) win the bake-off, be
// applied to the original kernel, and be curated into the KB — a number with a known reason not to
// be believed, promoted by every downstream step. Reported in review of #411. `laneRows` still
// carries every lane for transparency; only eligibility to WIN is tightened.
const ACCEPTED = (c) => String(c.validation_status || '').toLowerCase() === 'accepted';
const ranked = cands.filter(c => c.speedup > 1.0 && (c.kind !== 'lane' || ACCEPTED(c)))
  .sort((a, b) => b.speedup - a.speedup);
const rejectedByGate = cands.filter(c => c.speedup > 1.0 && c.kind === 'lane' && !ACCEPTED(c));
if (rejectedByGate.length) {
  log(`bake-off: ${rejectedByGate.length} lane(s) beat the baseline but are NOT eligible to win ` +
      `(validation_status != accepted): ${rejectedByGate.map(c => `${c.lang}=${c.validation_status}`).join(', ')}`);
}
const winner = ranked[0] || null;
const laneRows = cands;
const bestSpeedup = cands.reduce((m, c) => Math.max(m, Number(c.speedup) || 0), 0);

// Apply instruction depends on the winner class: a lane winner is a source patch (git apply); an
// env-tune winner is a config/env deploy (no source change — record apply_env + tuning_artifact).
const applyStep = (APPLY_TO_ORIGINAL !== 'true' || !winner)
  ? `Do NOT modify the original kernel (report only). Set applied_to_original to "".`
  : (winner.kind === 'env')
    ? `The winner is an ENV/CONFIG tune (backend ${winner.lang}), not a source patch. Do NOT edit kernel ` +
      `source. Record in the report the deploy: apply_env='${winner.apply_env}' and tuning_artifact=` +
      `'${winner.tuning_artifact}'. Set applied_to_original to "${winner.lang} (env: ${winner.apply_env})".`
    : (winner.patch
        ? `APPLY the winning lane patch: cd ${KERNEL_PATH_ORIG} && (git apply ${winner.patch} || git apply --3way ${winner.patch}); ` +
          `if it does not apply cleanly, apply it manually to match intent. Set applied_to_original to the winning language. Do NOT delete anything.`
        : `The winner has no patch file; report only and set applied_to_original to "".`);

const rep = await agentT(
  `You are the bake-off reporter. Write a concise markdown comparison report and, if requested, apply the winner.

## Task
1. Write ${EVAL_DIR}/bakeoff_report.md containing:
   - The op: ${oracle.op_kind} from ${KERNEL_PATH_ORIG} (task_dir ${oracle.task_dir}).
   - The FROZEN baseline (the ONE denominator every candidate shares): the input kernel itself
     (baseline_ms = ${bake.baseline_ms != null ? bake.baseline_ms : 'unknown'} ms; op_bench best_known_ms = ${bake.best_known_ms != null ? bake.best_known_ms : 'unknown'} ms).
   - A candidate table sorted fastest-first with columns: language | mode | kind (lane|env) | speedup vs frozen baseline | validation_status | eval_dir | patch/env.
   - The WINNER (fastest verified) and a one-paragraph rationale. All speedups are directly comparable
     because every candidate — the input-language optimize lane, each authored language, AND the tuned
     env backend — was scored against the SAME frozen input kernel.
     NOTE: WINNER may be null — this means NO candidate beat the frozen baseline (every speedup <= 1.0x,
     e.g. all lanes failed or were slower than the original). In that case state clearly that no
     candidate beat the baseline and the ORIGINAL kernel is RETAINED (do not present a slower-than-1.0x
     candidate as a "winner by default"); still list every candidate in the table for transparency.
2. ${applyStep}

## Inputs
${cfg({
    OP_KIND: oracle.op_kind, KERNEL_PATH: KERNEL_PATH_ORIG, TASK_DIR: oracle.task_dir,
    BASELINE_MS: bake.baseline_ms != null ? bake.baseline_ms : '',
    BEST_KNOWN_MS: bake.best_known_ms != null ? bake.best_known_ms : '',
    CANDIDATES: laneRows,
    WINNER: winner,
    APPLY_TO_ORIGINAL,
  })}

Return ONLY {report_path, applied_to_original, note} as StructuredOutput.`,
  { phase: 'Report', label: 'bakeoff:report', schema: REPORT_SCHEMA });

log(winner
  ? `Bake-off COMPLETE. winner=${winner.lang}:${winner.mode} ${winner.speedup.toFixed(2)}x. Results in ${EVAL_DIR}`
  : `Bake-off COMPLETE. NO candidate beat the frozen baseline (best ${bestSpeedup.toFixed(2)}x <= 1.0x across ${cands.length} candidate(s)) — keeping the ORIGINAL kernel. Results in ${EVAL_DIR}`);

// ---------------------------------------------------------------------------
// PHASE: UpdateExperience — bake-off curates CENTRALLY here (its lanes run with
// update_experience=off): the reusable lesson is the cross-language routing outcome, which
// no single lane can see. Sink is THIS workflow's knowledge/learned/ (see its README.md),
// never e2e's. Only on a measured win, and ADD-only — a failed step is byte-neutral.
if (winner && winner.speedup > 1.0) {
  const LEARNED_DIR = `${WORKFLOW_DIR}/knowledge/learned`;
  try {
    const ue = await agentT(
      roleAgent('update_experience', 'Report',
        'Curate one distilled learned card from this bake-off win (ADD-only, measured evidence, ' +
        'ratios not wall-clock; record the pitfalls hit).', {
          SCOPE: 'bakeoff', LEARNED_DIR, SKILL_DIR: WORKFLOW_DIR, EVAL_DIR,
          PERF_KNOWLEDGE_DIR: KERNEL_KNOWLEDGE_DIR,
          CURATION_ISOLATE: RDNA4_ISOLATE ? 'true' : 'false',
          REQUIRED_PLATFORMS: RDNA4_ISOLATE ? ['gfx1201'] : [],
          WINNER: winner, CANDIDATES: laneRows,
          REPORT_PATH: rep ? rep.report_path : `${EVAL_DIR}/bakeoff_report.md`,
          OP_SPEC,
        }),
      { phase: 'Report', label: 'update_experience', schema: UPDATE_EXPERIENCE_SCHEMA });
    if (ue && ue.card_path) log(`[kb] learned card ${ue.action || 'written'}: ${ue.card_path}`);
  } catch (e) {
    log(`[kb] update_experience skipped: ${e && e.message ? e.message : e}`);
  }
}

// Persist the agent timeline, then render the run report (ledger -> clickable role-execution-tree
// HTML + MD twin) as the very last step. kernel_workflow is always a TOP-LEVEL entry point (E2E nests
// kernel_lane.js, not this dispatcher), so it owns its own ledger + report. The nested lane timelines
// are already merged into LLM_TL.nested, so this one report covers the whole bake-off. The script has
// no filesystem access, so a tiny agent writes the timeline and runs the report driver. Entirely
// best-effort: accounting must never fail a run that produced a real speedup.
if (EVAL_DIR && LLM_STATS) {
  try {
    const tlJson = JSON.stringify({ ...LLM_TL, instance: EVAL_DIR });
    await agentT(persistLlmStatsPrompt(EVAL_DIR, tlJson),
      { phase: 'Report', label: 'file_writer:persist_llm_stats',
        schema: obj({ written: { type: 'boolean' }, path: { type: 'string' }, ok: { type: 'boolean' } }, []) });
    const nestedNote = LLM_TL.nested.length ? ' plus ' + LLM_TL.nested.length + ' nested lane(s)' : '';
    log(`LLM token+time+cost ledger -> ${EVAL_DIR}/reports/trace/ and run report (HTML+MD) -> ` +
        `${EVAL_DIR}/report/. ${LLM_TL.events.length} agent attempts recorded${nestedNote}.`);
  } catch (e) {
    log(`LLM stats/report emit failed (NON-FATAL — the run is unaffected): ${String(e)}`);
  }
}

return {
  mode: MODE,
  // `instance` = this run's eval dir: a stable identity so the parser dedupes a timeline reached
  // twice without collapsing two distinct same-shape lanes.
  llm_timeline: LLM_STATS ? { ...LLM_TL, instance: EVAL_DIR } : undefined,
  task_dir: oracle.task_dir,
  eval_dir: EVAL_DIR,
  baseline_ms: bake.baseline_ms != null ? bake.baseline_ms : (bake.best_known_ms != null ? bake.best_known_ms : null),
  candidates: laneRows,
  winner: winner && {
    lang: winner.lang, mode: winner.mode, kind: winner.kind, speedup: winner.speedup,
    eval_dir: winner.eval_dir, final_patch: winner.patch,
    apply_env: winner.apply_env, tuning_artifact: winner.tuning_artifact,
    validation_status: winner.validation_status,
  },
  report_path: rep ? rep.report_path : `${EVAL_DIR}/bakeoff_report.md`,
  applied_to_original: rep ? rep.applied_to_original : '',
  validation_status: winner ? 'ok' : 'no_winner',
};
