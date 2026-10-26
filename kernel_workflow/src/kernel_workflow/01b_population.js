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
      && Object.keys(value).every(key => ['id', 'optimizer', 'imports'].includes(key)),
    'each lineage supports only id, optimizer and imports');
    need(typeof value.id === 'string' && /^[a-z][a-z0-9_-]{0,63}$/.test(value.id), 'lineage IDs must be safe lowercase names');
    need(['greedy', 'evo'].includes(value.optimizer), 'each lineage requires optimizer=greedy or evo');
    const imports = value.imports === undefined ? [] : value.imports;
    need(Array.isArray(imports) && imports.every(id => typeof id === 'string')
      && new Set(imports).size === imports.length, 'lineage imports must be distinct peer IDs');
    need(value.optimizer === 'evo' || imports.length === 0, 'greedy lineages require imports=[]');
    return { id: value.id, optimizer: value.optimizer, imports: [...imports] };
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
