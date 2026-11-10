// Experimental decision-only entry, before Setup. The host reads exact trusted seed files;
// the native planner constructs one new decision. No archive constructor or source application.
if (NATIVE_PLANNER_SEED) return await nativeSeededPlannerDecision();

async function nativeSeededPlannerDecision() {
  const ref = A.native_planner_seed_ref;
  const read = async () => {
    const result = await runProcess({cmd: 'python3', args: ['-I', '-B',
      `${WORKFLOW_DIR}/scripts/read_native_planner_seed.py`, ref.path, ref.sha256, String(ref.bytes)],
      timeoutMs: Math.min(30000, AGENT_TIMEOUT_MS), graceMs: 2000, settleMs: 2000});
    if (result.status !== 'completed' || result.code !== 0 || !result.group_cleared
        || result.strays_killed || result.stdio_held || typeof result.stdout !== 'string'
        || result.stdout.length > 4194304)
      throw new Error('native planner seed source closure rejected');
    const value = JSON.parse(result.stdout);
    if (value?.seed?.schema !== 'geak.native_planner_seed/1' || !value.archive
        || !Array.isArray(value.verified_refs)) throw new Error('invalid native planner seed reader output');
    return value;
  };
  const bound = await read(), {seed, archive} = bound, c = seed.context;
  if (KERNEL_PATH_ORIG !== seed.task_root)
    throw new Error('native planner seed task_root must match kernel_path');
  const handoffMetric = EXTERNAL_OBJECTIVE || {kind: HAS_WORKLOAD ? 'time_weighted' : 'geomean'};
  const evo = evolutionDecisionView({base: archive.base, baseCommit: archive.base_commit,
    archive: seed.archive.path, entries: archive.entries, getBestId: () => archive.best_id,
    metric: archive.metric ?? null});
  const archiveHasSource = () => evo.entries.some(row => row.patch_ref && !row.retention_error);
  if (!archiveHasSource()) throw new Error('native planner seed has no retained source');
  const planner = nativePlannerCall({EVAL_DIR: c.eval_dir, round: c.round,
    remaining: c.budget_remaining, cumulative: c.cumulative_speedup,
    handoffMetric,
    BASELINE_GEOMEAN_MS: c.baseline_geomean_ms, profileSummary: c.profile_summary,
    bestPerCase: c.current_best_per_case, finalWinner: c.final_winner,
    measurementRef: () => c.current_best_ref || {unavailable: 'no bound current measurement'},
    history: c.history, archiveHasSource, evo, GFX: c.gfx, DEVICE_TARGET: c.device_target,
    PHYSICAL_CU_COUNT: c.physical_cu_count, ROOFLINE_STATUS: c.roofline_status,
    researchBriefPath: c.research_brief_path, BASELINE_PER_CASE: c.baseline_per_case,
    getLeft: () => Infinity, KK_OPERATOR: c.kk_operator, KK_LANGUAGE: c.kk_language, KK_REFS: c.kk_refs,
    ENGINEER_CALL: {processDeadlineEpoch: A.native_planner_deadline_epoch, processTimeoutMs: AGENT_TIMEOUT_MS},
    DEADLINE_EPOCH: 0, ENGINEER_RESERVE_S: 0, ENGINEER_CUTOFF_EPOCH: null,
    evoPlanSchema: nativeEvolutionPlannerSchema(true), allowInitialBenchmark: false,
    recordPlannerInputs: true, factsInput: async () => ({NATIVE_PLANNER_SEED_CONTEXT: {
      id: seed.id, original_archive: seed.original_archive, archive: seed.archive,
      authority: 'prospective decision from source-bound historical evidence; no current measurement or Git reconstruction',
      historical_archive_metric: archive.metric ?? null,
      historical_metric_scope: 'The prospective PRIMARY_METRIC does not identify historical archive aggregates; absent archival metric identity remains unknown.',
      facts: c.facts}})});
  const plan = await planner.callPlanner('initial',
    'Decide this round\'s orthogonal directions (or stop).' +
      ` Read ${WORKFLOW_DIR}/roles/evolution.md for the retained-approach policy.`,
    false, `tech_lead:plan r${c.round}`);
  const requested = Array.isArray(plan?.directions) ? plan.directions.map((direction, index) => ({
    index, requested_id: direction?.id ?? null,
    requested_parent_id: direction?.parent_id ?? null,
    requested_donor_ids: direction?.donor_ids ?? null})) : [];
  const result = {schema: 'geak.native_planner_decision/1', seed_id: seed.id, seed_ref: ref,
    original_archive_ref: seed.original_archive, archive_ref: seed.archive,
    verified_refs: bound.verified_refs, constructed_planner_call: planner.constructedCall(),
    agent_timeline: LLM_TL,
    requested_plan: plan ?? null, requested_selections: requested, resolved_selections: [],
    case_measurement_observation: planner.caseMeasurementObservation(),
    continuation_materialization_qualified: false,
    authority: 'selection resolution only; historical archive verification is not current source qualification',
    direction_allocation: c.budget_remaining === null ? {unavailable: 'budget_remaining is unbound'}
      : {budget_remaining: c.budget_remaining},
    source_closure_rechecked: false};
  // A model tool may have altered a referenced file. Never return usable resolutions without rereading.
  try {
    const after = await read();
    if (JSON.stringify(after) !== JSON.stringify(bound)) throw new Error('seed changed');
    result.source_closure_rechecked = true;
  } catch (_) { return {...result, status: 'source_closure_changed'}; }
  if (!plan) return {...result, status: 'planner_unavailable'};
  if (plan.stop === true) return {...result, status: 'stopped'};
  if (!Array.isArray(plan.directions) || !plan.directions.length)
    return {...result, status: 'selection_rejected', reference_errors: ['no requested directions']};
  const retained = id => id === 'ORIGINAL' || evo.entries.some(row => row.id === id
    && row.patch_ref && !row.retention_error);
  const errors = [];
  for (const row of requested) {
    if (typeof row.requested_parent_id !== 'string' || !retained(row.requested_parent_id))
      errors.push({index: row.index, field: 'parent_id', error: 'missing or unavailable requested parent'});
    if (!Array.isArray(row.requested_donor_ids)
        || row.requested_donor_ids.some(id => typeof id !== 'string' || !retained(id)))
      errors.push({index: row.index, field: 'donor_ids', error: 'missing or unavailable requested donors'});
  }
  if (errors.length) return {...result, status: 'selection_rejected', reference_errors: errors};
  // Unknown allocation is left unknown. IDs can still be resolved against the retained source set.
  const directions = c.budget_remaining === null
    ? plan.directions.map((d, idx) => ({...d, idx, id: d.id || `r${c.round}_d${idx}`}))
    : nativeDirectionSelection(plan, c.budget_remaining, c.round, true);
  const resolved = evo.prepare(directions, c.round);
  if (!resolved.length) return {...result, status: 'budget_exhausted'};
  if (resolved.some(d => d.evolution_context.parent_resolution !== 'explicit'
      || d.evolution_context.reference_errors.length))
    return {...result, status: 'selection_rejected', reference_errors: ['native resolution fallback refused']};
  const sourceRef = id => id === 'ORIGINAL' ? {id, task_root: seed.task_root, task_files: seed.task_files}
    : {id, ...seed.source_refs.find(ref => ref.entry_id === id)};
  result.resolved_selections = resolved.map(d => ({
    direction_index: d.idx, id: d.id, requested_id: requested[d.idx].requested_id,
    requested_parent_id: requested[d.idx].requested_parent_id,
    requested_donor_ids: requested[d.idx].requested_donor_ids,
    parent_id: d.parent_id, donor_ids: d.donor_ids,
    parent_ref: sourceRef(d.parent_id), donor_refs: d.donor_ids.map(sourceRef),
    native_resolution: d.evolution_context}));
  return {...result, status: 'resolved'};
}
