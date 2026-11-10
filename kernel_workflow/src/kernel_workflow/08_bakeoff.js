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
      // Preserve explicit search policy and evidence controls in every language lane.
      ...Object.fromEntries(['optimizer', 'evolution_start', 'evolution_visibility', 'evolution_return_development',
        'evolution_import_stall_sources', 'evolution_case_evidence_view',
        'evolution_evidence_handoff', 'evolution_preserve_founder_attempts',
        'record_planner_inputs', 'case_measurement_view'].filter(key => A[key] !== undefined)
        .map(key => [key, A[key]])),
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
      // The explicit bakeoff argument object must preserve the caller's role allowance too.
      ...(A.agent_timeout_ms != null ? { agent_timeout_ms: A.agent_timeout_ms } : {}),
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
    const failure = {lane: l.key, eval_dir: `${EVAL_DIR}/bakeoff/${l.key}`,
      code: e && e.code || 'lane_execution_failed', message: String(e && e.message || e)};
    return {lane: l, r: {eval_dir: failure.eval_dir, validation_status: 'unavailable', process_validated: false,
      final_speedup: null, final_patch: '', delivery_incomplete: [failure]}, speedup: null,
      failure};
  }
})));
