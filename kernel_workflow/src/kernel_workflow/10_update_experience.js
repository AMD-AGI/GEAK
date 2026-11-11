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
