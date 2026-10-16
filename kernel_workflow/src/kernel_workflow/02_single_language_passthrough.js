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

