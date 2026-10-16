// ---------------------------------------------------------------------------
// Canonical handoff to run_e2e.py — write the SCHEMA-VALIDATED return to a FIXED
// on-disk path from INSIDE the workflow.
//
// WHY (the perfect-cooperation contract): run_e2e.py must produce a TRUTHFUL
// result.json regardless of HOW the agent ran this Workflow. Newer Claude CLIs
// run a Workflow invocation as a NON-BLOCKING background task: the orchestrating
// turn ends before completion, so this return value is NEVER echoed to the
// transcript for run_e2e.py to scrape. The script itself has no fs, so we have an
// agent (the only thing with Write) persist `wfReturn` to
// <eval_dir>/workflow_return.json as the workflow's FINAL act. run_e2e.py reads
// THAT one contracted file first (deterministic, no transcript scrape, no parsing
// of free-form per-candidate files).
//
// Best-effort by design: if this persist fails (agent variance), the returned
// value is UNCHANGED and run_e2e.py still falls back to its layered, schema-robust
// disk recovery (director_e2e_validation.json -> accepted intermediate win ->
// measured-baseline no_gain). So correctness never DEPENDS on this single write —
// it is the preferred fast path, not the only path.
if (EVAL_DIR) {
  try {
    const json = JSON.stringify(wfReturn, null, 2);
    await safeAgent(
      `You are a file writer. Use the Write tool to create the file ` +
      `"${EVAL_DIR}/workflow_return.json" with EXACTLY the content below, verbatim. ` +
      `Do NOT reformat, truncate, summarize, or add/remove any keys or values:\n\n` +
      '```json\n' + json + '\n```\n\n' +
      `Then return {"written": true, "path": "${EVAL_DIR}/workflow_return.json"}.`,
      { phase: 'Validate', label: 'persist-workflow-return',
        schema: obj({ written: { type: 'boolean' }, path: { type: 'string' } }, []) },
      2);
    log(`Persisted canonical workflow_return.json -> ${EVAL_DIR}/workflow_return.json (run_e2e.py reads this first).`);
  } catch (e) {
    log(`persist workflow_return.json failed (NON-FATAL — run_e2e.py will recover from disk): ${String(e)}`);
  }
}

