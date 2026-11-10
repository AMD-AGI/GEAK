// ---- Canonical handoff to run_e2e.py ----
// The return value may never reach the transcript (the Workflow can run as a background task) and
// the script has no fs, so an agent persists `wfReturn` to <eval_dir>/workflow_return.json, which
// run_e2e.py reads first. Best-effort: on failure run_e2e.py recovers from disk.
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

