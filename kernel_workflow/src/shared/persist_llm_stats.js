
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
