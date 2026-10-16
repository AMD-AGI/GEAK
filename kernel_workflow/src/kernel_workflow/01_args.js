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
