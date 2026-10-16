// ---------------------------------------------------------------------------
// Args + defaults. A JS workflow can't read its own path, so workflow_dir is passed in.
// ---------------------------------------------------------------------------
let _rawArgs = args;
let _parseErr = '';
// The driver agent sometimes passes `args` as a JSON STRING rather than an object. The
// string is COMPLETE -- measured 2026-09-15: a valid 32-key object with `state` intact --
// but carries a stray trailing character (one spurious '}'), so a plain JSON.parse throws
// 'Extra data', the coercion fails, workflow_dir is undefined and the run aborts in 28ms.
// This is NOT truncation and NOT a payload-size problem; trimming the state does not fix
// it. Parse tolerantly: retry while dropping trailing junk, bounded so a genuinely
// malformed payload still fails loudly instead of looping.
if (typeof _rawArgs === 'string') {
  let _s = _rawArgs.trim();
  for (let _i = 0; _i < 8 && _s.length > 1; _i++) {
    try { _rawArgs = JSON.parse(_s); _parseErr = ''; break; }
    catch (_e) { _parseErr = String((_e && _e.message) || _e); _s = _s.slice(0, -1).trim(); }
  }
}
const A = _rawArgs || {};
if (Object.keys(A).some((key) => key.startsWith('ablation_'))) {
  throw new Error('ablation_* study controls are retired; use the original revision to replay those studies.');
}
// Agent timeline for the LLM ledger (LLM_STATS + tlAgent: src/shared/tl_agent.js).
const LLM_TL = { schema: 'geak.agent_timeline/1', workflow: 'e2e_workflow', events: [], nested: [] };
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  let _preview;
  try {
    _preview = typeof _rawArgs === 'object' && _rawArgs !== null
      ? 'object keys=[' + Object.keys(_rawArgs).join(',') + ']'
      : JSON.stringify(_rawArgs);
  } catch (_e) { _preview = '<unstringifiable>'; }
  const _s = typeof args === 'string' ? args : '';
  throw new Error('args.workflow_dir is required. DIAGNOSTIC: typeof args=' + (typeof args) +
    ' typeof _rawArgs=' + (typeof _rawArgs) +
    ' rawStrLen=' + _s.length +
    ' parseErr=' + _parseErr +
    ' tail=' + JSON.stringify(_s.slice(-200)) +
    ' preview=' + String(_preview).slice(0, 200));
}
// The UNCHANGED single-kernel workflow. Default: sibling "kernel_workflow" dir next to this one.
const KERNEL_WF_DIR = String(A.kernel_workflow_dir ||
  (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kernel_workflow')).replace(/\/+$/, '');
// The single-language WORKER lane. kernel_workflow.js is now a dispatcher (mode=optimize/author ->
// this worker; mode=bakeoff -> multi-language fan-out), so e2e MUST call the worker DIRECTLY
// (kernel_lane.js) — routing through the dispatcher would add a nesting level (e2e -> dispatcher ->
// worker = 3 levels) and the runtime forbids it. The worker's behavior/args are unchanged.
const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/kernel_lane.js`;

