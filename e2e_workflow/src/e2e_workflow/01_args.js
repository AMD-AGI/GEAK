// RUNTIME SANDBOX CONTRACT -- read before adding any Node API to this file.
// The script runs on two runtimes with different globals:
//                                             process   require   import()
//   Claude Code's native Workflow tool          no        no      REFUSED AT LAUNCH
//   interface/runtime/engine/run_workflow.mjs   yes       no      yes
// The native sandbox has only the JS built-ins plus agent/parallel/pipeline/phase/log/workflow/
// args/budget/console/setTimeout: no filesystem, no child_process, and one `import(` in code
// rejects the whole script at launch. So: no `import(` and no load-bearing `require(`; file work
// goes in a scripts/ CLI that an agent runs. test_llm_ledger.py enforces the `import(` rule.

// Args + defaults. A JS workflow can't read its own path, so workflow_dir is passed in.
let _rawArgs = args;
let _parseErr = '';
// The driver agent may pass `args` as a complete JSON string with stray trailing characters
// (e.g. an extra '}'). Parse tolerantly: retry while trimming the tail, bounded so a truly
// malformed payload still fails loudly.
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
// Call the single-language worker lane directly: going through the kernel_workflow.js dispatcher
// would nest e2e -> dispatcher -> worker (3 levels), which the runtime forbids.
const KERNEL_WF_SCRIPT = `${KERNEL_WF_DIR}/kernel_lane.js`;

