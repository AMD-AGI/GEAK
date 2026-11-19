export const meta = {
  name: 'hip-ep-e2e-workflow',
  description: 'End-to-end loop for an OGA model served by hip-ep (the ONNX Runtime EP for AMD GPUs). Rebuilds the release kernel library, measures TTFT and decode with model_benchmark (with an A/A of the rebuild), profiles the prefill and decode forwards, and reports. The tools it drives live next to this script (tools/hip_ep_e2e).',
  whenToUse: 'Run GEAK end to end on an OGA-exported model on hip-ep. Pass args.model_path (OGA model dir), args.hipep_pkg (release package dir), args.tools_dir (this script\'s dir), args.exp_root, and the machine paths args.therock, args.sysroot, args.py314, args.hipep_src and args.hipep_build (tools/hip_ep_e2e/README.md).',
  phases: [
    { title: 'Setup', detail: 'Director rebuilds the release library and measures it against the release (A/A, baseline)' },
    { title: 'Profile', detail: 'Profiler traces one run -> per-phase kernel shares with sources' },
    { title: 'Validate', detail: 'Validator writes the report' },
  ],
};

// RUNTIME SANDBOX CONTRACT: no filesystem, no child_process, no dynamic module loading. Every file
// or shell action is done by an agent running the tools in TOOLS_DIR; this script only orchestrates.

let _raw = args;
if (typeof _raw === 'string') {
  try { _raw = JSON.parse(_raw); } catch (e) { throw new Error(`args is not a JSON object: ${e}`); }
}
const A = _raw || {};
const need = (k) => {
  const v = String(A[k] || '').replace(/\/+$/, '');
  if (!v) throw new Error(`args.${k} is required`);
  return v;
};
const MODEL_PATH = need('model_path');
const BASE_PKG = need('hipep_pkg');
const TOOLS_DIR = need('tools_dir');
const EXP_ROOT = need('exp_root');
// Every machine path the tools read (hipep_env.PATHS). None has a default.
const TOOL_ENV = {
  HIPEP_PKG: BASE_PKG, HIPEP_THEROCK: need('therock'), HIPEP_SYSROOT: need('sysroot'),
  HIPEP_PY: need('py314'), HIPEP_SRC: need('hipep_src'), HIPEP_BUILD: need('hipep_build'),
};
const LENS = String(A.lens || '1024,2048');
const GEN = parseInt(A.gen != null ? A.gen : 128, 10);
const REPS = parseInt(A.reps != null ? A.reps : 3, 10);
const ROUNDS = parseInt(A.rounds != null ? A.rounds : 2, 10);
const AGENT_RETRIES = 2;
const MIN = 60000;

function cfg(o) {
  return Object.entries(o).map(([k, v]) =>
    `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');
}

// An A/B, a parity run or a table refresh outlasts one Bash call, and a shell the agent's tool
// backgrounds is reaped on the tool's schedule: e2e_workflow.js's LONG MEASUREMENTS rule.
const LONG_COMMANDS = `## Long commands
Steps marked LONG take 10-30 minutes, longer than one Bash call may last. Do not run one in your Bash
tool's background mode: a shell the tool manages for you is reaped on its own schedule. DIR and NAME
are given with each step. If DIR/NAME.pid names a live process (\`ps -o stat= -p $(cat DIR/NAME.pid)\`
prints a state other than Z), the step is still running from an earlier attempt: wait for it, do not
start it again. Otherwise start it orphaned to init, with its pid written first and its exit code
last, as two commands:
    mkdir -p DIR && rm -f DIR/NAME.rc DIR/NAME.pid
    setsid nohup bash -c 'echo $$ > DIR/NAME.pid; <the command>; echo $? > DIR/NAME.rc' > DIR/NAME.log 2>&1 < /dev/null &
Then wait in separate foreground calls, each one ending inside its Bash timeout (10 minutes):
    timeout 540 bash -c 'until [ -s DIR/NAME.rc ]; do sleep 30; done'; cat DIR/NAME.rc; tail -5 DIR/NAME.log
Repeat until DIR/NAME.rc exists; a non-zero code means the step failed, and DIR/NAME.log says why. If it
never appears and the pid shows no live process, the step died: read the log. Never start a command
with a sleep, run one GPU step at a time, and do not return before every LONG step you started has its
.rc: your reply ends your turn, and a result that lands later is lost.`;

function roleAgent(role, ph, intro, inputs) {
  return `You are the hip-ep ${role}. PHASE=${ph}.
First Read ${TOOLS_DIR}/roles/${role}.md and follow it for PHASE=${ph}. Read ${TOOLS_DIR}/DESIGN.md
for the measurement rules. Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
Export every TOOL_ENV variable before running a tool. Never kill or signal a process you did not
start, never pattern-match processes, and never touch another container. GPU tools take the
serving-GPU lock themselves.

${LONG_COMMANDS}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies.`;
}

// Hang guard: a hung agent resolves null after timeoutMs; an API error is retried once.
async function agentT(prompt, opts, timeoutMs) {
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    try {
      if (typeof setTimeout !== 'function') return await agent(prompt, opts);
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`[hung-agent guard] ${opts.label} exceeded ${Math.round(timeoutMs / MIN)} min`);
          resolve(null);
        }, timeoutMs);
      });
      return await Promise.race([
        agent(prompt, opts).then((r) => { clearTimeout(to); return r; },
          (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
    } catch (e) {
      log(`[api-fault guard] ${opts.label} attempt ${attempt}: ${String(e).slice(0, 200)}`);
    }
  }
  return null;
}

const S = (props, req) => ({ type: 'object', properties: props, required: req,
  additionalProperties: true });
const str = { type: 'string' };
const bool = { type: 'boolean' };
const PERLEN = { type: 'object', additionalProperties: true };

// ===========================================================================
phase('Setup');
const setup = await agentT(roleAgent('director', 'setup',
  'Create the eval dir, rebuild the release library and measure it against the release.', {
    MODEL_PATH, BASE_PKG, TOOLS_DIR, EXP_ROOT, LENS, GEN, REPS, ROUNDS, TOOL_ENV,
  }), { label: 'director:setup', phase: 'Setup', schema: S({
  ok: bool, error: str, eval_dir: str, device_gfx: str, aa_json: str, aa_parity_exact: bool,
  baseline: PERLEN,
}, ['ok', 'eval_dir', 'device_gfx', 'aa_json']) }, 150 * MIN);
const setupError = !setup ? 'the director returned no result'
  : !setup.ok ? setup.error || 'the director reported a failure'
    : !setup.eval_dir || !setup.aa_json ? 'no eval dir or no A/A result' : '';
if (setupError) throw new Error(`Setup failed: ${setupError}`);
const EVAL_DIR = setup.eval_dir;
const AA_JSON = setup.aa_json;
log(`baseline ${JSON.stringify(setup.baseline || {})}; A/A ${AA_JSON}`);

// ===========================================================================
phase('Profile');
const prof = await agentT(roleAgent('profiler', 'profile',
  'Trace one run and summarize the prefill and decode forwards.', {
    EVAL_DIR, MODEL_PATH, PKG: BASE_PKG, TOOLS_DIR, LEN: LENS.split(',')[0], AA_JSON, TOOL_ENV,
  }), { label: 'profiler:profile', phase: 'Profile', schema: S({
  heads_json: str,
  prefill: { type: 'array', items: { type: 'object', additionalProperties: true } },
  decode: { type: 'array', items: { type: 'object', additionalProperties: true } },
}, ['heads_json', 'prefill', 'decode']) }, 60 * MIN);
if (!prof || !prof.heads_json) throw new Error('Profile failed: no heads.json');

// ===========================================================================
phase('Validate');
const val = await agentT(roleAgent('validator', 'validate', 'Write the report.', {
  EVAL_DIR, MODEL_PATH, BASE_PKG, AA_JSON, HEADS_JSON: prof.heads_json,
  DEVICE_GFX: setup.device_gfx, AA_PARITY_EXACT: setup.aa_parity_exact,
}), { label: 'validator:validate', phase: 'Validate', schema: S({ report_md: str }, ['report_md']) },
60 * MIN);

const ret = {
  eval_dir: EVAL_DIR, model_path: MODEL_PATH, base_pkg: BASE_PKG, device_gfx: setup.device_gfx,
  aa_json: AA_JSON, aa_parity_exact: setup.aa_parity_exact, baseline: setup.baseline || null,
  report_md: val ? val.report_md : '',
};
await agentT(`Write the JSON below verbatim to "${EVAL_DIR}/workflow_return.json" with the Write ` +
  'tool. Then run exactly this command, which saves its report_md as report.md next to it:\n' +
  `python3 -c 'import json, sys; open(sys.argv[2], "w").write(json.load(open(sys.argv[1]))["report_md"])' ` +
  `"${EVAL_DIR}/workflow_return.json" "${EVAL_DIR}/report.md"\n` +
  `Return {"written": true}.\n\n${JSON.stringify(ret, null, 1)}`,
{ label: 'persist', phase: 'Validate', schema: S({ written: bool }, ['written']) }, 10 * MIN);
return ret;
