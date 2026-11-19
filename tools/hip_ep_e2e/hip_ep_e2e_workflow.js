export const meta = {
  name: 'hip-ep-e2e-workflow',
  description: 'End-to-end optimizer for an OGA model served by hip-ep (the ONNX Runtime EP for AMD GPUs). Measures TTFT and decode with model_benchmark and profiles the prefill and decode forwards. For the ops whose kernels carry the prefill or decode (matmul_nbits, gqa), it refreshes hip-ep\'s own autotune tables for the model (the tuning lane). The candidate is a rebuilt libcustom_kernels library in a copy of the release package, accepted only through an interleaved TTFT/decode A/B and a teacher-forced logits parity gate. The tools it drives live next to this script (tools/hip_ep_e2e).',
  whenToUse: 'Optimize an OGA-exported model on hip-ep. Pass args.model_path (OGA model dir), args.hipep_pkg (release package dir), args.tools_dir (this script\'s dir), args.exp_root, and the machine paths args.therock, args.sysroot, args.py314, args.hipep_src and args.hipep_build (tools/hip_ep_e2e/README.md).',
  phases: [
    { title: 'Setup', detail: 'Director rebuilds the release library, records the baseline and the A/A' },
    { title: 'Profile', detail: 'Profiler traces one run -> per-phase kernel shares with sources' },
    { title: 'Strategize', detail: 'The tuning lane runs for the tunable ops that carry enough of the prefill or decode' },
    { title: 'Tuning', detail: 'Tuner refreshes those ops\' autotune tables for the model -> rebuild -> gate' },
    { title: 'Validate', detail: 'Validator re-measures the final package against the release and writes the report' },
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
const AA = String(A.aa != null ? A.aa : 'true') !== 'false';
// The ops with a hip-ep autotune table the tuning lane can refresh, the kernels each table picks
// configs for, and the phases those kernels run in (matmul_nbits' table covers the prefill GEMM).
// An op is tuned when its kernels hold at least MIN_SHARE of a phase.
const TUNABLE = [
  { op: 'matmul_nbits', kernels: /^MatMulNBitsWMMA/, phases: ['prefill'] },
  { op: 'gqa', kernels: /^gqa_flash/, phases: ['prefill', 'decode'] },
];
const MIN_SHARE = 0.2;
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
const num = { type: 'number' };
const str = { type: 'string' };
const bool = { type: 'boolean' };
const PERLEN = { type: 'object', additionalProperties: true };
const GATE_SCHEMA = S({
  accepted: bool, reason: str, pkg: str, overlay: str, kernel_lib_md5: str,
  ab_json: str, parity_json: str, per_len: PERLEN, parity_top1: num, parity_rel_l2: num,
}, ['accepted', 'reason', 'pkg', 'overlay', 'ab_json', 'parity_json']);

// ===========================================================================
phase('Setup');
const setup = await agentT(roleAgent('director', 'setup',
  'Check the runtime, create the eval dir, rebuild the release library, record the baseline and ' +
  '(if AA) the A/A of the rebuild.', {
    MODEL_PATH, BASE_PKG, TOOLS_DIR, EXP_ROOT, LENS, GEN, REPS, ROUNDS, AA: String(AA), TOOL_ENV,
  }), { label: 'director:setup', phase: 'Setup', schema: S({
  ok: bool, error: str, eval_dir: str, device_gfx: str, base_kernel_lib_md5: str,
  rebuilt_kernel_lib_md5: str, baseline_json: str, noise_from: str, aa_parity_exact: bool,
  baseline: PERLEN,
}, ['ok', 'eval_dir', 'device_gfx', 'baseline_json', 'noise_from']) }, 150 * MIN);
const setupError = !setup ? 'the director returned no result'
  : !setup.ok ? setup.error || 'the director reported a failure'
    : !setup.eval_dir || !setup.baseline_json ? 'no eval dir or no baseline'
      : AA && !setup.noise_from ? 'no A/A result' : '';
if (setupError) throw new Error(`Setup failed: ${setupError}`);
const EVAL_DIR = setup.eval_dir;
const NOISE_FROM = setup.noise_from || '';
log(`baseline ${JSON.stringify(setup.baseline || {})}; noise band from ${NOISE_FROM || 'the 1% floor'}`);

// ===========================================================================
phase('Profile');
const prof = await agentT(roleAgent('profiler', 'profile',
  'Trace one run and summarize the prefill and decode forwards.', {
    EVAL_DIR, MODEL_PATH, PKG: BASE_PKG, TOOLS_DIR, LEN: LENS.split(',')[0], TOOL_ENV,
  }), { label: 'profiler:profile', phase: 'Profile', schema: S({
  heads_json: str,
  prefill: { type: 'array', items: { type: 'object', additionalProperties: true } },
  decode: { type: 'array', items: { type: 'object', additionalProperties: true } },
}, ['heads_json', 'prefill', 'decode']) }, 60 * MIN);
if (!prof || !prof.heads_json) throw new Error('Profile failed: no heads.json');

// ===========================================================================
phase('Strategize');
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const shares = {};
for (const t of TUNABLE) {
  shares[t.op] = Math.max(...t.phases.map((ph) => (prof[ph] || [])
    .filter((r) => t.kernels.test(String(r.base || ''))).reduce((s, r) => s + (Number(r.share) || 0), 0)));
}
const tuneOps = TUNABLE.map((t) => t.op).filter((op) => shares[op] >= MIN_SHARE);
log(`largest phase share per tunable op: ${TUNABLE.map((t) => `${t.op} ${pct(shares[t.op])}`).join(', ')}` +
  ` (the tuning lane needs ${pct(MIN_SHARE)})`);

// The candidate: the release, unless the tuning lane's package is accepted.
let cur = { pkg: BASE_PKG, overlay: '', label: 'release' };
const history = [];
if (tuneOps.length) {
  phase('Tuning');
  const laneDir = `${EVAL_DIR}/lanes/0_tuning`;
  const r = await agentT(roleAgent('tuner', 'tuning',
    `Refresh the ${tuneOps.join(' and ')} autotune tables for this model, rebuild, and gate the candidate.`, {
      EVAL_DIR, LANE_DIR: laneDir, MODEL_PATH, BASE_PKG, NOISE_FROM, TOOLS_DIR, LENS, GEN, REPS,
      ROUNDS, TOOL_ENV, ARCH: setup.device_gfx, OPS: tuneOps.join(','),
    }), { label: 'tuner:tuning', phase: 'Tuning', schema: GATE_SCHEMA }, 360 * MIN);
  history.push({ lane: 'tuning', ops: tuneOps, shares,
    result: r || { accepted: false, reason: 'the tuner returned no result' } });
  if (r && r.accepted) cur = { pkg: r.pkg, overlay: r.overlay, label: 'tuning' };
  log(`tuning: ${r ? (r.accepted ? 'ACCEPTED' : `rejected (${r.reason})`) : 'no result'}`);
} else {
  history.push({ lane: 'tuning', ops: [], shares, result: { accepted: false,
    reason: `not run: no tunable op holds ${pct(MIN_SHARE)} of a phase` } });
}

// ===========================================================================
phase('Validate');
const val = await agentT(roleAgent('validator', 'validate',
  'Independently re-measure the final package against the release and write the report.', {
    EVAL_DIR, MODEL_PATH, BASE_PKG, FINAL_PKG: cur.pkg, FINAL_OVERLAY: cur.overlay,
    HISTORY: history, BASELINE_JSON: setup.baseline_json, HEADS_JSON: prof.heads_json,
    DEVICE_GFX: setup.device_gfx, AA_PARITY_EXACT: setup.aa_parity_exact, NOISE_FROM, TOOLS_DIR,
    LENS, GEN, REPS, ROUNDS, TOOL_ENV,
  }), { label: 'validator:validate', phase: 'Validate', schema: S({
  validation_status: { type: 'string', enum: ['validated', 'not_reproduced', 'parity_failed', 'no_change'] },
  report_md: str, ab_json: str, parity_json: str,
  per_len: PERLEN, parity_top1: num, parity_rel_l2: num,
}, ['validation_status', 'report_md']) }, 120 * MIN);

// A candidate the validator could not reproduce against the release is reported, not returned as
// the final package.
const status = val ? val.validation_status : 'unknown';
const kept = status === 'validated';
const ret = {
  eval_dir: EVAL_DIR, model_path: MODEL_PATH, base_pkg: BASE_PKG, device_gfx: setup.device_gfx,
  final_pkg: kept ? cur.pkg : BASE_PKG, final_overlay: kept ? cur.overlay : '',
  final_label: kept ? cur.label : 'release',
  candidate: cur.pkg === BASE_PKG ? null : { pkg: cur.pkg, overlay: cur.overlay, label: cur.label },
  noise_from: NOISE_FROM, aa_parity_exact: setup.aa_parity_exact, baseline: setup.baseline || null,
  lanes: history, validation_status: status, per_len: val ? val.per_len : null,
  report_md: val ? val.report_md : '',
};
await agentT(`Write the JSON below verbatim to "${EVAL_DIR}/workflow_return.json" with the Write ` +
  'tool. Then run exactly this command, which saves its report_md as report.md next to it:\n' +
  `python3 -c 'import json, sys; open(sys.argv[2], "w").write(json.load(open(sys.argv[1]))["report_md"])' ` +
  `"${EVAL_DIR}/workflow_return.json" "${EVAL_DIR}/report.md"\n` +
  `Return {"written": true}.\n\n${JSON.stringify(ret, null, 1)}`,
{ label: 'persist', phase: 'Validate', schema: S({ written: bool }, ['written']) }, 10 * MIN);
return ret;
