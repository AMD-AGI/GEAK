export const meta = {
  name: 'hip-ep-e2e-workflow',
  description: 'End-to-end optimizer for an OGA model served by hip-ep (the ONNX Runtime EP for AMD GPUs). Measures TTFT and decode with model_benchmark and profiles the prefill and decode forwards. For the ops whose kernels carry the prefill or decode, it runs two lanes: a tuning lane (refresh hip-ep\'s own autotune tables, matmul_nbits and gqa, for the model) and a kernel lane (the UNCHANGED kernel_workflow on a task built from hip-ep\'s own unit-test leaves; the 4-bit MatMulNBits prefill GEMM is the op it has a task for). Every candidate is a rebuilt libcustom_kernels library in a copy of the release package, accepted only through an interleaved TTFT/decode A/B and a teacher-forced logits parity gate. The tools it drives live next to this script (tools/hip_ep_e2e).',
  whenToUse: 'Optimize an OGA-exported model on hip-ep. Pass args.model_path (OGA model dir), args.hipep_pkg (release package dir), args.tools_dir (this script\'s dir), args.kernel_workflow_dir (GEAK kernel_workflow/ dir), args.exp_root, and the machine paths args.therock, args.sysroot, args.py314, args.hipep_src and args.hipep_build (tools/hip_ep_e2e/README.md). Optional: args.kernel_budget, args.kernel_hours.',
  phases: [
    { title: 'Setup', detail: 'Director rebuilds the release library, records the baseline and the A/A' },
    { title: 'Profile', detail: 'Profiler traces one run -> per-phase kernel shares with sources' },
    { title: 'Strategize', detail: 'Each lane runs for the ops that carry enough of the prefill or decode' },
    { title: 'Tuning', detail: 'Tuner refreshes those ops\' autotune tables for the model -> rebuild -> gate' },
    { title: 'HeadKernel', detail: 'Extractor builds the kernel task -> nested kernel_workflow lane' },
    { title: 'Integrate', detail: 'Integrator rebuilds the library with the kernel patch (and re-tuned tables) -> gate' },
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
const KERNEL_WF_DIR = need('kernel_workflow_dir');
// The single-language worker, called directly: going through the kernel_workflow.js dispatcher would
// nest this workflow -> dispatcher -> worker (3 levels), which the runtime forbids (as in
// e2e_workflow.js).
const KERNEL_LANE_SCRIPT = `${KERNEL_WF_DIR}/kernel_lane.js`;
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
const KERNEL_BUDGET = parseInt(A.kernel_budget != null ? A.kernel_budget : 6, 10);
const KERNEL_HOURS = Number(A.kernel_hours != null ? A.kernel_hours : 6);
// The op the kernel lane has a GEAK task for (make_task.py --op), the kernels its editable sources
// compile, and the autotune tables to re-sweep on the edited kernel. It runs under the same
// MIN_SHARE rule as the tuning lane.
const KERNEL_OP = { op: 'matmul_nbits_u4_prefill', kernels: /^MatMulNBitsWMMA/, phases: ['prefill'],
  retune: ['matmul_nbits'] };
// A free-string verdict is read by its leading word, as kernel_lane.js reads its own: "PASS - 15/15"
// is a pass, "FAIL" and "did not pass" are not.
const says = (v, w) => String(v == null ? '' : v).trim().toLowerCase().startsWith(w);
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
// An op's share: its kernels' largest share of one phase.
const shareOf = (t) => Math.max(...t.phases.map((ph) => (prof[ph] || [])
  .filter((r) => t.kernels.test(String(r.base || ''))).reduce((s, r) => s + (Number(r.share) || 0), 0)));
const shares = {};
for (const t of TUNABLE) shares[t.op] = shareOf(t);
const tuneOps = TUNABLE.map((t) => t.op).filter((op) => shares[op] >= MIN_SHARE);
const kernelShare = shareOf(KERNEL_OP);
log(`largest phase share per tunable op: ${TUNABLE.map((t) => `${t.op} ${pct(shares[t.op])}`).join(', ')}; ` +
  `kernel lane op ${KERNEL_OP.op} ${pct(kernelShare)} (a lane needs ${pct(MIN_SHARE)})`);

// The current accepted state: each candidate is gated against it, and parity is always checked
// against the release, so accepted changes cannot drift step by step.
let cur = { pkg: BASE_PKG, overlay: '', label: 'release' };
const history = [];

async function tuningLane() {
  phase('Tuning');
  const laneDir = `${EVAL_DIR}/lanes/0_tuning`;
  const r = await agentT(roleAgent('tuner', 'tuning',
    `Refresh the ${tuneOps.join(' and ')} autotune tables for this model, rebuild, and gate the candidate.`, {
      EVAL_DIR, LANE_DIR: laneDir, MODEL_PATH, BASE_PKG, NOISE_FROM, TOOLS_DIR, LENS, GEN, REPS,
      ROUNDS, TOOL_ENV, ARCH: setup.device_gfx, OPS: tuneOps.join(','),
    }), { label: 'tuner:tuning', phase: 'Tuning', schema: GATE_SCHEMA }, 360 * MIN);
  if (r && r.accepted) cur = { pkg: r.pkg, overlay: r.overlay, label: 'tuning' };
  log(`tuning: ${r ? (r.accepted ? 'ACCEPTED' : `rejected (${r.reason})`) : 'no result'}`);
  return { lane: 'tuning', ops: tuneOps, shares,
    result: r || { accepted: false, reason: 'the tuner returned no result' } };
}

async function kernelLane() {
  const entry = (result, extra = {}) => ({ lane: 'kernel', op: KERNEL_OP.op, share: kernelShare, ...extra, result });
  phase('HeadKernel');
  const laneDir = `${EVAL_DIR}/lanes/1_kernel`;
  // kernel_lane.js names the kernel after the task dir (minus `_task`); the name addresses the
  // experience store.
  const taskDir = `${laneDir}/${KERNEL_OP.op}_task`;
  const ex = await agentT(roleAgent('extractor', 'extract',
    'Generate the GEAK kernel task for this op and record its baseline.', {
      EVAL_DIR, LANE_DIR: laneDir, TASK_DIR: taskDir, OP: KERNEL_OP.op, MODEL_PATH,
      CUR_OVERLAY: cur.overlay, LENS, TOOLS_DIR, TOOL_ENV, ARCH: setup.device_gfx,
    }), { label: 'extractor:extract', phase: 'HeadKernel', schema: S({
    status: str, workload_json: str, steer: str,
  }, ['status', 'workload_json']) }, 60 * MIN);
  if (!ex || !says(ex.status, 'pass')) {
    return entry({ accepted: false, extract: ex,
      reason: ex ? `the task failed at baseline: ${ex.status}` : 'the extractor returned no result' });
  }
  const clock = await agentT('Run `date +%s` with Bash and return {"epoch": <that integer>}.',
    { label: 'clock', phase: 'HeadKernel', schema: S({ epoch: num }, ['epoch']) }, 5 * MIN);
  const windowS = Math.round(KERNEL_HOURS * 3600);
  // A cap, not a fixed window: no round starts after the deadline, and a no_stop_s past the window
  // leaves the TechLead free to end early (e2e_workflow.js's laneDeadlineArgs does the same).
  const deadlineEpoch = clock && clock.epoch ? Math.round(clock.epoch) + windowS : null;
  if (deadlineEpoch === null) log('kernel lane: the clock is unreadable, so the lane runs without a deadline');
  const deadline = deadlineEpoch ? { deadline_epoch: deadlineEpoch, no_stop_s: windowS + 86400 } : {};
  let kr = null;
  let kerr = '';
  try {
    kr = await workflow({ scriptPath: KERNEL_LANE_SCRIPT }, {
      kernel_path: taskDir, workflow_dir: KERNEL_WF_DIR, mode: 'optimize', target_language: 'hip',
      budget: KERNEL_BUDGET, exp_root: `${EVAL_DIR}/kernel`, workload_spec_path: ex.workload_json,
      task: ex.steer || '', ...deadline,
    });
  } catch (e) {
    kerr = String(e).slice(0, 500);
    log(`kernel lane failed to run: ${kerr}`);
  }
  const ksp = kr ? Number(kr.final_speedup || kr.final_geomean || 0) : 0;
  const kernel = kr && { eval_dir: kr.eval_dir, final_speedup: ksp, validation_status: kr.validation_status,
    reason: kr.reason || '' };
  log(`kernel lane ${KERNEL_OP.op}: isolated speedup ${ksp}x, ${kr ? kr.validation_status : 'no result'}`);
  // Only a result the lane's own Director re-measured and accepted counts, the bar kernel_lane.js
  // itself sets before it writes to the experience store.
  if (!kr || !(ksp > 1.0) || !kr.final_patch || !says(kr.validation_status, 'accepted')) {
    return entry({ accepted: false, reason: kerr ? `kernel lane did not run: ${kerr}`
      : !kr ? 'kernel lane returned no result'
        : `no verified isolated speedup (${ksp}x, ${kr.validation_status})` +
          (kr.reason ? `: ${kr.reason}` : '') }, { kernel, deadline_epoch: deadlineEpoch });
  }

  phase('Integrate');
  const ig = await agentT(roleAgent('integrator', 'integrate',
    'Rebuild the kernel library with the kernel patch and re-tuned tables, then gate it.', {
      EVAL_DIR, LANE_DIR: laneDir, MODEL_PATH, BASE_PKG, CUR_PKG: cur.pkg,
      CUR_OVERLAY: cur.overlay, TASK_DIR: taskDir, FINAL_PATCH: kr.final_patch,
      KERNEL_SPEEDUP: ksp, NOISE_FROM, TOOLS_DIR, LENS, GEN, REPS, ROUNDS, TOOL_ENV,
      ARCH: setup.device_gfx, RETUNE_OPS: KERNEL_OP.retune.join(','),
    }), { label: 'integrator:integrate', phase: 'Integrate', schema: GATE_SCHEMA }, 240 * MIN);
  if (ig && ig.accepted) cur = { pkg: ig.pkg, overlay: ig.overlay, label: `kernel:${KERNEL_OP.op}` };
  log(`integrate ${KERNEL_OP.op}: ${ig ? (ig.accepted ? 'ACCEPTED' : `rejected (${ig.reason})`) : 'no result'}`);
  return entry(ig || { accepted: false, reason: 'the integrator returned no result' },
    { kernel, deadline_epoch: deadlineEpoch });
}

if (tuneOps.length) {
  history.push(await tuningLane());
} else {
  history.push({ lane: 'tuning', ops: [], shares, result: { accepted: false,
    reason: `not run: no tunable op holds ${pct(MIN_SHARE)} of a phase` } });
}
if (kernelShare >= MIN_SHARE) {
  history.push(await kernelLane());
} else {
  const reason = `not run: ${KERNEL_OP.op} is ${pct(kernelShare)} of its phase, under ${pct(MIN_SHARE)}`;
  history.push({ lane: 'kernel', op: KERNEL_OP.op, share: kernelShare, result: { accepted: false, reason } });
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
