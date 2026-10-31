export const meta = {
  name: 'kernel-fusion',
  description: 'KernelFusion stage, run BEFORE the GEAK e2e workflow: capture a serving trace (GEAK profiler), pick the Top-K fusion patterns with a user-supplied analysis prompt, build a region UT and have an agent write each fused kernel against it, and apply the winners back through the GEAK e2e integrator A/B. Returns the accepted fused overlay, which the e2e workflow then uses as its baseline.',
  whenToUse: 'Started by interface/run_e2e.py before the e2e workflow unless switched off (handoff "kernel_fusion": false, GEAK_KERNEL_FUSION=0 or --no-kernel-fusion). Takes the same args as e2e_workflow.js plus kf_dir, kf_prompt and the optional kf_* knobs documented in kernel_fusion/README.md.',
  phases: [
    { title: 'Setup', detail: 'e2e Director builds the KF eval dir + baseline throughput (roles/director.md PHASE=setup)' },
    { title: 'Trace', detail: 'AgentX: kernel-extract skill sampling capture (points from the baseline log, one probe each); else roles/profiler.md; or kf_trace_dir' },
    { title: 'TopK', detail: 'fusion analyst: kf_trace_digest.py + the user prompt -> ranked Top-K fusion candidates (topk.json)' },
    { title: 'Author', detail: 'per recommended candidate + its fallback: region UT (AgentX: kernel-extract skill prepare; else roles/kernel_extractor.md), then one author agent writes the fused kernel against that UT' },
    { title: 'ApplyBack', detail: 'per slot, serially: e2e_integrator A/B (primary, else its fallback), accepted/stack overlays compound; director validate confirms the whole stack' },
  ],
};

// ---------------------------------------------------------------------------
// Args. Everything a GEAK role needs comes in under the SAME names e2e_workflow.js reads, because
// run_e2e.py builds both arg sets with one map_args(); only the kf_* keys are new.
// ---------------------------------------------------------------------------
const A = args || {};
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');   // e2e_workflow/
if (!WORKFLOW_DIR) throw new Error('args.workflow_dir is required (the e2e_workflow dir)');
const KF_DIR = String(A.kf_dir || '').replace(/\/+$/, '');
if (!KF_DIR) throw new Error('args.kf_dir is required (the KernelFusion eval dir)');
const KF_HOME = `${WORKFLOW_DIR}/kernel_fusion`;

const MODEL_PATH = String(A.model_path || '');
const LAUNCH_SCRIPT = String(A.launch_script || '');
const BACKEND = String(A.backend || 'sglang');
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map((s) => s.trim()).filter(Boolean);
const SERVING_TP = parseInt(A.tp != null ? A.tp : 1, 10);
const SERVING_GPU = String(A.serving_gpu != null ? A.serving_gpu
  : GPU_LIST.slice(0, Math.max(1, SERVING_TP)).join(',') || '0');
const WORKLOAD = { isl: parseInt(A.isl != null ? A.isl : 1024, 10), osl: parseInt(A.osl != null ? A.osl : 1024, 10),
  conc: parseInt(A.conc != null ? A.conc : 64, 10), shape_provenance: String(A.workload_shape_provenance || 'handoff_workload') };
const INIT_FLAGS = String(A.initial_extra_server_args || '');
const INIT_ENV = String(A.initial_extra_env || '');
// Operator constraints for every agent of this stage (e.g. paths it must not read).
const KF_TASK = String(A.kf_task || '');
// Simulated speculative acceptance (run_e2e.py passes the names it found); same rule and wording as e2e_workflow.js.
const CORRECTNESS_UNSET_ENV = String(A.correctness_unset_env || '').split(',')
  .map((s) => s.trim()).filter((s) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(s));
const CORRECTNESS_PROBE_ROLES = new Set(['e2e_integrator', 'director']);
function correctnessEnvBlock(role) {
  if (!CORRECTNESS_UNSET_ENV.length || !CORRECTNESS_PROBE_ROLES.has(role)) return '';
  const unset = CORRECTNESS_UNSET_ENV.map((v) => `-u ${v}`).join(' ');
  return `\n\n## CORRECTNESS PROBES RUN WITHOUT SIMULATED ACCEPTANCE\n` +
    `This stack serves with simulated speculative-decoding acceptance (${CORRECTNESS_UNSET_ENV.join(', ')}). ` +
    `It keeps throughput comparable, but what it serves is neither deterministic nor the model's own ` +
    `output: the unmodified reference does not reproduce itself, so a greedy-parity or task-accuracy probe ` +
    `run under it measures the simulator, not the change. For EVERY correctness probe, launch BOTH servers ` +
    `(the TRUE baseline and the candidate) with these variables removed, from the process env ` +
    `(\`env ${unset} ...\`) and from EXTRA_ENV, and change nothing else: same flags, same overlay, same ` +
    `TP/GPU. Throughput legs keep them, because those legs ARE the measurement, so a correctness server ` +
    `is never one of the throughput servers. Say in your notes that the probe ran with simulated ` +
    `acceptance off.`;
}

const INIT_OVERLAY = String(A.initial_overlay_pythonpath || '');
const EFFECTIVE_CONFIG_DIGEST = String(A.effective_config_digest || '');
const MEASUREMENT_MODE = String(A.measurement_mode || 'warm_server');
const PARITY_REPLICAS = parseInt(A.parity_replicas != null ? A.parity_replicas : 1, 10);
const SEARCH_REPLICAS = parseInt(A.search_replicas != null ? A.search_replicas : 1, 10);
const NOISE_BAND_DEFAULT = parseFloat(A.noise_band_pct != null ? A.noise_band_pct : 0.5);
const AB_FINISH_RETRIES = parseInt(A.ab_finish_retries != null ? A.ab_finish_retries : 2, 10);

// KernelFusion knobs.
const PROMPT = String(A.kf_prompt || `${KF_HOME}/default_prompt.md`);
const PROMPTS = PROMPT.split(',').map((p) => p.trim()).filter(Boolean);   // several files drive one analysis
const TRACE_DIR_IN = String(A.kf_trace_dir || '');          // existing traces: skip the capture
const WORKLOAD_SUMMARY = String(A.kf_workload_summary || ''); // per-point weights, when the caller has them
const TOPK_IN = String(A.kf_topk_json || '');                 // a finished analysis: reused, not re-run
const BASELINE_DIR_IN = String(A.kf_baseline_dir || '');     // a measured baseline of this same stack: reused, not re-measured
const SOURCE_DIRS = String(A.kf_source_dirs || '');           // framework source, when not importable here
const STEPS = new Set(String(A.kf_steps || 'trace,topk,author,apply').split(',').map((s) => s.trim()).filter(Boolean));
const want = (s) => STEPS.has(s);
const VALIDATION_MODE = String(A.kf_validation_mode || 'isolated_server');   // fresh servers: boot-to-boot noise
const VALIDATION_REPLICAS = parseInt(A.kf_validation_replicas != null ? A.kf_validation_replicas : 2, 10);
// AgentX trace replays use the kernel-extract skill for both trace capture and UT preparation, as
// e2e_workflow.js does (its KERNEL_EXTRACT_ON); other workloads keep the profiler / kernel_extractor roles.
const WORKLOAD_SPEC = A.workload_spec && typeof A.workload_spec === 'object' ? A.workload_spec : {};
const IS_AGENTX = String(A.workload_kind || WORKLOAD_SPEC.kind || '') === 'agentx_trace_replay';
const KERNEL_EXTRACT_DIR = `${WORKFLOW_DIR}/knowledge/kernel_extraction_skills/kernel-extract`;
const CAPTURE_PERSIST_POLICY = String(A.capture_persist_policy || 'structure_only');
const CAPTURE_STORAGE_ENV = `CAPTURE_BYTE_BUDGET=${A.capture_byte_budget || '8GiB'} ` +
  `CAPTURE_CASE_BYTE_LIMIT=${A.capture_case_byte_limit || '2GiB'} CAPTURE_PERSIST_POLICY=${CAPTURE_PERSIST_POLICY}`;
const AUTHOR_K = parseInt(A.kf_author_k != null ? A.kf_author_k : 3, 10);
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 7200000, 10);

// Stop STARTING author/apply work once 85% of the stage budget is gone; in-flight work finishes and the
// stage returns whatever was accepted. Inert without time_budget_s.
const TIME_BUDGET_MS = A.time_budget_s ? parseInt(A.time_budget_s, 10) * 1000 : null;
let DEADLINE_HIT = false;
if (TIME_BUDGET_MS && typeof setTimeout === 'function') {
  setTimeout(() => { DEADLINE_HIT = true; log('[kf] 85% of the stage budget spent: no new author/apply work starts.'); },
    Math.floor(TIME_BUDGET_MS * 0.85));
}

// Same deployment requirement e2e_workflow.js injects into every kernel task (derived the same way).
const GRAPH_REQ = ((BACKEND === 'vllm' || BACKEND === 'sglang') && !/enforce[-_]eager/i.test(INIT_FLAGS))
  ? ' DEPLOYMENT REQUIREMENT: the server captures decode into a CUDA/HIP graph, so the fused kernel runs inside ' +
    'that graph. It must be graph-capture-safe (no .item()/.cpu()/.tolist()/synchronize(), no Python branch on a ' +
    'GPU scalar, shape-stable, weight prep cached once by data_ptr) and its speedup must be intrinsic (fewer ' +
    'launches / less HBM traffic inside the captured region), never a per-call graph-replay wrapper.'
  : '';

// ---------------------------------------------------------------------------
// Agent plumbing: the e2e_workflow.js roleAgent/safeAgent contract, trimmed to what this stage uses.
// ---------------------------------------------------------------------------
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
const arrStr = { type: 'array', items: { type: 'string' } };
const cfg = (o) => Object.entries(o).map(([k, v]) => `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// Same text as e2e_workflow.js PROCESS_SAFETY (its final-phase section does not apply to this stage).
const PROCESS_SAFETY = `## PROCESS SAFETY (a violation can kill the caller's orchestrator, failing the whole task)
This container's PID 1 is the CALLER's orchestrator process, not yours, and it is NOT restartable.
NEVER run global or pattern-matched process cleanup: no \`pkill -f\` / \`pgrep -f ... | xargs kill\` /
\`killall\` / \`ps aux | grep ... | xargs kill\`, and never \`kill -- -PGID\` for a group you did not
create. A pattern as innocent as \`-f vllm\` matches the orchestrator's own command line and TERMs it.
Manage ONLY processes you started, by the pid you captured at launch. When a script of yours launches a
server, do NOT hand-roll the shutdown — source the shared teardown contract, which verifies process
identity (pid, pgid, /proc start time) before signalling anything:
    source ${WORKFLOW_DIR}/scripts/server_teardown.sh
    trap server_teardown EXIT
    \${SERVER_LAUNCH_PREFIX:-} <launch server> &   # own session => pgid == pid
    SERVER_PID=\$!; server_record_identity "\$SERVER_PID"

## LONG MEASUREMENTS OUTLIVE YOUR TOOL CALL — DETACH THEM FROM THE SHELL MANAGER
A serving measurement is longer than any bound your Bash tool can give it. One canonical trace-replay
leg is a 3600s window PLUS warmup and finalization (~70min wall), and a cold server boot alone is ~9min.
Your foreground bound is BASH_MAX_TIMEOUT_MS, and your tool's OWN background mode is worse: a shell the
tool manages for you is reaped on the manager's schedule, and the reap is a GROUP kill that takes the
server down with the client — you get a truncated log, an unloaded GPU, and no summary, from a leg that
was measuring perfectly. Do NOT run a bench through your tool's background/\`run_in_background\` mode.
Launch it so it is orphaned to init, OUTSIDE the manager's reach, then poll its log:
    cd "\$EVAL_DIR" && setsid nohup bash run_bench.sh > run_bench.out 2>&1 < /dev/null &
    # returns immediately; the leg's parent becomes PID 1 and no tool-call bound applies
Then poll in SEPARATE short calls (\`sleep 300; tail -5 run_bench.out\`) until the summary lands. Before
you conclude a leg died, check whether it is still alive (\`ps -o pid,stat,etime -p <pid>\`) and whether
the GPUs still hold weights — a reaped wrapper with a live leg underneath looks identical to a crash.

## INSIDE THIS WORKFLOW, ENDING YOUR TURN IS YOUR FINAL ANSWER
Ending your turn to wait for a background task or notification forces your structured result at once:
you are finalized mid-job, and the server and captures you are waiting on are lost with you. Wait only in
the foreground, in calls of at most ~9 minutes each (a loop of short sleeps + a check), re-issued until the
job is done. Profiler traces with Python stacks of a large TP model take many minutes to flush: keep polling
until every rank's file is complete, and only then tear the server down.

## THE GEAK REPO IS READ-ONLY
Never edit, add or delete anything under WORKFLOW_DIR's repository (scripts, skills, roles, tools). When a
GEAK tool does not handle this model (an unparsed log format, a missing case), work around it with your own
script under EVAL_DIR and say so in notes.

`;

function roleAgent(role, phase, intro, inputs) {
  const inall = { BACKEND, SERVING_TP, SERVING_GPU, MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST,
    ...(correctnessEnvBlock(role) ? { CORRECTNESS_UNSET_ENV: CORRECTNESS_UNSET_ENV.join(' ') } : {}), ...inputs };
  return `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
You are running inside the GEAK KernelFusion stage: EVAL_DIR is the KernelFusion eval dir, and everything
the role file says about EVAL_DIR applies to it unchanged.

## SERVING CONFIG INVARIANT
Every e2e serving launch (baseline, profile, integrate ref/cand) uses BACKEND=${BACKEND} TP=${SERVING_TP}
GPU=${SERVING_GPU} in the bench_e2e.sh env. GPU_IDS=${GPU_IDS} is only for single-GPU isolated work.

## Inputs
${cfg(inall)}

${KF_TASK ? `\n## Operator constraints\n${KF_TASK}\n` : ''}
Return ONLY the structured JSON the role file specifies.` + correctnessEnvBlock(role);
}

// The kernel-extract skill, invoked the way e2e_workflow.js roleAgent does for kernel_extractor on AgentX.
function skillAgent(phase, intro, inputs) {
  const inall = { BACKEND, SERVING_TP, SERVING_GPU, MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST, ...inputs,
    SKILL_ROOT: KERNEL_EXTRACT_DIR, SHARED_TOOLS: `${WORKFLOW_DIR}/scripts`, WORKFLOW_DIR,
    PARSE_PROFILE_STRICT_PHASES: '1', KERNEL_ENV_SKIP_ENUM_REAP: '1', CAPTURE_PERSIST_POLICY, CAPTURE_STORAGE_ENV };
  return `You are the kernel_extractor. PHASE=${phase}.
First Read ${KERNEL_EXTRACT_DIR}/SKILL.md and ${KERNEL_EXTRACT_DIR}/references/workflow-handoff.md.
Use the complete extraction procedure and its shared and skill-specific tools. The handoff reference supplies only workflow inputs and output packaging.
Do not load roles/kernel_extractor.md; it is the legacy procedure for other workloads.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
You are running inside the GEAK KernelFusion stage: EVAL_DIR is the KernelFusion eval dir.
Every serving launch uses BACKEND=${BACKEND} TP=${SERVING_TP} GPU=${SERVING_GPU}.

## Inputs
${cfg(inall)}
${KF_TASK ? `\n## Operator constraints\n${KF_TASK}\n` : ''}
Return ONLY the structured JSON requested above.`;
}

function bounded(prompt, opts) {
  const p = agent(PROCESS_SAFETY + prompt, opts);
  if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) return p;
  let to;
  const guard = new Promise((resolve) => { to = setTimeout(() => resolve(null), AGENT_TIMEOUT_MS); });
  return Promise.race([p.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }), guard]);
}

async function safeAgent(prompt, opts, tries = 3) {
  for (let i = 0; i < tries; i++) {
    try {
      const r = await bounded(prompt, opts);
      if (r) return r;
    } catch (e) { log(`agent[${opts.label}] attempt ${i + 1}/${tries}: ${String(e).slice(0, 160)}`); }
  }
  log(`agent[${opts.label}] gave no result after ${tries} tries.`);
  return null;
}

const SETUP_SCHEMA = obj({ eval_dir: { type: 'string' }, baseline_throughput_tok_s: { type: 'number' },
  noise_band_pct: { type: 'number' }, server_flags: { type: 'object', additionalProperties: true },
  server_env: { type: 'string' }, notes: { type: 'string' } }, ['eval_dir', 'baseline_throughput_tok_s']);
const PROFILE_SCHEMA = obj({ profile_topN_json: { type: 'string' }, top_kernels: arrObj, notes: { type: 'string' } },
  ['profile_topN_json']);
const BASELINE_WAIT_SCHEMA = obj({ status: { type: 'string', enum: ['done', 'running', 'dead'] },
  baseline_throughput_tok_s: { type: 'number' }, notes: { type: 'string' } }, ['status']);
const SKILL_TRACE_SCHEMA = obj({ trace_dir: { type: 'string' }, workload_json: { type: 'string' },
  sampling_points_json: { type: 'string' }, ranking_json: { type: 'string' }, points: arrObj, notes: { type: 'string' } },
  ['trace_dir', 'workload_json', 'points']);
const AGENTX_PREPARATION_SCHEMA = obj({ delivery_path: { type: 'string' },
  prepared: obj({ head_candidates: arrObj, kernel_candidates: arrObj, failed_tasks: arrObj, deferred_tasks: arrObj },
    ['head_candidates', 'kernel_candidates']) }, ['delivery_path', 'prepared']);
const TOPK_SCHEMA = obj({
  topk_json: { type: 'string' }, report_path: { type: 'string' }, digest_md: { type: 'string' },
  candidates: arrObj, recommended: arrStr, notes: { type: 'string' },
  report_md: { type: 'string' }, model_notes_md: { type: 'string' },
}, ['topk_json', 'candidates', 'recommended', 'report_md']);
const EXTRACT_SCHEMA = obj({ short_name: { type: 'string' }, editable: { type: 'boolean' }, task_dir: { type: 'string' },
  target_callable: { type: 'string' }, unittest_smoke: { type: 'string' }, candidate_bind: { type: 'object', additionalProperties: true },
  baseline_frozen: { type: 'boolean' },
  notes: { type: 'string' } }, ['editable', 'task_dir', 'unittest_smoke']);
const AUTHOR_SCHEMA = obj({ final_patch: { type: 'string' }, work_dir: { type: 'string' },
  overlay_dir: { type: 'string' }, banner: { type: 'string' },
  isolated_speedup: { type: 'number' }, correctness: { type: 'string' }, launches_after: { type: 'number' },
  ut_command: { type: 'string' }, notes: { type: 'string' } }, ['final_patch', 'isolated_speedup', 'correctness']);
const VALIDATE_SCHEMA = obj({ validation_status: { type: 'string' }, throughput_speedup: { type: 'number' },
  director_verified_throughput_tok_s: { type: 'number' }, output_parity: { type: 'string' }, notes: { type: 'string' } },
  ['validation_status']);
const INTEGRATE_SCHEMA = obj({ short_name: { type: 'string' }, e2e_throughput_tok_s: { type: 'number' },
  e2e_delta_pct: { type: 'number' }, ref_med: { type: 'number' }, cand_med: { type: 'number' },
  ab_complete: { type: 'boolean' }, output_parity: { type: 'string' }, parity_kind: { type: 'string' },
  gate: { type: 'string', enum: ['accepted', 'stack', 'rejected', 'incomplete'] },
  accepted_overlay: { type: 'string' }, reason: { type: 'string' } }, ['gate', 'e2e_throughput_tok_s']);

// A director that follows the detach rule may return while its baseline leg is still running, with no
// number yet. Waiting is then a job of its own: an agent that starts nothing and only polls, re-issued
// until the summary lands or the leg is gone.
async function awaitBaseline(dir) {
  for (let i = 1; i <= 6 && !DEADLINE_HIT; i++) {
    const w = await safeAgent(`You wait for one measurement and start nothing. A baseline leg (bench_e2e.sh) was ` +
      `launched detached under ${dir} (see ${dir}/logs, ${dir}/*.out and ${dir}/baseline). Poll in separate ` +
      `foreground calls of at most ~9 minutes each (a loop of sleep + check), for up to ~100 minutes in total, until ` +
      `${dir}/baseline/bench_summary.json exists or the leg is no longer alive (its pid / process group, and whether ` +
      `the GPUs still hold weights). Return {status: "done" with baseline_throughput_tok_s = that summary's ` +
      `throughput_tok_s_median, or "running" if it is still going when you stop, or "dead" if it exited without a ` +
      `summary, notes}.`, { phase: 'Setup', label: `kf:baseline wait ${i}`, schema: BASELINE_WAIT_SCHEMA });
    if (w && w.status === 'done' && w.baseline_throughput_tok_s > 0) return w.baseline_throughput_tok_s;
    if (w && w.status === 'dead') return 0;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Setup (GEAK director) -- needed whenever this stage touches a server or a GPU.
// ---------------------------------------------------------------------------
const needsServer = (want('trace') && !TRACE_DIR_IN) || want('author') || want('apply');
let EVAL_DIR = KF_DIR, baseTput = 0, noiseBand = NOISE_BAND_DEFAULT;
if (needsServer) {
  phase('Setup');
  const setup = await safeAgent(roleAgent('director', 'setup',
    'Build the isolated KernelFusion eval dir (use EVAL_DIR_OVERRIDE exactly) and record the baseline throughput.' +
    (BASELINE_DIR_IN ? ` The baseline of this exact stack (same flags, env, overlay, workload) was already measured in ` +
      `${BASELINE_DIR_IN}: do every setup step EXCEPT the baseline measurement -- launch no server and no bench; copy ` +
      `that dir to EVAL_DIR/baseline (cp -a) and return its bench_summary.json throughput_tok_s_median as ` +
      'baseline_throughput_tok_s.' : ''), {
      LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT: KF_DIR.replace(/\/[^/]*$/, ''), EVAL_DIR_OVERRIDE: KF_DIR,
      MODEL_NAME_HINT: MODEL_PATH.replace(/\/+$/, '').split('/').pop(), TASK: '', GPU_IDS, WORKLOAD,
      INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY: INIT_OVERLAY, MEASUREMENT_PURPOSE: 'parity',
      REPLICAS: PARITY_REPLICAS, SKILL_DIR: WORKFLOW_DIR,
    }), { phase: 'Setup', label: 'kf:director:setup', schema: SETUP_SCHEMA });
  if (!setup || !setup.eval_dir) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'setup failed', accepted: [], final_overlay: INIT_OVERLAY };
  }
  EVAL_DIR = setup.eval_dir;
  baseTput = setup.baseline_throughput_tok_s || 0;
  if (!(baseTput > 0)) baseTput = await awaitBaseline(EVAL_DIR);
  if (!(baseTput > 0) && want('trace') && !TRACE_DIR_IN) {
    // The trace step plans its sampling points from this baseline's serving log: without it there is
    // nothing to sample from. One more attempt, then the stage stops (the e2e workflow still runs).
    const again = await safeAgent(roleAgent('director', 'setup',
      `The eval dir ${EVAL_DIR} exists but its baseline was not measured. Reuse the dir and measure the baseline ` +
      '(step 5 of PHASE=setup) to completion.', {
        LAUNCH_SCRIPT, MODEL_PATH, EXP_ROOT: KF_DIR.replace(/\/[^/]*$/, ''), EVAL_DIR_OVERRIDE: EVAL_DIR,
        MODEL_NAME_HINT: MODEL_PATH.replace(/\/+$/, '').split('/').pop(), TASK: '', GPU_IDS, WORKLOAD,
        INIT_FLAGS, INIT_ENV, INIT_BASE_OVERLAY: INIT_OVERLAY, MEASUREMENT_PURPOSE: 'parity',
        REPLICAS: PARITY_REPLICAS, SKILL_DIR: WORKFLOW_DIR,
      }), { phase: 'Setup', label: 'kf:director:setup (baseline)', schema: SETUP_SCHEMA });
    baseTput = (again && again.baseline_throughput_tok_s) || 0;
    if (!(baseTput > 0)) baseTput = await awaitBaseline(EVAL_DIR);
    if (!(baseTput > 0)) {
      return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'baseline not measured (twice)',
        accepted: [], final_overlay: INIT_OVERLAY };
    }
  }
  noiseBand = setup.noise_band_pct || NOISE_BAND_DEFAULT;
  log(`[kf] setup: baseline ${baseTput} tok/s, noise band ${noiseBand}%`);
}

// ---------------------------------------------------------------------------
// Trace: on AgentX the kernel-extract skill's workload measurement and sampling capture (stages 1-2:
// sampling points planned from the baseline log, one bounded probe per point); otherwise the GEAK
// profiler. Or the caller's traces.
// ---------------------------------------------------------------------------
let TRACE_DIR = TRACE_DIR_IN;
let profileTopN = '';
let workloadJson = WORKLOAD_SUMMARY;
let rankingJson = '';
const skillInputs = () => ({
  EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, TASK: '',
  BASELINE_LOG: `${EVAL_DIR}/baseline/server.log`, BASELINE_BENCH_DIR: `${EVAL_DIR}/baseline`,
  BASELINE_SUMMARY: `${EVAL_DIR}/baseline/bench_summary.json`, EVIDENCE_DIR: `${EVAL_DIR}/kernel_extraction/round_kf`,
  BASELINE_THROUGHPUT: baseTput, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV, CURRENT_OVERLAY: INIT_OVERLAY,
  PREVIOUS_WORKLOAD_JSON: workloadJson, PREVIOUS_RANKING_JSON: rankingJson, SKILL_DIR: WORKFLOW_DIR,
});
if (want('trace') && !TRACE_DIR_IN && IS_AGENTX) {
  phase('Trace');
  const traceIntro = 'Run ONLY skill stage 1 and the workload-measurement and capture part of stage 2 for the ' +
    'completed baseline in the inputs: observed_workload.py -> plan_sampling_points.py -> serving_probe.py once per ' +
    'point (traces with record_shapes and Python stacks, steady state, verified) -> coverage_ledger.py, then write ' +
    'kernel_workload.json with each point\'s weight. Do NOT select targets, declare tasks, extract source or author UTs: ' +
    'a fusion analysis consumes these traces next. Return {trace_dir: the directory holding one sub-directory of ' +
    'per-rank traces per sampling point, workload_json, sampling_points_json, ranking_json, points: [{point_id, phase, ' +
    'trace_files, verified}], notes}.';
  let tr = await safeAgent(skillAgent('trace', traceIntro, skillInputs()),
    { phase: 'Trace', label: 'kf:kernel-extract trace', schema: SKILL_TRACE_SCHEMA });
  const verified = (t) => t && t.trace_dir && (t.points || []).some((p) => p && p.verified !== false);
  if (!verified(tr)) {
    log('[kf] sampling capture returned no verified trace; retrying once.');
    tr = await safeAgent(skillAgent('trace', 'A previous attempt returned no verified trace. Reuse what it saved, ' +
      'recapture each failed point once, and wait for the captures to finish. ' + traceIntro, skillInputs()),
      { phase: 'Trace', label: 'kf:kernel-extract trace (retry)', schema: SKILL_TRACE_SCHEMA });
  }
  if (!verified(tr)) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'sampling capture produced no verified trace (twice)',
      accepted: [], final_overlay: INIT_OVERLAY };
  }
  TRACE_DIR = tr.trace_dir;
  workloadJson = WORKLOAD_SUMMARY || tr.workload_json;
  rankingJson = tr.ranking_json || '';
} else if (want('trace') && !TRACE_DIR_IN) {
  phase('Trace');
  let prof = await safeAgent(roleAgent('profiler', 'baseline',
    'Capture a warm serving trace and emit the standardized Top-N. Keep the raw per-rank trace files on disk.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 0,
      OVERLAY_PYTHONPATH: INIT_OVERLAY, EXTRA_SERVER_ARGS: INIT_FLAGS, EXTRA_ENV: INIT_ENV, SKILL_DIR: WORKFLOW_DIR,
    }), { phase: 'Trace', label: 'kf:profiler', schema: PROFILE_SCHEMA });
  let got = prof && prof.profile_topN_json && (prof.top_kernels || []).length;
  if (!got) {
    // A profiler that returned without a trace (killed capture, unfinished wait) must not hand TopK an
    // empty dir: retry once, then stop the stage -- the e2e workflow still runs on the original stack.
    log('[kf] profiler returned no Top-N; retrying the capture once.');
    prof = await safeAgent(roleAgent('profiler', 'baseline',
      'A previous attempt returned before its trace existed. Capture the warm serving trace again, wait for it ' +
      'to finish, and emit the standardized Top-N. Keep the raw per-rank trace files on disk.', {
        EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], WORKLOAD, ROUND: 0,
        OVERLAY_PYTHONPATH: INIT_OVERLAY, EXTRA_SERVER_ARGS: INIT_FLAGS, EXTRA_ENV: INIT_ENV, SKILL_DIR: WORKFLOW_DIR,
      }), { phase: 'Trace', label: 'kf:profiler (retry)', schema: PROFILE_SCHEMA });
    got = prof && prof.profile_topN_json && (prof.top_kernels || []).length;
  }
  if (!got) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'trace capture produced no Top-N (twice)',
      accepted: [], final_overlay: INIT_OVERLAY };
  }
  profileTopN = prof.profile_topN_json;
  TRACE_DIR = `${EVAL_DIR}/profile`;
}

// ---------------------------------------------------------------------------
// TopK: deterministic digest + the user's analysis prompt.
// ---------------------------------------------------------------------------
let topk = null;
if (want('topk')) {
  phase('TopK');
  if (!TRACE_DIR) {
    return { eval_dir: KF_DIR, kf_dir: KF_DIR, status: 'error', reason: 'no trace: pass kf_trace_dir or enable the trace step',
      accepted: [], final_overlay: INIT_OVERLAY };
  }
  const OUT = `${EVAL_DIR}/analysis`;
  topk = TOPK_IN ? await safeAgent(`You load a finished KernelFusion analysis; you do not analyse anything.
Copy ${TOPK_IN} to ${OUT}/topk.json (mkdir -p ${OUT}; cp). Read it, and the REPORT.md / MODEL_NOTES.md next to it
when they exist. Return JSON: {"topk_json": "${OUT}/topk.json", "candidates": <its candidates, verbatim>,
"recommended": <its recommended ids, verbatim and in order>, "report_md": <REPORT.md content or "">,
"model_notes_md": <MODEL_NOTES.md content or "">, "notes": "reused ${TOPK_IN}"}.`,
    { phase: 'TopK', label: 'kf:topk (given)', schema: TOPK_SCHEMA })
  : await safeAgent(`You are the GEAK KernelFusion analyst. You choose which kernel-fusion patterns to implement.

## Your method is the user's prompt
Read every one of these files in full: ${PROMPTS.join(', ')}. Together they are the AUTHORITATIVE analysis
method, ranking rules, deliverable list, deployment and workload description for this task: follow them step
by step. A file may describe a whole run (launching servers, running GEAK, where outputs go): take from it the
deployment, workload, metric and constraints, but your job is only the analysis below. They may also name a
container, a working directory or a materials/ layout from the setting they were written in; those are NOT
this environment. Map them onto the inputs below instead:
- its traces / sampling points            -> the trace files under TRACE_DIR (any depth; one dir = one point)
- its workload summary / weights           -> WORKLOAD_SUMMARY when given; otherwise derive the weights from the
                                              baseline bench + server log under EVAL_DIR and say so
- its deploy/server args + env             -> SERVER_FLAGS / SERVER_ENV / LAUNCH_SCRIPT (and EVAL_DIR/baseline/server.log)
- its model config                         -> MODEL_PATH/config.json
- framework + operator-library source      -> ${SOURCE_DIRS ? `SOURCE_DIRS (${SOURCE_DIRS})` : `the installed packages (python3 -c "import ${BACKEND}; print(${BACKEND}.__file__)", aiter, ...)`}
- its out/ directory                       -> ${OUT}/
Where the prompt says not to write kernels, that is right for you too: the NEXT step of this stage authors
your recommended candidates with the GEAK kernel layer and A/B-tests them on the live server.

## Do this first (deterministic, reproducible)
    python3 ${KF_HOME}/kf_trace_digest.py --trace-dir "${TRACE_DIR}" --out "${OUT}/digest"
    # optional per-kernel Top-N of any one trace file:  python3 ${WORKFLOW_DIR}/scripts/parse_profile.py --torch-trace <file> --top 40 --out ${OUT}/topn_<point>
${OUT}/digest/digest.md has, per trace and per step kind: steps, busy/span/idle per step, kernels/step, short-kernel
share, the per-layer repeating block, the top kernels, and seq_*.tsv (the median step's full ordered kernel
sequence with gaps and, for eager launches, the launching cpu_op + framework frame). Use these numbers; quote
them in your report. Graph-replayed steps have no launch frames: map those kernels to source by name.

This environment does not let you write .md report files: return the full report the prompt asks for as markdown
in report_md, and the model / deployment notes in model_notes_md; the stage writes them to ${OUT}/REPORT.md and
${OUT}/MODEL_NOTES.md.

## Machine-readable result (in addition to everything the prompt asks for)
Write ${OUT}/topk.json: {"candidates": [...], "recommended": [ids in the order to implement]}, every candidate:
  id (short, [A-Za-z0-9_]), rank, level ("L1".."L4"), region (one line), kernels (device kernel names, in order),
  phases (step kinds it runs in), occurrences {per_layer, per_step, layers}, m_range,
  seam {target_callable: "pkg.module:Class.method or pkg.module:function" -- ONE Python callable whose body
        covers the whole region, so the region can be rebound to a fused implementation; source_file; lines},
  inputs_outputs (tensors that cross the region boundary, with shapes/dtypes), launches_before, launches_after,
  region_pct_device_time (weighted), est_gain_pct_upper, est_gain_pct_expected (weighted e2e %),
  difficulty (1-5), bitwise_exact (bool), exclusive_with [ids], language ("triton" unless you have a reason),
  fallback (id of the smaller-scope candidate to implement instead if this one fails, or null),
  fused_design (how the fused kernel works: tiling, what stays in registers/LDS, what it no longer writes to HBM).
"recommended" = the non-conflicting set the prompt's ranking recommends, best first. A candidate whose region
cannot be reached through one Python seam can still be ranked, but must not be recommended.

## Inputs
${cfg({ EVAL_DIR, TRACE_DIR, WORKLOAD_SUMMARY: workloadJson || '(none)', PROFILE_TOPN: profileTopN || '(none)',
    MODEL_PATH, BACKEND, SERVING_TP, SERVER_FLAGS: INIT_FLAGS, SERVER_ENV: INIT_ENV, LAUNCH_SCRIPT: LAUNCH_SCRIPT || '(none)',
    WORKLOAD, OVERLAY: INIT_OVERLAY || '(none)' })}

${KF_TASK ? `## Operator constraints\n${KF_TASK}\n\n` : ''}Return JSON: {"topk_json": "${OUT}/topk.json", "report_path": "${OUT}/REPORT.md", "report_md": "<full report>",
"model_notes_md": "<model notes>", "digest_md": "${OUT}/digest/digest.md", "candidates": <topk.json candidates>,
"recommended": <ids>, "notes": "..."}`,
  { phase: 'TopK', label: 'kf:analyst', schema: TOPK_SCHEMA });
  log(`[kf] topk: ${topk ? topk.candidates.length : 0} candidates, recommended ${topk ? JSON.stringify(topk.recommended) : '[]'}`);
}

const result = {
  eval_dir: KF_DIR, kf_dir: KF_DIR, kf_eval_dir: EVAL_DIR, trace_dir: TRACE_DIR, prompt: PROMPT,
  topk_json: topk ? topk.topk_json : '', analysis_report: topk ? `${EVAL_DIR}/analysis/REPORT.md` : '',
  analysis_files: topk ? { [`${EVAL_DIR}/analysis/REPORT.md`]: topk.report_md || '',
    [`${EVAL_DIR}/analysis/MODEL_NOTES.md`]: topk.model_notes_md || '' } : {},
  baseline_throughput_tok_s: baseTput, final_throughput_tok_s: baseTput, throughput_speedup: 1,
  final_overlay: INIT_OVERLAY, final_flags: INIT_FLAGS, final_env: INIT_ENV,
  accepted: [], rejected: [], status: 'analysis_only',
};
if (!topk || !(want('author') || want('apply'))) {
  if (!topk && want('topk')) { result.status = 'error'; result.reason = 'analyst returned nothing'; }
  return result;
}
// ---------------------------------------------------------------------------
// Author: one slot per recommended candidate. A slot also authors its FALLBACK -- the smaller-scope
// pattern the prompt's ranking names for when the recommended one does not pan out (the analyst's
// `fallback`, else the lowest-difficulty exclusive alternative of a difficulty>=3 candidate). Each
// job: region unittest (GEAK kernel-extract skill / kernel_extractor) -> fused kernel (one author agent).
// ---------------------------------------------------------------------------
phase('Author');
const byId = new Map((topk.candidates || []).map((c) => [String(c.id), c]));
const seamOf = (c) => (c && c.seam && c.seam.target_callable) || '';
function fallbackOf(c, taken) {
  const named = c.fallback != null ? byId.get(String(c.fallback)) : null;
  if (named && seamOf(named) && !taken.has(String(named.id))) return named;
  if (!((c.difficulty || 0) >= 3)) return null;
  const alts = (c.exclusive_with || []).map((id) => byId.get(String(id)))
    .filter((a) => a && seamOf(a) && !taken.has(String(a.id)) && (a.difficulty || 9) < (c.difficulty || 0))
    .sort((a, b) => (b.est_gain_pct_expected || 0) - (a.est_gain_pct_expected || 0));
  return alts[0] || null;
}
const primaries = (topk.recommended || []).map((id) => byId.get(String(id))).filter(Boolean).slice(0, AUTHOR_K);
const taken = new Set(primaries.map((c) => String(c.id)));
const slots = primaries.map((c) => {
  const fb = fallbackOf(c, taken);
  if (fb) taken.add(String(fb.id));
  return { primary: c, fallback: fb };
});
const jobs = slots.flatMap((s) => [s.primary, s.fallback].filter(Boolean));
log(`[kf] authoring ${jobs.length} job(s): ${slots.map((s) => s.primary.id + (s.fallback ? `(fallback ${s.fallback.id})` : '')).join(', ')}`);

const shortOf = (c) => String(c.id).replace(/[^A-Za-z0-9_]+/g, '_');

// AgentX: the kernel-extract skill prepares the fusion UTs (PHASE=prepare), ONE invocation PER REGION, in
// parallel, each on its own GPU and evidence dir with the whole stage budget, reusing the trace step's
// evidence. A region whose UT does not pass gets one more attempt that starts from the first one's notes.
// Same gate as e2e_workflow.js preparedUtDefect: a UT without a passing smoke and a frozen baseline is unusable.
let preparedById = new Map();
// A fusion UT must let the candidate REPLACE the region: the per-kernel UT contract (every member kernel
// fires, the split is pinned, the timer launches the members directly) makes any fusion fail its gates.
const FUSION_UT_CONTRACT = 'FUSION UT CONTRACT -- this task exists so that the next step can REPLACE the whole ' +
  'region with fewer (usually one) kernel launches, so the UT must not pin the current kernel structure: ' +
  '(1) kernel_src/ holds an editable copy of the seam callable (the host-side function whose body launches the ' +
  'region) together with the kernels it launches; the case runner and the timer call that seam callable from ' +
  'kernel_src with the recorded inputs -- never the member kernels directly -- so a candidate may change how many ' +
  'kernels the region launches, their grid/split and their intermediates. ' +
  '(2) Correctness compares only the region\'s outputs at the seam boundary (and any declared in-place mutation) ' +
  'with the frozen live baseline; intermediates between member kernels are not checked. ' +
  '(3) Execution/binding proof that every member kernel fires is required for the BASELINE leg only; for the ' +
  'candidate leg require only that the kernel_src seam implementation ran, never that the original member ' +
  'kernels still launch or keep their count/split. ' +
  '(4) The negative control mutates the OUTPUT of the kernel_src seam implementation (e.g. perturb what the host ' +
  'function returns or writes), so it still triggers after the members are fused away. ' +
  '(5) Time one call of the whole seam per case (launch overhead included) -- the win of a fusion is fewer ' +
  'launches and fewer HBM round-trips. Record "fusion_contract": true in meta.json. ';
const PREPARE_INTRO = 'Explicit user scope (overrides the default selection): build exactly ONE task for the single ' +
  'entry of FUSION_SCOPE, named by its `task`. The entry is a FUSION REGION: declare its device_kernels as one joint ' +
  'boundary (joint_ut.phases) owned by that task, reached through the entry\'s seam (or the narrowest enclosing live ' +
  'callable; record why). The candidate leg will rebind that callable to one fused implementation, so the oracle is ' +
  'taken at the region boundary. ' + FUSION_UT_CONTRACT + 'Reuse the trace step\'s evidence (PREVIOUS_WORKLOAD_JSON / PREVIOUS_RANKING_JSON and ' +
  'the traces under TRACE_DIR); do not launch a new full replay. This region is your ONLY job and the time in ' +
  'REMAINING_MIN is all yours: do not defer or stop early for budget. When an input is hard to reconstruct (quantized ' +
  'checkpoint operands, split/partial buffers, library-op wrappers), build it the way the live path does -- load the ' +
  'real tensors from MODEL_PATH, call the live producer, or bind below the wrapper -- and verify by running it. ' +
  'Other regions are prepared in parallel on the other GPUs: use only GPU_ID for UT work and write only under ' +
  'EVIDENCE_DIR and your task dir. Complete source extraction, UT authoring and GPU validation, keep a failure in ' +
  'the delivery with its concrete blocker, pack with `agentx_kernel_preparation.mjs --emit-prepared` and return ONLY ' +
  '{delivery_path, prepared} (prepared = that command\'s stdout VERBATIM). Do not optimize or integrate.';
async function prepareOne(c, i, previous) {
  const t = shortOf(c);
  const scope = [{ task: t, fusion_level: c.level, region: c.region, device_kernels: c.kernels || [],
    phases: c.phases || [], seam: c.seam || {}, inputs_outputs: c.inputs_outputs || '', m_range: c.m_range || '' }];
  const intro = previous
    ? `A previous attempt at this region did not produce a passing UT. Its outcome: ${previous}. Reuse what it saved ` +
      'under EVIDENCE_DIR, fix that blocker (do not just restate it), and finish. ' + PREPARE_INTRO
    : PREPARE_INTRO;
  const prep = await safeAgent(skillAgent('prepare', intro, { ...skillInputs(), GPU_ID: GPU_LIST[i % GPU_LIST.length],
    EVIDENCE_DIR: `${EVAL_DIR}/kernel_extraction/round_kf_${t}`, TRACE_DIR, FUSION_SCOPE: scope,
    // one agent call is capped at AGENT_TIMEOUT_MS (workflow scripts have no clock)
    SOURCE_DELIVERY_REQUIRED: true, REMAINING_MIN: AGENT_TIMEOUT_MS > 0 ? Math.floor(AGENT_TIMEOUT_MS / 60000) - 10 : 240 }),
    { phase: 'Author', label: `kf:kernel-extract prepare ${t}${previous ? ' (retry)' : ''}`, schema: AGENTX_PREPARATION_SCHEMA });
  const p = (prep && prep.prepared) || {};
  const all = [...(p.head_candidates || []), ...(p.kernel_candidates || [])];
  for (const k of all) {
    const r = k && k.prepared_ut && k.prepared_ut.result;
    if (!r || typeof r.task_dir !== 'string' || !r.task_dir.trim()) continue;
    const ok = r.editable !== false && r.baseline_frozen === true &&
      (k.prepared_ut.phase === 'extract' ? r.unittest_smoke : r.smoke) === 'pass';
    if (ok || all.length === 1) return { ...r, ok };
  }
  const why = [...(p.failed_tasks || []), ...(p.deferred_tasks || [])].map((f) => f && (f.reason || f.notes || f.status))
    .filter(Boolean).join('; ');
  return { ok: false, notes: why || (prep ? 'no task in the delivery' : 'no result') };
}
if (IS_AGENTX && jobs.length && !DEADLINE_HIT) {
  const res = await parallel(jobs.map((c, i) => async () => {
    let r = await prepareOne(c, i, null);
    if (!(r && r.ok) && !DEADLINE_HIT) r = await prepareOne(c, i, String((r && r.notes) || 'no result').slice(0, 1500));
    return [shortOf(c), r];
  }));
  for (const x of res) if (x) preparedById.set(x[0], x[1]);
  log(`[kf] kernel-extract prepared ${[...preparedById.values()].filter((r) => r && r.ok).length}/${jobs.length} fusion UT(s)`);
}

async function authorOne(c, i) {
  if (DEADLINE_HIT) return { c, skip: 'stage deadline' };
  const gpu = GPU_LIST[i % GPU_LIST.length];
  const short = shortOf(c);
  const ext = IS_AGENTX ? (preparedById.get(short) || null) : await safeAgent(roleAgent('kernel_extractor', 'extract',
    'Build the immutable unittest task dir for a FUSION REGION, not a single kernel. The baseline leg is the ' +
    'live, unfused region reached through ONE Python seam (KERNEL.target_callable, a function whose body launches ' +
    'every kernel in KERNEL.device_kernels); candidate_bind rebinds that seam, so the candidate leg runs the fused ' +
    'implementation from kernel_src/. Record I/O at the seam boundary (KERNEL.inputs_outputs) for the oracle. If ' +
    'the given seam does not cover the region, use the narrowest enclosing callable that does and say so in notes; ' +
    'if none exists return editable:false with notes "no_rebind_seam". ' + FUSION_UT_CONTRACT, {
      EVAL_DIR, MODEL_PATH, GPU_ID: gpu, WORKLOAD, SKILL_DIR: WORKFLOW_DIR,
      KERNEL: { short_name: short, classification: 'fusion', fusion_level: c.level, region: c.region,
        device_kernel: (c.kernels || [])[0] || '', device_kernels: c.kernels || [],
        target_callable: seamOf(c), source_hint: (c.seam || {}).source_file || '',
        seam_candidates: [seamOf(c)].filter(Boolean), inputs_outputs: c.inputs_outputs || '',
        phases: c.phases || [], m_range: c.m_range || '', candidate_backends: [c.language || 'triton'] },
      CURRENT_OVERLAY: INIT_OVERLAY, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV,
    }), { phase: 'Author', label: `kf:extract ${short}`, schema: EXTRACT_SCHEMA });
  // Same guard as e2e extractWithBaseline: without a frozen live baseline the unittest would time the
  // fused kernel against its own scaffold.
  const extOk = IS_AGENTX ? ext && ext.ok
    : ext && ext.editable !== false && ext.unittest_smoke === 'pass' && ext.task_dir && ext.baseline_frozen !== false;
  if (!extOk) {
    return { c, skip: `extraction failed (${ext ? ext.notes || ext.unittest_smoke : 'none'})` };
  }
  const lang = c.language || 'triton';
  // The fused kernel is written by one agent from a plain instruction (not the GEAK kernel layer): the region,
  // the UT it must pass and the goal. It works on a copy of the task dir and hands back a diff for apply back.
  const work = `${EVAL_DIR}/kernels/${short}`;
  const au = await safeAgent(`You write ONE fused ${lang} kernel that replaces a whole kernel region of a live LLM serving stack.

## The region
${cfg({ id: c.id, level: c.level, region: c.region, kernels_in_order: c.kernels || [], phases: c.phases || [],
    m_range: c.m_range || '', seam: c.seam || {}, inputs_outputs: c.inputs_outputs || '',
    launches_before: c.launches_before || (c.kernels || []).length, launches_after: c.launches_after || 1,
    suggested_design: c.fused_design || '', bitwise_exact_expected: !!c.bitwise_exact })}

## The unit test
TASK_DIR=${ext.task_dir} is a prepared, GPU-validated unit test of this region at its seam
(${ext.target_callable || seamOf(c)}): kernel_src/ holds the editable copy of the seam callable and the kernels it
launches, cases.py the recorded serving cases, unittest.py the correctness check against the frozen live baseline
and the timing. Read its meta.json / README / run/ to see how to run it.

## Do
cp -a TASK_DIR to WORK_DIR=${work}/work and work only there (never edit TASK_DIR). Use GPU ${gpu} only. Design,
iterate and measure as you see fit, and deliver three things:
1. The fused kernel: a self-contained module under ${work}/kernel/ that replaces the region (fewer launches, usually
   one), with the region's outputs unchanged${c.bitwise_exact ? ' (bit-exact)' : ''}.
2. Its unit test: the fused version against the stock region on the recorded serving shapes (the prepared UT in
   TASK_DIR, with kernel_src/ pointed at your kernel, or your own test calling the live stock functions), checking
   every output of the region, and timing both sides with CUDA/HIP-graph replay (as the server runs decode).
3. The overlay that puts it in the server: ${work}/overlay/sitecustomize.py (the dir goes first on the server's
   PYTHONPATH). Patch the seam lazily -- after its module is first imported, never by importing it eagerly -- fall
   back to the stock path whenever the inputs do not match what the kernel supports, and print
   "[kf-${short}] ENGAGED" to stderr once, the first time the patched path actually runs. It must survive graph
   capture.${GRAPH_REQ}

## Return
Write the diff of WORK_DIR against TASK_DIR (diff -ruN) plus the kernel and overlay files to
${work}/final_patch.diff and return JSON {"final_patch": "${work}/final_patch.diff" or "" if nothing correct beat the
baseline, "work_dir": "${work}/work", "overlay_dir": "${work}/overlay", "banner": "[kf-${short}] ENGAGED",
"isolated_speedup": <stock region time / fused time, graph replay, weighted over the recorded cases>, "correctness":
"pass" | "fail", "launches_after": <kernel launches of the fused region>, "ut_command": "<how you ran the test>",
"notes": "<design and per-case measurements>"}.`
    + (KF_TASK ? `\n\n## Operator constraints\n${KF_TASK}` : ''),
    { phase: 'Author', label: `kf:author ${short}`, schema: AUTHOR_SCHEMA });
  const kl = au ? { eval_dir: work, final_patch: au.correctness === 'pass' ? au.final_patch || '' : '',
    overlay_dir: au.overlay_dir || '', banner: au.banner || '',
    final_geomean: au.isolated_speedup || 0, validation_status: au.correctness, reason: au.notes || '' }
    : { validation_status: 'error', reason: 'author agent returned nothing' };
  const iso = kl ? (kl.final_weighted != null ? kl.final_weighted : kl.final_geomean) : 0;
  return { c, ext, kl, lang, iso };
}
const authored = await parallel(jobs.map((c, i) => () => authorOne(c, i)));
const authoredById = new Map(authored.filter(Boolean).map((r) => [String(r.c.id), r]));
const usable = (r) => r && !r.skip && r.kl && r.kl.authored !== false && r.iso > 1.0 && r.kl.final_patch;
for (const r of authored) {
  if (r && !usable(r)) {
    const why = r.skip || `no isolated win (${r.iso || 0}x, ${r.kl ? r.kl.reason || r.kl.validation_status || '' : 'none'})`;
    log(`[kf] ${r.c.id}: ${why}`);
    result.rejected.push({ id: r.c.id, stage: r.skip ? 'extract' : 'author', reason: why, isolated_speedup: r.iso || 0 });
  }
}

// ---------------------------------------------------------------------------
// ApplyBack: per slot, serially, GEAK e2e_integrator A/B on the compounding overlay -- the primary
// first, its fallback if the primary has no isolated win or does not pass. 'accepted' and 'stack'
// (engaged, parity ok, not slower) both carry forward, exactly as in the e2e workflow; the GEAK
// director's validate then measures the WHOLE stack against the original baseline, and only a
// validated_win is handed to the main workflow.
// ---------------------------------------------------------------------------
phase('ApplyBack');
let curOverlay = INIT_OVERLAY, curTput = baseTput;
async function integrateOne(r) {
  const c = r.c;
  const inputs = {
    EVAL_DIR, MODEL_PATH, GPU_ID: SERVING_GPU, WORKLOAD, NOISE_BAND_PCT: noiseBand,
    MEASUREMENT_PURPOSE: 'search', REPLICAS: SEARCH_REPLICAS, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_RESULT: { short_name: r.ext.short_name || c.id, task_dir: r.ext.task_dir, op_kind: 'fusion',
      winner_kind: 'authored', winner_backend: r.lang, authored_language: r.lang,
      authored_kernel_eval_dir: r.kl.eval_dir || '', target_callable: r.ext.target_callable || seamOf(c),
      code_patch: r.kl.final_patch, final_patch: r.kl.final_patch, verified_isolated_speedup: r.iso,
      overlay_dir: r.kl.overlay_dir || '', engaged_banner: r.kl.banner || '',
      pct_gpu_time: c.region_pct_device_time || 0, live_call_seam: seamOf(c),
      parity_note: c.bitwise_exact ? 'expected_exact' : 'expected_close' },
    CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: INIT_FLAGS, CURRENT_ENV: INIT_ENV, CURRENT_THROUGHPUT: curTput,
      };
  const ready = r.kl.overlay_dir ? ` KERNEL_RESULT.overlay_dir is a READY overlay for this fused kernel (its ` +
    'sitecustomize.py patches the seam lazily and falls back to the stock path; the kernel module sits beside it): ' +
    'build the CAND overlay as CURRENT_OVERLAY plus that directory\'s files instead of re-materializing the kernel ' +
    'from the patch -- when both carry a sitecustomize.py, keep both by renaming them and writing one ' +
    'sitecustomize.py that imports each -- and confirm KERNEL_RESULT.engaged_banner in the CAND server log.' : '';
  let integ = await safeAgent(roleAgent('e2e_integrator', 'integrate',
    'Overlay this fused kernel back onto the live server and gate it on the e2e A/B.' + ready, inputs),
    { phase: 'ApplyBack', label: `kf:integrate ${c.id}`, schema: INTEGRATE_SCHEMA });
  for (let t = 1; t <= AB_FINISH_RETRIES && !(integ && integ.gate !== 'incomplete' && integ.ab_complete !== false); t++) {
    integ = await safeAgent(roleAgent('e2e_integrator', 'integrate',
      'FINISH this e2e A/B: reuse a reference leg already on disk if usable, run the missing leg, then return the gate.',
      { ...inputs, RESUME_AB: true }), { phase: 'ApplyBack', label: `kf:integrate ${c.id} (finish ${t})`, schema: INTEGRATE_SCHEMA });
  }
  return integ;
}
for (const s of slots) {
  if (!want('apply')) break;
  for (const c of [s.primary, s.fallback].filter(Boolean)) {
    const r = authoredById.get(String(c.id));
    if (!usable(r)) continue;
    if (DEADLINE_HIT) { result.rejected.push({ id: c.id, stage: 'apply', reason: 'stage deadline', isolated_speedup: r.iso }); continue; }
    const integ = await integrateOne(r);
    const rec = { id: c.id, level: c.level, slot: s.primary.id, isolated_speedup: r.iso,
      gate: integ ? integ.gate : 'incomplete', e2e_delta_pct: integ ? integ.e2e_delta_pct : null,
      ref_med: integ ? integ.ref_med : null, cand_med: integ ? integ.cand_med : null,
      parity: integ ? integ.output_parity : '', reason: integ ? integ.reason || '' : 'no integrator result' };
    if (integ && (integ.gate === 'accepted' || integ.gate === 'stack') && integ.ab_complete !== false && integ.accepted_overlay) {
      curOverlay = integ.accepted_overlay;
      curTput = integ.e2e_throughput_tok_s || curTput;
      result.accepted.push({ ...rec, overlay: curOverlay });
      log(`[kf] ${c.id}: carried (${rec.gate}, ${rec.e2e_delta_pct}%)`);
      break;                                   // slot filled; its fallback is not needed
    }
    result.rejected.push({ ...rec, stage: 'apply' });
    log(`[kf] ${c.id}: not applied (${rec.gate}: ${rec.reason})`);
  }
}
if (!want('apply')) {
  for (const r of authored) if (usable(r)) result.rejected.push({ id: r.c.id, stage: 'apply', reason: 'apply step disabled', isolated_speedup: r.iso });
}

// Combined confirmation: the stack vs the TRUE baseline, fresh servers, GEAK director validate.
if (result.accepted.length) {
  phase('ApplyBack');
  const v = await safeAgent(roleAgent('director', 'validate',
    'Independently re-measure the KernelFusion stack against the true baseline + output parity and arbitrate. ' +
    'This stage has no architect/final report: skip report reconciliation (step 7), and APPLY_TO_ORIGINAL=false.', {
      EVAL_DIR, MODEL_PATH, GPU_ID: GPU_LIST[0], BASELINE_THROUGHPUT: baseTput, NOISE_BAND_PCT: noiseBand,
      BASELINE_OVERLAY: INIT_OVERLAY, FINAL_OVERLAY: curOverlay, FINAL_FLAGS: { flags: INIT_FLAGS, env: INIT_ENV },
      CLAIMED_THROUGHPUT: curTput, WORKLOAD, APPLY_TO_ORIGINAL: 'false',
      MEASUREMENT_MODE: VALIDATION_MODE, MEASUREMENT_PURPOSE: 'validation', REPLICAS: VALIDATION_REPLICAS,
      SKILL_DIR: WORKFLOW_DIR,     }), { phase: 'ApplyBack', label: 'kf:director:validate', schema: VALIDATE_SCHEMA });
  result.validation = v || { validation_status: 'error', notes: 'validate returned nothing' };
  if (v && v.validation_status === 'validated_win' && v.throughput_speedup > 1) {
    result.final_overlay = curOverlay;
    result.final_throughput_tok_s = v.director_verified_throughput_tok_s || curTput;
    result.throughput_speedup = v.throughput_speedup;
    result.status = 'accepted';
    return result;
  }
  log(`[kf] combined stack not confirmed (${result.validation.validation_status}, x${result.validation.throughput_speedup || '?'}); main workflow keeps the original stack.`);
}
result.status = 'no_gain';
return result;
