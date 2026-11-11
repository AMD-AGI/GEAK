// e2e warm start: identity formatting + the prompt hook. These `let`s stay at their off-value
// (null / '') unless Module A runs, and every consumer keys off them.
let KB_DIMS = null;        // the deployment dimensions the Director established during preflight
let KB_REF_DIR = '';       // where Module A left its references; '' when it never ran
let KB_CACHE_DIR = '';     // where Module A materialized the records' FILES (patches, tuned tables)
let KB_READ_PLANE = '';    // which plane answered the read (not implied by `both`)
// Everything this run recalled, report-ready: the address asked, what came back, what re-measuring
// it here produced, and whether it was accepted (fed to the Report role). `asked` is filled even
// with zero candidates: a miss and a never-recorded page are the same 404, so the ladder asked is
// the finding.
let KB_RECALL = { e2e: null, kernel: [] };

// One collection point for the KERNEL plane's recall: every lane return (its `warm_start` block,
// kernel_lane.js) passes through here, so the report shows stored patches a lane adopted.
function noteKernelKB(r, label) {
  // Absorb the nested run's agent timeline (llm_timeline) for the ledger's nested-run accounting.
  if (r && r.llm_timeline) LLM_TL.nested.push(r.llm_timeline);
  const w = r && r.warm_start;
  if (w && typeof w === 'object') {
    KB_RECALL.kernel.push({
      lane: String(label || ''),
      slug: String(w.slug || ''),
      read_reason: String(w.read_reason || ''), match_tier: String(w.match_tier || ''),
      filtered: w.filtered || null,
      candidates: Array.isArray(w.candidates) ? w.candidates.length : 0,
      adopted: !!w.adopted,
      adopted_speedup: w.adopted ? (w.adopted_speedup != null ? w.adopted_speedup : null) : null,
      // Carry the split, not just the geomean: an adopted patch with no own rounds flatters the lane.
      incremental_speedup: w.incremental_speedup != null ? w.incremental_speedup : null,
      rounds_committed: w.rounds_committed != null ? w.rounds_committed : null,
      no_rounds_after_adopt: !!w.no_rounds_after_adopt,
      final_geomean: r.final_geomean != null ? r.final_geomean : null,
      validation_status: String(r.validation_status || ''),
    });
  }
  return r;
}

const shq = (s) => "'" + String(s == null ? '' : s).replace(/'/g, "'\\''") + "'";

// The ONE place e2e KB identity argv is formatted, for both reader (Module A) and writer (Module B):
// a one-segment disagreement silently addresses a different page (see kb/identity.py).
function kbIdentityFlags() {
  if (!KB_DIMS) return '';
  return [`--model ${shq(KB_DIMS.model)}`, `--gfx ${shq(KB_DIMS.gfx)}`,
    `--framework ${shq(BACKEND)}`, `--framework-version ${shq(KB_DIMS.framework_version)}`,
    `--precision ${shq(KB_DIMS.precision)}`, `--rocm-version ${shq(KB_DIMS.rocm_version)}`,
    `--tp ${SERVING_TP}`, `--ep ${SERVING_EP}`,
    // The declared kind, never the shape provenance (whose fallback values would mis-file
    // degraded AgentX runs).
    `--workload-kind ${shq(WORKLOAD_KIND)}`,
    `--isl ${ISL}`, `--osl ${OSL}`, `--conc ${CONC}`].join(' ');
}

// Metadata shared by the persisted return and the KB writer.
function kbResultMetadata() {
  return {
    // Only tp/ep/conc address AgentX. Observed shape is retained outside the content digest.
    workload: Object.assign({}, WORKLOAD, {
      kind: WORKLOAD_KIND, tp: SERVING_TP, ep: SERVING_EP,
    }, AGENTX ? {
      scenario: AGENTX.scenario, corpus: AGENTX.corpus, num_entries: AGENTX.num_entries,
      duration_s: AGENTX.duration_s, metric_basis: AGENTX_METRIC_BASIS,
    } : {}),
    metric_basis: AGENTX ? AGENTX_METRIC_BASIS : '',
    comparability: A.comparability || {},
  };
}

// --store is what the local plane writes into and is meaningless to a remote-only run; open_plane()
// hard-errors without it on plane local|both, so it is not optional there.
function kbPlaneFlags(plane) {
  plane = plane || E2E_KB_PLANE;
  return `--plane ${plane}` + (plane === 'remote' ? '' : ` --store ${shq(E2E_KB_STORE_DIR)}`);
}

// A READ takes one plane: `cmd_resolve` picks it (kb.plane:read_planes; service first, local mirror
// as fallback) and reports it as `read_plane`. The graded axis is not part of the address (one page
// holds several bases), so the reader states it, or mixed units would be ranked together. An
// explicit recall policy requires an exact basis; otherwise the measured basis filters (unlabelled
// records still match).
function kbMetricBasisFlag() {
  return E2E_KB_METRIC.basis ? `--metric-basis ${shq(E2E_KB_METRIC.basis)} ` +
    (E2E_KB_METRIC.strict ? '--require-metric-basis ' : '') : '';
}

function kbResolveScript(args) {
  return KB_ENV_PRELUDE + '\\\n' +
    `python3 ${shq(E2E_STORE_SCRIPT)} resolve ${kbIdentityFlags()} \\\n` +
    `  ${kbMetricBasisFlag()}${kbPlaneFlags(E2E_KB_PLANE)} ${args}`;
}

// Warm-start prompt injection, like expertSkillsBlock: '' when the feature is off, the role is not a
// consumer, or Module A found nothing (KB_REF_DIR is set only by Module A).
function warmStartBlock(role) {
  if (!KB_REF_DIR || !WARM_START_ROLES.has(role)) return '';
  // `tuning_kb=false` is the blind-eval switch; the tuning track's prior lives in the shared KB, so
  // the switch also withholds this block.
  if (role === 'tuning_specialist' && !TUNING_KB_ENABLED) return '';
  return `\n\n## Warm start (a PRIOR run's record — already measured on this box)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/warm_start.md and follow it. The knowledge base ` +
    `offered prior configurations for this exact deployment; they are in ${KB_REF_DIR}/, and ` +
    `${KB_REF_DIR}/measured_on_this_box.md records what happened when THIS run benched them — that ` +
    `file OVERRIDES the stored claims in its siblings wherever the two disagree. A stored number is ` +
    `a hypothesis; only the measured column is evidence.` +
    // The tuning track reads the same pages to check whether its search was already run once, so it
    // gets an extra paragraph rather than a second block.
    (role === 'tuning_specialist' && KB_CACHE_DIR
      ? `\n\nFor YOUR track specifically: an accepted-kernel entry marked \`from tuning skillset\` is a ` +
        `tuned artifact a prior run produced and proved engaged. Its file was downloaded with the ` +
        `record — look under ${KB_CACHE_DIR}/*/files/tuning/ — and the entry names the env var that ` +
        `binds it. Before you start searching a shape, check whether it is already there. If it is, ` +
        `install the artifact, prove engagement the same way you would for your own, and measure it ` +
        `in your pre/post A/B; a recalled artifact still has to earn its accept on THIS box, but it ` +
        `costs one measurement instead of a search. Search only the shapes that came back empty.`
      : '');
}

// Run-wide workload-identity invariant for EVERY role prompt (a trace replay changes what a
// benchmark means); '' unless the caller declared a trace replay.
function workloadIdentityBlock() {
  if (!IS_AGENTX) return '';
  // Late-adopt here too: every role prompt passes through this block.
  ensureMeasuredShape();
  return `
## WORKLOAD IDENTITY — this run measures a TRACE REPLAY, not a fixed ISL/OSL sweep
The served load is the AgentX corpus \`${AGENTX.corpus}\` replayed by aiperf under scenario
\`${AGENTX.scenario}\` (${AGENTX.num_entries} entries, concurrency ${CONC}).
The CLIENT owns the load: it chooses the request mix, the arrival pattern, its own warmup, and the
measurement duration. Consequences you must respect:

* **ISL=${ISL} / OSL=${OSL} is a KERNEL-TARGETING shape, not a benchmark** (provenance:
  ${WORKLOAD_SHAPE_PROVENANCE}${SHAPE_IS_MEASURED ? '' : ' — A PLACEHOLDER, NOT THIS CORPUS'}).
  ${SHAPE_IS_MEASURED
    ? 'It is an average of what was actually served, so it is the right regime to size tiles and\n  synthesize GEMM/attention inputs against. Read it as a centre of mass, not as a shape every\n  request has.'
    : 'It is the synthetic default and does NOT describe this load: the real shape is read off the\n  baseline once it has run, so nothing here has a corpus average compiled into it.\n  Do NOT size tiles or synthesize GEMM shapes from this number — if your phase needs the served\n  shape, take it from the baseline summary (observed_isl/observed_osl), not from this line.'}
  A trace corpus spans orders of magnitude in request length, so there is no single ISL: read the
  real distribution off the run's own export rather than assuming one. NEVER build a synthetic sweep
  at these lengths and NEVER report one as an e2e number — measured on this stack, a synthetic sweep
  came out ~2.76x the trace-replay throughput, so such a number is not a smaller-scale version of
  the real one, it is a different workload.
* **Keep passing \`ISL=<isl> OSL=<osl> CONC=<conc>\` on the bench line exactly as your role file
  shows.** The trace client ignores them. Do not restructure your bench invocation for this workload.
* **Client selection and the metric axis are ALREADY configured** in \`$EVAL_DIR/bench_env.sh\`, which
  \`bench_e2e.sh\` sources beside itself on every invocation. Do NOT set \`BENCH_CLIENT\`,
  \`E2E_METRIC\` or any \`AGENTX_*\` variable yourself, and never delete or edit that file.
* **The graded axis is ${AXIS_LABEL[AGENTX_METRIC_BASIS]}.**
  \`bench_summary.json\` reports it as \`throughput_tok_s_median\` with
  \`metric_basis=${AGENTX_METRIC_BASIS}\`. Always read the metric-neutral key; ${/intvty/.test(AGENTX_METRIC_BASIS)
    ? 'on this axis it is tok/s/user, higher is better. Report `guard_aggregate_output_tok_s_median`\n  beside it.' + (AGENTX_METRIC_BASIS === 'e2e_norm_intvty_p50'
      ? ' Also report `guard_e2e_norm_intvty_p90_median`, the tail guard.' : '')
    : 'on a ~140:1\n  prefill:output trace the output-only axis barely moves for a large real change in work.'}
* **A measured window is LONG**: ${AGENTX.geak_loop_duration_s}s per search leg,
  ${AGENTX.duration_s}s for parity/validation. Budget your phase around that and do not retry a
  timed-out leg blindly.
* **Search legs are deliberately non-canonical.** They run the scenario's ${AGENTX.geak_loop_duration_s}s
  floor rather than the canonical ${AGENTX.duration_s}s, so the client stamps them
  \`submission_valid=false\` with the deviations attached. That is EXPECTED and does not invalidate an
  A/B comparison — both legs deviate identically. It does mean a search-leg number may never be
  reported as a submittable result.
`;
}

// The reserve holds ONE search leg for Finalize, and a search-leg claim is no yardstick for Validate's
// legs (a claim equal to the baseline is canonical). Both are '' off a trace replay.
function agentxFinalizeNote() {
  if (!IS_AGENTX) return '';
  return ` On this trace replay the final bench of step 2 is ONE leg on the ${AGENTX.geak_loop_duration_s}s ` +
    `search window. Do NOT also bench the bundle on the ${AGENTX.duration_s}s canonical window: Validate ` +
    'measures the stock and the final stack there right after this phase, and the final reserve holds time ' +
    "for those legs only. Return the search leg's median as final_throughput_tok_s and say in note that it " +
    'is a search-window number. Omit throughput_speedup: BASELINE_THROUGHPUT was measured on the canonical ' +
    'window, so the ratio of the two is not a speedup.';
}
function agentxClaimNote(claimed, baseline) {
  if (!IS_AGENTX || !(claimed > 0) || claimed === baseline) return '';
  return ` On this trace replay CLAIMED_THROUGHPUT comes from ${AGENTX.geak_loop_duration_s}s search legs, ` +
    `which replay a different, shorter slice of the corpus than your ${AGENTX.duration_s}s legs, so it is ` +
    'not comparable with your numbers. When you set validation_status, drop both claim conditions ' +
    '(validated_win\'s "within the noise band of the claim" and flagged\'s "lower than the claim by ' +
    'more than the band") and arbitrate on your same-session A/B, the baseline and parity. Still ' +
    'report claimed_throughput_tok_s, and say in arbitration_note that it is a search-window number.';
}
// Every signal must agree, so no single adoption path can pass a changed bundle off as Setup's.
function agentxAdoptedNothing(s) {
  return IS_AGENTX && !!s.setup && s.baseline > 0 && !s.accepted && s.tput === s.baseline
    && s.flags === s.setup.flags && s.env === s.setup.env && s.overlay === s.setup.overlay;
}
// Why a trace replay skips Validate, or '': nothing to compare, or less time than its timed windows.
function agentxValidateSkip(adoptedNothing, leftMs) {
  if (!IS_AGENTX) return '';
  if (adoptedNothing) return 'final_is_baseline';
  return leftMs < AGENTX_VALIDATE_WINDOWS_MS ? 'no_time' : '';
}

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

// Optimization roles consume the skill's delivery; only the extractor loads its full procedure.
function kernelExtractBlock(role) {
  if (!KERNEL_EXTRACT_ON || !['system_architect', 'op_benchmarker'].includes(role)) return '';
  return `\n\n## AgentX prepared UTs\n` +
    `Read ${KERNEL_EXTRACT_DIR}/references/workflow-handoff.md. Use the supplied prepared tasks ` +
    `and saved workload/ranking evidence. Selection, cases, weights and source bindings are already ` +
    `validated; do not repeat discovery, sampling or UT authoring, or replace them with fixed ISL/OSL ` +
    `assumptions. Run the existing immutable UT against optimization candidates, then E2E A/B.\n` +
    `Declared workload kind: ${WORKLOAD_KIND}; specification: ${JSON.stringify(WORKLOAD_SPEC)}`;
}

function roleAgent(role, phase, intro, inputs) {
  // BACKEND is injected for every role: any role that calls bench_e2e.sh must forward it
  // (BACKEND=<backend>) so the right serving adapter (scripts/adapters/<backend>.sh) is used.
  const standaloneExtraction = KERNEL_EXTRACT_ON && role === 'kernel_extractor';
  const inall = {
    BACKEND, SERVING_TP, SERVING_GPU,
    MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS, VALIDATION_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST,
    E2E_LEARNED_KB: E2E_LEARNED_KB_ENABLED ? 'on' : 'off',
    // Only surfaced on a trace replay.
    ...(IS_AGENTX ? { WORKLOAD_KIND } : {}),
    ...(correctnessEnvBlock(role) ? { CORRECTNESS_UNSET_ENV: CORRECTNESS_UNSET_ENV.join(' ') } : {}),
    ...inputs,
    ...(standaloneExtraction ? {
      SKILL_ROOT: KERNEL_EXTRACT_DIR, SHARED_TOOLS: `${WORKFLOW_DIR}/scripts`, WORKFLOW_DIR,
      PARSE_PROFILE_STRICT_PHASES: '1', KERNEL_ENV_SKIP_ENUM_REAP: '1',
      CAPTURE_PERSIST_POLICY, CAPTURE_STORAGE_ENV,
    } : {}),
  };
  const instruction = standaloneExtraction
    ? `First Read ${KERNEL_EXTRACT_DIR}/SKILL.md and ${KERNEL_EXTRACT_DIR}/references/workflow-handoff.md.
Use the complete extraction procedure and its shared and skill-specific tools. The handoff reference supplies only workflow inputs and output packaging.
Do not load roles/kernel_extractor.md; it is the legacy procedure for other workloads.`
    : `First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.`;
  const base = `You are the ${role}. PHASE=${phase}.
${instruction}
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
When you invoke bench_e2e.sh, pass BACKEND=${BACKEND} in its env so the correct serving adapter is used.

## SERVING CONFIG INVARIANT (do not violate — all e2e numbers must be comparable)
Every e2e SERVING benchmark in this run (baseline, config sweep, integrate ref/cand, validation,
profiler trace) MUST use the SAME serving config: tensor-parallel TP=${SERVING_TP} on the GPU set
GPU=${SERVING_GPU}. Whenever you invoke bench_e2e.sh for a SERVING throughput/profile measurement, pass
exactly these in its env:
    BACKEND=${BACKEND} TP=${SERVING_TP} GPU=${SERVING_GPU}
NEVER change TP or the GPU set between the baseline, a candidate, and validation — a TP/GPU mismatch
makes every delta meaningless. (If SERVING_TP=1, GPU=${SERVING_GPU} is a single id; if SERVING_TP>1 it
is a comma-separated set spanning exactly TP GPUs.)
GPU_IDS=${GPU_IDS} is a SEPARATE OPTIMIZATION-PARALLELISM pool: it is used ONLY for single-GPU isolated
work (op_bench bake-offs, shape-capture, the recursive kernel layer), where each task pins ONE id from
the pool via GPU_ID. Do NOT use the serving TP/GPU set for that isolated work, and do NOT use a single
optimization-pool id for a serving launch — keep the two separate.
${workloadIdentityBlock()}
## Inputs
${cfg(inall)}

Return ONLY the structured JSON the ${standaloneExtraction ? 'workflow handoff reference' : 'role file'} specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role) + warmStartBlock(role) + kernelExtractBlock(role) + correctnessEnvBlock(role);
}

// Resilient agent wrapper: retry a failed agent a few times, then degrade to null for the callers'
// null-guards (setup re-throws). Each attempt races a timeout that resolves null (the harness turns
// errors into null, not hangs): 120min (45min fast), since agents launch servers and run ~30min
// benches. args.agent_timeout_ms=0 disables it; without setTimeout it is a raw agent().
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : (FAST_MODE ? 2700000 : 7200000), 10);
// The guard abandons the PROMISE, not the agent: a timed-out attempt keeps running its servers and legs,
// so a retry runs beside it on the same GPUs and files. agent_timeout_retry=false degrades to null instead.
const AGENT_TIMEOUT_RETRY = String(A.agent_timeout_retry != null ? A.agent_timeout_retry : 'true') !== 'false';
// Process-safety rules prepended to EVERY agent prompt at the one `agent()` funnel: no
// pattern-matched kills (they can TERM the caller's orchestrator), detach long legs from the tool's
// shell manager (it group-reaps them), and never start a command with `sleep` (Claude Code 2.1.310+
// refuses one of 25s+ with background tasks on; run_e2e.py turns those off too).
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
Launch it so it is orphaned to init, OUTSIDE the manager's reach, then wait on its log:
    cd "\$EVAL_DIR" && setsid nohup bash run_bench.sh > run_bench.out 2>&1 < /dev/null &
    # returns immediately; the leg's parent becomes PID 1 and no tool-call bound applies
Then wait in SEPARATE foreground calls, each a bounded loop that checks BEFORE it sleeps and ends inside
the timeout you give that Bash call (the tool caps it at BASH_MAX_TIMEOUT_MS, 10 min by default):
    timeout 540 bash -c 'until [ -s <leg_dir>/bench_summary.json ]; do sleep 30; done'; tail -5 run_bench.out
Never START a command with a \`sleep\` (Claude Code refuses one of 25s or more), and never wait by ending
your turn: your turn ending IS your return, so a Monitor, background-watcher or \`run_in_background\`
notification lands after you are gone. Keep waiting until the summary lands or the leg is provably dead.
Before you conclude a leg died, check whether it is still alive (\`ps -o pid,stat,etime -p <pid>\`) and
whether the GPUs still hold weights — a reaped wrapper with a live leg underneath looks identical to a crash.

## ONCE THE FINAL PHASE STARTS, IT OWNS THE GPUS
When \$EVAL_DIR/.final_phase_fence exists the search is over: the workflow stopped every leg, tuner and
waiter still running from it, on purpose, so Finalize and Validate measure on quiet GPUs. Unless you are
the e2e_integrator finalizing or the director validating, start nothing more on the GPUs (no server, leg,
tuner or profiler), do not relaunch anything of yours that was stopped, and return what you already have.
`;
function withProcessSafety(prompt) {
  return typeof prompt === 'string' ? PROCESS_SAFETY + prompt : prompt;
}

// Near the deadline, cap each OPTIMIZATION agent's hung-guard at (remaining − reserve), so a chain
// of agents in one loop iteration can't overrun the reserve. Final-phase agents are exempt by
// position (FINAL_PHASE_STARTED, set at the phase('Finalize') call site), not label: the
// Finalize-gate tags its agents 'Finalize'. 2min floor; inert without time_budget_s.
let FINAL_PHASE_STARTED = false;
const FINALIZE_GATE_PHASE = 'Finalize-gate';   // display grouping only — not load-bearing for the cap
const AGENT_TIMEOUT_FLOOR_MS = 120000;
function agentTimeoutFor() {
  if (TIME_BUDGET_MS == null || FINAL_PHASE_STARTED) return AGENT_TIMEOUT_MS;
  return Math.max(AGENT_TIMEOUT_FLOOR_MS, Math.min(AGENT_TIMEOUT_MS, remainingMs() - FINAL_RESERVE_MS));
}
// True when a new agent would only get the floor (reserve begun or under one floor away, final phase
// not started): nothing measured fits, and the abandoned agent would keep using the GPUs.
function atReserveFloor() {
  return TIME_BUDGET_MS != null && !FINAL_PHASE_STARTED &&
    remainingMs() - FINAL_RESERVE_MS <= AGENT_TIMEOUT_FLOOR_MS;
}
// A required checkpoint write (copy + hash, no GPU, 4-8min) is not bounded by the reserve cap: up to
// 30min, passed only at requireE2EValidationCheckpoint. A hung writer is not retried beside itself
// (two writers would leave digests disagreeing with assets).
const CHECKPOINT_COMMIT_TIMEOUT_MS = 30 * 60000;
function checkpointTimeoutFor() {
  return Math.max(agentTimeoutFor(), Math.min(AGENT_TIMEOUT_MS, CHECKPOINT_COMMIT_TIMEOUT_MS));
}

