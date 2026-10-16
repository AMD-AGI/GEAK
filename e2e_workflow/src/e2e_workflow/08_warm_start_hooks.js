// ---------------------------------------------------------------------------
// e2e warm start: identity formatting + the prompt hook.
//
// These three `let`s are the ONLY state the feature adds above Module A, and all three stay at their
// off-value (null / '') for the whole run unless Module A actually executes. Every consumer below
// keys off them, which is what makes `warm_start=off` produce a byte-identical run rather than one
// that merely behaves the same.
// ---------------------------------------------------------------------------
let KB_DIMS = null;        // the deployment dimensions the Director established during preflight
let KB_REF_DIR = '';       // where Module A left its references; '' when it never ran
let KB_CACHE_DIR = '';     // where Module A materialized the records' FILES (patches, tuned tables)
let KB_READ_PLANE = '';    // which plane ANSWERED the read, which `both` alone does not tell you
// Everything this run RECALLED, in one report-ready object: which address was asked, what came back,
// what re-measuring it HERE produced, and whether it was accepted. Threaded into the Report role's
// Inputs because final_report.md was previously silent about recall altogether — a run that recalled
// nothing and a run that recalled a 1.12x record and then rejected it wrote the same report, and the
// second is the one a reader most needs to see.
//
// `asked` is filled in even when zero candidates come back. On an exact-lookup scheme a miss and a
// never-recorded page are the SAME 404, so the ladder that was asked IS the finding: it is the only
// artifact that distinguishes "no prior art" from "prior art exists one segment away".
let KB_RECALL = { e2e: null, kernel: [] };

// One collection point for the KERNEL plane's recall. Every nested kernel_workflow returns its own
// `warm_start` block (kernel_lane.js), and until now all of it died inside the lane: the e2e report
// could say a kernel was authored from scratch while the lane had in fact adopted a stored patch.
// Called from the two bounded wrappers and the two direct workflow() sites, i.e. every lane call.
function noteKernelKB(r, label) {
  // Absorb a nested kernel run's own agent timeline (kernel_workflow/kernel_lane return it on
  // llm_timeline). This is the one funnel every lane return passes through, so it captures every
  // nested run's attempts for the ledger's "plus N nested kernel run(s)" accounting.
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
      // A lane that adopted a stored patch and then committed nothing of its own is where the kernel
      // plane most often flatters itself, so carry the split rather than just the headline geomean.
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

// The ONE place e2e KB identity argv is formatted — called by both the reader (Module A) and the
// writer (Module B). kb/identity.py's own header names the failure this prevents: a reader and a
// writer that disagree by a single segment do not raise, they address two different pages, and the
// only symptom is that history quietly stops existing. Two call sites formatting the same flags
// independently is exactly how that drift starts.
function kbIdentityFlags() {
  if (!KB_DIMS) return '';
  return [`--model ${shq(KB_DIMS.model)}`, `--gfx ${shq(KB_DIMS.gfx)}`,
    `--framework ${shq(BACKEND)}`, `--framework-version ${shq(KB_DIMS.framework_version)}`,
    `--precision ${shq(KB_DIMS.precision)}`, `--rocm-version ${shq(KB_DIMS.rocm_version)}`,
    `--tp ${SERVING_TP}`, `--ep ${SERVING_EP}`,
    // The DECLARATION (A.workload_kind / workload_spec.kind), never the shape PROVENANCE: the
    // provenance values include agentx_pending_baseline and synthetic_fallback_on_agentx, which
    // would mis-file precisely the degraded AgentX runs that most need a stable address.
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

// A READ takes exactly one plane. `cmd_resolve` now picks it via kb.plane:read_planes (service
// first, local mirror only when the service has no answer) and reports which one spoke as
// `read_plane`, so the bash branch that used to do it here is gone — one copy of the rule, and it
// is the copy a human gets from the CLI too.
// The graded axis is NOT part of the address (bench_summarize's bases are five and growing, and a
// reader wants them side by side with a label rather than scattered over five sparse pages), so the
// reader has to state it instead. One page has already carried ~64763 (total tok/s), ~471 (output
// tok/s) and ~81 (P90 interactivity) at once; ranked together the top row wins by unit, not merit.
// An explicit recall policy requires an exact basis, including on synthetic runs. Without one,
// preserve the historical measurement-derived filter and its compatibility with unlabelled records.
function kbMetricBasisFlag() {
  return E2E_KB_METRIC.basis ? `--metric-basis ${shq(E2E_KB_METRIC.basis)} ` +
    (E2E_KB_METRIC.strict ? '--require-metric-basis ' : '') : '';
}

function kbResolveScript(args) {
  return KB_ENV_PRELUDE + '\\\n' +
    `python3 ${shq(E2E_STORE_SCRIPT)} resolve ${kbIdentityFlags()} \\\n` +
    `  ${kbMetricBasisFlag()}${kbPlaneFlags(E2E_KB_PLANE)} ${args}`;
}

// Warm-start prompt injection, mirroring expertSkillsBlock exactly: returns '' whenever the feature
// is off or the role is not a consumer, so roleAgent's output is byte-identical to the pre-feature
// build in those cases. KB_REF_DIR is assigned by Module A and nowhere else, so an off run — or one
// whose read found nothing — never reaches the template at all.
function warmStartBlock(role) {
  if (!KB_REF_DIR || !WARM_START_ROLES.has(role)) return '';
  // `tuning_kb=false` is the blind-evaluation switch: it exists so a run can measure what the tuning
  // track finds with NO prior knowledge in its context. Now that the tuning track's prior knowledge
  // lives in the shared KB rather than in a private store, the switch has to reach here too — one
  // flag, one meaning, whichever store the knowledge happens to sit in. Without this, "blind" would
  // have quietly stopped being blind the moment the two stores merged.
  if (role === 'tuning_specialist' && !TUNING_KB_ENABLED) return '';
  return `\n\n## Warm start (a PRIOR run's record — already measured on this box)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/warm_start.md and follow it. The knowledge base ` +
    `offered prior configurations for this exact deployment; they are in ${KB_REF_DIR}/, and ` +
    `${KB_REF_DIR}/measured_on_this_box.md records what happened when THIS run benched them — that ` +
    `file OVERRIDES the stored claims in its siblings wherever the two disagree. A stored number is ` +
    `a hypothesis; only the measured column is evidence.` +
    // The tuning track reads the same pages for a different purpose, so it gets the extra paragraph
    // here rather than a second block: the others are shopping for a CONFIG to adopt, it is checking
    // whether the search it is about to spend hours on has already been run to completion once.
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

// Run-wide workload-identity invariant, injected into EVERY role prompt exactly
// like the serving-config invariant above it — a trace replay changes what a
// benchmark MEANS, so no role may be left assuming the synthetic sweep.
// Returns '' unless the caller declared a trace replay, so a fixed-ISL/OSL run
// gets byte-identical prompts.
function workloadIdentityBlock() {
  if (!IS_AGENTX) return '';
  // Late-adopt here rather than only at Setup: this block is the one path every
  // role prompt takes, so the shape can never be stale by more than one agent.
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

function roleAgent(role, phase, intro, inputs) {
  // BACKEND is injected for every role: any role that calls bench_e2e.sh must forward it
  // (BACKEND=<backend>) so the right serving adapter (scripts/adapters/<backend>.sh) is used.
  const inall = {
    BACKEND, SERVING_TP, SERVING_GPU,
    MEASUREMENT_MODE, PARITY_REPLICAS, SEARCH_REPLICAS, VALIDATION_REPLICAS,
    EFFECTIVE_CONFIG_DIGEST,
    E2E_LEARNED_KB: E2E_LEARNED_KB_ENABLED ? 'on' : 'off',
    // Only surfaced on a trace replay; a synthetic run's Inputs block is unchanged.
    ...(IS_AGENTX ? { WORKLOAD_KIND } : {}),
    ...(correctnessEnvBlock(role) ? { CORRECTNESS_UNSET_ENV: CORRECTNESS_UNSET_ENV.join(' ') } : {}),
    ...inputs,
  };
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
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

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role) + warmStartBlock(role) + correctnessEnvBlock(role);
}

// Resilient agent wrapper: a single agent failure (transient API 502 / didn't emit StructuredOutput)
// must NOT kill a multi-hour run. Retry a few times, then DEGRADE to null so the caller's existing
// null-guards skip/continue gracefully (critical phases like setup re-throw on null themselves).
// Bound each attempt: an agent LLM call that HANGS (no response, no terminal error) would block this
// await forever — the harness resolves terminal errors to null but not an indefinite hang. Race the
// call against a VERY generous timeout that resolves null, which the loop below treats as a failed
// attempt (retry, then degrade). A true hang never returns, so a generous bound still catches it while
// NEVER killing a legitimately-long agent. The OUTER e2e agents orchestrate the serving stack (Director
// launches sglang + runs the baseline bench ~30min; ConfigTuner does multiple server-launch+bench
// cycles; the head e2e gate overlays + launches + A/B benches) — these run far longer than a kernel
// agent, so the bound must be large (default 120min). Too-short a bound here causes the long setup
// agent to be killed and retried, spawning duplicate exp dirs. args.agent_timeout_ms=0 disables;
// falls back to raw agent() if setTimeout is unavailable.
// Default 120min (generous — outer e2e agents launch servers + run ~30min benches). Fast mode tightens
// it to 45min so a single hung/slow agent can't blow the wall-clock budget (still ample for the director
// baseline + the head e2e A/B). Default mode keeps 120min → unchanged.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : (FAST_MODE ? 2700000 : 7200000), 10);
// The guard abandons the PROMISE, not the agent: a timed-out attempt keeps running its servers and legs,
// so a retry runs beside it on the same GPUs and files. agent_timeout_retry=false degrades to null instead.
const AGENT_TIMEOUT_RETRY = String(A.agent_timeout_retry != null ? A.agent_timeout_retry : 'true') !== 'false';
// Process-safety rule prepended to EVERY agent prompt. It is injected at this funnel
// (the one place `agent()` is ever called) rather than in roleAgent(), because several
// prompts are built inline and would otherwise never see it — and any future call site
// inherits it automatically instead of having to remember it. Bash-capable agents own
// server lifecycles, and a pattern-matched kill is indistinguishable from a correct one
// until it TERMs the caller's orchestrator and fails the whole task.
// The detachment rule rides along for the same reason: EVERY bash-capable role runs a long
// serving leg (director's baseline, tuner's launch+bench cycles, the head e2e A/B), and each
// one otherwise rediscovers at ~1h that a tool-managed background shell gets group-reaped
// mid-measurement — an hour of budget and a booted server lost per role that learns it.
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
`;
function withProcessSafety(prompt) {
  return typeof prompt === 'string' ? PROCESS_SAFETY + prompt : prompt;
}

// Option A (Hyperloom #1202): near the deadline, tighten each OPTIMIZATION agent's hung-guard so no
// in-flight step finishes later than (budget − reserve). The loop-tops stop STARTING new work at
// TIME_HEAD_DEADLINE_MS, but one iteration is a SEQUENCE of agents (extract → bakeoff → author → verify)
// each armed with the full AGENT_TIMEOUT_MS, and that sequence could overrun the reserve and starve
// Finalize/Report/Validate (the #1202 stall was a 62min extract_op running into the SIGKILL). Final-phase
// agents are EXEMPT — they run INSIDE the reserve by design. The 2min floor lets a deadline-killed agent's
// retries self-limit instead of burning the reserve. Inert when time_budget_s is absent (byte-identical).
//
// Exemption is by POSITION, not by a caller-supplied label. Keying it on opts.phase ∈ ['Finalize',
// 'Report','Validate'] failed OPEN: the Finalize-GATE runs before phase('Finalize') yet tagged its
// integrator agents 'Finalize', so optimization work claimed unlimited time. FINAL_PHASE_STARTED is set
// once, at the phase('Finalize') call site. Agents in flight when it flips keep the cap they were armed
// with — work started before the final phase stays bounded.
let FINAL_PHASE_STARTED = false;
const FINALIZE_GATE_PHASE = 'Finalize-gate';   // display grouping only — not load-bearing for the cap
const AGENT_TIMEOUT_FLOOR_MS = 120000;
function agentTimeoutFor() {
  if (TIME_BUDGET_MS == null || FINAL_PHASE_STARTED) return AGENT_TIMEOUT_MS;
  return Math.max(AGENT_TIMEOUT_FLOOR_MS, Math.min(AGENT_TIMEOUT_MS, remainingMs() - FINAL_RESERVE_MS));
}
// An agent started now would only get the floor above: the reserve has begun, or is under one floor
// away, and the final phase has not started. Nothing measured fits in two minutes, and the guard that
// fires then abandons the promise, not the agent -- so whatever starts here runs on, unwatched, on the
// GPUs Finalize and Validate are about to need.
function atReserveFloor() {
  return TIME_BUDGET_MS != null && !FINAL_PHASE_STARTED &&
    remainingMs() - FINAL_RESERVE_MS <= AGENT_TIMEOUT_FLOOR_MS;
}

