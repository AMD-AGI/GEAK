// ---------------------------------------------------------------------------
// Kernel-targeting shape — the shape roles synthesize kernel inputs against
// ---------------------------------------------------------------------------
// Nothing is MEASURED at isl/osl on a trace replay: aiperf owns the sequence
// lengths and the bench client ignores these. But roles still read them as the
// analytic serving call model when they pick tile sizes and synthesize GEMM
// shapes, so leaving them at the synthetic 1024/1024 default would aim the whole
// kernel search orders of magnitude below the real load. This mirrors
// interface/run_e2e.py::_targeting_shape so the standalone and Hyperloom entries
// resolve the shape identically, and so the provenance printed into every role
// prompt is never a claim we cannot back -- the old default asserted
// 'handoff_workload' even when no handoff existed.
function resolveTargetShape() {
  const haveDeclared = ISL_DECLARED > 0 && OSL_DECLARED > 0;
  const declaredOr = (v) => (v != null ? v : 1024);
  // Hyperloom resolves this upstream and passes it down; honour it verbatim.
  if (A.workload_shape_provenance) {
    return [declaredOr(ISL_DECLARED), declaredOr(OSL_DECLARED),
            String(A.workload_shape_provenance)];
  }
  if (!IS_AGENTX) {
    // Unchanged behaviour for the fixed ISL/OSL path, minus the borrowed label.
    return [declaredOr(ISL_DECLARED), declaredOr(OSL_DECLARED),
            haveDeclared ? 'declared_args' : 'synthetic_default'];
  }
  // An explicit shape is the caller stating what they measured on this stack.
  // Absent that, we do NOT guess and we do not compile a corpus average into this
  // file: the corpus is declarable, AgentX keeps publishing new ones, the numbers
  // move with the tokenizer and the window, and a stale constant would be read as
  // fact by every role. The run measures its own shape instead -- the baseline
  // bench reports it and adoptMeasuredShape() below installs it, which is also
  // why no kernel work is scheduled before the baseline lands.
  const obsIsl = parseInt(AGENTX.observed_isl, 10);
  const obsOsl = parseInt(AGENTX.observed_osl, 10);
  if (obsIsl > 0 && obsOsl > 0) return [obsIsl, obsOsl, 'agentx_observed'];
  if (haveDeclared) return [ISL_DECLARED, OSL_DECLARED, 'agentx_declared_args'];
  return [1024, 1024, 'agentx_pending_baseline'];
}
// Mutable: on a trace replay the real shape is not knowable until something has
// been measured, so the baseline installs it and every later phase reads it.
let [ISL, OSL, WORKLOAD_SHAPE_PROVENANCE] = resolveTargetShape();
const WORKLOAD = { isl: ISL, osl: OSL, conc: CONC, shape_provenance: WORKLOAD_SHAPE_PROVENANCE };
// Whether ISL/OSL describe the served load or are still a placeholder. A
// placeholder never corrupts the MEASUREMENT, but it would misdirect every
// kernel choice, so the prompt says which one a role is holding.
let SHAPE_IS_MEASURED = !/^(agentx_pending_baseline|synthetic_fallback_on_agentx)$/
  .test(WORKLOAD_SHAPE_PROVENANCE);

// Install the shape the baseline actually served. bench_e2e.sh reports it as
// observed_isl/observed_osl (averaged per request, medianed across replicas), so
// this is measured on THIS corpus, tokenizer, model and window rather than
// asserted. Called right after Setup, before any kernel work is scheduled.
function adoptMeasuredShape(isl, osl, where) {
  if (!(isl > 0 && osl > 0)) return false;
  ISL = Math.round(isl);
  OSL = Math.round(osl);
  WORKLOAD_SHAPE_PROVENANCE = 'agentx_measured_this_run';
  SHAPE_IS_MEASURED = true;
  WORKLOAD.isl = ISL;
  WORKLOAD.osl = OSL;
  WORKLOAD.shape_provenance = WORKLOAD_SHAPE_PROVENANCE;
  log(`Kernel-targeting shape measured from ${where}: isl=${ISL} osl=${OSL} `
    + `(ratio ${(ISL / Math.max(1, OSL)).toFixed(1)}:1). Roles size kernels against this.`);
  return true;
}
// The Setup probe reads the baseline summary ONCE, at the instant Setup returns,
// and that instant is not guaranteed to contain one. An agent whose leg dies to a
// hardware fault moves the wreckage aside and relaunches, so `baseline/` is
// legitimately EMPTY while a replacement leg runs, and the summary lands minutes
// after Setup reports done. A single read then loses the shape for the whole run
// rather than for a moment: on the 20260909 standalone run Setup returned at
// 10:40:49 with the directory just renamed away, the good summary was written at
// 11:57, and every later phase kept the 1024/1024 placeholder. "Not written yet"
// is a retryable state, not a verdict, so adoption is ALSO attempted lazily from
// the prompt builder every role passes through: the first prompt built after the
// summary appears installs the real shape. A no-op once measured, so a run that
// adopted at Setup pays nothing and its prompts are unchanged.
function ensureMeasuredShape() {
  if (SHAPE_IS_MEASURED || !IS_AGENTX) return SHAPE_IS_MEASURED;
  // EVAL_DIR is a `let` declared further down and only assigned by Setup, so it
  // is read through typeof: a prompt built before that declaration is evaluated
  // would otherwise die in the temporal dead zone, turning a missing shape into
  // a crashed run. Nothing to read before Setup anyway — the baseline is what
  // Setup produces.
  const evalDir = typeof EVAL_DIR === 'undefined' ? '' : (EVAL_DIR || '');
  if (!evalDir) return SHAPE_IS_MEASURED;
  let shape = null;
  try {
    // Same execFileSync/python3 route the Setup probe uses: this JS has no fs.
    const { execFileSync } = require('child_process');
    const out = execFileSync('python3', ['-c', `
import json, os, sys
p = os.path.join(sys.argv[1], 'baseline', 'bench_summary.json')
def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
try:
    with open(p) as fh:
        s = json.load(fh) or {}
    print(json.dumps([_num(s.get('observed_isl')), _num(s.get('observed_osl'))]))
except Exception:
    print('[null, null]')
`, evalDir], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 30000 });
    shape = JSON.parse(out);
  } catch (err) {
    return SHAPE_IS_MEASURED;   // unreadable right now; a later prompt retries
  }
  if (shape && shape[0] > 0 && shape[1] > 0) {
    adoptMeasuredShape(shape[0], shape[1],
      'the baseline bench (read after Setup returned, which had no summary yet)');
  }
  return SHAPE_IS_MEASURED;
}
// Which axis bench_e2e.sh medians. It defaults E2E_METRIC to output, and on a ~140:1
// prefill:output trace output-only tok/s moves almost not at all for a large change in total
// work, so the axis has to be SELECTED here.
//
// bench_summarize.py's _BASES, flattened: each E2E_METRIC it accepts -> the metric_basis its
// summary records. test_bench_env_channel.py holds this table to that one.
const METRIC_BASIS_OF_AXIS = {
  output: 'aggregate_output_tok_s',
  total: 'aggregate_total_token_tok_s',
  total_token: 'aggregate_total_token_tok_s',
  total_throughput: 'aggregate_total_token_tok_s',
  intvty: 'e2e_norm_intvty_p90',
  interactivity: 'e2e_norm_intvty_p90',
  e2e_norm_intvty_p50: 'e2e_norm_intvty_p50',
  e2e_norm_intvty_p90: 'e2e_norm_intvty_p90',
  p90_intvty_inferencex: 'p90_intvty_inferencex',
};
// The E2E_METRIC the declaration sets for each basis: the basis itself wherever _BASES takes it.
const AXIS_OF_METRIC_BASIS = {
  aggregate_output_tok_s: 'output',
  aggregate_total_token_tok_s: 'total',
  e2e_norm_intvty_p50: 'e2e_norm_intvty_p50',
  e2e_norm_intvty_p90: 'e2e_norm_intvty_p90',
  p90_intvty_inferencex: 'p90_intvty_inferencex',
};
const AXIS_LABEL = {
  aggregate_output_tok_s: 'OUTPUT tok/s',
  aggregate_total_token_tok_s: 'TOTAL (input+output) tok/s',
  e2e_norm_intvty_p50: 'TTFT-inclusive P50 interactivity: 1 / P50(E2EL/OSL) in tok/s/user',
  p90_intvty_inferencex: 'InferenceX P90 interactivity: 1000 / P90(ITL) in tok/s/user, decode only',
  e2e_norm_intvty_p90: 'TTFT-inclusive P90 interactivity: 1 / P90(E2EL/OSL) in tok/s/user',
};
// Precedence: the basis the caller declared; else an E2E_METRIC already exported, because an
// exported value outranks bench_env.sh and is what the bench will measure; else the standalone
// default. Anything bench_summarize.py cannot measure fails HERE: it would refuse the axis too,
// but only once the baseline replay had been paid for.
function resolveAgentxMetric() {
  if (!AGENTX) return { basis: '', source: '' };
  const unknown = (what, v) => new Error(`AgentX: ${what} '${v}' is not an axis ` +
    `bench_summarize.py measures (known: ${Object.keys(METRIC_BASIS_OF_AXIS).join(', ')}).`);
  let inherited = '';
  try { inherited = String(process.env.E2E_METRIC || '').trim().toLowerCase(); } catch (_) { /* no env */ }
  // The bench reads an exported value as an axis token, so that is the only way it is looked up.
  const inheritedBasis = inherited ? METRIC_BASIS_OF_AXIS[inherited] : '';
  if (inherited && !inheritedBasis) throw unknown('the exported E2E_METRIC', inherited);
  const declared = String(WORKLOAD_SPEC.metric_basis || '').trim().toLowerCase();
  if (!declared) {
    return inheritedBasis ? { basis: inheritedBasis, source: 'the exported E2E_METRIC' }
      : { basis: AGENTX_DEFAULTS.metric_basis, source: 'the standalone default' };
  }
  const basis = AXIS_OF_METRIC_BASIS[declared] ? declared : METRIC_BASIS_OF_AXIS[declared];
  if (!basis) throw unknown('workload_spec.metric_basis', declared);
  if (inheritedBasis && inheritedBasis !== basis) {
    throw new Error(`AgentX: workload_spec.metric_basis=${basis}, but E2E_METRIC=${inherited} is ` +
      'exported in this environment and outranks bench_env.sh, so every bench would grade a ' +
      'different axis than the one declared. Unset E2E_METRIC or declare the basis it names.');
  }
  return { basis, source: 'workload_spec.metric_basis' };
}
const { basis: AGENTX_METRIC_BASIS, source: AGENTX_METRIC_SOURCE } = resolveAgentxMetric();
const AGENTX_E2E_METRIC = AGENTX ? AXIS_OF_METRIC_BASIS[AGENTX_METRIC_BASIS] : '';
// Recall has its own policy. In particular, older orchestrators may grade output/total while
// the operator wants only p50 candidates. This never changes the bench axis or the writer's basis.
function resolveKbMetric() {
  let env = '';
  try { env = String(process.env.GEAK_E2E_KB_METRIC_BASIS || '').trim(); } catch (_) { /* no env */ }
  const arg = String(A.e2e_kb_metric_basis || '').trim();
  const requested = (arg || env).toLowerCase();
  if (!requested) return { basis: AGENTX_METRIC_BASIS, strict: false, source: 'measurement' };
  const basis = AXIS_OF_METRIC_BASIS[requested] ? requested : METRIC_BASIS_OF_AXIS[requested];
  if (!basis) throw new Error(`Unknown e2e KB metric basis '${requested}'. Known bases: ` +
    Object.keys(AXIS_OF_METRIC_BASIS).join(', '));
  return { basis, strict: true, source: arg ? 'e2e_kb_metric_basis' : 'GEAK_E2E_KB_METRIC_BASIS' };
}
const E2E_KB_METRIC = resolveKbMetric();
function kbMeasurementBasis() {
  if (AGENTX) return AGENTX_METRIC_BASIS;
  let axis = '';
  try { axis = String(process.env.E2E_METRIC || '').trim().toLowerCase(); } catch (_) { /* no env */ }
  return METRIC_BASIS_OF_AXIS[axis] || 'aggregate_output_tok_s';
}
function kbCanAttestMetric(candidate) {
  const measured = kbMeasurementBasis();
  // A fresh output/total A/B is not evidence for or against the stored p50 claim.
  return candidate.metric_basis ? candidate.metric_basis === measured : !E2E_KB_METRIC.strict;
}
// Body of the per-run bench_env.sh the Director drops beside the copied bench
// script (see roles/director.md PHASE=setup). Every line assigns ONLY when the
// name is unset or empty, so a real exported value -- an orchestrator's, or an
// operator's one-off override on the command line -- always outranks the
// declaration. Empty string when synthetic.
//
// The guard is spelled `[ -n "${K:-}" ] || K='v'` rather than the shorter
// `: "${K:='v'}"` on purpose: inside a `${K:=word}` expansion the quotes are
// NOT shell quoting, they are literal characters of `word`, so that spelling
// assigns the value WITH its quotes attached. This form puts the value in real
// single quotes, which is also what makes _shq's escaping mean anything.
const _shq = (v) => `'${String(v).replace(/'/g, "'\\''")}'`;
const _agentxEnvPairs = () => {
  if (!AGENTX) return [];
  const pairs = [
    ['GEAK_WORKLOAD_KIND', WORKLOAD_KIND],
    ['GEAK_ISL_OSL_INACTIVE', '1'],
    ['BENCH_CLIENT', 'agentx'],
    ['E2E_METRIC', AGENTX_E2E_METRIC],
    ['GEAK_METRIC_BASIS', AGENTX_METRIC_BASIS],
    // No REPEATS: an explicit one outranks REPLICAS in bench_e2e.sh, so pinning it here turned
    // isolated validation's three replicas into one. bench_e2e.sh defaults a trace-replay
    // client's legacy rounds to one itself, after the replica count is resolved.
    ['CONC', String(CONC)],
    ['GEAK_AGENTX_SCENARIO', AGENTX.scenario],
    ['AGENTX_DATASET', AGENTX.corpus],
    // What the run is compared AGAINST, so an override of the corpus or the window never
    // carries it along: a replay of anything else is then stamped non-canonical, which is
    // the point. Declare canonical_corpus / canonical_duration_s to move the reference itself.
    ['AGENTX_CANONICAL_DATASET', AGENTX.canonical_corpus || AGENTX_DEFAULTS.corpus],
    ['AGENTX_NUM_ENTRIES', AGENTX.num_entries],
    ['GEAK_AGENTX_DURATION_S', AGENTX.duration_s],
    ['AGENTX_CANONICAL_DURATION', AGENTX.canonical_duration_s || AGENTX_DEFAULTS.duration_s],
    ['GEAK_AGENTX_LOOP_DURATION_S', AGENTX.geak_loop_duration_s],
    ['AGENTX_WARMUP_REQUESTS_PER_LANE', AGENTX.warmup_requests_per_lane],
    ['AGENTX_WARMUP_GRACE_PERIOD', AGENTX.warmup_grace_period_s],
  ];
  // Optional, only when the caller supplied them: the aiperf binary, the
  // InferenceX checkout that may hold map_aiperf.py (GEAK vendors a fallback),
  // and the profile-window placement.
  const opt = [
    ['AGENTX_FAILED_REQUEST_THRESHOLD', AGENTX.failed_request_threshold],
    ['AIPERF_BIN', AGENTX.aiperf_bin],
    ['INFERENCEX_PATH', AGENTX.inferencex_path],
    ['AGENTX_PROFILE_WARMUP_S', AGENTX.profile_warmup_s],
    ['AGENTX_PROFILE_WINDOW_S', AGENTX.profile_window_s],
    ['AGENTX_MAX_CTX', AGENTX.max_context_length],
  ];
  for (const [k, v] of opt) if (v != null && String(v).trim() !== '') pairs.push([k, v]);
  return pairs;
};
const AGENTX_ENV = !AGENTX ? '' : [
  '#!/usr/bin/env bash',
  '# Per-run workload declaration, written by e2e_workflow.js at Setup.',
  '# Sourced by bench_e2e.sh / bench_replica.sh from beside themselves.',
  '# Assign-if-unset throughout: an inherited/exported value always wins here.',
  ...(_agentxEnvPairs().map(([k, v]) => `[ -n "\${${k}:-}" ] || ${k}=${_shq(v)}`)),
  `export ${_agentxEnvPairs().map(([k]) => k).join(' ')}`,
  '',
].join('\n');
// Seed config: when an external orchestrator (e.g. Hyperloom) already did
// config/param search, it passes its accepted best flags/env so the GEAK
// baseline is measured ON that config (fair engagement start), not the stack
// default. Serving TP/GPU are handled by SERVING_TP / SERVING_GPU above.
const INIT_FLAGS = String(A.initial_extra_server_args || '');
const INIT_ENV = String(A.initial_extra_env || '');
// Schema-v2 handoffs may carry the exact Python overlay/source snapshot stack
// that produced Hyperloom's current best.  This is part of the baseline
// configuration, not a GEAK-authored candidate, so it must remain underneath
// every later candidate overlay instead of being dropped at Setup.
const INIT_BASE_OVERLAY = String(A.initial_overlay_pythonpath || '');
const EFFECTIVE_CONFIG_DIGEST = String(A.effective_config_digest || '');
// One lifecycle for EVERY throughput number (baseline, search A/B, parity, validation), because
// a cache-cold search number and a cache-warm validation number are not comparable and the gains
// carried between phases used to mix the two. warm_server = Hyperloom's protocol: one server per
// leg, round 1 a full warmup that is discarded, round 2 IS the number.
// isolated_server (REPLICAS fresh servers per leg) stays available when boot-to-boot variance is
// what must be gated on. The bench dispatcher owns retry/degraded aggregation in both modes.
const MEASUREMENT_MODE = String(A.measurement_mode || 'warm_server');
const PARITY_REPLICAS = parseInt(A.parity_replicas != null ? A.parity_replicas : 1, 10);
const SEARCH_REPLICAS = parseInt(A.search_replicas != null ? A.search_replicas : 1, 10);
const VALIDATION_REPLICAS = parseInt(A.validation_replicas != null ? A.validation_replicas : 3, 10);
// Validation keeps its OWN knob so an operator can make the final re-measure stricter than the
// search that fed it (isolated_server + validation_replicas=3 buys the boot-to-boot dispersion
// test). Default is warm_server like everything else, which is what the orchestrator rebenches
// against — and the cheap one: isolated validation is 6-12 cold boots serialized behind the
// serving-GPU flock, enough to overrun FINAL_RESERVE_MS and ship no validated number at all.
const VALIDATION_MEASUREMENT_MODE = String(A.validation_measurement_mode || 'warm_server');
// Sample count follows the mode rather than a knob of its own: warm_server IS one timed round per
// leg, and a second knob could only disagree with the lifecycle it belongs to. Going isolated is
// what buys extra samples, and it already has `validation_replicas` to size them.
const VALIDATION_SAMPLES = VALIDATION_MEASUREMENT_MODE === 'warm_server' ? 1 : VALIDATION_REPLICAS;
// CUDA/HIP-graph deployment requirement (general; derived from the serving config, NOT hardcoded).
// vllm/sglang/atom capture the steady-state decode path into a FULL CUDA/HIP graph UNLESS --enforce-eager
// is set (ATOM exposes both --enforce-eager and --cudagraph-capture-sizes, so it belongs in this set).
// A kernel that wins only via its OWN per-call graph-capture+replay wrapper falls back to eager inside the
// server's graph, so the isolated win evaporates e2e (observed on M3: MoE 1.22x isolated -> 0% e2e). When
// graphs are on we inject an EXPLICIT requirement into every kernel-optimize task: the win must be intrinsic
// and graph-capture-safe. Detection is config-driven (enforce-eager absent + graph-capable backend), so it
// auto-disables for an enforce-eager run and applies to any future graph-capturing backend.
const CUDA_GRAPH_DEPLOY = (BACKEND === 'vllm' || BACKEND === 'sglang' || BACKEND === 'atom') && !/enforce[-_]eager/i.test(INIT_FLAGS);
const GRAPH_REQ = CUDA_GRAPH_DEPLOY ? (
  ' DEPLOYMENT REQUIREMENT — the server captures the steady-state decode path into a FULL CUDA/HIP graph, ' +
  'so this kernel runs INSIDE that captured graph. Your speedup MUST be INTRINSIC: better tiles/algorithm, ' +
  `fused quant (one fp8 ${MATRIX_CORE_NAME}, kill the dequant), or fewer ops/launches that reduce work INSIDE the captured ` +
  'region. Do NOT rely on a per-call CUDA/HIP-graph capture+replay WRAPPER for the speedup — inside the ' +
  "server's graph that wrapper falls back to eager and the win vanishes, and the e2e integrate gate WILL " +
  'reject a wrapper-only win (this already happened: a 1.22x isolated MoE GEMM gave 0% e2e because only its ' +
  'static tile change survived the graph). The steady-state decode call must be graph-capture-safe: ' +
  'host-sync-free (no .item()/.cpu()/.tolist()/.synchronize(), no Python branch on a GPU scalar), shape-stable, ' +
  'and prep/compile ONCE (cache by data_ptr) so the captured region only LAUNCHES the kernel. VERIFY your ' +
  'speedup holds when the op is replayed under a CUDA graph, not just in eager timing.'
) : '';
// Acceptance noise band (%). Isolated-server ref/candidate measurements, non-overlap, and engagement
// proof (see e2e_integrator) make a 0.5% default trustworthy for a fixed-shape workload, where a
// replica is a fixed amount of identical work. Prompt-tunable.
//
// A trace replay is not that workload. It is client-paced over trajectories of wildly unequal size
// (ISL p50 83k, p99 491k) against a prefix cache running 85-96% hits, so a replica completes a
// whole number of trajectories and the throughput it reports quantises on which ones it got
// through. The 20260912 validation measured that directly: two surviving replicas of the
// BASELINE, same config, same box, spread 3.43% on total tok/s (24054.0 / 23255.9) and 3.86% on
// InferenceX P90 interactivity (29.52 / 28.40 tok/s/user), the default axis. A 0.5% band against
// that is 7-8x too permissive -- it would call a pure-noise difference a win -- so this workload
// carries its own floor. The TTFT-inclusive axis moved 17% on the same pair; a run graded on it
// needs an explicit noise_band_pct. Hardcoding the floor is a placeholder for deriving it from
// the replica spread Setup already measures; until then it is at least the measured one.
const NOISE_BAND_AGENTX = 4.0;
const NOISE_BAND_DEFAULT = parseFloat(
  A.noise_band_pct != null ? A.noise_band_pct : (IS_AGENTX ? NOISE_BAND_AGENTX : 0.5));
// The band the run gates on, from the one the Director returned. Its role file starts every run
// at 0.5 and only ever widens it, so on a trace replay that 0.5 would replace the workload floor
// (and an explicit noise_band_pct) outright; there the Director may only raise the band. A
// fixed-shape run keeps the Director's value, as it always has.
function gateNoiseBand(reported) {
  const r = parseFloat(reported);
  const director = Number.isFinite(r) && r > 0 ? r : 0;
  if (!IS_AGENTX) return director || NOISE_BAND_DEFAULT;
  return Math.max(director, NOISE_BAND_DEFAULT);
}
// No timed-repeat knob on purpose: the round count belongs to the lifecycle (bench_e2e.sh derives
// it from GEAK_REPEAT_MODE + MEASUREMENT_PURPOSE), so a second knob could only disagree with it.
// Every integrate A/B MUST measure BOTH legs (reference + candidate). When the
// integrator returns gate:'incomplete'/ab_complete:false (it only ran ref, hung,
// or degraded), the orchestrator RE-INVOKES it to finish the missing leg up to
// this many times before falling back to carrying it as a pending integration.
// This is what guarantees "every A/B runs ref AND cand to completion regardless
// of pass/fail" — general, not per-kernel. Bump via args.ab_finish_retries.
const AB_FINISH_RETRIES = parseInt(A.ab_finish_retries != null ? A.ab_finish_retries : 3, 10);
// A resolvable FROZEN baseline (a seeded baseline_overlay/ + a declared meta.candidate_bind, or an
// importable meta.baseline_callable on the op track) is the speedup DENOMINATOR and is MANDATORY.
// If an extraction smoke-passes but froze no baseline, the
// unittest would silently time the candidate against its own naive same-language scaffold (the
// "optimized-HIP vs naive-HIP = fake 15.7× isolated, ~0% e2e" bug). When that happens we RE-EXTRACT
// up to this many times; if still missing, the extraction is treated as a FAILURE (flag dominant /
// skip others) — never a fake speedup. Bump via args.baseline_extract_retries.
const BASELINE_EXTRACT_RETRIES = parseInt(A.baseline_extract_retries != null ? A.baseline_extract_retries : 3, 10);
// Issue #429: force capture storage bounds into every extraction EXTRA_ENV (override via args).
const CAPTURE_BYTE_BUDGET = String(A.capture_byte_budget != null ? A.capture_byte_budget : '8GiB');
const CAPTURE_CASE_BYTE_LIMIT = String(A.capture_case_byte_limit != null ? A.capture_case_byte_limit : '2GiB');
const CAPTURE_PERSIST_POLICY = String(A.capture_persist_policy != null ? A.capture_persist_policy : 'share_large');
const CAPTURE_WORKSPACE_BUDGET = String(A.capture_workspace_budget != null ? A.capture_workspace_budget : '32GiB');
const CAPTURE_STORAGE_ENV = `CAPTURE_BYTE_BUDGET=${CAPTURE_BYTE_BUDGET} CAPTURE_CASE_BYTE_LIMIT=${CAPTURE_CASE_BYTE_LIMIT} CAPTURE_PERSIST_POLICY=${CAPTURE_PERSIST_POLICY}`;
const TASK = A.task || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
// The caller reads workflow_return.json and recovers results only from a pinned eval_dir, so a Director
// that builds a sibling directory leaves the whole run where the caller never looks.
function evalDirDivergence(override, returned) {
  const norm = (p) => String(p || '').trim().replace(/(.)\/+$/, '$1');
  if (!norm(override) || norm(returned) === norm(override)) return '';
  return `Director returned eval_dir ${norm(returned) || '(none)'}, but the caller pinned ${norm(override)}`;
}
const MODEL_NAME_HINT = (MODEL_PATH || KERNEL_PATH).replace(/\/+$/, '').split('/').pop();

