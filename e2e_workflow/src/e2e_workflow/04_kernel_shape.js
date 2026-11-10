// Kernel-targeting shape — the shape roles synthesize kernel inputs against. On a trace replay
// isl/osl aren't measured (aiperf owns lengths), but roles size tiles and GEMM shapes from them, so a
// 1024/1024 placeholder would aim the search far below the real load. Mirrors
// interface/run_e2e.py::_targeting_shape so both entries resolve the same shape and provenance.
function resolveTargetShape() {
  const haveDeclared = ISL_DECLARED > 0 && OSL_DECLARED > 0;
  const declaredOr = (v) => (v != null ? v : 1024);
  // Hyperloom resolves this upstream and passes it down; honour it verbatim.
  if (A.workload_shape_provenance) {
    return [declaredOr(ISL_DECLARED), declaredOr(OSL_DECLARED),
            String(A.workload_shape_provenance)];
  }
  if (!IS_AGENTX) {
    // Fixed ISL/OSL path: declared args, else the synthetic default.
    return [declaredOr(ISL_DECLARED), declaredOr(OSL_DECLARED),
            haveDeclared ? 'declared_args' : 'synthetic_default'];
  }
  // Use an explicit observed shape if given; never guess or hard-code a corpus average (it moves
  // with corpus, tokenizer and window). Otherwise the baseline bench measures it and
  // adoptMeasuredShape() installs it, which is why no kernel work starts before the baseline.
  const obsIsl = parseInt(AGENTX.observed_isl, 10);
  const obsOsl = parseInt(AGENTX.observed_osl, 10);
  if (obsIsl > 0 && obsOsl > 0) return [obsIsl, obsOsl, 'agentx_observed'];
  if (haveDeclared) return [ISL_DECLARED, OSL_DECLARED, 'agentx_declared_args'];
  return [1024, 1024, 'agentx_pending_baseline'];
}
// Mutable: on a trace replay the baseline installs the measured shape.
let [ISL, OSL, WORKLOAD_SHAPE_PROVENANCE] = resolveTargetShape();
const WORKLOAD = { isl: ISL, osl: OSL, conc: CONC, shape_provenance: WORKLOAD_SHAPE_PROVENANCE };
// Whether ISL/OSL describe the served load or a placeholder; prompts tell roles which, since a
// placeholder would misdirect kernel choices.
let SHAPE_IS_MEASURED = !/^(agentx_pending_baseline|synthetic_fallback_on_agentx)$/
  .test(WORKLOAD_SHAPE_PROVENANCE);

// Install the shape the baseline served (bench_e2e.sh observed_isl/observed_osl: per-request mean,
// median across replicas). Called right after Setup, before any kernel work.
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
// The baseline summary can land after Setup returns (a leg relaunched after a hardware fault), so
// the prompt builder also retries adoption lazily: the first prompt built after the summary
// appears installs the real shape. A no-op once measured.
function ensureMeasuredShape() {
  if (SHAPE_IS_MEASURED || !IS_AGENTX) return SHAPE_IS_MEASURED;
  // EVAL_DIR is assigned by Setup; before that there is no baseline to read.
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
// The axis bench_e2e.sh medians must be selected here: its default (output tok/s) barely moves on a
// prefill-heavy trace. This is bench_summarize.py's _BASES flattened (E2E_METRIC -> metric_basis);
// test_bench_env_channel.py keeps the two in sync.
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
// Precedence: the declared basis; else an exported E2E_METRIC (it outranks bench_env.sh); else
// the standalone default. An axis bench_summarize.py can't measure fails here, before the baseline.
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
// KB recall can use its own basis (e.g. only p50 candidates while the bench grades output/total);
// it never changes the bench axis or the writer's basis.
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
// Body of the per-run bench_env.sh the Director drops beside the bench script (roles/director.md
// PHASE=setup); empty when synthetic. Each line assigns only if the name is unset or empty, so an
// exported value always wins. Spelled `[ -n "${K:-}" ] || K='v'`, not `: "${K:='v'}"`, whose quotes
// would become part of the value.
const _shq = (v) => `'${String(v).replace(/'/g, "'\\''")}'`;
const _agentxEnvPairs = () => {
  if (!AGENTX) return [];
  const pairs = [
    ['GEAK_WORKLOAD_KIND', WORKLOAD_KIND],
    ['GEAK_ISL_OSL_INACTIVE', '1'],
    ['BENCH_CLIENT', 'agentx'],
    ['E2E_METRIC', AGENTX_E2E_METRIC],
    ['GEAK_METRIC_BASIS', AGENTX_METRIC_BASIS],
    // No REPEATS: an explicit one outranks REPLICAS in bench_e2e.sh, which sets trace-replay rounds
    // itself after resolving the replica count.
    ['CONC', String(CONC)],
    ['GEAK_AGENTX_SCENARIO', AGENTX.scenario],
    ['AGENTX_DATASET', AGENTX.corpus],
    // The reference the run is compared against; overriding the corpus, entry count or window
    // doesn't move it (so such a replay is non-canonical). Declare canonical_corpus /
    // canonical_num_entries / canonical_duration_s to move it.
    ['AGENTX_CANONICAL_DATASET', AGENTX.canonical_corpus || AGENTX_DEFAULTS.corpus],
    ['AGENTX_NUM_ENTRIES', AGENTX.num_entries],
    ['AGENTX_CANONICAL_NUM_ENTRIES', AGENTX.canonical_num_entries || AGENTX_DEFAULTS.num_entries],
    ['GEAK_AGENTX_DURATION_S', AGENTX.duration_s],
    ['AGENTX_CANONICAL_DURATION', AGENTX.canonical_duration_s || AGENTX_DEFAULTS.duration_s],
    ['GEAK_AGENTX_LOOP_DURATION_S', AGENTX.geak_loop_duration_s],
    ['AGENTX_WARMUP_REQUESTS_PER_LANE', AGENTX.warmup_requests_per_lane],
    ['AGENTX_WARMUP_GRACE_PERIOD', AGENTX.warmup_grace_period_s],
  ];
  // Optional, only when supplied: aiperf binary, InferenceX checkout (may hold map_aiperf.py; GEAK
  // vendors a fallback), and the profile-window placement.
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
// Kernel extraction is enabled only by an AgentX workload; model/backend features select its
// branches, never its activation.
const KERNEL_EXTRACT_ON = !!MODEL_PATH && WORKLOAD_KIND === 'agentx_trace_replay';
const KERNEL_EXTRACT_DIR = `${WORKFLOW_DIR}/knowledge/kernel_extraction_skills/kernel-extract`;
if (KERNEL_EXTRACT_ON) {
  WORKLOAD.kind = WORKLOAD_KIND;
  WORKLOAD.spec = WORKLOAD_SPEC;
}
if (KERNEL_EXTRACT_ON) log(`Kernel extraction skill enabled (workload=${WORKLOAD_KIND}; verify features from current evidence).`);
// Seed config: an orchestrator that already searched config passes its best flags/env, so the
// baseline is measured on that config, not the stack default (TP/GPU: SERVING_TP / SERVING_GPU).
const INIT_FLAGS = String(A.initial_extra_server_args || '');
const INIT_ENV = String(A.initial_extra_env || '');
// Schema-v2 handoffs may carry the overlay stack behind Hyperloom's current best. It is part of the
// baseline configuration, so it stays underneath every later candidate overlay.
const INIT_BASE_OVERLAY = String(A.initial_overlay_pythonpath || '');
const EFFECTIVE_CONFIG_DIGEST = String(A.effective_config_digest || '');
// One lifecycle for EVERY throughput number (baseline, search A/B, parity, validation), since cold
// and warm numbers aren't comparable. warm_server = Hyperloom's protocol: one server per leg, round
// 1 is a discarded warmup, round 2 IS the number (a trace replay warms each lane itself).
// isolated_server (REPLICAS fresh servers per leg) gates on boot-to-boot variance. The bench
// dispatcher owns retry/degraded aggregation in both modes.
const MEASUREMENT_MODE = String(A.measurement_mode || 'warm_server');
const PARITY_REPLICAS = parseInt(A.parity_replicas != null ? A.parity_replicas : 1, 10);
const SEARCH_REPLICAS = parseInt(A.search_replicas != null ? A.search_replicas : 1, 10);
const VALIDATION_REPLICAS = parseInt(A.validation_replicas != null ? A.validation_replicas : 3, 10);
// Validation has its own mode so the final re-measure can be stricter (isolated_server +
// validation_replicas=3 adds the boot-to-boot dispersion test). Default warm_server: isolated
// validation is 6-12 serialized cold boots and can overrun FINAL_RESERVE_MS.
const VALIDATION_MEASUREMENT_MODE = String(A.validation_measurement_mode || 'warm_server');
// Samples follow the mode: warm_server is one timed round per leg; isolated uses validation_replicas.
const VALIDATION_SAMPLES = VALIDATION_MEASUREMENT_MODE === 'warm_server' ? 1 : VALIDATION_REPLICAS;
// CUDA/HIP-graph deployment requirement (derived from the serving config). vllm/sglang/atom capture
// the steady-state decode path into a full graph unless --enforce-eager is set, and a win that relies
// on its own per-call graph wrapper falls back to eager inside it and vanishes e2e. So when graphs are
// on, every kernel-optimize task is told the win must be intrinsic and graph-capture-safe.
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
// Acceptance noise band (%): 0.5 for a fixed-shape workload (identical work per replica, isolated
// measurements, non-overlap and engagement proof). A trace replay varies with which trajectories a
// replica completes, so it gets a 4% floor; a TTFT-inclusive axis needs an explicit noise_band_pct.
const NOISE_BAND_AGENTX = 4.0;
const NOISE_BAND_DEFAULT = parseFloat(
  A.noise_band_pct != null ? A.noise_band_pct : (IS_AGENTX ? NOISE_BAND_AGENTX : 0.5));
// The gate band from the Director's value. The Director starts at 0.5, so on a trace replay it may
// only raise the band above the workload floor; a fixed-shape run uses the Director's value.
function gateNoiseBand(reported) {
  const r = parseFloat(reported);
  const director = Number.isFinite(r) && r > 0 ? r : 0;
  if (!IS_AGENTX) return director || NOISE_BAND_DEFAULT;
  return Math.max(director, NOISE_BAND_DEFAULT);
}
// No timed-repeat knob: bench_e2e.sh derives rounds from GEAK_REPEAT_MODE + MEASUREMENT_PURPOSE.
// Every integrate A/B must measure both legs: on gate:'incomplete' / ab_complete:false the
// integrator is re-invoked up to this many times to finish the missing leg, then the candidate is
// carried as a pending integration.
const AB_FINISH_RETRIES = parseInt(A.ab_finish_retries != null ? A.ab_finish_retries : 3, 10);
// A resolvable FROZEN baseline (seeded baseline_overlay/ + meta.candidate_bind, or an importable
// meta.baseline_callable on the op track) is the mandatory speedup denominator; without it the
// unittest would time the candidate against its own naive scaffold. Re-extract up to this many
// times, then treat the extraction as failed (flag if dominant, else skip).
const BASELINE_EXTRACT_RETRIES = parseInt(A.baseline_extract_retries != null ? A.baseline_extract_retries : 3, 10);
// Capture storage bounds forced into every extraction EXTRA_ENV (override via args).
const CAPTURE_BYTE_BUDGET = String(A.capture_byte_budget != null ? A.capture_byte_budget : '8GiB');
const CAPTURE_CASE_BYTE_LIMIT = String(A.capture_case_byte_limit != null ? A.capture_case_byte_limit : '2GiB');
const CAPTURE_PERSIST_POLICY = String(A.capture_persist_policy != null ? A.capture_persist_policy : (KERNEL_EXTRACT_ON ? 'structure_only' : 'share_large'));
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

