// ---------------------------------------------------------------------------
// Workload IDENTITY — the synthetic sweep, or a trace replay that owns its load
// ---------------------------------------------------------------------------
// GEAK runs both under an orchestrator and standalone. Under Hyperloom,
// interface/run_e2e.py reads handoff.workload_spec and EXPORTS the resulting
// BENCH_CLIENT / E2E_METRIC / AGENTX_* into the environment GEAK inherits, so
// this script never had to know what it was measuring. Standalone there is no
// such parent: the args land here directly, and without the block below an
// AgentX request would benchmark a synthetic ISL/OSL sweep on the output-token
// axis and call it an AgentX number.
//
// So the workload is declared HERE, in either of two spellings:
//   * args.workload_spec — the SAME object Hyperloom puts on its handoff, for
//     callers that want every field explicit; or
//   * args.workload_kind: "agentx_trace_replay" — the shorthand, which fills
//     the canonical AgentX setup below and lets workload_spec override any part.
// Neither present => 'synthetic', AGENTX_ENV is empty, no bench_env.sh is
// written, and the fixed ISL/OSL path is byte-identical to a build without this.
const WORKLOAD_SPEC = (A.workload_spec && typeof A.workload_spec === 'object') ? A.workload_spec : {};
const WORKLOAD_KIND = String(A.workload_kind || WORKLOAD_SPEC.kind || 'synthetic');
const IS_AGENTX = WORKLOAD_KIND === 'agentx_trace_replay';
// The canonical AgentX submission setup. These are the leaderboard's numbers,
// not tuning knobs: the 393-entry weka corpus at 3600s IS the graded workload,
// and a run that deviates is stamped non-canonical by the client adapter and
// can never be KEPT. The 900s loop duration is the scenario's own floor and is
// what the inner search legs run, which is why they are always non-canonical.
const AGENTX_DEFAULTS = {
  scenario: 'inferencex-agentx-mvp',
  // The full-context parent, not the _256k sibling: the latter drops every
  // request over 256k tokens (98.8k -> 68.3k) for ~256k-context servers, and
  // the campaign serves at --max-model-len 1048576.
  corpus: 'semianalysis_cc_traces_weka_062126',
  num_entries: 393,
  duration_s: 3600,           // canonical/parity window
  geak_loop_duration_s: 900,  // inner search legs (scenario floor)
  warmup_requests_per_lane: 10,
  warmup_grace_period_s: 1800,
  // Failed-request tolerance is deliberately NOT defaulted here. Pinning one number for the whole
  // run flattens the distinction between a leg that is exploring and a leg whose number will be
  // compared, and the client is the only place that knows which one it is running
  // (MEASUREMENT_PURPOSE). Declaring a value still overrides the client, for a run that needs a
  // specific tolerance end to end.
  failed_request_threshold: null,
  // InferenceX's P90 interactivity, 1000 / P90(ITL) in tok/s/user: the x-axis of the public
  // InferenceX pareto, decode-only (TTFT and queue wait are outside it). bench_summarize.py
  // reports total tok/s beside it as a guard. Hyperloom declares its own basis on the handoff
  // and exports the matching E2E_METRIC, so this default only decides a run nobody configured.
  metric_basis: 'p90_intvty_inferencex',
};
const AGENTX = IS_AGENTX ? Object.assign({}, AGENTX_DEFAULTS, WORKLOAD_SPEC) : null;
// ONE concurrency for the run. Roles put WORKLOAD.conc on every bench line as CONC=<conc>, and a
// command-line value outranks bench_env.sh, so a workload_spec.concurrency that reached only the
// declaration would be replaced by args.conc (or the synthetic 64) on every bench.
const CONC = parseInt(AGENTX && AGENTX.concurrency != null ? AGENTX.concurrency
  : (A.conc != null ? A.conc : 64), 10);

