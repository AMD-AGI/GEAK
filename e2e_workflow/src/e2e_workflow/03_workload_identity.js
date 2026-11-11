// Workload IDENTITY — the synthetic sweep, or a trace replay that owns its load. Under Hyperloom,
// run_e2e.py exports BENCH_CLIENT / E2E_METRIC / AGENTX_* from the handoff. Standalone, declare it
// as args.workload_spec (the handoff's object) or args.workload_kind="agentx_trace_replay" (the
// canonical setup below; workload_spec overrides any part). AgentX extraction resumes its saved
// declaration when no new workload is supplied.
const WORKLOAD_SPEC = A.workload_spec && typeof A.workload_spec === 'object'
  ? A.workload_spec
  : ((!A.workload_kind || A.workload_kind === 'agentx_trace_replay') &&
      A.state?.workload_spec?.kind === 'agentx_trace_replay'
    ? A.state.workload_spec : {});
const WORKLOAD_KIND = String(A.workload_kind || WORKLOAD_SPEC.kind || 'synthetic');
const IS_AGENTX = WORKLOAD_KIND === 'agentx_trace_replay';
// The canonical AgentX submission setup: leaderboard values, not tuning knobs. A deviating run is
// stamped non-canonical and never KEPT; the inner search legs run the scenario's 900s floor, so
// they are always non-canonical.
const AGENTX_DEFAULTS = {
  scenario: 'inferencex-agentx-mvp',
  // The full-context corpus, not the _256k sibling (which drops requests over 256k tokens);
  // the campaign serves at --max-model-len 1048576.
  corpus: 'semianalysis_cc_traces_weka_062126',
  num_entries: 393,
  duration_s: 3600,           // canonical/parity window
  geak_loop_duration_s: 900,  // inner search legs (scenario floor)
  warmup_requests_per_lane: 10,
  warmup_grace_period_s: 1800,
  // Not defaulted: the client picks the tolerance per leg (MEASUREMENT_PURPOSE: exploring vs
  // compared). A declared value overrides it for the whole run.
  failed_request_threshold: null,
  // InferenceX P90 interactivity, 1000 / P90(ITL) tok/s/user (decode-only; bench_summarize.py
  // also reports total tok/s as a guard). Hyperloom declares its own basis; this is the default.
  metric_basis: 'p90_intvty_inferencex',
};
const AGENTX = IS_AGENTX ? Object.assign({}, AGENTX_DEFAULTS, WORKLOAD_SPEC) : null;
// ONE concurrency for the run: roles pass CONC=<WORKLOAD.conc> on every bench line, which outranks
// bench_env.sh, so workload_spec.concurrency must win here.
const CONC = parseInt(AGENTX && AGENTX.concurrency != null ? AGENTX.concurrency
  : (A.conc != null ? A.conc : 64), 10);

