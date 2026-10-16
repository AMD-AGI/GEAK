// ---------------------------------------------------------------------------
// Schemas (obj/arrStr/arrObj: src/shared/helpers.js).
// ---------------------------------------------------------------------------
const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, model_name: { type: 'string' },
  baseline_throughput_tok_s: { type: 'number' }, baseline_spread_pct: { type: 'number' },
  noise_band_pct: { type: 'number' }, baseline_summary_path: { type: 'string' },
  server_flags: { type: 'object', additionalProperties: true }, server_env: { type: 'string' },
  tp: { type: 'number' }, workload: { type: 'object', additionalProperties: true },
  bench_script: { type: 'string' }, notes: { type: 'string' },
  // The four deployment dimensions the KB addresses a page by, beyond model/framework/workload which
  // this script already knows. The Director establishes all four during its existing preflight (it
  // has to, to launch the server at all) and previously just discarded them. gfx, product target,
  // and physical CU count are required; device_name remains optional telemetry. A GUESS is not
  // acceptable: the service has no DELETE, so a wrong-but-authoritative-looking page is permanent.
  gfx: { type: 'string' }, device_target: { type: 'string' },
  device_name: { type: 'string' }, physical_cu_count: { type: 'number' },
  precision: { type: 'string' },
  framework_version: { type: 'string' }, rocm_version: { type: 'string' },
}, ['eval_dir', 'baseline_throughput_tok_s', 'gfx', 'device_target', 'physical_cu_count']);

// admitHeads adopts entity_kind from these rows, so a row without it silently refuses every head.
// device_kernel is required on gpu_kernel rows too; admitHeads checks that (no conditional schema here).
const PROFILE_ROW_SCHEMA = obj({
  entity_kind: { type: 'string',
    enum: ['gpu_kernel', 'memory_op', 'dispatcher_op', 'python_launcher', 'unresolved'] },
  device_kernel: { type: 'string' },
}, ['entity_kind']);
const PROFILE_SCHEMA = obj({
  round: { type: 'number' }, profile_topN_json: { type: 'string' }, profile_topN_md: { type: 'string' },
  profile_workload_json: { type: 'string' }, // per-(shape,dtype) weighted workload model (optional)
  source: { type: 'string' }, total_gpu_time_ms: { type: 'number' },
  top_kernels: { type: 'array', items: PROFILE_ROW_SCHEMA },
  shift_note: { type: 'string' }, notes: { type: 'string' },
}, ['profile_topN_json', 'top_kernels']);

const STRATEGY_SCHEMA = obj({
  regime_summary: { type: 'string' }, config_directions: arrObj,
  head_candidates: { type: 'array', items: obj({
    short_name: { type: 'string' }, entity_kind: { type: 'string' }, device_kernel: { type: 'string' },
  }, ['short_name', 'entity_kind', 'device_kernel']) },
  kernel_candidates: arrObj,
  drop_list: arrObj, order_of_work: arrStr, strategy_path: { type: 'string' },
}, ['kernel_candidates']);

// Result of the ensure_flydsl provisioning gate (build-on-demand). ok=true iff flydsl is importable
// (flydsl + kernels.moe_gemm_2stage) after sourcing env_file; built distinguishes a fresh build from reuse.
const FLYDSL_GATE_SCHEMA = obj({
  ok: { type: 'boolean' }, built: { type: 'boolean' },
  env_file: { type: 'string' }, version: { type: 'string' }, reason: { type: 'string' },
}, ['ok']);

const SWEEP_SCHEMA = obj({
  trials: arrObj, accepted_flags: { type: 'string' }, accepted_env: { type: 'string' },
  best_throughput_tok_s: { type: 'number' }, throughput_speedup_vs_baseline: { type: 'number' },
  summary: { type: 'string' },
  // Only the warm-start repair pass fills this, and it stays OUT of `required` so every ordinary
  // sweep keeps validating unchanged. It is the structured half of "why did this not run here" —
  // read in place of guessing at the free-text notes with a regex.
  repair: obj({
    attempted: { type: 'boolean' },
    classification: { type: 'string' },   // upstream_gone | conflicts_with_baseline | env_unsupported | mixed | unknown
    dropped_flags: { type: 'array', items: { type: 'string' } },
    reason_by_flag: { type: 'object', additionalProperties: true },
    launch_error: { type: 'string' },
  }, []),
}, ['accepted_flags', 'best_throughput_tok_s']);

// `e2e_store.py resolve` output, passed through VERBATIM. Nothing is `required` and no field is
// typed as a number: a page that was never written answers with empty candidates and a
// `read_reason`, and a recorded run legitimately has `speedup: null` when it only ever measured
// absolute throughput. Typing either would turn a truthful answer into a schema failure and a
// cold start.
const KB_RESOLVE_SCHEMA = obj({
  tried: arrStr, canonical_id: { type: 'string' }, match_tier: { type: 'string' },
  ranked_by: { type: 'string' }, candidates: arrObj, read_reason: { type: 'string' },
  // `plane` is what was ASKED for; `read_plane` is which one answered. Under `both` they differ,
  // and only the second one explains where the offer came from.
  plane: { type: 'string' }, read_plane: { type: 'string' },
  // How the offer was ORDERED (throughput, high to low, on every rung) versus which metric that
  // rung crowns its champion on. Two different things that used to share one word: a coarse rung
  // is ranked by absolute throughput here but still promotes on speedup, and a reader told only
  // 'speedup' would mis-explain the order it is looking at.
  sorted_by: { type: 'string' }, champion_metric: { type: 'string' },
  // Why a rung came back empty, in counts — the only thing separating "nobody recorded this
  // deployment" from "the floor ate the records". Named explicitly so the resolver agent cannot
  // summarise it away. Its `read_plane` is the FIRST rung that saw anything, not necessarily the
  // one that answered.
  curation: { type: 'object', additionalProperties: true },
}, []);
// Result of the standalone tuning-skillset phase. pre/post are ITS OWN in-session isolated-server A/B
// legs (NOT the run baseline), which makes tuning_delta_pct an attributable share of the total gain.
// engagement_verified is load-bearing: the skillset's own thesis is that an unproven artifact is not a
// win, so the orchestrator refuses to bank an accept without it.
const TUNING_SCHEMA = obj({
  ran: { type: 'boolean' }, mode: { type: 'string' }, skills_used: arrStr,
  preflight: { type: 'object', additionalProperties: true },
  ops_tuned: arrObj, artifacts: arrStr,
  // Deploy bundle: how a tuned DATA artifact reaches production. GEAK's overlay is a PYTHONPATH
  // mechanism for code, but tuned config tables are data a library reads from its own package dir, and
  // some need a derived cache dropped before they take effect. Neither travels in the overlay — so the
  // role writes EVAL_DIR/tuning/deploy/{MANIFEST.json,tuning_patch.diff,files/,deploy.sh} and Finalize
  // folds it into EVAL_DIR/final/ (diff concatenated, deploy.sh invoked from final_launch.sh).
  deploy_bundle: { type: 'string' }, deploy_verified: { type: 'boolean' },
  cache_invalidation: arrStr,
  // DATA paths the deploy writes INSIDE an installed package tree (e.g. aiter/configs/*.csv). The
  // Integrator forbids live-tree edits and asserts `git status --porcelain` is empty before/after every
  // A/B leg; this list is the carve-out that keeps a banked tuning from being cleaned away. Data only —
  // a source change belongs in apply_overlay, see below.
  live_tree_files: arrStr,
  // A tuned artifact binds only if the live seam dispatches the backend that consumes it, so tuning
  // sometimes needs a CODE change to land (a routing switch, or a wrapper that drops the kernel
  // selection — tuning-aiter/SKILL.md §2b measures a correctly-tuned row going 85.7% SLOWER on a wrapper
  // that ignores kernelName). That is in scope, and it travels as a reversible overlay, exactly like a
  // head env-winner's routing patch: never as a live-tree .py edit, which cannot be varied per A/B leg.
  apply_overlay: { type: 'string' },
  apply_env: { type: 'string' }, apply_flags: { type: 'string' },
  correctness_gate: { type: 'string' }, accuracy_note: { type: 'string' },
  engagement_verified: { type: 'boolean' }, engagement_evidence: { type: 'string' },
  pre_tune_throughput_tok_s: { type: 'number' }, post_tune_throughput_tok_s: { type: 'number' },
  noise_floor_pct: { type: 'number' }, tuning_delta_pct: { type: 'number' },
  tuning_speedup: { type: 'number' },
  ab_interleaved: { type: 'boolean' }, ab_complete: { type: 'boolean' },
  gate: { type: 'string', enum: ['accepted', 'no_win', 'rejected', 'incomplete', 'skipped'] },
  report_path: { type: 'string' }, summary: { type: 'string' }, reason: { type: 'string' },
}, ['gate']);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  config_directions: arrObj, head_candidates: arrObj, kernel_candidates: arrObj,
}, ['stop']);

// Extraction smoke gate: EXACTLY lowercase "pass" or "fail". Every gate below tests `=== 'pass'`, and a
// free string let agents answer "PASS (exit 0) on GPU 1 ..." — a passing unittest then read as a failed
// extraction and the head was silently dropped. The enum makes StructuredOutput reject anything else so
// the agent must re-answer; the prose belongs in smoke_detail / notes.
const SMOKE_STATUS = { type: 'string', enum: ['pass', 'fail'] };

const EXTRACT_OP_SCHEMA = obj({
  short_name: { type: 'string' }, op_kind: { type: 'string' }, editable: { type: 'boolean' },
  task_dir: { type: 'string' }, shapes: { type: 'object', additionalProperties: true },
  workload_path: { type: 'string' }, // per-(shape,dtype) weighted workload model for this kernel (optional)
  dtype: { type: 'string' }, synthesized: { type: 'boolean' }, regimes_captured: arrStr,
  candidate_backends: arrStr, reference_io_sha256: { type: 'string' },
  target_callable: { type: 'string' }, // module:attr rebind seam for an authored kernel ('' if none)
  baseline_callable: { type: 'string' }, // module:attr of the FROZEN real online kernel (the speedup denominator)
  baseline_frozen: { type: 'boolean' }, // true only when baseline_callable resolves outside the task dir
  device_kernel: { type: 'string' },
  seam_candidates: arrObj,
  selection_validation: { type: 'object', additionalProperties: true },
  smoke: SMOKE_STATUS, smoke_detail: { type: 'string' }, notes: { type: 'string' },
}, ['op_kind', 'task_dir', 'smoke']);

const OPBENCH_SCHEMA = obj({
  short_name: { type: 'string' }, op_kind: { type: 'string' }, provenance_ok: { type: 'boolean' },
  winner_backend: { type: 'string' }, winner_kind: { type: 'string' },
  // null when nothing was timed -- a speedup is a measurement, and 0.0 read as
  // 'benched, not faster'. `measured` is the discriminator; gate on it, not on the number.
  isolated_speedup: { type: ['number', 'null'] }, measured: { type: 'boolean' },
  winner_editable: { type: 'boolean' },
  best_known_ms: { type: 'number' },
  recommend_tier_c: { type: 'boolean' }, author_plan: arrObj, tuning_artifact: { type: 'string' },
  apply_env: { type: 'string' }, apply_flags: { type: 'string' }, code_patch: { type: 'string' },
  per_backend: arrObj, parity_note: { type: 'string' },
  gate: { type: 'string', enum: ['have_winner', 'author_recommended', 'no_win', 'harness_error', 'tamper'] },
  harness_suspect: { type: 'boolean' }, reason: { type: 'string' },
}, ['gate', 'isolated_speedup']);

const EXTRACT_SCHEMA = obj({
  short_name: { type: 'string' }, editable: { type: 'boolean' }, task_dir: { type: 'string' },
  source_path_in_sglang: { type: 'string' }, target_callable: { type: 'string' },
  num_cases: { type: 'number' }, regimes_captured: arrStr, candidate_backends: arrStr,
  build: { type: 'boolean' }, unittest_smoke: SMOKE_STATUS, smoke_detail: { type: 'string' },
  // the baseline leg is an ENVIRONMENT (baseline_overlay/ = a frozen CURRENT_OVERLAY snapshot), and
  // candidate_bind is the ONE entry layered on top of it to make the candidate leg.
  candidate_bind: { type: 'object', additionalProperties: true },
  baseline_overlay: { type: 'string' },
  baseline_frozen: { type: 'boolean' }, // true only when baseline_overlay/ was seeded AND candidate_bind is declared
  device_kernel: { type: 'string' },
  seam_candidates: arrObj,
  selection_validation: { type: 'object', additionalProperties: true },
  reference_io_sha256: { type: 'string' }, notes: { type: 'string' },
}, ['editable', 'task_dir', 'unittest_smoke']);

const INTEGRATE_SCHEMA = obj({
  short_name: { type: 'string' }, provenance_ok: { type: 'boolean' },
  isolated_speedup: { type: 'number' }, pct_gpu_time: { type: 'number' },
  e2e_throughput_tok_s: { type: 'number' }, e2e_delta_pct: { type: 'number' },
  // A/B completion signals: the integrator MUST set ab_complete=true ONLY when
  // BOTH the reference and candidate serving legs were measured. When it ran out
  // of time / hung mid-gate it returns gate:'incomplete' (ab_complete=false) with
  // whatever partial legs it has (ref_med / cand_med) — that is NOT a rejection.
  ref_med: { type: 'number' }, cand_med: { type: 'number' },
  ab_complete: { type: 'boolean' },
  output_parity: { type: 'string' },
  // How the correctness of an ACCEPT was established: 'byte_exact' (hard greedy byte-parity vs the true
  // baseline), 'accuracy' (soft sampled task-accuracy probe — quant / accuracy_gate), or 'none'. The
  // implausible-speedup guard only distrusts an 'accuracy'/soft accept; a byte_exact accept is trusted.
  parity_kind: { type: 'string' },
  gate: { type: 'string', enum: ['accepted', 'stack', 'rejected', 'incomplete'] },
  accepted_overlay: { type: 'string' }, reason: { type: 'string' },
}, ['gate', 'e2e_throughput_tok_s']);

const FINALIZE_SCHEMA = obj({
  final_overlay: { type: 'string' }, final_patch: { type: 'string' }, final_launch_script: { type: 'string' },
  final_throughput_tok_s: { type: 'number' }, throughput_speedup: { type: 'number' },
  accepted_kernels: arrStr, accepted_config: { type: 'object', additionalProperties: true }, note: { type: 'string' },
  // Did the tuning deploy bundle make it into final/ AND still engage on the assembled bundle? Reported
  // rather than assumed: a data artifact that silently fails to install would leave a bundle that looks
  // complete and reproduces none of the tuning gain.
  tuning_in_bundle: { type: 'boolean' }, tuning_engagement_recheck: { type: 'string' },
}, ['final_throughput_tok_s']);

const EXPERIENCE_SCHEMA = obj({
  playbook_appended: { type: 'boolean' }, insights: arrStr, ledger: arrObj,
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const REPORT_SCHEMA = obj({
  baseline_throughput_tok_s: { type: 'number' }, final_throughput_tok_s: { type: 'number' },
  throughput_speedup: { type: 'number' }, accepted_config: { type: 'object', additionalProperties: true },
  accepted_kernels: arrObj, milestones: { type: 'number' }, report_path: { type: 'string' },
}, ['throughput_speedup', 'report_path']);

const VALIDATE_SCHEMA = obj({
  model_name: { type: 'string' }, baseline_throughput_tok_s: { type: 'number' },
  director_verified_throughput_tok_s: { type: 'number' }, throughput_speedup: { type: 'number' },
  claimed_throughput_tok_s: { type: 'number' }, validation_status: { type: 'string' },
  output_parity: { type: 'string' }, applied_to_original: { type: 'string' },
  final_overlay: { type: 'string' }, final_launch_script: { type: 'string' },
  arbitration_note: { type: 'string' },
}, ['director_verified_throughput_tok_s', 'validation_status']);

const CHECKPOINT_WRITE_SCHEMA = obj({
  written: { type: 'boolean' }, path: { type: 'string' }, checkpoint_sha256: { type: 'string' },
}, ['written', 'path', 'checkpoint_sha256']);

