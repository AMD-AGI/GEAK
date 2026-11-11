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
  // Deployment dimensions the KB addresses a page by, from the Director's preflight. gfx, product
  // target and physical CU count are required (device_name is optional telemetry). Never guess: the
  // service has no DELETE, so a wrong page is permanent.
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

// Returned after source extraction and GPU UT validation. Ranking metadata stays in files, but the
// finished envelope travels in the reply (this script can't read files; see RUNTIME SANDBOX
// CONTRACT): `prepared` is the verbatim, gated stdout of `agentx_kernel_preparation.mjs
// --emit-prepared`, shaped like STRATEGY_SCHEMA's queues.
const AGENTX_PREPARATION_SCHEMA = obj({
  delivery_path: { type: 'string' },
  prepared: obj({
    head_candidates: arrObj, kernel_candidates: arrObj,
    failed_tasks: arrObj, deferred_tasks: arrObj,
    profile_workload_json: { type: 'string' }, kernel_ranking_json: { type: 'string' },
    kernel_extraction_delivery_json: { type: 'string' },
  }, ['head_candidates', 'kernel_candidates']),
}, ['delivery_path', 'prepared']);

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
  // Filled only by the warm-start repair pass (so not `required`): the structured "why did this not
  // run here", read instead of regexing the notes.
  repair: obj({
    attempted: { type: 'boolean' },
    classification: { type: 'string' },   // upstream_gone | conflicts_with_baseline | env_unsupported | mixed | unknown
    dropped_flags: { type: 'array', items: { type: 'string' } },
    reason_by_flag: { type: 'object', additionalProperties: true },
    launch_error: { type: 'string' },
  }, []),
}, ['accepted_flags', 'best_throughput_tok_s']);

// `e2e_store.py resolve` output, passed through verbatim. Nothing is required or typed as a number:
// an unwritten page answers with empty candidates + `read_reason`, and `speedup` can be null.
const KB_RESOLVE_SCHEMA = obj({
  tried: arrStr, canonical_id: { type: 'string' }, match_tier: { type: 'string' },
  ranked_by: { type: 'string' }, candidates: arrObj, read_reason: { type: 'string' },
  // `plane` is what was asked for; `read_plane` is which one answered (they differ under `both`).
  plane: { type: 'string' }, read_plane: { type: 'string' },
  // How the offer is ordered (throughput, high to low) vs the metric the rung crowns its champion
  // on; a coarse rung is ordered by throughput but still promotes on speedup.
  sorted_by: { type: 'string' }, champion_metric: { type: 'string' },
  // Why a rung came back empty, in counts ("never recorded" vs "filtered by the floor"). Its
  // `read_plane` is the first rung that saw anything, not necessarily the one that answered.
  curation: { type: 'object', additionalProperties: true },
}, []);
// Result of the tuning-skillset phase. pre/post are its own isolated-server A/B legs (not the run
// baseline), so tuning_delta_pct is an attributable share. No accept is banked without
// engagement_verified.
const TUNING_SCHEMA = obj({
  ran: { type: 'boolean' }, mode: { type: 'string' }, skills_used: arrStr,
  preflight: { type: 'object', additionalProperties: true },
  ops_tuned: arrObj, artifacts: arrStr,
  // Deploy bundle for tuned DATA (config tables a library reads from its package dir, caches to
  // drop), which the PYTHONPATH overlay can't carry: EVAL_DIR/tuning/deploy/{MANIFEST.json,
  // tuning_patch.diff,files/,deploy.sh}, folded into EVAL_DIR/final/ by Finalize.
  deploy_bundle: { type: 'string' }, deploy_verified: { type: 'boolean' },
  cache_invalidation: arrStr,
  // DATA paths the deploy writes inside an installed package tree (e.g. aiter/configs/*.csv): the
  // carve-out from the Integrator's clean-live-tree check. Data only; code goes in apply_overlay.
  live_tree_files: arrStr,
  // Code a tuned artifact needs in order to bind (e.g. a routing switch), shipped as a reversible
  // overlay like a head env-winner's routing patch, never as a live-tree .py edit.
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

// Extraction smoke gate: exactly lowercase "pass" or "fail" (every gate tests `=== 'pass'`); the
// enum makes StructuredOutput reject free text, which belongs in smoke_detail / notes.
const SMOKE_STATUS = { type: 'string', enum: ['pass', 'fail'] };
// Large machine verdicts travel as immutable task-local files, not LLM summaries.
const SELECTION_VERDICT_REF_SCHEMA = obj({
  path: { type: 'string' }, sha256: { type: 'string' },
}, ['path', 'sha256']);
const EXTRACTION_SELECTION_FIELDS = {
  selection_validation_ref: SELECTION_VERDICT_REF_SCHEMA,
  source_delivery_ref: SELECTION_VERDICT_REF_SCHEMA,
  joint_ut: { type: 'object', additionalProperties: true },
  phase_selection_validations: { type: 'array', items: obj({
    regime: { type: 'string' }, target_callable: { type: 'string' }, seam_candidates: arrObj,
    selection_validation: { type: 'object', additionalProperties: true },
    selection_validation_ref: SELECTION_VERDICT_REF_SCHEMA,
  }, ['regime', 'target_callable']) },
};

const EXTRACT_OP_SCHEMA = obj({
  ...EXTRACTION_SELECTION_FIELDS,
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
  // null when nothing was timed (0.0 would read as "not faster"); gate on `measured`, not the number.
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
  ...EXTRACTION_SELECTION_FIELDS,
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

const KERNEL_LAYER_SCHEMA = obj({
  ran: { type: 'boolean' }, kernel_eval_dir: { type: 'string' },
  final_patch: { type: 'string' }, final_geomean: { type: 'number' },
  validation_status: { type: 'string' }, note: { type: 'string' },
}, ['ran', 'final_patch', 'final_geomean']);

const MEASUREMENT_EVIDENCE_SCHEMA = obj({
  measurement_id: { type: 'string' }, candidate_digest: { type: 'string' },
  reference_digest: { type: 'string' }, workload_digest: { type: 'string' }, protocol_digest: { type: 'string' },
  ref_summary: { type: 'string' }, cand_summary: { type: 'string' },
  ref_replicas: { type: 'number' }, cand_replicas: { type: 'number' },
  fresh_pair: { type: 'boolean' }, configs_matched: { type: 'boolean' }, workload_matched: { type: 'boolean' },
}, []);
const INTEGRATE_SCHEMA = obj({
  short_name: { type: 'string' }, provenance_ok: { type: 'boolean' },
  isolated_speedup: { type: 'number' }, pct_gpu_time: { type: 'number' },
  affected_gpu_time: obj({
    candidate_digest: { type: 'string' },
    profile_path: { type: 'string' }, reference_digest: { type: 'string' }, workload_digest: { type: 'string' },
    kernels: { type: 'array', items: obj({
      id: { type: 'string' }, pct_gpu_time: { type: 'number' }, isolated_speedup: { type: 'number' },
      removed: { type: 'boolean' },
    }, ['id', 'pct_gpu_time']) },
  }, ['profile_path', 'candidate_digest', 'reference_digest', 'workload_digest', 'kernels']),
  measurement_evidence: MEASUREMENT_EVIDENCE_SCHEMA,
  speedup_review: obj({
    status: { type: 'string', enum: ['pending', 'confirmed', 'inconclusive', 'rejected'] },
    original_measurement: MEASUREMENT_EVIDENCE_SCHEMA,
    estimate: { type: 'object', additionalProperties: true },
  }, ['status']),
  e2e_throughput_tok_s: { type: 'number' }, e2e_delta_pct: { type: 'number' },
  // ab_complete=true only when both serving legs were measured; a timed-out or hung gate returns
  // gate:'incomplete' (ab_complete=false) with any partial ref_med / cand_med — not a rejection.
  ref_med: { type: 'number' }, cand_med: { type: 'number' },
  ab_complete: { type: 'boolean' },
  output_parity: { type: 'string' },
  // How an accept's correctness was established: 'byte_exact' (sampled greedy byte parity vs the
  // original baseline), 'accuracy' (sampled task-accuracy probe) or 'none'. Neither sampled check
  // proves correctness for every input.
  parity_kind: { type: 'string' },
  gate: { type: 'string', enum: ['accepted', 'stack', 'rejected', 'incomplete'] },
  accepted_overlay: { type: 'string' }, reason: { type: 'string' },
}, ['gate', 'e2e_throughput_tok_s']);

const FINALIZE_SCHEMA = obj({
  final_overlay: { type: 'string' }, final_patch: { type: 'string' }, final_launch_script: { type: 'string' },
  final_throughput_tok_s: { type: 'number' }, throughput_speedup: { type: 'number' },
  accepted_kernels: arrStr, accepted_config: { type: 'object', additionalProperties: true }, note: { type: 'string' },
  // Whether the tuning deploy bundle reached final/ and still engages there (reported, not assumed:
  // a silently failed data install reproduces none of the tuning gain).
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
