// ---------------------------------------------------------------------------
// Schemas (obj/arrStr/arrObj: src/shared/helpers.js).
// ---------------------------------------------------------------------------
// oracle_freezer output — the standalone counterpart of e2e's kernel_extractor.extract_op, built from a
// kernel dir instead of a live server. Same op-task-dir contract; also creates the run dir (EVAL_DIR).
const FREEZE_SCHEMA = obj({
  eval_dir: { type: 'string' },            // the isolated run dir this freeze created under EXP_ROOT
  op_kind: { type: 'string' },             // gemm|attn|elementwise|moe|other
  task_dir: { type: 'string' },            // the immutable op task dir (unittest.py + meta.json + baseline_src/)
  live_backend: { type: 'string' },        // the input kernel's language, e.g. "hip"
  candidate_backends: arrStr,
  baseline_frozen: { type: 'boolean' },
  baseline_callable: { type: 'string' },
  device_gfx: { type: 'string' },
  device_target: { type: 'string' },
  device_name: { type: 'string' },
  physical_cu_count: { type: 'number' },
  // Always "" from oracle_freezer — a freezer-built task dir records NO golden tensors (correctness is
  // live parity vs baseline_src/). Kept in the schema because an e2e kernel_extractor task dir, which
  // captures unsynthesizable real routing / paged-KV metadata, does ship a reference_io.pt and fills it.
  reference_io_sha256: { type: 'string' },
  op_spec: { type: 'object', additionalProperties: true },
  workload_path: { type: 'string' },
  smoke: { type: 'string' },               // "pass" required to proceed
  notes: { type: 'string' },
}, ['op_kind', 'task_dir', 'smoke']);

// op_benchmarker output — the role file is reused VERBATIM from e2e (referenced in place, never edited);
// the standalone-tune behavior is steered entirely by the discover prompt below. `baseline_ms` +
// `tuned_speedup` are extra fields this dispatcher asks the agent to fill (in the prompt) so a tuned env
// backend can be ranked against the SAME frozen input-kernel baseline as the author/optimize lanes.
const OPBENCH_SCHEMA = obj({
  short_name: { type: 'string' }, op_kind: { type: 'string' }, provenance_ok: { type: 'boolean' },
  winner_backend: { type: 'string' }, winner_kind: { type: 'string' },
  // null when nothing was timed -- a speedup is a measurement, and 0.0 read as
  // 'benched, not faster'. `measured` is the discriminator; gate on it, not on the number.
  isolated_speedup: { type: ['number', 'null'] }, measured: { type: 'boolean' },
  winner_editable: { type: 'boolean' },
  best_known_ms: { type: 'number' },
  baseline_ms: { type: 'number' },     // ms of the FROZEN input kernel (baseline_src/) on the same oracle
  tuned_speedup: { type: 'number' },   // best tuned env backend's speedup vs the FROZEN baseline (baseline_ms/tuned_ms)
  recommend_tier_c: { type: 'boolean' }, author_plan: arrObj, tuning_artifact: { type: 'string' },
  apply_env: { type: 'string' }, apply_flags: { type: 'string' }, code_patch: { type: 'string' },
  per_backend: arrObj, parity_note: { type: 'string' },
  gate: { type: 'string' }, harness_suspect: { type: 'boolean' }, reason: { type: 'string' },
}, ['gate', 'isolated_speedup']);

const REPORT_SCHEMA = obj({
  report_path: { type: 'string' }, applied_to_original: { type: 'string' }, note: { type: 'string' },
}, ['report_path']);

