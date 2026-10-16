# Config Tuner

Improve serving configuration through measured flag/env changes. Do not rewrite kernels or perform
per-operation GEMM tuning; those belong to the head track. `CONFIG_TUNE_ENABLED=false` disables this
exploratory sweep, not a kernel candidate's required backend-selection env/overlay.

Consult relevant `SKILL_DIR/knowledge/{e2e_optimization,sglang_internals,backend_playbook}.md` and
backend-specific references. When `E2E_LEARNED_KB=on`, relevant learned cards may add candidates.
When `E2E_LEARNED_KB=off`, do not open, list or cite `knowledge/learned/`.

## PHASE=sweep

Inputs: `EVAL_DIR`, `MODEL_PATH`, `BACKEND`, `GPU_ID`, `SERVING_TP`, `SERVING_GPU`, `WORKLOAD`,
`BASELINE_THROUGHPUT`, `NOISE_BAND_PCT`, `CONFIG_DIRECTIONS`, `CURRENT_FLAGS`, `CURRENT_ENV`,
`CURRENT_OVERLAY`, `MEASUREMENT_MODE`, `MEASUREMENT_PURPOSE`, `REPLICAS`, `EFFECTIVE_CONFIG_DIGEST`,
`ENABLE_FP8`, `SKILL_DIR`; warm-start replay may add `MERGE_OVERRIDES` / `MERGE_ADDED`.

For each ranked direction:
1. Add one configuration axis to the currently accepted stack. Merge flags/env by key, replacing
   existing values; do not concatenate duplicate keys. Only genuinely repeatable flags may repeat.
2. Use the run's shared benchmark with the exact backend, serving GPU/TP, workload, overlay and
   measurement lifecycle. Preserve caller-owned client/protocol settings, including `bench_env.sh`.
   ```bash
   BACKEND="$BACKEND" OUT_DIR="$EVAL_DIR/config/<dir_id>" GPU="$SERVING_GPU" TP="$SERVING_TP" MODEL="$MODEL_PATH" \
   ISL=<isl> OSL=<osl> CONC=<conc> PROFILE=0 \
   GEAK_REPEAT_MODE="$MEASUREMENT_MODE" MEASUREMENT_PURPOSE=search REPLICAS="${REPLICAS:-1}" \
   EFFECTIVE_CONFIG_DIGEST="$EFFECTIVE_CONFIG_DIGEST" OVERLAY_PYTHONPATH="$CURRENT_OVERLAY" \
   EXTRA_SERVER_ARGS="<complete candidate flags>" EXTRA_ENV="<complete candidate env>" \
     bash "$EVAL_DIR/bench_e2e.sh" 2>&1 | tee "$EVAL_DIR/logs/cfg_<dir_id>.log"
   ```
3. Read the summary's declared metric basis, median, completion and acceptance fields. Compare
   `(candidate_median-current_median)/current_median*100`. Missing/partial measurements are not wins.
4. Prove the requested value/backend actually engaged from runtime evidence. For changes that can
   alter numerics, compare fixed-seed greedy outputs against baseline. Intentional quantization
   requires the supplied task-accuracy gate/tolerance; do not invent an acceptable quality loss.
   With `CORRECTNESS_UNSET_ENV`, remove those names from process env AND EXTRA_ENV for both
   correctness servers; throughput keeps them and must use separate launches.
5. Keep only an engaged, correct change whose gain exceeds `NOISE_BAND_PCT`; otherwise retain current
   config. Accepted changes become the reference for later directions. Record every trial, including
   failures, in `EVAL_DIR/config/sweep_results.json`.

Supported axes depend on the installed backend: attention backend, graph/compile, scheduling,
memory/KV settings and stack-level backend switches. FP8 changes require `ENABLE_FP8=true` and an
accuracy gate. GEMM DB/config tuning belongs to Op Benchmarker. Never change the required workload.

### Warm-start replay and repair

An axis named `compound (recovered configuration — do not split)` arrives already merged. Run its
complete `flags`/`env` verbatim; do not append the current strings or split the compound record.
Check `MERGE_OVERRIDES` (`flag`, `baseline_value`, `recalled_value`) and `MERGE_ADDED`, and verify the
recalled values actually took effect.

For an explicit repair pass after a failed launch, inspect the actual error. Classify unsupported
knobs as `upstream_gone`, `conflicts_with_baseline`, `env_unsupported`, or a mixed/unknown case.
Adapt/drop only what is necessary, benchmark, and disclose every change in `repair`. If it still
cannot run, return `best_throughput_tok_s:0` with the launch error; a partial replay is not measurement
of the complete recalled record. Include `repair` only on this repair path.

Return the full final accepted flags/env (the caller replaces its current values with them), not
just this sweep's additions. Keep rejected trial evidence separate.

Return JSON:
```json
{
  "trials": [
    {"id": "cfg0", "axis": "attention-backend", "change": "--attention-backend aiter",
     "throughput_tok_s": 0.0, "delta_pct": 0.0, "parity": "pass|fail|n/a",
     "swap_verified": true, "kept": true, "note": "..."}
  ],
  "accepted_flags": "<final kept extra server flags>",
  "accepted_env": "<final kept extra env KEY=VAL ...>",
  "best_throughput_tok_s": 0.0,
  "throughput_speedup_vs_baseline": 1.0,
  "summary": "what worked, what didn't, what to re-profile against",
  "repair": {
    "attempted": true,
    "classification": "upstream_gone|conflicts_with_baseline|env_unsupported|mixed|unknown",
    "dropped_flags": ["--flag"],
    "reason_by_flag": {"--flag": "why it had to go"},
    "launch_error": "the line that actually killed the server"
  }
}
```
