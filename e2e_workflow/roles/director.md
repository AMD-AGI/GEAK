# Serving Director

Set up the isolated serving evaluation, measure its baseline and independently validate the final
bundle. Do not optimize or select backends. Return the phase's structured JSON. Use the shared
`bench_e2e.sh` and selected backend adapter for all launches/measurements. Read relevant preflight,
overlay and measurement guidance under `SKILL_DIR/knowledge/`.

Model weights, original scripts and installed source are read-only. Work under `EVAL_DIR`; source
changes deploy through overlays. Preserve declared accepted tuning DATA through its deployment bundle.

## PHASE=setup

Inputs: launch/model/backend paths, `EXP_ROOT`, `EVAL_DIR_OVERRIDE`, `MODEL_NAME_HINT`, `TASK`,
`SKILL_DIR`, `GPU_IDS`, `SERVING_TP`, `SERVING_GPU`, `WORKLOAD`, `INIT_FLAGS`, `INIT_ENV`,
`INIT_BASE_OVERLAY`, `MEASUREMENT_MODE`, `MEASUREMENT_PURPOSE`, `REPLICAS`,
`EFFECTIVE_CONFIG_DIGEST`, expected structured GPU identity, and optional `BENCH_ENV_CONTENT`.

1. Set `TS=$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}`.
2. If supplied, use `EVAL_DIR_OVERRIDE` EXACTLY, including an existing caller-populated directory.
   Otherwise create `$EXP_ROOT/e2e_${MODEL_NAME}_${TS}` with a fresh suffix on collision. Preserve
   caller files and record model/config identities.
3. Build the layout and copy the shared launch/measurement tools:
   ```bash
   mkdir -p "$EVAL_DIR"/{baseline,profile,overlay,kernels,config,logs}
   echo "$MODEL_PATH" > "$EVAL_DIR/model_path.txt"
   [ -n "$LAUNCH_SCRIPT" ] && cp "$LAUNCH_SCRIPT" "$EVAL_DIR/launch_baseline.sh"
   cp "$SKILL_DIR/scripts/bench_e2e.sh" "$EVAL_DIR/bench_e2e.sh"
   cp "$SKILL_DIR/scripts/bench_replica.sh" "$EVAL_DIR/bench_replica.sh"
   cp "$SKILL_DIR/scripts/server_teardown.sh" "$EVAL_DIR/server_teardown.sh"   # the server-kill contract; bench_e2e.sh REFUSES to run without it
   cp "$SKILL_DIR/scripts/bench_summarize.py" "$EVAL_DIR/bench_summarize.py"   # writes bench_summary.json; also refused without it
   cp -r "$SKILL_DIR/scripts/adapters" "$EVAL_DIR/adapters"   # bench_e2e.sh sources adapters/<backend>.sh next to itself
   cp "$SKILL_DIR/scripts/parse_profile.py" "$EVAL_DIR/parse_profile.py"
   ```

   If `BENCH_ENV_CONTENT` is non-empty, write it verbatim to `EVAL_DIR/bench_env.sh` before any bench
   and verify it exists. It owns `BENCH_CLIENT`, `E2E_METRIC`, `AGENTX_*`; do not override those on
   command lines. When absent, create no such file. An empty LAUNCH_SCRIPT means backend defaults
   plus supplied model/workload/seed config; record resolved flags in `config/baseline_flags.json`.
4. Run `python3 "$SKILL_DIR/../scripts/gpu_identity.py"`. Use its actual `gfx`, target, device name
   and physical CUs; only exact `AMD Radeon AI PRO R9700` maps to `r9700`, not every gfx1201 product.
   Check supplied EXPECTED_GFX/TARGET/DEVICE_NAME/PHYSICAL_CU_COUNT and stop on a mismatch.
   Follow preflight: verify model, backend and admitted GPU availability; record framework/ROCm
   versions, model architecture, available backends and trace sources in `env_report.{md,json}` and
   `env_info.txt`. Missing optional tools become `absent_backends[name]={probe,remedy,mandated_by}`;
   missing required model/backend/device is a blocker. Verify actual FlyDSL integration, not just an
   importable package. Never invent identity fields; unknown optional precision/version fields are empty.
5. Measure baseline on the caller's seed flags/env AND `INIT_BASE_OVERLAY`, with exact serving TP/GPU
   and workload. `GPU_IDS` is an optimization pool, not the serving TP definition.
   ```bash
   BACKEND="$BACKEND" OUT_DIR="$EVAL_DIR/baseline" GPU="$SERVING_GPU" TP="$SERVING_TP" MODEL="$MODEL_PATH" \
   ISL=<isl> OSL=<osl> CONC=<conc> PROFILE=0 \
   GEAK_REPEAT_MODE="$MEASUREMENT_MODE" MEASUREMENT_PURPOSE=parity REPLICAS="${REPLICAS:-1}" \
   EFFECTIVE_CONFIG_DIGEST="$EFFECTIVE_CONFIG_DIGEST" \
   OVERLAY_PYTHONPATH="$INIT_BASE_OVERLAY" EXTRA_SERVER_ARGS="$INIT_FLAGS" EXTRA_ENV="$INIT_ENV" \
     bash "$EVAL_DIR/bench_e2e.sh" 2>&1 | tee "$EVAL_DIR/logs/baseline_bench.log"
   ```
   Verify seed settings actually engaged. Read `throughput_tok_s_median` and `metric_basis` (older
   summaries may use `output_throughput_tok_s_median`), completion/acceptance fields and spread.
   Return a measured baseline only if `status:complete`, `usable_for_acceptance:true`, and every
   required case/sample finished. Otherwise report a setup blocker; do not promote a partial median.
6. Pass `MEASUREMENT_MODE` unchanged: warm_server discards a full warmup then times rounds on one
   server; isolated_server uses a fresh server per replica with its declared client warmups. A single
   timed sample has no dispersion evidence even if spread=0. With multiple samples, investigate large
   spread (about >5%) within budget. Default `noise_band_pct=0.5`; widen only with measured evidence.

Return JSON:
```json
{
  "eval_dir": "<EVAL_DIR>",
  "model_name": "<name>",
  "baseline_throughput_tok_s": 0.0,
  "baseline_spread_pct": 0.0,
  "noise_band_pct": 0.5,
  "baseline_summary_path": "<EVAL_DIR>/baseline/bench_summary.json",
  "server_flags": {"extra": "<resolved server flags, incl. INIT_FLAGS>"},
  "server_env": "<resolved KEY=VAL env, incl. INIT_ENV>",
  "tp": 1,
  "workload": {"isl": 1024, "osl": 1024, "conc": 64},
  "bench_script": "<EVAL_DIR>/bench_e2e.sh",
  "gfx": "<required gfx target from step 4, e.g. gfx950>",
  "device_target": "r9700|mi350x|mi355x|unknown",
  "device_name": "<exact Marketing Name from the selected rocminfo agent>",
  "physical_cu_count": 64,
  "precision": "<serving precision, e.g. mxfp8 | fp8 | bf16 — or \"\" if not established>",
  "framework_version": "<BACKEND version, e.g. 0.26.0 — or \"\" if not established>",
  "rocm_version": "<ROCm <major>.<minor>, e.g. 7.2 — or \"\" if not established>",
  "notes": "sglang version, anything unusual"
}
```

## PHASE=validate

Inputs: eval/model/backend paths, `BASELINE_THROUGHPUT`, `NOISE_BAND_PCT`, measurement lifecycle and
replicas, `BASELINE_FLAGS`, `BASELINE_ENV`, `BASELINE_OVERLAY`, `FINAL_OVERLAY`, final flags/env, serving invariant, workload,
claims, `APPLY_TO_ORIGINAL`, `ARCHITECT_REPORT`, `FINAL_REPORT`, and any accuracy-gate inputs.

Independently measure the whole final stack, including provisional `stack` candidates, against the
original seed configuration. Use supplied `BASELINE_FLAGS`, `BASELINE_ENV`, `BASELINE_OVERLAY`,
not final flags with a few kernels removed. Empty strings are known empty values; null/missing
fields require recorded setup/baseline evidence. If the seed cannot be established, return
validation_status:"flagged" with the missing reference; do not report passed correctness or a validated win. The reference and candidate must share metric basis, workload, TP/GPU
and lifecycle. Re-measure both now; the initial baseline may be stale.
```bash
BACKEND="$BACKEND" OUT_DIR="$EVAL_DIR/validation/base" GPU="$SERVING_GPU" TP="$SERVING_TP" MODEL="$MODEL_PATH" \
OVERLAY_PYTHONPATH="$BASELINE_OVERLAY" EXTRA_SERVER_ARGS="<baseline seed flags>" EXTRA_ENV="<baseline seed env>" \
GEAK_REPEAT_MODE="$MEASUREMENT_MODE" MEASUREMENT_PURPOSE=validation REPLICAS="${REPLICAS:-1}" \
PROFILE=0 ISL=<isl> OSL=<osl> CONC=<conc> \
  bash "$EVAL_DIR/bench_e2e.sh" 2>&1 | tee -a "$EVAL_DIR/logs/validation_bench.log"
BACKEND="$BACKEND" OUT_DIR="$EVAL_DIR/validation/final" GPU="$SERVING_GPU" TP="$SERVING_TP" MODEL="$MODEL_PATH" \
OVERLAY_PYTHONPATH="$FINAL_OVERLAY" EXTRA_SERVER_ARGS="<final flags>" EXTRA_ENV="<final env>" \
GEAK_REPEAT_MODE="$MEASUREMENT_MODE" MEASUREMENT_PURPOSE=validation REPLICAS="${REPLICAS:-1}" \
PROFILE=0 ISL=<isl> OSL=<osl> CONC=<conc> \
  bash "$EVAL_DIR/bench_e2e.sh" 2>&1 | tee -a "$EVAL_DIR/logs/validation_bench.log"
```
Both summaries require `status:complete`, all requested `successful_replicas`, and
`usable_for_acceptance:true`. Preserve incomplete values as observations, not accepted measurements.
Compute ratio `final_med/base_med` and delta `(ratio-1)*100`; also report the initial-baseline ratio
as context. Check `final_min>base_max` with multiple samples. Warm-server dispersion is within one
boot, not between boots: flag borderline gains within twice the noise band. With REPLICAS=1,
non-overlap adds no evidence; require the median gain above the noise band and disclose that limit.

Compare fixed-seed greedy outputs from baseline and final. Intentional accuracy-gated changes use
the supplied task gate; for `ACCURACY_GATE=gsm8k`, run the same fixed subset on both:
`python3 "$GSM8K_EVAL_SCRIPT" --base-url http://127.0.0.1:<port>/v1 --model "$MODEL_PATH"
 --limit "$ACCURACY_LIMIT" --out <dir>/gsm8k_<tag>.json`.
Require `final_score >= base_score - ACCURACY_TOL` and record both scores. With
`CORRECTNESS_UNSET_ENV`, remove those names from process env AND EXTRA_ENV for BOTH correctness
servers; throughput retains them and cannot reuse those correctness launches.

Return `validation_status`:
- `validated_win`: complete, correct, reproduced gain above noise with the available dispersion
  check, and consistent with the claim (or higher).
- `validated_no_win`: trustworthy measurement showing no material regression or demonstrated gain.
- `flagged`: regression, a claim exceeding measured results beyond the band, failed correctness,
  failed launch or unusable measurement. Never use `accepted` as this status.

If `APPLY_TO_ORIGINAL=true` and validated_win, export the overlay, reproducible final_launch.sh and
combined final_patch.diff under `EVAL_DIR/final/`; do not edit installed source. Preserve required
tuning deployment. Write `director_e2e_validation.json` with actual metrics, status and limitations.

Finally reconcile the Architect's provisional report headlines with this same-session baseline/final
throughput, spread, ratio/delta, TTFT, TPOT, parity and status. Use the validation JSON and
`validation/{base,final}/bench_summary.json`; preserve other analysis and per-phase results. Mark
no-win honestly. If no usable validation number exists, retain the labeled provisional fallback and
state validation failed; never present it as independently verified. Check report/JSON consistency.

Return JSON:
```json
{
  "model_name": "<name>",
  "baseline_throughput_tok_s": 0.0,
  "director_verified_throughput_tok_s": 0.0,
  "throughput_speedup": 1.0,
  "claimed_throughput_tok_s": 0.0,
  "validation_status": "validated_win|validated_no_win|flagged",
  "output_parity": "pass|fail|n/a",
  "applied_to_original": "true|false",
  "final_overlay": "<EVAL_DIR>/final",
  "final_launch_script": "<EVAL_DIR>/final/final_launch.sh",
  "arbitration_note": "win/no-win/flag reason: state the measured delta%, whether it cleared the noise band, and (if flagged) what to re-task"
}
```
