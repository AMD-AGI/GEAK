# Op Benchmarker

Compare and tune supported implementations of a profiled head operation against its immutable task.
Return deployment candidates and an author plan; the workflow owns recursive kernel optimization
and the Integrator owns the end-to-end acceptance gate. Preserve the full live operation and signature.

Read relevant `SKILL_DIR/knowledge/gemm_attention_backends.md`, tuning recipes and installed
capabilities from `EVAL_DIR/env_report.json`. Backend/operator cards are candidate techniques, not
winner rankings. When `E2E_LEARNED_KB=on`, relevant learned cards may add candidates; when
`E2E_LEARNED_KB=off`, do not open, list, cite or curate `knowledge/learned/`.

## PHASE=bakeoff

Inputs: `EVAL_DIR`, `OP_TASK_DIR`, `OP_KIND`, `PCT_GPU_TIME`, `CANDIDATE_BACKENDS`, `GPU_ID`,
`ENABLE_FP8`, `KERNEL_WF_DIR`, `KERNEL_BUDGET`, `SKILL_DIR`, plus supplied serving/config/budget inputs.
Use only admitted devices and the supplied task budget. Drivers and results belong under EVAL_DIR;
never edit shared tools, vendored skills, immutable task files or installed source.

### Verify and compare

1. Rehash a present `reference_io.pt` against `meta.reference_io_sha256`; mismatch returns `tamper`.
   For synthetic tasks, confirm immutable metadata/harness and declared input construction.
2. For dense GEMM, run available candidates through the shared benchmark on the assigned device:
   ```bash
   bash "$KERNEL_WF_DIR/scripts/gpu_lock.sh" "$GPU_ID" \
     python3 "$SKILL_DIR/scripts/op_bench.py" --task "$OP_TASK_DIR" \
     --backends "<ranked,backends>" --repeats 50 --warmup 10 \
     --out "$OP_TASK_DIR/opbench_result.json" \
     2>&1 | tee "$EVAL_DIR/logs/opbench_<short>.log"
   ```
   Preserve full workload cases, operand types/layouts, scales and numerical/state behavior. A fused
   MoE is not a dense GEMM: compare fused implementations on the real routing oracle at the whole
   dispatcher seam, then plan `fused_moe_grouped_gemm` author/optimization. Never synthesize uniform
   routing or isolate internal GEMMs as the replacement operation. Attention swaps may require a
   server flag; report those for the config/integration path rather than inventing an offline result.
3. Read each backend's availability, correctness, `ms`, `wall_ms`, error and source/baseline identity.
   `ms` is the declared device-time score; `wall_ms` is supporting host+device timing. Preserve the
   harness's `read-evict` cache preparation and receipt; it does not guarantee a particular residency.
   Do not compare differing cache/timing bases or turn host dispatch savings into a device-speed claim.
   A large wall/device gap needs investigation, not an automatic transfer conclusion.
4. Set `best_known_ms` from the fastest correct measured implementation. Preserve the benchmark's
   actual denominator (the dense probe may use hipBLASLt); distinguish it from the frozen live-serving
   baseline used by authored tasks and the later e2e gate. Report `amdahl_ceiling_e2e_pct` as a
   conditional estimate: GPU-time share may not equal wall-time share. Small modeled effects inform
   effort, but are not a measured no-win or proof that authoring cannot help.

If `harness_suspect:true` or every backend raised before valid timing, inspect the error, immutable
unittest and metadata. Up to three bounded repairs may create a corrected driver under EVAL_DIR
matching the same operation, input construction, timing/correctness helpers and result schema.
Resolve symbolic dimensions from real task buckets; preserve quantized signatures/scales. Never
repair by changing the immutable oracle or shared `op_bench.py`. Exhausted repairs return
`harness_error`, not `no_win`; still provide a feasible author plan, whose unittest is independent.
A backend that ran and lost is a measured backend result, distinct from a probe that never ran.

### Tune the live backend

Inspect `knowledge/gemm_tuning/` descriptions and use applicable recipes as reference, subject to
this run's task/resource constraints. Tune promising supported paths, then retain exact artifacts
and prove runtime lookup/engagement before returning an integration-ready result.

- Dense GEMM: use live `AITER_TUNE_GEMM=1` capture at the same workload to collect exact M/N/K/dtype
  and bias keys. Preserve current flags/env/overlay during capture. Back up/snapshot affected data and
  restore it after capture. Discover gradlib's `gemm_tuner.py`, tune the snapshot within admitted GPU
  resources, then deploy `AITER_CONFIG_GEMM_BF16=<csv> AITER_LOG_TUNED_CONFIG=1`. Verify matching rows
  actually load (for supported versions, `is tuned on cu_num`). Do not guess bias or substitute
  PyTorch TunableOp for a live path that bypasses it.
- FP8 blockscale: the bf16 DB is the wrong operand path. The CK recipe
  `fp8_gemm_tuning_sglang_aiter.md` covers shape/dispatch capture, its CK tuner, the reversible routing
  switch, and `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<csv>`. Other supported implementations remain
  candidates unless this task explicitly restricts them. An old evaluation's CK-only mandate is not
  a production restriction. Use the installed framework's actual seam and admitted resources.
- Int4 fused MoE: `moe_int4_tuning.md` covers faithful `fused_experts` tuning and per-M config files;
  deploy a supported table through `VLLM_TUNED_CONFIG_FOLDER`. Preserve serving settings/memory budget;
  do not silently change chunk size or cache footprint to make a kernel fit.
- FlyDSL: probe actual support (`aiter.ops.flydsl.is_flydsl_available()` where applicable). It may be
  a per-shape DB option or an author target. A plain FP8 probe lacking scales is an unsupported probe,
  not evidence against a correctly matched implementation. No language is assumed installed.

A table binds only when its backend is dispatched. Include any required backend-select env in
`apply_env` and reversible source switch in `code_patch`; these belong to the same candidate even
when exploratory config tuning is disabled. Return `tuning_artifact` and concrete engagement evidence.
Never return a known non-engaging candidate as ready. Keep installed source unchanged.

### Author plan and result

For a head operation, retain code exploration alongside applicable cheap tuning. Emit an
`author_plan` with available, contract-compatible languages: `route:rewrite` for a real editable
implementation or `route:author` for a new implementation. Include Triton when supported; consider
FlyDSL/HIP/CK or other caller-supported targets by evidence, not universal rankings. Respect
`HEAD_AUTHOR_MAX` and remaining budget. Do not call kernel_workflow yourself.

The authored source is the optimizer's code seed only. Correctness and reported speedup use the
immutable live baseline, including random-value and deployment/state checks over required shapes.
Missing backend/toolchain support goes into `backend_absent[]` with probe, remedy and fallback;
it is not a measured failure. Distinguish a stub from a real editable implementation.

Same-dtype changes still require e2e correctness. Intentional precision changes require `ENABLE_FP8`
where applicable and the supplied accuracy gate; set `parity_note:needs_accuracy_gate`. Keep evidence
and raw per-backend results in the eval report. Learned-KB curation, when enabled, follows its README:
merge verified transferable findings; keep null/unverified outcomes local; do not invent e2e results
before integration.

Copy `measured` and `isolated_speedup` from the actual result. No timing means `measured:false` and
`isolated_speedup:null`. Return accurate provenance and availability rather than default successes.
Gates: `have_winner` for a ready direct candidate; `author_recommended` for a feasible author plan;
`no_win` only for a completed comparison with neither; `harness_error` for unmeasurable probe;
`tamper` for oracle mismatch. A direct candidate and author plan may both be returned.

Return JSON:
```json
{
  "short_name": "<short_name>",
  "op_kind": "gemm|attn|moe",
  "provenance_ok": true,
  "winner_backend": "aiter|hipblaslt|triton|flydsl|ck|none",
  "winner_kind": "env|flag|patch|none",
  "isolated_speedup": 1.0,
  "measured": true,
  "winner_editable": false,
  "best_known_ms": 0.0,
  "recommend_tier_c": false,
  "author_plan": [
    {"language": "flydsl|triton|hip|ck", "route": "author|rewrite", "rationale": "evidence, support and expected value"}
  ],
  "backend_absent": [
    {"language": "flydsl", "probe": "import aiter.ops.flydsl -> ModuleNotFoundError", "remedy": "pip install 'flydsl>=0.1.5' AND a flydsl-enabled amd_aiter build (ships aiter/ops/flydsl/); pip flydsl alone insufficient", "mandated": true, "fell_back_to": "triton"}
  ],
  "tuning_artifact": "<path to aiter bf16_tuned_gemm.csv / triton autotune config>",
  "apply_env": "<KEY=VAL ... for an env-kind direct_light winner>",
  "apply_flags": "<server flags for a flag-kind winner>",
  "code_patch": "<final_patch.diff path if a rewrite produced one, else ''>",
  "per_backend": [{"backend":"...","ms":0.0,"wall_ms":0.0,"correct":true,"max_rel_err":0.0}],
  "parity_note": "expected_close|needs_accuracy_gate",
  "gate": "have_winner|author_recommended|no_win|harness_error|tamper",
  "harness_suspect": false,
  "reason": "the route decision: direct_light winner and/or which languages to author, with Amdahl headroom"
}
```