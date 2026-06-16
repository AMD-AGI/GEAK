---
title: Recipes index — durable how-to levers (reference only)
kind: reference
updated: 2026-06-09
---

# Recipes index — durable e2e levers (procedures, not rankings)

Reference material for the workflows. These are **durable "how-to" procedures** (API entrypoints,
tuning flows, knobs, fusion passes, config switches, verification) — the parts of kernel knowledge that
**do not go stale** as ROCm/aiter versions change. They are deliberately free of "which is fastest" /
TFLOPS / `status` claims (those are time-sensitive — see `sota_registry.yaml`, treated as dated evidence).

> **Consumption contract.** This is *reference material a workflow may consult*, not instructions it must
> obey. The workflow/agent decides what to try; **on-box measurement decides what wins**. Knowledge here
> can only *seed candidates* and *show how to do a thing correctly* — it never gates, never prunes the
> agent's own ideas, and never substitutes for measuring. If a recipe is stale or wrong, the worst case
> is a candidate that loses the bake-off; it can never lower the result below the measured baseline.

## How to use (candidate enumeration → decide → measure)
1. **Enumerate** applicable backends for the op from `capability_index.yaml` (filter by gen/dtype/regime).
2. **Read** the relevant card(s) `operators/<op>/backends/<backend>.md` + `languages/<lang>/` for the
   *mechanism + knobs + code skeleton* — the "how", not the "who's best".
3. **Always keep a baseline candidate** (canonical/current impl) and **measure every candidate** on the
   immutable oracle / e2e bench. Pick the winner by measurement only.

## E2E levers, cheapest-first (the durable ordering — Amdahl, not rank)
Order by leverage = `pct_gpu_time × achievable_speedup`, and by cost (cheap switches before rewrites).
This ordering is structural and does not age; the specific winners do, so always measure.

1. **Config / backend switch (cheapest, no source change)** — flags, env, backend selection (e.g.
   `--attention-backend`, `SGLANG_USE_AITER` / `VLLM_ROCM_USE_AITER`, `--quantization`, cuda-graph,
   torch.compile). See `backends/<backend>/` for the knob list. → consumed by the e2e Config Tuner.
2. **Per-shape GEMM tuning DB (cheap, high-leverage)** — GEMM is often the largest single bucket. The
   durable recipe: `AITER_TUNE_GEMM=1` live capture → `gradlib gemm_tuner.py` (err-ratio gate) →
   deploy via `AITER_CONFIG_GEMM_*` → verify engagement with `AITER_LOG_TUNED_CONFIG=1`. Full procedure:
   [[kernel_workflow/gemm_tuning_workflow.md]]; mechanics: [[backends/aiter/tuned_gemm.md]].
3. **Fusion (medium)** — fold elementwise/quant/norm/rope into neighbors; vLLM Inductor fusion passes
   (RMSNorm+quant, SiLU+quant, RoPE+KV). See [[optimization/kernel_fusion_strategy.md]].
4. **Quantization path (medium, parity-gated)** — fp8/mxfp paths + the FNUZ-vs-OCP correctness trap.
   See [[quantization/deployment_recipes.md]], [[quantization/fnuz_vs_ocp.md]].
5. **Kernel rewrite (most expensive)** — author a faster impl in a tile DSL when a hot op has no good
   editable kernel. Authoring ladder + the immutable-unittest loop: [[kernel_workflow/optimize_single_kernel.md]],
   [[kernel_workflow/authoring_a_kernel_with_geak.md]]. CDNA scheduling prior (durable): prefer 8-wave
   ping-pong / 4-wave interleave over NVIDIA wave-specialization — [[optimization/mfma_scheduling.md]].

## Verification recipes (always, regardless of lever)
- **Engagement proof** — confirm the kernel/config is actually used (don't trust the swap took effect):
  [[profiling/engagement_verification.md]] (e.g. `AITER_LOG_TUNED_CONFIG=1` → "is tuned on cu_num").
- **Measurement discipline** — warmup, repeats, noise band, same-session A/B, parity check:
  [[profiling/benchmarking_methodology.md]], [[profiling/common_pitfalls.md]].

## Durable knob dictionaries (the search space, by authoring language)
- Triton/Gluon: [[languages/triton_amd/knobs.md]], [[languages/gluon/programming_model.md]]
- HIP/CK/asm: [[languages/hip_cpp/patterns.md]], [[languages/composable_kernel/ck_tile.md]], [[languages/asm_mfma/overview.md]]
- FlyDSL / TileLang / HipKittens: [[languages/flydsl/overview.md]], [[languages/tilelang/overview.md]], [[languages/hipkittens/primitives.md]]

## Sources
- Procedures cross-link to the per-file docs above, each of which carries its own `## Sources`.
- On-box `ROCm/aiter@a6bb4993` for the tuning/engagement entrypoints.
