---
title: autotuning methodology (triton / hipBLASLt / aiter)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, both]
updated: 2026-06-05
sources:
  - https://github.com/ROCm/aiter/tree/main/gradlib
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
---

# autotuning methodology

> **Canonical deep version:** [`../expert_skills/tuning/tuning-core/search_strategy.md`](../expert_skills/tuning/tuning-core/search_strategy.md)
> in the vendored tuning skillset. That copy is the one whose claims are executable
> (`perf_knowledge/expert_skills/tuning/validate/claims.py`) and re-checked per image, and it is
> what the e2e tuning phase actually runs. This card stays because the three-tier map and the 9-tuple aiter key are the
> orientation a reader needs before opening the skillset at all
> — where the two touch the same ground, the deep version wins and this one is the index entry
> into it. Do not grow tuning procedure here; send the fix upstream and re-sync the tree.

## TL;DR
Three tuner tiers exist, but **only one engages the live sglang/vllm GEMM path: aiter's per-shape DB**.
The recipe is: capture real shapes with **`AITER_TUNE_GEMM`**, race candidates with **gradlib
`gemm_tuner.py`** gated at **`err_ratio < 0.05`**, deploy via **`AITER_CONFIG_GEMM_BF16=<tuned.csv>`**,
and verify engagement in the log. The lookup key is a **9-tuple** — get any field (esp. `bias`) wrong
and you hit 0% engagement. Validated **+2.23% e2e** on Qwen3.5-27B / sglang 0.5.11 / aiter, 2026-06-08.
triton `@autotune` and `hipblaslt-bench` are useful for *authoring* kernels but don't reach the live
dispatch unless rebound through aiter. See `[[operators/dense_gemm/tuning.md]]` and
`[[operators/dense_gemm/backends/aiter.md]]`.

## The three tiers
| tier | tool | reaches live serving path? |
|---|---|---|
| author-time kernel | triton `@triton.autotune` over configs | only if the kernel is the live dispatch |
| library offline | `hipblaslt-bench` / TensileLite; PyTorch `TunableOp` (`HIPBLASLT_TUNING_FILE`) | **no** for sglang/aiter — aiter bypasses these hooks |
| **live dispatch** | **aiter gradlib per-shape DB** | **yes** — this is the lever |

## aiter live-tuning recipe (the lever)
1. **Capture** real shapes on a *warm* server: `AITER_TUNE_GEMM=1` (or replace `F.linear`→`tgemm.mm`);
   shapes append to `aiter/configs/bf16_untuned_gemm.csv`. **Bias/scale/dtype must match live calls.**
2. **Race**: `python3 gradlib/gradlib/gemm_tuner.py --input_file …/bf16_untuned_gemm.csv
   --tuned_file …/bf16_tuned_gemm.csv --indtype bf16 --mp <ngpus>`; gate accepted kernels at
   **`err_ratio < 0.05`** (max tolerable numeric error vs reference). If OOM, set
   `CACHE_INVALIDATE_BUFFERS` to a small prime (e.g. 11/7/3/1).
3. **Deploy**: `AITER_CONFIG_GEMM_BF16=<tuned.csv>` and restart; **verify** with
   `grep -c 'is tuned on cu_num'` in the server log (non-zero ⇒ engaged).

### The 9-tuple lookup key
aiter selects by `(cu_num, M, N, K, bias, dtype_in, dtype_out, scaleAB, bpreshuffle)`. The tuned CSV
columns mirror this: `cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle, libtype, solidx,
splitK, soltimes, kernelName, tflops, bw`. `libtype ∈ {hipblaslt, rocblas, asm, triton}`; `splitK` is
valid when `libtype==asm`. **A single mismatched field (commonly `bias=true` tuned vs `bias=false`
live) ⇒ 100% lookup miss, 0 engagement** — the most common silent failure.

## Search-space pruning
Superseded in full by [`../expert_skills/tuning/tuning-core/search_strategy.md`](../expert_skills/tuning/tuning-core/search_strategy.md): prune-before-you-race, M bucketing,
ordering by expected payoff, `--mp` parallelism and its cross-GPU variance trap, and when to stop.
The three bullets that used to sit here were a subset of it and had already drifted; a subset that
drifts is worse than a pointer.

## Caching / reuse
- Commit the tuned CSV per (model, dtype, hardware) and load via the env var — do **not** edit
  site-packages. Re-tune when shapes, dtype, ROCm/aiter version, or GPU SKU (304 vs 256 CUs) change.
- triton: persist the `@autotune` cache keyed on shape; the first call per shape pays the search.

## Pitfalls
- Tuning synthetic `bias=true` shapes when live is `bias=false` ⇒ 0 engagement (verify with `grep`).
- Relying on `TunableOp` / `HIPBLASLT_TUNING_FILE` for an aiter-dispatched server — aiter bypasses it.
- Reporting theoretical peak as the bar; the real target is the best *tuned library* kernel
  (`[[optimization/roofline_and_bottlenecks.md]]`).
- Forgetting to re-tune after a ROCm/aiter bump or SKU change.

## Verify
- Engagement: `grep -c 'is tuned on cu_num' server.log` > 0.
- Gate: accepted kernels satisfy `err_ratio < 0.05`; e2e delta beyond noise band with non-overlapping
  repeats (`[[operators/dense_gemm/tuning.md]]`).
- Counters: tuned config should show higher MFMA busy / closer-to-roofline TFLOP/s (`[[profiling/]]`).

## Sources
- gradlib `gemm_tuner.py`, `err_ratio`, CSV columns (9-tuple key), `CACHE_INVALIDATE_BUFFERS`: ROCm/aiter gradlib + ROCm hipBLASLt tuning blogs.
- `AITER_TUNE_GEMM` / `AITER_CONFIG_GEMM_BF16` recipe, `mfma_16x16`, ≥1024 WG, `OPTIMIZE_EPILOGUE`: ROCm MI300X workload guide.
- +2.23% e2e validation: perf_knowledge e2e run 2026-06-08 (see `[[operators/dense_gemm/backends/aiter.md]]`).
