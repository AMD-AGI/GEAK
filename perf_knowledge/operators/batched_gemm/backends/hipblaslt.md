---
title: batched_gemm on hipblaslt — SOTA card
kind: sota_card
operator: batched_gemm
backend: hipblaslt
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCm/hipBLASLt
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# batched_gemm × hipblaslt

## TL;DR
> hipBLASLt provides the **executed** strided-batched GEMM kernels (Tensile `Cijk_*`) that aiter
> dispatches — it is the production batched-GEMM library. You rarely call it directly in serving; you
> tune *which* hipBLASLt solution runs via aiter ([aiter.md](aiter.md)). Direct use is for offline
> benching and non-aiter stacks.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| hipBLASLt strided-batched GEMM (`hipblaslt-bench` matmul w/ batch_count) | `ROCm/hipBLASLt` | gfx942/950; bf16/fp16/fp8 | dense fp8 reference: ~2750 TFLOP/s @ M=N=K=4096, ~3130 @ 8192 on MI355X (CDNA4); bf16 ceiling ~890 TFLOP/s on MI300X (single-GEMM proxies; batched of small shapes is occupancy-bound, lower) | the executed kernel on/under aiter |

## Config space / knobs
- `--batch_count B`, layout (NT for attention), `--algo_method all`/index to enumerate solutions.
- Solution selection is the knob — done by aiter's race, not by hand.
- For small batched: rely on solutions that pack the batch into ≥1024 WGs; `HIPBLASLT_TUNING_FILE`
  exists but **does not engage sglang's aiter path** (it hooks PyTorch dispatch).

## Numerics / parity
fp32 accumulate per batch; same-math solution swap parity-safe. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
In serving, hipBLASLt is reached **through aiter** (no direct rebind seam in sglang). For a non-aiter
stack, `cublasLt`-style API or PyTorch with TunableOp. To change which solution serving uses → tune
aiter's DB, not `HIPBLASLT_TUNING_FILE`.

## Pitfalls & anti-patterns
- `HIPBLASLT_TUNING_FILE` / TunableOp assumed to speed serving — aiter bypasses that dispatch → 0 effect.
- Benching a single large GEMM and extrapolating to small batched — occupancy behavior differs.

## How to verify
`hipblaslt-bench` on the exact (B,M,N,K,dtype) for the isolated number; for serving, verify via the
aiter engagement marker ([aiter.md](aiter.md)).

## Alternatives / cross-links
[aiter.md](aiter.md) (dispatch/tune lever) · [ck.md](ck.md) · [asm.md](asm.md) · [../overview.md](../overview.md).
Dense equivalent: [[operators/dense_gemm/backends/hipblaslt.md]].

## Sources
- hipBLASLt: https://github.com/ROCm/hipBLASLt.
- fp8 GEMM TFLOP/s reference (MI355X): ROCm CDNA4 GEMM blog.
- TunableOp/tuning-file bypass note: perf_knowledge dense_gemm/backends/aiter.md validation.
