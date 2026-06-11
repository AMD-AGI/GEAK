---
title: cast_fill_copy — fusion (cast into epilogues, elide the copy)
kind: operator_overview
operator: cast_fill_copy
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/ROCm/aiter
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
---

# cast_fill_copy — fusion

Data movement is the purest fusion donor: a standalone cast/copy/fill does **zero useful compute** for a
full HBM round-trip. The two rules: **fuse the cast into the producer/consumer**, and **elide the copy by
making the consumer stride-aware**.

## The patterns

| op | fuse into / elide | result | where |
|---|---|---|---|
| **bf16→fp8 cast** | norm/GEMM **epilogue** (fused-norm-quant, GEMM→fp8 out) | quant happens in the fp32 accumulator, no separate pass | [`../fused_norm_quant/overview.md`](../fused_norm_quant/overview.md), [`../quant_dequant_fp8/overview.md`](../quant_dequant_fp8/overview.md) |
| **fp32→bf16 cast** | GEMM/norm epilogue (`OPTIMIZE_EPILOGUE=1`) | the convert is the store, free | [`../dense_gemm/fusion.md`] |
| **`.contiguous()` / transpose copy** | **elide** — make consumer stride-aware | no materialization at all | [`../transpose/overview.md`](../transpose/overview.md) |
| **strided gather copy** | fuse into the consuming kernel's load | no intermediate contiguous buffer | [`../gather_scatter/overview.md`](../gather_scatter/overview.md) |
| **KV-cache fill/cast-on-write** | fuse into the KV-write kernel | one pass | [`../paged_kv_copy/overview.md`](../paged_kv_copy/overview.md), [`../kv_cache_quant/overview.md`](../kv_cache_quant/overview.md) |
| **zero-init fill** | fuse into the first kernel that writes (accumulate-from-zero) | skip the memset | split-K / reduction init |

## How it gets fused
- **GEMM/norm epilogue quant (the big one)**: the fp8 cast + scale happens in the GEMM's fp32 accumulator
  *before* the store — the output is written directly as fp8, no separate cast kernel. `OPTIMIZE_EPILOGUE=1`
  fuses the plain dtype-convert at the GEMM store. This removes the highest-traffic standalone casts on the
  quant path.
- **aiter (library-fused)**: cast lives inside `fused_norm_quant`, fp8-output GEMM, and KV-cache-quant ops
  — never a standalone cast on the serving path. See [`../../backends/aiter/overview.md`](../../backends/aiter/overview.md).
- **Inductor (automatic)**: elides redundant casts and fuses dtype-convert into adjacent pointwise/reduction
  kernels; it also avoids many `.contiguous()` copies by tracking strides through the graph. See
  [backends/pytorch_inductor.md](backends/pytorch_inductor.md).
- **Triton/HIP (manual)**: apply the cast at the store of the producing kernel; pass strides to the consumer
  instead of materializing a contiguous copy.

## The `.contiguous()` trap
The most common avoidable copy: `x.transpose(0,1).contiguous()` materializes a strided→contiguous pass
(slow, non-coalesced read). Most GEMM/attention kernels accept a transpose flag (`transpose_b`, layout
arg) — pass it and the copy disappears. Only materialize when the same transposed tensor is read many times
*and* the strided reads dominate (measure).

## Anti-patterns
- A standalone `.to(fp8)` on the serving path that a fused-norm-quant would absorb.
- `.contiguous()` to "be safe" when the consumer is already stride-aware → a full pointless copy.
- A separate `zeros_like` + accumulate when the kernel could initialize from zero in-pass.
- Fusing a strided gather into a kernel that then can't 128-bit-coalesce — sometimes a contiguous copy +
  fast consumer wins (measure both).

## Verify
rocprof: the fused quant path shows **no** standalone cast kernel and no extra `[tokens, hidden]` write;
`.contiguous()` elision shows the copy kernel gone. `TORCH_COMPILE_DEBUG=1` for the Inductor path.

## Sources
- aiter fused norm/quant/KV-cast ops: https://github.com/ROCm/aiter
- `OPTIMIZE_EPILOGUE=1` convert fusion: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- Inductor cast elision / stride tracking / fusion: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
