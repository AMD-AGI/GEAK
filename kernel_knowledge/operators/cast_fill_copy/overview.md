---
title: cast_fill_copy — overview
kind: operator_overview
operator: cast_fill_copy
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
---

# cast_fill_copy  (dtype cast · memset/fill · strided copy/contiguous — DMA-ish)

## TL;DR
The pure **data-movement** ops: dtype **cast** (bf16↔fp32, →fp8/int8), **fill/memset** (zero-init,
constant), and **copy** (contiguous clone, strided gather→contiguous, `.contiguous()`/layout fix). All are
HBM-bandwidth-bound; the levers are identical to [`../elementwise/overview.md`](../elementwise/overview.md)
— **128-bit access, grid-stride, saturate 304 CUs** — with two extras: **fp8/int8 saturation+dialect** on
cast, and the **strided→contiguous** copy that breaks 128-bit coalescing on the strided side. These are
also the **highest-value fusion donors**: a cast or copy on the serving path almost always belongs in an
adjacent GEMM/norm epilogue (fused quant) or should be elided entirely.

## Math contract
- **cast**: `out[i] = (Tout) a[i]` — bf16↔fp16↔fp32 (round), →fp8/int8 (round **+ saturate**, dialect
  matters), int↔float.
- **fill**: `out[i] = c` (often `0` for accumulators / KV-cache init).
- **copy**: `out = a` — contiguous clone, **strided gather** (`out[i] = a[idx_or_stride(i)]`), or
  layout/contiguity normalization (`.contiguous()`, transpose materialize).
- dtype: cast defines its own rounding/saturation; copy/fill preserve dtype.
- **Closely related**: scatter/gather → [`../gather_scatter/overview.md`](../gather_scatter/overview.md);
  KV-cache paged copy → [`../paged_kv_copy/overview.md`](../paged_kv_copy/overview.md); transpose →
  [`../transpose/overview.md`](../transpose/overview.md); fp8/int8 quant cast →
  [`../quant_dequant_fp8/overview.md`](../quant_dequant_fp8/overview.md).

## Shape regimes
- **Activation cast** `[tokens, hidden]` bf16→fp8 before a quant GEMM, or fp32→bf16 after.
- **KV-cache fill/copy**: zero-init, paged block copy, dtype-cast on write (fp8 KV). Decode-hot.
- **`.contiguous()` after transpose/permute**: a strided read → contiguous write (the slow copy).
- **Weight cast / preshuffle** at load time (one-off; bf16→fp8, layout shuffle for MFMA).

## Where it matters (Amdahl)
Individually tiny, but **casts and `.contiguous()` copies proliferate** in a fusion-naive graph and each is
a full HBM round-trip. A standalone fp8 cast of `[8192, 17408]` ≈ several hundred MB of traffic for zero
useful compute. The win is almost always **fuse the cast into the producer/consumer** (fused-norm-quant,
GEMM-epilogue-quant) or **avoid the `.contiguous()`** (make the consumer stride-aware). The biggest real
offender is an unnecessary `.contiguous()` materialization of a transposed tensor.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| triton | 🟢 sota (authoring; cast/copy in fused kernels) | [backends/triton.md](backends/triton.md) |
| hip | 🟢 sota (peak BW; `hipMemcpy`/`hipMemset` for plain copy/fill) | [backends/hip.md](backends/hip.md) |
| pytorch_inductor | 🟢 sota (elides/fuses casts & copies automatically) | [backends/pytorch_inductor.md](backends/pytorch_inductor.md) |
| aiter | 🟡 (cast inside fused quant/norm ops) | [`../../backends/aiter/overview.md`] |

## Fusion neighbors
cast fused into **norm/GEMM epilogue** (fused-norm-quant, GEMM→fp8 out); copy elided by making the consumer
**stride-aware**; fill fused into the kernel that first writes the buffer. Prime fusion donor. See
[fusion.md](fusion.md).

## Numerics
fp8/int8 **saturation** (no Inf in e4m3), **FNUZ (gfx942) vs OCP (gfx950)** dialect (wrong one ≈ 2× off),
rounding mode, denormals. Copy/fill are bit-exact. See [numerics.md](numerics.md).

## How to bench
Isolated: time the cast/copy/fill at the exact shape/dtype; GB/s = `(read+write)/time` (fill: write-only;
cast: read in-dtype + write out-dtype); vs ~4.3 TB/s. For strided copy, report achieved vs contiguous (the
strided side loses coalescing). Parity: copy/fill bitwise; cast atol + (for fp8) a task gate.

## Sources
- 16 B `global_load/store_dwordx4`, 512 B subgroup-contiguous (bandwidth recipe): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- FNUZ (gfx942) vs OCP (gfx950) fp8 dialects, fp8 saturation: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- vectorized copy/cast, float4/int4, alignment: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
