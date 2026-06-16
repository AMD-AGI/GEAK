---
title: cast_fill_copy — tuning (vectorized DMA, strided copy, fill)
kind: operator_overview
operator: cast_fill_copy
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
---

# cast_fill_copy — tuning

Same bandwidth recipe as elementwise (128-bit access, grid-stride, block=256, grid `304×k`), plus the
two data-movement specifics: the **mixed-width cast** and the **strided copy** that breaks coalescing.

## 1. Plain copy / fill — use the runtime first
- **Contiguous copy**: `hipMemcpyAsync(d2d)` — the HIP runtime's DMA path is already at peak HBM; don't
  hand-write a kernel unless you're fusing something into the copy.
- **Fill with 0 / repeating byte pattern**: `hipMemsetAsync` — peak write bandwidth.
- **Fill with a non-byte-repeating constant** (e.g. fp32 `1.0`): needs a kernel (memset is byte-wise) —
  vectorized `float4` store, grid-stride. Write-only, so it hits ~peak write BW.

## 2. Cast — mixed load/store width
The read is in the **input** dtype and the write in the **output** dtype, so the two sides have *different*
elements-per-128-bit. Vectorize **both** to the widest each allows:
| cast | read width (per 128-bit) | write width | note |
|---|---|---|---|
| bf16 → fp32 | 8 bf16 | 4 fp32 (2× the store bytes) | store-bound |
| fp32 → bf16 | 4 fp32 | 8 bf16 | read-bound |
| bf16 → fp8 | 8 bf16 | 16 fp8 | write packs 2× — read-bound |
Pick the grid/loop so each lane handles a chunk that's a multiple of **both** widths (e.g. 16 elements:
2× `int4`-bf16 read, 1× `int4`-fp8 write). Keep the convert in fp32. See [backends/hip.md](backends/hip.md).

## 3. Strided / `.contiguous()` copy — the coalescing problem
A `.contiguous()` of a transposed `[M,N]` reads `a[j*ldA + i]` (strided, **non-coalesced**) and writes
contiguous. The strided side can't issue 128-bit coalesced loads → bandwidth collapses.
- **First choice: avoid it.** Make the consumer stride-aware (most GEMM/attention kernels take a `transpose`
  flag) so the materialization never happens.
- If unavoidable, treat it as a **transpose**: tile through **LDS** (read a coalesced tile, transpose in
  LDS with XOR-swizzle/padding to dodge bank conflicts, write coalesced) — see
  [`../transpose/overview.md`](../transpose/overview.md) and
  [`../../languages/hip_cpp/lds_async.md`](../../languages/hip_cpp/lds_async.md) §2.

## 4. Occupancy & grid (shared with elementwise)
- **block 256** (4 wave64s); **grid `304×k`** + grid-stride loop.
- High occupancy to hide HBM latency: `__launch_bounds__(256, 4)` / Triton `waves_per_eu=3/4`,
  `num_warps=2/4`, `num_stages=1`.
- **alignment**: 128-bit needs 16-byte-aligned pointers + length divisible by the vec width; runtime-branch
  to a scalar tail/fallback otherwise.
- `__restrict__` on src/dst → wider loads + reorder.

## 5. Async overlap (DMA-ish)
For host↔device or overlapping copy with compute, use **separate streams** + pinned host memory
(`hipHostMalloc`) for true async DMA, sequenced with events. **HIP graphs** kill per-launch overhead for
the many small KV-cache copies in a decode loop. See
[`../../languages/hip_cpp/patterns.md`](../../languages/hip_cpp/patterns.md) §4.

## 6. Don't tune — fuse or elide
A cast or copy on the serving hot path usually shouldn't exist as its own kernel: fuse the cast into the
producing norm/GEMM (fused-norm-quant), or elide the copy with stride-awareness. See [fusion.md](fusion.md).

## Sources
- 16 B access, 512 B subgroup-contiguous, block=256: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- hipMemcpy/hipMemset, streams/graphs, pinned async, __restrict__: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- vectorized cast/copy widths, alignment, register pressure: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
