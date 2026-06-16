---
title: cast_fill_copy on HIP / C++ — SOTA card
kind: sota_card
operator: cast_fill_copy
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# cast_fill_copy × hip

## TL;DR
For **plain contiguous copy/fill**, the HIP runtime (`hipMemcpyAsync` / `hipMemsetAsync`) is already at peak
HBM — use it, don't hand-write. HIP kernels earn their place for **mixed-width casts**, **non-byte-pattern
fills**, **strided/`.contiguous()` copies** (LDS-tiled like a transpose), and **fused** data movement; plus
the async/streams/graphs plumbing for overlapping copies in a decode loop.

## SOTA implementation(s)
| impl | source | gens/dtypes | mechanism | when best |
|---|---|---|---|---|
| `hipMemcpyAsync` (D2D) / `hipMemsetAsync` | runtime | all | DMA, peak HBM | plain contiguous copy / zero-fill |
| vectorized cast kernel (`int4` in/out) | [`../../../languages/hip_cpp/patterns.md`](../../../languages/hip_cpp/patterns.md) §2 | all gfx9 | grid-stride 128-bit | mixed-width cast, non-pattern fill |
| LDS-tiled strided copy / transpose | [`../../../languages/hip_cpp/lds_async.md`](../../../languages/hip_cpp/lds_async.md) §2 | all gfx9 | coalesced read → LDS swizzle → coalesced write | `.contiguous()` you can't elide |

```cpp
// bf16 -> fp8 (fnuz on gfx942), 16 elems/thread, grid-stride
__global__ void cast_bf16_fp8(int n16, const int4* __restrict__ in,  // 16 bf16 = 32 B? -> 2x int4 read
                              __hip_fp8_storage_t* out, float scale) {
    for (int i = blockIdx.x*256 + threadIdx.x; i < n16; i += 256*gridDim.x) {
        // load 2x int4 (16 bf16), convert each to fp32*scale, pack 16 fp8 -> int4 store
        // use __hip_cvt_float_to_fp8(..., __HIP_E4M3_FNUZ) on gfx942
    }
}
// grid = 304*8, block = 256
```

## Config space / knobs
- **copy/fill**: `hipMemcpyAsync`/`hipMemsetAsync` first; kernel only for non-byte fills or fused casts.
- **cast**: vectorize **both** sides to the widest each dtype allows (mixed width — see [../tuning.md](../tuning.md));
  convert in fp32; fp8 via `__hip_cvt_float_to_fp8` with the **right dialect** (`E4M3_FNUZ` on gfx942).
- **strided copy**: LDS tile + XOR-swizzle/pad to coalesce both read and write (transpose pattern).
- **block 256**, **grid `304×k`** + grid-stride; `__launch_bounds__(256, 4)`, `__restrict__`.
- **async**: separate streams + pinned host mem for true DMA; **HIP graphs** for many small KV copies.

## Numerics / parity
copy/fill bit-exact; float→float RNE; **fp8 dialect FNUZ (gfx942) / OCP (gfx950)** — wrong one ≈ 2× off;
saturate to max-normal; int8 clamp `[-128,127]`. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
`hipcc --offload-arch=gfx942` → `.so`, torch custom op. The runtime copy/fill needs no binding. HIP source
is the edit surface for a fused cast/copy.

## Pitfalls & anti-patterns
- Hand-writing a contiguous copy/fill kernel that `hipMemcpy`/`hipMemset` already does at peak.
- Wrong fp8 dialect for the arch (silent 2× scale error).
- Strided `.contiguous()` read without LDS-tiling → non-coalesced, bandwidth collapse.
- Per-launch overhead from many tiny KV copies in decode → use HIP graphs.
- Misaligned 128-bit access → undefined / scalar fallback (runtime-branch).

## How to verify
`--save-temps` ISA: `global_load/store_dwordx4`, `ds_*` for the LDS-tiled path, no `scratch_`; GB/s vs
~4.3 TB/s; copy/fill bitwise vs torch; cast atol + fp8 dialect check + task gate.

## Alternatives / cross-links
[triton.md](triton.md) (fused authoring) · [pytorch_inductor.md](pytorch_inductor.md) (auto) ·
[../tuning.md](../tuning.md) · [`../../transpose/overview.md`](../../transpose/overview.md) ·
[`../../../languages/hip_cpp/patterns.md`](../../../languages/hip_cpp/patterns.md) §2,§4.

## Sources
- hipMemcpy/hipMemset, streams/graphs, pinned async, __restrict__, fp8 cvt: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- 16 B access, block=256, ≥1024 grid: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- FNUZ vs OCP fp8 dialects, saturation: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
