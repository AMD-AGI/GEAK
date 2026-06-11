---
title: elementwise on HIP / C++ — SOTA card
kind: sota_card
operator: elementwise
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# elementwise × hip

## TL;DR
HIP/C++ gives **full control of vectorization, alignment, and occupancy** → the actual HBM-bandwidth
ceiling for an elementwise op, and the editable seam for a Tier-C custom kernel. The recipe is fixed:
128-bit `float4`/`int4` access in a **grid-stride loop**, `block=256`, grid `≈304×8`, `__restrict__` +
`__launch_bounds__` to hold VGPRs and lift occupancy. Reach for HIP over Triton when you need exact ISA
control or a fusion neither Triton nor the library expresses.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vectorized grid-stride pointwise (`float4`/`int4`, scalar tail) | [`../../../languages/hip_cpp/patterns.md`](../../../languages/hip_cpp/patterns.md) §2 | all gfx9, all dtypes | bandwidth-bound: **~4.3 TB/s** (~81% peak, BabelStream-class) @ MI300X gfx942 | peak BW, fused custom epilogue, Tier-C rewrite |

```cpp
template<typename T4>                                  // float4 / __half2-packed int4
__global__ void __launch_bounds__(256, 4)             // 4 waves/EU -> VGPR<=128, hide latency
emul_vec(int nv, const T4* __restrict__ a, const T4* __restrict__ b,
         T4* __restrict__ out, float s, float lo, float hi) {
    for (int i = blockIdx.x*256 + threadIdx.x; i < nv; i += 256*gridDim.x) {
        T4 av = a[i], bv = b[i];                       // global_load_dwordx4 ×2
        // unpack -> fp32 -> a*s+b -> clamp -> repack (per lane, no cross-lane)
        out[i] = fma_clamp4(av, bv, s, lo, hi);        // global_store_dwordx4
    }   // + scalar tail for n % 4
}
// grid = 304*8, block = 256
```

## Config space / knobs
- **vector type**: `float4` (fp32), `int4` reinterpret for 8×bf16/fp16 or 16×fp8/int8. 16 B = 128 bit.
- **`block`**: 256 (4 SIMDs); 128 only for power.
- **`grid`**: `304*k` (k≈8) + grid-stride loop → fixed grid, any size, `dwordx4` stays in the loop.
- **`__launch_bounds__(256, minWavesPerEU)`**: 4 → VGPR≤128 (high occupancy, good for BW); back off if it
  spills. `__restrict__` enables wider loads + reorder.
- **`-munsafe-fp-atomics`** only relevant if the op writes via atomics (rare for plain pointwise).

## Numerics / parity
Load → `float` → compute in fp32 → round to out dtype (matches torch). `v_max_f32`/`v_min_f32` differ
from `std::max` on NaN — replicate torch `min(max())` order if NaN parity matters. fp8 = FNUZ on gfx942.
See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Build with `hipcc --offload-arch=gfx942` → a `.so`, bind via a torch custom op
(`TORCH_LIBRARY`/`direct_register_custom_op`) so it survives `torch.compile`. This is the same seam vLLM
uses for its `csrc/rocm` kernels. The HIP source *is* the edit surface (rebuild after editing).

## Pitfalls & anti-patterns
- Compiler **not emitting `dwordx4`** for some vector types (historic ROCm#341) — verify ISA
  (`--save-temps`, grep `_dwordx4`).
- Misaligned pointer / `n` not divisible by vec → undefined or scalar fallback; runtime-branch on
  `(ptr%16==0 && n%vec==0)`.
- Over-vectorizing a register-heavy kernel crosses a 16-VGPR boundary → fewer waves/EU → *slower*. Check
  `-Rpass-analysis=kernel-resource-usage`.
- Scalar grid (`<<<n/256,256>>>` with no stride) for huge n → too many blocks, scheduler overhead; prefer
  fixed grid + stride loop.

## How to verify
`--save-temps` ISA shows `global_load_dwordx4`/`global_store_dwordx4`, no `scratch_`; achieved GB/s vs
~4.3 TB/s; `rocprofv3` `FETCH_SIZE`/`WRITE_SIZE` = exactly the tensor bytes (no extra traffic); atol vs
torch.

## Alternatives / cross-links
[triton.md](triton.md) (faster to author/fuse) · [pytorch_inductor.md](pytorch_inductor.md) (auto) ·
[../tuning.md](../tuning.md) · [`../../../languages/hip_cpp/patterns.md`](../../../languages/hip_cpp/patterns.md) §2,
[`../../../languages/hip_cpp/lds_async.md`](../../../languages/hip_cpp/lds_async.md) (128-bit access).

## Sources
- 16 B optimal access, 512 B subgroup-contiguous, block=256: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- float4/int4, grid-stride, register-pressure / alignment tradeoffs: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- warpSize=64, __launch_bounds__, __restrict__, v_max NaN behavior: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
