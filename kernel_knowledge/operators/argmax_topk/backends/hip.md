---
title: argmax_topk on HIP / C++ — SOTA card
kind: sota_card
operator: argmax_topk
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://moderngpu.github.io/scan.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# argmax_topk × hip

## TL;DR
HIP gives full control of the `(value, index)` wave/LDS reduce and the **tie rule**, so you can match a
specific reference/sampler tie convention exactly (the thing that bites the Triton path). It's the seam for
a fused greedy lm_head or a fast bitonic/selection top-k. The AMD must-knows: shuffle **both** val and idx
over **64 lanes** (6 steps), and there's **no fp `atomicAdd`** for argmax → use the two-call combine.

## SOTA implementation(s)
| impl | source | gens/dtypes | strategy | when best |
|---|---|---|---|---|
| `(val,idx)` wave-shuffle + LDS reduce | [`../../../languages/hip_cpp/patterns.md`](../../../languages/hip_cpp/patterns.md) §1 | all gfx9, fp32 compare | argmax | greedy decode top-1 |
| chunk vocab + 2-call winner combine | this card | all gfx9 | split | batch=1, huge vocab |
| local heap → LDS merge → bitonic | [moderngpu](https://moderngpu.github.io/scan.html) | all gfx9 | top-k | sampling k≤256 |

```cpp
struct VI { float v; int i; };
__device__ VI vi_max(VI a, VI b) {                     // left-most tie
    if (b.v > a.v || (b.v == a.v && b.i < a.i)) return b; return a;
}
__device__ VI wave_argmax(VI x) {                      // 64-lane, 6 steps
    for (int o = warpSize/2; o > 0; o >>= 1) {
        VI y { __shfl_down(x.v, o), __shfl_down(x.i, o) };
        x = vi_max(x, y);
    }
    return x;                                            // lane 0 holds (max, argmax)
}
// per-thread best over grid-stride -> wave_argmax -> LDS[nwaves] -> wave0 reduce -> store
```

## Config space / knobs
- **shuffle the pair**: val + idx (or bit-pack into one 64-bit word so a single `__shfl_down` carries both
  — high bits = order-preserving float key, low bits = idx).
- **tie rule** in the combine (`b.i < a.i` for left-most) — set to match the reference exactly.
- **block 256**, LDS = `nwaves` pairs. **fp32 compare** for bf16/fp16.
- **split**: chunk vocab across blocks → per-block `(val,idx)` → two-call combine (no fp atomic for max).
- **top-k**: per-thread local top-k buffer → LDS merge → bitonic; or threshold-select (find k-th, gather
  ≥). **vectorize the logits load** (128-bit) — the bandwidth-bound part.

## Numerics / parity
You own the tie rule and NaN handling → can match torch/sampler exactly (unlike Triton #802/#6635). `v_max`
returns non-NaN operand — add an explicit NaN check if the reference propagates NaN. fp32 compare. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
`hipcc --offload-arch=gfx942` → `.so`, torch custom op. HIP source is the edit surface. The path for a
fused greedy lm_head that never materializes full logits, or a production top-k sampler.

## Pitfalls & anti-patterns
- **5-step / 32-lane** shuffle from CUDA → wrong argmax (half the wave).
- Shuffling only the value, not the index → wrong index.
- Forgetting the tie rule in **both** the wave and LDS combine → inconsistent ties.
- `atomicAdd`-style combine for max → no fp atomic; use two-call or `atomicMax` on int-bits.
- Non-vectorized logits load → bandwidth-bound on the wrong thing.

## How to verify
`--save-temps` ISA: `global_load_dwordx4`, `ds_*` LDS, no `scratch_`; index match vs torch on tie/±Inf
edge cases; greedy temp=0 token-match; GB/s = `logits_bytes/time` vs ~4.3 TB/s.

## Alternatives / cross-links
[triton.md](triton.md) (faster to author; parity caveats) · [../tuning.md](../tuning.md) ·
[../fusion.md](../fusion.md) · [`../../reduction/backends/hip.md`](../../reduction/backends/hip.md)
(the value-only reduce this extends).

## Sources
- wave64 `__shfl_down` (val+idx), 64-bit masks, v_max NaN behavior: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- selection / bitonic top-k: https://moderngpu.github.io/scan.html
- block=256, ≥1024 grid, CU saturation: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
