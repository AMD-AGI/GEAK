---
title: cumsum_scan on HIP / C++ — SOTA card
kind: sota_card
operator: cumsum_scan
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32, int64]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://moderngpu.github.io/scan.html
  - https://github.com/proger/accelerated-scan
---

# cumsum_scan × hip

## TL;DR
HIP gives full control of the scan **order and hierarchy** — the reason to drop to it is the Triton
`associative_scan` non-commutative bug ([../numerics.md](../numerics.md)): in HIP you write the wave
`__shfl_up` scan and the LDS/global carry yourself, so a non-commutative SSM recurrence is correct at any
length. Use Hillis-Steele within a wave/small block (shallow depth) and Blelloch for the block/grid level
(work-efficient).

## SOTA implementation(s)
| impl | source | gens/dtypes | algo | when best |
|---|---|---|---|---|
| wave `__shfl_up` scan + LDS block carry | this card | all gfx9, fp32 acc | Hillis-Steele (wave) | short axis / one block |
| Blelloch up/down-sweep block scan | [moderngpu scan](https://moderngpu.github.io/scan.html) | all gfx9 | Blelloch | large N, work-bound |
| chunked 3-stage (block-scan/carry/add) | this card + tuning.md | all gfx9 | hierarchical | long sequence axis |
| pair-scan recurrence (SSM/gated-delta) | `accelerated-scan` CUDA→HIP | all gfx9 | warp→block Blelloch | linear-attn / SSM, **correct at any seq** |

```cpp
__device__ float wave_scan_inclusive(float v) {        // 64-lane Hillis-Steele
    int lane = threadIdx.x % warpSize;
    for (int off = 1; off < warpSize; off <<= 1) {     // 1,2,4,8,16,32  (6 steps, wave64)
        float n = __shfl_up(v, off);
        if (lane >= off) v += n;
    }
    return v;                                           // lane i holds prefix sum 0..i
}
// block: each wave scans 64 lanes -> last lane total to LDS -> wave0 scans partials ->
//        each wave adds its exclusive carry -> store (vectorized 128-bit)
```

## Config space / knobs
- **wave scan** `__shfl_up` 6 steps (64 lanes — not 5). **block 256** (4 waves), tiny LDS for carries.
- **algorithm**: Hillis-Steele for the wave/small block (depth `log N`); Blelloch up/down-sweep for large
  N (work `O(N)`).
- **chunking**: 3-stage stitch for long axes (block-scan → scan chunk-totals → add carry-in); the same
  kernel runs stages 1 and 3 with a carry-in arg.
- **fp32 accumulate**; vectorize **load and store** (scan touches the whole tensor both ways).
- **pair-scan**: carry `(a,b)` per lane (or bitcast-pack into int64); combine `(a₁a₂, a₂b₁+b₂)` in the
  shuffle — **you control operand order**, so no #2359.

## Numerics / parity
fp32 accumulate; tree order differs from torch sequential (bf16 LSB). Integer MoE scans are exact
(bitwise parity). No Triton non-commutative bug here. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
`hipcc --offload-arch=gfx942` → `.so`, torch custom op. HIP source is the edit surface. This is the path
for a production SSM/linear-attention scan that must be correct at long sequence length.

## Pitfalls & anti-patterns
- **5-step (32-lane) `__shfl_up`** from CUDA → wrong scan (drops half the wave).
- `__shfl_up` is **not** memory-barriered — `__syncthreads()` between the LDS carry write and read.
- Half-float `__shfl` unsupported → scan as int/float and repack.
- Forgetting the exclusive carry-add in the block scan → per-wave-correct but block-wrong result.
- Inclusive vs exclusive off-by-one (MoE offsets want exclusive).

## How to verify
`--save-temps` ISA: `ds_*` for LDS carries, vectorized `dwordx4` load/store, no `scratch_`; fp32 atol vs
`torch.cumsum`; **test the SSM recurrence at seq ≥ 128** (where Triton fails) against a reference.

## Alternatives / cross-links
[triton.md](triton.md) (faster to author, but #2359 for non-commutative) · [../tuning.md](../tuning.md) ·
[../fusion.md](../fusion.md) · [`../../../languages/hip_cpp/patterns.md`](../../../languages/hip_cpp/patterns.md) §1
(cross-lane), [`../../../languages/hip_cpp/lds_async.md`](../../../languages/hip_cpp/lds_async.md) (barriers).

## Sources
- wave64 `__shfl_up`, 64-bit masks, no implicit barrier, half-float shuffle unsupported: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- Hillis-Steele vs Blelloch up/down-sweep: https://moderngpu.github.io/scan.html
- warp→block hierarchical scan, first-order recurrence (CUDA ref to port): https://github.com/proger/accelerated-scan
