---
title: depthwise_conv on HIP/C++ — SOTA card
kind: sota_card
operator: depthwise_conv
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
---

# depthwise_conv × HIP/C++

## TL;DR
A hand-written HIP kernel is the **last-resort** depthwise backend: use it only when MIOpen has no fast
solver for your exact shape and Triton can't express the fusion. Depthwise conv is memory-bound (no
channel reduction, no matrix core), so a good HIP kernel is a coalesced NHWC load + a small in-register
filter + an LDS-staged spatial halo if reused — there is no MFMA to write. For nearly all production
depthwise conv, prefer **MIOpen** ([miopen.md](miopen.md)). The HIP value here is full control of LDS
halo layout and the conv+bias+act epilogue for one pinned shape. (The 1D causal variant has a real,
shipping HIP kernel — see [[causal_conv1d]]/hip; 2D depthwise does not, on this box.)

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Hand-written HIP depthwise conv (NHWC, register filter, LDS halo) | author via kernel layer; HIP kernel-language + workload guide | gfx942/950; fp16/bf16/fp32 | shape-specific; no on-box measurement (vision tail) | a single pinned shape MIOpen/Triton don't cover well |

Honest gap: there is **no on-box 2D depthwise HIP kernel** to cite — this card is a design recipe, not a
measured artifact. Benchmark vs MIOpen before committing.

## Config space / knobs
- Block = multiple of 64 (wave64); one wave per channel-tile is a natural mapping (cf. the causal_conv1d
  HIP kernel's 1-wave-per-batch pattern).
- `BLOCK_C` channel-tile coalesced along the contiguous (NHWC) axis; spatial tile `BLOCK_H×BLOCK_W`.
- Hold the `R×S` filter in registers (`#pragma unroll`); LDS-stage the input halo only if the spatial
  tile reuses overlapping rows (3×3 stride-1 → halo of 1).
- `__launch_bounds__(TPB, minWavesPerEU)` to cap VGPRs (avoid scratch spill); `__restrict__` for
  `global_load_dwordx4`; `-munsafe-fp-atomics` only if you split-reduce (depthwise usually doesn't).
- Grid ≥1024 workgroups to feed 304 CUs.

## Numerics / parity
fp32 accumulate over the spatial window, cast on store; same-math vs `F.conv2d(groups=C)`, `atol≈1e-2`
bf16. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Register as a custom op and rebind the model's conv call site; e2e-gate. No library dispatch DB to hook —
this fully replaces the conv for the pinned shape, so guard it to that shape and fall back to MIOpen
otherwise.

## Pitfalls & anti-patterns
- ⚠ Re-implementing what MIOpen Winograd already does well (3×3 stride-1) — you will rarely beat it; only
  author for shapes it lacks.
- ⚠ Over-aggressive `minWavesPerEU` → VGPR/scratch spill (HBM) → 3–5× slower.
- ⚠ Forgetting NHWC coalescing → uncoalesced loads dominate a memory-bound op.
- First call JIT/AOT compile cost; warm before timing.

## How to verify
`-Rpass-analysis=kernel-resource-usage` (VGPR/LDS/scratch); `--save-temps` ISA (vectorized loads, no
`scratch_`); isolated bench vs MIOpen `MIOpenDriver --group-count C` at the same shape; parity vs
`F.conv2d(groups=C)`.

## Alternatives / cross-links
[miopen.md](miopen.md) (production default) · [triton.md](triton.md) (easier authoring) ·
[../overview.md](../overview.md) · language: [`../../../languages/hip_cpp/`](../../../languages/hip_cpp/) ·
LLM 1D variant with a real HIP kernel: [[causal_conv1d]].

## Sources
- HIP kernel language (wave64, __launch_bounds__, __restrict__): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- MI300X workload optimization (≥1024 grid, VGPR/LDS budgets, memory-bound tuning): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- LDS banks / occupancy (halo staging): https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
