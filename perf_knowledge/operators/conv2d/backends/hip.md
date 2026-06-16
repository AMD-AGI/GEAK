---
title: conv2d on HIP/C++ — SOTA card
kind: sota_card
operator: conv2d
backend: hip
gens: [gfx942, gfx950]
dtypes: [fp32, bf16, fp16]
regimes: [both]
status: competitive
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# conv2d × HIP/C++

## TL;DR
Hand-writing conv2d in HIP means hand-writing an **implicit-GEMM with MFMA** — i.e. a GEMM µkernel whose
A-tile load folds the `p·s+r·d` gather (the conv→GEMM coordinate transform). This is **expensive to author
and you will rarely beat MIOpen/CK**, which already have tuned XDL implicit-GEMM solvers. Reach for HIP
only when (a) no MIOpen/CK instance covers your shape, or (b) you need a fusion neither can express, for a
single pinned shape. For the LLM 1D conv, HIP *is* the right tool (small, register-window) — see
[[causal_conv1d]]/hip; for 2D, prefer the libraries.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Hand-written HIP implicit-GEMM conv (MFMA, conv→GEMM gather in the A-load, CShuffle-style epilogue) | author via kernel layer; HIP MFMA intrinsics + workload guide | gfx942/950; fp16/bf16/fp32 | only worthwhile if it beats MIOpen/CK on a shape they miss; no on-box measurement | uncovered shape / custom fusion, pinned |

Honest gap: no on-box 2D conv HIP kernel to cite — this is a design recipe. Benchmark vs MIOpen/CK first.

## Config space / knobs
This is GEMM tuning (because conv2d is implicit-GEMM): tile `MPerBlock/NPerBlock/KPerBlock` (M=N·P·Q,
N=K, K_gemm=C·R·S), `mfma_16x16` (`__builtin_amdgcn_mfma_*`), LDS A/B staging with XOR swizzle, double-
buffer pipeline, CShuffle epilogue (re-tile the scattered MFMA accumulator through LDS before the global
store). Block = multiple of 64; grid ≥1024 workgroups; `__launch_bounds__` to cap VGPR; `-munsafe-fp-
atomics` if split-K. See [`../../../languages/hip_cpp/`](../../../languages/hip_cpp/) (intrinsics,
lds_async, patterns) and [`../../../languages/asm_mfma/`](../../../languages/asm_mfma/).

The conv-specific part is the **A-tile gather**: instead of a contiguous `A[m,k]` load, compute the
`(n,c,h,w)` source index from `(m=n·P·Q, k=c·R·S)` and load with bounds-checked padding — the
`transform_conv_fwd_to_gemm` logic, hand-rolled. Forward maps to K-contiguous (no bank conflict).

## Numerics / parity
fp32 MFMA accumulate, cast on store; same-math vs `F.conv2d`, `atol≈1e-2` bf16. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
Register a custom op, guard it to the pinned shape, fall back to MIOpen otherwise; e2e-gate. No library
dispatch DB to hook.

## Pitfalls & anti-patterns
- ⚠ Re-implementing MIOpen/CK XDL implicit-GEMM — they're hard to beat; only for uncovered shapes.
- ⚠ Naive im2col (materializing the gather to HBM) — defeats the point; fold the gather into the A-load.
- ⚠ `num_warps`/VGPR over-subscription → scratch spill (HBM) → 3–5× slower.
- ⚠ Padding/OOB in the gather must be guarded (pad reads → 0), or you read garbage at the borders.
- First call JIT/AOT compile; warm before timing.

## How to verify
`-Rpass-analysis=kernel-resource-usage`; `--save-temps` ISA (want `v_mfma_*16x16`, `global_load_dwordx4`,
LDS `ds_read_b128`, no `scratch_`); isolated bench vs MIOpen/CK at the same shape; parity vs `F.conv2d`.

## Alternatives / cross-links
[miopen.md](miopen.md) (production default) · [composable_kernel.md](ck.md) (tuned XDL
implicit-GEMM — prefer this over hand HIP) · [../overview.md](../overview.md) · languages:
[`../../../languages/hip_cpp/`](../../../languages/hip_cpp/),
[`../../../languages/asm_mfma/`](../../../languages/asm_mfma/) · LLM 1D variant: [[causal_conv1d]].

## Sources
- HIP kernel language + MFMA intrinsics (wave64, __launch_bounds__, builtins): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- Implicit-GEMM convolution (conv→GEMM, gather in A-load, no im2col): https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html
- MI300X workload optimization (mfma_16x16, ≥1024 grid, VGPR/LDS, ~45–55% of peak): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
