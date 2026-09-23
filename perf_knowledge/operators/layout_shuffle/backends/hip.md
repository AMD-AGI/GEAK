---
title: layout_shuffle on HIP — SOTA card
kind: sota_card
operator: layout_shuffle
backend: hip
gens: [gfx942, gfx950, gfx1151]
dtypes: [bf16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/shuffle.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
  - https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
---

# layout_shuffle × HIP

## TL;DR
You author a HIP layout permute only when writing a **bespoke MFMA microkernel** whose operand fragment order
differs from aiter's `shuffle_weight` layouts — then you define the matching offline shuffle yourself. For any
weight consumed by an aiter GEMM/MoE, use [backends/aiter.md](aiter.md)'s `shuffle_weight` (it already matches
the kernels). The shuffle is one-time; the design work is making the on-device layout LDS-bank-conflict-free
for your kernel's `ds_*_b128` reads.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| custom offline permute matching your MFMA fragment | this card / AMD lab notes | gfx942/950, all dtypes | one-time; enables conflict-free operand load in your kernel — measure the kernel, not the shuffle | bespoke MFMA microkernel |
| reuse aiter `shuffle_weight` layout | [backends/aiter.md](aiter.md) | both | matched to aiter kernels | weight for an aiter GEMM |

## Config space / knobs
- Fragment shape: N-lane × K-pack to match your `__builtin_amdgcn_mfma_*` operand layout.
- On-device layout must give **conflict-free `ds_*_b128`** operand loads (apply the XOR-swizzle / padding
  rules of [[operators/transpose/tuning.md]] §2–3, [[languages/hip_cpp/lds_async.md]]).
- Offline transform: a torch `view`+`permute`+`contiguous` (CPU/GPU) is usually enough — the shuffle itself
  rarely needs a hand HIP kernel; HIP is for the **consuming GEMM**.

## Numerics / parity
value-preserving (exact); risk is layout-vs-kernel mismatch → GEMM `allclose` catches it. See
[[operators/layout_shuffle/numerics.md]].

## Integration (rebind seam)
Shuffle at load (torch), consume in your HIP MFMA kernel (the Tier-C seam). No aiter bpreshuffle key unless
you route through aiter's dispatch — a fully custom kernel owns its own layout contract.

## Pitfalls & anti-patterns
- ⚠ Re-inventing aiter's layout for an aiter kernel — just call `shuffle_weight`.
- ⚠ A custom layout that isn't conflict-free for your `ds_read` pattern → bank conflicts (the −75% BW class).
- ⚠ Shuffling at runtime per call instead of once at load.

## How to verify
GEMM output `allclose`; rocprofv3 → your MFMA kernel's operand `ds_read` has ~0 bank conflicts; the layout
matches the fragment (no in-kernel reshuffle).

## Alternatives / cross-links
[backends/aiter.md](aiter.md) · [backends/triton.md](triton.md) · [[operators/transpose/tuning.md]] ·
[[languages/hip_cpp/lds_async.md]] · [[languages/hip_cpp/intrinsics.md]].

## Sources
- aiter shuffle layouts (reference for fragment order): ROCm/aiter@a6bb49937:aiter/ops/shuffle.py.
- MFMA operand layout / LDS staging: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html · https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/

## On gfx1151 (RDNA3.5, Strix Halo)
`gfx1151` appears in `gens:` because this backend's **source is portable** to RDNA — it
compiles/JITs there with no vendor asset table. It is **not** a claim that anything on this
card was measured on RDNA: every perf number, ranking and tuning recipe above is CDNA
(gfx90a/942/950) evidence. Three differences bite before any of it transfers — WMMA not MFMA,
wave32 not wave64, and 40 CU behind a 32 MB MALL on ~229 GB/s shared LPDDR5X rather than HBM —
so tile shapes, occupancy targets and the roofline ceiling all move. Any `fp8_*` entry in
`dtypes:` above is CDNA-only: gfx1151 has **no fp8 matrix instruction and no block-scaled
FP4/FP6**, so an fp8 candidate there runs EMULATED — it passes correctness and loses
performance silently. See [`../../../hardware/rdna35_gfx1151/`](../../../hardware/rdna35_gfx1151/)
and MEASURE on the box.
