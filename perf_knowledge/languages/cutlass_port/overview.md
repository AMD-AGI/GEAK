---
title: CUTLASS / CuTe on ROCm — portability overview
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [both]
status: na
updated: 2026-06-08
sources:
  - https://github.com/nvidia/cutlass
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
  - https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html
---

# CUTLASS / CuTe on ROCm

## TL;DR
CUTLASS (and its **CuTe** core: layout algebra + tiled-MMA/copy) is **NVIDIA-native** and does **not** run
on AMD/CDNA. Direct hipification is impractical: it depends on **libcu++ (`cuda/std/*`) headers**,
NVIDIA-only **TMA descriptors** and **`mma`/`wgmma`** instructions, the **32-vs-64 warp/wave** mismatch,
and the CuTe **DSL ships under NVIDIA EULA** (not a permissive license). AMD's response is **not a line-by-
line port** but functional equivalents that keep CuTe's *tile/layout-algebra programming model* on CDNA:
**Composable Kernel (CK)**, **hipBLASLt**, ROCm **Triton**, and the new MLIR-native **FlyDSL** (which
explicitly borrows CuTe layout-algebra *concepts* from the BSD-3 parts only). For the AMD status of the
header port specifically, see [status_amd.md](status_amd.md). In perf_knowledge the `cutlass` backend is recorded
as **`na` (NVIDIA-native)**; this card explains why and what to use instead.

## Concepts (what CUTLASS/CuTe is, so the gap is clear)
- **CUTLASS** — NVIDIA's C++ template library for high-performance GEMM/conv/attention, built around
  composable warp/threadblock/tile collectives targeting Tensor Cores (Ampere SM80 → Blackwell).
- **CuTe** — the modern core: a **layout algebra** (shapes + strides as composable `Layout` objects) plus
  tiled `MMA`/`Copy` atoms. The CuTe **DSL** is a Python front-end (pip `nvidia-cutlass-dsl`).
- **Hard NVIDIA couplings:** `cuda/std/*` (libcu++) headers; **TMA** (`cp.async.bulk` + tensor-map
  descriptors); **`wgmma`/`tcgen05`** async tensor-core matmuls; warp = 32 threads baked into atoms;
  `mma.sync` PTX. None of these exist on CDNA (no TMA, no wgmma, wave = 64 — see
  [../hipkittens/primitives.md](../hipkittens/primitives.md)).

## The AMD equivalents (use these instead)
| need | AMD path | note |
|---|---|---|
| production dense/grouped GEMM, FMHA | **CK / ck_tile** | the CUTLASS analogue on CDNA; turnkey templates. Long build times, brittle compiler interactions. |
| default BLAS GEMM (fused epilogues, fp8) | **hipBLASLt** | Tensile-generated `Cijk_*`; the executed dense-GEMM kernel on Instinct |
| CuTe-style layout-algebra DSL authoring | **FlyDSL** | Python-first MLIR-native; references CuTe layout-algebra *concepts* (BSD-3 parts only, no EULA code) to ease porting CuTe/CUTLASS-derived workloads (DeepGEMM, FlashMLA, XFormers) |
| portable WMMA-style fragments | **rocWMMA** | mirrors `nvcuda::wmma`, not CUTLASS, but portable ([../rocwmma/overview.md](../rocwmma/overview.md)) |
| fast-iteration tile kernels | **Triton** (ROCm) | CDNA3 backend; underperforms tuned libs on plain GEMM |

## Pitfalls
- **Do not try to HIPIFY CUTLASS wholesale.** HIPIFY converts runtime API calls but cannot supply
  libcu++, TMA, or `wgmma`; you hit hard compile failures (cf. real-world reports where CUTLASS deps block
  HIP builds). The wave-size assumption alone breaks CuTe atoms.
- **CuTe DSL is EULA-licensed** (pip `nvidia-cutlass-dsl` bundles a compiler under NVIDIA Software EULA) —
  not redistributable like permissive OSS. AMD ports reference only the BSD-3 layout-algebra concepts.
- **CUTLASS-derived workloads** (DeepGEMM, FlashMLA, XFormers, FasterTransformer) are exactly the ones that
  "are difficult to hipify" — plan to **re-implement on CK/FlyDSL**, not translate.
- Don't list a `cutlass` SOTA card as a real AMD backend — mark `na` with this reason.

## Verify
- Confirm a given CUTLASS-based repo's NVIDIA couplings (`grep` for `cuda/std`, `cp.async.bulk`, `wgmma`,
  `tcgen05`, `__shfl_*` with 32-lane assumptions) before estimating port cost.
- Check the FlyDSL blog / aiter FlyDSL ops for the equivalent kernel (`languages/flydsl/`,
  `operators/dense_gemm/backends/flydsl.md`).

## Sources
- NVIDIA CUTLASS repo / CuTe DSL (NVIDIA-native, SM80+): https://github.com/nvidia/cutlass
- HIP porting guide (HIPIFY scope, 32-vs-64 warp size, unsupported capabilities):
  https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- FlyDSL (CuTe-concept-inspired AMD DSL; BSD-3 references; targets DeepGEMM/FlashMLA/XFormers):
  https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html
- CK as CUTLASS-interface alternative request: https://github.com/ROCm/composable_kernel/issues/900
- detail: [status_amd.md](status_amd.md)
