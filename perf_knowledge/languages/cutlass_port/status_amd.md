---
title: CUTLASS / CuTe on AMD — port status & blockers
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3]
regimes: [both]
status: na
updated: 2026-06-08
sources:
  - https://github.com/nvidia/cutlass
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
  - https://github.com/ROCm/composable_kernel/issues/900
  - https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html
---

# CUTLASS / CuTe on AMD — status

## TL;DR
**Status: `na` — no usable CUTLASS/CuTe on CDNA, and none planned.** NVIDIA targets CUTLASS at NVIDIA GPUs
only (Ampere SM80 → Blackwell); CUTLASS 3.9 (Apr 2026) continues to add only NVIDIA tile shapes (Blackwell
FP8 MMA) and makes CuTe layout algebra the primary API while CUTLASS 2.x goes to maintenance. ROCm fully
supports CDNA hardware (MI350X/MI355X since ROCm 7.0; ROCm 7.2 current), but **CUTLASS itself has no CDNA
backend**. The practical path on AMD is **CK / hipBLASLt / Triton / FlyDSL**, not a port.

## The four hard blockers
1. **libcu++ headers.** CUTLASS pulls `cuda/std/*` (NVIDIA-only). Builds proceed until CUTLASS, then fail
   on these includes — the documented workaround is "use hipBLASLt or Composable Kernel instead."
2. **NVIDIA-only instructions.** Tensor-core ops are emitted as `mma`/`wgmma`; bulk copy as **TMA**
   (`cp.async.bulk` + tensor-map descriptors). CDNA has **none** of these (only MFMA; async
   buffer-load-to-lds for copy). HIPIFY does not synthesize them.
3. **Warp/wave size.** CuTe atoms assume **warp = 32**; CDNA wave = **64**. Code must use `warpSize`/device
   query, not a literal 32 — a pervasive assumption throughout CUTLASS that breaks tile/atom math.
4. **Licensing.** The CuTe **DSL** (pip `nvidia-cutlass-dsl`) ships a compiler under **NVIDIA Software
   EULA** with CUDA-SDK-like restrictions — not permissively redistributable. Only the BSD-3-licensed
   layout-algebra *concepts* can be reused downstream.

## What HIPIFY can and can't do
- **Can:** mechanically translate runtime API (e.g. `cuEventCreate → hipEventCreate`), driver→HIP error
  codes, naming conventions.
- **Can't:** provide libcu++, emit TMA/`wgmma`, or fix the 32→64 wave assumption. "Additional porting is
  required for architecture feature queries / CUDA capabilities HIP doesn't support" — and CUTLASS leans
  *entirely* on those capabilities. Net: wholesale hipification of CUTLASS is not viable.

## AMD's actual strategy: equivalents, not a port
- **Composable Kernel (CK)** — AMD's tile-based GEMM/attention library, the CUTLASS analogue on CDNA.
  Long-requested as a "CUTLASS interface" because CUTLASS-based projects (FasterTransformer, xformers) are
  hard to hipify. Trade-offs: long build times, slow iteration, brittle compiler interactions, steep
  onboarding.
- **hipBLASLt** — Tensile `Cijk_*`; the default executed dense-GEMM kernel on Instinct (fused epilogues,
  fp8).
- **FlyDSL** — the most direct CuTe-equivalent: Python-first, MLIR-native, **references CuTe layout-algebra
  concepts (BSD-3 only, no EULA code)**. Built explicitly to "open a smoother path for AMD enablement of
  open-source projects already based on CuTe DSL / CUTLASS-like abstractions" and to accelerate
  CUTLASS-derived ODM workloads — **DeepGEMM, FlashMLA, XFormers**. See
  `operators/dense_gemm/backends/flydsl.md`.

## Porting a CUTLASS-derived workload to AMD (the realistic plan)
1. Identify the NVIDIA couplings (`grep` `cuda/std`, `cp.async.bulk`, `wgmma`, `tcgen05`, 32-lane shuffles).
2. **Re-implement the hot GEMM/attention in CK or FlyDSL**, not translate CUTLASS line-by-line.
3. Keep the host/glue logic, swap the kernel layer.
4. Validate parity + bench vs hipBLASLt/aiter; e2e-gate.

## perf_knowledge recording
- Backend id `cutlass` = **`na` (NVIDIA-native)** per taxonomy. This file is the documented reason.
- The *authorable* AMD equivalents are `ck`, `hipblaslt` (lib), `flydsl`, `triton`, `rocwmma`.

## Sources
- NVIDIA CUTLASS/CuTe (NVIDIA-only target, SM80→Blackwell; 3.9 maintenance note):
  https://github.com/nvidia/cutlass
- HIP porting guide (HIPIFY scope, 32-vs-64 warp size): https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- CUTLASS-deps-block-HIP real report (workaround = hipBLASLt/CK): https://github.com/ROCm/composable_kernel/issues/900
- FlyDSL (CuTe-concept-based AMD DSL; BSD-3; DeepGEMM/FlashMLA/XFormers):
  https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html
- ROCm CDNA support / version matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
- overview: [overview.md](overview.md)
