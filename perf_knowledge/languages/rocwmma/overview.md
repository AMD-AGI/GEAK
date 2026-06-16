---
title: rocWMMA — WMMA-style C++ Matrix-Core fragment API on CDNA
kind: language
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, int8, fp32]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/rocWMMA/en/develop/api-reference/api-reference-guide.html
  - ROCm/rocWMMA@develop:samples/simple_hgemm.cpp
  - https://gpuopen.com/learn/wmma_on_rdna3/
---

# rocWMMA

## TL;DR
rocWMMA is AMD's **header-only C++ library** that exposes the Matrix Cores through a typed
`fragment`/`mma_sync` API deliberately **mirroring NVIDIA's `nvcuda::wmma`**. It is the *portable*
authoring middle ground: more ergonomic than raw `__builtin_amdgcn_mfma_*` intrinsics, but you still own
LDS staging and pipelining, so it rarely beats CK/ck_tile on a turnkey dense GEMM. Reach for it when you
need a **custom fused kernel** with a matmul inside that must also port to CUDA via WMMA, not to ship a
production GEMM. Min ROCm 6.4; header at `/opt/rocm/include/rocwmma/rocwmma.hpp`. The standalone
`ROCm/rocWMMA` repo is deprecated — it now lives in `ROCm/rocm-libraries`.

## Concepts
- **Fragment = typed view of MFMA lane registers.** A `rocwmma::fragment<...>` is a small object held in
  packed VGPRs that holds *this lane's share* of a BlockM×BlockN×BlockK tile. rocWMMA hides the scattered
  MFMA lane layout: `load_matrix_sync` distributes a tile from memory into the right lanes, `mma_sync`
  issues the right `v_mfma_*`, `store_matrix_sync` gathers back.
- **No element locality.** Vector elements inside a fragment have *no guaranteed order or locality* —
  never index `frag.x[i]` assuming a row/col; only do *elementwise* math (alpha/beta scaling, activation).
- **Wave-cooperative.** `mma_sync`/`load`/`store` are warp-synchronous: all **64 lanes** of the wavefront
  (WaveSize = 64 on CDNA) execute them together. With LDS source/dest you may need an explicit
  `synchronize_workgroup()` between produce and consume.
- **fp32 accumulate always.** On CDNA the matrix unit accumulates in 32-bit and converts to the output
  type — so for bf16/fp8 inputs the `accumulator` fragment is `float` even if you store bf16.
- **Maps 1:1 to MFMA.** `fragment<...,16,16,16,bf16>` + `mma_sync` lowers to `v_mfma_f32_16x16x16_bf16`;
  `<32,32,8>` lowers to `v_mfma_f32_32x32x8_*`. Verify with `ROCm/amd_matrix_instruction_calculator`.

## The levers
- **Fragment tile shape** — `16×16×16` (default) vs `32×32×8`. Prefer **16×16** on MI300X: the same
  power/clock fact as raw MFMA holds — 16×16 generally yields higher achievable FLOPs than 32×32.
- **dtype triple** `<Ti/To/Tc>` — bf16/bf16/**f32**, fp8/f32/f32 (BlockK=32, 2× K density; gfx942/950
  only), i8/i32/i32. Match the gfx942 FNUZ fp8 format unless OCP fp8 is explicitly selected.
- **Waves/block** — up to 4 wavefronts per thread block; valid `(TBlock_X, TBlock_Y)` include
  `WaveSize×{1,2,4}`, `2·WaveSize×{1,2}`, `4·WaveSize×1`. Larger output tile per block trades occupancy.
- **Scheduler template param** — `default_schedule` vs the `coop_*` schedulers (see
  [patterns.md](patterns.md)) for multi-wave shared-tile coalescing.
- **LDS staging** — the *real* perf lever, and entirely manual. Stage tiles through LDS, reuse across the
  block, software-pipeline. The `samples/perf_hgemm.cpp` sample does this; at that point you are
  re-implementing what CK already does.

## Pitfalls
- **Naïve global-load GEMM is not competitive.** The `simple_hgemm` pattern (load A/B from global each
  K-step) is for learning only. Real perf needs LDS + reuse + pipelining you must write yourself.
- **Don't index fragment elements positionally** — only elementwise ops are layout-safe.
- **BlockK is small in practice (≤ 32).** `BlockM/N` below the minimum recommended → padding → perf hit.
  Partial fragments (FragMNK < BlockMNK) are internally padded to the nearest supported block (costs perf).
- **fp8/f64 require gfx942/gfx950.** On gfx908/gfx90a these dtype triples are unavailable; bf16 BlockK was
  8 on gfx908, 16 on gfx942+.
- **For production dense GEMM/FMHA/MoE, use CK/ck_tile, not rocWMMA.** rocWMMA's strength is portability
  and fusing MFMA into bespoke kernels, not beating a tuned library.

## Verify
- ISA: `hipcc --offload-arch=gfx942 -O3 -I/opt/rocm/include hgemm.cpp -S` and confirm the expected
  `v_mfma_f32_16x16x16_bf16` and `ds_read`/`global_load` placement.
- Confirm the fragment→MFMA mapping with `ROCm/amd_matrix_instruction_calculator`.
- Bench against hipBLASLt/CK on the exact `(M,N,K,dtype)`; expect to *lose* on plain GEMM unless you have
  built full LDS staging + pipelining.

## Sources
- rocWMMA API Reference (fragment template, load/mma/store_matrix_sync, tile-shape & dtype tables,
  schedulers): https://rocm.docs.amd.com/projects/rocWMMA/en/develop/api-reference/api-reference-guide.html
- rocWMMA GitHub (samples `simple_hgemm.cpp`, `perf_hgemm.cpp`; now under ROCm/rocm-libraries):
  https://github.com/ROCm/rocWMMA — `ROCm/rocWMMA@develop:samples/simple_hgemm.cpp`
- AMD GPUOpen, "How to accelerate AI with WMMA" (fragment model, load/store/mma_sync semantics):
  https://gpuopen.com/learn/wmma_on_rdna3/
- Matrix Core Programming on CDNA3/CDNA4 (MFMA shapes/dtypes rocWMMA fragments map to):
  https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMD CDNA3 ISA Ref Ch.7 Matrix Arithmetic (instructions behind `mma_sync`):
  https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- amd_matrix_instruction_calculator (verify fragment→MFMA lowering):
  https://github.com/ROCm/amd_matrix_instruction_calculator
