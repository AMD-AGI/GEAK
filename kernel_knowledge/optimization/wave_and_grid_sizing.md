---
title: wave and grid sizing (wave64, workgroup, launch bounds, persistent)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
---

# wave and grid sizing

## TL;DR
CDNA is **wave64** (64 lanes/wavefront) — *not* wave32. A workgroup is `num_warps` wavefronts; choose it
to balance LDS/register footprint against waves/CU. The grid must cover all CUs: **304 CUs on MI300X**,
**256 CUs on MI350X**, with **8 XCDs** (`[[optimization/xcd_l2_locality.md]]`). Use **`__launch_bounds__`**
to bound register use for a target occupancy, and **persistent kernels** when you want explicit
tile→CU/XCD scheduling instead of trusting the dispatcher. See
`[[hardware/shared/wavefront_simd_vgpr_agpr.md]]`, `[[hardware/cdna3_mi300/arch.md]]`,
`[[hardware/cdna4_mi350/arch.md]]`.

## Concepts (the hardware)
- **Wave64**: every wavefront = 64 lanes; divergence and coalescing windows are 64-wide
  (`[[optimization/vectorization_and_coalescing.md]]`). Triton on AMD maps `num_warps` to wave64
  wavefronts.
- **Workgroup**: 1–N wavefronts (≤ slot limits) sharing one CU's LDS. Up to **8 wave slots/SIMD ⇒
  32 waves/CU** (`[[optimization/occupancy_and_registers.md]]`).
- **CU counts**: **MI300X = 304 CUs (8 XCD × 38)**; **MI350X = 256 CUs** (CDNA4 reduced CU count but
  ~2× per-CU matrix throughput). The grid should produce **≥1024 workgroups** to fill all CUs/XCDs on
  prefill-scale work (`[[operators/dense_gemm/tuning.md]]`).

## The levers
- **`num_warps` (triton)**: wavefronts/block. 4–8 typical for GEMM; more warps = bigger block (more
  LDS/regs/block, fewer blocks/CU). Tune jointly with `num_stages` and tile size.
- **Workgroup size (HIP)**: `blockDim` a multiple of 64. Larger blocks share LDS better but raise
  per-block resource use (`[[optimization/occupancy_and_registers.md]]`).
- **`__launch_bounds__(maxThreadsPerBlock, minWavesPerCU)`**: caps VGPRs so the requested occupancy is
  guaranteed; matches the kernel's real block size to avoid the compiler over-allocating registers.
- **Grid coverage**: aim for an integer (ideally ≥1, often several) waves of workgroups across all CUs;
  **≥1024 WGs** and **tile count a multiple of 8** for XCD balance (`[[optimization/xcd_l2_locality.md]]`).
- **Persistent kernels**: launch exactly `num_CU × blocks_per_CU` workgroups that loop over output
  tiles (`for tile in my_tiles: ...`). Benefits: amortize launch overhead, keep weights/state resident,
  explicit tile→XCD mapping for L2 locality, and natural Stream-K reduction
  (`[[operators/splitk_streamk_gemm/overview.md]]`). Cost: you own load balancing.

## Decode vs prefill sizing
- **Prefill (large M)**: compute-bound; large tiles, ≥1024 WGs, full-CU coverage, high MFMA occupancy
  (`[[optimization/mfma_scheduling.md]]`).
- **Decode (skinny M, GEMV-like)**: memory/latency-bound; small `BLOCK_M` (16/32), more SPLIT_K to
  create enough workgroups to fill 304/256 CUs, otherwise CUs starve
  (`[[operators/skinny_gemv_decode/overview.md]]`, `[[operators/splitk_streamk_gemm/overview.md]]`).

## Pitfalls
- Writing/tuning as if wave32 (CUDA habit) — all lane math is **64-wide** on CDNA.
- Grids that produce <304 (MI300X) / <256 (MI350X) workgroups ⇒ idle CUs; <1024 leaves XCDs underfed.
- `__launch_bounds__` set below the actual block size ⇒ forced register spills.
- Persistent kernels with naive static tile partition ⇒ tail imbalance (use atomic work-queue or Stream-K).
- Hardcoding 304 CUs on MI350X — it is **256**; query `hipGetDeviceProperties` for portability.

## Verify
- `hipGetDeviceProperties` → `multiProcessorCount` (CU count), confirm grid ≥ CU count and ≥1024 WGs.
- Omniperf: per-CU occupancy / idle CUs, wavefront launch count (`[[profiling/]]`).
- ISA dump: VGPR count vs `__launch_bounds__` target, no scratch (`[[languages/triton_amd/isa_verify.md]]`).
- A/B: `num_warps ∈ {4,8}`, persistent vs non-persistent for decode.

## Sources
- Wave64, wave slots, `__launch_bounds__` semantics: AMD CDNA3 ISA reference.
- MI300X = 304 CUs / 8 XCD; ≥1024 WG guidance: ROCm workload guide + MI300X architecture deck.
- MI350X = 256 CUs (CU reduction, 2× per-CU matrix throughput): AMD CDNA4 whitepaper.
