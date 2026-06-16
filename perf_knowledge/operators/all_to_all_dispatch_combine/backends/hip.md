---
title: all_to_all_dispatch_combine on HIP — SOTA card
kind: sota_card
operator: all_to_all_dispatch_combine
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: competitive
updated: 2026-06-08
sources:
  - https://gau-nernst.github.io/amd-a2a/
  - https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
  - https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
  - https://arxiv.org/abs/2506.04667
---

# all_to_all_dispatch_combine × HIP

## TL;DR
Authoring the dispatch/combine directly in HIP (over **symmetric memory / GPU-initiated xGMI**, e.g.
MoRI-SHMEM / rocSHMEM primitives) is the research path and the way to understand the collective — the
reference single-kernel a2a study took a naive **93,540 µs → 292 µs** on MI300X with hand-written kernels.
For production prefer MoRI-EP ([backends/mori.md](mori.md)); author HIP to push past it or to
prototype the **combine→GEMM-epilogue** single-kernel ideal.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| authored P2P symmetric-memory dispatch+combine | gau-nernst (GPU MODE AMD Distributed Challenge) | gfx942, bf16/fp8 | **292 µs** (256 experts, topk 8, hidden 7168, 256 tok, world 8); ladder 517→345→303→292 µs, MI300X | research / pushing past MoRI |
| MoRI-SHMEM / rocSHMEM primitives | mori / ROCm/DeepEP | gfx942/950 | underlies MoRI-EP & DeepEP-ROCm | building a custom EP path |

## Config space / knobs
- **`grid_size = 304`** (exact MI300X CU count) — gave ~3× combine-send speedup vs 256 (the headline finding).
- **Hoist `torch.empty`/memset** out of the kernel (caching-allocator `hipMalloc` dominated traces) — use
  pre-registered symmetric buffers.
- 128-bit vectorized token copies; FP8 dispatch / BF16 combine to halve dispatch bytes.
- Stay in one xGMI island (≤8 GPU) intra-node; RDMA (IBGDA) inter-node.

## Numerics / parity
combine reduction order + FP8 dispatch → accuracy-gate. See [[operators/all_to_all_dispatch_combine/numerics.md]].

## Integration (rebind seam)
`.hip` over rocSHMEM/MoRI-SHMEM symmetric memory, bound via a torch custom op + process-group init. The Tier-C
seam for a bespoke EP collective; otherwise rebind to MoRI-EP.

## Pitfalls & anti-patterns
- ⚠ `grid_size != 304` (e.g. 256) → much slower / "didn't work" (authored-kernel finding).
- ⚠ Allocation/memset inside the hot path → `hipMalloc` in traces dominates.
- ⚠ Re-implementing MoRI by hand for production — MoRI-EP is faster and maintained; author only to beat it or
  to reach the single-kernel combine→GEMM ideal (FlashDMoE north-star, not yet on AMD).
- ⚠ xGMI link health: baseline with `rccl-tests` first (~316–330 GB/s) — a slow link masquerades as a slow kernel.

## How to verify
`rccl-tests` fabric baseline; rocprofv3 → kernels present, **grid 304**, no `hipMalloc` steady-state,
overlapping grouped-GEMM; numeric parity vs torch MoE.

## Alternatives / cross-links
[backends/mori.md](mori.md) (production SOTA) · [backends/aiter.md](aiter.md) ·
[[backends/mori_rccl/mori_ep.md]] · [[operators/all_to_all_dispatch_combine/tuning.md]] ·
[[languages/hip_cpp/patterns.md]].

## Sources
- Authored a2a (grid_size=304, 292 µs ladder, malloc/memset hoist): https://gau-nernst.github.io/amd-a2a/
- MoRI-SHMEM / symmetric-memory primitives: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- xGMI fabric baseline: https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- Single-kernel combine→GEMM north-star (FlashDMoE): https://arxiv.org/abs/2506.04667
