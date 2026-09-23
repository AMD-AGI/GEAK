---
title: embedding on hip — SOTA card
kind: sota_card
operator: embedding
backend: hip
gens: [gfx908, gfx90a, gfx942, gfx950, gfx1151]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
---

# embedding × hip

## TL;DR
The de-facto HIP embedding is **`torch.embedding`'s ROCm gather** (a rocPRIM/library index gather),
which both vLLM and SGLang call via `F.embedding`. Hand-writing a HIP gather is a one-afternoon kernel
(coalesced per-row copy) that **matches but does not beat** the library — it is bandwidth-bound. The only
reason to author one in HIP is to **fuse** the vocab-parallel mask/zero-fill into the same kernel without
relying on `torch.compile`, or to fuse the gather with a downstream cast/quant — niche, ≪1% Amdahl.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `torch.embedding` ROCm gather (library) | PyTorch ROCm `aten::embedding` → rocPRIM gather | all gfx9, bf16/fp16/fp32 | bandwidth-bound row copy; ≪1% GPU time (no isolated gate) | the default and recommended path |
| hand-HIP coalesced gather `out[t]=W[ids[t]]` | author-it-yourself (`__restrict__`, `dwordx4` row copy) | all gfx9 | equals library; only wins by *fusing* mask/cast into the same launch | when you must avoid `torch.compile` yet still fuse the mask |

## Config space / knobs
For a hand-HIP gather (see [[hip_cpp]] patterns.md/intrinsics.md):
- **One workgroup per few rows**, threads stride the `d`-element row; use `global_load_dwordx4` /
  `__restrict__` for wide coalesced reads (row = `d·dtype` bytes, naturally 128B-aligned for hidden
  ∈ {4096,5120,8192}).
- Block = multiple of 64 (wave64); 256 threads typical. `__launch_bounds__` rarely needed (VGPR-light).
- Grid ≥1024 WGs to fill 304 CUs only reachable at large `T`; for decode (`T`≤256) it is launch-latency
  bound — keep it a single small kernel, or skip the custom kernel entirely.
- `-munsafe-fp-atomics` is irrelevant (no reduction); the gather is pure load/store.

## Numerics / parity
Bit-exact copy. Fused mask is integer remap + conditional zero — exact. No tolerance band. The correctness
risk is the **mask boundary** at vocab-parallel shard edges (see [../numerics.md](../numerics.md)).

## Integration (rebind seam)
No framework exposes a named HIP embedding op — the live call is `F.embedding` inside
`VocabParallelEmbedding.forward`. A hand-HIP kernel would be wired as a custom op
(`torch.library` / `csrc/rocm/torch_bindings.cpp`-style) and a rebuild; not a runtime swap. There is no
production demand for this, so the seam is theoretical.

## Pitfalls & anti-patterns
- Writing a HIP gather expecting a speedup — there is none; you only take on maintenance + a rebuild.
- `warpSize==32` grid math (it is **64** on CDNA) → wrong occupancy.
- Skipping the OOB mask for vocab-parallel ids → OOB global read (segfault or garbage row).

## How to verify
rocprofv3: the gather row (`embedding`/`CopyKernel`) is ≪1% of GPU time. Oracle: `W[ids]` bit-exact.

## Alternatives / cross-links
[triton.md](triton.md) · [vllm_kernels.md](vllm_kernels.md) (the live path) · [../overview.md](../overview.md) ·
[[hip_cpp]] · [[gather_scatter]].

## Sources
- HIP kernel language (wave64, `__restrict__`, `__launch_bounds__`): https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- Coalesced reads / ≥1024 grid: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Live `F.embedding` path: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py

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
