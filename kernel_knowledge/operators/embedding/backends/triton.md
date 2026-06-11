---
title: embedding on triton — SOTA card
kind: sota_card
operator: embedding
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/pytorch/pytorch/pull/143286
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# embedding × triton

## TL;DR
Triton is **not** the production embedding path — nobody ships a hand-written Triton token-gather, because
`torch.embedding` (library gather) already saturates HBM for `H = W[ids]`. Triton's role is purely
**TorchInductor codegen**: when the model graph is `torch.compile`d, Inductor emits a Triton kernel that
fuses the vocab-parallel mask/zero-fill *with* the gather. Reach for hand-written Triton only to fuse the
gather into an exotic neighbor (rare, ≪1% Amdahl — usually not worth it).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Inductor-emitted fused gather+mask | TorchInductor (Triton AMD backend) `triton-lang/triton@HEAD:third_party/amd` | gfx942/950, bf16/fp16 | not separately benched — rides `torch.compile`; ≪1% GPU time, no isolated gate | when the model is `torch.compile`d (mask + gather collapse to 1 kernel) |
| hand-written `tl.load(W + ids[:,None]*d + arange)` gather | author-it-yourself | any | matches library gather (bandwidth-bound); no win over `torch.embedding` | only to fuse with a custom epilogue |

A standalone Triton gather is a memory-bound `tl.load` with a row offset — it equals, not beats, the
library. The honest recommendation: use Inductor, don't author one.

## Config space / knobs
For a hand-written gather (memory-bound — see [[triton_amd]] knobs.md):
- `BLOCK` = rows per program (one token row of `d` elements each); `num_warps=2..4` (memory-bound, **not**
  8 — wave64 spill trap); `num_stages=1` (no GEMM). `knobs.amd.use_buffer_ops=1` for bounds-checked
  masked loads of foreign/OOB ids.
- Grid ≥1024 WGs to fill 304 CUs only matters at large `T`; with `T`≤16k and one row/program you usually
  can't reach it → the library grid-strided gather is the safer fill.

## Numerics / parity
Bit-exact gather (copy); Inductor fusion of the mask is exact (integer remap + `where(mask,0,row)`). No
tolerance band. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
No serving framework calls a named Triton embedding op — it appears only as an Inductor-generated kernel
inside a compiled graph. To engage: `torch.compile` the model (or vLLM/SGLang's compiled path) and confirm
via `TORCH_LOGS=+inductor` that the embedding + mask lowered to one Triton kernel. AMD GEMM/elementwise
knobs are wired into Inductor (pytorch/pytorch #143286) but the gather has no meaningful knob.

## Pitfalls & anti-patterns
- Authoring a Triton gather expecting a speedup — there is none over `torch.embedding`; you only add
  maintenance.
- `num_warps=8` carried from NVIDIA → VGPR spill on wave64 (see [[triton_amd]] pitfalls).
- Forgetting `mask=` on `tl.load` for vocab-parallel OOB ids → OOB read.

## How to verify
rocprofv3 Top-N: the gather should be a single fused Triton row ≪1% of GPU time. Oracle: `W[ids]`
bit-exact vs the compiled output.

## Alternatives / cross-links
[hip.md](hip.md) · [vllm_kernels.md](vllm_kernels.md) (the live path) · [../overview.md](../overview.md) ·
[[triton_amd]] · [[gather_scatter]].

## Sources
- AMD Triton backend / Inductor codegen path: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- AMD GEMM/elementwise knobs in Inductor: https://github.com/pytorch/pytorch/pull/143286
- Memory-bound op tuning (coalescing, grid): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
