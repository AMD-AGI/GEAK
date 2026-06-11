---
title: mrope on hip — SOTA card
kind: sota_card
operator: mrope
backend: hip
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: competitive
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu
  - /sgl-workspace/aiter/aiter/ops/fused_qk_norm_mrope_cache_quant.py
  - https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
---

# mrope × hip

## TL;DR
There is no separate generic HIP mRoPE kernel in the public serving stacks — mRoPE is delivered either as
the aiter **HIP fused** kernel (`fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`, the recommended path) or
as a **Triton** kernel (vLLM). To hand-write mRoPE in HIP, extend the RoPE `rotary_embedding` kernel
(`csrc/pos_encoding.cu`) with a per-section position lookup. Use HIP only to own a custom fused attention
entry; otherwise [aiter.md](aiter.md) / [triton.md](triton.md).

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter HIP fused mrope | `aiter/ops/fused_qk_norm_mrope_cache_quant.py` | gfx942/950, bf16/fp16/fp8 | the serving fused path (HIP-compiled) | VLM serving — [aiter.md](aiter.md) |
| RoPE HIP kernel extended with section split | `vllm/csrc/pos_encoding.cu` (base) | gfx942/950 | bandwidth-bound | custom fused attention entry |
| (no standalone generic mrope HIP) | — | — | — | use aiter/Triton |

## Config space / knobs
- Extend `rotary_embedding`: per dim, select the axis from `mrope_section`, use that axis's position for
  the cos/sin lookup. Precompute the dim→axis map.
- Bound by `rotary_dim` (partial); `is_neox`; vector I/O; in-place; cos/sin fp32.
- `__launch_bounds__(block, 4)`; `hipcc --offload-arch=gfx942 -O3`.

## Numerics / parity
cos/sin fp32; section partition + per-axis positions correct; partial-rotary bound; deterministic parity.
See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- aiter HIP: JIT `module_fused_qk_norm_mrope_cache_quant_shuffle`.
- Custom: edit/extend `csrc/pos_encoding.cu`, rebuild.

## Pitfalls & anti-patterns
- ⚠ Hand-writing a generic mRoPE HIP kernel when aiter's fused one exists = wasted effort.
- ⚠ Partial-rotary OOB; wrong section.

## How to verify
`--save-temps` ISA; isolated vs fp64 per-section; partial-rotary test; VLM greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [[rope/backends/hip]] · [[languages/hip_cpp/patterns]] §2.

## Sources
- aiter HIP fused mrope: `/sgl-workspace/aiter/aiter/ops/fused_qk_norm_mrope_cache_quant.py`.
- RoPE HIP base to extend: https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu.
- wave64 / vector I/O: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html.
