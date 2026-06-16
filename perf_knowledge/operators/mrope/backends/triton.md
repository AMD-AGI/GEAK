---
title: mrope on triton — SOTA card
kind: sota_card
operator: mrope
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/pull/22593
  - https://github.com/vllm-project/vllm/pull/16457
  - /sgl-workspace/aiter/aiter/ops/triton/rope/rope.py
---

# mrope × triton

## TL;DR
Triton is the authorable SOTA and the path vLLM uses for MRoPE — the MRoPE Triton kernel (with the
partial-rotary fix, #22593) handles Qwen2-VL/2.5-VL and GLM-4.1V. Memory-bound; the section split is the
only delta from a plain RoPE Triton kernel.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| vLLM MRoPE Triton kernel | vLLM #22593 (partial-rotary), #16457 (PyTorch→Triton) | gfx942/950, bf16/fp16 | replaced 40–60%-latency PyTorch MRoPE | vLLM VLM RoPE path |
| aiter Triton rope (+ section split for mrope) | `aiter/ops/triton/rope/rope.py` | gfx942/950 | `grid=(b,h,cdiv(s,32))`, num_warps=4 | authorable mrope |

## Config space / knobs
- `mrope_section` (constexpr partition); per-axis cos/sin selection per section.
- `is_neox`, `rotary_dim` (partial — bound the loop), positions `[3, ...]`.
- `BLOCK_S=32`, `num_warps=4`, `BLOCK_D` per RoPE; in-place; cos/sin fp32.

## Numerics / parity
cos/sin fp32; `mrope_section` correct; **partial-rotary bound by rotary_dim** (#22593 OOB if not); per-axis
positions; deterministic parity. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- vLLM: the MRoPE layer dispatches to the Triton kernel; `VLLM_ROCM_USE_AITER_TRITON_ROPE` for the AITER
  Triton variant.
- Direct: extend `aiter.ops.triton.rope` with the section split.

## Pitfalls & anti-patterns
- ⚠ `rotary_dim == head_size` assumption → OOB on partial-rotary VLMs (#22593) — the canonical MRoPE bug.
- ⚠ Wrong section partition or per-axis position.

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; isolated vs fp64 per-section; partial-rotary test; VLM greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [[rope/backends/triton]] · [[languages/triton_amd/patterns]].

## Sources
- MRoPE Triton + partial-rotary fix: https://github.com/vllm-project/vllm/pull/22593.
- PyTorch→Triton RoPE: https://github.com/vllm-project/vllm/pull/16457.
- aiter Triton rope base: `/sgl-workspace/aiter/aiter/ops/triton/rope/rope.py`.
