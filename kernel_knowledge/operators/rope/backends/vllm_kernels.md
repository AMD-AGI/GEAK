---
title: rope on vllm_kernels — SOTA card
kind: sota_card
operator: rope
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu
  - https://github.com/vllm-project/vllm/pull/16457
  - https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/rotary_embedding.html
---

# rope × vllm_kernels

## TL;DR
vLLM has three RoPE paths: native HIP `rotary_embedding` (`csrc/pos_encoding.cu`), a **Triton** rope from
flash_attn (#16457, the fix for the 40–60%-latency PyTorch RoPE), and **AITER** rope
(`VLLM_ROCM_USE_AITER_TRITON_ROPE`). The `RotaryEmbedding` layer (`model_executor/layers/rotary_embedding`)
dispatches `forward_hip`/`forward_cuda`/`forward_native`, with YaRN/NTK/Linear/DualChunk scaling subclasses.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| AITER rope (incl. fused QK-norm+RoPE) | via `VLLM_ROCM_USE_AITER` | gfx942/950 | Qwen3 fused win | MI300X serving — [aiter.md](aiter.md) |
| native HIP `rotary_embedding` | `csrc/pos_encoding.cu` | gfx942/950, bf16/fp16 | bandwidth-bound, in-place | `USE_AITER=0` / Tier-C |
| Triton rope (flash_attn) | vLLM #16457 | gfx942/950 | replaced 40–60% PyTorch RoPE | `VLLM_ROCM_USE_AITER_TRITON_ROPE` |

## Config space / knobs
- `VLLM_ROCM_USE_AITER=1` (+ `VLLM_ROCM_USE_AITER_TRITON_ROPE` for the Triton path).
- `RotaryEmbedding(is_neox, rotary_dim, ...)`; scaling subclasses build their cos/sin cache (multiple
  caches for multi-LoRA).
- Native HIP: vector width, in-place.

## Numerics / parity
cos/sin fp32; `is_neox` matches config; **partial rotation** (#22593 MRoPE fix, #39625 shape mismatch);
scaling cache correct; deterministic → token-identical parity. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- Python: `model_executor/layers/rotary_embedding.py` (`forward_hip`).
- Native HIP: `csrc/pos_encoding.cu` + `torch_bindings.cpp` (`_C::rotary_embedding`); rebuild to edit.
- torch.compile: RoPE custom-op preserved; the op is on the migration list to PluggableLayer/vLLM-IR
  (#32676) — expect the surface to change.

## Pitfalls & anti-patterns
- ⚠ Partial-rotary OOB (#22593, #39625) — bound by `rotary_dim`.
- ⚠ Wrong scaling cache / `is_neox`.
- ⚠ Image mismatch with `USE_AITER=1`.

## How to verify
rocprofv3 RoPE kernel (AITER vs Triton vs native); isolated vs fp64; partial-rotary test; greedy parity.

## Alternatives / cross-links
[aiter.md](aiter.md) · [hip.md](hip.md) · [triton.md](triton.md) · [[mrope]] ·
[[backends/vllm_kernels/aiter_integration]].

## Sources
- vLLM HIP rotary_embedding: https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding.cu.
- PyTorch→Triton RoPE: https://github.com/vllm-project/vllm/pull/16457.
- RotaryEmbedding layer / scaling variants: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/rotary_embedding.html.
