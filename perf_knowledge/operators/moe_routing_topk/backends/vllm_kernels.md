---
title: moe_routing_topk on vllm_kernels — SOTA card
kind: sota_card
operator: moe_routing_topk
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/pull/17955
  - https://github.com/vllm-project/vllm/pull/16752
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
---

# moe_routing_topk × vllm_kernels

## TL;DR
> vLLM doesn't author its own router HIP kernel — it **wires aiter's** (`rocm_aiter_topk_softmax`,
> `rocm_aiter_biased_grouped_topk`) as torch custom ops and otherwise uses its **own
> Triton/CUDA `topk_softmax`/`grouped_topk`** reference. The card here is the *dispatch + registration*
> view: how the router engages on ROCm vLLM and the env hierarchy that gates it.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `rocm_aiter_biased_grouped_topk` (registered aiter op) | vLLM PR #17955 | gfx942/950, DeepSeek-V3/R1 | inherits aiter DPP 1.66× | grouped sigmoid routing under `VLLM_ROCM_USE_AITER=1` |
| `rocm_aiter_topk_softmax` (registered aiter op) | vLLM PR #16752 (Fused-MoE V1) | gfx942/950 | — | non-grouped softmax under AITER |
| vLLM `grouped_topk` / `fused_grouped_topk` (Triton/torch) | `fused_moe.py` | gfx942/950 | reference | `VLLM_ROCM_USE_AITER=0` fallback / softmax models |

Recommend: `VLLM_ROCM_USE_AITER=1` (+ `_MOE=1`) to engage the aiter DPP router; fall to the Triton
reference if the aiter path mis-scores.

## Config space / knobs
- `VLLM_ROCM_USE_AITER=1` (master, default 0) + `VLLM_ROCM_USE_AITER_MOE=1` (default 1 once master on).
- Routing dispatch: `is_rocm_aiter_moe_enabled()` → import `torch.ops.vllm.rocm_aiter_biased_grouped_topk`
  as `grouped_topk`; else vLLM's `grouped_topk`.
- `AITER_ONLINE_TUNE=1` for missing-shape retries on the downstream grouped GEMM (not the router itself).

## Numerics / parity
The aiter biased path is **sigmoid-only** — vLLM review noted `scoring_func` is "assumed"; for
softmax-scored MoE the aiter biased path is wrong (aiter #2153). vLLM's own `fused_grouped_topk` honors
`scoring_func` and is the safe fallback. Run `tests/kernels/moe/test_routing.py::test_grouped_topk` with
`VLLM_ROCM_USE_AITER=1`. See [numerics.md](../numerics.md).

## Integration (rebind seam)
Registered via `direct_register_custom_op` (fake/meta impl) → survives torch.compile (Inductor fuses
around it, doesn't decompose). Selected in the FusedMoE layer's `select_experts` path.

## Pitfalls & anti-patterns
- aiter biased sigmoid hardcode → softmax model regression; assert or fall back.
- Image mismatch: `VLLM_ROCM_USE_AITER=1` with no aiter wheel → ImportError.
- `num_expert_group` outside {8,16,32} (wave64) → incomplete reduction in the aiter kernel.

## How to verify
rocprof: confirm the aiter routing op ran (not the Triton reference); `test_grouped_topk` parity; e2e
greedy parity `AITER=1` vs `=0`.

## Alternatives / cross-links
[aiter.md](aiter.md) · [triton.md](triton.md) · [hip.md](hip.md) ·
[`backends/vllm_kernels/aiter_integration.md`](../../../backends/vllm_kernels/aiter_integration.md) ·
[overview.md](../overview.md).

## Sources
- biased group topk for DeepSeek-V3: https://github.com/vllm-project/vllm/pull/17955
- AITER Fused-MoE V1 (`rocm_aiter_topk_softmax` registered): https://github.com/vllm-project/vllm/pull/16752
- reference routing: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
- sigmoid/reduction bug: https://github.com/ROCm/aiter/issues/2153
