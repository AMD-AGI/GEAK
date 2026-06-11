---
title: kv_cache_quant on triton — SOTA card
kind: sota_card
operator: kv_cache_quant
backend: triton
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp8_e4m3]
regimes: [decode, prefill, both]
status: competitive
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton
  - vllm-project/vllm@HEAD:vllm/v1/attention/backends/triton_attn.py
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# kv_cache_quant × triton

## TL;DR
Triton's KV-quant role is inside the **Triton attention backend** (`TRITON_ATTN` / Triton MLA): the FP8 KV
store/load + scaled convert is done in the Triton paged-attention kernel rather than as a standalone op.
It is the portable, correct fallback when AITER/vLLM-HIP KV paths don't cover a shape; on the hot decode
path the fused aiter/HIP kernels usually win. The standalone KV cast is memory-bound, so Triton loses
nothing there — the question is whether the read-side attention is also Triton.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Triton unified/paged attention FP8 KV read+write | `vllm:.../triton_attn.py`; aiter Triton attn | gfx942/950, e4m3 | fp32 online softmax, FP8 KV scaled | TRITON_ATTN fallback |
| `dynamic_per_token_quant_fp8_i8` (for a standalone KV cast) | `aiter/ops/triton/quant/quant.py` | gfx942/950 | row reduce | rare standalone KV quant |

## Config space / knobs
- KV dtype + `k_scale`/`v_scale` passed into the Triton attention kernel.
- `BLOCK_SIZE` (paged block) must match the cache; `num_warps`/`num_stages` for the attention loop.
- fnuz fp8 dtype on gfx942.

## Numerics / parity
fp32 online softmax; FP8 KV scaled read; fnuz gfx942 / ocp gfx950. Triton reduction order differs from
HIP/AITER → re-check greedy/temp=0 parity after a backend swap. Gate on gsm8k → [[numerics.md]].

## Integration (rebind seam)
`--attention-backend TRITON_ATTN` (+ `--kv-cache-dtype fp8`). Python kernel → autotune overlay; no
site-packages edit.

## Pitfalls & anti-patterns
- OCP fp8 dtype on gfx942 → compile error; use fnuz.
- Reduction-order parity drift vs HIP/AITER.
- Using Triton KV path when the fused aiter chain (norm+RoPE+write+quant) would be faster on the hot path.

## How to verify
`AMDGCN_ENABLE_DUMP=1`; rocprof confirm the Triton attention kernel; gsm8k parity vs ROCM_AITER_FA.

## Alternatives / cross-links
[aiter.md](aiter.md) · [vllm_kernels.md](vllm_kernels.md) · [hip.md](hip.md) · [[languages/triton_amd]] ·
[[operators/attention_decode_paged]] · [overview.md](../overview.md).

## Sources
- Triton attention backend (FP8 KV): `vllm-project/vllm@HEAD:vllm/v1/attention/backends/triton_attn.py`; aiter Triton attn `ROCm/aiter@a6bb49937:aiter/ops/triton`.
- Triton AMD knobs / fnuz fp8: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
