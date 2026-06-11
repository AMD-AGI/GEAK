---
title: chunked_prefill on sglang_kernels — SOTA card
kind: sota_card
operator: chunked_prefill
backend: sglang_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py
  - https://docs.sglang.io/platforms/amd_gpu.html
---

# chunked_prefill × sglang_kernels

## TL;DR
This card is the **sglang dispatch** for chunked prefill: how sglang selects unified vs split attention
for mixed prefill+decode batches on MI300X. sglang's scheduler does chunked prefill at the policy layer
(`--chunked-prefill-size`) and routes the attention to the AITER unified kernel
(`SGLANG_USE_AITER_UNIFIED_ATTN=1`) or the per-regime backend (`--prefill-attention-backend` /
`--decode-attention-backend`). Use AITER unified for launch-bound small batch; bake off vs the split
per-regime backends.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| AITER unified attention (sglang `SGLANG_USE_AITER_UNIFIED_ATTN=1`) | `aiter/ops/triton/attention/unified_attention.py` via `triton_ops/aiter_unified_attention.py` | gfx942/950; bf16/fp16/fp8 | launch-bound small batch (one kernel) | mixed batches, small/medium concurrency |
| per-regime split (`--prefill-attention-backend aiter` + `--decode-attention-backend aiter`) | sglang attention registry | as above | two-kernel; large batch | large per-step batches |

## Config space / knobs
`--chunked-prefill-size N` (scheduler chunk → throughput vs TTFT); `SGLANG_USE_AITER_UNIFIED_ATTN=1`
(unified kernel); `--attention-backend {aiter,tilelang,triton}`; split via
`--prefill-attention-backend` / `--decode-attention-backend`; `SGLANG_USE_AITER=1` (master);
`HSA_NO_SCRATCH_RECLAIM=1` (near-mandatory MI300X). See [../tuning.md](../tuning.md) and
`backends/sglang_kernels/attention_backends.md`.

## Numerics / parity
fp32 online-softmax over segmented KV; chunk-invariance gate; reduction order differs across unified/split
and across aiter/triton/tilelang → re-check greedy temp=0 parity after a swap. See [../numerics.md](../numerics.md).

## Integration (rebind seam)
The dispatch surface is `attention_registry.py` (`@register_attention_backend`) and `aiter_backend.py`'s
`from aiter.ops.triton.attention.unified_attention import unified_attention`. Select via flags; confirm the
backend banner in the server log.

## Pitfalls & anti-patterns
- Unified kernel is causal only.
- `SGLANG_USE_AITER=1` with no aiter wheel → `ImportError` (image mismatch).
- AITER CK can crash under HIP-graph capture for novel shapes (#16025) → force Triton for that model.
- Chunk size too large → decode TPOT spikes; too small → scheduling overhead.

## How to verify
Confirm the backend banner; rocprofv3 Top-N (one unified kernel vs two split); chunk-invariance test;
greedy temp=0 e2e parity after a swap; throughput + TPOT + TTFT at the chosen chunk size.

## Alternatives / cross-links
[triton.md](triton.md) · [aiter.md](aiter.md) · [vllm_kernels.md](vllm_kernels.md) ·
`backends/sglang_kernels/attention_backends.md`, `overview.md`, `where_kernels_live.md` · [[../overview.md]].

## Sources
- sglang attention registry + unified-attn flag: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/attention_registry.py
- AITER unified kernel: on-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/attention/unified_attention.py`.
- sglang AMD docs (TileLang default, AITER): https://docs.sglang.io/platforms/amd_gpu.html
