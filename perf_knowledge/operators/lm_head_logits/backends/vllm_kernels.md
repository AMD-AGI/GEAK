---
title: lm_head_logits on vllm_kernels — SOTA card
kind: sota_card
operator: lm_head_logits
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/rocm/skinny_gemms.cu
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
---

# lm_head_logits × vllm_kernels

## TL;DR
vLLM owns the **wiring** of the head (the `LogitsProcessor`: GEMM → soft_cap/scale/bias epilogue →
vocab-parallel gather/all-gather, plus the greedy **vocab-parallel argmax** that skips the all-gather), and
on ROCm it also ships its **own skinny GEMM kernels** (`csrc/rocm/skinny_gemms.cu`: `wvSplitK`, `LLMM1`,
`wvSplitKQ`) that can serve the small-M decode head when AITER isn't selected. So "vllm_kernels" here = the
LogitsProcessor orchestration + the editable HIP skinny GEMM fallback; the GEMM itself is usually AITER or
hipBLASLt.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `LogitsProcessor._get_logits` (`quant_method.apply` GEMM + `_gather_logits`) | `vllm-project/vllm@HEAD:vllm/model_executor/layers/logits_processor.py` | gfx942/950, bf16/fp16 | the wiring; GEMM delegated to AITER/hipBLASLt | always the orchestration layer |
| `get_top_tokens` vocab-parallel argmax (greedy) | same file | — | replaces `O(M·V)` all-gather with `O(M·2·tp)` (val,idx) gather | greedy decoding at TP>1 |
| vLLM skinny GEMM `wvSplitK`/`LLMM1`/`wvSplitKQ` | `vllm-project/vllm@HEAD:csrc/rocm/skinny_gemms.cu` | gfx942/950, bf16/fp16/fp8 | decode skinny GEMM; can beat a generic library kernel at small M | decode head when `VLLM_ROCM_USE_SKINNY_GEMM=1` and AITER not selected |

## Config space / knobs
- `VLLM_ROCM_USE_AITER=1` (+`_LINEAR=1`) → head GEMM goes to AITER (recommended). With AITER off,
  `VLLM_ROCM_USE_SKINNY_GEMM=1` (default) routes small-M to `wvSplitK*`.
- `--logprobs-mode` controls whether processed/raw logits are needed (affects whether the full `[M,V]` must
  be kept). Greedy + no-logprobs lets `get_top_tokens` skip the all-gather.
- Editing `csrc/rocm/skinny_gemms.cu` (Tier-C): autotune MFMA `matrix_instr_nonkdim`, `waves_per_eu`,
  split-K for the `(M,N=V,K=d)` shape, then rebuild vLLM. `torch_bindings.cpp` is the registration surface.

## Numerics / parity
fp32 accumulate, **fp32 logits**; soft_cap (`tanh`)/scale/bias in the processor (keep on fp32 before
downcast). all-gather is bit-exact reconstruction; the **vocab-parallel argmax tie-break must match
single-GPU (lowest index)** or greedy diverges (see [../numerics.md](../numerics.md)). fp8 KV/head → fnuz
gate on gfx942.

## Integration (rebind seam)
The head is built from `ParallelLMHead` (weight; its `forward` raises) consumed by `LogitsProcessor`. GEMM
backend chosen by the `VLLM_ROCM_USE_AITER*` hierarchy; skinny fallback by `VLLM_ROCM_USE_SKINNY_GEMM`.
Verify with rocprofv3: AITER/hipBLASLt `Cijk_*` vs vLLM `wvSplitK*`/`LLMM1` rows for the `N=V` GEMM.

## Pitfalls & anti-patterns
- Projecting all prefill tokens (not just last-token) → O(chunk·V) instead of O(batch·V) GEMM. Confirm the
  runner prunes hidden states.
- `wvSplitK*` is decode (small-M); it won't help prefill-shaped heads.
- Editing `csrc/rocm/*.cu` requires a vLLM rebuild (not Python-only).
- fp16 logits out → range clip; keep fp32.

## How to verify
rocprofv3 → confirm which GEMM kernel ran for the `N=V` shape; greedy/temp=0 e2e parity after toggling
AITER vs skinny; isolated skinny-GEMM bench at the served decode batch.

## Alternatives / cross-links
[aiter.md](aiter.md) (live GEMM) · [triton.md](triton.md) · [hip.md](hip.md) · [../overview.md](../overview.md) ·
[[skinny_gemv_decode]] · [[argmax_topk]] · [[vllm_kernels]].

## Sources
- LogitsProcessor (GEMM, gather/all-gather, soft_cap/scale/bias, `get_top_tokens`): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
- vLLM ROCm skinny GEMM kernels: https://github.com/vllm-project/vllm/blob/main/csrc/rocm/skinny_gemms.cu
- ROCm GEMM dispatch / env hierarchy: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
