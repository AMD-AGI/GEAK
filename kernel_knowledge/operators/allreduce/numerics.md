---
title: allreduce — numerics
kind: technique
operator: allreduce
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, int6, int4]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
  - https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
---

# allreduce — numerics & parity

## RCCL (default) — parity-safe
RCCL all-reduce accumulates in fp32 with deterministic-enough ordering for inference; switching ring↔tree
or channel counts changes the reduction **order** → benign bf16 rounding deltas, not regressions. Gate with
greedy/temp=0 parity over ≥10 prompts; don't require byte parity across algo changes.

## Quick Reduce / quantized custom AR — an accuracy gate
vLLM Quick Reduce **quantizes the reduction** to int8/int6/int4/fp8 to cut wire bytes
(`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`). Any non-`NONE` setting **changes the reduced values** — this is a
real accuracy knob:
- `FP`/`INT8` are usually safe for inference; `INT6`/`INT4` are aggressive — re-run a small eval (gsm8k).
- `_CAST_BF16_TO_FP16=1` changes the compute dtype of the reduction — verify no overflow on large
  activations.
- Never ship a quantized AR without an eval.

## fnuz fp8 on gfx942
If the AR quantizes to fp8, gfx942 is **fnuz** (exponent bias off-by-one vs OCP) — a wrong-dialect
interpretation is off by exactly 2×. Match the dialect to the arch.

## AITER custom AR stability
AITER custom AR has had **segfaults** on MI300X (aiter #1542) — a stability gate, not just accuracy. Fall
back to `SGLANG_USE_AITER_AR=0` (RCCL) if it crashes.

## Verification recipe
1. Numeric: custom/quantized AR output vs an fp32 RCCL reference for a random tensor — within the quant
   tolerance for Quick Reduce, near-exact for AITER fp16/bf16 AR.
2. e2e: greedy parity + a small eval whenever a non-`NONE` Quick Reduce or any quantized AR is enabled.
3. Stability: soak the custom-AR path before trusting it (the #1542 class of bug).

## Sources
- Quick Reduce quantization levels: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- RCCL algo/order: https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- AITER AR segfault: https://github.com/ROCm/aiter/issues/1542
