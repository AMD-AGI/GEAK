---
title: lm_head_logits on aiter — SOTA card
kind: sota_card
operator: lm_head_logits
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# lm_head_logits × aiter

## TL;DR
On sglang/vLLM the LM-head projection is **just another linear**, so it is **dispatched by aiter's tuned
GEMM** (`tuned_gemm.gemm_a16w16` → hipBLASLt / asm / skinny / triton per shape). aiter has **no dedicated
"lm_head" kernel** — the head is a row in the same per-shape DB as every body GEMM, distinguished only by
its **large-N (`N=V`=128k–256k), skinny-M** shape. To tune the head you capture and tune that shape exactly
like [[dense_gemm]] × aiter. The aiter-specific value: its DB will pick a **skinny/split-K** kernel for the
small-M decode head so the giant `[V,d]` weight read fills 304 CUs.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter `tuned_gemm` for `(M=batch, N=V, K=d)` | `ROCm/aiter@a6bb49937:aiter/tuned_gemm.py` | gfx942/950, bf16/fp16 (fp8 head via `gemm_a8w8`) | same-mechanism as dense GEMM (+2.23% e2e blended tune included head shapes implicitly); head GEMM is bandwidth-bound — wins come from skinny/split-K libtype selection, not MFMA | the live serving head GEMM on sglang/vLLM |
| `gemm_a8w8` (fp8/int8 quantized head) | aiter `gemm_a8w8` + dequant epilogue | gfx942/950, fp8_e4m3_fnuz | needs task-accuracy gate (not parity) | memory-constrained large-vocab head |

## Config space / knobs
- **Capture**: `AITER_TUNE_GEMM=1` on a warm server → the head's true `(M=batch, N=V, K=d, bias, dtype)`
  appended to `bf16_untuned_gemm.csv` (capture live so `M`=batch, not chunk).
- **Tune**: `gradlib/gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>` races libtypes; for the head's
  small M it should favor `skinny`/split-K, but **N=V≫cu_num often steers it to hipBLASLt/asm** — let the
  tuner decide, then read it back.
- **Deploy**: `AITER_CONFIG_GEMM_BF16=<tuned.csv>` `AITER_LOG_TUNED_CONFIG=1`.
- The vocab-parallel layout (N split across TP) means each rank's head GEMM is `(M, V/tp, d)` — capture on
  the deployed TP degree.

## Numerics / parity
Same-math bf16/fp16→**fp32** GEMM (parity-safe libtype swap, gradlib `err_ratio<0.05`). **Emit fp32
logits.** fp8 quantized head → task-accuracy gate (gsm8k/ppl), fnuz dialect on gfx942. soft_cap/scale/bias
applied by LogitsProcessor *after* the aiter GEMM (not inside aiter). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
Live call: the head's `nn.Linear`/`quant_method.apply` resolves through aiter's `tuned_gemm` exactly like
body linears (`gemm_a16w16` / `tgemm.mm`). Lookup key = `(cu_num, padded_M, N=V, K=d, bias, dtype, otype,
scaleAB, bpreshuffle)`. Verify: `grep -c 'is tuned on cu_num' server.log` includes a hit for the
`N=V` shape; `AITER_LOG_MORE=1` to see which libtype the head landed on.

## Pitfalls & anti-patterns
- Capturing with `M`=chunk (all prefill tokens) → head shape never matches live `M`=batch → DB miss.
- Assuming the `skinny` libtype engages for the head — with `N=V` it frequently picks hipBLASLt/asm;
  **check**, don't assume (`AITER_LOG_MORE=1`).
- fp16 logits out (range clip) — keep fp32 ([../numerics.md](../numerics.md)).
- Tuning hipBLASLt's override file does nothing (aiter bypasses PyTorch BLAS) — same trap as [[dense_gemm]].

## How to verify
Isolated GEMM bench at `(M∈{1,16,64,256}, N=V, K=d, bias=<live>)`; e2e **decode-latency** A/B (head is a
tail-latency item at low batch), gate on delta + non-overlap + engagement (`is tuned on cu_num` hit for the
`N=V` row).

## Alternatives / cross-links
[vllm_kernels.md](vllm_kernels.md) (wiring + skinny GEMM) · [triton.md](triton.md) · [hip.md](hip.md) ·
[../overview.md](../overview.md) · [[dense_gemm]] × aiter · [[skinny_gemv_decode]] · [[aiter]].

## Sources
- aiter tuned_gemm dispatch + libtypes (hipblaslt/asm/skinny/triton): `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py` (on-box `/sgl-workspace/aiter`).
- Head is a linear routed through the same path: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
- Skinny/split-K decode GEMM tuning: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
