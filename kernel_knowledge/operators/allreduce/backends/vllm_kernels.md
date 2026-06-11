---
title: allreduce on vllm_kernels (Quick Reduce) — SOTA card
kind: sota_card
operator: allreduce
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, int6, int4]
regimes: [decode, prefill, both]
status: sota
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
---

# allreduce × vllm_kernels (Quick Reduce + AITER custom AR)

## TL;DR
> vLLM ships **Quick Reduce** — a quantized custom all-reduce that quantizes the reduction (FP/INT8/INT6/
> INT4) to cut xGMI wire bytes for small/decode messages — and wires **AITER CustomAllreduce** into the
> communicator. Use Quick Reduce for latency-bound decode TP when the accuracy budget allows; it's a
> deliberate accuracy knob.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Quick Reduce (quantized custom AR) | vLLM `csrc/` + `VLLM_ROCM_QUICK_REDUCE_*` | gfx942/950; FP/INT8/INT6/INT4 | faster small-msg AR (quant gate) | decode TP, latency-bound |
| AITER CustomAllreduce (in cuda communicator) | vLLM × aiter integration | gfx942/950; bf16/fp16 | beats RCCL small msgs | non-quantized custom AR |
| RCCL (fallback) | `ROCm/rccl` | gfx942/950 | ~316–330 GB/s busbw | large msgs / Quick Reduce off |

## Config space / knobs
- `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION={NONE,FP,INT8,INT6,INT4}` (the accuracy/perf dial).
- `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1` (reduce in fp16), `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB`
  (size cap above which it falls to RCCL).
- `VLLM_ROCM_USE_AITER=1` to enable the AITER custom AR path.
- Multi-GPU baseline env: `NCCL_MIN_NCHANNELS=112`.

## Numerics / parity
Any non-`NONE` Quick Reduce **changes reduced values** → accuracy gate (FP/INT8 usually safe, INT6/INT4
aggressive — re-run gsm8k). `CAST_BF16_TO_FP16` changes compute dtype. fnuz fp8 on gfx942. See
[numerics.md](../numerics.md).

## Integration (rebind seam)
Quick Reduce / AITER AR selected in the vLLM distributed communicator on ROCm; gated by the env above.
rocprof to confirm the custom AR kernel ran (not RCCL).

## Pitfalls & anti-patterns
- INT6/INT4 Quick Reduce shipped without an eval = accuracy regression.
- AITER AR segfaults (#1542) — fall back to RCCL.
- `MAX_SIZE_BYTES_MB` too high → quantizing large msgs (bad bandwidth + accuracy); cap it.

## How to verify
e2e greedy + gsm8k with Quick Reduce on/off; rocprof confirms the custom kernel; numeric vs fp32 RCCL.

## Alternatives / cross-links
[mori_rccl.md](rccl.md) · [hip.md](hip.md) ·
[`backends/vllm_kernels/overview.md`](../../../backends/vllm_kernels/overview.md) ·
[`backends/mori_rccl/rccl_tuning.md`](../../../backends/mori_rccl/rccl_tuning.md) · [overview.md](../overview.md).

## Sources
- Quick Reduce env / quantization: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- vLLM ROCm envs: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
- AITER AR segfault: https://github.com/ROCm/aiter/issues/1542
