---
title: allgather — fusion
kind: technique
operator: allgather
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
  - https://docs.vllm.ai/en/latest/design/fusions/
---

# allgather — fusion

## Fusions that exist on AMD today
| fusion | what it merges | where | payoff |
|---|---|---|---|
| **reduce-scatter + RMSNorm + (fp8 quant) + all-gather** | the full SP collective in one pipeline | aiter Triton `reduce_scatter_rmsnorm_quant_all_gather.py` | one fused kernel for the SP rewrite of all-reduce |
| **GEMM → all-gather (AsyncTP)** | producer GEMM overlapped/fused with the AG of its output | vLLM AsyncTP pass | hides AG under compute |
| **all-gather → GEMM** | AG of a sharded input overlapped with the consuming GEMM | AsyncTP | hides AG under compute |

## The aiter fused SP kernel (the headline)
`aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py` is a single **fused pipeline
kernel** (`fused_pipeline_kernel`, with `_quantize_fp8_stage`) that does **reduce-scatter → RMSNorm →
optional fp8 quant → all-gather** — i.e. the entire sequence-parallel replacement for an all-reduce+norm,
fused. This is the AMD realization of the SP rewrite (replace AR with RS+norm+AG) and ties [[allgather]],
[[reduce_scatter]], and [[fused_allreduce_rmsnorm]] together in one kernel. Built on Iris GPU-initiated
comm.

## AsyncTP overlap
When not fully fused, the AsyncTP compile pass overlaps the AG with the adjacent GEMM so comm hides under
compute (the producer GEMM emits shards that AG streams while the next GEMM starts). Offload the AG copy to
SDMA (`HSA_ENABLE_SDMA=1`) so it doesn't steal CU cycles.

## Cross-links
[[reduce_scatter]] · [[fused_allreduce_rmsnorm]] · [[allreduce]] (the SP rewrite source) ·
[`backends/mori_rccl/rccl_tuning.md`](../../backends/mori_rccl/rccl_tuning.md).

## Sources
- aiter fused RS+RMSNorm+quant+AG: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`.
- AsyncTP / SP fusion: https://docs.vllm.ai/en/latest/design/fusions/
