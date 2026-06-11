---
title: fused_allreduce_rmsnorm — overview
kind: operator_overview
operator: fused_allreduce_rmsnorm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://docs.vllm.ai/en/latest/design/fusions/
  - https://github.com/vllm-project/vllm/issues/26768
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
---

# fused_allreduce_rmsnorm  (comm + norm overlap/fusion)

## TL;DR
Every transformer TP layer does `all_reduce` immediately followed by `RMSNorm` (often `+residual add`, and
on fp8 models `+static fp8 quant`). Fusing them removes a full read/write of the activation between the
collective and the norm and lets comm overlap compute. On AMD there are **two realizations**: (1) a
**torch.compile fusion pass** that matches the AR→(add)→RMSNorm(→fp8 quant) chain (vLLM
`allreduce_rms_fusion.py` patterns), and (2) the **sequence-parallel rewrite** baked into one kernel —
aiter's `reduce_scatter_rmsnorm_quant_all_gather` (RS + norm + quant + AG fused). The most important fact:
the norm is **memory-bound**, so fusing it onto the collective's epilogue (or running it on the **sharded**
tensor under SP) is a near-free latency cut; AMD reports the related qknorm+AR fusion at **~10–15% prefill
TPS on TP=2/4**.

## Math contract
`y = RMSNorm( AllReduce(Σ_r x_r) [+ residual] ) [→ fp8 quant]`.
- RMSNorm: `y = x / rms(x) · γ`, `rms(x)=sqrt(mean(x²)+ε)`, fp32 reduction over the hidden dim.
- AR: fp32 accumulate across ranks. Fused variants: `AllReduceRMSNorm`, `AllReduceFusedAddRMSNorm`,
  `AllReduceFusedRMSNormStaticQuantFP8`, `AllReduceFusedAddRMSNormStaticQuantFP8`.
- SP form: `AllReduce → ReduceScatter + local RMSNorm + AllGather` (norm on the 1/P shard).

## Shape regimes
- **prefill (large M)**: large AR message + norm over `[M, H]` — fusion saves the intermediate pass; SP
  saves norm work.
- **decode (M=batch)**: small AR + norm — latency-bound; fusing onto a custom-AR epilogue + graph capture.

## Where it matters (Amdahl)
The norm is small but the **fused read/write saving** + **comm overlap** is a steady per-layer cut. AMD's
qknorm+AR fusion (a sibling) gave **~10–15% prefill TPS at TP=2/4** — the AR+RMSNorm fusion is in the same
family. Largest on TP-heavy layers and sub-island TP where AR is already a visible cost.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (fused Triton SP kernel; qknorm+AR; fused add+rmsnorm+quant ops) | [backends/aiter.md](backends/aiter.md) |
| hip | 🟢 (custom-AR epilogue norm; the fused SP kernel internals) | [backends/hip.md](backends/hip.md) |
| mori_rccl | 🟡 (RCCL AR + separate norm; SP rewrite uses RCCL RS/AG; MoRI-CCL) | [backends/rccl.md](backends/rccl.md) |

## Fusion neighbors
- builds on [[allreduce]] / [[reduce_scatter]] / [[allgather]]; the norm is [[rmsnorm]] / [[fused_add_rmsnorm]];
  the fp8 epilogue is [[fused_norm_quant]] / [[scaled_quant_gemm]]. See [fusion.md](fusion.md).

## Numerics
RMSNorm fp32 reduction; AR fp32 accumulate → parity-safe. The **fp8 static quant** epilogue is an accuracy
gate (fnuz on gfx942). See [numerics.md](numerics.md).

## How to bench
Isolated: fused AR+RMSNorm vs separate AR then RMSNorm at (M, H, TP). e2e: TP serving tok/s with the fusion
pass on/off (`enable_fi_allreduce_fusion`); trace that the fused kernel replaced the AR+norm pair.

## Sources
- vLLM AR+RMSNorm fusion passes / patterns: https://docs.vllm.ai/en/latest/design/fusions/ ; https://github.com/vllm-project/vllm/issues/26768
- aiter fused SP kernel (RS+norm+quant+AG): `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`.
- qknorm+AR fusion ~10–15% TPS: https://github.com/ROCm/aiter/releases.
