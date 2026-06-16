---
title: lm_head_logits — fusion
kind: technique
operator: lm_head_logits
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# lm_head_logits — fusion

## TL;DR
Two fusions pay off at V=128k–256k: **(1) GEMM + epilogue** (scale / soft_cap-`tanh` / bias) to avoid a
separate `[M,V]` bandwidth pass, and **(2) GEMM + argmax** for greedy decoding to skip materializing full
`[M,V]` logits *and* the vocab-parallel all-gather. The largest "fusion" is upstream and structural:
**only the last token per sequence is projected** (hidden pruned before the head) — not a kernel fusion but
the single biggest work reduction.

## Fusion neighbors
| neighbor | type | payoff | note |
|---|---|---|---|
| `scale` / `soft_cap` / `bias` | GEMM epilogue (elementwise over `[M,V]`) | medium–high at large V | one fused pass vs read+write `[M,V]` again → [[gemm_epilogue_fused]] |
| **argmax** (greedy) | GEMM + reduction | **high** | `get_top_tokens`: local argmax per V-shard, gather `(val,idx)` only → skips `O(M·V)` all-gather → [[argmax_topk]], [[allgather]] |
| temperature + softmax + sample | downstream | n/a (separate) | feeds [[sampling_topk_topp]]; sampling needs full logits so it can't fuse into the head |
| last-token prune | upstream gather | **highest** | model runner prunes `hidden` → `M`=batch; structural, see [tuning.md](tuning.md) |
| tied weight with [[embedding]] | memory share | — | one `[V,d]` tensor, two access patterns |

## Epilogue fusion (scale / soft_cap / bias)
At V=256k, the soft-cap (`soft_cap·tanh(logits/soft_cap)`) + scale + bias is a full `[M,V]` elementwise. Run
unfused it costs a read **and** write of the logits matrix; fused into the GEMM epilogue (or a single
trailing kernel that consumes the GEMM output in registers/LDS) it costs one pass. For Gemma-2/3 (256k
vocab + soft-cap) this is the meaningful logits-side fusion. Keep soft-cap on **fp32 logits before any
downcast** (parity — [numerics.md](numerics.md)).

## Argmax fusion (greedy fast path)
For temp=0, you never need the full `[M,V]` logits — only the argmax. `get_top_tokens` does a
**vocab-parallel argmax** (local argmax per shard, gather only `(value,index)`), avoiding both the full
logits materialization across ranks and the `O(M·V)` all-gather. This is the highest-leverage fusion for
greedy serving at large TP. Tie-break must match single-GPU argmax (lowest index) — see [numerics.md].

## What can't fuse into the head
- **Sampling** (top-k/top-p/min-p): needs the full logits row → it is a *consumer* of the head, fused
  internally on its own side (see [[sampling_topk_topp]]), not into the GEMM.
- **all-gather for sampling**: when sampling (not greedy) you must materialize full `[M,V]` → the
  all-gather is unavoidable; the argmax fusion only helps greedy.

## Cross-links
[overview.md](overview.md) · [tuning.md](tuning.md) · [numerics.md](numerics.md) ·
[[gemm_epilogue_fused]] · [[argmax_topk]] · [[sampling_topk_topp]] · [[allgather]] · [[embedding]].

## Sources
- soft_cap/scale/bias epilogue + `get_top_tokens` (vocab-parallel argmax, skips all-gather): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
- Epilogue/bandwidth fusion guidance at large N: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
