---
title: lm_head_logits — numerics
kind: technique
operator: lm_head_logits
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both]
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# lm_head_logits — numerics

## TL;DR
Two rules dominate: **(1) accumulate in fp32 and emit fp32 logits** (the downstream softmax/sampling needs
the dynamic range; bf16/fp16 logits clip and bias the distribution), and **(2) the greedy argmax tie-break
and the vocab-parallel reduction order are the parity-sensitive seams** after a backend or TP swap. The
GEMM itself is a same-math bf16/fp16→fp32 op (parity-safe across hipBLASLt/asm/skinny solution swaps), so
the risk is in the **epilogue** (soft_cap `tanh`, scale) and the **distributed** path, not the matmul.

## dtype / accumulation
| stage | dtype | why |
|---|---|---|
| hidden in | bf16/fp16 | model activation dtype |
| weight `Wₑ` | bf16/fp16 (or quantized) | tied to [[embedding]] |
| **accumulate** | **fp32** | MFMA accumulates in fp32; required for V=128k–256k sums |
| **logits out** | **fp32** | sampling/softmax range; downcasting biases top-p/min-p thresholds and logprobs |

Quantized heads (fp8/int8 weight) need a **task-accuracy gate** (gsm8k / perplexity), not byte parity —
fp8 fnuz on gfx942 is off-dialect-by-2× if read as OCP (see [[quant_dequant_fp8]], [[vllm_kernels]]).

## The epilogue ordering (Gemma soft-cap)
`LogitsProcessor.forward` computes `soft_cap · tanh(logits / soft_cap)` **then** `*= scale`. Order matters
for parity: do the soft-cap on **fp32 logits before** any downcast, and keep the exact `tanh` (not a
polynomial approx) to match reference. A reordered or low-precision soft-cap shifts every logit slightly →
changes which tokens survive top-p/top-k → visible sampling divergence even though "the GEMM is correct."

## Greedy argmax tie-break (the classic divergence)
At temp=0 the head feeds an argmax (often fused: `get_top_tokens`). **Tie-break must be deterministic and
match the reference** — on exact ties, return the **lowest index** (the FlashInfer/aiter sampling kernels
use `atomicMin` on the index for exactly this reason; see [[sampling_topk_topp]] numerics). A vocab-parallel
local-argmax-then-gather must break cross-shard ties the same way the single-GPU argmax would, or greedy
decoding diverges at tie tokens.

## Vocab-parallel reduction / all-gather parity
- **all-gather path**: concatenates per-shard logits → bit-exact reconstruction (no FP reduction), then
  trims vocab padding to `org_vocab_size`. Parity-safe.
- **vocab-parallel argmax path** (`get_top_tokens`): correctness hinges on the cross-shard `(val,idx)`
  reduction using the **same tie-break** as a single-GPU argmax. This is where greedy TP parity breaks if
  done naively.
- Padding rows (vocab padded up to a multiple for the TP split) must be masked to `-inf` so they never win
  argmax/top-k.

## The one real gate
Greedy/temp=0 e2e parity after any head backend or TP-degree change: same prompts → same token stream. If
they diverge, suspect (in order) logits dtype downcast, soft-cap precision/order, then argmax tie-break.
For sampling backends, compare the **post-softmax distribution** (KL/entropy), not exact tokens.

## Cross-links
[overview.md](overview.md) · [tuning.md](tuning.md) · [fusion.md](fusion.md) ·
[[sampling_topk_topp]] (tie-break + fp32) · [[argmax_topk]] · [[dense_gemm]] (same-math GEMM parity) ·
[[allgather]].

## Sources
- soft_cap (`tanh`)/scale/bias epilogue + all-gather + `get_top_tokens`: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
- fp32 logits / argmax-invariant ordering in the sampler: https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py
- fp32 MFMA accumulate / fnuz fp8 dialect: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
