---
title: lm_head_logits — overview
kind: operator_overview
operator: lm_head_logits
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
---

# lm_head_logits  (`logits = hidden · Wₑᵀ [+ bias]`, large-N skinny GEMM)

## TL;DR
The LM head is the **final projection GEMM** from hidden `d` to the full vocab `V` (128k–256k): a
**large-N, skinny-M** GEMM (`M`=tokens-to-sample, often the batch only). It is a *real* GEMM — so its
backend story **is** the GEMM story: at decode it is a [[skinny_gemv_decode]] (`M`≤256, `N`=V huge), at
prefill (only the last token per sequence is projected) it stays skinny. On sglang/vLLM it is **dispatched
by aiter's tuned GEMM** like every other linear (so [[dense_gemm]] × [[aiter]] applies), but with two
twists: (1) huge `N`=V makes it a **bandwidth-bound** weight read of the `[V,d]` matrix (~4 GB at
V=256k/d=8k/bf16), and (2) it is **vocab-parallel** (split along N), so the logits need an all-gather — or,
for greedy, a **vocab-parallel argmax** that skips the all-gather entirely.

## Math contract
- `logits[M, V] = hidden[M, d] · Wₑ[V, d]ᵀ (+ bias[V])`, `Wₑ` = lm_head weight (often tied to
  [[embedding]]). dtype: hidden bf16/fp16, weight bf16/fp16, **fp32 accumulate**, logits output **fp32**
  (sampling math wants fp32 — see [numerics.md](numerics.md)).
- **Epilogue extras** (in `LogitsProcessor.forward`): optional `scale` (`logits *= scale`), **soft-cap**
  (`soft_cap · tanh(logits / soft_cap)`, Gemma-2/3), and `bias`. These ride the GEMM epilogue or a cheap
  follow-up elementwise.
- **Only the sampled positions are projected**: the model runner prunes `hidden` to the last token of each
  sequence *before* the LM head, so `M` = number of sequences sampling this step, not total tokens. This is
  the single biggest cost control.
- **Vocab-parallel** (TP>1, split along V): each rank computes `logits` for its V-shard, then
  `tensor_model_parallel_all_gather` (or `gather`) reconstructs full `[M,V]` and trims vocab padding to
  `org_vocab_size`. Greedy can instead do a **local argmax + gather of (val,idx)** (`get_top_tokens`),
  cutting comms from `O(M·V)` to `O(M·2·tp)`.

## Shape regimes
- **decode** (dominant for the head): `M` = running batch (1..256), `N=V` (128k–256k), `K=d` (4k–8k). This
  is a **skinny GEMV/GEMM** — split-K / skinny kernels matter, and the **`[V,d]` weight read dominates**
  (compute-light, bandwidth-bound). → [[skinny_gemv_decode]].
- **prefill**: still skinny because only the **last token** per sequence is projected (`M`=batch, not
  chunk-length). Unlike the body GEMMs (large-M), the head does **not** grow with sequence length.
- Large vocab is the defining feature: `N`=V is 5–20× the hidden GEMM `N`, so the head is a
  disproportionately large weight read relative to its FLOPs.

## Where it matters (Amdahl)
The head is **one GEMM per step** but with the largest single `N` in the model. At **decode** with small
batch it can be a visible slice of step latency (the `[V,d]` read is ~4 GB at V=256k) — a tail-latency
item, not a throughput head. At prefill it is negligible vs the body [[dense_gemm]] mass (~80%). The wins:
(1) keep `M` minimal (project only sampled positions), (2) use the skinny/split-K path so the weight read
is bandwidth-efficient, (3) for greedy avoid the full-logits all-gather via vocab-parallel argmax.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (live path — dispatches the head GEMM, skinny/hipBLASLt/asm per shape) | [backends/aiter.md](backends/aiter.md) |
| vllm_kernels | 🟢 sota (LogitsProcessor wiring + vLLM skinny GEMM `wvSplitK*` for decode) | [backends/vllm_kernels.md](backends/vllm_kernels.md) |
| triton | 🟡 competitive (skinny split-K GEMM; loses to tuned hipBLASLt on plain, wins by fusing epilogue) | [backends/triton.md](backends/triton.md) |
| hip | 🟡 competitive (vLLM `csrc/rocm/skinny_gemms.cu` is the editable decode kernel) | [backends/hip.md](backends/hip.md) |

## Fusion neighbors
- **GEMM + epilogue**: `scale`, `soft_cap` (`tanh`), `bias` fuse onto the GEMM or a trailing elementwise →
  [fusion.md](fusion.md), [[gemm_epilogue_fused]].
- **GEMM + argmax** (greedy): fuse the projection with a vocab-parallel argmax to skip materializing full
  `[M,V]` logits and the all-gather (`get_top_tokens`) → [[argmax_topk]].
- Feeds directly into [[sampling_topk_topp]] (logits → temperature → softmax → sample).
- Weight is tied to [[embedding]] (`tie_word_embeddings`).

## Numerics
fp32-accumulate, **fp32 logits out** (sampling/softmax wants the full range); soft-cap `tanh` ordering and
greedy argmax tie-break are the parity-sensitive parts → [numerics.md](numerics.md).

## How to bench
Isolated: GEMM bench at `(M=batch, N=V, K=d, bias, dtype)` for your model (e.g. `M∈{1,16,64,256}`,
`N=128256`, `K=8192`) + the soft-cap/scale epilogue; oracle = `hidden @ W.T` fp32 ref. e2e: this is a
**tail-latency** item — measure decode step latency at low batch, not just throughput.

## Sources
- LogitsProcessor GEMM + all-gather + soft_cap/scale/bias + vocab-parallel argmax (`get_top_tokens`):
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
- `ParallelLMHead` weight (tied; `forward` raises → used in sampler):
  https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/vocab_parallel_embedding.py
- Head GEMM dispatched by aiter tuned_gemm (live path): `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py`.
- Skinny/decode GEMM tuning (split-K, ≥1024 grid): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
