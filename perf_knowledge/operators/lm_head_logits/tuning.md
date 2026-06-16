---
title: lm_head_logits — tuning
kind: technique
operator: lm_head_logits
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [decode, prefill, both]
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
  - https://github.com/vllm-project/vllm/tree/main/csrc/rocm
---

# lm_head_logits — tuning

## TL;DR
Tune it as a **large-N skinny GEMM**, because that is what it is. The four levers, in order of payoff:
(1) **minimize M** — project only the sampled positions; (2) pick the **skinny / split-K** kernel so the
huge `[V,d]` weight read fills 304 CUs and is bandwidth-efficient; (3) **avoid the full-logits all-gather**
for greedy via vocab-parallel argmax; (4) **fuse the epilogue** (scale/soft_cap/bias). On the live serving
path the GEMM kernel choice is made by **aiter's tuned DB**, so the real lever is capturing the head's
`(M,N=V,K=d,bias)` shape into that DB — same mechanism as [[dense_gemm]] × [[aiter]].

## Lever 1 — minimize M (the biggest free win)
The model runner prunes `hidden` to the **last token per sequence** before the head, so `M`=batch, not
total tokens. Confirm this is happening (a regression that projects all prefill tokens turns an O(batch·V)
GEMM into O(chunk·V) — easily 100× more work). At decode `M`≤256; keep it there.

## Lever 2 — the skinny GEMM kernel (N=V is huge, K=d, M tiny)
- This is `skinny_gemv_decode` territory: `M`≤16 → small-M HIP kernels (`wvSplitK`/`LLMM1`), `M`≤256 →
  split-K to reach ≥1024 programs across the 8 XCDs so the `[V,d]` weight read is fully parallel.
- **SPLIT_K** is the key knob: with `N=V` large and `M` tiny, M·N tiles alone may not fill 304 CUs at the
  smallest batches; split the K=d reduction (atomic accumulate) to add programs. See [[triton_amd]]
  knobs.md §6 and [[skinny_gemv_decode]].
- aiter chooses `skinny` libtype for `M≤16, N≤cu_num`-ish; for the head `N=V≫cu_num`, so it more often
  lands on hipBLASLt/asm — **verify** the dispatched libtype (`AITER_LOG_MORE=1`) rather than assuming.
- Weight read dominates: this op is **bandwidth-bound**, so the goal is HBM efficiency (coalesced,
  `dwordx4`, full grid), not MFMA utilization.

## Lever 3 — skip the all-gather for greedy
`tensor_model_parallel_all_gather` of `[M,V]` is `O(M·V)` comms (M·256k floats). For greedy decoding,
`get_top_tokens` does a **local argmax per V-shard** then gathers only `(value, index)` pairs →
`O(M·2·tp)`. This removes a large collective from the decode critical path. Engage it for greedy/temp=0;
sampling still needs full logits (so it pays the all-gather). See [[argmax_topk]], [[allgather]].

## Lever 4 — fuse the epilogue
`scale` (`logits *= scale`), `soft_cap` (`soft_cap·tanh(logits/soft_cap)`, Gemma-2/3), and `bias` are a
trailing elementwise over `[M,V]`. Fuse into the GEMM epilogue (or a single follow-up kernel) to avoid an
extra `[M,V]` read/write — at V=256k this is a non-trivial bandwidth pass. See [[gemm_epilogue_fused]].

## Lever 0 — the aiter DB (live-path engagement)
Because the head GEMM dispatches through `aiter.tuned_gemm`, the only tuning that touches the **serving
path** is getting the head's shape into aiter's per-shape DB: `AITER_TUNE_GEMM=1` captures live shapes
(incl. the true `bias` flag and `M=batch`), then `gradlib` tunes, deploy by `AITER_CONFIG_GEMM_BF16=...`.
The head's `(M,N=V,K=d)` is just another row. See [[aiter]] tuned_gemm.md and [[dense_gemm]] × aiter.

## Pitfalls
- Tuning the head with `M`=chunk (all prefill tokens) when live is `M`=batch → DB miss / wrong kernel.
- Assuming `skinny` libtype engages — with `N=V` it often does not; check `AITER_LOG_MORE=1`.
- fp16 logits output: clips the dynamic range sampling needs → keep **fp32 logits** ([numerics.md](numerics.md)).
- Forgetting to fuse soft_cap → a wasted `[M,V]` pass at 256k vocab.

## Cross-links
[overview.md](overview.md) · [numerics.md](numerics.md) · [fusion.md](fusion.md) ·
[[skinny_gemv_decode]] · [[dense_gemm]] · [[aiter]] · [[argmax_topk]] · [[sampling_topk_topp]].

## Sources
- Skinny/split-K decode GEMM, ≥1024 grid, coalesced read: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- aiter tuned_gemm dispatch (skinny/hipblaslt/asm libtypes): `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/tuned_gemm.py`.
- LogitsProcessor epilogue (scale/soft_cap/bias) + `get_top_tokens` vocab-parallel argmax: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py
- vLLM skinny GEMM kernels (`wvSplitK`, `LLMM1`): https://github.com/vllm-project/vllm/tree/main/csrc/rocm
