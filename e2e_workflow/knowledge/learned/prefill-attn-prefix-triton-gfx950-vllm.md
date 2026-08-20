---
key: dense GQA chunked-prefill attention (prefix_prefill) · gfx950 · vLLM
type: routing
confidence: ★★★
confirms: 5
effect: no op-level env/flag lever exists — the live path already IS the editable in-tree Triton kernel, so Tier-C rewrite is the only route. Head 8.6–25% GPU across 5 models; e2e transfer still unmeasured.
last_seen: 2026-08-13
---
# vLLM V1 ROCM_ATTN chunked-prefill — the live path IS the editable Triton `_fwd_kernel`

- path: (1) the head is `vllm.v1.attention.ops.prefix_prefill:context_attention_fwd` (the `_fwd_kernel`
  flash-attn), routed via `chunked_prefill_paged_decode` when max_query_len>1. (2) Do not expect an
  op-level env winner: the aiter/CK cross-backend swap is a SERVER flag (`--attention-backend`), i.e.
  the Config Tuner's job, so `op_bench.py:bench_attn` only validates the oracle. `current correct
  rel=0`, `winner=none`, `harness_suspect=false`, smoke speedup ~1.0 is the EXPECTED reading here, not
  a fault — target==baseline because both are the same live seam. (3) Go straight to Tier-C Triton
  `route=rewrite` (`mode=optimize`, an editable impl exists): autotune BLOCK_M/BLOCK_N, num_warps,
  num_stages, waves_per_eu, matrix_instr_nonkdim for GQA 32q/8kv, head_dim=128, causal.
- expected gain: purely Amdahl-scaled off the head share, so screen before spending a round —
  at 8.58% head a 1.2×/1.5× buys +1.5%/+2.9% e2e; at 16.94% → +2.8%/+5.6%; at 25.11% → +4.4%/+9.1%.
- apply: in the served regime **context_len=0** (fresh chunks) → the paged K/V cache is NEVER
  dereferenced (the context loop is `range(0, ctx=0)`). Optimize the current-chunk causal QKᵀ·softmax·V
  path, not the prefix-cache loop. Oracle cases are many SHORT varlen prefill chunks (M∈{16..336}), not
  one long sequence — the win comes from small-tile efficiency.
- verify: judge against the immutable `unittest.py` (bf16, 3 random draws) with
  baseline_callable==target_callable==`context_attention_fwd` — never a naive scaffold. Prefill-ONLY
  head (decode is served by a separate paged_attention kernel) → `served_regimes=['prefill']`, no decode
  bucket. Must stay CUDA-graph capture-safe (no host sync).
- caution: flydsl is a GEMM DSL, not an attention author target; ck/hip are absent-gated on this image
  (no ckProfiler). e2e rebind seam = `context_attention_fwd`.
- source: 2026-08-13, gfx950 / vLLM 0.26.0 — Llama-3.1-8B TP1 (head 14.98% and 16.94%), Qwen3-0.6B TP1
  (25.11%), Qwen3-8B TP1 (16.59%, GQA 32q/8kv hd128), Mixtral-8x7B TP8 (8.58% prefill, per-rank GQA
  4q/1kv). All five: op_bench current correct, winner=none, harness_suspect=false — as predicted.
