---
key: paged decode attention · gfx942 · vLLM, non-pow2 KV block
type: routing
confidence: ★★
effect: head ~8% GPU; decode-regime Triton/HIP rewrite → ~+1% e2e ceiling (modest, real)
confirms: 1
last_seen: 2026-06-15
---
# vLLM paged attention with a non-pow2 KV block → the live path is the editable in-tree Triton kernel
- lever: when the KV `block_size` is non-pow2 (e.g. 784), `use_rocm_custom_paged_attention()` returns
  False → the ROCm custom CK/aiter paged-attn is **structurally disabled**, and the live path falls to
  the in-tree **Triton** `kernel_paged_attention_2d`. So CK/aiter are NOT op-level candidates here; the
  op-level lever is a Triton (or HIP) rewrite of that kernel.
- apply: Tier-C author, Triton route=rewrite FIRST (editable kernel exists, mode=optimize): autotune
  BLOCK_M/BLOCK_N/num_warps/num_stages/waves_per_eu, MUST win/not-regress decode M-buckets {1,64} and
  stay HIP-graph-capturable. HIP author is a second lever if Triton plateaus. Reaching CK/aiter instead
  needs a server `--page-size`/`--attention-backend` change = the Config Tuner's job, not op-level.
- verify: `op_bench.py:bench_attn` needs a `reference_io.pt`; when absent (synthesized oracle) it yields
  no op-level bake-off BY DESIGN → run the immutable `unittest.py` directly as the bench. e2e rebind seam
  = `vllm.v1.attention.ops.chunked_prefill_paged_decode:chunked_prefill_paged_decode`.
- source: exp/e2e_*Qwen3.5-27B-FP8*/ 2026-06-15
