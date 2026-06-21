---
key: block-sparse NSA GQA attention (prefill) · gfx950 · vLLM
type: routing
confidence: ★★
effect: ~5.6% GPU (MiniMax-M3-MXFP8 prefill); editable in-tree Triton kernel → Tier-C rewrite is the lever (e2e transfer TBD)
last_seen: 2026-06-20
---
# MiniMax-M3 / NSA block-sparse GQA attention → the live path IS an editable in-tree Triton kernel
- lever: this op (`_gqa_sparse_fwd_kernel`, host wrapper `minimax_m3_sparse_attn`) is a custom
  block-sparse flash kernel — there is NO library (CK/aiter) equivalent to swap to, so the op-level
  bake-off has no env/flag winner. `op_bench.py:bench_attn` only validates the oracle and delegates
  any `--attention-backend` swap to the Config Tuner; it produces NO isolated ms. The actual baseline
  bar = the IMMUTABLE `unittest.py` (here ~1.40 ms/case at M~8192, topk=16 blocks, 16q/1kv GQA, fp8 KV).
- apply: Tier-C author, **Triton route=rewrite FIRST** (editable impl exists → `mode=optimize`): the
  kernel is BLOCK_SIZE_Q=1 with a per-query-token inner loop over selected blocks (BLOCK_SIZE_K=128,
  one sparse block == one page). Tuning surface: pack multiple query tokens per program (BLOCK_SIZE_Q>1
  / num_q_loop) to amortize topk/page-table loads and feed wider MFMA; num_warps/num_stages for gfx950
  (note: the kernel pins num_stages=1 ONLY on gfx942 for LDS; gfx950 keeps Triton default — a tuning
  knob); fp8 KV dequant happens per-load (`k.to(q.dtype)`) — fuse into the dot. HIP author is a
  documented second lever (capability_index `sparse_attention_nsa` backend=hip, gfx950, fp8_e4m3_fnuz;
  TileLang on gfx942 only).
- verify: judge against the immutable oracle (bf16 rtol=atol=2e-2). This is prefill-only (decode is a
  SEPARATE callable `_gqa_sparse_decode_kernel`), so REQUIRE_DECODE_BUCKET does not apply. e2e rebind
  seam = `vllm.models.minimax_m3.common.sparse_attention:minimax_m3_sparse_attn`. Amdahl: 5.6% head →
  a 2x kernel ≈ +2.8% e2e ceiling — modest, confirm it clears the e2e gate, not just isolated.
- caution: BLOCK_SIZE_Q>1 packing must keep the causal mask correct per query row and not regress the
  topk/real_topk early-exit; also verify the prefill chunk shapes (M not a multiple of 128) still pass.
- source: exp/e2e_MiniMax-M3-MXFP8_20260620_093154_1073576_26273/ 2026-06-20
