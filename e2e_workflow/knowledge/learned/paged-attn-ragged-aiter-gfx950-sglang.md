---
name: sglang aiter paged_attention_ragged decode head (block_size=1 token-indexed KV)
description: On sglang+aiter the decode head is `aiter:paged_attention_ragged` (JIT-built HIP/CK
  `paged_attention_ll4mi_QKV_mfma16_kernel`). No op-level env/flag knob exists, the stock Triton paged
  decode is 17-24x SLOWER at block_size=1, `partition_size` != 256 silently returns GARBAGE (a fake
  2.05x), and the only large lever is kv-fp8 (a server flag, lossy).
keywords: [paged attention, decode, aiter, paged_attention_ragged, ragged, block_size 1, partition_size,
  ll4mi, mfma16, kv-fp8, sglang, gfx950, memory bound]
kernels: [paged_attention_ragged, paged_attention_ll4mi_QKV_mfma16_kernel, paged_attn_decode_v2_w_dot_kernel]
platforms: [gfx950]
kernel_class: attention-decode
regime: decode, bf16 KV, GQA 40q/8kv, head_dim 128, sglang aiter backend
key: paged decode attention (ragged, block_size=1) · gfx950 · sglang + aiter
type: routing
confidence: ★★
effect: head 56.76% GPU but memory-bound at 85% of roofline (attainable ~1.057x -> ~+3% e2e ceiling);
  kv-fp8 server flag measured 1.477x serving-weighted -> +22.45% e2e ceiling (LOSSY, accuracy gate)
confirms: 1
lifecycle: active
last_seen: 2026-08-21
---
# sglang + aiter paged decode: the seam is `aiter:paged_attention_ragged`, and it has NO op-level knob
- lever (routing): the live decode attention on sglang's aiter backend is
  `sglang.srt.layers.attention.aiter_backend:paged_attention_ragged` -> `aiter:paged_attention_ragged`,
  which `compile_template_op`-JITs `paged_attention_ll4mi_QKV_mfma16_kernel` from
  `aiter/csrc/cpp_itfs/pa/pa_kernels.cuh` + `pa_ragged.cuh`. The task `meta.editable=false` refers to the
  *installed* kernel, but the HIP/CK **template source ships in the image and is JIT-recompiled at
  import**, so a Tier-C `ck`/HIP route is a **rewrite** (reversible overlay of those headers), not an
  author-from-zero. `op_bench.py:bench_attn` takes NO timing here by design (backend choice is a server
  flag) -> `measured:false`, `harness_suspect:false` is expected; run the immutable `unittest.py` and a
  hand-written driver for the real op-level bake-off.
- caution (**the fake-win trap on this op**): `partition_size` is the only knob the seam exposes
  (sglang hardcodes the module constant `_AITER_PARTITION_SIZE_ROCM = 256`, no env). Sweeping it looks
  like a free win — ps=512 times 1.25x and ps=1024 times **2.05x** — but the kernel is compiled for
  256, so every value != 256 returns GARBAGE (max_rel_err 10^2..10^4 vs the fp32 oracle; ps=128 returns
  NaN). ps=64/128 are also slower. Always re-check correctness before believing a partition/split knob.
- caution: the stock editable Triton paged decode
  `aiter.ops.triton.attention.pa_decode:paged_attention_decode` IS correct on these shapes (rel 0.044 -
  0.363, i.e. at/below the incumbent's own oracle error) but runs **0.057x serving-weighted** at the
  b64/s8704 bucket because the deployment uses **block_size = 1 token-indexed pages**, so its
  per-block dot degenerates to one token per tile. It is still the right **author SEED** for a Triton
  Tier-C lane, but the lane must specialize to blk=1 ragged KV or it cannot approach the bar. Also pass
  its `compute_type` as a **triton** dtype (`tl.bfloat16`); a `torch.dtype` raises
  `'torch.dtype' object has no attribute 'scalar'` at compile time (looks like a harness fault, is not).
- **kv-fp8 is by far the biggest lever on this head and it is a SERVER FLAG.** Re-timing the identical
  seam/shapes with the paged K/V pool in `fp8_e4m3` + per-tensor k_scale/v_scale
  (`kv_cache_dtype="fp8_e4m3"`, in-kernel dequant) gives **1.477x serving-weighted** (b64/s8704 1.477x,
  ragged 1.569x, b1 1.188x) = **+22.45% e2e ceiling at a 56.76% head** — 7x anything a rewrite can reach
  (roofline says the bf16 kernel is already at 85% of achievable BW, attainable ~1.057x). It is
  `--kv-cache-dtype fp8_e4m3` (Config Tuner), LOSSY, gsm8k-style accuracy gate, never byte parity.
  Surface it even when `ENABLE_FP8=false`.
- verify: the incumbent's own error vs the frozen fp32 oracle is LARGE here (max_rel_err 0.365 / 0.217 /
  0.063), so an absolute `tol=2e-2` band rejects the baseline itself — judge candidates against the
  incumbent's band (`_oracle_provenance.json:incumbent_max_rel_err`), which is what the immutable
  `unittest.py` effectively does (identity run: PASS, weighted 1.0004).
- source: e2e_cycle4 Qwen3-14B-FP8 gfx950 MI355X sglang 0.5.17 TP1 (ISL 8192/OSL 1024/conc 64), head
  56.76%; identity baseline captured for decode_b64_s8704 / decode_b1_s8704 / decode_b64_ragged
  (absolute timings in EVAL_DIR); drivers in `<eval>/config/pa_ragged_partition_sweep.py`, `pa_ragged_backend_bakeoff.py`,
  `pa_kvfp8_probe.py`. e2e transfer: not yet gated (bake-off produced no direct winner).
