---
key: paged decode attention · gfx942 · vLLM (pow2 + non-pow2 KV block; incl. MLA TRITON_MLA)
type: routing
confidence: ★★★
effect: head ~8-21% GPU; decode-regime Triton/HIP rewrite → ~+1-4% e2e ceiling (modest, real); op-level backend bake-off is N/A (server-flag swap)
confirms: 6
last_seen: 2026-07-05
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
- verify: `op_bench.py:bench_attn` does NOT do a cross-backend op bake-off — it only validates the oracle
  (harness_suspect=false is expected, NOT a fault). Run the immutable `unittest.py` directly as the bench.
  e2e rebind seam (pow2 block): `vllm._custom_ops:paged_attention_rocm` (attribute-access call site →
  monkeypatch rebinds cleanly); non-pow2 seam: `chunked_prefill_paged_decode`.
- caution: with a **pow2** KV block_size (e.g. 16, Qwen3-14B), `use_rocm_custom_paged_attention()` is True
  so the live path is the NON-editable ROCm/CK `paged_attention_ll4mi_QKV_mfma16` kernel → Tier-C is
  route=**author** (fresh Triton paged-decode), not rewrite. Also verify the unittest's per-call ms is
  rebuild-dominated (large paged KV copied to device each call → ~66ms baseline regardless of kernel), so
  trust geomean speedup ratio, not the absolute ms, and re-confirm the win at the e2e gate (decode is
  launch/rebuild-bound — a host-heavy rewrite can net ~0). Cross-backend aiter↔triton swap = Config Tuner
  server flag, not op-level.
- MLA decode (DeepSeek/Kimi weight-absorbed MQA): the live MLA backend on gfx942 is TRITON_MLA = the
  editable in-tree Triton stage1 grouped kernel (`vllm.v1.attention.ops.triton_decode_attention:`
  `_fwd_grouped_kernel_stage1`, seam = host wrapper `decode_attention_fwd`). Same routing as paged
  decode: aiter/CK MLA swap = a server `--attention-backend` flag (Config Tuner), so op-level is a
  Tier-C Triton route=rewrite (mode=optimize). Knobs: BLOCK_N (kv tile, 16 on HIP), BLOCK_H (16),
  num_warps {4,8}, num_stages (1 on HIP), waves_per_eu, matrix_instr_nonkdim {16}, kpack {2},
  num_kv_splits. TritonMLA runs PIECEWISE cudagraph (AttentionCGSupport.NEVER) → tile/schedule autotune
  is parity-safe (oracle rel=0) and capture-safe. op_bench bench_attn does NOT do a cross-backend
  bake-off (validates oracle; harness_suspect=false expected) — run immutable unittest.py directly.
- caution (Kimi-K2.6 ROCM_AITER_MLA persistent `_ps` asm path): when the deployed MLA backend is the
  aiter persistent asm kernel (seam `aiter.mla:mla_decode_fwd`, NOT TRITON_MLA), the live path is
  NON-editable asm → Tier-C is route=**author** (fresh Triton MLA decode against the immutable synth
  oracle), not rewrite. ROCM_AITER_TRITON_MLA crashes at launch so the server-flag swap is dead. Oracle
  is synthesized in-unittest (value-independent decode; no reference_io.pt → provenance = unittest_sha256,
  reference_io_sha256 empty by design). The author kernel must accept the same paged-KV +
  get_mla_metadata_v1 work/reduce metadata signature (or build its own), stay UNIFORM_BATCH cudagraph-safe
  (no host sync), in-place into o, no persistent HBM. Amdahl is modest here (7.52% head → ~+0.7% e2e at 1.1x).
- caution (vLLM 0.21 UNIFIED_ATTENTION triton_attn backend, VLLM_ROCM_USE_AITER=0): live seam =
  editable in-tree Triton `vllm.v1.attention.ops.triton_unified_attention:unified_attention` (the
  @triton.jit `kernel_unified_attention` 2D/3D + `reduce_segments`). aiter's copy
  `aiter.ops.triton.attention.unified_attention` is NOT the live path — do not bench/rebind it. No
  op-level env/flag win: aiter↔triton is a `--attention-backend` server flag (Config Tuner). op_bench
  bench_attn only validates the oracle (harness_suspect=false expected, rel=0) → run immutable
  unittest.py directly; baseline vs current is ~1.0x self (identity, no overlay). Tier-C = Triton
  route=**rewrite** (mode=optimize): autotune BLOCK_M/BLOCK_Q/BLOCK_SIZE/TILE_SIZE, num_warps,
  num_stages, waves_per_eu; serves BOTH configs (sliding head_dim=256 GQA16/8 + full head_dim=512
  GQA16/2) across prefill+decode — MUST not regress decode M-buckets {1,64} and stay HIP-graph-safe.
  Integrator also rebinds the consumers' imported ref in vllm.v1.attention.backends.triton_attn /
  rocm_aiter_unified_attn. CK attention candidate here needs ckProfiler (absent on this image → advisory,
  fall back to triton/hip).
- source: exp/e2e_*Qwen3.5-27B-FP8*/ 2026-06-15 (non-pow2); e2e_Qwen-Qwen3-14B_20260622 (pow2 bk=16, 21% head);
  e2e_moonshotai-Kimi-K2.6_20260622 (MLA decode stage1 TRITON_MLA, 17.23% head, baseline decode M1=0.219ms/M64=0.265ms, oracle rel=0;
  AND aiter `_ps` asm head h1 7.52%, baseline M1=0.062ms/M64=0.080ms/M64-long=0.312ms, oracle rel pass tol2e-2 → route=author triton);
  e2e_moonshotai-Kimi-K2.6_20260623 (re-confirm aiter `_ps` asm path, mla_a16w16_qh16..._ps.co loaded; 7.5% head, synth oracle PASS rel=0,
  baseline M1=0.184ms/M64=0.438ms; no op-level env/flag win → author_plan triton FIRST then hip, gate=author_recommended)
