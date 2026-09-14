---
name: sglang triton decode-attention split-KV over-splitting (hybrid SWA)
key: grouped decode attention (`_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2`) · gfx950/MI355X · sglang bf16, hybrid sliding-window + full-attention
description: On sglang/ROCm the triton decode backend over-splits KV for sliding-window layers because forward_decode never consumes the window_num_kv_splits it already computes; a 3-line launcher overlay fixes it (capping max_kv_splits via a flag is the cruder variant, and the obvious flag is dead on HIP).
keywords: [decode attention, flash-decoding, split-kv, num_kv_splits, window_num_kv_splits, sliding window, hybrid SWA, triton_backend, launcher overlay, gfx950, sglang]
kernels: [_fwd_grouped_kernel_stage1, _fwd_kernel_stage2, decode_attention_fwd, get_num_kv_splits_triton, TritonAttnBackend.forward_decode]
platforms: [gfx950/MI355X, ROCm 7.2, sglang 0.5.17]
kernel_class: attention-decode
regime: decode, page_size=1, bf16 KV, TP=2, conc=64
type: lever
confidence: ★★★
effect: ROOT CAUSE = forward_decode ignores window_num_kv_splits; overlay iso serving-wtd 1.10x with NO bucket regression, zero HBM, engagement verified. E2E VERDICT (now measured): the split-count overlay alone is SUB-NOISE (+0.19% vs a same-session paired ref, byte-exact -> `stack`); the REAL win on the same op was a BIT-NEUTRAL authored Triton rewrite (.cg cache_modifier + per-family LOOP_STAGES/waves_per_eu/schedule_hint) = iso 1.92x, **+20.53% e2e, byte-exact, ACCEPTED**. Stacked on that kernel the split lever decays 1.10 -> 1.07, and a batch-aware per-bucket split schedule tops out at 1.117.
confirms: 3
lifecycle: active
last_seen: 2026-08-21
---
# sglang triton decode attention — the split-KV count is the op-level lever, and one flag is a trap

- observed: the LIVE captured `num_kv_splits` for the **sliding-window** layers is 10/14/16 even though
  the window only spans 1023 KV slots — the launcher's own CU-aware heuristic
  (`sglang.kernels.ops.attention.metadata:get_num_kv_splits_triton`) picks **4** for that window.
  Over-splitting costs both stage1 (tiny per-split tiles) and the stage2 reduce.
- 🎯 ROOT CAUSE (found 2026-08-21, confirm #2): `TritonAttnBackend` DOES compute
  `window_num_kv_splits` from the window lengths — eager (`init_forward_metadata`) and cuda-graph
  (`_apply_cuda_graph_metadata` refills it at every replay) — and stores it on `ForwardMetadata`, but
  **`forward_decode` never consumes it**: it swaps only `kv_indptr`/`kv_indices` to the window variants
  and always passes `forward_metadata.num_kv_splits` (sized from the FULL seq_lens) to
  `decode_attention_fwd`. So both the sliding and the full-attention layers get the SAME split vector.
  The fix is a 3-hunk launcher patch / reversible sitecustomize overlay that binds
  `num_kv_splits := window_num_kv_splits` for `layer.sliding_window_size > -1` — no flag, no new buffer,
  ZERO HBM, HIP-graph-safe (both are persistent capture-stable buffers of identical shape/dtype).
  Measured (same oracle replay): `sw_B64_kv1023` 0.1660 -> 0.1438 (**1.154x**), `fa_B64_kv8167` 0.999x,
  m=1 buckets 1.02x / 1.06x -> serving-weighted **1.101x, no bucket regresses**. Prefer it over the flag:
  the flag lowers the cap for the full-attention layers too, which is what tanks the m=1 long-KV bucket.
- measured (immutable oracle replay, cuda_event_graph, gfx950, Gemma-4-26B-A4B TP=2):
  dominant `sw_B64_kv1023` **1.19x**, `fa_B64_kv8167` 1.08x
  at a uniform 4 splits; serving-weighted 1.159 (cap 4) / 1.118 (cap 8). Small-batch buckets go the OTHER
  way (m=1 long-KV 0.40x at cap 4, 0.69x at cap 8) — they carry ~0 analytic serving weight at conc=64,
  but this is the risk to gate at e2e for low-concurrency deployments.
- 🔴 TRAP: `--triton-attention-num-kv-splits` is a **dead flag on ROCm** —
  `ServerArgs._handle_amd_specifics()` unconditionally re-sets it to 16 AFTER parsing, so passing it
  changes nothing. Use **`--triton-attention-split-tile-size T`** instead: `TritonAttnBackend.__init__`
  derives `max_kv_splits = ceil(max_context_len / T)` from it, after the AMD override, on the normal
  (non-deterministic) path. e.g. ctx 13312: T=1664 -> cap 8, T=3328 -> cap 4. Bonus: the cap also sizes
  the `attn_logits`/`attn_lse` graph scratch, so lowering it FREES HBM (memory-gate friendly).
  `SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS=1` (fill cap for every seq) is the companion env.
- parity: same dtype, but changing the split count reassociates the fp32 split reduce -> rel err up to
  4.5e-3 vs the frozen golden (tol 2e-2). Not byte-identical; re-check greedy parity at the e2e gate.
- op-level bake-off is N/A as always for attention (`op_bench.py:bench_attn` takes no timing and returns
  `winner=none`, `measured=false`, `harness_suspect=false` — expected, not a fault). Bench by replaying
  the immutable `unittest.py` cases directly with the alternative launcher metadata.
- Tier-C: the live path IS the editable in-tree Triton kernel
  (`sglang/kernels/ops/attention/decode_attention.py`), so route=**rewrite**. Knobs: BLOCK_N (32; 16 when
  Lk>=576), BLOCK_H 16, num_warps, num_stages (1 on HIP), waves_per_eu, matrix_instr_nonkdim 16
  (`kpack` is deprecated on gfx950 and silently forced to 1). One kernel must serve BOTH geometries
  (sliding hd256 GQA 8/4, full hd512 GQA 8/1) and both M buckets {1,64}, and stay HIP-graph-capturable.
- engagement check: launch with `OVERLAY_PYTHONPATH=<overlay dir>` and `SGLANG_SWA_KVSPLIT_DEBUG=1`;
  the server log must show `[overlay] swa_window_kvsplits: patched TritonAttnBackend.forward_decode`
  once per TP rank + `[swa_kvsplits] engaged` (verified: 3 ranks, healthy server, no traceback).
- 🏁 **E2E OUTCOME (confirm #3, same eval, both candidates gated on a TP=2 GPU=1,2 paired A/B):**
  - split-KV launcher overlay ALONE: engaged (3 ranks), byte-exact greedy parity, **+0.19%** vs a
    same-session paired no-overlay ref (a −2.98% box drift made the stale shared ref look like a
    regression — always re-run a drift-control ref block) ⇒ **`stack`, not accept**. A 1.10x iso at a
    22.8% head buys +3.1% on paper and delivered nothing measurable: size split-count fixes accordingly.
  - **the accepted lever was a BIT-NEUTRAL authored Triton rewrite, iso 1.92x / +20.53% e2e byte-exact.**
    An earlier aggressive rewrite (USE_EXP2, constexpr split count, wider BLOCK_N, SPLITS==1 stage-2
    elision, KV-head fold) measured **+38.05% e2e but was REJECTED on greedy byte-parity** (13/16 prompts
    diverge vs a provably deterministic no-overlay baseline). The corrective re-author kept ONE numeric
    change — none — and shipped only cache/scheduling knobs: `cache_modifier=".cg"` on the two gathered
    paged-KV loads (stream past L1/LLC; the tiles are never reused) plus per-family
    `(LOOP_STAGES, waves_per_eu, schedule_hint)` — sw64 (1,4,""), fa64 (1,2,"attention"),
    m=1 (2,2,"attention,memory-bound-attention"). Family is picked from STATIC shapes only
    (Lk, workgroup count), the same way the reference launcher derives its grid, so it stays
    graph-capturable. Bit-identical `o` on all 4 buckets + the ragged/min-len replay shapes.
- 📉 **The split lever DECAYS once that kernel is live** (re-measured on the immutable oracle with the
  accepted kernel as BOTH legs, GPU 1, cuda_event_graph 10/50): window-splits overlay serving-wtd
  **1.0708** (was 1.1008 on the stock kernel), sw_B64 1.111 / fa_B64 1.006 / sw_B1 1.020 / fa_B1 1.050.
- 🔎 **NEW: the full-attention family has its OWN split optimum, and the live heuristic misses it too.**
  Sweeping the per-seq split VALUE at the DEPLOYED cap (max_kv_splits=16, scratch unchanged, zero HBM):
  sw_B64_kv1023 best **3** (1.146; 4 = 1.126, live 14 = 1.00), fa_B64_kv8167 best **8** (1.070; live 14 =
  0.988, 16 = 1.006), sw_B1 best 14 (1.014), fa_B1 best 16 (1.056). A batch-aware schedule hitting all
  four = serving-wtd **1.1169 with NO bucket regression** — strictly better than both the window-splits
  overlay (1.071) and the `--triton-attention-split-tile-size` flag (1.111 but fa_B1 0.726). ⚠ the curve
  is NON-MONOTONE (fa_B64: 8 → 1.070 but 10 → 0.869) — tile/split-boundary alignment, so tune the value,
  don't extrapolate a rule from two points. Deployable shape: clamp the per-seq split VALUES (host-side
  metadata, identical buffer shape/dtype ⇒ graph-safe); do NOT lower `max_kv_splits`, which also shrinks
  the attn_logits scratch and forces the same cap on every family (that is what tanks m=1 long-KV).
- source: 2026-08-21, e2e_cycle3 Gemma-4-26B-A4B-it (hybrid SWA 25+5 layers), gfx950 TP=2.
