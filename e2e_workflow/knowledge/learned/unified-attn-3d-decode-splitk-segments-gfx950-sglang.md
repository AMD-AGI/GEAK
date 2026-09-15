---
name: unified-attn-3d-decode-splitk-segments-gfx950-sglang
description: sglang's live decode attention IS aiter's editable Triton unified_attention 3d; its MIN_SEGMENTS=8 split-K floor over-splits whenever the un-split grid already fills the CUs — an occupancy-derived per-call cap that collapses to NS=1 (one dispatch) plus num_warps 2->4 is 1.234x isolated byte-exact and rode a Director-validated bundle.
keywords: [unified_attention, hip-graph, capture-time-reachability, engagement-proof, reduce_segments, split-k, num-segments-per-seq, select_3d_config, occupancy, launch-shape, num-warps, paged-decode, aiter-triton, sglang, gpt-oss]
kernels: [kernel_unified_attention_3d, reduce_segments, select_3d_config]
platforms: [gfx950]
kernel_class: attention_decode
regime: decode, bf16 q + bf16 paged KV, block_size 64, head_size 64, 64 q-heads / 8 kv-heads, max_seqlen_k ~1.1k, m in {1,64}
key: unified_attention 3d ALL_DECODE paged decode · gfx950 · sglang + aiter Triton (bf16 KV, head_size 64, GQA 8)
type: lever
confidence: ★★
effect: iso 1.2336x serving-weighted, byte-identical (max_rel_err 0.0); m=64 1.16-1.19x, m=1 held at 1.00x by the per-call cap. e2e TRANSFER MEASURED and small, as Amdahl predicts at a 5.01% head: kernel-only same-session A/B +0.402% (6047.2 -> 6071.5 tok/s, engagement-proven 4 inject hits/replica, ranges overlap, below a 0.5% band -> gate=stack, not standalone-accept); it then shipped inside a bundle the Director validated at +1.809% (5942.2 -> 6049.8, 5 fresh servers/leg, permutation p=0.048, gsm8k 0.950 -> 0.940 = 1 sigma), where the decode half shows as TPOT 10.162 -> 10.069 ms (-0.9%). Stack-only: real, reproducible, too small to headline alone. A LATER re-authoring of the same seam on the same stack transferred NOTHING (+0.186% e2e, inside the noise band and inside its own +0.586% Amdahl ceiling) purely because its fast path was guarded on a condition the HIP-graph CAPTURE shapes did not satisfy — see the capture-reachability caution.
confirms: 2
lifecycle: active
last_seen: 2026-09-10
---
# sglang's live decode attention IS aiter's editable Triton `unified_attention` — its split-K floor ignores occupancy
- routing: on sglang+aiter (gfx950) the live decode seam is
  `sglang.srt.layers.attention.aiter_backend:unified_attention`, the module-global alias of
  **`aiter.ops.triton.attention.unified_attention`** — editable in-tree Triton, so Tier-C is
  route=**rewrite** (mode=optimize). (Contrast the vLLM UNIFIED_ATTENTION card: there the aiter copy is
  NOT live.) There is **no op-level env/flag knob**: `select_3d_config()` is a hardcoded heuristic with
  no config-JSON/env path, so `op_bench.py bench_attn` correctly takes no timing (`measured:false`,
  `harness_suspect:false` is EXPECTED, not a fault) — bench with the immutable `unittest.py`.
- lever: derive the segment cap from THIS call's grid instead of the constant floor. `select_3d_config()`
  forces `MIN_SEGMENTS = min(8, MAX_SEGMENTS)`; recompute `cap = next_pow2(ceil(cu_count / num_2d_prgms))`
  (~1 workgroup per CU, `cu_count = target_num_prgms // 4`) and only clamp when the stock count exceeds it.
  At m=64 that collapses NS 8/4 -> **1**, which removes the fp32 `segm_output/max/expsum` round-trip
  (~8.4 MB of ~152 MB per-call traffic at NS=4) AND the whole second `reduce_segments` dispatch (~4.4 us
  device + a second host launch-prep). Then restore the memory-level parallelism the smaller grid cost:
  `num_warps` 2 -> 4 **on the NS=1 path only** (4.34 CTA/CU x 2 waves = 2.17 waves/SIMD, only 43% of the
  5-wave VGPR ceiling; LDS=scratch=0 so wave count is the only latency cover). Measured m=64 k=1088
  0.05108 -> 0.04304 ms. A flat clamp to 4 was the weaker earlier form (1.069x); NS=2 measured 1.066x —
  the full collapse, not a halved split, is what pays.
- apply: overlay the module (`[overlay] injected module aiter.ops.triton.attention.unified_attention`);
  ordering matters — apply the num_warps override AFTER `select_3d_config` returns, because it computes
  `occ = waves_per_eu * 4 // attn_warps` and scales `target_num_prgms` by it, so editing warps inside it
  silently halves the stock split before the fill-aware cap sees it.
- verify: immutable `unittest.py` -> CORRECTNESS PASS `max_rel_err=0.0` on eager, random-draw and
  graph_replay legs (constexpr launch geometry only: no host sync, no per-call allocation growth, so
  HIP-graph-capture safe). Live: log the chosen (NS, warps) per call — the shipped config fired on 702 of
  1080 live decode calls, i.e. the cap is genuinely per-call, not latched.
- caution: also verify the SMALL-batch bucket separately. A flat NS<=4 clamp regressed m=1 to 0.907x and
  NS=1 to 0.679x (1 seq x 8 kv heads = 8 CTAs on 118 CUs is block-starved); the occupancy-derived cap is
  what keeps m=1 at 1.00x, so a batch-blind constant is the wrong shape of fix. And size the reward before
  the budget: at a ~5% head even 1.23x caps e2e near +1%, so expect gate=stack — bank it with other wins
  rather than expecting it to clear a noise band alone.
- caution: also verify the new branch is reachable AT HIP/CUDA-GRAPH CAPTURE, not merely on the live
  decode path. `select_3d_config` is a HOST heuristic, and the decode path is graph-REPLAYED: Python runs
  once, during capture, and never again. A re-authored variant of this same lever gated its split-K=1
  fast path on `num_2d_prgms >= 2*get_num_sms()` AND `max_seqlen_k <= 4096`; the module injected into all
  4 workers, yet the fast-path marker fired ZERO times across boot, capture, warmup and the timed round,
  because the capture-time bucket did not satisfy the guard — a complete, provenance-clean A/B that
  measured stock against stock (+0.186%, noise). So: (a) key any config decision on the CAPTURED
  batch/seq bucket rather than on a runtime-varying quantity, and (b) count the engagement marker during
  CAPTURE, not only at replay — module-injection hits prove loading, never execution.
- roofline note: the prior called this kernel `saturated`, `attainable_speedup=1.0`,
  `expected_e2e_gain_pct=0.0` (confidence low) and the measurement beat it by 1.23x — a device-time
  byte/FLOP model cannot see a dispatch-collapse win. Do not let a saturated verdict retire this seam.
- technique card (kernel sink, cite don't copy):
  `kernel_workflow/knowledge/learned/split-only-up-to-one-workgroup-per-cu-and-make-pipeline-dept-attention-decode-gfx950-decode.md`
- source: exp/e2e_*gpt-oss-120b*/ 2026-09-07..08 (gfx950, sglang 0.5.x, aiter c16d44b9) —
  tuning/attn3d_{seed_result,knob_sweep}*.json, kernels/_exp/*unified_attn_3d*/final_patch.diff,
  overlay/cand_unified_attn_3d_decode/integrate_result.json, director_e2e_validation.json.
