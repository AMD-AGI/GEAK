---
key: fp8 a8w8 blockscale BPRESHUFFLE GEMM · gfx950 (MI355X) · sglang decode-heavy
type: lever
confidence: ★★
effect: iso serving-weighted 1.022x (immutable UT, RESULT PASS) — prefill M15360/16384 1.09–1.13x, decode M1–M32 1.42–1.52x, but decode M64/M128 1.00x → e2e Amdahl ceiling only ~+1.0% at 45.7% GPU time
last_seen: 2026-08-16
---
# gfx950 sglang fp8 a8w8 blockscale **bpreshuffle** — CK per-shape tune DB (`--preshuffle` variant)
- live path: sglang `fp8_utils.aiter_w8a8_block_fp8_linear` already binds **CK**
  `aiter.gemm_a8w8_blockscale_bpreshuffle` (`_use_aiter_bpreshuffle_gfx95=True`,
  `use_aiter_triton_gemm_w8a8_tuned_gfx950(N,K)=False` for all four (N,K)) → the CK skill's **§9
  fp8_utils Triton→CK import overlay is a NO-OP here; NO code_patch, apply_env alone binds.** Verify this
  per box — the same skill on gfx942 sglang DID need the overlay (Triton was live there).
- lever: the CK skill, `--preshuffle` variant. Tuner = the SAME
  `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py` with `--preshuffle --libtype all`
  (races ck / cktile / gfx950 asm `fp8gemm_bf16_blockscale_BpreShuffle_*x128`); it then writes/reads
  **`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE`** (a DIFFERENT env from the non-preshuffle
  `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` — setting the wrong one binds to nothing).
  Shipped `aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv` has **zero rows** for Qwen3-14B's
  (N,K) ∈ {(7168,5120),(5120,5120),(34816,5120),(5120,17408)} → live = CK *default* kernel = real headroom.
  Cost: 64 shapes tuned in ~6 min on ONE MI355X, errRatio 0.0 on every row.
- verify: `AITER_LOG_TUNED_CONFIG=1` → "is tuned on cu_num = 256" on 20/20 live (M,N,K). Winners are
  `ck` for M≤1024 and `cktile` (`192x256x128_4x2x1_16x16x128_intrawave`) for M≥15360.
- 🔑 **caution — the tuner's `us` column massively overstates the decode win, and the M-profile is
  non-monotonic.** Tuner (hot, LLC-resident) said 121 us for the M=64 four-family sum vs 214 us measured;
  the production op under **cold-cache** CUDA-event timing (what the UT and the live server see — the 14B
  weight set ≫ the 256 MB Infinity Cache, so every decode step reads weights cold from HBM) gives:
  M=1 1.41x · M=2–8 1.43x · M=16–32 1.49–1.52x · **M=64 1.007x · M=128 1.00x** · M=256 1.01x ·
  M=512–2048 1.09–1.12x · M=15360/16384 1.09/1.13x. At M=64–128 the CK *default* heuristic already picks
  a near-cold-HBM-bound kernel, so the tune buys nothing exactly where a conc=64 decode mix lives.
  ⇒ ALWAYS re-measure the tune with cold-flush device timing on the production op before crediting it;
  never take the tuner CSV `us` (or the vLLM gfx950 non-preshuffle card's +16.1%) as the expected e2e.
- outcome here: UT `RESULT PASS`, max_rel_err 0.0 vs the live baseline on every random draw (the tuned
  kernels are bit-identical on the covered shapes) → `parity_note=expected_close`, zero HBM cost.
  GEAK_WEIGHTED_SPEEDUP 1.0225 / geomean 1.1194 → **Amdahl ceiling only ~+1.0% e2e** despite the head
  owning 45.66% of GPU time. Ship it (free, safe, helps TTFT ~1.13x on prefill) but do NOT expect it to
  clear the noise band alone — the real head lever on this box is the Tier-C author route (an authored
  kernel at 2x would be +29.6% e2e).
- source: e2e_cycle0 Qwen3-14B-FP8 TP=1 gfx950/MI355X cu_num=256, 2026-08-16;
  tuned CSV + A/B driver in `<eval>/config/ck_tune/`.
