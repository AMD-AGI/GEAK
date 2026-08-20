---
key: fp8 a8w8 blockscale GEMM · gfx950 · sglang (ROCm>=7.2) prefill+decode
type: lever
confidence: ★★★
effect: bpreshuffle CK per-shape tune DB → iso 1.46× geomean (per-shape 1.01–2.28×, no regressions); e2e VERIFIED +20.91% (Qwen3-14B-FP8 TP1, Director validated_win, 5046.9→6102.0 tok/s, byte-identical parity 8/8, TPOT −17.5%) at a 53.51% head — i.e. AT the Amdahl ceiling.
last_seen: 2026-08-17
---
# gfx950 sglang fp8 a8w8 blockscale — the live kernel is CK **bpreshuffle**, so tune THAT DB

- path: (1) read the live dispatch first — on gfx950 + sglang>=0.5.17 + ROCm>=7.2 the live kernel is CK
  `gemm_a8w8_blockscale_bpreshuffle`, NOT plain CK and NOT Triton. `fp8_utils.aiter_w8a8_block_fp8_linear`
  sets `_use_aiter_bpreshuffle_gfx95 = _use_aiter_gfx95 and hip>=7.2` and dispatches bpreshuffle for any
  (n,k) outside the hardcoded `use_aiter_triton_gemm_w8a8_tuned_gfx950` allowlist — the head families
  typically fall outside it. (2) aiter is already ON in the stock server, so there is NO swap and NO
  fp8_utils overlay to apply. (3) The whole lever is the per-shape tune DB for the bpreshuffle variant.
- expected gain: gated by shipped coverage, which is thin — the shipped
  `a8w8_blockscale_bpreshuffle_tuned_gemm.csv` (58 gfx950 rows) covered ZERO of the four head families,
  so all 12 head shapes logged `will use default config!`. Tuning them gave 1.28–1.93× at M1, 1.17–2.28×
  at M32, 1.19–1.86× at large M, with no regressions (unlike the vLLM plain-CK default, which regresses
  prefill). At a 53.51% head that banked +20.91% e2e.
- apply: `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --preshuffle --libtype both`
  (the bpreshuffle dir has no tuner of its own; it reuses the plain one with `--preshuffle`). Deploy via
  **`AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=<csv>`** — a DIFFERENT env from the plain
  `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE`. Merge your rows over the shipped table. winner_kind=env, ZERO
  extra HBM, no code_patch. ckProfiler not needed (JIT ck+cktile).
- verify: `AITER_LOG_TUNED_CONFIG=1` → `is tuned on cu_num` for the head shapes, 0 `use default`.
  Residual `use default` hits on M:248/256 are cudagraph-capture buckets, not head-shape misses.
  fp8 → accuracy gate rather than byte-exact (tuner rel<0.01 all shapes).
- caution: ALWAYS re-read the live `fp8_utils` dispatch — bpreshuffle vs plain-CK vs Triton is coupled to
  sglang version, ROCm version, and the (n,k) allowlist. On ROCm<7.2 gfx950 bpreshuffle is disabled
  (hipcc miscompile #23319) and plain CK is used with a per-shape NaN M-cap. Probe which kernel imports
  first, then pick the matching tune-DB env. Note sglang handles the bpreshuffle scale layout
  (`materialize_bpreshuffle_fp8_scale`), so bpreshuffle is correct here — contrast the vLLM/gfx942 cards
  where rebinding a PLAIN-blockscale call to bpreshuffle without the layout fix gave rel≈42 garbage.
  Author lanes are low-ROI here: flydsl is role-forbidden for fp8 blockscale, CK author needs the absent
  ckProfiler, and Triton is unlikely to beat tuned CK bpreshuffle.
- source: exp/e2e_*Qwen3-14B-FP8*_sglang_*/ 2026-08-17 (bakeoff + tuned CSV in `ck_tune/`; 34/36 shapes
  updated, all correct; Director validated_win +20.91%, non-overlapping, gsm8k unchanged).
