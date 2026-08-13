---
key: fp8 a8w8 blockscale GEMM · gfx950 · vLLM prefill+decode
type: lever
confidence: ★★★
effect: iso 3.17× serving-weighted (prefill 3.3×, decode 1.14×) default-vs-tuned CK on the head; e2e GATED +16.1% (verified, cold 1.178× / hot 1.155×, 3651.6→4198.4 tok/s, TTFT 646→389ms) — Qwen3.5-27B-FP8 TP4 1024/1024/conc64
last_seen: 2026-08-12
---
# gfx950 vLLM fp8 a8w8 blockscale — CK per-shape tune DB (no overlay needed)
- lever: the MANDATED CK skill `gemm_tuning/fp8_gemm_tuning_sglang_aiter.md`, but on vLLM the ONLY moving
  part is the per-shape CK tune DB. Unlike sglang, vLLM ALREADY dispatches the CK `gemm_a8w8_blockscale`
  (live baseline_callable == `aiter:gemm_a8w8_blockscale`); it just runs the UNTUNED default config
  (0 gfx950 blockscale coverage in shipped tables) → real headroom.
- apply: run aiter CK tuner (`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py --libtype both
  --mp <all GPUs>`) on the captured head (M,N,K) shapes → tuned CSV; deploy `winner_kind=env`,
  `apply_env=AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=<tuned.csv>`. NO fp8_utils Triton→CK overlay and NO
  code_patch (the sglang seam) — vLLM needs none; the tuned CSV alone binds.
- verify: `AITER_LOG_TUNED_CONFIG=1` → grep `is tuned on cu_num` (got 15/15 shapes engaged). Parity
  unchanged (rel_err ~2e-4 vs fp32 dequant oracle, << TOL 0.05) → parity_note=expected_close.
- caution: gains are prefill-driven (M=4096 ~3.3×); decode M1/M64 only ~1.14× — also verify the e2e gate,
  since a decode-bound serving mix will bank far less than the serving-weighted 3.17×. If ckProfiler is
  absent from the image the CK *author* lane is unavailable, but the per-shape tune DB still applies (it
  is the only moving part here).
- e2e outcome: the iso 3.17× DID transfer through the Integrator gate — accepted head `gemm_a8w8_blockscale`
  (backend=aiter, winner_kind=env), e2e_delta +16.1%, throughput 3651.6→4198.4 tok/s (cold 1.178×), TTFT
  646→389ms, TPOT 16.9→14.8ms; gsm8k parity clean. So on this decode-heavy serving mix the prefill-driven
  iso win still banks a large e2e gain — the earlier "decode mix banks far less" caution held only partially.
- source: exp/e2e_*Qwen3.5-27B-FP8*/ 2026-08-12 (verified A/B, status ok, throughput_speedup 1.178;
  tuned CSV in config/ck_tune/, recipe in gemm_tuning/fp8_gemm_tuning_sglang_aiter.md)
