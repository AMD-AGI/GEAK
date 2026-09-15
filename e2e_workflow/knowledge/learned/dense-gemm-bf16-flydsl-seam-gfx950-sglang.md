---
name: dense-gemm-bf16-flydsl-seam-gfx950-sglang
description: bf16 dense Linear (qkv/o_proj/router) on sglang+aiter/gfx950 where the aiter tuned DB routes the shape to FlyDSL — what the DB already closes, where its gfx950 coverage silently ends, and why the remaining win is prefill-only.
key: dense_gemm_bf16 flydsl-routed seam · gfx950 · sglang+aiter (mxfp4 MoE model, unquantized attn/router path)
keywords: [flydsl_hgemm, tuned_gemm, gemm_a16w16, AITER_CONFIG_GEMM_BF16, gradlib, model_configs, gfx1250, qkv_proj, router_gate, prefill-only]
kernels: [hgemm_bf16_16x64x64x8_SPK1_W1x1x2_BLDS1_TN_AS1_BIAS_0, hgemm_bf16_32x64x64x5_SPK9_W1x4x1]
platforms: [gfx950]
kernel_class: dense_gemm
regime: sglang + aiter, bf16 a16w16 with fused bias, decode M=1/64 + prefill M=1024
type: lever
confidence: ★★
confirms: 1
effect: Tier-A backend swap and the aiter DB tune are CLOSED for every decode bucket (the shipped gptoss model_config already picked FlyDSL and it wins by 1.2–1.5x over both hipBLASLt and aiter-Triton). The only headroom is the two PREFILL M=1024 families, which have NO gfx950 row at all — a tuned/plain hipBLASLt row buys 1.24x/1.22x there, but served-weighted that is only 1.029x -> 0.17% e2e ceiling at a 6.02% seam. Stack-only.
last_seen: 2026-09-09
---
# bf16 dense GEMM routed to FlyDSL (sglang+aiter, gfx950)

- **Check `model_configs/<model>_bf16_tuned_gemm.csv` BEFORE tuning anything.** On this build
  `configs/bf16_tuned_gemm.csv` has ZERO rows for the model's families, but the per-model CSV is merged
  in automatically and is exhaustively tuned per M bucket — so "0 shipped coverage" read off the base
  CSV is wrong, and a re-tune of the covered buckets is a measured no-op.
- **Read the `gfx` column, not just (M,N,K).** The gptoss table's M=1024 rows for N=5120/K=2880 and
  N=2880/K=4096 are **gfx1250**; on gfx950 those two prefill shapes are uncovered and fall through to
  `torch`. That is the entire headroom at this seam, and grepping without `$1=="gfx950"` hides it.
- **gradlib on this build is hipBLASLt-only** (`GemmTuner (hipblaslt-only)`; asm/opus/flydsl/triton/
  skinny search paths deleted). So "one aiter tune covers per-backend GEMM tuning" no longer holds —
  it can only ever propose hipBLASLt rows, and it cannot re-tune the FlyDSL geometry.
- **The tuner's `us` is a HOT microbench.** Cold-cache (the serving reality: 69 GB of weights, nothing
  resident) its "best" solidx lost to the plain `torch`/hipBLASLt heuristic on both winning shapes
  (50.68 vs 46.24; 47.12 vs 46.88). Re-measure cold before deploying a solidx; a `libtype=torch,
  solidx=0` row is a legitimate deployable outcome.
- **Deploy a SINGLE self-contained CSV rebuilt from the full merge, not a colon list.** The colon-merge
  engages, but `jit/core.update_config_files`' dedup is non-deterministic on this build: a plain
  re-import regenerated `/tmp/aiter_configs/bf16_tuned_gemm.csv` MISSING gfx950 rows the live server had
  been using (`not found tuned config … using torch`). One path skips that code entirely; build it from
  all 15 source CSVs so no shipped coverage is dropped.
- **Prefill-only wins do not bank.** analytic passes are decode 1024 vs prefill 64, so a 1.24x on both
  prefill families is 1.029x served-weighted = 0.17% e2e at a 6.02% seam. Size the lever on the
  DECODE bucket before spending a round.
- caution: `aiter.ops.triton.gemm.basic.gemm_a16w16` hard-aborts the Triton compiler
  (`llvm iota_range Begin <= End` assertion) at M=1/N=2880/K=4096 on this image — bench it in a
  subprocess or one abort takes the whole bake-off with it.
- caution: the shared `scripts/op_bench.py` mis-scores this seam — it benches only the primary member,
  probes flydsl with DEFAULT tiling (4x off the live tuned geometry) and uses hipBLASLt as the baseline,
  yielding winner=hipblaslt speedup=1.0. Write the corrected per-case driver (role step 2b).
- open: Tier-C not yet run. qkv decode M=64 moves ~30.5 MB in 24.36 us ≈ 1.25 TB/s cold — well off the
  gfx950 roofline, so the decode mass is unclosed; flydsl rewrite (editable live source) then triton author.
- source: exp e2e_gpt-oss-120b_20260909_025102 head bake-off (GPU 2), 2026-09-09; artifacts in that
  eval dir's `config/head_gemm_bakeoff_report.md`.
