---
name: mxfp4 fused-MoE on sglang+aiter (gfx950) — the shipped tuned_fmoe CSV is already the optimum; the only env lever left is the 1-STAGE fuse at the decode tier
description: >
  Bake-off result for an aiter `fused_moe` mxfp4 (fp4_e2m1 weights + e8m0 per_1x32 scales, bf16 act)
  grouped-expert head on sglang/gfx950 (Qwen3.5-397B-A17B-MXFP4, TP4, inter_dim/rank 256, E=512,
  topk=10). Racing the FULL flydsl stage-2 kernel registry against the shipped model-specific
  `*_fp4_tuned_fmoe.csv` finds NO tile that beats the shipped election. The one real env lever is
  electing aiter's 1-STAGE fused asm kernel (stage1+silu/mul+stage2 in one launch) at the decode
  token tier. Also records a HARD deployment trap in aiter's AITER_CONFIG_* colon-merge.
keywords: [fused_moe, mxfp4, fp4_e2m1, e8m0, per_1x32, AITER_CONFIG_FMOE, tuned_fmoe, flydsl,
  1stage, xbf16, grouped GEMM, MoE, sglang, aiter, gfx950, MI355X, quark]
kernels: [mfma_moe1_silu_mul_afp4_wfp4_bf16, mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256,
  flydsl_moe2_afp4_wfp4_bf16_*, fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256]
platforms: [gfx950 · sglang 0.5.17 + aiter · ROCm 7.2 · quark mxfp4]
kernel_class: MoE grouped GEMM (fused dispatcher)
regime: decode-dominated serving (ISL 8192 / OSL 1024 / conc 64, TP4)
confidence: "★★"
confirms: 1
lifecycle: active
last_seen: 2026-08-21
---

# mxfp4 fused-MoE (sglang + aiter, gfx950): where the levers actually are

## Seam
`aiter.fused_moe:fused_moe` — ONE dispatcher call runs stage-1 (gate+up, mxfp4×mxfp4) → silu/mul →
stage-2 (down-proj) → top-k reduce. The profiled "head kernel"
(`mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_..._persist_cu256`) is **the emitted device name of a
flydsl recipe** (`flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2`) elected from
`aiter/configs/model_configs/<model>_fp4_tuned_fmoe.csv`. Do not go looking for a CK/asm source for
it and do not treat stage-1/stage-2 as separately bindable — they are two rows of ONE CSV row.

## Measured (Qwen3.5-397B-A17B-MXFP4, TP4/rank inter_dim 256, oracle = live capture, cold-flushed)
- **Tier-B stage-2 tile tune = EXHAUSTED.** Racing every flydsl stage-2 kernel with `tile_m ==
  block_m` (64 variants at the decode tier 64, 65 at prefill tier 16384; the rest are numerically
  invalid because `block_m` is the moe_sorting block and MUST equal the kernel's `tile_m`) against
  the shipped row: best alternative **1.044×** at M=64 and **1.006×** at M=16384, both inside the
  paired control-vs-control noise (±4% at M=64, ±0.5% at M=16384). The shipped model CSV is already
  the registry optimum. Do not spend a round re-tuning `kernelName2` on a model that ships a
  `*_fp4_tuned_fmoe.csv` covering the live `(inter_dim, E, topk)`.
- **The one env win = the 1-STAGE fuse at the decode tier.** Flipping the token-tier-64 row to
  `run_1stage=1, xbf16=1, flat=1, block_m=16,
  kernelName1=_ZN5aiter50fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256E, kernelName2=""` folds
  both stages into one launch (the silu/mul intermediate never round-trips to HBM): **1.036×
  (median) / 1.085× (min) at the dominant decode M=64 bucket**, neutral elsewhere ⇒ serving-weighted
  **1.029×**, Amdahl ceiling only **+0.30%** at a 10.51% head (**+0.18%** at the 6.33% live share) ⇒
  **stack-only, below the e2e noise band on its own.** ⚠ It is *not* free: `xbf16` keeps the
  intermediate in bf16, so `max_rel` vs the golden goes 0.0076 → 0.019 (tol 0.03) — needs the
  accuracy gate, and the sibling `..._16x128` kernel FAILS tol (0.146). At PREFILL the same fuse is
  a **30× regression and numerically wrong** — restrict the row swap to the decode tier.
- **CK is not a candidate on this image**: the `moe_ck2stages_gemm2_*FP4X2_FP4X2_B16` names exist in
  the CSVs but the instances are not compiled in (`[aiter WARNING] ck kernel not found` → silent
  fallback + garbage, `max_rel` 3.18). Probe with a correctness check, never by name presence.
- Roofline agreed with the outcome: stage-2 sits at ~85% of the HBM pin (attainable 1.053×).

## ⚠ Deployment trap: `AITER_CONFIG_FMOE` colon-merge MUTATES the shipped image CSV
`aiter.jit.core.AITER_CONFIG.update_config_files` merges the colon list, and on a **duplicate shape
key** it auto-resolves by keeping the lowest `us` **and writes the pruned result BACK into the source
config files**, then raises `RuntimeError: ... Please re-run`. Appending a small override CSV
alongside the shipped `model_configs/*_tuned_fmoe.csv` therefore silently edits the image (verified:
49 → 48 rows in the shipped file; restored with `git checkout`). Two further gotchas: a non-empty
`_tag` makes your row a NON-duplicate so it is kept but LOSES to the earlier file ("keep first" in
`get_cfg_2stages`) ⇒ **zero engagement**, and the merge result is cached under `/tmp/aiter_configs/`.
**Correct recipe:** ship a FULL edited COPY of the model CSV and build `AITER_CONFIG_FMOE` from the
default list with the original model CSV EXCLUDED and your copy appended — no duplicate keys, no
write-back, fully reversible. Verify engagement by grepping the one-shot
`[fused_moe] using 1stage|2stage (kernelName1=..., kernelName2=...)` line for the live token tier.

## Route
Live backend is already flydsl (aiter's SOTA mxfp4 MoE DSL), so Tier-A backend swap and Tier-B tile
tune are both dead ends here; the remaining headroom is Tier-C **fusion** work on the editable
sources that DO exist on the image: `aiter/ops/flydsl/moe_kernels.py` (JIT DSL, route=rewrite) and
`aiter/ops/triton/moe/moe_op_mxfp4.py` + `moe_op_mxfp4_silu_fused.py` (route=rewrite). The 1-stage
asm result above is the empirical proof that fusing the two stages is the right direction at decode
— and also bounds it (~3.6% of the whole dispatcher call).

caution: also verify the served decode batch actually lands on the tier you tuned — the lookup key is
`nextPow2(M)`, so a conc-64 steady state uses tier 64 while ramp batches fall on tier 32 and keep the
2-stage row.
