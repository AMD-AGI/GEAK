---
key: dense bf16 GEMM · gfx942 · sglang
type: lever
confidence: ★★★
effect: +2.23% e2e VERIFIED (1548.9→1583.5 tok/s, non-overlapping 5-repeat A/B); ~+6% cumulative w/ attn-triton
confirms: 2
last_seen: 2026-06-08
---
# Dense bf16 GEMM → tune aiter's per-shape DB (the #1 verified e2e win on this stack)
- lever: the live dense-GEMM path on sglang/gfx942 is aiter `tuned_gemm.py` (executing hipBLASLt
  `Cijk_*`), seam `aiter.tuned_gemm:gemm_a16w16`. Tune its per-shape DB — this is THE GEMM lever, and
  the strongest *transferred-to-e2e* win recorded.
- apply: capture real shapes `AITER_TUNE_GEMM=1` → `gradlib/gemm_tuner.py --indtype bf16 --mp <ngpus>`
  → deploy `AITER_CONFIG_GEMM_BF16=<tuned.csv>` (pure env, no package edit). FlyDSL races inside this DB
  (`libtype=flydsl`) and is auto-selected where it wins.
- verify: `AITER_LOG_TUNED_CONFIG=1` → count `is tuned on cu_num` hits (>0 = engaged; the winning run
  had 246 hits). The capture's correct `bias=False` + full shape coverage is what makes it both ENGAGE
  and WIN — a bias-mismatched/partial tune reads ~0/−0.6% (superseded).
- caution: NOT TunableOp / `HIPBLASLT_TUNING_FILE` — aiter bypasses the PyTorch/hipBLASLt C dispatch
  for its tuned shapes, so those hooks don't touch the live path.
- source: exp/e2e_*Qwen3.5-27B*/ 2026-06-08 (verified A/B, full recipe in `aiter_gemm_tuning.md`)
