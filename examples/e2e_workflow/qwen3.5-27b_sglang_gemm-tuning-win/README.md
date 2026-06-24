# Example run #2 — Qwen3.5-27B, sglang, ISL/OSL=1024, conc=64 (the GEMM-tuning win)

Second `team_workflow_e2e` run on `Qwen-Qwen3.5-27B`, same workload as the
[run #1 example](../qwen3.5-27b_sglang_isl1024_osl1024_conc64/). The difference: this run carries the
**bias-correct aiter GEMM tuning fix**, so dense-GEMM tuning **engages and wins** instead of being a
no-op.

## Result (headline)
- Baseline **1492.7** tok/s (TP=1).
- `--attention-backend triton` (accepted) → reference ~1548.9.
- **aiter per-shape GEMM DB tune** (live `AITER_TUNE_GEMM=1` capture → gradlib → `AITER_CONFIG_GEMM_BF16`):
  **+2.23% e2e** (ref 1548.9 → cand **1583.5**, non-overlapping 5-repeat A/B, **246 `is tuned on cu_num`
  engagement hits**) → **accepted**. **Cumulative ≈ +6%** vs baseline.
- Single editable kernels were optimized too (chunk_fwd_kernel_o **1.228×**, causal_conv1d 1.066×,
  chunk_gated_delta_h 1.004×, an authored Triton GEMM 1.466× isolated).

## Honest caveat
The run was **stopped during the Milestone stage**: the `recompute_w_u` recursive optimization wedged
(~3h) and triggered a host fork-storm (hundreds of `rocm_agent_enumerator` procs — see finding #8 in
`workflow_e2e_team/knowledge/gemm_tuning/aiter_gemm_tuning.md`), which blocked the milestone's parallel barrier. So
the single-kernel cluster's **e2e stacking number was not obtained**, and there is no auto-generated
Finalize/Validate (`final_report.md` here was compiled by hand from the run's artifacts).

## Why this example matters
It's the concrete proof that **GEMM tuning is a real lever on this model/stack (+2.23%)** once the tune
input comes from a live bias-correct capture — correcting the earlier "GEMM tune nets ~0" conclusion.
See `final_report.md` for the per-stage 阶段树, the artifact 产物树, and the full A/B numbers.

(The tuned CSV itself is intentionally **not** shipped — solution indices are ROCm/aiter-build-specific;
regenerate per stack via the recipe in `workflow_e2e_team/knowledge/gemm_tuning/aiter_gemm_tuning.md`.)
