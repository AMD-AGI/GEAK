---
key: dense GEMM · gfx950 · vLLM MXFP8 E8M0 decode-bound
type: lever
confidence: ★★★
effect: dense MXFP8 linear Triton split-K/fused decode-tile rewrite → +21.8% e2e (1709→2082) at conc=64; quality-clean
confirms: 2
last_seen: 2026-06-22
---
# MXFP8 dense-linear decode-tile rewrite is the main e2e lever (decode-bound serving)
- lever: rewrite the dense MXFP8 linear kernel (qkv / o_proj / up-gate / down — vLLM `rocm_native` `tl.dot_scaled`)
  as a **split-K + fused decode-tile** Triton kernel. This is the dominant *convertible* e2e lever for
  MiniMax-M3-MXFP8: **+21.8% e2e** (1709→2082 tok/s), parity PASS, gsm8k 0.955==0.955.
- apply: overlay via sitecustomize.py (apply_to_original=false); engages live on all 4 TP workers, cudagraph-safe,
  equal mem-fraction. Decode tpot −18%.
- verify: same-session non-overlapping A/B (cand_min > ref_max) + greedy parity + gsm8k subset == baseline.
- dead-end: **the win is DECODE-driven → it only converts at high concurrency.** At conc=64 (decode-bound)
  it gives +21.8%; at conc=32 (prefill/GEMM-heavy 8192/1024/32) the *same* 1.3× isolated kernel converts to
  ~0% e2e — gate rejects it. Pick a decode-heavy workload to measure/realize this lever.
- dead-end: **MoE grouped MXFP8 GEMM resists** — native E8M0 `dot_scaled` is already near-optimal (~1.1× isolated
  ceiling, no e2e movement). Don't expect a big grouped-GEMM win on an already-tuned seed.
- dead-end: **+31.5%/+50% are not reachable** on this run's already-strong seed baseline (1709-936); v1's +31.5%
  was vs a lower 762 baseline (more headroom). Deep stacking / re-seeding plateaus at ~+21.8% (kernels near
  ceiling); over-deep bursts (6 rounds) *slow* exploration and underperform faster 3-round bursts.
- dead-end: cross-backend (flydsl/aiter) self-reported 3-7× are SELF-RELATIVE-to-first-port inflation, not
  vs-live — the e2e gate correctly rejects them; rank only by vs-live / e2e.
- source: GEAK/worktree/deep/exp/e2e_MiniMax-M3-MXFP8_20260621_144547_3794_10149/overlay/accepted__mxfp8_linear_kernel/integrate_result.json
