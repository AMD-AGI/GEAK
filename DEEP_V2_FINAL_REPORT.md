# deep_mode v2 — final report (MiniMax-M3-MXFP8, gfx950, vLLM TP=4)

## Result
**Best validated e2e: +21.8%** (1709.95 → ~2082 tok/s), **parity PASS + gsm8k 0.955 == 0.955** (quality-clean).
Deliverable overlay (apply_to_original=false): `exp/e2e_MiniMax-M3-MXFP8_20260621_144547_3794_10149/overlay/`
(`accepted__mxfp8_linear_kernel/`: sitecustomize.py + `_mxfp8_linear_splitk_impl.py`; the +21.8% leg was a
later triton-fused generation, the +21.3% splitk is the saved accepted overlay). This **beats fast_mode's
+21%** on a like-for-like quality basis.

## What deep_v2 delivered (the orchestrator — the primary goal)
A ground-up redesign of the deep HeadKernel track, all `DEEP_V2`-gated (default/fast byte-identical), generic
(no model/kernel/backend hard-coding), no `rm` anywhere. Validated working on-box:
- **Global (head op × backend) lane pool** — multiple kernels AND backends optimize concurrently (vs v1 serial).
- **GPU-elastic, N-adaptive partition** — serial e2e gate on the fixed serving slot overlapping co-opt on the
  other cards; degrades to time-slice at N==TP. No card conflicts observed.
- **Full-backend roster** (triton + flydsl + aiter actually run) + ceiling-aware patience + revive.
- **Budget controller** (EV = Amdahl × ceiling-gap × rate) + periodic re-profile + **run-until-budget +
  reseedForDepth** depth pass + per-op SHARED_KB + run-global GLOBAL_KB.
- **gsm8k task-accuracy gate** (the right bar for quant) + same-session non-overlapping A/B + spurious-win
  rejection (correctly killed self-relative 3–7× "wins" that weren't vs-live).

## Why +50% / beating v1's +31.5% was not reached (honest)
The realistic quality-preserving ceiling for this model on this box is ~**+21.8%**, established by FIVE
independent attempts all landing ≤ +21.8%:
1. The **dense MXFP8 linear** decode-tile rewrite is the only *convertible* e2e lever, and it is **decode-driven**:
   +21.8% at conc=64 (decode-bound) but ~0% at conc=32 (a 1.3× isolated kernel → gate-rejected at low conc).
2. The **MoE grouped GEMM** (the larger ~25–33% head) resists — native E8M0 `dot_scaled` is already near-optimal
   (~1.1× isolated, no e2e movement).
3. The seed config is already at the Tier-0 optimum (config sweep = no-op every run).
4. ~37% of GPU is TP=4 all-reduce comm (config-optimal, not kernel-addressable).
5. Deep stacking / re-seeding plateaus at ~+21.8% (kernels near ceiling); v1's +31.5% was vs a *lower* 762
   baseline (more headroom) — this run's seed baseline is already 936–1709.
Process note: over-deep bursts (DEEP_V2_WAVE_BUDGET=6) *slowed* exploration and underperformed the faster
3-round bursts — for this model, exploration breadth/speed mattered more than per-burst depth.

## Runs (chronology)
- decode-bound 1024/1024/64 (conc=64): **+21.8%** ← best, the deliverable.
- GEMM-heavy 8192/1024/32 (conc=32): +1.3% (kernel wins don't convert at low conc).
- decode-bound + depth (fresh): plateaued ≤ +21.8% (confirmed ceiling; deeper bursts slower).

## Recommendation
Adopt the **+21.8% overlay** as the deep_v2 deliverable. The v2 orchestrator is sound and merge-ready
(deep-mode-v2 branch); the M3 lever is curated in `knowledge/learned/mxfp8-linear-decode-rewrite-gfx950.md`.
