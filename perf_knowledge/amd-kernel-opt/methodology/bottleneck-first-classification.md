---
type: Playbook
title: Bottleneck-first optimization methodology
description: Classify a kernel's dominant bottleneck before tuning; the bottleneck class dictates which lever to pull.
tags: [methodology, all-domains, all-bottlenecks]
timestamp: 2026-06-22T00:00:00Z
---

# Why
Across every campaign, the wins that landed came from tuning the *dominant* lever,
and the wasted attempts came from turning knobs blindly. The recurring failure mode:
spending effort on the kernel body when the op was launch-floor-bound, so no
GPU-internal change could ever move wall-clock (see
[body opts invisible when launch-bound](/anti-patterns/launch-bound-body-opts-invisible.md)).

# Step 1 — Measure GPU utilization first
If overall GPU utilization < 30%, the problem is **launch / host / sync**, not the
kernel internals. Fix that before micro-tuning. Tiny ops (runtime ≈ host dispatch,
~20–25 µs on this stack) are almost always here.

| Class | Signature | Lever | Patterns |
|---|---|---|---|
| Launch / occupancy | TFLOPS far below peak; improves with larger shapes; GPU util low | graph replay, persistent kernel, raise occupancy | [host-graph-replay](/patterns/host-graph-replay.md), [triton launcher](/patterns/triton-launcher-do-not-specialize.md), [ck V3→V1](/patterns/ck-pipeline-v1-occupancy.md) |
| Memory / HBM | HBM BW near peak, MFMA-issue ratio low (large K) | bigger tiles, async prefetch, L2/XCD locality, split-K, cut bytes | [L2 pid remap](/patterns/l2-locality-pid-remap.md), [int4 unpack](/patterns/int4-load-once-unpack.md) |
| Compute / MFMA | MFMA-issue ratio high, HBM has slack | wider-K MFMA atom, AccVGPR, raise occupancy | [launch-config autotune](/patterns/launch-config-autotune.md), [hoist math](/patterns/hoist-kloop-invariant-math.md) |
| Stall | both BW and MFMA low; high `s_barrier`/`s_waitcnt` | LDS ping-pong, scheduling hints, swizzle | LDS double-buffer + scheduling hints |

# Step 2 — Prefer host-side levers
On this stack the highest-ROI wins repeatedly came from **outside** the kernel body:
routing to a faster prebuilt kernel ([backend dispatch](/patterns/backend-dispatch-swap.md)),
replaying a captured graph, or changing a host-side `block_m` schedule
([routing sparsity](/patterns/block-m-routing-sparsity.md)). These need no rebuild,
are bit-exact, and transfer cleanly to e2e.

# Step 3 — Single-variable iteration
One hypothesis, one change, re-measure on representative shapes (small/medium/large).
Accept only if the mean gain exceeds run-to-run variation (>2%). Roll back cleanly;
never stack on a rolled-back change. Diagnose *why* before the next attempt.

# Step 4 — Guard against false wins
Correctness before performance. Verify any backend swap with a numerics gate (cos/SNR)
and a try/except fallback. Suspect a result that is larger than the change can
structurally produce — it is usually [benchmark over-fit](/anti-patterns/benchmark-overfit.md)
or a harness bug (the retracted 17.39× on chunk_scaled_dot_kkt was a harness grid bug).

# Citations
1. head_kernels/campaign20/FINAL_REPORT.md
2. KernelForge `system_prompt` (bottleneck-first catalog), e.g. KernelForge/results/*/run.log config
