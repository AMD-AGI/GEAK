---
type: Kernel Case Study
title: _topk_forward (Triton MoE router top-k)
description: A launch-overhead-bound Triton MoE router top-k whose only transferable win is host-side do_not_specialize on pointer args (~1.02x); a graph-replay run measured 1.90x but is benchmark-overfit.
tags: [domain-moe, bottleneck-launch, lever-host-side, gfx942]
speedup: 1.90x (graph-replay, overfit-flagged) / ~1.02x (transferable)
correctness: PASS — indices + bitmatrix bit-exact; values cosine ~1.0
kept: kept-deployed (do_not_specialize pointers); graph-replay flagged-risky
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
Triton `@triton.jit` MoE router top-k on MI300X (gfx942/CDNA3), Triton 3.6.0.
Harness pins launch params: `BLOCK_M=32, BLOCK_N=32, N_EXPTS_PAD=128, N_EXPTS_ACT=4,
n_expts_tot=128, APPLY_SOFTMAX=True`, `grid=(cdiv(n_rows,32),)`, `n_rows=B*1024`.

| case | n_rows | grid | latency (ms) |
|------|--------|------|--------------|
| c2   | 2048   | 64   | 0.0349 |
| c32  | 32768  | 1024 | 0.0344 |
| c64  | 65536  | 2048 | 0.0362 |

Latency is ~constant across a 32x work increase => **host-launch-bound**, not
compute-bound. rocprofv3: true GPU body ~13.7us (c64); event wall-clock ~36us; an
EMPTY-body kernel with the identical signature measures the same ~33us. Launch
floor (noop) ~21us; the extra ~12us is Triton's per-arg specialization / cache-key
construction over a large signature (many scalars + 3 tuple args).

# What changed (the win)
Two reports, reconciled honestly:

- **KernelForge (transferable lever):** `do_not_specialize` on the **pointer args
  only** (`X, PeerYvs, PeerYis, PeerBits`). Trims per-launch host specialization
  overhead — the only lever that touches a host-bound launch. Restricting to
  pointers is deliberate: `do_not_specialize` on scalar ints HURTS (esp. c32, -9%)
  by dropping divisibility hints. Saved as `v6_dns_pointers.py`.
- **campaign20 (headline):** CUDA/HIP-graph capture+replay of the same kernel,
  plus num_warps/num_stages, collapsing the per-call host/dispatch floor.

# Result
| source | c2 | c32 | c64 | geomean | note |
|--------|----|-----|-----|---------|------|
| KernelForge v6 (do_not_specialize pointers) | 1.050 | 0.954 | 1.045 | **1.016x** | transferable, verified over 4 rounds |
| campaign20 (graph-replay + warps/stages) | 1.932 | 1.972 | 1.798 | **1.90x** | benchmark-overfit risk |

Correctness PASS in both: indices + bitmatrix bit-exact, values cosine ~1.0.
The honest transferable gain is **~1.02x** — near the structural ceiling (~60% of
measured time is an irreducible launch floor, GPU compute hidden underneath). The
1.90x is real CUDA-event timing but comes from amortizing the host/dispatch floor
via graph replay; KernelForge flags this class as benchmark-overfit because it
collapses a per-call cost the real training loop still pays unless it also captures
a graph. Treat 1.90x as benchmark-specific, 1.02x as what transfers.

# What was tried and reverted
| attempt | result | why |
|---------|--------|-----|
| `@triton.autotune` over num_warps/num_stages | 0.74x REGRESSION | per-call host lookup dwarfs a ~0.035ms host-bound kernel; warps flat (0.034-0.039ms over 1..16) |
| vectorize bitmatrix build (4 word-iters -> 1) | ~1.00x flat | bits identical but not the bottleneck (GPU body hidden) |
| single-pass `tl.topk` over all 128 experts | ~1.00x flat | shortens GPU critical path, but GPU time is fully hidden under host floor |
| `do_not_specialize` (all args) + single-pass body | 0.89x (c64) REGRESSION | single `tl.topk` compiles to heavier selection net / higher regs; GPU time leaks above the host floor |
| `do_not_specialize` on scalar ints | -9% (c32) | drops divisibility hints |

Lesson: every GPU-body change on a launch-bound kernel is invisible or harmful.

# Patterns
- [Triton launcher do_not_specialize](/patterns/triton-launcher-do-not-specialize.md)
- [Host-side graph replay](/patterns/host-graph-replay.md)
- [Launch-bound body opts are invisible](/anti-patterns/launch-bound-body-opts-invisible.md)
- [Benchmark overfit](/anti-patterns/benchmark-overfit.md)

# Citations
1. KernelForge/results/_topk_forward/tasks/cli/7472ba62-29b9-4a4c-80a3-051b26b4c8ed/workspace/optimization_report.md
2. head_kernels/campaign20/FINAL_REPORT.md
