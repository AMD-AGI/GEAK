---
type: Kernel Case Study
title: write_req_to_token_pool_triton (SGLang KV pool index write)
description: A launch-overhead-bound tiny Triton op whose 2.05x geomean win came entirely from host-side dispatch-path cuts (do_not_specialize + thin cached launcher + skip launch_metadata), not GPU-internal levers.
tags: [domain-kv-cache, bottleneck-launch, lever-host-side, gfx942]
speedup: "2.05x geomean"
correctness: PASS — exact integer match vs golden + torch reference; fast path re-verified with fresh tensors each step
kept: kept-deployed
timestamp: 2026-06-22T00:00:00Z
---

# Baseline
- Original SGLang Triton kernel writing KV pool indices: grid=(B,), one program per request.
- Target: AMD Instinct MI300X (gfx942). Workload: seq_len=1024, pre_len=16, extend_len=1008, B in {2,32,64}.
- Measured v0 latency:

| case | latency (ms) |
|------|--------------|
| c2   | 0.0377 |
| c32  | 0.0366 |
| c64  | 0.0352 |

- Latencies flat across B → launch/latency-bound; tiny work per program.
- rocprofv3: true GPU kernel duration ~5us (min 1.9, max 8.2), but an EMPTY kernel measures the same ~37us in harness → score is essentially 100% host-side Triton launch overhead; GPU work is fully hidden.

# What changed (the win)
Three cumulative host-side levers (v5 left in place):
1. `@triton.jit(do_not_specialize=[...])` on all non-constexpr runtime args — skips per-launch alignment/divisibility specialization analysis (~37us → ~23us).
2. Thin cached launcher — after the first compiling call, launch the already-compiled kernel directly, skipping binder + cache-key compute + cache lookup. Cache keyed on (grid, constexpr) only, NOT tensor identity; args re-read every call. Requires do_not_specialize so bypassing the binder is alignment-safe (~23us → ~17us).
3. Skip `launch_metadata` when no launch hooks are registered (common case incl. training); passing None avoids wasted host work (~17us → ~15us fast path).
- Constexpr-guard + try/except fallback to the safe JIT path guarantees correctness.

# Result
Clean separate-process benchmark (median of stable runs):

| case | v0 baseline (ms) | v5 best (ms) | speedup |
|------|------------------|--------------|---------|
| c2   | ~0.0383 | ~0.0187 | ~2.05x |
| c32  | ~0.0372 | ~0.0178 | ~2.09x |
| c64  | ~0.0362 | ~0.0177 | ~2.05x |

- Overall ~2.05x geomean. Best version v5, left in place at source_file_path.
- Correctness: exact integer match vs golden and torch reference on all cases. Bit-exact (integer indices). Fast path independently re-verified with FRESH tensors each step (real-training simulation) — ALL PASS.
- Transfers 1:1 to real training: no tensor-identity caching; only stable (grid, constexpr) compiled artifacts cached.

# What was tried and reverted
| attempt | lever | result |
|---------|-------|--------|
| v1 vectorize serial cumsum | replace O(pid) loop-carried serial accumulation with block load + tl.sum tree reduction | ~1.0x (slight regression, noise). Cumsum is NOT the bottleneck. REVERTED |
| v2 BLOCK_SIZE 512→1024 | single-pass extend write | ~1.0x (slight regression). No effect — launch-bound, not compute-bound. REVERTED |

Both GPU-internal levers did nothing: the score is pure host-side launch overhead.

# Patterns
- [Host graph replay](/patterns/host-graph-replay.md)
- [Triton launcher do_not_specialize](/patterns/triton-launcher-do-not-specialize.md)
- [Anti-pattern: launch-bound body opts invisible](/anti-patterns/launch-bound-body-opts-invisible.md)

# Citations
1. KernelForge/results/write_req_to_token_pool_triton/tasks/cli/86053804-a4c0-49af-8134-2252a4cd9ea1/workspace/optimization_report.md
