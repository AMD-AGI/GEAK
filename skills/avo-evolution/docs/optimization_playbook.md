# Optimization Playbook (kernel-class menus + non-negotiable rules)

Distilled from CuTeGen (arXiv:2604.01489) Appendices A/B. Use this when
implementing an optimization in a variation step. Two parts: a **kernel-class
menu** (pick optimizations that match the workload) and **non-negotiable rules**
(how to fix bugs and optimize without cheating).

## Step 0 — classify the kernel

Before optimizing, infer the class and pick matching optimizations only:

- **(A) GEMM / matmul-like** (`C = A@B`, batched/grouped GEMM, linear layers, attention GEMMs)
- **(B) Activation / elementwise** (relu, gelu, sigmoid, softmax, add+bias, clamp, norm-ish)
- **(C) Reduction / other** (sum/mean/argmax, scan, fused/irregular)

Applying GEMM-only tricks (tensor cores, split-K, warp tiling) to an elementwise
kernel — or vice versa — wastes steps and regresses. Match the menu to the class.

## (A) GEMM-like — pick ONE per step

1. Threadblock (CTA) tiling for M/N/K
2. Warp-level tiling
3. Thread-level tiling / vectorized loads
4. Tensor-core MMA (only if dtype + layout allow)
5. Shared-memory bank-conflict reduction (swizzle / padded leading dim)
6. Pipelining / multi-stage loads (cp.async / async copy if supported)
7. Split-K (only if K is very large AND the K-reduction is the bottleneck)
8. Asynchronous loads/stores

Rules: reason about (M, N, K) and layouts/strides before choosing; pick the
strongest match for the actual shapes; only specialize to a shape/property
(e.g. symmetric, diagonal) when the reference code **explicitly** states it.

## (B) Activation / elementwise — pick ONE per step

1. Vectorized global load/store (float4 / half2) when contiguous + aligned
2. Grid-stride loop over `numel`
3. Fuse simple elementwise ops **already present** (do not invent new semantics)
4. Fast-math intrinsics (only for tanh/sigmoid/GELU-like; keep outputs stable)
5. Correct tail handling + bounds checks
6. Reduce register pressure / temporaries
7. Better block size (128/256/512) without changing semantics

Rules: treat as a 1D array of length `numel`; ensure coalesced access; avoid
shared memory unless there is clear data reuse (elementwise usually has none);
handle the tail when `numel` is not divisible by the vector width.

## (C) Reduction / other — pick ONE safe optimization

Vectorized load/store, grid-stride loop, better bounds handling, or a correct
block/warp reduction. Do not introduce GEMM-specific assumptions.

## Non-negotiable rules (read first)

**Do NOT "fix" or "optimize" by making the kernel simpler or by removing the
optimization structure.** Specifically:

- Do NOT replace tiling / partitioning / MMA / pipeline with a naive kernel "to
  pass correctness". Repair layouts, strides, partitions, synchronization,
  predication — keep the same algorithm and structure.
- Do NOT fall back to PyTorch ops or vendor libraries (cuBLAS / cuDNN / cuBLASLt)
  unless the reference's intent already was to call them. A trivial rewrite that
  scores ≈1.0x is rejected by the commit gate anyway (`min_commit_speedup`).
- Do NOT change precision or input/output shapes to make a number look better.
- Do NOT remove boundary checks / masks, or introduce race conditions, or change
  `@triton.jit` / kernel signatures.

## Debug discipline (diagnose → repair, smallest fix)

1. **Freeze intent**: state what the kernel computes and the invariants that must
   hold (tiling, pipeline stages, smem staging).
2. **Localize** without de-optimizing (add guarded prints/asserts/temporary sync;
   you may temporarily serialize a pipeline to `stages=1` for diagnosis, then
   restore it).
3. **Smallest structural fix**: wrong layout/stride, wrong tile/partition mapping,
   missing predicate for edge tiles, missing barrier/wait for async copy, wrong
   accumulator dtype / late cast. One change, re-test.
4. **OOM during correctness check is usually a harness/integration bug** (repeated
   `.contiguous()`, duplicate references, materialized broadcasts), not the kernel
   math — fix the integration path, do not simplify the kernel.

## Stage awareness (delayed profiling)

- **Early steps (structural)**: prioritize tiling/decomposition/data-movement.
  Do not micro-tune tile sizes from profiler output yet — premature profiling
  drives the search into poor local optima.
- **Later steps (profiling-guided)**: once structure is sound, use `profile.json`
  / `profile_kernel` for occupancy, tile sizes, bank conflicts, and fences.

The controller signals the current stage in the task body
(`Optimization stage: STRUCTURAL` vs `PROFILING-GUIDED`) — follow it.
