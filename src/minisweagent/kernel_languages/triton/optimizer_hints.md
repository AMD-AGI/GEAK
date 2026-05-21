# Triton — optimizer hints (worker-side)

Concrete language-specific patterns the OptimizationAgent may try
when the profile supports them. Appended to the task body by
``compose_task_body`` when available.

## Memory-bound kernels

- **Reduce HBM traffic**: fold loads across reduction steps, reuse
  values across inner-loop iterations, prefer `tl.dot` accumulation
  over explicit multiply-add loops.
- **Vectorised loads**: ensure load/store widths match the pointer's
  natural alignment. A 128-bit load is worth more than four 32-bit
  loads even when the latter looks simpler.
- **Shared-memory (LDS) staging**: stage tiles through LDS only when
  they're reused; single-use tiles should stay in registers.

## Compute-bound kernels

- **`tl.dot` favours MFMA lanes**: reshape inner loops so the matmul
  shapes map onto `16x16` / `32x32` MFMA tiles. 8-wide K-dim loops
  are usually worse than 16-wide.
- **Approximate when safe**: `tl.exp2` + log2 ratio is cheaper than
  `tl.exp` on some generations; verify correctness tolerance first.
- **Branch flattening**: `tl.where` is cheaper than a Python-level
  conditional tile.

## Latency-bound kernels

- **Persistent-kernel pattern**: amortise launch overhead by doing
  more work per program. One 2-ms kernel beats ten 0.3-ms kernels.
- **Shape specialisation**: when callers span very different regimes
  (tiny / medium / large), emit separate kernels and dispatch.

## LDS-bound kernels

- **Bank-conflict audit**: check for repeat-patterns of stride-1
  access that collide on the same bank. Often fixed by swizzling
  tile indices.
- **Register-promote transients**: temps that only live within a
  single inner-loop iteration belong in registers, not LDS.
