# HIP — optimizer hints (worker-side)

Concrete language-specific patterns the OptimizationAgent may try.

## Memory-bound kernels

- **Coalesced global access**: adjacent threads within a warp should
  hit adjacent addresses. Stride-1 access patterns are preferable
  to gather-style.
- **Vectorised loads**: `float4` / `int4` loads are free on GFX9/10;
  prefer them when 16-byte alignment is known.
- **LDS staging**: stage tiles through `__shared__` memory only when
  they're reused multiple times per thread block.

## Compute-bound kernels

- **MFMA-friendly tile shapes**: reshape inner loops so matmul
  operands map onto `16x16x16` / `32x32x8` MFMA tiles.
- **Branch simplification**: HIP GPUs execute in wavefronts (64
  lanes) — divergent branches within a wavefront stall the other
  lanes. Use `?:` / `hmul` / `hadd` intrinsics instead.
- **Loop unrolling**: `#pragma unroll N` in hot inner loops when
  the compiler hasn't done it already.

## Latency-bound kernels

- **Wavefront-cooperative**: use `__ballot_sync` / `__shfl_sync` for
  wave-wide reductions instead of per-thread memory round-trips.
- **Size-specialisation**: emit separate kernels for small / medium /
  large regimes and dispatch based on input shape.
- **Persistent kernels**: amortise launch overhead by having each
  block process multiple tiles.

## Search-like workloads

- **Branchless binary search**: use `__builtin_expect` hints +
  conditional moves instead of `if/else` chains.
- **Coarse-index narrowing**: precompute a coarse index to prune
  the search space, then finish with a tight linear scan.
- **Do NOT default to bandwidth maximisation** for these — the
  critical path is latency, not throughput.

## LDS-bound kernels

- **Bank-conflict audit**: check stride-1 patterns that collide on
  the same bank; fix by swizzling indices.
- **Register-promote transients**: data used only within one
  inner-loop iteration belongs in registers, not LDS.
