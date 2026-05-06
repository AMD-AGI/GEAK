# HIP → Triton translation hints

Pair-specific guidance for rewriting a HIP C++ kernel as a Triton
(Python) kernel.

## Structural mapping

- `__global__ void kernel(...)` -> `@triton.jit def kernel(...)`.
- `blockIdx.x` -> `tl.program_id(axis=0)`.
- `threadIdx.x` + `blockDim.x` -> `tl.arange(0, BLOCK_SIZE)` combined
  with a pid offset.
- `__shared__ T buf[N]` -> Triton keeps tile data in registers by
  default; explicit LDS staging is rarely needed.
- `__syncthreads()` has no direct analogue in Triton; the JIT handles
  tile-level synchronisation implicitly.

## Compute primitives

- MFMA intrinsics (`__builtin_amdgcn_mfma_*`) -> `tl.dot(a, b, acc)`.
  Triton chooses the right MFMA shape from the operand shapes.
- Wave shuffles (`__shfl_*`) -> typically unnecessary; Triton's
  block-scoped operations replace most shuffle use cases.
- Masked loads in HIP (`if (idx < n) { ... }`) -> `tl.load(ptr + offs,
  mask=mask, other=0.0)` in Triton.

## Launch wiring

- HIP: `hipLaunchKernelGGL(kernel, grid, block, 0, 0, *args);` with
  explicit `dim3` values.
- Triton: `kernel[grid](*args, BLOCK_SIZE=...)` where grid is a
  lambda / tuple; block size is a compile-time constexpr.

## Python wrapper

- Triton kernels are invoked from Python — emit a Python launcher
  function alongside the `@triton.jit` kernel so the harness can
  import it directly (no C++ build step needed).

## Common gotchas

- HIP kernels with explicit block geometry often need reshaping to
  fit Triton's tile-first mental model.
- Wavefront-cooperative patterns (binary search, ballot) frequently
  rewrite as tile-wide Triton operations; don't port the shuffle
  primitives literally.
- Triton doesn't have a direct equivalent of `__restrict__`; trust
  the compiler's pointer analysis.
