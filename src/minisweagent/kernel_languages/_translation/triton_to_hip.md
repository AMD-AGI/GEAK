# Triton → HIP translation hints

Pair-specific guidance for rewriting a Triton kernel as a HIP C++
kernel. Appended to the TranslationAgent prompt when the source is
Triton and the target is HIP.

## Structural mapping

- A `@triton.jit` function becomes a `__global__ void` kernel.
- `tl.program_id(axis=0)` -> `blockIdx.x`; `axis=1` -> `blockIdx.y`.
- `tl.arange(0, BLOCK)` -> `threadIdx.x` (stride-1 within block).
- `tl.load(ptr + offs, mask=mask)` -> bounds-checked `ptr[idx]`
  with an `if (idx < n)` guard.
- `tl.store(ptr + offs, val, mask=mask)` -> same, for writes.

## Compute primitives

- `tl.dot(a, b)` -> either `__builtin_amdgcn_mfma_f32_16x16x16f16`
  (when shapes match MFMA tiles) or an explicit multiply-add loop.
- `tl.sum(x, axis=...)` -> wave-level reduction via
  `__shfl_xor_sync` on GFX10+, or LDS-staged reduction otherwise.
- Triton implicitly autotunes block size via `@triton.autotune`;
  HIP requires you to pick a concrete `dim3 block(256)` up front.

## Launch wiring

- Triton: `kernel[grid](*args, BLOCK_SIZE=...)` — Python-level launch.
- HIP: `hipLaunchKernelGGL(kernel, grid, block, 0, 0, *args);` from
  a C++ launcher function (which you must emit alongside the kernel).

## Wrapper shape for the evaluation harness

The generated HIP file typically needs a `pybind11` / `torch.utils
.cpp_extension.load_inline` wrapper so the harness can invoke it
from Python. Emit both the kernel and the wrapper in one file.

## Common gotchas

- Triton masks allow out-of-bounds reads with `other=0.0`; HIP
  requires explicit `if (idx < n)`.
- Triton's `num_warps` is a compile-time hint that controls thread
  count per block; in HIP you set it directly via `block.x`.
- `tl.constexpr` values can become template parameters or
  `__launch_bounds__` hints; don't hardcode the default block size
  into the kernel body.
