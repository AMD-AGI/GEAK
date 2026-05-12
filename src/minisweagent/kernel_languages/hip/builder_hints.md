# HIP — HarnessBuilder hints

Language-specific idioms for producing a universal-contract harness
from a user's HIP test file.

## Universal contract the harness must satisfy

Same as all languages: argparse with `--correctness`, `--benchmark`,
`--full-benchmark`, `--profile`; emit `GEAK_RESULT_LATENCY_MS` and
`GEAK_RESULT_SPEEDUP` markers on stdout.

## HIP-specific inputs

HIP kernels come in one of three common shapes:

1. **Pybind11 / torch-extension wrapper**. Python-callable after a
   `torch.utils.cpp_extension.load_inline` or `load(...)` call. The
   harness imports the compiled module and invokes the Python-level
   function.
2. **Standalone `make` + `./bench`**. The harness shells out to the
   compiled binary and parses its stdout.
3. **Raw `hipcc` + host-side launcher**. Same shape as (2) but
   compiled per-invocation.

The builder picks the shape based on evidence in the kernel file and
the user's test file:

- `torch.utils.cpp_extension` / `pybind11::module` visible anywhere in
  the kernel file -> shape 1.
- `Makefile` at `repo_root` + existing `./bench` binary -> shape 2.
- Raw `__global__ void ...` without a Python binding -> shape 3.

## Reference selection

- For shape 1 (pybind11 wrapper), the same wrapper exposes a
  reference implementation (usually a PyTorch fallback); use it.
- For shapes 2 and 3, the user test file contains either a
  CPU reference or a separate validation run; preserve that path.

## Timing loop

- Warmup 5 iterations; measure 100 and take the median (not mean).
- `hipDeviceSynchronize()` before/after each measurement.
