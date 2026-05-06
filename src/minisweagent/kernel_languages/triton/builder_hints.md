# Triton — HarnessBuilder hints

Language-specific idioms for producing a universal-contract harness
from a user's Triton test file.

## Universal contract the harness must satisfy

The generated `harness.py` must expose argparse with these flags:

    --correctness       run correctness against a reference
    --benchmark         time the kernel; print `GEAK_RESULT_LATENCY_MS=<float>`
    --full-benchmark    time + verify; print `GEAK_RESULT_SPEEDUP=<float>`
    --profile           run under the profiler with device_launch capture

## Triton-specific inputs

- The kernel is a `@triton.jit`-decorated function. Locate it by the
  decorator, not by the function name — users rename things.
- The Python-level launcher (typically named after the kernel,
  e.g. `def silu(x, y, out):` then `silu_kernel[(grid,)](...)`) is
  what the harness imports.
- When the test file imports the kernel via `from package.file import
  kernel`, preserve that module path in the harness so reruns stay
  stable.

## Reference selection

- Prefer `torch` reference implementations (correctness-only) over
  homemade NumPy ones — faster to write, well-vetted.
- Use `torch.allclose(candidate, reference, atol=1e-4, rtol=1e-4)`
  unless the user test specifies tighter tolerances.

## Timing loop

- Warmup 5 iterations before the timed run; measure 100 iterations
  and take the median (not mean).
- Synchronize (`torch.cuda.synchronize()`) before and after timing.
