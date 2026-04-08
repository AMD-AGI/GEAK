#!/usr/bin/env python3
"""GEAK Test Harness - Softmax PoC

Loads baseline and optimized CK softmax kernel .so files via ctypes,
verifies the optimized kernel against the baseline (ground truth), and
reports bandwidth and speedup.

The .so files (libbaseline.so, liboptimized.so) are auto-discovered in the
same directory as this script.

Modes:
    --correctness     Verify optimized against baseline on HARNESS_SHAPES
    --profile         Run optimized kernel once per PROFILE_SHAPE (for rocprofv3)
    --benchmark       Benchmark both kernels on HARNESS_SHAPES, report speedup
    --full-benchmark  Benchmark both kernels on ALL_SHAPES, report speedup

Usage:
    python test_harness.py --correctness
    python test_harness.py --benchmark
    python test_harness.py --benchmark --iterations 50
    python test_harness.py --full-benchmark
    python test_harness.py --profile
"""

import argparse
import ctypes
import os
import statistics
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_SO = SCRIPT_DIR / "libbaseline.so"
OPTIMIZED_SO = SCRIPT_DIR / "liboptimized.so"

# -- Shape lists (sorted by element count) ------------------------------------
# 3D softmax shapes: (Batch, SeqLen, Hidden) with reduction on last dim.

ALL_SHAPES: list[list[int]] = [
    [1, 8, 256],
    [1, 8, 1024],
    [1, 8, 4096],
    [1, 32, 512],
    [4, 16, 1024],
    [1, 32, 4096],
    [4, 32, 1024],
    [2, 64, 1024],
    [4, 32, 2048],
    [8, 32, 1024],
    [4, 64, 2048],
    [8, 64, 1024],
    [8, 128, 1024],
    [8, 128, 2048],
    [16, 64, 2048],
    [8, 128, 4096],
    [16, 128, 2048],
    [32, 64, 2048],
    [16, 128, 4096],
    [32, 128, 2048],
    [32, 64, 4096],
    [32, 128, 4096],
    [64, 128, 2048],
    [64, 128, 4096],
    [128, 128, 4096],
]

HARNESS_SHAPES: list[list[int]] = ALL_SHAPES

PROFILE_SHAPES: list[list[int]] = [ALL_SHAPES[i] for i in range(0, len(ALL_SHAPES), len(ALL_SHAPES) // 5)][:5]


# -- Kernel loading via ctypes ------------------------------------------------


def load_kernel(path: str):
    """Load a kernel .so and set up the run_kernel function signature."""
    lib = ctypes.CDLL(path)
    lib.run_kernel.restype = ctypes.c_float
    lib.run_kernel.argtypes = [
        ctypes.c_void_p,  # in_dev
        ctypes.c_void_p,  # out_dev
        ctypes.POINTER(ctypes.c_int64),  # lengths
        ctypes.POINTER(ctypes.c_int64),  # strides
        ctypes.c_int,  # ndims
        ctypes.POINTER(ctypes.c_int),  # reduce_dims
        ctypes.c_int,  # n_reduce_dims
        ctypes.c_double,  # alpha
        ctypes.c_double,  # beta
        ctypes.c_bool,  # time_kernel
        ctypes.c_int,  # warmup
        ctypes.c_int,  # nrepeat
    ]
    return lib


def call_kernel(
    lib,
    x: torch.Tensor,
    y: torch.Tensor,
    reduce_dims: list[int],
    alpha: float = 1.0,
    beta: float = 0.0,
    time_kernel: bool = False,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Call run_kernel from a loaded .so. Returns time in ms (0 if not timing)."""
    ndims = x.ndim
    lengths = (ctypes.c_int64 * ndims)(*x.shape)
    strides = (ctypes.c_int64 * ndims)(*x.stride())
    rdims = (ctypes.c_int * len(reduce_dims))(*reduce_dims)

    torch.cuda.synchronize()
    ms = lib.run_kernel(
        x.data_ptr(),
        y.data_ptr(),
        lengths,
        strides,
        ndims,
        rdims,
        len(reduce_dims),
        alpha,
        beta,
        time_kernel,
        warmup,
        nrepeat,
    )
    return ms


# -- Test logic ---------------------------------------------------------------


def run_kernel_output(
    lib, x: torch.Tensor, reduce_dim: int
) -> torch.Tensor | None:
    """Run kernel on input x. Returns output tensor, or None if unsupported."""
    y = torch.empty_like(x)
    ms = call_kernel(lib, x, y, [reduce_dim], alpha=1.0, beta=0.0, time_kernel=False)
    if ms < 0:
        return None
    return y


def benchmark_kernel(
    lib,
    shape: list[int],
    reduce_dim: int,
    warmup: int = 5,
    nrepeat: int = 20,
) -> float:
    """Time the kernel. Returns average time in ms, or -1 if unsupported."""
    x = torch.randn(shape, dtype=torch.float16, device="cpu").to("cuda") # IMPORTANT: Use device='cpu' then .to('cuda') to avoid polluting the profiler trace with RNG/memset kernels.
    y = torch.empty_like(x)

    ms = call_kernel(
        lib,
        x,
        y,
        [reduce_dim],
        alpha=1.0,
        beta=0.0,
        time_kernel=True,
        warmup=warmup,
        nrepeat=nrepeat,
    )
    return ms


def compute_bandwidth(shape: list[int], time_ms: float) -> float:
    """Compute effective bandwidth in GB/s (read input + write output, FP16)."""
    numel = 1
    for s in shape:
        numel *= s
    bytes_moved = numel * 2 * 2  # 2 bytes per FP16, read + write
    return bytes_moved / time_ms / 1e6  # bytes / ms = KB/s, / 1e6 = GB/s


# -- GPU warmup ---------------------------------------------------------------

_GPU_WARMUP_ROUNDS = 3


def _gpu_warmup(base_lib, opt_lib, shapes, reduce_dim_arg: int):
    """Run the largest shape untimed on both kernels to bring GPU to peak clock.

    MI300X (and other GPUs with DVFS) idle at low frequency and need sustained
    work to ramp up.  CK's 5 warmup iterations inside the timed region only
    produce microseconds of work for small shapes — not enough.  This function
    forces a ramp-up *before* any timed work begins.
    """
    warmup_shape = shapes[-1]  # shapes are sorted small→large
    reduce_dim = reduce_dim_arg if reduce_dim_arg >= 0 else len(warmup_shape) - 1
    x = torch.randn(warmup_shape, dtype=torch.float16, device="cpu").to("cuda")
    y = torch.empty_like(x)
    for _ in range(_GPU_WARMUP_ROUNDS):
        call_kernel(base_lib, x, y, [reduce_dim], time_kernel=False)
        call_kernel(opt_lib, x, y, [reduce_dim], time_kernel=False)
    torch.cuda.synchronize()


# -- Mode implementations ----------------------------------------------------


def mode_correctness(base_lib, opt_lib, shapes: list[list[int]], reduce_dim_arg: int) -> bool:
    """Verify optimized kernel against baseline on each shape. Returns True if all pass."""
    torch.manual_seed(42)
    any_failed = False

    for shape in shapes:
        reduce_dim = reduce_dim_arg if reduce_dim_arg >= 0 else len(shape) - 1
        x = torch.randn(shape, dtype=torch.float16, device="cpu").to("cuda") # IMPORTANT: Use device='cpu' then .to('cuda') to avoid polluting the profiler trace with RNG/memset kernels.

        y_base = run_kernel_output(base_lib, x, reduce_dim)
        if y_base is None:
            print(f"  SKIP  {shape}  (baseline: UNSUPPORTED)")
            continue

        y_opt = run_kernel_output(opt_lib, x, reduce_dim)
        if y_opt is None:
            print(f"  SKIP  {shape}  (optimized: UNSUPPORTED)")
            continue

        try:
            torch.testing.assert_close(y_opt, y_base, atol=1e-3, rtol=1e-3)
            max_err = (y_opt.float() - y_base.float()).abs().max().item()
            print(f"  PASS  {shape}  max_err={max_err:.2e}")
        except AssertionError as e:
            max_err = (y_opt.float() - y_base.float()).abs().max().item()
            print(f"  FAIL  {shape}  max_err={max_err:.2e}")
            print(f"        {e}")
            any_failed = True

    return not any_failed


def mode_profile(opt_lib, shapes: list[list[int]], reduce_dim_arg: int) -> None:
    """Run optimized kernel once per shape for profiler capture."""
    for shape in shapes:
        reduce_dim = reduce_dim_arg if reduce_dim_arg >= 0 else len(shape) - 1
        x = torch.randn(shape, dtype=torch.float16, device="cpu").to("cuda") # IMPORTANT: Use device='cpu' then .to('cuda') to avoid polluting the profiler trace with RNG/memset kernels.
        y = torch.empty_like(x)
        ms = call_kernel(opt_lib, x, y, [reduce_dim], alpha=1.0, beta=0.0, time_kernel=False)
        if ms < 0:
            print(f"  SKIP  {shape}  (UNSUPPORTED)")


def mode_benchmark(
    base_lib, opt_lib, shapes: list[list[int]], reduce_dim_arg: int,
    warmup: int, nrepeat: int, base_label: str, opt_label: str,
) -> bool:
    """Benchmark both kernels on shapes. Returns True if all shapes ran."""
    _gpu_warmup(base_lib, opt_lib, shapes, reduce_dim_arg)

    opt_times_us: list[float] = []
    base_times_us: list[float] = []
    speedups: list[float] = []

    print(f"{'Shape':>30s}  {'Baseline ms':>11s}  {'Optimized ms':>12s}  {'BW GB/s':>8s}  {'Speedup':>7s}")
    print("-" * 80)

    for shape in shapes:
        reduce_dim = reduce_dim_arg if reduce_dim_arg >= 0 else len(shape) - 1

        ms_base = benchmark_kernel(base_lib, shape, reduce_dim, warmup=warmup, nrepeat=nrepeat)
        if ms_base < 0:
            print(f"{str(shape):>30s}  {'UNSUPPORTED':>11s}")
            continue

        ms_opt = benchmark_kernel(opt_lib, shape, reduce_dim, warmup=warmup, nrepeat=nrepeat)
        if ms_opt < 0:
            print(f"{str(shape):>30s}  {ms_base:11.4f}  {'UNSUPPORTED':>12s}")
            continue

        bw_opt = compute_bandwidth(shape, ms_opt)
        speedup = ms_base / ms_opt if ms_opt > 0 else float("inf")

        base_times_us.append(ms_base * 1000)
        opt_times_us.append(ms_opt * 1000)
        speedups.append(speedup)

        print(f"{str(shape):>30s}  {ms_base:11.4f}  {ms_opt:12.4f}  {bw_opt:8.1f}  {speedup:6.2f}x")

    if not opt_times_us:
        print("\nNo shapes were supported by both kernels.")
        return False

    median_opt = statistics.median(opt_times_us)
    median_base = statistics.median(base_times_us)
    mean_speedup = statistics.mean(speedups)
    median_speedup = median_base / median_opt if median_opt > 0 else float("inf")

    print()
    print(f"Shapes benchmarked: {len(opt_times_us)}")
    print(f"median_wall_time_us: {median_opt:.2f}")
    print(f"median_baseline_us:  {median_base:.2f}")
    print(f"median_speedup:      {median_speedup:.4f}")
    print(f"mean_speedup:        {mean_speedup:.4f}")

    return True


# -- Main ---------------------------------------------------------------------


def get_iterations(args_iterations: int | None) -> int:
    """Resolve iteration count: CLI flag -> env var -> default 20."""
    if args_iterations is not None:
        return args_iterations
    env_val = os.environ.get("GEAK_BENCHMARK_ITERATIONS")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return 20


def main():
    parser = argparse.ArgumentParser(description="GEAK Softmax Test Harness")

    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--correctness", action="store_true", help="Verify optimized against baseline")
    modes.add_argument("--profile", action="store_true", help="Run optimized kernel once per shape (for rocprofv3)")
    modes.add_argument("--benchmark", action="store_true", help="Benchmark on HARNESS_SHAPES")
    modes.add_argument("--full-benchmark", action="store_true", help="Benchmark on ALL_SHAPES")

    parser.add_argument("--iterations", type=int, default=None, help="Timed iterations (default: env GEAK_BENCHMARK_ITERATIONS or 20)")
    parser.add_argument("--reduce-dim", type=int, default=-1, help="Reduction dimension (default: last dim)")
    args = parser.parse_args()

    for path in [BASELINE_SO, OPTIMIZED_SO]:
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    base_label = BASELINE_SO.name
    opt_label = OPTIMIZED_SO.name
    base_lib = load_kernel(str(BASELINE_SO))
    opt_lib = load_kernel(str(OPTIMIZED_SO))

    if args.correctness:
        print(f"Correctness check: {opt_label} vs {base_label} (ground truth)")
        print(f"Shapes: {len(HARNESS_SHAPES)}")
        print()
        ok = mode_correctness(base_lib, opt_lib, HARNESS_SHAPES, args.reduce_dim)
        print()
        print("RESULT: PASS" if ok else "RESULT: FAIL")
        sys.exit(0 if ok else 1)

    elif args.profile:
        print(f"Profile mode: {opt_label}")
        print(f"Shapes: {len(PROFILE_SHAPES)}")
        print()
        mode_profile(opt_lib, PROFILE_SHAPES, args.reduce_dim)

    elif args.benchmark:
        nrepeat = get_iterations(args.iterations)
        print(f"Benchmark: {opt_label} vs {base_label}")
        print(f"Shapes: {len(HARNESS_SHAPES)}, iterations: {nrepeat}")
        print()
        ok = mode_benchmark(base_lib, opt_lib, HARNESS_SHAPES, args.reduce_dim,
                            warmup=5, nrepeat=nrepeat, base_label=base_label, opt_label=opt_label)
        sys.exit(0 if ok else 1)

    elif args.full_benchmark:
        nrepeat = get_iterations(args.iterations)
        print(f"Full benchmark: {opt_label} vs {base_label}")
        print(f"Shapes: {len(ALL_SHAPES)}, iterations: {nrepeat}")
        print()
        ok = mode_benchmark(base_lib, opt_lib, ALL_SHAPES, args.reduce_dim,
                            warmup=5, nrepeat=nrepeat, base_label=base_label, opt_label=opt_label)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()