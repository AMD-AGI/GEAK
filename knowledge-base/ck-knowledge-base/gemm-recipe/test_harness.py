#!/usr/bin/env python3
"""GEAK Test Harness - GEMM+Bias

Loads baseline and optimized CK GEMM+bias kernel .so files via ctypes,
verifies the optimized kernel against the baseline (ground truth), and
reports TFLOPS and speedup.

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
# GEMM shapes: (M, N, K). A is (M, K) row-major, B is (K, N) col-major.
# D (bias) is (M, N) row-major with StrideD=0 (broadcast along M).

ALL_SHAPES: list[tuple[int, int, int]] = [
    (256, 256, 128),
    (512, 512, 256),
    (1024, 1024, 512),
    (1920, 2048, 2048),
    (2048, 2048, 2048),
    (3840, 4096, 2048),
    (4096, 4096, 4096),
]

HARNESS_SHAPES: list[tuple[int, int, int]] = ALL_SHAPES

PROFILE_SHAPES: list[tuple[int, int, int]] = [
    ALL_SHAPES[i]
    for i in range(0, len(ALL_SHAPES), max(1, len(ALL_SHAPES) // 3))
][:3]


# -- Kernel loading via ctypes ------------------------------------------------


def load_kernel(path: str):
    """Load a kernel .so and set up the run_kernel function signature."""
    lib = ctypes.CDLL(path)
    lib.run_kernel.restype = ctypes.c_float
    lib.run_kernel.argtypes = [
        ctypes.c_void_p,   # p_a
        ctypes.c_void_p,   # p_b
        ctypes.c_void_p,   # p_d0 (bias)
        ctypes.c_void_p,   # p_e  (output)
        ctypes.c_int64,    # M
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # StrideA
        ctypes.c_int64,    # StrideB
        ctypes.c_int64,    # StrideD0
        ctypes.c_int64,    # StrideE
        ctypes.c_bool,     # time_kernel
        ctypes.c_int,      # warmup
        ctypes.c_int,      # nrepeat
    ]
    return lib


def call_kernel(
    lib,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    time_kernel: bool = False,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Call run_kernel from a loaded .so. Returns time in ms (0 if not timing)."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # A row-major: StrideA = K
    # B col-major (K,N) with strides (1, K): StrideB = K
    # D broadcast bias: StrideD0 = 0
    # E row-major: StrideE = N
    StrideA = a.stride(0)
    StrideB = b.stride(1)
    StrideD0 = 0
    StrideE = e.stride(0)

    torch.cuda.synchronize()
    ms = lib.run_kernel(
        a.data_ptr(),
        b.data_ptr(),
        d.data_ptr(),
        e.data_ptr(),
        M, N, K,
        StrideA, StrideB, StrideD0, StrideE,
        time_kernel,
        warmup,
        nrepeat,
    )
    return ms


# -- Test logic ---------------------------------------------------------------


def make_tensors(M: int, N: int, K: int):
    """Create A(M,K) row-major, B(N,K) whose .t() is (K,N) col-major, D(1,N), E(M,N)."""
    a = torch.randn(M, K, dtype=torch.float16, device="cpu").to("cuda")
    # B: CK expects (K,N) column-major, i.e. contiguous along K.
    # Create (N,K) contiguous, then .t() gives logical (K,N) with strides (1, K) = col-major.
    b_storage = torch.randn(N, K, dtype=torch.float16, device="cpu").to("cuda")
    b = b_storage.t()  # shape (K,N), strides (1, K) — column-major
    d = torch.randn(1, N, dtype=torch.float16, device="cpu").to("cuda")
    e = torch.empty(M, N, dtype=torch.float16, device="cuda")
    return a, b, d, e


def run_kernel_output(lib, M: int, N: int, K: int) -> torch.Tensor | None:
    """Run kernel, return output tensor or None if unsupported."""
    a, b, d, e = make_tensors(M, N, K)
    ms = call_kernel(lib, a, b, d, e, time_kernel=False)
    if ms < 0:
        return None
    return e, a, b, d


def benchmark_kernel(lib, M: int, N: int, K: int, warmup: int = 5, nrepeat: int = 20) -> float:
    """Time the kernel. Returns average time in ms, or -1 if unsupported."""
    a, b, d, e = make_tensors(M, N, K)
    ms = call_kernel(lib, a, b, d, e, time_kernel=True, warmup=warmup, nrepeat=nrepeat)
    return ms


def compute_tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Compute TFLOPS for GEMM (2*M*N*K FLOPs)."""
    flops = 2.0 * M * N * K
    return flops / time_ms / 1e9


# -- GPU warmup ---------------------------------------------------------------

_GPU_WARMUP_ROUNDS = 3


def _gpu_warmup(base_lib, opt_lib, shapes):
    """Run the largest shape untimed on both kernels to bring GPU to peak clock."""
    M, N, K = shapes[-1]
    a, b, d, e = make_tensors(M, N, K)
    for _ in range(_GPU_WARMUP_ROUNDS):
        call_kernel(base_lib, a, b, d, e, time_kernel=False)
        call_kernel(opt_lib, a, b, d, e, time_kernel=False)
    torch.cuda.synchronize()


# -- Mode implementations ----------------------------------------------------


def mode_correctness(base_lib, opt_lib, shapes) -> bool:
    """Verify optimized kernel against baseline on each shape."""
    torch.manual_seed(42)
    any_failed = False

    for shape in shapes:
        M, N, K = shape
        a, b, d, e_base = make_tensors(M, N, K)
        e_opt = torch.empty_like(e_base)

        ms_base = call_kernel(base_lib, a, b, d, e_base, time_kernel=False)
        if ms_base < 0:
            print(f"  SKIP  ({M}, {N}, {K})  (baseline: UNSUPPORTED)")
            continue

        ms_opt = call_kernel(opt_lib, a, b, d, e_opt, time_kernel=False)
        if ms_opt < 0:
            print(f"  SKIP  ({M}, {N}, {K})  (optimized: UNSUPPORTED)")
            continue

        try:
            torch.testing.assert_close(e_opt, e_base, atol=1e-2, rtol=1e-2)
            max_err = (e_opt.float() - e_base.float()).abs().max().item()
            print(f"  PASS  ({M}, {N}, {K})  max_err={max_err:.2e}")
        except AssertionError as e:
            max_err = (e_opt.float() - e_base.float()).abs().max().item()
            print(f"  FAIL  ({M}, {N}, {K})  max_err={max_err:.2e}")
            print(f"        {e}")
            any_failed = True

    return not any_failed


def mode_profile(opt_lib, shapes) -> None:
    """Run optimized kernel once per shape for profiler capture."""
    for shape in shapes:
        M, N, K = shape
        a, b, d, e = make_tensors(M, N, K)
        ms = call_kernel(opt_lib, a, b, d, e, time_kernel=False)
        if ms < 0:
            print(f"  SKIP  ({M}, {N}, {K})  (UNSUPPORTED)")


def mode_benchmark(base_lib, opt_lib, shapes, warmup: int, nrepeat: int) -> bool:
    """Benchmark both kernels on shapes. Returns True if all shapes ran."""
    _gpu_warmup(base_lib, opt_lib, shapes)

    opt_times_us: list[float] = []
    base_times_us: list[float] = []
    speedups: list[float] = []

    print(f"{'Shape':>25s}  {'Baseline ms':>11s}  {'Optimized ms':>12s}  {'TFLOPS':>8s}  {'Speedup':>7s}")
    print("-" * 75)

    for shape in shapes:
        M, N, K = shape
        label = f"({M}, {N}, {K})"

        ms_base = benchmark_kernel(base_lib, M, N, K, warmup=warmup, nrepeat=nrepeat)
        if ms_base < 0:
            print(f"{label:>25s}  {'UNSUPPORTED':>11s}")
            continue

        ms_opt = benchmark_kernel(opt_lib, M, N, K, warmup=warmup, nrepeat=nrepeat)
        if ms_opt < 0:
            print(f"{label:>25s}  {ms_base:11.4f}  {'UNSUPPORTED':>12s}")
            continue

        tflops = compute_tflops(M, N, K, ms_opt)
        speedup = ms_base / ms_opt if ms_opt > 0 else float("inf")

        base_times_us.append(ms_base * 1000)
        opt_times_us.append(ms_opt * 1000)
        speedups.append(speedup)

        print(f"{label:>25s}  {ms_base:11.4f}  {ms_opt:12.4f}  {tflops:8.1f}  {speedup:6.2f}x")

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
    parser = argparse.ArgumentParser(description="GEAK GEMM+Bias Test Harness")

    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--correctness", action="store_true")
    modes.add_argument("--profile", action="store_true")
    modes.add_argument("--benchmark", action="store_true")
    modes.add_argument("--full-benchmark", action="store_true")

    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    for path in [BASELINE_SO, OPTIMIZED_SO]:
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)

    base_lib = load_kernel(str(BASELINE_SO))
    opt_lib = load_kernel(str(OPTIMIZED_SO))

    if args.correctness:
        print(f"Correctness check: optimized vs baseline (ground truth)")
        print(f"Shapes: {len(HARNESS_SHAPES)}")
        print()
        ok = mode_correctness(base_lib, opt_lib, HARNESS_SHAPES)
        print()
        print("RESULT: PASS" if ok else "RESULT: FAIL")
        sys.exit(0 if ok else 1)

    elif args.profile:
        print(f"Profile mode: optimized kernel")
        print(f"Shapes: {len(PROFILE_SHAPES)}")
        print()
        mode_profile(opt_lib, PROFILE_SHAPES)

    elif args.benchmark:
        nrepeat = get_iterations(args.iterations)
        print(f"Benchmark: optimized vs baseline")
        print(f"Shapes: {len(HARNESS_SHAPES)}, iterations: {nrepeat}")
        print()
        ok = mode_benchmark(base_lib, opt_lib, HARNESS_SHAPES, warmup=5, nrepeat=nrepeat)
        sys.exit(0 if ok else 1)

    elif args.full_benchmark:
        nrepeat = get_iterations(args.iterations)
        print(f"Full benchmark: optimized vs baseline")
        print(f"Shapes: {len(ALL_SHAPES)}, iterations: {nrepeat}")
        print()
        ok = mode_benchmark(base_lib, opt_lib, ALL_SHAPES, warmup=5, nrepeat=nrepeat)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
