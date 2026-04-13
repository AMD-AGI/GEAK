#!/usr/bin/env python3
"""GEAK Test Harness - 2D Grouped Convolution Forward

Loads baseline and optimized CK conv-fwd kernel .so files via ctypes,
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
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_SO = SCRIPT_DIR / "libbaseline.so"
OPTIMIZED_SO = SCRIPT_DIR / "liboptimized.so"

# -- Shape lists (sorted by element count) ------------------------------------
# Conv shapes: (G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad)
# G=groups, N=batch, K=out-channels-per-group, C=in-channels-per-group
# Hi,Wi=input spatial, Y,X=filter spatial, stride/dilation/pad applied to both H and W

ALL_SHAPES: list[tuple[int, ...]] = [
    # (G, N,  K,   C,  Hi, Wi, Y, X, stride, dilation, pad)
    (1, 1, 128, 192, 28, 28, 3, 3, 1, 1, 1),
    (1, 1, 256, 192, 28, 28, 3, 3, 2, 1, 1),
    (1, 2, 128, 192, 56, 56, 3, 3, 1, 1, 1),
    (1, 4, 128, 192, 28, 28, 3, 3, 1, 1, 1),
    (1, 1, 256, 192, 71, 71, 3, 3, 2, 1, 1),
    (1, 4, 256, 192, 56, 56, 3, 3, 2, 1, 1),
    (1, 8, 128, 192, 28, 28, 3, 3, 1, 1, 1),
    (1, 8, 256, 192, 56, 56, 3, 3, 2, 1, 1),
    (1, 16, 256, 192, 56, 56, 3, 3, 2, 1, 1),
    (1, 128, 256, 192, 71, 71, 3, 3, 2, 1, 1),
]

HARNESS_SHAPES: list[tuple[int, ...]] = ALL_SHAPES

PROFILE_SHAPES: list[tuple[int, ...]] = [
    ALL_SHAPES[i]
    for i in range(0, len(ALL_SHAPES), max(1, len(ALL_SHAPES) // 3))
][:3]


# -- Kernel loading via ctypes ------------------------------------------------


def load_kernel(path: str):
    """Load a kernel .so and set up the run_kernel function signature."""
    lib = ctypes.CDLL(path)
    lib.run_kernel.restype = ctypes.c_float
    lib.run_kernel.argtypes = [
        ctypes.c_void_p,   # p_in
        ctypes.c_void_p,   # p_wei
        ctypes.c_void_p,   # p_out
        ctypes.c_int64,    # G
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # C
        ctypes.c_int64,    # Hi
        ctypes.c_int64,    # Wi
        ctypes.c_int64,    # Y
        ctypes.c_int64,    # X
        ctypes.c_int64,    # stride_h
        ctypes.c_int64,    # stride_w
        ctypes.c_int64,    # dilation_h
        ctypes.c_int64,    # dilation_w
        ctypes.c_int64,    # pad_h
        ctypes.c_int64,    # pad_w
        ctypes.c_bool,     # time_kernel
        ctypes.c_int,      # warmup
        ctypes.c_int,      # nrepeat
    ]
    return lib


def call_kernel(
    lib,
    p_in, p_wei, p_out,
    G, N, K, C, Hi, Wi, Y, X,
    stride, dilation, pad,
    time_kernel: bool = False,
    warmup: int = 5,
    nrepeat: int = 50,
) -> float:
    """Call run_kernel from a loaded .so. Returns time in ms (0 if not timing)."""
    torch.cuda.synchronize()
    ms = lib.run_kernel(
        p_in, p_wei, p_out,
        G, N, K, C, Hi, Wi, Y, X,
        stride, stride,
        dilation, dilation,
        pad, pad,
        time_kernel,
        warmup,
        nrepeat,
    )
    return ms


# -- Tensor helpers -----------------------------------------------------------

def compute_output_size(Hi, Y, stride, dilation, pad):
    """Compute output spatial dimension."""
    return (Hi + 2 * pad - dilation * (Y - 1) - 1) // stride + 1


def make_tensors(G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad):
    """Allocate input, weight, and output tensors in CK's GNHWC / GKYXC / GNHWK layout."""
    Ho = compute_output_size(Hi, Y, stride, dilation, pad)
    Wo = compute_output_size(Wi, X, stride, dilation, pad)

    # CK layout: GNHWC for input, GKYXC for weight, GNHWK for output
    inp = torch.randn(G, N, Hi, Wi, C, dtype=torch.float16, device="cpu").to("cuda")
    wei = torch.randn(G, K, Y, X, C, dtype=torch.float16, device="cpu").to("cuda")
    out = torch.empty(G, N, Ho, Wo, K, dtype=torch.float16, device="cuda")
    return inp, wei, out, Ho, Wo


def ck_to_torch_ref(inp, wei, G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad):
    """Run PyTorch conv2d as reference. Converts CK layouts to/from PyTorch NCHW."""
    # inp is GNHWC -> reshape to (N, G*C, Hi, Wi)
    inp_nchw = inp.permute(1, 0, 4, 2, 3).reshape(N, G * C, Hi, Wi)
    # wei is GKYXC -> reshape to (G*K, C, Y, X)
    wei_oiyx = wei.permute(0, 1, 4, 2, 3).reshape(G * K, C, Y, X)

    ref = F.conv2d(inp_nchw, wei_oiyx, stride=stride, padding=pad,
                   dilation=dilation, groups=G)
    # ref is (N, G*K, Ho, Wo) -> reshape to GNHWK = (G, N, Ho, Wo, K)
    Ho, Wo = ref.shape[2], ref.shape[3]
    ref = ref.reshape(N, G, K, Ho, Wo).permute(1, 0, 3, 4, 2)
    return ref


def compute_tflops(G, N, K, C, Ho, Wo, Y, X, time_ms):
    """2 * G * N * K * C * Ho * Wo * Y * X FLOPs for grouped conv."""
    flops = 2.0 * G * N * K * C * Ho * Wo * Y * X
    return flops / time_ms / 1e9


# -- GPU warmup ---------------------------------------------------------------

_GPU_WARMUP_ROUNDS = 3


def _gpu_warmup(base_lib, opt_lib, shapes):
    """Run the largest shape untimed on both kernels to bring GPU to peak clock."""
    G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad = shapes[-1]
    inp, wei, out, _, _ = make_tensors(G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad)
    for _ in range(_GPU_WARMUP_ROUNDS):
        call_kernel(base_lib, inp.data_ptr(), wei.data_ptr(), out.data_ptr(),
                    G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad, time_kernel=False)
        call_kernel(opt_lib, inp.data_ptr(), wei.data_ptr(), out.data_ptr(),
                    G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad, time_kernel=False)
    torch.cuda.synchronize()


# -- Mode implementations ----------------------------------------------------


def mode_correctness(base_lib, opt_lib, shapes) -> bool:
    """Verify optimized kernel against baseline on each shape."""
    torch.manual_seed(42)
    any_failed = False

    for shape in shapes:
        G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad = shape
        label = f"G={G} N={N} K={K} C={C} {Hi}x{Wi} f{Y}x{X} s{stride} d{dilation} p{pad}"

        inp, wei, out_base, Ho, Wo = make_tensors(G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad)
        out_opt = torch.empty_like(out_base)

        ms_base = call_kernel(base_lib, inp.data_ptr(), wei.data_ptr(), out_base.data_ptr(),
                              G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad, time_kernel=False)
        if ms_base < 0:
            print(f"  SKIP  {label}  (baseline: UNSUPPORTED)")
            continue

        ms_opt = call_kernel(opt_lib, inp.data_ptr(), wei.data_ptr(), out_opt.data_ptr(),
                             G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad, time_kernel=False)
        if ms_opt < 0:
            print(f"  SKIP  {label}  (optimized: UNSUPPORTED)")
            continue

        try:
            torch.testing.assert_close(out_opt, out_base, atol=1e-2, rtol=1e-2)
            max_err = (out_opt.float() - out_base.float()).abs().max().item()
            print(f"  PASS  {label}  max_err={max_err:.2e}")
        except AssertionError as e:
            max_err = (out_opt.float() - out_base.float()).abs().max().item()
            print(f"  FAIL  {label}  max_err={max_err:.2e}")
            print(f"        {e}")
            any_failed = True

    return not any_failed


def mode_profile(opt_lib, shapes) -> None:
    """Run optimized kernel once per shape for profiler capture."""
    for shape in shapes:
        G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad = shape
        inp, wei, out, _, _ = make_tensors(G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad)
        ms = call_kernel(opt_lib, inp.data_ptr(), wei.data_ptr(), out.data_ptr(),
                         G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad, time_kernel=False)
        if ms < 0:
            print(f"  SKIP  (UNSUPPORTED)")


def mode_benchmark(base_lib, opt_lib, shapes, warmup: int, nrepeat: int) -> bool:
    """Benchmark both kernels on shapes."""
    _gpu_warmup(base_lib, opt_lib, shapes)

    opt_times_us: list[float] = []
    base_times_us: list[float] = []
    speedups: list[float] = []

    print(f"{'Shape':>55s}  {'Base ms':>8s}  {'Opt ms':>8s}  {'TFLOPS':>7s}  {'Speedup':>7s}")
    print("-" * 95)

    for shape in shapes:
        G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad = shape
        label = f"G={G} N={N} K={K} C={C} {Hi}x{Wi} f{Y}x{X} s{stride}"
        Ho = compute_output_size(Hi, Y, stride, dilation, pad)
        Wo = compute_output_size(Wi, X, stride, dilation, pad)

        inp, wei, out, _, _ = make_tensors(G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad)

        ms_base = call_kernel(base_lib, inp.data_ptr(), wei.data_ptr(), out.data_ptr(),
                              G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad,
                              time_kernel=True, warmup=warmup, nrepeat=nrepeat)
        if ms_base < 0:
            print(f"{label:>55s}  {'UNSUP':>8s}")
            continue

        out2 = torch.empty_like(out)
        ms_opt = call_kernel(opt_lib, inp.data_ptr(), wei.data_ptr(), out2.data_ptr(),
                             G, N, K, C, Hi, Wi, Y, X, stride, dilation, pad,
                             time_kernel=True, warmup=warmup, nrepeat=nrepeat)
        if ms_opt < 0:
            print(f"{label:>55s}  {ms_base:8.4f}  {'UNSUP':>8s}")
            continue

        tflops = compute_tflops(G, N, K, C, Ho, Wo, Y, X, ms_opt)
        speedup = ms_base / ms_opt if ms_opt > 0 else float("inf")

        base_times_us.append(ms_base * 1000)
        opt_times_us.append(ms_opt * 1000)
        speedups.append(speedup)

        print(f"{label:>55s}  {ms_base:8.4f}  {ms_opt:8.4f}  {tflops:7.1f}  {speedup:6.2f}x")

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
    parser = argparse.ArgumentParser(description="GEAK Conv2D Test Harness")

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
