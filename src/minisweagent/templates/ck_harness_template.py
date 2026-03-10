#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
#
# CK (Composable Kernel) test harness template.
#
# This template is used by the UnitTestAgent to generate a test harness for
# CK kernel examples.  The agent fills in the uppercase placeholders below
# based on the specific kernel being optimized.
#
# Correctness checking compares the tensor outputs of the ORIGINAL (pre-
# optimization) binary against the OPTIMIZED (post-optimization) binary.
# Both binaries are run with the same deterministic inputs and the env var
# GEAK_DUMP_OUTPUT is used to dump the GPU output tensor to a text file
# (one float per line, via CK's Tensor::savetxt).

import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Configuration -- filled in by UnitTestAgent / preprocessor
# ---------------------------------------------------------------------------

ORIGINAL_BINARY = "__ORIGINAL_BINARY__"
BUILD_DIR = "__BUILD_DIR__"
CMAKE_SOURCE_DIR = "__CMAKE_SOURCE_DIR__"
CMAKE_TARGET = "__CMAKE_TARGET__"

# Tolerances for original-vs-optimized comparison on AMD GPUs.
# Different tile/block configurations change FP16 accumulation order,
# so we need relaxed tolerances compared to GPU-vs-CPU checks.
RTOL_FP16 = 1e-2
ATOL_FP16 = 1e-2
RTOL_FP32 = 1e-4
ATOL_FP32 = 1e-5

# Default data type for tolerance selection
KERNEL_DTYPE = "fp16"

# ---------------------------------------------------------------------------
# Shape definitions
# Each shape is a list of extra CLI args appended after "verify init time".
# For kernels that only accept 3 args (e.g. example 41), use [[]].
# For kernels that accept conv params (e.g. example 30), each entry is a
# list of strings like ["2", "32", "2", "256", "192", "3", "3", ...].
# ---------------------------------------------------------------------------

ALL_SHAPES = [
    [],
]

def _sample_shapes(shapes, n):
    if len(shapes) <= n:
        return list(shapes)
    step = len(shapes) / n
    return [shapes[int(i * step)] for i in range(n)]

HARNESS_SHAPES = _sample_shapes(ALL_SHAPES, 25)
PROFILE_SHAPES = _sample_shapes(ALL_SHAPES, 5)

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _rebuild_optimized():
    """Rebuild the optimized binary from current (possibly modified) source."""
    if not os.path.isdir(BUILD_DIR):
        os.makedirs(BUILD_DIR, exist_ok=True)
        configure_cmd = [
            "cmake",
            "-DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc",
            f"-DCMAKE_PREFIX_PATH=/opt/rocm",
            "-S", CMAKE_SOURCE_DIR,
            "-B", BUILD_DIR,
        ]
        result = subprocess.run(configure_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"CMAKE CONFIGURE FAILED:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

    build_cmd = ["cmake", "--build", BUILD_DIR, "--target", CMAKE_TARGET, "-j"]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"BUILD FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def _optimized_binary_path():
    """Return the path to the built optimized binary."""
    return os.path.join(BUILD_DIR, "bin", CMAKE_TARGET)


def _ensure_original():
    """Verify the original binary exists and is executable."""
    if not os.path.isfile(ORIGINAL_BINARY):
        print(f"ERROR: Original binary not found: {ORIGINAL_BINARY}", file=sys.stderr)
        sys.exit(1)
    if not os.access(ORIGINAL_BINARY, os.X_OK):
        print(f"ERROR: Original binary not executable: {ORIGINAL_BINARY}", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Binary execution helpers
# ---------------------------------------------------------------------------

def _run_binary(binary_path, verify, init_method, time_kernel, shape_args,
                dump_output_path=None, capture=True):
    """Run a CK binary. Returns (returncode, stdout, stderr)."""
    cmd = [binary_path, str(verify), str(init_method), str(time_kernel)] + shape_args

    env = os.environ.copy()
    if dump_output_path:
        env["GEAK_DUMP_OUTPUT"] = dump_output_path

    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(cmd, env=env)
        return result.returncode, "", ""


def _parse_perf_ms(output):
    """Parse 'Perf: X ms, ...' from binary output. Returns float ms or None."""
    m = re.search(r'Perf:\s+([\d.]+)\s+ms', output)
    if m:
        return float(m.group(1))
    return None


def _get_tolerances():
    """Return (rtol, atol) based on kernel data type."""
    if KERNEL_DTYPE.lower() in ("fp16", "half", "f16", "bf16"):
        return RTOL_FP16, ATOL_FP16
    return RTOL_FP32, ATOL_FP32

# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def run_correctness(shapes):
    """Compare tensor outputs of original vs optimized binary."""
    _ensure_original()
    _rebuild_optimized()
    opt_binary = _optimized_binary_path()

    if not os.path.isfile(opt_binary):
        print(f"ERROR: Optimized binary not found after build: {opt_binary}",
              file=sys.stderr)
        return False

    rtol, atol = _get_tolerances()
    print(f"Running correctness check on {len(shapes)} shapes "
          f"(rtol={rtol}, atol={atol})...")

    all_pass = True
    for i, shape_args in enumerate(shapes):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dump = os.path.join(tmpdir, "orig_output.txt")
            opt_dump = os.path.join(tmpdir, "opt_output.txt")

            rc_orig, stdout_orig, stderr_orig = _run_binary(
                ORIGINAL_BINARY, 0, 1, 0, shape_args,
                dump_output_path=orig_dump,
            )
            if rc_orig != 0:
                print(f"  FAIL shape[{i}]: original binary exit code {rc_orig}")
                if stderr_orig:
                    print(f"    stderr: {stderr_orig.strip()[:200]}")
                all_pass = False
                continue

            rc_opt, stdout_opt, stderr_opt = _run_binary(
                opt_binary, 0, 1, 0, shape_args,
                dump_output_path=opt_dump,
            )
            if rc_opt != 0:
                print(f"  FAIL shape[{i}]: optimized binary exit code {rc_opt}")
                if stderr_opt:
                    print(f"    stderr: {stderr_opt.strip()[:200]}")
                all_pass = False
                continue

            if not os.path.isfile(orig_dump):
                print(f"  FAIL shape[{i}]: original binary did not dump output "
                      f"(GEAK_DUMP_OUTPUT not supported?)")
                all_pass = False
                continue

            if not os.path.isfile(opt_dump):
                print(f"  FAIL shape[{i}]: optimized binary did not dump output")
                all_pass = False
                continue

            try:
                orig_data = np.loadtxt(orig_dump, dtype=np.float32)
                opt_data = np.loadtxt(opt_dump, dtype=np.float32)
            except Exception as exc:
                print(f"  FAIL shape[{i}]: could not load dump files: {exc}")
                all_pass = False
                continue

            if orig_data.shape != opt_data.shape:
                print(f"  FAIL shape[{i}]: output shape mismatch: "
                      f"orig={orig_data.shape} vs opt={opt_data.shape}")
                all_pass = False
                continue

            if np.allclose(orig_data, opt_data, rtol=rtol, atol=atol):
                print(f"  PASS shape[{i}] ({len(orig_data)} elements)")
            else:
                diff = np.abs(orig_data - opt_data)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                num_mismatched = np.sum(
                    ~np.isclose(orig_data, opt_data, rtol=rtol, atol=atol)
                )
                print(f"  FAIL shape[{i}]: tensor mismatch -- "
                      f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
                      f"{num_mismatched}/{len(orig_data)} elements differ")
                all_pass = False

    return all_pass


def run_benchmark(shapes, iterations):
    """Benchmark the optimized binary. Returns list of median latencies in ms."""
    _rebuild_optimized()
    opt_binary = _optimized_binary_path()

    latencies = []
    for i, shape_args in enumerate(shapes):
        times = []
        for _ in range(iterations):
            rc, stdout, stderr = _run_binary(opt_binary, 0, 1, 1, shape_args)
            if rc != 0:
                print(f"  ERROR shape[{i}]: exit code {rc}", file=sys.stderr)
                break
            t = _parse_perf_ms(stdout)
            if t is not None and t > 0:
                times.append(t)
        if times:
            med = statistics.median(times)
            latencies.append(med)
            print(f"  shape[{i}]: median={med:.4f} ms "
                  f"(over {len(times)} runs)")
        else:
            print(f"  shape[{i}]: no valid timing data")
    return latencies


def geomean(values):
    """Compute geometric mean of a list of positive values."""
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def main():
    parser = argparse.ArgumentParser(
        description="CK test harness (original-vs-optimized tensor comparison)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--correctness", action="store_true",
                      help="Compare tensor outputs of original vs optimized")
    mode.add_argument("--profile", action="store_true",
                      help="Run optimized kernel once for profiling")
    mode.add_argument("--benchmark", action="store_true",
                      help="Benchmark on HARNESS_SHAPES")
    mode.add_argument("--full-benchmark", action="store_true",
                      help="Benchmark on ALL_SHAPES")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of benchmark iterations "
                             "(default: 20 or GEAK_BENCHMARK_ITERATIONS)")
    args = parser.parse_args()

    if args.iterations is not None:
        iterations = args.iterations
    else:
        env_iters = os.environ.get("GEAK_BENCHMARK_ITERATIONS")
        iterations = int(env_iters) if env_iters else 20

    if args.correctness:
        print(f"=== Correctness Check ({len(HARNESS_SHAPES)} shapes) ===")
        ok = run_correctness(HARNESS_SHAPES)
        if ok:
            print("All correctness checks PASSED.")
            sys.exit(0)
        else:
            print("Some correctness checks FAILED.", file=sys.stderr)
            sys.exit(1)

    elif args.profile:
        _rebuild_optimized()
        opt_binary = _optimized_binary_path()
        print(f"=== Profile Mode ({len(PROFILE_SHAPES)} shapes) ===")
        for i, shape_args in enumerate(PROFILE_SHAPES):
            print(f"  Profiling shape[{i}]")
            _run_binary(opt_binary, 0, 1, 1, shape_args, capture=False)
        sys.exit(0)

    elif args.benchmark or args.full_benchmark:
        shapes = ALL_SHAPES if args.full_benchmark else HARNESS_SHAPES
        mode_name = "Full Benchmark" if args.full_benchmark else "Benchmark"
        print(f"=== {mode_name} ({len(shapes)} shapes, "
              f"{iterations} iterations each) ===")
        latencies = run_benchmark(shapes, iterations)
        if not latencies:
            print("ERROR: No valid benchmark results.", file=sys.stderr)
            sys.exit(1)
        overall = geomean(latencies)
        print(f"\nGeometric mean latency: {overall:.4f} ms")
        print(f"GEAK_RESULT_LATENCY_MS={overall:.6f}")
        sys.exit(0)


if __name__ == "__main__":
    main()
