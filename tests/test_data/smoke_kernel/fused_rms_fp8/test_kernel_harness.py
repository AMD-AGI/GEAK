#!/usr/bin/env python3
# GEAK materialized harness bootstrap
import importlib.util
import os
import sys
import types
from pathlib import Path

def _find_baseline_kernel_dir():
    """Find preprocess dir (has benchmark_baseline.txt) by walking up from GEAK_WORK_DIR."""
    work = os.environ.get("GEAK_WORK_DIR", "").strip()
    if not work:
        return None
    d = Path(work).resolve()
    for _ in range(10):
        if d is None or not d.exists():
            break
        bb = d / "benchmark_baseline.txt"
        if bb.is_file():
            return str(d)
        d = d.parent
    return None

def _load_baseline_triton(baseline_dir, module_alias, entry_name):
    """Load kernel from baseline_dir. Returns callable or None."""
    entry_file = Path(baseline_dir) / "kernel.py"
    if not entry_file.is_file():
        return None
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)
    spec = importlib.util.spec_from_file_location(module_alias, entry_file)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    try:
        spec.loader.exec_module(module)
        return getattr(module, entry_name, None)
    except Exception:
        return None

def _resolve_geak_kernel_dir():
    candidates = []
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir:
        candidates.append(work_dir)
    repo_root = os.environ.get("GEAK_REPO_ROOT", "").strip()
    rel_kernel_dir = '.'
    if repo_root and rel_kernel_dir:
        candidates.append(os.path.join(repo_root, rel_kernel_dir))
    original_kernel_dir = os.path.dirname(os.path.abspath(__file__))
    if original_kernel_dir:
        candidates.append(original_kernel_dir)
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "kernel.py")):
            return candidate
    return original_kernel_dir or os.getcwd()

def _ensure_geak_package(module_name):
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        prefix = ".".join(parts[:idx])
        if prefix in sys.modules:
            continue
        pkg = types.ModuleType(prefix)
        pkg.__path__ = []
        sys.modules[prefix] = pkg

def _ensure_geak_aiter_fp8_dtype(module):
    fp8_value = getattr(module, "fp8_dtype", None)
    if fp8_value is None:
        return
    aiter_mod = sys.modules.get("aiter")
    if aiter_mod is None:
        try:
            import aiter as aiter_mod
        except Exception:
            _ensure_geak_package("aiter")
            aiter_mod = sys.modules.get("aiter")
    if aiter_mod is None:
        return
    dtypes_obj = getattr(aiter_mod, "dtypes", None)
    if dtypes_obj is None:
        dtypes_obj = types.SimpleNamespace()
        setattr(aiter_mod, "dtypes", dtypes_obj)
    if getattr(dtypes_obj, "fp8", None) is None:
        setattr(dtypes_obj, "fp8", fp8_value)

def _register_geak_aliases(kernel_dir):
    aliases = ['fused_rms_fp8', 'aiter.ops.triton.fused_fp8_quant']
    entry_file = os.path.join(kernel_dir, "kernel.py")
    if not os.path.isfile(entry_file):
        return
    for alias in aliases:
        if alias in sys.modules:
            continue
        _ensure_geak_package(alias)
        spec = importlib.util.spec_from_file_location(alias, entry_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        _ensure_geak_aiter_fp8_dtype(module)

_KERNEL_DIR = _resolve_geak_kernel_dir()
if _KERNEL_DIR and _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
_register_geak_aliases(_KERNEL_DIR)

"""
Test harness for fused_fp8_quant kernel (aiter reference).

Modes: --correctness, --profile, --benchmark, --full-benchmark

This file is structurally identical to the test harness embedded in
kernel.py, except it imports the kernel from the aiter package rather
than using the inlined implementation.
"""
import argparse
import math
import torch
import torch.nn.functional as F

from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
import aiter

fp8_dtype = aiter.dtypes.fp8


# ============================================================================
# TEST CONFIGURATIONS
# ============================================================================

# (M, N1, N2) -- batch/tokens, hidden dimension 1, hidden dimension 2
ALL_SHAPES = [
    (1, 128, 128),
    (4, 128, 128),
    (1, 128, 4096),
    (8, 128, 128),
    (1, 128, 7168),
    (1, 4096, 4096),
    (1, 128, 8192),
    (1, 4096, 8192),
    (1, 7168, 7168),
    (1, 8192, 8192),
    (32, 128, 128),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (256, 128, 128),
    (32, 128, 7168),
    (1024, 128, 128),
    (256, 128, 7168),
    (256, 4096, 4096),
    (8192, 128, 128),
    (32, 7168, 7168),
    (256, 7168, 7168),
    (1024, 4096, 4096),
    (1024, 8192, 8192),
    (8192, 7168, 7168),
]

seen = set()
unique_shapes = []
for s in ALL_SHAPES:
    if s not in seen:
        seen.add(s)
        unique_shapes.append(s)
ALL_SHAPES = sorted(unique_shapes, key=lambda s: s[0] * (s[1] + s[2]))

# HARNESS_SHAPES: uniformly sample 25 shapes from ALL_SHAPES
_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _harness_indices = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _harness_indices]

# PROFILE_SHAPES: exactly 5 shapes evenly spaced
_profile_indices = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _profile_indices]

# For backward compatibility
EVAL_CONFIGS = HARNESS_SHAPES
PROFILE_CONFIGS = PROFILE_SHAPES

RTOL, ATOL = 0.1, 0.1


# ============================================================================
# REFERENCE IMPLEMENTATIONS
# ============================================================================


def rmsnorm(input, weight, eps=1e-6):
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps)
    rms_norm = input * norm_factor[:, None] * weight[None, :]
    return rms_norm


def per_token_fp8_group_quant(x, dtype_quant, group_size=128):
    DTYPE_MAX = torch.finfo(dtype_quant).max
    M, N = x.shape
    if N % group_size > 0:
        num_pad = group_size - (N % group_size)
        x_reshape = F.pad(x, (0, num_pad, 0, 0), "constant", 0)
        x_reshape = x_reshape.reshape(
            M, (N + group_size - 1) // group_size, group_size
        ).to(torch.float32)
    else:
        x_reshape = x.reshape(M, N // group_size, group_size).to(torch.float32)
    x_max = torch.max(torch.abs(x_reshape), dim=-1, keepdim=True)[0]
    x_max = torch.where(x_max < 1e-10, 1e-10, x_max).to(torch.float32)
    x_scale = x_max / DTYPE_MAX
    scale_recip = 1.0 / x_scale
    x_quant = torch.clamp(x_reshape * scale_recip, -DTYPE_MAX, DTYPE_MAX).to(
        dtype_quant
    )
    x_quant = x_quant.reshape(M, (N + group_size - 1) // group_size * group_size)[:, :N]
    x_scale = x_scale.squeeze(-1)
    return x_quant, x_scale


def upcast(x, s, dtype, group_size=128):
    x_N = x.shape[1]
    x = x.reshape(-1, x_N // group_size, group_size).to(torch.float32) * s.reshape(
        -1, s.shape[1], 1
    )
    x = x.reshape(-1, x_N)
    return x.to(dtype=dtype)


def run_torch_rms_fp8_group_quant(
    x1, w1, eps1, x2, w2, eps2, res1, dtype_quant, group_size
):
    s = x1 + res1
    y1 = rmsnorm(s, w1, eps1)
    y2 = rmsnorm(x2, w2, eps2)
    y1_q, y1_s = per_token_fp8_group_quant(y1, dtype_quant, group_size)
    return (y1_q, y1_s), y1.to(x1.dtype), y2.to(x1.dtype), s.to(x1.dtype)


# ============================================================================
# INPUT GENERATION
# ============================================================================


def generate_inputs(M, N1, N2, dtype=torch.bfloat16):
    """Generate inputs on CPU then move to GPU."""
    torch.manual_seed(42)
    x1 = (torch.randn((M, N1), dtype=dtype, device="cpu") / 10).to("cuda")
    x2 = (torch.randn((M, N2), dtype=dtype, device="cpu") / 10).to("cuda")
    w1 = torch.ones((N1,), dtype=torch.float32, device="cpu").to("cuda")
    w2 = torch.ones((N2,), dtype=torch.float32, device="cpu").to("cuda")
    res1 = (torch.randn((M, N1), dtype=dtype, device="cpu") / 10).to("cuda")
    return x1, w1, x2, w2, res1


# ============================================================================
# TEST HARNESS
# ============================================================================


def run_correctness(shapes=None, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    group_size = 128
    dtype = torch.bfloat16
    results, failures = [], []

    for i, (M, N1, N2) in enumerate(shapes):
        try:
            x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)

            (y1_q_torch, y1_s_torch), y1_torch, y2_torch, y1_res_torch = \
                run_torch_rms_fp8_group_quant(
                    x1, w1, 1e-6, x2, w2, 1e-6, res1, fp8_dtype, group_size
                )

            (y1_q_triton, y1_s_triton), y1_triton, y2_triton, y1_res_triton = \
                fused_rms_fp8_group_quant(
                    x1, w1, 1e-6,
                    inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                    group_size=group_size,
                    dtype_quant=fp8_dtype,
                    res1=res1,
                    output_unquantized_inp1=True,
                )

            torch.testing.assert_close(y1_torch, y1_triton, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(y2_torch, y2_triton, atol=ATOL, rtol=RTOL)
            torch.testing.assert_close(y1_res_torch, y1_res_triton, atol=ATOL, rtol=RTOL)

            y1_upcast_torch = upcast(
                y1_q_torch, y1_s_torch, dtype=torch.float32, group_size=group_size
            )
            y1_upcast_triton = upcast(
                y1_q_triton, y1_s_triton, dtype=torch.float32, group_size=group_size
            )
            torch.testing.assert_close(y1_upcast_torch, y1_upcast_triton, atol=ATOL, rtol=RTOL)

            results.append({"config": (M, N1, N2), "correct": True})
            if verbose:
                print(f"  PASS: ({M}, {N1}, {N2})")

            del x1, x2, w1, w2, res1
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append({"config": (M, N1, N2), "error": str(e)})
            if verbose:
                print(f"  FAIL: ({M}, {N1}, {N2}) - {str(e)[:50]}")

    if verbose:
        print("-" * 62)
        print(
            f"{'Status:':<22} {'ALL PASS' if not failures else f'FAILED ({len(failures)}/{len(shapes)})'}"
        )

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = PROFILE_SHAPES
    group_size = 128
    dtype = torch.bfloat16

    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for M, N1, N2 in shapes:
        x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)
        for _ in range(warmup):
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()
        for _ in range(iters):
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()
        if verbose:
            print(f"  ({M},{N1},{N2}) done")
        del x1, x2, w1, w2, res1
        torch.cuda.empty_cache()


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    """Benchmark kernel vs reference. Uses baseline Triton when available; else PyTorch."""
    if shapes is None:
        shapes = HARNESS_SHAPES
    group_size = 128
    dtype = torch.bfloat16
    baseline_dir = _find_baseline_kernel_dir()
    kernel_dir = _resolve_geak_kernel_dir()
    baseline_fn = None
    if baseline_dir and baseline_dir != kernel_dir:
        baseline_fn = _load_baseline_triton(baseline_dir, "baseline_fused_rms_fp8", "fused_rms_fp8_group_quant")
    ref_label = "baseline_triton" if baseline_fn else "PyTorch"

    latencies = []
    speedups = []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    print(f"  Comparing kernel vs {ref_label}")
    print(f"{'Config (M,N1,N2)':<22} {'Ref':>10} {'Triton':>10} {'Speedup':>10}")
    print("-" * 62)

    for M, N1, N2 in shapes:
        x1, w1, x2, w2, res1 = generate_inputs(M, N1, N2, dtype)

        for _ in range(warmup):
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fused_rms_fp8_group_quant(
                x1, w1, 1e-6,
                inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                group_size=group_size,
                dtype_quant=fp8_dtype,
                res1=res1,
                output_unquantized_inp1=True,
            )
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]

        ref_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if baseline_fn is not None:
                _ = baseline_fn(
                    x1, w1, 1e-6,
                    inp2=x2, inp2_weight=w2, inp2_epsilon=1e-6,
                    group_size=group_size,
                    dtype_quant=fp8_dtype,
                    res1=res1,
                    output_unquantized_inp1=True,
                )
            else:
                _ = run_torch_rms_fp8_group_quant(
                    x1, w1, 1e-6, x2, w2, 1e-6, res1, fp8_dtype, group_size
                )
            end.record()
            torch.cuda.synchronize()
            ref_times.append(start.elapsed_time(end))

        ref_ms = sorted(ref_times)[len(ref_times) // 2]
        speedup = ref_ms / triton_ms if triton_ms > 0 else 1.0

        latencies.append(triton_ms)
        speedups.append(speedup)

        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(f"({M:>6}, {N1:>5}, {N2:>5}){' ':4} {ref_ms:>8.4f}ms {triton_ms:>8.4f}ms {speedup:>8.2f}x{marker}", flush=True)

    log_sum = sum(math.log(l) for l in latencies)
    geomean_latency = math.exp(log_sum / len(latencies))

    log_sum_speedup = sum(math.log(s) for s in speedups)
    geomean_speedup = math.exp(log_sum_speedup / len(speedups))

    print("-" * 62)
    print(f"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {
        "geomean_latency_ms": geomean_latency,
        "geomean_speedup": geomean_speedup,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fused RMS + FP8 Kernel Test Harness")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness tests on benchmark shapes",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run minimal profiling workload"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark on HARNESS_SHAPES (25 uniformly sampled)",
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run benchmark on ALL_SHAPES (complete set)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup iterations (default: 50)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
        help="Number of benchmark iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("Fused RMSNorm + FP8 Quantization Kernel")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        run_correctness(HARNESS_SHAPES)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)