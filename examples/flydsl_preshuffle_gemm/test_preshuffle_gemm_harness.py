#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Self-contained test harness for FlyDSL preshuffle GEMM kernel.
# Supports --correctness, --benchmark, --full-benchmark, --profile.
#
# Focuses on FP8 per-token quantised GEMM (the primary optimisation target).
# Dependencies: flydsl, torch (with ROCm).

import argparse
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
_REPO_ROOT = os.environ.get("GEAK_REPO_ROOT", SCRIPT_DIR)
_WORK_DIR = os.environ.get("GEAK_WORK_DIR", "")

for p in [_WORK_DIR, _REPO_ROOT, SCRIPT_DIR]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

import flydsl.compiler as flyc
import torch
from flydsl.runtime.device import get_rocm_arch
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

ARCH = str(get_rocm_arch())
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz

# (M, N, K, tile_m, tile_n, tile_k, in_dtype)
ALL_CONFIGS = [
    (16, 5120, 8192, 16, 64, 512, "fp8"),
    (33, 1024, 2048, 32, 64, 512, "fp8"),
    (5120, 5120, 8320, 64, 256, 128, "fp8"),
    (5120, 2048, 8320, 128, 128, 128, "fp8"),
    (9728, 8192, 8320, 128, 128, 128, "fp8"),
    (5133, 5120, 8320, 64, 256, 128, "fp8"),
    (16, 5120, 8192, 16, 64, 512, "bf16"),
    (5120, 5120, 8320, 64, 256, 128, "bf16"),
]


# ---------------------------------------------------------------------------
# Inlined helpers (from tests.utils) to keep harness self-contained
# ---------------------------------------------------------------------------


def _get_dtype_max(dtype):
    try:
        return torch.finfo(dtype).max
    except Exception:
        return torch.iinfo(dtype).max


def pertoken_quant(x, quant_dtype=torch.int8, dtypeMax=None):
    x = x.to(torch.float32)
    if dtypeMax is None:
        dtypeMax = _get_dtype_max(quant_dtype)
    x = torch.nan_to_num(x, nan=0.0, posinf=float(dtypeMax), neginf=-float(dtypeMax))
    per_token_max = torch.amax(x, dim=-1, keepdim=True)
    per_token_min = torch.amin(x, dim=-1, keepdim=True)
    per_token_amax = torch.maximum(per_token_max, -per_token_min)
    per_token_scale = per_token_amax / dtypeMax
    per_token_scale[per_token_scale == 0] = 1
    per_token_scale = torch.nan_to_num(per_token_scale, nan=1.0, posinf=1.0, neginf=1.0)
    y = (x / per_token_scale).to(dtype=quant_dtype)
    return y, per_token_scale.to(torch.float32)


def shuffle_weight(x, layout=(16, 16)):
    x_type = x.dtype
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size()
    BN = IN
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    return x_.permute(0, 1, 3, 4, 2, 5).contiguous().view(*x.shape).view(x_type)


def run_torch_ref(a, b, scale_a, scale_b):
    if scale_a is not None and scale_b is not None:
        a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
        b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    else:
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
    return torch.mm(a_f32, b_f32.T)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs))), configs
    n = len(configs)
    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    return indices, [configs[i] for i in indices]


def config_str(cfg):
    M, N, K, tm, tn, tk, dt = cfg
    return f"M={M} N={N} K={K} tile={tm}x{tn}x{tk} dtype={dt}"


def _build_and_prepare(M, N, K, tile_m, tile_n, tile_k, in_dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")
    torch_out_dtype = torch.bfloat16

    launch_fn = compile_preshuffle_gemm_a8(
        M=M,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype="bf16",
        lds_stage=2,
    )

    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)

    if in_dtype in ("fp16", "bf16"):
        torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        a_q = a_fp32.to(torch_dtype)
        b_q = b_fp32_t.to(torch_dtype)
        scale_a, scale_b = None, None
    else:
        quant_dtype = torch.int8 if in_dtype == "int8" else DTYPE_FP8
        a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=quant_dtype)
        b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=quant_dtype)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    b_shuffled = shuffle_weight(b_q, layout=(16, 16))

    c_ref = run_torch_ref(a_q, b_q, scale_a, scale_b)
    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    if scale_a is None:
        sa_flat = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        sa_flat = scale_a.contiguous().view(-1)
    if scale_b is None:
        sb_flat = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        sb_flat = scale_b.contiguous().view(-1)

    def _gemm_args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            _as_i8(a.contiguous().view(-1)),
            _as_i8(b.contiguous().view(-1)),
            sa.contiguous().view(-1) if sa.numel() > 0 else sa,
            sb.contiguous().view(-1) if sb.numel() > 0 else sb,
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled_fn = flyc.compile(launch_fn, *_gemm_args(c_out, a_q, b_shuffled, sa_flat, sb_flat))

    def kernel_fn():
        compiled_fn(*_gemm_args(c_out, a_q, b_shuffled, sa_flat, sb_flat))

    return kernel_fn, c_out, c_ref


def check_correctness(c_out, c_ref, rtol=0.1, atol=0.1):
    c_f32 = c_out.to(torch.float32)
    close = torch.isclose(c_f32, c_ref, rtol=rtol, atol=atol)
    if close.all():
        return True, 0.0
    err_ratio = (~close).sum().item() / c_ref.numel()
    return err_ratio < 0.05, err_ratio


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_correctness(configs, indices):
    failures = 0
    for idx, cfg in zip(indices, configs):
        M, N, K, tm, tn, tk, dt = cfg
        label = config_str(cfg)
        print(f"[Correctness] Config idx={idx}: {label}")
        try:
            kernel_fn, c_out, c_ref = _build_and_prepare(M, N, K, tm, tn, tk, dt)
            kernel_fn()
            torch.cuda.synchronize()
            passed, err_ratio = check_correctness(c_out, c_ref)
            if passed:
                print(f"  PASSED (err_ratio={err_ratio:.4f})")
            else:
                print(f"  FAILED (err_ratio={err_ratio:.4f})")
                failures += 1
        except Exception as e:
            print(f"  FAILED with exception: {type(e).__name__}: {e}")
            failures += 1
        finally:
            torch.cuda.empty_cache()

    print(f"\nGEAK_SHAPES_USED={indices}")
    if failures > 0:
        print(f"\n{failures} correctness test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll correctness tests PASSED")


def run_benchmark(configs, indices, label="benchmark"):
    latencies = []
    for idx, cfg in zip(indices, configs):
        M, N, K, tm, tn, tk, dt = cfg
        clabel = config_str(cfg)
        print(f"[{label}] Config idx={idx}: {clabel}", end="  ")
        try:
            kernel_fn, _, _ = _build_and_prepare(M, N, K, tm, tn, tk, dt)

            for _ in range(WARMUP):
                kernel_fn()
            torch.cuda.synchronize()

            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(ITERATIONS)]
            for i in range(ITERATIONS):
                start_events[i].record()
                kernel_fn()
                end_events[i].record()
            torch.cuda.synchronize()

            times_ms = sorted(start_events[i].elapsed_time(end_events[i]) for i in range(ITERATIONS))
            median_ms = times_ms[len(times_ms) // 2]

            flops = 2 * M * N * K
            tflops = flops / (median_ms / 1e3) / 1e12
            latencies.append(median_ms)
            print(f"{median_ms:.4f}ms  {tflops:.2f} TFLOPS")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            latencies.append(float("inf"))
        finally:
            torch.cuda.empty_cache()

    print(f"\nGEAK_SHAPES_USED={indices}")
    finite = [l for l in latencies if l != float("inf") and l > 0]
    if finite:
        geo_mean = math.exp(sum(math.log(l) for l in finite) / len(finite))
        print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")
    else:
        print("No successful benchmarks")
        sys.exit(1)


def run_profile(configs, indices):
    for idx, cfg in zip(indices, configs):
        M, N, K, tm, tn, tk, dt = cfg
        label = config_str(cfg)
        print(f"[profile] Config idx={idx}: {label}")
        try:
            kernel_fn, _, _ = _build_and_prepare(M, N, K, tm, tn, tk, dt)
            for _ in range(3):
                kernel_fn()
            torch.cuda.synchronize()
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
        finally:
            torch.cuda.empty_cache()

    print(f"\nGEAK_SHAPES_USED={indices}")


def main():
    parser = argparse.ArgumentParser(description="Preshuffle GEMM kernel test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true", help="Run correctness checks")
    group.add_argument("--profile", action="store_true", help="Run configs for profiling")
    group.add_argument("--benchmark", action="store_true", help="Run sampled configs benchmark")
    group.add_argument("--full-benchmark", action="store_true", help="Run all configs benchmark")
    parser.add_argument("--iterations", type=int, default=None, help="Override number of benchmark iterations")
    args = parser.parse_args()

    global ITERATIONS
    if args.iterations is not None:
        ITERATIONS = args.iterations

    if args.correctness:
        indices, configs = _pick(ALL_CONFIGS, 25)
        run_correctness(configs, indices)
    elif args.profile:
        indices, configs = _pick(ALL_CONFIGS, 5)
        run_profile(configs, indices)
    elif args.benchmark:
        indices, configs = _pick(ALL_CONFIGS, 25)
        run_benchmark(configs, indices, label="benchmark")
    elif args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        run_benchmark(ALL_CONFIGS, indices, label="full-benchmark")


if __name__ == "__main__":
    main()
