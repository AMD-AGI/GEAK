#!/usr/bin/env python3
"""GEAK-compatible harness for an official TileLang benchmark GEMM derivative."""

from __future__ import annotations

import argparse
import functools
import math
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
for raw_path in (
    str(SCRIPT_DIR),
    os.environ.get("GEAK_REPO_ROOT", ""),
    os.environ.get("GEAK_WORK_DIR", ""),
):
    if raw_path:
        while raw_path in sys.path:
            sys.path.remove(raw_path)
        sys.path.insert(0, raw_path)

from kernel import (  # noqa: E402
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    ENABLE_RASTERIZATION,
    K_PACK,
    NUM_STAGES,
    THREADS,
    matmul_kernel,
)

BENCHMARK_SHAPES = [(2048, 2048, 2048)]
FULL_BENCHMARK_SHAPES = [(2048, 2048, 2048), (4096, 1024, 4096)]
PROFILE_SHAPE = (2048, 2048, 2048)
DEFAULT_ITERATIONS = 20
DEFAULT_WARMUP = 5


def _seed(seed: int = 123) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_input(m: int, n: int, k: int, seed: int = 123) -> tuple[torch.Tensor, torch.Tensor]:
    _seed(seed)
    a_cpu = torch.randn((m, k), dtype=torch.float16)
    b_cpu = torch.randn((n, k), dtype=torch.float16)
    return a_cpu.cuda(), b_cpu.cuda()


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.T


@functools.cache
def _compiled_kernel(m: int, n: int, k: int):
    return matmul_kernel(m, n, k, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, THREADS, ENABLE_RASTERIZATION, K_PACK)


def _run_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _compiled_kernel(a.shape[0], b.shape[0], a.shape[1])(a, b)


def run_correctness() -> None:
    for idx, (m, n, k) in enumerate(BENCHMARK_SHAPES):
        a, b = _make_input(m, n, k, 123 + idx)
        out = _run_kernel(a, b)
        ref = _reference(a, b)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        print(f"PASS shape=({m},{n},{k}) max_abs_diff={(out - ref).abs().max().item():.6e}")
    print("CORRECTNESS: PASS")


def run_profile(iterations: int) -> None:
    a, b = _make_input(*PROFILE_SHAPE, seed=987)
    _run_kernel(a, b)
    torch.cuda.synchronize()
    for _ in range(iterations):
        _run_kernel(a, b)
    torch.cuda.synchronize()
    print(f"PROFILE: PASS iterations={iterations}")


def _median_ms(fn, iterations: int) -> float:
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return float(times[len(times) // 2])


def run_benchmark(shapes: list[tuple[int, int, int]], iterations: int, warmup: int) -> dict[str, float]:
    latencies = []
    speedups = []
    for idx, (m, n, k) in enumerate(shapes):
        a, b = _make_input(m, n, k, 456 + idx)
        _run_kernel(a, b)
        torch.cuda.synchronize()
        for _ in range(warmup):
            _run_kernel(a, b)
        torch.cuda.synchronize()

        kernel_ms = _median_ms(lambda: _run_kernel(a, b), iterations)
        ref_ms = _median_ms(lambda: _reference(a, b), iterations)
        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0.0
        tflops = (2.0 * m * n * k) / (kernel_ms * 1e-3) / 1e12
        latencies.append(kernel_ms)
        speedups.append(max(speedup, 1e-9))
        print(
            f"shape=({m},{n},{k}) reference_ms={ref_ms:.6f} "
            f"tilelang_ms={kernel_ms:.6f} speedup={speedup:.4f}x tflops={tflops:.3f}"
        )

    geomean_latency = math.exp(sum(math.log(max(v, 1e-9)) for v in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(v) for v in speedups) / len(speedups))
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.6f}")
    print(f"GEAK_RESULT_SPEEDUP={geomean_speedup:.6f}")
    return {"latency_ms": geomean_latency, "speedup": geomean_speedup}


def _iteration_count(cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    for name in ("GEAK_BENCHMARK_ITERATIONS", "GEAK_EVAL_BENCHMARK_ITERATIONS"):
        value = os.getenv(name)
        if value:
            return int(value)
    return DEFAULT_ITERATIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA/HIP device is not visible", file=sys.stderr)
        return 2

    iterations = _iteration_count(args.iterations)
    if args.correctness:
        run_correctness()
    elif args.profile:
        run_profile(iterations)
    elif args.full_benchmark:
        run_benchmark(FULL_BENCHMARK_SHAPES, iterations, args.warmup)
    else:
        run_benchmark(BENCHMARK_SHAPES, iterations, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
