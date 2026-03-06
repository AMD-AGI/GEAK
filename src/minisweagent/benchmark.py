"""
Standardized benchmarking framework for GEAK agent.
Internal module - used automatically by agent when benchmarking kernels.

Always outputs to: <kernel_dir>/benchmark/baseline/metrics.json
Always includes: latency, throughput, FLOPS, bandwidth (when calculable)
"""

import datetime
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StandardMetrics:
    """Standard metrics for any kernel - always present for baseline."""

    # Timing (always required)
    latency_ms: float
    latency_us: float
    latency_min_ms: float
    latency_max_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_std_ms: float

    # Throughput (always required)
    throughput_ops_per_sec: float

    # Compute (if calculable)
    flops: float | None = None
    tflops: float | None = None

    # Memory (if calculable)
    bandwidth_gb_s: float | None = None
    memory_mb: float | None = None

    # Metadata
    num_iterations: int = 1000
    num_warmup: int = 100


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a kernel."""

    kernel_name: str
    kernel_path: str
    timestamp: str

    # Standard metrics (always present)
    baseline_metrics: StandardMetrics

    # Test case information
    test_cases: list[dict[str, Any]]

    # Additional kernel-specific data
    kernel_info: dict[str, Any]

    # Environment info
    device_info: dict[str, str]

    # Correctness
    all_tests_passed: bool


class StandardBenchmark:
    """
    Standardized benchmark runner.
    Works with any kernel type and always produces consistent metrics.
    """

    def __init__(
        self, kernel_path: Path, output_dir: Path | None = None, warmup_iters: int = 100, benchmark_iters: int = 1000
    ):
        self.kernel_path = Path(kernel_path)
        self.output_dir = output_dir or self.kernel_path.parent / "benchmark"
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters

        # Standard output location
        self.baseline_dir = self.output_dir / "baseline"
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.baseline_dir / "metrics.json"

    def measure_timing(self, func: Callable, *args, **kwargs) -> dict[str, float]:
        """
        Measure timing statistics for any function.
        Returns standardized timing metrics.
        """
        times_ms = []

        # Warmup
        for _ in range(self.warmup_iters):
            try:
                _ = func(*args, **kwargs)
            except Exception:
                pass

        # Benchmark
        for _ in range(self.benchmark_iters):
            try:
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                times_ms.append((end - start) * 1000)  # Convert to ms
            except Exception as e:
                print(f"Warning: Benchmark iteration failed: {e}")
                continue

        if not times_ms:
            raise RuntimeError("All benchmark iterations failed")

        times_ms = np.array(times_ms)

        return {
            "latency_ms": float(np.mean(times_ms)),
            "latency_us": float(np.mean(times_ms) * 1000),
            "latency_min_ms": float(np.min(times_ms)),
            "latency_max_ms": float(np.max(times_ms)),
            "latency_p50_ms": float(np.median(times_ms)),
            "latency_p95_ms": float(np.percentile(times_ms, 95)),
            "latency_p99_ms": float(np.percentile(times_ms, 99)),
            "latency_std_ms": float(np.std(times_ms)),
            "throughput_ops_per_sec": 1.0 / (np.mean(times_ms) / 1000) if np.mean(times_ms) > 0 else 0.0,
            "num_iterations": self.benchmark_iters,
            "num_warmup": self.warmup_iters,
        }

    def calculate_flops(
        self, kernel_info: dict[str, Any], test_case: dict[str, Any], latency_s: float
    ) -> dict[str, float | None]:
        """
        Calculate FLOPS metrics if possible.
        Returns None if not calculable for this kernel type.
        """
        # Try to infer operations from kernel info
        flops = None

        # Example: For element-wise ops (add, mul, etc.)
        if "operation_count" in kernel_info:
            flops = kernel_info["operation_count"]
        elif "n_elements" in test_case:
            # Assume 1 FLOP per element for simple ops
            flops = test_case["n_elements"]

        if flops and latency_s > 0:
            tflops = (flops / latency_s) / 1e12
            return {"flops": float(flops), "tflops": float(tflops)}

        return {"flops": None, "tflops": None}

    def calculate_bandwidth(self, test_case: dict[str, Any], latency_s: float) -> float | None:
        """
        Calculate memory bandwidth if possible.
        """
        if "memory_bytes" in test_case and latency_s > 0:
            bandwidth_gb_s = (test_case["memory_bytes"] / latency_s) / 1e9
            return float(bandwidth_gb_s)
        return None

    def benchmark_kernel(
        self, kernel_func: Callable, test_cases: list[dict[str, Any]], kernel_info: dict[str, Any] | None = None
    ) -> BenchmarkResult:
        """
        Benchmark a kernel with discovered test cases.
        Always produces standardized metrics.
        """
        import datetime

        kernel_info = kernel_info or {}

        # Use first test case as baseline (typically medium size)
        baseline_test = test_cases[0] if test_cases else {}

        # Extract args/kwargs from test case
        args = baseline_test.get("args", [])
        kwargs = baseline_test.get("kwargs", {})

        # Measure timing
        timing_metrics = self.measure_timing(kernel_func, *args, **kwargs)

        # Calculate FLOPS
        latency_s = timing_metrics["latency_ms"] / 1000
        flops_metrics = self.calculate_flops(kernel_info, baseline_test, latency_s)

        # Calculate bandwidth
        bandwidth = self.calculate_bandwidth(baseline_test, latency_s)

        # Create standardized metrics
        baseline_metrics = StandardMetrics(
            latency_ms=timing_metrics["latency_ms"],
            latency_us=timing_metrics["latency_us"],
            latency_min_ms=timing_metrics["latency_min_ms"],
            latency_max_ms=timing_metrics["latency_max_ms"],
            latency_p50_ms=timing_metrics["latency_p50_ms"],
            latency_p95_ms=timing_metrics["latency_p95_ms"],
            latency_p99_ms=timing_metrics["latency_p99_ms"],
            latency_std_ms=timing_metrics["latency_std_ms"],
            throughput_ops_per_sec=timing_metrics["throughput_ops_per_sec"],
            flops=flops_metrics.get("flops"),
            tflops=flops_metrics.get("tflops"),
            bandwidth_gb_s=bandwidth,
            memory_mb=baseline_test.get("memory_mb"),
            num_iterations=self.benchmark_iters,
            num_warmup=self.warmup_iters,
        )

        # Device info (try to get GPU info if available)
        device_info = self._get_device_info()

        return BenchmarkResult(
            kernel_name=self.kernel_path.stem,
            kernel_path=str(self.kernel_path),
            timestamp=datetime.datetime.now().isoformat(),
            baseline_metrics=baseline_metrics,
            test_cases=test_cases,
            kernel_info=kernel_info,
            device_info=device_info,
            all_tests_passed=True,  # Updated by test runner
        )

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.datetime.now().isoformat()

    def _get_device_info(self) -> dict[str, str]:
        """Get device information (GPU, CPU, etc.)."""
        device_info = {"type": "unknown", "name": "unknown"}

        try:
            import torch

            if torch.cuda.is_available():
                device_info["type"] = "cuda"
                device_info["name"] = torch.cuda.get_device_name(0)
                device_info["compute_capability"] = str(torch.cuda.get_device_capability(0))
                device_info["memory_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}"
        except ImportError:
            pass

        return device_info

    def save_metrics(self, result: BenchmarkResult) -> Path:
        """
        Save standardized metrics to benchmark/baseline/metrics.json
        """
        # Convert to dict
        result_dict = {
            "kernel_name": result.kernel_name,
            "kernel_path": result.kernel_path,
            "timestamp": result.timestamp,
            "baseline_metrics": asdict(result.baseline_metrics),
            "test_cases": result.test_cases,
            "kernel_info": result.kernel_info,
            "device_info": result.device_info,
            "all_tests_passed": result.all_tests_passed,
        }

        # Save to standard location
        with open(self.metrics_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        print(f"✓ Baseline metrics saved to: {self.metrics_path}")
        return self.metrics_path

    def load_metrics(self) -> BenchmarkResult | None:
        """Load metrics from standard location."""
        if not self.metrics_path.exists():
            return None

        with open(self.metrics_path) as f:
            data = json.load(f)

        baseline_metrics = StandardMetrics(**data["baseline_metrics"])
        return BenchmarkResult(
            kernel_name=data["kernel_name"],
            kernel_path=data["kernel_path"],
            timestamp=data["timestamp"],
            baseline_metrics=baseline_metrics,
            test_cases=data["test_cases"],
            kernel_info=data["kernel_info"],
            device_info=data["device_info"],
            all_tests_passed=data["all_tests_passed"],
        )

    def print_metrics(self, result: BenchmarkResult):
        """Print standardized metrics in human-readable format."""
        print("\n" + "=" * 80)
        print("BASELINE METRICS")
        print("=" * 80)
        print(f"Kernel: {result.kernel_name}")
        print(f"Path: {result.kernel_path}")
        print(f"Device: {result.device_info.get('name', 'unknown')}")
        print(f"Timestamp: {result.timestamp}")
        print()

        m = result.baseline_metrics
        print("LATENCY:")
        print(f"  Mean:   {m.latency_ms:.4f} ms ({m.latency_us:.2f} μs)")
        print(f"  Median: {m.latency_p50_ms:.4f} ms")
        print(f"  P95:    {m.latency_p95_ms:.4f} ms")
        print(f"  P99:    {m.latency_p99_ms:.4f} ms")
        print(f"  Min:    {m.latency_min_ms:.4f} ms")
        print(f"  Max:    {m.latency_max_ms:.4f} ms")
        print(f"  Std:    {m.latency_std_ms:.4f} ms")
        print()

        print("THROUGHPUT:")
        print(f"  {m.throughput_ops_per_sec:.2f} ops/sec")
        print()

        if m.tflops is not None:
            print("COMPUTE:")
            print(f"  {m.tflops:.4f} TFLOPS")
            print(f"  {m.flops:.2e} FLOPS total")
            print()

        if m.bandwidth_gb_s is not None:
            print("MEMORY:")
            print(f"  {m.bandwidth_gb_s:.2f} GB/s bandwidth")
            if m.memory_mb:
                print(f"  {m.memory_mb:.2f} MB used")
            print()

        print("ITERATIONS:")
        print(f"  Warmup: {m.num_warmup}")
        print(f"  Benchmark: {m.num_iterations}")
        print()

        print("=" * 80)


# Main API - used internally by GEAK agent
__all__ = ["StandardBenchmark", "StandardMetrics", "BenchmarkResult"]


"""
AGENT USAGE:
-----------
When the agent benchmarks a kernel, it should:

1. Create StandardBenchmark instance:
   benchmark = StandardBenchmark(kernel_path)

2. Run benchmark with discovered test cases:
   result = benchmark.benchmark_kernel(
       kernel_func=discovered_kernel_func,
       test_cases=discovered_test_cases,
       kernel_info={'operation': 'add', 'block_size': 1024}
   )

3. Save to standard location (benchmark/baseline/metrics.json):
   benchmark.save_metrics(result)

4. Optionally print results:
   benchmark.print_metrics(result)

This ensures ALL kernels get consistent metrics stored in the same format and location.
"""
