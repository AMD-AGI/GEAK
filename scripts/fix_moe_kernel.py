#!/usr/bin/env python3
"""Fix moe_routing_sigmoid_top1 kernel.py: benchmark with speedup and default mode."""

import math

path = "/home/sdubagun/work/repos/AIG-Eval/tasks/geak_eval/moe_routing_sigmoid_top1/kernel.py"
with open(path) as f:
    lines = f.readlines()

benchmark_start = None
main_start = None
main_end = None

for i, line in enumerate(lines):
    if line.startswith("def run_benchmark("):
        benchmark_start = i
    if line.startswith("def main("):
        main_start = i
    if line.startswith("if __name__"):
        main_end = i

print(f"benchmark: {benchmark_start}, main: {main_start}, __name__: {main_end}")

new_benchmark_lines = [
    'def run_benchmark(shapes, warmup=5, iters=20):\n',
    '    """Benchmark kernel vs reference; report per-shape speedups and geo-mean."""\n',
    '    torch.manual_seed(42)\n',
    '    device = "cuda"\n',
    '    dtype = torch.bfloat16\n',
    '    TOPK = 1\n',
    '\n',
    '    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")\n',
    '\n',
    '    latencies = []\n',
    '    speedups = []\n',
    '\n',
    "    print(f\"{'Config':<30} {'Reference':>10} {'Kernel':>10} {'Speedup':>10}\")\n",
    '    print("-" * 62)\n',
    '\n',
    '    for i, (M, N, K) in enumerate(shapes):\n',
    '        x = torch.randn((M, K), dtype=dtype, device=device)\n',
    '        w = torch.randn((K, N), dtype=dtype, device=device) * 0.1\n',
    '        dummy_ids = torch.ones((M, 1), dtype=torch.int32, device=device) * N\n',
    '        dummy_weights = torch.ones((M, 1), dtype=torch.float32, device=device)\n',
    '\n',
    '        _eager = partial(\n',
    '            torch_routing_sigmoid_top1, dummy_ids=dummy_ids, dummy_weights=dummy_weights\n',
    '        )\n',
    '\n',
    '        for _ in range(warmup):\n',
    '            routing_sigmoid_top1(x, w, TOPK, fused_shared_experts=True)\n',
    '        torch.cuda.synchronize()\n',
    '\n',
    '        kernel_times = []\n',
    '        for _ in range(iters):\n',
    '            start = torch.cuda.Event(enable_timing=True)\n',
    '            end = torch.cuda.Event(enable_timing=True)\n',
    '            start.record()\n',
    '            routing_sigmoid_top1(x, w, TOPK, fused_shared_experts=True)\n',
    '            end.record()\n',
    '            torch.cuda.synchronize()\n',
    '            kernel_times.append(start.elapsed_time(end))\n',
    '\n',
    '        kernel_ms = sorted(kernel_times)[len(kernel_times) // 2]\n',
    '\n',
    '        for _ in range(warmup):\n',
    '            _eager(x, w, TOPK, fused_shared_experts=True)\n',
    '        torch.cuda.synchronize()\n',
    '\n',
    '        ref_times = []\n',
    '        for _ in range(iters):\n',
    '            start = torch.cuda.Event(enable_timing=True)\n',
    '            end = torch.cuda.Event(enable_timing=True)\n',
    '            start.record()\n',
    '            _eager(x, w, TOPK, fused_shared_experts=True)\n',
    '            end.record()\n',
    '            torch.cuda.synchronize()\n',
    '            ref_times.append(start.elapsed_time(end))\n',
    '\n',
    '        ref_ms = sorted(ref_times)[len(ref_times) // 2]\n',
    '\n',
    '        speedup = ref_ms / kernel_ms if kernel_ms > 0 else float("inf")\n',
    '        speedups.append(speedup)\n',
    '        latencies.append(kernel_ms)\n',
    '\n',
    '        marker = " *" if speedup > 1.0 else ""\n',
    '        shape_str = f"M={M}, N={N}, K={K}"\n',
    '        print(f"  {shape_str:<28} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}")\n',
    '\n',
    '    log_sum = sum(math.log(t) for t in latencies)\n',
    '    geomean_latency = math.exp(log_sum / len(latencies))\n',
    '\n',
    '    log_sum_speedup = sum(math.log(s) for s in speedups)\n',
    '    geomean_speedup = math.exp(log_sum_speedup / len(speedups))\n',
    '\n',
    '    print("-" * 62)\n',
    "    print(f\"{'Geometric mean latency:':<22} {geomean_latency:.4f} ms\")\n",
    "    print(f\"{'Geometric mean speedup:':<22} {geomean_speedup:.2f}x\")\n",
    '    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}")\n',
    '    print(f"GEAK_RESULT_SPEEDUP={geomean_speedup:.2f}")\n',
    '\n',
    '\n',
]

new_main_lines = [
    'def main():\n',
    '    parser = argparse.ArgumentParser(description="Test harness for moe_routing_sigmoid_top1_fused kernel")\n',
    '    parser.add_argument("--correctness", action="store_true", help="Run correctness tests")\n',
    '    parser.add_argument("--profile", action="store_true", help="Run kernel once for profiling")\n',
    '    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on HARNESS_SHAPES")\n',
    '    parser.add_argument("--full-benchmark", action="store_true", help="Run benchmark on ALL_SHAPES")\n',
    '    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations (default: 5)")\n',
    '    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations (default: 20)")\n',
    '\n',
    '    args = parser.parse_args()\n',
    '\n',
    '    if args.correctness:\n',
    '        success = run_correctness(HARNESS_SHAPES)\n',
    '        sys.exit(0 if success else 1)\n',
    '    elif args.profile:\n',
    '        run_profile(PROFILE_SHAPES)\n',
    '    elif args.full_benchmark:\n',
    '        print("\\n[Full Benchmark Mode]")\n',
    '        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)\n',
    '    elif args.benchmark:\n',
    '        print("\\n[Benchmark Mode]")\n',
    '        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)\n',
    '    else:\n',
    '        print("\\n[Benchmark Mode]")\n',
    '        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)\n',
    '\n',
    '\n',
]

result = lines[:benchmark_start] + new_benchmark_lines + new_main_lines + lines[main_end:]

with open(path, "w") as f:
    f.writelines(result)

print("Done! Replaced run_benchmark and main.")
