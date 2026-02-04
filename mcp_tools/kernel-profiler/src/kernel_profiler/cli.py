"""CLI wrapper for kernel-profiler MCP tools.

Enables calling MCP tools via bash commands.

Usage:
    kernel-profiler profile <kernel_file> [--function <name>] [--gpu <id>]
    kernel-profiler benchmark <kernel_file> [--function <name>] [--gpu <id>]
    kernel-profiler roofline --latency <us> [--flops <n>] [--bytes <n>]
    kernel-profiler suggestions <bottleneck>
"""

import argparse
import json
import sys

from .server import (
    profile_kernel,
    benchmark_kernel,
    get_roofline_analysis,
    get_bottleneck_suggestions,
)


def main():
    parser = argparse.ArgumentParser(
        prog="kernel-profiler",
        description="GPU kernel profiling tools for AMD GPUs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # profile command
    profile_parser = subparsers.add_parser("profile", help="Profile a GPU kernel")
    profile_parser.add_argument("kernel_file", help="Path to kernel file")
    profile_parser.add_argument("--function", "-f", default="run_baseline", help="Function to profile")
    profile_parser.add_argument("--gpu", "-g", default="0", help="GPU device ID")
    profile_parser.add_argument("--warmup", "-w", type=int, default=100, help="Warmup iterations")
    profile_parser.add_argument("--iters", "-i", type=int, default=100, help="Profile iterations")
    profile_parser.add_argument("--docker-image", help="Docker image to use")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Quick latency benchmark")
    bench_parser.add_argument("kernel_file", help="Path to kernel file")
    bench_parser.add_argument("--function", "-f", default="run_baseline", help="Function to benchmark")
    bench_parser.add_argument("--gpu", "-g", default="0", help="GPU device ID")
    bench_parser.add_argument("--warmup", "-w", type=int, default=1000, help="Warmup iterations")
    bench_parser.add_argument("--iters", "-i", type=int, default=3000, help="Benchmark iterations")
    bench_parser.add_argument("--docker-image", help="Docker image to use")

    # roofline command
    roof_parser = subparsers.add_parser("roofline", help="Roofline analysis")
    roof_parser.add_argument("--latency", "-l", type=float, required=True, help="Latency in microseconds")
    roof_parser.add_argument("--flops", type=int, default=0, help="FLOPs per kernel call")
    roof_parser.add_argument("--bytes", type=int, default=0, help="Bytes transferred per call")

    # suggestions command
    sugg_parser = subparsers.add_parser("suggestions", help="Get optimization suggestions")
    sugg_parser.add_argument("bottleneck", choices=["latency", "memory", "compute", "lds", "cache", "balanced"],
                             help="Bottleneck type")

    args = parser.parse_args()

    if args.command == "profile":
        result = profile_kernel(
            kernel_file=args.kernel_file,
            function_name=args.function,
            gpu_device=args.gpu,
            warmup_iters=args.warmup,
            profile_iters=args.iters,
            docker_image=args.docker_image
        )
    elif args.command == "benchmark":
        result = benchmark_kernel(
            kernel_file=args.kernel_file,
            function_name=args.function,
            gpu_device=args.gpu,
            warmup_iters=args.warmup,
            bench_iters=args.iters,
            docker_image=args.docker_image
        )
    elif args.command == "roofline":
        result = get_roofline_analysis(
            latency_us=args.latency,
            flops=args.flops,
            bytes_transferred=args.bytes
        )
    elif args.command == "suggestions":
        result = get_bottleneck_suggestions(bottleneck=args.bottleneck)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
