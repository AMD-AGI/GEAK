"""CLI wrapper for kernel-evolve MCP tools.

Enables calling MCP tools via bash commands.

Usage:
    kernel-evolve generate <kernel_file> [--bottleneck <type>] [--strategy <name>]
    kernel-evolve mutate <kernel_file> [--type <mutation_type>] [--latency <us>] [--speedup <x>]
    kernel-evolve crossover <kernel1> <kernel2> [--speedup1 <x>] [--speedup2 <x>]
    kernel-evolve strategies <bottleneck>
    kernel-evolve params <kernel_type> [--size <n>]
"""

import argparse
import json
import sys
from pathlib import Path

from .server import (
    generate_optimization,
    mutate_kernel,
    crossover_kernels,
    get_optimization_strategies,
    suggest_kernel_params,
)


def main():
    parser = argparse.ArgumentParser(
        prog="kernel-evolve",
        description="Evolutionary GPU kernel optimization using LLM"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate optimized kernel")
    gen_parser.add_argument("kernel_file", help="Path to kernel file")
    gen_parser.add_argument("--bottleneck", "-b", default="balanced",
                            choices=["latency", "memory", "compute", "lds", "balanced"],
                            help="Bottleneck type")
    gen_parser.add_argument("--strategy", "-s", help="Specific strategy to apply")
    gen_parser.add_argument("--model", "-m", default="amd/claude-sonnet-4-20250514", help="LLM model")

    # mutate command
    mut_parser = subparsers.add_parser("mutate", help="Mutate existing kernel")
    mut_parser.add_argument("kernel_file", help="Path to kernel file")
    mut_parser.add_argument("--type", "-t", default="parameter",
                            choices=["parameter", "algorithm", "hybrid"],
                            help="Mutation type")
    mut_parser.add_argument("--latency", "-l", type=float, default=0.0, help="Current latency (us)")
    mut_parser.add_argument("--speedup", "-s", type=float, default=1.0, help="Current speedup")
    mut_parser.add_argument("--model", "-m", default="amd/claude-sonnet-4-20250514", help="LLM model")

    # crossover command
    cross_parser = subparsers.add_parser("crossover", help="Combine two kernels")
    cross_parser.add_argument("kernel1", help="Path to first kernel")
    cross_parser.add_argument("kernel2", help="Path to second kernel")
    cross_parser.add_argument("--speedup1", type=float, default=1.0, help="Speedup of kernel1")
    cross_parser.add_argument("--speedup2", type=float, default=1.0, help="Speedup of kernel2")
    cross_parser.add_argument("--model", "-m", default="amd/claude-sonnet-4-20250514", help="LLM model")

    # strategies command
    strat_parser = subparsers.add_parser("strategies", help="Get strategies for bottleneck")
    strat_parser.add_argument("bottleneck", choices=["latency", "memory", "compute", "lds", "balanced"],
                              help="Bottleneck type")

    # params command
    params_parser = subparsers.add_parser("params", help="Suggest kernel parameters")
    params_parser.add_argument("kernel_type", choices=["elementwise", "reduction", "matmul", "attention"],
                               help="Kernel type")
    params_parser.add_argument("--size", "-n", type=int, default=1048576, help="Problem size")

    args = parser.parse_args()

    if args.command == "generate":
        kernel_code = Path(args.kernel_file).read_text()
        result = generate_optimization(
            kernel_code=kernel_code,
            bottleneck=args.bottleneck,
            strategy=args.strategy,
            model=args.model
        )
    elif args.command == "mutate":
        kernel_code = Path(args.kernel_file).read_text()
        result = mutate_kernel(
            kernel_code=kernel_code,
            mutation_type=args.type,
            latency_us=args.latency,
            speedup=args.speedup,
            model=args.model
        )
    elif args.command == "crossover":
        kernel1 = Path(args.kernel1).read_text()
        kernel2 = Path(args.kernel2).read_text()
        result = crossover_kernels(
            kernel1=kernel1,
            kernel2=kernel2,
            speedup1=args.speedup1,
            speedup2=args.speedup2,
            model=args.model
        )
    elif args.command == "strategies":
        result = get_optimization_strategies(bottleneck=args.bottleneck)
    elif args.command == "params":
        result = suggest_kernel_params(kernel_type=args.kernel_type, problem_size=args.size)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
