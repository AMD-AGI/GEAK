"""CLI for GEAK Agent.

Usage:
    geak-agent <kernel_path> [options]
    geak-agent --help

Options:
    --test "command"     Explicit test command (skip discovery)
    --bench "command"    Explicit benchmark command (skip discovery)
    --gpu N              GPU device to use (default: 0)
    --no-confirm         Skip confirmation prompts
    --runtime TYPE       Runtime environment: local, docker, or auto (default: auto)
    --docker-image IMG   Docker image to use
"""

import argparse
import sys
from pathlib import Path

# Add parent src to path to import geakagent modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from .mcp_tools.discovery import DiscoveryPipeline, discover

from geakagent.runtime_env import (
    prompt_runtime_environment,
    display_runtime_info,
    RuntimeType,
    RuntimeEnvironment,
    DEFAULT_DOCKER_IMAGE,
)


def main():
    parser = argparse.ArgumentParser(
        description="GEAK Agent - GPU Evolutionary Agent for Kernel optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover tests automatically
  geak-agent /path/to/kernel.py

  # Provide explicit test command
  geak-agent /path/to/kernel.py --test "pytest test_kernel.py -v"

  # Provide both test and benchmark
  geak-agent /path/to/kernel.py --test "pytest test.py" --bench "python bench.py"

  # Skip confirmation prompts
  geak-agent /path/to/kernel.py --no-confirm
"""
    )
    
    parser.add_argument(
        "kernel_path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Path to kernel file or directory (default: current directory)"
    )
    
    parser.add_argument(
        "--test", "-t",
        dest="test_command",
        help="Explicit test command (e.g., 'pytest test_kernel.py -v')"
    )
    
    parser.add_argument(
        "--bench", "-b",
        dest="bench_command",
        help="Explicit benchmark command (e.g., 'python benchmark.py')"
    )
    
    parser.add_argument(
        "--gpu", "-g",
        default="0",
        help="GPU device to use (default: 0)"
    )
    
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only run discovery, don't start agent loop"
    )
    
    parser.add_argument(
        "--runtime", "-r",
        choices=["local", "docker", "auto"],
        default="auto",
        help="Runtime environment (default: auto)"
    )
    
    parser.add_argument(
        "--docker-image",
        help=f"Docker image to use (default: {DEFAULT_DOCKER_IMAGE})"
    )
    
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Workspace directory to mount in Docker (default: auto-detected)"
    )
    
    parser.add_argument(
        "--no-runtime-check",
        action="store_true",
        help="Skip runtime environment detection"
    )
    
    args = parser.parse_args()
    
    # Run discovery pipeline
    print("\n" + "=" * 60)
    print("  GEAK Agent - GPU Evolutionary Agent for Kernels")
    print("=" * 60)
    print(f"\n  Kernel path: {args.kernel_path}")
    print(f"  GPU device: {args.gpu}")
    
    # Runtime environment detection
    runtime_env = None
    if not args.no_runtime_check:
        print("\n" + "=" * 60)
        print("  RUNTIME ENVIRONMENT")
        print("=" * 60)
        
        if args.runtime == "local":
            # Force local environment
            runtime_env = RuntimeEnvironment(runtime_type=RuntimeType.LOCAL)
            display_runtime_info(runtime_env)
        elif args.runtime == "docker":
            # Force Docker environment
            image = args.docker_image or DEFAULT_DOCKER_IMAGE
            runtime_env = RuntimeEnvironment(
                runtime_type=RuntimeType.DOCKER,
                docker_image=image,
                docker_devices=["/dev/kfd", "/dev/dri"],
                has_gpu=True,
                has_triton=True,
                has_torch=True
            )
            display_runtime_info(runtime_env)
        else:
            # Auto-detect (default)
            runtime_env = prompt_runtime_environment(auto_confirm=args.no_confirm)
    
    # Determine workspace
    if args.kernel_path.is_file():
        workspace = args.kernel_path.parent
        kernel_path = args.kernel_path
    else:
        workspace = args.kernel_path
        kernel_path = None
    
    # Run discovery
    result = discover(
        workspace=workspace,
        kernel_path=kernel_path,
        test_command=args.test_command,
        bench_command=args.bench_command,
        interactive=not args.no_confirm
    )
    
    if args.discover_only:
        print("\n  Discovery complete. Exiting (--discover-only).")
        return
    
    # Get commands
    test_cmd = result.user_provided_test or (result.tests[0].command if result.tests else None)
    bench_cmd = result.user_provided_bench or (result.benchmarks[0].command if result.benchmarks else None)
    
    if not test_cmd and not bench_cmd:
        print("\n  ERROR: No tests or benchmarks found.")
        print("  Please provide --test or --bench, or run in a directory with tests.")
        return
    
    # Build context for agent
    workspace_path = args.workspace or workspace
    context = {
        "workspace": str(workspace_path),
        "kernel_path": str(kernel_path) if kernel_path else None,
        "test_command": test_cmd,
        "bench_command": bench_cmd,
        "gpu_device": args.gpu,
        "runtime_env": {
            "type": runtime_env.runtime_type.value if runtime_env else "local",
            "docker_image": runtime_env.docker_image if runtime_env else None,
            "has_gpu": runtime_env.has_gpu if runtime_env else False,
        } if runtime_env else None,
        "kernels": [
            {
                "name": k.kernel_name,
                "file": str(k.file_path),
                "type": k.kernel_type,
                "functions": k.function_names
            }
            for k in result.kernels
        ]
    }
    
    print("\n" + "=" * 60)
    print("  AGENT CONTEXT")
    print("=" * 60)
    print(f"\n  Workspace: {context['workspace']}")
    print(f"  Test command: {context['test_command']}")
    print(f"  Bench command: {context['bench_command']}")
    print(f"  GPU: {context['gpu_device']}")
    print(f"  Runtime: {context['runtime_env']['type'] if context['runtime_env'] else 'local'}")
    if context['runtime_env'] and context['runtime_env'].get('docker_image'):
        print(f"  Docker Image: {context['runtime_env']['docker_image']}")
    print(f"  Kernels: {len(context['kernels'])}")
    
    # TODO: Start agent loop
    print("\n" + "=" * 60)
    print("  AGENT LOOP")
    print("=" * 60)
    print("\n  (Agent loop not yet implemented)")
    print("  Context is ready for the agent to use.")
    print("\n  The agent will:")
    print("    1. Run bash commands to test/benchmark")
    print("    2. Call MCP tools (kernel-ercs, kernel-profiler, kernel-evolve)")
    print("    3. Decide next actions based on results")
    print("    4. Iterate until target speedup achieved")


if __name__ == "__main__":
    main()
