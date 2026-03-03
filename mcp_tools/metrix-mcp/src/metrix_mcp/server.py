"""
Metrix MCP Server - GPU Kernel Profiling

Provides GPU kernel profiling using AMD Metrix with hardware metrics,
bottleneck classification, and factual observations.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .core import MetrixTool

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create MCP server
mcp = FastMCP(
    name="metrix-profiler",
    instructions="GPU kernel profiling using AMD Metrix profiler with hardware metrics and bottleneck analysis",
)


def _profile_kernel_impl(
    command: str,
    num_replays: int = 3,
    kernel_filter: str | None = None,
    auto_select: bool = False,
    quick: bool = False,
    gpu_devices: str | list[str] | None = None,
) -> dict[str, Any]:
    """
    Profile a GPU kernel using Metrix.

    Args:
        command: Command to execute for profiling (e.g., 'python3 kernel.py')
        num_replays: Number of profiling replays for statistics (default: 3)
        kernel_filter: Kernel name pattern to filter (e.g., '*topk*')
        auto_select: Automatically select main kernel (default: False)
        quick: Use quick profile (3 metrics, 1 pass) vs memory profile (12 metrics, 2 passes)
        gpu_devices: GPU device ID(s) to profile on (default: HIP_VISIBLE_DEVICES or "0")

    Returns:
        {
            "success": bool,
            "results": [
                {
                    "device_id": "0",
                    "gpu_info": {
                        "vendor": "AMD",
                        "model": "AMD Instinct MI300X",
                        "architecture": "gfx942",
                        "compute_units": 228,
                        "peak_hbm_bandwidth_gbs": 5200.0,
                        ...
                    },
                    "kernels": [
                        {
                            "name": "kernel_name",
                            "duration_us": 123.45,
                            "bottleneck": "memory-bound",
                            "observations": [
                                "High HBM bandwidth utilization (67%)",
                                "Low L2 cache hit rate (12%)"
                            ],
                            "metrics": {
                                "duration_us": 123.45,
                                "memory.hbm_bandwidth_utilization": 0.67,
                                "memory.l2_hit_rate": 0.12,
                                ...
                            }
                        }
                    ]
                }
            ]
        }
    """
    try:
        logger.info("=" * 60)
        logger.info("Metrix MCP Tool Called")
        logger.info("=" * 60)
        logger.info(f"Command: {command}")
        logger.info(f"GPU devices: {gpu_devices or 'default'}")
        logger.info(f"Profile mode: {'quick' if quick else 'memory (full)'}")
        logger.info(f"Num replays: {num_replays}")
        logger.info(f"Kernel filter: {kernel_filter or 'None'}")
        logger.info(f"Auto select: {auto_select}")

        # Initialize MetrixTool
        tool = MetrixTool(gpu_devices=gpu_devices)

        # Run profiling
        result = tool.profile(
            command=command, num_replays=num_replays, kernel_filter=kernel_filter, auto_select=auto_select, quick=quick
        )

        logger.info("=" * 60)
        logger.info("✓ Profiling Complete!")
        logger.info(f"  Profiled {sum(len(r['kernels']) for r in result['results'])} kernel(s)")
        logger.info(f"  Across {len(result['results'])} GPU(s)")
        logger.info("=" * 60)

        full_result = {"success": True, **result}

        # If the serialized result exceeds 32KB, write it to a temp file
        # and return a file reference instead.  This avoids hitting the
        # asyncio StreamReader readline limit on the JSON-RPC stdio
        # transport when kernel names contain very long C++ mangled symbols.
        _LARGE_RESULT_THRESHOLD = 32 * 1024  # 32 KB
        result_json = json.dumps(full_result)
        if len(result_json) > _LARGE_RESULT_THRESHOLD:
            result_dir = Path(tempfile.gettempdir()) / "mcp_results" / "metrix"
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file = result_dir / f"{os.getpid()}_{id(full_result)}.json"
            result_file.write_text(result_json)
            logger.info(f"Large result ({len(result_json)} bytes) written to {result_file}")
            return {"_result_file": str(result_file)}

        return full_result

    except Exception as e:
        logger.error(f"Metrix profiling failed: {e}", exc_info=True)
        return {"success": False, "error": str(e), "results": []}


@mcp.tool()
def profile_kernel(
    command: str,
    num_replays: int = 3,
    kernel_filter: str | None = None,
    auto_select: bool = False,
    quick: bool = False,
    gpu_devices: str | list[str] | None = None,
) -> dict[str, Any]:
    """
    Profile a GPU kernel using AMD Metrix profiler.

    Provides hardware-level metrics including:
    - Memory: HBM bandwidth, L1/L2 hit rates, coalescing efficiency
    - Compute: Arithmetic intensity, TFLOPS
    - LDS: Bank conflicts, utilization
    - Duration: Kernel execution time

    Automatically classifies bottlenecks:
    - memory-bound: High HBM usage, low compute intensity
    - compute-bound: Low HBM usage, high L2 hit rate
    - latency-bound: Very short duration, low resource usage
    - lds-bound: High LDS bank conflicts
    - balanced: No clear bottleneck

    MCP tool wrapper - calls implementation function.
    """
    return _profile_kernel_impl(
        command=command,
        num_replays=num_replays,
        kernel_filter=kernel_filter,
        auto_select=auto_select,
        quick=quick,
        gpu_devices=gpu_devices,
    )


def main():
    """Run MCP server."""
    logger.info("Starting Metrix MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
