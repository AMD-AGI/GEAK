"""Unified GPU Profiler MCP Server.

Wraps two profiling backends behind a single `profile_kernel` tool:
- metrix: AMD Metrix API (structured JSON, bottleneck classification, hardware metrics)
- rocprof-compute: rocprof-compute CLI (deep roofline, instruction mix, cache, wavefront)

Usage:
    profiler-mcp                  # Run as MCP server
    python -m profiler_mcp.server # Same thing
"""

import logging
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

mcp = FastMCP(
    name="profiler",
    instructions=(
        "Unified GPU kernel profiling. Use backend='metrix' for structured metrics "
        "with bottleneck classification, or backend='rocprof-compute' for deep "
        "roofline and instruction-level analysis."
    ),
)


# ---------------------------------------------------------------------------
# Backend: Metrix
# ---------------------------------------------------------------------------


def _profile_with_metrix(
    command: str,
    num_replays: int = 3,
    kernel_filter: str | None = None,
    auto_select: bool = False,
    quick: bool = False,
    gpu_devices: str | list[str] | None = None,
) -> dict[str, Any]:
    """Profile using AMD Metrix API. Returns structured JSON."""
    # Import MetrixTool from the installed metrix-mcp or in-tree copy
    try:
        from metrix_mcp.core import MetrixTool
    except ImportError:
        # Fallback: look in the agent package
        _agent_root = Path(__file__).resolve().parent.parent.parent.parent
        _metrix_src = _agent_root / "metrix-mcp" / "src"
        if str(_metrix_src) not in sys.path:
            sys.path.insert(0, str(_metrix_src))
        from metrix_mcp.core import MetrixTool

    tool = MetrixTool(gpu_devices=gpu_devices)
    try:
        result = tool.profile(
            command=command,
            num_replays=num_replays,
            kernel_filter=kernel_filter,
            auto_select=auto_select,
            quick=quick,
        )
    except Exception as e:
        return {
            "success": False,
            "backend": "metrix",
            "error": str(e),
            "results": [],
        }
    return {"success": True, "backend": "metrix", **result}


# ---------------------------------------------------------------------------
# Backend: rocprof-compute
# ---------------------------------------------------------------------------


def _profile_with_rocprof(
    command: str,
    workdir: str | None = None,
    profiling_type: str = "profiling",
) -> dict[str, Any]:
    """Profile using rocprof-compute. Returns backend-neutral structured JSON.

    Args:
        command: Command to execute for profiling.
        workdir: Working directory (defaults to cwd).
        profiling_type: One of 'profiling' (full), 'roofline', 'profiler_analyzer'.
    """
    try:
        from minisweagent.kernel_profile import _build_rocprof_result
        from minisweagent.tools.profiling_tools import ProfilingAnalyzer
    except ImportError:
        _agent_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        _src = _agent_root / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from minisweagent.kernel_profile import _build_rocprof_result
        from minisweagent.tools.profiling_tools import ProfilingAnalyzer

    analyzer = ProfilingAnalyzer(profiling_type=profiling_type)
    try:
        raw = analyzer.profile_structured(
            profiling_workdir=workdir or str(Path.cwd()),
            profiling_cmd=command,
        )
    finally:
        analyzer.cleanup()

    if not raw.get("success"):
        return {
            "success": False,
            "backend": "rocprof-compute",
            "error": raw.get("error", "rocprof-compute profiling failed"),
            "results": [],
        }

    result = _build_rocprof_result(raw)
    result["success"] = True
    return result


# ---------------------------------------------------------------------------
# Unified MCP tool
# ---------------------------------------------------------------------------


@mcp.tool()
def profile_kernel(
    command: str,
    backend: str = "metrix",
    workdir: str | None = None,
    profiling_type: str = "profiling",
    num_replays: int = 3,
    kernel_filter: str | None = None,
    auto_select: bool = False,
    quick: bool = False,
    gpu_devices: str | list[str] | None = None,
) -> dict[str, Any]:
    """Profile a GPU kernel.

    Args:
        command: Command to execute (e.g. 'python3 kernel.py').
        backend: 'metrix' for structured AMD Metrix profiling, or
                 'rocprof-compute' for deep roofline/instruction analysis.
        workdir: Working directory for the command (rocprof-compute only).
        profiling_type: For rocprof-compute: 'profiling' (full), 'roofline', or
                        'profiler_analyzer'. Ignored for metrix.
        num_replays: Number of profiling replays (metrix only, default 3).
        kernel_filter: Kernel name pattern filter (metrix only).
        auto_select: Auto-select main kernel (metrix only).
        quick: Quick profile with fewer metrics (metrix only).
        gpu_devices: GPU device ID(s) to profile on.

    Returns:
        {
            "success": bool,
            "backend": str,
            # metrix returns: "results" with structured kernel data
            # rocprof-compute returns: "analysis" with text output
        }
    """
    logger.info("=" * 60)
    logger.info(f"Profiler MCP: backend={backend}, command={command}")
    logger.info("=" * 60)

    try:
        if backend == "metrix":
            return _profile_with_metrix(
                command=command,
                num_replays=num_replays,
                kernel_filter=kernel_filter,
                auto_select=auto_select,
                quick=quick,
                gpu_devices=gpu_devices,
            )
        elif backend == "rocprof-compute":
            return _profile_with_rocprof(
                command=command,
                workdir=workdir,
                profiling_type=profiling_type,
            )
        else:
            return {
                "success": False,
                "backend": backend,
                "error": f"Unknown backend '{backend}'. Use 'metrix' or 'rocprof-compute'.",
                "results": [],
            }
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        return {
            "success": False,
            "backend": backend,
            "error": str(e),
            "results": [],
        }


def main():
    """Run the unified profiler MCP server."""
    logger.info("Starting Unified Profiler MCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
