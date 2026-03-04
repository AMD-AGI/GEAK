# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Unified GPU Profiler MCP Server.

Wraps two profiling backends behind a single `profile_kernel` tool:
- metrix: AMD Metrix API (structured JSON, bottleneck classification, hardware metrics)
- rocprof-compute: rocprof-compute CLI (deep roofline, instruction mix, cache, wavefront)

Usage:
    profiler-mcp                  # Run as MCP server
    python -m profiler_mcp.server # Same thing
"""

import logging
import os
import re
import shlex
import subprocess
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
# Command normalisation
# ---------------------------------------------------------------------------

_SHELL_META = re.compile(r'[&|;$`(){}<>!\\]|&&|\|\||<<|>>|\bcd\b|\bsource\b|\bexport\b')

# Detects inline env-var assignment: "VAR=value command ..."
# rocprofv3 treats "VAR=value" as the executable name and crashes.
_INLINE_ENV = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*=\S+\s')


def _normalize_command(command: str) -> str:
    """Wrap *command* in ``bash -c`` if it contains shell constructs.

    ``rocprofv3`` uses ``os.execvpe`` to launch the profiled command, which
    bypasses the shell entirely.  This means shell builtins (``cd``),
    environment variable expansion (``$VAR``), compound operators
    (``&&``, ``|``), and inline env-var assignments
    (``HIP_VISIBLE_DEVICES=0 python3 ...``) will all fail.  Wrapping in
    ``bash -c`` ensures the command is interpreted by a real shell.

    Simple commands (e.g. ``python3 kernel.py --profile``) are left as-is
    so that ``execvpe`` can launch them directly without the extra process.
    """
    if command.startswith("bash -c ") or command.startswith("bash -c'"):
        return command
    if _SHELL_META.search(command) or _INLINE_ENV.match(command):
        logger.info(f"Command contains shell constructs, wrapping in bash -c: {command[:120]}")
        return f"bash -c {shlex.quote(command)}"
    return command


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

    # Empty HIP_VISIBLE_DEVICES hides all GPUs from ROCm.  We need to
    # remove it for the profiling subprocess.  profiler-mcp runs as a
    # dedicated single-threaded MCP server process (not inside the
    # multi-threaded parallel agent), so a save/restore of os.environ
    # is safe here -- no concurrent threads can observe the temporary gap.
    _hip_removed: str | None = None
    if os.environ.get("HIP_VISIBLE_DEVICES") == "":
        _hip_removed = os.environ.pop("HIP_VISIBLE_DEVICES")

    analyzer = ProfilingAnalyzer(profiling_type=profiling_type)
    try:
        raw = analyzer.profile_structured(
            profiling_workdir=workdir or str(Path.cwd()),
            profiling_cmd=command,
        )
    finally:
        analyzer.cleanup()
        if _hip_removed is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = _hip_removed

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
# Warmup helper (backend-agnostic)
# ---------------------------------------------------------------------------


def _warmup(command: str, warmup_runs: int) -> None:
    """Run the profiling command without instrumentation to warm caches.

    Warms Triton JIT cache, GPU instruction/data caches, GPU clock
    frequencies, and HIP runtime so that the first instrumented run
    reflects steady-state performance.  Failures are tolerated (best-effort).
    """
    if warmup_runs <= 0:
        return
    for i in range(warmup_runs):
        logger.info(f"Warmup run {i + 1}/{warmup_runs}: {command}")
        try:
            subprocess.run(
                command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
        except Exception as exc:
            logger.warning(f"Warmup run {i + 1} failed (non-fatal): {exc}")


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
    warmup_runs: int = 2,
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
        warmup_runs: Number of un-instrumented warmup executions before
                     profiling (default 2).  Set to 0 to skip warmup.

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

    command = _normalize_command(command)

    _warmup(command, warmup_runs)

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
