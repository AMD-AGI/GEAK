"""Unified GPU Profiler MCP Server.

Wraps two profiling backends behind a single `profile_kernel` tool:
- metrix: AMD Metrix API (structured JSON, bottleneck classification, hardware metrics)
- rocprof-compute: rocprof-compute CLI (deep roofline, instruction mix, cache, wavefront)

Usage:
    profiler-mcp                  # Run as MCP server
    python -m profiler_mcp.server # Same thing
"""

import csv
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU arch detection (kept inline; profiler-mcp must not depend on minisweagent)
# ---------------------------------------------------------------------------


def _detect_gpu_arch() -> str:
    """Return the GFX architecture string (e.g. 'gfx942', 'gfx1201') or '' on failure."""
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "gfx" in line.lower() and "name:" in line.lower():
                for p in line.split():
                    if p.startswith("gfx"):
                        return p
    except Exception:
        pass
    return ""


def _guard_rocprof_compute(backend: str) -> tuple[str, str]:
    """If *backend* is 'rocprof-compute' on RDNA, return ('metrix', arch). Otherwise (backend, '')."""
    if backend != "rocprof-compute":
        return backend, ""
    arch = _detect_gpu_arch()
    if arch.startswith(("gfx10", "gfx11", "gfx12")):
        return "metrix", arch
    return backend, ""


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

_SHELL_META = re.compile(r"[&|;$`(){}<>!\\]|&&|\|\||<<|>>|\bcd\b|\bsource\b|\bexport\b")

# Detects inline env-var assignment: "VAR=value command ..."
# rocprofv3 treats "VAR=value" as the executable name and crashes.
_INLINE_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S+\s")


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
    cwd: str | None = None,
) -> dict[str, Any]:
    """Profile using AMD Metrix API. Returns structured JSON."""
    from .core import MetrixTool

    tool = MetrixTool(gpu_devices=gpu_devices)
    try:
        result = tool.profile(
            command=command,
            num_replays=num_replays,
            kernel_filter=kernel_filter,
            auto_select=auto_select,
            quick=quick,
            cwd=cwd,
        )
    except Exception as e:
        logger.warning("Metrix profiling failed: %s", e)
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
        from minisweagent.run.preprocess.kernel_profile import _build_rocprof_result
        from minisweagent.tools.profiling_tools import ProfilingAnalyzer
    except ImportError:
        _agent_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        _src = _agent_root / "src"
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        from minisweagent.run.preprocess.kernel_profile import _build_rocprof_result
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


def _to_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _parse_rocprof_legacy_outputs(output_base: Path) -> list[dict[str, Any]]:
    """Parse legacy rocprof stats/trace CSV into backend-neutral kernels."""
    stats_path = output_base.with_suffix(".stats.csv")
    trace_path = output_base.with_suffix(".csv")

    trace_by_name: dict[str, list[dict[str, str]]] = {}
    if trace_path.exists():
        with trace_path.open(newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("KernelName")
                if name:
                    trace_by_name.setdefault(name, []).append(row)

    kernels: list[dict[str, Any]] = []
    if stats_path.exists():
        with stats_path.open(newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("Name")
                if not name:
                    continue
                calls = _to_int(row.get("Calls"))
                total_ns = _to_float(row.get("TotalDurationNs"))
                avg_ns = _to_float(row.get("AverageNs"))
                percentage = _to_float(row.get("Percentage"))
                traces = trace_by_name.get(name, [])
                durations = [_to_float(r.get("DurationNs")) for r in traces if r.get("DurationNs")]
                sample = traces[0] if traces else {}
                total_us = total_ns / 1000.0
                avg_us = avg_ns / 1000.0
                metrics: dict[str, Any] = {
                    "duration_us": total_us,
                    "duration_us_avg": avg_us,
                    "dispatch_count": calls,
                    "profile_percentage": percentage,
                    "timing_only": True,
                }
                if durations:
                    metrics["duration_us_min"] = min(durations) / 1000.0
                    metrics["duration_us_max"] = max(durations) / 1000.0
                for key, column in (
                    ("grid_size", "grd"),
                    ("workgroup_size", "wgr"),
                    ("lds_bytes", "lds"),
                    ("scratch_bytes", "scr"),
                    ("arch_vgpr", "arch_vgpr"),
                    ("accum_vgpr", "accum_vgpr"),
                    ("sgpr", "sgpr"),
                    ("wave_size", "wave_size"),
                ):
                    if column in sample:
                        metrics[key] = _to_int(sample.get(column))
                kernels.append(
                    {
                        "name": name,
                        "duration_us": total_us,
                        "bottleneck": "unknown",
                        "observations": [
                            "Captured by legacy rocprof timing-only backend.",
                            f"Calls={calls}, avg={avg_us:.3f} us, total={total_us:.3f} us.",
                        ],
                        "metrics": metrics,
                    }
                )

    if not kernels and trace_by_name:
        for name, traces in trace_by_name.items():
            durations = [_to_float(r.get("DurationNs")) for r in traces if r.get("DurationNs")]
            total_us = sum(durations) / 1000.0
            avg_us = total_us / len(durations) if durations else 0.0
            kernels.append(
                {
                    "name": name,
                    "duration_us": total_us,
                    "bottleneck": "unknown",
                    "observations": [
                        "Captured by legacy rocprof timing-only backend.",
                        f"Calls={len(traces)}, avg={avg_us:.3f} us, total={total_us:.3f} us.",
                    ],
                    "metrics": {
                        "duration_us": total_us,
                        "duration_us_avg": avg_us,
                        "dispatch_count": len(traces),
                        "timing_only": True,
                    },
                }
            )

    kernels.sort(key=lambda k: k.get("duration_us", 0), reverse=True)
    return kernels


def _profile_with_rocprof_legacy(
    command: str,
    workdir: str | None = None,
    gpu_devices: str | list[str] | None = None,
) -> dict[str, Any]:
    """Profile Python/TileLang-friendly kernel timing with legacy rocprof.

    ROCm 7.0.2's rocprofv3 can abort while Python GPU runtimes dynamically
    load extensions. The legacy rocprof path does not expose Metrix counters,
    but it reliably captures kernel names and dispatch timing for Python
    workloads on this MI300 stack.
    """
    if isinstance(gpu_devices, list):
        device = str(gpu_devices[0]) if gpu_devices else "0"
    elif gpu_devices is None:
        device = os.environ.get("HIP_VISIBLE_DEVICES") or "0"
    else:
        device = str(gpu_devices).split(",", 1)[0] or "0"

    with tempfile.TemporaryDirectory(prefix="geak-rocprof-legacy-") as tmp:
        output_base = Path(tmp) / "rocprof_legacy"
        cmd = ["rocprof", "--timestamp", "on", "--stats", "-o", str(output_base.with_suffix(".csv"))]
        cmd.extend(shlex.split(command))
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = device
        result = subprocess.run(
            cmd,
            cwd=workdir or None,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            return {
                "success": False,
                "backend": "rocprof-legacy",
                "error": (
                    f"rocprof legacy failed with exit code {result.returncode}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                ),
                "results": [],
            }

        kernels = _parse_rocprof_legacy_outputs(output_base)
        return {
            "success": bool(kernels),
            "backend": "rocprof-legacy",
            "warning": "timing-only profile; hardware counters are unavailable from legacy rocprof",
            "results": [
                {
                    "device_id": device,
                    "gpu_info": {"detected": False, "device_id": device},
                    "kernels": kernels,
                }
            ],
            "error": "" if kernels else "legacy rocprof completed but produced no kernels",
        }


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
    backend: Literal["metrix", "rocprof-compute", "rocprof-legacy"],
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
        backend: Required. 'metrix' for structured AMD Metrix profiling,
                 'rocprof-compute' for roofline/instruction-level analysis, or
                 'rocprof-legacy' for timing-only Python/TileLang-friendly traces.
        workdir: Working directory for the command.
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
    logger.info("Profiler MCP: backend=%s, command=%s", backend, command)

    backend, rdna_arch = _guard_rocprof_compute(backend)
    if rdna_arch:
        logger.warning(
            "rocprof-compute does not support RDNA (%s). Falling back to metrix backend.",
            rdna_arch,
        )

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
                cwd=workdir or str(Path.cwd()),
            )
        elif backend == "rocprof-compute":
            return _profile_with_rocprof(
                command=command,
                workdir=workdir,
                profiling_type=profiling_type,
            )
        elif backend == "rocprof-legacy":
            return _profile_with_rocprof_legacy(
                command=command,
                workdir=workdir,
                gpu_devices=gpu_devices,
            )
        else:
            return {
                "success": False,
                "backend": backend,
                "error": f"Unknown backend '{backend}'. Use 'metrix', 'rocprof-compute', or 'rocprof-legacy'.",
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
