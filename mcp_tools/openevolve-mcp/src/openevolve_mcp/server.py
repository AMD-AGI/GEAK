# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""
OpenEvolve Optimizer MCP Server - COMMANDMENT-based Multi-file Evaluation

Provides GPU kernel optimization using LLM-guided evolution via run_openevolve.py.
Supports multi-file kernels with deterministic COMMANDMENT.md evaluation.

The server invokes run_openevolve.py as a subprocess, which:
1. Auto-detects GPUs and builds COMMANDMENT.md (or uses pre-built)
2. Validates commands on baseline kernel
3. Freezes COMMANDMENT.md (immutable during evolution)
4. Runs OpenEvolve evolutionary optimization with GPU-isolated evaluation
"""

import json
import logging
import os
import subprocess
from pathlib import Path

from fastmcp import FastMCP

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create MCP server
mcp = FastMCP(
    name="openevolve-optimizer",
    instructions=(
        "GPU kernel optimization using LLM-guided evolution (OpenEvolve). "
        "Uses COMMANDMENT.md for deterministic, reproducible evaluation with "
        "Metrix hardware profiling and GPU isolation."
    ),
)

# Auto-detect GEAK_OE_ROOT from environment or common locations
GEAK_OE_ROOT = os.environ.get("GEAK_OE_ROOT", "")
if not GEAK_OE_ROOT:
    for candidate in [
        "/opt/geak-oe",
        "/workspace/geak-oe",
        str(Path(__file__).parent.parent.parent.parent.parent / "geak-oe"),
        str(Path.home() / "geak-oe-fresh"),
    ]:
        cand = Path(candidate)
        run_script = cand / "examples" / "geak_eval" / "run_openevolve.py"
        if cand.is_dir() and run_script.is_file():
            GEAK_OE_ROOT = candidate
            break

RUN_OPENEVOLVE_SCRIPT = str(Path(GEAK_OE_ROOT) / "examples" / "geak_eval" / "run_openevolve.py") if GEAK_OE_ROOT else ""


def _find_run_openevolve() -> str:
    """Locate run_openevolve.py, raising if not found."""
    if RUN_OPENEVOLVE_SCRIPT and Path(RUN_OPENEVOLVE_SCRIPT).is_file():
        return RUN_OPENEVOLVE_SCRIPT

    # Try environment again at call time (may have been set after import)
    oe_root = os.environ.get("GEAK_OE_ROOT", "")
    if oe_root:
        script = Path(oe_root) / "examples" / "geak_eval" / "run_openevolve.py"
        if script.is_file():
            return str(script)

    raise FileNotFoundError(
        "run_openevolve.py not found. Set GEAK_OE_ROOT environment variable "
        "to the OpenEvolve repository root (e.g. /opt/geak-oe)."
    )


@mcp.tool()
def optimize_kernel(
    kernel_path: str,
    max_iterations: int = 10,
    gpu: str | int = 0,
    output_dir: str | None = None,
    commandment_path: str | None = None,
    baseline_metrics_path: str | None = None,
    config_path: str | None = None,
) -> dict:
    """
    Optimize a GPU kernel using OpenEvolve LLM-guided evolution.

    Uses COMMANDMENT.md for deterministic evaluation with Metrix hardware profiling.
    Supports multi-file kernels (directories) and single-file kernels.

    Args:
        kernel_path: Path to kernel file (.py) or directory containing kernel files.
        max_iterations: Max evolution iterations (default: 10).
        gpu: GPU device ID(s) -- single int or comma-separated string (e.g. "0,1,2").
        output_dir: Output directory for results. Defaults to <kernel_dir>/optimization_output.
        commandment_path: Optional path to pre-built COMMANDMENT.md (skips auto-build).
        baseline_metrics_path: Optional path to pre-computed baseline metrics JSON.
        config_path: Optional path to OpenEvolve config.yaml (overrides bundled config).
                     Use this to change the LLM model, population size, etc.

    Returns:
        {
            "success": bool,
            "speedup": float,
            "best_score": float,
            "iterations_completed": int,
            "baseline_latency_us": float,
            "best_latency_us": float,
            "output_dir": str,
            "best_kernel_path": str,
            "commandment_path": str,
            "error": str | None
        }
    """
    try:
        logger.info("=" * 60)
        logger.info("OpenEvolve MCP Tool: optimize_kernel")
        logger.info("=" * 60)

        # Validate kernel path
        kernel_path = str(Path(kernel_path).resolve())
        if not Path(kernel_path).exists():
            return {"success": False, "error": f"Kernel not found: {kernel_path}"}

        # Find run_openevolve.py
        script = _find_run_openevolve()
        logger.info(f"Using run_openevolve.py: {script}")

        # Determine output directory
        if not output_dir:
            kp = Path(kernel_path)
            if kp.is_dir():
                output_dir = str(kp / "optimization_output")
            else:
                output_dir = str(kp.parent / "optimization_output")
        output_dir = str(Path(output_dir).resolve())
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Build command
        # --gpu 0 always: physical GPU selection is handled via HIP_VISIBLE_DEVICES
        # in the subprocess env (set below). The script sees device 0 = first visible GPU.
        cmd = [
            "python3",
            script,
            kernel_path,
            "--iterations",
            str(max_iterations),
            "--gpu",
            "0",
            "--output",
            output_dir,
        ]

        if config_path:
            resolved = str(Path(config_path).resolve())
            if not Path(resolved).is_file():
                return {"success": False, "error": f"Config not found: {resolved}"}
            cmd.extend(["--config", resolved])
            logger.info(f"Using custom config: {resolved}")
        if commandment_path:
            cmd.extend(["--commandment", str(Path(commandment_path).resolve())])
        if baseline_metrics_path:
            cmd.extend(["--baseline-metrics", str(Path(baseline_metrics_path).resolve())])

        logger.info(f"Command: {' '.join(cmd)}")

        # Set up environment (pass through API keys, don't hardcode)
        env = os.environ.copy()
        if GEAK_OE_ROOT:
            env["GEAK_OE_ROOT"] = GEAK_OE_ROOT
        gpu_str = str(gpu)
        env["HIP_VISIBLE_DEVICES"] = gpu_str
        # Pre-set GEAK_GPU_IDS so run_openevolve.py's evaluator pool uses
        # exactly the GPUs we assigned, not all physical GPUs via rocm-smi.
        env["GEAK_GPU_IDS"] = gpu_str

        # Ensure PYTHONPATH includes OpenEvolve
        oe_root = env.get("GEAK_OE_ROOT", GEAK_OE_ROOT)
        if oe_root:
            existing_pp = env.get("PYTHONPATH", "")
            if oe_root not in existing_pp:
                env["PYTHONPATH"] = f"{oe_root}:{existing_pp}" if existing_pp else oe_root

        # Run the subprocess
        logger.info("Starting OpenEvolve optimization (this may take a while)...")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max
            env=env,
            cwd=oe_root if oe_root else None,
        )

        logger.info(f"Process exited with code: {proc.returncode}")
        if proc.stdout:
            # Log last 50 lines of stdout
            lines = proc.stdout.strip().split("\n")
            for line in lines[-50:]:
                logger.info(f"  [stdout] {line}")
        if proc.stderr:
            lines = proc.stderr.strip().split("\n")
            for line in lines[-20:]:
                logger.warning(f"  [stderr] {line}")

        # Parse results from output JSON
        out = Path(output_dir)
        result_path = out / "openevolve_result.json"
        if result_path.is_file():
            with open(result_path) as f:
                result_data = json.load(f)

            best_metrics = result_data.get("best_metrics", {})
            baseline_metrics = result_data.get("baseline_metrics", {})
            best_kernel = out / "best_kernel.py"
            commandment = out / "COMMANDMENT.md"

            response = {
                "success": result_data.get("success", proc.returncode == 0),
                "speedup": result_data.get("best_score", 1.0),
                "best_score": result_data.get("best_score", 1.0),
                "iterations_completed": result_data.get("iterations_completed", 0),
                "baseline_latency_us": baseline_metrics.get("duration_us", 0),
                "best_latency_us": best_metrics.get("duration_us", 0),
                "output_dir": output_dir,
                "best_kernel_path": str(best_kernel) if best_kernel.is_file() else "",
                "commandment_path": str(commandment) if commandment.is_file() else "",
                "error": None,
            }
            logger.info(f"Optimization complete: speedup={response['speedup']:.4f}x")
            return response

        elif proc.returncode == 0:
            # Process succeeded but no result JSON -- check for best_kernel.py
            best_kernel = Path(output_dir) / "best_kernel.py"
            return {
                "success": True,
                "speedup": 1.0,
                "best_score": 1.0,
                "iterations_completed": 0,
                "baseline_latency_us": 0,
                "best_latency_us": 0,
                "output_dir": output_dir,
                "best_kernel_path": str(best_kernel) if best_kernel.is_file() else "",
                "commandment_path": "",
                "error": "No result JSON found but process succeeded",
            }
        else:
            error_msg = proc.stderr[-2000:] if proc.stderr else "Unknown error"
            return {
                "success": False,
                "speedup": 1.0,
                "error": f"Process exited with code {proc.returncode}: {error_msg}",
            }

    except subprocess.TimeoutExpired:
        logger.error("OpenEvolve optimization timed out (2h limit)")
        return {"success": False, "error": "Optimization timed out after 2 hours"}
    except Exception as e:
        logger.error(f"MCP tool execution failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@mcp.tool()
def check_openevolve_status() -> dict:
    """
    Check if OpenEvolve is properly installed and configured.

    Returns availability status, GEAK_OE_ROOT path, and run_openevolve.py location.
    """
    status = {
        "geak_oe_root": GEAK_OE_ROOT,
        "run_openevolve_script": RUN_OPENEVOLVE_SCRIPT,
        "openevolve_available": False,
        "kernel_profile_available": False,
        "errors": [],
    }

    # Check run_openevolve.py
    try:
        script = _find_run_openevolve()
        status["run_openevolve_script"] = script
        status["openevolve_available"] = True
    except FileNotFoundError as e:
        status["errors"].append(str(e))

    # Check OpenEvolve importable
    try:
        import openevolve  # noqa: F401

        status["openevolve_importable"] = True
    except ImportError:
        status["openevolve_importable"] = False
        status["errors"].append("Cannot import openevolve package")

    # Check kernel-profile tool
    try:
        result = subprocess.run(["kernel-profile", "--help"], capture_output=True, text=True, timeout=10)
        status["kernel_profile_available"] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        status["errors"].append("kernel-profile command not found")

    return status


def main():
    """Run MCP server."""
    logger.info("Starting OpenEvolve MCP Server (COMMANDMENT-based)...")
    logger.info(f"  GEAK_OE_ROOT: {GEAK_OE_ROOT}")
    logger.info(f"  run_openevolve.py: {RUN_OPENEVOLVE_SCRIPT}")
    mcp.run()


if __name__ == "__main__":
    main()
