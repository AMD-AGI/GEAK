# Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.

"""
Bridge evaluator: connects OpenEvolve to deterministic COMMANDMENT-based
evaluation using Metrix hardware profiler.

This evaluator is called by OpenEvolve's standard 10-parameter evaluate()
interface.  For each candidate program it:
  1. Writes candidate file(s) to a working directory
  2. Runs CORRECTNESS commands from COMMANDMENT.md
  3. Runs PROFILE commands from COMMANDMENT.md
  4. Parses hardware metrics (duration_us, bandwidth, etc.) from Metrix
  5. Calculates speedup = baseline_latency_us / candidate_latency_us

IMPORTANT: This evaluator does NOT know or care what the CORRECTNESS and
PROFILE commands actually are.  They could invoke correctness_check.py (for
AIG-Eval Triton kernels), a custom pytest script, a CK verification binary,
or anything else.  The COMMANDMENT.md is the only contract -- the evaluator
just executes whatever commands are in it.

NO Python time.perf_counter() is used for scoring -- all benchmarking
is done through the Metrix hardware profiler via COMMANDMENT shell commands.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Make openevolve importable
_GEAK_OE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _GEAK_OE_ROOT not in sys.path:
    sys.path.insert(0, _GEAK_OE_ROOT)


# Cache the CommandmentEvaluator instance across calls (singleton per process)
_CMD_EVALUATOR = None
_CMD_EVALUATOR_PATH = None


def _get_commandment_evaluator():
    """Get or create the CommandmentEvaluator singleton."""
    global _CMD_EVALUATOR, _CMD_EVALUATOR_PATH

    from openevolve.commandment_evaluator import CommandmentEvaluator

    commandment_path = os.environ.get("GEAK_COMMANDMENT_PATH", "")
    baseline_metrics_path = os.environ.get("GEAK_BASELINE_METRICS", "")
    kernel_dir = os.environ.get("GEAK_KERNEL_DIR", "")

    # Re-create if the path changed (shouldn't happen, but be safe)
    if _CMD_EVALUATOR is None or _CMD_EVALUATOR_PATH != commandment_path:
        if not commandment_path or not os.path.isfile(commandment_path):
            raise FileNotFoundError(
                f"COMMANDMENT.md not found at '{commandment_path}'. "
                f"Set GEAK_COMMANDMENT_PATH env var or run run_openevolve.py "
                f"which generates it."
            )

        _CMD_EVALUATOR = CommandmentEvaluator(
            commandment_path=commandment_path,
            baseline_metrics_path=baseline_metrics_path,
            kernel_dir=kernel_dir,
            timeout=600,
        )
        _CMD_EVALUATOR_PATH = commandment_path
        logger.info(f"Loaded CommandmentEvaluator from {commandment_path}")

    return _CMD_EVALUATOR


# ---------------------------------------------------------------------------
# OpenEvolve evaluate() entry point
# ---------------------------------------------------------------------------

def evaluate(
    test_suite_path: str,
    program_text: str,
    ref_wrapper_path: str,
    wrapper_fn_name: str,
    unit_tests_path: str,
    n_warmup: int,
    n_iters: int,
    atol: float,
    rtol: float,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Evaluation function called by OpenEvolve.

    This is the bridge: it takes the standard 10-parameter OpenEvolve
    signature and delegates to the CommandmentEvaluator, which runs
    shell commands for correctness checking (correctness_check.py) and
    hardware profiling (Metrix / kernel-profile) as specified in
    COMMANDMENT.md.

    Args:
        test_suite_path: Not used directly
        program_text:    Path to the candidate kernel.py file (written by
                         OpenEvolve's Evaluator to a temp dir)
        ref_wrapper_path: Not used
        wrapper_fn_name:  Not used
        unit_tests_path:  Not used
        n_warmup:        Not used (Metrix handles warmup internally)
        n_iters:         Not used (Metrix handles iterations internally)
        atol:            Not used (correctness_check.py reads from kernel)
        rtol:            Not used (correctness_check.py reads from kernel)
        verbose:         Verbose logging

    Returns:
        Dict with keys: success, combined_score, final_score,
        correctness_score, performance_metrics, error, ...
    """
    start_time = time.time()

    try:
        # Validate the candidate file exists
        if not os.path.isfile(program_text):
            return _failure_result(f"Candidate file not found: {program_text}")

        print(f"\n{'='*60}")
        print(f"GEAK-Eval COMMANDMENT Evaluator: {os.path.basename(program_text)}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # Read the candidate code and prepare as program_files dict
        # ------------------------------------------------------------------
        with open(program_text, "r") as f:
            candidate_code = f.read()

        # Use the basename from the original kernel (e.g., kernel.py)
        # so that COMMANDMENT.md commands referencing ${GEAK_WORK_DIR}/kernel.py work
        baseline_kernel = os.environ.get("GEAK_BASELINE_KERNEL", "")
        if baseline_kernel:
            filename = os.path.basename(baseline_kernel)
        else:
            filename = os.path.basename(program_text)

        program_files = {filename: candidate_code}

        # ------------------------------------------------------------------
        # Run COMMANDMENT evaluation
        # ------------------------------------------------------------------
        cmd_eval = _get_commandment_evaluator()

        # Use the temp dir that OpenEvolve already created (parent of program_text)
        work_dir = os.path.dirname(os.path.abspath(program_text))

        # Read assigned GPU ID from .gpu_id file written by the evaluator.
        # This ensures each candidate is benchmarked on its own exclusive GPU
        # when parallel_evaluations > 1.
        gpu_id = None
        gpu_id_file = os.path.join(work_dir, ".gpu_id")
        if os.path.isfile(gpu_id_file):
            try:
                with open(gpu_id_file, "r") as f:
                    gpu_id = int(f.read().strip())
                print(f"  GPU assignment: device {gpu_id} (exclusive)")
            except (ValueError, IOError):
                pass

        result = cmd_eval.evaluate(
            program_files=program_files,
            program_id=os.path.basename(work_dir),
            work_dir=work_dir,
            gpu_id=gpu_id,
        )

        elapsed = time.time() - start_time

        # ------------------------------------------------------------------
        # Build OpenEvolve-compatible result dict
        # ------------------------------------------------------------------
        if not result.success:
            error_msg = result.error or "Unknown COMMANDMENT failure"
            if not result.correctness_passed:
                error_msg = f"CORRECTNESS failed: {error_msg}"
            return _failure_result(error_msg)

        # Extract speedup from COMMANDMENT metrics
        speedup = result.speedup
        duration_us = result.metrics.get("duration_us", 0)
        baseline_dur = cmd_eval.baseline_metrics.get("duration_us", 0)

        # If Metrix produced duration_us, use it for speedup
        if duration_us > 0 and baseline_dur > 0:
            speedup = baseline_dur / duration_us

        summary_parts = [
            f"CORRECTNESS: PASS",
            f"Metrix duration_us: {duration_us:.2f}" if duration_us > 0 else "Metrix: N/A",
            f"Baseline duration_us: {baseline_dur:.2f}" if baseline_dur > 0 else "",
            f"Speedup: {speedup:.4f}x",
            f"Eval time: {elapsed:.1f}s",
        ]
        # Add extra metrics
        for key in ["memory.hbm_bandwidth_utilization", "memory.coalescing_efficiency",
                     "memory.l2_hit_rate", "bottleneck"]:
            val = result.metrics.get(key)
            if val is not None:
                summary_parts.append(f"{key}: {val}")

        summary = "\n".join(p for p in summary_parts if p)
        print(summary)

        return {
            "success": True,
            "combined_score": speedup,
            "final_score": speedup,
            "correctness_score": 1.0,
            "performance_metrics": speedup,
            "duration_us": duration_us,
            "baseline_duration_us": baseline_dur,
            "speedup": speedup,
            "hw_metrics": {
                k: v for k, v in result.metrics.items()
                if isinstance(v, (int, float))
            },
            "summary": summary,
            "error": None,
        }

    except Exception as e:
        traceback.print_exc()
        return _failure_result(f"Evaluation crashed: {e}")


def _failure_result(error_msg: str) -> Dict[str, Any]:
    """Return a standard failure dict that OpenEvolve understands."""
    print(f"  FAIL: {error_msg}")
    return {
        "success": False,
        "combined_score": 0.0,
        "final_score": 0.0,
        "correctness_score": 0.0,
        "performance_metrics": 0.0,
        "error": error_msg,
        "summary": f"Evaluation failed: {error_msg}",
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the GEAK-Eval evaluator")
    parser.add_argument("kernel_path", help="Path to a kernel.py file")
    args = parser.parse_args()

    # For self-test, require env vars to be set
    if not os.environ.get("GEAK_COMMANDMENT_PATH"):
        print("ERROR: Set GEAK_COMMANDMENT_PATH env var first.")
        print("Run run_openevolve.py to generate COMMANDMENT.md.")
        sys.exit(1)

    result = evaluate(
        test_suite_path="",
        program_text=args.kernel_path,
        ref_wrapper_path="",
        wrapper_fn_name="",
        unit_tests_path="",
        n_warmup=0,
        n_iters=0,
        atol=0,
        rtol=0,
        verbose=True,
    )
    print(f"\nResult: combined_score={result['combined_score']}")
