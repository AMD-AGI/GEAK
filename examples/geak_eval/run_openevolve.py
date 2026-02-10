#!/usr/bin/env python3
# Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.

"""
Run OpenEvolve evolutionary optimisation on a GPU kernel.

Architecture:
  COMMANDMENT.md is the ONLY contract between the caller and OpenEvolve.
  It specifies exact shell commands for SETUP, CORRECTNESS, and PROFILE.
  OpenEvolve does NOT care what those commands point to -- it just executes
  them.  The caller (mini-SWE-agent, run.sh, or any other orchestrator) is
  responsible for creating:
    - correctness checking scripts (tailored to the specific kernel)
    - profiling/benchmarking scripts (using Metrix / kernel-profile)
    - the COMMANDMENT.md itself (after validating commands on baseline)

  This makes the system generic: it works for Triton, CK, HIP, ASM, or any
  other kernel type, with one file or many, in any language.

Usage modes:

  1. PRE-BUILT COMMANDMENT (recommended for agents):
     The caller creates and validates COMMANDMENT.md + baseline_metrics.json
     BEFORE invoking this script, then passes them in:

       python run_openevolve.py /path/to/kernel.py \\
           --commandment /path/to/COMMANDMENT.md \\
           --baseline-metrics /path/to/baseline_metrics.json

     In this mode, steps 1-4 are skipped entirely.  OpenEvolve starts
     immediately using the frozen COMMANDMENT.

  2. AUTO-BUILD (AIG-Eval convenience):
     If --commandment is NOT provided, the script auto-builds commands
     using the AIG-Eval kernel interface (triton_op, torch_op, EVAL_CONFIGS,
     --profile).  This is a convenience for the AIG-Eval kernel suite only.

       python run_openevolve.py /path/to/kernel.py --iterations 10

Pipeline (auto-build mode):
  1. Sanity-check profiling tools
  2. Build evaluation commands (includes warm-up + Metrix profiling)
  3. Validate commands on baseline → profile metrics become BASELINE
     (baseline is measured using the EXACT same commands that will be
     frozen in COMMANDMENT.md, including warm-up, ensuring apple-to-apple
     comparison with every candidate)
  4. Write COMMANDMENT.md (FROZEN -- warm-up + profiling commands included)
  5. Run OpenEvolve (each candidate evaluated with frozen COMMANDMENT)

Pipeline (pre-built mode):
  1. Load COMMANDMENT.md and baseline metrics
  2. Run OpenEvolve
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Bootstrap: make sure openevolve is importable
# ---------------------------------------------------------------------------
_GEAK_OE_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _GEAK_OE_ROOT not in sys.path:
    sys.path.insert(0, _GEAK_OE_ROOT)

_GEAK_EVAL_DIR = str(Path(__file__).resolve().parent)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("run_openevolve")


# ===========================================================================
# GPU detection
# ===========================================================================

def _detect_available_gpus() -> list:
    """
    Detect available GPU device IDs.

    Tries rocm-smi first (works even when HIP_VISIBLE_DEVICES is set),
    then falls back to torch.cuda.device_count(), then to a single GPU 0.
    """
    import re as _re

    # Try rocm-smi (most reliable for AMD GPUs)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpu_ids = sorted(set(
                int(m.group(1))
                for m in _re.finditer(r"GPU\[(\d+)\]", result.stdout)
            ))
            if gpu_ids:
                logger.info(f"Detected {len(gpu_ids)} GPUs via rocm-smi")
                return gpu_ids
    except Exception:
        pass

    # Fallback: torch
    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            logger.info(f"Detected {count} GPUs via torch.cuda")
            return list(range(count))
    except Exception:
        pass

    # Last resort
    logger.warning("Could not detect GPUs -- assuming single GPU 0")
    return [0]


# ===========================================================================
# Metrix profiling (shared between modes)
# ===========================================================================

def profile_with_metrix(
    kernel_path: str, gpu_device: int = 0, timeout: int = 300
) -> Optional[Dict[str, Any]]:
    """
    Run Metrix hardware profiler (kernel-profile) on a kernel.

    Includes a warm-up pass to ensure stable measurements:
      1. Dry-run the kernel once (populates Triton JIT cache, ramps GPU power)
      2. Run Metrix profiling with 5 replays for statistical stability

    Returns a dict of hardware metrics or None if Metrix is unavailable.
    """
    if shutil.which("kernel-profile") is None:
        logger.warning("kernel-profile not found -- skipping Metrix profiling")
        return None

    try:
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu_device)

        # --- Warm-up passes ---
        # Run the kernel twice without profiling to populate Triton JIT cache
        # and ramp GPU power state to steady operation.  A single warm-up is
        # not always sufficient after a long idle period.
        warmup_cmd = f"python {kernel_path} --profile"
        for i in range(2):
            logger.info(f"  Warm-up {i+1}/2: {warmup_cmd}")
            subprocess.run(
                warmup_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

        # --- Actual profiling ---
        cmd = [
            "kernel-profile",
            f"python {kernel_path} --profile",
            "--gpu-devices", str(gpu_device),
            "--auto-select",
            "--replays", "5",
        ]
        logger.info(f"  Profile: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            logger.warning(f"kernel-profile exited {result.returncode}")
            logger.debug(f"stderr: {result.stderr[:500]}")
            return None

        metrics = _parse_metrix_output(result.stdout + "\n" + result.stderr)
        if metrics:
            logger.info(f"Metrix: {len(metrics)} metrics collected")
        return metrics

    except subprocess.TimeoutExpired:
        logger.warning("kernel-profile timed out")
        return None
    except Exception as e:
        logger.warning(f"Metrix profiling failed: {e}")
        return None


def _parse_metrix_output(output: str) -> Dict[str, Any]:
    """Parse kernel-profile / Metrix output into a metrics dict."""
    import re

    metrics: Dict[str, Any] = {}

    # Parse key: value lines (e.g., "duration_us: 104.27")
    kv_pattern = r"^\s*([\w.]+)\s*:\s*([\d.]+(?:e[+-]?\d+)?)"
    for match in re.finditer(kv_pattern, output, re.MULTILINE):
        key = match.group(1).strip()
        try:
            metrics[key] = float(match.group(2))
        except ValueError:
            pass

    # Also look for Bottleneck and Kernel name
    bottleneck_match = re.search(r"Bottleneck:\s*(\w+)", output)
    if bottleneck_match:
        metrics["bottleneck"] = bottleneck_match.group(1)

    kernel_match = re.search(r"Kernel:\s*(\S+)", output)
    if kernel_match:
        metrics["kernel_name"] = kernel_match.group(1)

    return metrics


def _check_profiling_target_mismatch(
    metrics: Dict[str, Any], kernel_path: str
) -> None:
    """
    Warn if the profiled kernel function is NOT defined in the editable
    kernel.py file.

    This catches cases like nsa_backward where kernel-profile auto-selects
    a Triton kernel that lives in a *dependency* (e.g. triton_nsa_kernel)
    rather than in the editable source.  When this happens, OpenEvolve's
    mutations to kernel.py will have no effect on the profiled bottleneck.
    """
    profiled_name = metrics.get("kernel_name", "")
    if not profiled_name:
        return

    # Read the kernel source and extract function definitions
    try:
        with open(kernel_path, "r") as f:
            source = f.read()
    except Exception:
        return

    import re
    # Match "def <name>" and "@triton.jit" decorated functions
    fn_names = set(re.findall(r"(?:^|\n)\s*def\s+(\w+)\s*\(", source))
    # Also match Triton JIT kernel names (may differ from Python def names)
    jit_names = set(re.findall(r"@triton\.(?:jit|autotune)[\s\S]*?\ndef\s+(\w+)\s*\(", source))
    all_names = fn_names | jit_names

    # The profiled kernel name often has a suffix (e.g. "_0d1d2d3d")
    # Try to match the base name against known function names.
    base_profiled = profiled_name.split("_0d")[0]  # strip Triton mangled suffix

    matched = False
    for name in all_names:
        if name in base_profiled or base_profiled in name:
            matched = True
            break

    if not matched:
        logger.warning("")
        logger.warning("=" * 60)
        logger.warning("PROFILING TARGET MISMATCH WARNING")
        logger.warning("=" * 60)
        logger.warning(
            f"  Metrix profiled kernel: '{profiled_name}'"
        )
        logger.warning(
            f"  But this function is NOT found in: {kernel_path}"
        )
        logger.warning(
            f"  Functions in kernel.py: {sorted(all_names)}"
        )
        logger.warning(
            "  The profiled bottleneck may be in a dependency rather than"
        )
        logger.warning(
            "  the editable code. OpenEvolve mutations may not affect it."
        )
        logger.warning(
            "  Consider: (1) profile a specific Triton kernel from the"
        )
        logger.warning(
            "  editable file, or (2) include the dependency in the"
        )
        logger.warning(
            "  multi-file program so OpenEvolve can modify it."
        )
        logger.warning("=" * 60)
        logger.warning("")


# ===========================================================================
# AIG-Eval auto-build helpers (only used when --commandment is NOT provided)
# ===========================================================================

def build_evaluation_commands(
    kernel_path: str,
    gpu_device: int,
) -> Dict[str, list]:
    """
    Build shell commands for AIG-Eval kernels that implement the standard
    interface (triton_op, torch_op, EVAL_CONFIGS, --profile).

    NOTE: This is an AIG-Eval convenience function.  For generic kernels
    (CK, HIP, ASM, multi-file, etc.), the mini-SWE-agent should create
    its own correctness and profiling scripts and pass a pre-built
    COMMANDMENT.md via --commandment.

    Returns:
        dict with keys "setup", "correctness", "profile" -- each a list
        of shell command strings.
    """
    correctness_script = os.path.join(_GEAK_EVAL_DIR, "correctness_check.py")
    baseline_abs = os.path.abspath(kernel_path)

    # NOTE: All commands use ${GEAK_GPU_DEVICE} instead of a hardcoded GPU ID.
    # This variable is set by the CommandmentEvaluator at runtime, allowing
    # different candidates to be evaluated on different GPUs in parallel
    # without any two candidates sharing a GPU.
    setup_commands = [
        "export HIP_VISIBLE_DEVICES=${GEAK_GPU_DEVICE}",
        "export PYTHONPATH=${GEAK_WORK_DIR}:${PYTHONPATH}",
    ]

    # CORRECTNESS: compare generated triton_op against baseline torch_op
    # ${GEAK_WORK_DIR}/kernel.py is where the candidate file is written
    correctness_commands = [
        f"python {correctness_script} --baseline {baseline_abs} --generated ${{GEAK_WORK_DIR}}/kernel.py",
    ]

    # PROFILE: run Metrix hardware profiler on the candidate
    # IMPORTANT: warm-up command FIRST to ensure:
    #   - Triton JIT cache is populated (first compile is slow)
    #   - GPU power state has ramped up to steady state
    #   - HBM/cache state is representative of steady operation
    # This warm-up is part of the COMMANDMENT and is executed identically
    # for BOTH baseline validation AND every candidate evaluation.
    # NOTE: --gpu-devices uses ${GEAK_GPU_DEVICE} for GPU isolation.
    # Two warm-up runs are needed to handle cold GPU (idle power state,
    # Triton JIT compilation, HBM warmup).  A single warm-up is not
    # always sufficient after a long idle period.
    profile_commands = [
        # Warm-up 1: trigger Triton JIT compilation + initial GPU ramp
        "python ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true",
        # Warm-up 2: ensure GPU is at steady power state
        "python ${GEAK_WORK_DIR}/kernel.py --profile > /dev/null 2>&1 || true",
        # Actual Metrix profiling (now with warm cache + steady GPU)
        'kernel-profile "python ${GEAK_WORK_DIR}/kernel.py --profile" --gpu-devices ${GEAK_GPU_DEVICE} --auto-select --replays 5',
    ]

    return {
        "setup": setup_commands,
        "correctness": correctness_commands,
        "profile": profile_commands,
    }


def _run_command(cmd: str, work_dir: str, env: dict, timeout: int = 300):
    """Run a single shell command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
            env=env,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def validate_commands_on_baseline(
    commands: Dict[str, list],
    kernel_path: str,
    gpu_device: int,
) -> Dict[str, Any]:
    """
    Run evaluation commands on the baseline kernel to confirm they work
    BEFORE writing COMMANDMENT.md.

    Returns:
        dict with "success", "profile_metrics", "stdout", "stderr", "error"
    """
    import tempfile

    kernel_path = os.path.abspath(kernel_path)
    kernel_basename = os.path.basename(kernel_path)

    work_dir = tempfile.mkdtemp(prefix="oe_validate_baseline_")
    logger.info(f"Validating commands on baseline (work_dir={work_dir})...")

    try:
        # Copy baseline kernel into the temp work_dir
        dst = os.path.join(work_dir, kernel_basename)
        shutil.copy2(kernel_path, dst)

        # Build env with GEAK_WORK_DIR, GEAK_KERNEL_DIR, and GEAK_GPU_DEVICE
        # GEAK_GPU_DEVICE is the variable used in COMMANDMENT.md commands
        # so that the same commands work for any GPU.
        env = os.environ.copy()
        env["GEAK_WORK_DIR"] = work_dir
        env["GEAK_KERNEL_DIR"] = os.path.dirname(kernel_path)
        env["GEAK_GPU_DEVICE"] = str(gpu_device)
        env["HIP_VISIBLE_DEVICES"] = str(gpu_device)

        all_stdout = []
        all_stderr = []

        # ---- SETUP ----
        for cmd in commands.get("setup", []):
            logger.info(f"  [SETUP] {cmd}")
            ok, stdout, stderr = _run_command(cmd, work_dir, env)
            all_stdout.append(stdout)
            all_stderr.append(stderr)
            if not ok:
                msg = f"SETUP command failed: {cmd}\nstderr: {stderr[-500:]}"
                logger.error(msg)
                return {"success": False, "error": msg,
                        "stdout": "\n".join(all_stdout),
                        "stderr": "\n".join(all_stderr)}

        # ---- CORRECTNESS ----
        for cmd in commands.get("correctness", []):
            logger.info(f"  [CORRECTNESS] {cmd}")
            ok, stdout, stderr = _run_command(cmd, work_dir, env)
            all_stdout.append(stdout)
            all_stderr.append(stderr)
            if not ok:
                msg = (f"CORRECTNESS command failed on baseline (this should "
                       f"never happen -- baseline vs baseline must pass).\n"
                       f"Command: {cmd}\nstderr: {stderr[-500:]}")
                logger.error(msg)
                return {"success": False, "error": msg,
                        "stdout": "\n".join(all_stdout),
                        "stderr": "\n".join(all_stderr)}
            logger.info("  [CORRECTNESS] PASS")

        # ---- PROFILE ----
        profile_metrics: Dict[str, Any] = {}
        for cmd in commands.get("profile", []):
            logger.info(f"  [PROFILE] {cmd}")
            ok, stdout, stderr = _run_command(cmd, work_dir, env, timeout=300)
            all_stdout.append(stdout)
            all_stderr.append(stderr)
            if not ok:
                msg = (f"PROFILE command failed on baseline.\n"
                       f"Command: {cmd}\nstderr: {stderr[-500:]}")
                logger.error(msg)
                return {"success": False, "error": msg,
                        "stdout": "\n".join(all_stdout),
                        "stderr": "\n".join(all_stderr)}
            # Parse Metrix output
            profile_metrics = _parse_metrix_output(stdout + "\n" + stderr)

        # Verify that profiling produced a latency metric
        dur = profile_metrics.get("duration_us", 0)
        if dur <= 0:
            logger.warning(
                f"PROFILE commands ran successfully but did not produce "
                f"duration_us metric (got: {profile_metrics}). "
                f"Profiling may fall back to other metrics."
            )
        else:
            logger.info(f"  [PROFILE] duration_us = {dur:.2f}")

        # Check if the profiled kernel is actually in the editable file
        _check_profiling_target_mismatch(profile_metrics, kernel_path)

        logger.info("All commands validated successfully on baseline.")
        return {
            "success": True,
            "profile_metrics": profile_metrics,
            "stdout": "\n".join(all_stdout),
            "stderr": "\n".join(all_stderr),
            "error": None,
        }

    finally:
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


def write_frozen_commandment(
    commands: Dict[str, list],
    output_dir: str,
    kernel_path: str,
    baseline_metrics: Dict[str, Any],
) -> str:
    """
    Write COMMANDMENT.md with the validated, working commands.

    Called ONLY after validate_commands_on_baseline() passes.
    From this point forward, COMMANDMENT.md is FROZEN.

    Returns:
        Path to the written COMMANDMENT.md
    """
    from openevolve.commandment_evaluator import CommandmentGenerator

    gen = CommandmentGenerator()
    commandment_path = os.path.join(output_dir, "COMMANDMENT.md")
    gen.save(
        output_path=commandment_path,
        kernel_dir=os.path.dirname(os.path.abspath(kernel_path)),
        setup_commands=commands["setup"],
        correctness_commands=commands["correctness"],
        profile_commands=commands["profile"],
        profiling_results=baseline_metrics,
    )

    # Log the frozen hash for integrity verification
    import hashlib
    with open(commandment_path, "r") as f:
        content = f.read()
    frozen_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    logger.info("=" * 60)
    logger.info("COMMANDMENT.md WRITTEN AND FROZEN")
    logger.info(f"  Path:   {commandment_path}")
    logger.info(f"  SHA256: {frozen_hash}")
    logger.info("  These commands are now IMMUTABLE for the entire")
    logger.info("  OpenEvolve evolution run.")
    logger.info("=" * 60)

    return commandment_path


# ===========================================================================
# OpenEvolve configuration
# ===========================================================================

def build_config(
    config_path: Optional[str],
    kernel_path: str,
    max_iterations: int,
    output_dir: str,
    api_key: Optional[str] = None,
) -> "Config":
    """Build an OpenEvolve Config, merging defaults with user overrides."""
    from openevolve.config import Config, load_config

    if config_path and os.path.isfile(config_path):
        config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        bundled = Path(__file__).parent / "config.yaml"
        if bundled.is_file():
            config = load_config(str(bundled))
            logger.info(f"Loaded bundled config from {bundled}")
        else:
            config = Config()
            logger.info("Using default OpenEvolve config")

    config.max_iterations = max_iterations

    # Ensure eval_dir is an absolute path under output_dir
    eval_dir = os.path.join(output_dir, "evals")
    os.makedirs(eval_dir, exist_ok=True)
    config.evaluator.eval_dir = eval_dir

    # Set API key from environment if not in config
    resolved_key = (
        api_key
        or os.environ.get("AMD_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if resolved_key:
        for model_cfg in config.llm.models:
            if not model_cfg.api_key:
                model_cfg.api_key = resolved_key
        for model_cfg in config.llm.evaluator_models:
            if not model_cfg.api_key:
                model_cfg.api_key = resolved_key
        os.environ["OPENAI_API_KEY"] = resolved_key

    return config


# ===========================================================================
# Main pipeline
# ===========================================================================

async def run_openevolve(
    kernel_path: str,
    output_dir: str,
    max_iterations: int,
    gpu_device: int,
    config_path: Optional[str],
    api_key: Optional[str],
    skip_profiling: bool,
    commandment_path: Optional[str] = None,
    baseline_metrics_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full OpenEvolve pipeline on a kernel.

    Two modes:

    A) Pre-built COMMANDMENT (commandment_path is provided):
       The caller (mini-SWE-agent) has already created and validated
       COMMANDMENT.md and baseline_metrics.json.  We skip straight to
       running OpenEvolve.  This is the generic path for any kernel type.

    B) Auto-build (commandment_path is None):
       We auto-build commands using the AIG-Eval kernel interface,
       validate them on the baseline, write COMMANDMENT.md, then run
       OpenEvolve.  This is the convenience path for AIG-Eval kernels.
    """
    from openevolve import OpenEvolve

    kernel_path = os.path.abspath(kernel_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    evaluator_path = str(Path(__file__).parent / "geak_eval_evaluator.py")

    logger.info("=" * 60)
    logger.info("OpenEvolve Kernel Optimisation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Kernel:      {kernel_path}")
    logger.info(f"Evaluator:   {evaluator_path}")
    logger.info(f"Output:      {output_dir}")
    logger.info(f"Iterations:  {max_iterations}")
    logger.info(f"GPU:         {gpu_device}")

    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_device)

    # ==================================================================
    # Mode A: Pre-built COMMANDMENT (agent-provided)
    # ==================================================================
    if commandment_path:
        commandment_path = os.path.abspath(commandment_path)
        if not os.path.isfile(commandment_path):
            return {"success": False, "error": f"COMMANDMENT.md not found: {commandment_path}"}

        logger.info("")
        logger.info("Mode: PRE-BUILT COMMANDMENT (agent-provided)")
        logger.info(f"  COMMANDMENT: {commandment_path}")
        logger.info("  Skipping auto-build -- COMMANDMENT was created and")
        logger.info("  validated by the calling agent.")

        # Load baseline metrics
        baseline_metrics = _load_baseline_metrics(
            baseline_metrics_path, commandment_path, output_dir
        )

        # If no baseline metrics file was provided but we have a kernel,
        # run Metrix profiling to get them
        if not baseline_metrics and not skip_profiling:
            logger.info("  No baseline metrics provided -- profiling with Metrix...")
            baseline_metrics = profile_with_metrix(kernel_path, gpu_device) or {}
            if baseline_metrics:
                bm_path = os.path.join(output_dir, "baseline_metrics.json")
                with open(bm_path, "w") as f:
                    json.dump(baseline_metrics, f, indent=2)
                baseline_metrics_path = bm_path

        # Save baseline metrics for the evaluator
        if baseline_metrics and not baseline_metrics_path:
            baseline_metrics_path = os.path.join(output_dir, "baseline_metrics.json")
            with open(baseline_metrics_path, "w") as f:
                json.dump(baseline_metrics, f, indent=2)

    # ==================================================================
    # Mode B: Auto-build COMMANDMENT (AIG-Eval convenience)
    # ==================================================================
    else:
        logger.info("")
        logger.info("Mode: AUTO-BUILD (AIG-Eval kernel interface)")
        baseline_metrics, commandment_path, baseline_metrics_path = (
            await _auto_build_commandment(
                kernel_path, output_dir, gpu_device, skip_profiling
            )
        )
        if commandment_path is None:
            # _auto_build_commandment already logged the error
            return {"success": False, "error": "Auto-build COMMANDMENT failed"}

    # ==================================================================
    # Run OpenEvolve (common to both modes)
    # ==================================================================
    os.environ["GEAK_COMMANDMENT_PATH"] = commandment_path
    if baseline_metrics_path:
        os.environ["GEAK_BASELINE_METRICS"] = baseline_metrics_path
    os.environ["GEAK_KERNEL_DIR"] = os.path.dirname(kernel_path)
    os.environ["GEAK_BASELINE_KERNEL"] = kernel_path

    # ------------------------------------------------------------------
    # Detect available GPUs for parallel evaluation
    # ------------------------------------------------------------------
    available_gpus = _detect_available_gpus()
    logger.info("")
    logger.info(f"Available GPUs: {len(available_gpus)} -- {available_gpus}")

    # Set GEAK_GPU_IDS so the Evaluator's GPUPool knows which GPUs to use
    os.environ["GEAK_GPU_IDS"] = ",".join(str(g) for g in available_gpus)

    logger.info("")
    logger.info("Running OpenEvolve evolutionary optimisation...")
    logger.info(f"  COMMANDMENT: {commandment_path}")
    logger.info("  Every candidate scored by frozen COMMANDMENT commands")
    config = build_config(
        config_path, kernel_path, max_iterations, output_dir, api_key
    )

    # Set parallel_evaluations to match the number of available GPUs
    # so each candidate gets its own exclusive GPU.
    config.evaluator.parallel_evaluations = len(available_gpus)

    logger.info(f"  Population:  {config.database.population_size}")
    logger.info(f"  Islands:     {config.database.num_islands}")
    logger.info(f"  Max iters:   {config.max_iterations}")
    logger.info(f"  Diff-based:  {config.diff_based_evolution}")
    logger.info(f"  GPUs:        {len(available_gpus)} ({available_gpus})")
    logger.info(f"  Parallel eval: {config.evaluator.parallel_evaluations}")
    logger.info(f"  GPU isolation: each candidate on exclusive GPU")
    logger.info("")

    start_time = time.time()

    oe = OpenEvolve(
        initial_program_path=kernel_path,
        evaluation_file=evaluator_path,
        config=config,
        output_dir=output_dir,
    )

    best_program = await oe.run()

    elapsed = time.time() - start_time
    logger.info(f"OpenEvolve completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    result = {
        "success": True,
        "elapsed_seconds": elapsed,
        "iterations": best_program.generation if best_program else 0,
        "best_score": (
            best_program.metrics.get("combined_score", 0)
            if best_program else 0
        ),
        "best_metrics": best_program.metrics if best_program else {},
        "baseline_metrics": baseline_metrics,
        "kernel_path": kernel_path,
        "output_dir": output_dir,
        "commandment_path": commandment_path,
    }

    if best_program and best_program.code:
        best_kernel_path = os.path.join(output_dir, "best_kernel.py")
        with open(best_kernel_path, "w") as f:
            f.write(best_program.code)
        result["best_kernel_path"] = best_kernel_path
        logger.info(f"Saved best kernel to {best_kernel_path}")

        opt_path = os.path.join(
            os.path.dirname(kernel_path),
            os.path.splitext(os.path.basename(kernel_path))[0]
            + "_optimized.py",
        )
        with open(opt_path, "w") as f:
            f.write(best_program.code)
        result["optimized_kernel_path"] = opt_path
        logger.info(f"Saved optimised kernel to {opt_path}")

    summary_path = os.path.join(output_dir, "openevolve_result.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved result summary to {summary_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Best score (speedup):   {result['best_score']}")
    logger.info(f"  Iterations completed:   {result['iterations']}")
    logger.info(f"  Total time:             {elapsed:.1f}s")
    if baseline_metrics.get("duration_us"):
        logger.info(f"  Baseline latency (us):  {baseline_metrics['duration_us']:.2f}")
    if baseline_metrics.get("bottleneck"):
        logger.info(f"  Baseline bottleneck:    {baseline_metrics['bottleneck']}")
    logger.info(f"  COMMANDMENT:            {commandment_path}")
    logger.info("=" * 60)

    return result


# ===========================================================================
# Helper: load baseline metrics from various sources
# ===========================================================================

def _load_baseline_metrics(
    metrics_path: Optional[str],
    commandment_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Load baseline metrics from (in priority order):
      1. Explicit --baseline-metrics JSON file
      2. BASELINE METRICS section embedded in COMMANDMENT.md
      3. baseline_metrics.json in output_dir
    """
    # 1. Explicit file
    if metrics_path and os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        logger.info(f"  Baseline metrics loaded from {metrics_path} ({len(metrics)} keys)")
        return metrics

    # 2. Embedded in COMMANDMENT.md
    try:
        with open(commandment_path, "r") as f:
            content = f.read()
        import re
        json_match = re.search(
            r"## BASELINE METRICS.*?```json\s*\n(.*?)```",
            content, re.DOTALL
        )
        if json_match:
            metrics = json.loads(json_match.group(1))
            logger.info(f"  Baseline metrics extracted from COMMANDMENT.md ({len(metrics)} keys)")
            return metrics
    except Exception as e:
        logger.debug(f"  Could not extract baseline metrics from COMMANDMENT: {e}")

    # 3. Default location in output_dir
    default_path = os.path.join(output_dir, "baseline_metrics.json")
    if os.path.isfile(default_path):
        with open(default_path, "r") as f:
            metrics = json.load(f)
        logger.info(f"  Baseline metrics loaded from {default_path} ({len(metrics)} keys)")
        return metrics

    logger.warning("  No baseline metrics found -- speedup calculation may be inaccurate")
    return {}


# ===========================================================================
# Helper: auto-build COMMANDMENT for AIG-Eval kernels
# ===========================================================================

async def _auto_build_commandment(
    kernel_path: str,
    output_dir: str,
    gpu_device: int,
    skip_profiling: bool,
) -> tuple:
    """
    Auto-build and validate COMMANDMENT.md for AIG-Eval kernels.

    This is the convenience path for kernels that implement the AIG-Eval
    interface (triton_op, torch_op, EVAL_CONFIGS, --profile).  For generic
    kernels, use --commandment instead.

    Returns:
        (baseline_metrics, commandment_path, baseline_metrics_path)
        commandment_path is None if validation failed.
    """
    baseline_metrics: Dict[str, Any] = {}
    baseline_metrics_path = os.path.join(output_dir, "baseline_metrics.json")

    # Step 1: Sanity-check that profiling tools are available
    if not skip_profiling:
        logger.info("")
        logger.info("Step 1/5: Checking profiling tools...")
        if shutil.which("kernel-profile") is None:
            logger.warning(
                "kernel-profile (Metrix) not found -- profiling commands "
                "may fail during validation."
            )
        else:
            logger.info("  kernel-profile (Metrix): available")
    else:
        logger.info("Step 1/5: Profiling skipped (--skip-profiling)")

    # Step 2: Build evaluation commands (AIG-Eval specific)
    # NOTE: PROFILE commands include a warm-up run BEFORE the actual
    # Metrix profiling.  This ensures Triton JIT cache is populated
    # and GPU power state is ramped up.  The SAME warm-up + profile
    # commands are used for baseline validation AND every candidate.
    logger.info("")
    logger.info("Step 2/5: Building evaluation commands (AIG-Eval interface)...")
    commands = build_evaluation_commands(kernel_path, gpu_device)
    logger.info(f"  SETUP:       {len(commands['setup'])} command(s)")
    logger.info(f"  CORRECTNESS: {len(commands['correctness'])} command(s)")
    logger.info(f"  PROFILE:     {len(commands['profile'])} command(s)")
    for i, cmd in enumerate(commands['profile']):
        logger.info(f"    PROFILE[{i}]: {cmd}")

    # Step 3: Validate commands on baseline kernel
    # CRITICAL: The profile metrics from THIS step become the baseline.
    # This ensures the baseline was measured using the EXACT same commands
    # (including warm-up) that will be used for every candidate.
    # NO separate profiling step is used -- apple-to-apple comparison.
    logger.info("")
    logger.info("Step 3/5: Validating commands on baseline kernel...")
    logger.info("  (commands are NOT yet written to COMMANDMENT.md)")
    logger.info("  Profile metrics from this run will be the BASELINE.")
    validation = validate_commands_on_baseline(
        commands=commands,
        kernel_path=kernel_path,
        gpu_device=gpu_device,
    )
    if not validation["success"]:
        logger.error(
            f"Command validation FAILED on baseline -- aborting.\n"
            f"  Error: {validation['error']}"
        )
        return baseline_metrics, None, baseline_metrics_path

    # ALWAYS use validation profile metrics as the baseline.
    # This is the cornerstone of deterministic evaluation: the baseline
    # was measured with the EXACT same PROFILE commands (including warm-up)
    # that will be written to COMMANDMENT.md and used for every candidate.
    val_profile_metrics = validation.get("profile_metrics", {})
    if val_profile_metrics:
        baseline_metrics = val_profile_metrics
        with open(baseline_metrics_path, "w") as f:
            json.dump(baseline_metrics, f, indent=2)
        logger.info(
            f"Baseline metrics from validation profiling "
            f"({len(baseline_metrics)} keys):"
        )
        dur = baseline_metrics.get("duration_us", 0)
        if dur > 0:
            logger.info(f"  duration_us = {dur:.2f}")
        bn = baseline_metrics.get("bottleneck", "")
        if bn:
            logger.info(f"  bottleneck  = {bn}")

        # Also save to kernel directory for reference
        kernel_metrics = os.path.join(
            os.path.dirname(kernel_path), "metrics.json"
        )
        with open(kernel_metrics, "w") as f:
            json.dump(baseline_metrics, f, indent=2)
    else:
        logger.warning(
            "Validation profiling did not produce metrics -- "
            "speedup calculation will be inaccurate."
        )

    logger.info("All commands validated successfully on baseline.")

    # Step 4: Write COMMANDMENT.md (FROZEN from here)
    logger.info("")
    logger.info("Step 4/5: Writing COMMANDMENT.md (frozen)...")
    commandment_path = write_frozen_commandment(
        commands=commands,
        output_dir=output_dir,
        kernel_path=kernel_path,
        baseline_metrics=baseline_metrics,
    )

    return baseline_metrics, commandment_path, baseline_metrics_path


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve evolutionary optimisation on a GPU kernel"
    )
    parser.add_argument(
        "kernel_path",
        help="Path to the kernel file(s) to optimise (main entry point)",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=10,
        help="Max evolution iterations (default: 10)",
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory (default: <kernel_dir>/optimization_output)",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to OpenEvolve config.yaml (default: bundled config)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="LLM API key (default: from AMD_LLM_API_KEY env var)",
    )
    parser.add_argument(
        "--skip-profiling", action="store_true",
        help="Skip Metrix baseline profiling",
    )

    # ---- Pre-built COMMANDMENT mode ----
    parser.add_argument(
        "--commandment", type=str, default=None,
        help=(
            "Path to a pre-built COMMANDMENT.md (created by mini-SWE-agent "
            "or other orchestrator).  When provided, auto-build steps 1-4 "
            "are skipped and OpenEvolve starts immediately."
        ),
    )
    parser.add_argument(
        "--baseline-metrics", type=str, default=None,
        help=(
            "Path to baseline_metrics.json (pre-computed by agent).  "
            "If not provided, metrics are extracted from COMMANDMENT.md "
            "or computed via Metrix."
        ),
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(os.path.abspath(args.kernel_path)),
            "optimization_output",
        )

    result = asyncio.run(
        run_openevolve(
            kernel_path=args.kernel_path,
            output_dir=args.output,
            max_iterations=args.iterations,
            gpu_device=args.gpu,
            config_path=args.config,
            api_key=args.api_key,
            skip_profiling=args.skip_profiling,
            commandment_path=args.commandment,
            baseline_metrics_path=args.baseline_metrics,
        )
    )

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
