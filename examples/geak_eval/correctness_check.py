#!/usr/bin/env python3
# Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.

"""
Correctness checker for AIG-Eval GPU kernels (Triton / Python interface).

NOTE: This script is an AIG-Eval convenience helper, NOT a required part of
the OpenEvolve pipeline.  It works specifically with kernels that implement
the AIG-Eval interface (triton_op, torch_op, check_correctness, EVAL_CONFIGS).

For generic kernels (CK, HIP, ASM, multi-file, etc.), the mini-SWE-agent
should create its own correctness checking script tailored to that kernel,
and reference it in the COMMANDMENT.md.  The COMMANDMENT.md is the only
contract between the agent and OpenEvolve.

Strategy:
  1. Import both baseline and generated kernel modules
  2. If the baseline has its own check_correctness(), use it:
     - Monkey-patch baseline_mod.triton_op = generated_mod.triton_op
     - Run baseline's check_correctness() for each EVAL_CONFIG
     - This ensures the baseline's torch_op reference, custom tolerances,
       multi-output handling, gradient checks, etc. are all respected
  3. Fallback (no check_correctness): generic torch.allclose comparison

Exit code 0 = all configs pass, non-zero = at least one failed.

Usage:
    python correctness_check.py --baseline /path/to/baseline/kernel.py \\
                                --generated /path/to/candidate/kernel.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from typing import Any, Dict, List, Tuple

# Only set PYTORCH_CUDA_ALLOC_CONF on NVIDIA/CUDA -- on ROCm/HIP the
# "expandable_segments" allocator option is not supported and triggers a
# UserWarning that pollutes stderr, causing false correctness failures.
def _is_rocm() -> bool:
    """Detect whether we are running on a ROCm/HIP platform."""
    # Fast env-var check (set by many ROCm containers and scripts)
    if os.environ.get("HIP_VISIBLE_DEVICES") is not None:
        return True
    if os.environ.get("ROCM_HOME") is not None:
        return True
    # Check for rocm-smi binary
    import shutil
    if shutil.which("rocm-smi") is not None:
        return True
    return False

if not _is_rocm():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


def _load_module(filepath: str, name: str):
    """Dynamically import a Python file as a module."""
    filepath = os.path.abspath(filepath)
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {filepath}")
    mod = importlib.util.module_from_spec(spec)
    # Add the module's directory to sys.path so local imports work
    mod_dir = os.path.dirname(filepath)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Strategy 1: Delegate to kernel's own check_correctness (preferred)
# ---------------------------------------------------------------------------

def _check_via_kernel_own_method(
    baseline_mod,
    generated_mod,
    eval_configs: list,
    seed: int,
    verbose: bool,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Use the baseline kernel's own check_correctness() function but with
    the generated kernel's triton_op substituted in.

    This is a monkey-patch approach: we temporarily replace
    baseline_mod.triton_op with generated_mod.triton_op, then call the
    baseline's check_correctness().  Because Python resolves module-level
    names at call time, the baseline's check_correctness() will call
    generated_mod.triton_op while still using baseline's torch_op,
    get_inputs, tolerances, and comparison logic.
    """
    results = []
    all_passed = True

    # Save original triton_op so we can restore it
    original_triton_op = baseline_mod.triton_op

    # Also patch any internal name that might reference triton_op
    # (some kernels may import helper functions that call triton_op)
    baseline_mod.triton_op = generated_mod.triton_op

    if verbose:
        print(f"{'Config':<35} {'Status':>8} {'Details':>20}")
        print("-" * 68)

    try:
        for config in eval_configs:
            config_str = str(config)
            try:
                # Set deterministic seed
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                # Call the baseline's own check_correctness with this config
                result = baseline_mod.check_correctness(*config)

                passed = result.get("correct", False)
                error = result.get("error", None)

                if not passed:
                    all_passed = False

                # Build detail string from result dict
                detail_parts = []
                for k, v in result.items():
                    if k not in ("correct", "error") and isinstance(v, bool):
                        detail_parts.append(f"{k}={'OK' if v else 'FAIL'}")
                detail = ", ".join(detail_parts) if detail_parts else "-"

                results.append({
                    "config": config,
                    "passed": passed,
                    "max_diff": 0.0 if passed else float("inf"),
                    "error": error,
                    "details": result,
                })

                if verbose:
                    status = "PASS" if passed else "FAIL"
                    if error:
                        detail = str(error)[:30]
                    print(f"{config_str:<35} {status:>8} {detail:>20}")

            except Exception as e:
                all_passed = False
                results.append({
                    "config": config,
                    "passed": False,
                    "max_diff": float("inf"),
                    "error": str(e),
                })
                if verbose:
                    print(f"{config_str:<35} {'ERROR':>8} {str(e)[:30]:>20}")

    finally:
        # Restore original triton_op
        baseline_mod.triton_op = original_triton_op

    if verbose:
        print("-" * 68)
        n_pass = sum(1 for r in results if r["passed"])
        n_total = len(results)
        status = "ALL PASS" if all_passed else f"FAILED ({n_total - n_pass}/{n_total})"
        print(f"Result: {status}")

    return all_passed, results


# ---------------------------------------------------------------------------
# Strategy 2: Generic torch.allclose comparison (fallback)
# ---------------------------------------------------------------------------

def _check_via_generic_comparison(
    baseline_mod,
    generated_mod,
    eval_configs: list,
    rtol: float,
    atol: float,
    seed: int,
    verbose: bool,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Generic comparison: baseline.torch_op(*inputs) vs generated.triton_op(*inputs).

    Handles:
    - Single tensor outputs: direct torch.allclose
    - Tuple outputs: element-wise torch.allclose on each tensor
    """
    # Get tolerances from baseline if available
    rtol = getattr(baseline_mod, "RTOL", rtol)
    atol = getattr(baseline_mod, "ATOL", atol)

    get_inputs = getattr(baseline_mod, "get_inputs", None)
    if get_inputs is None:
        raise AttributeError("Baseline kernel has no get_inputs()")

    for mod, fn_name in [
        (baseline_mod, "torch_op"),
        (generated_mod, "triton_op"),
    ]:
        if not hasattr(mod, fn_name):
            raise AttributeError(f"{fn_name}() not found")

    results = []
    all_passed = True

    if verbose:
        print(f"{'Config':<25} {'Status':>8} {'Max Diff':>12}")
        print("-" * 50)

    for config in eval_configs:
        config_str = str(config)
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            inputs = get_inputs(*config)

            baseline_ref = baseline_mod.torch_op(*inputs)
            generated_out = generated_mod.triton_op(*inputs)

            # Handle tuple outputs (element-wise comparison)
            passed, max_diff = _compare_outputs(
                baseline_ref, generated_out, rtol, atol
            )

            if not passed:
                all_passed = False

            results.append({
                "config": config,
                "passed": passed,
                "max_diff": max_diff,
                "error": None,
            })

            if verbose:
                status = "PASS" if passed else "FAIL"
                diff_str = f"{max_diff:.2e}" if max_diff > 0 else "-"
                print(f"{config_str:<25} {status:>8} {diff_str:>12}")

        except Exception as e:
            all_passed = False
            results.append({
                "config": config,
                "passed": False,
                "max_diff": float("inf"),
                "error": str(e),
            })
            if verbose:
                print(f"{config_str:<25} {'ERROR':>8} {str(e)[:30]}")

    if verbose:
        print("-" * 50)
        n_pass = sum(1 for r in results if r["passed"])
        n_total = len(results)
        status = "ALL PASS" if all_passed else f"FAILED ({n_total - n_pass}/{n_total})"
        print(f"Result: {status}")

    return all_passed, results


def _compare_outputs(
    ref_output, gen_output, rtol: float, atol: float
) -> Tuple[bool, float]:
    """
    Compare outputs, handling single tensors, tuples, and named tuples.

    Returns (passed: bool, max_diff: float).
    """
    if isinstance(ref_output, torch.Tensor) and isinstance(gen_output, torch.Tensor):
        passed = torch.allclose(gen_output, ref_output, rtol=rtol, atol=atol)
        if passed:
            return True, 0.0
        max_diff = torch.max(torch.abs(gen_output.float() - ref_output.float())).item()
        return False, max_diff

    if isinstance(ref_output, (tuple, list)) and isinstance(gen_output, (tuple, list)):
        if len(ref_output) != len(gen_output):
            return False, float("inf")
        all_ok = True
        worst_diff = 0.0
        for r, g in zip(ref_output, gen_output):
            if isinstance(r, torch.Tensor) and isinstance(g, torch.Tensor):
                ok = torch.allclose(g, r, rtol=rtol, atol=atol)
                if not ok:
                    all_ok = False
                    diff = torch.max(torch.abs(g.float() - r.float())).item()
                    worst_diff = max(worst_diff, diff)
            # Skip non-tensor elements (e.g., None, int metadata)
        return all_ok, worst_diff

    # Fallback: try direct comparison
    try:
        passed = torch.allclose(gen_output, ref_output, rtol=rtol, atol=atol)
        return passed, 0.0 if passed else float("inf")
    except Exception:
        return False, float("inf")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_correctness(
    baseline_path: str,
    generated_path: str,
    rtol: float = 1e-1,
    atol: float = 1e-1,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Compare generated kernel against baseline using the best available
    strategy.

    Strategy 1 (preferred): Use kernel's own check_correctness() with
    monkey-patched triton_op.  Handles multi-output, custom validation,
    gradient checks, etc.

    Strategy 2 (fallback): Generic torch.allclose comparison with tuple
    support.

    Returns:
        (all_passed: bool, results: list of per-config dicts)
    """
    baseline_mod = _load_module(baseline_path, "baseline_kernel")
    generated_mod = _load_module(generated_path, "generated_kernel")

    eval_configs = getattr(baseline_mod, "EVAL_CONFIGS", None)
    if eval_configs is None:
        raise AttributeError(
            f"Baseline kernel {baseline_path} has no EVAL_CONFIGS"
        )

    has_check_fn = hasattr(baseline_mod, "check_correctness")
    has_triton_op = hasattr(generated_mod, "triton_op")

    if has_check_fn and has_triton_op:
        if verbose:
            print(f"[Strategy] Using kernel's own check_correctness() "
                  f"with generated triton_op")
        return _check_via_kernel_own_method(
            baseline_mod, generated_mod, eval_configs, seed, verbose,
        )
    else:
        if verbose:
            reason = ("no check_correctness" if not has_check_fn
                      else "no triton_op in generated")
            print(f"[Strategy] Generic comparison ({reason})")
        return _check_via_generic_comparison(
            baseline_mod, generated_mod, eval_configs,
            rtol, atol, seed, verbose,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Check correctness of a generated kernel against baseline"
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Path to the baseline kernel.py",
    )
    parser.add_argument(
        "--generated", required=True,
        help="Path to the generated (candidate) kernel.py",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-1,
        help="Relative tolerance (default: 1e-1)",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-1,
        help="Absolute tolerance (default: 1e-1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic inputs (default: 42)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-config output",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.baseline):
        print(f"ERROR: Baseline file not found: {args.baseline}")
        sys.exit(2)
    if not os.path.isfile(args.generated):
        print(f"ERROR: Generated file not found: {args.generated}")
        sys.exit(2)

    try:
        all_passed, results = check_correctness(
            baseline_path=args.baseline,
            generated_path=args.generated,
            rtol=args.rtol,
            atol=args.atol,
            seed=args.seed,
            verbose=not args.quiet,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR: {e}")
        sys.exit(2)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
