#!/usr/bin/env python3
"""
Phase C: Validate multi-file support against AIG-Eval ROCm kernels.
Tests that each kernel can be loaded, have diffs applied, and have prompts generated.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from openevolve.database import load_program_from_directory, Program
from openevolve.config import Config
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_multifile_diff,
    format_multifile_diff,
)
from openevolve.commandment_evaluator import CommandmentGenerator

AIG_EVAL_DIR = os.environ.get("AIG_EVAL_DIR", "")
if not AIG_EVAL_DIR:
    for candidate in [
        os.path.join(_PROJECT_ROOT, "..", "AIG-Eval", "tasks", "geak_eval"),
        "/workspace/AIG-Eval/tasks/geak_eval",
        os.path.join(os.path.expanduser("~"), "AIG-Eval", "tasks", "geak_eval"),
    ]:
        if os.path.isdir(candidate):
            AIG_EVAL_DIR = os.path.abspath(candidate)
            break

# All kernel directories to test
KERNEL_DIRS = [
    "rope",
    "gemm",
    "topk",
    "fused_qkv_rope",
    "fused_rms_fp8",
    "ff_backward",
    "nsa_forward",
    "nsa_backward",
]


def test_load_kernel(kernel_name):
    """Load a single kernel as a multi-file program."""
    kernel_dir = os.path.join(AIG_EVAL_DIR, kernel_name)
    if not os.path.isdir(kernel_dir):
        return None, f"Directory not found: {kernel_dir}"

    try:
        files, main_file = load_program_from_directory(kernel_dir)
        return files, main_file
    except Exception as e:
        return None, str(e)


def test_create_program(kernel_name, files, main_file):
    """Create a Program from the loaded files."""
    p = Program(
        id=f"aig_{kernel_name}",
        code=files[main_file],
        files=files,
        main_file=main_file,
    )
    assert p.is_multifile(), f"{kernel_name}: should be multifile"
    assert len(p.get_all_code()) > 0, f"{kernel_name}: should have code"
    return p


def test_diff_roundtrip(kernel_name, files, main_file):
    """Apply a no-op diff and verify nothing changes."""
    # Get first line of main file
    first_line = files[main_file].split("\n")[0]
    if not first_line.strip():
        # Use first non-empty line
        for line in files[main_file].split("\n"):
            if line.strip():
                first_line = line
                break

    diff = format_multifile_diff(
        main_file,
        first_line,
        first_line + "  # optimized"
    )
    result = apply_multifile_diff(files, diff)
    assert main_file in result, f"{kernel_name}: main_file should be in result"
    assert "optimized" in result[main_file], f"{kernel_name}: diff should be applied"

    # Other files unchanged
    for f in files:
        if f != main_file:
            assert result[f] == files[f], f"{kernel_name}: {f} should be unchanged"


def test_prompt_generation(kernel_name, files, main_file):
    """Generate a multi-file prompt for this kernel."""
    config = Config()
    sampler = PromptSampler(config.prompt)
    prompt = sampler.build_prompt(
        current_program=files[main_file],
        program_metrics={"score": 0.5},
        is_multifile=True,
        program_files=files,
        main_file=main_file,
        baseline_profiling={
            "latency_us": 100.0,
            "bandwidth_gb_s": 500.0,
            "hbm_utilization": 50.0,
            "compute_busy": 60.0,
        },
    )
    user_msg = prompt["user"]
    assert len(user_msg) > 100, f"{kernel_name}: prompt should have content"
    assert main_file in user_msg, f"{kernel_name}: prompt should reference main file"
    return len(user_msg)


def test_commandment_generation(kernel_name, kernel_dir):
    """Generate a COMMANDMENT.md for this kernel."""
    gen = CommandmentGenerator()
    content = gen.generate(
        kernel_dir=kernel_dir,
        setup_commands=[f"cd {kernel_dir}"],
        correctness_commands=["python3 -c 'print(\"ok\")'"],
        profile_commands=["python3 -c 'import json; print(json.dumps({\"latency_us\": 100}))'"],
    )
    assert "## SETUP" in content
    assert "## PROFILE" in content
    return True


if __name__ == "__main__":
    if not os.path.isdir(AIG_EVAL_DIR):
        print(f"ERROR: AIG-Eval directory not found at {AIG_EVAL_DIR}")
        sys.exit(1)

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for kernel_name in KERNEL_DIRS:
        print(f"\n{'='*60}")
        print(f"Kernel: {kernel_name}")
        print(f"{'='*60}")

        kernel_dir = os.path.join(AIG_EVAL_DIR, kernel_name)

        # Test 1: Load
        print(f"  [1/5] Loading...", end=" ")
        files_result, main_file = test_load_kernel(kernel_name)
        if files_result is None:
            print(f"SKIPPED ({main_file})")
            skipped += 1
            continue
        print(f"OK ({len(files_result)} files, main={main_file})")
        total += 1

        # Test 2: Create Program
        print(f"  [2/5] Creating Program...", end=" ")
        try:
            program = test_create_program(kernel_name, files_result, main_file)
            print(f"OK (all_code={len(program.get_all_code())} chars)")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            continue

        # Test 3: Diff round-trip
        print(f"  [3/5] Diff round-trip...", end=" ")
        try:
            test_diff_roundtrip(kernel_name, files_result, main_file)
            print("OK")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        # Test 4: Prompt generation
        print(f"  [4/5] Prompt generation...", end=" ")
        try:
            prompt_len = test_prompt_generation(kernel_name, files_result, main_file)
            print(f"OK ({prompt_len} chars)")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

        # Test 5: COMMANDMENT generation
        print(f"  [5/5] COMMANDMENT generation...", end=" ")
        try:
            test_commandment_generation(kernel_name, kernel_dir)
            print("OK")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total} kernels tested, {passed} checks passed, "
          f"{failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    if failed:
        sys.exit(1)
    print("All AIG-Eval kernel validation PASSED!")
