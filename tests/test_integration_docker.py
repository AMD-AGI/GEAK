#!/usr/bin/env python3
"""
Integration test for multi-file support with the GEAK-msa add_kernel.
Run inside the Docker container.
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
    extract_multifile_diffs,
    format_multifile_diff,
)
from openevolve.commandment_evaluator import CommandmentGenerator

ADD_KERNEL_DIR = os.environ.get("GEAK_ADD_KERNEL_DIR", "")
if not ADD_KERNEL_DIR:
    for candidate in [
        os.path.join(_PROJECT_ROOT, "..", "GEAK-msa", "examples", "add_kernel"),
        "/workspace/examples/add_kernel",
        os.path.join(os.path.expanduser("~"), "GEAK-msa", "examples", "add_kernel"),
    ]:
        if os.path.isdir(candidate):
            ADD_KERNEL_DIR = os.path.abspath(candidate)
            break

def test_load_add_kernel():
    """Test 1: Load add_kernel as multi-file program."""
    files, main_file = load_program_from_directory(ADD_KERNEL_DIR)
    print(f"Loaded {len(files)} files, main_file={main_file}")
    for f in sorted(files.keys()):
        print(f"  {f}: {len(files[f])} chars")
    assert len(files) > 0, "No files loaded"
    assert main_file == "kernel.py", f"Expected kernel.py, got {main_file}"
    assert "triton" in files["kernel.py"].lower() or "import" in files["kernel.py"]
    print("  PASSED")

def test_create_multifile_program():
    """Test 2: Create Program with multi-file."""
    files, main_file = load_program_from_directory(ADD_KERNEL_DIR)
    p = Program(id="test_add", code=files[main_file], files=files, main_file=main_file)
    assert p.is_multifile(), "Should be multifile"
    assert len(p.get_all_code()) > 0, "get_all_code should return content"
    print(f"  is_multifile={p.is_multifile()}, all_code_len={len(p.get_all_code())}")
    print("  PASSED")

def test_multifile_diff_roundtrip():
    """Test 3: Multi-file diff round-trip on add_kernel."""
    files, main_file = load_program_from_directory(ADD_KERNEL_DIR)

    # Generate a diff
    if "mask = offsets < n_elements" in files.get("kernel.py", ""):
        diff = format_multifile_diff(
            "kernel.py",
            "    mask = offsets < n_elements",
            "    mask = offsets < n_elements  # bounds check"
        )
    else:
        # Generic diff
        first_line = files[main_file].split("\n")[0]
        diff = format_multifile_diff(
            main_file,
            first_line,
            first_line + "  # modified"
        )

    # Apply diff
    result = apply_multifile_diff(files, diff)
    assert main_file in result, "Main file should be in result"
    assert "modified" in result[main_file] or "bounds check" in result[main_file], \
        "Diff should have been applied"

    # Other files should be unchanged
    for f in files:
        if f != main_file:
            assert result[f] == files[f], f"File {f} should be unchanged"

    print("  PASSED")

def test_multifile_prompt_generation():
    """Test 4: Generate multi-file prompt with profiling data."""
    files, main_file = load_program_from_directory(ADD_KERNEL_DIR)

    config = Config()
    sampler = PromptSampler(config.prompt)
    prompt = sampler.build_prompt(
        current_program=files[main_file],
        program_metrics={"score": 0.5},
        is_multifile=True,
        program_files=files,
        main_file=main_file,
        baseline_profiling={
            "latency_us": 50.0,
            "bandwidth_gb_s": 800.0,
            "hbm_utilization": 65.0,
            "compute_busy": 15.0,
            "l2_hit_rate": 40.0,
        },
    )

    user_msg = prompt["user"]
    assert "kernel.py" in user_msg, "Should contain file listing"
    assert "Latency" in user_msg or "latency" in user_msg, "Should contain profiling"
    assert "Memory-bound" in user_msg, "Should contain bottleneck analysis"
    assert "file:" in user_msg, "Should contain multi-file diff format"
    print(f"  Prompt length: {len(user_msg)} chars")
    print("  PASSED")

def test_commandment_generator():
    """Test 5: Generate COMMANDMENT.md for add_kernel."""
    gen = CommandmentGenerator()
    content = gen.generate(
        kernel_dir=ADD_KERNEL_DIR,
        setup_commands=[
            "export PYTHONPATH=${GEAK_WORK_DIR}:${PYTHONPATH}",
            "mkdir -p ${GEAK_WORK_DIR}/results",
        ],
        correctness_commands=[
            "python -c \"import torch; x=torch.randn(1024, device='cuda'); print('OK')\"",
        ],
        profile_commands=[
            "python -c \"from metrix import Metrix; m=Metrix(arch='gfx942'); print('Profiling done')\"",
        ],
        profiling_results={
            "latency_us": 50.0,
            "bandwidth_gb_s": 800.0,
        },
    )
    assert "## SETUP" in content
    assert "## CORRECTNESS" in content
    assert "## PROFILE" in content
    assert "VALIDATED" in content
    assert "Config Hash:" in content
    print("  PASSED")

def test_metrix_available():
    """Test 6: Check Metrix is importable inside Docker."""
    try:
        from metrix import Metrix
        print(f"  Metrix imported successfully: {Metrix}")
        print("  PASSED")
    except ImportError as e:
        print(f"  WARNING: Metrix not available: {e}")
        print("  SKIPPED (optional)")

if __name__ == "__main__":
    tests = [
        ("Load add_kernel as multi-file", test_load_add_kernel),
        ("Create multifile Program", test_create_multifile_program),
        ("Multi-file diff round-trip", test_multifile_diff_roundtrip),
        ("Multi-file prompt generation", test_multifile_prompt_generation),
        ("COMMANDMENT generator", test_commandment_generator),
        ("Metrix availability", test_metrix_available),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        print(f"\nTest: {name}")
        try:
            func()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
    print("All integration tests PASSED!")
