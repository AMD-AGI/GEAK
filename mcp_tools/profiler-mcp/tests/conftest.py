"""Shared fixtures for profiler-mcp tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure profiler-mcp and metrix-mcp are importable
_repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
for sub in [
    str(Path(_repo_root) / "mcp_tools" / "profiler-mcp" / "src"),
    str(Path(_repo_root) / "mcp_tools" / "metrix-mcp" / "src"),
    str(Path(_repo_root) / "src"),
]:
    if sub not in sys.path:
        sys.path.insert(0, sub)


def _has_gpu():
    """Check if an AMD GPU is available via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


HAS_GPU = _has_gpu()

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="No AMD GPU available")

DEFAULT_TEST_COMMAND = (
    "python3 /home/sdubagun/work/repos/AIG-Eval/tasks/geak_eval/rope/kernel.py --profile"
)


@pytest.fixture
def test_command():
    """Kernel profiling command. Override with PROFILER_TEST_COMMAND env var."""
    return os.environ.get("PROFILER_TEST_COMMAND", DEFAULT_TEST_COMMAND)


@pytest.fixture
def test_workdir():
    """Working directory for rocprof-compute. Override with PROFILER_TEST_WORKDIR."""
    return os.environ.get(
        "PROFILER_TEST_WORKDIR",
        "/home/sdubagun/work/repos/AIG-Eval/tasks/geak_eval/rope",
    )
