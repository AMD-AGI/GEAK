import os
import sys
import threading
from pathlib import Path

import pytest

# Make MCP tool packages importable when running outside Docker.
# In-container, these are installed via `pip install -e mcp_tools/*/` in the
# Dockerfile; for local dev we add their src/ directories to sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _mcp_src in sorted((_REPO_ROOT / "mcp_tools").glob("*/src")):
    _p = str(_mcp_src)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from minisweagent.models import GLOBAL_MODEL_STATS


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-fire",
        action="store_true",
        default=False,
        help="Run fire tests (real API calls that cost money)",
    )


@pytest.fixture(autouse=True)
def _no_git_prompt(monkeypatch):
    """Prevent git from prompting for credentials during tests."""
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


_global_stats_lock = threading.Lock()


@pytest.fixture
def reset_global_stats():
    """Reset global model stats and ensure exclusive access for tests that need it."""
    with _global_stats_lock:
        GLOBAL_MODEL_STATS._cost = 0.0
        GLOBAL_MODEL_STATS._n_calls = 0
        yield
        GLOBAL_MODEL_STATS._cost = 0.0
        GLOBAL_MODEL_STATS._n_calls = 0
