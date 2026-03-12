# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.pipeline.task_runner`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.task_runner is deprecated, "
    "use minisweagent.run.pipeline.task_runner",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.pipeline import task_runner as _mod  # noqa: E402
from minisweagent.run.pipeline.task_runner import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
