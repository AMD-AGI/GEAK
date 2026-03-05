# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.pipeline.task_planner`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.task_planner is deprecated, "
    "use minisweagent.run.pipeline.task_planner",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.pipeline import task_planner as _mod  # noqa: E402
from minisweagent.run.pipeline.task_planner import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
