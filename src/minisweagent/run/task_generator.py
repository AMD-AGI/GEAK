# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.pipeline.task_generator`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.task_generator is deprecated, "
    "use minisweagent.run.pipeline.task_generator",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.pipeline import task_generator as _mod  # noqa: E402
from minisweagent.run.pipeline.task_generator import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
