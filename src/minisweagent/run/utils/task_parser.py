# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.config.task_parser`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.utils.task_parser is deprecated, "
    "use minisweagent.run.config.task_parser",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.config import task_parser as _mod  # noqa: E402
from minisweagent.run.config.task_parser import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
