# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.pipeline.codebase_context`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.codebase_context is deprecated, "
    "use minisweagent.run.pipeline.codebase_context",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.pipeline import codebase_context as _mod  # noqa: E402
from minisweagent.run.pipeline.codebase_context import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
