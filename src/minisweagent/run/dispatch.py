# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.pipeline.dispatch`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.dispatch is deprecated, "
    "use minisweagent.run.pipeline.dispatch",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.pipeline import dispatch as _mod  # noqa: E402
from minisweagent.run.pipeline.dispatch import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
