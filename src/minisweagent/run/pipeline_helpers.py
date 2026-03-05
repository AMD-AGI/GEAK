# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.pipeline.helpers`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.pipeline_helpers is deprecated, "
    "use minisweagent.run.pipeline.helpers",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.pipeline import helpers as _mod  # noqa: E402
from minisweagent.run.pipeline.helpers import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
