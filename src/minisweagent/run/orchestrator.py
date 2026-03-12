# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.pipeline.orchestrator`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.orchestrator is deprecated, "
    "use minisweagent.run.pipeline.orchestrator",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.pipeline import orchestrator as _mod  # noqa: E402
from minisweagent.run.pipeline.orchestrator import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
