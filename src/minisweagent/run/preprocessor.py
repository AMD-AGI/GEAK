# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.pipeline.preprocessor`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.preprocessor is deprecated, "
    "use minisweagent.run.pipeline.preprocessor",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.pipeline import preprocessor as _mod  # noqa: E402
from minisweagent.run.pipeline.preprocessor import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
