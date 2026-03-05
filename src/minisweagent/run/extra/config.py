# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.config.global_config`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.extra.config is deprecated, "
    "use minisweagent.run.config.global_config",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.config import global_config as _mod  # noqa: E402
from minisweagent.run.config.global_config import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
