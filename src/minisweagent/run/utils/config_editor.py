# ruff: noqa: I001
"""Backward-compatible shim. Use ``minisweagent.run.config.editor`` instead."""
import warnings as _w
_w.warn(
    "minisweagent.run.utils.config_editor is deprecated, "
    "use minisweagent.run.config.editor",
    DeprecationWarning,
    stacklevel=2,
)
from minisweagent.run.config import editor as _mod  # noqa: E402
from minisweagent.run.config.editor import *  # noqa: F401,F403,E402

def __getattr__(name):
    return getattr(_mod, name)
