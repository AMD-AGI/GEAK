"""Deprecated: ``classify_kernel_category`` has moved to ``memory.cross_session``.

This module is retained for one release as a deprecation shim.  New code
should import from ``minisweagent.memory.cross_session`` directly:

    from minisweagent.memory.cross_session import classify_kernel_category

The shim will be removed in a follow-up cleanup commit (see plan §13.2-D
row 25 and §I-G).
"""

from __future__ import annotations

import warnings

from minisweagent.memory.cross_session import classify_kernel_category as _classify_kernel_category


def classify_kernel_category(kernel_path: str) -> str:  # noqa: D401 — see canonical docstring
    """Backwards-compatible re-export.  Emits DeprecationWarning on first use."""
    warnings.warn(
        "minisweagent.memory.cross_session_memory.classify_kernel_category is "
        "deprecated; import from minisweagent.memory.cross_session instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _classify_kernel_category(kernel_path)


__all__ = ["classify_kernel_category"]
