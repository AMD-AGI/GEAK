"""Deprecated compatibility shim for cross-session memory helpers."""

from __future__ import annotations

import warnings

from minisweagent.memory.cross_session import classify_kernel_category as _classify_kernel_category


def classify_kernel_category(kernel_path: str) -> str:  # noqa: D401
    """Backwards-compatible re-export."""
    warnings.warn(
        "minisweagent.memory.cross_session_memory.classify_kernel_category is deprecated; "
        "import from minisweagent.memory.cross_session instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _classify_kernel_category(kernel_path)


__all__ = ["classify_kernel_category"]
