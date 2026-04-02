"""Memory system integration layer for GEAK (working-session only).

Environment toggles:
  GEAK_MEMORY_DISABLE=1           -- disable all memory
  GEAK_MEMORY_NO_WORKING=1        -- disable within-session working memory
"""

from __future__ import annotations

import os


def _is_disabled(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip() in ("1", "true", "yes")


def is_memory_enabled() -> bool:
    return not _is_disabled("GEAK_MEMORY_DISABLE")


def is_working_memory_enabled() -> bool:
    return is_memory_enabled() and not _is_disabled("GEAK_MEMORY_NO_WORKING")


def assemble_memory_context(**kwargs) -> str:
    """Placeholder — cross-session memory not yet integrated."""
    return ""


def record_optimization_outcome(**kwargs) -> None:
    """Placeholder — cross-session recording not yet integrated."""
    pass
