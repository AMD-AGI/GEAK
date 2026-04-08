"""Memory system integration layer for GEAK.

Environment toggles:
  GEAK_MEMORY_DISABLE=1              -- disable all memory
  GEAK_MEMORY_NO_WORKING=1           -- disable within-session working memory
  GEAK_MEMORY_NO_CROSS_SESSION=1     -- disable cross-session memory
  GEAK_CROSS_SESSION_MEMORY_URL=...  -- point to shared memory server
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
    """Retrieve relevant cross-session optimization experiences for prompt injection."""
    if not is_memory_enabled():
        return ""
    try:
        from minisweagent.memory.cross_session import retrieve

        return retrieve(**kwargs)
    except Exception:
        return ""


def record_optimization_outcome(**kwargs) -> None:
    """Persist an optimization outcome to cross-session memory."""
    if not is_memory_enabled():
        return
    try:
        from minisweagent.memory.cross_session import record

        record(**kwargs)
    except Exception:
        pass
