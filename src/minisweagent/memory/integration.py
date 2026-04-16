"""Memory system integration layer for GEAK.

Environment flags:
  GEAK_MEMORY_DISABLE=1             -- turn off all memory (within-session + cross-session)
  GEAK_USE_KNOWLEDGE_BASE=0         -- turn off reading from the knowledge base (on by default)
  GEAK_SAVE_TO_KNOWLEDGE_BASE=1     -- turn on saving run insights to KB after each run (off by default)
  GEAK_MEMORY_MIN_SPEEDUP=1.10      -- minimum speedup required to save an experience (default 1.10x)
  GEAK_CROSS_SESSION_MEMORY_URL=... -- point to a shared memory server (default: local SQLite)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _is_on(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip().lower() in ("1", "true", "yes")


def _is_off(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip().lower() in ("0", "false", "no")


def is_memory_enabled() -> bool:
    """All memory is on unless GEAK_MEMORY_DISABLE=1."""
    return not _is_on("GEAK_MEMORY_DISABLE")


def is_retrieve_enabled() -> bool:
    """Reading from the knowledge base is on by default.

    Set GEAK_USE_KNOWLEDGE_BASE=0 to turn off.
    """
    if not is_memory_enabled():
        return False
    if _is_off("GEAK_USE_KNOWLEDGE_BASE"):
        return False
    if _is_on("GEAK_MEMORY_NO_CROSS_SESSION"):
        return False
    return True


def is_record_enabled() -> bool:
    """Saving to the knowledge base is off by default.

    Set GEAK_SAVE_TO_KNOWLEDGE_BASE=1 to turn on.
    """
    if not is_memory_enabled():
        return False
    return _is_on("GEAK_SAVE_TO_KNOWLEDGE_BASE")


def is_working_memory_enabled() -> bool:
    """Within-session working memory is on by default."""
    return is_memory_enabled() and not _is_on("GEAK_MEMORY_NO_WORKING")


def assemble_memory_context(**kwargs) -> str:
    """Retrieve relevant cross-session optimization experiences for prompt injection."""
    if not is_retrieve_enabled():
        return ""
    try:
        from minisweagent.memory.cross_session import retrieve

        return retrieve(**kwargs)
    except Exception:
        return ""


def record_optimization_outcome(**kwargs) -> None:
    """Persist an optimization outcome to cross-session memory.

    Validates that critical fields are not empty/null before storing.
    """
    if not is_record_enabled():
        return
    try:
        from minisweagent.memory.cross_session import record

        _validate_record_kwargs(kwargs)
        record(**kwargs)
    except ValueError as exc:
        logger.warning("Skipping record: validation failed: %s", exc)
    except Exception as exc:
        logger.warning("Cross-session record failed: %s", exc)


def _validate_record_kwargs(kwargs: dict) -> None:
    """Raise ValueError if critical fields are missing or empty."""
    kernel_path = kwargs.get("kernel_path", "")
    if not kernel_path:
        raise ValueError("kernel_path is empty")

    speedup = kwargs.get("speedup_achieved")
    if speedup is None or float(speedup) <= 0:
        raise ValueError("speedup_achieved is missing or <= 0")

    strategy = kwargs.get("strategy_name", "")
    if not strategy:
        raise ValueError("strategy_name is empty")
