"""Memory system integration layer for GEAK.

Environment toggles:
  GEAK_MEMORY_DISABLE=1              -- disable all memory (read + write)
  GEAK_MEMORY_NO_WORKING=1           -- disable within-session working memory
  GEAK_MEMORY_NO_CROSS_SESSION=1     -- disable cross-session memory entirely
  GEAK_MEMORY_NO_RETRIEVE=1          -- disable reading from knowledge base
  GEAK_MEMORY_NO_RECORD=1            -- disable writing to knowledge base
  GEAK_CROSS_SESSION_MEMORY_URL=...  -- point to shared memory server
  GEAK_MEMORY_MIN_SPEEDUP=1.10       -- minimum speedup to store (default 1.10x)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _is_disabled(env_var: str) -> bool:
    return os.environ.get(env_var, "").strip() in ("1", "true", "yes")


def is_memory_enabled() -> bool:
    return not _is_disabled("GEAK_MEMORY_DISABLE")


def is_retrieve_enabled() -> bool:
    """Check if reading from the knowledge base is enabled."""
    return is_memory_enabled() and not _is_disabled("GEAK_MEMORY_NO_RETRIEVE")


def is_record_enabled() -> bool:
    """Check if writing to the knowledge base is enabled."""
    return is_memory_enabled() and not _is_disabled("GEAK_MEMORY_NO_RECORD")


def is_working_memory_enabled() -> bool:
    return is_memory_enabled() and not _is_disabled("GEAK_MEMORY_NO_WORKING")


def assemble_memory_context(**kwargs) -> str:
    """Retrieve relevant cross-session optimization experiences for prompt injection.

    Controlled by GEAK_MEMORY_NO_RETRIEVE=1 (or GEAK_MEMORY_DISABLE=1).
    """
    if not is_retrieve_enabled():
        return ""
    try:
        from minisweagent.memory.cross_session import retrieve

        return retrieve(**kwargs)
    except Exception:
        return ""


def record_optimization_outcome(**kwargs) -> None:
    """Persist an optimization outcome to cross-session memory.

    Controlled by GEAK_MEMORY_NO_RECORD=1 (or GEAK_MEMORY_DISABLE=1).
    Validates that critical fields are not empty before storing.
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
