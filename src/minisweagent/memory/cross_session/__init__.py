"""Cross-session memory for GEAK kernel optimization.

Public API -- only two functions are exposed:
  record(**kwargs)  -- persist an optimization outcome (write path)
  retrieve(**kwargs) -- retrieve relevant past experiences (read path)

Backend selection is automatic based on GEAK_CROSS_SESSION_MEMORY_URL:
  - URL set   -> remote backend (REST client to shared server)
  - URL unset -> local backend (SQLite)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from minisweagent.memory.cross_session.backends.base import MemoryBackend
from minisweagent.memory.cross_session.config import CrossSessionConfig

logger = logging.getLogger(__name__)

_backend: MemoryBackend | None = None
_backend_lock = threading.Lock()
_config: CrossSessionConfig | None = None


def _get_config() -> CrossSessionConfig:
    global _config
    if _config is None:
        _config = CrossSessionConfig.from_env()
    return _config


def get_backend() -> MemoryBackend | None:
    """Return the configured backend, or None if disabled."""
    global _backend
    cfg = _get_config()
    if not cfg.enabled:
        return None

    if _backend is not None:
        return _backend

    with _backend_lock:
        if _backend is not None:
            return _backend

        try:
            if cfg.is_remote:
                from minisweagent.memory.cross_session.backends.remote import RemoteBackend

                _backend = RemoteBackend(
                    base_url=cfg.url,
                    api_key=cfg.api_key,
                    timeout=cfg.request_timeout,
                )
            else:
                from minisweagent.memory.cross_session.backends.local import LocalSQLiteBackend

                _backend = LocalSQLiteBackend(db_path=cfg.local_db_path)
        except Exception as exc:
            logger.warning("Failed to initialize cross-session memory backend: %s", exc)
            return None

    return _backend


def record(**kwargs: Any) -> None:
    """Write path: extract and persist an ExperienceRecord from run artifacts."""
    backend = get_backend()
    if backend is None:
        return

    try:
        from minisweagent.memory.cross_session.extractor import extract_experience

        experience = extract_experience(**kwargs)

        min_speedup = float(_get_config().min_store_speedup)
        if experience.best_speedup < min_speedup:
            logger.debug(
                "Skipping experience (%.3fx < %.2fx threshold): %s",
                experience.best_speedup,
                min_speedup,
                experience.record_id,
            )
            return

        backend.store_experience(experience)
        logger.debug("Stored cross-session experience: %s", experience.record_id)
    except Exception as exc:
        logger.warning("Cross-session record failed: %s", exc)


def retrieve(**kwargs: Any) -> str:
    """Retrieve relevant past experiences and format as context.

    Accepts either ``target_code`` (the raw kernel source string, preferred)
    or ``kernel_path`` (a filesystem path we read once here). Supplying
    both is fine -- ``target_code`` wins. Supplying neither yields no
    retrieval. The retriever itself never touches the filesystem.
    """
    backend = get_backend()
    if backend is None:
        return ""

    target_code = kwargs.get("target_code", "") or ""
    if not target_code:
        kernel_path = kwargs.get("kernel_path", "") or ""
        if kernel_path:
            from pathlib import Path

            try:
                target_code = Path(kernel_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                target_code = ""

    try:
        from minisweagent.memory.cross_session.retriever import retrieve_context

        cfg = _get_config()
        return retrieve_context(
            backend=backend,
            target_code=target_code,
            bottleneck_type=kwargs.get("bottleneck_type", ""),
            profiling_metrics=kwargs.get("profiling_metrics") or {},
            limit=cfg.retrieval_limit,
            top_k=cfg.retrieval_top_k,
            compact=bool(kwargs.get("compact", False)),
        )
    except Exception as exc:
        logger.warning("Cross-session retrieve failed: %s", exc)
        return ""
