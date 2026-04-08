"""Configuration for cross-session memory.

Single-knob design: set GEAK_CROSS_SESSION_MEMORY_URL to point at a shared
server. Everything else auto-configures.

  - URL set   -> remote backend (REST client)
  - URL unset -> local backend (SQLite at ~/.cache/geak/memory.db)
  - GEAK_MEMORY_NO_CROSS_SESSION=1 -> disabled entirely
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_local_db_path() -> str:
    return str(Path.home() / ".cache" / "geak" / "memory.db")


@dataclass
class CrossSessionConfig:
    enabled: bool = True
    url: str = ""
    local_db_path: str = field(default_factory=_default_local_db_path)
    api_key: str = ""
    retrieval_limit: int = 30
    retrieval_top_k: int = 5
    context_max_tokens: int = 800
    consolidation_threshold: int = 10
    request_timeout: float = 5.0

    @property
    def is_remote(self) -> bool:
        return bool(self.url)

    @classmethod
    def from_env(cls) -> CrossSessionConfig:
        disabled = os.environ.get("GEAK_MEMORY_NO_CROSS_SESSION", "").strip().lower()
        if disabled in ("1", "true", "yes"):
            return cls(enabled=False)

        also_global = os.environ.get("GEAK_MEMORY_DISABLE", "").strip().lower()
        if also_global in ("1", "true", "yes"):
            return cls(enabled=False)

        url = os.environ.get("GEAK_CROSS_SESSION_MEMORY_URL", "").strip()
        api_key = os.environ.get("GEAK_MEMORY_API_KEY", "").strip()
        local_db = os.environ.get("GEAK_MEMORY_STORE_PATH", "").strip() or _default_local_db_path()
        timeout = float(os.environ.get("GEAK_MEMORY_TIMEOUT", "5.0"))
        limit = int(os.environ.get("GEAK_MEMORY_RETRIEVAL_LIMIT", "30"))

        return cls(
            enabled=True,
            url=url,
            local_db_path=local_db,
            api_key=api_key,
            retrieval_limit=limit,
            request_timeout=timeout,
        )
