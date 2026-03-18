"""Storage adapter for cross-session memory backends.

Provides an ABC (MemoryStore) that all backends implement, and a factory
function that instantiates the correct backend based on GEAK_MEMORY_BACKEND.

Supported backends:
  sqlite   -- current SQLite implementation (default, control group)
  mem0     -- Mem0 platform (vector + graph, ADD/UPDATE/DELETE reconciliation)
  memgraph -- Memgraph cognitive memory (Events/Episodes/Beliefs)
  lancedb  -- LanceDB embedded vector DB (file-based, no server)
  falkordb -- FalkorDB graph DB (C-optimized, strategy relationships)
  redis    -- Redis KV + vector search
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """Abstract interface for cross-session memory storage backends."""

    @abstractmethod
    def store(self, outcome: dict[str, Any]) -> str:
        """Store an optimization outcome. Returns a unique ID."""

    @abstractmethod
    def retrieve(
        self,
        kernel_category: str | None = None,
        kernel_language: str | None = None,
        bottleneck_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve outcomes by exact field matching."""

    @abstractmethod
    def search_similar(
        self,
        query_text: str,
        profiling_metrics: dict | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Semantic/vector search for similar outcomes."""

    @abstractmethod
    def update(self, outcome_id: str, updates: dict[str, Any]) -> bool:
        """Update an existing outcome record."""

    @abstractmethod
    def delete(self, outcome_id: str) -> bool:
        """Delete an outcome record."""

    @abstractmethod
    def get_strategy_stats(
        self,
        kernel_category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get aggregate strategy effectiveness statistics."""

    def close(self):
        """Clean up resources."""

    def get_all_outcomes(self) -> list[dict[str, Any]]:
        """Return all stored outcomes (for export/migration)."""
        return self.retrieve(limit=10000)


def get_backend_name() -> str:
    return os.environ.get("GEAK_MEMORY_BACKEND", "sqlite").lower().strip()


def create_memory_store(**kwargs) -> MemoryStore:
    """Factory: create the appropriate MemoryStore backend.

    The backend is selected via GEAK_MEMORY_BACKEND env var.
    """
    backend = get_backend_name()
    logger.info("Creating memory store with backend: %s", backend)

    if backend == "sqlite":
        from minisweagent.memory.backends.sqlite_backend import SQLiteMemoryStore
        return SQLiteMemoryStore(**kwargs)
    elif backend == "mem0":
        from minisweagent.memory.backends.mem0_backend import Mem0MemoryStore
        return Mem0MemoryStore(**kwargs)
    elif backend == "memgraph":
        from minisweagent.memory.backends.memgraph_backend import MemgraphMemoryStore
        return MemgraphMemoryStore(**kwargs)
    elif backend == "lancedb":
        from minisweagent.memory.backends.lancedb_backend import LanceDBMemoryStore
        return LanceDBMemoryStore(**kwargs)
    elif backend == "falkordb":
        from minisweagent.memory.backends.falkordb_backend import FalkorDBMemoryStore
        return FalkorDBMemoryStore(**kwargs)
    elif backend == "redis":
        from minisweagent.memory.backends.redis_backend import RedisMemoryStore
        return RedisMemoryStore(**kwargs)
    else:
        logger.warning("Unknown backend '%s', falling back to sqlite", backend)
        from minisweagent.memory.backends.sqlite_backend import SQLiteMemoryStore
        return SQLiteMemoryStore(**kwargs)
