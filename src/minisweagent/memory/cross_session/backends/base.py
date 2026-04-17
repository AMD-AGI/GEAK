"""Abstract interface for cross-session memory backends.

Any backend (SQLite, REST, PostgreSQL, ...) implements this protocol.
The retriever and extractor only depend on this interface.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill


@runtime_checkable
class MemoryBackend(Protocol):
    """Pluggable storage backend for cross-session memory."""

    def store_experience(self, record: ExperienceRecord) -> None:
        """Persist a new experience record (append-only)."""
        ...

    def store_skill(self, skill: StrategySkill) -> None:
        """Persist or update a strategy skill."""
        ...

    def search_experiences(
        self,
        *,
        category: str | None = None,
        bottleneck: str | None = None,
        language: str | None = None,
        hardware: str | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        """Search experiences by structured filters."""
        ...

    def search_skills(
        self,
        *,
        category: str | None = None,
        bottleneck: str | None = None,
        language: str | None = None,
        limit: int = 10,
    ) -> list[StrategySkill]:
        """Search consolidated skills by filters."""
        ...

    def list_experiences(self, *, limit: int = 100, offset: int = 0) -> list[ExperienceRecord]:
        """List experiences ordered by recency."""
        ...

    def list_skills(self) -> list[StrategySkill]:
        """List all consolidated skills."""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics (counts, category distribution, etc.)."""
        ...

    def experience_count(self) -> int:
        """Total number of stored experiences."""
        ...
