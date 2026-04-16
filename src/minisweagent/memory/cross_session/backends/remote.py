"""REST client backend -- delegates all operations to a shared memory server.

The server exposes the same MemoryBackend interface over HTTP.
Devs configure via: GEAK_CROSS_SESSION_MEMORY_URL=http://geak-memory.internal:8642
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill

logger = logging.getLogger(__name__)


class RemoteBackend:
    """HTTP client that implements MemoryBackend by calling the REST server."""

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 5.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _url(self, path: str) -> str:
        return f"{self._base}/api/v1{path}"

    def _post(self, path: str, data: dict) -> dict:
        resp = requests.post(self._url(path), json=data, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        resp = requests.get(self._url(path), params=params, headers=self._headers, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    # ── Experience CRUD ──────────────────────────────────────────────

    def store_experience(self, record: ExperienceRecord) -> None:
        self._post("/experiences", record.to_dict())

    def search_experiences(
        self,
        *,
        category: str | None = None,
        bottleneck: str | None = None,
        language: str | None = None,
        hardware: str | None = None,
        limit: int = 50,
    ) -> list[ExperienceRecord]:
        params: dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        if bottleneck:
            params["bottleneck"] = bottleneck
        if language:
            params["language"] = language
        if hardware:
            params["hardware"] = hardware

        rows = self._get("/experiences/search", params)
        if not isinstance(rows, list):
            rows = rows.get("results", [])
        return [ExperienceRecord.from_dict(r) for r in rows]

    def list_experiences(self, *, limit: int = 100, offset: int = 0) -> list[ExperienceRecord]:
        rows = self._get("/experiences", {"limit": limit, "offset": offset})
        if not isinstance(rows, list):
            rows = rows.get("results", [])
        return [ExperienceRecord.from_dict(r) for r in rows]

    def experience_count(self) -> int:
        stats = self.get_stats()
        return stats.get("experience_count", 0)

    # ── Skill CRUD ───────────────────────────────────────────────────

    def store_skill(self, skill: StrategySkill) -> None:
        self._post("/skills", skill.to_dict())

    def search_skills(
        self,
        *,
        category: str | None = None,
        bottleneck: str | None = None,
        language: str | None = None,
        limit: int = 10,
    ) -> list[StrategySkill]:
        params: dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        if bottleneck:
            params["bottleneck"] = bottleneck
        if language:
            params["language"] = language

        rows = self._get("/skills/search", params)
        if not isinstance(rows, list):
            rows = rows.get("results", [])
        return [StrategySkill.from_dict(r) for r in rows]

    def list_skills(self) -> list[StrategySkill]:
        rows = self._get("/skills")
        if not isinstance(rows, list):
            rows = rows.get("results", [])
        return [StrategySkill.from_dict(r) for r in rows]

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        result = self._get("/stats")
        return result if isinstance(result, dict) else {}
