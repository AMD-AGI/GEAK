"""Memgraph backend for cross-session memory.

Memgraph's cognitive AI memory layer with three-tier architecture:
Events -> Episodes -> Beliefs. Features automatic consolidation
("cognitive dreaming"), semantic deduplication, and sub-50ms retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from minisweagent.memory.storage_adapter import MemoryStore

logger = logging.getLogger(__name__)


class MemgraphMemoryStore(MemoryStore):
    """Memgraph-backed memory using its Python client.

    If memgraph is available as an AI memory service, we use the high-level
    API (add/search). Otherwise we fall back to Cypher queries via GQLAlchemy
    or the bolt driver.
    """

    def __init__(self, **kwargs):
        self._use_ai_api = False
        self._client = None
        self._driver = None

        try:
            import memgraph
            self._client = memgraph.Memgraph()
            self._use_ai_api = True
            logger.info("Using Memgraph AI memory API")
        except (ImportError, Exception):
            pass

        if not self._use_ai_api:
            try:
                from neo4j import GraphDatabase
                host = os.environ.get("GEAK_MEMGRAPH_HOST", "localhost")
                port = os.environ.get("GEAK_MEMGRAPH_PORT", "7687")
                self._driver = GraphDatabase.driver(f"bolt://{host}:{port}")
                self._ensure_schema()
                logger.info("Using Memgraph via Bolt driver")
            except ImportError:
                raise ImportError(
                    "Neither memgraph nor neo4j driver installed. "
                    "Run: pip install memgraph  OR  pip install neo4j"
                )

        self._counter = 0

    def _ensure_schema(self):
        if not self._driver:
            return
        with self._driver.session() as s:
            try:
                s.run("CREATE INDEX ON :Outcome(kernel_category)")
            except Exception:
                pass
            try:
                s.run("CREATE INDEX ON :Strategy(name)")
            except Exception:
                pass

    def _outcome_to_text(self, outcome: dict) -> str:
        cat = outcome.get("kernel_category", "unknown")
        lang = outcome.get("kernel_language", "unknown")
        speedup = outcome.get("speedup_achieved", 1.0)
        technique = outcome.get("optimization_technique", "")
        strategy = outcome.get("strategy_name", "")
        success = outcome.get("success", False)
        parts = [f"{'Success' if success else 'Failure'}: {cat} {lang} kernel"]
        parts.append(f"speedup={speedup:.2f}x")
        if strategy:
            parts.append(f"strategy={strategy}")
        if technique:
            parts.append(f"technique={technique[:100]}")
        return ", ".join(parts)

    def store(self, outcome: dict[str, Any]) -> str:
        self._counter += 1
        oid = f"mg_{int(time.time())}_{self._counter}"

        if self._use_ai_api and self._client:
            text = self._outcome_to_text(outcome)
            self._client.add(text, user_id="geak_agent")
            return oid

        if self._driver:
            cat = outcome.get("kernel_category", "unknown")
            strategy = outcome.get("strategy_name", "")
            with self._driver.session() as s:
                s.run(
                    """MERGE (c:Category {name: $cat})
                       MERGE (st:Strategy {name: $strat})
                       CREATE (o:Outcome {
                           id: $oid, kernel_category: $cat,
                           kernel_language: $lang, bottleneck_type: $bn,
                           strategy_name: $strat, optimization_technique: $tech,
                           speedup_achieved: $sp, success: $success,
                           timestamp: $ts
                       })
                       MERGE (o)-[:BELONGS_TO]->(c)
                       MERGE (o)-[:USED_STRATEGY]->(st)""",
                    oid=oid, cat=cat,
                    lang=outcome.get("kernel_language", ""),
                    bn=outcome.get("bottleneck_type", "unknown"),
                    strat=strategy,
                    tech=outcome.get("optimization_technique", ""),
                    sp=float(outcome.get("speedup_achieved", 1.0)),
                    success=bool(outcome.get("success", False)),
                    ts=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            return oid

        return oid

    def retrieve(
        self,
        kernel_category: str | None = None,
        kernel_language: str | None = None,
        bottleneck_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if self._use_ai_api and self._client:
            query_parts = [kernel_category or "", kernel_language or "", bottleneck_type or ""]
            query = " ".join(p for p in query_parts if p) or "kernel optimization"
            results = self._client.search(query, user_id="geak_agent")
            return [{"memory": r.get("memory", ""), **r.get("metadata", {})} for r in (results or [])][:limit]

        if self._driver:
            conditions = []
            params: dict[str, Any] = {"lim": limit}
            if kernel_category:
                conditions.append("o.kernel_category = $cat")
                params["cat"] = kernel_category
            if kernel_language:
                conditions.append("o.kernel_language = $lang")
                params["lang"] = kernel_language
            if bottleneck_type:
                conditions.append("o.bottleneck_type = $bn")
                params["bn"] = bottleneck_type
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            with self._driver.session() as s:
                result = s.run(
                    f"MATCH (o:Outcome) {where} RETURN o ORDER BY o.speedup_achieved DESC LIMIT $lim",
                    **params,
                )
                return [dict(record["o"]) for record in result]
        return []

    def search_similar(
        self,
        query_text: str,
        profiling_metrics: dict | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if self._use_ai_api and self._client:
            results = self._client.search(query_text, user_id="geak_agent")
            return [{"memory": r.get("memory", ""), **r.get("metadata", {})} for r in (results or [])][:limit]
        return self.retrieve(limit=limit)

    def update(self, outcome_id: str, updates: dict[str, Any]) -> bool:
        if not self._driver:
            return False
        set_parts = [f"o.{k} = ${k}" for k in updates]
        if not set_parts:
            return False
        try:
            with self._driver.session() as s:
                s.run(
                    f"MATCH (o:Outcome {{id: $oid}}) SET {', '.join(set_parts)}",
                    oid=outcome_id, **updates,
                )
            return True
        except Exception:
            return False

    def delete(self, outcome_id: str) -> bool:
        if not self._driver:
            return False
        try:
            with self._driver.session() as s:
                s.run("MATCH (o:Outcome {id: $oid}) DETACH DELETE o", oid=outcome_id)
            return True
        except Exception:
            return False

    def get_strategy_stats(
        self,
        kernel_category: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._driver:
            return []
        if kernel_category:
            query = """MATCH (o:Outcome)-[:USED_STRATEGY]->(s:Strategy)
                       WHERE o.kernel_category = $cat
                       RETURN s.name, COUNT(o), SUM(CASE WHEN o.success THEN 1 ELSE 0 END),
                              AVG(o.speedup_achieved), MAX(o.speedup_achieved)
                       ORDER BY AVG(o.speedup_achieved) DESC"""
            params = {"cat": kernel_category}
        else:
            query = """MATCH (o:Outcome)-[:USED_STRATEGY]->(s:Strategy)
                       RETURN s.name, COUNT(o), SUM(CASE WHEN o.success THEN 1 ELSE 0 END),
                              AVG(o.speedup_achieved), MAX(o.speedup_achieved)
                       ORDER BY AVG(o.speedup_achieved) DESC"""
            params = {}
        try:
            with self._driver.session() as s:
                result = s.run(query, **params)
                return [
                    {
                        "strategy_name": r[0],
                        "total_attempts": r[1],
                        "successes": r[2],
                        "avg_speedup": r[3],
                        "max_speedup": r[4],
                    }
                    for r in result
                ]
        except Exception:
            return []

    def close(self):
        if self._driver:
            self._driver.close()
