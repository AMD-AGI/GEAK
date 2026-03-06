"""FalkorDB backend for cross-session memory.

Graph database that tracks strategy relationships and enables
graph-based retrieval of optimization outcomes. C-optimized,
purpose-built for AI/GraphRAG workloads.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from minisweagent.memory.storage_adapter import MemoryStore

logger = logging.getLogger(__name__)


class FalkorDBMemoryStore(MemoryStore):

    def __init__(self, **kwargs):
        try:
            from falkordb import FalkorDB
        except ImportError:
            raise ImportError("falkordb not installed. Run: pip install falkordb")

        host = os.environ.get("GEAK_FALKORDB_HOST", "localhost")
        port = int(os.environ.get("GEAK_FALKORDB_PORT", "6379"))
        self._client = FalkorDB(host=host, port=port)
        self._graph = self._client.select_graph("geak_memory")
        self._ensure_schema()
        self._counter = 0

    def _ensure_schema(self):
        try:
            self._graph.query(
                "CREATE INDEX FOR (o:Outcome) ON (o.kernel_category)"
            )
        except Exception:
            pass
        try:
            self._graph.query(
                "CREATE INDEX FOR (s:Strategy) ON (s.name)"
            )
        except Exception:
            pass

    def store(self, outcome: dict[str, Any]) -> str:
        self._counter += 1
        oid = f"o_{int(time.time())}_{self._counter}"
        cat = outcome.get("kernel_category", "unknown")
        lang = outcome.get("kernel_language", "")
        bottleneck = outcome.get("bottleneck_type", "unknown")
        strategy = outcome.get("strategy_name", "")
        technique = outcome.get("optimization_technique", "")
        speedup = float(outcome.get("speedup_achieved", 1.0))
        success = bool(outcome.get("success", False))

        self._graph.query(
            """MERGE (c:Category {name: $cat})
               MERGE (s:Strategy {name: $strat})
               CREATE (o:Outcome {
                   id: $oid, kernel_category: $cat, kernel_language: $lang,
                   bottleneck_type: $bottleneck, strategy_name: $strat,
                   optimization_technique: $technique,
                   speedup_achieved: $speedup, success: $success,
                   timestamp: $ts
               })
               MERGE (o)-[:BELONGS_TO]->(c)
               MERGE (o)-[:USED_STRATEGY]->(s)""",
            params={
                "oid": oid, "cat": cat, "lang": lang, "bottleneck": bottleneck,
                "strat": strategy, "technique": technique,
                "speedup": speedup, "success": success,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        if success and speedup > 1.2:
            self._graph.query(
                """MATCH (s:Strategy {name: $strat})
                   SET s.win_count = COALESCE(s.win_count, 0) + 1,
                       s.best_speedup = CASE
                           WHEN COALESCE(s.best_speedup, 0) < $speedup
                           THEN $speedup ELSE s.best_speedup END""",
                params={"strat": strategy, "speedup": speedup},
            )

        return oid

    def retrieve(
        self,
        kernel_category: str | None = None,
        kernel_language: str | None = None,
        bottleneck_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        conditions = []
        params: dict[str, Any] = {"limit": limit}
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
        result = self._graph.query(
            f"""MATCH (o:Outcome) {where}
                RETURN o ORDER BY o.speedup_achieved DESC LIMIT $limit""",
            params=params,
        )
        return [dict(row[0].properties) for row in result.result_set]

    def search_similar(
        self,
        query_text: str,
        profiling_metrics: dict | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        result = self._graph.query(
            """MATCH (o:Outcome)-[:USED_STRATEGY]->(s:Strategy)
               WHERE o.success = true
               RETURN o, s.name as strategy
               ORDER BY o.speedup_achieved DESC LIMIT $limit""",
            params={"limit": limit},
        )
        return [
            {**dict(row[0].properties), "strategy": row[1]}
            for row in result.result_set
        ]

    def update(self, outcome_id: str, updates: dict[str, Any]) -> bool:
        set_parts = [f"o.{k} = ${k}" for k in updates]
        if not set_parts:
            return False
        try:
            self._graph.query(
                f"MATCH (o:Outcome {{id: $oid}}) SET {', '.join(set_parts)}",
                params={"oid": outcome_id, **updates},
            )
            return True
        except Exception:
            return False

    def delete(self, outcome_id: str) -> bool:
        try:
            self._graph.query(
                "MATCH (o:Outcome {id: $oid}) DETACH DELETE o",
                params={"oid": outcome_id},
            )
            return True
        except Exception:
            return False

    def get_strategy_stats(
        self,
        kernel_category: str | None = None,
    ) -> list[dict[str, Any]]:
        if kernel_category:
            result = self._graph.query(
                """MATCH (o:Outcome)-[:USED_STRATEGY]->(s:Strategy)
                   WHERE o.kernel_category = $cat
                   RETURN s.name as strategy_name,
                          COUNT(o) as total_attempts,
                          SUM(CASE WHEN o.success THEN 1 ELSE 0 END) as successes,
                          AVG(o.speedup_achieved) as avg_speedup,
                          MAX(o.speedup_achieved) as max_speedup
                   ORDER BY avg_speedup DESC""",
                params={"cat": kernel_category},
            )
        else:
            result = self._graph.query(
                """MATCH (o:Outcome)-[:USED_STRATEGY]->(s:Strategy)
                   RETURN s.name as strategy_name,
                          COUNT(o) as total_attempts,
                          SUM(CASE WHEN o.success THEN 1 ELSE 0 END) as successes,
                          AVG(o.speedup_achieved) as avg_speedup,
                          MAX(o.speedup_achieved) as max_speedup
                   ORDER BY avg_speedup DESC""",
            )
        return [
            {
                "strategy_name": row[0],
                "total_attempts": row[1],
                "successes": row[2],
                "avg_speedup": row[3],
                "max_speedup": row[4],
            }
            for row in result.result_set
        ]

    def close(self):
        pass
