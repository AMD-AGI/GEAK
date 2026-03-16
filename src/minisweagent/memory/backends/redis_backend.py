"""Redis backend for cross-session memory.

Uses Redis as a KV store with JSON support. Vector search requires
Redis Stack (redis-stack-server) with the RediSearch module.
Falls back to keyword matching if vector search is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from minisweagent.memory.storage_adapter import MemoryStore

logger = logging.getLogger(__name__)


class RedisMemoryStore(MemoryStore):

    def __init__(self, **kwargs):
        try:
            import redis
        except ImportError:
            raise ImportError("redis not installed. Run: pip install redis")

        host = os.environ.get("GEAK_REDIS_HOST", "localhost")
        port = int(os.environ.get("GEAK_REDIS_PORT", "6379"))
        db = int(os.environ.get("GEAK_REDIS_DB", "0"))
        self._r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._prefix = "geak:outcome:"
        self._counter_key = "geak:outcome_counter"

    def _next_id(self) -> str:
        return str(self._r.incr(self._counter_key))

    def store(self, outcome: dict[str, Any]) -> str:
        oid = self._next_id()
        key = f"{self._prefix}{oid}"
        record = {
            "id": oid,
            "kernel_type": outcome.get("kernel_type", "unknown"),
            "kernel_category": outcome.get("kernel_category", "unknown"),
            "kernel_language": outcome.get("kernel_language", ""),
            "bottleneck_type": outcome.get("bottleneck_type", "unknown"),
            "gpu_architecture": outcome.get("gpu_architecture", "unknown"),
            "strategy_name": outcome.get("strategy_name", ""),
            "optimization_technique": outcome.get("optimization_technique", ""),
            "speedup_achieved": float(outcome.get("speedup_achieved", 1.0)),
            "success": int(outcome.get("success", False)),
            "failure_reason": outcome.get("failure_reason") or "",
            "profiling_metrics": json.dumps(outcome.get("profiling_metrics") or {}),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._r.hset(key, mapping=record)
        self._r.sadd("geak:outcome_ids", oid)
        cat = record["kernel_category"]
        self._r.sadd(f"geak:cat:{cat}", oid)
        return oid

    def _get_outcome(self, oid: str) -> dict[str, Any] | None:
        data = self._r.hgetall(f"{self._prefix}{oid}")
        if not data:
            return None
        data["speedup_achieved"] = float(data.get("speedup_achieved", 1.0))
        data["success"] = bool(int(data.get("success", 0)))
        return data

    def retrieve(
        self,
        kernel_category: str | None = None,
        kernel_language: str | None = None,
        bottleneck_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if kernel_category:
            ids = self._r.smembers(f"geak:cat:{kernel_category}")
        else:
            ids = self._r.smembers("geak:outcome_ids")
        results = []
        for oid in sorted(ids, reverse=True):
            o = self._get_outcome(oid)
            if not o:
                continue
            if kernel_language and o.get("kernel_language") != kernel_language:
                continue
            if bottleneck_type and o.get("bottleneck_type") != bottleneck_type:
                continue
            results.append(o)
            if len(results) >= limit:
                break
        return results

    def search_similar(
        self,
        query_text: str,
        profiling_metrics: dict | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        all_ids = self._r.smembers("geak:outcome_ids")
        query_lower = query_text.lower()
        scored: list[tuple[float, dict]] = []
        for oid in all_ids:
            o = self._get_outcome(oid)
            if not o:
                continue
            text = " ".join([
                o.get("kernel_category", ""),
                o.get("strategy_name", ""),
                o.get("optimization_technique", ""),
                o.get("bottleneck_type", ""),
            ]).lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append((score, o))
        scored.sort(key=lambda x: (-x[0], -x[1].get("speedup_achieved", 0)))
        return [o for _, o in scored[:limit]]

    def update(self, outcome_id: str, updates: dict[str, Any]) -> bool:
        key = f"{self._prefix}{outcome_id}"
        if not self._r.exists(key):
            return False
        self._r.hset(key, mapping=updates)
        return True

    def delete(self, outcome_id: str) -> bool:
        key = f"{self._prefix}{outcome_id}"
        if not self._r.exists(key):
            return False
        data = self._r.hgetall(key)
        self._r.delete(key)
        self._r.srem("geak:outcome_ids", outcome_id)
        cat = data.get("kernel_category", "")
        if cat:
            self._r.srem(f"geak:cat:{cat}", outcome_id)
        return True

    def get_strategy_stats(
        self,
        kernel_category: str | None = None,
    ) -> list[dict[str, Any]]:
        outcomes = self.retrieve(kernel_category=kernel_category, limit=1000)
        stats: dict[str, dict] = {}
        for o in outcomes:
            strat = o.get("strategy_name", "unknown")
            if strat not in stats:
                stats[strat] = {"total": 0, "successes": 0, "speedups": []}
            stats[strat]["total"] += 1
            if o.get("speedup_achieved", 1.0) > 1.0:
                stats[strat]["successes"] += 1
            stats[strat]["speedups"].append(o.get("speedup_achieved", 1.0))
        return [
            {
                "strategy_name": k,
                "total_attempts": v["total"],
                "successes": v["successes"],
                "avg_speedup": sum(v["speedups"]) / len(v["speedups"]) if v["speedups"] else 0,
                "max_speedup": max(v["speedups"]) if v["speedups"] else 0,
            }
            for k, v in stats.items()
        ]

    def close(self):
        self._r.close()
