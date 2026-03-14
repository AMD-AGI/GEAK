"""Mem0 backend for cross-session memory.

Uses the Mem0 platform for ADD/UPDATE/DELETE/NOOP reconciliation
and dual vector+graph storage. Mem0 automatically handles deduplication,
contradiction detection, and memory evolution.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from minisweagent.memory.storage_adapter import MemoryStore

logger = logging.getLogger(__name__)


class Mem0MemoryStore(MemoryStore):

    def __init__(self, **kwargs):
        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError("mem0ai not installed. Run: pip install mem0ai")

        config = {
            "version": "v1.1",
        }

        llm_key = os.environ.get("AMD_LLM_API_KEY", "")
        llm_base = os.environ.get("AMD_LLM_BASE_URL", "")
        if llm_key and llm_base:
            config["llm"] = {
                "provider": "openai",
                "config": {
                    "api_key": llm_key,
                    "base_url": llm_base,
                    "model": os.environ.get("GEAK_MODEL", "amd-llama-135b"),
                },
            }

        self._mem = Memory.from_config(config)
        self._user_id = "geak_agent"

    def _outcome_to_text(self, outcome: dict) -> str:
        """Convert outcome dict to a natural language description for Mem0."""
        parts = []
        cat = outcome.get("kernel_category", "unknown")
        lang = outcome.get("kernel_language", "unknown")
        speedup = outcome.get("speedup_achieved", 1.0)
        technique = outcome.get("optimization_technique", "")
        bottleneck = outcome.get("bottleneck_type", "unknown")
        strategy = outcome.get("strategy_name", "")
        success = outcome.get("success", False)

        if success:
            parts.append(
                f"Successfully optimized {cat} kernel ({lang}) achieving {speedup:.2f}x speedup."
            )
        else:
            parts.append(
                f"Failed to optimize {cat} kernel ({lang}), speedup was {speedup:.2f}x."
            )

        if bottleneck and bottleneck != "unknown":
            parts.append(f"Bottleneck: {bottleneck}.")
        if strategy:
            parts.append(f"Strategy: {strategy}.")
        if technique:
            parts.append(f"Technique: {technique}.")

        return " ".join(parts)

    def store(self, outcome: dict[str, Any]) -> str:
        text = self._outcome_to_text(outcome)
        metadata = {
            "kernel_category": outcome.get("kernel_category", "unknown"),
            "kernel_language": outcome.get("kernel_language", ""),
            "bottleneck_type": outcome.get("bottleneck_type", "unknown"),
            "speedup_achieved": float(outcome.get("speedup_achieved", 1.0)),
            "strategy_name": outcome.get("strategy_name", ""),
            "gpu_architecture": outcome.get("gpu_architecture", "unknown"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        result = self._mem.add(text, user_id=self._user_id, metadata=metadata)
        mem_id = ""
        if isinstance(result, dict):
            results_list = result.get("results", [])
            if results_list:
                mem_id = results_list[0].get("id", "")
        return str(mem_id)

    def retrieve(
        self,
        kernel_category: str | None = None,
        kernel_language: str | None = None,
        bottleneck_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query_parts = []
        if kernel_category:
            query_parts.append(f"{kernel_category} kernel optimization")
        if kernel_language:
            query_parts.append(f"{kernel_language}")
        if bottleneck_type:
            query_parts.append(f"{bottleneck_type} bottleneck")
        query = " ".join(query_parts) if query_parts else "kernel optimization outcomes"

        results = self._mem.search(query, user_id=self._user_id, limit=limit)
        return [
            {
                "id": r.get("id", ""),
                "memory": r.get("memory", ""),
                "score": r.get("score", 0.0),
                **r.get("metadata", {}),
            }
            for r in (results.get("results", []) if isinstance(results, dict) else results)
        ]

    def search_similar(
        self,
        query_text: str,
        profiling_metrics: dict | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        results = self._mem.search(query_text, user_id=self._user_id, limit=limit)
        return [
            {
                "id": r.get("id", ""),
                "memory": r.get("memory", ""),
                "score": r.get("score", 0.0),
                **r.get("metadata", {}),
            }
            for r in (results.get("results", []) if isinstance(results, dict) else results)
        ]

    def update(self, outcome_id: str, updates: dict[str, Any]) -> bool:
        try:
            self._mem.update(outcome_id, data=json.dumps(updates))
            return True
        except Exception:
            return False

    def delete(self, outcome_id: str) -> bool:
        try:
            self._mem.delete(outcome_id)
            return True
        except Exception:
            return False

    def get_strategy_stats(
        self,
        kernel_category: str | None = None,
    ) -> list[dict[str, Any]]:
        results = self.retrieve(kernel_category=kernel_category, limit=100)
        stats: dict[str, dict] = {}
        for r in results:
            strat = r.get("strategy_name", "unknown")
            if strat not in stats:
                stats[strat] = {"total": 0, "successes": 0, "speedups": []}
            stats[strat]["total"] += 1
            if r.get("speedup_achieved", 1.0) > 1.0:
                stats[strat]["successes"] += 1
            stats[strat]["speedups"].append(r.get("speedup_achieved", 1.0))
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
        pass
