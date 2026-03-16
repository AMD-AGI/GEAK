"""SQLite backend for cross-session memory (control group).

Wraps the existing CrossSessionMemory class to conform to the MemoryStore ABC.
"""

from __future__ import annotations

from typing import Any

from minisweagent.memory.storage_adapter import MemoryStore
from minisweagent.memory.cross_session_memory import CrossSessionMemory


class SQLiteMemoryStore(MemoryStore):

    def __init__(self, db_path: str | None = None, **kwargs):
        self._db = CrossSessionMemory(db_path=db_path)

    def store(self, outcome: dict[str, Any]) -> str:
        row_id = self._db.record_outcome(
            kernel_type=outcome.get("kernel_type", "unknown"),
            kernel_category=outcome.get("kernel_category", "unknown"),
            bottleneck_type=outcome.get("bottleneck_type", "unknown"),
            gpu_architecture=outcome.get("gpu_architecture", "unknown"),
            strategy_name=outcome.get("strategy_name", ""),
            speedup_achieved=outcome.get("speedup_achieved", 1.0),
            success=outcome.get("success", False),
            failure_reason=outcome.get("failure_reason"),
            cost_dollars=outcome.get("cost_dollars", 0.0),
            steps_taken=outcome.get("steps_taken", 0),
            commandment_worked=outcome.get("commandment_worked", False),
            profiling_metrics=outcome.get("profiling_metrics"),
            optimization_technique=outcome.get("optimization_technique", ""),
            kernel_language=outcome.get("kernel_language", ""),
        )
        return str(row_id)

    def retrieve(
        self,
        kernel_category: str | None = None,
        kernel_language: str | None = None,
        bottleneck_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        results = self._db.query_outcomes(
            kernel_category=kernel_category,
            bottleneck_type=bottleneck_type,
            limit=limit,
        )
        if kernel_language:
            results = [r for r in results if r.get("kernel_language") == kernel_language]
        return results

    def search_similar(
        self,
        query_text: str,
        profiling_metrics: dict | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if profiling_metrics:
            return self._db.query_by_bottleneck_similarity(
                bottleneck_type="",
                profiling_metrics=profiling_metrics,
                limit=limit,
            )
        return self._db.query_outcomes(limit=limit)

    def update(self, outcome_id: str, updates: dict[str, Any]) -> bool:
        try:
            set_clauses = ", ".join(f"{k} = ?" for k in updates)
            self._db._conn.execute(
                f"UPDATE optimization_outcomes SET {set_clauses} WHERE id = ?",
                list(updates.values()) + [int(outcome_id)],
            )
            self._db._conn.commit()
            return True
        except Exception:
            return False

    def delete(self, outcome_id: str) -> bool:
        try:
            self._db._conn.execute(
                "DELETE FROM optimization_outcomes WHERE id = ?",
                (int(outcome_id),),
            )
            self._db._conn.commit()
            return True
        except Exception:
            return False

    def get_strategy_stats(
        self,
        kernel_category: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._db.query_strategy_effectiveness(kernel_category=kernel_category)

    def close(self):
        self._db.close()
