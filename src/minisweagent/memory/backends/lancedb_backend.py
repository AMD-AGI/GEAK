"""LanceDB backend for cross-session memory.

Embedded vector DB, file-based (no server process required).
Each experiment can have its own isolated DB directory.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from minisweagent.memory.storage_adapter import MemoryStore

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = os.path.expanduser("~/.config/mini-swe-agent/lancedb_memory")


class LanceDBMemoryStore(MemoryStore):

    def __init__(self, db_path: str | None = None, **kwargs):
        try:
            import lancedb
        except ImportError:
            raise ImportError("lancedb not installed. Run: pip install lancedb")

        self._db_path = db_path or os.environ.get("GEAK_LANCEDB_PATH", _DEFAULT_DB_DIR)
        Path(self._db_path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self._db_path)
        self._table_name = "optimization_outcomes"
        self._ensure_table()

    def _ensure_table(self):
        if self._table_name not in self._db.table_names():
            import pyarrow as pa
            schema = pa.schema([
                ("id", pa.string()),
                ("kernel_type", pa.string()),
                ("kernel_category", pa.string()),
                ("kernel_language", pa.string()),
                ("bottleneck_type", pa.string()),
                ("gpu_architecture", pa.string()),
                ("strategy_name", pa.string()),
                ("optimization_technique", pa.string()),
                ("speedup_achieved", pa.float64()),
                ("success", pa.bool_()),
                ("failure_reason", pa.string()),
                ("profiling_metrics_json", pa.string()),
                ("timestamp", pa.string()),
                ("text", pa.string()),
            ])
            self._db.create_table(self._table_name, schema=schema)
        self._table = self._db.open_table(self._table_name)

    def _make_text(self, outcome: dict) -> str:
        """Build a text field for vector search from outcome fields."""
        parts = [
            outcome.get("kernel_category", ""),
            outcome.get("kernel_language", ""),
            outcome.get("bottleneck_type", ""),
            outcome.get("strategy_name", ""),
            outcome.get("optimization_technique", ""),
        ]
        return " ".join(p for p in parts if p)

    def store(self, outcome: dict[str, Any]) -> str:
        oid = f"{int(time.time() * 1000)}_{outcome.get('kernel_category', 'unk')}"
        row = {
            "id": oid,
            "kernel_type": outcome.get("kernel_type", "unknown"),
            "kernel_category": outcome.get("kernel_category", "unknown"),
            "kernel_language": outcome.get("kernel_language", ""),
            "bottleneck_type": outcome.get("bottleneck_type", "unknown"),
            "gpu_architecture": outcome.get("gpu_architecture", "unknown"),
            "strategy_name": outcome.get("strategy_name", ""),
            "optimization_technique": outcome.get("optimization_technique", ""),
            "speedup_achieved": float(outcome.get("speedup_achieved", 1.0)),
            "success": bool(outcome.get("success", False)),
            "failure_reason": outcome.get("failure_reason") or "",
            "profiling_metrics_json": json.dumps(outcome.get("profiling_metrics") or {}),
            "timestamp": outcome.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "text": self._make_text(outcome),
        }
        self._table.add([row])
        return oid

    def retrieve(
        self,
        kernel_category: str | None = None,
        kernel_language: str | None = None,
        bottleneck_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        try:
            df = self._table.to_pandas()
        except Exception:
            return []
        if kernel_category:
            df = df[df["kernel_category"] == kernel_category]
        if kernel_language:
            df = df[df["kernel_language"] == kernel_language]
        if bottleneck_type:
            df = df[df["bottleneck_type"] == bottleneck_type]
        df = df.sort_values("speedup_achieved", ascending=False).head(limit)
        return df.to_dict("records")

    def search_similar(
        self,
        query_text: str,
        profiling_metrics: dict | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        try:
            results = self._table.search(query_text, query_type="fts").limit(limit).to_pandas()
            return results.to_dict("records")
        except Exception:
            return self.retrieve(limit=limit)

    def update(self, outcome_id: str, updates: dict[str, Any]) -> bool:
        try:
            where_clause = f"id = '{outcome_id}'"
            self._table.update(where=where_clause, values=updates)
            return True
        except Exception:
            return False

    def delete(self, outcome_id: str) -> bool:
        try:
            self._table.delete(f"id = '{outcome_id}'")
            return True
        except Exception:
            return False

    def get_strategy_stats(
        self,
        kernel_category: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            df = self._table.to_pandas()
            if kernel_category:
                df = df[df["kernel_category"] == kernel_category]
            if df.empty:
                return []
            stats = df.groupby("strategy_name").agg(
                total_attempts=("id", "count"),
                successes=("success", "sum"),
                avg_speedup=("speedup_achieved", "mean"),
                max_speedup=("speedup_achieved", "max"),
            ).reset_index()
            return stats.to_dict("records")
        except Exception:
            return []

    def close(self):
        pass
