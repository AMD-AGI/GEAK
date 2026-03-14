"""JSONL + TF-IDF semantic store for cross-session memory.

Alternative to SQLite that supports semantic retrieval over technique
descriptions, enabling cross-kernel transfer based on *what was done*
rather than just kernel category matching.

4-path retrieval (from PPT architecture, <=500 tokens total):
  1. Category + Language exact match
  2. Bottleneck similarity (same bottleneck type)
  3. Profile-metric cosine similarity (reuse profile_similarity.py)
  4. Belief confidence filter (only outcomes with speedup > 1.0x)

Storage: append-only .jsonl file, one JSON object per line.
Retrieval: scikit-learn TfidfVectorizer over technique + category + bottleneck.

References:
  - ExpeL (AAAI 2024): FAISS-based insight retrieval
  - ContextEvolve (2025): bottleneck-driven strategy routing
  - DAM (2025): procedural vs episodic separation
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from minisweagent import global_config_dir

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = global_config_dir / "geak_semantic_memory.jsonl"


class SemanticStore:
    """Append-only JSONL store with TF-IDF retrieval."""

    def __init__(self, store_path: str | Path | None = None):
        self.store_path = Path(store_path) if store_path else DEFAULT_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict[str, Any]] = []
        self._tfidf = None
        self._tfidf_matrix = None
        self._dirty = True
        self._load()

    def _load(self):
        self._entries = []
        if self.store_path.exists():
            for line in self.store_path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        self._entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        self._dirty = True

    def store(self, outcome: dict[str, Any]) -> None:
        entry = {
            "kernel_path": outcome.get("kernel_path", ""),
            "kernel_type": outcome.get("kernel_type", "unknown"),
            "kernel_category": outcome.get("kernel_category", "unknown"),
            "bottleneck_type": outcome.get("bottleneck_type", "unknown"),
            "gpu_architecture": outcome.get("gpu_architecture", "unknown"),
            "strategy_name": outcome.get("strategy_name", ""),
            "optimization_technique": outcome.get("optimization_technique", ""),
            "speedup_achieved": float(outcome.get("speedup_achieved", 1.0)),
            "success": bool(outcome.get("success", False)),
            "failure_reason": outcome.get("failure_reason"),
            "profiling_metrics": outcome.get("profiling_metrics", {}),
            "steps_taken": int(outcome.get("steps_taken", 0)),
            "cost_dollars": float(outcome.get("cost_dollars", 0.0)),
            "timestamp": outcome.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        }
        with open(self.store_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._entries.append(entry)
        self._dirty = True

    def update(self, entry_id: str, updates: dict[str, Any]) -> None:
        idx = int(entry_id) if entry_id.isdigit() else -1
        if 0 <= idx < len(self._entries):
            self._entries[idx].update(updates)
            self._rewrite()

    def delete(self, entry_id: str) -> None:
        idx = int(entry_id) if entry_id.isdigit() else -1
        if 0 <= idx < len(self._entries):
            self._entries.pop(idx)
            self._rewrite()

    def _rewrite(self):
        with open(self.store_path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry) + "\n")
        self._dirty = True

    def retrieve(
        self,
        kernel_category: str | None = None,
        bottleneck_type: str | None = None,
        kernel_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        results = self._entries
        if kernel_category:
            results = [e for e in results if e.get("kernel_category") == kernel_category]
        if bottleneck_type:
            results = [e for e in results if e.get("bottleneck_type") == bottleneck_type]
        if kernel_type:
            results = [e for e in results if e.get("kernel_type") == kernel_type]
        return results[-limit:]

    def _build_tfidf_index(self):
        if not self._dirty and self._tfidf is not None:
            return
        if not self._entries:
            self._tfidf = None
            self._tfidf_matrix = None
            self._dirty = False
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            logger.debug("scikit-learn not available, TF-IDF retrieval disabled")
            self._tfidf = None
            self._dirty = False
            return

        corpus = []
        for e in self._entries:
            doc = " ".join(filter(None, [
                e.get("optimization_technique", ""),
                e.get("strategy_name", ""),
                e.get("kernel_category", ""),
                e.get("bottleneck_type", ""),
                e.get("kernel_type", ""),
            ]))
            corpus.append(doc if doc.strip() else "empty")

        self._tfidf = TfidfVectorizer(max_features=500, stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(corpus)
        self._dirty = False

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> list[tuple[dict[str, Any], float]]:
        """TF-IDF similarity search over stored outcomes."""
        self._build_tfidf_index()
        if self._tfidf is None or self._tfidf_matrix is None:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self._tfidf.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        results = []
        for idx, score in ranked[:top_k]:
            if score >= min_score:
                results.append((self._entries[idx], float(score)))
        return results

    def four_path_retrieve(
        self,
        kernel_category: str | None = None,
        kernel_type: str | None = None,
        bottleneck_type: str | None = None,
        profiling_metrics: dict | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """4-path retrieval pipeline (PPT architecture).

        1. Category + Language exact match
        2. Bottleneck similarity
        3. Profile-metric cosine similarity
        4. Confidence / success filter
        """
        seen_ids: set[int] = set()
        results: list[dict[str, Any]] = []

        # Path 1: Category + Language match
        for i, e in enumerate(self._entries):
            if kernel_category and e.get("kernel_category") == kernel_category:
                if kernel_type and e.get("kernel_type") != kernel_type:
                    continue
                if i not in seen_ids:
                    seen_ids.add(i)
                    results.append(e)

        # Path 2: Bottleneck similarity (cross-kernel transfer)
        if bottleneck_type:
            for i, e in enumerate(self._entries):
                if e.get("bottleneck_type") == bottleneck_type and i not in seen_ids:
                    seen_ids.add(i)
                    results.append(e)

        # Path 3: Profile-metric cosine similarity
        if profiling_metrics:
            try:
                from minisweagent.memory.profile_similarity import (
                    find_similar_profiles,
                )
                similar = find_similar_profiles(profiling_metrics, self._entries, top_k=5)
                for outcome, _sim in similar:
                    oid = id(outcome)
                    for i, e in enumerate(self._entries):
                        if id(e) == oid and i not in seen_ids:
                            seen_ids.add(i)
                            results.append(e)
                            break
            except Exception:
                pass

        # Path 4: Confidence / success filter
        results = [r for r in results if r.get("success") or r.get("speedup_achieved", 1.0) > 1.0]

        results.sort(key=lambda r: -float(r.get("speedup_achieved", 0)))
        return results[:limit]

    def format_memory_context(
        self,
        kernel_category: str | None = None,
        kernel_type: str | None = None,
        bottleneck_type: str | None = None,
        profiling_metrics: dict | None = None,
    ) -> str:
        """Format retrieved outcomes for injection into agent prompt."""
        results = self.four_path_retrieve(
            kernel_category=kernel_category,
            kernel_type=kernel_type,
            bottleneck_type=bottleneck_type,
            profiling_metrics=profiling_metrics,
        )
        if not results:
            return ""

        lines = [f"--- Optimization Memory ({len(results)} relevant outcomes) ---"]

        successful = [r for r in results if r.get("success")]
        if successful:
            avg_sp = sum(r["speedup_achieved"] for r in successful) / len(successful)
            lines.append(f"Successful: {len(successful)}/{len(results)} (avg {avg_sp:.2f}x)")

        by_strategy: dict[str, list[dict]] = {}
        for r in results:
            sn = r.get("strategy_name", "unknown")
            by_strategy.setdefault(sn, []).append(r)

        worked = {s: rs for s, rs in by_strategy.items()
                  if any(r.get("success") for r in rs)}
        failed = {s: rs for s, rs in by_strategy.items()
                  if not any(r.get("success") for r in rs)}

        if worked:
            lines.append("\nStrategies that worked:")
            for s, rs in sorted(worked.items(), key=lambda x: -max(r["speedup_achieved"] for r in x[1]))[:5]:
                best = max(r["speedup_achieved"] for r in rs)
                lines.append(f"  - {s}: best {best:.2f}x ({len(rs)} trials)")

        if failed:
            lines.append("\nStrategies that failed:")
            for s, rs in list(failed.items())[:3]:
                reasons = set(r.get("failure_reason", "") for r in rs if r.get("failure_reason"))
                lines.append(f"  - {s}: 0/{len(rs)} success" +
                             (f" ({'; '.join(reasons)})" if reasons else ""))

        cross_kernel = [r for r in results
                        if r.get("kernel_category") != kernel_category and r.get("success")]
        if cross_kernel:
            lines.append(f"\nCross-kernel insights (similar bottleneck):")
            for r in cross_kernel[:3]:
                lines.append(
                    f"  - {r.get('kernel_category', '?')} ({r.get('strategy_name', '?')}): "
                    f"{r['speedup_achieved']:.2f}x"
                )

        lines.append("---")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        total = len(self._entries)
        successful = sum(1 for e in self._entries if e.get("success"))
        categories = list({e.get("kernel_category", "unknown") for e in self._entries})
        return {
            "total_outcomes": total,
            "successful_outcomes": successful,
            "kernel_categories": categories,
            "store_path": str(self.store_path),
        }

    def close(self):
        pass
