"""Multi-stage retrieval funnel for cross-session memory.

Stage 1: Coarse SQL filter (language, hardware)
Stage 2: Profiling fingerprint scoring (weighted cosine similarity)
Stage 3: Category/context re-ranking with diversity penalty
Stage 4: Landscape aggregation via formatter
"""

from __future__ import annotations

import logging
from typing import Any

from minisweagent.memory.cross_session.backends.base import MemoryBackend
from minisweagent.memory.cross_session.fingerprint import (
    bottleneck_bonus,
    build_fingerprint,
    category_bonus,
    fingerprint_similarity,
)
from minisweagent.memory.cross_session.formatter import format_landscape_context
from minisweagent.memory.cross_session.schemas import ExperienceRecord

logger = logging.getLogger(__name__)


def retrieve_context(
    backend: MemoryBackend,
    kernel_path: str = "",
    bottleneck_type: str = "",
    profiling_metrics: dict[str, Any] | None = None,
    limit: int = 30,
    top_k: int = 5,
) -> str:
    """Run the full retrieval funnel and return formatted context."""
    profiling_metrics = profiling_metrics or {}

    kernel_category = _infer_category(kernel_path)
    language = _infer_language(kernel_path)

    # Stage 1: coarse filter
    candidates = _stage1_coarse_filter(backend, language=language, limit=limit)
    if not candidates:
        candidates = _stage1_coarse_filter(backend, language=None, limit=limit)
    if not candidates:
        return ""

    # Stage 2: fingerprint scoring
    query_fp = build_fingerprint(profiling_metrics.get("metrics", profiling_metrics))
    scored = _stage2_fingerprint_scoring(candidates, query_fp, bottleneck_type, kernel_category)

    # Stage 3: re-rank with diversity
    top = _stage3_rerank_diverse(scored, top_k=top_k)

    if not top:
        return ""

    # Also fetch consolidated skills
    skills = []
    try:
        skills = backend.search_skills(
            category=kernel_category if kernel_category != "unknown" else None,
            bottleneck=bottleneck_type or None,
            language=language if language != "unknown" else None,
            limit=5,
        )
    except Exception:
        pass

    # Stage 4: format as landscape
    return format_landscape_context(
        experiences=top,
        skills=skills,
        query_category=kernel_category,
        query_bottleneck=bottleneck_type,
    )


def _stage1_coarse_filter(
    backend: MemoryBackend,
    language: str | None = None,
    limit: int = 30,
) -> list[ExperienceRecord]:
    """SQL-level coarse filter."""
    lang = language if language and language != "unknown" else None
    return backend.search_experiences(language=lang, limit=limit)


def _stage2_fingerprint_scoring(
    candidates: list[ExperienceRecord],
    query_fp: list[float],
    query_bottleneck: str,
    query_category: str,
) -> list[tuple[float, ExperienceRecord]]:
    """Score each candidate by profiling fingerprint similarity + bonuses."""
    scored: list[tuple[float, ExperienceRecord]] = []

    for exp in candidates:
        exp_fp = build_fingerprint(exp.profiling_metrics)
        sim = fingerprint_similarity(query_fp, exp_fp)
        bn_bonus = bottleneck_bonus(query_bottleneck, exp.bottleneck_type)
        cat_bonus = category_bonus(query_category, exp.kernel_category)

        success_bonus = 0.1 if exp.success else 0.0
        total = sim + bn_bonus + cat_bonus + success_bonus
        scored.append((total, exp))

    scored.sort(key=lambda x: -x[0])
    return scored


def _stage3_rerank_diverse(
    scored: list[tuple[float, ExperienceRecord]],
    top_k: int = 5,
) -> list[ExperienceRecord]:
    """Select top-K with diversity penalty to avoid strategy monoculture."""
    if not scored:
        return []

    selected: list[ExperienceRecord] = []
    seen_categories: dict[str, int] = {}

    for score, exp in scored:
        if len(selected) >= top_k:
            break

        cat = exp.best_change_category or "other"
        cat_count = seen_categories.get(cat, 0)

        # Diversity: penalty for 3rd+ instance of same strategy category
        if cat_count >= 2:
            adjusted = score * 0.5
            # Skip if there are better-scoring candidates still available
            remaining = [(s, e) for s, e in scored if e not in selected and (e.best_change_category or "other") != cat]
            if remaining and remaining[0][0] > adjusted:
                continue

        selected.append(exp)
        seen_categories[cat] = cat_count + 1

    return selected


def _infer_category(kernel_path: str) -> str:
    """Quick category inference from path (delegates to existing classifier when available)."""
    try:
        from minisweagent.memory.cross_session_memory import classify_kernel_category
        return classify_kernel_category(kernel_path)
    except ImportError:
        pass

    p = kernel_path.lower()
    for tag, cat in [
        ("gemm", "gemm"), ("matmul", "gemm"), ("attention", "attention"),
        ("mla", "attention"), ("sdpa", "attention"), ("norm", "normalization"),
        ("moe", "moe"), ("rope", "positional_encoding"), ("rotary", "positional_encoding"),
        ("topk", "topk"), ("softmax", "softmax"), ("reduce", "reduction"),
    ]:
        if tag in p:
            return cat
    return "unknown"


def _infer_language(kernel_path: str) -> str:
    p = kernel_path.lower()
    if any(k in p for k in (".hip", "hip")):
        return "hip"
    if "triton" in p or p.endswith(".py"):
        return "triton"
    if any(k in p for k in (".cu", "cuda")):
        return "cuda"
    return "unknown"
