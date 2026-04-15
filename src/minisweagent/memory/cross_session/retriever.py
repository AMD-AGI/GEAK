"""Multi-stage retrieval funnel for cross-session memory.

Stage 1: Fetch all candidates (broad, no hard filters)
Stage 2: Text similarity scoring (keyword overlap between query context
         and stored insights/strategies/what_worked)
Stage 3: Soft boosts (category, bottleneck, success) + diversity penalty
Stage 4: Landscape aggregation via formatter

Text similarity is the primary signal because optimization strategies
and code patterns transfer across hardware and kernel categories.
A code diff showing "fuse QK+softmax" is useful for any attention-like
kernel regardless of whether profiling metrics match.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from minisweagent.memory.cross_session.backends.base import MemoryBackend
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
    compact: bool = False,
) -> str:
    """Run the full retrieval funnel and return formatted context."""
    kernel_category = _infer_category(kernel_path)
    language = _infer_language(kernel_path)
    logger.info("Retriever: category=%s language=%s bottleneck=%s path=...%s",
                kernel_category, language, bottleneck_type, kernel_path[-80:])

    # Stage 1: fetch all candidates (broad)
    candidates = backend.search_experiences(limit=limit)
    if not candidates:
        return ""

    # Stage 2: text similarity scoring
    query_terms = _build_query_terms(kernel_path, kernel_category, bottleneck_type)
    scored = _stage2_text_similarity(candidates, query_terms, kernel_category, bottleneck_type, language)

    # Relevance gate: require at least one genuine relevance signal before
    # injecting memory. Language boost alone isn't enough -- it causes
    # irrelevant same-language experiences to distract the agent.
    # Signals: (a) known category match, OR (b) meaningful text overlap.
    _MIN_TEXT_SIM = 0.05
    best_text_sim = max(
        (_text_similarity(query_terms, _experience_text(exp)) for _, exp in scored),
        default=0.0,
    )
    has_category_match = (
        kernel_category != "unknown"
        and any(exp.kernel_category == kernel_category for _, exp in scored)
    )
    if not has_category_match and best_text_sim < _MIN_TEXT_SIM:
        logger.info(
            "Retriever: no category match and best text_sim=%.4f < %.2f, skipping",
            best_text_sim, _MIN_TEXT_SIM,
        )
        return ""

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
        query_language=language,
        compact=compact,
    )


def _build_query_terms(kernel_path: str, category: str, bottleneck: str) -> set[str]:
    """Build query keywords from kernel path AND source code identifiers."""
    terms: set[str] = set()

    path_lower = kernel_path.lower()
    for word in re.split(r"[/_.\-\s]+", path_lower):
        if len(word) >= 3:
            terms.add(word)

    if category and category != "unknown":
        terms.add(category)
    if bottleneck and bottleneck != "unknown":
        terms.add(bottleneck)

    terms.update(_extract_source_terms(kernel_path))

    _DOMAIN_SYNONYMS = {
        "attention": {"attention", "attn", "mla", "sdpa", "softmax", "qkv", "kv", "rope"},
        "gemm": {"gemm", "matmul", "mm", "matrix", "multiply", "linear"},
        "normalization": {"norm", "rms", "layernorm", "rmsnorm", "normalization"},
        "moe": {"moe", "expert", "routing", "gating", "topk", "dispatch"},
        "positional_encoding": {"rope", "rotary", "positional", "embedding", "cos", "sin"},
        "memory": {"memory", "bandwidth", "coalescing", "vectorized", "loads", "hbm"},
        "compute": {"compute", "flops", "mfma", "arithmetic", "intensity"},
        "latency": {"latency", "launch", "overhead", "pipeline", "stall"},
        "fusion": {"fuse", "fused", "fusion", "merge", "combine", "single-pass"},
        "spatial_search": {
            "nearest", "neighbor", "radius", "spatial", "distance",
            "point_cloud", "interpolate", "search", "gather", "scatter",
        },
    }
    expanded: set[str] = set()
    for term in terms:
        for _group_key, synonyms in _DOMAIN_SYNONYMS.items():
            if term in synonyms:
                expanded.update(synonyms)
    terms.update(expanded)
    return terms


def _experience_text(exp: ExperienceRecord) -> str:
    """Concatenate all text fields of an experience into a searchable blob."""
    parts = [
        exp.kernel_name,
        exp.kernel_category,
        exp.bottleneck_type,
        exp.best_strategy,
        exp.best_change_category,
        exp.key_insight,
        exp.trajectory_sketch,
        exp.code_changes_summary,
        exp.profiling_insight,
        exp.kernel_url,
        exp.kernel_structure,
    ]
    parts.extend(exp.what_worked)
    parts.extend(exp.what_failed)
    parts.extend(exp.dead_ends)
    parts.extend(exp.round_insights)
    for strat in exp.strategies:
        parts.append(strat.get("task", ""))
    if exp.patch_content:
        code_words = re.findall(r'\b[a-zA-Z_]\w{3,}\b', exp.patch_content[:2000])
        parts.extend(code_words[:50])
    return " ".join(p for p in parts if p).lower()


def _text_similarity(query_terms: set[str], doc_text: str) -> float:
    """Simple keyword overlap score (Jaccard-like)."""
    if not query_terms or not doc_text:
        return 0.0

    doc_words = set(re.split(r"[/_.\-\s,;:()]+", doc_text))
    doc_words = {w for w in doc_words if len(w) >= 3}

    if not doc_words:
        return 0.0

    overlap = query_terms & doc_words
    return len(overlap) / (len(query_terms) + len(doc_words) - len(overlap))


def _stage2_text_similarity(
    candidates: list[ExperienceRecord],
    query_terms: set[str],
    query_category: str,
    query_bottleneck: str,
    query_language: str = "",
) -> list[tuple[float, ExperienceRecord]]:
    """Score each candidate by text similarity + soft boosts."""
    scored: list[tuple[float, ExperienceRecord]] = []

    for exp in candidates:
        doc_text = _experience_text(exp)
        text_sim = _text_similarity(query_terms, doc_text)

        cat_boost = 0.15 if (
            query_category and query_category != "unknown"
            and exp.kernel_category == query_category
        ) else 0.0

        if query_bottleneck and query_bottleneck != "unknown":
            if exp.bottleneck_type == query_bottleneck:
                bn_boost = 0.25
            elif exp.bottleneck_type != "unknown":
                bn_boost = -0.15
            else:
                bn_boost = 0.0
        else:
            bn_boost = 0.0

        exp_lang = getattr(exp, "kernel_language", "") or ""
        if query_language and query_language != "unknown" and exp_lang:
            lang_boost = 0.20 if exp_lang == query_language else -0.10
        else:
            lang_boost = 0.0

        success_boost = 0.05 if exp.success and exp.best_speedup > 1.05 else 0.0

        total = text_sim + cat_boost + bn_boost + lang_boost + success_boost
        scored.append((total, exp))

    scored.sort(key=lambda x: -x[0])
    if scored:
        logger.info("Retriever scoring: top=%s(%.3f) bottom=%s(%.3f)",
                     scored[0][1].kernel_name, scored[0][0],
                     scored[-1][1].kernel_name, scored[-1][0])
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

        if cat_count >= 2:
            adjusted = score * 0.5
            remaining = [
                (s, e) for s, e in scored
                if e not in selected and (e.best_change_category or "other") != cat
            ]
            if remaining and remaining[0][0] > adjusted:
                continue

        selected.append(exp)
        seen_categories[cat] = cat_count + 1

    return selected


def _infer_category(kernel_path: str) -> str:
    """Quick category inference from path."""
    try:
        from minisweagent.memory.cross_session_memory import classify_kernel_category
        return classify_kernel_category(kernel_path)
    except ImportError:
        pass

    p = kernel_path.lower()
    for tag, cat in [
        ("gemm", "gemm"), ("matmul", "gemm"), ("attention", "attention"),
        ("mla", "attention"), ("sdpa", "attention"), ("norm", "normalization"),
        ("rms", "normalization"), ("moe", "moe"), ("expert", "moe"),
        ("rope", "positional_encoding"), ("rotary", "positional_encoding"),
        ("topk", "topk"), ("softmax", "softmax"), ("reduce", "reduction"),
        ("ff", "ffn"), ("feedforward", "ffn"), ("linear", "gemm"),
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


_NOISE_WORDS = frozenset({
    "int", "float", "void", "const", "return", "bool", "auto", "char",
    "include", "define", "pragma", "ifdef", "endif", "nullptr", "true",
    "false", "this", "struct", "class", "template", "typename", "static",
    "inline", "extern", "restrict", "volatile", "unsigned", "size_t",
    "blockidx", "blockdim", "threadidx", "griddim", "warpsize",
    "hipstream_t", "cudastream_t", "hipstream", "cudastream",
    "hiperror_t", "cudaerror_t", "hipmalloc", "cudamalloc",
})


def _extract_source_terms(kernel_path: str, max_tokens: int = 80) -> set[str]:
    """Extract meaningful identifiers from kernel source code.

    Reads the first ~4KB of the kernel file and pulls out C/HIP/Triton
    identifiers (function names, variable names, algorithmic keywords).
    These terms dramatically improve retrieval relevance compared to
    path-only matching.
    """
    from pathlib import Path

    p = Path(kernel_path)
    if not p.is_file():
        return set()

    try:
        text = p.read_text(errors="ignore")[:4096]
    except Exception:
        return set()

    raw = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", text))
    terms = {w.lower() for w in raw} - _NOISE_WORDS
    if len(terms) > max_tokens:
        terms = set(sorted(terms)[:max_tokens])
    return terms
