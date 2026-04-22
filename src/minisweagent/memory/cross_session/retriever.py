"""Pure code-similarity retrieval for cross-session memory.

The retriever takes the target kernel's **source code** (as a string) and
ranks KB entries by Jaccard similarity to each entry's stored
``original_kernel_code``. There is no path handling here — the caller is
responsible for obtaining the source before invoking retrieval.

Why this matters: sub-agent contexts often pass a working-dir path that
doesn't exist yet (the bootstrap hasn't copied the kernel into that
location yet), which used to silently yield ``target_code=""`` and made
every code-similarity score zero. By only accepting the raw source
string, the retriever can never regress due to path-resolution quirks.

Scoring:

  total = code_sim  +  scaled_success_boost

  * code_sim in [0, 1]: Jaccard over whitespace-normalized non-trivial
    lines. 1.0 = byte-identical source (the stored patch applies
    verbatim). ~0.5 = sibling / stripped variant. < 0.2 = unrelated.
  * scaled_success_boost in [0, 0.30]: rewards higher verified speedups
    so two code-identical entries surface in descending-speedup order.

Tie-break: raw ``best_speedup`` (so 2.23x same-kernel surfaces before
1.80x same-kernel when both have code_sim=1.0).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

from minisweagent.memory.cross_session.backends.base import MemoryBackend
from minisweagent.memory.cross_session.formatter import format_landscape_context
from minisweagent.memory.cross_session.schemas import ExperienceRecord

logger = logging.getLogger(__name__)

# Single-source-of-truth speedup threshold (mirrors config.min_store_speedup
# and formatter._SPEEDUP_THRESHOLD). Any strategy >= this is "WORKED";
# below 1.0 is "REGRESSED"; in-between is "MARGINAL".
_SPEEDUP_THRESHOLD = float(os.environ.get("GEAK_MEMORY_MIN_SPEEDUP", "1.10"))

# Weight of verified speedup vs code similarity in the final score.
# Code similarity dominates (0..1 scale). Speedup boost is capped at 0.30
# so a strong but unrelated-code entry cannot outrank a weak-speedup but
# byte-identical entry.
_SUCCESS_BOOST_MAX = 0.30
_CODE_SIM_WEIGHT = 1.0

# Minimum code_sim / success_boost score required before the retriever
# emits anything. Prevents unrelated KB entries from being shown when
# no candidate has meaningful overlap. The floor is set just above zero
# so a lone high-success (e.g., 3.92x success_boost=0.20) entry still
# qualifies when no same-kernel entry exists.
_MIN_RELEVANCE = 0.02


def retrieve_context(
    backend: MemoryBackend,
    target_code: str = "",
    bottleneck_type: str = "",
    profiling_metrics: dict[str, Any] | None = None,
    limit: int = 30,
    top_k: int = 5,
    compact: bool = False,
) -> str:
    """Rank KB entries by code similarity to ``target_code`` and return formatted context.

    ``target_code`` is the raw source of the kernel being optimized right
    now. The caller reads the file (or constructs the string however) and
    passes it in directly — the retriever never touches the filesystem.
    """
    logger.info(
        "Retriever: target_code=%dB bottleneck=%s",
        len(target_code),
        bottleneck_type or "unknown",
    )

    # Stage 1: fetch all candidates (broad)
    candidates = backend.search_experiences(limit=limit)
    if not candidates:
        return ""

    # Stage 2: code-similarity scoring
    scored = _stage2_code_similarity(candidates, target_code)

    best_score = scored[0][0] if scored else 0.0
    if best_score < _MIN_RELEVANCE:
        logger.info(
            "Retriever: best score %.4f below %.2f relevance gate, skipping",
            best_score,
            _MIN_RELEVANCE,
        )
        return ""

    # Stage 3: take top-k by score (no diversity penalty — we want
    # multiple same-kernel entries when they exist)
    top = [exp for _, exp in scored[:top_k]]
    if not top:
        return ""

    # Compute per-entry code similarity once so the formatter can surface
    # the raw number alongside each entry. The agent needs this signal
    # explicitly: when ALL top entries have low code_sim the KB is a
    # DISTANT reference set (cross-family at best), and the agent should
    # lean on its own kernel + profile for strategy ideas. Without this
    # number the agent tends to anchor on whatever entry ranks highest
    # -- even if that entry is only 13% code-similar to the current
    # kernel, which is fine as a weak hint but misleading as a blueprint.
    per_entry_code_sim: list[float] = []
    for exp in top:
        kb_code = getattr(exp, "original_kernel_code", "") or ""
        per_entry_code_sim.append(_code_similarity(target_code, kb_code) if kb_code else 0.0)

    # Optional: skills are still retrieved using whatever metadata the
    # backend has; they're a lightweight supplement, not the primary signal.
    skills = []
    try:
        skills = backend.search_skills(bottleneck=bottleneck_type or None, limit=5)
    except Exception as exc:
        logger.debug("search_skills failed (non-fatal): %s", exc)

    # NOTE: Domain-KB (generic AMD/ROCm/HIP documentation) lives in a
    # separate path — the rag-mcp MCP server (enabled via ``tools.rag``).
    # Agent calls its ``query`` / ``optimize`` tools on demand rather
    # than receiving pre-injected snippets. This file is the EXPERIENCE
    # side only (per-kernel verified runs with real diffs + full kernel
    # code + profiler metrics + dead-ends).
    return format_landscape_context(
        experiences=top,
        skills=skills,
        query_bottleneck=bottleneck_type,
        compact=compact,
        target_code=target_code,
        per_entry_code_sim=per_entry_code_sim,
    )


def _normalized_lines(code: str) -> set[str]:
    """Split code into a set of non-trivial normalized lines.

    Drops blank lines, comments, and lines shorter than 4 chars after
    whitespace collapse. This makes the set-Jaccard metric robust to
    reformatting while still capturing structural similarity.
    """
    if not code:
        return set()
    out: set[str] = set()
    for line in code.splitlines():
        s = re.sub(r"\s+", " ", line).strip()
        if len(s) < 4:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        out.add(s)
    return out


def _code_similarity(target_code: str, kb_code: str) -> float:
    """Jaccard similarity over normalized non-trivial lines.

    Returns 1.0 iff the two source files are byte-identical under
    whitespace normalization. Returns 0.0 for completely disjoint
    implementations. Values in between scale roughly with the fraction
    of shared structural lines.
    """
    if not target_code or not kb_code:
        return 0.0
    if hashlib.sha256(target_code.encode()).hexdigest() == hashlib.sha256(kb_code.encode()).hexdigest():
        return 1.0
    t = _normalized_lines(target_code)
    k = _normalized_lines(kb_code)
    if not t or not k:
        return 0.0
    overlap = len(t & k)
    union = len(t | k)
    return overlap / union if union else 0.0


def _stage2_code_similarity(
    candidates: list[ExperienceRecord],
    target_code: str,
) -> list[tuple[float, ExperienceRecord]]:
    """Score candidates by code similarity + capped speedup boost.

    Scoring contract:
      * Primary: code_sim in [0, 1]. 1.0 means byte-identical source.
      * Secondary: scaled verified speedup in [0, _SUCCESS_BOOST_MAX=0.30].

    A byte-identical entry with any verified speedup always outranks a
    non-identical entry (since 1.0 + 0 > 0.99 + 0.30).
    """
    scored: list[tuple[float, ExperienceRecord]] = []

    for exp in candidates:
        kb_code = getattr(exp, "original_kernel_code", "") or ""
        code_sim = _code_similarity(target_code, kb_code) if kb_code else 0.0
        success_boost = min(
            _SUCCESS_BOOST_MAX,
            _scaled_success_boost(exp.best_speedup, bool(exp.success)),
        )
        total = _CODE_SIM_WEIGHT * code_sim + success_boost
        scored.append((total, exp))

    # Sort by total score desc, with verified speedup as tie-break so that
    # among byte-identical same-kernel entries (all at code_sim=1.0), the
    # higher-speedup one surfaces first in the formatter.
    scored.sort(key=lambda x: (-x[0], -x[1].best_speedup))
    if scored:
        top = scored[0]
        logger.info(
            "Retriever scoring: top=%s sp=%.3fx total=%.3f (code_sim strong=%s)",
            top[1].kernel_name,
            top[1].best_speedup,
            top[0],
            "yes" if top[0] >= 0.6 else ("partial" if top[0] >= 0.2 else "low"),
        )
    return scored


def _scaled_success_boost(speedup: float, success: bool) -> float:
    """Reward KB experiences that achieved higher verified speedups — these
    contain more transferable signal than borderline wins. The lower bound
    ``_SPEEDUP_THRESHOLD`` mirrors the KB-write threshold (1.10x), so any
    entry that survived KB filtering qualifies for at least the base boost.
    """
    if not success or speedup < _SPEEDUP_THRESHOLD:
        return 0.0
    if speedup >= 2.5:
        return 0.20
    if speedup >= 1.5:
        return 0.10
    if speedup >= 1.2:
        return 0.07
    return 0.05
