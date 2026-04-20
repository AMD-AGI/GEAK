"""Code-similarity-first retrieval for cross-session memory.

The primary signal is direct source-code similarity between the target
kernel and each KB entry's stored ``original_kernel_code``. A byte-for-byte
match means the stored patch applies verbatim; a high line-set overlap
means the technique likely applies with minor adaptation; a low overlap
means any reuse requires non-trivial translation.

Secondary signal is verified speedup magnitude (higher peak = more
transferable). Name-stem overlap is a tie-breaker for runs whose KB
does not yet contain ``original_kernel_code`` (pre-backfill entries).

Removed in this simplification:
  * category boost (redundant with code similarity)
  * bottleneck match +/-0.25 (was penalizing same-kernel entries whose
    stored bottleneck classification differed from runtime — the very
    regression this module was meant to fix)
  * language boost (code similarity already implies same language)
  * diversity penalty on best_change_category (was hiding multiple
    same-kernel entries in favor of strategy variety)

The result: when the KB has N entries for the same kernel, all N surface
in the top-K, ranked by verified speedup. When the KB has no same-kernel
entries, cross-kernel entries are ranked by code similarity — same-family
seeds (e.g. shared RMS reduction pattern) rise above unrelated ones.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
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
_STEM_BOOST_MAX = 0.10
_CODE_SIM_WEIGHT = 1.0


def retrieve_context(
    backend: MemoryBackend,
    kernel_path: str = "",
    bottleneck_type: str = "",
    profiling_metrics: dict[str, Any] | None = None,
    limit: int = 30,
    top_k: int = 5,
    compact: bool = False,
) -> str:
    """Rank KB entries by code similarity to the current kernel and return formatted context."""
    kernel_category = _infer_category(kernel_path)
    language = _infer_language(kernel_path)
    target_code = _read_target_code(kernel_path)

    logger.info(
        "Retriever: path=...%s target_code=%dB category=%s language=%s bottleneck=%s",
        kernel_path[-80:],
        len(target_code),
        kernel_category,
        language,
        bottleneck_type,
    )

    # Stage 1: fetch all candidates (broad)
    candidates = backend.search_experiences(limit=limit)
    if not candidates:
        return ""

    # Stage 2: code-similarity-based scoring
    scored = _stage2_code_similarity(candidates, target_code, kernel_path)

    # Relevance gate: emit something only if at least one entry has a
    # non-trivial code overlap OR shares name stem with the target.
    # Prevents unrelated KB entries from being shown when none apply.
    best_score = scored[0][0] if scored else 0.0
    if best_score < 0.02:
        logger.info(
            "Retriever: best score %.4f below 0.02 relevance gate, skipping",
            best_score,
        )
        return ""

    # Stage 3: take top-k by score (no diversity penalty — we want
    # multiple same-kernel entries when they exist)
    top = [exp for _, exp in scored[:top_k]]
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
        query_category=kernel_category,
        query_bottleneck=bottleneck_type,
        query_language=language,
        compact=compact,
        target_kernel_path=kernel_path,
    )


def _read_target_code(kernel_path: str) -> str:
    """Read the current kernel's full source for similarity comparison."""
    if not kernel_path:
        return ""
    try:
        return Path(kernel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


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
    # Fast byte-identical short-circuit
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
    target_kernel_path: str,
) -> list[tuple[float, ExperienceRecord]]:
    """Score candidates by code similarity + capped speedup + name-stem boost.

    Ranking contract:
      * Primary: code_sim in [0, 1]. 1.0 means byte-identical source.
      * Secondary: scaled verified speedup in [0, _SUCCESS_BOOST_MAX=0.30].
      * Tie-break: kernel-name stem overlap in [0, _STEM_BOOST_MAX=0.10].

    A byte-identical entry with any verified speedup always outranks a
    non-identical entry (since 1.0 + 0 > 0.99 + 0.30 + 0.10).
    """
    scored: list[tuple[float, ExperienceRecord]] = []

    for exp in candidates:
        kb_code = getattr(exp, "original_kernel_code", "") or ""
        code_sim = _code_similarity(target_code, kb_code) if kb_code else 0.0

        success_boost = min(
            _SUCCESS_BOOST_MAX,
            _scaled_success_boost(exp.best_speedup, bool(exp.success)),
        )
        stem_boost = min(
            _STEM_BOOST_MAX,
            _kernel_stem_overlap(target_kernel_path, exp.kernel_name),
        )

        total = _CODE_SIM_WEIGHT * code_sim + success_boost + stem_boost
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


def _kernel_stem_overlap(target_path: str, kb_kernel_name: str) -> float:
    """Compute a fine-grained name-stem overlap between the target kernel's
    path and a KB experience's kernel name.

    Returns a boost in [0.0, 0.30] proportional to the number of shared
    distinctive tokens (len >= 3, stopwords removed). This makes same-family
    seeds — e.g. ``fused_rms_fp8`` vs ``fast_rms_layernorm`` (shared:
    ``rms``), or ``fused_qkv_rope`` vs ``fused_qk_rope_cache_mla``
    (shared: ``fused``, ``rope``) — rank above random same-language
    experiences even when path-level text_sim is noisy.
    """

    if not target_path or not kb_kernel_name:
        return 0.0

    stopwords = {"the", "and", "kernel", "triton", "hip", "py", "cpp", "cu", "tasks", "geak", "eval"}

    def _tokens(s: str) -> set[str]:
        out: set[str] = set()
        for w in re.split(r"[/_.\-\s]+", s.lower()):
            if len(w) >= 3 and w not in stopwords:
                out.add(w)
        return out

    # Target tokens: only the trailing directory name (kernel family)
    tail = target_path.rstrip("/").rsplit("/", 1)[-1]
    if tail.endswith(".py"):
        tail = target_path.rstrip("/").rsplit("/", 2)[-2] if "/" in target_path else tail
    tgt_tokens = _tokens(tail)
    if not tgt_tokens:
        return 0.0
    kb_tokens = _tokens(kb_kernel_name)
    if not kb_tokens:
        return 0.0

    overlap = tgt_tokens & kb_tokens
    if not overlap:
        return 0.0
    # Scale: 1 token shared → +0.10, 2 → +0.20, 3+ → +0.30
    return min(0.30, 0.10 * len(overlap))


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


def _infer_category(kernel_path: str) -> str:
    """Quick category inference from path."""
    try:
        from minisweagent.memory.cross_session_memory import classify_kernel_category

        return classify_kernel_category(kernel_path)
    except ImportError:
        pass

    p = kernel_path.lower()
    for tag, cat in [
        ("gemm", "gemm"),
        ("matmul", "gemm"),
        ("attention", "attention"),
        ("mla", "attention"),
        ("sdpa", "attention"),
        ("norm", "normalization"),
        ("rms", "normalization"),
        ("moe", "moe"),
        ("expert", "moe"),
        ("rope", "positional_encoding"),
        ("rotary", "positional_encoding"),
        ("topk", "topk"),
        ("softmax", "softmax"),
        ("reduce", "reduction"),
        ("ff", "ffn"),
        ("feedforward", "ffn"),
        ("linear", "gemm"),
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
