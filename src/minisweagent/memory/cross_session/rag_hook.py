"""Lightweight RAG hook over the rag-mcp knowledge-base markdown corpus.

Complements the experience-based cross-session memory with domain-grounded
optimization knowledge (AMD aiter reports, Triton-on-ROCm guides, layer-6
kernel case studies) from the PR #90 `rag-mcp` knowledge base.

No external dependencies, no server startup, no FAISS / BGE reranker.
A simple stdlib BM25 index is lazily built once per process against a
pre-filtered subset of the KB most relevant to Triton kernel optimization:

  * layer-3-libraries/compilers/triton-on-rocm.md
  * layer-6-extended/optimize-guides/customer-case/**/Report_*.md
  * layer-6-extended/optimize-guides/L1-important/performance-guidelines.md
  * layer-6-extended/optimize-guides/L1-important/programming-patterns.md
  * best-practices/performance/*.md

The hook returns structured snippets (title, layer, body excerpt) that
``format_landscape_context()`` splices into the LLM's context right after
the FIRST-MOVE directive.

Design choices:
  * Lazy index build: first ``query_rag()`` call scans, tokenizes, and
    caches. All subsequent calls reuse the in-memory index.
  * Empty-return semantics: if the KB path is missing or no docs match,
    return ``[]``. The formatter then cleanly skips the Domain-KB block.
  * Snippet-level retrieval: a doc is sliced into its ``##`` section
    subsections; the highest-matching subsection is returned (not the
    whole doc), keeping the context budget tight.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Search these subpaths (relative to the KB root) — highest-signal for Triton kernel work.
_KB_SEARCH_SUBPATHS = (
    "amd-knowledge-base/layer-3-libraries/compilers",
    "amd-knowledge-base/layer-6-extended/optimize-guides/customer-case",
    "amd-knowledge-base/layer-6-extended/optimize-guides/L1-important",
    "amd-knowledge-base/best-practices/performance",
)

# Candidate KB root locations (first existing is used).
_KB_CANDIDATES = (
    # 1. Env override
    "GEAK_RAG_KB_PATH",
    # 2. Main repo's cached copy (populated via git archive)
    "mcp_tools/rag-mcp-kb-cache/knowledge-base",
    # 3. Repo's own rag-mcp checkout (if PR #90 is merged)
    "mcp_tools/rag-mcp/knowledge-base",
)

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "into", "are", "was",
        "were", "has", "have", "but", "not", "can", "will", "only", "use", "uses",
        "used", "its", "when", "what", "any", "all", "one", "two", "more", "also",
        "their", "them", "they", "there", "which", "where", "these", "those", "such",
        "some", "most", "many", "much", "very", "just", "like", "than", "per",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _tokenize(text: str) -> list[str]:
    """Extract alphanumeric tokens (len >= 3), lowercased, stopword-filtered."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


@dataclass
class _KbDoc:
    """In-memory representation of a single KB markdown subsection."""

    path: str
    title: str
    layer: str
    body: str
    tokens: list[str] = field(default_factory=list)
    tf: dict[str, int] = field(default_factory=dict)


# Path-based multiplicative boosts reward high-signal doc subtrees:
# aiter customer-case reports are the closest analogue to our GEAK eval
# kernels (same codebase lineage), so they outrank generic tutorials.
_PATH_BOOST_RULES = (
    ("customer-case/github_repo/aiter/", 3.0),
    ("customer-case/github_repo/composable_kernel/", 2.0),
    ("customer-case/github_repo/sglang/", 1.7),
    ("customer-case/github_repo/vllm/", 1.4),
    ("customer-case/github_repo/", 1.5),  # fallback for other customer-case
    ("optimize-guides/L1-important/performance-guidelines", 1.5),
    ("optimize-guides/L1-important/programming-patterns", 1.5),
    ("best-practices/performance/", 1.3),
)

# Kernel-family keyword boosts: if query contains these family markers
# AND doc path also does, multiply score.
_FAMILY_KEYWORDS = (
    ({"rms", "rmsnorm", "layernorm"}, "rmsnorm", 2.0),
    ({"rms", "rmsnorm", "layernorm"}, "norm", 1.4),
    ({"rope", "rotary"}, "rope", 2.5),
    ({"rope", "rotary"}, "rotary_embedding", 2.5),
    ({"fp4", "mxfp4", "afp4", "wfp4", "e2m1"}, "afp4wfp4", 3.0),
    ({"fp4", "mxfp4"}, "fp4", 2.0),
    ({"fp8", "e4m3", "e5m2"}, "fp8", 2.0),
    ({"gemm", "matmul"}, "gemm", 1.5),
    ({"moe", "expert", "routing", "topk"}, "moe", 1.5),
    ({"attention", "attn", "flash"}, "attention", 1.5),
)


def _compute_path_boost(path: str, query_token_set: set[str]) -> float:
    """Multiplicative boost combining path subtree + family keyword match."""
    boost = 1.0
    for prefix, multiplier in _PATH_BOOST_RULES:
        if prefix in path:
            boost *= multiplier
            break  # only apply one path-prefix rule (most specific wins)
    for query_family, path_tag, multiplier in _FAMILY_KEYWORDS:
        if query_family & query_token_set and path_tag in path.lower():
            boost *= multiplier
    return boost


@dataclass
class _BM25Index:
    docs: list[_KbDoc]
    df: dict[str, int]
    avgdl: float
    n_docs: int

    def score(self, query_tokens: list[str], k1: float = 1.4, b: float = 0.75) -> list[tuple[float, _KbDoc]]:
        """Standard BM25 scoring + path/family boosts over query_tokens."""
        if not self.docs:
            return []
        query_set = set(query_tokens)
        scored: list[tuple[float, _KbDoc]] = []
        for doc in self.docs:
            dl = max(len(doc.tokens), 1)
            score = 0.0
            for q in query_tokens:
                if q not in self.df:
                    continue
                idf = math.log(1.0 + (self.n_docs - self.df[q] + 0.5) / (self.df[q] + 0.5))
                tf = doc.tf.get(q, 0)
                numer = tf * (k1 + 1)
                denom = tf + k1 * (1.0 - b + b * dl / self.avgdl)
                score += idf * (numer / denom if denom else 0.0)
            if score <= 0:
                continue
            boost = _compute_path_boost(doc.path, query_set)
            scored.append((score * boost, doc))
        scored.sort(key=lambda x: -x[0])
        return scored


_INDEX_CACHE: _BM25Index | None = None
_KB_ROOT_RESOLVED: Path | None = None


def _resolve_kb_root() -> Path | None:
    """Find the first existing KB root from the candidate list."""
    # 1. Env override
    env_path = os.environ.get("GEAK_RAG_KB_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir():
            return p
        logger.info("GEAK_RAG_KB_PATH=%s not a directory, ignoring", env_path)

    # 2. Walk up from this module to find the repo root, then check known paths.
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        for rel in (
            "mcp_tools/rag-mcp-kb-cache/knowledge-base",
            "mcp_tools/rag-mcp/knowledge-base",
        ):
            candidate = parent / rel
            if candidate.is_dir():
                return candidate
    return None


def _extract_title(body: str, path: Path) -> str:
    """First `#` heading of the doc, falling back to the filename stem."""
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()[:120]
    return path.stem.replace("_", " ")


def _extract_layer(path: Path) -> str:
    """Infer a `layer-X` tag from the path for UI display."""
    parts = path.parts
    for p in parts:
        if p.startswith("layer-"):
            return p
    return "unknown"


def _split_subsections(body: str) -> list[tuple[str, str]]:
    """Split a markdown doc into (heading, section_body) chunks.

    The chunk unit is the `##` or `###` section (top `#` is the doc title).
    A doc with no sections returns a single ("", full_body) chunk.
    """
    lines = body.splitlines(keepends=True)
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_buf: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("## ", "### ")):
            if current_buf:
                sections.append((current_heading, "".join(current_buf)))
            current_heading = stripped.lstrip("#").strip()[:140]
            current_buf = []
        else:
            current_buf.append(line)
    if current_buf:
        sections.append((current_heading, "".join(current_buf)))
    if not sections:
        sections = [("", body)]
    return sections


def _build_index(kb_root: Path) -> _BM25Index:
    """Walk the KB subtree, build BM25 index over each subsection of each doc."""
    docs: list[_KbDoc] = []
    df: dict[str, int] = {}
    total_tokens = 0

    for sub in _KB_SEARCH_SUBPATHS:
        sub_root = kb_root / sub
        if not sub_root.exists():
            continue
        for md_path in sub_root.rglob("*.md"):
            try:
                text = md_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text) < 200:  # skip index / stub files
                continue
            doc_title = _extract_title(text, md_path)
            layer = _extract_layer(md_path)
            for heading, body in _split_subsections(text):
                if len(body.strip()) < 120:
                    continue
                combined_title = f"{doc_title} / {heading}" if heading else doc_title
                tokens = _tokenize(body + " " + combined_title)
                if len(tokens) < 20:
                    continue
                tf: dict[str, int] = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                # Per-doc DF increments (not per-token)
                for t in tf:
                    df[t] = df.get(t, 0) + 1
                docs.append(
                    _KbDoc(
                        path=str(md_path.relative_to(kb_root)),
                        title=combined_title,
                        layer=layer,
                        body=body,
                        tokens=tokens,
                        tf=tf,
                    )
                )
                total_tokens += len(tokens)

    avgdl = total_tokens / max(len(docs), 1)
    return _BM25Index(docs=docs, df=df, avgdl=avgdl, n_docs=len(docs))


def _ensure_index() -> _BM25Index | None:
    """Lazy singleton index — built once per process."""
    global _INDEX_CACHE, _KB_ROOT_RESOLVED
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    kb_root = _resolve_kb_root()
    if kb_root is None:
        logger.info("RAG hook: no KB root found in candidates, hook disabled")
        return None
    logger.info("RAG hook: building BM25 index from %s", kb_root)
    _INDEX_CACHE = _build_index(kb_root)
    _KB_ROOT_RESOLVED = kb_root
    logger.info(
        "RAG hook: indexed %d subsections across %d source paths (avgdl=%.1f)",
        _INDEX_CACHE.n_docs, len(_KB_SEARCH_SUBPATHS), _INDEX_CACHE.avgdl,
    )
    return _INDEX_CACHE


def _build_query(kernel_path: str, kernel_source: str) -> list[str]:
    """Tokenize kernel path + source identifiers into a BM25 query."""
    tokens: list[str] = []
    tokens.extend(_tokenize(kernel_path))
    if kernel_source:
        tokens.extend(_tokenize(kernel_source[:8000]))
    # Dedupe while preserving order (keep first occurrence for idf-style emphasis).
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def query_rag(
    kernel_path: str,
    kernel_source: str | None = None,
    top_k: int = 2,
    max_body_chars: int = 1500,
) -> list[dict]:
    """Return top-`top_k` KB subsections for this kernel.

    Each result dict has keys: ``title``, ``path``, ``layer``, ``body``, ``score``.
    Empty list on index-unavailable or no-match (formatter skips Domain-KB block).
    """
    index = _ensure_index()
    if index is None or not index.docs:
        return []

    if kernel_source is None and kernel_path:
        try:
            kernel_source = Path(kernel_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            kernel_source = ""

    query_tokens = _build_query(kernel_path, kernel_source or "")
    if not query_tokens:
        return []

    scored = index.score(query_tokens)
    if not scored:
        return []

    results: list[dict] = []
    for score, doc in scored[:top_k]:
        results.append(
            {
                "title": doc.title,
                "path": doc.path,
                "layer": doc.layer,
                "body": doc.body[:max_body_chars],
                "score": round(score, 3),
            }
        )
    return results
