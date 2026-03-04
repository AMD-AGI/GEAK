"""
RAG Knowledge Base MCP Server.

Provides GPU/ROCm/HIP knowledge retrieval via hybrid search
(embedding + BM25 + RRF fusion + BGE reranker).

Tools:
  - query:    Search the knowledge base by topic.
  - optimize: Retrieve optimization suggestions for GPU kernels.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from fastmcp import FastMCP

from rag_mcp.retrieval import DEFAULT_INDEX_PATH, HybridRetriever, _parse_tags

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load RAG config from file or environment variable.

    Lookup order:
    1. RAG_MCP_CONFIG env var (highest priority)
    2. ~/.config/rag-mcp/config.yaml (user-level override)
    3. Bundled config/rag_config.yaml (package default)
    """
    config_path = os.environ.get("RAG_MCP_CONFIG")
    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                logger.info("Loaded config from %s", p)
                return yaml.safe_load(f) or {}
    candidates = [
        Path.home() / ".config" / "rag-mcp" / "config.yaml",
        Path(__file__).parent / "config" / "rag_config.yaml",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                logger.info("Loaded config from %s", c)
                return yaml.safe_load(f) or {}
    logger.info("No config file found, using defaults")
    return {}


def _build_retriever(cfg: dict) -> HybridRetriever:
    """Construct a HybridRetriever from config dict."""
    retrieval = cfg.get("retrieval", {})
    reranker = cfg.get("reranker", {})
    fusion = cfg.get("fusion", {})

    index_path = os.environ.get("RAG_INDEX_PATH", str(DEFAULT_INDEX_PATH))

    enable_bm25 = retrieval.get("enable_bm25", True)
    bm25_top_k = retrieval.get("bm25_top_k", 25) if enable_bm25 else 0

    return HybridRetriever(
        index_path=Path(index_path),
        embed_top_k=retrieval.get("embed_top_k", 25),
        bm25_top_k=bm25_top_k,
        rrf_k=fusion.get("rrf_k", 60),
        semantic_weight=fusion.get("semantic_weight", 0.7),
        bm25_weight=fusion.get("bm25_weight", 0.3),
        enable_reranker=reranker.get("enable_reranker", True),
    )


# ---------------------------------------------------------------------------
# Server & retriever setup
# ---------------------------------------------------------------------------

_cfg = _load_config()
_retriever = _build_retriever(_cfg)
_top_k = _cfg.get("retrieval", {}).get("mcp_top_k", 8)

mcp = FastMCP(
    name="rag-knowledge-base",
    instructions=(
        "GPU/ROCm/HIP knowledge base retrieval. "
        "Use 'query' for general knowledge search and "
        "'optimize' for kernel optimization suggestions."
    ),
)


# ---------------------------------------------------------------------------
# Stats logging helpers
# ---------------------------------------------------------------------------

def _format_preview(content: str, max_len: int = 200) -> str:
    """Single-line truncated preview of content."""
    preview = ' '.join(content.replace('\n', ' ').split())
    if len(preview) > max_len:
        preview = preview[:max_len] + "..."
    return preview


def _log_results(tool_name: str, query_str: str, results: list[tuple[Any, float, str, float]]) -> None:
    """Log RAG retrieval statistics."""
    embed_count = sum(1 for r in results if r[2] == "embedding")
    bm25_count = sum(1 for r in results if r[2] == "bm25")
    logger.info(
        "[RAG-STATS] %s returned %d results (top_k=%d) | embedding: %d, bm25: %d",
        tool_name, len(results), _top_k, embed_count, bm25_count,
    )
    for i, (doc, score, source, orig_score) in enumerate(results, 1):
        title = doc.metadata.get('section', doc.metadata.get('title', 'Unknown'))[:40]
        content_len = len(doc.page_content)
        layer_info = doc.metadata.get('layer', 'unknown')
        method_tag = "[EMB]" if source == "embedding" else "[BM25]"
        logger.info(
            "  [%d] %s Score=%.4f (orig=%.4f) Layer=%s Len=%d Title=%s",
            i, method_tag, score, orig_score, layer_info, content_len, title,
        )
        logger.debug("      Content: %s", _format_preview(doc.page_content))


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query(
    topic: str,
    layer: str | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Search the GPU/ROCm/HIP knowledge base.

    Args:
        topic:  Search query describing the information you need.
        layer:  Optional layer filter (e.g. "hip", "rocm", "ai_frameworks").
        top_k:  Number of results to return (default from server config).

    Returns:
        Dictionary with 'results' list and retrieval metadata.
    """
    k = top_k or _top_k
    logger.info("[RAG-STATS] Calling query | topic=%r, layer=%r, top_k=%d", topic, layer, k)

    filters: dict[str, Any] = {}
    if layer:
        filters['layers'] = [layer]

    raw_results = _retriever.search(topic, k=k, filters=filters or None)
    _log_results("query", topic, raw_results)

    if not raw_results:
        return {"results": [], "message": f"No results found for topic: {topic}"}

    output = f"Found {len(raw_results)} results for '{topic}':\n\n"
    for i, (doc, score, source, orig_score) in enumerate(raw_results, 1):
        title = doc.metadata.get("section", doc.metadata.get("source", "Unknown")[:50])
        layer_info = doc.metadata.get("layer", "unknown")
        category = doc.metadata.get("category", "unknown")
        tags = _parse_tags(doc.metadata.get("tags", []))

        output += f"## Result {i}: {title}\n\n"
        output += f"**Layer**: {layer_info} | **Category**: {category}\n"
        if tags:
            output += f"**Tags**: {', '.join(tags)}\n"
        output += f"\n{doc.page_content}\n\n"
        if i < len(raw_results):
            output += "---\n\n"

    return {
        "results": output,
        "count": len(raw_results),
        "query": topic,
    }


@mcp.tool()
def optimize(
    code_type: str,
    context: str | None = None,
    gpu_model: str | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Retrieve optimization suggestions for GPU kernels.

    Args:
        code_type:  Type of code/kernel to optimize (e.g. "matrix multiplication", "convolution").
        context:    Additional context about the optimization goal.
        gpu_model:  Target GPU model (e.g. "MI300X", "MI250").
        top_k:      Number of results to return (default from server config).

    Returns:
        Dictionary with optimization suggestions and retrieval metadata.
    """
    k = top_k or _top_k
    parts = [gpu_model or "", code_type, "optimization", context or ""]
    query_str = " ".join(p for p in parts if p).strip()
    if not query_str or query_str == "optimization":
        query_str = "GPU kernel optimization best practices"

    logger.info(
        "[RAG-STATS] Calling optimize | code_type=%r, context=%r, gpu_model=%r, top_k=%d",
        code_type, context, gpu_model, k,
    )

    raw_results = _retriever.search(query_str, k=k)
    _log_results("optimize", query_str, raw_results)

    if not raw_results:
        return {"results": [], "message": f"No optimization suggestions found for: {code_type}"}

    output = f"Optimization suggestions for {code_type}:\n\n"
    for i, (doc, score, source, orig_score) in enumerate(raw_results, 1):
        title = doc.metadata.get("section", doc.metadata.get("source", "Unknown")[:50])
        output += f"## Suggestion {i}: {title}\n\n"
        output += f"{doc.page_content}\n\n"
        if i < len(raw_results):
            output += "---\n\n"

    return {
        "results": output,
        "count": len(raw_results),
        "query": query_str,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run MCP server."""
    logger.info("Starting RAG Knowledge Base MCP Server...")
    logger.info("  Index path: %s", _retriever.index_path)
    logger.info("  embed_top_k=%d, bm25_top_k=%d, enable_reranker=%s",
                _retriever.embed_top_k, _retriever.bm25_top_k, _retriever.enable_reranker)
    mcp.run()


if __name__ == "__main__":
    main()
