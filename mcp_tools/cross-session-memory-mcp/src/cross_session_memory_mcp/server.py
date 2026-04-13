"""Cross-session memory MCP server for GEAK.

Exposes the knowledge base (knowledge_base.json) as searchable MCP tools
so optimization agents can query past experiences during kernel optimization.

Tools:
  search_memory   -- find relevant experiences by category/bottleneck/keyword
  get_code_patterns -- get proven code templates for a kernel type
  record_result   -- store a new optimization result (if speedup >= threshold)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="cross-session-memory",
    instructions=(
        "Cross-session memory for kernel optimization. "
        "Use search_memory to find strategies that worked on similar kernels. "
        "Use get_code_patterns to get proven code templates you can adapt. "
        "Use record_result after optimization to store your result for future runs."
    ),
)

_KB_PATH = os.environ.get(
    "GEAK_KNOWLEDGE_BASE_PATH",
    str(Path(__file__).parent.parent.parent.parent / "src" / "minisweagent" / "memory" / "cross_session" / "knowledge_base.json"),
)


def _load_kb() -> list[dict[str, Any]]:
    path = Path(_KB_PATH)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data.get("experiences", [])
    except Exception:
        return []


def _save_kb(experiences: list[dict[str, Any]]) -> None:
    path = Path(_KB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "description": "GEAK cross-session memory knowledge base",
        "experience_count": len(experiences),
        "experiences": experiences,
    }
    path.write_text(json.dumps(data, indent=2, default=str))


@mcp.tool()
def search_memory(
    kernel_category: str = "",
    bottleneck_type: str = "",
    keyword: str = "",
    min_speedup: float = 1.0,
) -> dict[str, Any]:
    """Search past kernel optimization experiences.

    Args:
        kernel_category: Filter by category (normalization, gemm, ffn, attention, moe).
        bottleneck_type: Filter by bottleneck (memory, compute, latency, balanced).
        keyword: Free-text search across strategies, insights, and code changes.
        min_speedup: Minimum speedup to include (default 1.0 = all).

    Returns:
        Matching experiences with strategies, speedups, and code patterns.
    """
    experiences = _load_kb()
    results = []

    for exp in experiences:
        if kernel_category and exp.get("kernel_category", "") != kernel_category:
            continue
        if bottleneck_type and exp.get("bottleneck_type", "") != bottleneck_type:
            continue
        if exp.get("best_speedup", 0) < min_speedup:
            continue
        if keyword:
            searchable = " ".join([
                exp.get("key_insight", ""),
                exp.get("code_changes_summary", ""),
                exp.get("best_strategy", ""),
                " ".join(exp.get("what_worked", [])),
                exp.get("trajectory_sketch", ""),
            ]).lower()
            if keyword.lower() not in searchable:
                continue

        results.append({
            "kernel_category": exp.get("kernel_category"),
            "bottleneck_type": exp.get("bottleneck_type"),
            "best_speedup": exp.get("best_speedup"),
            "best_strategy": exp.get("best_strategy", "")[:200],
            "what_worked": exp.get("what_worked", [])[:10],
            "what_failed": exp.get("what_failed", [])[:5],
            "dead_ends": exp.get("dead_ends", []),
            "code_changes_summary": exp.get("code_changes_summary", "")[:300],
            "trajectory_sketch": exp.get("trajectory_sketch", "")[:300],
        })

    return {
        "count": len(results),
        "total_in_kb": len(experiences),
        "results": results,
    }


@mcp.tool()
def get_code_patterns(
    kernel_category: str = "",
    bottleneck_type: str = "",
    top_k: int = 3,
) -> dict[str, Any]:
    """Get proven code patterns from past optimizations that you can copy and adapt.

    Args:
        kernel_category: Filter by category (normalization, gemm, ffn).
        bottleneck_type: Filter by bottleneck (memory, compute, latency).
        top_k: Maximum number of patterns to return.

    Returns:
        Reusable code templates extracted from successful patches.
    """
    experiences = _load_kb()
    patterns = []

    sorted_exp = sorted(
        [e for e in experiences if e.get("best_speedup", 0) > 1.05],
        key=lambda e: -e.get("best_speedup", 0),
    )

    for exp in sorted_exp:
        if kernel_category and exp.get("kernel_category", "") != kernel_category:
            continue
        if bottleneck_type and exp.get("bottleneck_type", "") != bottleneck_type:
            continue

        patch = exp.get("patch_content", "")
        if not patch:
            continue

        added_blocks = _extract_added_blocks(patch)
        if added_blocks:
            patterns.append({
                "source_kernel": exp.get("kernel_category"),
                "speedup": exp.get("best_speedup"),
                "summary": exp.get("code_changes_summary", "")[:200],
                "code_blocks": added_blocks[:5],
            })

        if len(patterns) >= top_k:
            break

    return {"count": len(patterns), "patterns": patterns}


def _extract_added_blocks(patch: str) -> list[dict[str, str]]:
    """Extract named code blocks from patch + lines."""
    blocks: list[dict[str, str]] = []
    current_lines: list[str] = []
    current_name = ""

    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            if current_lines and current_name:
                blocks.append({"name": current_name, "code": "\n".join(current_lines)})
                current_lines = []
                current_name = ""
            continue

        content = line[1:]
        stripped = content.strip()

        if not stripped or stripped.startswith("#"):
            if current_lines and current_name:
                blocks.append({"name": current_name, "code": "\n".join(current_lines)})
                current_lines = []
                current_name = ""
            continue

        if stripped.startswith("def ") or stripped.startswith("class "):
            if current_lines and current_name:
                blocks.append({"name": current_name, "code": "\n".join(current_lines)})
            current_name = stripped.split("(")[0].replace("def ", "").replace("class ", "")
            current_lines = [content]
        elif re.match(r"^_\w+\s*=\s*\{\}", stripped):
            if current_lines and current_name:
                blocks.append({"name": current_name, "code": "\n".join(current_lines)})
            current_name = f"cache_{stripped.split('=')[0].strip()}"
            current_lines = [content]
        elif current_lines:
            current_lines.append(content)

    if current_lines and current_name:
        blocks.append({"name": current_name, "code": "\n".join(current_lines)})

    return blocks


@mcp.tool()
def record_result(
    kernel_category: str,
    bottleneck_type: str,
    best_speedup: float,
    strategy_name: str = "",
    what_worked: str = "",
    what_failed: str = "",
    code_changes_summary: str = "",
    patch_content: str = "",
) -> dict[str, Any]:
    """Record a new optimization result to the knowledge base.

    Only stores results with speedup >= 1.10x (configurable via GEAK_MEMORY_MIN_SPEEDUP).

    Args:
        kernel_category: Category (normalization, gemm, ffn, attention, moe).
        bottleneck_type: Bottleneck type (memory, compute, latency, balanced).
        best_speedup: Best verified speedup achieved.
        strategy_name: Name of the winning strategy.
        what_worked: Comma-separated list of what worked.
        what_failed: Comma-separated list of what failed.
        code_changes_summary: Summary of code changes.
        patch_content: The actual patch diff (optional, can be large).

    Returns:
        Confirmation of storage or rejection reason.
    """
    min_speedup = float(os.environ.get("GEAK_MEMORY_MIN_SPEEDUP", "1.10"))
    if best_speedup < min_speedup:
        return {
            "stored": False,
            "reason": f"Speedup {best_speedup:.3f}x below threshold {min_speedup}x",
        }

    import uuid
    from datetime import datetime, timezone

    experience = {
        "record_id": uuid.uuid4().hex[:16],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kernel_category": kernel_category,
        "bottleneck_type": bottleneck_type,
        "best_speedup": best_speedup,
        "success": best_speedup > 1.05,
        "best_strategy": strategy_name[:200],
        "best_change_category": "",
        "what_worked": [w.strip() for w in what_worked.split(",") if w.strip()] if what_worked else [],
        "what_failed": [w.strip() for w in what_failed.split(",") if w.strip()] if what_failed else [],
        "dead_ends": [],
        "key_insight": strategy_name,
        "trajectory_sketch": "",
        "code_changes_summary": code_changes_summary[:500],
        "patch_content": patch_content[:10000],
        "kernel_name": "",
        "kernel_path": "",
        "kernel_language": "triton",
        "repo_url": "",
        "baseline_latency_ms": 0.0,
        "top_kernels": [],
        "hardware": "",
        "profiling_metrics": {},
        "best_latency_ms": 0.0,
        "patch_file": "",
        "final_report_path": "",
        "notebook_dir": "",
    }

    experiences = _load_kb()
    experiences.append(experience)
    _save_kb(experiences)

    return {
        "stored": True,
        "record_id": experience["record_id"],
        "experience_count": len(experiences),
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
