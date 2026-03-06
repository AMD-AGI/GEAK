"""ReMe-inspired Memory: Multi-faceted Distillation + Utility-based Refinement.

Implements the core mechanisms from ReMe (arxiv 2512.10696):
1. Multi-faceted distillation: extract success patterns, failure triggers, comparative insights
2. Context-adaptive reuse: tailor historical insights to new contexts
3. Utility-based refinement: prune low-utility memories, keep compact pool

Reference code: agentscope-ai/ReMe (pip install reme-ai)
We adapt their Task Memory concept for kernel optimization.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from minisweagent import global_config_dir

REME_MEMORY_PATH = global_config_dir / "reme_memory.json"


@dataclass
class DistilledInsight:
    """A multi-faceted insight extracted from an optimization run."""
    id: str
    kernel_category: str
    insight_type: str  # "success_pattern", "failure_trigger", "comparative"
    content: str
    evidence: str = ""
    utility_score: float = 1.0  # starts at 1.0, decays if not useful
    retrieval_count: int = 0
    use_count: int = 0  # times the agent actually used this insight
    created_at: float = field(default_factory=time.time)
    last_retrieved: float = 0.0


class ReMeMemory:
    """ReMe-inspired memory with distillation and utility pruning."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else REME_MEMORY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.insights = [DistilledInsight(**d) for d in data]
        else:
            self.insights = []

    def _save(self):
        self.path.write_text(json.dumps([asdict(i) for i in self.insights], indent=2))

    def add_insight(self, insight: DistilledInsight):
        """Add a new insight, merging with existing if duplicate."""
        for existing in self.insights:
            if existing.content == insight.content and existing.kernel_category == insight.kernel_category:
                existing.utility_score = min(2.0, existing.utility_score + 0.2)
                existing.evidence += f"; {insight.evidence}"
                self._save()
                return
        self.insights.append(insight)
        self._save()

    def retrieve(self, kernel_category: str, top_k: int = 5) -> list[DistilledInsight]:
        """Retrieve relevant insights, sorted by utility score."""
        relevant = [i for i in self.insights
                    if i.kernel_category == kernel_category or i.kernel_category == "all"]
        relevant.sort(key=lambda x: -x.utility_score)
        for i in relevant[:top_k]:
            i.retrieval_count += 1
            i.last_retrieved = time.time()
        self._save()
        return relevant[:top_k]

    def mark_used(self, insight_id: str):
        """Mark an insight as actually used by the agent."""
        for i in self.insights:
            if i.id == insight_id:
                i.use_count += 1
                i.utility_score = min(2.0, i.utility_score + 0.3)
                self._save()
                return

    def prune(self, min_utility: float = 0.3, max_age_days: int = 30):
        """Prune low-utility and stale insights."""
        now = time.time()
        max_age_seconds = max_age_days * 86400
        before = len(self.insights)
        self.insights = [
            i for i in self.insights
            if i.utility_score >= min_utility
            and (now - i.created_at) < max_age_seconds
        ]
        after = len(self.insights)
        if before != after:
            self._save()
        return before - after

    def decay_unused(self, decay_rate: float = 0.05):
        """Decay utility of insights that are retrieved but not used."""
        for i in self.insights:
            if i.retrieval_count > 0 and i.use_count == 0:
                i.utility_score = max(0.1, i.utility_score - decay_rate)
        self._save()

    def format_for_prompt(self, kernel_category: str) -> str:
        """Format retrieved insights for injection into agent prompt."""
        insights = self.retrieve(kernel_category)
        if not insights:
            return ""

        lines = ["\nReMe insights (from past optimization experience):"]
        for i in insights:
            tag = {"success_pattern": "WIN", "failure_trigger": "AVOID", "comparative": "NOTE"}.get(i.insight_type, "?")
            lines.append(f"  [{tag}] {i.content} (utility: {i.utility_score:.1f}, evidence: {i.evidence[:50]})")
        return "\n".join(lines)


def distill_from_log(log_path: str | Path, kernel_category: str) -> list[DistilledInsight]:
    """Extract multi-faceted insights from an agent log (ReMe-style distillation)."""
    path = Path(log_path)
    if not path.exists():
        return []

    text = path.read_text()
    insights = []
    ts = str(int(time.time()))

    # Success patterns
    speedups = re.findall(r'Speedup \(geomean\):\s+(\d+\.\d+)x', text)
    if speedups:
        best = max(float(s) for s in speedups)
        if best > 1.0:
            insights.append(DistilledInsight(
                id=f"success_{kernel_category}_{ts}",
                kernel_category=kernel_category,
                insight_type="success_pattern",
                content=f"Achieved {best:.2f}x geomean speedup on {kernel_category}",
                evidence=f"Best of {len(speedups)} measurements",
            ))

    # Failure triggers
    if "BrokenPipeError" in text:
        insights.append(DistilledInsight(
            id=f"fail_brokenpipe_{kernel_category}_{ts}",
            kernel_category=kernel_category,
            insight_type="failure_trigger",
            content="OpenEvolve BrokenPipeError -- use manual optimization as fallback",
            evidence="OE process crash detected",
        ))

    if "rocBLAS" in text or "F.linear" in text:
        insights.append(DistilledInsight(
            id=f"fail_vendor_blas_{kernel_category}_{ts}",
            kernel_category=kernel_category,
            insight_type="failure_trigger",
            content="Triton kernel competes against vendor BLAS -- cannot win on standard shapes",
            evidence="rocBLAS/F.linear reference detected in log",
            utility_score=1.5,
        ))

    # Comparative insights
    if "COMMANDMENT.md validation: OK" in text:
        insights.append(DistilledInsight(
            id=f"comp_cmd_ok_{kernel_category}_{ts}",
            kernel_category="all",
            insight_type="comparative",
            content="Auto-generated COMMANDMENT.md validated successfully -- use this approach",
            evidence="Validation passed",
        ))

    return insights


def seed_reme_memory(path: str | Path | None = None) -> ReMeMemory:
    """Seed ReMe memory with insights from eval runs."""
    mem = ReMeMemory(path)

    mem.add_insight(DistilledInsight(
        id="gemm_vendor_blas", kernel_category="gemm",
        insight_type="failure_trigger",
        content="Triton GEMM cannot beat vendor BLAS for standard shapes. Focus on fused/quantized operations.",
        evidence="6 runs, max 0.51x", utility_score=1.8,
    ))
    mem.add_insight(DistilledInsight(
        id="rope_half_loading", kernel_category="rope",
        insight_type="success_pattern",
        content="Direct half-loading (x1/x2 separately) reduces instruction count for RoPE. +2-6% speedup.",
        evidence="rope 2.87x->2.94x", utility_score=1.5,
    ))
    mem.add_insight(DistilledInsight(
        id="attention_manual", kernel_category="attention",
        insight_type="failure_trigger",
        content="OpenEvolve produces worse candidates for attention backward. Use manual Triton kernel optimization.",
        evidence="nsa_backward OE=0.76x, manual=1.52x", utility_score=1.3,
    ))
    mem.add_insight(DistilledInsight(
        id="commandment_auto", kernel_category="all",
        insight_type="success_pattern",
        content="Auto-generated COMMANDMENT.md succeeds 100% vs ~60% for manual creation.",
        evidence="All experiments", utility_score=1.6,
    ))
    mem.add_insight(DistilledInsight(
        id="topk_stage_specific", kernel_category="topk",
        insight_type="success_pattern",
        content="TopK needs stage-specific optimization: Stage1=latency-bound, Stage2=LDS-bound.",
        evidence="topk 1.46x->2.07x", utility_score=1.4,
    ))

    return mem
