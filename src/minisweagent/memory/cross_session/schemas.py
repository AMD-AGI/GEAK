"""Data schemas for cross-session kernel optimization memory.

Two tracks (MetaClaw-inspired):
  - ExperienceRecord: episodic -- what happened in a specific optimization run
  - StrategySkill: semantic -- consolidated knowledge about what works for a class of kernels
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class ExperienceRecord:
    """Append-only record of a single kernel optimization run."""

    record_id: str = field(default_factory=_new_id)
    timestamp: str = field(default_factory=_now_iso)

    # Kernel identity
    kernel_path: str = ""
    kernel_name: str = ""
    kernel_category: str = "unknown"
    kernel_language: str = "unknown"  # triton, hip, cuda
    repo_url: str = ""

    # Profiling fingerprint
    bottleneck_type: str = "unknown"  # memory, compute, latency, balanced
    baseline_latency_ms: float = 0.0
    top_kernels: list[dict[str, Any]] = field(default_factory=list)
    hardware: str = ""
    profiling_metrics: dict[str, Any] = field(default_factory=dict)

    # Optimization outcome
    best_speedup: float = 1.0
    best_latency_ms: float = 0.0
    success: bool = False
    best_strategy: str = ""
    best_change_category: str = ""  # algorithmic, fusion, tuning, wrapper

    # Extracted insights
    what_worked: list[str] = field(default_factory=list)
    what_failed: list[str] = field(default_factory=list)
    dead_ends: list[str] = field(default_factory=list)
    key_insight: str = ""
    trajectory_sketch: str = ""

    # Code artifacts (the most transferable knowledge)
    patch_content: str = ""  # actual diff/patch text that produced the speedup
    code_changes_summary: str = ""  # human-readable summary of what changed in the code

    # Linkage
    patch_file: str = ""
    final_report_path: str = ""
    notebook_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperienceRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, s: str) -> ExperienceRecord:
        return cls.from_dict(json.loads(s))


@dataclass
class StrategySkill:
    """Consolidated reusable knowledge about what works for a class of kernels."""

    skill_id: str = field(default_factory=_new_id)
    title: str = ""
    kernel_categories: list[str] = field(default_factory=list)
    bottleneck_types: list[str] = field(default_factory=list)
    kernel_languages: list[str] = field(default_factory=list)
    strategy_description: str = ""
    change_category: str = ""  # algorithmic, fusion, tuning, wrapper
    expected_speedup: str = ""
    evidence_count: int = 0
    success_rate: float = 0.0
    contraindications: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=_now_iso)
    source_records: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategySkill:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, s: str) -> StrategySkill:
        return cls.from_dict(json.loads(s))
