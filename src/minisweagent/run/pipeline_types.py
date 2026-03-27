"""Typed dataclasses for the 3 module boundaries in the GEAK pipeline.

Boundary 1: preprocess -> orchestration  (PreprocessContext)
Boundary 2: postprocess -> orchestration (RoundEvaluation + FullBenchmarkResult)
Boundary 3: postprocess -> output        (FinalReport)

Each dataclass is serialized to a JSON manifest on disk:
- preprocess_context.json  (written by geak-preprocess)
- round_N_evaluation.json  (written by postprocess after each round)
- final_report.json        (written by postprocess at the end)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    """Remove keys with None values for clean JSON output."""
    return {k: v for k, v in d.items() if v is not None}


# ── Boundary 1: preprocess -> orchestration ───────────────────────────


@dataclass
class PreprocessContext:
    """Handover from preprocessing to orchestration.

    All path fields are required.  If the preprocessor cannot produce an
    artifact, it should fail early rather than pass None downstream.

    Serialized to ``preprocess_context.json`` by the preprocessor.
    """

    kernel_path: str
    repo_root: str
    harness_path: str
    preprocess_dir: str
    commandment_path: str
    codebase_context_path: str
    baseline_metrics_path: str
    profiling_result_path: str
    discovery: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return _strip_none({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreprocessContext:
        return cls(
            kernel_path=d.get("kernel_path", ""),
            repo_root=d.get("repo_root", ""),
            harness_path=d.get("harness_path", ""),
            preprocess_dir=d.get("preprocess_dir", ""),
            commandment_path=d.get("commandment_path", ""),
            codebase_context_path=d.get("codebase_context_path", ""),
            baseline_metrics_path=d.get("baseline_metrics_path", ""),
            profiling_result_path=d.get("profiling_result_path", ""),
            discovery=d.get("discovery"),
        )


# ── Boundary 2: postprocess -> orchestration ──────────────────────────


@dataclass
class FullBenchmarkResult:
    """Result of running FULL_BENCHMARK on a candidate patch.

    ``failure_reason`` is ``None`` on success.  On failure it holds a
    short explanation (config mismatch, crash, etc.).  Detailed output
    is written to files on disk, not stored here.
    """

    verified_speedup: float | None = None
    baseline_ms: float | None = None
    candidate_ms: float | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _strip_none({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FullBenchmarkResult:
        if not isinstance(d, dict):
            return cls()
        return cls(
            verified_speedup=d.get("verified_speedup"),
            baseline_ms=d.get("baseline_ms"),
            candidate_ms=d.get("candidate_ms"),
            failure_reason=d.get("failure_reason"),
        )


@dataclass
class RoundEvaluation:
    """Per-round evaluation result returned by postprocess.

    Serialized to ``round_N_evaluation.json`` after each round.
    """

    round: int = 0
    best_patch: str = ""
    best_task: str = ""
    benchmark_speedup: float = 1.0
    full_benchmark: FullBenchmarkResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                continue
            if isinstance(val, FullBenchmarkResult):
                d[f.name] = val.to_dict()
            else:
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoundEvaluation:
        if not isinstance(d, dict):
            return cls()
        fb_raw = d.get("full_benchmark")
        fb = FullBenchmarkResult.from_dict(fb_raw) if isinstance(fb_raw, dict) else None
        return cls(
            round=int(d.get("round", 0)),
            best_patch=str(d.get("best_patch", "")),
            best_task=str(d.get("best_task", "")),
            benchmark_speedup=float(d.get("benchmark_speedup", 1.0)),
            full_benchmark=fb,
        )


# ── Boundary 3: postprocess -> output ─────────────────────────────────


@dataclass
class FinalReport:
    """Final optimization report written at the end of a run.

    Serialized to ``final_report.json``.
    """

    status: str = "complete"
    summary: str = ""
    best_patch: str | None = None
    best_round: str | None = None
    best_task: str | None = None
    best_speedup: float | None = None
    verified_speedup_unclamped: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _strip_none({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FinalReport:
        if not isinstance(d, dict):
            return cls()
        return cls(
            status=str(d.get("status", "complete")),
            summary=str(d.get("summary", "")),
            best_patch=d.get("best_patch"),
            best_round=d.get("best_round"),
            best_task=d.get("best_task"),
            best_speedup=d.get("best_speedup"),
            verified_speedup_unclamped=d.get("verified_speedup_unclamped", d.get("verified_speedup_raw")),
        )
