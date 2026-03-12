"""File-backed within-session working notebook for GEAK.

The notebook is intentionally local to a single optimization run.  It stores
append-only JSONL events per writer so multiple agents can contribute evidence
without overwriting each other.  A compact summary can then be injected back
into prompts for later rounds or later steps within the same session.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


_OVERALL_SPEEDUP_RE = re.compile(
    r"Overall:\s*([0-9]+(?:\.[0-9]+)?)x\s*"
    r"\(([0-9]+(?:\.[0-9]+)?)\s*ms\s*->\s*([0-9]+(?:\.[0-9]+)?)\s*ms\)",
    re.IGNORECASE,
)
_PER_SHAPE_RE = re.compile(
    r"^\s*(\([^)]*\)):\s*([0-9]+(?:\.[0-9]+)?)x\s*"
    r"\(([0-9]+(?:\.[0-9]+)?)\s*ms\s*->\s*([0-9]+(?:\.[0-9]+)?)\s*ms\)",
    re.IGNORECASE | re.MULTILINE,
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_writer_id(writer_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", writer_id.strip())
    return cleaned.strip("-") or "default"


def parse_speedup_report(text: str) -> dict[str, Any]:
    """Parse the save_and_test speedup summary block.

    Returns a dict containing an optional overall speedup and a per-shape map.
    Missing information is represented with ``None`` / empty dicts.
    """

    report: dict[str, Any] = {
        "overall_speedup": None,
        "baseline_ms": None,
        "candidate_ms": None,
        "per_shape": {},
    }
    if not text:
        return report

    overall = _OVERALL_SPEEDUP_RE.search(text)
    if overall:
        report["overall_speedup"] = float(overall.group(1))
        report["baseline_ms"] = float(overall.group(2))
        report["candidate_ms"] = float(overall.group(3))

    per_shape: dict[str, dict[str, float]] = {}
    for match in _PER_SHAPE_RE.finditer(text):
        per_shape[match.group(1)] = {
            "speedup": float(match.group(2)),
            "baseline_ms": float(match.group(3)),
            "candidate_ms": float(match.group(4)),
        }
    report["per_shape"] = per_shape
    return report


class WorkingNotebook:
    """Append-only within-session notebook shared by a kernel run."""

    def __init__(self, root_dir: str | Path, *, writer_id: str = "default"):
        self.root_dir = Path(root_dir).resolve()
        self.events_dir = self.root_dir / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.writer_id = _sanitize_writer_id(writer_id)
        self.writer_path = self.events_dir / f"{self.writer_id}.jsonl"

    def append_event(self, kind: str, **data: Any) -> None:
        entry = {
            "ts": time.time(),
            "kind": kind,
            **data,
        }
        with open(self.writer_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")

    def record_baseline(
        self,
        *,
        baseline_latency_ms: float | None,
        bottleneck_type: str | None,
        kernel_category: str | None,
    ) -> None:
        self.append_event(
            "baseline",
            baseline_latency_ms=baseline_latency_ms,
            bottleneck_type=bottleneck_type,
            kernel_category=kernel_category,
        )

    def record_attempt(
        self,
        *,
        strategy: str,
        change_category: str | None,
        step: int | None = None,
    ) -> None:
        self.append_event(
            "attempt",
            strategy=strategy,
            change_category=change_category,
            step=step,
        )

    def record_result(
        self,
        *,
        output: str,
        returncode: int,
        strategy: str | None = None,
        change_category: str | None = None,
        tag: str | None = None,
        message: str | None = None,
        step: int | None = None,
    ) -> None:
        parsed = parse_speedup_report(output)
        self.append_event(
            "result",
            strategy=strategy,
            change_category=change_category,
            tag=tag,
            message=message,
            step=step,
            returncode=returncode,
            overall_speedup=parsed.get("overall_speedup"),
            baseline_ms=parsed.get("baseline_ms"),
            candidate_ms=parsed.get("candidate_ms"),
            per_shape=parsed.get("per_shape", {}),
        )

    def record_round_evaluation(
        self,
        *,
        round_num: int,
        best_task: str | None,
        verified_speedup: float | None,
        baseline_ms: float | None,
        candidate_ms: float | None,
        per_shape_speedups: dict[str, Any] | None = None,
    ) -> None:
        self.append_event(
            "round_evaluation",
            round_num=round_num,
            strategy=best_task,
            change_category=None,
            overall_speedup=verified_speedup,
            baseline_ms=baseline_ms,
            candidate_ms=candidate_ms,
            per_shape=per_shape_speedups or {},
        )

    @staticmethod
    def _load_events(root_dir: str | Path | None) -> list[dict[str, Any]]:
        if not root_dir:
            return []
        events_dir = Path(root_dir).resolve() / "events"
        if not events_dir.is_dir():
            return []

        events: list[dict[str, Any]] = []
        for path in sorted(events_dir.glob("*.jsonl")):
            try:
                with open(path, encoding="utf-8") as handle:
                    for raw in handle:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            events.append(json.loads(raw))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue
        return sorted(events, key=lambda item: _safe_float(item.get("ts")) or 0.0)

    @classmethod
    def summarize_dir(cls, root_dir: str | Path | None) -> str:
        events = cls._load_events(root_dir)
        if not events:
            return ""

        baseline_latency_ms: float | None = None
        bottleneck_type = ""
        kernel_category = ""
        best_event: dict[str, Any] | None = None
        best_speedup = 1.0
        tried: dict[str, dict[str, Any]] = {}
        recent_lines: list[str] = []
        dead_end_lines: list[str] = []
        best_shape: dict[str, tuple[float, str]] = {}
        worst_shape: dict[str, tuple[float, str]] = {}

        for event in events:
            kind = str(event.get("kind", ""))
            strategy = str(event.get("strategy") or "").strip()
            category = str(event.get("change_category") or "").strip()

            if kind == "baseline":
                baseline_latency_ms = _safe_float(event.get("baseline_latency_ms")) or baseline_latency_ms
                bottleneck_type = str(event.get("bottleneck_type") or bottleneck_type)
                kernel_category = str(event.get("kernel_category") or kernel_category)

            if strategy:
                info = tried.setdefault(
                    strategy,
                    {
                        "attempts": 0,
                        "category": category,
                        "best_speedup": 0.0,
                        "last_status": "",
                    },
                )
                if kind == "attempt":
                    info["attempts"] += 1
                if category and not info.get("category"):
                    info["category"] = category

            overall_speedup = _safe_float(event.get("overall_speedup"))
            if overall_speedup is not None and strategy:
                info = tried.setdefault(
                    strategy,
                    {"attempts": 0, "category": category, "best_speedup": 0.0, "last_status": ""},
                )
                info["best_speedup"] = max(float(info["best_speedup"]), overall_speedup)
                info["last_status"] = "win" if overall_speedup > 1.0 else "no-improvement"

            if overall_speedup is not None and overall_speedup > best_speedup:
                best_speedup = overall_speedup
                best_event = event

            if kind in ("result", "round_evaluation") and overall_speedup is not None:
                label = strategy or kind
                baseline_ms = _safe_float(event.get("baseline_ms"))
                candidate_ms = _safe_float(event.get("candidate_ms"))
                if baseline_ms is not None and candidate_ms is not None:
                    recent_lines.append(
                        f"{label}: {overall_speedup:.4f}x ({baseline_ms:.4f} ms -> {candidate_ms:.4f} ms)"
                    )
                else:
                    recent_lines.append(f"{label}: {overall_speedup:.4f}x")
                if overall_speedup <= 1.0:
                    suffix = f" [{category}]" if category else ""
                    dead_end_lines.append(f"{label}{suffix}: {overall_speedup:.4f}x")

            per_shape = event.get("per_shape") or {}
            if isinstance(per_shape, dict):
                for shape, metrics in per_shape.items():
                    if not isinstance(metrics, dict):
                        continue
                    shape_speedup = _safe_float(metrics.get("speedup"))
                    if shape_speedup is None:
                        continue
                    label = strategy or kind or "unknown"
                    if shape not in best_shape or shape_speedup > best_shape[shape][0]:
                        best_shape[shape] = (shape_speedup, label)
                    if shape not in worst_shape or shape_speedup < worst_shape[shape][0]:
                        worst_shape[shape] = (shape_speedup, label)

        lines = ["--- Within-Session Working Notebook ---"]
        head: list[str] = []
        if kernel_category:
            head.append(f"kernel={kernel_category}")
        if bottleneck_type:
            head.append(f"bottleneck={bottleneck_type}")
        if baseline_latency_ms:
            head.append(f"baseline={baseline_latency_ms:.4f}ms")
        if head:
            lines.append("Baseline: " + " | ".join(head))

        if best_event is not None:
            best_strategy = str(best_event.get("strategy") or best_event.get("kind") or "unknown")
            best_category = str(best_event.get("change_category") or "").strip()
            category_suffix = f" [{best_category}]" if best_category else ""
            lines.append(f"Best so far: {best_speedup:.4f}x via {best_strategy}{category_suffix}")

        if tried:
            ranked = sorted(
                tried.items(),
                key=lambda item: (-float(item[1].get("best_speedup", 0.0)), -int(item[1].get("attempts", 0))),
            )[:5]
            tried_bits = []
            for strategy, info in ranked:
                category = f"/{info['category']}" if info.get("category") else ""
                best = float(info.get("best_speedup", 0.0))
                attempts = int(info.get("attempts", 0))
                if best > 0:
                    tried_bits.append(f"{strategy}{category} ({attempts} try, best {best:.4f}x)")
                else:
                    tried_bits.append(f"{strategy}{category} ({attempts} try)")
            if tried_bits:
                lines.append("Tried families: " + "; ".join(tried_bits))

        if dead_end_lines:
            uniq_dead = []
            seen_dead = set()
            for item in reversed(dead_end_lines):
                if item in seen_dead:
                    continue
                seen_dead.add(item)
                uniq_dead.append(item)
                if len(uniq_dead) >= 4:
                    break
            lines.append("Dead ends: " + "; ".join(reversed(uniq_dead)))

        shape_lines: list[str] = []
        for shape, (speedup, label) in list(sorted(best_shape.items(), key=lambda item: -item[1][0]))[:2]:
            shape_lines.append(f"{shape} best {speedup:.4f}x via {label}")
        shape_regressions = [
            f"{shape} weak {speedup:.4f}x via {label}"
            for shape, (speedup, label) in sorted(worst_shape.items(), key=lambda item: item[1][0])[:2]
            if speedup < 1.0
        ]
        if shape_lines or shape_regressions:
            lines.append("Per-shape notes: " + "; ".join(shape_lines + shape_regressions))

        if recent_lines:
            lines.append("Recent evidence: " + "; ".join(recent_lines[-3:]))

        lines.append("---")
        return "\n".join(lines)


def summarize_working_notebook(root_dir: str | Path | None) -> str:
    """Public helper for prompt injection from a notebook directory."""
    return WorkingNotebook.summarize_dir(root_dir)
