"""Consolidation engine: merge experience episodes into StrategySkill entries.

Runs periodically (every N experiences or on-demand) to distill reusable
optimization knowledge from accumulated experiences.

Inspired by MetaClaw skill evolution and HiMem conflict-aware reconsolidation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from minisweagent.memory.cross_session.backends.base import MemoryBackend
from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill

logger = logging.getLogger(__name__)

_MIN_EVIDENCE = 3


def consolidate(backend: MemoryBackend) -> list[StrategySkill]:
    """Group experiences and produce/update StrategySkill entries.

    Groups by (kernel_category, bottleneck_type, best_change_category).
    For each group with enough evidence, creates a skill with:
      - success rate from the group
      - contraindications from what_failed patterns
      - evidence count and source record IDs
    """
    experiences = backend.list_experiences(limit=10000)
    if not experiences:
        return []

    groups = _group_experiences(experiences)
    skills: list[StrategySkill] = []

    for key, exps in groups.items():
        if len(exps) < _MIN_EVIDENCE:
            continue

        cat, bottleneck, change_cat = key
        skill = _build_skill(cat, bottleneck, change_cat, exps)
        if skill:
            backend.store_skill(skill)
            skills.append(skill)

    logger.info(
        "Consolidation complete: %d groups, %d skills produced from %d experiences",
        len(groups),
        len(skills),
        len(experiences),
    )
    return skills


def _group_experiences(
    experiences: list[ExperienceRecord],
) -> dict[tuple[str, str, str], list[ExperienceRecord]]:
    """Group experiences by (category, bottleneck, change_category)."""
    groups: dict[tuple[str, str, str], list[ExperienceRecord]] = defaultdict(list)
    for exp in experiences:
        cat = exp.kernel_category or "unknown"
        bn = exp.bottleneck_type or "unknown"
        cc = exp.best_change_category or "wrapper"
        groups[(cat, bn, cc)].append(exp)
    return dict(groups)


def _build_skill(
    category: str,
    bottleneck: str,
    change_category: str,
    experiences: list[ExperienceRecord],
) -> StrategySkill | None:
    """Build a StrategySkill from a group of experiences."""
    n = len(experiences)
    successes = [e for e in experiences if e.success and e.best_speedup > 1.01]
    success_rate = len(successes) / n if n > 0 else 0.0

    speedups = [e.best_speedup for e in successes]
    if speedups:
        avg_sp = sum(speedups) / len(speedups)
        min_sp = min(speedups)
        max_sp = max(speedups)
        expected = f"{min_sp:.2f}x-{max_sp:.2f}x (avg {avg_sp:.2f}x)"
    else:
        expected = "no improvement observed"

    languages = sorted(set(e.kernel_language for e in experiences if e.kernel_language != "unknown"))

    title = f"{_label(bottleneck)}-bound {category}: {_change_label(change_category)}"
    description = _build_description(change_category, successes, experiences)
    contras = _extract_contraindications(experiences)
    source_ids = [e.record_id for e in experiences]

    return StrategySkill(
        title=title,
        kernel_categories=[category],
        bottleneck_types=[bottleneck],
        kernel_languages=languages or ["unknown"],
        strategy_description=description,
        change_category=change_category,
        expected_speedup=expected,
        evidence_count=n,
        success_rate=round(success_rate, 3),
        contraindications=contras[:5],
        source_records=source_ids[:50],
    )


def _build_description(
    change_category: str,
    successes: list[ExperienceRecord],
    all_exps: list[ExperienceRecord],
) -> str:
    """Build a concise strategy description from evidence."""
    parts: list[str] = []

    worked_counts: dict[str, int] = defaultdict(int)
    for exp in successes:
        for w in exp.what_worked:
            worked_counts[w] += 1

    top_worked = sorted(worked_counts.items(), key=lambda x: -x[1])[:3]
    if top_worked:
        patterns = "; ".join(f"{w} ({c}x)" for w, c in top_worked)
        parts.append(f"Winning patterns: {patterns}")

    n_total = len(all_exps)
    n_success = len(successes)
    parts.append(f"Track record: {n_success}/{n_total} successful ({100*n_success//n_total}%)")

    best = max(successes, key=lambda e: e.best_speedup) if successes else None
    if best and best.trajectory_sketch:
        parts.append(f"Best trajectory: {best.trajectory_sketch}")

    return ". ".join(parts)


def _extract_contraindications(experiences: list[ExperienceRecord]) -> list[str]:
    """Extract recurring failure patterns as contraindications."""
    fail_counts: dict[str, int] = defaultdict(int)
    for exp in experiences:
        if not exp.success or exp.best_speedup <= 1.01:
            for f in exp.what_failed:
                fail_counts[f] += 1
            for d in exp.dead_ends:
                fail_counts[d] += 1

    contras: list[str] = []
    for pattern, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
        if count >= 2:
            contras.append(f"{pattern} (failed {count}x)")
    return contras


def _label(bottleneck: str) -> str:
    return bottleneck.capitalize() if bottleneck and bottleneck != "unknown" else "General"


def _change_label(change_category: str) -> str:
    labels = {
        "algorithmic": "algorithmic rewrites",
        "fusion": "operation fusion",
        "tuning": "parameter tuning",
        "wrapper": "dispatch/wrapper changes",
    }
    return labels.get(change_category, change_category)


def reflect_on_transfer(
    seed_experience: "ExperienceRecord",
    test_experience: "ExperienceRecord",
    backend: Any,
) -> None:
    """After a cross-kernel transfer experiment, create/update a StrategySkill
    reflecting what transferred and what didn't.

    Called after a test kernel run that had access to seed experiences.
    """
    seed_sp = seed_experience.best_speedup
    test_sp = test_experience.best_speedup

    transferred = test_sp > 1.02

    common_patterns = []
    if seed_experience.what_worked and test_experience.what_worked:
        seed_strats = {w.split(":")[0].strip() for w in seed_experience.what_worked if ":" in w}
        test_strats = {w.split(":")[0].strip() for w in test_experience.what_worked if ":" in w}
        common_patterns = list(seed_strats & test_strats)

    skill = StrategySkill(
        title=f"Transfer: {seed_experience.kernel_category} -> {test_experience.kernel_category}",
        kernel_categories=list({seed_experience.kernel_category, test_experience.kernel_category}),
        bottleneck_types=list({seed_experience.bottleneck_type, test_experience.bottleneck_type}),
        kernel_languages=[seed_experience.kernel_language],
        strategy_description=(
            f"Strategies from {seed_experience.kernel_name} ({seed_sp:.2f}x) "
            f"{'transferred successfully' if transferred else 'did not transfer'} "
            f"to {test_experience.kernel_name} ({test_sp:.2f}x). "
            f"Common patterns: {', '.join(common_patterns) if common_patterns else 'none identified'}."
        ),
        change_category=seed_experience.best_change_category,
        expected_speedup=f"{test_sp:.2f}x" if transferred else "no improvement",
        evidence_count=2,
        success_rate=1.0 if transferred else 0.0,
        contraindications=(
            [f"Does not transfer between {seed_experience.bottleneck_type}-bound and {test_experience.bottleneck_type}-bound kernels"]
            if not transferred and seed_experience.bottleneck_type != test_experience.bottleneck_type
            else []
        ),
        source_records=[seed_experience.record_id, test_experience.record_id],
    )

    try:
        backend.store_skill(skill)
        logger.info(
            "Reflection skill created: %s (transferred=%s, test_speedup=%.2fx)",
            skill.title, transferred, test_sp,
        )
    except Exception as exc:
        logger.warning("Failed to store reflection skill: %s", exc)
