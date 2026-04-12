"""Format retrieved experiences into a landscape-oriented context block.

Guides exploration budget allocation, not specific actions.
Includes: strategy landscape, dead-end warnings, trajectory sketch,
exploration nudges.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from minisweagent.memory.cross_session.schemas import ExperienceRecord, StrategySkill

_CATEGORY_LABELS = {
    "algorithmic": "Algorithmic rewrites",
    "fusion": "Operation fusion",
    "tuning": "Parameter tuning",
    "wrapper": "Wrapper / dispatch-path",
}


def format_landscape_context(
    experiences: list[ExperienceRecord],
    skills: list[StrategySkill],
    query_category: str = "",
    query_bottleneck: str = "",
) -> str:
    """Format into a ~600-800 token landscape block for prompt injection."""
    if not experiences and not skills:
        return ""

    parts: list[str] = []
    n = len(experiences)
    parts.append(f"### Cross-Session Memory (from {n} similar kernel{'s' if n != 1 else ''})")

    landscape = _build_strategy_landscape(experiences)
    if landscape:
        bn_label = f" ({query_bottleneck}-bound)" if query_bottleneck else ""
        cat_label = query_category or "similar"
        parts.append(f"**Strategy landscape for {cat_label} kernels{bn_label}:**")
        for cat_key in ("algorithmic", "fusion", "tuning", "wrapper"):
            stats = landscape.get(cat_key)
            if not stats:
                continue
            label = _CATEGORY_LABELS.get(cat_key, cat_key)
            attempts = stats["attempts"]
            improved = stats["improved"]
            avg_sp = stats["avg_speedup"]
            parts.append(f"  {label}: {attempts} attempts, {improved} improved, avg {avg_sp:.2f}x")

        best_cat = max(landscape.items(), key=lambda x: x[1]["improved"] / max(x[1]["attempts"], 1))
        worst_cats = [k for k, v in landscape.items() if v["attempts"] >= 3 and v["improved"] == 0]

        hint_parts = []
        if best_cat[1]["improved"] > 0:
            hint_parts.append(f"{_CATEGORY_LABELS.get(best_cat[0], best_cat[0])} has the best track record")
        if worst_cats:
            names = ", ".join(_CATEGORY_LABELS.get(c, c) for c in worst_cats)
            hint_parts.append(f"{names} rarely helps alone")
        if hint_parts:
            parts.append(f"  -> Budget hint: {'; '.join(hint_parts)}.")

    dead_ends = _collect_dead_ends(experiences, skills)
    if dead_ends:
        parts.append("**Dead ends to avoid:**")
        for de in dead_ends[:4]:
            parts.append(f"  - {de}")

    sketch = _best_trajectory_sketch(experiences)
    if sketch:
        parts.append("**Similar kernel trajectory (reference, not prescription):**")
        parts.append(f"  {sketch}")

    nudges = _exploration_nudges(landscape, experiences)
    if nudges:
        parts.append("**Under-explored for this profile:**")
        for nudge in nudges[:2]:
            parts.append(f"  - {nudge}")

    # Code insights: show actual code changes that produced speedups
    code_section = _best_code_insights(experiences)
    if code_section:
        parts.append(code_section)

    return "\n".join(parts)


def _build_strategy_landscape(experiences: list[ExperienceRecord]) -> dict[str, dict[str, Any]]:
    """Group experiences by change_category and compute stats."""
    landscape: dict[str, dict[str, Any]] = {}

    for exp in experiences:
        cat = exp.best_change_category or "wrapper"
        if cat not in landscape:
            landscape[cat] = {"attempts": 0, "improved": 0, "speedups": []}
        landscape[cat]["attempts"] += 1
        if exp.success and exp.best_speedup > 1.01:
            landscape[cat]["improved"] += 1
            landscape[cat]["speedups"].append(exp.best_speedup)

        for wf in exp.what_failed:
            fail_cat = _infer_category_from_text(wf)
            if fail_cat and fail_cat != cat:
                if fail_cat not in landscape:
                    landscape[fail_cat] = {"attempts": 0, "improved": 0, "speedups": []}
                landscape[fail_cat]["attempts"] += 1

    for stats in landscape.values():
        sps = stats["speedups"]
        stats["avg_speedup"] = sum(sps) / len(sps) if sps else 1.0
        del stats["speedups"]

    return landscape


def _collect_dead_ends(
    experiences: list[ExperienceRecord],
    skills: list[StrategySkill],
) -> list[str]:
    """Collect high-confidence dead-end warnings."""
    dead: list[str] = []

    de_counts: dict[str, int] = defaultdict(int)
    for exp in experiences:
        for de in exp.dead_ends:
            de_counts[de] += 1
    for de_text, cnt in sorted(de_counts.items(), key=lambda x: -x[1]):
        if cnt >= 2:
            dead.append(f"{de_text} (seen in {cnt} kernels)")

    for skill in skills:
        for ci in skill.contraindications:
            dead.append(f"{ci} (from {skill.evidence_count} experiences)")

    seen: set[str] = set()
    unique: list[str] = []
    for d in dead:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _best_trajectory_sketch(experiences: list[ExperienceRecord]) -> str:
    """Pick the most informative trajectory sketch from the best experience."""
    best = None
    best_speedup = 0.0
    for exp in experiences:
        if exp.trajectory_sketch and exp.best_speedup > best_speedup:
            best = exp
            best_speedup = exp.best_speedup

    if not best:
        return ""

    bn = best.bottleneck_type
    bl = f"{best.baseline_latency_ms:.3f}ms" if best.baseline_latency_ms > 0 else "?"
    return (
        f"{best.kernel_name} ({bn}-bound, baseline {bl}): "
        f"{best.trajectory_sketch} -> {best.best_speedup:.2f}x"
    )


def _exploration_nudges(
    landscape: dict[str, dict[str, Any]],
    experiences: list[ExperienceRecord],
) -> list[str]:
    """Suggest under-explored strategy categories."""
    nudges: list[str] = []
    all_cats = {"algorithmic", "fusion", "tuning"}

    explored = {cat for cat, stats in landscape.items() if stats["attempts"] >= 2}
    unexplored = all_cats - explored

    high_impact_cats = sorted(
        [(cat, stats) for cat, stats in landscape.items() if stats["improved"] > 0],
        key=lambda x: -x[1]["improved"],
    )

    for cat in unexplored:
        label = _CATEGORY_LABELS.get(cat, cat)
        if high_impact_cats and high_impact_cats[0][0] == cat:
            nudges.append(f"{label} is under-explored but has the highest success rate elsewhere.")
        else:
            nudges.append(f"{label} has few data points for this profile; consider exploring.")

    return nudges


def _best_code_insights(experiences: list[ExperienceRecord]) -> str:
    """Extract reusable code patterns from top experiences as actionable templates."""
    best = None
    best_speedup = 0.0
    for exp in experiences:
        if exp.success and exp.best_speedup > best_speedup and (exp.patch_content or exp.code_changes_summary):
            best = exp
            best_speedup = exp.best_speedup

    if not best:
        return ""

    parts = [f"**Proven code patterns from {best.kernel_name} ({best.best_speedup:.2f}x speedup):**"]

    if best.code_changes_summary:
        parts.append(f"  Summary: {best.code_changes_summary[:300]}")

    if best.patch_content:
        templates = _extract_code_templates(best.patch_content)
        if templates:
            parts.append("  **Reusable code templates (copy and adapt):**")
            for name, code in templates[:3]:
                parts.append(f"  *{name}*:")
                for line in code.splitlines()[:8]:
                    parts.append(f"    {line}")

    return "\n".join(parts)


def _extract_code_templates(patch_content: str) -> list[tuple[str, str]]:
    """Extract self-contained code blocks from a patch as reusable templates."""
    templates: list[tuple[str, str]] = []

    added_lines = []
    for line in patch_content.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

    current_block: list[str] = []
    current_name = ""

    for line in added_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            if current_block and current_name:
                templates.append((current_name, "\n".join(current_block)))
                current_block = []
                current_name = ""
            continue

        if stripped.startswith("def ") or stripped.startswith("class "):
            if current_block and current_name:
                templates.append((current_name, "\n".join(current_block)))
            current_name = stripped.split("(")[0].replace("def ", "").replace("class ", "")
            current_block = [line]
        elif (stripped.endswith("= {}") or stripped.endswith("= []")) and "=" in stripped:
            var_name = stripped.split("=")[0].strip()
            if current_block and current_name:
                templates.append((current_name, "\n".join(current_block)))
            current_name = f"Cache: {var_name}"
            current_block = [line]
        elif current_block:
            current_block.append(line)

    if current_block and current_name:
        templates.append((current_name, "\n".join(current_block)))

    return templates


def _infer_category_from_text(text: str) -> str:
    """Best-effort classification of a what_worked/what_failed entry."""
    t = text.lower()
    if any(k in t for k in ("algorithm", "rewrite", "restructur")):
        return "algorithmic"
    if any(k in t for k in ("fuse", "fusion", "merge")):
        return "fusion"
    if any(k in t for k in ("tune", "block", "warp", "tile", "autotune")):
        return "tuning"
    return ""
