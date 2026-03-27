"""Scan prior-round results and task directories.

Shared utility used by both the orchestrator's ``tool_collect_results``
and the task generator's ``_run_task_agent`` to summarize what happened
in previous rounds.
"""

from __future__ import annotations

import json
import re as _re
from pathlib import Path

_LATENCY_RE = _re.compile(r"GEAK_RESULT_LATENCY_MS=([\d.]+(?:e[+-]?\d+)?)")
_SPEEDUP_RE = _re.compile(r"GEAK_RESULT_SPEEDUP=([\d.]+(?:e[+-]?\d+)?)")
_GEOMEAN_SPEEDUP_RE = _re.compile(r"GEAK_RESULT_GEOMEAN_SPEEDUP=([\d.]+(?:e[+-]?\d+)?)")
_CORRECTNESS_RE = _re.compile(r"(ALL\s+PASS|CORRECTNESS\s+TEST\s+PASSED|Status:\s*ALL\s+PASS)", _re.IGNORECASE)
_CORRECTNESS_FAIL_RE = _re.compile(r"(FAIL|CORRECTNESS\s+TEST\s+FAILED|Status:\s*FAIL)", _re.IGNORECASE)


def _extract_metrics(content: str) -> dict[str, str]:
    """Extract structured metrics from test output content."""
    metrics: dict[str, str] = {}

    m = _GEOMEAN_SPEEDUP_RE.search(content) or _SPEEDUP_RE.search(content)
    if m:
        metrics["speedup"] = f"{float(m.group(1)):.4f}x"

    m = _LATENCY_RE.search(content)
    if m:
        metrics["latency_ms"] = m.group(1)

    if _CORRECTNESS_RE.search(content):
        metrics["correctness"] = "PASS"
    elif _CORRECTNESS_FAIL_RE.search(content):
        metrics["correctness"] = "FAIL"

    return metrics


def scan_single_round_results(results_dir: Path) -> list[str]:
    """Scan a single round's results directory and return Markdown sections.

    For each task directory:
    - Reads ``best_results.json`` for the authoritative best-patch selection
    - Scans ALL test outputs for structured GEAK_RESULT markers
    - Includes full content of the best patch's test output
    - Reports log status (errors/completed)
    """
    sections: list[str] = []
    task_dirs = sorted(
        d for d in results_dir.iterdir() if d.is_dir() and d.name not in ("worktrees",) and not d.name.startswith(".")
    )
    if not task_dirs:
        return sections

    for td in task_dirs:
        label = td.name
        patches = sorted(td.glob("patch_*.patch"))
        test_outputs = sorted(td.glob("patch_*_test.txt"))
        log_files = sorted(td.glob("*.log"))

        section = [f"### {label}"]
        section.append(f"- Patches produced: {len(patches)}")

        best_patch_id = None
        best_results_path = td / "best_results.json"
        if best_results_path.exists():
            try:
                br = json.loads(best_results_path.read_text())
                best_patch_id = br.get("best_patch_id")
                speedup = br.get("best_patch_speedup")
                baseline_ms = br.get("baseline_latency_ms")
                candidate_ms = br.get("candidate_latency_ms")
                section.append(
                    f"- **Best patch**: {best_patch_id}"
                    f" (speedup={speedup}x, baseline={baseline_ms}ms, candidate={candidate_ms}ms)"
                )
                if br.get("llm_selection_analysis"):
                    section.append(f"- Selection: {br['llm_selection_analysis']}")
            except (json.JSONDecodeError, OSError):
                pass

        for tf in test_outputs:
            try:
                content = tf.read_text(errors="replace")
                metrics = _extract_metrics(content)

                patch_id = tf.stem.replace("_test", "")
                is_best = patch_id == best_patch_id

                if metrics:
                    parts = [f"{k}={v}" for k, v in metrics.items()]
                    marker = " **[BEST]**" if is_best else ""
                    section.append(f"- {tf.name}: {', '.join(parts)}{marker}")
                else:
                    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                    tail = lines[-3:] if len(lines) >= 3 else lines
                    section.append(f"- {tf.name} (tail): {' | '.join(tail)}")

                if is_best:
                    tail_content = content[-3000:] if len(content) > 3000 else content
                    section.append(f"\n<details><summary>{tf.name} (best patch full output)</summary>\n")
                    section.append(f"```\n{tail_content.strip()}\n```\n</details>")
            except Exception:
                section.append(f"- {tf.name}: (unreadable)")

        for lf in log_files[:2]:
            try:
                content = lf.read_text(errors="replace")[-1500:]
                if "ERROR" in content or "Traceback" in content:
                    error_lines = [
                        ln.strip() for ln in content.splitlines() if "error" in ln.lower() or "traceback" in ln.lower()
                    ]
                    section.append(
                        f"- Log ({lf.name}): ERRORS -- {error_lines[-1][:200] if error_lines else 'see log'}"
                    )
                else:
                    section.append(f"- Log ({lf.name}): completed")
            except Exception:
                pass

        sections.append("\n".join(section))

    return sections


def scan_previous_results(results_dir: Path) -> str:
    """Scan previous round results and build a combined summary.

    Accepts either a single round directory (e.g. results/round_1) or
    the parent results/ directory containing multiple round_N subdirs.
    Returns a Markdown summary covering ALL prior rounds.
    """
    sections: list[str] = []

    round_subdirs = (
        sorted(d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("round_"))
        if results_dir.is_dir()
        else []
    )

    if round_subdirs:
        for rd in round_subdirs:
            round_sections = scan_single_round_results(rd)
            if round_sections:
                sections.append(f"## {rd.name.replace('_', ' ').title()} Results\n")
                sections.extend(round_sections)
    else:
        single_sections = scan_single_round_results(results_dir)
        if single_sections:
            sections.append("## Previous Round Results\n")
            sections.extend(single_sections)

    if not sections:
        return ""

    return "\n\n".join(sections) + "\n"


def scan_previous_tasks(tasks_dir: Path, current_round: int) -> str:
    """Scan prior rounds' task directories and summarize what was planned.

    Reads YAML frontmatter (label, agent_type, priority) and the first
    ~200 chars of the task body from each .md task file under
    tasks/round_1 through tasks/round_{current_round - 1}.
    """
    sections: list[str] = []
    for r in range(1, current_round):
        round_dir = tasks_dir / f"round_{r}"
        if not round_dir.is_dir():
            continue
        task_files = sorted(round_dir.glob("*.md"))
        if not task_files:
            continue
        round_items: list[str] = []
        for tf in task_files:
            try:
                text = tf.read_text(errors="replace")
                parts = _re.split(r"^---\s*$", text, maxsplit=2, flags=_re.MULTILINE)
                if len(parts) >= 3:
                    import yaml as _yaml

                    fm = _yaml.safe_load(parts[1]) or {}
                    body_preview = parts[2].strip()[:200]
                else:
                    fm = {}
                    body_preview = text.strip()[:200]
                label = fm.get("label", tf.stem)
                agent_type = fm.get("agent_type", "unknown")
                priority = fm.get("priority", "?")
                round_items.append(
                    f"- **{label}** (agent={agent_type}, priority={priority}): "
                    f"{body_preview}{'...' if len(body_preview) >= 200 else ''}"
                )
            except Exception:
                round_items.append(f"- {tf.name}: (unreadable)")

        if round_items:
            sections.append(f"## Round {r} Planned Tasks\n\n" + "\n".join(round_items))

    if not sections:
        return ""

    return "\n\n".join(sections) + "\n"
