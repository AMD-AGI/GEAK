"""Render and publish reports from normalized e2e results."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

WORKFLOW_RETURN_FILE = "workflow_return.json"
KERNEL_JOURNEY_FILE = "kernel_journey.json"
FINAL_REPORT_FILE = "final_report.md"
BASELINE_ALIGNMENT_BEGIN = "<!-- GEAK_BASELINE_ALIGNMENT_BEGIN -->"
BASELINE_ALIGNMENT_END = "<!-- GEAK_BASELINE_ALIGNMENT_END -->"


def _existing_report_path(eval_dir: Path) -> str:
    """Absolute path of the run's final report, or "" when none exists yet."""
    try:
        candidate = eval_dir / FINAL_REPORT_FILE
        return str(candidate) if candidate.is_file() else ""
    except OSError:
        return ""


def _format_optional_number(
    value: Any,
    *,
    digits: int = 2,
    suffix: str = "",
) -> str:
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "unavailable"


def _render_baseline_alignment_section(result: dict[str, Any]) -> str:
    """Render a deterministic, Setup-seed-first handoff report section."""
    basis = result.get("baseline_basis") or {}
    alignment = (
        result.get("handoff_alignment") or result.get("baseline_alignment") or {}
    )
    geak_baseline = _format_optional_number(
        basis.get("setup_baseline_tok_s") or basis.get("geak_measured_baseline_tok_s"),
        digits=3,
        suffix=" tok/s",
    )
    same_config_baseline = _format_optional_number(
        basis.get("orchestrator_best_tput_same_config"),
        digits=3,
        suffix=" tok/s",
    )
    same_config_divergence = _format_optional_number(
        basis.get("current_best_same_config_divergence_pct"),
        digits=2,
        suffix="%",
    )
    raw_baseline = _format_optional_number(
        basis.get("orchestrator_baseline_tok_s"), digits=3, suffix=" tok/s"
    )
    raw_divergence = _format_optional_number(
        basis.get("raw_session_baseline_divergence_pct"),
        digits=2,
        suffix="%",
    )
    threshold = _format_optional_number(
        alignment.get("warning_threshold_pct"), digits=1, suffix="%"
    )
    status = str(alignment.get("status") or "unavailable")
    identity = alignment.get("server_identity") or result.get("server_identity") or {}
    identity_status = str(identity.get("status") or "unavailable")
    identity_paths = identity.get("evidence_paths") or []
    stack = result.get("serving_stack") or {}
    launcher = str(stack.get("launcher") or "unknown")
    recipe_aligned = bool(alignment.get("recipe_aligned_with_orchestrator", True))
    # "unavailable" alone reads as "GEAK failed to compute it". Say whose side
    # the gap is on when the orchestrator itself never verified the reference.
    reference_caveat = (
        [
            "",
            (
                "The same-config number above is missing because the upstream "
                "orchestrator shipped it `"
                f"{alignment.get('same_config_reference_status') or 'unverified'}"
                "` — it never re-measured GEAK's seed config, so there is no "
                "reference to diverge from. This is an upstream handoff gap, not "
                "a GEAK measurement failure, and nothing on the GEAK side can "
                "fill it in. Until it is verified, read `hot_geak_speedup` (not "
                "`hot_speedup`) for GEAK's own contribution: `hot_speedup` is "
                "measured against the raw session baseline and therefore still "
                "carries the orchestrator's own accepted-config gain."
            ),
        ]
        if status == "unavailable_reference_unverified"
        else []
    )
    recipe_caveat = (
        []
        if recipe_aligned
        else [
            "",
            (
                "The servers behind these numbers were launched by GEAK's own "
                "backend adapter, not by the upstream orchestrator's launch "
                "script. That script owns the platform kernel preset, "
                "`--trust-remote-code` and the gpu-memory-utilization default, so "
                "the two harnesses did not serve the same stack. Read the "
                "same-config divergence above as a launch-recipe difference, not "
                "as a measurement or hardware signal, and check "
                "`serving_stack.baseline.kernel_picks` for which kernels each "
                "side actually selected."
            ),
        ]
    )
    return "\n".join(
        [
            BASELINE_ALIGNMENT_BEGIN,
            "## Baseline alignment",
            "",
            "Primary same-config comparison (Setup seed):",
            "",
            f"- GEAK measured baseline: {geak_baseline} (Setup seed)",
            (
                "- Upstream current-best baseline on the same config: "
                f"{same_config_baseline}"
            ),
            f"- Same-config divergence: {same_config_divergence}",
            f"- Alignment status: `{status}` (warning threshold: ±{threshold})",
            f"- Observed server identity: `{identity_status}`",
            (
                "- Server identity evidence: "
                + (
                    ", ".join(f"`{path}`" for path in identity_paths)
                    if identity_paths
                    else "unavailable"
                )
            ),
            f"- Server launch recipe: `{launcher}`",
            *reference_caveat,
            *recipe_caveat,
            "",
            "Raw-session audit comparison:",
            "",
            f"- Upstream raw-session baseline: {raw_baseline}",
            f"- Raw-session baseline divergence: {raw_divergence}",
            "",
            (
                "The raw-session baseline predates accepted upstream configuration "
                "changes and is not expected to match the GEAK baseline. Its "
                "divergence includes previously accepted configuration gains and "
                "is not a pure measurement-drift signal."
            ),
            BASELINE_ALIGNMENT_END,
        ]
    )


def _upsert_marked_markdown_section(
    text: str,
    section: str,
    *,
    begin_marker: str,
    end_marker: str,
) -> str:
    """Replace a generated section or insert it after the first H1."""
    begin = text.find(begin_marker)
    end = text.find(end_marker)
    if begin >= 0 and end >= begin:
        end += len(end_marker)
        return text[:begin] + section + text[end:]

    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("# "):
            return (
                "".join(lines[: index + 1])
                + "\n"
                + section
                + "\n\n"
                + "".join(lines[index + 1 :])
            )
    return section + "\n\n" + text


def _update_baseline_alignment_reports(result: dict[str, Any]) -> list[str]:
    """Idempotently add baseline alignment to every existing primary report."""
    section = _render_baseline_alignment_section(result)
    eval_dir = str(result.get("eval_dir") or "").strip()
    if not eval_dir:
        return []
    eval_root = Path(eval_dir).resolve(strict=False)

    candidates: list[Path] = []
    report_path = str(result.get("report_path") or "").strip()
    if report_path:
        report_candidate = Path(report_path)
        if not report_candidate.is_absolute():
            report_candidate = eval_root / report_candidate
        candidates.append(report_candidate)
    candidates.append(eval_root / "final_report.md")

    updated: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        resolved_path = path.resolve(strict=False)
        try:
            resolved_path.relative_to(eval_root)
        except ValueError:
            continue
        key = str(resolved_path)
        if key in seen:
            continue
        seen.add(key)
        if not resolved_path.is_file():
            continue
        original = resolved_path.read_text(encoding="utf-8")
        rendered = _upsert_marked_markdown_section(
            original,
            section,
            begin_marker=BASELINE_ALIGNMENT_BEGIN,
            end_marker=BASELINE_ALIGNMENT_END,
        )
        if rendered != original:
            tmp = resolved_path.with_name(resolved_path.name + ".tmp")
            tmp.write_text(rendered, encoding="utf-8")
            os.replace(tmp, resolved_path)
        updated.append(str(resolved_path))
    return updated



def _md_table(rows: list[tuple[str, Any]]) -> str:
    """Two-column markdown table; ``None``/"" render as an explicit em dash."""
    out = ["| item | value |", "|---|---|"]
    for key, value in rows:
        text = "—" if value is None or value == "" else str(value)
        out.append(f"| {key} | {text} |")
    return "\n".join(out)


def _render_synthesized_final_report(normalized: dict, wf: dict | None) -> str:
    """Render a final report from the normalized result + on-disk recovery.

    Deliberately not an LLM call: this runs on the timeout path where the only
    budget left is the flush grace, so it must never hang. A dump of recovered
    numbers, not an analysis — the banner says so.
    """
    status = str(normalized.get("status") or "unknown")
    speedup = normalized.get("throughput_speedup")
    kernels = normalized.get("accepted_kernels") or []
    heads = normalized.get("accepted_heads") or []
    config = normalized.get("accepted_config") or {}
    evidence = normalized.get("validation_evidence") or {}
    pending = (wf or {}).get("pending_integrations") or []

    parts: list[str] = []
    parts.append("# GEAK e2e — final report (SYNTHESIZED)\n")
    parts.append(
        "> **This report was not written by the Report phase.** The workflow did not reach\n"
        "> Finalize/Report/Validate before its wall-clock budget expired, so `run_e2e.py`\n"
        "> synthesized this file from the artifacts already on disk. Every number below was\n"
        "> really measured, but there was **no independent Director re-measurement**, so treat\n"
        "> the headline as provisional and re-bench before promoting it.\n"
    )
    parts.append("\n## Result\n")
    parts.append(_md_table([
        ("status", status),
        ("result_source", normalized.get("result_source")),
        ("baseline tok/s", normalized.get("baseline_throughput_tok_s")),
        ("final tok/s", normalized.get("final_throughput_tok_s")),
        ("speedup", f"{speedup:.4f}x" if isinstance(speedup, (int, float)) else None),
        ("output parity", normalized.get("output_parity")),
        ("TTFT ms", normalized.get("ttft_ms")),
        ("TPOT ms", normalized.get("tpot_ms")),
        ("validation_status", evidence.get("validation_status")),
        ("speedup basis", evidence.get("speedup_basis")),
        ("bench client", normalized.get("bench_client")),
        ("eval_dir", normalized.get("eval_dir")),
    ]))

    parts.append("\n\n## Accepted work\n")
    if not kernels and not heads:
        parts.append("\nNothing was accepted — the run is a do-no-harm no-gain.\n")
    for label, entries in (("kernels", kernels), ("heads", heads)):
        if not entries:
            continue
        parts.append(f"\n### Accepted {label}\n\n")
        parts.append(_md_table([
            (str(e.get("short_name") or "?"),
             f"{e.get('e2e_delta_pct')}% e2e, backend={e.get('backend') or '?'}, gate={e.get('gate') or '?'}")
            for e in entries if isinstance(e, dict)
        ]))
        parts.append("\n")
    if config.get("flags") or config.get("env"):
        parts.append("\n### Accepted config\n\n")
        parts.append(f"- flags: `{config.get('flags') or ''}`\n")
        parts.append(f"- env: `{config.get('env') or ''}`\n")
        if config.get("env_unparsed"):
            # Not cosmetic: whatever is listed here is NOT in env_map, so a
            # consumer replaying this config will not set it. Naming it is the
            # difference between a lossy replay and an unexplained one.
            parts.append(
                "- env fragments that are not assignments (dropped from "
                f"`env_map`): `{' '.join(config['env_unparsed'])}`\n"
            )

    if pending:
        parts.append("\n## Deferred (verified-isolated, e2e A/B never completed)\n\n")
        parts.append(
            "These are REAL isolated wins whose end-to-end A/B ran out of time. They are not\n"
            "rejections — resume the run against the same eval_dir to finish them.\n\n"
        )
        for entry in pending:
            if isinstance(entry, dict):
                parts.append(
                    f"- `{entry.get('short_name') or '?'}` — "
                    f"{entry.get('isolated') or '?'}x isolated, "
                    f"{entry.get('pct_gpu_time') or '?'}% GPU time\n"
                )

    parts.append("\n## Artifacts\n\n")
    parts.append(f"- `{KERNEL_JOURNEY_FILE}` — per-kernel journey\n")
    parts.append(f"- `{WORKFLOW_RETURN_FILE}` — recovered workflow return\n")
    parts.append("- `result.json` — the full normalized result this report was rendered from\n")
    if normalized.get("final_overlay"):
        parts.append(f"- `{normalized['final_overlay']}` — final overlay\n")
    return "".join(parts)


def _write_final_report_fallback(eval_dir: Path, normalized: dict,
                                 wf: dict | None) -> str:
    """Guarantee a readable final report; return its path ("" on failure).

    Same guaranteed-emit contract as ``result.json`` / ``kernel_journey.json``.
    When the Report phase already wrote one this is a no-op returning the
    existing path — the architect's report is never overwritten.
    """
    existing = _existing_report_path(eval_dir)
    if existing:
        return existing
    path = eval_dir / FINAL_REPORT_FILE
    eval_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_render_synthesized_final_report(normalized, wf), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a kill mid-write never yields a partial file
    return str(path)

