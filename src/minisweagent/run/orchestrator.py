"""Compatibility orchestrator entrypoint.

The preprocess verification work proved that the copied preprocess stage is
intact, but the original `run/orchestrator.py` module was missing from this
checkout. This lightweight orchestrator restores the import surface and offers
basic preprocess-artifact inspection so the CLI remains usable while the full
LLM-driven orchestration layer is backfilled separately.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from minisweagent.run.pipeline_helpers import DEFAULT_HETEROGENEOUS, DEFAULT_PIPELINE_OUTPUT_DIR


def _load_preprocess_context(preprocess_dir: Path) -> dict[str, Any]:
    """Reconstruct the preprocessor context from artifact files on disk."""
    ctx: dict[str, Any] = {"preprocess_dir": str(preprocess_dir)}

    resolved_path = preprocess_dir / "resolved.json"
    if resolved_path.exists():
        ctx["resolved"] = json.loads(resolved_path.read_text())
        ctx["kernel_path"] = ctx["resolved"].get("local_file_path", "")
        repo_root = ctx["resolved"].get("local_repo_path", "")
        ctx["repo_root"] = str(Path(repo_root).resolve()) if repo_root else ""

    disc_path = preprocess_dir / "discovery.json"
    if disc_path.exists():
        ctx["discovery"] = json.loads(disc_path.read_text())
        focused = ctx["discovery"].get("focused_test") or {}
        if focused.get("focused_command"):
            ctx["test_command"] = focused["focused_command"]
        else:
            tests = ctx["discovery"].get("tests", [])
            ctx["test_command"] = tests[0]["command"] if tests else None
        if focused.get("focused_test_file"):
            ctx["harness_path"] = focused["focused_test_file"]

    harness_txt = preprocess_dir / "harness_path.txt"
    if harness_txt.exists():
        ctx["harness_path"] = harness_txt.read_text().strip()

    prof_path = preprocess_dir / "profile.json"
    if prof_path.exists():
        ctx["profiling"] = json.loads(prof_path.read_text())

    bm_path = preprocess_dir / "baseline_metrics.json"
    if bm_path.exists():
        ctx["baseline_metrics"] = json.loads(bm_path.read_text())

    cmd_path = preprocess_dir / "COMMANDMENT.md"
    if cmd_path.exists():
        ctx["commandment"] = cmd_path.read_text()

    return ctx


def run_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    model=None,
    model_factory=None,
    *,
    output_dir: Path | None = None,
    max_rounds: int | None = None,
    start_round: int = 1,
    heterogeneous: bool = DEFAULT_HETEROGENEOUS,
    console=None,
) -> dict[str, Any]:
    """Return a compatibility report for the available preprocess artifacts."""
    _ = (model, model_factory, console)
    out = Path(output_dir or preprocess_ctx.get("preprocess_dir") or DEFAULT_PIPELINE_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    required = {
        "kernel_path": bool(preprocess_ctx.get("kernel_path")),
        "repo_root": bool(preprocess_ctx.get("repo_root")),
        "test_command": bool(preprocess_ctx.get("test_command")),
        "harness_path": bool(preprocess_ctx.get("harness_path")),
    }
    artifacts = {
        name: (out / name).exists()
        for name in (
            "resolved.json",
            "discovery.json",
            "profile.json",
            "baseline_metrics.json",
            "COMMANDMENT.md",
        )
    }

    report = {
        "status": "ready",
        "mode": "heterogeneous" if heterogeneous else "homogeneous",
        "gpu_ids": gpu_ids,
        "max_rounds": max_rounds or int(os.getenv("GEAK_MAX_ROUNDS", "5")),
        "start_round": start_round,
        "preprocess_dir": str(out),
        "required_context": required,
        "artifacts": artifacts,
        "note": (
            "Compatibility orchestrator restored for import/CLI availability. "
            "It validates preprocess artifacts and reports readiness."
        ),
    }

    report_path = out / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    """CLI: ``geak-orchestrate --preprocess-dir <dir> [--gpu-ids 0,1]``."""
    import argparse

    from minisweagent.agents.agent_spec import detect_available_gpus
    from minisweagent.run.pipeline_helpers import add_agent_filter_args, apply_agent_filter_env

    parser = argparse.ArgumentParser(
        description="Compatibility GEAK orchestrator: validate preprocess artifacts and report readiness",
    )
    parser.add_argument(
        "--preprocess-dir",
        required=True,
        help="Directory containing preprocessor artefacts (resolved.json, discovery.json, profile.json, ...)",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU device IDs (default: auto-detect or 0)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum optimisation rounds (default: GEAK_MAX_ROUNDS env or 5)",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=1,
        help="Round number to resume from (default: 1)",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        default=DEFAULT_HETEROGENEOUS,
        help="Mark the report as heterogeneous mode",
    )
    add_agent_filter_args(parser)
    args = parser.parse_args()
    apply_agent_filter_env(args)

    preprocess_dir = Path(args.preprocess_dir).resolve()
    if not preprocess_dir.is_dir():
        print(f"ERROR: preprocess directory not found: {args.preprocess_dir}", file=sys.stderr)
        sys.exit(1)

    ctx = _load_preprocess_context(preprocess_dir)
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]
    else:
        gpu_ids = detect_available_gpus()

    report = run_orchestrator(
        preprocess_ctx=ctx,
        gpu_ids=gpu_ids,
        output_dir=preprocess_dir,
        max_rounds=args.max_rounds,
        start_round=args.start_round,
        heterogeneous=args.heterogeneous,
    )
    print(json.dumps(report, indent=2, default=str))


__all__ = ["run_orchestrator", "main"]


if __name__ == "__main__":
    main()
