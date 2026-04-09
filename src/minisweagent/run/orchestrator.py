"""Orchestrator: dispatch to homogeneous or heterogeneous optimization mode.

This module is the thin entry point for ``geak-orchestrate``.  It reads
preprocessor artefacts, resolves configuration, and delegates to:

- ``agents.heterogeneous.orchestrator`` -- LLM-generated diverse tasks

Note: homogeneous mode is handled directly by ``mini`` CLI via
``agents.homogeneous.homogeneous_agent`` and is not supported here.

All heavy lifting lives in those modules and in the shared postprocess
package (``run.postprocess.evaluation``, ``run.postprocess.results``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from minisweagent.run.pipeline_helpers import DEFAULT_HETEROGENEOUS, DEFAULT_PIPELINE_OUTPUT_DIR

logger = logging.getLogger(__name__)


def run_orchestrator(
    preprocess_ctx: dict[str, Any],
    gpu_ids: list[int],
    model,
    model_factory,
    *,
    output_dir: Path | None = None,
    max_rounds: int | None = None,
    start_round: int = 1,
    heterogeneous: bool = DEFAULT_HETEROGENEOUS,
) -> dict[str, Any]:
    """Run the orchestrator agent loop.

    Parameters
    ----------
    preprocess_ctx:
        Context dict returned by ``run_preprocessor()``.
    gpu_ids:
        List of GPU device IDs available for task execution.
    model:
        LLM model instance for the orchestrator.
    model_factory:
        Callable returning a new model instance (for sub-agents).
    output_dir:
        Override output directory (defaults to preprocess_ctx source).
    max_rounds:
        Maximum optimisation rounds (default: from GEAK_MAX_ROUNDS env or 5).
    start_round:
        Round number to start from (1-based, default 1).
    heterogeneous:
        If True, use LLM-generated diverse tasks per round.
        If False (default), use homogeneous mode where all agents get the same task.
    """
    _out = output_dir or Path(preprocess_ctx.get("output_dir", DEFAULT_PIPELINE_OUTPUT_DIR))
    _out = Path(_out)
    _out.mkdir(parents=True, exist_ok=True)

    _env_rounds = os.getenv("GEAK_MAX_ROUNDS")
    max_rounds = max_rounds or int(_env_rounds or "5")
    logger.info(
        "run_orchestrator: output_dir=%s, max_rounds=%d (source=%s), start_round=%d, heterogeneous=%s",
        _out,
        max_rounds,
        "arg" if (max_rounds != int(_env_rounds or "5") if _env_rounds else max_rounds != 5) else ("env" if _env_rounds else "default"),
        start_round,
        heterogeneous,
    )

    if not heterogeneous:
        raise NotImplementedError("Homogeneous mode is not supported via geak-orchestrate. Use the 'mini' CLI instead.")

    from minisweagent.agents.heterogeneous.orchestrator import run_heterogeneous_orchestrator

    return run_heterogeneous_orchestrator(
        preprocess_ctx,
        gpu_ids,
        model,
        model_factory,
        _out,
        max_rounds,
        start_round,
    )


# ── CLI entry point ──────────────────────────────────────────────────


def _probe_preprocess_dir(pp_dir: Path):
    """Backward-compatible fallback: reconstruct PreprocessContext by probing files."""
    from minisweagent.run.pipeline_types import PreprocessContext

    logger.debug("_probe_preprocess_dir: probing %s for preprocessor artefacts.", pp_dir)
    kernel_path = ""
    repo_root = str(pp_dir)
    harness_path = ""

    resolved_path = pp_dir / "resolved.json"
    if resolved_path.exists():
        resolved = json.loads(resolved_path.read_text())
        kernel_path = resolved.get("local_file_path", "")
        repo_path = resolved.get("local_repo_path")
        if kernel_path:
            kp = Path(kernel_path).resolve()
            git_root = None
            cur = kp if kp.is_dir() else kp.parent
            while cur != cur.parent:
                if (cur / ".git").exists():
                    git_root = cur
                    break
                cur = cur.parent
            if git_root:
                repo_root = str(git_root)
                logger.debug("_probe_preprocess_dir: repo_root from git walk: %s", repo_root)
            elif repo_path:
                repo_root = repo_path
                logger.debug("_probe_preprocess_dir: repo_root from resolved.json: %s", repo_root)
            else:
                repo_root = str(Path(kernel_path).parent)
                logger.debug("_probe_preprocess_dir: repo_root defaulted to kernel parent: %s", repo_root)

    testcase_sel_path = pp_dir / "testcase_selection.json"
    if testcase_sel_path.exists():
        try:
            ts = json.loads(testcase_sel_path.read_text())
            if isinstance(ts, dict):
                harness_path = ts.get("harness_path", "")
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("_probe_preprocess_dir: failed to read testcase_selection.json: %s", exc)

    discovery = None
    discovery_path = pp_dir / "discovery.json"
    if discovery_path.exists():
        try:
            discovery = json.loads(discovery_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("_probe_preprocess_dir: failed to read discovery.json: %s", exc)

    return PreprocessContext(
        kernel_path=kernel_path,
        repo_root=repo_root,
        harness_path=harness_path,
        preprocess_dir=str(pp_dir),
        commandment_path=str(pp_dir / "COMMANDMENT.md") if (pp_dir / "COMMANDMENT.md").exists() else "",
        codebase_context_path=str(pp_dir / "CODEBASE_CONTEXT.md") if (pp_dir / "CODEBASE_CONTEXT.md").exists() else "",
        baseline_metrics_path=str(pp_dir / "baseline_metrics.json")
        if (pp_dir / "baseline_metrics.json").exists()
        else "",
        profiling_result_path=str(pp_dir / "profile.json") if (pp_dir / "profile.json").exists() else "",
        discovery=discovery,
    )


def main() -> None:
    """CLI: ``geak-orchestrate --preprocess-dir <dir> [--gpu-ids 0,1] [--max-rounds 3]``."""
    import argparse

    from minisweagent.run.pipeline_helpers import DEFAULT_HETEROGENEOUS

    parser = argparse.ArgumentParser(
        description="GEAK orchestrator: LLM-driven task generation, dispatch, and iteration loop",
    )
    parser.add_argument(
        "--preprocess-dir",
        required=True,
        help="Directory containing preprocessor artefacts (resolved.json, discovery.json, profile.json, ...)",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU device IDs (default: all detected GPUs, or 0 as fallback)",
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
        help="Round to resume from (1-based, default: 1). "
        "Skips exploration and loads prior round evaluations from disk.",
    )
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        default=DEFAULT_HETEROGENEOUS,
        help="Use LLM-generated diverse tasks per round. Default: homogeneous (all agents get the same task).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: from GEAK_MODEL env or geak.yaml)",
    )
    from minisweagent.run.pipeline_helpers import add_agent_filter_args, apply_agent_filter_env

    add_agent_filter_args(parser)
    args = parser.parse_args()
    apply_agent_filter_env(args)

    pp_dir = Path(args.preprocess_dir).resolve()
    if not pp_dir.is_dir():
        logger.error("Preprocess directory not found: %s", args.preprocess_dir)
        sys.exit(1)

    from minisweagent.run.pipeline_types import PreprocessContext

    manifest_path = pp_dir / "preprocess_context.json"
    if manifest_path.exists():
        preprocess_ctx = PreprocessContext.from_dict(json.loads(manifest_path.read_text()))
    else:
        logger.warning("preprocess_context.json not found in %s, falling back to file probing", pp_dir)
        preprocess_ctx = _probe_preprocess_dir(pp_dir)

    ctx: dict[str, Any] = {
        "kernel_path": preprocess_ctx.kernel_path,
        "repo_root": preprocess_ctx.repo_root,
        "harness_path": preprocess_ctx.harness_path,
        "output_dir": preprocess_ctx.preprocess_dir,
        "preprocess_dir": preprocess_ctx.preprocess_dir,
        "commandment_path": preprocess_ctx.commandment_path,
        "codebase_context_path": preprocess_ctx.codebase_context_path,
        "baseline_metrics_path": preprocess_ctx.baseline_metrics_path,
        "profiling_path": preprocess_ctx.profiling_result_path,
        "discovery": preprocess_ctx.discovery,
    }
    if preprocess_ctx.commandment_path and Path(preprocess_ctx.commandment_path).exists():
        ctx["commandment"] = Path(preprocess_ctx.commandment_path).read_text()
    if preprocess_ctx.baseline_metrics_path and Path(preprocess_ctx.baseline_metrics_path).exists():
        ctx["baseline_metrics"] = json.loads(Path(preprocess_ctx.baseline_metrics_path).read_text())
    if preprocess_ctx.profiling_result_path and Path(preprocess_ctx.profiling_result_path).exists():
        ctx["profiling"] = json.loads(Path(preprocess_ctx.profiling_result_path).read_text())

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip()]
        logger.info("GPU IDs from CLI: %s", gpu_ids)
    else:
        try:
            from minisweagent.agents.agent_spec import detect_available_gpus

            gpu_ids = detect_available_gpus()
            logger.info("Auto-detected GPU IDs: %s", gpu_ids)
        except Exception as exc:
            logger.warning("GPU auto-detection failed (%s); falling back to [0].", exc)
            gpu_ids = [0]

    from minisweagent.run.pipeline_helpers import geak_model_factory, load_geak_model

    model_name = args.model or os.getenv("GEAK_MODEL")
    model = load_geak_model(model_name)
    factory = geak_model_factory(model_name)

    report = run_orchestrator(
        preprocess_ctx=ctx,
        gpu_ids=gpu_ids,
        model=model,
        model_factory=factory,
        output_dir=pp_dir,
        max_rounds=args.max_rounds,
        start_round=args.start_round,
        heterogeneous=args.heterogeneous,
    )

    if report:
        report_dict = report.to_dict() if hasattr(report, "to_dict") else report
        logger.info("Orchestrator report (truncated): %s", json.dumps(report_dict, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
