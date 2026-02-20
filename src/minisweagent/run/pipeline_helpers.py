"""Shared helpers for the GEAK preprocessing and orchestration pipelines.

All CLI entry points (``geak``, ``geak-preprocess``, ``geak-orchestrate``,
``run-tasks``, ``task-generator``) import from this module so that harness
extraction, validation, profiling, model loading, agent filtering, and
pipeline-context injection are always identical regardless of entry point.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

REQUIRED_HARNESS_FLAGS = ("--profile", "--correctness")

MAX_HARNESS_RETRIES = 2


# ── agent filtering ──────────────────────────────────────────────────


def add_agent_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--allowed-agents`` and ``--excluded-agents`` to *parser*."""
    parser.add_argument(
        "--allowed-agents",
        default=None,
        help=(
            "Comma-separated list of allowed agent types "
            "(e.g. swe_agent,strategy_agent). Sets GEAK_ALLOWED_AGENTS."
        ),
    )
    parser.add_argument(
        "--excluded-agents",
        default=None,
        help=(
            "Comma-separated list of excluded agent types "
            "(e.g. openevolve). Sets GEAK_EXCLUDED_AGENTS."
        ),
    )


def apply_agent_filter_env(args: argparse.Namespace) -> None:
    """Propagate ``--allowed-agents`` / ``--excluded-agents`` to env vars."""
    if getattr(args, "allowed_agents", None):
        os.environ["GEAK_ALLOWED_AGENTS"] = args.allowed_agents
    if getattr(args, "excluded_agents", None):
        os.environ["GEAK_EXCLUDED_AGENTS"] = args.excluded_agents


# ── model loading ────────────────────────────────────────────────────


def load_geak_model(
    model_name: str | None,
    *,
    config_spec: str = "geak",
) -> Any:
    """Load an LLM model using the standard GEAK config-resolution pattern.

    Reads the YAML config for *config_spec*, extracts the ``model`` section,
    and delegates to ``get_model``.  Falls back to the ``GEAK_MODEL``
    environment variable when *model_name* is ``None``.
    """
    import yaml

    from minisweagent.config import get_config_path
    from minisweagent.models import get_model

    resolved_name = model_name or os.environ.get("GEAK_MODEL")
    cfg_path = get_config_path(config_spec)
    model_config: dict[str, Any] = {}
    if cfg_path.exists():
        full_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_config = full_cfg.get("model", {})

    return get_model(resolved_name, config=model_config)


def geak_model_factory(
    model_name: str | None,
    *,
    config_spec: str = "geak",
):
    """Return a zero-arg callable that creates a fresh model each time."""
    import yaml

    from minisweagent.config import get_config_path
    from minisweagent.models import get_model

    resolved_name = model_name or os.environ.get("GEAK_MODEL")
    cfg_path = get_config_path(config_spec)
    model_config: dict[str, Any] = {}
    if cfg_path.exists():
        full_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_config = full_cfg.get("model", {})

    def _factory():
        return get_model(resolved_name, config=copy.deepcopy(model_config))

    return _factory


def _ensure_mcp_importable() -> None:
    """Add MCP tool source directories to sys.path if not already present."""
    for sub in (
        "mcp_tools/profiler-mcp/src",
        "mcp_tools/metrix-mcp/src",
        "mcp_tools/automated-test-discovery/src",
    ):
        p = str(_REPO_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


# ── harness path extraction ──────────────────────────────────────────


def extract_harness_path(test_command: str) -> str:
    """Extract the harness script path from a test command string.

    Handles patterns like::

        'pytest /path/to/test.py -v'                -> '/path/to/test.py'
        'python /path/to/harness.py --correctness'  -> '/path/to/harness.py'
        '/path/to/harness.py'                       -> '/path/to/harness.py'
    """
    try:
        tokens = shlex.split(test_command)
    except ValueError:
        tokens = test_command.split()

    for token in tokens:
        if token.endswith(".py") and "/" in token:
            return token

    for token in tokens:
        if token.endswith(".py"):
            return token

    return tokens[-1] if tokens else test_command


# ── harness validation ───────────────────────────────────────────────


def validate_harness(harness_path: str) -> tuple[bool, list[str]]:
    """Static-analyse a harness script to verify it supports required CLI flags.

    Checks that the harness uses an argument-parsing library (argparse, click,
    or typer) and defines ``--profile`` and ``--correctness`` flags.  This is a
    fast, no-GPU, no-runtime check that catches the exact class of LLM
    compliance bug where the harness ignores CLI flags.

    Returns ``(valid, errors)`` where *errors* is empty when *valid* is True.
    """
    harness = Path(harness_path)
    errors: list[str] = []

    if not harness.is_file():
        return False, [f"Harness file not found: {harness}"]

    source = harness.read_text()

    has_parser = (
        "argparse" in source
        or "ArgumentParser" in source
        or "click" in source
        or "typer" in source
    )
    if not has_parser:
        errors.append(
            "Harness does not use argparse/click/typer -- "
            "CLI flags like --profile and --correctness will be silently ignored"
        )

    for flag in REQUIRED_HARNESS_FLAGS:
        if flag not in source:
            errors.append(f"Harness source does not define '{flag}' flag")

    return len(errors) == 0, errors


# ── validated harness creation (UnitTestAgent + retry) ───────────────


def create_validated_harness(
    *,
    model: Any,
    repo: Path,
    kernel_name: str,
    log_dir: Path | None,
    discovery_context: str,
    max_retries: int = MAX_HARNESS_RETRIES,
) -> str:
    """Run UnitTestAgent with harness validation and retry loop.

    After the agent produces a harness, :func:`validate_harness` checks that
    it actually defines ``--profile`` and ``--correctness``.  If validation
    fails the errors are fed back into the discovery context and the agent is
    re-invoked, up to *max_retries* additional attempts.

    Returns the ``test_command`` string on success.

    Raises
    ------
    RuntimeError
        If validation still fails after all retries.
    """
    from minisweagent.agents.unit_test_agent import run_unit_test_agent

    max_attempts = max_retries + 1
    harness_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        ctx = discovery_context
        if harness_errors:
            ctx += (
                f"\n\nHARNESS VALIDATION FAILED (attempt {attempt}/{max_attempts}):\n"
                + "\n".join(f"- {e}" for e in harness_errors)
                + "\n\nYou MUST add argparse with --profile and --correctness modes. "
                "See INSTRUCTIONS.md sections 1a and 1b."
            )

        test_command = run_unit_test_agent(
            model=model,
            repo=repo,
            kernel_name=kernel_name,
            log_dir=log_dir,
            discovery_context=ctx,
        )
        logger.info("UnitTestAgent test_command (attempt %d): %s", attempt, test_command)

        harness = extract_harness_path(test_command)
        valid, harness_errors = validate_harness(harness)
        if valid:
            logger.info("Harness validation: OK")
            return test_command

        logger.warning(
            "Harness validation failed (attempt %d/%d): %s",
            attempt,
            max_attempts,
            harness_errors,
        )

        if attempt == max_attempts:
            raise RuntimeError(
                f"Harness validation failed after {max_attempts} attempts: "
                + "; ".join(harness_errors)
            )

    raise AssertionError("unreachable")  # pragma: no cover


# ── pipeline context injection ───────────────────────────────────────


def inject_pipeline_context(
    task_body: str,
    config: dict,
    *,
    commandment_text: str | None = None,
    baseline_metrics: dict | None = None,
    profiling_path: str | None = None,
    kernel_path: str | None = None,
    repo_root: str | None = None,
    test_command: str | None = None,
    codebase_context: str | None = None,
) -> tuple[str, dict]:
    """Prepend pipeline context to *task_body* and augment *config*.

    This is the single canonical context-injection path.  Both
    ``dispatch.task_file_to_agent_task`` and the ``geak`` parallel path
    call this so that every agent -- regardless of dispatch route --
    receives identical pipeline context.

    Returns ``(enriched_body, updated_config)``.
    """

    cfg = dict(config)
    ctx: list[str] = [
        "## Pipeline Context (auto-injected from task metadata)",
        "",
    ]

    if kernel_path:
        ctx.append(f"KERNEL FILE TO EDIT: {kernel_path}")
    if repo_root:
        ctx.append(f"REPO ROOT: {repo_root}")
    if test_command:
        ctx.append(f"TEST COMMAND: {test_command}")
    ctx.append("")

    ctx.append(
        "IMPORTANT: Only edit files within your REPO ROOT directory. "
        "Do NOT search or modify files outside of it. "
        "The KERNEL FILE TO EDIT path above is the exact file you should optimize."
    )
    ctx.append("")

    if commandment_text:
        ctx.append("## COMMANDMENT (evaluation contract -- you MUST follow these rules)")
        ctx.append(commandment_text.strip())
        ctx.append("")

    if baseline_metrics:
        dur = baseline_metrics.get("duration_us", "unknown")
        bn = baseline_metrics.get("bottleneck", "unknown")
        ctx.append("## Baseline Performance (your optimization must improve on these)")
        ctx.append(f"Total duration: {dur} us")
        ctx.append(f"Bottleneck: {bn}")
        top = baseline_metrics.get("top_kernels", [])
        if top:
            ctx.append("Top kernels by duration:")
            for k in top[:5]:
                bn_tag = f" [{k['bottleneck']}]" if k.get("bottleneck") else ""
                ctx.append(
                    f"  - {k.get('name', '?')}: {k.get('duration_us', '?')} us "
                    f"({k.get('pct_of_total', '?')}%){bn_tag}"
                )
        ctx.append("")

    if profiling_path and Path(profiling_path).exists():
        ctx.append(f"PROFILING DATA: {profiling_path}")
        ctx.append("(Read this file for detailed per-kernel profiling metrics)")
        ctx.append("")

    if codebase_context:
        ctx.append("## Codebase Context (repo structure and key files)")
        ctx.append(codebase_context.strip())
        ctx.append("")
        cfg["codebase_context"] = codebase_context.strip()

    enriched = "\n".join(ctx) + "\n" + task_body
    return enriched, cfg


# ── baseline profiling (via profiler-mcp, with warmup) ───────────────


def run_baseline_profile(test_command: str, gpu_id: int = 0) -> dict:
    """Profile the test harness via profiler-mcp (includes warmup).

    Uses ``profiler_mcp.server.profile_kernel`` which performs backend-agnostic
    warmup runs before the actual instrumented profiling pass.
    """
    _ensure_mcp_importable()
    from profiler_mcp.server import profile_kernel

    harness = extract_harness_path(test_command)
    profile_cmd = f"python {harness} --profile"

    _profile_fn = getattr(profile_kernel, "fn", profile_kernel)
    return _profile_fn(
        command=profile_cmd,
        backend="metrix",
        num_replays=3,
        quick=True,
        gpu_devices=str(gpu_id),
    )
