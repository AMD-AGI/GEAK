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
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from minisweagent import get_repo_root
from minisweagent.run.utils.pipeline_context import (
    build_bottleneck_guidance as _shared_bottleneck_guidance,
)
from minisweagent.run.utils.pipeline_context import (
    format_gpu_info as _shared_format_gpu_info,
)
from minisweagent.run.utils.pipeline_context import (
    gpu_arch_context as _shared_gpu_arch_context,
)
from minisweagent.run.utils.pipeline_context import (
    inject_pipeline_context as _shared_inject_pipeline_context,
)
from minisweagent.run.utils.pipeline_context import (
    run_baseline_profile as _shared_run_baseline_profile,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = get_repo_root()

REQUIRED_HARNESS_FLAGS = ("--profile", "--correctness", "--benchmark", "--full-benchmark")

MAX_HARNESS_RETRIES = 2

# Use one canonical benchmark definition everywhere. The legacy
# GEAK_AGENT_BENCHMARK_ITERATIONS split is intentionally ignored so
# agent-time patch testing and final verification stay apples-to-apples.
DEFAULT_EVAL_BENCHMARK_ITERATIONS = int(os.getenv("GEAK_EVAL_BENCHMARK_ITERATIONS", "30"))
DEFAULT_AGENT_BENCHMARK_ITERATIONS = DEFAULT_EVAL_BENCHMARK_ITERATIONS
DEFAULT_PIPELINE_OUTPUT_DIR = "geak_output"
DEFAULT_HETEROGENEOUS = False


# ── agent filtering ──────────────────────────────────────────────────


def add_agent_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--allowed-agents`` and ``--excluded-agents`` to *parser*."""
    parser.add_argument(
        "--allowed-agents",
        default=None,
        help=("Comma-separated list of allowed agent types (e.g. strategy_agent). Sets GEAK_ALLOWED_AGENTS."),
    )
    parser.add_argument(
        "--excluded-agents",
        default=None,
        help=("Comma-separated list of excluded agent types (e.g. openevolve). Sets GEAK_EXCLUDED_AGENTS."),
    )


def apply_agent_filter_env(args: argparse.Namespace) -> None:
    """Propagate ``--allowed-agents`` / ``--excluded-agents`` to env vars."""
    configure_agent_filter_env(
        getattr(args, "allowed_agents", None),
        getattr(args, "excluded_agents", None),
    )


def configure_agent_filter_env(
    allowed_agents: str | None,
    excluded_agents: str | None,
) -> None:
    """Apply generic default agent filters.

    Default behavior excludes ``openevolve`` unless the user explicitly
    supplies an allowlist/excludelist or pre-sets ``GEAK_EXCLUDED_AGENTS``.
    This keeps the default pipeline focused on the lighter-weight agents while
    still allowing users to opt in deliberately.
    """

    if allowed_agents:
        os.environ["GEAK_ALLOWED_AGENTS"] = allowed_agents
        if excluded_agents is not None:
            os.environ["GEAK_EXCLUDED_AGENTS"] = excluded_agents
        return

    if excluded_agents:
        os.environ["GEAK_EXCLUDED_AGENTS"] = excluded_agents
        return

    os.environ.setdefault("GEAK_EXCLUDED_AGENTS", "openevolve")


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

    resolved_name = model_name or os.environ.get("GEAK_MODEL") or "claude-opus-4.6"
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

    resolved_name = model_name or os.environ.get("GEAK_MODEL") or "claude-opus-4.6"
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


def _preferred_harness_path(log_dir: Path, kernel_path: Path | None) -> Path:
    if kernel_path is not None:
        stem = kernel_path.stem or "kernel"
        return log_dir / f"test_{stem}_harness.py"
    return log_dir / "geak_test_harness.py"


def _materialized_harness_bootstrap(
    *,
    repo_root: Path,
    kernel_path: Path | None,
) -> str:
    kernel_dir = kernel_path.resolve().parent if kernel_path is not None else None
    rel_kernel_dir: Path | None = None
    if kernel_dir is not None:
        try:
            rel_kernel_dir = kernel_dir.relative_to(repo_root.resolve())
        except ValueError:
            rel_kernel_dir = None

    rel_kernel_dir_text = str(rel_kernel_dir).replace("\\", "/") if rel_kernel_dir is not None else ""
    original_kernel_dir = str(kernel_dir) if kernel_dir is not None else ""
    return (
        "# GEAK materialized harness bootstrap\n"
        "def _resolve_geak_kernel_dir():\n"
        "    candidates = []\n"
        '    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()\n'
        "    if work_dir:\n"
        "        candidates.append(work_dir)\n"
        '    repo_root = os.environ.get("GEAK_REPO_ROOT", "").strip()\n'
        f"    rel_kernel_dir = {rel_kernel_dir_text!r}\n"
        "    if repo_root and rel_kernel_dir:\n"
        "        candidates.append(os.path.join(repo_root, rel_kernel_dir))\n"
        f"    original_kernel_dir = {original_kernel_dir!r}\n"
        "    if original_kernel_dir:\n"
        "        candidates.append(original_kernel_dir)\n"
        "    for candidate in candidates:\n"
        '        if candidate and os.path.isfile(os.path.join(candidate, "kernel.py")):\n'
        "            return candidate\n"
        "    return original_kernel_dir or os.getcwd()\n"
        "\n"
        "_KERNEL_DIR = _resolve_geak_kernel_dir()\n"
        "if _KERNEL_DIR not in sys.path:\n"
        "    sys.path.insert(0, _KERNEL_DIR)\n"
    )


def _rewrite_materialized_harness_source(
    source_text: str,
    *,
    repo_root: Path,
    kernel_path: Path | None,
) -> str:
    bootstrap = _materialized_harness_bootstrap(
        repo_root=repo_root,
        kernel_path=kernel_path,
    )
    legacy_patterns = [
        re.compile(
            r"(?ms)^# Ensure the kernel directory is importable\n"
            r"_KERNEL_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n"
            r"if _KERNEL_DIR not in sys\.path:\n"
            r"\s+sys\.path\.insert\(0, _KERNEL_DIR\)\n"
        ),
        re.compile(
            r"(?ms)^_KERNEL_DIR = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n"
            r"if _KERNEL_DIR not in sys\.path:\n"
            r"\s+sys\.path\.insert\(0, _KERNEL_DIR\)\n"
        ),
    ]
    for pattern in legacy_patterns:
        if pattern.search(source_text):
            return pattern.sub(bootstrap, source_text, count=1)

    import_block = re.compile(r"(?ms)\A((?:from __future__ import annotations\n)?(?:import .+\n|from .+ import .+\n)+)")
    match = import_block.match(source_text)
    if match:
        return source_text[: match.end()] + "\n" + bootstrap + source_text[match.end() :]
    return bootstrap + "\n" + source_text


def _materialize_validated_harness(
    *,
    test_command: str,
    harness_path: str,
    repo_root: Path,
    log_dir: Path | None,
    kernel_path: Path | None,
    gpu_id: int,
) -> tuple[str, str, list[dict[str, Any]]] | None:
    if log_dir is None:
        return None

    source_harness = Path(harness_path).resolve()
    target_dir = log_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_harness = _preferred_harness_path(target_dir, kernel_path)
    if source_harness == target_harness:
        return None

    rewritten_text = _rewrite_materialized_harness_source(
        source_harness.read_text(),
        repo_root=repo_root,
        kernel_path=kernel_path,
    )
    target_harness.write_text(rewritten_text)
    shutil.copymode(source_harness, target_harness)

    materialized_command = test_command.replace(str(source_harness), str(target_harness))
    valid, static_errors = validate_harness(str(target_harness))
    if not valid:
        raise RuntimeError("Materialized harness static validation failed: " + "; ".join(static_errors))

    exec_ok, exec_errors, harness_results = execute_harness_validation(
        str(target_harness),
        repo_root=str(repo_root),
        gpu_id=gpu_id,
    )
    if not exec_ok:
        raise RuntimeError(
            "Materialized harness runtime validation failed: " + "; ".join(e.splitlines()[0] for e in exec_errors)
        )
    return materialized_command, str(target_harness), harness_results


# ── harness validation ───────────────────────────────────────────────


_GPU_ALLOC_IN_PROFILE_RE = re.compile(
    r"""torch\.(?:randn?|empty|zeros|ones|full)\s*\("""
    r"""[^)]*device\s*=\s*["']cuda["']""",
)


def validate_harness(harness_path: str) -> tuple[bool, list[str]]:
    """Static-analyse a harness script to verify it supports required CLI flags.

    Checks that the harness uses an argument-parsing library (argparse, click,
    or typer) and defines all four required flags: ``--correctness``,
    ``--profile``, ``--benchmark``, and ``--full-benchmark``.  Also checks
    that the ``run_profile`` function (if present) does not allocate tensors
    directly on CUDA, which would pollute the profiler trace with GPU RNG /
    memset kernels.

    Returns ``(valid, errors)`` where *errors* is empty when *valid* is True.
    """
    harness = Path(harness_path)
    errors: list[str] = []

    if not harness.is_file():
        return False, [f"Harness file not found: {harness}"]

    source = harness.read_text()

    has_parser = "argparse" in source or "ArgumentParser" in source or "click" in source or "typer" in source
    if not has_parser:
        errors.append(
            "Harness does not use argparse/click/typer -- "
            "CLI flags like --profile and --correctness will be silently ignored"
        )

    for flag in REQUIRED_HARNESS_FLAGS:
        if flag not in source:
            errors.append(f"Harness source does not define '{flag}' flag")

    # Check for GPU-side tensor allocation inside the profile function.
    # rocprofv3 captures ALL GPU kernels, so torch.randn(..., device='cuda')
    # inside run_profile pollutes the trace with RNG kernels.
    _in_profile_fn = False
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("def ") and "profile" in stripped:
            _in_profile_fn = True
            continue
        if _in_profile_fn and stripped.startswith("def "):
            _in_profile_fn = False
        if _in_profile_fn and _GPU_ALLOC_IN_PROFILE_RE.search(line):
            errors.append(
                f"Line {lineno}: GPU tensor allocation inside profile function "
                f"(device='cuda'). Use device='cpu' then .to('cuda') to avoid "
                f"polluting the profiler trace with RNG/memset kernels. "
                f"See src/minisweagent/run/preprocess/INSTRUCTIONS.md point 8."
            )
            break  # one warning is enough

    return len(errors) == 0, errors


# ── harness runtime execution ─────────────────────────────────────────


def execute_harness_validation(
    harness_path: str,
    repo_root: str | None = None,
    gpu_id: int = 0,
    benchmark_extra_args: str | None = None,
) -> tuple[bool, list[str], list[dict]]:
    """Run the harness across all modes and return ``(ok, errors, results)``.

    Delegates to :func:`minisweagent.tools.run_harness.run_harness` with
    ``mode="all"`` which executes correctness -> profile -> benchmark ->
    full-benchmark in sequence, short-circuiting on first failure.

    Parameters
    ----------
    benchmark_extra_args:
        Extra benchmark tuning args. ``--iterations N`` is normalized into
        ``GEAK_BENCHMARK_ITERATIONS`` so harnesses do not need to expose it
        as a CLI flag. Any remaining args are passed via
        ``GEAK_BENCHMARK_EXTRA_ARGS``.

    Returns
    -------
    ok : bool
        True if every mode passed.
    errors : list[str]
        Human-readable error descriptions for failed modes (empty on success).
    results : list[dict]
        Per-mode result dicts from :func:`run_harness`.
    """
    from minisweagent.run.preprocess.run_harness import results_errors, run_harness

    env_overrides: dict[str, str] = {}
    # Keep validation fast: override iterations to a small number unless
    # the caller explicitly provides benchmark_extra_args.
    if not benchmark_extra_args:
        env_overrides["GEAK_BENCHMARK_ITERATIONS"] = "5"
    else:
        import re as _re

        remaining_extra_args = benchmark_extra_args.strip()
        _iter_match = _re.search(r"(?:^|\s)--iterations\s+(\d+)(?=\s|$)", remaining_extra_args)
        if _iter_match:
            env_overrides["GEAK_BENCHMARK_ITERATIONS"] = _iter_match.group(1)
            remaining_extra_args = _re.sub(
                r"(?:^|\s)--iterations\s+\d+(?=\s|$)",
                " ",
                remaining_extra_args,
            ).strip()
        if remaining_extra_args:
            env_overrides["GEAK_BENCHMARK_EXTRA_ARGS"] = remaining_extra_args

    results = run_harness(
        harness_path,
        mode="all",
        repo_root=repo_root,
        gpu_id=gpu_id,
        env_overrides=env_overrides,
    )
    if not isinstance(results, list):
        results = [results]

    ok = all(r["success"] for r in results)
    errors = results_errors(results) if not ok else []
    return ok, errors, results


# ── validated harness creation (UnitTestAgent + retry) ───────────────


def create_validated_harness(
    *,
    model: Any,
    repo: Path,
    kernel_name: str,
    log_dir: Path | None,
    kernel_path: Path | None,
    discovery_context: str,
    max_retries: int = MAX_HARNESS_RETRIES,
    gpu_id: int = 0,
) -> tuple[str, list[dict]]:
    """Run UnitTestAgent with static + runtime validation and retry loop.

    After the agent produces a harness:
      1. :func:`validate_harness` performs static analysis (argparse,
         ``--profile``, ``--correctness`` flags, GPU allocation patterns).
      2. :func:`execute_harness_validation` actually runs the harness in
         all four modes (correctness, profile, benchmark, full-benchmark)
         to catch import errors, shape mismatches, OOM, etc.

    If either step fails the errors are fed back into the discovery context
    and the agent is re-invoked, up to *max_retries* additional attempts.

    Returns ``(test_command, harness_results)`` on success where
    *harness_results* is the list of per-mode result dicts.

    Raises
    ------
    RuntimeError
        If validation still fails after all retries.
    """
    from minisweagent.run.preprocess.unit_test_agent import run_unit_test_agent

    max_attempts = max_retries + 1
    harness_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        ctx = discovery_context
        if harness_errors:
            ctx += (
                f"\n\nHARNESS VALIDATION FAILED (attempt {attempt}/{max_attempts}):\n"
                + "\n".join(f"- {e}" for e in harness_errors)
                + "\n\nYou MUST fix the harness so that ALL modes work: "
                "--correctness, --profile, --benchmark, --full-benchmark. "
                "See src/minisweagent/run/preprocess/INSTRUCTIONS.md sections 1a and 1b."
            )

        test_command = run_unit_test_agent(
            model=model,
            repo=repo,
            kernel_name=kernel_name,
            log_dir=log_dir,
            preferred_harness_path=_preferred_harness_path(log_dir, kernel_path) if log_dir else None,
            kernel_path=kernel_path,
            discovery_context=ctx,
        )
        logger.info("UnitTestAgent test_command (attempt %d): %s", attempt, test_command)

        harness = extract_harness_path(test_command)

        # Phase 1: static analysis
        valid, harness_errors = validate_harness(harness)
        if not valid:
            logger.warning(
                "Harness static validation failed (attempt %d/%d): %s",
                attempt,
                max_attempts,
                harness_errors,
            )
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Harness validation failed after {max_attempts} attempts: " + "; ".join(harness_errors)
                )
            continue

        logger.info("Harness static validation: OK")

        # Phase 2: runtime execution of all modes
        repo_root = str(repo) if repo else None
        exec_ok, exec_errors, harness_results = execute_harness_validation(
            harness,
            repo_root=repo_root,
            gpu_id=gpu_id,
        )
        if exec_ok:
            try:
                materialized = _materialize_validated_harness(
                    test_command=test_command,
                    harness_path=harness,
                    repo_root=repo,
                    log_dir=log_dir,
                    kernel_path=kernel_path,
                    gpu_id=gpu_id,
                )
            except Exception as exc:
                harness_errors = [str(exc)]
                logger.warning(
                    "Harness materialization failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    harness_errors,
                )
                if attempt == max_attempts:
                    raise RuntimeError(f"Harness materialization failed after {max_attempts} attempts: {exc}")
                continue
            if materialized is not None:
                test_command, harness, harness_results = materialized
                logger.info("Materialized harness to %s", harness)
            logger.info("Harness runtime validation: ALL MODES PASSED")
            return test_command, harness_results

        harness_errors = exec_errors
        logger.warning(
            "Harness runtime validation failed (attempt %d/%d): %s",
            attempt,
            max_attempts,
            [e.splitlines()[0] for e in exec_errors],
        )

        if attempt == max_attempts:
            raise RuntimeError(
                f"Harness runtime validation failed after {max_attempts} attempts: "
                + "; ".join(e.splitlines()[0] for e in exec_errors)
            )

    raise AssertionError("unreachable")  # pragma: no cover


# ── shared pipeline context and profiling guidance wrappers ──────────


def _bottleneck_guidance(metrics: dict) -> list[str]:
    return _shared_bottleneck_guidance(metrics)


def _format_gpu_info(gpu_info: dict) -> list[str]:
    return _shared_format_gpu_info(gpu_info)


def _gpu_arch_context(profiling_path: str) -> list[str]:
    return _shared_gpu_arch_context(profiling_path)


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
    benchmark_baseline: str | None = None,
) -> tuple[str, dict]:
    return _shared_inject_pipeline_context(
        task_body,
        config,
        commandment_text=commandment_text,
        baseline_metrics=baseline_metrics,
        profiling_path=profiling_path,
        kernel_path=kernel_path,
        repo_root=repo_root,
        test_command=test_command,
        codebase_context=codebase_context,
        benchmark_baseline=benchmark_baseline,
    )


def run_baseline_profile(test_command: str, gpu_id: int = 0) -> dict:
    return _shared_run_baseline_profile(
        test_command,
        gpu_id=gpu_id,
        ensure_mcp_importable=_ensure_mcp_importable,
        extract_harness_path=extract_harness_path,
    )
