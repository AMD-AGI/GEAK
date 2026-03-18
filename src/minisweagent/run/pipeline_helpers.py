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

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

REQUIRED_HARNESS_FLAGS = ("--profile", "--correctness", "--benchmark", "--full-benchmark")

MAX_HARNESS_RETRIES = 2

# Use one canonical benchmark definition everywhere. The legacy
# GEAK_AGENT_BENCHMARK_ITERATIONS split is intentionally ignored so
# agent-time patch testing and final verification stay apples-to-apples.
DEFAULT_EVAL_BENCHMARK_ITERATIONS = int(
    os.getenv("GEAK_EVAL_BENCHMARK_ITERATIONS", "30")
)
DEFAULT_AGENT_BENCHMARK_ITERATIONS = DEFAULT_EVAL_BENCHMARK_ITERATIONS
DEFAULT_PIPELINE_OUTPUT_DIR = "geak_output"
DEFAULT_HETEROGENEOUS = False


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
        "    work_dir = os.environ.get(\"GEAK_WORK_DIR\", \"\").strip()\n"
        "    if work_dir:\n"
        "        candidates.append(work_dir)\n"
        "    repo_root = os.environ.get(\"GEAK_REPO_ROOT\", \"\").strip()\n"
        f"    rel_kernel_dir = {rel_kernel_dir_text!r}\n"
        "    if repo_root and rel_kernel_dir:\n"
        "        candidates.append(os.path.join(repo_root, rel_kernel_dir))\n"
        f"    original_kernel_dir = {original_kernel_dir!r}\n"
        "    if original_kernel_dir:\n"
        "        candidates.append(original_kernel_dir)\n"
        "    for candidate in candidates:\n"
        "        if candidate and os.path.isfile(os.path.join(candidate, \"kernel.py\")):\n"
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

    import_block = re.compile(
        r"(?ms)\A((?:from __future__ import annotations\n)?(?:import .+\n|from .+ import .+\n)+)"
    )
    match = import_block.match(source_text)
    if match:
        return source_text[: match.end()] + "\n" + bootstrap + source_text[match.end():]
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
        raise RuntimeError(
            "Materialized harness static validation failed: " + "; ".join(static_errors)
        )

    exec_ok, exec_errors, harness_results = execute_harness_validation(
        str(target_harness),
        repo_root=str(repo_root),
        gpu_id=gpu_id,
    )
    if not exec_ok:
        raise RuntimeError(
            "Materialized harness runtime validation failed: "
            + "; ".join(e.splitlines()[0] for e in exec_errors)
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
                f"See INSTRUCTIONS.md point 8."
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
        Extra CLI args appended to benchmark/full-benchmark invocations
        (e.g. ``"--iterations 50"``).  Passed via the
        ``GEAK_BENCHMARK_EXTRA_ARGS`` env var so both direct invocations
        and COMMANDMENT-based scripts use the same settings.

    Returns
    -------
    ok : bool
        True if every mode passed.
    errors : list[str]
        Human-readable error descriptions for failed modes (empty on success).
    results : list[dict]
        Per-mode result dicts from :func:`run_harness`.
    """
    from minisweagent.tools.run_harness import results_errors, run_harness

    env_overrides: dict[str, str] | None = None
    if benchmark_extra_args:
        env_overrides = {"GEAK_BENCHMARK_EXTRA_ARGS": benchmark_extra_args}

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
    from minisweagent.agents.unit_test_agent import run_unit_test_agent

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
                "See INSTRUCTIONS.md sections 1a and 1b."
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
                    f"Harness validation failed after {max_attempts} attempts: "
                    + "; ".join(harness_errors)
                )
            continue

        logger.info("Harness static validation: OK")

        # Phase 2: runtime execution of all modes
        repo_root = str(repo) if repo else None
        exec_ok, exec_errors, harness_results = execute_harness_validation(
            harness, repo_root=repo_root, gpu_id=gpu_id,
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
                    raise RuntimeError(
                        f"Harness materialization failed after {max_attempts} attempts: {exc}"
                    )
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


# ── bottleneck-specific optimization guidance ────────────────────────

_BOTTLENECK_GUIDANCE: dict[str, str] = {
    "balanced": (
        "## Optimization Guidance (bottleneck: balanced)\n"
        '"Balanced" means no single resource is saturated. Actionable kernel-body approaches:\n'
        "1. INCREASE ARITHMETIC INTENSITY: Fuse adjacent operations into the kernel loop "
        "so more compute happens per memory access.\n"
        "2. REDUCE MEMORY TRAFFIC: Cache intermediate results in registers or LDS "
        "instead of reading/writing global memory.\n"
        "3. IMPROVE PARALLELISM: Restructure loops to expose more independent work per "
        "wavefront; consider split-K or multi-pass approaches.\n"
        "4. ALTERNATIVE ALGORITHMS: Try a fundamentally different algorithm for the same "
        "computation (different reduction tree, different scan, tiled vs non-tiled, etc.).\n"
        "5. COMPILER GUIDANCE: Restructure Triton/HIP code to help the compiler generate "
        "better ISA -- avoid tl.where in hot loops, use tl.constexpr aggressively, "
        "minimize live variables across tl.dot calls.\n"
    ),
    "memory-bound": (
        "## Optimization Guidance (bottleneck: memory-bound)\n"
        "The kernel is limited by memory bandwidth. Focus on kernel-body changes:\n"
        "1. VECTORIZED LOADS: Use float4/float2 vector loads to maximize HBM throughput.\n"
        "2. COALESCED ACCESS: Ensure adjacent threads access adjacent memory addresses.\n"
        "3. LDS STAGING: Stage global memory reads through LDS to improve access patterns.\n"
        "4. REDUCE DATA MOVEMENT: Recompute values instead of storing and reloading them.\n"
        "5. OPERATION FUSION: Fuse the memory-bound kernel with adjacent elementwise ops "
        "to amortize memory access cost over more computation.\n"
        "6. TILING / BLOCKING: Increase tile sizes to improve data reuse from L2 cache.\n"
    ),
    "compute-bound": (
        "## Optimization Guidance (bottleneck: compute-bound)\n"
        "The kernel is limited by arithmetic throughput. Focus on kernel-body changes:\n"
        "1. REDUCE INSTRUCTION COUNT: Simplify expressions, use hardware intrinsics "
        "(tl.math.rsqrt, fma), eliminate redundant computations.\n"
        "2. USE MFMA INSTRUCTIONS: On AMD GPUs, restructure computation to use Matrix "
        "Fused Multiply-Add for dense linear algebra.\n"
        "3. STRENGTH REDUCTION: Replace expensive ops (div, mod, pow) with cheaper "
        "equivalents (shifts, masks, lookup tables).\n"
        "4. LOOP UNROLLING: Manually unroll inner loops to help the compiler schedule "
        "instructions more aggressively.\n"
        "5. ALGORITHM CHANGE: Switch to an algorithm with lower computational complexity "
        "(e.g., O(n log n) vs O(n^2), approximate methods).\n"
    ),
    "latency-bound": (
        "## Optimization Guidance (bottleneck: latency-bound)\n"
        "The kernel is too short to saturate any resource. Focus on kernel-body changes:\n"
        "1. INCREASE WORK PER KERNEL: Process more elements per thread or per block "
        "to amortize kernel launch overhead.\n"
        "2. FUSE KERNELS: Merge this kernel with adjacent ones to eliminate launch gaps.\n"
        "3. PERSISTENT KERNEL: Convert to a persistent kernel pattern that stays resident "
        "and processes multiple tiles without relaunching.\n"
        "4. INCREASE BLOCK SIZE: Use larger thread blocks to improve GPU occupancy for "
        "this short-running kernel.\n"
    ),
    "lds-bound": (
        "## Optimization Guidance (bottleneck: lds-bound)\n"
        "The kernel is limited by LDS (Local Data Share) bandwidth or capacity.\n"
        "1. REDUCE LDS BANK CONFLICTS: Pad shared memory arrays to avoid stride-32 "
        "access patterns (on AMD: 32 banks, 4 bytes each).\n"
        "2. REDUCE LDS USAGE: Move data from LDS to registers where possible to free "
        "LDS capacity and improve occupancy.\n"
        "3. OPTIMIZE LDS ACCESS PATTERN: Restructure loops so that LDS reads/writes "
        "are coalesced within each wavefront.\n"
        "4. SPLIT COMPUTATION: Break the kernel into phases that use LDS at different "
        "times to reduce peak LDS pressure.\n"
    ),
}


_SEARCH_WORKLOAD_HINTS = (
    "binary_search",
    "lower_bound",
    "upper_bound",
    "search_n",
    "device_search",
    "haystack",
    "needle",
)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _search_workload_guidance(metrics: dict) -> list[str]:
    """Add narrower guidance for latency-bound HIP search workloads."""
    evidence_chunks = [str(metrics.get("kernel_name", ""))]
    for top in metrics.get("top_kernels", []) or []:
        evidence_chunks.append(str(top.get("name", "")))
    haystack = " ".join(evidence_chunks).lower()
    if not any(hint in haystack for hint in _SEARCH_WORKLOAD_HINTS):
        return []

    bottleneck = str(metrics.get("bottleneck", "")).lower()
    if "latency" not in bottleneck:
        return []

    derived = metrics.get("metrics", {}) or {}
    hbm_util = _safe_float(derived.get("memory.hbm_bandwidth_utilization"))
    l2_hit = _safe_float(derived.get("memory.l2_hit_rate"))
    if hbm_util is not None and hbm_util >= 10.0:
        return []

    hbm_text = f"{hbm_util:.1f}%" if hbm_util is not None else "unknown"
    l2_text = f"{l2_hit:.1f}%" if l2_hit is not None else "unknown"
    return [
        "## Workload Guidance (HIP search / pointer-chasing)",
        (
            "Profiler evidence suggests a latency-bound search workload: "
            f"HBM utilization={hbm_text}, L2 hit rate={l2_text}."
        ),
        "Prioritize branchless search logic, operation-specific specialization, and size-specialized variants.",
        "Also consider wavefront-cooperative upper-level search and amortized pivot-table narrowing when correctness rules allow it.",
        "Deprioritize generic vectorization or bandwidth-maximization ideas unless later profiling shows memory throughput is actually the limiter.",
        "",
    ]


def _bottleneck_guidance(bottleneck: str, metrics: dict) -> list[str]:
    """Return actionable optimization guidance lines based on bottleneck type."""
    bn_lower = bottleneck.lower().strip()
    bn_aliases = {
        "latency": "latency-bound",
        "memory": "memory-bound",
        "compute": "compute-bound",
        "lds": "lds-bound",
    }
    bn_lower = bn_aliases.get(bn_lower, bn_lower)
    for key, text in _BOTTLENECK_GUIDANCE.items():
        if key in bn_lower:
            lines = text.strip().splitlines()
            lines.extend(_search_workload_guidance(metrics))
            lines.append("")
            return lines

    lines = _BOTTLENECK_GUIDANCE["balanced"].strip().splitlines()
    lines.extend(_search_workload_guidance(metrics))
    lines.append("")
    return lines


# ── GPU architecture context from profiling data ─────────────────────

def _gpu_arch_context(profiling_path: str) -> list[str]:
    """Extract GPU architecture info from profile.json and format it."""
    import json as _json

    try:
        data = _json.loads(Path(profiling_path).read_text())
    except Exception:
        return []

    results = data.get("results", [])
    if not results:
        return []

    gpu_info = results[0].get("gpu_info", {}) if isinstance(results[0], dict) else {}
    if not gpu_info:
        for r in results:
            if isinstance(r, dict) and r.get("gpu_info"):
                gpu_info = r["gpu_info"]
                break

    if not gpu_info:
        return []

    arch = gpu_info.get("architecture", gpu_info.get("gfx_version", "unknown"))
    name = gpu_info.get("name", gpu_info.get("model", "AMD GPU"))
    cus = gpu_info.get("compute_units", "?")
    hbm_bw = gpu_info.get("peak_hbm_bandwidth_gbps", gpu_info.get("hbm_bandwidth", "?"))
    lds_per_cu = gpu_info.get("lds_per_cu_kb", 64)
    vgprs = gpu_info.get("vgprs_per_cu", 512)

    return [
        f"## GPU Architecture: {name} ({arch})",
        f"- Architecture: {arch}",
        f"- Compute Units: {cus}",
        f"- Peak HBM bandwidth: {hbm_bw} GB/s",
        f"- LDS per CU: {lds_per_cu} KB (32 banks on gfx9xx)",
        f"- VGPRs per CU: {vgprs}",
        "- Wavefront size: 64 (AMD default), some kernels can use 32",
        "- MFMA (Matrix Fused Multiply-Add) instructions available for dense math",
        "- Use these specs to guide your kernel optimizations (tile sizes, occupancy, LDS usage).",
        "",
    ]


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
    benchmark_baseline: str | None = None,
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

        ctx.extend(_bottleneck_guidance(str(bn), baseline_metrics))

    if profiling_path and Path(profiling_path).exists():
        ctx.append(f"PROFILING DATA: {profiling_path}")
        ctx.append("(Read this file for detailed per-kernel profiling metrics)")
        ctx.append("")

        ctx.extend(_gpu_arch_context(profiling_path))

    if benchmark_baseline:
        ctx.append("## Benchmark Baseline (compare your save_and_test output against this)")
        ctx.append("This is the original kernel's canonical benchmark output from the same full benchmark contract used for patch testing.")
        ctx.append("Your save_and_test output includes canonical benchmark results -- compare against these numbers.")
        ctx.append(f"```\n{benchmark_baseline.strip()}\n```")
        ctx.append("")

    if codebase_context:
        ctx.append("## Codebase Context (kernel dependency tree)")
        ctx.append(
            "The dependency tree below shows in-repo files the target kernel "
            "imports. Every listed dependency is a potential optimization "
            "target -- improving any of them can reduce overall latency."
        )
        ctx.append(codebase_context.strip())
        ctx.append("")
        cfg["codebase_context"] = codebase_context.strip()

    ctx.append(
        "IMPORTANT: Baseline profiling and performance metrics are already "
        "established and provided above. Do NOT run save_and_test for a "
        "baseline run. Start optimizing immediately."
    )
    ctx.append("")

    try:
        from minisweagent.memory.integration import assemble_memory_context
        _bm = baseline_metrics or {}
        _mem_ctx = assemble_memory_context(
            kernel_path=kernel_path,
            bottleneck_type=_bm.get("bottleneck"),
            profiling_metrics=_bm,
        )
        if _mem_ctx:
            ctx.append("## Optimization Memory (from past kernel optimization runs)")
            ctx.append(_mem_ctx.strip())
            ctx.append("")
    except Exception:
        pass

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
