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
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

REQUIRED_HARNESS_FLAGS = ("--profile", "--correctness", "--benchmark", "--full-benchmark")

MAX_HARNESS_RETRIES = 2

DEFAULT_AGENT_BENCHMARK_ITERATIONS = int(os.getenv("GEAK_AGENT_BENCHMARK_ITERATIONS", "10"))

DEFAULT_EVAL_BENCHMARK_ITERATIONS = DEFAULT_AGENT_BENCHMARK_ITERATIONS


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

    cfg_path = get_config_path(config_spec)
    model_config: dict[str, Any] = {}
    if cfg_path.exists():
        full_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_config = full_cfg.get("model", {})

    return get_model(model_name, config=model_config)


def geak_model_factory(
    model_name: str | None,
    *,
    config_spec: str = "geak",
):
    """Return a zero-arg callable that creates a fresh model each time."""
    import yaml

    from minisweagent.config import get_config_path
    from minisweagent.models import get_model

    cfg_path = get_config_path(config_spec)
    model_config: dict[str, Any] = {}
    if cfg_path.exists():
        full_cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_config = full_cfg.get("model", {})

    def _factory():
        return get_model(model_name, config=copy.deepcopy(model_config))

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


def _bottleneck_guidance(bottleneck: str, metrics: dict) -> list[str]:
    """Return actionable optimization guidance lines based on bottleneck type."""
    bn_lower = bottleneck.lower().strip()
    for key, text in _BOTTLENECK_GUIDANCE.items():
        if key in bn_lower:
            lines = text.strip().splitlines()
            lines.append("")
            return lines

    lines = _BOTTLENECK_GUIDANCE["balanced"].strip().splitlines()
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
        ctx.append("This is the original kernel's --benchmark output on HARNESS_SHAPES (20-25 shapes).")
        ctx.append("Your save_and_test output includes benchmark results -- compare against these numbers.")
        ctx.append(f"```\n{benchmark_baseline.strip()}\n```")
        ctx.append("")

    if codebase_context:
        ctx.append("## Codebase Context (repo structure and key files)")
        ctx.append(codebase_context.strip())
        ctx.append("")
        cfg["codebase_context"] = codebase_context.strip()

    ctx.append(
        "IMPORTANT: Baseline profiling and performance metrics are already "
        "established and provided above. Do NOT run save_and_test for a "
        "baseline run. Start optimizing immediately."
    )
    ctx.append("")

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


# ── kernel resolution helpers (extracted from mini.py) ───────────────


def run_discovery(kernel_path: str, kernel_name: str | None = None) -> str:
    """Run test discovery on the resolved kernel and return formatted results for the task prompt.

    The raw DiscoveryResult is stashed on ``run_discovery._last_result`` so the
    caller can pass it to the task planner without a second discovery pass.
    """
    from rich.console import Console

    try:
        from automated_test_discovery.server import discover as atd_discover

        from minisweagent.tools.discovery_types import DiscoveryResult
    except ImportError:
        return ""

    console = Console(highlight=False)
    console.print("\n[bold cyan]--- Test Discovery ---[/bold cyan]")
    if kernel_name:
        console.print(f"[dim]Kernel function: {kernel_name}[/dim]")
    try:
        kp = Path(kernel_path)
        _discover_fn = getattr(atd_discover, "fn", atd_discover)
        disc_dict = _discover_fn(
            kernel_path=str(kp),
            output_dir=str(kp.parent),
        )

        result = DiscoveryResult.from_dict(disc_dict, kp)
        run_discovery._last_result = result

        lines = []
        kernel_stem = kp.stem.lower()
        match_terms = [kernel_stem]
        if kernel_name:
            match_terms.extend([p for p in kernel_name.lower().split("_") if len(p) > 2])

        def _is_relevant(path_str):
            return any(term in path_str.lower() for term in match_terms)

        relevant_tests = [t for t in result.tests if _is_relevant(str(t.file_path))]
        other_tests = [t for t in result.tests if not _is_relevant(str(t.file_path))][:3]
        all_display = relevant_tests + other_tests
        if all_display:
            console.print(
                f"[bold green]Found {len(result.tests)} test(s) ({len(relevant_tests)} matching '{kernel_stem}'):[/bold green]"
            )
            for t in all_display[:5]:
                marker = "+" if kernel_stem in str(t.file_path).lower() else "-"
                console.print(f"  [green]{marker}[/green] {t.file_path} [dim](confidence: {t.confidence:.1f})[/dim]")
                lines.append(f"  - {t.file_path} (confidence: {t.confidence:.1f}, command: {t.command})")
        else:
            console.print("[yellow]No existing tests found.[/yellow]")
        relevant_bench = [b for b in result.benchmarks if kernel_stem in str(b.file_path).lower()]
        if relevant_bench:
            console.print(f"[bold green]Found {len(relevant_bench)} matching benchmark(s):[/bold green]")
            for b in relevant_bench[:3]:
                console.print(f"  [green]+[/green] {b.file_path} [dim](confidence: {b.confidence:.1f})[/dim]")
                lines.append(f"  - Benchmark: {b.file_path} (confidence: {b.confidence:.1f})")
        console.print("[bold cyan]---------------------[/bold cyan]\n")

        if lines:
            return (
                "\n--- Discovered Tests ---\n"
                + "\n".join(lines)
                + "\nRead these test files and reuse their reference implementations, input patterns, and tolerances.\n---\n"
            )
    except Exception as e:
        console.print(f"[yellow]Discovery failed: {e}[/yellow]")
    return ""


def inject_resolved_kernel(kernel_url: str, workspace: str | None, task: str) -> tuple[str, str | None]:
    """Resolve kernel URL to local path/line/kernel name and append workflow block to task."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from minisweagent.tools.resolve_kernel_url_impl import get_kernel_name_at_line, resolve_kernel_url
    except ImportError as e:
        raise SystemExit(f"Cannot resolve --kernel-url: resolve_kernel_url_impl not found ({e}).") from e
    clone_into = Path(workspace) if workspace else Path.cwd()
    resolved = resolve_kernel_url(kernel_url, clone_into=clone_into)
    if resolved.get("error"):
        raise SystemExit(f"Kernel URL resolve failed: {resolved['error']}")
    path = resolved["local_file_path"]
    line_num = resolved.get("line_number")
    kernel_name = get_kernel_name_at_line(path, line_num) if line_num else None
    if line_num:
        line_info = f" Line: {line_num}"
        kernel_info = f", kernel name: {kernel_name!r}" if kernel_name else ""
        profile_hint = "When profiling, all kernels are reported; the agent can choose which to use."
    else:
        line_info = ""
        kernel_info = ""
        profile_hint = "Line number was not specified; discovery should identify the kernel(s) in the file."
    kernel_dir = str(Path(path).parent)
    output_dir = f"{kernel_dir}/optimization_output"
    oe_script = "${GEAK_OE_ROOT:-/opt/geak-oe}/examples/geak_eval/run_openevolve.py"
    block = f"""\n
--- Resolved kernel (from --kernel-url) ---
Kernel path: {path}{(" |" + line_info + kernel_info) if line_info else ""}
---

--- Workflow ---
Follow these steps IN ORDER. Do one step per response.

Step 1 - DISCOVER: Read and analyse the kernel file. Identify the kernel function, its inputs/outputs, dependencies, and any existing tests in the repo.

Step 2 - TEST GEN: Create a standalone test harness that can (a) verify correctness and (b) benchmark performance. Save it next to the kernel (e.g. {kernel_dir}/test_harness.py).

Step 3 - BENCHMARK & COMMANDMENT: Profile the baseline kernel with kernel-profile and create two artifacts:
  a) baseline_metrics.json -- latency/bandwidth numbers from profiling.
  b) COMMANDMENT.md -- the evaluation contract for OpenEvolve.
  {profile_hint}
  Profile command example: kernel-profile 'python3 {path} --profile'

Step 4 - OPTIMIZE: Do NOT edit the kernel by hand. Run the OpenEvolve optimizer:
  python3 {oe_script} {path} --iterations 10 --gpu 0 --output {output_dir}
  If you created COMMANDMENT.md and baseline_metrics.json, add:
  --commandment <path_to_COMMANDMENT.md> --baseline-metrics <path_to_baseline_metrics.json>

Step 5 - REPORT: Summarise results (speedup, best score, any errors) and submit:
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
---
"""
    return task + block, kernel_name
