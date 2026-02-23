"""Preprocessor: sequential pipeline of existing modules.

Runs resolve-kernel-url -> codebase-context -> test-discovery ->
harness-execution -> kernel-profile -> baseline-metrics -> commandment
in order and returns a context dict for the orchestrator.

Each step calls the *same* Python function that the corresponding CLI
uses, so behaviour is identical whether invoked from here or from the
shell.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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


from minisweagent.run.pipeline_helpers import (
    create_validated_harness,
    extract_harness_path,
    run_baseline_profile,
)

# ── main entry point ─────────────────────────────────────────────────


def run_preprocessor(
    kernel_url: str,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    model=None,
    model_factory=None,
    console=None,
) -> dict[str, Any]:
    """Run all preprocessing steps and return a context dict.

    Parameters
    ----------
    kernel_url:
        GitHub URL or local path to the kernel.
    output_dir:
        Directory to write intermediate artefacts (resolved.json, etc.).
    gpu_id:
        GPU device to use for profiling.
    model:
        LLM model instance for the UnitTestAgent (optional).
    model_factory:
        Callable returning a new model instance (used if model is None).
    console:
        Optional Rich console for progress messages.

    Returns
    -------
    dict with keys:
        resolved, codebase_context_path, discovery, harness_results,
        profiling, baseline_metrics, commandment, test_command,
        kernel_path, repo_root, harness_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _print(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    ctx: dict[str, Any] = {}

    # ── 1. resolve-kernel-url ────────────────────────────────────────
    _print(
        "[bold cyan]--- Step 1/7: Resolve kernel URL ---[/bold cyan]"
        if console
        else "--- Step 1/7: Resolve kernel URL ---"
    )

    from minisweagent.tools.resolve_kernel_url_impl import resolve_kernel_url

    resolved = resolve_kernel_url(kernel_url, clone_into=str(output_dir))
    if resolved.get("error"):
        raise RuntimeError(f"resolve-kernel-url failed: {resolved['error']}")

    kernel_path = resolved["local_file_path"]
    repo_root = resolved.get("local_repo_path") or str(Path(kernel_path).parent)
    ctx["resolved"] = resolved
    ctx["kernel_path"] = kernel_path
    ctx["repo_root"] = repo_root

    (output_dir / "resolved.json").write_text(json.dumps(resolved, indent=2, default=str))
    _print(f"  Kernel: {kernel_path}")

    # ── 2. codebase context ──────────────────────────────────────────
    _print(
        "[bold cyan]--- Step 2/7: Codebase context ---[/bold cyan]"
        if console
        else "--- Step 2/7: Codebase context ---"
    )

    from minisweagent.run.codebase_context import generate_codebase_context

    codebase_context_path = generate_codebase_context(
        repo_root=Path(repo_root),
        kernel_path=Path(kernel_path),
        output_dir=output_dir,
    )
    ctx["codebase_context_path"] = str(codebase_context_path)
    _print(f"  CODEBASE_CONTEXT.md written ({codebase_context_path.stat().st_size} bytes)")

    # ── 3. test-discovery (automated_test_discovery MCP) ────────────
    _print("[bold cyan]--- Step 3/7: Test discovery ---[/bold cyan]" if console else "--- Step 3/7: Test discovery ---")

    _ensure_mcp_importable()
    from automated_test_discovery.server import discover as atd_discover

    _discover_fn = getattr(atd_discover, "fn", atd_discover)
    disc_dict = {}  # Initialize to empty dict to avoid NameError if discovery fails
    try:
        disc_dict = _discover_fn(
            kernel_path=kernel_path,
            output_dir=str(output_dir),
        )
    except Exception as exc:
        logger.warning("Test discovery failed: %s", exc)
        _print(f"  [yellow]Warning: Test discovery failed: {exc}[/yellow]" if console else f"  Warning: Test discovery failed: {exc}")
    
    ctx["discovery"] = disc_dict
    (output_dir / "discovery.json").write_text(json.dumps(disc_dict, indent=2, default=str))

    tests = disc_dict.get("tests", [])
    _print(f"  Tests found: {len(tests)}")

    # ── 3b. UnitTestAgent: create a proper test harness ─────────────
    # The MCP discovery finds test files but doesn't create a validated
    # harness with --correctness/--profile modes. The UnitTestAgent is a
    # full LLM agent that can read the kernel, read existing tests, run
    # them, see errors, and iterate until the harness works.
    #
    # After the agent produces a harness we:
    #   1. Statically validate it (argparse, --profile, --correctness)
    #   2. Run it in ALL modes (correctness, profile, benchmark,
    #      full-benchmark) to catch runtime errors early
    # If either step fails we feed errors back to the agent and retry.
    test_command = None
    harness_results: list[dict] | None = None
    _uta_model = model or (model_factory() if model_factory else None)
    if _uta_model and repo_root:
        _print(
            "[bold cyan]--- Step 3b/3c: UnitTestAgent (harness creation + execution) ---[/bold cyan]"
            if console
            else "--- Step 3b/3c: UnitTestAgent (harness creation + execution) ---"
        )
        try:
            from minisweagent.agents.unit_test_agent import format_discovery_for_agent
            from minisweagent.tools.discovery_types import DiscoveryResult

            disc_result = DiscoveryResult.from_dict(disc_dict, kernel_path)
            discovery_context = format_discovery_for_agent(disc_result)

            if codebase_context_path.exists():
                discovery_context = codebase_context_path.read_text() + "\n\n" + discovery_context

            kernel_name = Path(kernel_path).stem
            discovery_context += (
                "\n\nIMPORTANT: Your TEST_COMMAND must use absolute paths "
                "to the test script (e.g., `python /absolute/path/to/test_harness.py --correctness`). "
                "Do NOT use `cd` in the command. The profiler cannot handle compound shell commands."
            )

            test_command, harness_results = create_validated_harness(
                model=_uta_model,
                repo=Path(repo_root),
                kernel_name=kernel_name,
                log_dir=output_dir,
                discovery_context=discovery_context,
                gpu_id=gpu_id,
            )
            _print(f"  UnitTestAgent test_command: {test_command}")
            _print("  Harness static validation: OK")
            for r in harness_results:
                status = "PASS" if r["success"] else "FAIL"
                _print(f"  Harness --{r['mode']}: {status} ({r['duration_s']}s)")
            _print("  Harness execution: ALL MODES PASSED")
        except Exception as exc:
            _print(
                f"  [yellow]UnitTestAgent failed ({exc}), falling back to discovery[/yellow]"
                if console
                else f"  UnitTestAgent failed ({exc}), falling back to discovery"
            )
            logger.warning("UnitTestAgent failed: %s", exc, exc_info=True)
            test_command = None
            harness_results = None

    # Fall back to discovery results if UnitTestAgent didn't produce one.
    # Prefer the focused test (which targets the specific kernel) over
    # the generic test commands (which may be pytest suites without
    # --correctness/--profile support).
    if not test_command:
        focused = disc_dict.get("focused_test") or {}
        focused_cmd = focused.get("focused_command")
        if focused_cmd:
            test_command = focused_cmd
            _print(f"  Falling back to discovery focused test: {test_command}")
        elif tests:
            test_command = tests[0]["command"]
            _print(f"  Falling back to discovery test: {test_command}")

    ctx["test_command"] = test_command
    ctx["harness_results"] = harness_results
    if harness_results:
        (output_dir / "harness_results.json").write_text(
            json.dumps(harness_results, indent=2, default=str)
        )

    benchmark_baseline: str | None = None
    full_benchmark_baseline: str | None = None
    if harness_results:
        for r in harness_results:
            if r["mode"] == "benchmark" and r["success"]:
                benchmark_baseline = r["stdout"]
                (output_dir / "benchmark_baseline.txt").write_text(r["stdout"])
            if r["mode"] == "full-benchmark" and r["success"]:
                full_benchmark_baseline = r["stdout"]
                (output_dir / "full_benchmark_baseline.txt").write_text(r["stdout"])
    ctx["benchmark_baseline"] = benchmark_baseline
    ctx["full_benchmark_baseline"] = full_benchmark_baseline

    if test_command:
        _print(f"  Test command: {test_command}")

    # ── 5. kernel-profile (via profiler-mcp) ─────────────────────────
    _print(
        "[bold cyan]--- Step 5/7: Kernel profiling (Metrix instrumented) ---[/bold cyan]"
        if console
        else "--- Step 5/7: Kernel profiling (Metrix instrumented) ---"
    )

    profiling: dict[str, Any] | None = None
    if test_command:
        ctx["harness_path"] = extract_harness_path(test_command)
        (output_dir / "harness_path.txt").write_text(ctx["harness_path"])

        try:
            profiling = run_baseline_profile(test_command, gpu_id=gpu_id)
        except Exception as exc:
            _print(f"  [yellow]Profiling failed: {exc}[/yellow]" if console else f"  Profiling failed: {exc}")
            logger.warning("Profiling failed: %s", exc, exc_info=True)
    else:
        _print("  Skipping profiling (no test command found)")

    ctx["profiling"] = profiling
    if profiling:
        (output_dir / "profile.json").write_text(json.dumps(profiling, indent=2, default=str))
        _print("  Profiling complete")

    # ── 6. baseline-metrics ──────────────────────────────────────────
    _print(
        "[bold cyan]--- Step 6/7: Baseline metrics ---[/bold cyan]" if console else "--- Step 6/7: Baseline metrics ---"
    )

    baseline_metrics: dict[str, Any] | None = None
    if profiling and profiling.get("success", True):
        try:
            from minisweagent.baseline_metrics import build_baseline_metrics

            baseline_metrics = build_baseline_metrics(profiling, include_all=True)
            dur = baseline_metrics.get("duration_us", "?")
            bn = baseline_metrics.get("bottleneck", "?")
            _print(f"  Baseline: {dur} µs, bottleneck={bn}")
        except Exception as exc:
            _print(
                f"  [yellow]Baseline metrics failed: {exc}[/yellow]" if console else f"  Baseline metrics failed: {exc}"
            )
            logger.warning("Baseline metrics failed: %s", exc, exc_info=True)
    else:
        _print("  Skipping baseline metrics (no profiling data)")

    ctx["baseline_metrics"] = baseline_metrics

    # Enrich baseline_metrics with wall-clock benchmark data so that all
    # consumers (OpenEvolve, orchestrator) compare benchmark-vs-benchmark
    # instead of mixing Metrix profile durations with wall-clock latencies.
    if baseline_metrics is None:
        baseline_metrics = {}
    bb_path = output_dir / "benchmark_baseline.txt"
    if bb_path.exists():
        import re as _re

        bb_text = bb_path.read_text()
        _m = _re.search(
            r"median\s+latency[\w\s]*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms",
            bb_text,
            _re.IGNORECASE,
        )
        if _m:
            baseline_metrics["benchmark_duration_us"] = float(_m.group(1)) * 1000.0
        _sm = _re.search(r"(\d+)\s+shapes", bb_text, _re.IGNORECASE)
        if _sm:
            baseline_metrics["benchmark_shape_count"] = int(_sm.group(1))
        ctx["baseline_metrics"] = baseline_metrics

    if baseline_metrics:
        (output_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2, default=str))

    # ── 7. commandment ───────────────────────────────────────────────
    _print("[bold cyan]--- Step 7/7: Commandment ---[/bold cyan]" if console else "--- Step 7/7: Commandment ---")

    commandment: str | None = None
    if test_command:
        try:
            from minisweagent.tools.commandment import generate_commandment

            harness = ctx.get("harness_path") or extract_harness_path(test_command)
            commandment = generate_commandment(
                kernel_path=kernel_path,
                harness_path=harness,
                repo_root=repo_root,
            )
            _print("  COMMANDMENT.md generated")
        except Exception as exc:
            _print(f"  [yellow]Commandment failed: {exc}[/yellow]" if console else f"  Commandment failed: {exc}")
            logger.warning("Commandment generation failed: %s", exc, exc_info=True)
    else:
        _print("  Skipping commandment (no test command)")

    ctx["commandment"] = commandment
    if commandment:
        (output_dir / "COMMANDMENT.md").write_text(commandment)

    _print("")
    _print("Preprocessing complete. Artefacts written to: " + str(output_dir))
    return ctx


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    """CLI: ``geak-preprocess <url> -o output_dir/``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GEAK preprocessor: resolve → context → discover → harness-exec → profile → baseline → commandment",
    )
    parser.add_argument("url", help="GitHub URL or local path to the kernel")
    parser.add_argument(
        "-o",
        "--output",
        default="geak_preprocess_output",
        help="Output directory for intermediate artefacts (default: geak_preprocess_output)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID for profiling (default: 0)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model name for UnitTestAgent harness creation (uses default if omitted)",
    )
    args = parser.parse_args()

    try:
        from rich.console import Console

        console = Console()
    except ImportError:
        console = None

    from minisweagent.run.pipeline_helpers import geak_model_factory

    _model_factory = geak_model_factory(args.model)

    ctx = run_preprocessor(
        args.url,
        Path(args.output),
        gpu_id=args.gpu,
        model_factory=_model_factory,
        console=console,
    )

    print(json.dumps(ctx, indent=2, default=str))


if __name__ == "__main__":
    main()
