"""Preprocessor: sequential pipeline of existing modules.

Runs resolve-kernel-url -> test-discovery -> kernel-profile ->
baseline-metrics -> commandment in order and returns a context dict
for the orchestrator.

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


# ── helpers ──────────────────────────────────────────────────────────


def _extract_harness_path(test_command: str) -> str:
    """Extract the harness script path from a test command string."""
    import shlex

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


# ── main entry point ─────────────────────────────────────────────────


def run_preprocessor(
    kernel_url: str,
    output_dir: Path,
    gpu_id: int = 0,
    *,
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
    console:
        Optional Rich console for progress messages.

    Returns
    -------
    dict with keys:
        resolved, discovery, profiling, baseline_metrics,
        commandment, test_command, kernel_path, repo_root,
        harness_path
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
        "[bold cyan]--- Step 1/5: Resolve kernel URL ---[/bold cyan]"
        if console
        else "--- Step 1/5: Resolve kernel URL ---"
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

    # ── 2. test-discovery (automated_test_discovery MCP) ────────────
    _print("[bold cyan]--- Step 2/5: Test discovery ---[/bold cyan]" if console else "--- Step 2/5: Test discovery ---")

    _ensure_mcp_importable()
    from automated_test_discovery.server import discover as atd_discover

    _discover_fn = getattr(atd_discover, "fn", atd_discover)
    disc_dict = _discover_fn(
        kernel_path=kernel_path,
        output_dir=str(output_dir),
    )
    ctx["discovery"] = disc_dict
    (output_dir / "discovery.json").write_text(json.dumps(disc_dict, indent=2, default=str))

    tests = disc_dict.get("tests", [])
    focused = disc_dict.get("focused_test") or {}

    # Prefer the verified focused test command over the raw test file command.
    # Fall back to the original test if the focused harness failed verification.
    if focused.get("focused_command") and focused.get("verified", True):
        test_command = focused["focused_command"]
        _print(f"  Using focused test: {focused.get('focused_test_file', '?')}")
    elif tests:
        test_command = tests[0]["command"]
    else:
        test_command = None

    ctx["test_command"] = test_command
    _print(f"  Tests found: {len(tests)}")
    if test_command:
        _print(f"  Test command: {test_command}")

    # ── 3. kernel-profile (via profiler-mcp) ─────────────────────────
    _print(
        "[bold cyan]--- Step 3/5: Kernel profiling ---[/bold cyan]" if console else "--- Step 3/5: Kernel profiling ---"
    )

    profiling: dict[str, Any] | None = None
    if test_command:
        from profiler_mcp.server import profile_kernel

        harness = focused.get("focused_test_file") or _extract_harness_path(test_command)
        profile_cmd = focused.get("focused_command") or test_command
        ctx["harness_path"] = harness

        try:
            _profile_fn = getattr(profile_kernel, "fn", profile_kernel)
            profiling = _profile_fn(
                command=profile_cmd,
                backend="metrix",
                num_replays=3,
                quick=True,
                gpu_devices=str(gpu_id),
            )
        except Exception as exc:
            _print(f"  [yellow]Profiling failed: {exc}[/yellow]" if console else f"  Profiling failed: {exc}")
            logger.warning("Profiling failed: %s", exc, exc_info=True)
    else:
        _print("  Skipping profiling (no test command found)")

    ctx["profiling"] = profiling
    if profiling:
        (output_dir / "profile.json").write_text(json.dumps(profiling, indent=2, default=str))
        _print("  Profiling complete")

    # ── 4. baseline-metrics ──────────────────────────────────────────
    _print(
        "[bold cyan]--- Step 4/5: Baseline metrics ---[/bold cyan]" if console else "--- Step 4/5: Baseline metrics ---"
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
    if baseline_metrics:
        (output_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2, default=str))

    # ── 5. commandment ───────────────────────────────────────────────
    _print("[bold cyan]--- Step 5/5: Commandment ---[/bold cyan]" if console else "--- Step 5/5: Commandment ---")

    commandment: str | None = None
    if test_command:
        try:
            from minisweagent.tools.commandment import generate_commandment

            harness = ctx.get("harness_path") or _extract_harness_path(test_command)
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
        description="GEAK preprocessor: resolve → discover → profile → baseline → commandment",
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
    args = parser.parse_args()

    try:
        from rich.console import Console

        console = Console()
    except ImportError:
        console = None

    ctx = run_preprocessor(args.url, Path(args.output), gpu_id=args.gpu, console=console)

    print(json.dumps(ctx, indent=2, default=str))


if __name__ == "__main__":
    main()
