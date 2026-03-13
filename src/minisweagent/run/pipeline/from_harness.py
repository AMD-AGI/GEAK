"""geak-from-harness: build a preprocess directory from an existing test harness.

Bypasses the non-deterministic parts of ``geak-preprocess`` (UnitTestAgent
harness generation, MCP test discovery, repo cloning) and instead accepts a
user-provided harness file, a repo path, and a kernel path (relative to repo).

The output directory is fully compatible with ``geak-orchestrate``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_from_harness(
    harness: Path,
    repo: Path,
    kernel_relpath: str,
    output_dir: Path,
    gpu_id: int = 0,
    *,
    console=None,
) -> dict[str, Any]:
    """Build a preprocess directory from an existing test harness.

    Parameters
    ----------
    harness:
        Path to a test harness script (must support --correctness,
        --profile, --benchmark, --full-benchmark).
    repo:
        Path to the repository checkout (e.g. ``/workspace/aiter``).
    kernel_relpath:
        Kernel file path *relative* to *repo*
        (e.g. ``aiter/ops/triton/_triton_kernels/topk.py``).
    output_dir:
        Directory to write preprocessor artifacts into.
    gpu_id:
        GPU device ID for profiling and benchmarking.
    console:
        Optional Rich console for styled progress messages.
    """
    from minisweagent.run.pipeline.helpers import (
        DEFAULT_EVAL_BENCHMARK_ITERATIONS,
        execute_harness_validation,
        run_baseline_profile,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(repo).resolve()
    harness = Path(harness).resolve()
    kernel_path = (repo / kernel_relpath).resolve()

    def _print(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg, file=sys.stderr)

    if not repo.is_dir():
        raise FileNotFoundError(f"Repository not found: {repo}")
    if not harness.is_file():
        raise FileNotFoundError(f"Harness not found: {harness}")
    if not kernel_path.is_file():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    # ── 1. Copy harness into output dir ──────────────────────────────
    _print("--- Step 1/6: Copy harness ---")
    dest_harness = output_dir / harness.name
    shutil.copy2(harness, dest_harness)
    harness_abs = str(dest_harness.resolve())
    (output_dir / "harness_path.txt").write_text(harness_abs)
    _print(f"  {dest_harness}")

    # ── 2. Synthesise resolved.json ──────────────────────────────────
    _print("--- Step 2/6: Write resolved.json ---")
    resolved = {
        "is_weblink": False,
        "local_repo_path": str(repo),
        "local_file_path": str(kernel_path),
        "original_spec": str(kernel_path),
        "line_number": None,
        "line_end": None,
        "error": None,
    }
    (output_dir / "resolved.json").write_text(json.dumps(resolved, indent=2))
    _print(f"  kernel: {kernel_path}")

    # ── 3. Generate CODEBASE_CONTEXT.md ──────────────────────────────
    _print("--- Step 3/6: Codebase context ---")
    from minisweagent.run.pipeline.codebase_context import generate_codebase_context

    ctx_path = generate_codebase_context(
        repo_root=repo,
        kernel_path=kernel_path,
        output_dir=output_dir,
    )
    _print(f"  {ctx_path.name} ({ctx_path.stat().st_size} bytes)")

    # ── 4. Test discovery ──────────────────────────────────────────────
    _print("--- Step 4/6: Test discovery ---")
    from minisweagent.run.pipeline.helpers import _ensure_mcp_importable
    from minisweagent.tools.discovery_types import _infer_kernel_language

    _ensure_mcp_importable()
    from automated_test_discovery.server import discover as atd_discover

    _discover_fn = getattr(atd_discover, "fn", atd_discover)
    kernel_name = kernel_path.stem
    kernel_type = "triton" if kernel_path.suffix == ".py" else "cpp"
    try:
        discovery = _discover_fn(
            kernel_path=str(kernel_path),
            output_dir=str(output_dir),
            use_llm=False,
        )
        _print(
            f"  kernel.functions: {discovery.get('kernel', {}).get('functions', [])}"
        )
    except Exception as exc:
        _print(f"  Discovery failed ({exc}), using minimal stub")
        logger.warning("Test discovery failed: %s", exc)
        discovery = {
            "kernel": {
                "name": kernel_name,
                "type": kernel_type,
                "file": str(kernel_path),
                "functions": [],
            },
            "workspace": str(repo),
            "tests": [],
            "benchmarks": [],
            "total_tests_found": 0,
            "total_benchmarks_found": 0,
        }

    discovery["focused_test"] = {
        "focused_test_file": harness_abs,
        "focused_command": f"python {harness_abs} --correctness",
        "top_test_is_relevant": True,
        "reason": "User-provided harness",
    }
    (output_dir / "discovery.json").write_text(json.dumps(discovery, indent=2))

    # ── 5. Run harness + profile + baseline metrics ──────────────────
    _print("--- Step 5/6: Harness execution + profiling + baseline ---")

    test_command = f"python {harness_abs} --correctness"
    eval_iters = DEFAULT_EVAL_BENCHMARK_ITERATIONS
    extra = f"--iterations {eval_iters}"

    _print(f"  Running all modes (extra: {extra}) ...")
    ok, errors, results = execute_harness_validation(
        harness_abs,
        repo_root=str(repo),
        gpu_id=gpu_id,
        benchmark_extra_args=extra,
    )
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        _print(f"    --{r['mode']}: {status} ({r['duration_s']}s)")
    if not ok:
        _print(f"  WARNING: harness had failures: {errors}")

    (output_dir / "harness_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )

    for r in results:
        if r["mode"] == "benchmark" and r["success"]:
            (output_dir / "benchmark_baseline.txt").write_text(r["stdout"])
        if r["mode"] == "full-benchmark" and r["success"]:
            (output_dir / "full_benchmark_baseline.txt").write_text(r["stdout"])

    # Profiling
    _print("  Profiling (Metrix) ...")
    profiling: dict[str, Any] | None = None
    try:
        profiling = run_baseline_profile(test_command, gpu_id=gpu_id)
    except Exception as exc:
        _print(f"  Profiling failed: {exc}")
        logger.warning("Profiling failed: %s", exc, exc_info=True)

    if profiling:
        (output_dir / "profile.json").write_text(
            json.dumps(profiling, indent=2, default=str)
        )
        _print("  Profiling complete")

    # Baseline metrics
    baseline_metrics: dict[str, Any] = {}
    if profiling and profiling.get("success", True):
        try:
            from minisweagent.baseline_metrics import build_baseline_metrics

            baseline_metrics = build_baseline_metrics(profiling, include_all=True)
            _print(
                f"  Baseline: {baseline_metrics.get('duration_us', '?')} µs, "
                f"bottleneck={baseline_metrics.get('bottleneck', '?')}"
            )
        except Exception as exc:
            _print(f"  Baseline metrics failed: {exc}")
            logger.warning("Baseline metrics failed: %s", exc, exc_info=True)

    bb_path = output_dir / "benchmark_baseline.txt"
    if bb_path.exists():
        bb_text = bb_path.read_text()
        bm_val: float | None = None
        m = re.search(
            r"BENCHMARK_LATENCY_MS:\s*([\d.]+(?:e[+-]?\d+)?)", bb_text, re.IGNORECASE
        )
        if m:
            bm_val = float(m.group(1))
        if bm_val is None:
            m = re.search(
                r"median\s+(?:latency|time)[\w\s]*:\s*([\d.]+(?:e[+-]?\d+)?)\s*ms",
                bb_text,
                re.IGNORECASE,
            )
            if m:
                bm_val = float(m.group(1))
        if bm_val is not None:
            baseline_metrics["benchmark_duration_us"] = bm_val * 1000.0
        sm = re.search(r"(\d+)\s+shapes", bb_text, re.IGNORECASE)
        if sm:
            baseline_metrics["benchmark_shape_count"] = int(sm.group(1))

    if baseline_metrics:
        (output_dir / "baseline_metrics.json").write_text(
            json.dumps(baseline_metrics, indent=2, default=str)
        )

    # ── 6. Generate COMMANDMENT.md ───────────────────────────────────
    _print("--- Step 6/6: Commandment ---")
    from minisweagent.tools.commandment import generate_commandment

    kl = _infer_kernel_language(kernel_path, kernel_type)
    commandment = generate_commandment(
        kernel_path=kernel_path,
        harness_path=dest_harness,
        repo_root=repo,
        kernel_language=kl,
    )
    (output_dir / "COMMANDMENT.md").write_text(commandment)
    _print("  COMMANDMENT.md generated")

    _print("")
    _print(f"Done. Output directory: {output_dir}")
    return {
        "resolved": resolved,
        "kernel_path": str(kernel_path),
        "repo_root": str(repo),
        "harness_path": harness_abs,
        "test_command": test_command,
        "discovery": discovery,
        "profiling": profiling,
        "baseline_metrics": baseline_metrics or None,
        "commandment": commandment,
    }


def main() -> None:
    """CLI: ``geak-from-harness --harness <path> --repo <path> --kernel <relpath> -o <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build a geak-orchestrate-compatible preprocess directory "
            "from an existing test harness (no LLM, no cloning)."
        ),
    )
    parser.add_argument(
        "--harness",
        required=True,
        help="Path to the test harness script",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Path to the repository checkout (e.g. /workspace/aiter)",
    )
    parser.add_argument(
        "--kernel",
        required=True,
        help="Kernel file path relative to --repo (e.g. aiter/ops/triton/_triton_kernels/topk.py)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="geak_output",
        help="Output directory (default: geak_output)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    args = parser.parse_args()

    try:
        from rich.console import Console

        console = Console()
    except ImportError:
        console = None

    ctx = run_from_harness(
        harness=Path(args.harness),
        repo=Path(args.repo),
        kernel_relpath=args.kernel,
        output_dir=Path(args.output),
        gpu_id=args.gpu,
        console=console,
    )

    print(json.dumps(ctx, indent=2, default=str))


if __name__ == "__main__":
    main()
