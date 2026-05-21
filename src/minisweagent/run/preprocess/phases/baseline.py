"""Baseline phase — correctness pass + profile + per-op metrics.

Inputs  (read from ctx):
  - kernel_path, repo_root, output_dir, gpu_id
  - harness_path, test_command, eval_command, correctness_command,
    performance_command, benchmark_timeout
  - profiling (when a preceding phase already ran the profiler)

Outputs (written to ctx):
  - profiling (dict or None)
  - benchmark_baseline, full_benchmark_baseline (str stdout)
  - baseline_metrics (dict)
  - baseline_metrics_path (str)

Absorbs steps 5 + 6 of the legacy monolith.  For the explicit
``--harness`` path (where HarnessPhase already populated
``test_command`` + ``harness_path``) this phase:
  1. Runs the harness in ``--benchmark`` mode to capture baseline
     latency into ``benchmark_baseline.txt``.
  2. Calls the profiler-mcp on the harness in ``--profile`` mode.
  3. Builds baseline_metrics (profiler data + harness latency merge).

For the eval-command path, the phase:
  1. Runs ``correctness_command`` as a gate (fails fast on rc≠0).
  2. Runs ``performance_command`` under the profiler for
     ``profile.json``, then again unwrapped to capture
     ``benchmark_baseline.txt``.
  3. Builds baseline_metrics the same way.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from minisweagent.run.preprocess.benchmark_parsing import extract_latency_ms
from minisweagent.run.preprocess.phases.base import Phase, PhaseContext
from minisweagent.run.preprocess.repo_paths import ensure_preprocess_mcp_importable

logger = logging.getLogger(__name__)


def _ensure_mcp_importable() -> None:
    ensure_preprocess_mcp_importable(
        "mcp_tools/profiler-mcp/src",
        "mcp_tools/metrix-mcp/src",
        "mcp_tools/automated-test-discovery/src",
    )


def _join_cmd(cmd: str | list[str] | None) -> str | None:
    if cmd is None:
        return None
    if isinstance(cmd, list):
        return " && ".join(c.strip() for c in cmd if c.strip()) or None
    return cmd.strip() or None


def _run_shell(
    cmd: str,
    *,
    cwd: str | None,
    timeout: int,
    output_dir: Path,
    tag: str,
) -> subprocess.CompletedProcess:
    """Run a shell command and dump stdout/stderr to named sidecar files."""
    result = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    (output_dir / f"{tag}_stdout.txt").write_text(result.stdout or "")
    (output_dir / f"{tag}_stderr.txt").write_text(result.stderr or "")
    return result


def _call_profiler(perf_cmd: str, *, cwd: str | None, gpu_id: int) -> dict[str, Any] | None:
    """Invoke profiler-mcp's ``profile_kernel`` with metrix backend."""
    try:
        _ensure_mcp_importable()
        profiler_server = importlib.import_module("profiler_mcp.server")
        profile_kernel = profiler_server.profile_kernel
        profile_fn = getattr(profile_kernel, "fn", profile_kernel)
        return profile_fn(
            command=perf_cmd,
            backend="metrix",
            num_replays=3,
            quick=False,
            gpu_devices=str(gpu_id),
            workdir=cwd,
        )
    except Exception as exc:
        logger.warning("[yellow]Profiling failed: %s[/yellow]", exc, exc_info=True)
        return None


def _enrich_metrics_with_benchmark(
    baseline_metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Overlay harness-measured wall-clock latency onto profiler metrics.

    Consumers downstream compare baseline-vs-candidate benchmarks, not
    Metrix profile durations — so we standardise ``duration_us`` on
    the harness-measured value and preserve the profiler duration as
    ``profiler_duration_us``.
    """
    bb_path = output_dir / "benchmark_baseline.txt"
    if not bb_path.exists():
        return baseline_metrics
    bb_text = bb_path.read_text()
    bm_val = extract_latency_ms(bb_text)
    if bm_val is None:
        logger.warning(
            "Could not parse benchmark latency from %s — "
            "verified speedups will not be available for this kernel. "
            "Output preview: %s",
            bb_path,
            bb_text[:200].replace("\n", " "),
        )
    if bm_val is not None:
        baseline_metrics["benchmark_duration_us"] = bm_val * 1000.0
        if "duration_us" in baseline_metrics:
            baseline_metrics["profiler_duration_us"] = baseline_metrics["duration_us"]
        baseline_metrics["duration_us"] = bm_val * 1000.0
        profiler_us = baseline_metrics.get("profiler_duration_us")
        if profiler_us and profiler_us > 0:
            baseline_metrics["benchmark_profiler_ratio"] = round(
                baseline_metrics["benchmark_duration_us"] / profiler_us, 2
            )
    sm = re.search(r"(\d+)\s+shapes", bb_text, re.IGNORECASE)
    if sm:
        baseline_metrics["benchmark_shape_count"] = int(sm.group(1))
    return baseline_metrics


class BaselinePhase(Phase):
    """Capture baseline correctness, profile, and per-op metrics."""

    name = "baseline"

    def run(self, ctx: PhaseContext) -> None:
        self._log_enter()

        # Require the harness step to have produced a path before us.
        # If upstream phases haven't populated it (transition period),
        # the orchestrator's legacy fallback handles the pipeline
        # instead of us running half-way.
        if not ctx.harness_path and not ctx.test_command and not ctx.eval_command:
            logger.debug("BaselinePhase: no harness/test/eval command set; deferring to legacy fallback.")
            return

        output_dir = Path(ctx.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cwd = ctx.repo_root or None

        correctness_cmd = _join_cmd(ctx.correctness_command)
        perf_cmd = _join_cmd(ctx.performance_command)
        # Backfill from the legacy single eval_command if the structured pair wasn't given.
        if not correctness_cmd and not perf_cmd and ctx.eval_command:
            # Mirrors the legacy preprocessor's behaviour: with a single
            # eval_command we treat it as the performance command.
            perf_cmd = ctx.eval_command.strip() or None

        # ── Profile ─────────────────────────────────────────────────
        profile_t0 = time.monotonic()
        profiling: dict[str, Any] | None = None
        benchmark_baseline: str | None = None
        full_benchmark_baseline: str | None = None

        if ctx.eval_command:
            # eval_command path: optional correctness gate + performance run
            if correctness_cmd:
                logger.info("  Running correctness_command: %s", correctness_cmd)
                result = _run_shell(
                    correctness_cmd,
                    cwd=cwd,
                    timeout=3600,
                    output_dir=output_dir,
                    tag="correctness",
                )
                # §13.2-A row 1: populate ctx.correctness so downstream
                # consumers (reports, debugging) have the same structure
                # the legacy monolith produced.
                ctx.correctness = {
                    "command": correctness_cmd,
                    "returncode": result.returncode,
                    "stdout_path": str(output_dir / "correctness_stdout.txt"),
                    "stderr_path": str(output_dir / "correctness_stderr.txt"),
                }
                if result.returncode != 0:
                    raise RuntimeError(
                        f"correctness_command failed (returncode={result.returncode}). "
                        f"See {output_dir / 'correctness_stderr.txt'}"
                    )

            if not perf_cmd:
                logger.info("  Skipping profiling (no performance_command)")
            else:
                logger.info("  Profiling with performance_command: %s", perf_cmd)
                profiling = _call_profiler(perf_cmd, cwd=cwd, gpu_id=ctx.gpu_id)

                logger.info("  Capturing benchmark baseline...")
                try:
                    bench_result = subprocess.run(
                        perf_cmd,
                        shell=True,
                        executable="/bin/bash",
                        capture_output=True,
                        text=True,
                        timeout=ctx.benchmark_timeout,
                        cwd=cwd,
                    )
                    if bench_result.returncode == 0:
                        benchmark_baseline = bench_result.stdout
                        full_benchmark_baseline = bench_result.stdout
                        (output_dir / "benchmark_baseline.txt").write_text(bench_result.stdout)
                        (output_dir / "full_benchmark_baseline.txt").write_text(bench_result.stdout)
                        logger.info("  Baseline saved (%d bytes)", len(bench_result.stdout))
                    else:
                        logger.warning("  Baseline capture: FAILED (rc=%d)", bench_result.returncode)
                        if bench_result.stderr:
                            logger.warning("  stderr: %s", bench_result.stderr[:500])
                except Exception as exc:
                    logger.warning("Baseline capture failed: %s", exc, exc_info=True)
        elif ctx.test_command:
            # test_command path: use the legacy run_baseline_profile helper
            # (reads the harness, runs it under the profiler).
            from minisweagent.run.preprocess.harness_utils import (
                DEFAULT_EVAL_BENCHMARK_ITERATIONS,
                execute_harness_validation,
                extract_harness_path,
                run_baseline_profile,
            )

            if not ctx.harness_path:
                ctx.harness_path = extract_harness_path(ctx.test_command)
                (output_dir / "harness_path.txt").write_text(ctx.harness_path)

            # §13.2-A row 4: canonical harness baseline re-run.  Legacy
            # preprocessor.py:988-1027 runs the harness in all four modes
            # with ``--iterations DEFAULT_EVAL_BENCHMARK_ITERATIONS`` BEFORE
            # profiling, to capture a benchmark baseline that uses the same
            # iteration count as the orchestrator evaluation.  We preserve
            # that behaviour here so speedups stay benchmark-vs-benchmark
            # on a consistent contract.
            harness_path_for_baseline = ctx.harness_path or extract_harness_path(ctx.test_command)
            if harness_path_for_baseline and ctx.harness_results:
                logger.info(
                    "  Canonical baseline re-run: all modes with --iterations %d",
                    DEFAULT_EVAL_BENCHMARK_ITERATIONS,
                )
                try:
                    bl_ok, bl_errors, baseline_results = execute_harness_validation(
                        harness_path_for_baseline,
                        repo_root=ctx.repo_root,
                        gpu_id=ctx.gpu_id,
                        benchmark_extra_args=f"--iterations {DEFAULT_EVAL_BENCHMARK_ITERATIONS}",
                    )
                    for r in baseline_results:
                        status = "PASS" if r["success"] else "FAIL"
                        logger.info("    --%s: %s (%ss)", r["mode"], status, r["duration_s"])
                    if not bl_ok:
                        logger.warning("  Baseline re-run had failures: %s", bl_errors)
                    for r in baseline_results:
                        if r["mode"] == "benchmark" and r["success"]:
                            benchmark_baseline = r["stdout"]
                        if r["mode"] == "full-benchmark" and r["success"]:
                            full_benchmark_baseline = r["stdout"]
                except Exception as exc:
                    logger.warning(
                        "[yellow]Canonical baseline re-run failed: %s[/yellow]",
                        exc,
                        exc_info=True,
                    )
            elif ctx.harness_results:
                # Fallback: use the stdout captured during HarnessPhase's
                # execute_harness_validation call.  Matches legacy
                # preprocessor.py:1012-1017.
                for r in ctx.harness_results:
                    if r.get("mode") == "benchmark" and r.get("success"):
                        benchmark_baseline = r.get("stdout")
                    if r.get("mode") == "full-benchmark" and r.get("success"):
                        full_benchmark_baseline = r.get("stdout")

            # Canonicalize: prefer full-benchmark stdout over plain
            # benchmark stdout; write BOTH files for downstream consumers.
            canonical = full_benchmark_baseline or benchmark_baseline
            if canonical:
                benchmark_baseline = canonical
                full_benchmark_baseline = canonical
                (output_dir / "benchmark_baseline.txt").write_text(canonical)
                (output_dir / "full_benchmark_baseline.txt").write_text(canonical)

            try:
                profiling = run_baseline_profile(ctx.test_command, gpu_id=ctx.gpu_id)
            except Exception as exc:
                logger.warning("[yellow]Profiling failed: %s[/yellow]", exc, exc_info=True)

        ctx.profiling = profiling
        ctx.benchmark_baseline = benchmark_baseline
        ctx.full_benchmark_baseline = full_benchmark_baseline
        profile_elapsed = time.monotonic() - profile_t0

        if profiling:
            (output_dir / "profile.json").write_text(json.dumps(profiling, indent=2, default=str))
            if ctx.repo_root:
                (Path(ctx.repo_root) / "profile.json").write_text(json.dumps(profiling, indent=2, default=str))
                logger.info(
                    "  Profiling complete in %.0fs (also saved to %s/profile.json)",
                    profile_elapsed,
                    ctx.repo_root,
                )
            else:
                logger.info("  Profiling complete in %.0fs", profile_elapsed)

        # ── Baseline metrics ────────────────────────────────────────
        baseline_metrics: dict[str, Any] | None = None
        if profiling and profiling.get("success", True):
            try:
                from minisweagent.run.preprocess.baseline import build_baseline_metrics

                baseline_metrics = build_baseline_metrics(profiling, include_all=True)
                dur = baseline_metrics.get("duration_us", "?")
                bn = baseline_metrics.get("bottleneck", "?")
                logger.info("  Baseline: %s µs, bottleneck=%s", dur, bn)
            except Exception as exc:
                logger.warning("[yellow]Baseline metrics build failed: %s[/yellow]", exc, exc_info=True)
        else:
            logger.info("  Skipping baseline metrics (no profiling data)")

        if baseline_metrics is None:
            baseline_metrics = {}
        baseline_metrics = _enrich_metrics_with_benchmark(baseline_metrics, output_dir)

        ctx.baseline_metrics = baseline_metrics if baseline_metrics else None
        if baseline_metrics:
            metrics_path = output_dir / "baseline_metrics.json"
            metrics_path.write_text(json.dumps(baseline_metrics, indent=2, default=str))
            ctx.baseline_metrics_path = str(metrics_path)
            if ctx.repo_root:
                repo_metrics_path = Path(ctx.repo_root) / "baseline_metrics.json"
                repo_metrics_path.write_text(json.dumps(baseline_metrics, indent=2, default=str))
                logger.info("  Baseline metrics saved to %s", repo_metrics_path)

        ctx.phases_run.append(self.name)


__all__ = ["BaselinePhase"]
