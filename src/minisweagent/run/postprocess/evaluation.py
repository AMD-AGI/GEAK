"""Per-round evaluation: apply best patch, run FULL_BENCHMARK, profile, verify speedup.

After each optimization round, the orchestrator calls ``evaluate_round_best``
to independently verify the best agent's result.  This module handles:

1. Creating a clean git worktree and applying the winning patch.
2. Running the COMMANDMENT's SETUP + FULL_BENCHMARK sections.
3. Comparing candidate latency against the preprocessor baseline.
4. Profiling the patched kernel and comparing against baseline metrics.
5. Detecting benchmark config mismatches (agent-modified shapes/params).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from minisweagent.run.postprocess.benchmark_parsing import (
    extract_benchmark_config_lines as _extract_benchmark_config_lines,
    extract_latency_ms as _extract_latency_ms,
    extract_reported_speedup as _extract_reported_speedup,
    parse_shape_count as _parse_shape_count,
    parse_total_kernel_time_ms as _parse_total_kernel_time_ms,
)
from minisweagent.run.utils.generated_artifacts import (
    apply_patch_with_generated_helper_fallback,
)
from minisweagent.run.utils.git_safe_env import get_git_safe_env

logger = logging.getLogger(__name__)


class PatchApplyError(Exception):
    """Raised when a patch fails to apply to the evaluation worktree."""
    pass


def setup_eval_worktree(repo_root: str, patch_file: str, output_dir: Path) -> Path:
    """Create a temporary worktree and apply the best patch.

    Returns the worktree path.  The caller is responsible for cleanup
    via ``cleanup_eval_worktree``.

    Raises:
        PatchApplyError: If the patch fails to apply.
    """
    eval_dir = (output_dir / "_eval_worktree").resolve()
    if eval_dir.exists():
        shutil.rmtree(eval_dir, ignore_errors=True)

    repo = Path(repo_root).resolve()
    is_git = (repo / ".git").exists() or (repo / ".git").is_file()

    git_env = get_git_safe_env(output_dir)
    if is_git:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(eval_dir)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
            env=git_env,
        )
    else:
        shutil.copytree(str(repo), str(eval_dir), dirs_exist_ok=True)

    patch_path = Path(patch_file)
    if patch_path.exists() and patch_path.stat().st_size > 0:
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        apply_result, removed_paths = apply_patch_with_generated_helper_fallback(
            patch_text=patch_text,
            cwd=eval_dir,
            env=git_env,
        )
        if removed_paths:
            logger.warning(
                "Retrying evaluation patch apply without generated helper artifacts: %s",
                ", ".join(removed_paths[:5]),
            )
        if apply_result.returncode != 0:
            error_msg = f"git apply failed (rc={apply_result.returncode}): {apply_result.stderr[:500]}"
            logger.warning(error_msg)
            cleanup_eval_worktree(repo_root, eval_dir)
            raise PatchApplyError(error_msg)
    return eval_dir


def cleanup_eval_worktree(repo_root: str, eval_dir: Path) -> None:
    """Remove the temporary evaluation worktree."""
    repo = Path(repo_root).resolve()
    is_git = (repo / ".git").exists() or (repo / ".git").is_file()
    if is_git:
        git_env = get_git_safe_env(eval_dir.parent)
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(eval_dir)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            env=git_env,
        )
    if eval_dir.exists():
        shutil.rmtree(eval_dir, ignore_errors=True)


def build_eval_env(
    work_dir: Path,
    repo_root: str,
    harness_path: str,
    gpu_id: int,
    *,
    benchmark_iterations: int | None = None,
) -> dict[str, str]:
    """Build the GEAK_* environment dict for evaluation subprocesses.

    ``benchmark_iterations`` overrides the default iteration count used by
    BENCHMARK / FULL_BENCHMARK commands in the COMMANDMENT.  When ``None``
    the shared ``DEFAULT_EVAL_BENCHMARK_ITERATIONS`` is used.
    """
    from minisweagent.run.pipeline_helpers import DEFAULT_EVAL_BENCHMARK_ITERATIONS

    iters = benchmark_iterations or DEFAULT_EVAL_BENCHMARK_ITERATIONS
    env = os.environ.copy()
    env["GEAK_WORK_DIR"] = str(work_dir)
    env["GEAK_REPO_ROOT"] = repo_root
    env["GEAK_HARNESS"] = harness_path
    env["GEAK_GPU_DEVICE"] = str(gpu_id)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["GEAK_BENCHMARK_EXTRA_ARGS"] = f"--iterations {iters}"
    env["PYTHONPATH"] = f"{work_dir}:{repo_root}:{env.get('PYTHONPATH', '')}"
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in alloc_conf:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return env


def build_eval_script(
    commandment_path: str,
    sections: list[str],
) -> str | None:
    """Build a shell script from one or more COMMANDMENT sections.

    Returns the path to the written script, or None if no commands.
    """
    from minisweagent.run.dispatch import _read_commandment_section

    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    has_commands = False
    for sec in sections:
        body = _read_commandment_section(commandment_path, sec)
        if body:
            lines.append(f"# --- {sec} ---")
            lines.append(body)
            has_commands = True
    if not has_commands:
        return None
    script_dir = Path(commandment_path).parent
    script_path = script_dir / "_geak_eval_cmd.sh"
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    return str(script_path)


def evaluate_round_best(
    ctx: dict[str, Any],
    round_num: int,
    results_dir: Path,
    _print,
) -> dict[str, Any] | None:
    """Evaluate the single best kernel from a round with FULL_BENCHMARK + PROFILE.

    Creates a temporary worktree, applies the best patch, sets all GEAK_*
    env vars, runs SETUP + FULL_BENCHMARK, then profiles with PYTHONPATH
    pointing at the patched worktree.

    Returns a round evaluation dict, or None if no valid candidates exist.
    """
    output_dir = Path(ctx["output_dir"])
    pp_dir = Path(ctx.get("preprocess_dir", ctx.get("output_dir", ".")))

    best_patch_file: str | None = None
    best_speedup: float = 0.0
    best_task: str = ""
    best_kernel_time: float = float("inf")

    if not results_dir.is_dir():
        return None

    candidates: list[dict[str, Any]] = []
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name in ("worktrees",):
            continue
        br_file = task_dir / "best_results.json"
        if not br_file.exists():
            continue
        try:
            br = json.loads(br_file.read_text())
            speedup = float(br.get("best_patch_speedup", 0))
            patch_file = br.get("best_patch_file")
            if not patch_file or speedup <= 0:
                continue

            kernel_time: float | None = None
            test_output_path = br.get("best_patch_test_output", "")
            if test_output_path:
                test_path = Path(test_output_path)
                if test_path.exists():
                    kernel_time = _parse_total_kernel_time_ms(
                        test_path.read_text()
                    )

            candidates.append({
                "task": task_dir.name,
                "patch_file": patch_file,
                "speedup": speedup,
                "kernel_time_ms": kernel_time,
                "per_shape_speedups": br.get("per_shape_speedups") or {},
                "baseline_shape_latency_ms": br.get("baseline_shape_latency_ms") or {},
                "candidate_shape_latency_ms": br.get("candidate_shape_latency_ms") or {},
            })
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    if not candidates:
        _print(f"  Round {round_num}: no valid candidates for evaluation")
        return None

    all_have_kernel_time = all(
        c["kernel_time_ms"] is not None for c in candidates
    )

    if all_have_kernel_time:
        best = min(candidates, key=lambda c: c["kernel_time_ms"])  # type: ignore[arg-type]
    else:
        best = max(candidates, key=lambda c: c["speedup"])

    best_task = best["task"]
    best_patch_file = best["patch_file"]
    best_speedup = best["speedup"]
    if best["kernel_time_ms"] is not None:
        best_kernel_time = best["kernel_time_ms"]

    selection_method = "kernel_time" if all_have_kernel_time else "speedup"
    if best_kernel_time < float("inf"):
        _print(
            f"  Round {round_num} best: {best_task} "
            f"({best_speedup:.2f}x, {best_kernel_time:.4f}ms, "
            f"selected by {selection_method})"
        )
    else:
        _print(f"  Round {round_num} best: {best_task} ({best_speedup:.2f}x)")

    round_eval: dict[str, Any] = {
        "round": round_num,
        "best_patch": best_patch_file,
        "best_task": best_task,
        "benchmark_speedup": best_speedup,
    }
    if best.get("per_shape_speedups"):
        round_eval["per_shape_speedups"] = best["per_shape_speedups"]
    if best.get("baseline_shape_latency_ms"):
        round_eval["baseline_shape_latency_ms"] = best["baseline_shape_latency_ms"]
    if best.get("candidate_shape_latency_ms"):
        round_eval["candidate_shape_latency_ms"] = best["candidate_shape_latency_ms"]

    commandment_path = pp_dir / "COMMANDMENT.md"
    if not commandment_path.exists():
        eval_path = output_dir / f"round_{round_num}_evaluation.json"
        eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
        return round_eval

    repo_root = ctx.get("repo_root", "")
    harness_path = ctx.get("harness_path", "")
    gpu_id = ctx.get("gpu_ids", [0])[0]

    eval_worktree: Path | None = None
    try:
        try:
            eval_worktree = setup_eval_worktree(repo_root, best_patch_file, output_dir)
        except PatchApplyError as exc:
            _print(f"  Patch apply failed: {exc}")
            round_eval["patch_apply_error"] = str(exc)
            round_eval["status"] = "patch_failed"
            eval_path = output_dir / f"round_{round_num}_evaluation.json"
            eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
            return round_eval

        eval_harness_path = harness_path
        if harness_path and eval_worktree:
            harness_name = Path(harness_path).name
            eval_harness = eval_worktree / harness_name
            if eval_harness.exists():
                eval_harness_path = str(eval_harness)

        eval_env = build_eval_env(eval_worktree, repo_root, eval_harness_path, gpu_id)
        _print(f"  Eval worktree: {eval_worktree}")

        full_benchmark_baseline_path = pp_dir / "full_benchmark_baseline.txt"
        full_benchmark_baseline = (
            full_benchmark_baseline_path.read_text().strip()
            if full_benchmark_baseline_path.exists()
            else None
        )
        benchmark_baseline_path = pp_dir / "benchmark_baseline.txt"
        benchmark_baseline = (
            benchmark_baseline_path.read_text().strip()
            if benchmark_baseline_path.exists()
            else None
        )

        # --- FULL_BENCHMARK ---
        fb_script = build_eval_script(str(commandment_path), ["SETUP", "FULL_BENCHMARK"])
        if fb_script:
            _print(f"  Running FULL_BENCHMARK on best kernel from round {round_num}...")
            try:
                fb_result = subprocess.run(
                    ["bash", fb_script],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    cwd=str(eval_worktree),
                    env=eval_env,
                )
                fb_stdout = fb_result.stdout
                round_eval["full_benchmark"] = {
                    "stdout": fb_stdout[:5000],
                    "returncode": fb_result.returncode,
                    "success": fb_result.returncode == 0,
                }
                if full_benchmark_baseline:
                    round_eval["full_benchmark"]["baseline"] = full_benchmark_baseline[:2000]

                if fb_result.returncode == 0:
                    candidate_ms = _extract_latency_ms(fb_stdout)
                    baseline_ref = full_benchmark_baseline or benchmark_baseline
                    baseline_ms = _extract_latency_ms(baseline_ref) if baseline_ref else None
                    if candidate_ms and baseline_ms and baseline_ms > 0:
                        verified_speedup = baseline_ms / candidate_ms
                        round_eval["full_benchmark"]["verified_speedup"] = round(verified_speedup, 4)
                        round_eval["full_benchmark"]["candidate_ms"] = candidate_ms
                        round_eval["full_benchmark"]["baseline_ms"] = baseline_ms
                        _print(
                            f"  FULL_BENCHMARK verified speedup: {verified_speedup:.4f}x "
                            f"({baseline_ms:.4f} ms -> {candidate_ms:.4f} ms)"
                        )
                    else:
                        candidate_reported_speedup = _extract_reported_speedup(fb_stdout)
                        baseline_reported_speedup = (
                            _extract_reported_speedup(baseline_ref) if baseline_ref else None
                        )
                        if (
                            isinstance(candidate_reported_speedup, (int, float))
                            and isinstance(baseline_reported_speedup, (int, float))
                            and baseline_reported_speedup > 0
                        ):
                            verified_speedup = (
                                float(candidate_reported_speedup)
                                / float(baseline_reported_speedup)
                            )
                            round_eval["full_benchmark"]["verified_speedup"] = round(verified_speedup, 4)
                            round_eval["full_benchmark"]["candidate_reported_speedup"] = round(
                                float(candidate_reported_speedup), 6
                            )
                            round_eval["full_benchmark"]["baseline_reported_speedup"] = round(
                                float(baseline_reported_speedup), 6
                            )
                            _print(
                                "  FULL_BENCHMARK verified speedup: "
                                f"{verified_speedup:.4f}x "
                                f"(reported speedup {baseline_reported_speedup:.4f}x "
                                f"-> {candidate_reported_speedup:.4f}x)"
                            )
                    candidate_configs = _extract_benchmark_config_lines(fb_stdout)
                    baseline_configs = _extract_benchmark_config_lines(baseline_ref) if baseline_ref else None
                    if candidate_configs and baseline_configs:
                        if candidate_configs != baseline_configs:
                            _print(
                                f"  WARNING: Benchmark config mismatch detected! "
                                f"Agent may have modified benchmark parameters. "
                                f"Rejecting speedup."
                            )
                            logger.warning(
                                "Benchmark config mismatch: agent modified benchmark configs. "
                                "baseline_configs=%d lines, candidate_configs=%d lines",
                                len(baseline_configs), len(candidate_configs),
                            )
                            round_eval["full_benchmark"]["config_mismatch"] = True
                            round_eval["full_benchmark"]["config_mismatch_detail"] = (
                                f"baseline={len(baseline_configs)} configs, "
                                f"candidate={len(candidate_configs)} configs"
                            )
                            round_eval["full_benchmark"].pop("verified_speedup", None)
                            _print("  Verified speedup INVALIDATED due to config mismatch")
                    elif candidate_configs or baseline_configs:
                        candidate_shapes = _parse_shape_count(fb_stdout)
                        baseline_shapes = _parse_shape_count(baseline_ref) if baseline_ref else None
                        if candidate_shapes and baseline_shapes and candidate_shapes != baseline_shapes:
                            logger.warning(
                                "Shape count mismatch: baseline=%d, candidate=%d",
                                baseline_shapes, candidate_shapes,
                            )
                            round_eval["full_benchmark"]["shape_count_warning"] = (
                                f"baseline={baseline_shapes}, candidate={candidate_shapes}"
                            )

                _print(f"  FULL_BENCHMARK: {'PASS' if fb_result.returncode == 0 else 'FAIL'}")
            except Exception as exc:
                _print(f"  FULL_BENCHMARK failed: {exc}")
                round_eval["full_benchmark"] = {"error": str(exc)}

        # --- PROFILE ---
        _print(f"  Running PROFILE on best kernel from round {round_num}...")
        baseline_metrics_path = pp_dir / "baseline_metrics.json"
        baseline_metrics = None
        if baseline_metrics_path.exists():
            try:
                baseline_metrics = json.loads(baseline_metrics_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        try:
            from minisweagent.run.pipeline_helpers import _ensure_mcp_importable

            _ensure_mcp_importable()
            from profiler_mcp.server import profile_kernel

            _profile_fn = getattr(profile_kernel, "fn", profile_kernel)
            if harness_path:
                prev_pythonpath = os.environ.get("PYTHONPATH", "")
                os.environ["PYTHONPATH"] = f"{eval_worktree}:{repo_root}:{prev_pythonpath}"
                try:
                    profile_result = _profile_fn(
                        command=f"python {harness_path} --profile",
                        backend="metrix",
                        num_replays=3,
                        quick=True,
                        gpu_devices=str(gpu_id),
                    )
                finally:
                    if prev_pythonpath:
                        os.environ["PYTHONPATH"] = prev_pythonpath
                    else:
                        os.environ.pop("PYTHONPATH", None)

                if baseline_metrics and profile_result:
                    from minisweagent.run.preprocess.baseline import build_baseline_metrics

                    optimized_metrics = build_baseline_metrics(
                        profile_result, include_all=True
                    )
                    profile_comparison: dict[str, Any] = {}
                    for key in ("duration_us", "bottleneck"):
                        if key in baseline_metrics and key in optimized_metrics:
                            base_val = baseline_metrics[key]
                            opt_val = optimized_metrics[key]
                            if key == "duration_us" and isinstance(base_val, (int, float)):
                                change_pct = ((opt_val - base_val) / base_val * 100) if base_val else 0
                                profile_comparison[key] = {
                                    "baseline": base_val,
                                    "optimized": opt_val,
                                    "change_pct": round(change_pct, 1),
                                }
                            else:
                                profile_comparison[key] = {
                                    "baseline": base_val,
                                    "optimized": opt_val,
                                }

                    opt_bn = optimized_metrics.get("bottleneck", "unknown")
                    base_bn = baseline_metrics.get("bottleneck", "unknown")
                    if base_bn != opt_bn:
                        profile_comparison["bottleneck_shift"] = f"{base_bn} -> {opt_bn}"

                    round_eval["profile_comparison"] = profile_comparison
                    _print(f"  PROFILE comparison: {json.dumps(profile_comparison)[:300]}")
                else:
                    _print("  PROFILE: completed (no baseline for comparison)")
        except Exception as exc:
            _print(f"  PROFILE failed: {exc}")
            round_eval["profile_comparison"] = {"error": str(exc)}

    finally:
        if eval_worktree:
            cleanup_eval_worktree(repo_root, eval_worktree)

    eval_path = output_dir / f"round_{round_num}_evaluation.json"
    eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
    _print(f"  Round evaluation written to: {eval_path}")

    return round_eval
