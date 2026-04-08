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
    extract_benchmark_config_lines,
    extract_latency_ms,
    parse_shape_count,
    parse_total_kernel_time_ms,
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

    For git repos, creates a detached worktree.  For non-git directories,
    copies the tree and initialises a temporary git repo so that
    ``git apply`` works uniformly.

    Returns the worktree path.  The caller is responsible for cleanup
    via ``cleanup_eval_worktree``.

    Raises:
        PatchApplyError: If the patch fails to apply.
    """
    patch_path = Path(patch_file)
    if not patch_path.exists():
        raise PatchApplyError(f"Patch file does not exist: {patch_file}")
    if patch_path.stat().st_size == 0:
        raise PatchApplyError(f"Patch file is empty: {patch_file}")

    eval_dir = (output_dir / "_eval_worktree").resolve()
    if eval_dir.exists():
        logger.warning("Removing existing evaluation worktree: %s", eval_dir)
        shutil.rmtree(eval_dir, ignore_errors=True)

    repo = Path(repo_root).resolve()
    is_git = (repo / ".git").exists()

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
        subprocess.run(["git", "init"], cwd=str(eval_dir), capture_output=True, text=True, check=True, env=git_env)
        subprocess.run(["git", "add", "."], cwd=str(eval_dir), capture_output=True, text=True, check=True, env=git_env)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=str(eval_dir),
            capture_output=True,
            text=True,
            check=True,
            env=git_env,
        )
        logger.warning("Initialised temporary git repo in non-git eval worktree: %s", eval_dir)

    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    # errors="replace" is the Unicode error handling mode for str.decode()
    apply_result, removed_paths = apply_patch_with_generated_helper_fallback(
        patch_text=patch_text,
        cwd=eval_dir,
        env=git_env,
    )
    if removed_paths:
        logger.warning(
            "Retrying evaluation patch apply without generated helper artifacts: %s",
            ", ".join(removed_paths),
        )
    if apply_result.returncode != 0:
        error_msg = f"git apply failed (rc={apply_result.returncode}): {apply_result.stderr}"
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
    script_path = (script_dir / "_geak_eval_cmd.sh").resolve()
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    return str(script_path)


def resolve_eval_worktree(
    repo_root: str,
    best_patch_file: str,
    harness_path: str,
    output_dir: Path,
    gpu_id: int,
) -> tuple[Path, dict[str, str]]:
    """Create a clean evaluation worktree, apply the patch, build env dict.

    Returns ``(eval_worktree, eval_env)``.
    Raises ``PatchApplyError`` if the patch fails to apply.
    """
    eval_worktree = setup_eval_worktree(repo_root, best_patch_file, output_dir)
    logger.info("Eval worktree: %s", eval_worktree)

    # The harness_path comes from the original location. We assume it does not
    # import the kernel from relative locations.
    eval_env = build_eval_env(eval_worktree, repo_root, harness_path, gpu_id)
    return eval_worktree, eval_env


def run_correctness_and_benchmark(
    eval_worktree: Path,
    eval_env: dict[str, str],
    commandment_path: Path,
    pp_dir: Path,
    round_eval: dict[str, Any],
    round_num: int,
) -> None:
    """Run CORRECTNESS then FULL_BENCHMARK, compute verified speedup.

    Runs the COMMANDMENT CORRECTNESS section first as a safety gate.
    If correctness fails, the benchmark is skipped.
    Falls back to BENCHMARK if FULL_BENCHMARK baseline is not found.
    """
    correctness_script = build_eval_script(str(commandment_path), ["SETUP", "CORRECTNESS"])
    if correctness_script:
        logger.info("Running CORRECTNESS on best kernel from round %d...", round_num)
        try:
            correctness_result = subprocess.run(
                ["bash", correctness_script],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(eval_worktree),
                env=eval_env,
            )
        except Exception as exc:
            logger.warning("CORRECTNESS execution failed: %s", exc)
            round_eval["correctness"] = {"error": str(exc)}
            round_eval["status"] = "correctness_failed"
            return

        round_eval["correctness"] = {
            "returncode": correctness_result.returncode,
            "success": correctness_result.returncode == 0,
        }
        if correctness_result.returncode != 0:
            logger.warning(
                "CORRECTNESS failed (rc=%d): %s",
                correctness_result.returncode,
                correctness_result.stderr,
            )
            round_eval["status"] = "correctness_failed"
            return
        logger.info("CORRECTNESS: PASS")
    else:
        logger.warning("No CORRECTNESS commands found in COMMANDMENT")

    for baseline_section_name in ["FULL_BENCHMARK", "BENCHMARK"]:
        section_key = baseline_section_name.lower()
        baseline_path = pp_dir / (section_key + "_baseline.txt")

        if not baseline_path.exists():
            logger.warning("%s does not exist", baseline_path)
            continue

        baseline_text = baseline_path.read_text().strip()
        logger.info("%s baseline found: %s", section_key, baseline_path)

        benchmark_script = build_eval_script(str(commandment_path), ["SETUP", baseline_section_name])
        if not benchmark_script:
            logger.warning("No %s commands found in COMMANDMENT", baseline_section_name)
            continue

        logger.info("Running %s on best kernel from round %d...", section_key, round_num)
        try:
            candidate_result = subprocess.run(
                ["bash", benchmark_script],
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(eval_worktree),
                env=eval_env,
            )
        except Exception as exc:
            logger.warning("%s execution failed: %s", section_key, exc)
            round_eval[section_key] = {"error": str(exc)}
            continue

        if candidate_result.returncode != 0:
            logger.warning("%s execution failed: %s", baseline_section_name, candidate_result.stderr)
            round_eval[section_key] = {"error": candidate_result.stderr}
            continue

        candidate_stdout = candidate_result.stdout
        round_eval[section_key] = {
            "stdout": candidate_stdout,
            "returncode": candidate_result.returncode,
            "success": candidate_result.returncode == 0,
            "baseline": baseline_text,
        }

        _check_config_mismatch(candidate_stdout, baseline_text, round_eval, section_key)
        if not round_eval[section_key].get("config_mismatch"):
            _compute_verified_speedup(candidate_stdout, baseline_text, round_eval, section_key)

        logger.info("%s: PASS", baseline_section_name)
        break

    else:
        logger.warning("No full benchmark baseline or benchmark baseline found")
        return


def _compute_verified_speedup(
    candidate_stdout: str,
    baseline_text: str,
    round_eval: dict[str, Any],
    section_key: str,
) -> None:
    """Compute verified speedup from latency measurements."""
    candidate_ms = extract_latency_ms(candidate_stdout)
    baseline_ms = extract_latency_ms(baseline_text)

    if not candidate_ms or not baseline_ms or baseline_ms <= 0:
        logger.warning("Could not extract latency: candidate_ms=%s, baseline_ms=%s", candidate_ms, baseline_ms)
        return

    verified_speedup = baseline_ms / candidate_ms
    round_eval[section_key]["verified_speedup"] = round(verified_speedup, 4)
    round_eval[section_key]["candidate_ms"] = candidate_ms
    round_eval[section_key]["baseline_ms"] = baseline_ms
    logger.info(f"  Verified speedup: {verified_speedup:.4f}x ({baseline_ms:.4f} ms -> {candidate_ms:.4f} ms)")


def _check_config_mismatch(
    candidate_stdout: str,
    baseline_text: str,
    round_eval: dict[str, Any],
    section_key: str,
) -> None:
    """Detect and flag benchmark config or shape count mismatches."""
    candidate_configs = extract_benchmark_config_lines(candidate_stdout)
    baseline_configs = extract_benchmark_config_lines(baseline_text)

    if candidate_configs and baseline_configs:
        if candidate_configs != baseline_configs:
            logger.warning(
                "Benchmark config mismatch: baseline_configs=%d lines, candidate_configs=%d lines. Rejecting speedup.",
                len(baseline_configs),
                len(candidate_configs),
            )
            round_eval[section_key]["config_mismatch"] = True
            round_eval[section_key]["config_mismatch_detail"] = (
                f"baseline={len(baseline_configs)} configs, candidate={len(candidate_configs)} configs"
            )
    elif candidate_configs or baseline_configs:
        candidate_shapes = parse_shape_count(candidate_stdout)
        baseline_shapes = parse_shape_count(baseline_text)
        if candidate_shapes and baseline_shapes and candidate_shapes != baseline_shapes:
            logger.warning(
                "Shape count mismatch: baseline=%d, candidate=%d",
                baseline_shapes,
                candidate_shapes,
            )
            round_eval[section_key]["shape_count_warning"] = f"baseline={baseline_shapes}, candidate={candidate_shapes}"


def run_profile(
    eval_worktree: Path,
    eval_env: dict[str, str],
    commandment_path: Path,
    pp_dir: Path,
    round_eval: dict[str, Any],
    round_num: int,
    results_dir: Path,
) -> None:
    """Run the COMMANDMENT PROFILE section, save output, compare against baseline.

    The COMMANDMENT's PROFILE section handles warmup and runs
    ``kernel-profile --json -o ${GEAK_WORK_DIR}/profile.json``.
    This function picks up the resulting file, copies it to *results_dir*,
    and builds a comparison against ``baseline_metrics.json``.
    Mutates ``round_eval["profile_comparison"]`` in place.
    """
    profile_script = build_eval_script(str(commandment_path), ["SETUP", "PROFILE"])
    if not profile_script:
        logger.warning("No PROFILE commands found in COMMANDMENT")
        return

    logger.info("Running PROFILE on best kernel from round %d...", round_num)
    try:
        subprocess.run(
            ["bash", profile_script],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(eval_worktree),
            env=eval_env,
        )
    except Exception as exc:
        logger.warning("PROFILE execution failed: %s", exc)
        round_eval["profile_comparison"] = {"error": str(exc)}
        return

    profile_output = eval_worktree / "profile.json"
    if not profile_output.exists():
        logger.warning("COMMANDMENT PROFILE did not produce profile.json")
        return

    dest = results_dir / "profile.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile_output, dest)
    logger.info("Raw profile saved to %s", dest)

    try:
        profile_result = json.loads(dest.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse profile output: %s", exc)
        return

    baseline_metrics_path = pp_dir / "baseline_metrics.json"
    if not baseline_metrics_path.exists():
        logger.info("PROFILE: completed (no baseline_metrics.json for comparison)")
        return

    try:
        baseline_metrics = json.loads(baseline_metrics_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to parse baseline metrics: %s", baseline_metrics_path, exc_info=True)
        return

    from minisweagent.run.preprocess.baseline import build_baseline_metrics

    optimized_metrics = build_baseline_metrics(profile_result, include_all=True)
    comparison: dict[str, Any] = {
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
    }

    base_bn = baseline_metrics.get("bottleneck", "unknown")
    opt_bn = optimized_metrics.get("bottleneck", "unknown")
    if base_bn != opt_bn:
        comparison["bottleneck_shift"] = f"{base_bn} -> {opt_bn}"

    comparison_path = results_dir / "profile_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, default=str))
    round_eval["profile_comparison"] = comparison
    logger.info("Profile comparison saved to %s", comparison_path)


def write_eval_results(
    round_eval: dict[str, Any],
    output_dir: Path,
    round_num: int,
) -> Any:
    """Write evaluation artifacts to disk and return a typed RoundEvaluation."""
    fb_raw_check = round_eval.get("full_benchmark") or round_eval.get("benchmark") or {}
    if isinstance(fb_raw_check, dict) and fb_raw_check.get("verified_speedup") is not None:
        round_eval["speedup_source"] = "FULL_BENCHMARK verified result"
    else:
        round_eval["speedup_source"] = "agent-reported benchmark (no FULL_BENCHMARK verified result available)"
    eval_path = output_dir / f"round_{round_num}_evaluation.json"
    eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
    logger.info("Round evaluation written to: %s", eval_path)

    fb_raw = round_eval.get("full_benchmark") or round_eval.get("benchmark") or {}
    if isinstance(fb_raw, dict) and fb_raw.get("stdout"):
        fb_output_path = output_dir / f"round_{round_num}_full_benchmark.txt"
        fb_output_path.write_text(fb_raw["stdout"])

    from minisweagent.run.pipeline_types import FullBenchmarkResult, RoundEvaluation

    fb_typed = None
    if isinstance(fb_raw, dict):
        failure = None
        if fb_raw.get("error"):
            failure = str(fb_raw["error"])
        elif fb_raw.get("config_mismatch"):
            failure = f"config mismatch: {fb_raw.get('config_mismatch_detail', '')}"
        elif not fb_raw.get("success", True) and fb_raw.get("returncode", 0) != 0:
            failure = f"benchmark failed (exit code {fb_raw.get('returncode')})"
        fb_typed = FullBenchmarkResult(
            verified_speedup=fb_raw.get("verified_speedup"),
            baseline_ms=fb_raw.get("baseline_ms"),
            candidate_ms=fb_raw.get("candidate_ms"),
            failure_reason=failure,
        )

    return RoundEvaluation(
        round=round_num,
        best_patch=round_eval.get("best_patch", ""),
        best_task=round_eval.get("best_task", ""),
        benchmark_speedup=round_eval.get("benchmark_speedup", 1.0),
        full_benchmark=fb_typed,
    )


def evaluate_round_best(
    ctx: dict[str, Any],
    round_num: int,
    results_dir: Path,
) -> Any:
    """Evaluate the single best kernel from a round with FULL_BENCHMARK + PROFILE.

    Creates a temporary worktree, applies the best patch, sets all GEAK_*
    env vars, runs SETUP + FULL_BENCHMARK, then profiles with PYTHONPATH
    pointing at the patched worktree.

    Returns a typed ``RoundEvaluation``, or ``None`` if no valid candidates exist.
    """
    output_dir = Path(ctx["output_dir"])
    pp_dir = Path(ctx.get("preprocess_dir", ctx.get("output_dir", ".")))

    if not results_dir.is_dir():
        logger.warning("results_dir does not exist: %s", results_dir)
        return None

    # --- Collect candidates ---
    candidates: list[dict[str, Any]] = []
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name == "worktrees":
            continue
        br_file = task_dir / "best_results.json"
        if not br_file.exists():
            logger.warning("No best_results.json in %s", task_dir.name)
            continue
        try:
            br = json.loads(br_file.read_text())
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse %s: %s", br_file, exc)
            continue

        speedup = float(br.get("best_patch_speedup", 0))
        patch_file = br.get("best_patch_file")
        if not patch_file:
            logger.warning("No patch file in %s", br_file)
            continue
        if speedup <= 0:
            logger.info("No improvement (speedup=%.4f) in %s", speedup, task_dir.name)
            continue

        kernel_time: float | None = None
        test_output_path = br.get("best_patch_test_output", "")
        if test_output_path:
            test_path = Path(test_output_path)
            if test_path.exists():
                try:
                    kernel_time = parse_total_kernel_time_ms(test_path.read_text())
                except (OSError, ValueError) as exc:
                    logger.warning("Failed to parse kernel time from %s: %s", test_output_path, exc)
            else:
                logger.warning("Test output file missing: %s", test_output_path)

        candidates.append(
            {
                "task": task_dir.name,
                "patch_file": patch_file,
                "speedup": speedup,
                "kernel_time_ms": kernel_time,
                "per_shape_speedups": br.get("per_shape_speedups") or {},
                "baseline_shape_latency_ms": br.get("baseline_shape_latency_ms") or {},
                "candidate_shape_latency_ms": br.get("candidate_shape_latency_ms") or {},
            }
        )

    if not candidates:
        logger.info("Round %d: no valid candidates for evaluation", round_num)
        no_improvement_eval = {
            "round": round_num,
            "best_patch": None,
            "best_task": None,
            "benchmark_speedup": 1.0,
            "status": "no_candidates",
        }
        eval_path = output_dir / f"round_{round_num}_evaluation.json"
        eval_path.write_text(json.dumps(no_improvement_eval, indent=2))
        return None

    # --- Select best candidate ---
    all_have_kernel_time = all(c["kernel_time_ms"] is not None for c in candidates)
    if all_have_kernel_time:
        best = min(candidates, key=lambda c: c["kernel_time_ms"])  # type: ignore[arg-type]
    else:
        best = max(candidates, key=lambda c: c["speedup"])

    best_task: str = best["task"]
    best_patch_file: str = best["patch_file"]
    best_speedup: float = best["speedup"]
    best_kernel_time: float = best["kernel_time_ms"] if best["kernel_time_ms"] is not None else float("inf")

    selection_method = "kernel_time" if all_have_kernel_time else "speedup"
    if best_kernel_time < float("inf"):
        logger.info(
            "Round %d best: %s (%.2fx, %.4fms, selected by %s)",
            round_num,
            best_task,
            best_speedup,
            best_kernel_time,
            selection_method,
        )
    else:
        logger.info("Round %d best: %s (%.2fx)", round_num, best_task, best_speedup)

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

    # --- Validate required context ---
    commandment_path = pp_dir / "COMMANDMENT.md"
    if not commandment_path.exists():
        logger.warning("COMMANDMENT.md not found at %s; skipping FULL_BENCHMARK and PROFILE", commandment_path)
        eval_path = output_dir / f"round_{round_num}_evaluation.json"
        eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
        from minisweagent.run.pipeline_types import RoundEvaluation

        return RoundEvaluation(
            round=round_num, best_patch=best_patch_file or "", best_task=best_task, benchmark_speedup=best_speedup
        )

    repo_root = ctx.get("repo_root")
    if not repo_root:
        raise ValueError("ctx['repo_root'] is required for evaluation")
    harness_path = ctx.get("harness_path", "")
    if not harness_path:
        logger.warning("No harness_path in ctx; PROFILE step will be skipped")
    gpu_id = ctx.get("gpu_ids", [0])[0]

    # --- Resolve worktree, run benchmark + profile, clean up ---
    try:
        eval_worktree, eval_env = resolve_eval_worktree(
            repo_root,
            best_patch_file,
            harness_path,
            output_dir,
            gpu_id,
        )
    except PatchApplyError as exc:
        logger.warning("Patch apply failed: %s", exc)
        round_eval["patch_apply_error"] = str(exc)
        round_eval["status"] = "patch_failed"
        eval_path = output_dir / f"round_{round_num}_evaluation.json"
        eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
        from minisweagent.run.pipeline_types import FullBenchmarkResult, RoundEvaluation

        return RoundEvaluation(
            round=round_num,
            best_patch=best_patch_file or "",
            best_task=best_task,
            benchmark_speedup=best_speedup,
            full_benchmark=FullBenchmarkResult(failure_reason=f"patch apply failed: {exc}"),
        )

    try:
        run_correctness_and_benchmark(eval_worktree, eval_env, commandment_path, pp_dir, round_eval, round_num)
        run_profile(eval_worktree, eval_env, commandment_path, pp_dir, round_eval, round_num, results_dir)
    finally:
        cleanup_eval_worktree(repo_root, eval_worktree)

    return write_eval_results(round_eval, output_dir, round_num)
