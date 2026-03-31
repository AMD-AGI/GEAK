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
)
from minisweagent.run.postprocess.benchmark_parsing import (
    extract_latency_ms as _extract_latency_ms,
)
from minisweagent.run.postprocess.benchmark_parsing import (
    extract_reported_speedup as _extract_reported_speedup,
)
from minisweagent.run.postprocess.benchmark_parsing import (
    parse_shape_count as _parse_shape_count,
)
from minisweagent.run.postprocess.benchmark_parsing import (
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


def _find_agent_worktree_slot(results_dir: Path, best_task: str, repo_root: str, output_dir: Path) -> Path | None:
    """Find the agent's worktree slot that contains the best task's modified kernel.

    Searches for a modified kernel file across all slot directories using
    two strategies:
    1. The repo task path (tasks/.../<kernel_name>/kernel.py) if derivable.
    2. Workspace-relative paths (ws_*/run_*/<task_id>_*/kernel.py) where
       agents actually perform modifications.

    The search patterns are derived from ``output_dir.name`` so they work
    for any task type, not just a specific naming convention.
    """
    import re

    worktrees_dir = results_dir / "worktrees"
    if not worktrees_dir.is_dir():
        return None

    # Extract task identifier by stripping trailing timestamp + "_logs" suffix.
    # e.g. "some_task_type_L1_kernel_name_20260329_103747_logs"
    #    -> task_id = "some_task_type_L1_kernel_name"
    dir_name = output_dir.name.replace("_logs", "")
    m = re.match(r"(.+?)_\d{8}_\d{6,}$", dir_name)
    task_id = m.group(1) if m else dir_name

    # Try to derive the repo-relative task path from the task directory
    # structure.  The "tasks/" tree mirrors the task_id with path separators
    # instead of underscores, but we cannot reliably reverse-engineer the
    # path since underscores are ambiguous.  Instead, search the repo tasks/
    # tree for a directory whose underscored name matches task_id.
    task_rel: Path | None = None
    repo = Path(repo_root)
    tasks_root = repo / "tasks"
    if tasks_root.is_dir():
        for candidate in tasks_root.rglob("kernel.py"):
            # Build the underscored equivalent of the relative path
            rel = candidate.parent.relative_to(tasks_root)
            rel_underscored = "_".join(rel.parts)
            if rel_underscored == task_id:
                task_rel = Path("tasks") / rel
                break

    # Read baseline kernel for comparison
    baseline_kernel = None
    if task_rel:
        baseline_path = repo / task_rel / "kernel.py"
        if baseline_path.exists():
            baseline_kernel = baseline_path.read_text()

    for slot_dir in sorted(worktrees_dir.glob("slot_*")):
        if not slot_dir.is_dir() or "_logs" in slot_dir.name:
            continue

        # Look for kernel.py at the specific task path
        if task_rel:
            specific_kernel = slot_dir / task_rel / "kernel.py"
            if specific_kernel.exists():
                try:
                    content = specific_kernel.read_text()
                    if baseline_kernel is None or content != baseline_kernel:
                        logger.info("Found modified kernel at %s", specific_kernel.parent)
                        return specific_kernel.parent
                except OSError:
                    pass

        # Fallback: look for kernel.py directly in slot root
        root_kernel = slot_dir / "kernel.py"
        if root_kernel.exists():
            try:
                content = root_kernel.read_text()
                if baseline_kernel is None or content != baseline_kernel:
                    return root_kernel.parent
            except OSError:
                pass

    # Fallback: search workspace-relative paths where agents actually modify
    # kernels.  Agents work inside ws_*/run_*/<task_dir>/ directories that
    # contain the task_id in their name.
    for slot_dir in sorted(worktrees_dir.glob("slot_*")):
        if not slot_dir.is_dir() or "_logs" in slot_dir.name:
            continue
        for ws_kernel in slot_dir.glob("ws_*/run_*/*/kernel.py"):
            # Only consider directories whose name contains the task_id
            if task_id not in ws_kernel.parent.name:
                continue
            try:
                ws_content = ws_kernel.read_text()
                if baseline_kernel is None or ws_content != baseline_kernel:
                    logger.info("Found modified kernel at workspace path %s", ws_kernel.parent)
                    return ws_kernel.parent
            except OSError:
                pass

    return None


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


def _neutralize_namespace_stubs(worktree_dir) -> list:
    """Rename namespace package stubs anywhere inside an eval worktree.

    Eval worktrees are full repo checkouts that may contain namespace
    package directories (e.g. ``aiter/``) with stub ``__init__.py``
    files using ``pkgutil.extend_path``.  These stubs shadow real
    installed packages when the worktree (or any sub-directory such as the
    harness parent directory) is on ``sys.path``, causing ImportError.
    Renaming them lets the real packages load.

    The search is recursive because Python automatically prepends the
    directory of the executed script to ``sys.path[0]``, and that
    directory may be deeply nested inside the worktree.

    Returns a list of renamed directories (for cleanup/restore).
    """
    renamed: list[Path] = []
    if not worktree_dir:
        return renamed
    wt = Path(worktree_dir)
    if not wt.is_dir():
        return renamed

    # Collect all candidate directories first to avoid issues with
    # rglob iteration order (sub-packages may appear before parents).
    candidates: list[Path] = []
    for init_py in wt.rglob("__init__.py"):
        parent = init_py.parent
        try:
            init_text = init_py.read_text(errors="ignore").strip()
        except OSError:
            continue
        if "extend_path" in init_text and len(init_text) < 1000:
            candidates.append(parent)

    # Sort by path depth (shallowest first) so we rename the topmost
    # stub directory and automatically neutralise its children.
    candidates.sort(key=lambda p: len(p.parts))

    for candidate in candidates:
        # Skip if this candidate is inside an already-renamed directory.
        # Compare against original (pre-rename) paths.
        skip = False
        for r in renamed:
            # Reconstruct the original path from the disabled name
            orig_name = r.name.replace("_disabled", "").lstrip("_")
            orig_path = r.with_name(orig_name)
            try:
                if candidate == orig_path or candidate.is_relative_to(orig_path):
                    skip = True
                    break
            except (ValueError, TypeError):
                if str(candidate).startswith(str(orig_path) + "/"):
                    skip = True
                    break
        if skip:
            continue
        disabled = candidate.with_name("_" + candidate.name + "_disabled")
        try:
            candidate.rename(disabled)
            renamed.append(disabled)
        except OSError:
            pass
    return renamed


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
    # Only add --iterations if the harness actually accepts it.
    # Check for an argparse add_argument("--iterations"...) definition, or
    # parse_known_args (which tolerates unknown flags).  A bare mention of
    # "iterations" in comments is NOT sufficient.
    harness_text = Path(harness_path).read_text(errors="ignore") if harness_path and Path(harness_path).exists() else ""
    accepts_iterations = ("--iterations" in harness_text or "parse_known_args" in harness_text)
    if accepts_iterations:
        env["GEAK_BENCHMARK_EXTRA_ARGS"] = f"--iterations {iters}"
    else:
        env["GEAK_BENCHMARK_EXTRA_ARGS"] = ""
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
) -> Any:
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
                    kernel_time = _parse_total_kernel_time_ms(test_path.read_text())

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
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    if not candidates:
        _print(f"  Round {round_num}: no valid candidates for evaluation")
        # Still write a round evaluation so finalize_run() can find it.
        # Without this, finalize_run() has no round data and may skip
        # writing final_report.json entirely.
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

    all_have_kernel_time = all(c["kernel_time_ms"] is not None for c in candidates)

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
    _use_commandment = commandment_path.exists()
    if not _use_commandment:
        _print("  WARNING: COMMANDMENT.md not found, will use direct harness invocation for FULL_BENCHMARK")

    repo_root = ctx.get("repo_root", "")
    harness_path = ctx.get("harness_path", "")
    gpu_id = ctx.get("gpu_ids", [0])[0]

    eval_worktree: Path | None = None
    try:
        # Prefer the agent's actual worktree slot over a clean worktree.
        # This ensures FULL_BENCHMARK measures the EXACT same kernel.py
        # that save_and_test benchmarked — no discrepancies from patch
        # application, CUDA graph warm state, or missing runtime files.
        agent_slot_dir = _find_agent_worktree_slot(results_dir, best_task, repo_root, output_dir)
        if agent_slot_dir:
            eval_worktree = agent_slot_dir
            _print(f"  Using agent worktree slot: {eval_worktree.name}")
        else:
            # Fallback to clean worktree + patch
            try:
                eval_worktree = setup_eval_worktree(repo_root, best_patch_file, output_dir)
                _print("  Using clean eval worktree (agent slot not found)")
            except PatchApplyError as exc:
                _print(f"  Patch apply failed: {exc}")
                round_eval["patch_apply_error"] = str(exc)
                round_eval["status"] = "patch_failed"
                eval_path = output_dir / f"round_{round_num}_evaluation.json"
                eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
                from minisweagent.run.pipeline_types import FullBenchmarkResult as _FB
                from minisweagent.run.pipeline_types import RoundEvaluation as _RE

                return _RE(
                    round=round_num,
                    best_patch=best_patch_file or "",
                    best_task=best_task,
                    benchmark_speedup=best_speedup,
                    full_benchmark=_FB(failure_reason=f"patch apply failed: {exc}"),
                )

        eval_harness_path = harness_path
        if harness_path and eval_worktree:
            harness_name = Path(harness_path).name
            eval_harness = eval_worktree / harness_name
            if eval_harness.exists():
                eval_harness_path = str(eval_harness)

        # Fix A: Discover harness in eval worktree when preprocessor failed
        if not eval_harness_path and eval_worktree:
            _discovered = eval_worktree / 'test_kernel_harness.py'
            if _discovered.exists():
                eval_harness_path = str(_discovered)
                _print(f"  Discovered harness in eval worktree: {_discovered.name}")

        # Neutralize namespace package stubs that shadow installed packages
        _disabled_stubs = _neutralize_namespace_stubs(eval_worktree)
        eval_env = build_eval_env(eval_worktree, repo_root, eval_harness_path, gpu_id)
        _print(f"  Eval worktree: {eval_worktree}")

        full_benchmark_baseline_path = pp_dir / "full_benchmark_baseline.txt"
        full_benchmark_baseline = (
            full_benchmark_baseline_path.read_text().strip() if full_benchmark_baseline_path.exists() else None
        )
        benchmark_baseline_path = pp_dir / "benchmark_baseline.txt"
        benchmark_baseline = benchmark_baseline_path.read_text().strip() if benchmark_baseline_path.exists() else None

        # Fallback: if no baseline files exist, run baseline benchmark now
        if not full_benchmark_baseline and not benchmark_baseline:
            try:
                _print("  No baseline files found — running baseline benchmark...")
                baseline_env = build_eval_env(eval_worktree, repo_root, eval_harness_path, gpu_id)
                bl_result = subprocess.run(
                    ["python3", eval_harness_path, "--full-benchmark"],
                    capture_output=True, text=True, timeout=600,
                    cwd=str(eval_worktree), env=baseline_env,
                )
                if bl_result.returncode == 0 and bl_result.stdout.strip():
                    full_benchmark_baseline = bl_result.stdout.strip()
                    (pp_dir / "full_benchmark_baseline.txt").write_text(full_benchmark_baseline)
                    _bl_ms = _extract_latency_ms(full_benchmark_baseline)
                    _print(f"  Generated baseline: {_bl_ms}ms")
            except Exception as exc:
                _print(f"  WARNING: baseline generation failed: {exc}")

        # --- FULL_BENCHMARK ---
        fb_script = build_eval_script(str(commandment_path), ["SETUP", "FULL_BENCHMARK"]) if _use_commandment else None
        _ran_fb = False
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
                _ran_fb = True
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
                        baseline_reported_speedup = _extract_reported_speedup(baseline_ref) if baseline_ref else None
                        if (
                            isinstance(candidate_reported_speedup, (int, float))
                            and isinstance(baseline_reported_speedup, (int, float))
                            and baseline_reported_speedup > 0
                        ):
                            verified_speedup = float(candidate_reported_speedup) / float(baseline_reported_speedup)
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
                                "  WARNING: Benchmark config mismatch detected! "
                                "Agent may have modified benchmark parameters. "
                                "Rejecting speedup."
                            )
                            logger.warning(
                                "Benchmark config mismatch: agent modified benchmark configs. "
                                "baseline_configs=%d lines, candidate_configs=%d lines",
                                len(baseline_configs),
                                len(candidate_configs),
                            )
                            round_eval["full_benchmark"]["config_mismatch"] = True
                            round_eval["full_benchmark"]["config_mismatch_detail"] = (
                                f"baseline={len(baseline_configs)} configs, candidate={len(candidate_configs)} configs"
                            )
                            round_eval["full_benchmark"].pop("verified_speedup", None)
                            _print("  Verified speedup INVALIDATED due to config mismatch")
                    elif candidate_configs or baseline_configs:
                        candidate_shapes = _parse_shape_count(fb_stdout)
                        baseline_shapes = _parse_shape_count(baseline_ref) if baseline_ref else None
                        if candidate_shapes and baseline_shapes and candidate_shapes != baseline_shapes:
                            logger.warning(
                                "Shape count mismatch: baseline=%d, candidate=%d",
                                baseline_shapes,
                                candidate_shapes,
                            )
                            round_eval["full_benchmark"]["shape_count_warning"] = (
                                f"baseline={baseline_shapes}, candidate={candidate_shapes}"
                            )

                _print(f"  FULL_BENCHMARK: {'PASS' if fb_result.returncode == 0 else 'FAIL'}")
            except Exception as exc:
                _print(f"  FULL_BENCHMARK failed: {exc}")
                round_eval["full_benchmark"] = {"error": str(exc)}

        if not _ran_fb and eval_harness_path:
            _print(f"  Running FULL_BENCHMARK via direct harness invocation...")
            try:
                fb_result = subprocess.run(
                    ["python3", eval_harness_path, "--full-benchmark"],
                    capture_output=True, text=True, timeout=1800,
                    cwd=str(eval_worktree), env=eval_env,
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
                            f"  FULL_BENCHMARK (direct) verified speedup: {verified_speedup:.4f}x "
                            f"({baseline_ms:.4f} ms -> {candidate_ms:.4f} ms)"
                        )
                    else:
                        candidate_reported_speedup = _extract_reported_speedup(fb_stdout)
                        baseline_reported_speedup = _extract_reported_speedup(baseline_ref) if baseline_ref else None
                        if (
                            isinstance(candidate_reported_speedup, (int, float))
                            and isinstance(baseline_reported_speedup, (int, float))
                            and baseline_reported_speedup > 0
                        ):
                            verified_speedup = float(candidate_reported_speedup) / float(baseline_reported_speedup)
                            round_eval["full_benchmark"]["verified_speedup"] = round(verified_speedup, 4)
                            _print(f"  FULL_BENCHMARK (direct) verified speedup: {verified_speedup:.4f}x (from reported)")
                _fb_status = "PASS" if fb_result.returncode == 0 else "FAIL"
                _print(f"  FULL_BENCHMARK (direct): {_fb_status}")
            except Exception as exc:
                _print(f"  FULL_BENCHMARK (direct) failed: {exc}")
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
                    from concurrent.futures import ThreadPoolExecutor as _TPE
                    from concurrent.futures import TimeoutError as _FTE
                    _tpx = _TPE(max_workers=1)
                    try:
                        _pf = _tpx.submit(
                            _profile_fn,
                            command=f"python {harness_path} --profile",
                            backend="metrix",
                            num_replays=3,
                            quick=True,
                            gpu_devices=str(gpu_id),
                        )
                        profile_result = _pf.result(timeout=1200)
                    except _FTE:
                        _print("  PROFILE timed out after 20 minutes, skipping")
                        round_eval["profile_comparison"] = {"error": "PROFILE timeout after 1200s"}
                        profile_result = None
                        _tpx.shutdown(wait=False, cancel_futures=True)
                finally:
                    if prev_pythonpath:
                        os.environ["PYTHONPATH"] = prev_pythonpath
                    else:
                        os.environ.pop("PYTHONPATH", None)

                if baseline_metrics and profile_result:
                    from minisweagent.run.preprocess.baseline import build_baseline_metrics

                    optimized_metrics = build_baseline_metrics(profile_result, include_all=True)
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
        # Only clean up if we created a temporary eval worktree, not if
        # we reused an agent's existing slot directory.
        if eval_worktree and not agent_slot_dir:
            cleanup_eval_worktree(repo_root, eval_worktree)

    # Write full detail dict for backward compatibility and debugging
    eval_path = output_dir / f"round_{round_num}_evaluation.json"
    eval_path.write_text(json.dumps(round_eval, indent=2, default=str))
    _print(f"  Round evaluation written to: {eval_path}")

    # Write stdout/profile to files instead of embedding in the typed return
    fb_raw = round_eval.get("full_benchmark") or {}
    if isinstance(fb_raw, dict) and fb_raw.get("stdout"):
        fb_output_path = output_dir / f"round_{round_num}_full_benchmark.txt"
        fb_output_path.write_text(fb_raw["stdout"])
    profile_raw = round_eval.get("profile_comparison")
    if isinstance(profile_raw, dict) and profile_raw:
        profile_path = output_dir / f"round_{round_num}_profile_comparison.json"
        profile_path.write_text(json.dumps(profile_raw, indent=2, default=str))

    # Convert to typed boundary object
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
