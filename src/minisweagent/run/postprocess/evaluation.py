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


class CommandmentExecutionError(RuntimeError):
    """Raised when a COMMANDMENT subprocess fails for a non-kernel-logic reason.

    This signals that the evaluation contract itself is broken (e.g. the
    harness rejects `--iterations`, a wrapper script is missing, the script
    crashes before kernel logic runs, or the subprocess raises before
    completion). The orchestrator treats this as a hard failure and aborts
    the run rather than silently continuing with no validated speedup.

    Distinguished from kernel-level correctness failures, which remain
    recoverable per-round events (status="correctness_failed").
    """

    def __init__(self, section: str, returncode: int | None, detail: str) -> None:
        self.section = section
        self.returncode = returncode
        self.detail = detail
        super().__init__(f"COMMANDMENT {section} failed (rc={returncode}): {detail}")


# Stderr signatures that indicate the harness/commandment itself is broken,
# not the kernel under optimization. When any of these match a non-zero
# subprocess exit, we raise CommandmentExecutionError instead of treating
# the failure as a recoverable kernel-correctness miss.
_CONTRACT_BROKEN_PATTERNS: tuple[str, ...] = (
    "unrecognized arguments",
    "the following arguments are required",
    "argument --",
    "Harness file not found",
    "No such file or directory",
    "ModuleNotFoundError",
    "ImportError",
    "command not found",
    "Permission denied",
    "Errno 2",
)


def _stderr_indicates_broken_contract(stderr: str) -> str | None:
    """Return the first contract-broken signature found in *stderr*, or None."""
    if not stderr:
        return None
    for pattern in _CONTRACT_BROKEN_PATTERNS:
        if pattern in stderr:
            return pattern
    return None


def _format_stderr_tail(stderr: str, *, max_lines: int = 20) -> str:
    """Return the last *max_lines* of *stderr*, joined for log inclusion."""
    if not stderr:
        return ""
    return "\n".join(stderr.strip().splitlines()[-max_lines:])


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

    # Discover sibling sub-agent worktree object stores so a 3-way fallback
    # can bridge patch-lineage mismatches (e.g. when sub-agents were seeded
    # with a pre-wrapped kernel blob whose ancestry the eval worktree does
    # not share). We walk up from the patch file, find the first ``worktrees``
    # directory, and collect the ``.git/objects`` path of each slot.
    sibling_alternates: list[Path] = []
    for ancestor in patch_path.resolve().parents:
        wt_root = ancestor / "worktrees"
        if wt_root.is_dir():
            for slot in sorted(wt_root.iterdir()):
                objects_dir = slot / ".git" / "objects"
                if objects_dir.is_dir():
                    sibling_alternates.append(objects_dir)
            break

    apply_result, removed_paths = apply_patch_with_generated_helper_fallback(
        patch_text=patch_text,
        cwd=eval_dir,
        env=git_env,
        object_alternates=sibling_alternates,
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
    from minisweagent.run.preprocess.harness_utils import harness_supports_iterations

    iters = benchmark_iterations or DEFAULT_EVAL_BENCHMARK_ITERATIONS
    env = os.environ.copy()
    env["GEAK_WORK_DIR"] = str(work_dir)
    env["GEAK_REPO_ROOT"] = repo_root
    env["GEAK_HARNESS"] = harness_path
    env["GEAK_GPU_DEVICE"] = str(gpu_id)
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    env["GEAK_BENCHMARK_ITERATIONS"] = str(iters)
    if harness_supports_iterations(harness_path):
        env["GEAK_BENCHMARK_EXTRA_ARGS"] = f"--iterations {iters}"
    else:
        # Harness doesn't declare ``--iterations``; passing it on the CLI
        # would crash argparse with "unrecognized arguments". The harness
        # can still honour the iteration count by reading the
        # ``GEAK_BENCHMARK_ITERATIONS`` env var we set above.
        logger.debug(
            "build_eval_env: harness %s does not declare --iterations; relying on GEAK_BENCHMARK_ITERATIONS=%s only",
            harness_path,
            iters,
        )
    pp_parts = [str(work_dir), repo_root]
    if "/tests/" in harness_path:
        try:
            hp = Path(harness_path).resolve()
            for _ in range(3):
                hp = hp.parent
            pp_parts.append(str(hp))
        except (OSError, ValueError):
            pass
    pp_parts.append(env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = ":".join(p for p in pp_parts if p)
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in alloc_conf:
        logger.debug("build_eval_env: removing PYTORCH_CUDA_ALLOC_CONF with expandable_segments.")
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


def preflight_commandment_contract(
    commandment_path: Path,
    repo_root: str,
    harness_path: str,
    gpu_id: int,
    *,
    timeout_s: int = 600,
) -> None:
    """Smoke-test SETUP + CORRECTNESS once before fanning out sub-agents.

    Runs the COMMANDMENT against the *unpatched* repo with
    ``--iterations 1`` so a broken contract (e.g. a harness that rejects
    ``--iterations``) is surfaced as a single ``CommandmentExecutionError``
    instead of being burned into every sub-agent's iteration loop.

    Skipped silently when ``GEAK_SKIP_COMMANDMENT_PREFLIGHT=1`` is set.

    Raises
    ------
    CommandmentExecutionError
        If the SETUP+CORRECTNESS script can't run, returns non-zero, or
        prints a stderr matching the contract-broken signature set.
    """
    if os.environ.get("GEAK_SKIP_COMMANDMENT_PREFLIGHT", "").strip() == "1":
        logger.info("preflight_commandment_contract: skipped (GEAK_SKIP_COMMANDMENT_PREFLIGHT=1)")
        return

    if not commandment_path.exists():
        raise CommandmentExecutionError("PREFLIGHT", None, f"COMMANDMENT.md not found at {commandment_path}")

    script = build_eval_script(str(commandment_path), ["SETUP", "CORRECTNESS"])
    if not script:
        logger.warning(
            "preflight_commandment_contract: no SETUP/CORRECTNESS commands in %s; skipping",
            commandment_path,
        )
        return

    from minisweagent.run.preprocess.harness_utils import harness_supports_iterations

    repo_root_path = Path(repo_root).resolve()
    env = build_eval_env(repo_root_path, str(repo_root_path), harness_path, gpu_id)
    env["GEAK_BENCHMARK_ITERATIONS"] = "1"
    if harness_supports_iterations(harness_path):
        env["GEAK_BENCHMARK_EXTRA_ARGS"] = "--iterations 1"
    else:
        # See ``build_eval_env`` for the reasoning. Drop any ``--iterations``
        # tokens the upstream env-builder may have emitted so we don't pass
        # them on the harness CLI.
        env.pop("GEAK_BENCHMARK_EXTRA_ARGS", None)

    logger.info(
        "preflight_commandment_contract: smoke-testing COMMANDMENT against %s with iterations=1",
        repo_root_path,
    )
    try:
        result = subprocess.run(
            ["bash", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(repo_root_path),
            env=env,
        )
    except Exception as exc:
        raise CommandmentExecutionError(
            "PREFLIGHT", None, f"SETUP+CORRECTNESS subprocess failed to complete: {exc}"
        ) from exc

    if result.returncode != 0:
        stderr_tail = _format_stderr_tail(result.stderr)
        broken = _stderr_indicates_broken_contract(result.stderr)
        detail = (
            f"contract-broken signature {broken!r}; stderr tail:\n{stderr_tail}"
            if broken is not None
            else f"non-zero exit; stderr tail:\n{stderr_tail}"
        )
        logger.error(
            "preflight_commandment_contract: FAILED (rc=%d) -- aborting before sub-agent fan-out:\n%s",
            result.returncode,
            stderr_tail,
        )
        raise CommandmentExecutionError("PREFLIGHT", result.returncode, detail)

    logger.info("preflight_commandment_contract: PASS")


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
            # The subprocess could not complete (timeout, OS error, etc.).
            # This is a contract-level failure; record it on round_eval for
            # any callers that catch the exception, then escalate.
            round_eval["correctness"] = {"error": str(exc)}
            round_eval["status"] = "commandment_execution_failed"
            raise CommandmentExecutionError("CORRECTNESS", None, f"subprocess failed to complete: {exc}") from exc

        round_eval["correctness"] = {
            "returncode": correctness_result.returncode,
            "success": correctness_result.returncode == 0,
        }
        if correctness_result.returncode != 0:
            stderr_tail = _format_stderr_tail(correctness_result.stderr)
            broken = _stderr_indicates_broken_contract(correctness_result.stderr)
            if broken is not None:
                round_eval["status"] = "commandment_execution_failed"
                logger.error(
                    "CORRECTNESS failed because the COMMANDMENT contract is broken (matched %r):\n%s",
                    broken,
                    stderr_tail,
                )
                raise CommandmentExecutionError(
                    "CORRECTNESS",
                    correctness_result.returncode,
                    f"contract-broken signature {broken!r}; stderr tail:\n{stderr_tail}",
                )
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
            round_eval[section_key] = {"error": str(exc)}
            round_eval["status"] = "commandment_execution_failed"
            raise CommandmentExecutionError(
                baseline_section_name, None, f"subprocess failed to complete: {exc}"
            ) from exc

        if candidate_result.returncode != 0:
            stderr_tail = _format_stderr_tail(candidate_result.stderr)
            round_eval[section_key] = {
                "error": candidate_result.stderr,
                "returncode": candidate_result.returncode,
            }
            round_eval["status"] = "commandment_execution_failed"
            broken = _stderr_indicates_broken_contract(candidate_result.stderr)
            detail = (
                f"contract-broken signature {broken!r}; stderr tail:\n{stderr_tail}"
                if broken is not None
                else f"non-zero exit; stderr tail:\n{stderr_tail}"
            )
            logger.error(
                "%s execution failed (rc=%d) -- treating as COMMANDMENT failure:\n%s",
                baseline_section_name,
                candidate_result.returncode,
                stderr_tail,
            )
            raise CommandmentExecutionError(baseline_section_name, candidate_result.returncode, detail)

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
    """Compute verified speedup from latency measurements.

    Falsy-check pitfall: ``not candidate_ms`` collapses ``0.0`` and ``None``,
    so a real ``0.0`` candidate latency (e.g. divide-by-zero shortcut, broken
    benchmark loop) used to silently look identical to "couldn't parse". Use
    explicit ``is None`` / ``<= 0`` so an unparseable result and a degenerate
    measurement get distinct ``failure_reason`` strings, which then flow
    through to ``RoundEvaluation.full_benchmark.failure_reason``.
    """
    candidate_ms = extract_latency_ms(candidate_stdout)
    baseline_ms = extract_latency_ms(baseline_text)

    if candidate_ms is None or baseline_ms is None:
        msg = f"latency parse failed (candidate_ms={candidate_ms}, baseline_ms={baseline_ms})"
        logger.warning("Could not extract latency: %s", msg)
        round_eval[section_key]["failure_reason"] = msg
        return
    if baseline_ms <= 0:
        msg = f"baseline latency non-positive (baseline_ms={baseline_ms})"
        logger.warning("%s", msg)
        round_eval[section_key]["failure_reason"] = msg
        return
    if candidate_ms <= 0:
        msg = f"candidate latency non-positive (candidate_ms={candidate_ms}); rejecting as broken measurement"
        logger.warning("%s", msg)
        round_eval[section_key]["failure_reason"] = msg
        round_eval[section_key]["candidate_ms"] = candidate_ms
        round_eval[section_key]["baseline_ms"] = baseline_ms
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
        else:
            logger.debug("_check_config_mismatch: both sides have matching configs.")
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
    else:
        logger.debug("_check_config_mismatch: neither side has config lines; skipping comparison.")


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
        profile_result = subprocess.run(
            ["bash", profile_script],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(eval_worktree),
            env=eval_env,
        )
    except Exception as exc:
        round_eval["profile_comparison"] = {"error": str(exc)}
        round_eval["status"] = "commandment_execution_failed"
        raise CommandmentExecutionError("PROFILE", None, f"subprocess failed to complete: {exc}") from exc

    if profile_result.returncode != 0:
        stderr_tail = _format_stderr_tail(profile_result.stderr)
        round_eval["profile_comparison"] = {
            "error": profile_result.stderr,
            "returncode": profile_result.returncode,
        }
        round_eval["status"] = "commandment_execution_failed"
        broken = _stderr_indicates_broken_contract(profile_result.stderr)
        detail = (
            f"contract-broken signature {broken!r}; stderr tail:\n{stderr_tail}"
            if broken is not None
            else f"non-zero exit; stderr tail:\n{stderr_tail}"
        )
        logger.error(
            "PROFILE execution failed (rc=%d) -- treating as COMMANDMENT failure:\n%s",
            profile_result.returncode,
            stderr_tail,
        )
        raise CommandmentExecutionError("PROFILE", profile_result.returncode, detail)

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
        round_eval["speedup_source"] = (
            "agent-reported benchmark (no FULL_BENCHMARK verified result available — "
            "the orchestrator will run FULL_BENCHMARK automatically after this round; "
            "do not use this speedup for final selection)"
        )
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
        elif fb_raw.get("failure_reason"):
            # Set by ``_compute_verified_speedup`` when latency parsing failed
            # despite a clean exit; without this propagation a return-code-0
            # benchmark with unparseable output looked like "everything passed
            # but no verified speedup" with no diagnostic.
            failure = str(fb_raw["failure_reason"])
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
        # Fallback: check for best_patch.diff directly in the round directory.
        # The heterogeneous orchestrator LLM sometimes creates patches directly
        # (e.g. when dispatch_tasks fails and it edits kernel.py manually).
        for diff_name in ("best_patch.diff", "best_patch.patch"):
            fallback_patch = results_dir / diff_name
            if fallback_patch.exists() and fallback_patch.stat().st_size > 0:
                logger.info(
                    "Round %d: no best_results.json found, but found fallback patch: %s",
                    round_num,
                    fallback_patch,
                )
                candidates.append(
                    {
                        "task": "orchestrator_direct",
                        "patch_file": str(fallback_patch),
                        "speedup": 1.0,
                        "kernel_time_ms": None,
                        "per_shape_speedups": {},
                        "baseline_shape_latency_ms": {},
                        "candidate_shape_latency_ms": {},
                    }
                )
                break

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
        logger.debug("evaluate_round_best: selecting by min kernel_time_ms (%d candidates).", len(candidates))
    else:
        best = max(candidates, key=lambda c: c["speedup"])
        logger.debug("evaluate_round_best: selecting by max speedup (%d candidates).", len(candidates))

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
