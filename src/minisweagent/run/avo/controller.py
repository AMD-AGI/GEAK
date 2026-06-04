#!/usr/bin/env python3
"""AVO continuous-evolution controller + ``geak-avo`` CLI.

The controller is a deterministic outer loop. Its single responsibility is to
**never stop because a variation agent stopped**: a variation step may end via
``Submitted``, ``LimitsExceeded``, an exception, or a deadline — all are treated
as "this step is done", and the loop proceeds until the wall-clock budget's
``soft_stop`` fires.

It reuses GEAK wholesale and modifies nothing:

- preprocess: shells out to ``geak --preprocess-only`` (or accepts a prepared
  run directory via ``--prepared-dir``);
- budget: ``RunBudget`` / ``BudgetSpec`` (same as ``geak``);
- variation: ``OptimizationAgent`` via :mod:`minisweagent.run.avo.variation_step`;
- lineage + commit gate: :class:`LineageStore`;
- supervisor: :class:`StagnationDetector` + the ``avo-supervisor`` subagent;
- finalize: ``auto_finalize``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from minisweagent.config import load_config
from minisweagent.models import get_model
from minisweagent.run.avo.lineage_store import LineageStore
from minisweagent.run.avo.stagnation import StagnationDetector, StagnationLevel
from minisweagent.run.avo.supervisor import apply_directive, build_bundle, run_supervisor
from minisweagent.run.avo.variation_step import run_variation_step, write_evolution_entry
from minisweagent.run.budget import BudgetSpec, RunBudget
from minisweagent.utils.log import add_file_handler

logger = logging.getLogger(__name__)
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich", help="AVO — Agentic Variation Operators on GEAK.")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_avo_config(config_path: str | None) -> dict:
    """Load geak.yaml, deep-merge geak_avo.yaml, then an optional ``--config``."""
    try:
        base = load_config("geak")
    except FileNotFoundError:
        base = {}
    try:
        avo = load_config("geak_avo")
    except FileNotFoundError:
        avo = {}
    merged = _deep_merge(base, avo)
    if config_path:
        import yaml

        extra = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        merged = _deep_merge(merged, extra)
    return merged


def _build_budget(config: dict, mode: str, total_budget_s: float | None) -> RunBudget:
    budgets = (config.get("run") or {}).get("budgets", {}).get(mode) or {}
    if not budgets:
        raise typer.BadParameter(f"No run.budgets.{mode} block in merged config (geak.yaml + geak_avo.yaml).")
    spec = BudgetSpec(
        mode=mode,  # type: ignore[arg-type]
        total_s=float(total_budget_s if total_budget_s is not None else budgets["total_s"]),
        preprocess_soft_cap_s=float(budgets["preprocess_soft_cap_s"]),
        preprocess_hard_cap_fraction=float(budgets["preprocess_hard_cap_fraction"]),
        finalize_grace_s=float(budgets["finalize_grace_s"]),
        kill_buffer_s=float(budgets.get("kill_buffer_s", 60.0)),
    )
    return RunBudget(spec=spec)


# ---------------------------------------------------------------------------
# Preprocess (reuse geak --preprocess-only)
# ---------------------------------------------------------------------------


def _run_preprocess(
    repo: Path,
    task: str,
    output_dir: Path,
    model_name: str | None,
    gpu_ids: list[int] | None = None,
    mode: str | None = None,
    config_path: str | None = None,
) -> None:
    """Produce COMMANDMENT.md / baseline / profile via ``geak --preprocess-only``.

    Note: ``--kernel-language`` is intentionally NOT forwarded — ``geak`` has no
    such flag (it auto-detects the kernel type during preprocess); AVO's
    ``kernel_language`` is only used internally for skill selection / lineage.
    """
    if (output_dir / "COMMANDMENT.md").exists():
        logger.info("preprocess: COMMANDMENT.md already present in %s; skipping.", output_dir)
        return
    cmd = ["geak", "--repo", str(repo), "--task", task, "-o", str(output_dir), "--preprocess-only", "--yolo"]
    if model_name:
        cmd += ["--model", model_name]
    if gpu_ids:
        # Propagate GPU selection so baseline/profile run on the requested devices
        # (otherwise preprocess silently defaults to GPU 0).
        cmd += ["--gpu-ids", ",".join(str(g) for g in gpu_ids)]
    if mode:
        # Propagate the run mode so preprocess uses the matching budget/presets
        # instead of geak.yaml's default (otherwise --mode quick silently ran the
        # preprocess subprocess as mode=full).
        cmd += ["--mode", mode]
    if config_path:
        # Forward the user's extra config so preprocess honors the same overrides.
        # Note: geak's -c REPLACES geak.yaml as the user layer (it does not merge),
        # so an explicit AVO --config should be a complete config, not a snippet.
        cmd += ["--config", config_path]
    logger.info("preprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)
    if not (output_dir / "COMMANDMENT.md").exists():
        logger.warning("preprocess: COMMANDMENT.md not found after preprocess; the run may be degraded.")


# ---------------------------------------------------------------------------
# Isolated work repo (never modify the user's original repository)
# ---------------------------------------------------------------------------


def _prepare_work_repo(repo: Path, output_dir: Path) -> Path:
    """Create a **fully independent** working copy of ``repo`` so AVO never
    touches it — not even its ``.git`` (refs/tags/commits).

    AVO drives a coding agent and writes git commits/tags (``avo-v{N}``) as it
    evolves the kernel. To stay non-intrusive over a multi-day run, all
    variation steps, ``save_and_test`` edits, worktree resets
    (``git checkout -f`` / ``git clean -fd``), and lineage commits/tags happen
    inside this isolated repo at ``<output_dir>/avo_repo``. The original ``repo``
    is only **read** (for preprocess and as the snapshot source).

    Unlike a shared ``git worktree``, this uses an **independent clone** so the
    ``avo-v{N}`` tags live only in ``avo_repo/.git`` and never appear in the
    user's ``git tag`` list:

    - Git repos: ``git clone --local`` (objects hardlinked → ~no extra disk),
      then sync the **live** tree (dirty-tracked + untracked + JIT ``.so``
      symlinks) via GEAK's existing helpers so the snapshot matches the working
      tree preprocess profiled.
    - Non-git directories: a full copy bootstrapped into a throwaway git repo.

    **Resume protection.** The ``avo-v{N}`` commits/tags exist ONLY inside
    ``avo_repo/.git``; re-cloning on resume would pull them from the (clean)
    user repo and lose them, breaking ``reset_worktree_to_best``. So if a git
    repo already exists at the target path, reuse it as-is instead of recreating
    it — preserving the accumulated lineage history.
    """
    work_repo = (output_dir / "avo_repo").resolve()

    # Resume: an existing repo already holds the lineage's avo-v{N} history.
    # Never wipe/re-clone it, or the best-version refs are lost.
    if (work_repo / ".git").exists():
        logger.info("AVO: reusing existing isolated repo at %s (resume; lineage history preserved).", work_repo)
        return work_repo

    # A leftover dir without a valid .git (e.g. a crashed setup) would make
    # ``git clone`` fail on a non-empty target — clear it first.
    if work_repo.exists():
        import shutil

        logger.warning("AVO: removing stale %s (no .git) before fresh setup.", work_repo)
        shutil.rmtree(work_repo, ignore_errors=True)

    work_repo.parent.mkdir(parents=True, exist_ok=True)

    if (Path(repo) / ".git").exists():
        from minisweagent.run.task_file import (
            _apply_dirty_tracked_changes,
            _copy_nested_git_repos,
            _copy_untracked_files,
            _neutralize_nested_git_repos,
            _symlink_gitignored_so_files,
        )
        from minisweagent.run.utils.git_safe_env import get_git_safe_env

        git_env = get_git_safe_env(work_repo.parent)
        # --local hardlinks the object store (fast, ~no extra disk); the clone
        # gets its OWN .git, so commits/tags never reach the user's repo.
        subprocess.run(
            ["git", "clone", "--local", "--no-checkout", str(repo), str(work_repo)],
            check=True,
            capture_output=True,
            env=git_env,
        )
        subprocess.run(
            ["git", "-C", str(work_repo), "checkout", "HEAD"],
            check=False,
            capture_output=True,
            env=git_env,
        )
        # Match the user's *live* tree (clone only carries committed state).
        _apply_dirty_tracked_changes(repo, work_repo, git_env)
        _copy_untracked_files(repo, work_repo, git_env)
        _symlink_gitignored_so_files(repo, work_repo, git_env)
        # Submodules / nested repos are not recursed by clone; copy them as plain
        # files so the snapshot is complete and diffs stay clean.
        _copy_nested_git_repos(repo, work_repo)
        _neutralize_nested_git_repos(work_repo)
        logger.info("AVO: independent clone at %s (separate .git; original repo fully isolated).", work_repo)
    else:
        from minisweagent.run.utils.parallel_helpers import bootstrap_git_repo, create_copy_workdir

        create_copy_workdir(repo, work_repo)
        bootstrap_git_repo(work_repo)
        logger.info("AVO: isolated copy + bootstrapped git repo at %s (original repo untouched).", work_repo)
    return work_repo


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def run_avo(
    *,
    repo: Path,
    task: str,
    output_dir: Path,
    config: dict,
    budget: RunBudget,
    model_name: str | None,
    kernel_language: str = "python",
    gpu_ids: list[int] | None = None,
    mode: str = "full",
    config_path: str | None = None,
) -> dict:
    """Run one single-lineage AVO evolution; return the final report dict."""
    avo_cfg = dict(config.get("avo") or {})
    model_cfg = config.get("model", {})
    gpu_ids = gpu_ids or [0]
    # Make the requested GPUs visible to every stage: the variation-step agent's
    # save_and_test (via _build_agent), the ESCALATE rescue workers (which derive
    # their cfg from config["avo"]), and the verify context below. Without this,
    # --gpu-ids only reached verification and steps silently ran on GPU 0.
    avo_cfg["_gpu_ids"] = list(gpu_ids)
    config.setdefault("avo", {})["_gpu_ids"] = list(gpu_ids)

    def model_factory():
        return get_model(model_name, model_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── preprocess (reused) ───────────────────────────────────────────
    _run_preprocess(repo, task, output_dir, model_name, gpu_ids, mode, config_path)
    pp_elapsed = time.monotonic() - budget.started_at
    budget.commit_preprocess(pp_elapsed)
    budget.schedule_optimization_watchdog()

    # ── isolated work repo (never touch the user's original repo) ──────
    # Every variation step, agent edit, worktree reset, and lineage commit/tag
    # below operates on ``work_repo`` (an independent clone under output_dir),
    # not on ``repo``. The original repository — including its .git refs/tags —
    # is only read for preprocess + snapshot.
    work_repo = _prepare_work_repo(repo, output_dir)

    # ── state ─────────────────────────────────────────────────────────
    lineage = LineageStore(
        output_dir / "avo_state",
        epsilon=float(avo_cfg.get("commit_epsilon", 0.001)),
        language=kernel_language,
        min_commit_speedup=float(avo_cfg.get("min_commit_speedup", 1.0)),
        min_per_shape_speedup=float(avo_cfg.get("min_per_shape_speedup", 0.0)),
        significance_margin=float(avo_cfg.get("commit_significance_margin", 0.0)),
    )
    lineage.seed_from_baseline(output_dir, repo=work_repo)
    detector = StagnationDetector(avo_cfg.get("stagnation", {}))
    strategy_file = output_dir / ".optimization_strategies.md"
    min_commits = int(avo_cfg.get("min_commits_before_stop", 0))
    verify_each_step = bool(avo_cfg.get("verify_each_step", True))
    verify_repeats = int(avo_cfg.get("verify_repeats", 1))  # B1: noise-robust median over N re-measures
    skip_verify_on_no_gain = bool(avo_cfg.get("skip_verify_on_no_gain", True))  # C3
    min_commit_speedup = float(avo_cfg.get("min_commit_speedup", 1.0))
    profiling_after_step = int(avo_cfg.get("profiling_after_step", 3))

    # C2: verified-result cache (patch content hash → verified speedup), persisted.
    verify_cache_path = output_dir / "avo_state" / "verify_cache.json"
    verify_cache: dict = {}
    if verify_cache_path.exists():
        try:
            verify_cache = json.loads(verify_cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            verify_cache = {}

    # Context reused by GEAK's evaluate_round_best for verified per-shape geomean
    # scoring (P0) and by the ESCALATE rescue round (P1). Points at the isolated
    # work repo so the verified patch (best→candidate) applies onto the same base
    # the agent edited — and so verification never touches the user's repo.
    verify_ctx = _build_verify_ctx(work_repo, output_dir, gpu_ids, task)

    # Run-wide working notebook for cross-step memory (P-mem-2). One notebook for
    # the whole run; each step injects its summary and records its outcome.
    notebook_root = output_dir / "avo_state" / "notebook"

    step_idx = 0
    pending_nudge: str | None = None
    try:
        while not budget.soft_stop.is_set():
            step_idx += 1
            step_dir = output_dir / f"variation_{step_idx:04d}"
            lineage.reset_worktree_to_best(work_repo)
            lineage.heartbeat(step_index=step_idx)

            # Delayed profiling (CuTeGen): withhold profiler-driven micro-tuning
            # until structure is sound — i.e. past the step threshold OR once a
            # real improvement has been committed.
            profiling_enabled = step_idx > profiling_after_step or len(lineage.committed) > 1

            result = run_variation_step(
                repo=work_repo,
                base_task=task,
                step_dir=step_dir,
                lineage=lineage,
                direction=lineage.current_direction(),
                output_dir=output_dir,
                avo_config=avo_cfg,
                model_factory=model_factory,
                deadline=budget.optimization_deadline() if hasattr(budget, "optimization_deadline") else None,
                nudge=pending_nudge,
                notebook_root=notebook_root,
                profiling_enabled=profiling_enabled,
            )
            pending_nudge = None

            lineage.record_attempts(result)
            if verify_each_step:
                _apply_verified_score(
                    result,
                    verify_ctx,
                    step_idx,
                    output_dir,
                    repeats=verify_repeats,
                    min_commit_speedup=min_commit_speedup,
                    skip_on_no_gain=skip_verify_on_no_gain,
                    cache=verify_cache,
                    cache_path=verify_cache_path,
                )
            committed = lineage.maybe_commit(result, repo=work_repo)
            _record_to_notebook(notebook_root, step_idx, result, committed)
            write_evolution_entry(output_dir, result, committed)  # P-mem-3 causal log

            signal = detector.evaluate(result, committed)
            console.print(
                f"[cyan]AVO step {step_idx}[/cyan]: committed={committed} "
                f"level={signal.level.name} best={lineage.best_speedup:.3f}x — {signal.reason}"
            )

            if signal.level == StagnationLevel.NUDGE:
                pending_nudge = f"Progress is stalling ({signal.reason}). Try a different angle or mark the current strategy failed."
            elif signal.level == StagnationLevel.REDIRECT:
                _do_redirect(signal, lineage, step_dir, output_dir, strategy_file, detector, model_factory, work_repo)
            elif signal.level == StagnationLevel.ESCALATE:
                _do_escalate(lineage, output_dir, config, model_factory, verify_ctx, work_repo, base_task=task)
                detector.reset(partial=False)

        # Budget exhausted; honour min-commits as a best-effort note.
        if min_commits and len(lineage.committed) - 1 < min_commits:
            logger.warning(
                "AVO: budget elapsed with %d commits (< min_commits_before_stop=%d).",
                len(lineage.committed) - 1,
                min_commits,
            )
    finally:
        budget.cancel_all_timers()

    return _finalize(output_dir, lineage)


def _record_to_notebook(notebook_root: Path, step_index: int, result, committed: bool) -> None:
    """Append this step's outcome to the run-wide working notebook (P-mem-2).

    Best-effort: notebook I/O must never interrupt the evolution loop.
    """
    try:
        from minisweagent.memory.working_notebook import WorkingNotebook

        nb = WorkingNotebook(notebook_root, writer_id="avo")
        nb.record_attempt(strategy=result.strategy, change_category=None, step=step_index)
        nb.record_round_evaluation(
            round_num=step_index,
            best_task=result.strategy,
            verified_speedup=result.best_speedup,
            baseline_ms=None,
            candidate_ms=None,
            per_shape_speedups=None,
        )
        if not committed and not result.best_correct:
            nb.append_event(
                "result",
                strategy=result.strategy,
                tag="FAIL",
                message=f"step {step_index}: no committable candidate ({result.exit_status})",
                step=step_index,
                returncode=1,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("notebook record failed (non-fatal): %s", exc)


def _read_per_shape_speedups(output_dir: Path, step_index: int) -> dict:
    """Read per-shape verified speedups from round_{N}_evaluation.json (B2)."""
    import json as _json

    path = Path(output_dir) / f"round_{step_index}_evaluation.json"
    if not path.exists():
        return {}
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    raw = data.get("per_shape_speedups") or {}
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for shape, val in raw.items():
            try:
                out[str(shape)] = float(val)
            except (TypeError, ValueError):
                continue
    return out


_PROFILE_METRIC_HINTS = ("occupancy", "bandwidth", "util", "tflops", "duration", "latency", "throughput", "register", "lds", "smem")


def _read_profile_metrics(output_dir: Path, step_index: int) -> dict:
    """Best-effort numeric profiler metrics from round_{N}_evaluation.json (P-mem-3).

    Cross-backend: scans (one level deep) for scalar fields whose key looks like a
    perf metric. Empty when no profiler data is present.
    """
    import json as _json

    path = Path(output_dir) / f"round_{step_index}_evaluation.json"
    if not path.exists():
        return {}
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, float] = {}

    def _scan(prefix: str, obj) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            key = str(k).lower()
            if isinstance(v, (int, float)) and any(h in key for h in _PROFILE_METRIC_HINTS):
                out[f"{prefix}{k}"] = float(v)
            elif isinstance(v, dict) and prefix == "":
                _scan(f"{k}.", v)

    _scan("", data)
    return dict(list(out.items())[:8])  # cap


def _build_verify_ctx(repo: Path, output_dir: Path, gpu_ids: list[int], task: str) -> dict:
    """Build the ctx dict consumed by GEAK's ``evaluate_round_best``."""
    return {
        "output_dir": str(output_dir),
        "preprocess_dir": str(output_dir),
        "repo_root": str(repo),
        "harness_path": _discover_harness_path(output_dir),
        "gpu_ids": list(gpu_ids),
        "num_parallel": 1,
        "metric": None,
        "starting_patch": "",
        "_best_global_speedup": 0,
        "user_instructions": task,
    }


def _discover_harness_path(output_dir: Path) -> str:
    """Best-effort harness path from preprocess artifacts (PROFILE-only; non-fatal)."""
    import json as _json

    for name in ("testcase_selection.json", "preprocess_context.json"):
        path = output_dir / name
        if not path.exists():
            continue
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
            hp = data.get("harness_path") if isinstance(data, dict) else None
            if hp:
                return str(hp)
        except (OSError, ValueError):
            continue
    return ""


def _eval_once(verify_ctx: dict, step_index: int, output_dir: Path):
    """One independent verification via GEAK's ``evaluate_round_best``.

    Returns ``(verified_speedup, best_patch, per_shape)`` or ``None`` when there
    is no verifiable candidate. Each call applies the best patch in a fresh
    worktree and re-runs FULL_BENCHMARK, so repeated calls yield independent
    measurements of the *same* candidate (pure measurement noise).
    """
    from minisweagent.run.postprocess.evaluation import evaluate_round_best

    results_dir = output_dir / "results" / f"round_{step_index}"
    try:
        round_eval = evaluate_round_best(dict(verify_ctx), step_index, results_dir)
    except Exception as exc:  # noqa: BLE001 — verification failure must not kill the loop
        logger.warning("verify step %d: evaluate_round_best failed (%s).", step_index, exc)
        return None
    if round_eval is None or not getattr(round_eval, "best_patch", ""):
        return None
    fb = getattr(round_eval, "full_benchmark", None)
    verified = fb.verified_speedup if fb is not None and getattr(fb, "verified_speedup", None) is not None else None
    if verified is None:
        verified = getattr(round_eval, "benchmark_speedup", None)
    if verified is None:
        return None
    return float(verified), round_eval.best_patch, _read_per_shape_speedups(output_dir, step_index)


def _median_per_shape(per_shapes: list[dict]) -> dict:
    """Per-shape median across repeated measurements (outlier-robust, B1)."""
    import statistics

    keys: set[str] = set()
    for ps in per_shapes:
        keys.update(ps.keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [ps[k] for ps in per_shapes if k in ps]
        if vals:
            out[k] = float(statistics.median(vals))
    return out


def _peek_candidate(output_dir: Path, step_index: int) -> tuple[float | None, Path | None]:
    """Read the agent's self-reported best (speedup + patch file) before verifying.

    Used by C2 (dedup) and C3 (skip-on-no-gain) to avoid an expensive
    FULL_BENCHMARK when it cannot change the outcome.
    """
    import json as _json

    results_dir = output_dir / "results" / f"round_{step_index}"
    best_sp: float | None = None
    best_patch: Path | None = None
    if not results_dir.is_dir():
        return None, None
    for br in results_dir.glob("*/best_results.json"):
        try:
            data = _json.loads(br.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sp = data.get("best_patch_speedup")
        pf = data.get("best_patch_file")
        try:
            sp = float(sp) if sp is not None else None
        except (TypeError, ValueError):
            sp = None
        if sp is not None and (best_sp is None or sp > best_sp):
            best_sp = sp
            best_patch = Path(pf) if pf else None
    return best_sp, best_patch


def _patch_fingerprint(patch_file: Path | None) -> str | None:
    if patch_file is None or not Path(patch_file).exists():
        return None
    try:
        from minisweagent.run.avo.result import patch_hash

        return patch_hash(Path(patch_file).read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return None


def _apply_verified_score(
    result,
    verify_ctx: dict,
    step_index: int,
    output_dir: Path,
    *,
    repeats: int = 1,
    min_commit_speedup: float = 1.0,
    skip_on_no_gain: bool = True,
    cache: dict | None = None,
    cache_path: Path | None = None,
) -> None:
    """Overwrite the step's best score with a noise-robust verified speedup (B1).

    Runs GEAK's ``evaluate_round_best`` ``repeats`` times (each = fresh worktree +
    FULL_BENCHMARK over ``results/round_{step}/``) and uses the **median** speedup
    (and per-shape medians) to suppress measurement-noise spikes that would
    otherwise drive false commits or false stalls. ``repeats=1`` preserves the
    original single-shot behavior (no extra GPU cost).

    Cost guards: C3 skips verification when the agent's self-reported best shows
    no gain (cannot commit anyway); C2 reuses a cached verified result when the
    same patch (by content hash) was already benchmarked.
    """
    import statistics

    from minisweagent.run.avo.result import AttemptRecord

    self_sp, cand_patch = _peek_candidate(output_dir, step_index)

    # C3: agent itself reports no gain → cannot clear the commit floor; skip the
    # expensive FULL_BENCHMARK and treat as no committable candidate.
    if skip_on_no_gain and self_sp is not None and self_sp <= min_commit_speedup:
        logger.info(
            "verify step %d: agent self-reports %.4fx (<= floor %.4f); skipping verification.",
            step_index,
            self_sp,
            min_commit_speedup,
        )
        result.best_speedup = None
        result.best_correct = False
        result.best_patch_path = None
        return

    # C2: identical patch already verified → reuse cached result, skip re-bench.
    fp = _patch_fingerprint(cand_patch)
    if cache is not None and fp and fp in cache:
        cached = cache[fp]
        logger.info("verify step %d: cache hit (%s) → %.4fx (skipping re-benchmark).", step_index, fp, cached["speedup"])
        result.best_speedup = float(cached["speedup"])
        result.best_correct = True
        result.best_patch_path = Path(cached["patch"]) if cached.get("patch") else cand_patch
        result.per_shape_speedups = dict(cached.get("per_shape") or {})
        result.attempts.append(
            AttemptRecord(
                strategy=result.strategy,
                returncode=0,
                correctness_passed=True,
                verified_speedup=float(cached["speedup"]),
                patch_hash=fp,
                ts=time.time(),
            )
        )
        return

    n = max(1, int(repeats))
    samples: list[float] = []
    per_shapes: list[dict] = []
    best_patch: str | None = None
    for i in range(n):
        out = _eval_once(verify_ctx, step_index, output_dir)
        if out is None:
            continue
        sp, patch, per_shape = out
        samples.append(sp)
        per_shapes.append(per_shape)
        best_patch = patch

    if not samples or best_patch is None:
        result.best_speedup = None
        result.best_correct = False
        result.best_patch_path = None
        return

    verified = statistics.median(samples)
    if n > 1:
        spread = (max(samples) - min(samples)) / verified if verified else 0.0
        logger.info(
            "verify step %d: %d samples, median=%.4fx (min=%.4f max=%.4f spread=%.1f%%)",
            step_index,
            len(samples),
            verified,
            min(samples),
            max(samples),
            spread * 100.0,
        )

    result.best_speedup = float(verified)
    result.best_correct = True
    result.best_patch_path = Path(best_patch)
    result.per_shape_speedups = _median_per_shape(per_shapes) if n > 1 else (per_shapes[0] if per_shapes else {})
    result.profiling = _read_profile_metrics(output_dir, step_index)  # P-mem-3 causal signal
    # Record an authoritative attempt so the detector sees the verified outcome.
    result.attempts.append(
        AttemptRecord(
            strategy=result.strategy,
            returncode=0,
            correctness_passed=True,
            verified_speedup=float(verified),
            patch_hash=fp,
            ts=time.time(),
        )
    )
    # C2: remember this verified result keyed by patch content hash.
    if cache is not None and fp:
        cache[fp] = {"speedup": float(verified), "patch": str(best_patch), "per_shape": result.per_shape_speedups}
        if cache_path is not None:
            try:
                import json as _json

                cache_path.write_text(_json.dumps(cache, indent=2, default=str), encoding="utf-8")
            except OSError:
                logger.debug("verify cache write failed (non-fatal)")
    logger.info("verify step %d: verified geomean speedup = %.4fx", step_index, verified)


def _do_redirect(signal, lineage, step_dir, output_dir, strategy_file, detector, model_factory, repo) -> None:
    """Invoke the LLM supervisor, apply its directive (incl. backtrack), reset counters."""
    detector.note_supervisor_cycle()
    cycle = detector.supervisor_cycles_without_commit
    console.print(f"[yellow]AVO supervisor[/yellow] (cycle {cycle}): {signal.reason}")
    bundle = build_bundle(signal, lineage, step_dir, output_dir)
    model = None
    try:
        model = model_factory()
    except Exception as exc:  # noqa: BLE001
        logger.warning("supervisor: could not build model (%s); using fallback taxonomy.", exc)
    directive = run_supervisor(bundle, {}, model=model)
    apply_directive(directive, lineage, strategy_file, supervisor_cycle=cycle, repo=repo)
    detector.reset(partial=True)


def _do_escalate(lineage, output_dir, config, model_factory, verify_ctx, repo, *, base_task: str) -> None:
    """Diversified rescue: run a few variation steps under distinct directions,
    evaluate them together with GEAK's multi-candidate evaluator, and fold the
    best verified result into the lineage (P1).

    This reuses ``evaluate_round_best`` — which already selects the best among
    multiple worker dirs in a round — instead of constructing a full
    ``PipelineContext``, keeping the rescue self-contained and low-risk.
    """
    avo_cfg = dict(config.get("avo", {}))
    esc = dict(avo_cfg.get("escalate", {}))
    if not esc.get("enabled", True):
        return
    n_workers = int(esc.get("rescue_workers", 4))
    rescue_round = 9000 + len(lineage.committed)  # unique round id, away from normal steps
    console.print(f"[magenta]AVO escalate[/magenta]: diversified rescue with {n_workers} workers (round {rescue_round}).")

    lineage.reset_worktree_to_best(repo)
    directions = _diversified_directions(n_workers)
    rescue_dir = Path(output_dir) / "results" / f"round_{rescue_round}"
    rescue_dir.mkdir(parents=True, exist_ok=True)
    notebook_root = Path(output_dir) / "avo_state" / "notebook"

    for k, strat in enumerate(directions):
        worker_dir = rescue_dir / f"rescue-worker-{k}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        step_dir = Path(output_dir) / f"escalate_{rescue_round}_{k}"
        try:
            run_variation_step(
                repo=repo,
                base_task=base_task,
                step_dir=step_dir,
                lineage=lineage,
                direction={"strategy": strat, "assigned_by": "escalate", "supervisor_cycle": 0},
                output_dir=output_dir,
                avo_config=_with_patch_dir(avo_cfg, worker_dir),
                model_factory=model_factory,
                notebook_root=notebook_root,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AVO escalate: rescue worker %d failed (%s).", k, exc)

    try:
        from minisweagent.run.postprocess.evaluation import evaluate_round_best

        round_eval = evaluate_round_best(dict(verify_ctx), rescue_round, rescue_dir)
        if lineage.commit_from_round(round_eval, repo=repo):
            console.print(f"[green]AVO escalate[/green]: rescue produced a new best ({lineage.best_speedup:.3f}x).")
        else:
            logger.info("AVO escalate: rescue did not beat current best.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("AVO escalate: evaluation failed (non-fatal): %s", exc)


def _diversified_directions(n: int) -> list[str]:
    """Pick N distinct generic directions for an ESCALATE rescue."""
    from minisweagent.run.avo.supervisor import _FALLBACK_TAXONOMY

    names = [t["name"] for t in _FALLBACK_TAXONOMY]
    return names[:n] if n <= len(names) else names + names[: n - len(names)]


def _with_patch_dir(avo_cfg: dict, worker_dir: Path) -> dict:
    """ESCALATE workers write into their own rescue worker dir.

    ``run_variation_step`` derives the worker dir from ``output_dir`` + step
    index, so for the rescue we instead point the agent's patch_output_dir via a
    config hint consumed by variation_step. The default path is unchanged for
    normal steps.
    """
    cfg = dict(avo_cfg)
    cfg["_escalate_patch_dir"] = str(worker_dir)
    return cfg


def _finalize(output_dir: Path, lineage: LineageStore) -> dict:
    """Reuse GEAK's auto_finalize so final_report.json keeps its canonical shape."""
    try:
        from minisweagent.run.postprocess.results import auto_finalize

        ctx = lineage.build_postprocess_ctx(output_dir)
        report = auto_finalize(ctx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AVO finalize: auto_finalize failed (%s); writing minimal report.", exc)
        report = lineage.build_postprocess_ctx(output_dir)
    report["avo"] = {
        "committed_versions": len(lineage.committed),
        "best_id": lineage.best_id,
        "best_speedup": lineage.best_speedup,
    }
    import json

    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_trajectory(output_dir, lineage)
    console.print(
        f"[green]AVO done[/green]: {len(lineage.committed)} committed versions, "
        f"best={lineage.best_id} ({lineage.best_speedup:.3f}x)."
    )
    return report


def _write_trajectory(output_dir: Path, lineage: LineageStore) -> None:
    """Emit ``trajectory.json``: evolution health at a glance (observability).

    Best-effort: never raises. Captures the running-best curve over commits,
    total attempts vs commits, per-strategy stats, and supervisor interventions.
    """
    import json as _json

    try:
        curve = []
        running_best = 0.0
        per_strategy: dict[str, dict] = {}
        for node in lineage.committed:
            running_best = max(running_best, node.speedup)
            curve.append({"id": node.id, "speedup": node.speedup, "running_best": running_best, "strategy": node.strategy})
            st = per_strategy.setdefault(node.strategy or "?", {"commits": 0, "best": 0.0})
            st["commits"] += 1
            st["best"] = max(st["best"], node.speedup)

        attempts = 0
        attempts_path = lineage.attempts_path
        if attempts_path.exists():
            attempts = sum(1 for line in attempts_path.read_text(encoding="utf-8").splitlines() if line.strip())

        interventions = 0
        sup_log = lineage.state_dir / "supervisor_log.jsonl"
        if sup_log.exists():
            interventions = sum(1 for line in sup_log.read_text(encoding="utf-8").splitlines() if line.strip())

        committed = max(0, len(lineage.committed) - 1)  # exclude baseline v0
        trajectory = {
            "best_id": lineage.best_id,
            "best_speedup": lineage.best_speedup,
            "committed_versions": committed,
            "total_attempts": attempts,
            "commit_rate": round(committed / attempts, 4) if attempts else None,
            "supervisor_interventions": interventions,
            "best_speedup_curve": curve,
            "per_strategy": per_strategy,
        }
        (output_dir / "trajectory.json").write_text(_json.dumps(trajectory, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("trajectory.json write failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    repo: Path = typer.Option(..., "--repo", help="Target kernel repository root."),
    task: str = typer.Option(..., "-t", "--task", help="Optimization task description."),
    output: Path | None = typer.Option(None, "-o", "--output", help="Run output directory."),
    model: str | None = typer.Option(None, "-m", "--model", help="Model name override."),
    mode: str = typer.Option("full", "--mode", help="Budget mode: quick | full."),
    total_budget_s: float | None = typer.Option(None, "--total-budget-s", help="Override wall-clock cap (seconds)."),
    kernel_language: str = typer.Option("python", "--kernel-language", help="triton | hip | flydsl | python."),
    gpu_ids: str = typer.Option("0", "--gpu-ids", help="Comma-separated GPU device indices for evaluation."),
    config_path: str | None = typer.Option(None, "-c", "--config", help="Extra YAML config to merge last."),
) -> None:
    """Run a single-lineage AVO continuous-evolution session on a kernel repo."""
    config = load_avo_config(config_path)

    if output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = Path.cwd() / "optimization_logs" / f"avo_{repo.name}_{ts}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    add_file_handler(output / "avo_controller.log")

    budget = _build_budget(config, mode, total_budget_s)
    for line in budget.banner_lines():
        console.print(f"[bold cyan]{line}[/bold cyan]")

    parsed_gpu_ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()] or [0]

    try:
        run_avo(
            repo=repo.resolve(),
            task=task,
            output_dir=output,
            config=config,
            budget=budget,
            model_name=model,
            kernel_language=kernel_language,
            gpu_ids=parsed_gpu_ids,
            mode=mode,
            config_path=config_path,
        )
    except KeyboardInterrupt:
        console.print("[red]Interrupted.[/red]")
        sys.exit(130)


if __name__ == "__main__":
    app()
