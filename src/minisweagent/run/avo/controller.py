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
from minisweagent.run.avo.variation_step import run_variation_step
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


def _run_preprocess(repo: Path, task: str, output_dir: Path, model_name: str | None) -> None:
    """Produce COMMANDMENT.md / baseline / profile via ``geak --preprocess-only``."""
    if (output_dir / "COMMANDMENT.md").exists():
        logger.info("preprocess: COMMANDMENT.md already present in %s; skipping.", output_dir)
        return
    cmd = ["geak", "--repo", str(repo), "--task", task, "-o", str(output_dir), "--preprocess-only", "--yolo"]
    if model_name:
        cmd += ["--model", model_name]
    logger.info("preprocess: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)
    if not (output_dir / "COMMANDMENT.md").exists():
        logger.warning("preprocess: COMMANDMENT.md not found after preprocess; the run may be degraded.")


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
) -> dict:
    """Run one single-lineage AVO evolution; return the final report dict."""
    avo_cfg = dict(config.get("avo") or {})
    model_cfg = config.get("model", {})

    def model_factory():
        return get_model(model_name, model_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── preprocess (reused) ───────────────────────────────────────────
    _run_preprocess(repo, task, output_dir, model_name)
    pp_elapsed = time.monotonic() - budget.started_at
    budget.commit_preprocess(pp_elapsed)
    budget.schedule_optimization_watchdog()

    # ── state ─────────────────────────────────────────────────────────
    lineage = LineageStore(
        output_dir / "avo_state",
        epsilon=float(avo_cfg.get("commit_epsilon", 0.001)),
        language=kernel_language,
    )
    lineage.seed_from_baseline(output_dir)
    detector = StagnationDetector(avo_cfg.get("stagnation", {}))
    strategy_file = output_dir / ".optimization_strategies.md"
    min_commits = int(avo_cfg.get("min_commits_before_stop", 0))

    step_idx = 0
    pending_nudge: str | None = None
    try:
        while not budget.soft_stop.is_set():
            step_idx += 1
            step_dir = output_dir / f"variation_{step_idx:04d}"
            lineage.reset_worktree_to_best(repo)
            lineage.heartbeat(step_index=step_idx)

            result = run_variation_step(
                repo=repo,
                base_task=task,
                step_dir=step_dir,
                lineage=lineage,
                direction=lineage.current_direction(),
                output_dir=output_dir,
                avo_config=avo_cfg,
                model_factory=model_factory,
                deadline=budget.optimization_deadline() if hasattr(budget, "optimization_deadline") else None,
                nudge=pending_nudge,
            )
            pending_nudge = None

            lineage.record_attempts(result)
            committed = lineage.maybe_commit(result, repo=repo)

            signal = detector.evaluate(result, committed)
            console.print(
                f"[cyan]AVO step {step_idx}[/cyan]: committed={committed} "
                f"level={signal.level.name} best={lineage.best_speedup:.3f}x — {signal.reason}"
            )

            if signal.level == StagnationLevel.NUDGE:
                pending_nudge = f"Progress is stalling ({signal.reason}). Try a different angle or mark the current strategy failed."
            elif signal.level == StagnationLevel.REDIRECT:
                _do_redirect(signal, lineage, step_dir, output_dir, strategy_file, detector, model_factory)
            elif signal.level == StagnationLevel.ESCALATE:
                _do_escalate(lineage, output_dir, config, model_factory, budget)
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


def _do_redirect(signal, lineage, step_dir, output_dir, strategy_file, detector, model_factory) -> None:
    """Invoke the LLM supervisor, apply its directive, reset stall counters."""
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
    apply_directive(directive, lineage, strategy_file, supervisor_cycle=cycle)
    detector.reset(partial=True)


def _do_escalate(lineage, output_dir, config, model_factory, budget) -> None:
    """Run one GEAK parallel ``planned`` round as a rescue; fold best into lineage."""
    esc = dict(config.get("avo", {}).get("escalate", {}))
    if not esc.get("enabled", True):
        return
    console.print("[magenta]AVO escalate[/magenta]: running one GEAK planned rescue round.")
    try:
        from minisweagent.run.postprocess.results import post_round_evaluate

        ctx = lineage.build_postprocess_ctx(output_dir)
        # The rescue round writes into results/round_1; reuse GEAK's evaluator.
        round_eval = post_round_evaluate(ctx, round_num=1, output_dir=Path(output_dir))
        if round_eval is not None:
            sp = getattr(round_eval, "benchmark_speedup", None)
            logger.info("AVO escalate: rescue round speedup=%s", sp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AVO escalate: rescue round failed (non-fatal): %s", exc)


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
    console.print(
        f"[green]AVO done[/green]: {len(lineage.committed)} committed versions, "
        f"best={lineage.best_id} ({lineage.best_speedup:.3f}x)."
    )
    return report


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

    try:
        run_avo(
            repo=repo.resolve(),
            task=task,
            output_dir=output,
            config=config,
            budget=budget,
            model_name=model,
            kernel_language=kernel_language,
        )
    except KeyboardInterrupt:
        console.print("[red]Interrupted.[/red]")
        sys.exit(130)


if __name__ == "__main__":
    app()
