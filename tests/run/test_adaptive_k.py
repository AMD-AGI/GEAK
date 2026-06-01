"""Tests for the adaptive-K dispatcher and the kind-resolution contract.

Locks three things:

1. ``Dispatcher._k_for_mode`` — the adaptive allocation algorithm: planned/fixed
   passthroughs, mixed-mode round-1 fallback, proportional shift, exploration
   floor (clamping to ``[1, n-1]``), one-sided history, zero-speedup fallback,
   and the N=2 short-circuit.

2. ``_resolve_task_kind`` — YAML lookup from the task .md frontmatter, including
   substring label matching and the conservative ``"planned"`` default.

3. The architectural contract that ``kind`` is dispatcher scheduling metadata
   only and MUST NOT leak into agent ``cfg`` via the ``dispatch.py``
   ``_passthrough_key`` loop. This is the regression guard for the bug where
   passing ``kind`` through caused every worker to crash with
   ``TypeError: AgentConfig.__init__() got an unexpected keyword argument
   'kind'``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from minisweagent.run.dispatcher.selector import Dispatcher
from minisweagent.run.postprocess.evaluation import _resolve_task_kind, evaluate_round_best

# ---------------------------------------------------------------------------
# _k_for_mode behavior
# ---------------------------------------------------------------------------


class TestKForModeBasicModes:
    """fixed/planned modes are pure passthroughs; mixed has the interesting math."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16])
    def test_fixed_mode_returns_zero(self, n: int) -> None:
        assert Dispatcher._k_for_mode("fixed", n) == 0

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16])
    def test_planned_mode_returns_n(self, n: int) -> None:
        assert Dispatcher._k_for_mode("planned", n) == n

    def test_fixed_mode_ignores_round_evals(self) -> None:
        evals = [{"per_task": [{"kind": "planned", "speedup": 99.0}]}]
        assert Dispatcher._k_for_mode("fixed", 4, round_evals=evals) == 0

    def test_planned_mode_ignores_round_evals(self) -> None:
        evals = [{"per_task": [{"kind": "fixed", "speedup": 99.0}]}]
        assert Dispatcher._k_for_mode("planned", 4, round_evals=evals) == 4


class TestKForModeMixedFallback:
    """Round 1 (or any round without per_task data) falls back to N//2."""

    def test_mixed_no_round_evals_falls_back_to_half(self) -> None:
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=None) == 2

    def test_mixed_empty_round_evals_falls_back_to_half(self) -> None:
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=[]) == 2

    def test_mixed_n_equals_2_short_circuits_to_one(self) -> None:
        # N<=2 is the documented "no adaptive shift" boundary: there's no
        # interesting allocation to make with only 2 slots so we always
        # do 1+1 regardless of history.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 5.0},
                    {"kind": "fixed", "speedup": 1.0},
                ]
            }
        ]
        assert Dispatcher._k_for_mode("mixed", 2, round_evals=evals) == 1

    def test_mixed_per_task_present_but_all_zero_speedups_falls_back(self) -> None:
        # Zero/negative speedups are filtered out, leaving the algorithm with
        # no usable history → N//2 fallback.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 0.0},
                    {"kind": "planned", "speedup": -1.0},
                    {"kind": "fixed", "speedup": 0.0},
                ]
            }
        ]
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=evals) == 2

    def test_mixed_per_task_missing_kind_or_speedup_is_ignored(self) -> None:
        evals = [
            {
                "per_task": [
                    {"speedup": 5.0},  # no kind
                    {"kind": "planned"},  # no speedup
                    {"kind": "other", "speedup": 5.0},  # neither planned nor fixed
                ]
            }
        ]
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=evals) == 2


class TestKForModeMixedAdaptive:
    """The interesting case: per_task entries drive proportional K selection."""

    def test_planned_dominant_shifts_k_up(self) -> None:
        # planned avg = 2.0, fixed avg = 1.0 → K = round(4 * 2/3) = 3.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 2.0},
                    {"kind": "planned", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 1.0},
                    {"kind": "fixed", "speedup": 1.0},
                ]
            }
        ]
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=evals) == 3

    def test_fixed_dominant_shifts_k_down_clamped_at_one(self) -> None:
        # planned avg = 1.0, fixed avg = 2.0 → K = round(4 * 1/3) = 1.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 1.0},
                    {"kind": "fixed", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 2.0},
                ]
            }
        ]
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=evals) == 1

    def test_extreme_planned_dominance_clamps_to_n_minus_1(self) -> None:
        # Even with planned >> fixed, the exploration floor reserves 1 slot
        # for fixed → K never reaches N. (50.0 stays under MAX_PLAUSIBLE_SPEEDUP
        # so it counts as a real planned peak rather than being clamped away.)
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 50.0},
                    {"kind": "fixed", "speedup": 1.0},
                ]
            }
        ]
        k = Dispatcher._k_for_mode("mixed", 4, round_evals=evals)
        assert k == 3
        assert k <= 4 - 1, "exploration floor: at least 1 fixed slot must remain"

    def test_extreme_fixed_dominance_clamps_to_one(self) -> None:
        # Even with fixed >> planned, the exploration floor reserves 1 slot
        # for planned → K never reaches 0 (which would be fixed mode). (50.0
        # stays under MAX_PLAUSIBLE_SPEEDUP so it counts as a real fixed peak.)
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 1.0},
                    {"kind": "fixed", "speedup": 50.0},
                ]
            }
        ]
        k = Dispatcher._k_for_mode("mixed", 4, round_evals=evals)
        assert k == 1
        assert k >= 1, "exploration floor: at least 1 planned slot must remain"

    def test_one_sided_history_only_planned_uses_default_for_other(self) -> None:
        # planned avg = 3.0, fixed avg defaults to 1.0 → K = round(4 * 3/4) = 3.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 3.0},
                    {"kind": "planned", "speedup": 3.0},
                ]
            }
        ]
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=evals) == 3

    def test_one_sided_history_only_fixed_uses_default_for_other(self) -> None:
        # planned avg defaults to 1.0, fixed avg = 3.0 → K = round(4 * 1/4) = 1.
        evals = [
            {
                "per_task": [
                    {"kind": "fixed", "speedup": 3.0},
                    {"kind": "fixed", "speedup": 3.0},
                ]
            }
        ]
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=evals) == 1

    def test_history_aggregates_across_multiple_rounds(self) -> None:
        # Two rounds, planned 2.0 each, fixed 1.0 each → planned_avg=2.0,
        # fixed_avg=1.0 → K=3. Same answer either way, but exercises the
        # multi-round accumulation path.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 1.0},
                ]
            },
            {
                "per_task": [
                    {"kind": "planned", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 1.0},
                ]
            },
        ]
        assert Dispatcher._k_for_mode("mixed", 4, round_evals=evals) == 3


class TestKForModeMaxSeekingClampFailurePenalty:
    """Regression tests for the max-seeking + clamp + failure-penalty rework.

    The audit of the full 18-kernel run showed the old mean-based signal
    converged planned-vs-fixed to parity (K pinned at N//2). These lock the
    three properties that fix that: max-seeking (peak, not mean), the
    plausibility clamp, and the success-rate failure penalty.
    """

    def test_max_seeking_beats_mean_parity(self) -> None:
        # planned [5,1,1,1] and fixed [2,2,2,2] have the SAME mean (2.0): the
        # old mean-based signal tied them at K=n//2=6. Max-seeking rewards
        # planned's best (5.0) → K shifts decisively toward planned.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 5.0},
                    {"kind": "planned", "speedup": 1.0},
                    {"kind": "planned", "speedup": 1.0},
                    {"kind": "planned", "speedup": 1.0},
                    {"kind": "fixed", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 2.0},
                ]
            }
        ]
        # planned_score = 5.0 * (4/4) = 5.0; fixed_score = 2.0 * (4/4) = 2.0
        # K = round(12 * 5/7) = 9  (mean-based parity would have been 6)
        k = Dispatcher._k_for_mode("mixed", 12, round_evals=evals)
        assert k == 9
        assert k > 12 // 2, "max-seeking must beat the mean-parity K=n//2"

    def test_plausibility_clamp_drops_garbage_speedups(self) -> None:
        # 1e6 and a >=100 value are timing-saturation / garbage and must be
        # dropped. Without the clamp planned's 1e6 peak would pin K at n-1;
        # with it, K is driven by the plausible 3.0 vs fixed 2.0 only.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 1e6},
                    {"kind": "planned", "speedup": 120.0},
                    {"kind": "planned", "speedup": 3.0},
                    {"kind": "fixed", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 2.0},
                ]
            }
        ]
        # planned: plausible peak 3.0, success_rate 1/3 → score 1.0
        # fixed:   peak 2.0, success_rate 2/2 → score 2.0
        # K = round(8 * 1.0/3.0) = 3   (NOT 7 = n-1)
        k = Dispatcher._k_for_mode("mixed", 8, round_evals=evals)
        assert k == 3
        assert k < 8 - 1, "clamped garbage must not pin K at the planned ceiling"

    def test_failure_penalty_demotes_low_success_rate_source(self) -> None:
        # planned has a higher PEAK (5.0) than fixed (2.0) but only 1 of 6
        # dispatched planned tasks produced a plausible result; the success-rate
        # discount drops planned below fixed so K stays toward fixed.
        per_task: list[dict[str, Any]] = [{"kind": "planned", "speedup": 5.0, "status": "ok"}]
        per_task += [{"kind": "planned", "speedup": 0.0, "status": "failed"} for _ in range(5)]
        per_task += [{"kind": "fixed", "speedup": 2.0, "status": "ok"} for _ in range(6)]
        evals = [{"per_task": per_task}]
        # planned_score = 5.0 * (1/6) = 0.833; fixed_score = 2.0 * (6/6) = 2.0
        # K = round(8 * 0.833/2.833) = 2
        k = Dispatcher._k_for_mode("mixed", 8, round_evals=evals)
        assert k == 2
        assert k < 8 // 2, "failure-prone planned must lose slots despite higher peak"

    def test_all_failed_source_scores_neutral(self) -> None:
        # planned dispatched 3 tasks, all failed (no plausible speed) → neutral
        # score 1.0; fixed is strong → K clamps toward fixed but stays >= 1.
        per_task: list[dict[str, Any]] = [
            {"kind": "planned", "speedup": 0.0, "status": "failed"} for _ in range(3)
        ]
        per_task += [{"kind": "fixed", "speedup": 4.0, "status": "ok"} for _ in range(3)]
        evals = [{"per_task": per_task}]
        # planned_score = 1.0 (no plausible result); fixed_score = 4.0 * (3/3) = 4.0
        # K = round(8 * 1.0/5.0) = 2
        k = Dispatcher._k_for_mode("mixed", 8, round_evals=evals)
        assert k == 2
        assert k >= 1, "exploration floor keeps >= 1 planned slot even when all failed"

    def test_parity_returns_half(self) -> None:
        # Identical planned & fixed signals (same peak, same success rate) → n//2.
        evals = [
            {
                "per_task": [
                    {"kind": "planned", "speedup": 2.0},
                    {"kind": "planned", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 2.0},
                    {"kind": "fixed", "speedup": 2.0},
                ]
            }
        ]
        assert Dispatcher._k_for_mode("mixed", 8, round_evals=evals) == 8 // 2


class TestEdgePathCarriesPerTask:
    """Failure-path RoundEvaluations must carry per_task so partial-failure
    rounds still feed adaptive-K (regression guard for the 'K pinned' bug)."""

    def test_missing_commandment_path_carries_per_task(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        pp_dir = tmp_path / "pp"
        pp_dir.mkdir()  # intentionally NO COMMANDMENT.md → triggers the edge path
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        task_a = results_dir / "task_a"
        task_a.mkdir()
        (task_a / "best_results.json").write_text(
            json.dumps(
                {"best_patch_speedup": 2.0, "best_patch_file": str(tmp_path / "a.patch")}
            ),
            encoding="utf-8",
        )
        ctx = {"output_dir": str(output_dir), "preprocess_dir": str(pp_dir)}

        result = evaluate_round_best(ctx, round_num=1, results_dir=results_dir)

        assert result is not None
        per_task = result.to_dict().get("per_task")
        assert per_task, "missing-COMMANDMENT path must carry per_task, not drop it"
        assert any(o.get("label") == "task_a" for o in per_task)


# ---------------------------------------------------------------------------
# _resolve_task_kind YAML lookup
# ---------------------------------------------------------------------------


def _write_task_md(dir_: Path, fname: str, kind: str) -> Path:
    """Write a minimal task .md file with YAML frontmatter and a body."""
    path = dir_ / fname
    path.write_text(
        f"---\nkind: {kind}\nlabel: {Path(fname).stem}\npriority: 5\n---\n"
        "task body — content does not matter for kind resolution\n",
        encoding="utf-8",
    )
    return path


class TestResolveTaskKind:
    def test_returns_planned_for_planned_labeled_file(self, tmp_path: Path) -> None:
        _write_task_md(tmp_path, "00_single-pass-online-softmax.md", "planned")
        assert _resolve_task_kind(tmp_path, "single-pass-online-softmax") == "planned"

    def test_returns_fixed_for_fixed_labeled_file(self, tmp_path: Path) -> None:
        _write_task_md(tmp_path, "05_fixed-canonical.md", "fixed")
        assert _resolve_task_kind(tmp_path, "fixed-canonical") == "fixed"

    def test_returns_default_when_directory_does_not_exist(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_round_dir"
        assert _resolve_task_kind(missing, "anything") == "planned"

    def test_returns_default_when_no_matching_file(self, tmp_path: Path) -> None:
        _write_task_md(tmp_path, "05_fixed-canonical.md", "fixed")
        assert _resolve_task_kind(tmp_path, "totally-unrelated-label") == "planned"

    def test_substring_label_match(self, tmp_path: Path) -> None:
        # Numeric prefix and trailing kernel-name suffix should both match —
        # the writer-produced file naming includes both, but the evaluator
        # only knows the bare label from candidate metadata.
        _write_task_md(tmp_path, "00_single-pass-online-softmax.md", "planned")
        assert _resolve_task_kind(tmp_path, "single-pass-online") == "planned"

    def test_only_md_files_are_inspected(self, tmp_path: Path) -> None:
        # A non-.md file with a matching name must be ignored: the function
        # only treats files with .md suffix as task files.
        (tmp_path / "00_planned-thing.txt").write_text("kind: fixed\n", encoding="utf-8")
        assert _resolve_task_kind(tmp_path, "planned-thing") == "planned"

    def test_planned_and_fixed_in_same_dir_are_disambiguated_by_label(
        self, tmp_path: Path
    ) -> None:
        _write_task_md(tmp_path, "00_alpha.md", "planned")
        _write_task_md(tmp_path, "05_beta.md", "fixed")
        assert _resolve_task_kind(tmp_path, "alpha") == "planned"
        assert _resolve_task_kind(tmp_path, "beta") == "fixed"


# ---------------------------------------------------------------------------
# Architectural contract: kind MUST NOT leak into agent cfg
# ---------------------------------------------------------------------------


def _apply_dispatch_passthrough(meta: dict[str, Any]) -> dict[str, Any]:
    """Mirror the cfg-construction passthrough loop from
    ``minisweagent.run.dispatch.task_file_to_agent_task``.

    This deliberately re-implements the ~3-line loop instead of running the
    full dispatch flow so the contract is locked at the cfg-shape level — if
    the source loop drifts to re-include ``kind`` (or any other dispatcher
    metadata), this test will fail and force a documentation moment.
    """
    cfg: dict[str, Any] = {}
    for _passthrough_key in ("baseline_metrics", "benchmark_baseline"):
        if meta.get(_passthrough_key):
            cfg[_passthrough_key] = meta[_passthrough_key]
    return cfg


class TestKindIsNotInAgentCfg:
    """Regression guard for the AgentConfig(**cfg) crash bug."""

    def test_kind_in_meta_does_not_leak_into_cfg(self) -> None:
        meta = {
            "kind": "planned",
            "baseline_metrics": "/tmp/baseline.json",
            "benchmark_baseline": "/tmp/bench.json",
            "label": "single-pass-online-softmax",
        }
        cfg = _apply_dispatch_passthrough(meta)
        assert "kind" not in cfg, (
            "kind is dispatcher scheduling metadata and MUST NOT be passed to "
            "AgentConfig via the cfg passthrough loop in dispatch.py — that path "
            "previously caused every worker to crash with `AgentConfig.__init__() "
            "got an unexpected keyword argument 'kind'` because AgentConfig has no "
            "kind field. kind lives in the task .md YAML; the evaluator reads it "
            "via _resolve_task_kind. Do not re-add it to the passthrough tuple."
        )

    def test_baseline_metrics_still_passes_through(self) -> None:
        meta = {"kind": "planned", "baseline_metrics": "/tmp/baseline.json"}
        cfg = _apply_dispatch_passthrough(meta)
        assert cfg.get("baseline_metrics") == "/tmp/baseline.json"

    def test_benchmark_baseline_still_passes_through(self) -> None:
        meta = {"kind": "fixed", "benchmark_baseline": "/tmp/bench.json"}
        cfg = _apply_dispatch_passthrough(meta)
        assert cfg.get("benchmark_baseline") == "/tmp/bench.json"

    def test_source_passthrough_loop_is_kind_free(self) -> None:
        # Defensive: read the source line and assert it does not list "kind"
        # in the passthrough tuple. This catches a drift even if someone
        # bypasses the helper above by editing dispatch.py directly.
        import minisweagent.run.dispatch as dispatch_mod

        src = Path(dispatch_mod.__file__).read_text(encoding="utf-8")
        passthrough_lines = [
            line for line in src.splitlines() if "_passthrough_key in (" in line
        ]
        assert passthrough_lines, "passthrough loop not found in dispatch.py"
        for line in passthrough_lines:
            assert '"kind"' not in line and "'kind'" not in line, (
                f"dispatch.py passthrough loop must not include 'kind': {line!r}"
            )
