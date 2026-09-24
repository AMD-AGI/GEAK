#!/usr/bin/env python3
"""Workload-comparability tests for run_e2e.normalize_result.

A cross-harness ratio is only defined when both harnesses measured the same
WORKLOAD, and the way that assumption breaks is silent. In the orchestrator's
AgentX mode the served load is a replay of real agentic traces (p50 ~89k input
tokens, p99 past 500k) while the handoff still carries the CLI's synthetic
``isl``/``osl`` defaults of 1024/1024. A GEAK run that believes those defaults
measures a ~1k-token synthetic sweep and divides it into an agentic denominator.

That is not hypothetical. On Kimi-K3, GEAK's synthetic client measured 465.676
tok/s at ISL/OSL=1024 against Hyperloom's 168.998 tok/s agentic baseline, which
an ungated GEAK would publish as a 2.76x win with no kernel changed at all.

Two contracts are under test, and the FIRST one is the reason the second is safe
to ship:

  1. INVARIANT -- the established fixed-ISL/OSL path is untouched. Those
     handoffs classify as ``synthetic_isl_osl`` (or ``unknown`` on writers that
     predate any workload signal) and GEAK's synthetic client agrees, so every
     cross-harness field keeps exactly the value it has today. Suppression fires
     only on a POSITIVE mismatch between two KNOWN kinds.
  2. GUARD -- on a real mismatch every GEAK-over-orchestrator quantity becomes
     None with a recorded reason, while every WITHIN-GEAK ratio survives, since
     both of its legs were measured by the same client on the same workload.

Run: python3 -m pytest GEAK/interface/test_run_e2e_workload_comparability.py -v
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent

# The measured Kimi-K3 numbers behind this guard, kept as the fixture so a
# regression reproduces the exact false claim rather than an invented one.
HL_AGENTIC_BASELINE_TOK_S = 168.998
GEAK_SYNTHETIC_TOK_S = 465.676
FALSE_SPEEDUP = GEAK_SYNTHETIC_TOK_S / HL_AGENTIC_BASELINE_TOK_S  # ~2.76x

# Every field that divides a GEAK measurement by an orchestrator measurement.
CROSS_HARNESS_BASELINE_FIELDS = (
    "raw_session_baseline_divergence_pct",
    "current_best_same_config_divergence_pct",
    "measurement_divergence_pct",
    "gain_vs_orchestrator_baseline",
)
CROSS_HARNESS_ALIGNMENT_FIELDS = ("hot_speedup", "cold_speedup")


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


def _wf(eval_dir: Path, *, base: float, final: float, speedup: float) -> dict:
    return {
        "eval_dir": str(eval_dir),
        "baseline_throughput_tok_s": base,
        "final_throughput_tok_s": final,
        "throughput_speedup": speedup,
        "output_parity": "pass",
    }


def _recipe(tmp_path: Path, benchmark_script: str) -> str:
    """Write a launch recipe carrying just the fields run_e2e scans for."""
    path = tmp_path / f"recipe_{benchmark_script.replace('.', '_')}.yaml"
    path.write_text(
        "framework: vllm\n"
        "runner_type: mi355x\n"
        f"benchmark_script: {benchmark_script}\n"
        "inferencex_path: /src/Hyperloom/.cache/InferenceX@3d55815\n",
        encoding="utf-8",
    )
    return str(path)


def _cold_legs(eval_dir: Path, *, base_cold: float, final_cold: float) -> None:
    """Materialize the cold rounds so cold_speedup is actually exercised."""
    (eval_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (eval_dir / "validation" / "final").mkdir(parents=True, exist_ok=True)
    (eval_dir / "baseline" / "bench_summary.json").write_text(
        json.dumps(
            {
                "output_throughput_tok_s_median": base_cold * 1.02,
                "cold_output_throughput_tok_s": base_cold,
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "validation" / "final" / "bench_summary.json").write_text(
        json.dumps(
            {
                "output_throughput_tok_s_median": final_cold * 1.02,
                "cold_output_throughput_tok_s": final_cold,
            }
        ),
        encoding="utf-8",
    )


# ─────────────────────────── contract 1: the invariant ───────────────────────
# These must keep passing unchanged. They are the fixed-ISL/OSL path.


def test_synthetic_recipe_keeps_every_cross_harness_number(tmp_path: Path) -> None:
    """A fixed-ISL/OSL recipe stays fully comparable: nothing is suppressed."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "raw_baseline_tput": 1331.7541295483402,
        "orchestrator_best_tput_same_config": 1872.9515966860333,
        # The ordinary case: the recipe names a SERVER launcher, not a client.
        "launch_recipe": _recipe(tmp_path, "vllm_mi355x.sh"),
    }
    out = rx.normalize_result(
        h, _wf(eval_dir, base=1785.741, final=2869.795, speedup=1.607)
    )
    bb = out["baseline_basis"]
    wc = bb["workload_comparability"]

    assert wc["comparable"] is True
    assert wc["orchestrator_workload_kind"] == rx.WORKLOAD_KIND_SYNTHETIC
    assert wc["orchestrator_workload_kind_source"] == "launch_recipe.benchmark_script"
    assert wc["geak_workload_kind"] == rx.WORKLOAD_KIND_SYNTHETIC
    assert wc["suppressed_reason"] is None

    # The numbers this path has always published are still numbers.
    for field in CROSS_HARNESS_BASELINE_FIELDS:
        assert bb[field] is not None, f"{field} must survive on the synthetic path"
    assert bb["current_best_same_config_divergence_pct"] == pytest.approx(
        -4.66, abs=0.01
    )
    assert bb["measurement_divergence_pct"] == (
        bb["current_best_same_config_divergence_pct"]
    )


def test_handoff_without_any_workload_signal_stays_comparable(
    tmp_path: Path,
) -> None:
    """No recipe and no workload_spec must not be read as a mismatch.

    Every handoff written before a workload signal existed lands here, so
    inferring a mismatch from silence would retroactively blank the cross-harness
    fields on runs that are perfectly valid.
    """
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    h = {
        "workload": {"isl": 16384, "osl": 512, "conc": 16},
        "raw_baseline_tput": 498.81,
        "orchestrator_best_tput_same_config": 537.15,
    }
    out = rx.normalize_result(h, _wf(eval_dir, base=537.354, final=532.119, speedup=0.99))
    bb = out["baseline_basis"]
    wc = bb["workload_comparability"]

    assert wc["comparable"] is True
    assert wc["orchestrator_workload_kind"] == rx.WORKLOAD_KIND_UNKNOWN
    assert wc["orchestrator_workload_kind_source"] == "unavailable"
    assert bb["raw_session_baseline_divergence_pct"] == pytest.approx(7.73, abs=0.01)
    assert bb["current_best_same_config_divergence_pct"] == pytest.approx(
        0.04, abs=0.01
    )


# ──────────────────────────── contract 2: the guard ──────────────────────────


def test_agentx_baseline_vs_synthetic_geak_suppresses_the_false_speedup(
    tmp_path: Path,
) -> None:
    """The measured Kimi-K3 mismatch must not publish its 2.76x."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    _cold_legs(eval_dir, base_cold=450.0, final_cold=GEAK_SYNTHETIC_TOK_S)
    h = {
        # The CLI defaults the handoff carries -- ~100x off the real load.
        "workload": {"isl": 1024, "osl": 1024, "conc": 8},
        "raw_baseline_tput": HL_AGENTIC_BASELINE_TOK_S,
        "orchestrator_best_tput_same_config": HL_AGENTIC_BASELINE_TOK_S,
        # AgentX rewrote benchmark_script to the aiperf CLIENT: the sentinel.
        "launch_recipe": _recipe(tmp_path, rx.AGENTX_CLIENT_SCRIPT),
    }
    out = rx.normalize_result(
        h,
        _wf(
            eval_dir,
            base=GEAK_SYNTHETIC_TOK_S,
            final=GEAK_SYNTHETIC_TOK_S,
            speedup=1.0,
        ),
    )
    bb = out["baseline_basis"]
    am = out["alignment_metrics"]
    wc = bb["workload_comparability"]

    assert wc["comparable"] is False
    assert wc["orchestrator_workload_kind"] == rx.WORKLOAD_KIND_AGENTX
    assert wc["geak_workload_kind"] == rx.WORKLOAD_KIND_SYNTHETIC
    assert "kernel speedup" in wc["suppressed_reason"]

    for field in CROSS_HARNESS_BASELINE_FIELDS:
        assert bb[field] is None, f"{field} must be suppressed across workload kinds"
    for field in CROSS_HARNESS_ALIGNMENT_FIELDS:
        assert am[field] is None, f"{field} must be suppressed across workload kinds"

    # The specific number that must never appear anywhere in the payload.
    assert FALSE_SPEEDUP == pytest.approx(2.7555, abs=1e-3)
    published = json.dumps(out)
    assert f"{FALSE_SPEEDUP:.4f}" not in published
    assert f"{FALSE_SPEEDUP:.2f}" not in published


def test_suppression_keeps_raw_inputs_and_within_geak_ratios(
    tmp_path: Path,
) -> None:
    """Suppress the RATIOS, not the evidence.

    A reviewer still needs both sides' raw measurements to see why the comparison
    was refused, and the within-GEAK ratios remain valid because both of their
    legs came from the same client on the same workload.
    """
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    _cold_legs(eval_dir, base_cold=430.0, final_cold=480.0)
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 8},
        "raw_baseline_tput": HL_AGENTIC_BASELINE_TOK_S,
        "launch_recipe": _recipe(tmp_path, rx.AGENTX_CLIENT_SCRIPT),
    }
    out = rx.normalize_result(h, _wf(eval_dir, base=450.0, final=500.0, speedup=1.111))
    bb = out["baseline_basis"]
    am = out["alignment_metrics"]

    # Evidence survives.
    assert bb["orchestrator_baseline_tok_s"] == pytest.approx(
        HL_AGENTIC_BASELINE_TOK_S
    )
    assert bb["geak_measured_baseline_tok_s"] == pytest.approx(450.0)
    assert am["orchestrator_cold_baseline_tok_s"] == pytest.approx(
        HL_AGENTIC_BASELINE_TOK_S
    )

    # Within-GEAK ratios survive; cross-harness ones do not.
    assert am["hot_geak_speedup"] == pytest.approx(500.0 / 450.0, abs=1e-4)
    assert am["cold_geak_speedup"] is not None
    assert am["hot_speedup"] is None
    assert am["cold_speedup"] is None


def test_workload_spec_is_authoritative_over_the_recipe_sentinel(
    tmp_path: Path,
) -> None:
    """An explicit workload_spec.kind wins, and is reported as the source."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 8},
        "raw_baseline_tput": HL_AGENTIC_BASELINE_TOK_S,
        # No recipe at all: without the spec this would be "unknown".
        "workload_spec": {"kind": rx.WORKLOAD_KIND_AGENTX, "client": "aiperf"},
    }
    out = rx.normalize_result(h, _wf(eval_dir, base=465.0, final=470.0, speedup=1.01))
    wc = out["baseline_basis"]["workload_comparability"]

    assert wc["comparable"] is False
    assert wc["orchestrator_workload_kind"] == rx.WORKLOAD_KIND_AGENTX
    assert wc["orchestrator_workload_kind_source"] == "handoff.workload_spec.kind"


def test_agentx_on_both_sides_restores_the_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    """Matching agentic workloads are comparable again.

    This is the end state worth having: GEAK drives the same trace replay, so its
    number and Hyperloom's are the same measurement and the ratio is meaningful.
    """
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    monkeypatch.setenv("BENCH_CLIENT", rx.AGENTX_BENCH_CLIENT)
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 8},
        "raw_baseline_tput": HL_AGENTIC_BASELINE_TOK_S,
        "orchestrator_best_tput_same_config": HL_AGENTIC_BASELINE_TOK_S,
        "launch_recipe": _recipe(tmp_path, rx.AGENTX_CLIENT_SCRIPT),
    }
    # GEAK reproducing Hyperloom's baseline: the campaign's own accidental
    # agentic run landed at 170.106 against 168.998, i.e. +0.66%.
    out = rx.normalize_result(h, _wf(eval_dir, base=170.106, final=175.0, speedup=1.03))
    bb = out["baseline_basis"]
    wc = bb["workload_comparability"]

    assert wc["comparable"] is True
    assert wc["orchestrator_workload_kind"] == rx.WORKLOAD_KIND_AGENTX
    assert wc["geak_workload_kind"] == rx.WORKLOAD_KIND_AGENTX
    assert bb["raw_session_baseline_divergence_pct"] == pytest.approx(0.66, abs=0.01)
    assert bb["gain_vs_orchestrator_baseline"] is not None


def test_metric_basis_is_documented_as_not_discriminating(tmp_path: Path) -> None:
    """Both workloads report aggregate_output_tok_s, so bases cannot decide.

    Recorded explicitly because it is the trap: the obvious guard would compare
    metric_basis, see two matching strings, and conclude the loads matched.
    """
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 8},
        "raw_baseline_tput": HL_AGENTIC_BASELINE_TOK_S,
        "launch_recipe": _recipe(tmp_path, rx.AGENTX_CLIENT_SCRIPT),
    }
    out = rx.normalize_result(h, _wf(eval_dir, base=465.0, final=470.0, speedup=1.01))
    wc = out["baseline_basis"]["workload_comparability"]

    assert wc["metric_basis_discriminates"] is False


def test_kind_helpers_classify_directly() -> None:
    """Unit-level coverage of the two classifiers."""
    assert rx._geak_workload_kind() == rx.WORKLOAD_KIND_SYNTHETIC
    assert rx._baseline_workload_kind({}) == (
        rx.WORKLOAD_KIND_UNKNOWN,
        "unavailable",
    )
    # A garbage recipe path is unreadable, not a mismatch.
    assert rx._baseline_workload_kind({"launch_recipe": "/nope/missing.yaml"}) == (
        rx.WORKLOAD_KIND_UNKNOWN,
        "unavailable",
    )
