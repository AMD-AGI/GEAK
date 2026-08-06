#!/usr/bin/env python3
"""Cross-harness alignment / credibility tests for run_e2e.normalize_result.

These guard the numbers Hyperloom reports for a GEAK e2e win against
two failure modes that inflate the leaderboard:

  * conflating the explore/framework CONFIG gain (baked into GEAK's seeded
    baseline) with pure cross-harness measurement residue, and
  * presenting a hot-numerator-over-cold-denominator ratio as the win.

The contract under test (see run_e2e normalize_result / baseline_basis +
alignment_metrics):

  * ``current_best_same_config_divergence_pct`` = GEAK baseline vs the
    orchestrator's tput on the SAME accepted config — the primary clean residue.
  * ``measurement_divergence_pct`` is the backward-compatible alias for that
    same-config metric.
  * ``raw_session_baseline_divergence_pct`` = GEAK baseline vs the orchestrator
    RAW baseline (conflates config gain + residue) — audit only.
  * ``cold_speedup`` = GEAK cold final / orchestrator COLD baseline — the exact
    number Hyperloom promotes as its (cross-harness) PROVISIONAL gain, so it must
    equal current_best.tput / baseline_tput.

Run: python3 -m pytest GEAK/interface/test_run_e2e_alignment.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


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


def test_issue6_names_raw_and_same_config_divergence_explicitly(
    tmp_path: Path,
) -> None:
    """Issue 6 values distinguish accepted gain from measurement residue."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    geak_baseline = 537.354
    orch_raw = 498.81
    orch_same_cfg = 537.15
    h = {
        "workload": {"isl": 16384, "osl": 512, "conc": 16},
        "raw_baseline_tput": orch_raw,
        "orchestrator_best_tput_same_config": orch_same_cfg,
    }
    wf = _wf(eval_dir, base=geak_baseline, final=532.119, speedup=0.9903)

    out = rx.normalize_result(h, wf)
    bb = out["baseline_basis"]

    assert bb["raw_session_baseline_divergence_pct"] == pytest.approx(
        7.73, abs=0.01
    )
    assert bb["current_best_same_config_divergence_pct"] == pytest.approx(
        0.04, abs=0.01
    )
    assert (
        bb["measurement_divergence_pct"]
        == bb["current_best_same_config_divergence_pct"]
    )
    assert "baseline_divergence_pct" not in bb
    assert bb["orchestrator_best_tput_same_config"] == pytest.approx(orch_same_cfg)
    assert out["baseline_alignment"]["status"] == "aligned"


def test_same_config_divergence_above_threshold_is_warning(tmp_path: Path) -> None:
    """A real same-config mismatch warns without changing optimization status."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "raw_baseline_tput": 1331.7541295483402,
        "orchestrator_best_tput_same_config": 1872.9515966860333,
    }
    wf = _wf(eval_dir, base=1785.741, final=2869.795, speedup=1.607)

    out = rx.normalize_result(h, wf)

    assert out["baseline_basis"][
        "current_best_same_config_divergence_pct"
    ] == pytest.approx(-4.66, abs=0.01)
    assert out["baseline_alignment"]["status"] == "warning"
    assert out["status"] == "ok"


def test_large_raw_gain_does_not_trigger_alignment_warning(tmp_path: Path) -> None:
    """Accepted config gain is audit-only when same-config measurements align."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "raw_baseline_tput": 100.0,
        "orchestrator_best_tput_same_config": 120.0,
    }
    wf = _wf(eval_dir, base=120.2, final=125.0, speedup=1.04)

    out = rx.normalize_result(h, wf)
    bb = out["baseline_basis"]

    assert bb["raw_session_baseline_divergence_pct"] == pytest.approx(
        20.2, abs=0.01
    )
    assert bb["current_best_same_config_divergence_pct"] == pytest.approx(
        0.17, abs=0.01
    )
    assert out["baseline_alignment"]["status"] == "aligned"


def test_same_config_alignment_unavailable_without_reference(tmp_path: Path) -> None:
    """Older handoffs never fall back to raw divergence for alignment."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "raw_baseline_tput": 2844.209,
        # orchestrator_best_tput_same_config intentionally omitted
    }
    wf = _wf(eval_dir, base=2974.662, final=3236.489, speedup=1.088)

    out = rx.normalize_result(h, wf)
    bb = out["baseline_basis"]
    assert bb["measurement_divergence_pct"] is None
    assert bb["current_best_same_config_divergence_pct"] is None
    assert bb["raw_session_baseline_divergence_pct"] is not None
    assert out["baseline_alignment"]["status"] == "unavailable"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_divergence_inputs_are_unavailable(
    tmp_path: Path,
    value: float,
) -> None:
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    h = {
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "raw_baseline_tput": 100.0,
        "orchestrator_best_tput_same_config": value,
    }
    wf = _wf(eval_dir, base=120.0, final=125.0, speedup=1.04)

    out = rx.normalize_result(h, wf)

    assert out["baseline_basis"]["measurement_divergence_pct"] is None
    assert out["baseline_basis"]["orchestrator_best_tput_same_config"] is None
    assert out["baseline_alignment"]["status"] == "unavailable"
    json.dumps(out, allow_nan=False)


def test_alignment_report_is_same_config_first_and_idempotent(
    tmp_path: Path,
) -> None:
    report = tmp_path / "final_report.md"
    report.write_text("# GEAK final report\n\nExisting content.\n", encoding="utf-8")
    result = {
        "report_path": str(report),
        "eval_dir": str(tmp_path),
        "baseline_basis": {
            "geak_measured_baseline_tok_s": 537.354,
            "orchestrator_baseline_tok_s": 498.81,
            "raw_session_baseline_divergence_pct": 7.73,
            "current_best_same_config_divergence_pct": 0.04,
            "measurement_divergence_pct": 0.04,
            "orchestrator_best_tput_same_config": 537.15,
        },
        "baseline_alignment": {
            "status": "aligned",
            "primary_metric": "current_best_same_config_divergence_pct",
            "divergence_pct": 0.04,
            "warning_threshold_pct": 3.0,
            "raw_session_divergence_is_measurement_signal": False,
        },
    }

    rx._update_baseline_alignment_reports(result)
    rx._update_baseline_alignment_reports(result)

    rendered = report.read_text(encoding="utf-8")
    assert rendered.count(rx.BASELINE_ALIGNMENT_BEGIN) == 1
    assert rendered.count(rx.BASELINE_ALIGNMENT_END) == 1
    assert rendered.index("Primary same-config comparison") < rendered.index(
        "Raw-session audit comparison"
    )
    assert "not a pure measurement-drift signal" in rendered


def test_alignment_report_refuses_path_outside_eval_dir(tmp_path: Path) -> None:
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# External report\n", encoding="utf-8")
    result = {
        "report_path": str(outside),
        "eval_dir": str(eval_dir),
        "baseline_basis": {},
        "baseline_alignment": {"status": "unavailable"},
    }

    assert rx._update_baseline_alignment_reports(result) == []
    assert outside.read_text(encoding="utf-8") == "# External report\n"


@pytest.mark.parametrize("val,want", [(1800, "1800"), ("900", "900")])
def test_apply_server_timeouts_exports_ceiling(val, want, monkeypatch) -> None:
    monkeypatch.delenv("SERVER_STARTUP_TIMEOUT_SEC", raising=False)
    assert rx.apply_server_timeouts({"server_startup_timeout_sec": val}) == {
        "SERVER_STARTUP_TIMEOUT_SEC": want
    }


@pytest.mark.parametrize("bad", [None, "", "abc", 0, -5])
def test_apply_server_timeouts_noop_on_absent_or_bad(bad, monkeypatch) -> None:
    """Absent/garbage => nothing exported, so bench_e2e.sh keeps its own default."""
    monkeypatch.delenv("SERVER_STARTUP_TIMEOUT_SEC", raising=False)
    h = {} if bad is None else {"server_startup_timeout_sec": bad}
    assert rx.apply_server_timeouts(h) == {}
    assert "SERVER_STARTUP_TIMEOUT_SEC" not in os.environ


def test_map_args_forwards_serving_fidelity_when_present(tmp_path: Path) -> None:
    """max_model_len / mem_fraction in the handoff reach ps_args (GEAK launch)."""
    h = {
        "model_path": "/models/gpt-oss-120b",
        "exp_root": str(tmp_path),
        "eval_dir": str(tmp_path / "e2e"),
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "tp": 8,
        "max_model_len": 2248,
        "mem_fraction": 0.9,
    }
    ps = rx.map_args(h)
    assert ps["max_model_len"] == 2248
    assert ps["mem_fraction"] == pytest.approx(0.9)


def test_map_args_omits_serving_fidelity_when_absent(tmp_path: Path) -> None:
    """No knobs in the handoff => ps_args carries none (adapter keeps defaults)."""
    h = {
        "model_path": "/models/gpt-oss-120b",
        "exp_root": str(tmp_path),
        "eval_dir": str(tmp_path / "e2e"),
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "tp": 8,
    }
    ps = rx.map_args(h)
    assert "max_model_len" not in ps
    assert "mem_fraction" not in ps


def _fidelity_handoff(tmp_path: Path, **extra) -> dict:
    h = {
        "model_path": "/models/gpt-oss-120b",
        "exp_root": str(tmp_path),
        "eval_dir": str(tmp_path / "e2e"),
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
        "tp": 8,
    }
    h.update(extra)
    return h


def test_fold_forwards_fidelity_flags_vllm(tmp_path: Path) -> None:
    """vllm handoff knobs are folded into initial_extra_server_args as vllm flags.

    The workflow applies initial_extra_server_args to every serving launch, so
    this is what makes GEAK launch the identical engine Hyperloom measured.
    """
    h = _fidelity_handoff(
        tmp_path,
        framework="vllm",
        accepted_flags="--max-num-batched-tokens 24576",
        max_model_len=2248,
        mem_fraction=0.9,
    )
    ps = rx.map_args(h)
    flags = ps["initial_extra_server_args"]
    # Seed flags preserved …
    assert "--max-num-batched-tokens 24576" in flags
    # … plus the two fidelity knobs as vllm-named flags.
    assert "--max-model-len 2248" in flags
    assert "--gpu-memory-utilization 0.9" in flags
    # Advisory standalone keys still present (unchanged contract).
    assert ps["max_model_len"] == 2248
    assert ps["mem_fraction"] == pytest.approx(0.9)


def test_fold_uses_sglang_flag_names(tmp_path: Path) -> None:
    """Same knobs translate to the sglang adapter's own flag names."""
    h = _fidelity_handoff(
        tmp_path,
        framework="sglang",
        accepted_flags="",
        max_model_len=4096,
        mem_fraction=0.92,
    )
    flags = rx.map_args(h)["initial_extra_server_args"]
    assert "--context-length 4096" in flags
    assert "--mem-fraction-static 0.92" in flags
    # And NOT the vllm names.
    assert "--max-model-len" not in flags
    assert "--gpu-memory-utilization" not in flags


def test_fold_respects_explicit_caller_flag(tmp_path: Path) -> None:
    """A knob the caller already set in accepted_flags is never overridden."""
    h = _fidelity_handoff(
        tmp_path,
        framework="vllm",
        accepted_flags="--max-model-len 8192",
        max_model_len=2248,
        mem_fraction=0.9,
    )
    flags = rx.map_args(h)["initial_extra_server_args"]
    # Caller's explicit value wins; no duplicate max-model-len appended.
    assert flags.count("--max-model-len") == 1
    assert "--max-model-len 8192" in flags
    assert "--max-model-len 2248" not in flags
    # mem_fraction (not set by the caller) is still folded in.
    assert "--gpu-memory-utilization 0.9" in flags


def test_fold_noop_when_knobs_absent(tmp_path: Path) -> None:
    """No fidelity knobs => initial_extra_server_args is byte-identical to seed."""
    h = _fidelity_handoff(
        tmp_path,
        framework="vllm",
        accepted_flags="--max-num-batched-tokens 24576",
    )
    assert rx.map_args(h)["initial_extra_server_args"] == "--max-num-batched-tokens 24576"


def test_fold_unknown_backend_left_untouched(tmp_path: Path) -> None:
    """An unmapped backend never gets a guessed flag name (seed unchanged)."""
    h = _fidelity_handoff(
        tmp_path,
        framework="trtllm",
        accepted_flags="--foo bar",
        max_model_len=2248,
        mem_fraction=0.9,
    )
    assert rx.map_args(h)["initial_extra_server_args"] == "--foo bar"


def test_fold_helper_dedup_and_forms() -> None:
    """Direct helper coverage: --flag=value and --flag value both dedup."""
    # --flag=value form is detected.
    out = rx._fold_serving_fidelity_flags(
        "--max-model-len=8192", backend="vllm", max_model_len=2248, mem_fraction=0.0
    )
    assert out.count("--max-model-len") == 1
    assert "2248" not in out
    # Unknown backend returns input verbatim.
    assert rx._fold_serving_fidelity_flags(
        "--x 1", backend="mystack", max_model_len=10, mem_fraction=0.5
    ) == "--x 1"
    # Empty seed + both knobs => clean space-joined string, no leading space.
    out2 = rx._fold_serving_fidelity_flags(
        "", backend="sglang", max_model_len=4096, mem_fraction=0.9
    )
    assert out2 == "--context-length 4096 --mem-fraction-static 0.9"


def test_cold_speedup_equals_hyperloom_provisional_ratio() -> None:
    """cold_speedup (what Hyperloom promotes as provisional) == final / orch cold.

    Ground-truth cross-check against the real session artifact: the provisional
    gain Hyperloom records must be exactly current_best.tput / baseline_tput,
    i.e. GEAK cold final over the orchestrator COLD baseline — never the hot
    final over the cold baseline (which would overstate the win).
    """
    fixture = Path(
        "test_results/gemma-4-26B_session/gemma-4-26B-A4B-it"
        "/result.json"
    )
    if not fixture.exists():
        pytest.skip("session fixture not present")
    r = json.loads(fixture.read_text(encoding="utf-8"))
    am = r["alignment_metrics"]

    geak_cold_final = am["geak_cold_final_tok_s"]
    orch_cold = am["orchestrator_cold_baseline_tok_s"]
    promoted_final = r["final_throughput_tok_s"]

    # The promoted final IS the cold final (final_throughput_basis == "cold").
    assert r["final_throughput_basis"] == "cold"
    assert promoted_final == pytest.approx(geak_cold_final)

    # Hyperloom's provisional gain == cold_speedup == promoted_final / orch_cold.
    expected_provisional_pct = (promoted_final / orch_cold - 1.0) * 100.0
    cold_speedup_pct = (am["cold_speedup"] - 1.0) * 100.0
    assert cold_speedup_pct == pytest.approx(expected_provisional_pct, abs=0.05)

    # And it is STRICTLY below the discarded hot-final-over-cold-baseline ratio
    # (the inflated number the provisional must NOT use).
    hot_over_cold_pct = (am["geak_hot_final_tok_s"] / orch_cold - 1.0) * 100.0
    assert cold_speedup_pct < hot_over_cold_pct
