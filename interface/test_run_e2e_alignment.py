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
  * ``cold_speedup`` = GEAK cold final / the orchestrator anchor Hyperloom
    promotes, so it must equal current_best.tput / baseline_tput. Note the
    anchor is normally a HOT measure round (Hyperloom discards its warmup
    round); ``orchestrator_baseline_lifecycle`` says which, and
    ``hot_speedup`` is the hot-to-hot pairing.

Run: python3 -m pytest GEAK/interface/test_run_e2e_alignment.py -v
"""
from __future__ import annotations

import importlib.util
import json
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


def test_same_config_divergence_above_threshold_is_warning(
    tmp_path: Path, monkeypatch
) -> None:
    """A real same-config mismatch warns without changing optimization status.

    Both harnesses launched through the same script here, so the divergence is
    a measurement signal and gets the plain `warning`. See
    test_run_e2e_measurement_basis.py for the unaligned-recipe counterpart.
    """
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    monkeypatch.setenv("BENCH_LAUNCHER", "magpie")
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


def test_handoff_alignment_uses_setup_not_validate_base_and_verifies_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The late optimization denominator cannot overwrite the handoff verdict."""
    eval_dir = tmp_path / "e2e"
    (eval_dir / "validation" / "base").mkdir(parents=True)
    (eval_dir / "validation" / "final").mkdir(parents=True)
    (eval_dir / "baseline").mkdir()
    (eval_dir / "baseline" / "bench_summary.json").write_text(
        json.dumps({"throughput_tok_s_median": 1000.0}), encoding="utf-8"
    )
    (eval_dir / "validation" / "base" / "bench_summary.json").write_text(
        json.dumps({"throughput_tok_s_median": 900.0}), encoding="utf-8"
    )
    (eval_dir / "validation" / "final" / "bench_summary.json").write_text(
        json.dumps({"throughput_tok_s_median": 990.0}), encoding="utf-8"
    )
    server_log = eval_dir / "baseline" / "server.log"
    server_log.write_text(
        "server_args=ServerArgs(model_path='/models/qwen', tp_size=1, "
        "mem_fraction_static=0.92, context_length=8192)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BENCH_LAUNCHER", "magpie")
    handoff = {
        "orchestrator_best_tput_same_config": 1000.0,
        "same_config_observed_identity": {
            "backend": "sglang",
            "server_args": {
                "model_path": "/models/qwen",
                "tp_size": 1,
                "mem_fraction_static": 0.92,
            },
        },
    }

    out = rx.normalize_result(
        handoff, _wf(eval_dir, base=1000.0, final=990.0, speedup=1.1)
    )

    assert out["baseline_throughput_tok_s"] == 900.0
    assert out["throughput_speedup"] == pytest.approx(1.1)
    assert out["handoff_alignment"]["status"] == "aligned"
    assert out["handoff_alignment"]["divergence_pct"] == 0.0
    assert out["server_identity"]["status"] == "matched"
    assert out["server_identity"]["observed"]["server_args"]["context_length"] == 8192
    assert out["server_identity"]["evidence_paths"] == [str(server_log)]
    # Validation's -10% movement is a separate diagnostic, not handoff evidence.
    assert out["measurement_drift"]["status"] == "measured"
    assert out["measurement_drift"]["drift_pct"] == -10.0
    assert out["baseline_alignment"]["divergence_pct"] == 0.0


def test_handoff_identity_mismatch_is_exposed_with_its_evidence(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "e2e"
    (eval_dir / "baseline").mkdir(parents=True)
    (eval_dir / "baseline" / "bench_summary.json").write_text(
        json.dumps({"throughput_tok_s_median": 1000.0}), encoding="utf-8"
    )
    server_log = eval_dir / "baseline" / "server.log"
    server_log.write_text(
        "server_args=ServerArgs(model_path='/models/qwen', tp_size=1)\n",
        encoding="utf-8",
    )

    out = rx.normalize_result(
        {
            "orchestrator_best_tput_same_config": 1000.0,
            "observed_server_identity": {
                "backend": "sglang",
                "server_args": {"model_path": "/models/qwen", "tp_size": 2},
            },
        },
        _wf(eval_dir, base=1000.0, final=1000.0, speedup=1.0),
    )

    assert out["server_identity"]["status"] == "mismatched"
    assert out["handoff_alignment"]["status"] == "identity_mismatch"
    assert out["server_identity"]["evidence_paths"] == [str(server_log)]


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


def test_map_args_consumes_schema_v2_effective_config(tmp_path: Path) -> None:
    """The complete current-best descriptor, not only accepted_flags, seeds GEAK."""
    recipe = tmp_path / "baseline_config.with_envs.yaml"
    recipe.write_text(
        "benchmark:\n"
        "  envs:\n"
        "    EXTRA_SGLANG_ARGS: --trust-remote-code --context-length 8192\n"
        "    SGLANG_USE_AITER: '0'\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "base-overlay"
    snapshot = tmp_path / "snapshot"
    h = {
        "schema_version": 2,
        "model_path": "/models/gemma",
        "framework": "sglang",
        "exp_root": str(tmp_path),
        "eval_dir": str(tmp_path / "e2e"),
        "launch_recipe": str(recipe),
        "workload": {"isl": 8192, "osl": 1024, "conc": 64},
        "max_model_len": 13312,  # stale summary must not replace complete argv
        "accepted_flags": "--context-length=11264",
        "accepted_env": "SGLANG_USE_AITER=1",
        "baseline_env_spec": {
            "config": {
                "server_launch_flags": (
                    "--trust-remote-code --disable-radix-cache "
                    "--context-length 11264"
                ),
                "extra_server_args": "--context-length 11264",
                "extra_envs": {"SGLANG_USE_AITER": "1"},
            },
            "overlay_pythonpath": str(overlay),
            "source_snapshots": [
                {
                    "snapshot_dir": str(snapshot),
                    "reproducible": True,
                }
            ],
        },
    }

    ps = rx.map_args(h)
    flags = ps["initial_extra_server_args"]
    assert flags.count("--context-length") == 1
    assert "--context-length 11264" in flags
    assert "13312" not in flags
    assert "--disable-radix-cache" in flags
    assert "SGLANG_USE_AITER=1" in ps["initial_extra_env"]
    assert ps["initial_overlay_pythonpath"] == f"{overlay}:{snapshot}"
    assert len(ps["effective_config_digest"]) == 64
    # ONE lifecycle for the whole run, and it is Hyperloom's: 1 boot per leg, a discarded
    # full warmup round, then the timed round -- exactly two client passes with the second
    # one reported, which is what warmup_round/measure_round does.  The round count is not
    # a separate knob; it is what warm_server means.
    assert ps["measurement_mode"] == "warm_server"
    assert ps["validation_measurement_mode"] == "warm_server"
    assert "validation_rounds" not in ps
    # Only consulted if a caller pins validation back to isolated_server.
    assert ps["validation_replicas"] == 3


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


def test_promoted_final_is_hot_and_cold_stays_a_diagnostic(tmp_path: Path) -> None:
    """The promoted headline is ALWAYS the HOT median; cold is diagnostic only.

    The headline contract is fixed: ``final_throughput_basis == "hot"`` and the
    promoted final is the hot median (run_e2e.py sets ``final_basis = "hot"``
    unconditionally; BENCH_COLD_FINAL only adds a diagnostic cold round in
    alignment_metrics and never switches the headline).

    ``cold_speedup`` remains a SELF-CONSISTENT diagnostic — GEAK's cold final over
    the orchestrator's COLD baseline (never the hot final over the cold baseline,
    which would overstate the win) — even though it no longer drives the headline.

    This used to cross-check a real session artifact and was hidden behind a skip
    when that artifact was absent, so the cold-diagnostic ratio was never
    continuously exercised. It now builds the cold round synthetically (same shape
    as TestColdFinalBasis in test_run_e2e_dispatch.py) so it always runs.
    """
    # hot base/final drive the headline; cold rounds are strictly slower (a cold
    # round pays a cache-fill / JIT cost the hot median never does).
    eval_dir = tmp_path / "e2e_cold_diag"
    (eval_dir / "baseline").mkdir(parents=True)
    (eval_dir / "validation" / "final").mkdir(parents=True)
    (eval_dir / "baseline" / "bench_summary.json").write_text(
        json.dumps({"output_throughput_tok_s_median": 450.0,
                    "cold_output_throughput_tok_s": 460.0}),
        encoding="utf-8",
    )
    (eval_dir / "validation" / "final" / "bench_summary.json").write_text(
        json.dumps({"output_throughput_tok_s_median": 500.0,
                    "cold_output_throughput_tok_s": 480.0}),
        encoding="utf-8",
    )
    # raw_baseline_tput is the orchestrator's COLD leaderboard anchor.
    h = {"workload": {"isl": 1024, "osl": 1024, "conc": 64},
         "raw_baseline_tput": 440.0}
    r = rx.normalize_result(h, _wf(eval_dir, base=450.0, final=500.0, speedup=1.1111))
    am = r["alignment_metrics"]

    # Headline: hot basis, hot median promoted.
    assert r["final_throughput_basis"] == "hot"
    assert r["final_throughput_tok_s"] == 500.0
    assert r["final_throughput_tok_s"] == pytest.approx(am["geak_hot_final_tok_s"])

    # Cold diagnostic stays internally consistent: cold final over orch cold base,
    # and strictly below the inflated hot-final-over-cold-baseline ratio.
    geak_cold_final = am["geak_cold_final_tok_s"]
    orch_cold = am["orchestrator_cold_baseline_tok_s"]
    assert geak_cold_final == 480.0 and orch_cold == 440.0
    # cold_speedup is reported rounded (_safe_ratio -> 4 dp), so allow that.
    assert am["cold_speedup"] == pytest.approx(geak_cold_final / orch_cold, abs=1e-4)
    hot_over_cold = am["geak_hot_final_tok_s"] / orch_cold
    assert am["cold_speedup"] < hot_over_cold


# --- Regression: an unverified upstream same-config reference -----------------
#
# Anchored on the real Hyperloom run MiniMax-M3-MXFP4/20260904T002558Z-3de91cb3,
# which handed GEAK `orchestrator_best_tput_same_config: 0.0` with
# `same_config_reference_status: "unverified"` and zero accepted kernels. The
# session still reported hot_speedup ~1.19 -- entirely Hyperloom's OWN explore
# gain, because hot_speedup's denominator is the RAW session baseline
# (3693.71 tok/s hot measure round) and its numerator is the post-explore
# accepted config (measured on-hardware at 4379.15 tok/s under warm_server).


def _minimax_handoff(
    *,
    status: str | None = "unverified",
    exp_root: Path | None = None,
) -> dict:
    h = {
        "workload": {"isl": 8192, "osl": 1024, "conc": 64},
        "raw_baseline_tput": 3693.7114118953027,
        "orchestrator_best_tput_same_config": 0.0,
    }
    if status is not None:
        h["same_config_reference_status"] = status
    if exp_root is not None:
        # Hyperloom's double-run baseline: a full warmup round it discarded,
        # then the hot measure round that became state.baseline_tput. Without
        # this verdict the hot-to-hot pairings correctly degrade to None.
        (exp_root / "state.json").write_text(
            json.dumps({
                "baseline_tput": 3693.7114118953027,
                "baseline_warm_runtime_sec": 53.23,
                "baseline_measure_round_dropped": False,
            }),
            encoding="utf-8",
        )
        h["exp_root"] = str(exp_root)
    return h


def test_unverified_same_config_reference_is_named_as_upstream_gap(
    tmp_path: Path,
) -> None:
    """A 0.0 reference the orchestrator called unverified is not GEAK's failure."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    wf = _wf(eval_dir, base=4379.154, final=4379.154, speedup=1.0)

    out = rx.normalize_result(_minimax_handoff(), wf)
    al = out["baseline_alignment"]

    assert al["status"] == "unavailable_reference_unverified"
    assert al["same_config_reference_status"] == "unverified"
    # The reference really is absent -- the new status must not invent one.
    assert out["baseline_basis"]["current_best_same_config_divergence_pct"] is None
    json.dumps(out, allow_nan=False)


def test_verified_and_missing_status_keep_the_plain_unavailable_status(
    tmp_path: Path,
) -> None:
    """Only a non-"verified" upstream verdict earns the sharper status."""
    cases = (
        ("verified", "unavailable"),
        ("VERIFIED", "unavailable"),
        (None, "unavailable"),
        ("", "unavailable"),
        ("unverified", "unavailable_reference_unverified"),
        ("stale", "unavailable_reference_unverified"),
    )
    for index, (status, expected) in enumerate(cases):
        eval_dir = tmp_path / f"e2e_{index}"
        eval_dir.mkdir()
        out = rx.normalize_result(
            _minimax_handoff(status=status),
            _wf(eval_dir, base=4379.154, final=4379.154, speedup=1.0),
        )
        al = out["baseline_alignment"]
        assert al["status"] == expected, (status, al["status"])
        # Normalized to lowercase, None when the handoff said nothing at all.
        assert al["same_config_reference_status"] == (
            status.lower() if status else None
        )


def test_hot_speedup_declares_its_denominator_and_config_gain(
    tmp_path: Path,
) -> None:
    """hot_speedup must self-label as raw-baseline-relative, not GEAK-only."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    (eval_dir / "baseline").mkdir()
    (eval_dir / "validation" / "final").mkdir(parents=True)
    # GEAK's own warm_server measurement of the seeded (already-accepted) config:
    # the discarded cold outer round + the hot median, both real numbers.
    (eval_dir / "baseline" / "bench_summary.json").write_text(
        json.dumps({"output_throughput_tok_s_median": 4379.154,
                    "cold_output_throughput_tok_s": 2580.58}),
        encoding="utf-8",
    )
    (eval_dir / "validation" / "final" / "bench_summary.json").write_text(
        json.dumps({"output_throughput_tok_s_median": 4379.154,
                    "cold_output_throughput_tok_s": 2580.58}),
        encoding="utf-8",
    )
    wf = _wf(eval_dir, base=4379.154, final=4379.154, speedup=1.0)

    out = rx.normalize_result(_minimax_handoff(exp_root=tmp_path), wf)
    am = out["alignment_metrics"]

    # GEAK accepted nothing here, and hot_geak_speedup says so honestly...
    assert am["hot_geak_speedup"] == pytest.approx(1.0, abs=1e-4)
    # ...while hot_speedup is ~1.19 purely from Hyperloom's own config gain.
    assert am["hot_speedup"] == pytest.approx(4379.154 / 3693.7114118953027,
                                             abs=1e-3)
    assert am["hot_speedup"] > 1.15
    # Which is exactly why the block must say whose gain that is.
    assert am["hot_speedup_denominator"] == "raw_session_baseline"
    assert am["hot_speedup_includes_orchestrator_config_gain"] is True
    # The only same-config pairing -- absent, because upstream never verified it.
    assert am["hot_speedup_same_config"] is None
    json.dumps(out, allow_nan=False)


def test_unverified_reference_caveat_reaches_the_rendered_report(
    tmp_path: Path,
) -> None:
    """The status is useless if only the JSON carries the explanation."""
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    out = rx.normalize_result(
        _minimax_handoff(),
        _wf(eval_dir, base=4379.154, final=4379.154, speedup=1.0),
    )
    section = rx._render_baseline_alignment_section(out)

    assert "unavailable_reference_unverified" in section
    assert "upstream handoff gap" in section
    assert "hot_geak_speedup" in section
    # Idempotent, like the rest of the section.
    assert rx._render_baseline_alignment_section(out) == section


def test_verified_reference_report_has_no_unverified_caveat(
    tmp_path: Path,
) -> None:
    eval_dir = tmp_path / "e2e"
    eval_dir.mkdir()
    out = rx.normalize_result(
        _minimax_handoff(status="verified"),
        _wf(eval_dir, base=4379.154, final=4379.154, speedup=1.0),
    )
    section = rx._render_baseline_alignment_section(out)

    assert "upstream handoff gap" not in section
