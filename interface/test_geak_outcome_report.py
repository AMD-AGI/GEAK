"""Unit tests for interface/geak_outcome_report.py.

The module reads a run directory and reports what the run measured. The
contract worth pinning is the honesty of the absent case: a missing artifact
must never render as a zero, because that turns "we did not measure it" into
"it contributed nothing" — the exact confusion the report exists to prevent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent / "geak_outcome_report.py"
_spec = importlib.util.spec_from_file_location("geak_outcome_report", _MODULE_PATH)
assert _spec and _spec.loader
gor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gor)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """A miniature run directory with every stage present."""
    root = tmp_path / "e2e_model_20260101_000000_1_2"
    _write(root / "baseline" / "bench_summary.json", {
        "throughput_tok_s_median": 1000.0,
        "measurement_mode": "isolated_server",
        "ttft_ms_median": 900.0,
        "tpot_ms_median": 20.0,
    })
    _write(root / "config" / "sweep_results.json", {
        "baseline_throughput_tok_s": 1000.0,
        "best_throughput_tok_s": 1500.0,
        "throughput_speedup_vs_baseline": 1.5,
        "trials": [{"a": 1}, {"b": 2}],
        "measurement": {"mode": "isolated_server"},
        "accepted_flags": "--attention-backend X",
        "accepted_env": "FOO=1",
        "baseline_gsm8k_exact_match": 0.885,
    })
    _write(root / "kernels" / "h0_task" / "opbench_result.json", {
        "winner_backend": "aiter",
        "baseline_backend": "aiter",
        "winner_ms": 0.1,
        "baseline_ms": 0.1,
        "isolated_speedup": 1.0,
        "pct_gpu_time": 23.13,
        "amdahl_ceiling_e2e_pct": 0.0,
        "winner_editable": False,
        "measured": True,
    })
    _write(root / "tuning" / "tuning_result.json", {
        "pre_tune_throughput_tok_s": 1500.0,
        "post_tune_throughput_tok_s": 3000.0,
        "tuning_speedup": 2.0,
        "tuning_delta_pct": 100.0,
        "noise_floor_pct": 4.0,
        "gate": "accepted",
        "correctness_gate": "pass",
        "ab_interleaved": True,
        "ab_complete": True,
        "engagement_verified": True,
        "ops_tuned": [{"op": "gemm"}, "not-a-dict"],
    })
    return root


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_num_rejects_bools_and_junk() -> None:
    assert gor._num(True) is None
    assert gor._num(None) is None
    assert gor._num([1]) is None
    assert gor._num("nope") is None
    assert gor._num("1.5") == 1.5
    assert gor._num(float("inf")) is None
    assert gor._num(float("nan")) is None


def test_pct_and_speedup_guard_zero() -> None:
    assert gor._pct(0.0, 5.0) is None
    assert gor._pct(None, 5.0) is None
    assert gor._speedup(0.0, 5.0) is None
    assert gor._speedup(2.0, 3.0) == 1.5
    assert gor._pct(2.0, 3.0) == 50.0


def test_load_returns_none_for_unreadable(tmp_path: Path) -> None:
    assert gor._load(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert gor._load(bad) is None


def test_fmt_renders_absent_as_dash() -> None:
    assert gor._fmt(None) == "—"
    assert gor._fmt(1234.5) == "1,234.50"


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def test_baseline_stage(run_dir: Path) -> None:
    stage = gor.baseline_stage(run_dir)
    assert stage["present"] is True
    assert stage["after_tok_s"] == 1000.0
    assert stage["kind"] == "reference"


def test_sweep_stage(run_dir: Path) -> None:
    stage = gor.sweep_stage(run_dir)
    assert stage["present"] is True
    assert stage["speedup"] == 1.5
    assert stage["delta_pct"] == 50.0
    assert stage["trials"] == 2
    assert stage["accepted_flags"] == "--attention-backend X"


def test_tuning_stage(run_dir: Path) -> None:
    stage = gor.tuning_stage(run_dir)
    assert stage["present"] is True
    assert stage["speedup"] == 2.0
    assert stage["gate"] == "accepted"
    assert stage["measurement_mode"] == "isolated_server_ab"
    assert stage["ops_tuned"] == ["gemm"]


def test_headkernel_stages(run_dir: Path) -> None:
    stages = gor.headkernel_stages(run_dir)
    assert [s["task"] for s in stages] == ["h0_task"]
    assert stages[0]["amdahl_ceiling_e2e_pct"] == 0.0
    assert stages[0]["isolated_speedup"] == 1.0


def test_headkernel_skips_underscore_dirs_and_missing_result(run_dir: Path) -> None:
    (run_dir / "kernels" / "_exp").mkdir()
    (run_dir / "kernels" / "k9_task").mkdir()
    stages = gor.headkernel_stages(run_dir)
    names = {s["task"] for s in stages}
    assert "_exp" not in names
    absent = next(s for s in stages if s["task"] == "k9_task")
    assert absent["present"] is False
    assert absent["kind"] == "kernel"


def test_headkernel_missing_dir_is_empty(tmp_path: Path) -> None:
    assert gor.headkernel_stages(tmp_path / "nothing") == []


def test_absent_artifacts_report_absent_not_zero(tmp_path: Path) -> None:
    empty = tmp_path / "empty_run"
    empty.mkdir()
    record = gor.collect(empty)
    for stage in record["stages"]:
        assert stage["present"] is False
        assert "after_tok_s" not in stage or stage["after_tok_s"] is None
    assert record["summary"]["stages_measured"] == 0
    assert record["summary"]["compounded_speedup"] is None


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def test_summary_compounds_and_reports_no_seam(run_dir: Path) -> None:
    record = gor.collect(run_dir)
    summary = record["summary"]
    assert summary["stages_measured"] == 2
    assert summary["compounded_speedup"] == 3.0
    assert summary["observed_speedup_first_to_last"] == 3.0
    assert summary["compounded_is_estimate"] is False
    assert summary["handoff_seams"] == []


def test_summary_flags_a_handoff_seam(run_dir: Path) -> None:
    payload = json.loads((run_dir / "tuning" / "tuning_result.json").read_text(encoding="utf-8"))
    payload["pre_tune_throughput_tok_s"] = 1400.0  # sweep handed off 1500
    _write(run_dir / "tuning" / "tuning_result.json", payload)
    summary = gor.collect(run_dir)["summary"]
    assert summary["compounded_is_estimate"] is True
    seam = summary["handoff_seams"][0]
    assert seam["from_phase"] == gor.PHASE_SWEEP
    assert seam["to_phase"] == gor.PHASE_TUNING
    assert seam["gap_pct"] == pytest.approx(-6.6667, abs=1e-3)


# --------------------------------------------------------------------------- #
# spend
# --------------------------------------------------------------------------- #
def test_spend_by_phase_absent_is_none(tmp_path: Path) -> None:
    assert gor.spend_by_phase(tmp_path) is None


def test_spend_by_phase_aggregates_and_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "geak_calls.jsonl"
    path.write_text(
        json.dumps({"phase": "P1", "isl": 10, "osl": 2, "usd": 1.5, "tools": ["Read"]}) + "\n"
        + "\n"
        + "{not json\n"
        + json.dumps({"phase": "P1", "isl": 5, "osl": 1, "usd": 0.5, "tools": []}) + "\n"
        + json.dumps({"isl": 1, "osl": 1, "usd": 0.25, "tools": None}) + "\n",
        encoding="utf-8",
    )
    spend = gor.spend_by_phase(tmp_path)
    assert spend["P1"] == {"calls": 2, "isl": 15, "osl": 3, "usd": 2.0, "tool_calls": 1}
    assert spend["unknown"]["calls"] == 1


def test_spend_by_phase_empty_file_is_none(tmp_path: Path) -> None:
    (tmp_path / "geak_calls.jsonl").write_text("", encoding="utf-8")
    assert gor.spend_by_phase(tmp_path) is None


def test_spend_by_phase_unreadable_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "geak_calls.jsonl").write_text("{}\n", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("nope")

    monkeypatch.setattr(Path, "open", boom)
    assert gor.spend_by_phase(tmp_path) is None


# --------------------------------------------------------------------------- #
# rendering and writing
# --------------------------------------------------------------------------- #
def test_markdown_has_every_section(run_dir: Path) -> None:
    reports = run_dir / "reports"
    reports.mkdir()
    (reports / "geak_calls.jsonl").write_text(
        json.dumps({"phase": "P4 ConfigSweep", "isl": 9, "osl": 3, "usd": 2.25, "tools": ["Bash"]}) + "\n",
        encoding="utf-8",
    )
    text = gor.render_markdown(gor.collect(run_dir))
    assert "## Throughput" in text
    assert "## HeadKernel tasks" in text
    assert "## LLM spend by phase" in text
    assert "$2.25" in text
    assert "+50.00%" in text
    assert "1.0000x" in text


def test_markdown_says_spend_unknown_without_the_jsonl(run_dir: Path) -> None:
    text = gor.render_markdown(gor.collect(run_dir))
    assert "spend is unknown" in text


def test_markdown_renders_absent_rows(tmp_path: Path) -> None:
    empty = tmp_path / "empty_run"
    empty.mkdir()
    text = gor.render_markdown(gor.collect(empty))
    assert "*artifact absent*" in text


def test_markdown_reports_seams(run_dir: Path) -> None:
    payload = json.loads((run_dir / "tuning" / "tuning_result.json").read_text(encoding="utf-8"))
    payload["pre_tune_throughput_tok_s"] = 1400.0
    _write(run_dir / "tuning" / "tuning_result.json", payload)
    text = gor.render_markdown(gor.collect(run_dir))
    assert "seam" in text and "an estimate" in text


def test_markdown_renders_absent_kernel_row(run_dir: Path) -> None:
    (run_dir / "kernels" / "k9_task").mkdir()
    text = gor.render_markdown(gor.collect(run_dir))
    assert "*no opbench_result.json*" in text


def test_write_persists_both_files(run_dir: Path) -> None:
    result = gor.write(run_dir)
    assert result["status"] == "ok"
    assert result["stages_measured"] == 2
    reports = run_dir / "reports"
    record = json.loads((reports / gor.OUTCOME_JSON).read_text(encoding="utf-8"))
    assert record["run_id"] == run_dir.name
    assert (reports / gor.OUTCOME_MD).read_text(encoding="utf-8").startswith("# GEAK outcome")
    assert not list(reports.glob("*.tmp"))


def test_write_honours_an_explicit_reports_dir(run_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere"
    result = gor.write(run_dir, reports_dir=out)
    assert result["status"] == "ok"
    assert (out / gor.OUTCOME_JSON).is_file()


def test_write_reports_an_oserror(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(Path, "mkdir", boom)
    result = gor.write(run_dir)
    assert result["status"] == "error"
    assert "OSError" in result["error"]


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def test_main_writes_and_returns_zero(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gor.main([str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_main_stdout_mode_writes_nothing(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gor.main([str(run_dir), "--stdout"]) == 0
    assert "# GEAK outcome" in capsys.readouterr().out
    assert not (run_dir / "reports").exists()


def test_main_reports_failure(run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(gor, "write", lambda *_a, **_k: {"status": "error", "error": "x"})
    assert gor.main([str(run_dir)]) == 1
    capsys.readouterr()


def test_main_accepts_reports_dir_override(run_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "od"
    assert gor.main([str(run_dir), "--reports-dir", str(out)]) == 0
    capsys.readouterr()
    assert (out / gor.OUTCOME_MD).is_file()
