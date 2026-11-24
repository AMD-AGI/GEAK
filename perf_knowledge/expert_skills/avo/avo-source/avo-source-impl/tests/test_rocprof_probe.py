"""Unit tests for scripts/rocprof_probe.py (deterministic counter parse half, A2).

Run:  python -m pytest avo-workflow/tests/test_rocprof_probe.py -q
No GPU needed — pins the rocprof CSV parser + counter-vector derivation + graceful
ok=false. The rig-only `collect` is not exercised.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rocprof_probe.py"
_spec = importlib.util.spec_from_file_location("rocprof_probe", _SCRIPT)
rocprof_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rocprof_probe)

_FIX = Path(__file__).resolve().parent / "fixtures"


def test_parse_counters_csv_averages_rows():
    raw = rocprof_probe.parse_counters_csv((_FIX / "sample_rocprof.csv").read_text())
    assert raw["fetch_size"] == 1024     # averaged across the two identical rows
    assert raw["occupancy"] == pytest.approx(0.61)


def test_counters_to_vector_kb_units_and_pct():
    raw = rocprof_probe.parse_counters_csv((_FIX / "sample_rocprof.csv").read_text())
    v = rocprof_probe.counters_to_vector(raw, size_unit="kb")
    # (FETCH 1024 + WRITE 512) KB * 1024 = 1,572,864 bytes
    assert v["hbm_bytes"] == (1024 + 512) * 1024
    assert v["occupancy_pct"] == pytest.approx(61.0)   # 0.61 fraction -> 61%
    assert v["mfma_active_pct"] == pytest.approx(70.0)  # mean(0.71,0.69)=0.70 -> 70%


def test_counters_to_vector_bytes_units():
    v = rocprof_probe.counters_to_vector({"fetch_size": 2000, "write_size": 1000}, size_unit="bytes")
    assert v["hbm_bytes"] == 3000


def test_direct_hbm_bytes_alias_wins():
    v = rocprof_probe.counters_to_vector({"hbm_bytes": 4096, "fetch_size": 1})
    assert v["hbm_bytes"] == 4096


def test_probe_text_ok_vector():
    vec = rocprof_probe.probe_text((_FIX / "sample_rocprof.csv").read_text())
    assert vec["ok"] is True and vec["source"] == "rocprofv3"
    assert vec["counters"]["hbm_bytes"] == (1024 + 512) * 1024


def test_probe_text_empty_is_inert():
    assert rocprof_probe.probe_text("")["ok"] is False
    assert rocprof_probe.probe_text("no,csv,header\n")["ok"] is False  # no numeric -> inert


def test_cli_parse(capsys):
    rc = rocprof_probe.main(["parse", "--counters-file", str(_FIX / "sample_rocprof.csv")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["counters"]["occupancy_pct"] == pytest.approx(61.0)


def test_cli_parse_missing_file_inert(capsys):
    rc = rocprof_probe.main(["parse", "--counters-file", "/no/such.csv"])
    assert rc == 0 and json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_collect_is_rig_only_inert(capsys):
    rc = rocprof_probe.main(["collect", "--workspace", "."])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and "rig-only" in out["notes"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
