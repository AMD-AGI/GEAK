"""Operator timing reaches the original workflow without changing its measurements."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_e2e_timing", Path(__file__).with_name("run_e2e.py"))
rx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rx)

TIMING = {
    "GEAK_AGENT_TIMEOUT_MS": ("agent_timeout_ms", 14400000),
    "GEAK_TIME_TAIL_CAP_S": ("time_tail_cap_s", 3600),
    "GEAK_FINAL_RESERVE_S": ("final_reserve_s", 900),
}


def test_timing_overrides_reach_the_unchanged_workflow(monkeypatch):
    for name, (_, value) in TIMING.items():
        monkeypatch.setenv(name, str(value))
    args = rx.map_args({"model_path": "/models/x", "exp_root": "/tmp/exp"}, timeout_s=17820)
    assert args["time_budget_s"] == 17820
    for arg, value in TIMING.values():
        assert args[arg] == value
    assert args["measurement_mode"] == "warm_server"
    assert args["validation_measurement_mode"] == "warm_server"
    assert args["parity_replicas"] == 1
    assert "phases" not in args


@pytest.mark.parametrize("value", [None, "", "0", "-1", "bad"])
def test_absent_or_invalid_timing_keeps_workflow_defaults(monkeypatch, value):
    for name in TIMING:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    args = rx.map_args({"model_path": "/models/x", "exp_root": "/tmp/exp"})
    for arg, _ in TIMING.values():
        assert arg not in args
