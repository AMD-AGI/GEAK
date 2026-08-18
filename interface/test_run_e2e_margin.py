#!/usr/bin/env python3
"""Single-source contract for the implausible-speedup margin.

Run:  python3 -m pytest interface/test_run_e2e_margin.py -q

The JS live-path guard (``A.implausible_speedup_margin`` in e2e_workflow.js) and
THIS runner's recovery path must apply the SAME margin, or a speedup the live
path accepts could be re-flagged (or vice versa) on recovery. run_e2e.py makes
the Python constant the single source of truth: it validates the env-overridable
value (finite, non-negative -- an accepted nan/inf/negative would silently
disable the guard) and ``map_args`` forwards THAT value to the workflow
unconditionally. These tests pin both halves so a refactor cannot reintroduce
Python/JS drift or let a bad env value through.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load_with_env(monkeypatch, value):
    """Import run_e2e fresh with GEAK_IMPLAUSIBLE_SPEEDUP_MARGIN=value (or unset
    when value is None), so the import-time validation runs under that env."""
    if value is None:
        monkeypatch.delenv("GEAK_IMPLAUSIBLE_SPEEDUP_MARGIN", raising=False)
    else:
        monkeypatch.setenv("GEAK_IMPLAUSIBLE_SPEEDUP_MARGIN", value)
    spec = importlib.util.spec_from_file_location("run_e2e_margin_probe", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_margin_is_one(monkeypatch) -> None:
    rx = _load_with_env(monkeypatch, None)
    assert rx.IMPLAUSIBLE_SPEEDUP_MARGIN == 1.0


def test_valid_override_is_honoured_and_forwarded(monkeypatch) -> None:
    rx = _load_with_env(monkeypatch, "2.5")
    assert rx.IMPLAUSIBLE_SPEEDUP_MARGIN == 2.5
    # map_args forwards the SAME value to the JS workflow, unconditionally.
    ps = rx.map_args({"model_path": "/models/x", "exp_root": "/tmp/exp"})
    assert ps["implausible_speedup_margin"] == 2.5


def test_zero_is_a_valid_margin(monkeypatch) -> None:
    # 0.0 is a legitimate (strictest) margin, not the invalid-value fallback.
    rx = _load_with_env(monkeypatch, "0")
    assert rx.IMPLAUSIBLE_SPEEDUP_MARGIN == 0.0


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "-1", "-0.5", "abc", ""])
def test_non_finite_negative_or_garbage_falls_back_to_one(monkeypatch, bad) -> None:
    """A value that would silently disable the guard must fall back to 1.0, and
    the forwarded value must be that safe fallback (never nan/inf/negative)."""
    rx = _load_with_env(monkeypatch, bad)
    assert rx.IMPLAUSIBLE_SPEEDUP_MARGIN == 1.0
    assert math.isfinite(rx.IMPLAUSIBLE_SPEEDUP_MARGIN)
    ps = rx.map_args({"model_path": "/models/x", "exp_root": "/tmp/exp"})
    assert ps["implausible_speedup_margin"] == 1.0
