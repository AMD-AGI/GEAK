#!/usr/bin/env python3
"""Single-source contract for the implausible-speedup margin.

Run:  python3 -m pytest interface/test_run_e2e_margin.py -q

The JS live path and this runner's recovery path use the same advisory threshold,
retaining the legacy ``A.implausible_speedup_margin`` configuration name. It no
longer changes acceptance. run_e2e.py validates the env-overridable value (finite,
non-negative), and ``map_args`` forwards that value to the workflow unconditionally.
These tests preserve the configuration contract without treating the model as a
correctness gate.
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
    monkeypatch.setenv(
        "GEAK_GPU_IDENTITY_JSON",
        '{"gfx":"gfx950","target":"unknown","marketing_name":"AMD Instinct MI355X",'
        '"physical_cu_count":256}',
    )
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
    """Keep the existing fallback and forwarded value (never nan/inf/negative)."""
    rx = _load_with_env(monkeypatch, bad)
    assert rx.IMPLAUSIBLE_SPEEDUP_MARGIN == 1.0
    assert math.isfinite(rx.IMPLAUSIBLE_SPEEDUP_MARGIN)
    ps = rx.map_args({"model_path": "/models/x", "exp_root": "/tmp/exp"})
    assert ps["implausible_speedup_margin"] == 1.0
