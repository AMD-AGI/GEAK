"""Tests for exact-context measured implementation selection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "select_implementations", HERE / "benchmarks" / "select_implementations.py",
)
SELECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECT)


def record(name, language, speedup, gfx="gfx942:sramecc+", version="0.3.0", m=8):
    return {
        "measurement_id": f"measure_{name}",
        "context_fingerprint": "context_a",
        "context": {
            "suite": "fp8",
            "dtype": "fp8",
            "shape": {"m": m, "n": 256, "k": 128},
            "baseline": "triton",
            "environment": {
                "gfx": gfx, "flydsl_version": version, "aiter_dirty": False,
                "aiter_commit": "a" * 40, "torch": "2.9", "hip": "7.2",
            },
        },
        "implementation": {
            "implementation_id": f"impl_{name}",
            "name": name,
            "language": language,
            "status": "measured",
            "correctness": {"pass": True},
            "config": {"identity_complete": True},
            "speedup_vs_baseline": speedup,
        },
    }


def args(**overrides):
    base = {
        "suite": "fp8", "language": None, "gfx": "gfx942",
        "flydsl_version": "0.3.0", "dtype": "fp8",
        "m": 8, "n": 256, "k": 128, "top_k": 3,
        "baseline": "triton", "context_fingerprint": None,
        "math_contract": None, "aiter_commit": "a" * 40,
        "torch_version": "2.9", "hip_version": "7.2",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_exact_context_ranks_by_speedup_and_normalizes_gfx_suffix():
    registry = {"records": [
        record("slow", "flydsl", 1.1),
        record("fast", "flydsl", 1.4),
        record("wrong-shape", "flydsl", 9.0, m=16),
        record("wrong-gfx", "flydsl", 8.0, gfx="gfx950"),
    ]}
    got = SELECT.select(registry, args())
    assert got["matched"] == 2
    assert got["distinct_implementations"] == 2
    assert [row["implementation"]["name"] for row in got["candidates"]] == ["fast", "slow"]


def test_flydsl_version_filters_flydsl_but_not_other_language_baselines():
    registry = {"records": [
        record("old-flydsl", "flydsl", 2.0, version="0.2.0"),
        record("triton", "triton", 1.0, version="not_captured"),
    ]}
    got = SELECT.select(registry, args())
    assert [row["implementation"]["name"] for row in got["candidates"]] == ["triton"]


def test_failed_or_incorrect_implementation_never_becomes_a_candidate():
    bad = record("bad", "flydsl", 9.0)
    bad["implementation"]["correctness"] = {"pass": False}
    failed = record("failed", "flydsl", 8.0)
    failed["implementation"]["status"] = "unsupported_or_build_failed"
    assert SELECT.select({"records": [bad, failed]}, args())["candidates"] == []


def test_repeats_do_not_fill_top_k_and_use_median():
    first = record("same-a", "flydsl", 1.1)
    second = record("same-b", "flydsl", 1.3)
    second["implementation"]["implementation_id"] = first["implementation"]["implementation_id"]
    registry = {"records": [first, second, record("other", "flydsl", 1.15)]}
    got = SELECT.select(registry, args(top_k=3))
    assert got["matched"] == 3
    assert got["distinct_implementations"] == 2
    assert got["candidates"][0]["implementation"]["measured_speedup"]["median"] == pytest.approx(1.2)
    assert got["candidates"][0]["implementation"]["measured_speedup"]["replicates"] == 2


def test_multiple_comparison_contexts_require_explicit_fingerprint():
    first = record("one", "flydsl", 1.2)
    second = record("two", "flydsl", 9.0)
    second["context_fingerprint"] = "context_b"
    second["context"]["accounting"] = {"timer": "different"}
    got = SELECT.select({"records": [first, second]}, args(context_fingerprint=None))
    assert got["candidates"] == []
    assert got["ambiguous_contexts"] == ["context_a", "context_b"]


def test_nan_bool_dirty_or_incomplete_identity_is_rejected():
    import math

    nan = record("nan", "flydsl", math.nan)
    boolean = record("bool", "flydsl", True)
    dirty = record("dirty", "flydsl", 2.0)
    dirty["context"]["environment"]["aiter_dirty"] = True
    incomplete = record("incomplete", "flydsl", 2.0)
    incomplete["implementation"]["config"]["identity_complete"] = False
    assert SELECT.select(
        {"records": [nan, boolean, dirty, incomplete]}, args()
    )["candidates"] == []


def test_incomplete_query_context_is_safe_empty():
    got = SELECT.select(
        {"records": [record("one", "flydsl", 1.2)]},
        args(aiter_commit=None, torch_version=None, hip_version=None, baseline=None),
    )
    assert got["candidates"] == []
    assert set(got["missing_query_context"]) == {
        "baseline", "aiter_commit", "torch_version", "hip_version",
    }
