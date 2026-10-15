"""Tests for the normalized whole-implementation measurement registry."""

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "merge_benchmark_results", HERE / "benchmarks" / "merge_results.py",
)
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


def result(implementation_id="impl_a", speedup=1.2, run_id="run-a", baseline="triton"):
    return {
        "schema_version": 1,
        "run_id": run_id,
        "seed": 1907,
        "suite": "suite",
        "math_contract": "y=x",
        "shape": {"m": 8, "n": 16, "k": 32},
        "dtype": "fp8",
        "baseline": baseline,
        "accounting": {"timer": "events"},
        "environment": {"gfx": "gfx942", "flydsl_version": "0.3.0"},
        "implementations": [{
            "implementation_id": implementation_id,
            "name": "flydsl",
            "language": "flydsl",
            "status": "measured",
            "speedup_vs_baseline": speedup,
            "config": {
                "identity_complete": True,
                "tile_m": 16,
                "tile_n": 32,
                "tile_k": 64,
                "split_k": 2,
            },
        }],
    }


def write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_merge_flattens_and_deduplicates_identical_measurements(tmp_path):
    first = write(tmp_path / "a.json", result())
    second = write(tmp_path / "b.json", result())
    got = MERGE.merge([first, second])
    assert got["errors"] == []
    assert got["records_count"] == 1
    row = got["records"][0]
    assert row["measurement_id"].startswith("measure_")
    assert row["context"]["shape"]["m"] == 8
    assert row["implementation"]["speedup_vs_baseline"] == 1.2
    decisions = row["implementation"]["performance_decisions"]
    assert decisions["axes"]["workgroup_tile"]["values"] == {
        "k": 64, "m": 16, "n": 32,
    }
    assert decisions["axes"]["work_partition"]["values"]["split_k"] == 2


def test_environment_or_implementation_change_produces_a_new_record(tmp_path):
    one = result()
    two = result(implementation_id="impl_b")
    first = write(tmp_path / "a.json", one)
    second = write(tmp_path / "b.json", two)
    assert MERGE.merge([first, second])["records_count"] == 2


def test_physical_repeats_are_preserved_and_share_a_context(tmp_path):
    first = write(tmp_path / "a.json", result(run_id="run-a", speedup=1.1))
    second = write(tmp_path / "b.json", result(run_id="run-b", speedup=1.2))
    got = MERGE.merge([first, second])
    assert got["records_count"] == 2
    assert len({row["measurement_id"] for row in got["records"]}) == 2
    assert len({row["context_fingerprint"] for row in got["records"]}) == 1


def test_baseline_is_part_of_comparable_context(tmp_path):
    first = write(tmp_path / "a.json", result(run_id="run-a", baseline="triton"))
    second = write(tmp_path / "b.json", result(run_id="run-b", baseline="ck"))
    got = MERGE.merge([first, second])
    assert len({row["context_fingerprint"] for row in got["records"]}) == 2


def test_bad_input_is_reported_without_dropping_good_records(tmp_path):
    good = write(tmp_path / "good.json", result())
    bad = write(tmp_path / "bad.json", {"schema_version": 9})
    got = MERGE.merge([bad, good])
    assert got["records_count"] == 1
    assert len(got["errors"]) == 1
    assert got["contract"].startswith("Whole-implementation")
