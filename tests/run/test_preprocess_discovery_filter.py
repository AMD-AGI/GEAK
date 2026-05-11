"""Tests for ``_filter_discovery_to_repo_root`` in ``run/preprocess/preprocessor.py``.

Covers the bug where ``automated_test_discovery`` returned harnesses
from sibling kernel directories (``.../L3/fused_rms_fp8/...`` while the
kernel under optimization was ``.../L3/gemm_a16wfp4/kernel.py``), which
then triggered "outside repo_root" warnings and ran the wrong workload.
"""

from __future__ import annotations

import logging
from pathlib import Path

from minisweagent.run.preprocess.preprocessor import _filter_discovery_to_repo_root


def _disc(tests=None, benchmarks=None) -> dict:
    return {
        "kernel": {"name": "k", "type": "triton"},
        "tests": list(tests or []),
        "benchmarks": list(benchmarks or []),
        "total_tests_found": len(tests or []),
        "total_benchmarks_found": len(benchmarks or []),
    }


def test_drops_tests_outside_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "L3" / "gemm_a16wfp4"
    repo.mkdir(parents=True)
    sibling = tmp_path / "L3" / "fused_rms_fp8"
    sibling.mkdir(parents=True)

    inside = repo / "test_kernel_harness.py"
    inside.write_text("# in-scope\n")
    outside = sibling / "test_kernel_harness.py"
    outside.write_text("# out-of-scope\n")

    disc = _disc(tests=[{"file": str(inside), "name": "in"}, {"file": str(outside), "name": "out"}])

    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, r: logging.LogRecord) -> None:
            records.append(r)

    h = _H(level=logging.WARNING)
    logger_obj = logging.getLogger("minisweagent.run.preprocess.preprocessor")
    logger_obj.addHandler(h)
    try:
        out = _filter_discovery_to_repo_root(disc, str(repo))
    finally:
        logger_obj.removeHandler(h)

    assert out is not disc, "filter should return a new dict when it dropped entries"
    files = [t["file"] for t in out["tests"]]
    assert files == [str(inside)]
    assert out["total_tests_found"] == 1
    msgs = [r.getMessage() for r in records if "outside repo_root" in r.getMessage()]
    assert msgs, "must log a WARNING when discovery drops entries"
    assert "fused_rms_fp8" in msgs[0]


def test_drops_benchmarks_outside_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "kernel_dir"
    repo.mkdir()
    inside = repo / "bench.py"
    inside.write_text("# bench\n")
    outside = tmp_path / "elsewhere" / "bench.py"
    outside.parent.mkdir()
    outside.write_text("# different bench\n")

    disc = _disc(benchmarks=[{"file": str(inside)}, {"file": str(outside)}])
    out = _filter_discovery_to_repo_root(disc, str(repo))

    assert [b["file"] for b in out["benchmarks"]] == [str(inside)]
    assert out["total_benchmarks_found"] == 1


def test_returns_unchanged_when_all_inside(tmp_path: Path) -> None:
    repo = tmp_path / "k"
    repo.mkdir()
    a = repo / "a.py"
    a.write_text("")
    b = repo / "b.py"
    b.write_text("")

    disc = _disc(tests=[{"file": str(a)}, {"file": str(b)}])
    out = _filter_discovery_to_repo_root(disc, str(repo))
    # When nothing is dropped, the helper short-circuits and returns the
    # same dict by reference -- the caller can detect a no-op cheaply.
    assert out is disc


def test_handles_unparsable_paths_conservatively(tmp_path: Path) -> None:
    # If we cannot classify an entry (e.g. file is None or weird type),
    # keep it -- erring on the side of preserving discovery output.
    repo = tmp_path / "k"
    repo.mkdir()
    disc = _disc(tests=[{"file": None}, {"file": 42}, {"name": "no-file-key"}])
    out = _filter_discovery_to_repo_root(disc, str(repo))
    assert len(out["tests"]) == 3


def test_non_dict_disc_returned_unchanged() -> None:
    # Defensive: if discovery itself wasn't a dict (very rare), don't crash.
    assert _filter_discovery_to_repo_root("not-a-dict", "/tmp") == "not-a-dict"  # type: ignore[arg-type]
