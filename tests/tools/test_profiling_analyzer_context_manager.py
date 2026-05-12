"""Tests for ``ProfilingAnalyzer`` as a context manager.

The temp dir created in ``__init__`` was historically a leak waiting to
happen: out-of-tree callers had to remember to call ``cleanup()`` by hand,
and any exception path between construction and cleanup leaked. The
context-manager protocol makes the correct usage automatic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.tools.profiling_tools import ProfilingAnalyzer


def test_context_manager_cleans_temp_dir() -> None:
    """Exiting the ``with`` block removes the analyzer's temp output_path."""
    with ProfilingAnalyzer(profiling_type="profiling") as analyzer:
        out = analyzer.output_path
        assert isinstance(out, Path)
        assert out.is_dir(), f"{out} should be created by __init__"
    assert not out.exists(), f"{out} should be cleaned by __exit__"


def test_context_manager_calls_cleanup_on_exception() -> None:
    """An exception inside the ``with`` block propagates AND cleanup runs."""

    captured_path: Path | None = None

    def _body() -> None:
        nonlocal captured_path
        with ProfilingAnalyzer(profiling_type="profiling") as analyzer:
            captured_path = analyzer.output_path
            assert captured_path.is_dir()
            raise RuntimeError("simulated failure inside with-block")

    with pytest.raises(RuntimeError, match="simulated"):
        _body()

    assert captured_path is not None
    assert not captured_path.exists(), "cleanup must run even when the body raises"


def test_explicit_cleanup_still_works_for_out_of_tree_callers() -> None:
    """Back-compat: callers that don't use the ``with`` form still get cleanup."""
    analyzer = ProfilingAnalyzer(profiling_type="profiling")
    out = analyzer.output_path
    assert out.is_dir()
    analyzer.cleanup()
    assert not out.exists()


def test_cleanup_is_idempotent() -> None:
    """Repeated cleanup() must not raise (out-of-tree callers + the ``with``
    block double-call when both code paths are kept temporarily)."""
    analyzer = ProfilingAnalyzer(profiling_type="profiling")
    analyzer.cleanup()
    analyzer.cleanup()  # must not raise
