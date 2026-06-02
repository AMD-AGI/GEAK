"""Unit tests for the P-mem-3 (option C) cross-step evolution log."""

from __future__ import annotations

from pathlib import Path

from minisweagent.run.avo.result import VariationResult
from minisweagent.run.avo.variation_step import (
    _capture_agent_memory,
    build_evolution_log,
    write_evolution_entry,
)


def _write(out: Path, step, sp, strat, committed, rationale, raw_tail, prof, correct=True):
    r = VariationResult(
        step_index=step,
        step_dir=out,
        strategy=strat,
        best_speedup=sp,
        best_correct=correct,
        rationale=rationale,
        raw_tail=raw_tail,
        profiling=prof,
        exit_status="Submitted",
    )
    write_evolution_entry(out, r, committed)


def test_evolution_log_none_when_empty(tmp_path: Path):
    assert build_evolution_log(tmp_path, k_recent=2) is None


def test_evolution_log_structure_and_recent_tail(tmp_path: Path):
    _write(tmp_path, 1, 1.10, "tiling", True, "added CTA tiling", "[assistant] 128x128\n[tool] 1.10x", {"sm_occupancy": 0.35})
    _write(tmp_path, 2, 1.05, "vectorize", False, "float4 regressed", "[assistant] float4\n[tool] 1.05x", {"sm_occupancy": 0.40})
    _write(tmp_path, 3, 1.22, "pipeline", True, "double-buffer", "[assistant] cp.async\n[tool] 1.22x", {"sm_occupancy": 0.55})

    log = build_evolution_log(tmp_path, k_recent=1, max_versions=8)
    assert log is not None
    # older steps as one-liners with commit flags
    assert "step 1:" in log and "committed" in log
    assert "step 2:" in log and "rejected" in log
    # profiling delta surfaced
    assert "bottleneck" in log
    # most recent step shown verbatim
    assert "step 3 (recent)" in log and "cp.async" in log


def test_evolution_log_k_recent_zero_all_structured(tmp_path: Path):
    _write(tmp_path, 1, 1.10, "tiling", True, "r", "[assistant] x", {"occ": 0.3})
    assert "(recent)" not in build_evolution_log(tmp_path, k_recent=0, max_versions=8)


def test_capture_agent_memory():
    class _Agent:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "I tile the GEMM 128x128 and use cp.async."},
            {"role": "tool", "content": "correctness: pass; 1.22x"},
        ]

    rationale, raw_tail = _capture_agent_memory(_Agent())
    assert "tile" in rationale.lower()
    assert "[assistant]" in raw_tail and "[tool]" in raw_tail


def test_capture_handles_empty_agent():
    class _Empty:
        messages: list = []

    rationale, raw_tail = _capture_agent_memory(_Empty())
    assert rationale == "" and raw_tail == ""
