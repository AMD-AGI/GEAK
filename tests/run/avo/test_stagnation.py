"""Unit tests for the deterministic StagnationDetector (GPU-free)."""

from __future__ import annotations

from pathlib import Path

from minisweagent.run.avo.result import AttemptRecord, VariationResult
from minisweagent.run.avo.stagnation import StagnationDetector, StagnationLevel

_CFG = {
    "steps_without_commit": 5,
    "wall_time_without_commit_s": 1_000_000,
    "consecutive_correctness_failures": 3,
    "consecutive_no_improvement": 4,
    "patch_hash_repeat": 3,
    "supervisor_cycles_without_commit": 3,
}


def _result(*, speedup=None, correct=False, patch_hash=None, wall=1.0) -> VariationResult:
    return VariationResult(
        step_index=1,
        step_dir=Path("."),
        attempts=[AttemptRecord(correctness_passed=correct, verified_speedup=speedup, patch_hash=patch_hash)],
        best_speedup=speedup,
        best_correct=correct,
        wall_time_s=wall,
    )


def test_commit_resets_to_none():
    det = StagnationDetector(_CFG)
    det.consecutive_no_improvement = 3
    sig = det.evaluate(_result(speedup=1.2, correct=True), committed=True)
    assert sig.level == StagnationLevel.NONE
    assert det.consecutive_no_improvement == 0


def test_consecutive_no_improvement_triggers_redirect():
    det = StagnationDetector(_CFG)
    sig = None
    for _ in range(_CFG["consecutive_no_improvement"]):
        sig = det.evaluate(_result(speedup=1.0, correct=True), committed=False)
    assert sig.level == StagnationLevel.REDIRECT


def test_correctness_failures_trigger_redirect():
    det = StagnationDetector(_CFG)
    sig = None
    for _ in range(_CFG["consecutive_correctness_failures"]):
        sig = det.evaluate(_result(speedup=None, correct=False), committed=False)
    assert sig.level == StagnationLevel.REDIRECT


def test_patch_cycle_triggers_redirect():
    det = StagnationDetector(_CFG)
    sig = None
    for _ in range(_CFG["patch_hash_repeat"]):
        sig = det.evaluate(_result(speedup=1.0, correct=True, patch_hash="deadbeef"), committed=False)
    assert sig.level >= StagnationLevel.REDIRECT


def test_supervisor_cycles_escalate():
    det = StagnationDetector(_CFG)
    for _ in range(_CFG["supervisor_cycles_without_commit"]):
        det.note_supervisor_cycle()
    sig = det.evaluate(_result(speedup=1.0, correct=True), committed=False)
    assert sig.level == StagnationLevel.ESCALATE


def test_partial_reset_keeps_supervisor_cycles():
    det = StagnationDetector(_CFG)
    det.note_supervisor_cycle()
    det.steps_without_commit = 4
    det.reset(partial=True)
    assert det.steps_without_commit == 0
    assert det.supervisor_cycles_without_commit == 1


def test_full_reset_clears_supervisor_cycles():
    det = StagnationDetector(_CFG)
    det.note_supervisor_cycle()
    det.reset(partial=False)
    assert det.supervisor_cycles_without_commit == 0


def test_nudge_before_redirect():
    det = StagnationDetector(_CFG)
    # half of consecutive_no_improvement (4 // 2 = 2) → NUDGE, not yet REDIRECT
    det.evaluate(_result(speedup=1.0, correct=True), committed=False)
    sig = det.evaluate(_result(speedup=1.0, correct=True), committed=False)
    assert sig.level == StagnationLevel.NUDGE
