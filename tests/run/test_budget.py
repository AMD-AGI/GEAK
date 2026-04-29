"""Unit tests for ``run/budget.py``.

These tests cover the budget arithmetic, ``Deadline.cap`` clamping, the
watchdog ``threading.Timer`` flipping ``SoftStop`` on schedule, and the
phase-counter that defuses the ``Timer.cancel()``-vs-fire race.
"""

from __future__ import annotations

import threading
import time

import pytest

from minisweagent.run.budget import BudgetSpec, Deadline, RunBudget


def _spec(total_s: float = 10.0, soft_cap_s: float = 2.0, frac: float = 0.5, grace_s: float = 1.0) -> BudgetSpec:
    return BudgetSpec(
        mode="quick",
        total_s=total_s,
        preprocess_soft_cap_s=soft_cap_s,
        preprocess_hard_cap_fraction=frac,
        finalize_grace_s=grace_s,
    )


# ---------------------------------------------------------------------------
# Deadline.cap clamping
# ---------------------------------------------------------------------------


def test_deadline_cap_clamps_to_remaining():
    soft_stop = threading.Event()
    deadline = Deadline(time.monotonic() + 1.0, soft_stop)
    assert deadline.cap(10.0) <= 1.0 + 1e-3
    assert deadline.cap(0.5) == pytest.approx(0.5, abs=1e-3)


def test_deadline_cap_never_negative():
    soft_stop = threading.Event()
    deadline = Deadline(time.monotonic() - 5.0, soft_stop)  # already past
    assert deadline.cap(10.0) == 0.0
    assert deadline.expired()


def test_deadline_soft_stopped_reflects_event():
    soft_stop = threading.Event()
    deadline = Deadline(time.monotonic() + 60.0, soft_stop)
    assert not deadline.soft_stopped()
    soft_stop.set()
    assert deadline.soft_stopped()


# ---------------------------------------------------------------------------
# RunBudget.commit_preprocess: rollover + clamp
# ---------------------------------------------------------------------------


def test_commit_preprocess_under_cap_rolls_into_optimization():
    """Preprocess under the soft cap leaves more time for optimization."""
    budget = RunBudget(spec=_spec(total_s=10.0, soft_cap_s=2.0))
    deadline = budget.commit_preprocess(actual_preprocess_s=1.0)
    # opt budget = 10 - 1 = 9s remaining
    assert deadline.remaining() == pytest.approx(9.0, abs=0.5)


def test_commit_preprocess_over_cap_borrows_from_optimization():
    """Preprocess overrunning soft cap shrinks opt; budget arithmetic handles it."""
    budget = RunBudget(spec=_spec(total_s=10.0, soft_cap_s=2.0))
    deadline = budget.commit_preprocess(actual_preprocess_s=4.0)
    # opt budget = 10 - 4 = 6s remaining
    assert deadline.remaining() == pytest.approx(6.0, abs=0.5)


def test_commit_preprocess_at_or_past_total_clamps_to_zero():
    """If preprocess used the entire (or more than) total budget, opt = 0 (not negative)."""
    budget = RunBudget(spec=_spec(total_s=10.0))
    deadline = budget.commit_preprocess(actual_preprocess_s=10.0)
    assert deadline.remaining() == pytest.approx(0.0, abs=0.1)
    assert deadline.expired()

    deadline2 = RunBudget(spec=_spec(total_s=10.0)).commit_preprocess(actual_preprocess_s=15.0)
    assert deadline2.remaining() == pytest.approx(0.0, abs=0.1)


def test_commit_preprocess_negative_actual_clamps_to_zero():
    """Defensive: negative actual_preprocess_s is treated as zero."""
    budget = RunBudget(spec=_spec(total_s=10.0))
    deadline = budget.commit_preprocess(actual_preprocess_s=-5.0)
    # opt budget = 10 - max(0, -5) = 10
    assert deadline.remaining() == pytest.approx(10.0, abs=0.5)


def test_commit_preprocess_transitions_phase_to_optimization():
    budget = RunBudget(spec=_spec())
    assert budget.phase == "preprocess"
    budget.commit_preprocess(actual_preprocess_s=1.0)
    assert budget.phase == "optimization"


# ---------------------------------------------------------------------------
# Watchdog Timer flips SoftStop on schedule
# ---------------------------------------------------------------------------


def test_optimization_watchdog_flips_soft_stop_on_schedule():
    """At softstop_at = opt_deadline - finalize_grace, soft_stop is set."""
    # 200ms total, 100ms grace -> soft_stop fires at 100ms. Use generous
    # buffer (300ms) for thread scheduling jitter.
    spec = BudgetSpec(
        mode="quick",
        total_s=10.0,
        preprocess_soft_cap_s=2.0,
        preprocess_hard_cap_fraction=0.5,
        finalize_grace_s=0.1,
    )
    budget = RunBudget(spec=spec)
    # Pretend preprocess used (10 - 0.2) = 9.8s, leaving 0.2s for opt.
    budget.commit_preprocess(actual_preprocess_s=9.8)
    budget.schedule_optimization_watchdog()
    # softstop_at = 0.2 - 0.1 = 0.1 from now. Allow 200ms buffer.
    assert not budget.soft_stop.is_set()
    fired = budget.soft_stop.wait(timeout=0.5)
    assert fired, "watchdog should have fired soft_stop within 500ms"
    budget.cancel_all_timers()


def test_phase_guard_makes_stale_callback_a_noop():
    """A preprocess watchdog whose callback fires *after* phase has moved to
    'optimization' should be a no-op. This is the cancel-vs-fire race fix.
    """
    spec = _spec(total_s=10.0, soft_cap_s=0.05)  # 50ms preprocess soft cap
    budget = RunBudget(spec=spec)

    fired = threading.Event()

    def on_soft():
        fired.set()

    budget.schedule_preprocess_watchdogs(on_soft=on_soft, on_hard=lambda: None)
    # Transition phase to 'optimization' immediately, before timer fires.
    budget.commit_preprocess(actual_preprocess_s=0.01)
    # Wait long enough for the timer to expire.
    time.sleep(0.15)
    assert not fired.is_set(), "soft callback should not have fired post-phase-transition"
    budget.cancel_all_timers()


def test_phase_guard_logs_failures_not_swallow():
    """A callback that raises should be logged, not silently swallowed."""
    import logging

    spec = _spec(total_s=10.0, soft_cap_s=0.05)
    budget = RunBudget(spec=spec)

    def bad_callback():
        raise RuntimeError("synthetic failure")

    # Capture the budget logger.
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            records.append(record)

    handler = _Capture(level=logging.DEBUG)
    logger_obj = logging.getLogger("minisweagent.run.budget")
    logger_obj.addHandler(handler)
    try:
        budget.schedule_preprocess_watchdogs(on_soft=bad_callback, on_hard=lambda: None)
        time.sleep(0.2)
        assert any(
            "watchdog callback failed" in r.getMessage() and "synthetic failure" in (r.exc_text or "") for r in records
        ), "exception should be logged with traceback"
    finally:
        logger_obj.removeHandler(handler)
        budget.cancel_all_timers()


def test_cancel_all_timers_idempotent():
    budget = RunBudget(spec=_spec(soft_cap_s=10.0))
    budget.schedule_preprocess_watchdogs(on_soft=lambda: None, on_hard=lambda: None)
    budget.cancel_all_timers()
    budget.cancel_all_timers()  # should not raise
