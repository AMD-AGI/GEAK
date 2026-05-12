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


def _spec(
    total_s: float = 10.0,
    soft_cap_s: float = 2.0,
    frac: float = 0.5,
    grace_s: float = 1.0,
    kill_buffer_s: float = 1.0,
) -> BudgetSpec:
    return BudgetSpec(
        mode="quick",
        total_s=total_s,
        preprocess_soft_cap_s=soft_cap_s,
        preprocess_hard_cap_fraction=frac,
        finalize_grace_s=grace_s,
        kill_buffer_s=kill_buffer_s,
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


def test_deadline_cap_returns_zero_when_softstop_is_set():
    """Once SoftStop has fired, ``cap()`` returns 0.0 regardless of remaining
    wallclock so callers using ``cap()`` to size new subprocess timeouts will
    refuse to start new long-running work without needing a separate
    ``soft_stop.is_set()`` poll at every site.
    """
    soft_stop = threading.Event()
    deadline = Deadline(time.monotonic() + 60.0, soft_stop)
    assert deadline.cap(10.0) == pytest.approx(10.0, abs=1e-3)
    soft_stop.set()
    assert deadline.cap(10.0) == 0.0
    assert deadline.cap(0.5) == 0.0


# ---------------------------------------------------------------------------
# RunBudget.commit_preprocess: rollover + clamp
# ---------------------------------------------------------------------------


def test_commit_preprocess_under_cap_rolls_into_optimization():
    """Preprocess under the soft cap leaves more time for optimization.

    Formula: ``opt_budget = max(0, total_s - kill_buffer_s - preprocess_actual)``.
    """
    budget = RunBudget(spec=_spec(total_s=10.0, soft_cap_s=2.0, kill_buffer_s=1.0))
    deadline = budget.commit_preprocess(actual_preprocess_s=1.0)
    # opt budget = 10 - 1 (kill_buffer) - 1 (preprocess) = 8s remaining
    assert deadline.remaining() == pytest.approx(8.0, abs=0.5)


def test_commit_preprocess_over_cap_borrows_from_optimization():
    """Preprocess overrunning soft cap shrinks opt; budget arithmetic handles it."""
    budget = RunBudget(spec=_spec(total_s=10.0, soft_cap_s=2.0, kill_buffer_s=1.0))
    deadline = budget.commit_preprocess(actual_preprocess_s=4.0)
    # opt budget = 10 - 1 (kill_buffer) - 4 (preprocess) = 5s remaining
    assert deadline.remaining() == pytest.approx(5.0, abs=0.5)


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
    budget = RunBudget(spec=_spec(total_s=10.0, kill_buffer_s=1.0))
    deadline = budget.commit_preprocess(actual_preprocess_s=-5.0)
    # opt budget = 10 - 1 (kill_buffer) - max(0, -5) = 9
    assert deadline.remaining() == pytest.approx(9.0, abs=0.5)


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
    spec = BudgetSpec(
        mode="quick",
        total_s=10.0,
        preprocess_soft_cap_s=2.0,
        preprocess_hard_cap_fraction=0.5,
        finalize_grace_s=0.1,
        kill_buffer_s=0.0,  # don't reserve buffer from the tiny opt window
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


# ---------------------------------------------------------------------------
# Hard-kill watchdog (the absolute backstop)
# ---------------------------------------------------------------------------


def test_hard_kill_watchdog_fires_at_started_at_plus_total_s():
    """The hard-kill watchdog must fire at ``started_at + total_s`` -- the
    absolute wall-clock cap, NOT derived from ``opt_deadline + kill_buffer_s``.

    This is the backstop that protects against a sub-agent stuck inside a
    long-running ``subprocess.run`` (e.g. a 30-min benchmark) that never
    polls ``soft_stop``. Use sub-second delays to keep the test fast.
    """
    spec = BudgetSpec(
        mode="quick",
        total_s=0.5,
        preprocess_soft_cap_s=0.05,
        preprocess_hard_cap_fraction=0.5,
        finalize_grace_s=0.05,
        kill_buffer_s=0.1,
    )
    budget = RunBudget(spec=spec)
    # Pretend preprocess used effectively no time so the absolute cap is now + 0.5s.
    budget.commit_preprocess(actual_preprocess_s=0.0)
    budget.schedule_optimization_watchdog()

    fired = threading.Event()

    def _on_kill():
        fired.set()

    budget.schedule_optimization_hard_kill_watchdog(_on_kill)
    # Should fire at started_at + total_s = ~0.5s from start.
    # Generous timeout for thread scheduling jitter on shared CI.
    assert fired.wait(timeout=2.0), "hard-kill watchdog should have fired by now"
    budget.cancel_all_timers()


def test_hard_kill_watchdog_can_be_cancelled_before_firing():
    spec = BudgetSpec(
        mode="quick",
        total_s=10.0,
        preprocess_soft_cap_s=2.0,
        preprocess_hard_cap_fraction=0.5,
        finalize_grace_s=1.0,
        kill_buffer_s=0.5,
    )
    budget = RunBudget(spec=spec)
    budget.commit_preprocess(actual_preprocess_s=0.0)
    budget.schedule_optimization_watchdog()

    fired = threading.Event()
    budget.schedule_optimization_hard_kill_watchdog(lambda: fired.set())
    # Cancel immediately, before any timer could fire.
    budget.cancel_all_timers()
    # Wait long enough that the timer would have fired if not cancelled.
    time.sleep(0.3)
    assert not fired.is_set(), "cancelled hard-kill must not fire"


def test_hard_kill_watchdog_logs_callback_exception():
    """If the hard-kill callback raises (other than SystemExit), it's logged."""
    import logging

    spec = BudgetSpec(
        mode="quick",
        total_s=0.2,
        preprocess_soft_cap_s=0.05,
        preprocess_hard_cap_fraction=0.5,
        finalize_grace_s=0.05,
        kill_buffer_s=0.05,
    )
    budget = RunBudget(spec=spec)
    budget.commit_preprocess(actual_preprocess_s=0.0)

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.DEBUG)
    logger_obj = logging.getLogger("minisweagent.run.budget")
    logger_obj.addHandler(handler)
    try:

        def _bad():
            raise RuntimeError("synthetic hard-kill failure")

        budget.schedule_optimization_hard_kill_watchdog(_bad)
        time.sleep(0.5)
        assert any("hard-kill watchdog callback failed" in r.getMessage() for r in records), (
            "exception in hard-kill callback should be logged, not swallowed"
        )
    finally:
        logger_obj.removeHandler(handler)
        budget.cancel_all_timers()


def test_kill_buffer_default_is_60_seconds():
    """Belt-and-braces: confirm the documented default doesn't drift silently."""
    spec = BudgetSpec(
        mode="quick",
        total_s=3600.0,
        preprocess_soft_cap_s=900.0,
        preprocess_hard_cap_fraction=0.5,
        finalize_grace_s=300.0,
    )
    assert spec.kill_buffer_s == 60.0
