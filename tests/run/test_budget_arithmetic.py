"""Tests for the wall-clock cap arithmetic in ``RunBudget``.

The single load-bearing invariant: hard-kill fires at exactly
``started_at + total_s``, regardless of how preprocess plays out. The
``commit_preprocess`` ``kill_buffer_s`` subtraction is cooperative
headroom only; it must never alter the cap.
"""

from __future__ import annotations

import time

import pytest

from minisweagent.run.budget import BudgetSpec, RunBudget


def _spec(total_s: float = 10.0, kill_buffer_s: float = 2.0) -> BudgetSpec:
    return BudgetSpec(
        mode="quick",
        total_s=total_s,
        preprocess_soft_cap_s=2.0,
        preprocess_hard_cap_fraction=0.5,
        finalize_grace_s=1.0,
        kill_buffer_s=kill_buffer_s,
    )


def _captured_kill_timer(budget: RunBudget):
    """Return the threading.Timer object scheduled by the hard-kill watchdog."""
    matches = [t for t in budget._timers if t.name == "geak-optimization-hard-kill"]
    assert len(matches) == 1, f"expected exactly one hard-kill timer, got {matches}"
    return matches[0]


def test_hard_kill_fires_at_total_s_happy_path() -> None:
    """Short preprocess: hard-kill timer fires exactly total_s after started_at."""
    spec = _spec(total_s=10.0, kill_buffer_s=2.0)
    budget = RunBudget(spec=spec)
    started_at = budget.started_at

    # Simulate 3 s of preprocess elapsed.
    budget.commit_preprocess(actual_preprocess_s=3.0)
    schedule_time = time.monotonic()
    budget.schedule_optimization_hard_kill_watchdog(on_kill=lambda: None)

    timer = _captured_kill_timer(budget)
    expected_delay = (started_at + spec.total_s) - schedule_time
    # Timer.interval is set from the kill_delay_s we passed; epsilon for
    # scheduling jitter between commit_preprocess and the watchdog scheduling.
    assert timer.interval == pytest.approx(expected_delay, abs=0.05)

    budget.cancel_all_timers()


def test_hard_kill_fires_at_total_s_even_when_preprocess_overran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate wall-clock advance to ``started_at + total_s`` during
    preprocess. The hard-kill timer must anchor on ``started_at + total_s``
    (so its interval from "now" is near zero), NOT on
    ``opt_deadline_at + kill_buffer_s`` (which would push it ``kill_buffer_s``
    past the cap -- exactly the bug we're guarding against).
    """
    import minisweagent.run.budget as budget_mod

    real_monotonic = time.monotonic
    started_at = real_monotonic()
    # Pretend ``total_s`` has fully elapsed by the time commit_preprocess +
    # schedule_optimization_hard_kill_watchdog run.
    now = {"t": started_at + 10.0}

    def fake_monotonic() -> float:
        return now["t"]

    monkeypatch.setattr(budget_mod.time, "monotonic", fake_monotonic)

    spec = _spec(total_s=10.0, kill_buffer_s=2.0)
    # Build the budget with the fake clock active so started_at is captured
    # from the fake clock (avoids drift between real and fake started_at).
    monkeypatch.setattr(
        budget_mod,
        "time",
        type("FakeTime", (), {"monotonic": staticmethod(fake_monotonic)})(),
    )
    # Re-import-style reset: easier to construct RunBudget with explicit
    # started_at instead.
    budget = RunBudget(spec=spec, started_at=started_at)

    budget.commit_preprocess(actual_preprocess_s=spec.total_s)
    budget.schedule_optimization_hard_kill_watchdog(on_kill=lambda: None)

    timer = _captured_kill_timer(budget)
    # absolute_kill_at - now == (started_at + total_s) - (started_at + total_s) == 0
    assert timer.interval == pytest.approx(0.0, abs=0.05)
    # Crucially: NOT kill_buffer_s. That would mean the cap slipped.
    assert timer.interval < spec.kill_buffer_s

    budget.cancel_all_timers()


def test_commit_preprocess_clamps_opt_budget_to_zero_when_overrun() -> None:
    """preprocess_actual > total_s - kill_buffer_s clamps opt_budget to 0."""
    spec = _spec(total_s=10.0, kill_buffer_s=2.0)
    budget = RunBudget(spec=spec)
    budget.commit_preprocess(actual_preprocess_s=15.0)
    # opt_deadline_at lands at monotonic_now (opt_budget_s == 0).
    assert budget._opt_deadline_at is not None
    assert (budget._opt_deadline_at - time.monotonic()) <= 0.05
    budget.cancel_all_timers()


def test_commit_preprocess_reserves_kill_buffer_in_happy_path() -> None:
    """Short preprocess: cooperative opt_budget reserves kill_buffer_s.

    In production ``time.monotonic()`` at commit time is
    ``started_at + actual_preprocess_s``, so opt_deadline lands at
    ``started_at + total_s - kill_buffer_s``. The test (no real clock
    advance) verifies the formula directly: ``opt_deadline_at - now`` must
    equal ``total_s - kill_buffer_s - actual_preprocess_s``.
    """
    spec = _spec(total_s=10.0, kill_buffer_s=2.0)
    budget = RunBudget(spec=spec)
    budget.commit_preprocess(actual_preprocess_s=3.0)
    commit_time = time.monotonic()
    assert budget._opt_deadline_at is not None
    remaining_budget = budget._opt_deadline_at - commit_time
    expected = spec.total_s - spec.kill_buffer_s - 3.0  # 10 - 2 - 3 = 5
    assert remaining_budget == pytest.approx(expected, abs=0.05)
    budget.cancel_all_timers()


def test_banner_lines_advertises_kill_buffer() -> None:
    """Structural check: banner mentions total= and kill_buffer= so operators
    can correlate cap arithmetic from the log."""
    spec = _spec(total_s=10.0, kill_buffer_s=2.0)
    budget = RunBudget(spec=spec)
    lines = budget.banner_lines()
    text = "\n".join(lines)
    assert f"total={spec.total_s:.0f}s" in text
    assert f"kill_buffer={spec.kill_buffer_s:.0f}s" in text
    budget.cancel_all_timers()
