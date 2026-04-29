"""Run-level wall-clock budget for ``geak --mode quick|full``.

This module owns:

- ``BudgetSpec``                   -- per-mode time knobs loaded from YAML.
- ``RunBudget``                    -- monotonic clock, ``soft_stop`` event,
                                      a watchdog (``threading.Timer``) that flips
                                      ``soft_stop`` ~``finalize_grace_s`` before
                                      the optimization deadline, and a phase
                                      counter that defuses the
                                      ``Timer.cancel()``-vs-fire race.
- ``Deadline``                     -- a snapshot view of the optimization deadline
                                      that callers can poll cheaply.

The module is pure stdlib; it intentionally does not install OS signal
handlers (the SIGINT handler is owned by ``run/mini.py`` so it can also
reach the ``ProcessRegistry``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec & Phase
# ---------------------------------------------------------------------------


Mode = Literal["quick", "full"]
Phase = Literal["preprocess", "optimization", "finalize", "done"]


@dataclass
class BudgetSpec:
    """Per-mode budget knobs (loaded from ``run.budgets.<mode>`` in geak.yaml)."""

    mode: Mode
    total_s: float
    preprocess_soft_cap_s: float
    preprocess_hard_cap_fraction: float
    finalize_grace_s: float

    @property
    def preprocess_hard_cap_s(self) -> float:
        return self.total_s * self.preprocess_hard_cap_fraction


# ---------------------------------------------------------------------------
# Deadline (read-only view for polling)
# ---------------------------------------------------------------------------


class Deadline:
    """Read-only view of an absolute monotonic deadline.

    Callers poll ``soft_stopped()`` / ``expired()`` at loop boundaries and use
    ``cap()`` to clamp per-subprocess timeouts so they cannot run past the
    deadline.
    """

    __slots__ = ("_deadline_at", "_soft_stop")

    def __init__(self, deadline_at: float, soft_stop: threading.Event):
        self._deadline_at = float(deadline_at)
        self._soft_stop = soft_stop

    @property
    def at(self) -> float:
        return self._deadline_at

    def remaining(self) -> float:
        return max(0.0, self._deadline_at - time.monotonic())

    def expired(self) -> bool:
        return time.monotonic() >= self._deadline_at

    def soft_stopped(self) -> bool:
        return self._soft_stop.is_set()

    def cap(self, requested_s: float) -> float:
        """Clamp a requested timeout to ``[0, remaining]``."""
        try:
            requested = float(requested_s)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(requested, self.remaining()))


# ---------------------------------------------------------------------------
# RunBudget
# ---------------------------------------------------------------------------


@dataclass
class RunBudget:
    """Owns the monotonic clock, the ``soft_stop`` event, and watchdog timers.

    Lifecycle:

    1. Construct with ``BudgetSpec``. ``started_at = time.monotonic()`` is
       captured automatically.
    2. ``schedule_preprocess_watchdogs(on_soft, on_hard)`` -- two daemon
       ``threading.Timer`` instances (soft + hard) for the preprocess phase.
    3. After preprocess returns, call ``commit_preprocess(actual_s)`` to
       transition to the ``optimization`` phase. Returns the optimization
       ``Deadline``. The phase transition makes any preprocess timer that is
       *currently mid-callback* a no-op (the wrapper checks ``_phase``).
    4. ``schedule_optimization_watchdog()`` -- one daemon ``Timer`` at
       ``softstop_at = opt_deadline - finalize_grace_s`` that flips
       ``soft_stop``.
    5. ``cancel_all_timers()`` -- always called from a ``finally`` block.
    """

    spec: BudgetSpec
    started_at: float = field(default_factory=time.monotonic)
    soft_stop: threading.Event = field(default_factory=threading.Event)
    _phase: Phase = "preprocess"
    _timers: list[threading.Timer] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _opt_deadline_at: float | None = field(default=None, repr=False)

    # ----- phase management -----

    @property
    def phase(self) -> Phase:
        return self._phase

    def _set_phase(self, new_phase: Phase) -> None:
        with self._lock:
            old = self._phase
            self._phase = new_phase
        if old != new_phase:
            logger.debug("RunBudget phase transition: %s -> %s", old, new_phase)

    def _wrap(self, expected_phase: Phase, fn: Callable[[], None]) -> Callable[[], None]:
        """Wrap ``fn`` so that it is a no-op if the phase has moved on, and so
        any exception is logged rather than silently swallowed by ``Timer``.
        """

        def _wrapper() -> None:
            current = self._phase
            if current != expected_phase:
                logger.debug(
                    "watchdog callback skipped (expected phase=%s, current=%s)",
                    expected_phase,
                    current,
                )
                return
            try:
                fn()
            except Exception:
                logger.exception("watchdog callback failed (phase=%s)", expected_phase)

        return _wrapper

    # ----- timers -----

    def schedule_preprocess_watchdogs(
        self,
        on_soft: Callable[[], None],
        on_hard: Callable[[], None],
    ) -> None:
        """Schedule both soft and hard preprocess watchdogs (daemon timers)."""
        soft = threading.Timer(self.spec.preprocess_soft_cap_s, self._wrap("preprocess", on_soft))
        soft.daemon = True
        soft.name = "geak-preprocess-soft"
        hard = threading.Timer(self.spec.preprocess_hard_cap_s, self._wrap("preprocess", on_hard))
        hard.daemon = True
        hard.name = "geak-preprocess-hard"
        with self._lock:
            self._timers.extend([soft, hard])
        soft.start()
        hard.start()
        logger.debug(
            "preprocess watchdogs scheduled: soft @+%.0fs, hard @+%.0fs",
            self.spec.preprocess_soft_cap_s,
            self.spec.preprocess_hard_cap_s,
        )

    def cancel_preprocess_watchdogs(self) -> None:
        """Cancel any timers scheduled for the preprocess phase.

        Note: ``Timer.cancel()`` only prevents *not-yet-fired* timers from
        firing. A timer whose callback is already running cannot be stopped;
        the ``_wrap`` phase-guard handles that case by making it a no-op once
        we transition.
        """
        with self._lock:
            timers = list(self._timers)
            self._timers.clear()
        for t in timers:
            try:
                t.cancel()
            except Exception:
                logger.exception("cancelling timer %s failed", getattr(t, "name", "?"))

    def schedule_optimization_watchdog(self) -> Deadline:
        """Schedule the soft-stop watchdog for the optimization phase.

        Must be called *after* ``commit_preprocess``. Fires at
        ``opt_deadline - finalize_grace_s``. Returns the optimization Deadline
        (also accessible via :py:meth:`optimization_deadline`).
        """
        if self._phase != "optimization":
            raise RuntimeError(
                f"schedule_optimization_watchdog called in phase={self._phase!r}; "
                f"expected 'optimization'. Did you forget to call commit_preprocess()?"
            )
        if self._opt_deadline_at is None:
            raise RuntimeError("commit_preprocess must run before schedule_optimization_watchdog")

        softstop_delay_s = max(0.0, self._opt_deadline_at - time.monotonic() - self.spec.finalize_grace_s)
        timer = threading.Timer(softstop_delay_s, self._wrap("optimization", self.soft_stop.set))
        timer.daemon = True
        timer.name = "geak-optimization-softstop"
        with self._lock:
            self._timers.append(timer)
        timer.start()
        logger.debug(
            "optimization watchdog scheduled: softstop @+%.0fs (opt_deadline @+%.0fs)",
            softstop_delay_s,
            self._opt_deadline_at - time.monotonic(),
        )
        return Deadline(self._opt_deadline_at, self.soft_stop)

    def cancel_all_timers(self) -> None:
        with self._lock:
            timers = list(self._timers)
            self._timers.clear()
        for t in timers:
            try:
                t.cancel()
            except Exception:
                logger.exception("cancelling timer %s failed", getattr(t, "name", "?"))

    # ----- arithmetic / phase transitions -----

    def deadline_for_preprocess(self) -> Deadline:
        """Return a Deadline at ``T0 + preprocess_hard_cap_s`` for use by
        preprocess subprocesses to clamp their timeouts.
        """
        return Deadline(self.started_at + self.spec.preprocess_hard_cap_s, self.soft_stop)

    def commit_preprocess(self, actual_preprocess_s: float) -> Deadline:
        """Transition phase ``preprocess -> optimization``.

        ``actual_preprocess_s`` is the wall-clock duration the preprocess
        phase actually took. The optimization budget is
        ``max(0.0, total_s - actual_preprocess_s)`` (the rollover formula:
        any time preprocess didn't use, optimization gets; if preprocess
        overran, optimization shrinks; if preprocess used the entire budget,
        optimization gets 0 and finalize_grace_s).

        Phase is flipped *first* so that any preprocess timer mid-callback
        becomes a no-op as soon as it next checks ``_phase`` -- this is what
        defuses the cancel-vs-fire race.
        """
        self._set_phase("optimization")
        opt_budget_s = max(0.0, self.spec.total_s - max(0.0, float(actual_preprocess_s)))
        # opt_deadline is anchored on the *current* monotonic time so that the
        # split-allocation rule "opt = total - preprocess_actual" composes
        # correctly even when preprocess hit the hard cap.
        self._opt_deadline_at = time.monotonic() + opt_budget_s
        logger.info(
            "preprocess committed at +%.0fs; optimization budget = %.0fs (deadline @+%.0fs)",
            actual_preprocess_s,
            opt_budget_s,
            opt_budget_s,
        )
        return Deadline(self._opt_deadline_at, self.soft_stop)

    def optimization_deadline(self) -> Deadline:
        if self._opt_deadline_at is None:
            raise RuntimeError("optimization_deadline available only after commit_preprocess")
        return Deadline(self._opt_deadline_at, self.soft_stop)

    # ----- ergonomics -----

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def banner_lines(self) -> list[str]:
        """Human-readable banner showing the budget layout."""
        return [
            f"[budget] mode={self.spec.mode}, total={self.spec.total_s:.0f}s, "
            f"preprocess_soft_cap={self.spec.preprocess_soft_cap_s:.0f}s, "
            f"preprocess_hard_cap={self.spec.preprocess_hard_cap_s:.0f}s, "
            f"finalize_grace={self.spec.finalize_grace_s:.0f}s",
        ]
