"""Run-level wall-clock budget for ``geak --mode quick|full``.

This module owns:

- ``BudgetSpec``                   -- per-mode time knobs loaded from YAML.
                                      ``total_s`` is the absolute wall-clock
                                      cap (inclusive of ``kill_buffer_s``).
- ``RunBudget``                    -- monotonic clock, ``soft_stop`` event,
                                      a watchdog (``threading.Timer``) that
                                      flips ``soft_stop`` ~``finalize_grace_s``
                                      before the cooperative ``opt_deadline``,
                                      and the hard-kill watchdog anchored on
                                      ``started_at + total_s``.
- ``Deadline``                     -- a snapshot view of the optimization
                                      deadline that callers can poll cheaply.

Two distinct guarantees:

- **Cooperative**: ``soft_stop`` is set ~``finalize_grace_s`` before the
  cooperative ``opt_deadline``; the cooperative deadline sits
  ``kill_buffer_s`` before the absolute wall-clock cap.
- **Absolute**: the hard-kill watchdog fires at exactly
  ``started_at + total_s``. This is the cap enforcement -- it never
  derives from the cooperative deadline, so preprocess overrun cannot
  push it past ``total_s``.

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
    """Per-mode budget knobs (loaded from ``run.budgets.<mode>`` in geak.yaml).

    ``total_s`` is the **absolute wall-clock cap** for the run, inclusive of
    the internal ``kill_buffer_s`` margin. The hard-kill watchdog anchors on
    ``started_at + total_s`` (see
    :meth:`RunBudget.schedule_optimization_hard_kill_watchdog`); this anchor
    is what makes the cap a guarantee. Do not refactor the watchdog to derive
    its delay from ``opt_deadline_at + kill_buffer_s``: under preprocess
    overrun that would push hard-kill past ``total_s``.

    ``kill_buffer_s`` is the cooperative-finalize headroom the optimization
    phase reserves up front (see :meth:`RunBudget.commit_preprocess`). It
    widens the gap between the cooperative ``opt_deadline`` and the absolute
    hard-kill point in the happy path, but is **not** load-bearing for the
    cap promise -- removing it would only shrink finalize headroom, not push
    hard-kill past ``started_at + total_s``.

    Surprising-but-correct log surface: if a future config bumps
    ``preprocess_hard_cap_fraction`` to 1.0 and preprocess actually consumes
    the entire ``total_s``, the operator log will show
    ``"preprocess committed at +<total_s>s; optimization budget = 0s"``
    immediately followed by ``"[geak HARD-KILL] ..."``. This is the watchdog
    correctly enforcing the absolute cap with no remaining cooperative time.
    """

    mode: Mode
    total_s: float
    preprocess_soft_cap_s: float
    preprocess_hard_cap_fraction: float
    finalize_grace_s: float
    # Reserved up front by ``commit_preprocess`` so the cooperative
    # ``opt_deadline`` sits ``kill_buffer_s`` before the absolute hard-kill
    # anchor. Default 60s.
    kill_buffer_s: float = 60.0

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
        """Clamp a requested timeout to ``[0, remaining]``.

        Returns ``0.0`` once SoftStop has fired so that callers using
        ``cap()`` to size new subprocess timeouts will reject any new
        long-running work; the cooperative ``soft_stop.is_set()`` poll
        is still the primary signal but a 0-cap makes the Deadline API
        safe-by-default at this boundary.
        """
        try:
            requested = float(requested_s)
        except (TypeError, ValueError):
            return 0.0
        if self._soft_stop.is_set():
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

    def schedule_optimization_hard_kill_watchdog(self, on_kill: Callable[[], None]) -> None:
        """Schedule the unconditional hard-kill watchdog for the optimization phase.

        **Anchored on ``started_at + total_s``**, not on
        ``opt_deadline_at + kill_buffer_s``. This makes ``total_s`` the
        absolute wall-clock cap regardless of preprocess overrun. If
        preprocess used the entire budget and ``opt_budget_s`` clamped to 0,
        the timer simply fires close to immediately rather than slipping
        ``kill_buffer_s`` past the cap.

        Calls ``on_kill`` (typically ``state.registry.terminate_all`` followed
        by ``os._exit(124)``) so a stuck inner tool call -- one that never
        observes ``soft_stop`` because it is mid-``subprocess.run`` -- still
        terminates the process at the advertised wall-clock cap.

        The phase guard intentionally keeps this callable in 'optimization'
        and 'finalize' phases (we transition to 'finalize' inside the
        cooperative shutdown path; the killer must still fire if cooperative
        shutdown stalls). We accept this by not phase-guarding the kill
        callback.
        """
        if self._opt_deadline_at is None:
            raise RuntimeError("commit_preprocess must run before schedule_optimization_hard_kill_watchdog")

        absolute_kill_at = self.started_at + self.spec.total_s
        kill_delay_s = max(0.0, absolute_kill_at - time.monotonic())

        def _wrapper() -> None:
            # No phase guard: the kill is a backstop, not a cooperative
            # signal. If we've already cleanly exited, this Timer was
            # cancelled in cancel_all_timers().
            try:
                on_kill()
            except SystemExit:
                raise  # os._exit raises this; let it through
            except Exception:
                logger.exception("hard-kill watchdog callback failed")

        timer = threading.Timer(kill_delay_s, _wrapper)
        timer.daemon = True
        timer.name = "geak-optimization-hard-kill"
        with self._lock:
            self._timers.append(timer)
        timer.start()
        logger.debug(
            "optimization hard-kill watchdog scheduled: kill @+%.0fs "
            "(absolute_kill_at=started_at+%.0fs, opt_deadline @+%.0fs)",
            kill_delay_s,
            self.spec.total_s,
            self._opt_deadline_at - time.monotonic(),
        )

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
        phase actually took. The cooperative optimization budget reserves
        ``kill_buffer_s`` up front so the cooperative ``opt_deadline`` lands
        ``kill_buffer_s`` before the absolute hard-kill anchor in the happy
        path (see :meth:`schedule_optimization_hard_kill_watchdog`):

            opt_budget_s = max(0, total_s - kill_buffer_s - preprocess_actual)

        This subtraction is **headroom**, not the wall-clock cap enforcer.
        Even if preprocess overruns and clamps ``opt_budget_s`` to 0, the
        hard-kill anchor is unaffected: it fires at ``started_at + total_s``
        regardless.

        Phase is flipped *first* so that any preprocess timer mid-callback
        becomes a no-op as soon as it next checks ``_phase`` -- this is what
        defuses the cancel-vs-fire race.
        """
        self._set_phase("optimization")
        opt_budget_s = max(
            0.0,
            self.spec.total_s
            - self.spec.kill_buffer_s
            - max(0.0, float(actual_preprocess_s)),
        )
        # opt_deadline is anchored on the *current* monotonic time so that the
        # split-allocation rule "opt = total - kill_buffer - preprocess_actual"
        # composes correctly even when preprocess hit the hard cap.
        self._opt_deadline_at = time.monotonic() + opt_budget_s
        logger.info(
            "preprocess committed at +%.0fs; optimization budget = %.0fs "
            "(deadline @+%.0fs, kill_buffer=%.0fs reserved)",
            actual_preprocess_s,
            opt_budget_s,
            opt_budget_s,
            self.spec.kill_buffer_s,
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
            f"[budget] mode={self.spec.mode}, total={self.spec.total_s:.0f}s "
            f"(absolute wall-clock cap), "
            f"preprocess_soft_cap={self.spec.preprocess_soft_cap_s:.0f}s, "
            f"preprocess_hard_cap={self.spec.preprocess_hard_cap_s:.0f}s, "
            f"finalize_grace={self.spec.finalize_grace_s:.0f}s, "
            f"kill_buffer={self.spec.kill_buffer_s:.0f}s (internal margin)",
        ]
