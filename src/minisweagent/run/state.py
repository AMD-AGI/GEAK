"""Run state primitives shared by preprocess and optimization phases.

Two pieces:

- ``ProcessRegistry`` -- thread-safe registry of in-flight ``subprocess.Popen``
  / ``multiprocessing.Process`` / ``concurrent.futures.Future`` handles. The
  watchdog and SIGINT handler call ``terminate_all()`` here to actually kill
  in-flight work; without this layer, ``soft_stop`` polling alone cannot
  guarantee the finalize grace window because a hung subprocess can sit on the
  GIL-free side indefinitely.

- ``PreprocessState`` -- per-run state for the preprocess phase: the current
  pipeline stage, a few flags read by the soft-stop handler, and a reference
  to the registry so handlers can call ``terminate_all()`` and ``Popen``s can
  be tracked uniformly.

POSIX-only by design (the rest of GEAK is also POSIX-only). All ``Popen``
spawns *must* use ``start_new_session=True`` so we can ``os.killpg`` the whole
group; ``mp.Process`` targets must call ``os.setsid()`` on entry for the same
reason (mp.Process has no equivalent constructor flag).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


class PreprocessStage(str, Enum):
    """Stages of ``run_preprocessor`` in code order.

    ``HARNESS_INIT`` covers everything from kernel resolution through the
    initial harness execution but *before* the harness's benchmark mode runs
    (the benchmark mode is what produces ``benchmark_baseline.txt``, which is
    why we split it out into a separate stage -- the soft-stop handler treats
    "stuck during benchmark mode" differently from "stuck during initial
    harness setup").
    """

    HARNESS_INIT = "harness-init"
    HARNESS_BENCHMARK = "harness-benchmark"
    KERNEL_PROFILE = "kernel-profile"
    BASELINE_METRICS = "baseline-metrics"
    COMMANDMENT = "commandment"
    DONE = "done"


# ---------------------------------------------------------------------------
# ProcessRegistry
# ---------------------------------------------------------------------------


@dataclass
class ProcessRegistry:
    """Thread-safe registry of in-flight handles.

    Use the ``track*`` context managers from spawn sites:

    .. code-block:: python

        proc = subprocess.Popen([...], start_new_session=True)
        with registry.track(proc):
            out, err = proc.communicate(timeout=t)

    On ``terminate_all()`` we send SIGTERM to each process group and, after a
    short grace, SIGKILL to anything still alive. ``Future.cancel()`` is a
    best-effort hint -- only effective for futures that haven't started yet;
    running ones must be terminated by killing their underlying ``Popen``.
    """

    popens: list[subprocess.Popen] = field(default_factory=list)
    mp_procs: list = field(default_factory=list)  # multiprocessing.Process; target MUST call os.setsid()
    futures: list[Future] = field(default_factory=list)
    # ``RLock`` (not plain ``Lock``) so the ``register_future`` done-callback
    # can re-acquire the lock from the submitting thread when a future is
    # already done at registration time (the synchronous-callback case in
    # ``Future.add_done_callback``). Plain Lock would deadlock there.
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ----- tracking context managers -----

    @contextlib.contextmanager
    def track(self, popen: subprocess.Popen):
        with self.lock:
            self.popens.append(popen)
        try:
            yield popen
        finally:
            with self.lock:
                if popen in self.popens:
                    self.popens.remove(popen)

    @contextlib.contextmanager
    def track_mp(self, proc):
        with self.lock:
            self.mp_procs.append(proc)
        try:
            yield proc
        finally:
            with self.lock:
                if proc in self.mp_procs:
                    self.mp_procs.remove(proc)

    @contextlib.contextmanager
    def track_future(self, fut: Future):
        with self.lock:
            self.futures.append(fut)
        try:
            yield fut
        finally:
            with self.lock:
                if fut in self.futures:
                    self.futures.remove(fut)

    def register_future(self, fut: Future) -> None:
        """Add ``fut`` to the registry and arrange for it to be auto-removed
        on completion via ``Future.add_done_callback``.

        This is the right primitive for fire-and-forget submission from a
        dispatcher loop: the caller holds ``self.lock`` across the
        (soft_stop check + executor.submit + register_future) critical
        section to close the submit-and-track race, and the cleanup happens
        automatically when the worker thread completes (no need to remember
        to call ``futures.remove`` from the caller).

        Without auto-cleanup, completed futures linger in ``self.futures``
        and ``terminate_all`` reports misleading "futures=N" counts in its
        SIGTERM-wave log line.

        Ordering note: the append must precede ``add_done_callback`` so that
        the synchronous-callback case (future already done at registration)
        observes the future *in* ``self.futures`` and removes it. Reversing
        the order would leak entries in that path. Concurrent ``terminate_all``
        callers cannot observe the intermediate state because callers of
        ``register_future`` hold ``self.lock`` across submit-and-register
        per the docstring; the ``RLock`` makes the synchronous-callback
        re-entry safe.
        """
        self.futures.append(fut)
        fut.add_done_callback(self._on_future_done)

    def _on_future_done(self, fut: Future) -> None:
        """Remove a completed future from the registry. Safe to call from any
        thread (including the submitting thread, when the callback fires
        synchronously due to the future being already done at registration).
        """
        with self.lock:
            if fut in self.futures:
                self.futures.remove(fut)

    # ----- termination -----

    def _killpg(self, pid: int, sig: int, label: str) -> None:
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return
        except OSError as e:
            logger.warning("getpgid failed (%s, pid=%s): %s", label, pid, e)
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except OSError as e:
            logger.warning("killpg failed (%s, pid=%s, sig=%s): %s", label, pid, sig, e)

    def terminate_all(self, escalate_after_s: float = 5.0) -> None:
        """SIGTERM the world; wait up to ``escalate_after_s``; SIGKILL holdouts.

        The two-stage escalation is necessary because some benchmark wrappers
        and tools like ``ncu`` install their own SIGTERM handlers and ignore
        the first wave. 5s out of (typically) 300s of finalize grace is cheap
        insurance. ``Future.cancel()`` is called once -- it only affects
        not-yet-started futures; a running future's work dies via its
        underlying Popen.
        """
        with self.lock:
            popens = list(self.popens)
            mp_procs = list(self.mp_procs)
            futures = list(self.futures)

        if not popens and not mp_procs and not futures:
            return

        logger.info(
            "ProcessRegistry.terminate_all: SIGTERM wave (popens=%d, mp_procs=%d, futures=%d)",
            len(popens),
            len(mp_procs),
            len(futures),
        )

        # First wave: SIGTERM to the whole process group (to also catch GPU
        # benchmark grandchildren).
        for p in popens:
            self._killpg(p.pid, signal.SIGTERM, "Popen")
        for mp in mp_procs:
            try:
                if mp.is_alive():
                    self._killpg(mp.pid, signal.SIGTERM, "mp.Process")
            except Exception:
                logger.exception("checking mp.Process state failed")
        for f in futures:
            try:
                if not f.done():
                    f.cancel()
            except Exception:
                logger.exception("future.cancel() failed")

        # Grace period: poll for natural exit.
        deadline = time.monotonic() + escalate_after_s
        while time.monotonic() < deadline:
            still_alive = False
            for p in popens:
                try:
                    if p.poll() is None:
                        still_alive = True
                        break
                except Exception:
                    logger.exception("Popen.poll() raised")
            if not still_alive:
                for mp in mp_procs:
                    try:
                        if mp.is_alive():
                            still_alive = True
                            break
                    except Exception:
                        logger.exception("mp.Process.is_alive() raised")
            if not still_alive:
                logger.debug("ProcessRegistry.terminate_all: all targets exited gracefully")
                return
            time.sleep(0.1)

        # Second wave: SIGKILL the holdouts.
        killed = 0
        for p in popens:
            try:
                if p.poll() is None:
                    self._killpg(p.pid, signal.SIGKILL, "Popen")
                    killed += 1
            except Exception:
                logger.exception("Popen escalation to SIGKILL failed")
        for mp in mp_procs:
            try:
                if mp.is_alive():
                    self._killpg(mp.pid, signal.SIGKILL, "mp.Process")
                    killed += 1
            except Exception:
                logger.exception("mp.Process escalation to SIGKILL failed")
        if killed:
            logger.warning(
                "ProcessRegistry.terminate_all: escalated to SIGKILL for %d holdout(s)",
                killed,
            )


# ---------------------------------------------------------------------------
# PreprocessState
# ---------------------------------------------------------------------------


@dataclass
class PreprocessState:
    """Per-run state for the preprocess phase.

    Note: no explicit lock. ``current_stage`` and the bool flags are written
    only by the preprocessor (single thread) and the watchdog handler, both
    of which use atomic attribute writes (CPython guarantee). Multi-step
    state mutation lives in ``self.registry`` which has its own lock.
    """

    registry: ProcessRegistry = field(default_factory=ProcessRegistry)
    current_stage: PreprocessStage = PreprocessStage.HARNESS_INIT
    in_borrow_mode: bool = False
    skip_profiling: bool = False
    hard_fail: bool = False
    fail_reason: str | None = None
    output_dir: Path | None = None

    # ----- artifact-presence checks (used by soft-stop handler) -----

    def has_baseline_file(self) -> bool:
        if self.output_dir is None:
            return False
        return (self.output_dir / "benchmark_baseline.txt").exists()

    def has_commandment_file(self) -> bool:
        if self.output_dir is None:
            return False
        return (self.output_dir / "COMMANDMENT.md").exists()

    # ----- stage transitions -----

    @contextlib.contextmanager
    def enter(self, stage: PreprocessStage):
        """Mark a stage as current. Raises ``PreprocessAborted`` if a hard
        stop has fired before we even entered the stage.
        """
        self.set_stage(stage)
        try:
            yield
        finally:
            # We don't auto-advance current_stage on exit; the next enter()
            # will overwrite it. Leaving it as the last stage entered is
            # correct for the soft-stop handler if a stage raises.
            pass

    def set_stage(self, stage: PreprocessStage) -> None:
        """Bundle the "advance current_stage + raise on hard_fail" guard.

        Use this when a ``with state.enter(...)`` context isn't ergonomic
        (e.g. the stage spans a top-level ``if/elif`` chain that wouldn't
        nest cleanly in a context manager). It preserves the same hard-cap
        invariant: once ``state.hard_fail`` is set by the watchdog, any
        further stage transition aborts the preprocess pipeline.
        """
        if self.hard_fail:
            raise PreprocessAborted(self.fail_reason or "preprocess aborted by watchdog")
        self.current_stage = stage
        logger.debug("PreprocessState entered stage: %s", stage.value)

    def mark_done(self) -> None:
        self.current_stage = PreprocessStage.DONE


class PreprocessAborted(RuntimeError):
    """Raised when a stage is entered after the preprocess hard cap fired."""


# ---------------------------------------------------------------------------
# Soft / hard stop handlers (used as RunBudget watchdog callbacks)
# ---------------------------------------------------------------------------


def preprocess_soft_stop_handler(
    state: PreprocessState,
    *,
    soft_cap_s: float,
    hard_cap_s: float,
    console=None,
) -> None:
    """Apply the stage-aware soft-stop policy when the preprocess soft cap fires.

    See ``Preprocess fallback policy`` in the plan. Stage detection uses
    ``state.current_stage`` plus a quick artifact-presence check on
    ``output_dir``. Effects are flag mutations (``skip_profiling`` /
    ``hard_fail`` / ``in_borrow_mode``); the preprocessor itself is responsible
    for honoring those flags at the next stage boundary. We additionally
    terminate the in-flight profiler ``mp.Process`` directly when stuck in
    ``KERNEL_PROFILE`` so the borrowed time is spent on commandment generation
    rather than waiting for ncu to time out.
    """
    state.in_borrow_mode = True
    borrow_max_s = hard_cap_s - soft_cap_s
    new_opt_min = "(unknown)"  # caller can compute total - hard_cap if it cares to print

    stage = state.current_stage
    has_baseline = state.has_baseline_file()
    banner = (
        f"[preprocess] Original budget ({soft_cap_s:.0f}s) reached at stage '{stage.value}'.\n"
        f"[preprocess] Borrowing up to {borrow_max_s:.0f}s from optimization budget.\n"
        f"[preprocess] Optimization budget will shrink by however much preprocess overruns "
        f"(hard cap = {hard_cap_s:.0f}s; opt minimum = {new_opt_min})."
    )
    logger.warning(banner)
    if console is not None:
        try:
            console.print(f"[bold yellow]{banner}[/bold yellow]")
        except Exception:
            logger.exception("console.print of borrow banner failed (non-fatal)")

    # ---- stage classifier ----
    if stage == PreprocessStage.HARNESS_INIT and not has_baseline:
        # Hard fail: harness setup is broken; nothing useful to optimize.
        state.hard_fail = True
        state.fail_reason = (
            f"preprocess soft cap ({soft_cap_s:.0f}s) reached during '{stage.value}' "
            f"with no benchmark_baseline.txt produced; harness setup did not progress in time"
        )
        logger.error("[preprocess] %s -- terminating registry and aborting run", state.fail_reason)
        state.registry.terminate_all()
        return

    if stage == PreprocessStage.HARNESS_BENCHMARK:
        # Warn-and-continue, AND skip profiling. If benchmark mode is slow
        # enough to hit the soft cap, profiling (which is typically slower)
        # would blow the hard ceiling without producing any mandatory output.
        state.skip_profiling = True
        logger.warning("[preprocess] benchmark mode in flight; letting it complete and skipping profiling")
        return

    if stage == PreprocessStage.KERNEL_PROFILE:
        # Skip profiling: terminate the profiler subprocess directly so we
        # stop wasting borrowed time on it. baseline-metrics will fall back
        # to a thin metrics dict from benchmark_baseline.txt alone.
        state.skip_profiling = True
        logger.warning("[preprocess] profiler in flight; terminating mp.Process(es) and skipping profiling")
        # Kill *just* the mp_procs slice (don't terminate the whole registry,
        # which would also nuke the harness benchmark Popen if it's somehow
        # still around).
        with state.registry.lock:
            mp_procs = list(state.registry.mp_procs)
        for proc in mp_procs:
            try:
                if proc.is_alive():
                    state.registry._killpg(proc.pid, signal.SIGTERM, "mp.Process(profiler)")
            except Exception:
                logger.exception("terminating profiler mp.Process failed (non-fatal)")
        return

    # BASELINE_METRICS / COMMANDMENT / DONE: warn and continue. baseline-metrics
    # is fast so will likely finish in seconds; commandment is mandatory and
    # the only sensible action is to let it complete in borrowed time.
    logger.warning(
        "[preprocess] stage '%s' is in flight; letting it finish in borrowed time",
        stage.value,
    )


def preprocess_hard_stop_handler(
    state: PreprocessState,
    *,
    hard_cap_s: float,
    console=None,
) -> None:
    """Apply the preprocess hard-cap policy: terminate everything and fail."""
    state.hard_fail = True
    state.fail_reason = (
        f"preprocess hard cap ({hard_cap_s:.0f}s) exceeded at stage '{state.current_stage.value}' "
        f"(commandment_present={state.has_commandment_file()})"
    )
    logger.error("[preprocess] %s -- terminating registry", state.fail_reason)
    if console is not None:
        try:
            console.print("[bold red][preprocess] HARD CAP EXCEEDED -- aborting run.[/bold red]")
        except Exception:
            logger.exception("console.print of hard-cap banner failed (non-fatal)")
    state.registry.terminate_all()


# ---------------------------------------------------------------------------
# Thin baseline_metrics fallback (when profiling is skipped)
# ---------------------------------------------------------------------------


def build_thin_baseline_metrics(benchmark_baseline_text: str) -> dict[str, Any]:
    """Produce a minimal baseline_metrics dict from benchmark_baseline.txt alone.

    Used when ``skip_profiling`` is set and the profiler stage is bypassed.
    The orchestrator can still run with this; quality of optimization
    guidance just degrades (no ``bottleneck`` field).
    """
    from minisweagent.run.preprocess.benchmark_parsing import extract_latency_ms

    metrics: dict[str, Any] = {
        "bottleneck": "unknown",
        "profiling_skipped": True,
    }
    if benchmark_baseline_text:
        latency_ms = extract_latency_ms(benchmark_baseline_text)
        if latency_ms is not None:
            metrics["benchmark_duration_us"] = latency_ms * 1000.0
            metrics["duration_us"] = latency_ms * 1000.0
    return metrics
