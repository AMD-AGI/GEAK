"""Centralized in-process GPU manager.

Decouples sub-agent count from GPU count. Every GPU-bound subprocess is
submitted as a ``GpuJob``; the manager holds the GPU only while the job
runs and releases it on completion.

Scheduling: single dispatcher thread + unbounded worker pool.  The
dispatcher pops jobs from a FIFO queue, waits until enough GPUs are free,
then reserves them atomically and hands the job to a worker.  Workers
return GPUs via a done-callback.  This avoids the deadlock a fixed-size
worker pool would create when a multi-GPU job sits in the queue while
every worker holds zero GPUs.

GPU ownership is tracked via leases.  Each lease records which GPUs are
held, when they were acquired, and an optional execution deadline.  A
background reaper thread reclaims GPUs from expired leases and kills the
associated subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class GpuJob(Generic[T]):
    """A unit of GPU-bound work submitted to the manager."""

    fn: Callable[[list[int], dict[str, str]], T]
    num_gpus: int = 1
    timeout: float | None = None
    queue_timeout: float | None = None
    label: str = ""


@dataclass(frozen=True)
class GpuLease:
    """Immutable record of a GPU assignment."""

    lease_id: str
    job_label: str
    gpu_ids: tuple[int, ...]
    acquired_at: float
    deadline: float | None


@dataclass
class LeaseState:
    """Mutable tracking wrapper around a lease."""

    lease: GpuLease
    popen_pid: int | None = None
    released: bool = False
    released_at: float | None = None
    release_reason: str | None = None


class GPUManager:
    """Centralized GPU pool with FIFO scheduling and lease-based ownership.

    Parameters
    ----------
    gpu_ids:
        Physical GPU device IDs available (e.g. ``[0, 1, 2, 3]``).
    registry:
        A ``ProcessRegistry`` instance for SIGINT cleanup of futures.
    stats_log_interval_s:
        Seconds between periodic INFO log lines.  ``0`` disables.
    cpu_pressure_threshold:
        Max **per-core** load average (``loadavg / cpu_count``) before the
        dispatcher pauses.  Defaults to ``0.8`` (i.e. ~80% saturated).
        Internal knob — not surfaced through YAML config on purpose, since
        raw loadavg thresholds are a footgun across heterogeneous hosts.
    reaper_interval_s:
        Seconds between lease-reaper sweeps.  ``0`` disables.
    event_log_path:
        Path to a JSONL file for per-job lifecycle events.
        ``None`` disables file logging (events still go to ``logger.debug``).
    """

    def __init__(
        self,
        gpu_ids: list[int],
        registry: Any = None,
        *,
        stats_log_interval_s: float = 30.0,
        cpu_pressure_threshold: float | None = None,
        reaper_interval_s: float = 30.0,
        event_log_path: str | Path | None = None,
    ) -> None:
        if not gpu_ids:
            raise ValueError("gpu_ids must be non-empty")

        self._gpu_ids = list(gpu_ids)
        self._registry = registry
        self._stats_log_interval_s = stats_log_interval_s
        self._cpu_pressure_threshold = (
            cpu_pressure_threshold if cpu_pressure_threshold is not None else 0.8
        )
        self._reaper_interval_s = reaper_interval_s

        self._free: set[int] = set(gpu_ids)
        self._cond = threading.Condition(threading.Lock())
        self._job_queue: queue.Queue[tuple[GpuJob, Future] | None] = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=None)
        self._closed = False

        self._start_time = time.monotonic()
        self._busy_start: dict[int, float] = {}
        self._busy_total: dict[int, float] = {g: 0.0 for g in gpu_ids}
        self._stats_lock = threading.Lock()
        self._total_submitted = 0
        self._total_completed = 0
        self._in_flight = 0
        self._queue_depth = 0
        self._max_queue_depth = 0

        self._total_succeeded = 0
        self._total_failed = 0
        self._total_reaped = 0
        self._total_cancelled = 0

        self._active_leases: dict[str, LeaseState] = {}
        self._leases_lock = threading.Lock()

        self._event_log_file = open(event_log_path, "a", encoding="utf-8") if event_log_path else None
        self._event_log_lock = threading.Lock()

        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="gpu-manager-dispatch",
            daemon=True,
        )
        self._dispatcher.start()

        if stats_log_interval_s > 0:
            self._stats_logger = threading.Thread(
                target=self._stats_log_loop,
                name="gpu-manager-stats",
                daemon=True,
            )
            self._stats_logger.start()

        if reaper_interval_s > 0:
            self._reaper = threading.Thread(
                target=self._reaper_loop,
                name="gpu-lease-reaper",
                daemon=True,
            )
            self._reaper.start()

    def submit(self, job: GpuJob) -> Future:
        """Enqueue a job and return a ``Future`` for its result."""
        if self._closed:
            raise RuntimeError("GPUManager is shut down; cannot submit new jobs")
        if job.num_gpus > len(self._gpu_ids):
            raise ValueError(
                f"job '{job.label}' requests {job.num_gpus} GPU(s) but only {len(self._gpu_ids)} available"
            )

        fut: Future = Future()
        with self._stats_lock:
            self._total_submitted += 1
            self._queue_depth += 1
            self._max_queue_depth = max(self._max_queue_depth, self._queue_depth)
        self._emit_event("queued", job_label=job.label, num_gpus=job.num_gpus)
        self._job_queue.put((job, fut))
        return fut

    def run(self, job: GpuJob) -> Any:
        """Submit a job and block until it completes."""
        return self.submit(job).result(timeout=job.queue_timeout)

    def shutdown(self, *, cancel_pending: bool = False) -> None:
        """Stop accepting jobs and drain the dispatcher."""
        self._closed = True
        if cancel_pending:
            drained: list[tuple[GpuJob, Future]] = []
            while True:
                try:
                    item = self._job_queue.get_nowait()
                    if item is not None:
                        drained.append(item)
                except queue.Empty:
                    break
            for _job, fut in drained:
                fut.cancel()
                with self._stats_lock:
                    self._queue_depth = max(0, self._queue_depth - 1)

        self._job_queue.put(None)
        self._dispatcher.join(timeout=10)
        self._executor.shutdown(wait=True)
        logger.info("GPUManager shut down. Final stats: %s", self.stats)
        if self._event_log_file is not None:
            self._event_log_file.close()
            self._event_log_file = None

    @property
    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._stats_lock:
            busy = {}
            for g in self._gpu_ids:
                total = self._busy_total[g]
                if g in self._busy_start:
                    total += now - self._busy_start[g]
                busy[g] = round(total, 2)
            return {
                "busy_seconds_per_gpu": busy,
                "queue_depth": self._queue_depth,
                "in_flight": self._in_flight,
                "max_queue_depth": self._max_queue_depth,
                "total_jobs_submitted": self._total_submitted,
                "total_jobs_completed": self._total_completed,
                "total_succeeded": self._total_succeeded,
                "total_failed": self._total_failed,
                "total_reaped": self._total_reaped,
                "total_cancelled": self._total_cancelled,
                "manager_uptime_s": round(now - self._start_time, 2),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        try:
            while True:
                item = self._job_queue.get()
                if item is None:
                    return
                job, fut = item

                if fut.cancelled():
                    with self._stats_lock:
                        self._queue_depth = max(0, self._queue_depth - 1)
                    continue

                # CPU pressure gate — wait before acquiring GPUs.
                # Compares **per-core** load (loadavg / cpu_count) against the
                # threshold so the gate behaves the same on a 4-core dev box
                # and a 256-core server. Raw loadavg is meaningless without
                # normalizing by cpu_count.
                try:
                    cpus = max(1, os.cpu_count() or 1)
                    while (os.getloadavg()[0] / cpus) > self._cpu_pressure_threshold:
                        if self._closed:
                            fut.cancel()
                            return
                        time.sleep(1.0)
                except OSError:
                    pass

                with self._cond:
                    while len(self._free) < job.num_gpus:
                        if self._closed:
                            fut.cancel()
                            with self._stats_lock:
                                self._queue_depth = max(0, self._queue_depth - 1)
                            return
                        self._cond.wait(timeout=1.0)

                    assigned = [self._free.pop() for _ in range(job.num_gpus)]

                now = time.monotonic()
                deadline = (now + job.timeout) if job.timeout else None
                lease = GpuLease(
                    lease_id=str(uuid.uuid4()),
                    job_label=job.label,
                    gpu_ids=tuple(assigned),
                    acquired_at=now,
                    deadline=deadline,
                )
                state = LeaseState(lease=lease)
                with self._leases_lock:
                    self._active_leases[lease.lease_id] = state

                with self._stats_lock:
                    self._queue_depth = max(0, self._queue_depth - 1)
                    self._in_flight += 1
                    for g in assigned:
                        self._busy_start[g] = now

                self._emit_event(
                    "leased",
                    lease_id=lease.lease_id,
                    job_label=job.label,
                    gpu_ids=list(assigned),
                    deadline=deadline,
                )
                env_overrides = self._build_env_overrides(assigned)

                worker_fut = self._executor.submit(
                    self._run_job,
                    job,
                    lease,
                    state,
                    env_overrides,
                    fut,
                )
                if self._registry is not None:
                    try:
                        with self._registry.lock:
                            self._registry.register_future(worker_fut)
                    except Exception:
                        logger.debug("Failed to register worker future with registry", exc_info=True)
        finally:
            # B9 fix: drain remaining jobs on ANY exit path
            while True:
                try:
                    item = self._job_queue.get_nowait()
                    if item is not None:
                        item[1].cancel()
                        with self._stats_lock:
                            self._queue_depth = max(0, self._queue_depth - 1)
                except queue.Empty:
                    break

    def _run_job(
        self,
        job: GpuJob,
        lease: GpuLease,
        state: LeaseState,
        env_overrides: dict[str, str],
        fut: Future,
    ) -> None:
        reason = "completed"
        try:
            if fut.cancelled():
                logger.info("Job %s: cancelled before execution, skipping", job.label)
                reason = "cancelled"
                return
            self._emit_event("started", lease_id=lease.lease_id, job_label=job.label)
            result = job.fn(list(lease.gpu_ids), env_overrides)
            self._emit_event("completed", lease_id=lease.lease_id, job_label=job.label)
            try:
                fut.set_result(result)
            except InvalidStateError:
                logger.warning("Job %s: future already cancelled, discarding result", job.label)
                reason = "cancelled"
        except Exception as exc:
            reason = "failed"
            self._emit_event("failed", lease_id=lease.lease_id, job_label=job.label, error=str(exc))
            try:
                fut.set_exception(exc)
            except InvalidStateError:
                logger.warning("Job %s: future already cancelled, discarding exception", job.label)
                reason = "cancelled"
        finally:
            self._release_lease(lease.lease_id, reason=reason)

    def _release_lease(self, lease_id: str, reason: str) -> None:
        with self._leases_lock:
            state = self._active_leases.get(lease_id)
            if state is None or state.released:
                logger.warning("Lease %s: already released or unknown, skipping", lease_id)
                return
            state.released = True
            state.released_at = time.monotonic()
            state.release_reason = reason
            gpus = list(state.lease.gpu_ids)

        now = time.monotonic()
        with self._stats_lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._total_completed += 1
            if reason == "completed":
                self._total_succeeded += 1
            elif reason == "failed":
                self._total_failed += 1
            elif reason == "reaped":
                self._total_reaped += 1
            elif reason == "cancelled":
                self._total_cancelled += 1
            for g in gpus:
                if g in self._busy_start:
                    self._busy_total[g] += now - self._busy_start.pop(g)

        with self._cond:
            self._free.update(gpus)
            self._cond.notify_all()

        self._emit_event("released", lease_id=lease_id, reason=reason, gpu_ids=gpus)

    # kept for backward compat with tests that call it directly
    def _release_gpus(self, gpus: list[int]) -> None:
        now = time.monotonic()
        with self._stats_lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._total_completed += 1
            for g in gpus:
                if g in self._busy_start:
                    self._busy_total[g] += now - self._busy_start.pop(g)
        with self._cond:
            self._free.update(gpus)
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Lease reaper
    # ------------------------------------------------------------------

    def _reaper_loop(self) -> None:
        while not self._closed:
            time.sleep(self._reaper_interval_s)
            if self._closed:
                return
            now = time.monotonic()

            with self._leases_lock:
                expired = [
                    (lid, s)
                    for lid, s in self._active_leases.items()
                    if not s.released and s.lease.deadline is not None and now > s.lease.deadline
                ]

            for lease_id, state in expired:
                logger.warning(
                    "Lease %s expired (job: %s, ran %.0fs, limit was %.0fs). Killing subprocess and releasing GPUs %s.",
                    lease_id,
                    state.lease.job_label,
                    now - state.lease.acquired_at,
                    state.lease.deadline - state.lease.acquired_at,
                    list(state.lease.gpu_ids),
                )
                self._kill_lease_subprocess(state)
                self._release_lease(lease_id, reason="reaped")

            # Prune old completed leases to bound memory
            with self._leases_lock:
                cutoff = time.monotonic() - 300
                self._active_leases = {
                    k: v
                    for k, v in self._active_leases.items()
                    if not v.released or (v.released_at is not None and v.released_at > cutoff)
                }

    def _reaper_loop_once_for_test(self) -> None:
        """Run a single reaper iteration (for testing)."""
        now = time.monotonic()
        with self._leases_lock:
            expired = [
                (lid, s)
                for lid, s in self._active_leases.items()
                if not s.released and s.lease.deadline is not None and now > s.lease.deadline
            ]
        for lease_id, state in expired:
            self._kill_lease_subprocess(state)
            self._release_lease(lease_id, reason="reaped")
        with self._leases_lock:
            cutoff = time.monotonic() - 300
            self._active_leases = {
                k: v
                for k, v in self._active_leases.items()
                if not v.released or (v.released_at is not None and v.released_at > cutoff)
            }

    def _kill_lease_subprocess(self, state: LeaseState) -> None:
        if state.popen_pid is None:
            return
        try:
            pgid = os.getpgid(state.popen_pid)
        except (ProcessLookupError, OSError):
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(state.popen_pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def _emit_event(self, event: str, **fields: Any) -> None:
        now = time.monotonic()
        record = {"event": event, "t": round(now - self._start_time, 3), **fields}
        logger.debug("gpu_event: %s", record)
        if self._event_log_file is not None:
            line = json.dumps(record, default=str)
            with self._event_log_lock:
                self._event_log_file.write(line + "\n")
                self._event_log_file.flush()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_env_overrides(assigned: list[int]) -> dict[str, str]:
        devs = ",".join(str(g) for g in assigned)
        return {
            "HIP_VISIBLE_DEVICES": devs,
            "CUDA_VISIBLE_DEVICES": devs,
            "GEAK_GPU_DEVICE": devs,
        }

    def _stats_log_loop(self) -> None:
        while not self._closed:
            time.sleep(self._stats_log_interval_s)
            if self._closed:
                return
            s = self.stats
            uptime = s["manager_uptime_s"]
            parts = []
            for g in self._gpu_ids:
                busy = s["busy_seconds_per_gpu"].get(g, 0.0)
                pct = int(100 * busy / uptime) if uptime > 0 else 0
                parts.append(f"{g}:{pct}%")
            util_str = ",".join(parts)
            logger.debug(
                "gpu_manager.stats: util={%s} queue=%d in_flight=%d jobs=%d/%d uptime=%.0fs",
                util_str,
                s["queue_depth"],
                s["in_flight"],
                s["total_jobs_completed"],
                s["total_jobs_submitted"],
                uptime,
            )
