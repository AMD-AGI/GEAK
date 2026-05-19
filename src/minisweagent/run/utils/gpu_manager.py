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
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class GpuJob(Generic[T]):
    """A unit of GPU-bound work submitted to the manager."""

    fn: Callable[[list[int], dict[str, str]], T]
    num_gpus: int = 1
    timeout: float | None = None
    label: str = ""


class GPUManager:
    """Centralized GPU pool with FIFO scheduling.

    Parameters
    ----------
    gpu_ids:
        Physical GPU device IDs available (e.g. ``[0, 1, 2, 3]``).
    registry:
        A ``ProcessRegistry`` instance for SIGINT cleanup of futures.
    stats_log_interval_s:
        Seconds between periodic INFO log lines.  ``0`` disables.
    """

    def __init__(
        self,
        gpu_ids: list[int],
        registry: Any = None,
        *,
        stats_log_interval_s: float = 30.0,
    ) -> None:
        if not gpu_ids:
            raise ValueError("gpu_ids must be non-empty")

        self._gpu_ids = list(gpu_ids)
        self._registry = registry
        self._stats_log_interval_s = stats_log_interval_s

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

        self._dispatcher = threading.Thread(target=self._dispatch_loop, name="gpu-manager-dispatch", daemon=True)
        self._dispatcher.start()

        if stats_log_interval_s > 0:
            self._stats_logger = threading.Thread(target=self._stats_log_loop, name="gpu-manager-stats", daemon=True)
            self._stats_logger.start()

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
        self._job_queue.put((job, fut))
        return fut

    def run(self, job: GpuJob) -> Any:
        """Submit a job and block until it completes."""
        return self.submit(job).result(timeout=job.timeout)

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
                "manager_uptime_s": round(now - self._start_time, 2),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        while True:
            item = self._job_queue.get()
            if item is None:
                return
            job, fut = item

            if fut.cancelled():
                with self._stats_lock:
                    self._queue_depth = max(0, self._queue_depth - 1)
                continue

            with self._cond:
                while len(self._free) < job.num_gpus:
                    if self._closed:
                        fut.cancel()
                        with self._stats_lock:
                            self._queue_depth = max(0, self._queue_depth - 1)
                        return
                    self._cond.wait(timeout=1.0)

                assigned = []
                for _ in range(job.num_gpus):
                    assigned.append(self._free.pop())

            now = time.monotonic()
            with self._stats_lock:
                self._queue_depth = max(0, self._queue_depth - 1)
                self._in_flight += 1
                for g in assigned:
                    self._busy_start[g] = now

            env_overrides = self._build_env_overrides(assigned)

            worker_fut = self._executor.submit(self._run_job, job, assigned, env_overrides, fut)
            if self._registry is not None:
                try:
                    with self._registry.lock:
                        self._registry.register_future(worker_fut)
                except Exception:
                    logger.debug("Failed to register worker future with registry", exc_info=True)

    def _run_job(
        self,
        job: GpuJob,
        assigned: list[int],
        env_overrides: dict[str, str],
        fut: Future,
    ) -> None:
        try:
            result = job.fn(assigned, env_overrides)
            fut.set_result(result)
        except Exception as exc:
            fut.set_exception(exc)
        finally:
            self._release_gpus(assigned)

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
            logger.info(
                "gpu_manager.stats: util={%s} queue=%d in_flight=%d jobs=%d/%d uptime=%.0fs",
                util_str,
                s["queue_depth"],
                s["in_flight"],
                s["total_jobs_completed"],
                s["total_jobs_submitted"],
                uptime,
            )
