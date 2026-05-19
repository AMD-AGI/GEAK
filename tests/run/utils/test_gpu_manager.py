"""Tests for the centralized GPU manager."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from minisweagent.run.utils.gpu_manager import GpuJob, GPUManager


@pytest.fixture
def manager_2gpu():
    mgr = GPUManager([0, 1], stats_log_interval_s=0)
    yield mgr
    mgr.shutdown()


@pytest.fixture
def manager_4gpu():
    mgr = GPUManager([0, 1, 2, 3], stats_log_interval_s=0)
    yield mgr
    mgr.shutdown()


class TestBasicScheduling:
    def test_single_job_runs_and_returns(self, manager_2gpu):
        result = manager_2gpu.run(GpuJob(fn=lambda gpus, env: ("ok", gpus), label="test"))
        assert result[0] == "ok"
        assert len(result[1]) == 1
        assert result[1][0] in {0, 1}

    def test_env_overrides_contain_expected_keys(self, manager_2gpu):
        captured = {}

        def _fn(gpus, env):
            captured.update(env)
            return True

        manager_2gpu.run(GpuJob(fn=_fn, label="env-check"))
        assert "HIP_VISIBLE_DEVICES" in captured
        assert "CUDA_VISIBLE_DEVICES" in captured
        assert "GEAK_GPU_DEVICE" in captured

    def test_many_jobs_no_gpu_collision(self, manager_2gpu):
        active_gpus: set[int] = set()
        collision = threading.Event()
        lock = threading.Lock()

        def _fn(gpus, env):
            gpu = gpus[0]
            with lock:
                if gpu in active_gpus:
                    collision.set()
                active_gpus.add(gpu)
            time.sleep(0.05)
            with lock:
                active_gpus.discard(gpu)
            return gpu

        futs = [manager_2gpu.submit(GpuJob(fn=_fn, label=f"j{i}")) for i in range(8)]
        results = [f.result(timeout=10) for f in futs]
        assert not collision.is_set(), "Two concurrent jobs shared a GPU"
        assert len(results) == 8
        assert set(results).issubset({0, 1})

    def test_all_gpus_used(self, manager_2gpu):
        used = set()
        lock = threading.Lock()

        def _fn(gpus, env):
            with lock:
                used.update(gpus)
            time.sleep(0.02)
            return True

        futs = [manager_2gpu.submit(GpuJob(fn=_fn, label=f"j{i}")) for i in range(8)]
        for f in futs:
            f.result(timeout=10)
        assert used == {0, 1}


class TestMultiGpuAcquire:
    def test_multi_gpu_job_gets_correct_count(self, manager_4gpu):
        result = manager_4gpu.run(GpuJob(fn=lambda gpus, env: len(gpus), num_gpus=2, label="multi"))
        assert result == 2

    def test_multi_gpu_mixed_with_single(self, manager_4gpu):
        results = []
        lock = threading.Lock()

        def _fn(gpus, env):
            time.sleep(0.03)
            with lock:
                results.append(len(gpus))
            return len(gpus)

        futs = []
        futs.append(manager_4gpu.submit(GpuJob(fn=_fn, num_gpus=2, label="multi")))
        for i in range(4):
            futs.append(manager_4gpu.submit(GpuJob(fn=_fn, num_gpus=1, label=f"s{i}")))
        for f in futs:
            f.result(timeout=10)
        assert 2 in results
        assert results.count(1) == 4

    def test_num_gpus_exceeds_pool_raises(self, manager_2gpu):
        with pytest.raises(ValueError, match="requests 3 GPU"):
            manager_2gpu.submit(GpuJob(fn=lambda g, e: None, num_gpus=3, label="too-many"))


class TestShutdown:
    def test_submit_after_shutdown_raises(self, manager_2gpu):
        manager_2gpu.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            manager_2gpu.submit(GpuJob(fn=lambda g, e: None, label="late"))

    def test_shutdown_cancel_pending(self):
        mgr = GPUManager([0], stats_log_interval_s=0)
        blocker = threading.Event()

        def _block(gpus, env):
            blocker.wait(timeout=5)
            return "done"

        fut_blocking = mgr.submit(GpuJob(fn=_block, label="blocker"))
        time.sleep(0.1)
        for i in range(3):
            mgr.submit(GpuJob(fn=lambda g, e: "pending", label=f"p{i}"))
        blocker.set()
        mgr.shutdown(cancel_pending=True)
        assert fut_blocking.result(timeout=5) == "done"


class TestStats:
    def test_stats_fields(self, manager_2gpu):
        manager_2gpu.run(GpuJob(fn=lambda g, e: time.sleep(0.05), label="stat"))
        s = manager_2gpu.stats
        assert "busy_seconds_per_gpu" in s
        assert "queue_depth" in s
        assert "in_flight" in s
        assert "max_queue_depth" in s
        assert "total_jobs_submitted" in s
        assert "total_jobs_completed" in s
        assert "manager_uptime_s" in s
        assert s["total_jobs_submitted"] == 1
        assert s["total_jobs_completed"] == 1
        assert s["in_flight"] == 0
        assert s["queue_depth"] == 0

    def test_busy_seconds_accumulate(self, manager_2gpu):
        manager_2gpu.run(GpuJob(fn=lambda g, e: time.sleep(0.1), label="busy"))
        s = manager_2gpu.stats
        total_busy = sum(s["busy_seconds_per_gpu"].values())
        assert total_busy >= 0.08


class TestJobExceptions:
    def test_job_exception_propagated(self, manager_2gpu):
        def _fail(gpus, env):
            raise ValueError("test error")

        fut = manager_2gpu.submit(GpuJob(fn=_fail, label="fail"))
        with pytest.raises(ValueError, match="test error"):
            fut.result(timeout=5)

    def test_gpu_released_after_exception(self, manager_2gpu):
        def _fail(gpus, env):
            raise RuntimeError("boom")

        fut = manager_2gpu.submit(GpuJob(fn=_fail, label="fail"))
        with pytest.raises(RuntimeError):
            fut.result(timeout=5)

        result = manager_2gpu.run(GpuJob(fn=lambda g, e: "ok", label="after-fail"))
        assert result == "ok"


class TestRegistryIntegration:
    def test_futures_registered_with_registry(self):
        registry = MagicMock()
        registry.lock = threading.RLock()
        mgr = GPUManager([0], registry=registry, stats_log_interval_s=0)
        mgr.run(GpuJob(fn=lambda g, e: "ok", label="reg-test"))
        assert registry.register_future.called
        mgr.shutdown()


class TestEmptyOrInvalid:
    def test_empty_gpu_ids_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            GPUManager([])
