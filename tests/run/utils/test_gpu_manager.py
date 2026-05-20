"""Tests for the centralized GPU manager."""

from __future__ import annotations

import json
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.utils.gpu_manager import GpuJob, GpuLease, GPUManager, LeaseState


@pytest.fixture
def manager_2gpu():
    mgr = GPUManager([0, 1], stats_log_interval_s=0, reaper_interval_s=0)
    yield mgr
    mgr.shutdown()


@pytest.fixture
def manager_4gpu():
    mgr = GPUManager([0, 1, 2, 3], stats_log_interval_s=0, reaper_interval_s=0)
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


class TestCancelledFuture:
    """B1: _run_job skips execution when future is already cancelled."""

    def test_cancelled_future_skips_job_fn(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)
        executed = threading.Event()

        def _fn(gpus, env):
            executed.set()
            return "ran"

        blocker = threading.Event()
        mgr.submit(GpuJob(fn=lambda g, e: blocker.wait(timeout=5), label="blocker"))
        time.sleep(0.05)

        fut = mgr.submit(GpuJob(fn=_fn, label="to-cancel"))
        time.sleep(0.05)
        fut.cancel()
        blocker.set()
        time.sleep(0.3)

        assert not executed.is_set(), "Job ran despite future being cancelled"
        mgr.shutdown()


class TestDispatcherDrain:
    """B9: dispatcher drains remaining jobs on every exit path."""

    def test_closed_return_drains_queue(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)
        blocker = threading.Event()

        mgr.submit(GpuJob(fn=lambda g, e: blocker.wait(timeout=5), label="blocker"))
        time.sleep(0.05)

        pending_futs = [
            mgr.submit(GpuJob(fn=lambda g, e: "should-not-run", label=f"p{i}"))
            for i in range(3)
        ]

        blocker.set()
        mgr.shutdown(cancel_pending=True)

        for fut in pending_futs:
            assert fut.cancelled() or fut.done()


class TestSplitTimeout:
    """B2: queue_timeout is used for Future.result(), timeout sets lease deadline."""

    def test_queue_timeout_none_waits_forever(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)
        blocker = threading.Event()

        mgr.submit(GpuJob(fn=lambda g, e: blocker.wait(timeout=5), label="blocker"))
        time.sleep(0.05)

        result_holder = []

        def _run_in_thread():
            try:
                r = mgr.run(GpuJob(
                    fn=lambda g, e: "queued-result",
                    timeout=5.0,
                    queue_timeout=None,
                    label="waiter",
                ))
                result_holder.append(r)
            except Exception as exc:
                result_holder.append(exc)

        t = threading.Thread(target=_run_in_thread)
        t.start()
        time.sleep(0.2)
        blocker.set()
        t.join(timeout=10)
        mgr.shutdown()

        assert result_holder and result_holder[0] == "queued-result"

    def test_execution_timeout_sets_lease_deadline(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)
        mgr.run(GpuJob(fn=lambda g, e: "ok", timeout=60.0, label="with-deadline"))

        with mgr._leases_lock:
            leases = list(mgr._active_leases.values())
        assert len(leases) == 1
        assert leases[0].lease.deadline is not None
        assert leases[0].release_reason == "completed"
        mgr.shutdown()


class TestLeaseRelease:
    """B3: double release is detected and rejected."""

    def test_double_release_rejected(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)
        mgr.run(GpuJob(fn=lambda g, e: "ok", label="lease-test"))

        with mgr._leases_lock:
            lease_id = list(mgr._active_leases.keys())[0]

        stats_before = mgr.stats
        mgr._release_lease(lease_id, reason="spurious")
        stats_after = mgr.stats

        assert stats_before["total_jobs_completed"] == stats_after["total_jobs_completed"]
        assert stats_before["in_flight"] == stats_after["in_flight"]
        mgr.shutdown()


class TestLeaseReaper:
    """B2/B8: reaper releases GPUs from expired leases."""

    def test_reaper_releases_expired_lease(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=1)
        started = threading.Event()

        def _slow(gpus, env):
            started.set()
            time.sleep(30)
            return "should-be-reaped"

        fut = mgr.submit(GpuJob(fn=_slow, timeout=0.5, label="slow-job"))
        started.wait(timeout=5)
        time.sleep(3)

        assert 0 in mgr._free, "GPU should be returned to pool after reaping"
        mgr.shutdown()

    def test_lease_cleanup_removes_old_entries(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)

        for i in range(5):
            mgr.run(GpuJob(fn=lambda g, e: "ok", label=f"j{i}"))

        with mgr._leases_lock:
            assert len(mgr._active_leases) == 5
            for s in mgr._active_leases.values():
                s.released_at = time.monotonic() - 400

        mgr._reaper_loop_once_for_test()

        with mgr._leases_lock:
            assert len(mgr._active_leases) == 0
        mgr.shutdown()


class TestCPUPressureGate:
    """B7: dispatcher waits when CPU load is high."""

    def test_high_cpu_load_delays_dispatch(self):
        import types
        import minisweagent.run.utils.gpu_manager as gm_mod

        call_count = 0
        real_os = gm_mod.os

        def mock_getloadavg():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return (9999.0, 9999.0, 9999.0)
            return (0.1, 0.1, 0.1)

        fake_os = types.SimpleNamespace(
            getloadavg=mock_getloadavg,
            cpu_count=os.cpu_count,
            getpgid=os.getpgid,
            kill=os.kill,
            killpg=os.killpg,
        )
        gm_mod.os = fake_os
        try:
            mgr = GPUManager(
                [0], stats_log_interval_s=0, reaper_interval_s=0,
                cpu_pressure_threshold=1.0,
            )
            result = mgr.run(GpuJob(fn=lambda g, e: "dispatched", label="cpu-gate"))
            assert result == "dispatched"
            assert call_count >= 3
            mgr.shutdown()
        finally:
            gm_mod.os = real_os

    def test_getloadavg_oserror_skipped(self):
        with patch("minisweagent.run.utils.gpu_manager.os.getloadavg", side_effect=OSError):
            mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)
            result = mgr.run(GpuJob(fn=lambda g, e: "ok", label="no-loadavg"))
            assert result == "ok"
            mgr.shutdown()


class TestEventLogging:
    """Phase 2: JSONL event logging and per-outcome counters."""

    def test_lifecycle_events_written_to_file(self, tmp_path):
        log_file = tmp_path / "events.jsonl"
        mgr = GPUManager(
            [0], stats_log_interval_s=0, reaper_interval_s=0,
            event_log_path=str(log_file),
        )
        mgr.run(GpuJob(fn=lambda g, e: "ok", label="evt-test"))
        mgr.shutdown()

        lines = log_file.read_text().strip().splitlines()
        events = [json.loads(line)["event"] for line in lines]
        assert events == ["queued", "leased", "started", "completed", "released"]

    def test_failed_job_emits_failed_event(self, tmp_path):
        log_file = tmp_path / "events.jsonl"
        mgr = GPUManager(
            [0], stats_log_interval_s=0, reaper_interval_s=0,
            event_log_path=str(log_file),
        )
        fut = mgr.submit(GpuJob(fn=lambda g, e: (_ for _ in ()).throw(ValueError("boom")), label="fail-evt"))
        with pytest.raises(ValueError):
            fut.result(timeout=5)
        mgr.shutdown()

        lines = log_file.read_text().strip().splitlines()
        events = [json.loads(line)["event"] for line in lines]
        assert "failed" in events
        assert "released" in events

    def test_per_outcome_counters(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)
        mgr.run(GpuJob(fn=lambda g, e: "ok", label="s1"))
        mgr.run(GpuJob(fn=lambda g, e: "ok", label="s2"))

        def _fail(g, e):
            raise RuntimeError("err")

        fut = mgr.submit(GpuJob(fn=_fail, label="f1"))
        with pytest.raises(RuntimeError):
            fut.result(timeout=5)

        s = mgr.stats
        assert s["total_succeeded"] == 2
        assert s["total_failed"] == 1
        assert s["total_reaped"] == 0
        assert s["total_cancelled"] == 0
        assert s["total_jobs_completed"] == 3
        mgr.shutdown()

    def test_event_log_fields(self, tmp_path):
        log_file = tmp_path / "events.jsonl"
        mgr = GPUManager(
            [0], stats_log_interval_s=0, reaper_interval_s=0,
            event_log_path=str(log_file),
        )
        mgr.run(GpuJob(fn=lambda g, e: "ok", label="field-test"))
        mgr.shutdown()

        lines = log_file.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]

        leased = next(r for r in records if r["event"] == "leased")
        assert "lease_id" in leased
        assert "gpu_ids" in leased
        assert leased["job_label"] == "field-test"

        released = next(r for r in records if r["event"] == "released")
        assert released["reason"] == "completed"
        assert "t" in released

    def test_no_event_log_file_still_works(self):
        mgr = GPUManager([0], stats_log_interval_s=0, reaper_interval_s=0)
        mgr.run(GpuJob(fn=lambda g, e: "ok", label="no-file"))
        s = mgr.stats
        assert s["total_succeeded"] == 1
        mgr.shutdown()
