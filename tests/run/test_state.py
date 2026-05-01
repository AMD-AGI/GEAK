"""Unit tests for ``run/state.py``: ProcessRegistry + soft/hard stop handlers."""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from minisweagent.run.state import (
    PreprocessStage,
    PreprocessState,
    ProcessRegistry,
    build_thin_baseline_metrics,
    preprocess_hard_stop_handler,
    preprocess_soft_stop_handler,
)

# ---------------------------------------------------------------------------
# ProcessRegistry: SIGTERM -> SIGKILL escalation kills a stub group
# ---------------------------------------------------------------------------


def _spawn_sleeper(seconds: int = 60) -> subprocess.Popen:
    """Spawn a Python ``time.sleep`` subprocess in its own session."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        start_new_session=True,
    )


def test_terminate_all_kills_a_tracked_popen():
    registry = ProcessRegistry()
    proc = _spawn_sleeper(60)
    with registry.track(proc):
        # Tracked while it sits inside the with-block.
        assert proc in registry.popens
        registry.terminate_all(escalate_after_s=2.0)
        # Should have died via SIGTERM long before 2s escalation.
        assert proc.poll() is not None, "subprocess should be dead after terminate_all"


def test_terminate_all_escalates_to_sigkill_for_holdouts():
    """A child that traps SIGTERM should still die via SIGKILL escalation."""
    # Spawn a process that ignores SIGTERM (signal.SIG_IGN). The escalation
    # path should SIGKILL it after escalate_after_s.
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    registry = ProcessRegistry()
    with registry.track(proc):
        t0 = time.monotonic()
        registry.terminate_all(escalate_after_s=0.5)
        elapsed = time.monotonic() - t0
        # SIGKILL shouldn't be ignorable; the proc must be dead.
        assert proc.poll() is not None, "SIGTERM-trapping child should be SIGKILLed"
        # Total wait should be roughly escalate_after_s + small overhead.
        assert elapsed < 5.0


def test_track_removes_handle_on_exit():
    registry = ProcessRegistry()
    proc = _spawn_sleeper(2)
    try:
        with registry.track(proc):
            assert proc in registry.popens
        assert proc not in registry.popens
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass


def test_terminate_all_handles_already_dead_subprocess():
    registry = ProcessRegistry()
    proc = _spawn_sleeper(0)
    proc.wait(timeout=2)
    with registry.track(proc):
        # Should not raise even though pid is already gone.
        registry.terminate_all(escalate_after_s=0.5)


# ---------------------------------------------------------------------------
# register_future: auto-cleanup on completion (no stale "futures=N" in logs)
# ---------------------------------------------------------------------------


def test_register_future_auto_removes_on_completion():
    """After the future completes, it should no longer be in registry.futures.

    This is the cleanliness fix that stops ``terminate_all``'s SIGTERM-wave
    log line from misleadingly reporting "futures=N" for already-completed
    workers when the run finalizes naturally.
    """
    import concurrent.futures

    registry = ProcessRegistry()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        with registry.lock:
            fut = executor.submit(lambda: 42)
            registry.register_future(fut)
        # Future is in the registry while alive (or already removed if it
        # finished synchronously between submit and register; either is
        # acceptable as long as it's gone post-completion).
        result = fut.result(timeout=5)
        assert result == 42
        # add_done_callback may have fired in the executor thread; give it a
        # moment then assert the future is no longer tracked.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with registry.lock:
                if fut not in registry.futures:
                    break
            time.sleep(0.02)
        with registry.lock:
            assert fut not in registry.futures, (
                "register_future must auto-remove a completed future via add_done_callback"
            )


def test_register_future_handles_already_done_future():
    """If the future is already done at registration time (rare but possible
    for very fast tasks), the callback fires synchronously in the calling
    thread. Because ``ProcessRegistry.lock`` is now an ``RLock``, this must
    not deadlock when the caller is already inside ``with registry.lock:``.
    """
    import concurrent.futures

    registry = ProcessRegistry()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(lambda: "done")
        # Deliberately wait for completion BEFORE registering, so the
        # done-callback is invoked synchronously inside register_future.
        fut.result(timeout=5)

        with registry.lock:
            registry.register_future(fut)  # must not deadlock
        # The synchronous callback ran while we held the lock; the future
        # should already be gone (or we re-acquire the lock and verify).
        with registry.lock:
            assert fut not in registry.futures


def test_terminate_all_after_all_futures_done_reports_zero():
    """Integration: register a future, let it complete, then call
    ``terminate_all``. The SIGTERM wave should see ``futures=0`` (the whole
    point of the cleanliness fix).
    """
    import concurrent.futures
    import logging

    registry = ProcessRegistry()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with registry.lock:
            fut = executor.submit(lambda: None)
            registry.register_future(fut)
        fut.result(timeout=5)
        # Allow the done-callback to run.
        time.sleep(0.1)

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=logging.DEBUG)
    state_logger = logging.getLogger("minisweagent.run.state")
    state_logger.addHandler(handler)
    try:
        registry.terminate_all(escalate_after_s=0.1)
        # No SIGTERM-wave log emitted because we have nothing to terminate.
        assert not any("SIGTERM wave" in r.getMessage() for r in records), (
            "terminate_all should be a no-op when registry is fully cleaned up"
        )
    finally:
        state_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Stage classifier: preprocess_soft_stop_handler
# ---------------------------------------------------------------------------


def test_soft_stop_hard_fail_when_harness_init_without_baseline_file(tmp_path):
    state = PreprocessState(output_dir=tmp_path)
    state.current_stage = PreprocessStage.HARNESS_INIT
    preprocess_soft_stop_handler(state, soft_cap_s=900, hard_cap_s=1800)
    assert state.hard_fail is True
    assert state.fail_reason and "harness setup" in state.fail_reason.lower()


def test_soft_stop_warn_and_skip_profiling_when_harness_benchmark(tmp_path):
    state = PreprocessState(output_dir=tmp_path)
    state.current_stage = PreprocessStage.HARNESS_BENCHMARK
    preprocess_soft_stop_handler(state, soft_cap_s=900, hard_cap_s=1800)
    assert state.hard_fail is False
    assert state.skip_profiling is True
    assert state.in_borrow_mode is True


def test_soft_stop_skip_profiling_when_kernel_profile(tmp_path, monkeypatch):
    state = PreprocessState(output_dir=tmp_path)
    state.current_stage = PreprocessStage.KERNEL_PROFILE

    # Stand-in for an mp.Process. pid is intentionally a sentinel; we
    # monkeypatch _killpg below so we don't accidentally signal anything
    # real (pid=0 in os.getpgid() returns the current process group).
    class FakeMp:
        def __init__(self):
            self.pid = 999_999  # any value -- _killpg is mocked
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

    fake = FakeMp()
    state.registry.mp_procs.append(fake)

    kill_calls: list[tuple[int, int, str]] = []

    def _fake_killpg(pid, sig, label):  # noqa: ANN001
        kill_calls.append((pid, sig, label))
        fake._alive = False

    monkeypatch.setattr(state.registry, "_killpg", _fake_killpg)

    preprocess_soft_stop_handler(state, soft_cap_s=900, hard_cap_s=1800)

    assert state.skip_profiling is True
    assert state.in_borrow_mode is True
    assert state.hard_fail is False
    assert kill_calls and kill_calls[0][0] == fake.pid, "killpg should target the profiler mp.Process"


def test_soft_stop_warn_and_continue_for_baseline_metrics(tmp_path):
    state = PreprocessState(output_dir=tmp_path)
    (tmp_path / "benchmark_baseline.txt").write_text("3.5 ms\n10 shapes")
    state.current_stage = PreprocessStage.BASELINE_METRICS
    preprocess_soft_stop_handler(state, soft_cap_s=900, hard_cap_s=1800)
    assert state.hard_fail is False
    assert state.skip_profiling is False
    assert state.in_borrow_mode is True


def test_soft_stop_warn_and_continue_for_commandment(tmp_path):
    state = PreprocessState(output_dir=tmp_path)
    (tmp_path / "benchmark_baseline.txt").write_text("3.5 ms")
    state.current_stage = PreprocessStage.COMMANDMENT
    preprocess_soft_stop_handler(state, soft_cap_s=900, hard_cap_s=1800)
    assert state.hard_fail is False
    assert state.skip_profiling is False


def test_hard_stop_marks_state_and_terminates(tmp_path):
    state = PreprocessState(output_dir=tmp_path)
    state.current_stage = PreprocessStage.KERNEL_PROFILE
    preprocess_hard_stop_handler(state, hard_cap_s=1800)
    assert state.hard_fail is True
    assert "preprocess hard cap" in state.fail_reason.lower()


# ---------------------------------------------------------------------------
# Thin baseline_metrics fallback
# ---------------------------------------------------------------------------


def test_build_thin_baseline_metrics_extracts_latency():
    text = "Average latency: 3.5 ms over 30 iterations"
    metrics = build_thin_baseline_metrics(text)
    assert metrics["bottleneck"] == "unknown"
    assert metrics["profiling_skipped"] is True
    # 3.5 ms = 3500 us
    assert metrics["benchmark_duration_us"] == pytest.approx(3500.0, abs=1e-3)
    assert metrics["duration_us"] == pytest.approx(3500.0, abs=1e-3)


def test_build_thin_baseline_metrics_handles_empty():
    metrics = build_thin_baseline_metrics("")
    assert metrics["bottleneck"] == "unknown"
    assert metrics["profiling_skipped"] is True
    assert "duration_us" not in metrics


# ---------------------------------------------------------------------------
# PreprocessState.enter raises after hard_fail
# ---------------------------------------------------------------------------


def test_preprocess_state_enter_raises_after_hard_fail():
    from minisweagent.run.state import PreprocessAborted

    state = PreprocessState()
    state.hard_fail = True
    state.fail_reason = "synthetic"
    with pytest.raises(PreprocessAborted, match="synthetic"):
        with state.enter(PreprocessStage.COMMANDMENT):
            pass


def test_preprocess_state_enter_updates_current_stage():
    state = PreprocessState()
    assert state.current_stage == PreprocessStage.HARNESS_INIT
    with state.enter(PreprocessStage.COMMANDMENT):
        assert state.current_stage == PreprocessStage.COMMANDMENT
