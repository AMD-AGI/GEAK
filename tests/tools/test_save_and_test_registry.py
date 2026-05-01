"""Unit tests for the ``_tracked_subprocess_run`` helper in
``minisweagent.tools.save_and_test``.

The helper is the chokepoint that lets a homogeneous sub-agent's
long-running benchmark be killed by the budget watchdog. We don't run a
full agent here; we just verify the helper's drop-in semantics and that a
tracked Popen ends up registered with a ``ProcessRegistry`` so the
escalation path can reach it.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest

from minisweagent.run.state import ProcessRegistry
from minisweagent.tools.save_and_test import _tracked_subprocess_run


def test_returns_completed_process_with_stdout_stderr():
    """Drop-in replacement for ``subprocess.run`` -- callers depend on the
    ``CompletedProcess`` shape (stdout / stderr / returncode / args).
    """
    result = _tracked_subprocess_run(
        [sys.executable, "-c", "print('hello'); import sys; print('warn', file=sys.stderr); sys.exit(7)"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.stdout.strip() == "hello"
    assert "warn" in result.stderr
    assert result.returncode == 7


def test_registers_with_registry_during_execution():
    """While the subprocess is alive, it should be in ``registry.popens``.

    We use a child that signals via stdout when it has started, then
    sleeps; the parent thread checks ``registry.popens`` while the child
    is still alive.
    """
    registry = ProcessRegistry()
    seen_in_registry = threading.Event()

    def _watch():
        # Poll for up to 2s for the popen to show up.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with registry.lock:
                if registry.popens:
                    seen_in_registry.set()
                    return
            time.sleep(0.05)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()

    result = _tracked_subprocess_run(
        [sys.executable, "-c", "import time; time.sleep(0.4)"],
        registry=registry,
        timeout=5,
    )
    watcher.join(timeout=3)
    assert seen_in_registry.is_set(), "Popen should be tracked while alive"
    assert result.returncode == 0
    # And after the run returns, the registry should be empty again.
    assert registry.popens == [], "Popen must be removed from registry on exit"


def test_no_registry_still_works():
    """Calls with ``registry=None`` should behave like plain subprocess.run."""
    result = _tracked_subprocess_run(
        [sys.executable, "-c", "print('ok')"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_timeout_raises_timeout_expired_and_cleans_up():
    registry = ProcessRegistry()
    with pytest.raises(subprocess.TimeoutExpired):
        _tracked_subprocess_run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            registry=registry,
            timeout=0.3,
        )
    # After TimeoutExpired, the Popen should have been killed and removed
    # from the registry.
    time.sleep(0.2)
    assert registry.popens == [], "registry must be cleaned up after TimeoutExpired"


def test_terminate_all_kills_in_flight_tracked_subprocess(tmp_path):
    """Integration: the watchdog scenario.

    Start a long-running tracked subprocess in another thread; from the
    main thread, call ``registry.terminate_all()``; verify the subprocess
    is killed quickly and the runner returns (with a non-zero returncode
    or a TimeoutExpired-equivalent).
    """
    registry = ProcessRegistry()
    started = threading.Event()
    finished = threading.Event()
    result_holder: dict = {}

    def _runner():
        started.set()
        try:
            r = _tracked_subprocess_run(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                registry=registry,
                timeout=30,
            )
            result_holder["rc"] = r.returncode
        except subprocess.TimeoutExpired:
            result_holder["timeout"] = True
        finally:
            finished.set()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    assert started.wait(timeout=2.0)
    # Give the subprocess a moment to actually start before we terminate.
    time.sleep(0.3)
    registry.terminate_all(escalate_after_s=2.0)
    # Worker should observe the subprocess dying within a couple of seconds.
    assert finished.wait(timeout=10.0), "_tracked_subprocess_run did not return after terminate_all"
    # The subprocess died from a signal; returncode should be non-zero.
    rc = result_holder.get("rc")
    assert rc is None or rc != 0, f"expected nonzero returncode after kill, got {rc!r}"
