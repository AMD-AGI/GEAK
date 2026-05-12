"""``ProcessRegistry.terminate_all`` must be safe to call twice.

The second-SIGINT path calls ``terminate_all`` before raising
``KeyboardInterrupt``; the outer ``finally`` in ``mini.main`` then calls
it again. Reentrancy holes here would surface as ``ProcessLookupError``
/ ``ESRCH`` propagating out of the second call and masking the
exception that's actively propagating.

This test exercises the real failure modes by spawning genuine
``subprocess.Popen`` children, killing them, and calling
``terminate_all`` a second time after they're already gone.
"""

from __future__ import annotations

import subprocess
import sys
import time

from minisweagent.run.state import ProcessRegistry


def test_terminate_all_is_reentrant_with_real_subprocesses() -> None:
    """Two real children, two terminate_all() calls; no exception leaks; both reaped."""
    reg = ProcessRegistry()
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        for _ in range(2)
    ]
    try:
        with reg.lock:
            reg.popens.extend(procs)

        reg.terminate_all()
        # Second call must not raise: at this point the children may have
        # already been reaped, so killpg/getpgid hit ESRCH; those paths
        # must be tolerated.
        reg.terminate_all()

        # Wait for SIGTERM->SIGKILL escalation (5s default) to complete.
        # ``time.sleep(0.5)`` from earlier drafts of this test was too short.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and any(p.poll() is None for p in procs):
            time.sleep(0.05)
        assert all(p.poll() is not None for p in procs), (
            "all children should be reaped within 10s of the first terminate_all"
        )
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
                p.wait(timeout=2.0)


def test_terminate_all_on_empty_registry_is_a_noop() -> None:
    """No tracked processes -> immediate return, no log, no raise."""
    reg = ProcessRegistry()
    reg.terminate_all()
    reg.terminate_all()
