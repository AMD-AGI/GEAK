"""Run ``profiler_mcp.server.profile_kernel`` in a child process.

The profiler stage is the most likely place for ``run_preprocessor`` to wedge
(``ncu`` / ``rocprof`` calls into the GPU driver and can sit there for tens of
minutes). To make ``state.skip_profiling = True`` actually mean something, we
spawn the profile call as a ``multiprocessing.Process`` so we have a real
``terminate()`` handle.

Two non-obvious requirements:

1. ``mp.Process`` has no ``start_new_session=True`` equivalent. The child must
   call ``os.setsid()`` before importing the profiler so its grandchildren
   (``ncu`` / ``rocprof`` GPU subprocesses) inherit the new session and can be
   killed via ``os.killpg(child.pid, SIGTERM)`` from the registry.

2. ``mp.Queue`` deadlocks on ``proc.join()`` if the child is killed while in
   the middle of writing a large object (Metrix profiles can be ~MB-sized).
   We drain non-blockingly with ``q.get_nowait()`` before joining and treat an
   empty queue as "profiler terminated without a result".
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import signal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from minisweagent.run.state import PreprocessState, ProcessRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Child-process target
# ---------------------------------------------------------------------------


def _profile_in_subproc(
    perf_cmd: str,
    backend: str,
    gpu_id: int,
    workdir: str | None,
    num_replays: int,
    quick: bool,
    q: mp.Queue,
) -> None:
    """Child-process entry for profile_kernel.

    Becomes a session leader on the very first line so any GPU profiler this
    invocation forks (ncu / rocprof) is in our process group and can be
    cleaned up by ``os.killpg(self.pid, SIGTERM)``.
    """
    try:
        os.setsid()
    except OSError:
        # Already a session leader (rare but possible if Python invokes us
        # in an already-detached process). Continue regardless.
        pass

    # Make sure SIGTERM kills the whole group rather than orphaning the
    # ncu/rocprof grandchild. The default action for SIGTERM is to terminate;
    # we keep that, but explicitly install it so any inherited "ignore" from
    # the parent doesn't carry over.
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    except (ValueError, OSError):
        pass

    try:
        # Lazy import: keep ``profiler_mcp`` out of the parent's import graph
        # so a missing/broken profiler MCP doesn't kill the whole run.
        from minisweagent.run.preprocess.repo_paths import ensure_preprocess_mcp_importable

        ensure_preprocess_mcp_importable(
            "mcp_tools/profiler-mcp/src",
            "mcp_tools/metrix-mcp/src",
        )
        from profiler_mcp.server import profile_kernel  # type: ignore[import-not-found]

        _profile_fn = getattr(profile_kernel, "fn", profile_kernel)

        kwargs: dict[str, Any] = {
            "command": perf_cmd,
            "backend": backend,
            "num_replays": num_replays,
            "quick": quick,
            "gpu_devices": str(gpu_id),
        }
        if workdir is not None:
            kwargs["workdir"] = workdir

        result = _profile_fn(**kwargs)
        q.put(("ok", result))
    except Exception as e:
        logger.exception("profile_kernel raised in subproc")
        q.put(("err", repr(e)))


# ---------------------------------------------------------------------------
# Parent-side runner
# ---------------------------------------------------------------------------


def run_profiler_with_handle(
    state: PreprocessState | None = None,
    *,
    perf_cmd: str,
    backend: str = "metrix",
    gpu_id: int = 0,
    workdir: str | None = None,
    num_replays: int = 3,
    quick: bool = False,
    timeout_s: float | None = None,
    registry: ProcessRegistry | None = None,
) -> dict | None:
    """Run profile_kernel in a child process; return its result or ``None``.

    The process is tracked via *registry* (or ``state.registry``) so the
    watchdog / SIGINT handler can ``terminate_all()`` it. Returns ``None``
    if the profile was terminated, raised, or produced no result.
    """
    _registry = registry or (state.registry if state else None)

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_profile_in_subproc,
        args=(perf_cmd, backend, gpu_id, workdir, num_replays, quick, q),
        name="geak-profiler",
        daemon=False,
    )

    _ctx_mgr = _registry.track_mp(proc) if _registry else contextlib.nullcontext()
    with _ctx_mgr:
        proc.start()
        try:
            proc.join(timeout=timeout_s)
        finally:
            if proc.is_alive():
                # Either timed out or the soft-stop handler set
                # state.skip_profiling and signalled us; either way, kill.
                logger.warning(
                    "profiler still alive after join(timeout=%s); sending SIGTERM to pgid %s",
                    timeout_s,
                    proc.pid,
                )
                _kill_group(proc.pid, signal.SIGTERM)
                proc.join(5)
                if proc.is_alive():
                    logger.warning("profiler ignored SIGTERM; escalating to SIGKILL pid=%s", proc.pid)
                    _kill_group(proc.pid, signal.SIGKILL)
                    proc.join(2)

    # Drain non-blockingly. Joining the queue thread before draining can
    # deadlock if the child was killed mid-put on a large payload.
    payload: dict | None = None
    try:
        kind, value = q.get_nowait()
        if kind == "ok":
            payload = value
        else:
            logger.warning("profiler subproc returned error: %s", value)
    except queue_mod.Empty:
        logger.info("profiler subproc terminated without producing a result")
    finally:
        try:
            q.close()
            q.join_thread()
        except Exception:
            logger.exception("mp.Queue cleanup failed (non-fatal)")

    return payload


def _kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except OSError as e:
        logger.warning("profiler killpg(pid=%s, sig=%s) failed: %s", pid, sig, e)
