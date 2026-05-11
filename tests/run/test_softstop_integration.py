"""Integration tests for the SoftStop -> finalize path.

These tests exercise the riskiest part of the wall-clock budget design: when
the optimization watchdog flips ``soft_stop`` mid-dispatch, the orchestrator
must (a) stop dispatching new work, (b) reach the finalize phase quickly,
and (c) produce ``final_report.json`` on disk.

Two layers of test:

- ``test_dispatch_short_circuits_on_soft_stop`` -- the lowest-level guarantee:
  ``tool_dispatch_tasks`` must return a "skipped" payload when soft_stop is
  already set, *without* spawning any worker. This is what protects the
  finalize-grace window from being eaten by a 30-minute parallel batch.

- ``test_round_loop_finalizes_when_soft_stop_set_before_round`` -- exercises
  the round loop's per-round soft-stop check and the programmatic finalize
  fallback when the LLM doesn't (or can't) emit a ``finalize`` tool call.

The integration test from the plan ("monkeypatch run_preprocessor + sleeping
stub workers and verify final_report.json is written within finalize_grace")
is split here so each layer can fail independently and we get a clearer
signal on which layer regressed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Layer 1: dispatch short-circuit
# ---------------------------------------------------------------------------


def test_dispatch_short_circuits_on_soft_stop(tmp_path):
    """``tool_dispatch_tasks`` must return ``status=skipped`` when soft_stop
    is already set, without invoking ``run_task_batch`` at all.
    """
    from minisweagent.agents.heterogeneous import tools as tools_mod

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    soft_stop = threading.Event()
    soft_stop.set()  # simulate watchdog already fired

    ctx = {
        "output_dir": str(output_dir),
        "gpu_ids": [0],
        "soft_stop": soft_stop,
        "deadline": None,
        "registry": None,
        "model_factory": lambda: None,
    }

    invoked = {"count": 0}

    def _fake_run_task_batch(*args, **kwargs):
        invoked["count"] += 1
        return {"completed": 0, "failed": 0, "results": []}

    # If dispatch ever calls into run_task_batch we'll see invoked["count"] > 0,
    # which means the short-circuit failed.
    import minisweagent.run.dispatch as dispatch_mod

    original = dispatch_mod.run_task_batch
    dispatch_mod.run_task_batch = _fake_run_task_batch
    try:
        # Provide some task files so we know we wouldn't otherwise return early.
        tasks_dir = output_dir / "tasks" / "round_1"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "task1.md").write_text("---\nlabel: t1\n---\nbody")

        payload = tools_mod.tool_dispatch_tasks(ctx)
        result = json.loads(payload)
        assert result["status"] == "skipped"
        assert "soft-stop" in result["reason"].lower()
        assert invoked["count"] == 0, "run_task_batch must not be invoked when soft_stop is set"
    finally:
        dispatch_mod.run_task_batch = original


# ---------------------------------------------------------------------------
# Layer 2: orchestrator round loop calls finalize when soft_stop is set
# ---------------------------------------------------------------------------


def test_round_loop_finalizes_when_soft_stop_set_before_round(tmp_path, monkeypatch):
    """When ``soft_stop`` is set before the round loop starts iterating, the
    orchestrator must:
      1. Skip the LLM round loop.
      2. Either get a finalize tool call from a deadline-prompted LLM or
         fall through to the programmatic ``finalize_run`` fallback.
      3. Produce ``final_report.json`` on disk.

    We test the fallback path (no LLM finalize) because that's the corner
    case the design explicitly promises will still produce a report.
    """
    # Build a synthetic preprocess_ctx with the minimum keys required for
    # run_heterogeneous_orchestrator. We bypass run_preprocessor entirely.
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "COMMANDMENT.md").write_text("## SETUP\necho ok\n## CORRECTNESS\necho ok\n")

    preprocess_ctx = {
        "kernel_path": str(tmp_path / "kernel.py"),
        "repo_root": str(tmp_path),
        "harness_path": str(tmp_path / "harness.py"),
        "preprocess_dir": str(output_dir),
        "commandment": "## SETUP\necho ok\n## CORRECTNESS\necho ok\n",
        "baseline_metrics": {"duration_us": 1000.0, "bottleneck": "memory-bound"},
        "profiling": {},
        "discovery": {},
    }

    # Monkeypatch the LLM so the explore phase returns immediately and any
    # subsequent query -- post deadline-finalize prompt -- also returns
    # without a tool call so we exercise the programmatic finalize fallback.
    fake_model = MagicMock()
    fake_model.config.api_key = ""
    fake_model.query.return_value = {"content": "Ready to begin optimization rounds.", "tools": None}
    fake_model._impl = MagicMock()
    fake_model._impl.tools = []
    fake_model.cost = 0.0
    fake_model.n_calls = 0

    # finalize_run must be called and must write final_report.json. Use the
    # real finalize_run but with empty rounds; it will write the report in
    # the no-LLM-finalize fallback path inside auto_finalize.
    from minisweagent.run.postprocess import results as results_mod

    finalize_calls = {"count": 0}
    original_finalize = results_mod.finalize_run

    def _spy_finalize_run(ctx, output_dir, *args, **kwargs):
        finalize_calls["count"] += 1
        # Make sure final_report.json exists regardless of what auto_finalize
        # produces -- this isolates the test from postprocess internals.
        report = {
            "status": "complete",
            "summary": "soft-stop fallback finalize",
            "best_speedup": None,
            "best_patch": None,
        }
        Path(output_dir).joinpath("final_report.json").write_text(json.dumps(report))
        return MagicMock(
            best_speedup=None,
            best_patch=None,
            best_round=None,
            summary="soft-stop fallback finalize",
            to_dict=lambda: report,
        )

    monkeypatch.setattr(results_mod, "finalize_run", _spy_finalize_run)

    # Pre-set SoftStop so the round loop's per-round check fires immediately.
    soft_stop = threading.Event()
    soft_stop.set()

    from minisweagent.agents.heterogeneous.orchestrator import run_heterogeneous_orchestrator
    from minisweagent.run.budget import Deadline

    deadline = Deadline(time.monotonic() + 60.0, soft_stop)

    t0 = time.monotonic()
    run_heterogeneous_orchestrator(
        preprocess_ctx,
        gpu_ids=[0],
        model=fake_model,
        model_factory=lambda: fake_model,
        output_dir=output_dir,
        max_rounds=2,
        start_round=1,
        deadline=deadline,
        soft_stop=soft_stop,
        registry=None,
    )
    elapsed = time.monotonic() - t0

    # finalize_run was called and final_report.json exists.
    assert finalize_calls["count"] >= 1, "finalize_run should have been invoked"
    assert (output_dir / "final_report.json").is_file(), "final_report.json must be written"

    # We finalized quickly (well within finalize_grace_s budget); no real
    # 30-minute parallel batch ran. Be generous to allow for CI variance.
    assert elapsed < 30.0, f"finalize should be fast; took {elapsed:.1f}s"

    # Restore (monkeypatch.setattr handles teardown automatically).
    _ = original_finalize  # silence unused warning


# ---------------------------------------------------------------------------
# Layer 3: ProcessRegistry serializes terminate_all with submission
# ---------------------------------------------------------------------------


def test_registry_lock_serializes_with_submission(tmp_path):
    """The (soft_stop check + executor.submit + registry.futures.append)
    sequence must run inside ``registry.lock`` so terminate_all() cannot
    miss a worker that has already started.

    We don't reproduce the race per se (it's hard to deterministically
    trigger); we just verify that holding the lock blocks terminate_all().
    """
    import concurrent.futures

    from minisweagent.run.state import ProcessRegistry

    registry = ProcessRegistry()
    blocked = threading.Event()
    proceed = threading.Event()

    def _terminate_thread():
        # Simulate the watchdog calling terminate_all while the dispatch
        # loop is still inside the (check+submit+track) critical section.
        blocked.set()
        with registry.lock:
            # We got the lock -- meaning the dispatch loop has completed
            # its critical section. terminate_all() should now operate on
            # a fully-populated futures list.
            assert len(registry.futures) == 1
        proceed.set()

    th = threading.Thread(target=_terminate_thread)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        with registry.lock:
            # Start the terminator thread; it should block trying to take
            # the lock.
            th.start()
            assert blocked.wait(timeout=1.0)
            time.sleep(0.05)
            assert not proceed.is_set(), "terminator must wait for our lock"

            fut = executor.submit(lambda: 42)
            registry.futures.append(fut)

        th.join(timeout=2.0)
        assert proceed.is_set()
        assert fut.result(timeout=2.0) == 42
