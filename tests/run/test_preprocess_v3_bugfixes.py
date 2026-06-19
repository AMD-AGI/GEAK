from pathlib import Path
from types import SimpleNamespace

import pytest

import minisweagent.run.preprocess.resolve_kernel_url as resolve_kernel_url_module
from minisweagent.run.preprocess_v3.adapter import (
    _build_orchestrator_task,
    _preprocess_result_to_legacy_context,
    _resolve_kernel_and_repo,
)
from minisweagent.run.preprocess_v3.orchestrator import (
    FinishedSuccessfully,
    PreprocessOrchestratorAgent,
    PreprocessOrchestratorConfig,
)
from minisweagent.run.preprocess_v3.tools import (
    _ensure_preprocess_subagent_sandbox,
    _make_tool_collect_baseline,
    _make_tool_commandment_from_user_command,
    _make_tool_dispatch_subagent,
    _make_tool_finish_preprocess,
    _make_tool_warm_up_harness,
)


def test_resolve_kernel_path_relative_to_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    kernel = repo / "kernels" / "silu.hip"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("// hip kernel\n")

    resolved_kernel, resolved_repo = _resolve_kernel_and_repo("kernels/silu.hip", repo, console=None)

    assert resolved_kernel == kernel.resolve()
    assert resolved_repo == str(repo.resolve())


def test_resolve_kernel_fallback_uses_legacy_resolver_keys(tmp_path: Path, monkeypatch) -> None:
    cloned_repo = tmp_path / "cloned-repo"
    kernel = cloned_repo / "kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("# kernel\n")

    def fake_resolve_kernel_url(kernel_url: str, repo: str | None = None) -> dict:
        assert kernel_url == "https://example.test/repo/blob/main/kernel.py"
        assert repo == str(tmp_path / "repo")
        return {
            "error": None,
            "local_file_path": str(kernel),
            "local_repo_path": str(cloned_repo),
        }

    monkeypatch.setattr(resolve_kernel_url_module, "resolve_kernel_url", fake_resolve_kernel_url)

    resolved_kernel, resolved_repo = _resolve_kernel_and_repo(
        "https://example.test/repo/blob/main/kernel.py",
        tmp_path / "repo",
        console=None,
    )

    assert resolved_kernel == kernel.resolve()
    assert resolved_repo == str(cloned_repo.resolve())


def test_path_a_commandment_runs_user_command_through_run_sh(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    out_path = tmp_path / "COMMANDMENT.md"
    raw_command = (
        f"python3 {repo}/scripts/task_runner.py compile && "
        "python3 scripts/task_runner.py correctness && "
        "python3 scripts/task_runner.py performance"
    )
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )

    tool = _make_tool_commandment_from_user_command(agent)
    result = tool(
        run_command=raw_command,
        out_path=str(out_path),
        modes_covered=["benchmark"],
        inferred_modes=["correctness", "full_benchmark"],
    )

    text = out_path.read_text()
    assert result["ok"] is True
    assert "printf '#!/bin/bash" in text
    assert "exec bash -lc" in text
    assert "${GEAK_WORK_DIR}/run.sh" in text
    assert "cd ${GEAK_WORK_DIR} && python3" not in text
    assert str(repo) not in text
    assert "${GEAK_WORK_DIR}/scripts/task_runner.py" in text


def test_path_a_flagless_command_does_not_render_silent_duplicates(tmp_path: Path) -> None:
    """Issue #258: a flag-less Path-A command must NOT render an all-modes-identical
    COMMANDMENT, even when the LLM marks all four modes covered. The deterministic
    backstop refuses and signals PATH_A_FLAG_MISSING instead."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )

    tool = _make_tool_commandment_from_user_command(agent)
    result = tool(
        run_command="timeout 600 python op_tests/test_rmsnorm2dFusedAddQuant.py",
        out_path=str(out_path),
        # Even with all four marked covered, the backstop must still fire.
        modes_covered=["correctness", "profile", "benchmark", "full_benchmark"],
        inferred_modes=[],
    )

    assert result["ok"] is False
    assert result["error"] == "PATH_A_FLAG_MISSING"
    assert any("PATH_A_FLAG_MISSING" in w for w in result["warnings"])
    # No runnable COMMANDMENT may be written for the flag-less case.
    assert not out_path.exists()


def test_path_a_flag_aware_command_still_renders_four_modes(tmp_path: Path) -> None:
    """A1 happy path: a command that already carries a GEAK mode flag still renders
    four distinct mode sections. Confirms the is_flagless detector does NOT mis-fire."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )

    tool = _make_tool_commandment_from_user_command(agent)
    result = tool(
        run_command="python kernel_bench.py --benchmark",
        out_path=str(out_path),
        modes_covered=["correctness", "profile", "benchmark", "full_benchmark"],
        inferred_modes=[],
    )

    assert result["ok"] is True
    text = out_path.read_text()
    # Each section carries its own real flag (substituted from --benchmark).
    assert "--correctness" in text
    assert "--full-benchmark" in text
    assert "--profile" in text


def test_path_a_flagless_amalgamation_is_refused(tmp_path: Path) -> None:
    """A non-build ``&&`` amalgamation (same script run twice with different
    settings, no build step) must be refused with PATH_A_FLAG_MISSING rather than
    blindly split left=correctness / right=performance (which drops one metric)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )

    tool = _make_tool_commandment_from_user_command(agent)
    result = tool(
        run_command="python test.py --opt-a && python test.py --opt-b",
        out_path=str(out_path),
        modes_covered=["correctness", "benchmark"],
        inferred_modes=[],
    )

    assert result["ok"] is False
    assert result["error"] == "PATH_A_FLAG_MISSING"
    assert not out_path.exists()


def test_path_a_flag_bearing_amalgamation_is_refused(tmp_path: Path) -> None:
    """R2-1: a *flag-bearing* amalgamation would yield a harness path via
    _extract_harness_from_command and slip past the flag-less backstop, running
    only the first half. The flag-independent amalgamation guard (placed before
    harness extraction) must still refuse it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )

    tool = _make_tool_commandment_from_user_command(agent)
    result = tool(
        run_command="python test.py --benchmark --opt-a && python test.py --benchmark --opt-b",
        out_path=str(out_path),
        modes_covered=["benchmark"],
        inferred_modes=[],
    )

    assert result["ok"] is False
    assert result["error"] == "PATH_A_FLAG_MISSING"
    assert not out_path.exists()


def test_path_a_build_bearing_amalgamation_still_synthesizes(tmp_path: Path) -> None:
    """A build-bearing ``&&`` (compile + run) is NOT an amalgamation: it has a
    confident leading compile prefix, so the deterministic split path is preserved
    and the amalgamation guard does not mis-fire."""
    repo = tmp_path / "repo"
    repo.mkdir()
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )

    tool = _make_tool_commandment_from_user_command(agent)
    result = tool(
        run_command="make && python test.py --benchmark",
        out_path=str(out_path),
        modes_covered=["benchmark"],
        inferred_modes=["correctness"],
    )

    assert result["ok"] is True
    assert out_path.exists()


def test_baseline_build_env_exports_geak_work_dir(tmp_path: Path) -> None:
    """_build_env must export GEAK_WORK_DIR (and GEAK_REPO_ROOT) when work_dir is
    given, so a contract-compliant harness resolves the real source tree instead
    of falling back to its own directory (silent 'produced no latency')."""
    from minisweagent.run.preprocess_v3.baseline import _build_env

    work = tmp_path / "repo"
    work.mkdir()

    env = _build_env(work, gpu_id=0)
    assert env["GEAK_WORK_DIR"] == str(work)
    assert env["GEAK_REPO_ROOT"] == str(work)
    assert str(work) in env["PYTHONPATH"]

    # When work_dir is None, neither key is added (preserves prior no-op behavior).
    env_none = _build_env(None, gpu_id=0)
    assert "GEAK_WORK_DIR" not in env_none
    assert "GEAK_REPO_ROOT" not in env_none


def test_collect_baseline_defaults_work_dir_to_source_repo(tmp_path: Path, monkeypatch) -> None:
    """When the subagent omits work_dir, collect_baseline must fall back to the
    orchestrator's source repo so baseline runs with a valid GEAK_WORK_DIR."""
    import minisweagent.run.preprocess_v3.tools as tools_module

    repo = tmp_path / "repo"
    repo.mkdir()
    harness = tmp_path / "harness.py"
    harness.write_text("print('GEAK_RESULT_LATENCY_MS=1.0')\n")

    captured: dict[str, object] = {}

    def fake_collect_baseline_metrics(harness_path, *, repeats, work_dir, gpu_id, extra_env=None):
        captured["work_dir"] = work_dir
        captured["extra_env"] = extra_env
        return SimpleNamespace(
            success=True,
            median_ms=1.0,
            samples_ms=[1.0],
            stdev_ms=0.0,
            repeats=repeats,
            harness_path=harness_path,
            command="",
        )

    monkeypatch.setattr(tools_module, "collect_baseline_metrics", fake_collect_baseline_metrics)
    # capture_full_benchmark_stdout is imported lazily inside the tool from the
    # baseline module; stub it there so this unit test runs no real subprocess.
    import minisweagent.run.preprocess_v3.baseline as baseline_module

    monkeypatch.setattr(baseline_module, "capture_full_benchmark_stdout", lambda *a, **k: None)

    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )
    tool = _make_tool_collect_baseline(agent)
    # No work_dir passed -> must default to agent.config.repo.
    tool(harness_path=str(harness), repeats=1)

    assert captured["work_dir"] == repo


def test_dispatch_subagent_uses_sandbox_worktree_env(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.py").write_text("# kernel\n")
    output_dir = tmp_path / "out"
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )
    agent._extra_template_vars = {
        "repo_root": str(repo),
        "output_dir": str(output_dir),
        "gpu_id": 2,
    }
    seen: dict = {}

    class FakeDispatcher:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return {
                "name": kwargs["name"],
                "success": True,
                "output": f"HARNESS_PATH: {output_dir / '_preprocess_subagent_worktree' / 'harness.py'}",
            }

    tool = _make_tool_dispatch_subagent(agent, FakeDispatcher())
    result = tool(name="harness-generator", task="make a harness", context={"repo_root": str(repo)})

    sandbox = output_dir / "_preprocess_subagent_worktree"
    assert result["success"] is True
    assert Path(seen["cwd"]) == sandbox.resolve()
    assert sandbox.is_dir()
    assert (sandbox / "kernel.py").is_file()
    assert seen["context"]["sandbox_repo_root"] == str(sandbox.resolve())
    assert seen["context"]["_tool_env"]["GEAK_REPO_ROOT"] == str(repo.resolve())
    assert seen["context"]["_tool_env"]["GEAK_WORK_DIR"] == str(sandbox.resolve())
    assert seen["context"]["_tool_env"]["GEAK_GPU_DEVICE"] == "2"


def test_harness_generator_retry_cap_is_enforced(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.py").write_text("# kernel\n")
    output_dir = tmp_path / "out"
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )
    agent._extra_template_vars = {
        "repo_root": str(repo),
        "output_dir": str(output_dir),
        "gpu_id": 0,
    }
    calls = {"count": 0}

    class FakeDispatcher:
        def __call__(self, **kwargs):
            calls["count"] += 1
            return {"name": kwargs["name"], "success": False, "output": "HARNESS_VERIFIED=false"}

    tool = _make_tool_dispatch_subagent(agent, FakeDispatcher())
    for attempt in range(1, 4):
        result = tool(name="harness-generator", task="try", context={})
        assert result["success"] is False
        assert agent._collected["_harness_generator_attempts"] == attempt

    capped = tool(name="harness-generator", task="try again", context={})
    assert capped["success"] is False
    assert "retry budget exhausted" in capped["error"]
    assert calls["count"] == 3


def test_finish_preprocess_allows_failed_result_to_terminate() -> None:
    agent = PreprocessOrchestratorAgent(model=object())
    agent._collected = {}
    tool = _make_tool_finish_preprocess(agent)

    with pytest.raises(FinishedSuccessfully):
        tool(errors=["harness-generator retry budget exhausted"])


def test_legacy_context_recovers_harness_path_from_promoted_command(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    harness = repo / "tests" / "test_topk_harness.py"
    kernel = repo / "aiter" / "ops" / "triton" / "topk.py"
    output_dir = tmp_path / "out"
    harness.parent.mkdir(parents=True)
    kernel.parent.mkdir(parents=True)
    output_dir.mkdir()
    harness.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--profile', action='store_true')\n"
        "parser.add_argument('--correctness', action='store_true')\n"
        "parser.add_argument('--benchmark', action='store_true')\n"
        "parser.add_argument('--full-benchmark', action='store_true')\n"
        "parser.add_argument('--iterations', type=int, default=1)\n"
        "print('harness')\n"
    )
    kernel.write_text("# kernel\n")
    commandment = output_dir / "COMMANDMENT.md"
    commandment.write_text("# Commandment\n")
    baseline = SimpleNamespace(
        median_ms=1.25,
        samples_ms=[1.2, 1.3],
        stdev_ms=0.1,
        repeats=2,
        command="python harness --benchmark",
        success=True,
        raw_outputs=[
            {
                "returncode": 0,
                "stdout": "GEAK_RESULT_LATENCY_MS=1.25\n",
                "latency_ms": 1.25,
            }
        ],
    )
    result = SimpleNamespace(
        kernel_path=kernel,
        kernel_language=SimpleNamespace(name="triton"),
        baseline=baseline,
        full_benchmark_stdout=None,
        profile=None,
        commandment_path=commandment,
        codebase_context=None,
        harness_path=None,
        translation=None,
        subagent_runs=[],
        elapsed_s=1.0,
        path_taken="A",
    )

    ctx = _preprocess_result_to_legacy_context(
        result=result,
        repo_root=str(repo),
        output_dir=output_dir,
        kernel_path_input=kernel,
        eval_command=f"python {harness}",
    )

    assert ctx["test_command"] == f"python {harness}"
    assert ctx["harness_path"] == str(harness.resolve())
    assert ctx["benchmark_baseline"] == str(output_dir / "benchmark_baseline.txt")
    assert ctx["full_benchmark_baseline"] == str(output_dir / "full_benchmark_baseline.txt")
    assert (output_dir / "benchmark_baseline.txt").read_text() == "GEAK_RESULT_LATENCY_MS=1.25\n"
    assert ctx["v3_path_taken"] == "A"


def _harness_hint(harness: str = "/tmp/x_harness.py") -> str:
    """Render just the call-site hint block for a supplied harness."""
    return _build_orchestrator_task(
        user_task="Optimize this kernel.",
        harness=harness,
        eval_command=None,
        correctness_command=None,
        performance_command=None,
        benchmark_timeout=3600,
        translate_only=False,
    )


def test_harness_hint_contains_both_a1_and_a2_branches() -> None:
    """Deletion-guard for the shapes-override hint (prompt-only fix).

    The supplied-harness hint must carry BOTH routing branches so the
    orchestrator can pick A1 (no prompt shapes) or A2-with-shapes (prompt
    shapes override the harness). This is a presence check only — it does NOT
    prove routing (the LLM decides) nor that the generator's shape-compare
    works (covered by manual E2E). Presence != executability.
    """
    task = _harness_hint("/tmp/x_harness.py")

    # A1 branch: the all-four commandment_from_user_command call, fenced to the
    # no-prompt-shapes case, plus the skip-discovery fast path.
    assert "commandment_from_user_command" in task
    assert "['correctness', 'profile', 'benchmark', 'full_benchmark']" in task
    assert "skip `run_discovery`" in task

    # A2-override branch: prompt shapes divert to the generator with the harness
    # passed as a structural template; the orchestrator must NOT itself read or
    # compare shapes.
    assert "template_harness_path: /tmp/x_harness.py" in task
    assert "harness-generator" in task

    # The override must not be exempted by the harness passing the four-mode
    # contract: "pre-validated" must no longer appear as an unconditional A1 trigger.
    assert "pre-validated" not in task.lower()


def test_prevalidated_harness_bypasses_llm_orchestrator(tmp_path: Path, monkeypatch) -> None:
    """A pre-validated harness must run the deterministic Path-A sequence
    (collect_baseline -> collect_profile -> render_commandment) WITHOUT ever
    invoking the LLM orchestrator.

    Regression: the orchestrator's LLM classifier could misroute a shape-bearing
    task to the harness-generator or fail to converge, burning the whole
    preprocess budget with no baseline. ``_run_prevalidated_path_a`` short-circuits
    that. Here the model is a sentinel that raises if queried — proving the LLM
    loop is skipped entirely.
    """
    import minisweagent.run.preprocess_v3.adapter as adapter_module

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.py").write_text("# kernel\n")
    harness = tmp_path / "harness.py"
    harness.write_text("print('GEAK_RESULT_LATENCY_MS=2.0')\n")
    output_dir = tmp_path / "out"

    class _ExplodingModel:
        def query(self, *a, **k):  # pragma: no cover - must never be called
            raise AssertionError("LLM orchestrator was invoked for a pre-validated harness")

    # Stub the deterministic building blocks so the test runs no real subprocess.
    def fake_collect_baseline_metrics(harness_path, *, work_dir=None, gpu_id=0, repeats=5):
        return SimpleNamespace(
            success=True, median_ms=2.0, samples_ms=[2.0], stdev_ms=0.0,
            repeats=repeats, harness_path=harness_path, command="python harness",
            raw_outputs=[{"stdout": "GEAK_RESULT_LATENCY_MS=2.0", "returncode": 0, "latency_ms": 2.0}],
        )

    monkeypatch.setattr(adapter_module, "PreprocessOrchestratorAgent",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("orchestrator constructed")))
    # The worktree-bypass gate (contract.validate_harness) is a separate concern;
    # disable it here so the test exercises the bypass control-flow, not harness
    # contract validation (covered elsewhere).
    monkeypatch.setenv("GEAK_ALLOW_HARDCODED_PATHS", "1")
    import minisweagent.run.preprocess_v3.baseline as baseline_module
    monkeypatch.setattr(baseline_module, "collect_baseline_metrics", fake_collect_baseline_metrics)
    monkeypatch.setattr(baseline_module, "capture_full_benchmark_stdout", lambda *a, **k: "GEAK_RESULT_LATENCY_MS=2.0")
    monkeypatch.setattr(baseline_module, "collect_profile",
                        lambda *a, **k: SimpleNamespace(success=False, profile=None, command="", backend="metrix", profile_path=None))

    ctx = adapter_module.run_preprocess_v3(
        kernel_url=str(repo / "kernel.py"),
        output_dir=output_dir,
        gpu_id=0,
        model=_ExplodingModel(),
        harness=str(harness),
        repo=str(repo),
    )

    # Deterministic path produced the baseline + commandment, LLM never ran.
    assert ctx["v3_path_taken"] == "A"
    assert (output_dir / "COMMANDMENT.md").is_file()
    assert ctx["benchmark_baseline"] == str(output_dir / "benchmark_baseline.txt")


def test_prevalidated_bypass_opt_out_env(tmp_path: Path, monkeypatch) -> None:
    """GEAK_NO_PREVALIDATED_BYPASS=1 disables the deterministic short-circuit
    (falls back to the LLM orchestrator path)."""
    import minisweagent.run.preprocess_v3.adapter as adapter_module

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.py").write_text("# kernel\n")
    harness = tmp_path / "harness.py"
    harness.write_text("print('ok')\n")

    monkeypatch.setenv("GEAK_NO_PREVALIDATED_BYPASS", "1")

    called = {"bypass": False}
    monkeypatch.setattr(adapter_module, "_run_prevalidated_path_a",
                        lambda **k: called.__setitem__("bypass", True))
    # Make the orchestrator path raise immediately so we can detect we reached it
    # (and did NOT take the bypass).
    monkeypatch.setattr(adapter_module, "PreprocessOrchestratorAgent",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached-orchestrator")))

    with pytest.raises(RuntimeError, match="reached-orchestrator"):
        adapter_module.run_preprocess_v3(
            kernel_url=str(repo / "kernel.py"),
            output_dir=tmp_path / "out",
            model=object(),
            harness=str(harness),
            repo=str(repo),
        )
    assert called["bypass"] is False


def test_warm_up_harness_runs_once_with_baseline_cache(tmp_path, monkeypatch):
    """warm_up_harness runs the harness once in --correctness with the persistent
    baseline cache + a 1-shape cap, and is non-fatal on failure."""
    import minisweagent.run.preprocess_v3.baseline as baseline_module

    repo = tmp_path / "repo"
    repo.mkdir()
    harness = repo / "harness.py"
    harness.write_text("# harness\n")

    seen = {}

    def fake_run_once(hp, *, work_dir, gpu_id, timeout_s, flag, extra_env=None):
        seen["flag"] = flag
        seen["work_dir"] = work_dir
        seen["extra_env"] = extra_env or {}
        seen["timeout_s"] = timeout_s
        return {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 1.2, "latency_ms": None}

    monkeypatch.setattr(baseline_module, "_run_benchmark_once", fake_run_once)

    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )
    tool = _make_tool_warm_up_harness(agent)
    result = tool(harness_path=str(harness))

    assert result["ok"] is True
    # Warm the FULL config superset (--full-benchmark), not a single shape — each
    # shape compiles a separate artifact, so 1-shape warming leaves the rest cold.
    assert seen["flag"] == "--full-benchmark"
    assert seen["work_dir"] == repo
    # No shape cap: full coverage is the point (cost bounded by the timeout).
    assert "GEAK_MAX_BENCHMARK_SHAPES" not in seen["extra_env"]
    # The persistent baseline cache env is passed through.
    assert "/baseline/" in seen["extra_env"].get("GEAK_JIT_CACHE_DIR", "")
    assert "TRITON_CACHE_DIR" in seen["extra_env"]


def test_warm_up_harness_non_fatal_on_failure(tmp_path, monkeypatch):
    import minisweagent.run.preprocess_v3.baseline as baseline_module

    repo = tmp_path / "repo"
    repo.mkdir()
    harness = repo / "harness.py"
    harness.write_text("# harness\n")

    def fake_run_once(hp, *, work_dir, gpu_id, timeout_s, flag, extra_env=None):
        return {"returncode": -1, "stdout": "", "stderr": "TIMEOUT", "duration_s": 9.0, "latency_ms": None}

    monkeypatch.setattr(baseline_module, "_run_benchmark_once", fake_run_once)

    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )
    tool = _make_tool_warm_up_harness(agent)
    # Must NOT raise; ok=False but the run continues.
    result = tool(harness_path=str(harness))
    assert result["ok"] is False
    assert "advisory" in result["note"]


def test_preprocess_sandbox_env_carries_baseline_cache(tmp_path):
    """The preprocess subagent (verifier) env includes the persistent baseline
    JIT cache so its compiles hit the warm cache."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.py").write_text("# kernel\n")
    output_dir = tmp_path / "out"
    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=repo),
    )
    agent._extra_template_vars = {
        "repo_root": str(repo),
        "output_dir": str(output_dir),
        "gpu_id": 0,
    }
    _sandbox, env = _ensure_preprocess_subagent_sandbox(agent)
    assert "/baseline/" in env["GEAK_JIT_CACHE_DIR"]
    assert "TRITON_CACHE_DIR" in env and "AITER_JIT_DIR" in env
