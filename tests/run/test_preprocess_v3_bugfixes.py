from pathlib import Path
from types import SimpleNamespace

import minisweagent.run.preprocess.resolve_kernel_url as resolve_kernel_url_module
import pytest
from minisweagent.run.preprocess_v3.adapter import _preprocess_result_to_legacy_context, _resolve_kernel_and_repo
from minisweagent.run.preprocess_v3.orchestrator import (
    FinishedSuccessfully,
    PreprocessOrchestratorAgent,
    PreprocessOrchestratorConfig,
)
from minisweagent.run.preprocess_v3.tools import (
    _make_tool_collect_baseline,
    _make_tool_commandment_from_user_command,
    _make_tool_dispatch_subagent,
    _make_tool_finish_preprocess,
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

    def fake_collect_baseline_metrics(harness_path, *, repeats, work_dir, gpu_id):
        captured["work_dir"] = work_dir
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
