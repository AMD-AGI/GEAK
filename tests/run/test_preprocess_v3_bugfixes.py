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
    _make_tool_translate_to_flydsl,
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


@pytest.mark.parametrize(
    "translation, expected_skip",
    [
        (SimpleNamespace(success=True), True),    # translation validated -> skip gate
        (SimpleNamespace(success=False), False),  # translation failed -> keep gate
        (None, False),                            # user-supplied harness -> keep gate
    ],
)
def test_collect_baseline_skips_gate_only_on_translation_success(
    monkeypatch, tmp_path: Path, translation, expected_skip
) -> None:
    """The baseline correctness gate is skipped iff a translation succeeded.

    Translation runs its own correctness + perf-regression check, so re-gating
    on the stricter harness-generator harness discards already-validated kernels
    (the FAIL_PREPROCESS-on-translation bug). The skip must stay scoped to
    translation runs: user-supplied harnesses (no translation, or a failed one)
    must still be gated.
    """
    import minisweagent.run.preprocess_v3.tools as tools_module

    harness = tmp_path / "harness.py"
    harness.write_text("print('GEAK_RESULT_LATENCY_MS=1.0')\n")

    captured: dict[str, object] = {}

    def fake_collect_baseline_metrics(harness_path, *, repeats, work_dir, gpu_id, skip_correctness_gate=False):
        captured["skip_correctness_gate"] = skip_correctness_gate
        return SimpleNamespace(
            success=True, median_ms=1.0, samples_ms=[1.0], stdev_ms=0.0,
            repeats=repeats, harness_path=harness_path, command="",
        )

    monkeypatch.setattr(tools_module, "collect_baseline_metrics", fake_collect_baseline_metrics)
    import minisweagent.run.preprocess_v3.baseline as baseline_module

    monkeypatch.setattr(baseline_module, "capture_full_benchmark_stdout", lambda *a, **k: None)

    agent = PreprocessOrchestratorAgent(
        model=object(),
        config=PreprocessOrchestratorConfig(repo=tmp_path),
    )
    if translation is not None:
        agent._collected["translation"] = translation

    tool = _make_tool_collect_baseline(agent)
    tool(harness_path=str(harness), repeats=1)

    assert captured["skip_correctness_gate"] is expected_skip


def test_dispatch_subagent_injects_deterministic_kernel_path(monkeypatch, tmp_path: Path) -> None:
    """The orchestrator hands the harness subagents the exact worktree-relative
    kernel path so they never have to guess it from the source tree."""
    import minisweagent.run.preprocess_v3.tools as tools_module

    # Keep the test focused on injection — no real sandbox copy.
    monkeypatch.setattr(tools_module, "_ensure_preprocess_subagent_sandbox", lambda agent: (None, {}))

    repo = tmp_path / "repo"
    (repo / "level3").mkdir(parents=True)
    kernel = repo / "level3" / "1_MLP.py"
    kernel.write_text("# kernel\n")

    captured: dict[str, object] = {}

    def fake_dispatcher(*, name, task, model, cwd=None, context=None):
        captured["context"] = context
        return {"name": name, "success": True, "output": "HARNESS_PATH: /tmp/harness.py"}

    agent = PreprocessOrchestratorAgent(model=object(), config=PreprocessOrchestratorConfig(repo=repo))
    agent._extra_template_vars = {"kernel_path": str(kernel), "repo_root": str(repo)}

    tool = _make_tool_dispatch_subagent(agent, fake_dispatcher)
    tool(name="harness-generator", task="make a harness")

    assert captured["context"]["kernel_relpath"] == "level3/1_MLP.py"
    assert captured["context"]["kernel_path"] == str(kernel)


def _failed_baseline(stderr: str) -> SimpleNamespace:
    return SimpleNamespace(
        success=False, median_ms=None, samples_ms=[], stdev_ms=None,
        repeats=0, command="cmd", raw_outputs=[{"stderr": stderr, "stdout": ""}],
    )


def test_detect_kernel_resolution_failure() -> None:
    from minisweagent.run.preprocess_v3.baseline import detect_kernel_resolution_failure

    raw = [{"stderr": "Traceback\nFileNotFoundError: [Errno 2] No such file or directory: '/x/k.py'\n", "stdout": ""}]
    msg = detect_kernel_resolution_failure(raw)
    assert msg is not None and "/x/k.py" in msg and "FileNotFoundError" in msg
    assert detect_kernel_resolution_failure([{"stderr": "TIMEOUT after 600s", "stdout": ""}]) is None


def test_collect_baseline_fail_closed_after_retry_budget(monkeypatch, tmp_path: Path) -> None:
    """An empty baseline after the generator retry budget is exhausted terminates
    the run with a precise error instead of spinning / running on a broken harness."""
    import minisweagent.run.preprocess_v3.tools as tools_module
    from minisweagent.run.preprocess_v3.orchestrator import FinishedSuccessfully

    harness = tmp_path / "harness.py"
    harness.write_text("x")
    monkeypatch.setattr(
        tools_module, "collect_baseline_metrics",
        lambda *a, **k: _failed_baseline("FileNotFoundError: No such file or directory: '/x/k.py'"),
    )

    agent = PreprocessOrchestratorAgent(model=object(), config=PreprocessOrchestratorConfig(repo=tmp_path))
    agent._collected["_harness_generator_attempts"] = 3
    tool = _make_tool_collect_baseline(agent)

    with pytest.raises(FinishedSuccessfully) as exc_info:
        tool(harness_path=str(harness), repeats=1)
    assert "/x/k.py" in exc_info.value.payload["errors"][0]


def test_collect_baseline_returns_precise_error_within_budget(monkeypatch, tmp_path: Path) -> None:
    """Before the retry budget is exhausted, an empty baseline returns ok=False
    with the precise kernel-resolution reason (so the generator can be retried)."""
    import minisweagent.run.preprocess_v3.tools as tools_module

    harness = tmp_path / "harness.py"
    harness.write_text("x")
    monkeypatch.setattr(
        tools_module, "collect_baseline_metrics",
        lambda *a, **k: _failed_baseline("FileNotFoundError: No such file or directory: '/x/k.py'"),
    )

    agent = PreprocessOrchestratorAgent(model=object(), config=PreprocessOrchestratorConfig(repo=tmp_path))
    agent._collected["_harness_generator_attempts"] = 1
    tool = _make_tool_collect_baseline(agent)

    res = tool(harness_path=str(harness), repeats=1)
    assert res["ok"] is False
    assert "/x/k.py" in res["error"]


def test_translate_retargets_preprocess_state_to_opt_repo(monkeypatch, tmp_path: Path) -> None:
    """After translation, the orchestrator's kernel_path/repo_root point at the
    per-run _opt_repo (where optimization runs), not the source repo — so the
    harness sandbox + baseline resolve the translated kernel."""
    import minisweagent.run.preprocess_v3.tools as tools_module
    from minisweagent.run.preprocess_v3.translate import TranslationResult

    src_repo = tmp_path / "src"
    src_repo.mkdir()
    orig = src_repo / "k.py"
    orig.write_text("# orig\n")
    out = tmp_path / "out"
    out.mkdir()
    cand_dir = tmp_path / "cand"
    cand_dir.mkdir()
    cand_file = cand_dir / "k_flydsl.py"
    cand_file.write_text("# flydsl\n")

    result = TranslationResult(
        success=True, target_language="flydsl", translated_kernel_path=cand_file,
        speedup=None, self_review="", errors=[], elapsed_s=0.0, raw={},
    )
    monkeypatch.setattr(tools_module, "translate_to_flydsl", lambda **k: result)

    agent = PreprocessOrchestratorAgent(model=object(), config=PreprocessOrchestratorConfig(repo=src_repo))
    agent._extra_template_vars = {"kernel_path": str(orig), "repo_root": str(src_repo)}

    tool = _make_tool_translate_to_flydsl(agent)
    tool(source_path=str(orig), output_dir=str(out))

    opt_repo = (out / "_opt_repo").resolve()
    assert agent._extra_template_vars["repo_root"] == str(opt_repo)
    assert agent._extra_template_vars["kernel_path"] == str((opt_repo / "k_flydsl.py").resolve())


def test_collect_baseline_defaults_work_dir_to_effective_repo_root(monkeypatch, tmp_path: Path) -> None:
    """collect_baseline runs the harness with work_dir = the effective repo root
    (retargeted to _opt_repo after translation) so the kernel is resolvable."""
    import minisweagent.run.preprocess_v3.baseline as baseline_module
    import minisweagent.run.preprocess_v3.tools as tools_module

    captured: dict[str, object] = {}

    def fake_collect_baseline_metrics(harness_path, *, repeats, work_dir, gpu_id, skip_correctness_gate=False):
        captured["work_dir"] = work_dir
        return SimpleNamespace(
            success=True, median_ms=1.0, samples_ms=[1.0], stdev_ms=0.0,
            repeats=repeats, harness_path=harness_path, command="", raw_outputs=[],
        )

    monkeypatch.setattr(tools_module, "collect_baseline_metrics", fake_collect_baseline_metrics)
    monkeypatch.setattr(baseline_module, "capture_full_benchmark_stdout", lambda *a, **k: None)

    harness = tmp_path / "h.py"
    harness.write_text("x")
    opt_repo = tmp_path / "_opt_repo"
    opt_repo.mkdir()

    agent = PreprocessOrchestratorAgent(model=object(), config=PreprocessOrchestratorConfig(repo=tmp_path / "src"))
    agent._extra_template_vars = {"repo_root": str(opt_repo)}

    tool = _make_tool_collect_baseline(agent)
    res = tool(harness_path=str(harness), repeats=1)

    assert res["ok"] is True
    assert captured["work_dir"] == opt_repo


def test_copy_repo_sandbox_copies_repo_living_under_output_dir(tmp_path: Path) -> None:
    """When the repo to sandbox is the per-run _opt_repo (which lives UNDER
    output_dir), its own files must be copied — not ignored by the output-dir
    recursion guard (which would leave an empty sandbox)."""
    from minisweagent.run.preprocess_v3.tools import _copy_repo_sandbox

    output_dir = tmp_path / "out"
    opt_repo = output_dir / "_opt_repo"
    opt_repo.mkdir(parents=True)
    (opt_repo / "1_MLP_flydsl.py").write_text("# flydsl\n")
    (opt_repo / "1_MLP.py").write_text("# ref\n")
    sandbox = output_dir / "_preprocess_subagent_worktree"

    _copy_repo_sandbox(opt_repo, sandbox, output_dir)

    assert (sandbox / "1_MLP_flydsl.py").is_file()
    assert (sandbox / "1_MLP.py").is_file()


def test_copy_repo_sandbox_still_skips_nested_output_dir(tmp_path: Path) -> None:
    """The recursion guard still fires when output_dir lives INSIDE the repo:
    the output tree must not be copied into the sandbox."""
    from minisweagent.run.preprocess_v3.tools import _copy_repo_sandbox

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.py").write_text("# k\n")
    output_dir = repo / "optimization_logs" / "run1"
    output_dir.mkdir(parents=True)
    (output_dir / "log.txt").write_text("noise\n")
    sandbox = tmp_path / "sandbox"

    _copy_repo_sandbox(repo, sandbox, output_dir)

    assert (sandbox / "kernel.py").is_file()
    assert not (sandbox / "optimization_logs" / "run1" / "log.txt").exists()


def test_verifier_backstop_marks_verified_when_correctness_passes(monkeypatch, tmp_path: Path) -> None:
    """If the LLM verifier fails to confirm but the harness passes --correctness,
    the deterministic backstop marks it HARNESS_VERIFIED so the orchestrator
    proceeds to baseline instead of looping the generator."""
    import minisweagent.run.preprocess_v3.baseline as baseline_module
    import minisweagent.run.preprocess_v3.tools as tools_module

    monkeypatch.setattr(tools_module, "_ensure_preprocess_subagent_sandbox", lambda agent: (None, {}))
    monkeypatch.setattr(
        baseline_module, "_run_benchmark_once",
        lambda *a, **k: {"returncode": 0, "stdout": "", "stderr": "", "duration_s": 1.0, "latency_ms": None},
    )

    def fake_dispatcher(*, name, task, model, cwd=None, context=None):
        return {"name": name, "success": False, "output": "could not confirm"}

    harness = tmp_path / "harness.py"
    harness.write_text("x")
    agent = PreprocessOrchestratorAgent(model=object(), config=PreprocessOrchestratorConfig(repo=tmp_path))
    agent._collected["harness_path"] = str(harness)
    agent._extra_template_vars = {"repo_root": str(tmp_path)}

    tool = _make_tool_dispatch_subagent(agent, fake_dispatcher)
    res = tool(name="harness-verifier", task="verify")

    assert res["success"] is True
    assert "HARNESS_VERIFIED=true" in res["output"]


def test_verifier_backstop_no_false_positive_when_correctness_fails(monkeypatch, tmp_path: Path) -> None:
    """The backstop must NOT mark a harness verified when --correctness fails."""
    import minisweagent.run.preprocess_v3.baseline as baseline_module
    import minisweagent.run.preprocess_v3.tools as tools_module

    monkeypatch.setattr(tools_module, "_ensure_preprocess_subagent_sandbox", lambda agent: (None, {}))
    monkeypatch.setattr(
        baseline_module, "_run_benchmark_once",
        lambda *a, **k: {"returncode": 1, "stdout": "", "stderr": "FileNotFoundError", "duration_s": 1.0, "latency_ms": None},
    )

    def fake_dispatcher(*, name, task, model, cwd=None, context=None):
        return {"name": name, "success": False, "output": "nope"}

    harness = tmp_path / "harness.py"
    harness.write_text("x")
    agent = PreprocessOrchestratorAgent(model=object(), config=PreprocessOrchestratorConfig(repo=tmp_path))
    agent._collected["harness_path"] = str(harness)
    agent._extra_template_vars = {"repo_root": str(tmp_path)}

    tool = _make_tool_dispatch_subagent(agent, fake_dispatcher)
    res = tool(name="harness-verifier", task="verify")

    assert res["success"] is False
    assert "HARNESS_VERIFIED=true" not in (res.get("output") or "")
