"""Tests for :mod:`minisweagent.run.preprocess_v3.adapter`.

The adapter is the call-site wrapper that translates the v3 orchestrator's
typed :class:`PreprocessResult` into the legacy ``preprocess_ctx`` dict
shape so ``run/mini.py`` and ``run/unified.py`` can keep working
unchanged after the cutover.

The tests drive ``run_preprocess_v3`` end-to-end with a mocked model +
mocked deterministic tools, and assert:

* On success → returns a legacy-shaped dict with every key the existing
  downstream consumers read.
* On failure → raises ``RuntimeError`` (the legacy preprocess failure
  type), preserving the surrounding pipeline's exception flow.
* Field-by-field projection of ``PreprocessResult`` -> ``preprocess_ctx``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3 import adapter as adapter_module
from minisweagent.run.preprocess_v3 import baseline as baseline_mod
from minisweagent.run.preprocess_v3 import commandment as commandment_mod
from minisweagent.run.preprocess_v3 import explore as explore_mod
from minisweagent.run.preprocess_v3 import tools as tools_mod
from minisweagent.run.preprocess_v3 import translate as translate_mod
from minisweagent.run.preprocess_v3.baseline import BaselineMetrics, ProfileResult
from minisweagent.run.preprocess_v3.explore import CodebaseContext

_TRITON_KERNEL = """import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
"""


_HARNESS_BODY = """\
#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--correctness', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--benchmark', action='store_true')
parser.add_argument('--full-benchmark', action='store_true')
args = parser.parse_args()
print('GEAK_RESULT_LATENCY_MS=2.5')
"""


class _ScriptedModel:
    """Same shape as the integration-test scripted model."""

    def __init__(self, script):
        self.script = list(script)
        self.n_calls = 0
        self.cost = 0.0
        self._cid = 0

    def set_tools(self, _):
        pass

    def query(self, _messages):
        self.n_calls += 1
        if not self.script:
            return {"content": "all done"}
        entry = self.script.pop(0)
        if isinstance(entry, dict):
            return entry
        name, args = entry
        self._cid += 1
        return {
            "content": "",
            "tools": {
                "id": f"call_{self._cid}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            },
        }


def _patch_deterministic_tools(monkeypatch, harness_path: Path) -> None:
    """Same as the integration-test patches but with one shared body."""

    def _fake_explore(repo_root, kernel_path, kernel_language, *, out_path=None):
        text = f"# Codebase Context\n\nKernel: {kernel_path}\n"
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(text, encoding="utf-8")
        return CodebaseContext(
            text=text,
            files=[str(kernel_path)],
            out_path=Path(out_path) if out_path else None,
            kernel_language=kernel_language,
        )

    def _fake_baseline(harness_path_arg, *, repeats=5, work_dir=None, gpu_id=0):
        return BaselineMetrics(
            harness_path=Path(harness_path_arg),
            median_ms=3.14,
            samples_ms=[3.1, 3.14, 3.2],
            stdev_ms=0.05,
            repeats=repeats,
            command="bash -lc 'python harness.py --benchmark'",
            raw_outputs=[{"returncode": 0}],
        )

    def _fake_profile(harness_path_arg, *, work_dir=None, gpu_id=0, backend="metrix", **_kw):
        return ProfileResult(
            harness_path=Path(harness_path_arg),
            command="python3 harness.py --profile",
            profile={"success": True, "kernels": [{"name": "add"}]},
            profile_path=None,
            backend=backend,
        )

    def _fake_commandment(kernel_language, ctx, *, out_path=None):
        text = "# COMMANDMENT.md\n\nDo the thing.\n"
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(text, encoding="utf-8")
        return text

    def _fail_translate(**_):
        pytest.fail("translate_to_flydsl called in a same-language adapter test")

    monkeypatch.setattr(explore_mod, "explore_codebase", _fake_explore)
    monkeypatch.setattr(tools_mod, "explore_codebase", _fake_explore)
    monkeypatch.setattr(baseline_mod, "collect_baseline_metrics", _fake_baseline)
    monkeypatch.setattr(tools_mod, "collect_baseline_metrics", _fake_baseline)
    monkeypatch.setattr(baseline_mod, "collect_profile", _fake_profile)
    monkeypatch.setattr(tools_mod, "collect_profile", _fake_profile)
    monkeypatch.setattr(commandment_mod, "render_commandment", _fake_commandment)
    monkeypatch.setattr(tools_mod, "render_commandment", _fake_commandment)
    monkeypatch.setattr(translate_mod, "translate_to_flydsl", _fail_translate)
    monkeypatch.setattr(tools_mod, "translate_to_flydsl", _fail_translate)


def _build_fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()  # so _infer_repo_root finds it
    kernel = repo / "add_kernel.py"
    kernel.write_text(_TRITON_KERNEL, encoding="utf-8")
    return repo, kernel


def _happy_path_script(*, repo: Path, kernel_path: Path, out_dir: Path, harness_path: Path):
    commandment_path = out_dir / "COMMANDMENT.md"
    codebase_context_path = out_dir / "CODEBASE_CONTEXT.md"
    return [
        (
            "codebase_explore",
            {
                "repo_root": str(repo),
                "kernel_path": str(kernel_path),
                "out_path": str(codebase_context_path),
            },
        ),
        ("dispatch_subagent", {"name": "harness-generator", "task": "gen"}),
        ("dispatch_subagent", {"name": "harness-verifier", "task": "verify"}),
        ("collect_baseline", {"harness_path": str(harness_path)}),
        ("collect_profile", {"harness_path": str(harness_path)}),
        ("dispatch_subagent", {"name": "speedup-verify", "task": "speedup"}),
        (
            "render_commandment",
            {
                "kernel_path": str(kernel_path),
                "harness_path": str(harness_path),
                "repo_root": str(repo),
                "out_path": str(commandment_path),
            },
        ),
        (
            "finish_preprocess",
            {
                "harness_path": str(harness_path),
                "commandment_path": str(commandment_path),
                "errors": [],
                "summary": "Run successful.",
            },
        ),
    ]


def _happy_dispatcher(repo: Path, harness_path: Path):
    def _stub(*, name, task, model, cwd=None, context=None):
        if name == "harness-generator":
            return {
                "name": name,
                "success": True,
                "output": (
                    f"TEST_COMMAND: cd {repo} && python {harness_path} --correctness "
                    f"&& python {harness_path} --benchmark"
                ),
                "elapsed_s": 1.0,
            }
        if name == "harness-verifier":
            return {
                "name": name,
                "success": True,
                "output": f"HARNESS_VERIFIED=true\nHARNESS_PATH={harness_path}",
                "elapsed_s": 0.5,
            }
        if name == "speedup-verify":
            return {"name": name, "success": True, "output": "ok", "elapsed_s": 0.3}
        return {"name": name, "success": False, "output": "unknown"}

    return _stub


def _build_orchestrator(model, dispatcher, kernel_language):
    """Build an orchestrator with stub dispatcher injected.

    The adapter normally constructs the orchestrator + registers default
    tools internally. We mirror that here but with a stub dispatcher so
    the LLM tool calls route through canned responses.
    """
    from minisweagent.run.preprocess_v3.orchestrator import (
        PreprocessOrchestratorAgent,
        PreprocessOrchestratorConfig,
    )
    from minisweagent.run.preprocess_v3.registry import SubagentRegistry
    from minisweagent.run.preprocess_v3.tools import register_default_tools

    agent = PreprocessOrchestratorAgent(model=model, config=PreprocessOrchestratorConfig())
    register_default_tools(
        agent,
        kernel_language=kernel_language,
        registry=SubagentRegistry(root=Path("/nonexistent")),
        dispatcher=dispatcher,
    )
    return agent


@pytest.fixture
def patched_orchestrator(monkeypatch):
    """Replace ``register_default_tools`` so dispatcher hook is injectable."""
    captured: dict = {}

    real_register = adapter_module.register_default_tools

    def _patched(agent, *, kernel_language, registry=None, dispatcher=None):
        # Defer to caller-provided dispatcher when ``captured`` carries one.
        return real_register(
            agent,
            kernel_language=kernel_language,
            registry=registry,
            dispatcher=captured.get("dispatcher", dispatcher),
        )

    monkeypatch.setattr(adapter_module, "register_default_tools", _patched)
    return captured


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_run_preprocess_v3_returns_legacy_shaped_dict(tmp_path, monkeypatch, patched_orchestrator):
    """Happy path: every key downstream consumers read is populated."""
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)
    patched_orchestrator["dispatcher"] = _happy_dispatcher(repo, harness_path)

    model = _ScriptedModel(
        _happy_path_script(repo=repo, kernel_path=kernel_path, out_dir=out_dir, harness_path=harness_path)
    )

    ctx = adapter_module.run_preprocess_v3(
        kernel_url=str(kernel_path),
        output_dir=out_dir,
        gpu_id=2,
        model=model,
        repo=str(repo),
        user_task="Optimize add_kernel for B=1024, D=128.",
    )

    # Required-by-downstream keys are all present.
    assert ctx["kernel_path"] == str(kernel_path.resolve())
    assert ctx["repo_root"] == str(repo.resolve())
    assert ctx["output_dir"] == str(out_dir.resolve())
    assert ctx["kernel_type"] == "triton"
    assert ctx["discovery"]["kernel"]["type"] == "triton"
    assert ctx["discovery"]["kernel"]["path"] == str(kernel_path.resolve())
    assert ctx["harness_path"] == str(harness_path)
    assert ctx["test_command"] is not None and "python" in ctx["test_command"]
    assert ctx["commandment"] is not None and "COMMANDMENT" in ctx["commandment"]
    assert ctx["commandment_path"] == str(out_dir / "COMMANDMENT.md")
    assert ctx["baseline_metrics"]["median_ms"] == pytest.approx(3.14)
    assert ctx["baseline_metrics_path"] == str(out_dir / "baseline_metrics.json")
    assert ctx["profiling"]["success"] is True
    assert ctx["codebase_context_path"] == str(out_dir / "CODEBASE_CONTEXT.md")

    # Legacy fields we don't populate are explicit Nones, not missing.
    assert "harness_results" in ctx
    assert "kernel_analysis_md" in ctx
    assert ctx["evaluation_contract"] is None

    # v3-extras carried for the validation runbook / debugging.
    assert "v3_subagent_runs" in ctx
    assert ctx["v3_subagent_runs"]  # at least one dispatched
    assert "v3_elapsed_s" in ctx


def test_run_preprocess_v3_writes_baseline_metrics_json(tmp_path, monkeypatch, patched_orchestrator):
    """The adapter writes ``baseline_metrics.json`` so the file matches the path it returns."""
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)
    patched_orchestrator["dispatcher"] = _happy_dispatcher(repo, harness_path)

    model = _ScriptedModel(
        _happy_path_script(repo=repo, kernel_path=kernel_path, out_dir=out_dir, harness_path=harness_path)
    )

    ctx = adapter_module.run_preprocess_v3(
        kernel_url=str(kernel_path),
        output_dir=out_dir,
        model=model,
        repo=str(repo),
    )

    written = Path(ctx["baseline_metrics_path"])
    assert written.is_file()
    payload = json.loads(written.read_text())
    assert payload["median_ms"] == pytest.approx(3.14)
    assert payload["samples_ms"] == [3.1, 3.14, 3.2]


def test_run_preprocess_v3_extracts_test_command_from_subagent_output(tmp_path, monkeypatch, patched_orchestrator):
    """``test_command`` is recovered from ``harness-generator``'s output line."""
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)
    patched_orchestrator["dispatcher"] = _happy_dispatcher(repo, harness_path)

    model = _ScriptedModel(
        _happy_path_script(repo=repo, kernel_path=kernel_path, out_dir=out_dir, harness_path=harness_path)
    )

    ctx = adapter_module.run_preprocess_v3(
        kernel_url=str(kernel_path),
        output_dir=out_dir,
        model=model,
        repo=str(repo),
    )
    assert ctx["test_command"].startswith(f"cd {repo}")
    assert "--correctness" in ctx["test_command"]


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_run_preprocess_v3_raises_runtime_error_on_failure(tmp_path, monkeypatch, patched_orchestrator):
    """When ``PreprocessResult.success is False``, the adapter raises RuntimeError.

    The surrounding pipeline's ``except RuntimeError`` clauses (the
    legacy preprocess failure-handling path) keep working — the
    exception type is the same.
    """
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)

    def _always_reject(*, name, task, model, cwd=None, context=None):
        if name == "harness-verifier":
            return {"name": name, "success": False, "output": "HARNESS_VERIFIED=false", "elapsed_s": 0.1}
        return {"name": name, "success": True, "output": "ok", "elapsed_s": 0.1}

    patched_orchestrator["dispatcher"] = _always_reject

    codebase_context_path = out_dir / "CODEBASE_CONTEXT.md"
    model = _ScriptedModel(
        [
            (
                "codebase_explore",
                {
                    "repo_root": str(repo),
                    "kernel_path": str(kernel_path),
                    "out_path": str(codebase_context_path),
                },
            ),
            ("dispatch_subagent", {"name": "harness-generator", "task": "gen"}),
            ("dispatch_subagent", {"name": "harness-verifier", "task": "verify"}),
            (
                "finish_preprocess",
                {
                    "errors": ["harness-verifier rejected; giving up"],
                    "summary": "Give up.",
                },
            ),
        ]
    )

    with pytest.raises(RuntimeError) as excinfo:
        adapter_module.run_preprocess_v3(
            kernel_url=str(kernel_path),
            output_dir=out_dir,
            model=model,
            repo=str(repo),
        )
    assert "harness-verifier rejected" in str(excinfo.value)


def test_run_preprocess_v3_raises_runtime_error_when_no_model(tmp_path):
    """Missing ``model`` (and no ``model_factory``) is a configuration error."""
    repo, kernel_path = _build_fixture_repo(tmp_path)
    with pytest.raises(RuntimeError, match="model"):
        adapter_module.run_preprocess_v3(
            kernel_url=str(kernel_path),
            output_dir=tmp_path / "out",
            model=None,
            model_factory=None,
            repo=str(repo),
        )


# ---------------------------------------------------------------------------
# Translation field projection
# ---------------------------------------------------------------------------


def test_translation_field_skipped_when_same_language(tmp_path, monkeypatch, patched_orchestrator):
    """No translation -> ``v3_translation`` key is absent."""
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)
    patched_orchestrator["dispatcher"] = _happy_dispatcher(repo, harness_path)

    model = _ScriptedModel(
        _happy_path_script(repo=repo, kernel_path=kernel_path, out_dir=out_dir, harness_path=harness_path)
    )

    ctx = adapter_module.run_preprocess_v3(
        kernel_url=str(kernel_path),
        output_dir=out_dir,
        model=model,
        repo=str(repo),
    )
    assert "v3_translation" not in ctx


# ---------------------------------------------------------------------------
# Language detection wiring
# ---------------------------------------------------------------------------


def test_local_kernel_detects_triton_language(tmp_path, monkeypatch, patched_orchestrator):
    """The adapter forwards the detected ``KernelLanguage`` into the orchestrator inputs.

    Pre-step-0b runs ``preprocess_v3.lang.detect_language`` on the
    kernel path so the orchestrator's ``codebase_explore`` and
    ``render_commandment`` calls get the right language object. We
    verify this end-to-end by checking the projected ``kernel_type``
    in the returned dict.
    """
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)
    patched_orchestrator["dispatcher"] = _happy_dispatcher(repo, harness_path)

    model = _ScriptedModel(
        _happy_path_script(repo=repo, kernel_path=kernel_path, out_dir=out_dir, harness_path=harness_path)
    )

    ctx = adapter_module.run_preprocess_v3(
        kernel_url=str(kernel_path),
        output_dir=out_dir,
        model=model,
        repo=str(repo),
    )
    assert ctx["kernel_type"] == "triton"


# ---------------------------------------------------------------------------
# Wiring check (sanity)
# ---------------------------------------------------------------------------


def test_run_preprocessor_alias_in_mini_points_at_v3() -> None:
    """``mini.py`` imports the v3 adapter under the ``run_preprocessor`` name."""
    from minisweagent.run import mini as mini_module
    from minisweagent.run.preprocess_v3.adapter import run_preprocess_v3

    assert mini_module.run_preprocessor is run_preprocess_v3


# ---------------------------------------------------------------------------
# Result-to-context helper (unit-level)
# ---------------------------------------------------------------------------


def test_preprocess_result_to_legacy_context_handles_empty_baseline(tmp_path):
    """Direct unit test for the projection helper."""
    from minisweagent.run.preprocess_v3.orchestrator import PreprocessResult

    lang = KernelLanguage(
        name="hip",
        file_extensions=frozenset({".hip"}),
        detect_hints=(),
        kb_namespace="hip",
    )
    result = PreprocessResult(
        success=True,
        kernel_language=lang,
        kernel_path=Path("/tmp/k.hip"),
        harness_path=Path("/tmp/harness.py"),
        baseline=None,  # no metrics — adapter should not write the file
    )
    ctx = adapter_module._preprocess_result_to_legacy_context(
        result=result,
        repo_root="/tmp/repo",
        output_dir=tmp_path,
        kernel_path_input=Path("/tmp/k.hip"),
    )
    assert ctx["baseline_metrics"] is None
    assert ctx["baseline_metrics_path"] is None
    assert ctx["kernel_type"] == "hip"
    assert not (tmp_path / "baseline_metrics.json").exists()
