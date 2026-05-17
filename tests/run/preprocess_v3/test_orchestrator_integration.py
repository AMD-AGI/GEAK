"""End-to-end integration tests for the v3 preprocess orchestrator.

These tests drive :class:`PreprocessOrchestratorAgent.run` with a mocked
LLM model, mocked deterministic tool callables, and a stub
:class:`PreprocessSubagentDispatcher`. The goal is to verify the
**orchestration loop** — message flow, tool call sequencing, retry
semantics for harness-verifier, and final ``PreprocessResult``
assembly — without exercising any real subprocess, GPU, or LLM.

What we deliberately do NOT cover:

* Real ``run_translation`` end-to-end. Translation is mocked to a
  no-op-success here; commit set 5 ships the full FlyDSL-translate
  flow with the real legacy ``run_translation``.
* Real harness execution. Baseline / profile are mocked to canned
  metrics so we don't shell out.
* Real ``DefaultAgent`` subagent runs. The dispatcher is stubbed to
  return canned outputs.

The tests use the exact ``register_default_tools`` registration helper
the production agent uses; only the *underlying* deterministic functions
are replaced via monkeypatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3 import baseline as baseline_mod
from minisweagent.run.preprocess_v3 import commandment as commandment_mod
from minisweagent.run.preprocess_v3 import explore as explore_mod
from minisweagent.run.preprocess_v3 import tools as tools_mod
from minisweagent.run.preprocess_v3 import translate as translate_mod
from minisweagent.run.preprocess_v3.baseline import BaselineMetrics, ProfileResult
from minisweagent.run.preprocess_v3.explore import CodebaseContext
from minisweagent.run.preprocess_v3.orchestrator import (
    PreprocessOrchestratorAgent,
    PreprocessOrchestratorConfig,
)
from minisweagent.run.preprocess_v3.registry import SubagentRegistry
from minisweagent.run.preprocess_v3.tools import (
    PreprocessSubagentDispatcher,
    register_default_tools,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_LANG = KernelLanguage(
    name="triton",
    file_extensions=frozenset({".py"}),
    detect_hints=(r"@triton\.jit",),
    kb_namespace="triton",
)


_TRITON_KERNEL = '''"""Tiny Triton add kernel for the integration fixture."""

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
'''


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


def _build_fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Drop a synthetic 3-file repo. Returns (repo_root, kernel_path)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "__init__.py").write_text("", encoding="utf-8")
    kernel_path = repo / "add_kernel.py"
    kernel_path.write_text(_TRITON_KERNEL, encoding="utf-8")
    (repo / "README.md").write_text("# Repo\n", encoding="utf-8")
    return repo, kernel_path


class _ScriptedModel:
    """LLM stub that emits a scripted sequence of tool calls.

    Each entry in ``script`` is either:

    * a ``dict`` with ``tools`` already populated — used as-is.
    * a ``(name, args_dict)`` tuple — wrapped into the OpenAI tool-call
      shape automatically.

    After the script runs out, returns a "no tool call" response so the
    orchestrator escalates to ``FormatError`` (which the test then
    asserts didn't happen — i.e. the script was complete).
    """

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.queries: list[list[dict]] = []
        self.n_calls = 0
        self.cost = 0.0
        self._call_id = 0

    def set_tools(self, schemas: list) -> None:
        pass

    def query(self, messages):
        self.queries.append(list(messages))
        self.n_calls += 1
        if not self.script:
            return {"content": "All done."}

        entry = self.script.pop(0)
        if isinstance(entry, dict):
            return entry
        name, args = entry
        self._call_id += 1
        import json as _json

        return {
            "content": "",
            "tools": {
                "id": f"call_{self._call_id}",
                "type": "function",
                "function": {"name": name, "arguments": _json.dumps(args)},
            },
        }


def _make_dispatcher_with_canned_responses(responses_by_name: dict) -> PreprocessSubagentDispatcher:
    """Build a stub dispatcher returning canned subagent results."""

    class _Stub:
        def __init__(self, registry):
            self._registry = registry

        def __call__(self, *, name, task, model, cwd=None, context=None):
            response = responses_by_name.get(name, {"name": name, "success": True, "output": ""})
            # Track per-call invocations on the registry so retry tests can count.
            return dict(response, name=name)

    return _Stub(None)  # type: ignore[return-value]


@pytest.fixture
def empty_registry(tmp_path: Path) -> SubagentRegistry:
    """Empty SubagentRegistry rooted at an empty tmp dir.

    The dispatcher is mocked so the registry is never actually consulted
    in these tests — we just need a valid object to satisfy the type.
    """
    return SubagentRegistry(root=tmp_path / "empty_subagents")


# ---------------------------------------------------------------------------
# Mocks for the deterministic tools
# ---------------------------------------------------------------------------


def _patch_deterministic_tools(monkeypatch, *, harness_path: Path, profile_path: Path | None = None):
    """Patch the v3 modules so deterministic tools never shell out.

    ``explore_codebase`` returns a minimal :class:`CodebaseContext`,
    ``collect_baseline_metrics`` returns canned latency, ``collect_profile``
    returns a canned :class:`ProfileResult`, ``render_commandment`` writes
    a stub COMMANDMENT.md.

    ``translate_to_flydsl`` is patched too even though the integration
    test explicitly skips translation — we want a hard fail if the
    orchestrator tries to call it.
    """

    def _fake_explore(repo_root, kernel_path, kernel_language, *, out_path=None):
        text = "# Codebase Context\n\n## Files\n- add_kernel.py\n"
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
            median_ms=2.5,
            samples_ms=[2.4, 2.5, 2.6],
            stdev_ms=0.1,
            repeats=repeats,
            command="bash -lc 'python harness.py --benchmark'",
            raw_outputs=[{"returncode": 0, "stdout": "ok", "stderr": "", "duration_s": 0.1, "latency_ms": 2.5}],
        )

    def _fake_profile(
        harness_path_arg, *, work_dir=None, gpu_id=0, backend="metrix", num_replays=3, quick=False, out_path=None
    ):
        return ProfileResult(
            harness_path=Path(harness_path_arg),
            command="python3 harness.py --profile",
            profile={"success": True, "kernels": []},
            profile_path=Path(out_path) if out_path else profile_path,
            backend=backend,
        )

    def _fake_render_commandment(kernel_language, ctx, *, out_path=None):
        text = f"# COMMANDMENT\n\nKernel: {ctx.kernel_path}\nHarness: {ctx.harness_path}\n"
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(text, encoding="utf-8")
        return text

    def _fail_translate(**_):
        pytest.fail("translate_to_flydsl was called in a same-language integration test")

    monkeypatch.setattr(explore_mod, "explore_codebase", _fake_explore)
    monkeypatch.setattr(tools_mod, "explore_codebase", _fake_explore)
    monkeypatch.setattr(baseline_mod, "collect_baseline_metrics", _fake_baseline)
    monkeypatch.setattr(tools_mod, "collect_baseline_metrics", _fake_baseline)
    monkeypatch.setattr(baseline_mod, "collect_profile", _fake_profile)
    monkeypatch.setattr(tools_mod, "collect_profile", _fake_profile)
    monkeypatch.setattr(commandment_mod, "render_commandment", _fake_render_commandment)
    monkeypatch.setattr(tools_mod, "render_commandment", _fake_render_commandment)
    monkeypatch.setattr(translate_mod, "translate_to_flydsl", _fail_translate)
    monkeypatch.setattr(tools_mod, "translate_to_flydsl", _fail_translate)


# ---------------------------------------------------------------------------
# Happy-path test
# ---------------------------------------------------------------------------


def test_orchestrator_happy_path_completes_all_six_steps(
    tmp_path: Path,
    monkeypatch,
    empty_registry: SubagentRegistry,
) -> None:
    """Drive the full 6-step flow with a verifier that passes on attempt 1.

    Asserts:
    * ``PreprocessResult.success`` is True.
    * Every step's artifact path is populated.
    * The codebase_explore tool returned a populated text.
    * Two subagent runs (generator + verifier + speedup) are recorded.
    * Translation field is None (skipped).
    * The dispatcher was called exactly 3 times (gen, verify, speedup).
    """
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)

    dispatcher_calls: list[dict] = []

    class _CountingDispatcher:
        def __init__(self, registry):
            self.registry = registry

        def __call__(self, *, name, task, model, cwd=None, context=None):
            dispatcher_calls.append({"name": name, "task": task[:80]})
            if name == "harness-generator":
                return {
                    "name": name,
                    "success": True,
                    "output": f"TEST_COMMAND: cd {repo} && python {harness_path} --correctness && python {harness_path} --benchmark",
                    "elapsed_s": 1.0,
                }
            if name == "harness-verifier":
                return {
                    "name": name,
                    "success": True,
                    "output": (
                        "HARNESS_VERIFIED=true\n"
                        f"HARNESS_PATH={harness_path}\n"
                        "MODES_PASSED=correctness,profile,benchmark,full-benchmark\n"
                        "WARNINGS=none"
                    ),
                    "elapsed_s": 0.5,
                }
            if name == "speedup-verify":
                return {
                    "name": name,
                    "success": True,
                    "output": f"SPEEDUP_SCRIPT_PATH: {out_dir / 'compute_speedup.py'}",
                    "elapsed_s": 0.3,
                }
            return {"name": name, "success": False, "output": "unknown"}

    commandment_path = out_dir / "COMMANDMENT.md"
    codebase_context_path = out_dir / "CODEBASE_CONTEXT.md"

    script = [
        (
            "codebase_explore",
            {
                "repo_root": str(repo),
                "kernel_path": str(kernel_path),
                "out_path": str(codebase_context_path),
            },
        ),
        (
            "dispatch_subagent",
            {
                "name": "harness-generator",
                "task": "Build a harness for the kernel.",
            },
        ),
        (
            "dispatch_subagent",
            {
                "name": "harness-verifier",
                "task": "Verify the harness.",
            },
        ),
        ("collect_baseline", {"harness_path": str(harness_path)}),
        ("collect_profile", {"harness_path": str(harness_path)}),
        (
            "dispatch_subagent",
            {
                "name": "speedup-verify",
                "task": "Generate compute_speedup.py.",
            },
        ),
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

    model = _ScriptedModel(script)
    agent = PreprocessOrchestratorAgent(
        model=model,
        config=PreprocessOrchestratorConfig(step_limit=50, cost_limit=0.0),
    )
    register_default_tools(
        agent,
        kernel_language=_LANG,
        registry=empty_registry,
        dispatcher=_CountingDispatcher(empty_registry),
    )

    result = agent.run(
        task="Run the v3 preprocess flow on this fixture.",
        kernel_path=kernel_path,
        repo_root=repo,
        kernel_language=_LANG,
        source_language="triton",
        target_language="triton",
        output_dir=out_dir,
        gpu_id=0,
    )

    assert result.success is True, f"unexpected errors: {result.errors}"
    assert result.kernel_path == kernel_path
    assert result.kernel_language == _LANG
    assert result.harness_path == harness_path
    assert result.commandment_path == commandment_path
    assert result.codebase_context is not None
    assert result.codebase_context.text.startswith("# Codebase Context")
    assert result.baseline is not None
    assert result.baseline.median_ms == pytest.approx(2.5)
    assert result.profile is not None
    assert result.profile.success is True
    assert result.translation is None
    # 3 dispatcher calls: harness-generator, harness-verifier, speedup-verify.
    assert [c["name"] for c in dispatcher_calls] == [
        "harness-generator",
        "harness-verifier",
        "speedup-verify",
    ]
    # 3 subagent runs recorded on the result.
    assert [r["name"] for r in result.subagent_runs] == [
        "harness-generator",
        "harness-verifier",
        "speedup-verify",
    ]
    # No script entries left -> no truncation, no leftover steps.
    assert model.script == []


# ---------------------------------------------------------------------------
# Sad-path test: verifier rejects on attempt 1
# ---------------------------------------------------------------------------


def test_orchestrator_retries_harness_after_verifier_rejection(
    tmp_path: Path,
    monkeypatch,
    empty_registry: SubagentRegistry,
) -> None:
    """The orchestrator may re-dispatch harness-generator when verification fails.

    Sequence under test:
    1. harness-generator (attempt 1) -> output a TEST_COMMAND.
    2. harness-verifier -> rejects (HARNESS_VERIFIED=false, ESCALATE=false).
    3. harness-generator (attempt 2) -> revised output.
    4. harness-verifier -> approves.
    5. baseline / profile / speedup-verify / commandment / finish.

    Asserts:
    * Two harness-generator calls were made.
    * Two harness-verifier calls were made.
    * The retry's task string carries the verifier's feedback.
    * Final ``result.success`` is True.
    """
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)

    state = {"verifier_call_count": 0, "generator_call_count": 0}
    dispatcher_history: list[dict] = []

    class _RetryDispatcher:
        def __init__(self, registry):
            self.registry = registry

        def __call__(self, *, name, task, model, cwd=None, context=None):
            dispatcher_history.append({"name": name, "task": task})
            if name == "harness-generator":
                state["generator_call_count"] += 1
                return {
                    "name": name,
                    "success": True,
                    "output": f"TEST_COMMAND: cd {repo} && python {harness_path} --correctness",
                    "elapsed_s": 1.0,
                }
            if name == "harness-verifier":
                state["verifier_call_count"] += 1
                if state["verifier_call_count"] == 1:
                    return {
                        "name": name,
                        "success": False,
                        "output": (
                            "HARNESS_VERIFIED=false\n"
                            "ESCALATE=false\n"
                            "PHASE=runtime\n"
                            "FAILED_RULE=missing-latency-marker\n"
                            "FAILED_MODE=benchmark\n"
                            "EVIDENCE=last line was 'done' not 'GEAK_RESULT_LATENCY_MS=...'\n"
                            "CORRECTION_HINT=ensure --benchmark prints GEAK_RESULT_LATENCY_MS as last line"
                        ),
                        "elapsed_s": 0.4,
                    }
                return {
                    "name": name,
                    "success": True,
                    "output": (
                        "HARNESS_VERIFIED=true\n"
                        f"HARNESS_PATH={harness_path}\n"
                        "MODES_PASSED=correctness,profile,benchmark,full-benchmark"
                    ),
                    "elapsed_s": 0.5,
                }
            if name == "speedup-verify":
                return {"name": name, "success": True, "output": "ok", "elapsed_s": 0.3}
            return {"name": name, "success": False, "output": "unknown"}

    commandment_path = out_dir / "COMMANDMENT.md"
    codebase_context_path = out_dir / "CODEBASE_CONTEXT.md"

    script = [
        (
            "codebase_explore",
            {
                "repo_root": str(repo),
                "kernel_path": str(kernel_path),
                "out_path": str(codebase_context_path),
            },
        ),
        (
            "dispatch_subagent",
            {
                "name": "harness-generator",
                "task": "Build a harness for the kernel.",
            },
        ),
        (
            "dispatch_subagent",
            {
                "name": "harness-verifier",
                "task": "Verify the harness.",
            },
        ),
        # Verifier rejected -> retry generator with the correction hint.
        (
            "dispatch_subagent",
            {
                "name": "harness-generator",
                "task": (
                    "Re-build the harness. Verifier feedback: ensure --benchmark prints "
                    "GEAK_RESULT_LATENCY_MS as last line"
                ),
            },
        ),
        (
            "dispatch_subagent",
            {
                "name": "harness-verifier",
                "task": "Verify the new harness.",
            },
        ),
        ("collect_baseline", {"harness_path": str(harness_path)}),
        ("collect_profile", {"harness_path": str(harness_path)}),
        (
            "dispatch_subagent",
            {
                "name": "speedup-verify",
                "task": "Generate compute_speedup.py.",
            },
        ),
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
                "summary": "Run successful after one retry.",
            },
        ),
    ]

    model = _ScriptedModel(script)
    agent = PreprocessOrchestratorAgent(model=model)
    register_default_tools(
        agent,
        kernel_language=_LANG,
        registry=empty_registry,
        dispatcher=_RetryDispatcher(empty_registry),
    )

    result = agent.run(
        task="Run preprocess with a verifier rejection.",
        kernel_path=kernel_path,
        repo_root=repo,
        kernel_language=_LANG,
        source_language="triton",
        target_language="triton",
        output_dir=out_dir,
        gpu_id=0,
    )

    assert result.success is True, f"errors: {result.errors}"
    assert state["generator_call_count"] == 2, f"expected 2 generator dispatches, got {state['generator_call_count']}"
    assert state["verifier_call_count"] == 2, f"expected 2 verifier dispatches, got {state['verifier_call_count']}"

    # The retry's task carries the verifier's CORRECTION_HINT. The
    # dispatcher_history sequence is:
    #   [0] harness-generator (attempt 1)
    #   [1] harness-verifier  (rejects)
    #   [2] harness-generator (attempt 2 — retry with feedback)
    #   [3] harness-verifier  (approves)
    #   [4] speedup-verify
    retry_task = dispatcher_history[2]["task"]
    assert dispatcher_history[2]["name"] == "harness-generator"
    assert "GEAK_RESULT_LATENCY_MS" in retry_task

    # Subagent runs include all 5 dispatches (2 gen + 2 verify + 1 speedup).
    names = [r["name"] for r in result.subagent_runs]
    assert names.count("harness-generator") == 2
    assert names.count("harness-verifier") == 2
    assert names.count("speedup-verify") == 1


# ---------------------------------------------------------------------------
# Bound check: the orchestrator stops re-dispatching after the 3-attempt cap
# ---------------------------------------------------------------------------


def test_orchestrator_does_not_loop_forever_when_verifier_keeps_rejecting(
    tmp_path: Path,
    monkeypatch,
    empty_registry: SubagentRegistry,
) -> None:
    """Sad-path bound: with the LLM script capping itself at 3 generator
    attempts, the orchestrator records the failure and proceeds.

    The retry-loop bound is encoded in the LLM's prompt, not in the
    orchestrator's Python code (the system prompt says "Maximum 3
    generator attempts total"). Here we model an LLM that follows that
    instruction: 3 generator dispatches, then it gives up and calls
    finish_preprocess with the failure recorded in errors.
    """
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)

    class _AlwaysFailingDispatcher:
        def __init__(self, registry):
            self.registry = registry

        def __call__(self, *, name, task, model, cwd=None, context=None):
            if name == "harness-verifier":
                return {
                    "name": name,
                    "success": False,
                    "output": ("HARNESS_VERIFIED=false\nESCALATE=false\nPHASE=runtime\nFAILED_RULE=foo"),
                    "elapsed_s": 0.4,
                }
            return {"name": name, "success": True, "output": "ok", "elapsed_s": 1.0}

    codebase_context_path = out_dir / "CODEBASE_CONTEXT.md"
    script = [
        (
            "codebase_explore",
            {
                "repo_root": str(repo),
                "kernel_path": str(kernel_path),
                "out_path": str(codebase_context_path),
            },
        ),
        ("dispatch_subagent", {"name": "harness-generator", "task": "attempt 1"}),
        ("dispatch_subagent", {"name": "harness-verifier", "task": "verify 1"}),
        ("dispatch_subagent", {"name": "harness-generator", "task": "attempt 2"}),
        ("dispatch_subagent", {"name": "harness-verifier", "task": "verify 2"}),
        ("dispatch_subagent", {"name": "harness-generator", "task": "attempt 3"}),
        ("dispatch_subagent", {"name": "harness-verifier", "task": "verify 3"}),
        # LLM honours the 3-attempt cap and finishes with the error in the bag.
        (
            "finish_preprocess",
            {
                "errors": ["harness-verifier rejected 3 attempts; proceeding without harness"],
                "summary": "Verifier never approved the harness.",
            },
        ),
    ]

    model = _ScriptedModel(script)
    agent = PreprocessOrchestratorAgent(model=model)
    register_default_tools(
        agent,
        kernel_language=_LANG,
        registry=empty_registry,
        dispatcher=_AlwaysFailingDispatcher(empty_registry),
    )

    result = agent.run(
        task="Stress the retry path.",
        kernel_path=kernel_path,
        repo_root=repo,
        kernel_language=_LANG,
        source_language="triton",
        target_language="triton",
        output_dir=out_dir,
        gpu_id=0,
    )

    # Commit set 5a (fix for open question #7): when the LLM gracefully
    # calls ``finish_preprocess(errors=[...])`` because the harness
    # verifier rejected every attempt, the orchestrator MUST surface that
    # as a failure. Pre-fix this asserted ``success is True`` because the
    # finish payload was ignored — that was a silent contract bug.
    assert model.script == []  # script consumed cleanly
    names = [r["name"] for r in result.subagent_runs]
    assert names.count("harness-generator") == 3
    assert names.count("harness-verifier") == 3
    assert result.success is False, "give-up via finish_preprocess(errors=...) must mark failure"
    assert any("rejected 3 attempts" in err for err in result.errors), (
        f"errors from finish_preprocess must be folded into result.errors; got {result.errors!r}"
    )


def test_orchestrator_marks_failure_when_finish_omits_harness(
    tmp_path: Path,
    monkeypatch,
    empty_registry: SubagentRegistry,
) -> None:
    """``finish_preprocess`` with no harness_path AND errors yields success=False.

    Mirrors the realistic give-up case: the LLM hit the retry cap, never
    produced a verified harness, and finished with a non-empty errors
    array. Even if no Python exception was raised inside the loop,
    ``result.success`` must be ``False`` because the downstream pipeline
    cannot proceed without a harness_path.
    """
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _patch_deterministic_tools(monkeypatch, harness_path=tmp_path / "nonexistent_harness.py")

    class _AlwaysRejectingDispatcher:
        def __init__(self, registry):
            self.registry = registry

        def __call__(self, *, name, task, model, cwd=None, context=None):
            if name == "harness-verifier":
                return {
                    "name": name,
                    "success": False,
                    "output": "HARNESS_VERIFIED=false\nESCALATE=false",
                    "elapsed_s": 0.1,
                }
            return {"name": name, "success": True, "output": "ok", "elapsed_s": 0.1}

    codebase_context_path = out_dir / "CODEBASE_CONTEXT.md"
    script = [
        (
            "codebase_explore",
            {
                "repo_root": str(repo),
                "kernel_path": str(kernel_path),
                "out_path": str(codebase_context_path),
            },
        ),
        ("dispatch_subagent", {"name": "harness-generator", "task": "attempt 1"}),
        ("dispatch_subagent", {"name": "harness-verifier", "task": "verify 1"}),
        (
            "finish_preprocess",
            {
                "errors": ["max retries exhausted"],
                "summary": "Verifier never approved; giving up.",
            },
        ),
    ]

    model = _ScriptedModel(script)
    agent = PreprocessOrchestratorAgent(model=model)
    register_default_tools(
        agent,
        kernel_language=_LANG,
        registry=empty_registry,
        dispatcher=_AlwaysRejectingDispatcher(empty_registry),
    )

    result = agent.run(
        task="Give-up path: no harness, no baseline.",
        kernel_path=kernel_path,
        repo_root=repo,
        kernel_language=_LANG,
        source_language="triton",
        target_language="triton",
        output_dir=out_dir,
        gpu_id=0,
    )

    assert model.script == []  # script consumed cleanly
    assert result.success is False
    assert result.harness_path is None
    assert result.baseline is None
    assert "max retries exhausted" in result.errors


def test_orchestrator_marks_failure_when_baseline_missing(
    tmp_path: Path,
    monkeypatch,
    empty_registry: SubagentRegistry,
) -> None:
    """A finish_preprocess with no errors but no baseline still fails.

    Pins the strict success contract from commit set 5a: even an empty
    ``errors`` list isn't enough — ``baseline`` and ``harness_path`` must
    both be populated for ``success=True``. Otherwise the optimisation
    loop will hit ``KeyError`` on ``preprocess_ctx['baseline_metrics']``.
    """
    repo, kernel_path = _build_fixture_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    harness_path = out_dir / "test_harness.py"
    harness_path.write_text(_HARNESS_BODY, encoding="utf-8")

    _patch_deterministic_tools(monkeypatch, harness_path=harness_path)

    class _OnlyGenDispatcher:
        def __init__(self, registry):
            self.registry = registry

        def __call__(self, *, name, task, model, cwd=None, context=None):
            if name == "harness-verifier":
                return {
                    "name": name,
                    "success": True,
                    "output": f"HARNESS_VERIFIED=true\nHARNESS_PATH={harness_path}",
                    "elapsed_s": 0.1,
                }
            return {"name": name, "success": True, "output": "ok", "elapsed_s": 0.1}

    codebase_context_path = out_dir / "CODEBASE_CONTEXT.md"
    # Note: NO collect_baseline call in this script — the LLM skips it
    # (a realistic misbehaviour to guard against).
    script = [
        (
            "codebase_explore",
            {
                "repo_root": str(repo),
                "kernel_path": str(kernel_path),
                "out_path": str(codebase_context_path),
            },
        ),
        ("dispatch_subagent", {"name": "harness-generator", "task": "attempt 1"}),
        ("dispatch_subagent", {"name": "harness-verifier", "task": "verify 1"}),
        (
            "finish_preprocess",
            {
                "harness_path": str(harness_path),
                "errors": [],
                "summary": "Done (but baseline was never collected).",
            },
        ),
    ]

    model = _ScriptedModel(script)
    agent = PreprocessOrchestratorAgent(model=model)
    register_default_tools(
        agent,
        kernel_language=_LANG,
        registry=empty_registry,
        dispatcher=_OnlyGenDispatcher(empty_registry),
    )

    result = agent.run(
        task="Skip-baseline path.",
        kernel_path=kernel_path,
        repo_root=repo,
        kernel_language=_LANG,
        source_language="triton",
        target_language="triton",
        output_dir=out_dir,
        gpu_id=0,
    )

    assert model.script == []
    assert result.harness_path == harness_path
    assert result.baseline is None
    assert result.success is False
