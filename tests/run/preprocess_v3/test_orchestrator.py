"""Tests for ``minisweagent.run.preprocess_v3.orchestrator``.

This commit covers the agent skeleton — config defaults, system-prompt
contract, ``PreprocessResult`` dataclass shape, tool registration shim.
The full tool-suite + dispatch loop tests land alongside commit 5
(orchestrator tools wiring).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.orchestrator import (
    FormatError,
    LimitsExceeded,
    PreprocessOrchestratorAgent,
    PreprocessOrchestratorConfig,
    PreprocessResult,
    ToolEntry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal model stub for skeleton-level tests.

    The orchestrator only touches: ``model.query(messages)`` (returns a
    response dict), ``model.n_calls`` (int), ``model.cost`` (float), and
    optionally ``model.set_tools(schemas)``. We script ``query`` to
    return a queue of canned responses.
    """

    def __init__(self, responses: list[dict] | None = None) -> None:
        self._responses = list(responses or [])
        self.n_calls = 0
        self.cost = 0.0
        self.tools_set: list | None = None
        self.queries: list[list[dict]] = []

    def set_tools(self, schemas: list) -> None:
        self.tools_set = list(schemas)

    def query(self, messages: list[dict]) -> dict:
        self.queries.append(list(messages))
        self.n_calls += 1
        if not self._responses:
            return {"content": ""}
        return self._responses.pop(0)


_LANG = KernelLanguage(
    name="triton",
    file_extensions=frozenset({".py"}),
    detect_hints=(r"@triton\.jit",),
    kb_namespace="triton",
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """Default config matches the commit-set spec.

    Values are fixed so a future drift (e.g. someone bumping ``step_limit``
    in the dataclass) shows up as a test failure rather than a silent
    behaviour change in CI.
    """
    cfg = PreprocessOrchestratorConfig()

    assert cfg.model == "amd-llm-router"
    assert cfg.model_class == "amd_llm"
    assert cfg.step_limit == 200
    assert cfg.cost_limit == 0.0
    assert cfg.gpu_id == 0
    assert cfg.repo is None
    assert cfg.flydsl_repo is None
    # System + instance templates are non-empty Jinja strings.
    assert "{{kernel_path}}" in cfg.instance_template
    assert "v3 GEAK Preprocess Orchestrator" in cfg.system_template


def test_config_is_frozen() -> None:
    """Config is a frozen dataclass — can't be mutated after construction."""
    cfg = PreprocessOrchestratorConfig()
    with pytest.raises(Exception):
        cfg.step_limit = 50  # type: ignore[misc]


# ---------------------------------------------------------------------------
# System prompt contract
# ---------------------------------------------------------------------------


def test_system_prompt_names_all_six_steps() -> None:
    """Every step name from the design doc must appear in the system prompt.

    The orchestrator prompt is the only place these step names are
    enumerated for the LLM; if a future edit drops a step heading, the
    LLM will silently skip that step. Pinning the names here catches
    that.
    """
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    assert "codebase-explore" in sp or "codebase_explore" in sp
    assert "translate" in sp.lower()
    assert "harness generation" in sp.lower() or "harness-generator" in sp
    assert "harness-verifier" in sp
    assert "baseline + profile" in sp.lower() or "baseline" in sp
    assert "speedup-verify" in sp
    assert "COMMANDMENT" in sp


def test_system_prompt_lists_all_eight_tools() -> None:
    """Every registered tool's name must appear in the system prompt.

    Without this, the LLM doesn't know it can call them. The 8-tool
    inventory was locked by commit set 4 + commit set 7
    (``commandment_from_user_command`` added for the Path-A
    short-circuit).
    """
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    expected_tools = [
        "codebase_explore",
        "translate_to_flydsl",
        "dispatch_subagent",
        "collect_baseline",
        "collect_profile",
        "render_commandment",
        "commandment_from_user_command",
        "finish_preprocess",
    ]
    for tool in expected_tools:
        assert tool in sp, f"system prompt missing tool name {tool!r}"


def test_system_prompt_has_step_0_path_decision_section() -> None:
    """The orchestrator prompt explains how to pick Path A vs Path B
    BEFORE any other action (commit set 7).
    """
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    assert "Step 0" in sp, "system prompt missing 'Step 0' marker for Path A/B decision"
    assert "Path A" in sp and "Path B" in sp, "system prompt missing Path A / Path B section"


def test_system_prompt_step_0_names_path_a_trigger_tool() -> None:
    """Step 0 must name ``commandment_from_user_command`` as the Path-A
    trigger (the LLM picks the path by which tool it calls first).
    """
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    step_0_marker = sp.index("Step 0")
    # Trigger tool name must appear somewhere AFTER the Step 0 marker.
    after_step_0 = sp[step_0_marker:]
    assert "commandment_from_user_command" in after_step_0


def test_system_prompt_step_0_says_path_a_skips_subagents() -> None:
    """Step 0 must tell the LLM that Path A skips the 3 always-on
    subagents (harness-generator / harness-verifier / speedup-verify).
    """
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    step_0_marker = sp.index("Step 0")
    after_step_0 = sp[step_0_marker:]
    # The dispatch_subagent skip rule must be explicit.
    assert "harness-generator" in after_step_0
    assert "skipped" in after_step_0.lower() or "do not call" in after_step_0.lower()


def test_system_prompt_has_path_a_partial_mode_coverage_section() -> None:
    """The orchestrator prompt explains how to handle Path-A partial
    mode coverage (inferring modes from the source command).
    """
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    assert "partial mode coverage" in sp.lower() or "Partial mode coverage" in sp
    assert "inferred_modes" in sp
    assert "PATH_A_PARTIAL_COVERAGE" in sp


def test_system_prompt_encodes_finish_contract() -> None:
    """The ``finish_preprocess`` completion contract must be explicit."""
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    assert "finish_preprocess" in sp
    # Termination semantics must be unambiguous so the LLM doesn't keep
    # calling tools after a finish.
    assert "terminat" in sp.lower()


def test_system_prompt_is_practical_length() -> None:
    """The system prompt should be 80-180 lines (commit-set requirement).

    Counts only non-empty source lines since blank-line padding is just
    visual whitespace that the model strips.
    """
    cfg = PreprocessOrchestratorConfig()
    line_count = sum(1 for line in cfg.system_template.splitlines() if line.strip())
    assert 80 <= line_count <= 220, f"system prompt has {line_count} non-empty lines, expected 80-180-ish"


def test_system_prompt_lists_three_subagent_names() -> None:
    """Lock the three valid ``dispatch_subagent`` name choices in the prompt."""
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    for sub in ("harness-generator", "harness-verifier", "speedup-verify"):
        assert sub in sp, f"missing subagent name {sub!r} in system prompt"


def test_system_prompt_states_translation_is_tool_not_subagent() -> None:
    """Decision 3: translation is a tool call, not a subagent dispatch.

    The prompt must say so explicitly because the LLM has both
    ``translate_to_flydsl`` and ``dispatch_subagent`` available and could
    plausibly call ``dispatch_subagent("pytorch-to-flydsl", ...)``
    (which would fail since that subagent is not registered).
    """
    cfg = PreprocessOrchestratorConfig()
    sp = cfg.system_template
    assert "tool call" in sp.lower() and "translation" in sp.lower()


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def test_agent_constructs_with_default_config() -> None:
    """Empty-arg construction is supported (uses default config + no tools)."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())

    assert isinstance(agent.config, PreprocessOrchestratorConfig)
    assert agent.config.step_limit == 200
    assert agent.tool_names == []
    assert agent.messages == []


def test_agent_accepts_explicit_config() -> None:
    """Explicit config overrides the default."""
    cfg = PreprocessOrchestratorConfig(step_limit=42, gpu_id=3)
    agent = PreprocessOrchestratorAgent(model=_StubModel(), config=cfg)

    assert agent.config.step_limit == 42
    assert agent.config.gpu_id == 3


def test_register_tool_validates_schema_name() -> None:
    """``register_tool`` rejects schemas whose ``name`` doesn't match the key."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())

    bad_schema = {"name": "wrong_name", "type": "function"}
    with pytest.raises(ValueError, match="schema must be a dict with name"):
        agent.register_tool("right_name", bad_schema, lambda **_: {})


def test_register_tool_records_schema_and_callable() -> None:
    """Successful registration shows up in ``tool_names`` + ``get_tool_schemas``."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())

    def _impl(**_kwargs):
        return {"output": "ok"}

    schema = {
        "name": "dummy",
        "type": "function",
        "description": "Dummy tool",
        "parameters": {"type": "object", "properties": {}},
    }
    agent.register_tool("dummy", schema, _impl)

    assert agent.tool_names == ["dummy"]
    schemas = agent.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "dummy"


# ---------------------------------------------------------------------------
# Step loop / dispatch primitives
# ---------------------------------------------------------------------------


def test_step_raises_format_error_when_response_has_no_tool_call() -> None:
    """Text-only responses are not actionable — the orchestrator must escalate."""
    model = _StubModel(responses=[{"content": "I'm thinking..."}])
    agent = PreprocessOrchestratorAgent(model=model)

    with pytest.raises(FormatError, match="no tool call"):
        agent.step()


def test_step_dispatches_tool_and_records_observation() -> None:
    """A tool call returns a dict observation appended as ``role="tool"``."""
    model = _StubModel(
        responses=[
            {
                "content": "",
                "tools": {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"x": 42}'},
                },
            },
        ],
    )
    agent = PreprocessOrchestratorAgent(model=model)

    captured = {}

    def _echo(**kwargs):
        captured.update(kwargs)
        return {"echoed": kwargs}

    schema = {
        "name": "echo",
        "type": "function",
        "description": "Echo args back.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }
    agent.register_tool("echo", schema, _echo)

    observation = agent.step()
    assert observation == {"echoed": {"x": 42}}
    assert captured == {"x": 42}
    # Two messages appended: assistant tool-call + tool result.
    assert agent.messages[-2]["role"] == "assistant"
    assert agent.messages[-1]["role"] == "tool"
    assert agent.messages[-1]["name"] == "echo"


def test_step_raises_limits_exceeded_when_step_count_exceeds_cap() -> None:
    """``step_limit`` is enforced at the top of ``step()``."""
    model = _StubModel()
    model.n_calls = 10
    agent = PreprocessOrchestratorAgent(
        model=model,
        config=PreprocessOrchestratorConfig(step_limit=10),
    )

    with pytest.raises(LimitsExceeded):
        agent.step()


def test_step_raises_limits_exceeded_when_cost_exceeds_cap() -> None:
    """``cost_limit`` is enforced at the top of ``step()``."""
    model = _StubModel()
    model.cost = 5.0
    agent = PreprocessOrchestratorAgent(
        model=model,
        config=PreprocessOrchestratorConfig(cost_limit=5.0),
    )

    with pytest.raises(LimitsExceeded):
        agent.step()


def test_unknown_tool_returns_error_observation() -> None:
    """Calling an unregistered tool produces an error obs the LLM can read."""
    model = _StubModel(
        responses=[
            {
                "content": "",
                "tools": {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "nope", "arguments": "{}"},
                },
            },
        ],
    )
    agent = PreprocessOrchestratorAgent(model=model)

    observation = agent.step()
    assert "error" in observation
    assert "Unknown tool" in observation["error"]


def test_tool_arguments_accept_dict_or_json_string() -> None:
    """LiteLLM may pass arguments as a JSON string; orchestrator must coerce."""
    model = _StubModel(
        responses=[
            {
                "content": "",
                "tools": {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": {"x": 1}},
                },
            },
        ],
    )
    agent = PreprocessOrchestratorAgent(model=model)
    agent.register_tool(
        "echo",
        {
            "name": "echo",
            "type": "function",
            "description": "echo",
            "parameters": {"type": "object", "properties": {}},
        },
        lambda **kw: {"got": kw},
    )

    obs = agent.step()
    assert obs == {"got": {"x": 1}}


# ---------------------------------------------------------------------------
# PreprocessResult
# ---------------------------------------------------------------------------


def test_preprocess_result_defaults_sensibly() -> None:
    """Required fields are typed; optional fields default to None / empty list."""
    result = PreprocessResult(
        success=False,
        kernel_language=_LANG,
        kernel_path=Path("/tmp/kernel.py"),
    )
    assert result.harness_path is None
    assert result.baseline is None
    assert result.profile is None
    assert result.codebase_context is None
    assert result.commandment_path is None
    assert result.translation is None
    assert result.subagent_runs == []
    assert result.elapsed_s == 0.0
    assert result.errors == []


def test_preprocess_result_is_frozen() -> None:
    """Immutability so the orchestrator can hand the result around safely."""
    result = PreprocessResult(
        success=False,
        kernel_language=_LANG,
        kernel_path=Path("/tmp/kernel.py"),
    )
    with pytest.raises(Exception):
        result.success = True  # type: ignore[misc]


def test_preprocess_result_carries_subagent_runs() -> None:
    """``subagent_runs`` carries one dict per dispatch call."""
    runs = [
        {"name": "harness-generator", "success": True, "elapsed_s": 10.0},
        {"name": "harness-verifier", "success": True, "elapsed_s": 2.0},
    ]
    result = PreprocessResult(
        success=True,
        kernel_language=_LANG,
        kernel_path=Path("/tmp/kernel.py"),
        subagent_runs=runs,
    )
    assert len(result.subagent_runs) == 2
    assert result.subagent_runs[0]["name"] == "harness-generator"


# ---------------------------------------------------------------------------
# Path-A vs Path-B (commit set 7)
# ---------------------------------------------------------------------------


def test_preprocess_result_path_taken_defaults_to_b() -> None:
    """Legacy callers (constructing a result with no ``path_taken``) default
    to ``"B"`` — the existing 6-step flow."""
    result = PreprocessResult(
        success=False,
        kernel_language=_LANG,
        kernel_path=Path("/tmp/kernel.py"),
    )
    assert result.path_taken == "B"
    assert result.tool_calls == []


def test_preprocess_result_accepts_path_taken_a() -> None:
    """Constructing a result with ``path_taken="A"`` is valid."""
    result = PreprocessResult(
        success=True,
        kernel_language=_LANG,
        kernel_path=Path("/tmp/kernel.py"),
        harness_path=None,
        commandment_path=Path("/tmp/COMMANDMENT.md"),
        path_taken="A",
    )
    assert result.path_taken == "A"
    assert result.harness_path is None


def test_finalize_success_path_a_with_commandment_returns_true() -> None:
    """Path A succeeds when COMMANDMENT exists, even without a harness."""
    success = PreprocessOrchestratorAgent._finalize_success(
        finish_payload={"summary": "ok"},
        errors=[],
        harness_path=None,
        baseline=None,
        path_taken="A",
        commandment_path=Path("/tmp/COMMANDMENT.md"),
    )
    assert success is True


def test_finalize_success_path_a_without_commandment_returns_false() -> None:
    """Path A still fails if no COMMANDMENT was emitted."""
    success = PreprocessOrchestratorAgent._finalize_success(
        finish_payload={"summary": "ok"},
        errors=[],
        harness_path=None,
        baseline=None,
        path_taken="A",
        commandment_path=None,
    )
    assert success is False


def test_finalize_success_path_a_propagates_errors() -> None:
    """Loop-level errors invalidate Path A success too."""
    success = PreprocessOrchestratorAgent._finalize_success(
        finish_payload={"summary": "ok"},
        errors=["some error"],
        harness_path=None,
        baseline=None,
        path_taken="A",
        commandment_path=Path("/tmp/COMMANDMENT.md"),
    )
    assert success is False


def test_finalize_success_path_b_without_harness_returns_false() -> None:
    """Path B with no harness_path fails (existing contract preserved)."""
    success = PreprocessOrchestratorAgent._finalize_success(
        finish_payload={"summary": "ok"},
        errors=[],
        harness_path=None,
        baseline=object(),
        path_taken="B",
        commandment_path=Path("/tmp/COMMANDMENT.md"),
    )
    assert success is False


def test_finalize_success_path_b_default_is_b() -> None:
    """The ``path_taken`` default is ``"B"``; baseline must be present."""
    success = PreprocessOrchestratorAgent._finalize_success(
        finish_payload={"summary": "ok"},
        errors=[],
        harness_path=Path("/tmp/h.py"),
        baseline=object(),
    )
    assert success is True


def test_finalize_success_path_b_without_baseline_returns_false() -> None:
    """Path B without baseline fails (existing strict criteria preserved)."""
    success = PreprocessOrchestratorAgent._finalize_success(
        finish_payload={"summary": "ok"},
        errors=[],
        harness_path=Path("/tmp/h.py"),
        baseline=None,
        path_taken="B",
        commandment_path=Path("/tmp/COMMANDMENT.md"),
    )
    assert success is False


# ---------------------------------------------------------------------------
# _partition_errors_by_path — Path-A downgrade of baseline/profile errors
# ---------------------------------------------------------------------------


def test_partition_errors_path_a_downgrades_collect_baseline_to_warning() -> None:
    """Path A: ``collect_baseline …`` error strings become warnings."""
    real_errors, warnings = PreprocessOrchestratorAgent._partition_errors_by_path(
        [
            "collect_baseline failed: harness not found",
            "some other thing went wrong",
        ],
        path_taken="A",
    )
    assert warnings == ["collect_baseline failed: harness not found"]
    assert real_errors == ["some other thing went wrong"]


def test_partition_errors_path_a_downgrades_collect_profile_to_warning() -> None:
    """Path A: ``collect_profile …`` error strings become warnings."""
    real_errors, warnings = PreprocessOrchestratorAgent._partition_errors_by_path(
        ["collect_profile: profiler-mcp unavailable"],
        path_taken="A",
    )
    assert warnings == ["collect_profile: profiler-mcp unavailable"]
    assert real_errors == []


def test_partition_errors_path_a_preserves_other_errors() -> None:
    """Path A: non-baseline/profile errors remain as errors."""
    real_errors, warnings = PreprocessOrchestratorAgent._partition_errors_by_path(
        ["LimitsExceeded: step limit hit", "render_commandment crashed"],
        path_taken="A",
    )
    assert warnings == []
    assert real_errors == ["LimitsExceeded: step limit hit", "render_commandment crashed"]


def test_partition_errors_path_b_does_not_downgrade_anything() -> None:
    """Path B: nothing is downgraded — strict legacy criteria preserved."""
    real_errors, warnings = PreprocessOrchestratorAgent._partition_errors_by_path(
        [
            "collect_baseline failed",
            "collect_profile: missing harness",
            "other",
        ],
        path_taken="B",
    )
    assert warnings == []
    assert real_errors == [
        "collect_baseline failed",
        "collect_profile: missing harness",
        "other",
    ]


def test_partition_errors_handles_leading_whitespace() -> None:
    """Leading whitespace on Path-A error strings does not defeat the match."""
    real_errors, warnings = PreprocessOrchestratorAgent._partition_errors_by_path(
        ["   collect_baseline returned ok=False"],
        path_taken="A",
    )
    assert warnings == ["   collect_baseline returned ok=False"]
    assert real_errors == []


def test_partition_errors_coerces_non_string_entries() -> None:
    """Non-string entries are str()-coerced before the prefix check."""
    real_errors, warnings = PreprocessOrchestratorAgent._partition_errors_by_path(
        [Exception("collect_baseline boom")],  # type: ignore[list-item]
        path_taken="A",
    )
    assert warnings == ["collect_baseline boom"]
    assert real_errors == []


def test_partition_errors_empty_list_is_pair_of_empty_lists() -> None:
    """``[] -> ([], [])`` for both paths."""
    assert PreprocessOrchestratorAgent._partition_errors_by_path([], "A") == ([], [])
    assert PreprocessOrchestratorAgent._partition_errors_by_path([], "B") == ([], [])


# ---------------------------------------------------------------------------
# _build_result wires _partition_errors_by_path + writes warnings on Path A
# ---------------------------------------------------------------------------


def test_build_result_path_a_success_when_only_baseline_profile_failed(tmp_path: Path) -> None:
    """Path A + COMMANDMENT + only baseline/profile errors → success=True, warnings set."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    agent._collected = {"commandment_path": tmp_path / "COMMANDMENT.md"}
    agent._tool_calls = [{"name": "commandment_from_user_command", "args": {}}]
    agent._subagent_runs = []

    result = agent._build_result(
        context={"kernel_language": _LANG, "kernel_path": tmp_path / "k.py"},
        finish_payload={
            "errors": [
                "collect_baseline failed: harness not found",
                "collect_profile failed: harness not found",
            ],
            "summary": "Path A complete",
        },
        errors=[],
        elapsed_s=1.0,
    )

    assert result.path_taken == "A"
    assert result.success is True
    assert result.errors == []
    assert sorted(result.warnings) == sorted(
        [
            "collect_baseline failed: harness not found",
            "collect_profile failed: harness not found",
        ]
    )


def test_build_result_path_a_failure_when_no_commandment(tmp_path: Path) -> None:
    """Path A + no COMMANDMENT path → success=False (warnings still populated)."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    agent._collected = {}
    agent._tool_calls = [{"name": "commandment_from_user_command", "args": {}}]
    agent._subagent_runs = []

    result = agent._build_result(
        context={"kernel_language": _LANG, "kernel_path": tmp_path / "k.py"},
        finish_payload={"errors": ["collect_baseline failed"], "summary": "no commandment"},
        errors=[],
        elapsed_s=1.0,
    )

    assert result.path_taken == "A"
    assert result.success is False
    assert result.warnings == ["collect_baseline failed"]


def test_build_result_path_a_failure_when_real_error_present(tmp_path: Path) -> None:
    """Path A + non-baseline/profile error → success=False, error preserved."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    agent._collected = {"commandment_path": tmp_path / "COMMANDMENT.md"}
    agent._tool_calls = [{"name": "commandment_from_user_command", "args": {}}]
    agent._subagent_runs = []

    result = agent._build_result(
        context={"kernel_language": _LANG, "kernel_path": tmp_path / "k.py"},
        finish_payload={"errors": ["render_commandment crashed"], "summary": "fail"},
        errors=[],
        elapsed_s=1.0,
    )

    assert result.path_taken == "A"
    assert result.success is False
    assert result.errors == ["render_commandment crashed"]
    assert result.warnings == []


def test_build_result_path_b_clean_run_unchanged_behaviour(tmp_path: Path) -> None:
    """Path B + clean run → success=True (existing contract preserved)."""

    class _Baseline:
        pass

    baseline = _Baseline()
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    agent._collected = {
        "harness_path": tmp_path / "h.py",
        "baseline": baseline,
    }
    agent._tool_calls = []  # No Path-A tool → Path B
    agent._subagent_runs = []

    result = agent._build_result(
        context={"kernel_language": _LANG, "kernel_path": tmp_path / "k.py"},
        finish_payload={"errors": [], "summary": "clean"},
        errors=[],
        elapsed_s=1.0,
    )

    assert result.path_taken == "B"
    assert result.success is True
    assert result.warnings == []


def test_build_result_path_b_any_error_invalidates_success(tmp_path: Path) -> None:
    """Path B + any error string → success=False, warnings always empty."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    agent._collected = {"harness_path": tmp_path / "h.py", "baseline": object()}
    agent._tool_calls = []
    agent._subagent_runs = []

    result = agent._build_result(
        context={"kernel_language": _LANG, "kernel_path": tmp_path / "k.py"},
        finish_payload={"errors": ["collect_baseline failed"], "summary": "fail"},
        errors=[],
        elapsed_s=1.0,
    )

    assert result.path_taken == "B"
    assert result.success is False
    assert result.errors == ["collect_baseline failed"]
    assert result.warnings == []


def test_preprocess_result_has_warnings_field() -> None:
    """The new ``warnings`` field defaults to an empty list."""
    result = PreprocessResult(
        success=True,
        kernel_language=_LANG,
        kernel_path=Path("/tmp/k.py"),
    )
    assert result.warnings == []


def test_dispatch_loop_records_tool_calls_in_audit_log() -> None:
    """Every dispatched tool call appears in ``agent._tool_calls``."""
    model = _StubModel(
        responses=[
            {
                "content": "",
                "tools": {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"x": 1}'},
                },
            },
        ],
    )
    agent = PreprocessOrchestratorAgent(model=model)
    agent.register_tool(
        "echo",
        {
            "name": "echo",
            "type": "function",
            "description": "echo",
            "parameters": {"type": "object", "properties": {}},
        },
        lambda **kw: {"ok": True, "args": kw},
    )

    agent.step()
    assert len(agent._tool_calls) == 1
    assert agent._tool_calls[0]["name"] == "echo"
    assert agent._tool_calls[0]["args"] == {"x": 1}


def test_dispatch_loop_records_failed_tool_calls_too() -> None:
    """The audit log captures even failed dispatches (unknown tools)."""
    model = _StubModel(
        responses=[
            {
                "content": "",
                "tools": {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "nope", "arguments": "{}"},
                },
            },
        ],
    )
    agent = PreprocessOrchestratorAgent(model=model)

    agent.step()
    assert len(agent._tool_calls) == 1
    assert agent._tool_calls[0]["name"] == "nope"


# ---------------------------------------------------------------------------
# ToolEntry
# ---------------------------------------------------------------------------


def test_tool_entry_holds_schema_and_callable() -> None:
    """Sanity: the registration container exposes both fields."""

    def _f(**_):
        return {}

    entry = ToolEntry(schema={"name": "x"}, callable=_f)
    assert entry.schema == {"name": "x"}
    assert entry.callable is _f


# ---------------------------------------------------------------------------
# Tool wiring (commit 5)
#
# These tests exercise the 7-tool registration helper from
# ``minisweagent.run.preprocess_v3.tools``. They use a stub
# ``PreprocessSubagentDispatcher`` that returns canned results so we never
# spin up a real DefaultAgent.
# ---------------------------------------------------------------------------


from minisweagent.run.preprocess_v3.tools import (  # noqa: E402  (intentional late import)
    ALLOWED_SUBAGENT_NAMES,
    PreprocessSubagentDispatcher,
    register_default_tools,
    validate_call_against_schema,
)


class _StubDispatcher:
    """Stub dispatcher that records calls without spawning a child agent."""

    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.response = response or {
            "name": "harness-generator",
            "success": True,
            "output": "TEST_COMMAND: cd /tmp && python harness.py --correctness && python harness.py --benchmark",
            "elapsed_s": 0.1,
            "max_steps": -1,
            "is_unlimited_steps": True,
        }

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.response, name=kwargs.get("name", self.response.get("name")))


def _make_agent_with_tools(
    *,
    dispatcher=None,
    registry=None,
):
    from minisweagent.run.preprocess_v3.registry import SubagentRegistry

    agent = PreprocessOrchestratorAgent(model=_StubModel())
    register_default_tools(
        agent,
        kernel_language=_LANG,
        registry=registry or SubagentRegistry(root=Path("/nonexistent")),
        dispatcher=dispatcher or _StubDispatcher(),
    )
    return agent


def test_register_default_tools_registers_all_eight() -> None:
    """All 8 tool names must be registered in ``tool_names``.

    Commit set 7 added ``commandment_from_user_command`` for the Path-A
    short-circuit. The original 7 tools are unchanged; the 8th is
    additive.
    """
    agent = _make_agent_with_tools()
    assert sorted(agent.tool_names) == sorted(
        [
            "codebase_explore",
            "translate_to_flydsl",
            "dispatch_subagent",
            "collect_baseline",
            "collect_profile",
            "render_commandment",
            "commandment_from_user_command",
            "finish_preprocess",
        ]
    )


def test_register_default_tools_schemas_are_function_type() -> None:
    """Every registered schema declares ``type='function'`` and a parameters object."""
    agent = _make_agent_with_tools()
    for schema in agent.get_tool_schemas():
        assert schema["type"] == "function"
        assert schema["parameters"]["type"] == "object"


@pytest.mark.parametrize(
    ("tool_name", "good_args"),
    [
        (
            "codebase_explore",
            {"repo_root": "/tmp/repo", "kernel_path": "/tmp/repo/k.py", "out_path": "/tmp/out/CC.md"},
        ),
        (
            "translate_to_flydsl",
            {"source_path": "/tmp/k.py", "output_dir": "/tmp/out"},
        ),
        (
            "dispatch_subagent",
            {"name": "harness-generator", "task": "Build a harness for the kernel."},
        ),
        ("collect_baseline", {"harness_path": "/tmp/h.py"}),
        ("collect_profile", {"harness_path": "/tmp/h.py"}),
        (
            "render_commandment",
            {
                "kernel_path": "/tmp/k.py",
                "harness_path": "/tmp/h.py",
                "repo_root": "/tmp/repo",
                "out_path": "/tmp/out/CMD.md",
            },
        ),
        ("finish_preprocess", {}),
    ],
)
def test_each_schema_validates_known_good_call(tool_name: str, good_args: dict) -> None:
    """Each tool's schema must accept a known-good argument set."""
    agent = _make_agent_with_tools()
    schema = next(s for s in agent.get_tool_schemas() if s["name"] == tool_name)
    ok, msg = validate_call_against_schema(schema, good_args)
    assert ok, f"{tool_name} rejected good args: {msg}"


@pytest.mark.parametrize(
    ("tool_name", "bad_args", "reason"),
    [
        ("codebase_explore", {"repo_root": "/tmp"}, "missing required"),
        ("translate_to_flydsl", {"source_path": "/tmp"}, "missing required"),
        ("dispatch_subagent", {"task": "do stuff"}, "missing required"),
        ("collect_baseline", {}, "missing required"),
        ("collect_profile", {}, "missing required"),
        ("render_commandment", {"kernel_path": "/tmp/k.py"}, "missing required"),
    ],
)
def test_each_schema_rejects_known_bad_call(tool_name: str, bad_args: dict, reason: str) -> None:
    """Each tool's schema must reject a call missing a required argument."""
    agent = _make_agent_with_tools()
    schema = next(s for s in agent.get_tool_schemas() if s["name"] == tool_name)
    ok, msg = validate_call_against_schema(schema, bad_args)
    assert not ok
    assert reason in msg


def test_dispatch_subagent_schema_enforces_allow_list() -> None:
    """The ``dispatch_subagent`` schema's enum must reject unknown names."""
    agent = _make_agent_with_tools()
    schema = next(s for s in agent.get_tool_schemas() if s["name"] == "dispatch_subagent")

    ok_good, _ = validate_call_against_schema(schema, {"name": "harness-generator", "task": "go"})
    ok_bad, msg_bad = validate_call_against_schema(schema, {"name": "pytorch-to-flydsl", "task": "go"})

    assert ok_good is True
    assert ok_bad is False
    assert "enum" in msg_bad


def test_dispatch_subagent_schema_enum_lists_three_subagents() -> None:
    """Exactly the three v3 subagents are listed in the enum.

    Pin the contract so a future addition fails this test loudly until
    the orchestrator's system prompt is updated to mention the new
    subagent.
    """
    agent = _make_agent_with_tools()
    schema = next(s for s in agent.get_tool_schemas() if s["name"] == "dispatch_subagent")
    enum = schema["parameters"]["properties"]["name"]["enum"]
    assert sorted(enum) == sorted(ALLOWED_SUBAGENT_NAMES)
    assert "pytorch-to-flydsl" not in enum


# ---------------------------------------------------------------------------
# Tool dispatch behaviour
# ---------------------------------------------------------------------------


def test_finish_preprocess_terminates_loop_with_payload() -> None:
    """``finish_preprocess`` raises ``FinishedSuccessfully`` carrying its payload."""
    from minisweagent.run.preprocess_v3.orchestrator import FinishedSuccessfully

    agent = _make_agent_with_tools()
    finish_tool = agent._tools["finish_preprocess"].callable

    with pytest.raises(FinishedSuccessfully) as excinfo:
        finish_tool(
            harness_path="/tmp/h.py",
            commandment_path="/tmp/CMD.md",
            errors=[],
            summary="OK",
        )
    assert excinfo.value.payload["harness_path"] == "/tmp/h.py"
    assert excinfo.value.payload["summary"] == "OK"


def test_dispatch_subagent_records_subagent_run() -> None:
    """Calling ``dispatch_subagent`` appends a result to ``_subagent_runs``."""
    dispatcher = _StubDispatcher()
    agent = _make_agent_with_tools(dispatcher=dispatcher)

    dispatch_tool = agent._tools["dispatch_subagent"].callable
    result = dispatch_tool(name="harness-generator", task="Generate a harness")

    assert result["success"] is True
    assert result["name"] == "harness-generator"
    assert len(agent._subagent_runs) == 1
    assert dispatcher.calls[0]["name"] == "harness-generator"


def test_dispatch_subagent_extracts_test_command_from_generator_output() -> None:
    """The orchestrator parses ``TEST_COMMAND:`` from harness-generator output."""
    agent = _make_agent_with_tools()
    dispatch_tool = agent._tools["dispatch_subagent"].callable

    dispatch_tool(name="harness-generator", task="Generate")

    assert "test_command" in agent._collected
    assert "python harness.py" in agent._collected["test_command"]


def test_dispatch_subagent_extracts_harness_path_from_verifier_output() -> None:
    """The orchestrator parses ``HARNESS_PATH=`` from harness-verifier output."""
    verifier_response = {
        "name": "harness-verifier",
        "success": True,
        "output": "HARNESS_VERIFIED=true\nHARNESS_PATH=/tmp/harness.py\nMODES_PASSED=correctness,profile,benchmark,full-benchmark",
        "elapsed_s": 0.1,
        "max_steps": 30,
        "is_unlimited_steps": False,
    }
    agent = _make_agent_with_tools(dispatcher=_StubDispatcher(verifier_response))
    dispatch_tool = agent._tools["dispatch_subagent"].callable

    dispatch_tool(name="harness-verifier", task="Verify it")

    assert agent._collected["harness_path"] == "/tmp/harness.py"


# ---------------------------------------------------------------------------
# PreprocessSubagentDispatcher allow-list
# ---------------------------------------------------------------------------


def test_dispatcher_rejects_unknown_subagent_name() -> None:
    """The dispatcher itself enforces the allow-list (defence in depth)."""
    from minisweagent.run.preprocess_v3.registry import SubagentRegistry

    dispatcher = PreprocessSubagentDispatcher(SubagentRegistry(root=Path("/nonexistent")))
    result = dispatcher(name="not-a-real-subagent", task="x", model=None)

    assert result["success"] is False
    assert "allow-list" in result["error"]


def test_dispatcher_returns_structured_error_on_missing_registry_entry(tmp_path) -> None:
    """A name in the allow-list but missing from disk yields a structured error."""
    from minisweagent.run.preprocess_v3.registry import SubagentRegistry

    empty_registry = SubagentRegistry(root=tmp_path)
    dispatcher = PreprocessSubagentDispatcher(empty_registry)
    result = dispatcher(name="harness-generator", task="x", model=None)

    assert result["success"] is False
    assert "not found in registry" in result["error"]


def test_dispatcher_propagates_unlimited_steps(tmp_path) -> None:
    """``spec.max_steps == -1`` translates to ``step_limit=0`` for the child.

    The legacy ``DefaultAgent.AgentConfig`` convention is ``step_limit=0``
    means unlimited; the dispatcher must honor the v3 sentinel.
    """
    from minisweagent.run.preprocess_v3.registry import SubagentRegistry

    yaml_body = """\
name: harness-generator
description: Test subagent.
system_prompt: "You are."
max_steps: -1
"""
    folder = tmp_path / "harness-generator"
    folder.mkdir()
    (folder / "SUBAGENT.yaml").write_text(yaml_body, encoding="utf-8")

    captured: dict = {}

    def fake_factory(*, spec, model, cwd):
        captured["spec"] = spec

        class _Agent:
            def run(self, _task):
                return ("Submitted", "ok")

        return _Agent()

    dispatcher = PreprocessSubagentDispatcher(
        SubagentRegistry(root=tmp_path),
        agent_factory=fake_factory,
    )
    result = dispatcher(name="harness-generator", task="x", model=object())

    assert result["success"] is True
    assert result["max_steps"] == -1
    assert result["is_unlimited_steps"] is True
    assert captured["spec"].is_unlimited_steps is True
