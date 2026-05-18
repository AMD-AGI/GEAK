"""Unit tests for the v3 preprocess tool factories.

Most v3 tool wiring is tested via ``test_orchestrator.py`` (schemas +
default-tool registration) and ``test_orchestrator_integration.py``
(end-to-end loops). This module focuses on the **Path-A short-circuit**
tool (``commandment_from_user_command``) introduced in commit set 7 —
its dataclass shape, validation, partial-coverage warning emission, and
the rendered ``COMMANDMENT.md`` structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3 import tools as _tools_mod
from minisweagent.run.preprocess_v3.baseline import BaselineMetrics, ProfileResult
from minisweagent.run.preprocess_v3.commandment import (
    COMMANDMENT_SECTION_KEYS,
    render_commandment_from_sections,
)
from minisweagent.run.preprocess_v3.orchestrator import PreprocessOrchestratorAgent
from minisweagent.run.preprocess_v3.registry import SubagentRegistry
from minisweagent.run.preprocess_v3.tools import (
    PATH_A_MODES,
    RunInstructions,
    _make_tool_collect_baseline,
    _make_tool_collect_profile,
    _make_tool_commandment_from_user_command,
    _make_tool_render_commandment,
    _schema_commandment_from_user_command,
    _schema_render_commandment,
    register_default_tools,
    validate_call_against_schema,
)

_LANG = KernelLanguage(
    name="triton",
    file_extensions=frozenset({".py"}),
    detect_hints=(r"@triton\.jit",),
    kb_namespace="triton",
)


class _StubModel:
    """Minimal model stub — the tool tests never enter the LLM loop."""

    def __init__(self) -> None:
        self.n_calls = 0
        self.cost = 0.0

    def set_tools(self, schemas: list) -> None:
        pass

    def query(self, messages):
        raise AssertionError("tests in this module should not enter the LLM loop")


def _make_agent_with_path_a_tool(tmp_path: Path) -> PreprocessOrchestratorAgent:
    """Build an agent with the full 7-tool surface + the new Path-A tool."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    register_default_tools(
        agent,
        kernel_language=_LANG,
        registry=SubagentRegistry(root=tmp_path / "_empty_registry"),
    )
    return agent


# ---------------------------------------------------------------------------
# RunInstructions dataclass
# ---------------------------------------------------------------------------


def test_run_instructions_is_frozen_and_typed() -> None:
    """``RunInstructions`` is a frozen, slots dataclass."""
    ri = RunInstructions(
        raw_command="python kernel.py --benchmark",
        modes_covered=("benchmark",),
        inferred_modes=("correctness",),
        notes="benchmark-only command",
    )
    assert ri.raw_command == "python kernel.py --benchmark"
    assert ri.modes_covered == ("benchmark",)
    assert ri.inferred_modes == ("correctness",)
    assert ri.notes == "benchmark-only command"

    with pytest.raises(Exception):
        ri.notes = "mutated"  # type: ignore[misc]


def test_run_instructions_defaults() -> None:
    """``inferred_modes`` and ``notes`` are optional with empty defaults."""
    ri = RunInstructions(raw_command="cmd", modes_covered=("benchmark",))
    assert ri.inferred_modes == ()
    assert ri.notes == ""


def test_path_a_modes_pin_locks_to_four_canonical_names() -> None:
    """Pin: the four canonical mode names match the harness CLI flags."""
    assert PATH_A_MODES == ("correctness", "profile", "benchmark", "full_benchmark")


# ---------------------------------------------------------------------------
# render_commandment_from_sections
# ---------------------------------------------------------------------------


def test_render_commandment_from_sections_emits_all_five_headings(tmp_path: Path) -> None:
    """Every canonical section heading must appear in the rendered output."""
    sections = {
        "setup": "true",
        "correctness": "python h.py --correctness",
        "benchmark": "python h.py --benchmark",
        "full_benchmark": "python h.py --full-benchmark",
        "profile": "python h.py --profile",
    }
    out_path = tmp_path / "COMMANDMENT.md"
    text = render_commandment_from_sections(sections, out_path=out_path)

    assert "# Commandment" in text
    assert "## Setup" in text
    assert "## Correctness" in text
    assert "## Benchmark" in text
    assert "## Full Benchmark" in text
    assert "## Profile" in text
    assert out_path.exists()
    assert out_path.read_text() == text


def test_render_commandment_from_sections_preserves_section_order(tmp_path: Path) -> None:
    """The rendered headings appear in canonical Setup -> Profile order."""
    sections = {key: f"cmd_{key}" for key in COMMANDMENT_SECTION_KEYS}
    text = render_commandment_from_sections(sections)

    setup_pos = text.index("## Setup")
    correctness_pos = text.index("## Correctness")
    benchmark_pos = text.index("## Benchmark")
    full_pos = text.index("## Full Benchmark")
    profile_pos = text.index("## Profile")
    assert setup_pos < correctness_pos < benchmark_pos < full_pos < profile_pos


def test_render_commandment_from_sections_missing_keys_emit_warning_marker() -> None:
    """Missing section keys produce a ``PATH_A_PARTIAL_COVERAGE`` marker."""
    sections = {"setup": "true", "benchmark": "python h.py --benchmark"}
    text = render_commandment_from_sections(sections)

    assert "PATH_A_PARTIAL_COVERAGE: correctness not covered" in text
    assert "PATH_A_PARTIAL_COVERAGE: full_benchmark not covered" in text
    assert "PATH_A_PARTIAL_COVERAGE: profile not covered" in text


def test_render_commandment_from_sections_wraps_bodies_in_bash_fences() -> None:
    r"""Each section's body lives inside a ``\`\`\`bash`` fence so the
    commandment-contract validator's "fenced block must parse as shell"
    requirement holds."""
    sections = {"setup": "echo hi"}
    text = render_commandment_from_sections(sections)
    fence_count = text.count("```bash")
    assert fence_count == 5, f"expected 5 bash fences (one per section), got {fence_count}"


def test_render_commandment_from_sections_preamble_inserted_after_title() -> None:
    """The optional ``preamble`` argument is rendered between the title
    and the first ``## Setup`` heading."""
    text = render_commandment_from_sections(
        {"setup": "true"},
        preamble="<!-- audit: from user command -->",
    )
    title_pos = text.index("# Commandment")
    preamble_pos = text.index("<!-- audit:")
    setup_pos = text.index("## Setup")
    assert title_pos < preamble_pos < setup_pos


# ---------------------------------------------------------------------------
# commandment_from_user_command tool factory
# ---------------------------------------------------------------------------


def test_tool_factory_returns_callable() -> None:
    """``_make_tool_commandment_from_user_command`` returns a callable."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)
    assert callable(tool)


def test_tool_emits_commandment_with_expected_modes(tmp_path: Path) -> None:
    """A valid run_command produces a COMMANDMENT with every mode section."""
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)

    result = tool(
        run_command="python kernel.py --benchmark",
        out_path=str(out_path),
        modes_covered=["correctness", "profile", "benchmark", "full_benchmark"],
        inferred_modes=[],
        notes="all four modes provided",
    )

    assert result["ok"] is True
    assert result["commandment_path"] == str(out_path)
    assert sorted(result["modes_emitted"]) == sorted(PATH_A_MODES)
    assert result["warnings"] == []

    assert out_path.exists()
    body = out_path.read_text()
    for heading in ("## Setup", "## Correctness", "## Benchmark", "## Full Benchmark", "## Profile"):
        assert heading in body
    assert "python kernel.py --benchmark" in body


def test_tool_partial_coverage_emits_warning_markers(tmp_path: Path) -> None:
    """Modes in ``inferred_modes`` produce ``PATH_A_PARTIAL_COVERAGE`` markers."""
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)

    result = tool(
        run_command="python kernel.py --benchmark",
        out_path=str(out_path),
        modes_covered=["benchmark"],
        inferred_modes=["correctness", "profile", "full_benchmark"],
        notes="benchmark-only; inferring others",
    )

    assert result["ok"] is True
    assert "benchmark" in result["modes_emitted"]
    assert any("correctness inferred from benchmark" in w for w in result["warnings"])
    assert any("profile inferred from benchmark" in w for w in result["warnings"])
    assert any("full_benchmark inferred from benchmark" in w for w in result["warnings"])

    body = out_path.read_text()
    assert "PATH_A_PARTIAL_COVERAGE: correctness inferred from benchmark" in body
    assert "PATH_A_PARTIAL_COVERAGE: profile inferred from benchmark" in body
    assert "PATH_A_PARTIAL_COVERAGE: full_benchmark inferred from benchmark" in body


def test_tool_modes_neither_covered_nor_inferred_get_not_covered_marker(tmp_path: Path) -> None:
    """A mode missing from both lists yields a 'not covered' warning."""
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)

    result = tool(
        run_command="python kernel.py --benchmark",
        out_path=str(out_path),
        modes_covered=["benchmark"],
        inferred_modes=[],
    )

    body = out_path.read_text()
    assert "PATH_A_PARTIAL_COVERAGE: correctness not covered" in body
    assert "PATH_A_PARTIAL_COVERAGE: profile not covered" in body
    assert "PATH_A_PARTIAL_COVERAGE: full_benchmark not covered" in body
    assert any("not covered" in w for w in result["warnings"])


def test_tool_records_run_instructions_on_agent_state(tmp_path: Path) -> None:
    """The tool records a ``RunInstructions`` on ``agent._collected``."""
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)

    tool(
        run_command="python kernel.py --benchmark --shape 4096",
        out_path=str(out_path),
        modes_covered=["benchmark"],
        inferred_modes=["correctness"],
        notes="audit me",
    )

    assert "run_instructions" in agent._collected
    ri = agent._collected["run_instructions"]
    assert isinstance(ri, RunInstructions)
    assert ri.raw_command == "python kernel.py --benchmark --shape 4096"
    assert ri.modes_covered == ("benchmark",)
    assert ri.inferred_modes == ("correctness",)
    assert ri.notes == "audit me"

    assert agent._collected["commandment_path"] == str(out_path)


def test_tool_empty_run_command_raises_validation_error(tmp_path: Path) -> None:
    """Empty / whitespace ``run_command`` raises ``ValueError``."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)

    with pytest.raises(ValueError, match="non-empty shell command"):
        tool(run_command="", out_path=str(tmp_path / "CMD.md"))

    with pytest.raises(ValueError, match="non-empty shell command"):
        tool(run_command="   \n\t  ", out_path=str(tmp_path / "CMD.md"))


def test_tool_empty_out_path_raises_validation_error() -> None:
    """Empty ``out_path`` raises ``ValueError`` — we always write a file."""
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)

    with pytest.raises(ValueError, match="out_path"):
        tool(run_command="python kernel.py", out_path="")


def test_tool_commandment_path_resolves_to_readable_file(tmp_path: Path) -> None:
    """The returned ``commandment_path`` resolves to a file the test can read."""
    out_path = tmp_path / "out" / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)

    result = tool(
        run_command="./scripts/run.sh --benchmark",
        out_path=str(out_path),
        modes_covered=["benchmark"],
    )

    returned_path = Path(result["commandment_path"])
    assert returned_path.exists()
    assert returned_path == out_path
    assert "./scripts/run.sh --benchmark" in returned_path.read_text()


def test_tool_records_audit_preamble_with_user_command(tmp_path: Path) -> None:
    """The rendered COMMANDMENT carries the raw user command in an HTML
    comment preamble so downstream consumers can audit which command
    drove the Path-A short-circuit."""
    out_path = tmp_path / "COMMANDMENT.md"
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_commandment_from_user_command(agent)

    tool(
        run_command="make bench",
        out_path=str(out_path),
        modes_covered=["benchmark"],
        notes="make-target invocation",
    )

    body = out_path.read_text()
    assert "<!-- raw_command: make bench -->" in body
    assert "<!-- notes: make-target invocation -->" in body


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_has_required_run_command_and_out_path() -> None:
    """The schema declares ``run_command`` and ``out_path`` as required."""
    schema = _schema_commandment_from_user_command()
    assert schema["name"] == "commandment_from_user_command"
    assert schema["type"] == "function"
    required = schema["parameters"]["required"]
    assert "run_command" in required
    assert "out_path" in required


def test_schema_modes_enum_pins_four_canonical_names() -> None:
    """``modes_covered``/``inferred_modes`` enums are pinned to the 4 modes."""
    schema = _schema_commandment_from_user_command()
    props = schema["parameters"]["properties"]
    for field in ("modes_covered", "inferred_modes"):
        enum = props[field]["items"]["enum"]
        assert sorted(enum) == sorted(PATH_A_MODES)


def test_schema_accepts_known_good_call() -> None:
    """A call with all required fields validates against the schema."""
    schema = _schema_commandment_from_user_command()
    ok, msg = validate_call_against_schema(
        schema,
        {
            "run_command": "python k.py --benchmark",
            "out_path": "/tmp/COMMANDMENT.md",
        },
    )
    assert ok, msg


def test_schema_rejects_call_missing_run_command() -> None:
    """A call missing ``run_command`` fails validation."""
    schema = _schema_commandment_from_user_command()
    ok, msg = validate_call_against_schema(schema, {"out_path": "/tmp/CMD.md"})
    assert ok is False
    assert "run_command" in msg


def test_schema_rejects_call_missing_out_path() -> None:
    """A call missing ``out_path`` fails validation."""
    schema = _schema_commandment_from_user_command()
    ok, msg = validate_call_against_schema(schema, {"run_command": "python k.py --benchmark"})
    assert ok is False
    assert "out_path" in msg


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_default_tools_registers_commandment_from_user_command(tmp_path: Path) -> None:
    """The new tool is part of the default registration helper."""
    agent = _make_agent_with_path_a_tool(tmp_path)
    assert "commandment_from_user_command" in agent.tool_names


def test_register_default_tools_now_has_eight_tools(tmp_path: Path) -> None:
    """Default registration now includes the Path-A tool — 8 total."""
    agent = _make_agent_with_path_a_tool(tmp_path)
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


# ---------------------------------------------------------------------------
# collect_baseline / collect_profile tolerance of LLM-invented kwargs (B1)
# ---------------------------------------------------------------------------


def _stub_baseline(harness_path: Path, **_: object) -> BaselineMetrics:
    """Return a deterministic ``BaselineMetrics`` without running the harness."""
    return BaselineMetrics(
        harness_path=Path(harness_path),
        median_ms=1.25,
        samples_ms=[1.2, 1.25, 1.3],
        stdev_ms=0.05,
        repeats=3,
        command="<stub>",
    )


def _stub_profile(harness_path: Path, **kwargs: object) -> ProfileResult:
    """Return a deterministic ``ProfileResult`` without running the profiler."""
    out_path = kwargs.get("out_path")
    return ProfileResult(
        harness_path=Path(harness_path),
        command="<stub>",
        profile={"success": True, "stub": True},
        profile_path=Path(out_path) if out_path else None,
        backend="stub",
    )


def test_collect_baseline_tolerates_out_path_kwarg(monkeypatch, tmp_path: Path) -> None:
    """LLM-invented ``out_path=`` no longer aborts the tool with a TypeError."""
    monkeypatch.setattr(_tools_mod, "collect_baseline_metrics", _stub_baseline)
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_collect_baseline(agent)

    result = tool(harness_path=str(tmp_path / "h.py"), out_path=str(tmp_path / "baseline.json"))

    assert result["ok"] is True
    assert result["median_ms"] == 1.25
    assert result["repeats"] == 3


def test_collect_baseline_tolerates_repo_root_and_output_dir_kwargs(monkeypatch, tmp_path: Path) -> None:
    """LLM-invented ``repo_root=`` / ``output_dir=`` are silently ignored."""
    monkeypatch.setattr(_tools_mod, "collect_baseline_metrics", _stub_baseline)
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_collect_baseline(agent)

    result = tool(
        harness_path=str(tmp_path / "h.py"),
        repo_root="/some/repo",
        output_dir=str(tmp_path / "out"),
    )

    assert result["ok"] is True
    assert result["median_ms"] == 1.25


def test_collect_baseline_logs_extra_kwargs_at_debug(monkeypatch, tmp_path: Path) -> None:
    """Unexpected kwargs are recorded at DEBUG so the LLM behaviour is auditable.

    We bypass ``caplog`` and attach a local handler — caplog's interaction
    with pytest's log capture in this repo's test config doesn't reliably
    surface the records (the message goes to stdout via the streamhandler
    set up by other tests instead).
    """
    import logging as _logging

    monkeypatch.setattr(_tools_mod, "collect_baseline_metrics", _stub_baseline)
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_collect_baseline(agent)

    records: list[_logging.LogRecord] = []

    class _Capture(_logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(level=_logging.DEBUG)
    logger = _tools_mod.logger
    prev_level = logger.level
    logger.setLevel(_logging.DEBUG)
    logger.addHandler(handler)
    try:
        tool(harness_path=str(tmp_path / "h.py"), repo_root="/repo", out_path="/tmp/x.json")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    msgs = [r.getMessage() for r in records if r.levelno == _logging.DEBUG]
    assert any("collect_baseline ignored extra kwargs" in m for m in msgs)


def test_collect_baseline_no_extras_still_works(monkeypatch, tmp_path: Path) -> None:
    """The original call shape (no extras) returns the same dict it always did."""
    monkeypatch.setattr(_tools_mod, "collect_baseline_metrics", _stub_baseline)
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_collect_baseline(agent)

    result = tool(harness_path=str(tmp_path / "h.py"), repeats=3, gpu_id=0)

    assert set(result.keys()) == {"ok", "median_ms", "samples_ms", "stdev_ms", "repeats", "command"}
    assert agent._collected["baseline"].median_ms == 1.25


def test_collect_profile_tolerates_repo_root_and_output_dir_kwargs(monkeypatch, tmp_path: Path) -> None:
    """``collect_profile`` mirrors the same kwarg-tolerance contract as baseline."""
    monkeypatch.setattr(_tools_mod, "collect_profile", _stub_profile)
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_collect_profile(agent)

    result = tool(
        harness_path=str(tmp_path / "h.py"),
        repo_root="/some/repo",
        output_dir=str(tmp_path / "out"),
    )

    assert result["ok"] is True
    assert result["backend"] == "stub"


def test_collect_profile_no_extras_still_works(monkeypatch, tmp_path: Path) -> None:
    """Calling without extras still works (existing contract preserved)."""
    monkeypatch.setattr(_tools_mod, "collect_profile", _stub_profile)
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    tool = _make_tool_collect_profile(agent)

    result = tool(harness_path=str(tmp_path / "h.py"))

    assert result["ok"] is True
    assert result["command"] == "<stub>"


# ---------------------------------------------------------------------------
# render_commandment defaults out_path from output_dir (B2)
# ---------------------------------------------------------------------------


def _stub_render_commandment(_lang, _ctx, *, out_path):
    """Stub ``render_commandment`` — writes a tiny marker file + returns its text."""
    text = "# Commandment\n(stubbed)\n"
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text)
    return text


def test_render_commandment_defaults_out_path_from_output_dir(monkeypatch, tmp_path: Path) -> None:
    """When ``out_path`` is omitted, the tool writes to ``<output_dir>/COMMANDMENT.md``."""
    monkeypatch.setattr(_tools_mod, "render_commandment", _stub_render_commandment)
    output_dir = tmp_path / "outputs"
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    agent._extra_template_vars = {"output_dir": str(output_dir)}
    tool = _make_tool_render_commandment(agent, _LANG)

    result = tool(
        kernel_path=str(tmp_path / "k.py"),
        harness_path=str(tmp_path / "h.py"),
        repo_root=str(tmp_path),
    )

    expected = output_dir / "COMMANDMENT.md"
    assert result["ok"] is True
    assert result["out_path"] == str(expected)
    assert expected.exists()
    assert agent._collected["commandment_path"] == str(expected)


def test_render_commandment_explicit_out_path_preserved(monkeypatch, tmp_path: Path) -> None:
    """When ``out_path`` is explicit, it is used verbatim (existing behaviour)."""
    monkeypatch.setattr(_tools_mod, "render_commandment", _stub_render_commandment)
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    agent._extra_template_vars = {"output_dir": str(tmp_path / "ignored")}
    tool = _make_tool_render_commandment(agent, _LANG)

    explicit_path = tmp_path / "custom" / "C.md"
    result = tool(
        kernel_path=str(tmp_path / "k.py"),
        harness_path=str(tmp_path / "h.py"),
        repo_root=str(tmp_path),
        out_path=str(explicit_path),
    )

    assert result["out_path"] == str(explicit_path)
    assert explicit_path.exists()
    assert not (tmp_path / "ignored" / "COMMANDMENT.md").exists()


def test_render_commandment_missing_out_path_and_output_dir_raises(monkeypatch, tmp_path: Path) -> None:
    """No ``out_path`` and no ``output_dir`` on the agent is a hard error."""
    monkeypatch.setattr(_tools_mod, "render_commandment", _stub_render_commandment)
    agent = PreprocessOrchestratorAgent(model=_StubModel())
    agent._extra_template_vars = {}
    tool = _make_tool_render_commandment(agent, _LANG)

    with pytest.raises(ValueError, match="out_path"):
        tool(
            kernel_path=str(tmp_path / "k.py"),
            harness_path=str(tmp_path / "h.py"),
            repo_root=str(tmp_path),
        )


def test_render_commandment_schema_out_path_no_longer_required() -> None:
    """The schema declares ``out_path`` as optional so the LLM can omit it."""
    schema = _schema_render_commandment()
    required = schema["parameters"]["required"]
    assert "out_path" not in required
    assert set(required) == {"kernel_path", "harness_path", "repo_root"}
    out_path_desc = schema["parameters"]["properties"]["out_path"]["description"]
    assert "Optional" in out_path_desc
