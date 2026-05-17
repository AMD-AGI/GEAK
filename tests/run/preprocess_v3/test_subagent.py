"""Tests for :class:`PreprocessSubagent` — the v3-native subagent class.

These tests pin the contract the orchestrator dispatcher relies on:

* Step-limit interpretation (``0 == unlimited``, positive == bounded).
* Tool whitelisting against the v3 registry (unknown tool -> raise).
* Run-loop shape ``(exit_status, message)`` for the three terminating
  cases the dispatcher's success heuristic checks.
* Bash submit-marker handling (the v3 YAMLs prescribe these).
* OpenAI-style tool-call routing through the whitelisted tools.

The model is mocked end-to-end — no real LLM, no real subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from minisweagent.run.preprocess_v3.subagent import (
    PreprocessSubagent,
    UnknownToolError,
    known_tool_names,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Model stub that returns a queue of canned responses.

    Each entry can be:

    * a dict in the OpenAI response shape (``content`` / ``tools``)
    * a ``(name, args)`` tuple wrapped into a tool-call automatically
    * a plain string wrapped into ``{"content": "..."}``
    """

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.n_calls = 0
        self.cost = 0.0
        self.tools_set: list[dict[str, Any]] | None = None
        self.queries: list[list[dict[str, Any]]] = []
        self._call_id = 0

    def set_tools(self, schemas: list[dict[str, Any]]) -> None:
        self.tools_set = list(schemas)

    def query(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.queries.append(list(messages))
        self.n_calls += 1
        if not self.script:
            return {"content": "I'm done."}
        entry = self.script.pop(0)
        if isinstance(entry, dict):
            return entry
        if isinstance(entry, str):
            return {"content": entry}
        name, args = entry
        self._call_id += 1
        return {
            "content": "",
            "tools": {
                "id": f"call_{self._call_id}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            },
        }


@pytest.fixture
def stub_bash(monkeypatch):
    """Patch the BashCommand class so no real bash runs.

    The stub interprets simple ``echo`` chains (which is all the v3 YAMLs
    actually use for the submit marker) so the agent's submit-marker
    detection keeps working end-to-end. Anything more complex returns a
    canned ``ran: <command>`` string with rc=0.

    The returned ``calls`` list records every invocation so tests can
    assert dispatch behaviour without a subprocess.
    """
    calls: list[dict[str, Any]] = []

    class _StubBash:
        def __init__(self):
            self._cwd: str | None = None
            self._env_override: dict[str, str] | None = None

        def __call__(self, command: str = "", **_kwargs) -> dict[str, Any]:
            calls.append({"command": command, "cwd": self._cwd})
            # Cheap shell-ish echo simulator: each newline-separated line
            # whose first token is ``echo`` contributes the rest of the
            # line to the output. Anything else is summarised verbatim.
            # Just enough to drive the agent's submit-marker detection in
            # tests; not a real shell.
            out_lines: list[str] = []
            for raw_line in command.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                if stripped.startswith("echo "):
                    payload = stripped[len("echo ") :].strip()
                    if (payload.startswith("'") and payload.endswith("'")) or (
                        payload.startswith('"') and payload.endswith('"')
                    ):
                        payload = payload[1:-1]
                    out_lines.append(payload)
                else:
                    out_lines.append(f"ran: {stripped}")
            return {"output": "\n".join(out_lines), "returncode": 0}

    monkeypatch.setattr("minisweagent.tools.bash_command.BashCommand", _StubBash)
    return calls


@pytest.fixture
def stub_editor(monkeypatch):
    """Patch the str_replace_editor factory so no real file edits happen."""
    calls: list[dict[str, Any]] = []

    def _stub_factory():
        def _runner(**kwargs):
            calls.append(kwargs)
            return {"output": f"edited: {kwargs.get('path')}", "returncode": 0}

        return _runner

    monkeypatch.setattr("minisweagent.tools.str_replace_editor.str_replace_editor", _stub_factory)
    return calls


@pytest.fixture
def stub_save_and_test(monkeypatch):
    """Patch the SaveAndTestTool class so we don't need a save/test context."""
    calls: list[dict[str, Any]] = []

    class _StubSaveAndTest:
        def __call__(self, **kwargs) -> dict[str, Any]:
            calls.append(kwargs)
            return {"output": "saved & tested", "returncode": 0}

    monkeypatch.setattr("minisweagent.tools.save_and_test.SaveAndTestTool", _StubSaveAndTest)
    return calls


# ---------------------------------------------------------------------------
# Tool whitelisting
# ---------------------------------------------------------------------------


def test_known_tool_names_lists_v3_tools() -> None:
    """The v3 tool registry exposes the three preprocess-subagent tools."""
    names = known_tool_names()
    assert "bash" in names
    assert "str_replace_editor" in names
    assert "save_and_test" in names


def test_whitelist_empty_tool_list_runs_with_no_tools(stub_bash) -> None:
    """A subagent constructed with ``tools=[]`` has no tools but can still run."""
    agent = PreprocessSubagent(
        model=_ScriptedModel([]),
        system_prompt="You are a tool-less subagent.",
        tools=[],
        step_limit=1,
    )
    assert agent.tool_names == []


def test_whitelist_only_exposes_requested_tools(stub_bash) -> None:
    """Passing ``['bash']`` exposes only ``bash``; ``str_replace_editor`` is absent."""
    agent = PreprocessSubagent(
        model=_ScriptedModel([]),
        system_prompt="You are.",
        tools=["bash"],
    )
    assert agent.tool_names == ["bash"]


def test_whitelist_rejects_unknown_tool(stub_bash) -> None:
    """Unknown tool names raise :class:`UnknownToolError` at construction time."""
    with pytest.raises(UnknownToolError) as excinfo:
        PreprocessSubagent(
            model=_ScriptedModel([]),
            system_prompt="You are.",
            tools=["does_not_exist", "bash"],
        )
    assert "does_not_exist" in str(excinfo.value)
    assert "known tools" in str(excinfo.value)


def test_whitelist_propagates_cwd_to_bash(stub_bash) -> None:
    """``cwd`` is forwarded to the bash tool's internal ``_cwd`` slot."""
    agent = PreprocessSubagent(
        model=_ScriptedModel([]),
        system_prompt="You are.",
        tools=["bash"],
        cwd="/tmp/some_repo",
    )
    bash = agent._tool_table["bash"]
    assert getattr(bash, "_cwd", None) == "/tmp/some_repo"


# ---------------------------------------------------------------------------
# Step-limit interpretation
# ---------------------------------------------------------------------------


def test_step_limit_zero_is_unlimited(stub_bash) -> None:
    """``step_limit=0`` means "no cap" — must mirror DefaultAgent's convention.

    Drive the model with a long script that would exceed any positive
    limit and confirm the agent runs through it without raising
    ``LimitsExceeded``.
    """
    # 50 bash echoes then a submit marker.
    bash_block = "```bash\necho ok\n```"
    submit_block = "```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\necho final-message\n```"
    script = [bash_block] * 50 + [submit_block]

    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=0,
    )
    exit_status, _ = agent.run("just keep going")
    assert exit_status == "Submitted"


def test_step_limit_positive_caps_execution(stub_bash) -> None:
    """A positive ``step_limit`` is enforced when the model keeps producing turns."""
    # Bash block that NEVER prints the submit marker — agent should hit the cap.
    bash_block = "```bash\necho not-done\n```"
    agent = PreprocessSubagent(
        model=_ScriptedModel([bash_block] * 100),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=3,
    )
    exit_status, msg = agent.run("loop until cap")
    assert exit_status == "LimitsExceeded"
    assert "step_limit" in msg


def test_step_limit_one_runs_a_single_turn(stub_bash) -> None:
    """``step_limit=1`` allows exactly one turn before raising."""
    agent = PreprocessSubagent(
        model=_ScriptedModel(["```bash\necho once\n```"] * 5),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=1,
    )
    exit_status, _ = agent.run("one turn only")
    assert exit_status == "LimitsExceeded"
    assert agent._state.n_calls == 1


# ---------------------------------------------------------------------------
# Run loop shape
# ---------------------------------------------------------------------------


def test_run_returns_exit_status_and_message_tuple(stub_bash) -> None:
    """``run`` returns ``(exit_status, final_message)`` per the legacy contract."""
    script = ["```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\necho all-done\n```"]
    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=0,
    )
    result = agent.run("submit immediately")
    assert isinstance(result, tuple) and len(result) == 2
    exit_status, message = result
    assert exit_status == "Submitted"
    assert "all-done" in message


def test_run_clears_messages_on_re_entry(stub_bash) -> None:
    """``run`` resets the message log so a second invocation starts fresh."""
    bash_block = "```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\necho fin\n```"
    agent = PreprocessSubagent(
        model=_ScriptedModel([bash_block, bash_block]),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=0,
    )
    agent.run("task one")
    msgs_after_first = len(agent.messages)
    agent.run("task two")
    assert len(agent.messages) <= msgs_after_first  # not appended on top of first run


def test_run_renders_system_prompt_jinja_placeholders(stub_bash) -> None:
    """``{{kernel_path}}``-style placeholders in the system prompt interpolate."""
    script = ["```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\n```"]
    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You optimise {{kernel_path}} on gpu {{gpu_id}}.",
        tools=["bash"],
        step_limit=0,
        extra_template_vars={"kernel_path": "/tmp/k.py", "gpu_id": 7},
    )
    agent.run("go")
    system_msg = agent.messages[0]
    assert system_msg["role"] == "system"
    assert "/tmp/k.py" in system_msg["content"]
    assert "gpu 7" in system_msg["content"]


def test_run_handles_format_error_without_terminating(stub_bash) -> None:
    """A text-only response (no bash, no tool call) is recoverable, not terminal."""
    script = [
        "I'm thinking about the problem.",  # no bash, no tool call -> FormatError
        "```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\necho recovered\n```",
    ]
    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=10,
    )
    exit_status, msg = agent.run("recover from format error")
    assert exit_status == "Submitted"
    assert "recovered" in msg


# ---------------------------------------------------------------------------
# Submit semantics — both bash marker and submit tool
# ---------------------------------------------------------------------------


def test_bash_submit_marker_terminates_with_remainder(stub_bash) -> None:
    """The bash submit marker hands the rest of the output to ``Submitted``."""
    script = [
        "```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\necho 'TEST_COMMAND: cd /repo && python harness.py'\n```",
    ]
    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=0,
    )
    exit_status, msg = agent.run("submit with payload")
    assert exit_status == "Submitted"
    assert "TEST_COMMAND" in msg


def test_complete_task_marker_also_terminates(stub_bash) -> None:
    """The alternate ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` marker also works."""
    script = [
        "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho 'done'\n```",
    ]
    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=0,
    )
    exit_status, _ = agent.run("alternate marker")
    assert exit_status == "Submitted"


# ---------------------------------------------------------------------------
# Tool-call routing (OpenAI shape)
# ---------------------------------------------------------------------------


def test_tool_call_dispatches_to_whitelisted_tool(stub_bash, stub_editor) -> None:
    """An OpenAI tool call for ``str_replace_editor`` is routed and recorded."""
    script = [
        ("str_replace_editor", {"command": "create", "path": "/tmp/harness.py", "file_text": "x"}),
        "```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\n```",
    ]
    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You are.",
        tools=["bash", "str_replace_editor"],
        step_limit=0,
    )
    exit_status, _ = agent.run("create then submit")
    assert exit_status == "Submitted"
    assert len(stub_editor) == 1
    assert stub_editor[0]["path"] == "/tmp/harness.py"


def test_tool_call_for_unknown_tool_returns_error_obs(stub_bash) -> None:
    """An unknown-tool call produces an error observation but doesn't crash."""
    script = [
        ("not_a_real_tool", {"x": 1}),
        "```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\n```",
    ]
    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=0,
    )
    exit_status, _ = agent.run("attempt unknown tool")
    assert exit_status == "Submitted"
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "Unknown tool" in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_log_path_records_messages(stub_bash, tmp_path: Path) -> None:
    """When ``log_path`` is set, messages are teed to disk in append order."""
    log_path = tmp_path / "child.log"
    script = ["```bash\necho MINI_SWE_AGENT_FINAL_OUTPUT\n```"]
    agent = PreprocessSubagent(
        model=_ScriptedModel(script),
        system_prompt="You are.",
        tools=["bash"],
        step_limit=0,
        log_path=log_path,
    )
    agent.run("log me")
    assert log_path.is_file()
    content = log_path.read_text(encoding="utf-8")
    assert "[system]" in content
    assert "[user]" in content
