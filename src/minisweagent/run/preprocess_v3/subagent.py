"""v3-native subagent class consumed by :class:`PreprocessSubagentDispatcher`.

Replaces the legacy ``DefaultAgent`` fallback that ``tools.py`` previously
lazy-imported when no ``agent_factory`` was supplied. The legacy class
worked but dragged in a 600-line tool surface — strategy manager, save/test
context, working memory, RAG postprocessor, MCP bridge plumbing — none of
which is appropriate for a v3 preprocess subagent (those subagents only
need ``bash`` + ``str_replace_editor`` + optionally ``save_and_test`` per
their YAML ``tools:`` lists).

Design notes
------------

* **Minimal mirror of ``DefaultAgent.run`` shape.** Returns
  ``(exit_status, final_message)`` so the dispatcher's existing success
  detection (``exit_status in {"Submitted", "FinishedSuccessfully"} or
  "VERIFIED=true" in message``) keeps working unchanged.
* **Explicit tool whitelisting.** ``spec.tools`` is enforced at construction
  time against the built-in v3 tool registry (currently ``bash``,
  ``str_replace_editor``, ``save_and_test``). Unknown names raise a clear
  error — no silent skip, no defensive fallback.
* **``step_limit == 0`` means unlimited.** Matches the convention from
  ``DefaultAgent.AgentConfig`` and from the v3 ``UNLIMITED_MAX_STEPS``
  sentinel projection in :class:`PreprocessSubagentDispatcher`.
* **No skill runtime, no working memory, no select-patch agent.** Those
  are heterogeneous-optimization concerns. Preprocess subagents are
  short-lived and produce a single deliverable (harness file / verifier
  verdict / compute_speedup.py); the surrounding scaffolding only adds
  latency and failure modes here.

The class is intentionally not exported from
:mod:`minisweagent.run.preprocess_v3.tools` — callers that need a custom
runtime inject an ``agent_factory`` (matching the test seam). Production
goes through :class:`PreprocessSubagent` automatically.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions (mirror DefaultAgent's primitives, but local to v3)
# ---------------------------------------------------------------------------


class PreprocessSubagentError(Exception):
    """Base error for :class:`PreprocessSubagent` itself."""


class UnknownToolError(PreprocessSubagentError):
    """A YAML ``tools:`` entry doesn't map to a known v3 tool name."""


class _NonTerminating(Exception):
    """Local mirror of ``DefaultAgent.NonTerminatingException`` — recoverable."""


class _FormatError(_NonTerminating):
    """The model produced a response we could not parse (no tool call, no bash)."""


class _Terminating(Exception):
    """Local mirror of ``DefaultAgent.TerminatingException`` — ends the loop."""


class _Submitted(_Terminating):
    """The model signalled completion via the submit-marker bash echo."""


class _ToolSubmitted(_Terminating):
    """The model signalled completion via the ``submit`` tool."""


class _LimitsExceeded(_Terminating):
    """The step or cost budget was exhausted."""


# ---------------------------------------------------------------------------
# Tool registry (v3-native, whitelisted by name)
#
# Each entry is a zero-arg factory returning ``(callable, schema_dict)``.
# Callables accept kwargs from the LLM tool call and return a dict with
# ``output: str`` and ``returncode: int`` — same shape the legacy bash /
# str_replace_editor / save_and_test tools produce, so the dispatcher's
# success heuristics keep working.
# ---------------------------------------------------------------------------


def _factory_bash() -> tuple[Callable[..., dict[str, Any]], dict[str, Any]]:
    from minisweagent.tools.bash_command import BashCommand

    tool = BashCommand()
    schema = {
        "name": "bash",
        "type": "function",
        "description": (
            "Run a bash command in the subagent's working directory. Returns "
            "the command's combined stdout/stderr and exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["command"],
        },
    }
    return tool, schema


def _factory_str_replace_editor() -> tuple[Callable[..., dict[str, Any]], dict[str, Any]]:
    from minisweagent.tools.str_replace_editor import str_replace_editor

    tool = str_replace_editor()
    schema = {
        "name": "str_replace_editor",
        "type": "function",
        "description": ("View, create, or edit files (view / create / str_replace / insert subcommands)."),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                },
                "path": {"type": "string"},
                "file_text": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
                "insert_line": {"type": "integer"},
                "view_range": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["command", "path"],
        },
    }
    return tool, schema


def _factory_save_and_test() -> tuple[Callable[..., dict[str, Any]], dict[str, Any]]:
    from minisweagent.tools.save_and_test import SaveAndTestTool

    tool = SaveAndTestTool()
    schema = {
        "name": "save_and_test",
        "type": "function",
        "description": (
            "Save the current patch to the configured patch_output_dir and run the configured test_command against it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": [],
        },
    }
    return tool, schema


#: Known v3 tool names, mapped to zero-arg factories. The registry is
#: deliberately small — preprocess subagents do not need the full
#: optimisation tool surface (strategy manager, RAG, MCP bridges). Add
#: entries here when a new SUBAGENT.yaml legitimately needs them.
_TOOL_FACTORIES: dict[str, Callable[[], tuple[Callable[..., dict[str, Any]], dict[str, Any]]]] = {
    "bash": _factory_bash,
    "str_replace_editor": _factory_str_replace_editor,
    "save_and_test": _factory_save_and_test,
}


def known_tool_names() -> list[str]:
    """Sorted list of v3-known tool names. Exposed for tests."""
    return sorted(_TOOL_FACTORIES.keys())


# ---------------------------------------------------------------------------
# Config + agent
# ---------------------------------------------------------------------------


@dataclass
class _SubagentMessageState:
    """Mutable state the loop threads through; kept in a dataclass to make
    tests + debugging readable.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    n_calls: int = 0


_SUBMIT_MARKERS: tuple[str, ...] = ("MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")

_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)


class PreprocessSubagent:
    """v3-native subagent. Drop-in replacement for ``DefaultAgent`` inside
    :class:`PreprocessSubagentDispatcher`.

    Construction:

        agent = PreprocessSubagent(
            model=...,
            system_prompt="You are TestHarnessAgent. ...",
            tools=["bash", "str_replace_editor", "save_and_test"],
            step_limit=0,  # 0 == unlimited
            cwd="/path/to/repo",
        )

    Behaviour:

    * ``run(task)`` returns ``(exit_status, final_message)``.
    * ``exit_status`` is one of: ``"Submitted"``, ``"LimitsExceeded"``,
      or ``type(exc).__name__`` for any other terminating exception.
    * The system prompt is treated as a Jinja template so the YAML's
      ``{{kernel_path}}``-style placeholders interpolate against
      ``extra_template_vars`` (caller-supplied) without forcing the
      caller to pre-render.
    * Bash submit markers (``MINI_SWE_AGENT_FINAL_OUTPUT`` /
      ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``) are honoured so v3
      subagent YAMLs that prescribe them keep working unchanged.
    * The ``submit`` tool (if the LLM somehow calls it via the OpenAI
      tool-call channel) also ends the run cleanly.

    Not modelled (intentional simplifications vs. ``DefaultAgent``):

    * Trajectory dumping, skill runtime, working memory, select-patch
      agent, summary-on-cost-limit step. Preprocess subagents produce a
      single deliverable and don't benefit from those layers.
    """

    def __init__(
        self,
        *,
        model: Any,
        system_prompt: str,
        tools: list[str] | None = None,
        step_limit: int = 0,
        cost_limit: float = 0.0,
        cwd: str | Path | None = None,
        log_path: Path | None = None,
        extra_template_vars: dict[str, Any] | None = None,
        instance_template: str = "{{task}}",
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.instance_template = instance_template
        self.step_limit = int(step_limit)
        self.cost_limit = float(cost_limit)
        self.cwd = str(cwd) if cwd is not None else None
        self.log_path = Path(log_path) if log_path is not None else None
        self.extra_template_vars = dict(extra_template_vars or {})

        self._tool_table: dict[str, Callable[..., dict[str, Any]]] = {}
        self._tool_schemas: list[dict[str, Any]] = []
        self._register_tools(tools)

        self._state = _SubagentMessageState()

    # -----------------------------------------------------------------
    # Tool wiring (whitelisting against the v3 registry)
    # -----------------------------------------------------------------

    def _register_tools(self, names: list[str] | None) -> None:
        if not names:
            return
        unknown = [n for n in names if n not in _TOOL_FACTORIES]
        if unknown:
            raise UnknownToolError(
                f"PreprocessSubagent: unknown tool name(s) {unknown!r}; known tools are {known_tool_names()!r}"
            )
        for name in names:
            callable_, schema = _TOOL_FACTORIES[name]()
            self._tool_table[name] = callable_
            self._tool_schemas.append(schema)

        tool_env = self.extra_template_vars.get("_tool_env")
        if not isinstance(tool_env, dict):
            tool_env = {}
        if self.cwd:
            bash = self._tool_table.get("bash")
            if bash is not None and hasattr(bash, "_cwd"):
                bash._cwd = self.cwd
        for tool in self._tool_table.values():
            if hasattr(tool, "_env_override"):
                tool._env_override.update({str(k): str(v) for k, v in tool_env.items()})

    @property
    def tool_names(self) -> list[str]:
        """Sorted list of tool names registered on this subagent."""
        return sorted(self._tool_table.keys())

    # -----------------------------------------------------------------
    # Template rendering + message history
    # -----------------------------------------------------------------

    def render_template(self, template: str, **kwargs: Any) -> str:
        """Render a Jinja template with ``extra_template_vars`` merged in."""
        all_vars = {**self.extra_template_vars, **kwargs}
        return Template(template, undefined=StrictUndefined).render(**all_vars)

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        """Append a message to the history; optionally tee to ``log_path``."""
        self._state.messages.append({"role": role, "content": content, **extra})
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{role}] {content}\n")
            except Exception as exc:
                logger.debug("PreprocessSubagent log write failed: %s", exc)

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Current message history. Exposed for tests / debugging."""
        return list(self._state.messages)

    # -----------------------------------------------------------------
    # Tool dispatch
    # -----------------------------------------------------------------

    def _inject_tools_into_model(self) -> None:
        """Tell the model which tools the subagent can call.

        Same pattern as :class:`PreprocessOrchestratorAgent._inject_tools_into_model`.
        """
        if not self._tool_schemas:
            return
        if hasattr(self.model, "set_tools"):
            try:
                self.model.set_tools(self._tool_schemas)
                return
            except Exception as exc:
                logger.debug("PreprocessSubagent set_tools failed: %s", exc)
        impl = getattr(self.model, "_impl", self.model)
        try:
            impl.tools = self._tool_schemas
        except Exception as exc:
            logger.debug("PreprocessSubagent tool attribute set failed: %s", exc)

    # -----------------------------------------------------------------
    # Step loop
    # -----------------------------------------------------------------

    def _check_limits(self) -> None:
        if self.step_limit > 0 and self._state.n_calls >= self.step_limit:
            raise _LimitsExceeded(f"step_limit reached: {self._state.n_calls} >= {self.step_limit}")
        cost = float(getattr(self.model, "cost", 0.0) or 0.0)
        if self.cost_limit > 0 and cost >= self.cost_limit:
            raise _LimitsExceeded(f"cost_limit reached: ${cost:.2f} >= ${self.cost_limit:.2f}")

    def step(self) -> dict[str, Any]:
        """Run one LLM turn. Returns the observation dict."""
        self._check_limits()
        response = self.model.query(self._state.messages)
        self._state.n_calls = int(getattr(self.model, "n_calls", self._state.n_calls + 1))
        return self._handle_response(response)

    def _handle_response(self, response: dict[str, Any]) -> dict[str, Any]:
        content = response.get("content", "") if isinstance(response, dict) else ""
        tool_call = response.get("tools") if isinstance(response, dict) else None

        msg_extra: dict[str, Any] = {}
        if tool_call:
            msg_extra["tool_calls"] = tool_call
        self.add_message("assistant", content, **msg_extra)

        # Prefer a bash submit marker when present — it short-circuits
        # the rest of the response handling (matches DefaultAgent).
        bash_actions = _BASH_BLOCK_RE.findall(content or "")
        if len(bash_actions) == 1:
            return self._dispatch_bash(bash_actions[0].strip())

        if tool_call:
            return self._dispatch_tool_call(tool_call)

        raise _FormatError(
            "PreprocessSubagent: response had no parseable action (no single "
            "bash block, no tool call). Please reply with either one bash "
            "block in triple backticks or one tool call."
        )

    def _dispatch_bash(self, command: str) -> dict[str, Any]:
        bash = self._tool_table.get("bash")
        if bash is None:
            raise _FormatError(
                "PreprocessSubagent: bash block produced but the 'bash' tool "
                "is not in this subagent's whitelisted tools."
            )
        result = bash(command=command)
        output = result.get("output", "") if isinstance(result, dict) else str(result)
        rc = int(result.get("returncode", 0)) if isinstance(result, dict) else 0
        self.add_message("user", f"Observation (rc={rc}): {output}")
        self._check_submit_marker(output)
        return {"output": output, "returncode": rc}

    def _dispatch_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        name = function.get("name", "")
        raw_args = function.get("arguments", {})
        args = self._parse_args(raw_args)

        if name == "submit":
            final = args.get("output") or args.get("final_output") or ""
            raise _ToolSubmitted(str(final))

        callable_ = self._tool_table.get(name)
        if callable_ is None:
            obs = {
                "error": f"Unknown tool {name!r}. Available: {self.tool_names!r}",
                "output": "",
                "returncode": 1,
            }
            self.add_message(
                "tool",
                json.dumps(obs, default=str),
                tool_call_id=tool_call.get("id", ""),
                name=name,
            )
            return obs

        try:
            result = callable_(**args)
        except Exception as exc:
            logger.exception("PreprocessSubagent tool %r raised", name)
            obs = {"error": f"{type(exc).__name__}: {exc}", "output": "", "returncode": 1}
            self.add_message(
                "tool",
                json.dumps(obs, default=str),
                tool_call_id=tool_call.get("id", ""),
                name=name,
            )
            return obs

        if not isinstance(result, dict):
            result = {"output": str(result), "returncode": 0}
        self.add_message(
            "tool",
            json.dumps(result, default=str),
            tool_call_id=tool_call.get("id", ""),
            name=name,
        )
        self._check_submit_marker(result.get("output", "") or "")
        return result

    @staticmethod
    def _parse_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("PreprocessSubagent: tool args not valid JSON (raw=%r)", raw[:200])
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _check_submit_marker(self, output: str) -> None:
        """Honour the legacy bash submit markers used by v3 subagent YAMLs."""
        first_line = output.lstrip().splitlines()[0:1]
        if not first_line:
            return
        if first_line[0].strip() in _SUBMIT_MARKERS:
            remainder = output.lstrip().split("\n", 1)
            final = remainder[1] if len(remainder) > 1 else ""
            raise _Submitted(final)

    # -----------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------

    def run(self, task: str, **template_kwargs: Any) -> tuple[str, str]:
        """Run the loop until termination. Returns ``(exit_status, final_message)``.

        ``exit_status`` mirrors the legacy ``DefaultAgent.run`` contract so
        the dispatcher's success heuristics keep working:

        * ``"Submitted"`` — clean completion (bash marker or submit tool).
        * ``"LimitsExceeded"`` — step / cost budget exhausted.
        * Other terminating exception types surface as their class name.
        """
        self._state = _SubagentMessageState()
        self.extra_template_vars |= {"task": task, **template_kwargs}

        self._inject_tools_into_model()

        self.add_message("system", self.render_template(self.system_prompt))
        self.add_message("user", self.render_template(self.instance_template))

        while True:
            try:
                self.step()
            except _NonTerminating as nt:
                self.add_message("user", str(nt))
            except _Terminating as term:
                msg = str(term)
                self.add_message("user", msg)
                return type(term).__name__.lstrip("_"), msg
            except Exception as exc:
                logger.exception("PreprocessSubagent.run crashed with unhandled exception")
                return type(exc).__name__, f"{type(exc).__name__}: {exc}"


__all__ = [
    "PreprocessSubagent",
    "PreprocessSubagentError",
    "UnknownToolError",
    "known_tool_names",
]
