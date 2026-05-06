"""OptimizationAgent — THE unified agent class for GEAK.

Standalone (no inheritance). Replaces the 4-layer chain:

    DefaultAgent (611 LoC)
      └── InteractiveAgent (181 LoC)    — adds yolo/confirm/human modes, rich console
            └── StrategyAgent (156 LoC) — adds strategy_manager callback + tool_profile
                  └── StrategyInteractiveAgent (50 LoC) — overrides notify_strategy_changed

Production always uses `mode="yolo"` (verified by audit). Confirm/human modes
and user-confirmation prompts are DEAD CODE and are NOT ported into this class.

All other behavior is preserved byte-compatible with StrategyInteractiveAgent:
- Step loop, tool runtime, message history, trajectory saving
- Rich console output on add_message (inlined from InteractiveAgent)
- Strategy manager callback wiring (inlined from StrategyAgent)
- Strategy-list console formatting (inlined from StrategyInteractiveAgent)
- SelectPatchAgent dispatch on agent completion

See docs/refactor/EXECUTION_PLAN.md §4, §16.2.

NOTE on temporary transition:
  SelectPatchAgent, UnitTestAgent, and ShapeFixerAgent currently inherit from
  the old DefaultAgent. Until they migrate to SubagentBase (next phase of
  work), they inherit from OptimizationAgent as a single-level inheritance
  (cleaner than the old 4-layer chain). The `check_no_agent_inheritance.py`
  CI gate allows this transitional inheritance and flips to FAIL-strict after
  the SubagentBase migration.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from jinja2 import StrictUndefined, Template
from rich.console import Console
from rich.rule import Rule

from minisweagent import Environment, Model
from minisweagent.skills.skill_runtime import SkillRuntime
from minisweagent.tools.tools_runtime import ToolRuntime

logger = logging.getLogger(__name__)
console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Exceptions (shared primitives — callers import from this module)
# ---------------------------------------------------------------------------

class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Unified configuration for OptimizationAgent.

    Merged from AgentConfig (DefaultAgent) + InteractiveAgentConfig additions
    that production uses + StrategyAgent tool_profile. Dead interactive-mode
    fields (whitelist_actions, mode="confirm"/"human") are NOT here.
    """

    # ─── prompting ───
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be "
        "'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"

    # ─── budgets ───
    step_limit: int = 0
    cost_limit: float = 3.0
    summary_on_cost_limit: bool = False
    """When True, on LimitsExceeded allow one extra step (e.g. to write a summary)."""
    summary_on_limit_prompt: str = (
        "The cost limit has been reached. Before stopping, run exactly one command to document "
        "what you did so far (e.g. create a summary file or add to your final output)."
    )

    # ─── patch + test ───
    save_patch: bool = True
    test_command: str | None = None
    patch_output_dir: str | None = None
    metric: str | None = None

    # ─── strategy manager ───
    use_strategy_manager: bool = False
    strategy_file_path: str = ".optimization_strategies.md"

    # ─── context ───
    profiling_type: str | None = None
    codebase_context: str | None = None
    starting_patch: str | None = None

    # ─── exit / tool control ───
    # confirm_exit is kept for caller-config compatibility but is UNUSED in
    # production (audit §7: input_allowed=False everywhere). The on-exit-confirm
    # path has been removed from has_finished().
    confirm_exit: bool = True
    disabled_tools: list[str] = field(default_factory=list)
    source_file_paths: list[str] | None = None
    use_skills: bool = False

    # ─── tool profile (from StrategyAgent) ───
    tool_profile: str = "full"
    """ToolRuntime profile: 'swe' for reduced tool set, 'full' for all tools."""

    # ─── interactive-mode field kept for caller-config compatibility ───
    # Production always uses "yolo". Confirm/human modes removed — they were
    # dead code in GEAK's automated pipeline. Accepting this field (defaulting
    # to "yolo") means existing YAML configs that set mode="yolo" still load
    # cleanly.
    mode: Literal["yolo"] = "yolo"


# ---------------------------------------------------------------------------
# Observation truncation (from DefaultAgent — byte-identical)
# ---------------------------------------------------------------------------

OBSERVATION_MAX_LEN: int = 10000
OBSERVATION_HEAD_LEN: int = 5000
OBSERVATION_TAIL_LEN: int = 5000
OBSERVATION_TRUNCATED_NOTICE: str = (
    "\n<warning>\n"
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller "
    "number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more "
    "selective search pattern.\n"
    "If you really need to see something from the full command's output, you can "
    "redirect output to a file and then search in that file.\n"
    "</warning>\n"
)


def truncate_observation(text: str) -> str:
    """Truncate long observation to head + notice + elided + tail."""
    if not text or len(text) <= OBSERVATION_MAX_LEN:
        return text
    elided = len(text) - OBSERVATION_HEAD_LEN - OBSERVATION_TAIL_LEN
    return (
        text[:OBSERVATION_HEAD_LEN]
        + OBSERVATION_TRUNCATED_NOTICE
        + f"<elided>{elided} characters elided</elided>\n"
        + text[-OBSERVATION_TAIL_LEN:]
    )


# ---------------------------------------------------------------------------
# OptimizationAgent — the unified class
# ---------------------------------------------------------------------------

class OptimizationAgent:
    """THE GEAK agent. Merges the 4-layer chain into one standalone class.

    Peer class of SubagentBase (subagents COMPOSE OptimizationAgent via
    `_make_optimization_agent` helper; they do NOT inherit). See
    docs/refactor/EXECUTION_PLAN.md §4 for the design rationale.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: Callable = AgentConfig,
        **kwargs,
    ):
        # ── Config construction (kwargs-based to preserve caller compatibility) ──
        self.config = config_class(**kwargs)

        # ── Core state ──
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars: dict = {}
        self._allow_one_summary_step = False
        self.patch_counter = 0
        self.log_file: Path | None = None
        self.base_repo_path: Path | None = None
        self._traj_last_saved_idx: int = -1

        # ── ToolRuntime setup (combines DefaultAgent.__init__ + StrategyAgent
        #    tool_profile). The StrategyAgent layer used to rebuild toolruntime
        #    AFTER DefaultAgent had already built one — wasteful. Here we build
        #    it ONCE with the right profile.
        self.toolruntime = ToolRuntime(
            use_strategy_manager=self.config.use_strategy_manager,
            strategy_file=(
                self._get_strategy_file()
                if self.config.use_strategy_manager
                else ".optimization_strategies.md"
            ),
            on_strategy_change=self._on_strategy_changed
            if self.config.use_strategy_manager
            else None,
            patch_output_dir=self.config.patch_output_dir,
            tool_profile=self.config.tool_profile,
        )
        if self.config.disabled_tools:
            self.toolruntime.disable_tools(self.config.disabled_tools)

        # RAG MCP postprocessor wrapping (from DefaultAgent)
        try:
            self.toolruntime.wrap_rag_tools_with_postprocessor()
        except Exception as e:
            logger.warning("Failed to wrap RAG tools with postprocessor: %s", e)

        # Propagate env vars + working dir from env.config (from DefaultAgent)
        agent_env = getattr(self.env.config, "env", None)
        if agent_env:
            self.toolruntime.set_env(agent_env)
        agent_cwd = getattr(self.env.config, "cwd", None)
        if agent_cwd:
            self.toolruntime.set_cwd(agent_cwd)

        self._setup_save_and_test_context()

        # Sub-agent tool context (from DefaultAgent)
        if getattr(self.toolruntime, "_sub_agent_tool", None):
            self.toolruntime._sub_agent_tool.set_context(
                self.model,
                self.env,
                codebase_context=self.config.codebase_context,
                inherited_config={
                    "test_command": self.config.test_command,
                    "patch_output_dir": self.config.patch_output_dir,
                    "metric": self.config.metric,
                    "save_patch": self.config.save_patch,
                    "use_strategy_manager": self.config.use_strategy_manager,
                    "strategy_file_path": self.config.strategy_file_path,
                    "profiling_type": self.config.profiling_type,
                },
                save_and_test_context=getattr(self, "_save_and_test_context", None),
            )

        if self.config.codebase_context:
            self.toolruntime.set_codebase_context(self.config.codebase_context)

        self.skillruntime = SkillRuntime()

        # ── Tell the model which tools it can call (from StrategyAgent) ──
        if hasattr(self.model, "set_tools"):
            self.model.set_tools(self.toolruntime.get_tools_schema())
        else:
            model_impl = getattr(self.model, "_impl", self.model)
            model_impl.tools = self.toolruntime.get_tools_schema()

        # ── Initial strategy data ping for UI (from StrategyAgent) ──
        if self.config.use_strategy_manager:
            self._send_initial_strategy_data()

        logger.debug(
            "OptimizationAgent initialized (profile=%s, strategy_file=%s)",
            self.config.tool_profile,
            self.config.strategy_file_path,
        )

    # ===============================================================
    # Strategy manager integration (from StrategyAgent + StrategyInteractiveAgent)
    # ===============================================================

    def _get_strategy_file(self) -> str:
        """Resolve the strategy_list.json / .optimization_strategies.md file path.

        Prefers `patch_output_dir` (unique per dispatched task) so parallel
        agents on different GPUs don't clobber each other's strategy files.
        Falls back to env.config.cwd for standalone runs.
        """
        if getattr(self.config, "patch_output_dir", None):
            base = Path(self.config.patch_output_dir)
        else:
            base = Path(getattr(self.env.config, "cwd", None) or Path.cwd())
        strategy_file_path = self.config.strategy_file_path or ".optimization_strategies.md"
        strategy_path = Path(strategy_file_path)
        return str(strategy_path if strategy_path.is_absolute() else base / strategy_path)

    def _send_initial_strategy_data(self) -> None:
        """On startup, if a strategy file already exists, replay it to the UI."""
        from minisweagent.tools.strategy_manager import StrategyManager

        strategy_file = self._get_strategy_file()
        manager = StrategyManager(filepath=strategy_file)
        if manager.exists():
            try:
                strategy_list = manager.load()
                self._on_strategy_changed(strategy_list)
            except Exception as e:
                print(f"[WARNING] Failed to load initial strategy data: {e}", file=sys.stderr)

    def _on_strategy_changed(self, strategy_list) -> None:
        """Callback when strategy list changes. Formats + dispatches to UI.

        Combines StrategyAgent._on_strategy_changed (formatting) +
        StrategyInteractiveAgent.notify_strategy_changed (console print).
        """
        try:
            strategy_file = self._get_strategy_file()
            result = {
                "exists": True,
                "strategies": [],
                "baseline": None,
                "notes": strategy_list.notes,
                "filePath": strategy_file,
            }
            if strategy_list.baseline:
                result["baseline"] = {
                    "metrics": strategy_list.baseline.metrics,
                    "logFile": strategy_list.baseline.log_file,
                }
            for idx, strategy in enumerate(strategy_list.strategies, start=1):
                result["strategies"].append({
                    "index": idx,
                    "name": strategy.name,
                    "status": strategy.status.value,
                    "description": strategy.description,
                    "priority": strategy.priority,
                    "expected": strategy.expected,
                    "target": strategy.target,
                    "result": strategy.result,
                    "details": strategy.details,
                })
            self._print_strategy_changed(result)
            print(
                f"[DEBUG] Strategy data updated: {len(result['strategies'])} strategies",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process strategy data: {e}", file=sys.stderr)

    def _print_strategy_changed(self, strategy_data: dict) -> None:
        """Rich-console printer for strategy updates.

        Inlined from StrategyInteractiveAgent.notify_strategy_changed.
        """
        strategies = strategy_data.get("strategies", [])
        file_path = strategy_data.get("filePath", "")

        console.print(f"\n[bold green]Strategy list updated:[/bold green] {len(strategies)} strategies")
        logger.info("Strategy list updated: %d strategies (file: %s)", len(strategies), file_path)
        console.print(f"[dim]File: {file_path}[/dim]")

        if strategies:
            console.print("\n[bold]Current Strategies:[/bold]")
            for s in strategies[:5]:
                status_color = {
                    "pending": "yellow",
                    "exploring": "blue",
                    "successful": "green",
                    "failed": "red",
                    "partial": "orange",
                    "skipped": "dim",
                }.get(s["status"], "white")
                console.print(
                    f"  [{status_color}]{s['index']}. {s['name']}[/{status_color}] - {s['status']}"
                )
            if len(strategies) > 5:
                console.print(f"  [dim]... and {len(strategies) - 5} more[/dim]")

        console.print()  # trailing blank line

    # ===============================================================
    # save_and_test context (from DefaultAgent — byte-identical)
    # ===============================================================

    def _setup_save_and_test_context(self) -> None:
        from minisweagent.tools.save_and_test import SaveAndTestContext

        cwd = getattr(self.env.config, "cwd", None) or os.getcwd()
        source_file_paths = getattr(self.config, "source_file_paths", None)
        if source_file_paths is None:
            source_file_paths = getattr(self.config, "source_file_path", None)

        context = SaveAndTestContext(
            cwd=cwd,
            test_command=self.config.test_command,
            timeout=getattr(self.env.config, "timeout", 3600),
            patch_output_dir=self.config.patch_output_dir,
            env_vars=getattr(self.env.config, "env", None),
            base_repo_path=self.base_repo_path,
            log_fn=self._log_message,
            patch_counter=self.patch_counter,
            source_file_paths=source_file_paths,
        )

        save_and_test_tool = self.toolruntime._tool_table.get("save_and_test")
        if save_and_test_tool:
            save_and_test_tool.set_context(context)
            self._save_and_test_context = context

    # ===============================================================
    # Template rendering, message history, logging (from DefaultAgent)
    # ===============================================================

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = (
            asdict(self.config)
            | self.env.get_template_vars()
            | self.model.get_template_vars()
        )
        all_vars = template_vars | self.extra_template_vars | kwargs
        return Template(template, undefined=StrictUndefined).render(**all_vars)

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Append to messages + optional file log + optional console print.

        Merges DefaultAgent.add_message (message storage + file log) with
        InteractiveAgent.add_message (rich console output). In production, the
        console output ONLY fires when log_file is None — parallel runs redirect
        to per-worker log files, so console duplication is avoided.
        """
        self.messages.append({"role": role, "content": content, **kwargs})
        if self.log_file:
            try:
                log_content = self._format_log_entry(role, content, **kwargs)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_content)
            except Exception as exc:
                logger.debug("add_message: log file write failed: %s", exc)
            return

        # Console output (only when no log_file — avoids double-logging in parallel runs)
        if role == "assistant":
            console.print(
                f"\n[red][bold]mini-swe-agent[/bold] (step [bold]{self.model.n_calls}[/bold], "
                f"[bold]${self.model.cost:.2f}[/bold]):[/red]\n",
                end="",
                highlight=False,
            )
        else:
            console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
        console.print(content, highlight=False, markup=False)

    def _format_log_entry(self, role: str, content: str, **kwargs) -> str:
        """Build a human-readable log entry. Byte-identical to DefaultAgent."""
        if role == "assistant":
            header = f"mini-swe-agent step {self.model.n_calls} (${self.model.cost:.2f})"
            log = f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}\n"
            if content:
                log += content.rstrip() + "\n"
            if kwargs.get("tool_calls"):
                tc = kwargs["tool_calls"]
                func = tc["function"]
                log += f"\n> Tool Call: {func['name']}\n"
                args = func.get("arguments", {})
                log += json.dumps(args, indent=2, ensure_ascii=False) + "\n"
            return log

        if role == "tool":
            name = kwargs.get("name", "unknown")
            log = f"\nUser (tool_result: {name}):\n"
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    for k, v in data.items():
                        val = str(v)
                        log += f"  {k}: {val}\n"
                    return log
            except (json.JSONDecodeError, TypeError):
                pass
            log += content + "\n"
            return log

        # system / user / other
        return f"\n{role.capitalize()}:\n{content}\n"

    def _log_message(self, message: str) -> None:
        """Free-form log line (separate from message history)."""
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
                    f.flush()
            except Exception as exc:
                logger.debug("_log_message: file write failed: %s", exc)
        else:
            print(message, flush=True)

    def _save_traj(self) -> None:
        """Incremental append-only write of new messages to traj.json (JSONL)."""
        if not self.messages:
            return

        if getattr(self.config, "patch_output_dir", None):
            base_dir = Path(self.config.patch_output_dir).resolve()
        elif self.log_file:
            base_dir = self.log_file.parent.resolve()
        else:
            base_dir = Path(getattr(self.env.config, "cwd", None) or os.getcwd()).resolve()

        traj_path = base_dir / "traj.json"
        base_dir.mkdir(parents=True, exist_ok=True)

        start_idx = self._traj_last_saved_idx + 1
        if start_idx >= len(self.messages):
            return

        try:
            with open(traj_path, "a", encoding="utf-8") as f:
                for i in range(start_idx, len(self.messages)):
                    f.write(json.dumps(self.messages[i], ensure_ascii=False, default=str) + "\n")
                f.flush()
            self._traj_last_saved_idx = len(self.messages) - 1
        except Exception as exc:
            logger.debug("_save_traj: best-effort traj write failed: %s", exc)

    # ===============================================================
    # Step loop (from DefaultAgent)
    # ===============================================================

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation.

        Note: dropped InteractiveAgent.step() wrapper that added `console.print(Rule())`
        and KeyboardInterrupt handling — not needed in production automated runs.
        """
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response.

        Note: dropped InteractiveAgent.query() logic for:
        - mode="human" user-command injection
        - console.status spinner
        - limit-renegotiation via input()
        None of these fire in production (audit §7).
        """
        if not self._allow_one_summary_step and (
            0 < self.config.step_limit <= self.model.n_calls
            or 0 < self.config.cost_limit <= self.model.cost
        ):
            raise LimitsExceeded()
        if self._allow_one_summary_step:
            self._allow_one_summary_step = False

        # Working memory injection (from DefaultAgent)
        _wm = getattr(self, "_working_memory", None)
        if _wm:
            _wm.update_step(self.model.n_calls, self.model.cost)
            _wm_text = _wm.format_for_injection()
            if _wm_text and not any("[Working Memory" in m.get("content", "") for m in self.messages[-3:]):
                self.messages.append({"role": "user", "content": f"[Working Memory Update]\n{_wm_text}"})

        response = self.model.query(self.messages)
        output = response["content"]
        msg_kwargs = {}
        if response.get("tools") and not self._will_use_bash(response):
            msg_kwargs["tool_calls"] = response["tools"]
        if response.get("extra"):
            msg_kwargs["extra"] = response["extra"]
        self.add_message("assistant", output, **msg_kwargs)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.parse_action(response)

        last_msg = self.messages[-1] if self.messages else {}
        if last_msg.get("role") == "assistant" and last_msg.get("tool_calls") and response.get("tools"):
            tool_info = response["tools"]
            result_content = json.dumps(output) if isinstance(output, dict) else str(output)
            result_content = truncate_observation(result_content)
            self.add_message(
                "tool",
                result_content,
                tool_call_id=tool_info.get("id", ""),
                name=tool_info["function"]["name"],
            )
        else:
            output_for_render = {
                **output,
                "output": truncate_observation(output.get("output", "")),
            }
            observation = self.render_template(self.config.action_observation_template, output=output_for_render)
            self.add_message("user", observation)
        return output

    @staticmethod
    def _will_use_bash(response: dict) -> bool:
        content = response.get("content", "")
        if not content:
            return False
        return len(re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)) == 1

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the model response and execute it.

        Bash triple-backtick blocks take precedence; tool calls run when
        present; skills load if enabled.
        """
        all_action = {"output": "", "returncode": 0}
        content = response.get("content", "")
        actions = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL) if content else []
        if len(actions) == 1:
            bash_action = self.execute_action({"action": actions[0].strip(), **response})
            all_action["output"] += bash_action["output"]
            all_action["returncode"] = max(all_action["returncode"], bash_action["returncode"])
        if response.get("tools"):
            from minisweagent.tools.submit import Submitted as ToolSubmitted

            try:
                result = self.toolruntime.dispatch(tool_call=response["tools"]["function"])
                self.has_finished(result)
            except ToolSubmitted as e:
                raise Submitted(str(e))
            tool_action = self._handle_tool_result(result)
            all_action["output"] += tool_action["output"]
            all_action["returncode"] = max(all_action["returncode"], tool_action["returncode"])
        if self.config.use_skills:
            skills_action = self.skillruntime.load_skill(response)
            all_action["output"] += skills_action["output"]
            all_action["returncode"] = max(all_action["returncode"], skills_action["returncode"])
        if all_action["output"] or all_action["returncode"] == 0:
            return all_action
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def _handle_tool_result(self, result: dict) -> dict:
        """Post-process tool results. Handles working memory ingestion."""
        if hasattr(self, "_save_and_test_context"):
            self.patch_counter = self._save_and_test_context.patch_counter

        _wm = getattr(self, "_working_memory", None)
        if _wm and result:
            try:
                from minisweagent.memory.working_memory import (  # pylint: disable=import-error,no-name-in-module
                    classify_change,
                    extract_insight_from_tool_result,
                    extract_strategy_from_edit,
                )

                output_str = result.get("output", "") if isinstance(result, dict) else str(result)
                rc = result.get("returncode", 0) if isinstance(result, dict) else 0
                insight = extract_insight_from_tool_result("", output_str, rc)
                if insight:
                    _wm.ingest_insight(insight)
                    _wm.note_tool_result(
                        output_str,
                        rc,
                        tag=insight.tag,
                        message=insight.message,
                        skip_metrics=True,
                    )
                else:
                    _wm.note_tool_result(output_str, rc)
                if "has been edited" in output_str:
                    last_assistant = ""
                    for m in reversed(self.messages):
                        if m.get("role") == "assistant":
                            last_assistant = m.get("content", "")
                            tool_calls = m.get("tool_calls")
                            if tool_calls:
                                try:
                                    calls = tool_calls if isinstance(tool_calls, list) else [tool_calls]
                                    payloads = []
                                    for call in calls:
                                        if not isinstance(call, dict):
                                            continue
                                        tool_args = call.get("function", {}).get("arguments", {})
                                        if isinstance(tool_args, str):
                                            try:
                                                tool_args = json.loads(tool_args)
                                            except Exception:
                                                pass
                                        if isinstance(tool_args, dict):
                                            edit_keys = (
                                                "old_str", "new_str",
                                                "old_string", "new_string",
                                                "old_text", "new_text",
                                            )
                                            edit_args = {k: tool_args[k] for k in edit_keys if k in tool_args}
                                            if edit_args:
                                                payloads.append(json.dumps(edit_args, ensure_ascii=False))
                                        elif isinstance(tool_args, str) and tool_args.strip():
                                            payloads.append(tool_args)
                                    if payloads:
                                        last_assistant = "\n".join(payloads)
                                except Exception as exc:
                                    logger.debug("_handle_tool_result: WM edit extraction failed: %s", exc)
                            break
                    strat = extract_strategy_from_edit(last_assistant)
                    if strat:
                        change_type = classify_change(last_assistant)
                        _wm.record_strategy(strat, True)
                        _wm.record_change_category(change_type)
                        _wm.remember_pending_change(strat, change_type)
            except Exception as exc:
                logger.debug("_handle_tool_result: working memory integration failed: %s", exc)

        return result

    def execute_action(self, action: dict) -> dict:
        """Execute a bash command via env.

        Note: dropped InteractiveAgent.execute_action() wrapper that called
        ask_confirmation() in mode="confirm". Production uses mode="yolo".
        """
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output="")
            )
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]) -> None:
        """Raise Submitted if the output contains the final-output sentinel.

        Note: dropped InteractiveAgent.has_finished() wrapper that prompted
        the user on confirm_exit. Production exits immediately on Submitted.
        """
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in [
            "MINI_SWE_AGENT_FINAL_OUTPUT",
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ]:
            raise Submitted("".join(lines[1:]))

    # ===============================================================
    # run() — the entry point
    # ===============================================================

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent terminates. Return (exit_status, message)."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.extra_template_vars["tool_names"] = set(self.toolruntime._tool_table.keys())
        self.messages = []
        self._traj_last_saved_idx = -1
        if self.config.use_skills:
            self.config.system_template += self.skillruntime.build_system_prompt()
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                e_type = type(e)
                e_msg = str(e)
                self.add_message("user", e_msg)
                if e_type is LimitsExceeded and getattr(self.config, "summary_on_cost_limit", False):
                    self.add_message("user", self.config.summary_on_limit_prompt)
                    self._allow_one_summary_step = True
                    try:
                        self.step()
                    except (TerminatingException, NonTerminatingException):
                        pass
                    finally:
                        self._allow_one_summary_step = False
                self._run_select_patch_agent()
                return e_type.__name__, e_msg
            finally:
                self._save_traj()

    def _run_select_patch_agent(self) -> None:
        """On termination, run the SelectPatchAgent over parallel patch outputs.

        First tries deterministic benchmark parsing (cheap). Falls back to an
        LLM-based SelectPatchAgent run only if deterministic parsing fails.
        """
        if not self.config.patch_output_dir:
            return

        base_patch_dir = Path(self.config.patch_output_dir).resolve()
        if not base_patch_dir.exists():
            return

        from minisweagent.run.postprocess.benchmark_parsing import rewrite_best_results

        det_result = rewrite_best_results(base_patch_dir)
        if det_result:
            return

        try:
            from minisweagent.agents.select_patch_agent import SelectPatchAgent
            from minisweagent.config import load_agent_config
            from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

            parallel_ids: list[int] = []
            for d in base_patch_dir.glob("parallel_*"):
                if d.is_dir():
                    m = re.match(r"parallel_(\d+)$", d.name)
                    if m:
                        parallel_ids.append(int(m.group(1)))
            num_parallel = (max(parallel_ids) + 1) if parallel_ids else 1

            agent_config, _ = load_agent_config("mini_select_patch")
            env_config = LocalEnvironmentConfig(cwd=str(base_patch_dir))
            env = LocalEnvironment(**env_config.__dict__)
            select_agent = SelectPatchAgent(self.model, env, **agent_config)
            select_agent.log_file = base_patch_dir / "select_agent.log"

            task = select_agent.setup_selection_task(base_patch_dir, num_parallel, self.config.metric)
            if task:
                select_agent.run(task, _skip_select_patch=True)
            rewrite_best_results(base_patch_dir)
        except Exception as exc:
            logger.debug("_run_select_patch_agent: failed: %s", exc)


__all__ = [
    "OptimizationAgent",
    "AgentConfig",
    # Exceptions (re-exported for backward-compat with code that imported from default.py)
    "NonTerminatingException",
    "FormatError",
    "ExecutionTimeoutError",
    "TerminatingException",
    "Submitted",
    "LimitsExceeded",
    # Helpers
    "truncate_observation",
    "console",
]
