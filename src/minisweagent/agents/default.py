"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.tools.tools_runtime import ToolRuntime


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
    summary_on_cost_limit: bool = False
    """When True, on LimitsExceeded allow one extra step (e.g. to write a summary)."""
    summary_on_limit_prompt: str = (
        "The cost limit has been reached. Before stopping, run exactly one command to document "
        "what you did so far (e.g. create a summary file or add to your final output)."
    )
    # Save patch configuration (always enabled)
    save_patch: bool = True
    test_command: str | None = None
    patch_output_dir: str | None = None
    metric: str | None = None
    # Strategy manager configuration
    use_strategy_manager: bool = False
    strategy_file_path: str = ".optimization_strategies.md"
    profiling_type: str | None = None
    codebase_context: str | None = None
    starting_patch: str | None = None
    # Interactive/exit behaviour (set by --exit-immediately)
    confirm_exit: bool = True


# Unified observation truncation for both bash output and tool call results (head + tail).
OBSERVATION_MAX_LEN: int = 10000
OBSERVATION_HEAD_LEN: int = 5000
OBSERVATION_TAIL_LEN: int = 5000
OBSERVATION_TRUNCATED_NOTICE: str = (
    "\n<warning>\n"
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n"
    "</warning>\n"
)


def truncate_observation(text: str) -> str:
    """Truncate long observation to head + notice + elided + tail. Same logic for bash and tool results."""
    if not text or len(text) <= OBSERVATION_MAX_LEN:
        return text
    elided = len(text) - OBSERVATION_HEAD_LEN - OBSERVATION_TAIL_LEN
    return (
        text[:OBSERVATION_HEAD_LEN]
        + OBSERVATION_TRUNCATED_NOTICE
        + f"<elided>{elided} characters elided</elided>\n"
        + text[-OBSERVATION_TAIL_LEN:]
    )


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


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self._allow_one_summary_step = False
        # Initialize save_patch related attributes
        self.patch_counter = 0
        self.log_file: Path | None = None
        self.base_repo_path: Path | None = None
        # Incremental trajectory writing (traj.json)
        self._traj_last_saved_idx: int = -1
        # Initialize tool runtime with strategy manager settings
        # Subclasses (like StrategyAgent) can override _get_strategy_callback() for UI notifications
        self.toolruntime = ToolRuntime(
            use_strategy_manager=self.config.use_strategy_manager,
            strategy_file=self._get_strategy_file()
            if self.config.use_strategy_manager
            else ".optimization_strategies.md",
            on_strategy_change=self._get_strategy_callback(),
            patch_output_dir=self.config.patch_output_dir,
        )
        # Propagate agent's env vars (HIP_VISIBLE_DEVICES etc.) to tools
        agent_env = getattr(self.env.config, "env", None)
        if agent_env:
            self.toolruntime.set_env(agent_env)
        # Propagate working directory so bash tool commands run in the correct worktree
        agent_cwd = getattr(self.env.config, "cwd", None)
        if agent_cwd:
            self.toolruntime.set_cwd(agent_cwd)
        # Setup save_and_test tool context
        self._setup_save_and_test_context()
        # Wire sub_agent context (needs model + env for recursive agent calls)
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

    def _get_strategy_file(self) -> str:
        """Get the strategy file path.

        Prefers ``patch_output_dir`` (unique per dispatched task) so that
        parallel agents on different GPUs don't clobber each other's
        strategy files.  Falls back to ``cwd`` for standalone ``mini`` runs.
        """
        if getattr(self.config, "patch_output_dir", None):
            base = Path(self.config.patch_output_dir)
        else:
            base = Path(getattr(self.env.config, "cwd", None) or Path.cwd())
        strategy_file_path = self.config.strategy_file_path or ".optimization_strategies.md"
        strategy_path = Path(strategy_file_path)
        return str(strategy_path if strategy_path.is_absolute() else base / strategy_path)

    def _get_strategy_callback(self):
        """Get the callback for strategy changes. Override in subclasses for UI notifications."""
        return

    def _setup_save_and_test_context(self):
        """Setup context for save_and_test tool."""
        from minisweagent.tools.save_and_test import SaveAndTestContext

        cwd = getattr(self.env.config, "cwd", None) or os.getcwd()

        # source_file_paths: files the agent is allowed to modify.
        # Loaded from task metadata (if available) to prevent agents from
        # gaming benchmarks by modifying evaluation infrastructure.
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

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        all_vars = template_vars | self.extra_template_vars | kwargs
        return Template(template, undefined=StrictUndefined).render(**all_vars)

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})
        if self.log_file:
            try:
                log_content = self._format_log_entry(role, content, **kwargs)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_content)
            except Exception:
                pass

    def _format_log_entry(self, role: str, content: str, **kwargs) -> str:
        """Build a human-readable log entry."""
        if role == "assistant":
            header = f"mini-swe-agent step {self.model.n_calls} (${self.model.cost:.2f})"
            log = f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}\n"
            # Text content first
            if content:
                log += content.rstrip() + "\n"
            # Then tool call details (raw JSON)
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
            # Pretty-print JSON output when possible
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
        log = f"\n{role.capitalize()}:\n"
        log += content + "\n"
        return log

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self._traj_last_saved_idx = -1
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

    def _save_traj(self):
        """Incrementally append new messages to `traj.json` (JSONL style).

        Notes:
        - Writes only newly-added messages since the last call.
        - In parallel runs, each agent writes under its own `parallel_{idx}/` dir
          because `patch_output_dir` is set per agent.
        """
        if not self.messages:
            return

        # Prefer patch_output_dir (parallel-aware). Fall back to log_file dir, then cwd.
        base_dir: Path
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
        except Exception:
            # Best-effort: never block the agent on telemetry logging.
            return

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if not self._allow_one_summary_step and (
            0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost
        ):
            raise LimitsExceeded()
        if self._allow_one_summary_step:
            self._allow_one_summary_step = False

        _wm = getattr(self, "_working_memory", None)
        if _wm:
            _wm.update_step(self.model.n_calls, self.model.cost)
            _wm_text = _wm.format_for_injection()
            if _wm_text and not any("[Working Memory" in m.get("content", "") for m in self.messages[-3:]):
                self.messages.append({"role": "user", "content": f"[Working Memory Update]\n{_wm_text}"})

        response = self.model.query(self.messages)
        output = response["content"]
        # Include tool_calls in assistant message when the model requests a tool call
        # and there is no bash block (bash takes priority in parse_action).
        msg_kwargs = {}
        if response.get("tools") and not self._will_use_bash(response):
            msg_kwargs["tool_calls"] = response["tools"]
        # Attach per-request usage / response metadata for trajectory logging.
        if response.get("extra"):
            msg_kwargs["extra"] = response["extra"]
        self.add_message("assistant", output, **msg_kwargs)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.parse_action(response)

        # If the last assistant message has tool_calls, the tool was dispatched
        # → add a structured tool result message instead of a plain observation.
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
            # Bash: truncate output body in Python so template only renders returncode + (possibly truncated) output.
            output_for_render = {
                **output,
                "output": truncate_observation(output.get("output", "")),
            }
            observation = self.render_template(self.config.action_observation_template, output=output_for_render)
            self.add_message("user", observation)
        return output

    @staticmethod
    def _will_use_bash(response: dict) -> bool:
        """Return True when parse_action would execute a bash block for this response."""
        content = response.get("content", "")
        if not content:
            return False
        return len(re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)) == 1

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        content = response.get("content", "")
        actions = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL) if content else []
        if len(actions) == 1:
            return self.execute_action({"action": actions[0].strip(), **response})
        if response.get("tools"):
            from minisweagent.tools.submit import Submitted as ToolSubmitted

            try:
                result = self.toolruntime.dispatch(tool_call=response["tools"]["function"])
                self.has_finished(result)
            except ToolSubmitted as e:
                raise Submitted(str(e))
            # Handle tool results (sync state, etc.)
            return self._handle_tool_result(result)
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def _handle_tool_result(self, result: dict) -> dict:
        """Handle tool results. Submit tool raises Submitted, save_and_test handles itself."""
        if hasattr(self, "_save_and_test_context"):
            self.patch_counter = self._save_and_test_context.patch_counter

        _wm = getattr(self, "_working_memory", None)
        if _wm and result:
            try:
                from minisweagent.memory.working_memory import extract_insight_from_tool_result, extract_strategy_from_edit
                output_str = result.get("output", "") if isinstance(result, dict) else str(result)
                rc = result.get("returncode", 0) if isinstance(result, dict) else 0
                insight = extract_insight_from_tool_result("", output_str, rc)
                if insight:
                    insight.step = _wm.current_step
                    _wm.add_insight(insight.tag, insight.message)
                    import re as _re
                    _lat = _re.search(r'latency:\s*(\d+\.\d+)ms', insight.message, _re.IGNORECASE)
                    if _lat:
                        _wm.update_latency(float(_lat.group(1)))
                    else:
                        _sp = _re.search(r'(\d+\.\d+)x', insight.message)
                        if _sp:
                            _wm.update_speedup(float(_sp.group(1)))
                    _wm.note_tool_result(
                        output_str,
                        rc,
                        tag=insight.tag,
                        message=insight.message,
                    )
                else:
                    _wm.note_tool_result(output_str, rc)
                if "has been edited" in output_str:
                    from minisweagent.memory.working_memory import classify_change
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
                                                "old_str",
                                                "new_str",
                                                "old_string",
                                                "new_string",
                                                "old_text",
                                                "new_text",
                                            )
                                            edit_args = {
                                                k: tool_args[k]
                                                for k in edit_keys
                                                if k in tool_args
                                            }
                                            if edit_args:
                                                payloads.append(json.dumps(edit_args, ensure_ascii=False))
                                        elif isinstance(tool_args, str) and tool_args.strip():
                                            payloads.append(tool_args)
                                    if payloads:
                                        # Prefer real edit payloads over assistant prose so WM labels
                                        # are driven by the diff, not by task/context text.
                                        last_assistant = "\n".join(payloads)
                                except Exception:
                                    pass
                            break
                    strat = extract_strategy_from_edit(last_assistant)
                    if strat:
                        change_type = classify_change(last_assistant)
                        _wm.record_strategy(strat, True)
                        _wm.record_change_category(change_type)
                        _wm.remember_pending_change(strat, change_type)
            except Exception:
                pass

        return result

    def _run_select_patch_agent(self) -> None:
        if not self.config.patch_output_dir:
            return

        base_patch_dir = Path(self.config.patch_output_dir).resolve()
        if not base_patch_dir.exists():
            return

        # Try deterministic benchmark parsing first -- avoids LLM cost
        from minisweagent.benchmark_parsing import rewrite_best_results
        det_result = rewrite_best_results(base_patch_dir)
        if det_result:
            return

        # Fall back to LLM-based selection only if deterministic parsing failed
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

            # Final deterministic override as safety net
            rewrite_best_results(base_patch_dir)
        except Exception:
            return

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)

        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        # Legacy: Check for bash echo commands
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))

    # ============ Logging ============

    def _log_message(self, message: str):
        """Log a message to log file or console."""
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
                    f.flush()
            except Exception:
                pass
        else:
            print(message, flush=True)
