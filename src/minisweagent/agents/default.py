"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from jinja2 import StrictUndefined, Template
from pathlib import Path
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
    # Save patch configuration (always enabled)
    save_patch: bool = True
    test_command: str | None = None
    patch_output_dir: str | None = None
    metric: str | None = None
    # Strategy manager configuration
    use_strategy_manager: bool = False
    strategy_file_path: str = ".optimization_strategies.md"
    profiling_type: str | None = None


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
        # Initialize save_patch related attributes
        self.patch_counter = 0
        self.log_file: Path | None = None
        self.base_repo_path: Path | None = None
        # Initialize tool runtime with strategy manager settings
        # Subclasses (like StrategyAgent) can override _get_strategy_callback() for UI notifications
        self.toolruntime = ToolRuntime(
            profiling_type=self.config.profiling_type,
            llm_model=self.model,
            use_strategy_manager=self.config.use_strategy_manager,
            strategy_file=self._get_strategy_file()
            if self.config.use_strategy_manager
            else ".optimization_strategies.md",
            on_strategy_change=self._get_strategy_callback(),
            patch_output_dir=self.config.patch_output_dir,
        )
        # Setup test_perf tool context
        self._setup_test_perf_context()
    
    def _get_strategy_file(self) -> str:
        """Get the strategy file path. Override in subclasses to customize."""
        cwd = Path(getattr(self.env.config, "cwd", None) or Path.cwd())
        strategy_file_path = self.config.strategy_file_path or ".optimization_strategies.md"
        strategy_path = Path(strategy_file_path)
        return str(strategy_path if strategy_path.is_absolute() else cwd / strategy_path)
    
    def _get_strategy_callback(self):
        """Get the callback for strategy changes. Override in subclasses for UI notifications."""
        return None
    
    def _setup_test_perf_context(self):
        """Setup context for test_perf tool."""
        from minisweagent.tools.test_perf import TestPerfContext
        
        cwd = getattr(self.env.config, 'cwd', None) or os.getcwd()
        
        context = TestPerfContext(
            cwd=cwd,
            test_command=self.config.test_command,
            timeout=getattr(self.env.config, 'timeout', 3600),
            patch_output_dir=self.config.patch_output_dir,
            env_vars=getattr(self.env.config, 'env', None),
            base_repo_path=self.base_repo_path,
            log_fn=self._log_message,
            patch_counter=self.patch_counter,
        )
        
        test_perf_tool = self.toolruntime._tool_table.get("test_perf")
        if test_perf_tool:
            test_perf_tool.set_context(context)
            # Keep reference to sync state
            self._test_perf_context = context

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        all_vars = template_vars | self.extra_template_vars | kwargs
        return Template(template, undefined=StrictUndefined).render(**all_vars)

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})
        if self.log_file:
            try:
                if role == "assistant":
                    log_content = f"\nmini-swe-agent (step {self.model.n_calls}, ${self.model.cost:.2f}):\n"
                else:
                    log_content = f"\n{role.capitalize()}:\n"
                log_content += content + "\n"
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_content)
            except Exception:
                pass

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                self._run_select_patch_agent()
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        output = "<action>\n"+ response["content"] + f"\ntool call:\n   {json.dumps(response["tools"], indent=4)}" + "\n</action>"
        self.add_message("assistant", output)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.parse_action(response)
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        if response["content"]:
            actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
            if len(actions) == 1:
                actions = {"action": actions[0].strip(), **response}
                return self.execute_action(actions)
        if response["tools"]:
            from minisweagent.tools.submit import Submitted as ToolSubmitted
            try:
                result = self.toolruntime.dispatch(tool_call=response["tools"]["function"])
            except ToolSubmitted as e:
                raise Submitted(str(e))
            # Handle tool results (sync state, etc.)
            result = self._handle_tool_result(result)
            return result
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
    
    def _handle_tool_result(self, result: dict) -> dict:
        """Handle tool results. Submit tool raises Submitted, test_perf handles itself."""
        # Sync test_perf context state back to agent
        if hasattr(self, '_test_perf_context'):
            self.patch_counter = self._test_perf_context.patch_counter
        return result

    def _run_select_patch_agent(self) -> None:
        # Always try to run select patch agent if patch_output_dir is configured
        if not self.config.patch_output_dir:
            return

        base_patch_dir = Path(self.config.patch_output_dir).resolve()
        if not base_patch_dir.exists():
            return

        try:
            import yaml
            from minisweagent.config import get_config_path
            from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
            from minisweagent.agents.select_patch_agent import SelectPatchAgent

            parallel_ids: list[int] = []
            for d in base_patch_dir.glob("parallel_*"):
                if d.is_dir():
                    m = re.match(r"parallel_(\d+)$", d.name)
                    if m:
                        parallel_ids.append(int(m.group(1)))
            num_parallel = (max(parallel_ids) + 1) if parallel_ids else 1

            config_path = get_config_path("mini_select_patch")
            config = yaml.safe_load(config_path.read_text())
            agent_config = config.get("agent", {})

            env_config = LocalEnvironmentConfig(cwd=str(base_patch_dir))
            env = LocalEnvironment(**env_config.__dict__)
            select_agent = SelectPatchAgent(self.model, env, **agent_config)
            select_agent.log_file = base_patch_dir / "select_agent.log"

            task = select_agent.setup_selection_task(base_patch_dir, num_parallel, self.config.metric)
            if task:
                select_agent.run(task, _skip_select_patch=True)
        except Exception:
            # Best-effort: selection should not block returning the agent's final output.
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