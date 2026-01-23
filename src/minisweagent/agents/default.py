"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template
from pathlib import Path
from minisweagent import Environment, Model


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
        self.patch_results: dict[str, dict] = {}
        self.patch_counter = 0
        self.log_file: Path | None = None
        self.base_repo_path: Path | None = None

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        all_vars = template_vars | self.extra_template_vars | kwargs
        return Template(template, undefined=StrictUndefined).render(**all_vars)

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

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
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response["content"], re.DOTALL)
        if len(actions) == 1:
            return {"action": actions[0].strip(), **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

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
        
        # Check if output contains TEST_BASELINE_PERFORMANCE or SAVE_PATCH_AND_TEST
        if self.config.save_patch:
            lines = output.get("output", "").lstrip().splitlines(keepends=True)
            if lines and lines[0].strip() in ("TEST_BASELINE_PERFORMANCE", "SAVE_PATCH_AND_TEST"):
                patch_info = self._save_patch_and_test()
                if patch_info:
                    output["output"] = output.get("output", "") + "\n" + patch_info
        
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))
    
    # ============ Save Patch Functionality ============
    
    @staticmethod
    def _is_git_repo(path: Path) -> bool:
        """Return True if `path` is inside a git repository."""
        try:
            subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _diff_excludes(self) -> list[str]:
        """Best-effort excludes to prevent artifacts from polluting patches."""
        excludes = [".git", "__pycache__", "*.pyc"]
        if self.config.patch_output_dir:
            run_dir_name = Path(self.config.patch_output_dir).resolve().parent.name
            if run_dir_name:
                excludes.append(run_dir_name)
        return excludes
    
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
    
    def _save_patch_and_test(self) -> str | None:
        """Save current changes as patch and run test command."""
        patch_name = f"patch_{self.patch_counter}"
        self.patch_counter += 1
        self._log_message(f"\n[Agent] Saving patch and running test...")
        
        cwd = getattr(self.env, 'working_dir', None)
        if cwd is None:
            cwd = getattr(self.env.config, 'cwd', None) or os.getcwd()
        
        try:
            patch_content = ""
            if self._is_git_repo(Path(cwd)):
                git_diff_result = subprocess.run(
                    "git add -N . && git diff",
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=True,
                )
                patch_content = git_diff_result.stdout
            elif self.base_repo_path and self.base_repo_path.exists():
                excludes = self._diff_excludes()
                diff_result = subprocess.run(
                    [
                        "diff",
                        "-ruN",
                        "--exclude=.git",
                        "--exclude=__pycache__",
                        *[f"--exclude={p}" for p in excludes if p not in (".git", "__pycache__")],
                        str(self.base_repo_path),
                        str(cwd),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                patch_content = diff_result.stdout
            
            if not patch_content.strip():
                self._log_message(f"[Agent] No changes detected, baseline running.")
            else:
                self._log_message(f"[Agent] Patch {patch_name} captured, running test...")
            
            if not self.config.test_command:
                error_msg = "[Agent] ERROR: test_command is not configured. Cannot run test."
                self._log_message(error_msg)
                test_output = error_msg
                test_passed = False
                test_returncode = -1
            else:
                test_env = os.environ.copy()
                if hasattr(self.env.config, 'env'):
                    test_env.update(self.env.config.env)
                test_env["PYTHONUNBUFFERED"] = "1"
                test_command = self.config.test_command.replace("WORK_REPO", str(cwd))
                self._log_message(f"[Agent] Running test command: {test_command}")
                
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_file:
                    tmp_output_file = tmp_file.name
                
                try:
                    wrapped_command = f"({test_command}) > {tmp_output_file} 2>&1; echo $? > {tmp_output_file}.exitcode"
                    test_result = subprocess.run(
                        wrapped_command,
                        shell=True,
                        cwd=cwd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=self.env.config.timeout,
                        env=test_env,
                    )
                    
                    if Path(tmp_output_file).exists():
                        test_output = Path(tmp_output_file).read_text()
                    else:
                        test_output = test_result.stdout or ""
                    
                    exitcode_file = Path(f"{tmp_output_file}.exitcode")
                    if exitcode_file.exists():
                        try:
                            test_returncode = int(exitcode_file.read_text().strip())
                        except (ValueError, OSError):
                            test_returncode = test_result.returncode
                    else:
                        test_returncode = test_result.returncode
                    
                    test_passed = test_returncode == 0
                finally:
                    for tmp_file_path in [tmp_output_file, f"{tmp_output_file}.exitcode"]:
                        try:
                            if Path(tmp_file_path).exists():
                                Path(tmp_file_path).unlink()
                        except Exception:
                            pass
            
            self.patch_results[patch_name] = {
                "patch_file": f"{patch_name}.patch",
                "test_output_file": f"{patch_name}_test.txt",
                "test_passed": test_passed,
                "returncode": test_returncode,
            }
            status = "✓ PASSED" if test_passed else "✗ FAILED"
            self._log_message(f"[Agent] Test result for {patch_name}: {status}")
            
            metric_result = self._extract_metric(patch_name, test_output)
            self.patch_results[patch_name]["metric_result"] = metric_result
            
            if self.config.patch_output_dir:
                self._save_patch_file(patch_name, patch_content)
                self._save_test_output(patch_name, test_output)
                self._update_results_file()
            
            return self._format_patch_info(patch_name, patch_content, test_output, test_passed, test_returncode)
                
        except subprocess.TimeoutExpired:
            test_output = "Test command timed out"
            self.patch_results[patch_name] = {
                "patch_file": f"{patch_name}.patch",
                "test_output_file": f"{patch_name}_test.txt",
                "test_passed": False,
                "returncode": -1,
            }
            self._log_message(f"[Agent] Test for {patch_name}: ✗ TIMEOUT")
            self.patch_results[patch_name]["metric_result"] = None
            if self.config.patch_output_dir:
                self._save_patch_file(patch_name, patch_content)
                self._save_test_output(patch_name, test_output)
                self._update_results_file()
            return self._format_patch_info(patch_name, patch_content, test_output, False, -1)
        except Exception as e:
            test_output = str(e)
            self.patch_results[patch_name] = {
                "patch_file": f"{patch_name}.patch",
                "test_output_file": f"{patch_name}_test.txt",
                "test_passed": False,
                "returncode": -1,
            }
            self._log_message(f"[Agent] Test for {patch_name}: ✗ ERROR - {e}")
            self.patch_results[patch_name]["metric_result"] = None
            if self.config.patch_output_dir:
                self._save_patch_file(patch_name, patch_content)
                self._save_test_output(patch_name, test_output)
                self._update_results_file()
            return self._format_patch_info(patch_name, patch_content, test_output, False, -1)
    
    def _format_patch_info(self, patch_name: str, patch_content: str, test_output: str, test_passed: bool, returncode: int) -> str:
        """Format patch information for display."""
        status = "PASSED ✓" if test_passed else "FAILED ✗"
        info_parts = [
            f"\n{'='*60}",
            f"Patch saved: {patch_name}",
            f"Test status: {status}",
            f"Return code: {returncode}",
            f"{'='*60}",
            "\n## Test Output:",
            f"```\n{test_output}\n```",
            f"{'='*60}\n",
        ]
        return "\n".join(info_parts)
    
    def _extract_metric(self, patch_name: str, test_output: str):
        """Extract performance metrics from test output using LLM."""
        self._log_message(f"[Agent] Extracting metric for {patch_name}...")
        
        metric_prompt = self.config.metric or "Extract the performance metrics from the test output."
        prompt = (
            f"Task: {metric_prompt}\n\n"
            f"Test output:\n{test_output}\n\n"
            "Extract the requested metric from the test output above. "
            "Return ONLY a valid JSON object with the extracted data. "
            "If the metric cannot be found, return null. "
            "Examples:\n"
            '- For numeric value: {"value": 123.45}\n'
            '- For array: {"values": [1.2, 3.4, 5.6]}\n'
            '- For multiple metrics: {"bandwidth": 123.45, "latency": 0.5}\n'
        )
        response = self.model.query([
            {"role": "system", "content": "You are a helpful assistant to analyze kernel test output and extract metrics from test logs."},
            {"role": "user", "content": prompt}
        ])
        content = response.get("content", "")
        try:
            result = json.loads(content.strip())
            self._log_message(f"[Agent] Metric extracted: {result}")
            return result
        except json.JSONDecodeError:
            self._log_message(f"[Agent] Failed to parse metric response: {content}")
            return None
    
    def _save_patch_file(self, patch_name: str, patch_content: str):
        """Save patch content to file."""
        output_dir = Path(self.config.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_file = output_dir / f"{patch_name}.patch"
        patch_file.write_text(patch_content)
        self._log_message(f"[Agent] Patch saved to: {patch_file}")
    
    def _save_test_output(self, patch_name: str, test_output: str):
        """Save test output to file."""
        output_dir = Path(self.config.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / f"{patch_name}_test.txt"
        test_file.write_text(test_output)
        self._log_message(f"[Agent] Test output saved to: {test_file}")
    
    def _update_results_file(self):
        """Update results.json with all patch results."""
        output_dir = Path(self.config.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "results.json"
        simplified_results = {}
        for name, data in self.patch_results.items():
            result_entry = {
                "patch_file": data["patch_file"],
                "test_output_file": data["test_output_file"],
                "test_passed": data["test_passed"],
                "returncode": data["returncode"],
            }
            if "metric_result" in data:
                result_entry["metric_result"] = data["metric_result"]
            simplified_results[name] = result_entry
        results_file.write_text(json.dumps(simplified_results, indent=2, ensure_ascii=False))
        self._log_message(f"[Agent] Results updated: {results_file}")