"""Agent with git patch saving and test execution capability."""

import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.models import get_model
from minisweagent.agents.select_patch_agent import SelectPatchAgent


@dataclass
class BestPatchResult:
    """Result of selecting the best patch from parallel runs."""
    agent_id: int
    patch_id: str
    test_output: str
    metric_result: dict | None = None
    patch_dir: Path | None = None
    llm_conclusion: str | None = None


class Tee:
    """A class that writes to multiple file-like objects simultaneously."""
    def __init__(self, *files):
        self.files = files
        self.lock = threading.Lock()
        self.closed = False
    
    def write(self, text):
        if self.closed:
            return
        with self.lock:
            for f in self.files:
                try:
                    f.write(text)
                    f.flush()
                except (ValueError, OSError):
                    pass
    
    def flush(self):
        if self.closed:
            return
        with self.lock:
            for f in self.files:
                try:
                    f.flush()
                except (ValueError, OSError):
                    pass
    
    def close(self):
        """Mark this Tee as closed to prevent further writes."""
        self.closed = True


# Thread-local storage for stdout/stderr redirection
_thread_local = threading.local()
_stdout_lock = threading.Lock()


@contextmanager
def redirect_output_to_file(log_file: Path):
    """Context manager to redirect stdout and stderr to a file while keeping console output.
    Thread-safe version that uses thread-local storage and locking."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file_handle = open(log_file, "a", encoding="utf-8")
    
    try:
        _thread_local.original_stdout = sys.stdout
        _thread_local.original_stderr = sys.stderr
        
        tee_stdout = Tee(_thread_local.original_stdout, log_file_handle)
        tee_stderr = Tee(_thread_local.original_stderr, log_file_handle)
        
        with _stdout_lock:
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
        
        try:
            yield
        finally:
            with _stdout_lock:
                if isinstance(sys.stdout, Tee):
                    sys.stdout.close()
                if isinstance(sys.stderr, Tee):
                    sys.stderr.close()
                if hasattr(_thread_local, 'original_stdout'):
                    sys.stdout = _thread_local.original_stdout
                if hasattr(_thread_local, 'original_stderr'):
                    sys.stderr = _thread_local.original_stderr
    finally:
        log_file_handle.close()


@dataclass
class ParallelAgentConfig(AgentConfig):
    save_patch: bool = True
    test_command: str | None = None
    patch_output_dir: str | None = None
    metric: str | None = None
    mode: str | None = None
    num_parallel: int = 1
    repo: Path | None = None
    parallel_gpu_ids: list[int] | None = None


class ParallelAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=ParallelAgentConfig, **kwargs)
        self.patch_results: dict[str, dict] = {}
        self.patch_counter = 0
        self.log_file: Path | None = None
        self._last_action_hash: str | None = None

    def add_message(self, role: str, content: str, **kwargs):
        super().add_message(role, content, **kwargs)
        
        # Write to log file if available (thread-safe by appending)
        if self.log_file:
            try:
                log_content = ""
                if role == "assistant":
                    log_content = f"\nmini-swe-agent (step {self.model.n_calls}, ${self.model.cost:.2f}):\n"
                else:
                    log_content = f"\n{role.capitalize()}:\n"
                log_content += content + "\n"
                
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_content)
                    f.flush()
            except Exception:
                pass  # Ignore errors writing to log file
    
    def _log_message(self, message: str):
        """Log a message directly to the log file without going through stdout."""
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
                    f.flush()
            except Exception:
                pass
        # Also print to console if stdout is not redirected (not a Tee instance)
        if not isinstance(sys.stdout, Tee):
            print(message, flush=True)

    def execute_action(self, action: dict) -> dict:
        output = super().execute_action(action)
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ("TEST_BASELINE_PERFORMANCE", "SAVE_PATCH_AND_TEST"):
            patch_info = self._save_patch_and_test()
            if patch_info:
                output["output"] = output.get("output", "") + "\n" + patch_info
            
        return output

    def _save_patch_and_test(self) -> str | None:
        patch_name = f"patch_{self.patch_counter}"
        self.patch_counter += 1
        self._log_message(f"\n[ParallelAgent] Saving patch and running test...")
        
        cwd = getattr(self.env, 'working_dir', None)
        if cwd is None:
            cwd = getattr(self.env.config, 'cwd', None) or os.getcwd()
        
        try:
            git_diff_result = subprocess.run(
                ["git", "diff"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            patch_content = git_diff_result.stdout
            if not patch_content.strip():
                self._log_message(f"[ParallelAgent] No changes detected, baseline running.")
            else:
                self._log_message(f"[ParallelAgent] Patch {patch_name} captured, running test...")
            
            if not self.config.test_command:
                error_msg = "[ParallelAgent] ERROR: test_command is not configured. Cannot run test."
                self._log_message(error_msg)
                test_output = error_msg
                test_passed = False
                test_returncode = -1
            else:
                test_env = os.environ.copy()
                # Include environment variables from env config (e.g., HIP_VISIBLE_DEVICES for GPU isolation)
                if hasattr(self.env.config, 'env'):
                    test_env.update(self.env.config.env)
                    if 'HIP_VISIBLE_DEVICES' in self.env.config.env:
                        self._log_message(f"[ParallelAgent] Using GPU isolation: HIP_VISIBLE_DEVICES={self.env.config.env['HIP_VISIBLE_DEVICES']}")
                test_env["PYTHONUNBUFFERED"] = "1"
                # Replace WORK_REPO placeholder with actual working directory
                test_command = self.config.test_command.replace("WORK_REPO", str(cwd))
                self._log_message(f"[ParallelAgent] Running test command: {test_command}")
                
                # Create temporary file to capture ALL output
                # Direct redirection is more reliable than tee when scripts use subprocess.PIPE internally
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_file:
                    tmp_output_file = tmp_file.name
                
                try:
                    # Direct redirection to file: this captures ALL output that goes to stdout/stderr
                    # even if the script uses subprocess.PIPE internally, as long as it eventually prints
                    # This is more reliable than tee because it doesn't depend on the script's internal handling
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
                    
                    # Read output from file (captures everything that was printed)
                    if Path(tmp_output_file).exists():
                        test_output = Path(tmp_output_file).read_text()
                    else:
                        # Fallback to stdout if file doesn't exist
                        test_output = test_result.stdout or ""
                    
                    # Read exit code from file if available, otherwise use subprocess returncode
                    exitcode_file = Path(f"{tmp_output_file}.exitcode")
                    if exitcode_file.exists():
                        try:
                            test_returncode = int(exitcode_file.read_text().strip())
                        except (ValueError, OSError):
                            test_returncode = test_result.returncode
                    else:
                        test_returncode = test_result.returncode
                    
                    test_passed = test_returncode == 0
                    
                    if not test_output.strip():
                        self._log_message(
                            f"[ParallelAgent] Warning: Test command produced no output (returncode: {test_returncode}). "
                            f"This may indicate the script captured output with subprocess.PIPE but didn't print it. "
                            f"Output file: {tmp_output_file}"
                        )
                finally:
                    # Clean up temporary files
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
            self._log_message(f"[ParallelAgent] Test result for {patch_name}: {status}")
            
            if self.config.metric:
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
            self._log_message(f"[ParallelAgent] Test for {patch_name}: ✗ TIMEOUT")
            if self.config.metric:
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
            self._log_message(f"[ParallelAgent] Test for {patch_name}: ✗ ERROR - {e}")
            if self.config.metric:
                self.patch_results[patch_name]["metric_result"] = None
            if self.config.patch_output_dir:
                self._save_patch_file(patch_name, patch_content)
                self._save_test_output(patch_name, test_output)
                self._update_results_file()
            return self._format_patch_info(patch_name, patch_content, test_output, False, -1)

    def _format_patch_info(self, patch_name: str, patch_content: str, test_output: str, test_passed: bool, returncode: int) -> str:
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
        self._log_message(f"[ParallelAgent] Extracting metric for {patch_name}...")
        prompt = (
            f"Task: {self.config.metric}\n\n"
            f"Test output:\n{test_output}\n\n"
            "Extract the requested metric from the test output above. "
            "Return ONLY a valid JSON object with the extracted data. "
            "If the metric cannot be found, return null. "
            "Examples:\n"
            '- For numeric value: {"value": 123.45}\n'
            '- For array: {"values": [1.2, 3.4, 5.6]}\n'
            '- For multiple metrics: {"bandwidth": 123.45, "latency": 0.5}\n'
        )
        response = self.model.query([{"role": "system", "content": "You are a helpful assistant to analyze kernel test output and extract metrics from test logs."}, {"role": "user", "content": prompt}])
        content = response.get("content", "")
        try:
            result = json.loads(content.strip())
            self._log_message(f"[ParallelAgent] Metric extracted: {result}")
            return result
        except json.JSONDecodeError:
            self._log_message(f"[ParallelAgent] Failed to parse metric response: {content}")
            return None

    def _save_patch_file(self, patch_name: str, patch_content: str):
        output_dir = Path(self.config.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_file = output_dir / f"{patch_name}.patch"
        patch_file.write_text(patch_content)
        self._log_message(f"[ParallelAgent] Patch saved to: {patch_file}")
    
    def _save_test_output(self, patch_name: str, test_output: str):
        output_dir = Path(self.config.patch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / f"{patch_name}_test.txt"
        test_file.write_text(test_output)
        self._log_message(f"[ParallelAgent] Test output saved to: {test_file}")
    
    def _update_results_file(self):
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
        self._log_message(f"[ParallelAgent] Results updated: {results_file}")

    def run(self, task: str, **kwargs) -> tuple[str, str] | Any:
        # Handle parallel execution if num_parallel > 1
        if self.config.num_parallel > 1:
            if not self.config.repo:
                raise ValueError("repo is required when num_parallel > 1. Please specify the repository path.")
            repo_path = Path(self.config.repo) if isinstance(self.config.repo, (str, Path)) else self.config.repo
            repo_path = repo_path.resolve()
            if not repo_path.exists():
                raise ValueError(f"Repository path does not exist: {repo_path}")
            if not (repo_path / ".git").exists():
                raise ValueError(f"Repository path is not a git repository: {repo_path}")
            
            base_patch_dir = Path(self.config.patch_output_dir) if self.config.patch_output_dir else Path("patches")
            output = kwargs.get("output")
            save_traj_fn = kwargs.get("save_traj_fn")
            console = kwargs.get("console")
            model_factory = kwargs.get("model_factory")
            env_factory = kwargs.get("env_factory")
            
            if not model_factory or not env_factory:
                raise ValueError("model_factory and env_factory must be provided in kwargs when num_parallel > 1")
            
            return self.run_parallel(
                num_parallel=self.config.num_parallel,
                repo_path=repo_path,
                task_content=task,
                agent_config={k: v for k, v in self.config.__dict__.items() if k not in ('num_parallel', 'repo', 'parallel_gpu_ids')},
                model_factory=model_factory,
                env_factory=env_factory,
                base_patch_dir=base_patch_dir,
                output=output,
                metric_model_config=kwargs.get("metric_model_config"),
                parallel_gpu_ids=self.config.parallel_gpu_ids,
                save_traj_fn=save_traj_fn,
                console=console,
            )
        
        # Single agent execution
        init_msg = f"\n{'='*60}\n"
        init_msg += f"[ParallelAgent] Starting with patch saving enabled\n"
        init_msg += f"[ParallelAgent] Test command: {self.config.test_command}\n"
        init_msg += f"[ParallelAgent] Patch output directory: {self.config.patch_output_dir}\n"
        if self.config.metric:
            init_msg += f"[ParallelAgent] Metric extraction: {self.config.metric}\n"
        init_msg += f"[ParallelAgent] Triggers: Use 'TEST_BASELINE_PERFORMANCE' or 'SAVE_PATCH_AND_TEST' in command output\n"
        init_msg += f"{'='*60}\n\n"
        
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(init_msg)
                    f.flush()
            except Exception:
                pass
        
        exit_status, result = super().run(task, **kwargs)
        
        completion_msg = f"\n[ParallelAgent] Agent execution completed\n"
        completion_msg += f"[ParallelAgent] Exit status: {exit_status}\n"
        
        self._print_summary()
        completion_msg += f"[ParallelAgent] Trajectory will be saved by the runner\n"
        
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(completion_msg)
                    f.flush()
            except Exception:
                pass
        return exit_status, result

    def _print_summary(self):
        if not self.patch_results:
            return
        print(f"\n{'='*60}", flush=True)
        print(f"[ParallelAgent] Summary:", flush=True)
        print(f"  Total patches: {len(self.patch_results)}", flush=True)
        passed = sum(1 for data in self.patch_results.values() if data["test_passed"])
        failed = len(self.patch_results) - passed
        print(f"  Passed: {passed}", flush=True)
        print(f"  Failed: {failed}", flush=True)
        if self.config.patch_output_dir:
            print(f"  Results saved to: {Path(self.config.patch_output_dir) / 'results.json'}", flush=True)
        print(f"{'='*60}\n", flush=True)


    @staticmethod
    def _select_best_from_parallel_runs(base_patch_dir: Path, num_parallel: int, metric: str | None, metric_model_config: dict) -> BestPatchResult | None:
        """Select the best patch from multiple parallel runs using SelectPatchAgent."""
        from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
        
        print("[ParallelAgent] Using SelectPatchAgent for patch selection...", flush=True)
        
        # Create model and environment for the SelectPatchAgent
        model = get_model(config=metric_model_config)
        env_config = LocalEnvironmentConfig(cwd=str(base_patch_dir))
        env = LocalEnvironment(**env_config.__dict__)
        
        # Create SelectPatchAgent
        select_agent = SelectPatchAgent(model, env)
        
        # Setup the selection task
        task = select_agent.setup_selection_task(base_patch_dir, num_parallel, metric)
        
        if task is None:
            print("[ParallelAgent] Failed to setup selection task", flush=True)
            return None
        
        # Save agent conversation log
        log_file = base_patch_dir / "select_agent.log"
        select_agent.log_file = log_file
        
        print(f"[ParallelAgent] Running SelectPatchAgent (log: {log_file})...", flush=True)
        
        # Run the agent
        try:
            exit_status, result = select_agent.run(task)
            print(f"[ParallelAgent] SelectPatchAgent finished with status: {exit_status}", flush=True)
        except Exception as e:
            print(f"[ParallelAgent] SelectPatchAgent failed: {e}", flush=True)
            traceback.print_exc()
        
        # Extract the final result string (agent_id/patch_id)
        result_str = select_agent.extract_final_result()
        
        if not result_str:
            print("[ParallelAgent] SelectPatchAgent did not produce valid result", flush=True)
            return None
        
        print(f"[ParallelAgent] SelectPatchAgent returned: {result_str}", flush=True)
        
        # Read best_results.json saved by select agent
        best_results_file = base_patch_dir / "best_results.json"
        if not best_results_file.exists():
            print(f"[ParallelAgent] best_results.json not found at {best_results_file}", flush=True)
            return None
        
        try:
            best_results = json.loads(best_results_file.read_text())
        except json.JSONDecodeError as e:
            print(f"[ParallelAgent] Failed to parse best_results.json: {e}", flush=True)
            return None
        
        # Extract metadata from best_results.json
        best_agent_id = best_results.get("_selected_from_agent")
        best_patch_name = best_results.get("_selected_patch_id")
        best_patch_file = best_results.get("_selected_patch_file")
        best_test_output_file = best_results.get("_selected_test_output_file")
        llm_analysis = best_results.get("_llm_selection_analysis", "")
        parallel_dir_name = best_results.get("_selected_from_parallel_dir", f"parallel_{best_agent_id}")
        
        if best_agent_id is None or not best_patch_name:
            print("[ParallelAgent] Invalid best_results.json format", flush=True)
            return None
        
        print(f"[ParallelAgent] Selected best patch: agent_{best_agent_id}/{best_patch_name}", flush=True)
        
        # Get patch data
        best_patch_data = best_results.get(best_patch_name)
        if not best_patch_data:
            print(f"[ParallelAgent] Patch data for {best_patch_name} not found in best_results.json", flush=True)
            return None
        
        best_patch_dir = base_patch_dir / parallel_dir_name
        
        # Read test output
        test_output = ""
        if best_test_output_file:
            test_output_path = Path(best_test_output_file)
            if test_output_path.exists():
                test_output = test_output_path.read_text()
        
        return BestPatchResult(
            agent_id=best_agent_id,
            patch_id=best_patch_name,
            test_output=test_output,
            metric_result=best_patch_data.get("metric_result"),
            patch_dir=best_patch_dir,
            llm_conclusion=llm_analysis,
        )

    @staticmethod
    def _ensure_safe_directory(repo_path: Path):
        """Ensure repository is in git's safe.directory list."""
        repo_path_str = str(repo_path.resolve())
        try:
            result = subprocess.run(
                ["git", "config", "--global", "--get-all", "safe.directory"],
                capture_output=True,
                text=True,
            )
            safe_dirs = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if repo_path_str not in safe_dirs:
                subprocess.run(
                    ["git", "config", "--global", "--add", "safe.directory", repo_path_str],
                    check=True,
                    capture_output=True,
                    text=True,
                )
        except subprocess.CalledProcessError:
            try:
                subprocess.run(
                    ["git", "config", "--global", "--add", "safe.directory", repo_path_str],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                pass

    @staticmethod
    def _create_worktree(repo_path: Path, worktree_path: Path) -> Path:
        """Create a git worktree, cleaning up any existing one first."""
        worktree_str = str(worktree_path.resolve())
        
        # Clean up any existing worktree
        try:
            result = subprocess.run(
                ["git", "worktree", "list"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            worktree_exists = any(worktree_str in line or str(worktree_path) in line for line in result.stdout.splitlines())
            
            if worktree_exists:
                try:
                    subprocess.run(
                        ["git", "worktree", "remove", str(worktree_path), "--force"],
                        cwd=repo_path,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError:
                    subprocess.run(["git", "worktree", "prune"], cwd=repo_path, check=False, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "worktree", "prune"], cwd=repo_path, check=False, capture_output=True, text=True)
        except Exception:
            pass
        
        # Remove directory if it still exists
        if worktree_path.exists():
            try:
                shutil.rmtree(worktree_path)
            except Exception:
                pass
        
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        ParallelAgent._ensure_safe_directory(repo_path)
        
        # Create new worktree with detached HEAD to avoid branch name conflicts
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_path)],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else str(e))
            if "missing but already registered worktree" in error_msg.lower():
                subprocess.run(["git", "worktree", "prune"], cwd=repo_path, check=False, capture_output=True, text=True)
                subprocess.run(
                    ["git", "worktree", "add", "--detach", "-f", str(worktree_path)],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            elif "dubious ownership" in error_msg.lower():
                ParallelAgent._ensure_safe_directory(repo_path)
                ParallelAgent._ensure_safe_directory(worktree_path)
                subprocess.run(
                    ["git", "worktree", "add", "--detach", str(worktree_path)],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            elif "already used by worktree" in error_msg.lower():
                # Branch name conflict - remove old worktree and retry
                subprocess.run(["git", "worktree", "prune"], cwd=repo_path, check=False, capture_output=True, text=True)
                # Extract branch name from error message if possible, otherwise use worktree path
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree_path)],
                    cwd=repo_path,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "worktree", "add", "--detach", str(worktree_path)],
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                raise RuntimeError(f"Failed to create worktree: {error_msg}") from e
        
        # Ensure worktree path is also marked as safe
        ParallelAgent._ensure_safe_directory(worktree_path)
        
        return worktree_path

    @staticmethod
    def _replace_paths(text: str, repo_path: Path, worktree_path: Path) -> str:
        """Replace repository paths with worktree path in text."""
        repo_path_str = str(repo_path.resolve())
        worktree_path_str = str(worktree_path.resolve())
        text = text.replace("WORK_REPO", worktree_path_str)
        text = text.replace(repo_path_str, worktree_path_str)
        if str(repo_path) != repo_path_str:
            text = text.replace(str(repo_path), worktree_path_str)
        text = re.sub(r'/worktrees/agent_\d+', f'/worktrees/agent_{worktree_path.name.split("_")[-1]}', text)
        return text

    @classmethod
    def run_parallel(
        cls,
        num_parallel: int,
        repo_path: Path,
        task_content: str,
        agent_config: dict,
        model_factory,
        env_factory,
        base_patch_dir: Path,
        output: Path | None,
        metric_model_config: dict | None = None,
        parallel_gpu_ids: list[int] | None = None,
        redirect_output_fn=redirect_output_to_file,
        save_traj_fn=None,
        console=None,
    ) -> Any:
        """Run multiple parallel agents and select the best result."""
        if console:
            console.print(f"[bold green]Running {num_parallel} parallel patch agents...[/bold green]")
        
        worktree_base = base_patch_dir / "worktrees"
        worktree_base.mkdir(parents=True, exist_ok=True)
        repo_path_resolved = repo_path.resolve()
        repo_path_str = str(repo_path_resolved)
        
        if parallel_gpu_ids and len(parallel_gpu_ids) < num_parallel:
            if console:
                console.print(f"[bold yellow]Warning: Only {len(parallel_gpu_ids)} GPU IDs provided for {num_parallel} parallel agents. Some agents will not have GPU isolation.[/bold yellow]")
        
        def run_single_agent(agent_id: int):
            """Run a single parallel agent instance."""
            worktree_path = cls._create_worktree(repo_path, worktree_base / f"agent_{agent_id}")
            worktree_path_str = str(worktree_path.resolve())
            
            if console:
                console.print(f"[bold green]Created worktree for agent {agent_id}: {worktree_path}[/bold green]")
            
            parallel_patch_dir = base_patch_dir / f"parallel_{agent_id}"
            parallel_patch_dir.mkdir(parents=True, exist_ok=True)
            parallel_agent_config = agent_config.copy()
            parallel_agent_config["patch_output_dir"] = str(parallel_patch_dir)
            
            log_file = parallel_patch_dir / f"agent_{agent_id}.log"
            
            # Replace paths in test_command and task_content
            if parallel_agent_config.get("test_command"):
                parallel_agent_config["test_command"] = cls._replace_paths(
                    parallel_agent_config["test_command"], repo_path, worktree_path
                )
            
            task_with_repo = cls._replace_paths(task_content, repo_path, worktree_path)
            
            # Create model and environment
            parallel_model = model_factory()
            base_env = env_factory()
            env_config_dict = base_env.config.__dict__.copy() if hasattr(base_env, 'config') else {}
            env_config_dict["cwd"] = str(worktree_path)
            env_config_dict.setdefault("env", {})["WORK_REPO"] = worktree_path_str
            if parallel_gpu_ids and agent_id < len(parallel_gpu_ids):
                gpu_id = parallel_gpu_ids[agent_id]
                env_config_dict.setdefault("env", {})["HIP_VISIBLE_DEVICES"] = str(gpu_id)
                if console:
                    # Use lock to ensure console output completes before stdout redirection
                    with _stdout_lock:
                        console.print(f"[bold green]Parallel agent {agent_id} using GPU {gpu_id}[/bold green]")
                        # Force flush to ensure output is written before redirection
                        if hasattr(sys.stdout, 'flush'):
                            sys.stdout.flush()
            parallel_env = type(base_env)(**env_config_dict)
            
            parallel_output = None
            if output:
                parallel_output = output.parent / f"{output.stem}_parallel_{agent_id}{output.suffix}"
            
            agent = cls(parallel_model, parallel_env, **parallel_agent_config)
            agent.extra_template_vars["WORK_REPO"] = worktree_path_str
            agent.log_file = log_file
            
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Agent {agent_id} Conversation Log\n")
                f.write("=" * 60 + "\n\n")
            
            exit_status, result, extra_info = None, None, None
            with redirect_output_fn(log_file):
                try:
                    exit_status, result = agent.run(task_with_repo, _is_parallel_mode=True)
                except Exception as e:
                    exit_status, result = type(e).__name__, str(e)
                    extra_info = {"traceback": traceback.format_exc()}
                finally:
                    if parallel_output and save_traj_fn:
                        save_traj_fn(agent, parallel_output, exit_status=exit_status, result=result, extra_info=extra_info)
            
            return agent_id, agent, exit_status, result
        
        # Run parallel agents
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = {executor.submit(run_single_agent, i): i for i in range(num_parallel)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    agent_id = futures[future]
                    from minisweagent.utils.log import logger
                    logger.error(f"Error in parallel agent {agent_id}: {e}", exc_info=True)
        
        # Select best patch from all parallel runs
        if console:
            console.print(f"\n[bold green]Selecting best patch from {num_parallel} parallel runs...[/bold green]")
        # Use metric_model_config if provided, otherwise fall back to main model config
        if metric_model_config is None:
            metric_model_config = {}
        best_result = cls._select_best_from_parallel_runs(
            base_patch_dir, num_parallel, agent_config.get("metric"), metric_model_config
        )
        
        if best_result is not None:
            if console:
                if best_result.llm_conclusion:
                    console.print(f"\n[bold cyan]LLM Conclusion:[/bold cyan]")
                    console.print(best_result.llm_conclusion)
            
            base_patch_dir.mkdir(parents=True, exist_ok=True)
            
            # Read results from the best patch's parallel directory to get best patch data
            best_results_file = best_result.patch_dir / "results.json"
            if best_results_file.exists():
                results_data = json.loads(best_results_file.read_text())
                best_patch_data = results_data.get(best_result.patch_id, {}).copy()
                
                # Add agent_id and parallel_dir to the best patch
                best_patch_data["agent_id"] = best_result.agent_id
                best_patch_data["parallel_dir"] = f"parallel_{best_result.agent_id}"
                
                # Create output with only the best patch
                output_data = {
                    best_result.patch_id: best_patch_data,
                    "_selected_from_agent": best_result.agent_id,
                    "_selected_from_parallel_dir": f"parallel_{best_result.agent_id}",
                    "_selected_patch_id": best_result.patch_id,
                }
                
                # Add file paths for the best patch
                best_patch_file = best_result.patch_dir / best_patch_data.get("patch_file", f"{best_result.patch_id}.patch")
                output_data["_selected_patch_file"] = str(best_patch_file.resolve()) if best_patch_file.exists() else None
                
                best_test_file = best_result.patch_dir / best_patch_data.get("test_output_file", f"{best_result.patch_id}_test.txt")
                output_data["_selected_test_output_file"] = str(best_test_file.resolve()) if best_test_file.exists() else None
                
                # Add LLM analysis
                if best_result.llm_conclusion:
                    output_data["_llm_selection_analysis"] = best_result.llm_conclusion
                
                (base_patch_dir / "best_results.json").write_text(json.dumps(output_data, indent=2))
        
        if results:
            # Return (exit_status, result) tuple from the first successful result
            return results[0][2], results[0][3]
        else:
            # All agents failed - return error status
            return "Error", "All parallel agents failed"

        

