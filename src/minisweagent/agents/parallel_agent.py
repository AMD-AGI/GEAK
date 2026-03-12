"""Agent with git patch saving and test execution capability."""

import concurrent.futures
import json
import re
import shutil
import sys
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent, TerminatingException
from minisweagent.agents.select_patch_agent import run_select_patch


@dataclass
class BestPatchResult:
    """Result of selecting the best patch from parallel runs."""

    agent_id: int
    patch_id: str
    test_output: str
    metric_result: dict | None = None
    patch_dir: Path | None = None
    llm_conclusion: str | None = None


_stdout_lock = threading.Lock()


@contextmanager
def redirect_output_to_file(log_file: Path):
    """No-op context manager. Agent writes to log file directly via add_message/log_message.

    Stdout/stderr redirection doesn't work for parallel threads since sys.stdout is global.
    """
    yield


@dataclass
class ParallelAgentConfig(AgentConfig):
    # save_patch, test_command, patch_output_dir, metric are now inherited from AgentConfig
    mode: str | None = None
    num_parallel: int = 1
    repo: Path | None = None
    gpu_ids: list[int] | None = None
    agent_class: type | None = None
    agent_specs: list | None = None  # list[AgentSpec] for heterogeneous parallel
    tasks: list | None = None  # list[AgentTask] for GPU pool mode
    # Strategy agent compatibility
    strategy_file_path: str | None = None
    # Interactive/exit behaviour (passed through from --exit-immediately)
    confirm_exit: bool = True


class ParallelAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, **kwargs):
        super().__init__(model, env, config_class=ParallelAgentConfig, **kwargs)
        # patch_results, patch_counter, log_file, base_repo_path are now inherited from DefaultAgent
        self._last_action_hash: str | None = None

    # _is_git_repo(), _diff_excludes(), add_message, _log_message are inherited from DefaultAgent

    def run(self, task: str, **kwargs) -> tuple[str, str] | Any:
        num_parallel = self.config.num_parallel or 1
        console = kwargs.get("console")

        base_patch_dir = (
            Path(self.config.patch_output_dir) if self.config.patch_output_dir else Path("patches")
        ).resolve()
        model_factory = kwargs.get("model_factory") or (lambda: self.model)
        env_factory = kwargs.get("env_factory") or (lambda: self.env)

        if num_parallel == 1:
            # For single run, save patches directly to base_patch_dir (no parallel_0 subdirectory)
            base_patch_dir.mkdir(parents=True, exist_ok=True)
            prev_patch_output_dir = self.config.patch_output_dir
            self.config.patch_output_dir = str(base_patch_dir)
            try:
                exit_status, result = super().run(task, **(kwargs | {"_skip_select_patch": True}))
            finally:
                self.config.patch_output_dir = prev_patch_output_dir

            metric = (
                self.config.metric
                or "Extract the performance metrics from the test output and calculate the best speedup."
            )
            if console:
                console.print("\n[bold green]Selecting best patch from 1 run...[/bold green]")
            best_result = self._select_best_from_parallel_runs(base_patch_dir, 1, metric, model_factory)
            if best_result and console and best_result.llm_conclusion:
                console.print("\n[bold cyan]LLM Conclusion:[/bold cyan]")
                console.print(best_result.llm_conclusion)
            return exit_status, result

        if not self.config.repo:
            raise ValueError("Please specify the repository path.")
        repo_path = (
            Path(self.config.repo) if isinstance(self.config.repo, (str, Path)) else self.config.repo
        ).resolve()
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        # Determine whether repo_path is already a git repo. For non-git repos we
        # create per-agent isolated copies and bootstrap git *inside the copy* so
        # save_and_test can capture diffs without modifying the source directory.
        is_git_repo = (repo_path / ".git").exists() or (repo_path / ".git").is_file()

        output = kwargs.get("output")
        save_traj_fn = kwargs.get("save_traj_fn")

        results = self.run_parallel(
            num_parallel=num_parallel,
            repo_path=repo_path,
            is_git_repo=is_git_repo,
            task_content=task,
            agent_class=self.config.agent_class if self.config.agent_class else type(self),
            agent_config={
                k: v
                for k, v in self.config.__dict__.items()
                if k not in ("num_parallel", "repo", "gpu_ids", "agent_class", "agent_specs", "tasks")
            },
            model_factory=model_factory,
            env_factory=env_factory,
            base_patch_dir=base_patch_dir,
            output=output,
            gpu_ids=self.config.gpu_ids,
            save_traj_fn=save_traj_fn,
            console=console,
            agent_specs=self.config.agent_specs,
            tasks=self.config.tasks,
        )

        metric = (
            self.config.metric or "Extract the performance metrics from the test output and calculate the best speedup."
        )
        if console:
            console.print(f"\n[bold green]Selecting best patch from {num_parallel} parallel runs...[/bold green]")
        best_result = self._select_best_from_parallel_runs(base_patch_dir, num_parallel, metric, model_factory)
        if best_result and console and best_result.llm_conclusion:
            console.print("\n[bold cyan]LLM Conclusion:[/bold cyan]")
            console.print(best_result.llm_conclusion)

        if results:
            return results[0][2], results[0][3]
        return "Error", "All parallel agents failed"

    @staticmethod
    def _select_best_from_parallel_runs(
        base_patch_dir: Path, num_parallel: int, metric: str | None, model_factory
    ) -> BestPatchResult | None:
        """Select the best patch from multiple parallel runs using SelectPatchAgent."""
        print("[ParallelAgent] Using SelectPatchAgent for patch selection...", flush=True)

        model = model_factory()
        _, best_patch_id = run_select_patch(base_patch_dir, num_parallel, metric, model)

        # Override with deterministic benchmark parsing when possible
        from minisweagent.benchmark_parsing import rewrite_best_results
        det_result = rewrite_best_results(base_patch_dir)
        if det_result:
            best_patch_id = det_result.get("best_patch_id", best_patch_id)
            print(
                f"[ParallelAgent] Deterministic override: {best_patch_id} "
                f"({det_result.get('best_patch_speedup', '?')}x)",
                flush=True,
            )

        if not best_patch_id:
            print("[ParallelAgent] SelectPatchAgent did not produce best_results.json", flush=True)
            return None

        print(f"[ParallelAgent] Selected best patch: {best_patch_id}", flush=True)

        try:
            # Read the best_results.json for additional details
            best_results = json.loads((base_patch_dir / "best_results.json").read_text())

            # Parse best_patch_id: "parallel_X/patch_Y", "task_X/patch_Y", or "patch_Y"
            if "/" in best_patch_id:
                dir_name, patch_name = best_patch_id.split("/", 1)
                patch_dir = base_patch_dir / dir_name
                # Extract numeric ID from either "parallel_X" or "task_X"
                import re as _re

                id_match = _re.search(r"(\d+)", dir_name)
                agent_id = int(id_match.group(1)) if id_match else 0
            else:
                # Single run format: "patch_Y" (directly in base_patch_dir)
                patch_name = best_patch_id
                agent_id = 0
                patch_dir = base_patch_dir

            # metric_result is no longer persisted (results.json removed); rely on test logs if needed
            metric_result = None

            # Read test output if path provided
            test_output = ""
            test_output_path = best_results.get("best_patch_test_output")
            if test_output_path and Path(test_output_path).exists():
                test_output = Path(test_output_path).read_text()

            return BestPatchResult(
                agent_id=agent_id,
                patch_id=patch_name,
                test_output=test_output,
                metric_result=metric_result,
                patch_dir=patch_dir,
                llm_conclusion=best_results.get("llm_selection_analysis", ""),
            )
        except Exception as e:
            print(f"[ParallelAgent] Failed to process best_results.json: {e}", flush=True)
            return None

    @staticmethod
    def _ensure_safe_directory(repo_path: Path):
        """Ensure repository is in git's safe.directory list."""
        from minisweagent.run.task_file import _ensure_safe_directory

        _ensure_safe_directory(repo_path)

    @staticmethod
    def _bootstrap_git_repo(repo_path: Path, console=None) -> bool:
        """Bootstrap a minimal git repository for non-git directories.

        Creates .git, adds .gitignore to exclude build artifacts, and creates
        an initial commit. This allows unified git diff-based patch generation.

        Returns True if successful, False otherwise.
        """
        import subprocess

        try:
            # 1. git init
            subprocess.run(
                ["git", "init", "-b", "geak-bootstrap"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # 2. Create .gitignore with common build artifacts
            gitignore_path = repo_path / ".gitignore"
            gitignore_content = """# GEAK auto-generated gitignore for build artifacts
build/
*/build/
.rocprofv3/
__pycache__/
*.pyc
*.o
*.so
*.a
*.log
*.dat
optimization_logs/
*_logs/
CMakeCache.txt
CMakeFiles/
.pytest_cache/
*.egg-info/
.geak_resolved/
traj.json
"""
            # Append to existing .gitignore if present
            if gitignore_path.exists():
                existing = gitignore_path.read_text()
                if "# GEAK auto-generated" not in existing:
                    gitignore_path.write_text(existing + "\n" + gitignore_content)
            else:
                gitignore_path.write_text(gitignore_content)

            # 3. git add -A (respects .gitignore)
            subprocess.run(
                ["git", "add", "-A"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # 4. git commit (use -c to avoid needing global config)
            subprocess.run(
                [
                    "git",
                    "-c", "user.name=geak-bootstrap",
                    "-c", "user.email=geak@local",
                    "commit",
                    "-m", "GEAK bootstrap commit",
                    "--allow-empty",
                ],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if console:
                console.print("[bold green]Git repo bootstrapped successfully[/bold green]")
            return True

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or str(e)
            if console:
                console.print(f"[bold red]Failed to bootstrap git repo: {error_msg}[/bold red]")
            return False
        except Exception as e:
            if console:
                console.print(f"[bold red]Failed to bootstrap git repo: {e}[/bold red]")
            return False

    @staticmethod
    def _create_worktree(repo_path: Path, worktree_path: Path) -> Path:
        """Create a git worktree, cleaning up any existing one first."""
        from minisweagent.run.task_file import create_worktree

        return create_worktree(repo_path, worktree_path)

    @staticmethod
    def _copy_untracked_files(repo_path: Path, worktree_path: Path) -> None:
        """Copy untracked files from repo to worktree."""
        from minisweagent.run.task_file import _copy_untracked_files

        _copy_untracked_files(repo_path, worktree_path)

    @staticmethod
    def _create_copy_workdir(repo_path: Path, workdir_path: Path) -> Path:
        """Create an isolated work directory by copying `repo_path` (for non-git repos)."""
        if workdir_path.exists():
            try:
                shutil.rmtree(workdir_path)
            except Exception:
                pass
        workdir_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            repo_path,
            workdir_path,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        return workdir_path

    @staticmethod
    def _replace_paths(text: str, repo_path: Path, worktree_path: Path) -> str:
        """Replace repository paths with worktree path in text.

        Uses the provided repo_path (no hardcoded paths) to rewrite any absolute
        reference so that it points into the current worktree.
        """
        repo_path_str = str(repo_path.resolve())
        worktree_path_str = str(worktree_path.resolve())

        # If the text already contains paths pointing into a *previous* worktree
        # (e.g. "<repo>/optimization_logs/<run>/worktrees/agent_X/..."),
        # collapse that whole prefix back to the current worktree root first.
        # This prevents path "nesting" when replacement is applied more than once.
        prev_worktree_pat = re.compile(re.escape(repo_path_str) + r"/optimization_logs/\S*/worktrees/agent_\d+")
        text = prev_worktree_pat.sub(worktree_path_str, text)

        # Replace repo path (resolved and unresolved forms) with worktree path
        text = text.replace(repo_path_str, worktree_path_str)
        if str(repo_path) != repo_path_str:
            text = text.replace(str(repo_path), worktree_path_str)

        # Keep agent id in any remaining /worktrees/agent_<id> segments aligned
        # with this worktree.
        return re.sub(
            r"/worktrees/agent_\d+",
            f"/worktrees/agent_{worktree_path.name.split('_')[-1]}",
            text,
        )

    @classmethod
    def run_parallel(
        cls,
        num_parallel: int,
        repo_path: Path,
        is_git_repo: bool,
        task_content: str,
        agent_class: type,
        agent_config: dict,
        model_factory,
        env_factory,
        base_patch_dir: Path,
        output: Path | None,
        gpu_ids: list[int] | None = None,
        redirect_output_fn=redirect_output_to_file,
        save_traj_fn=None,
        console=None,
        agent_specs: list | None = None,
        tasks: list | None = None,
    ) -> list[tuple[int, Any, Any, Any]]:
        """Run multiple parallel agents and return their results.

        Supports three modes (checked in priority order):
        - Pool (preferred): pass tasks (list[AgentTask]) for M tasks on N GPUs.
          Tasks are decoupled from GPUs; overflow tasks queue and run as GPUs free up.
        - Heterogeneous (legacy): pass agent_specs (list[AgentSpec]) for different
          agent types with fixed GPU assignments.
        - Homogeneous (default): num_parallel identical agents, each with 1 GPU.
        """
        # Pool mode: M tasks on N GPU slots (preferred)
        if tasks:
            effective_gpu_ids = gpu_ids or [0]
            return cls._run_pool(
                tasks=tasks,
                gpu_ids=effective_gpu_ids,
                repo_path=repo_path,
                is_git_repo=is_git_repo,
                base_task_content=task_content,
                agent_config=agent_config,
                model_factory=model_factory,
                env_factory=env_factory,
                base_patch_dir=base_patch_dir,
                output=output,
                redirect_output_fn=redirect_output_fn,
                save_traj_fn=save_traj_fn,
                console=console,
            )

        # Heterogeneous mode: use agent_specs if provided (legacy)
        if agent_specs:
            return cls._run_parallel_heterogeneous(
                agent_specs=agent_specs,
                repo_path=repo_path,
                is_git_repo=is_git_repo,
                task_content=task_content,
                agent_config=agent_config,
                model_factory=model_factory,
                env_factory=env_factory,
                base_patch_dir=base_patch_dir,
                output=output,
                redirect_output_fn=redirect_output_fn,
                save_traj_fn=save_traj_fn,
                console=console,
            )

        # Homogeneous mode (original behavior)
        if console:
            console.print(f"[bold green]Running {num_parallel} parallel patch agents...[/bold green]")

        base_patch_dir = base_patch_dir.resolve()
        worktree_base = base_patch_dir / "worktrees"
        worktree_base.mkdir(parents=True, exist_ok=True)
        repo_path_resolved = repo_path.resolve()
        repo_path_str = str(repo_path_resolved)

        if gpu_ids and len(gpu_ids) < num_parallel:
            if console:
                console.print(
                    f"[bold yellow]Warning: Only {len(gpu_ids)} GPU IDs provided for {num_parallel} parallel agents. Some agents will not have GPU isolation.[/bold yellow]"
                )

        def run_single_agent(agent_id: int):
            """Run a single parallel agent instance."""
            if is_git_repo:
                worktree_path = cls._create_worktree(repo_path, worktree_base / f"agent_{agent_id}")
            else:
                worktree_path = cls._create_copy_workdir(repo_path, worktree_base / f"agent_{agent_id}")
                # Bootstrap a local git repo in the isolated copy so diffs are
                # captured from this workspace (avoid accidentally diffing a parent repo).
                cls._bootstrap_git_repo(worktree_path, console)
            worktree_path_str = str(worktree_path.resolve())

            if console:
                console.print(f"[bold green]Created worktree for agent {agent_id}: {worktree_path}[/bold green]")

            parallel_patch_dir = (base_patch_dir / f"parallel_{agent_id}").resolve()
            parallel_patch_dir.mkdir(parents=True, exist_ok=True)
            parallel_agent_config = agent_config.copy()
            parallel_agent_config["patch_output_dir"] = str(parallel_patch_dir)
            # Force yolo mode for parallel agents (no interactive confirmation prompts)
            parallel_agent_config["mode"] = "yolo"
            parallel_agent_config["confirm_exit"] = False

            log_file = parallel_patch_dir / f"agent_{agent_id}.log"

            # test_command should use relative paths, executed from worktree cwd
            # Path replacement kept for backward compatibility with absolute paths
            if parallel_agent_config.get("test_command"):
                parallel_agent_config["test_command"] = cls._replace_paths(
                    parallel_agent_config["test_command"], repo_path, worktree_path
                )

            task_with_repo = cls._replace_paths(task_content, repo_path, worktree_path)

            # Create model and environment
            parallel_model = model_factory()
            base_env = env_factory()
            env_config_dict = base_env.config.__dict__.copy() if hasattr(base_env, "config") else {}
            env_config_dict["cwd"] = worktree_path_str
            # Create a NEW dict to avoid shared-reference race across threads
            new_env = dict(env_config_dict.get("env") or {})
            new_env[repo_path_str] = worktree_path_str
            new_env["GEAK_WORK_DIR"] = worktree_path_str
            new_env["GEAK_REPO_ROOT"] = repo_path_str
            if gpu_ids and agent_id < len(gpu_ids):
                gpu_id = gpu_ids[agent_id]
                new_env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
                new_env["GEAK_GPU_DEVICE"] = str(gpu_id)
                if console:
                    # Use lock to ensure console output completes before stdout redirection
                    with _stdout_lock:
                        console.print(f"[bold green]Parallel agent {agent_id} using GPU {gpu_id}[/bold green]")
                        # Force flush to ensure output is written before redirection
                        if hasattr(sys.stdout, "flush"):
                            sys.stdout.flush()
            env_config_dict["env"] = new_env
            parallel_env = type(base_env)(**env_config_dict)

            parallel_output = None
            if output:
                parallel_output = output.parent / f"{output.stem}_parallel_{agent_id}{output.suffix}"

            agent = agent_class(parallel_model, parallel_env, **parallel_agent_config)
            # Set agent attributes if they exist (for ParallelAgent compatibility)
            if hasattr(agent, "extra_template_vars"):
                agent.extra_template_vars[repo_path_str] = worktree_path_str
            if hasattr(agent, "base_repo_path"):
                agent.base_repo_path = repo_path_resolved
                # Re-initialize test_perf context with updated base_repo_path
                if hasattr(agent, '_setup_test_perf_context'):
                    agent._setup_test_perf_context()
            if hasattr(agent, "log_file"):
                agent.log_file = log_file

            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Agent {agent_id} Conversation Log\n")
                f.write("=" * 60 + "\n\n")

            init_msg = (
                f"\n{'=' * 60}\n"
                "[ParallelAgent] Starting with patch saving enabled\n"
                f"[ParallelAgent] Test command: {parallel_agent_config.get('test_command')}\n"
                f"[ParallelAgent] Patch output directory: {parallel_agent_config.get('patch_output_dir')}\n"
                f"[ParallelAgent] Metric extraction: {parallel_agent_config.get('metric') or 'Automatic (LLM will extract performance metrics and calculate speedup)'}\n"
                f"{'=' * 60}\n\n"
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(init_msg)
                f.flush()

            exit_status, result, extra_info = None, None, None
            with redirect_output_fn(log_file):
                try:
                    exit_status, result = agent.run(task_with_repo, _is_parallel_mode=True)
                except Exception as e:
                    exit_status, result = type(e).__name__, str(e)
                    extra_info = {"traceback": traceback.format_exc()}
                    # Write error to log file
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n\nERROR: {exit_status}: {result}\n")
                        f.write(f"Traceback:\n{extra_info['traceback']}\n")
                finally:
                    if parallel_output and save_traj_fn:
                        save_traj_fn(
                            agent, parallel_output, exit_status=exit_status, result=result, extra_info=extra_info
                        )

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
        return results

    @classmethod
    def _run_parallel_heterogeneous(
        cls,
        agent_specs: list,
        repo_path: Path,
        is_git_repo: bool,
        task_content: str,
        agent_config: dict,
        model_factory,
        env_factory,
        base_patch_dir: Path,
        output: Path | None,
        redirect_output_fn=redirect_output_to_file,
        save_traj_fn=None,
        console=None,
    ) -> list[tuple[int, Any, Any, Any]]:
        """Run heterogeneous parallel agents from AgentSpec list."""
        num_agents = len(agent_specs)
        if console:
            labels = [s.label or s.agent_class.__name__ for s in agent_specs]
            console.print(f"[bold green]Running {num_agents} heterogeneous agents: {labels}[/bold green]")

        base_patch_dir = base_patch_dir.resolve()
        worktree_base = base_patch_dir / "worktrees"
        worktree_base.mkdir(parents=True, exist_ok=True)
        repo_path_resolved = repo_path.resolve()

        def run_spec_agent(agent_id: int, spec):
            """Run one agent from an AgentSpec."""
            if is_git_repo:
                worktree_path = cls._create_worktree(repo_path, worktree_base / f"agent_{agent_id}")
            else:
                worktree_path = cls._create_copy_workdir(repo_path, worktree_base / f"agent_{agent_id}")
                cls._bootstrap_git_repo(worktree_path, console)
            worktree_path_str = str(worktree_path.resolve())

            label = spec.label or spec.agent_class.__name__
            if console:
                with _stdout_lock:
                    console.print(
                        f"[bold green]Agent {agent_id} ({label}): "
                        f"GPU {spec.hip_visible_devices}, worktree {worktree_path}[/bold green]"
                    )

            parallel_patch_dir = (base_patch_dir / f"parallel_{agent_id}").resolve()
            parallel_patch_dir.mkdir(parents=True, exist_ok=True)

            # Merge base config with spec overrides
            parallel_agent_config = agent_config.copy()
            parallel_agent_config.update(spec.config)
            parallel_agent_config["patch_output_dir"] = str(parallel_patch_dir)
            parallel_agent_config["mode"] = "yolo"
            parallel_agent_config["confirm_exit"] = False
            if spec.step_limit:
                parallel_agent_config["step_limit"] = spec.step_limit
            if spec.cost_limit:
                parallel_agent_config["cost_limit"] = spec.cost_limit

            log_file = parallel_patch_dir / f"agent_{agent_id}.log"

            if parallel_agent_config.get("test_command"):
                parallel_agent_config["test_command"] = cls._replace_paths(
                    parallel_agent_config["test_command"], repo_path, worktree_path
                )

            task_with_repo = cls._replace_paths(task_content, repo_path, worktree_path)

            # Create model and environment with GPU assignment
            parallel_model = model_factory()
            base_env = env_factory()
            env_config_dict = base_env.config.__dict__.copy() if hasattr(base_env, "config") else {}
            env_config_dict["cwd"] = worktree_path_str
            # Create a NEW dict to avoid shared-reference race across threads
            env_config_dict["env"] = {
                **(env_config_dict.get("env") or {}),
                "HIP_VISIBLE_DEVICES": spec.hip_visible_devices,
                "GEAK_WORK_DIR": worktree_path_str,
                "GEAK_REPO_ROOT": str(repo_path.resolve()),
                "GEAK_GPU_DEVICE": spec.hip_visible_devices,
            }

            parallel_env = type(base_env)(**env_config_dict)

            parallel_output = None
            if output:
                parallel_output = output.parent / f"{output.stem}_parallel_{agent_id}{output.suffix}"

            agent = spec.agent_class(parallel_model, parallel_env, **parallel_agent_config)
            if hasattr(agent, "base_repo_path"):
                agent.base_repo_path = repo_path_resolved
            if hasattr(agent, "log_file"):
                agent.log_file = log_file

            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Agent {agent_id} ({label}) Conversation Log\n")
                f.write(f"GPU: {spec.hip_visible_devices}\n")
                f.write("=" * 60 + "\n\n")

            exit_status, result, extra_info = None, None, None
            with redirect_output_fn(log_file):
                try:
                    exit_status, result = agent.run(task_with_repo, _is_parallel_mode=True)
                except Exception as e:
                    exit_status, result = type(e).__name__, str(e)
                    extra_info = {"traceback": traceback.format_exc()}
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n\nERROR: {exit_status}: {result}\n")
                        f.write(f"Traceback:\n{extra_info['traceback']}\n")
                finally:
                    if parallel_output and save_traj_fn:
                        save_traj_fn(
                            agent, parallel_output, exit_status=exit_status, result=result, extra_info=extra_info
                        )

            return agent_id, agent, exit_status, result

        # Run all agents concurrently
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = {executor.submit(run_spec_agent, i, spec): i for i, spec in enumerate(agent_specs)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                except Exception as e:
                    agent_id = futures[future]
                    from minisweagent.utils.log import logger

                    logger.error(f"Error in heterogeneous agent {agent_id}: {e}", exc_info=True)
        return results

    @classmethod
    def _run_pool(
        cls,
        tasks: list,
        gpu_ids: list[int],
        repo_path: Path,
        is_git_repo: bool,
        base_task_content: str,
        agent_config: dict,
        model_factory,
        env_factory,
        base_patch_dir: Path,
        output: Path | None,
        redirect_output_fn=redirect_output_to_file,
        save_traj_fn=None,
        console=None,
    ) -> list[tuple[int, Any, Any, Any]]:
        """Run M tasks across N GPU slots with overflow queuing.

        Unlike _run_parallel_heterogeneous (which runs exactly N agents on N GPUs),
        this method accepts M tasks (where M can be > N) and schedules them across
        N GPU slots using a thread pool. When a task finishes and frees a GPU slot,
        the next queued task starts immediately -- like ProcessPoolExecutor.

        Args:
            tasks: List of AgentTask objects (from agent_spec.py), sorted by priority.
            gpu_ids: Available GPU device IDs (determines pool size N).
            base_task_content: Fallback task text if a task has no .task set.
            Other args: Same as run_parallel.
        """
        import queue as queue_mod

        n_slots = len(gpu_ids)
        n_tasks = len(tasks)

        if console:
            labels = [t.label or t.agent_class.__name__ for t in tasks]
            console.print(
                f"[bold green]GPU Pool: {n_tasks} tasks on {n_slots} GPU slots "
                f"(labels: {labels[:8]}{'...' if len(labels) > 8 else ''})[/bold green]"
            )

        base_patch_dir = base_patch_dir.resolve()
        worktree_base = base_patch_dir / "worktrees"
        worktree_base.mkdir(parents=True, exist_ok=True)
        repo_path_resolved = repo_path.resolve()

        # Thread-safe GPU pool: each GPU ID can be acquired/released
        gpu_queue = queue_mod.Queue()
        for gid in gpu_ids:
            gpu_queue.put(gid)

        # Map gpu_id -> slot index for worktree naming
        gpu_to_slot = {gid: idx for idx, gid in enumerate(gpu_ids)}

        # Sort tasks by priority (lower = runs first)
        sorted_tasks = sorted(enumerate(tasks), key=lambda t: t[1].priority)

        # Disambiguate duplicate labels so each task gets its own patch directory
        _label_counts: dict[str, int] = {}
        for _, t in sorted_tasks:
            _lbl = t.label or ""
            _label_counts[_lbl] = _label_counts.get(_lbl, 0) + 1
        _has_dup_labels = any(c > 1 for c in _label_counts.values())

        def execute_task(task_id: int, task) -> tuple[int, Any, Any, Any]:
            """Execute a single task on dynamically-assigned GPU(s)."""
            needed = getattr(task, "num_gpus", 1) or 1
            needed = min(needed, n_slots)
            acquired_gpus: list[int] = []
            for _ in range(needed):
                acquired_gpus.append(gpu_queue.get())  # blocks until a GPU is free
            gpu_id = acquired_gpus[0]
            slot_idx = gpu_to_slot[gpu_id]
            hip_devices = ",".join(str(g) for g in acquired_gpus)

            try:
                label = task.label or task.agent_class.__name__
                if console:
                    with _stdout_lock:
                        console.print(
                            f"[bold green]Task {task_id} ({label}): "
                            f"assigned to GPU(s) {hip_devices} (slot {slot_idx})[/bold green]"
                        )

                # Create or reset worktree for this slot
                wt_path = worktree_base / f"slot_{slot_idx}"
                if is_git_repo:
                    starting_patch = task.config.get("starting_patch")
                    if starting_patch:
                        from minisweagent.run.task_file import create_worktree_with_patch
                        create_worktree_with_patch(repo_path, wt_path, starting_patch)
                    else:
                        cls._create_worktree(repo_path, wt_path)
                else:
                    cls._create_copy_workdir(repo_path, wt_path)
                    cls._bootstrap_git_repo(wt_path, console)
                wt_path_str = str(wt_path.resolve())

                # Each task gets its own patch dir named by label (persists across worktree resets)
                if _has_dup_labels:
                    dir_name = f"{task.label}_{task_id}" if task.label else f"task_{task_id}"
                else:
                    dir_name = task.label if task.label else f"task_{task_id}"
                task_patch_dir = (base_patch_dir / dir_name).resolve()
                task_patch_dir.mkdir(parents=True, exist_ok=True)

                # Build agent config
                cfg = agent_config.copy()
                cfg.update(task.config)
                cfg["patch_output_dir"] = str(task_patch_dir)
                # Only set interactive-mode fields for agents that accept them
                # (OpenEvolveWorker extends DefaultAgent, not InteractiveAgent)
                from minisweagent.agents.interactive import InteractiveAgent
                if issubclass(task.agent_class, InteractiveAgent):
                    cfg.setdefault("mode", "yolo")
                    cfg.setdefault("confirm_exit", False)
                if task.step_limit:
                    cfg["step_limit"] = task.step_limit
                if task.cost_limit:
                    cfg["cost_limit"] = task.cost_limit

                log_file = task_patch_dir / f"task_{task_id}.log"

                if cfg.get("test_command"):
                    cfg["test_command"] = cls._replace_paths(cfg["test_command"], repo_path, wt_path)

                # Resolve task text
                agent_task = task.task if task.task else base_task_content
                agent_task = cls._replace_paths(agent_task, repo_path, wt_path)

                # Create model and environment with GPU assignment
                parallel_model = model_factory()
                base_env = env_factory()
                env_config_dict = base_env.config.__dict__.copy() if hasattr(base_env, "config") else {}
                env_config_dict["cwd"] = wt_path_str
                # Create a NEW dict to avoid shared-reference race across threads
                env_config_dict["env"] = {
                    **(env_config_dict.get("env") or {}),
                    "HIP_VISIBLE_DEVICES": hip_devices,
                    "GEAK_WORK_DIR": wt_path_str,
                    "GEAK_REPO_ROOT": str(repo_path.resolve()),
                    "GEAK_GPU_DEVICE": hip_devices,
                }
                parallel_env = type(base_env)(**env_config_dict)

                parallel_output = None
                if output:
                    parallel_output = output.parent / f"{output.stem}_task_{task_id}{output.suffix}"

                agent = task.agent_class(parallel_model, parallel_env, **cfg)
                if hasattr(agent, "base_repo_path"):
                    agent.base_repo_path = repo_path_resolved
                if hasattr(agent, "log_file"):
                    agent.log_file = log_file

                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"Task {task_id} ({label}) Conversation Log\n")
                    f.write(f"GPU: {hip_devices} | Priority: {task.priority} | Language: {task.kernel_language}\n")
                    f.write("=" * 60 + "\n\n")

                exit_status, result, extra_info = None, None, None
                with redirect_output_fn(log_file):
                    try:
                        exit_status, result = agent.run(agent_task, _is_parallel_mode=True)
                    except TerminatingException as e:
                        exit_status, result = type(e).__name__, str(e)
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(f"\n\n{exit_status}: {result}\n")
                    except Exception as e:
                        exit_status, result = type(e).__name__, str(e)
                        extra_info = {"traceback": traceback.format_exc()}
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(f"\n\nERROR: {exit_status}: {result}\n")
                            f.write(f"Traceback:\n{extra_info['traceback']}\n")
                    finally:
                        if parallel_output and save_traj_fn:
                            save_traj_fn(
                                agent, parallel_output, exit_status=exit_status, result=result, extra_info=extra_info
                            )

                if console:
                    with _stdout_lock:
                        console.print(f"[bold blue]Task {task_id} ({label}): completed on GPU(s) {hip_devices}[/bold blue]")

                return task_id, agent, exit_status, result

            finally:
                for g in acquired_gpus:
                    gpu_queue.put(g)

        # Submit ALL M tasks; ThreadPoolExecutor(max_workers=N) queues overflow
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_slots) as executor:
            futures = {executor.submit(execute_task, tid, task): tid for tid, task in sorted_tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                except Exception as e:
                    task_id = futures[future]
                    from minisweagent.utils.log import logger

                    logger.error(f"Error in pool task {task_id}: {e}", exc_info=True)

        return results
