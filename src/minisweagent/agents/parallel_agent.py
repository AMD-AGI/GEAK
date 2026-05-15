"""Agent with git patch saving and test execution capability."""

import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent import Environment, Model

logger = logging.getLogger(__name__)
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.select_patch_agent import run_select_patch
from minisweagent.run.task_file import _neutralize_nested_git_repos, create_worktree
from minisweagent.run.utils.parallel_helpers import (
    _stdout_lock as _stdout_lock,
)
from minisweagent.run.utils.parallel_helpers import (
    redirect_output_to_file,
    run_parallel_heterogeneous,
    run_pool,
)


@dataclass
class BestPatchResult:
    """Result of selecting the best patch from parallel runs."""

    agent_id: int
    patch_id: str
    test_output: str
    best_speedup: float | None = None
    best_patch_file: str | None = None
    patch_dir: Path | None = None
    llm_conclusion: str | None = None


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

    def run(self, task: str, **kwargs) -> BestPatchResult | None:
        num_parallel = self.config.num_parallel or 1
        console = kwargs.get("console")
        deadline = kwargs.get("deadline")
        soft_stop = kwargs.get("soft_stop")
        registry = kwargs.get("registry")

        # Validate repo path (required for worktree management)
        if not self.config.repo:
            raise ValueError("Please specify the repository path.")
        repo_path = (
            Path(self.config.repo) if isinstance(self.config.repo, (str, Path)) else self.config.repo
        ).resolve()
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        base_patch_dir = (
            Path(self.config.patch_output_dir) if self.config.patch_output_dir else Path("patches")
        ).resolve()
        model_factory = kwargs.get("model_factory") or (lambda: self.model)
        env_factory = kwargs.get("env_factory") or (lambda: self.env)
        is_git_repo = (repo_path / ".git").exists() and ParallelAgent._has_valid_head(repo_path)
        output = kwargs.get("output")
        save_traj_fn = kwargs.get("save_traj_fn")

        # Unified logic: always use run_parallel with git worktree management
        self.run_parallel(
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
            deadline=deadline,
            soft_stop=soft_stop,
            registry=registry,
        )

        metric = (
            self.config.metric or "Extract the performance metrics from the test output and calculate the best speedup."
        )
        if console:
            console.print(f"\n[bold green]Selecting best patch from {num_parallel} parallel runs...[/bold green]")
        logger.info("Selecting best patch from %d parallel runs...", num_parallel)
        results_dir = base_patch_dir / "results" / "round_1"
        best_result = self._select_best_from_parallel_runs(results_dir, num_parallel, metric, model_factory)
        if best_result and best_result.llm_conclusion:
            if console:
                console.print("\n[bold cyan]LLM Conclusion:[/bold cyan]")
                console.print(best_result.llm_conclusion)
            logger.info("LLM Conclusion: %s", best_result.llm_conclusion)

        # Return the best result object
        return best_result

    @staticmethod
    def _select_best_from_parallel_runs(
        base_patch_dir: Path, num_parallel: int, metric: str | None, model_factory
    ) -> BestPatchResult | None:
        """Select the best patch from multiple parallel runs using SelectPatchAgent."""
        logger.info("Selecting best patch from %d parallel runs via SelectPatchAgent.", num_parallel)

        model = model_factory()
        _, best_patch_id = run_select_patch(base_patch_dir, num_parallel, metric, model)

        # Only call rewrite_best_results when patch_*_test.txt files exist
        # directly in base_patch_dir (heterogeneous flat layout).  In
        # homogeneous/parallel mode the files live in subdirectories
        # (parallel_0/, parallel_1/) so compute_best_patch cannot find them
        # and the fallback would incorrectly clamp the LLM's speedup to 1.0.
        if list(base_patch_dir.glob("patch_*_test.txt")):
            from minisweagent.run.postprocess.benchmark_parsing import rewrite_best_results

            det_result = rewrite_best_results(base_patch_dir)
            if det_result:
                best_patch_id = det_result.get("best_patch_id", best_patch_id)
                logger.info(
                    "Deterministic override: %s (%sx)",
                    best_patch_id,
                    det_result.get("best_patch_speedup", "?"),
                )

        if not best_patch_id:
            logger.warning("SelectPatchAgent did not produce best_results.json.")
            return None

        logger.info("Selected best patch: %s", best_patch_id)

        try:
            # Read the best_results.json for additional details
            best_results = json.loads((base_patch_dir / "best_results.json").read_text())

            # Parse best_patch_id: "parallel_X/patch_Y", "task_X/patch_Y", or "patch_Y"
            if "/" in best_patch_id:
                dir_name, patch_name = best_patch_id.split("/", 1)
                patch_dir = base_patch_dir / dir_name
                # Extract numeric ID from either "parallel_X" or "task_X"
                id_match = re.search(r"(\d+)", dir_name)
                agent_id = int(id_match.group(1)) if id_match else 0
            else:
                # Single run format: "patch_Y" (directly in base_patch_dir)
                patch_name = best_patch_id
                agent_id = 0
                patch_dir = base_patch_dir

            # Read test output if path provided
            test_output = ""
            test_output_path = best_results.get("best_patch_test_output")
            if test_output_path and Path(test_output_path).exists():
                test_output = Path(test_output_path).read_text()

            # Extract speedup from best_results.json (written by select patch agent)
            raw_speedup = best_results.get("best_patch_speedup")
            best_speedup = float(raw_speedup) if raw_speedup is not None else None

            return BestPatchResult(
                agent_id=agent_id,
                patch_id=patch_name,
                test_output=test_output,
                best_speedup=best_speedup,
                best_patch_file=best_results.get("best_patch_file"),
                patch_dir=patch_dir,
                llm_conclusion=best_results.get("llm_selection_analysis", ""),
            )
        except Exception as e:
            logger.warning("Failed to process best_results.json: %s", e)
            return None

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
    def _has_valid_head(repo_path: Path) -> bool:
        """Check if the git repo has a valid HEAD (at least one commit)."""
        try:
            ParallelAgent._ensure_safe_directory(repo_path)
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception as exc:
            logger.debug("_has_valid_head: check failed for %s: %s", repo_path, exc)
            return False

    @staticmethod
    def _init_as_git_repo(repo_path: Path) -> None:
        """Initialize a non-git repo as a git repository with an initial commit.

        This allows unified git diff management for both git and non-git repos.
        Only initializes if the repo itself doesn't have a .git directory
        (ignores parent directories that might be git repos).

        Also handles nested git repos by neutralizing their .git directories
        so all content is properly included in the parent repo.

        If .git exists but has no valid HEAD (incomplete init), it will be removed
        and re-initialized.
        """
        git_dir = repo_path / ".git"

        # Check if .git exists and has valid HEAD
        if git_dir.exists():
            if ParallelAgent._has_valid_head(repo_path):
                return  # Already a valid git repo
            # Invalid git repo (no HEAD) - remove and reinitialize
            try:
                if git_dir.is_dir():
                    shutil.rmtree(git_dir)
                else:
                    git_dir.unlink()
            except Exception as exc:
                logger.debug("_init_as_git_repo: failed to remove invalid .git in %s: %s", repo_path, exc)
                pass

        try:
            # Neutralize nested git repos first (rename .git -> .git.bak)
            # This ensures nested content is added as regular files, not submodules
            _neutralize_nested_git_repos(repo_path)

            # Initialize git repo (use --initial-branch to ensure new repo creation)
            subprocess.run(
                ["git", "init", "--initial-branch=main"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # Add to safe.directory to avoid ownership issues
            ParallelAgent._ensure_safe_directory(repo_path)

            # Add all files
            subprocess.run(
                ["git", "add", "-A"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # Create initial commit with inline user config (avoids config issues when parent is git repo)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.email=agent@local",
                    "-c",
                    "user.name=Agent",
                    "commit",
                    "-m",
                    "Initial commit (auto-generated for worktree management)",
                ],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else str(e))
            raise RuntimeError(f"Failed to initialize git repo: {error_msg}") from e

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
        prev_worktree_pat = re.compile(
            re.escape(repo_path_str) + r"/optimization_logs/\S*/worktrees/(?:agent|slot|task)_\d+"
        )
        text = prev_worktree_pat.sub(worktree_path_str, text)

        # Replace repo path (resolved and unresolved forms) with worktree path
        text = text.replace(repo_path_str, worktree_path_str)
        if str(repo_path) != repo_path_str:
            text = text.replace(str(repo_path), worktree_path_str)

        # Keep slot id in any remaining /worktrees/slot_<id> segments aligned
        # with this worktree.
        return re.sub(
            r"/worktrees/(?:agent|slot|task)_\d+",
            f"/worktrees/{worktree_path.name}",
            text,
        )

    @staticmethod
    def _gpu_groups_for_homogeneous_parallel(gpu_ids: list[int], num_parallel: int) -> list[list[int]]:
        """Partition ``gpu_ids`` into ``num_parallel`` contiguous groups (sizes differ by at most one).

        - One parallel agent receives every ID (e.g. TP across all listed GPUs).
        - ``N`` agents split the list in order: ``[0,1,2,3]`` with ``N==2`` → ``[[0,1],[2,3]]``.
        """
        if num_parallel <= 0:
            return []
        n = len(gpu_ids)
        if n == 0:
            return [[] for _ in range(num_parallel)]
        base, rem = divmod(n, num_parallel)
        groups: list[list[int]] = []
        idx = 0
        for i in range(num_parallel):
            sz = base + (1 if i < rem else 0)
            groups.append(gpu_ids[idx : idx + sz])
            idx += sz
        return groups

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
        deadline=None,
        soft_stop=None,
        registry=None,
    ) -> list[tuple[int, Any, Any, Any]]:
        """Run multiple parallel agents and return their results.

        Supports three modes (checked in priority order):
        - Pool (preferred): pass tasks (list[AgentTask]) for M tasks on N GPUs.
        - Heterogeneous (legacy): pass agent_specs (list[AgentSpec]).
        - Homogeneous (default): num_parallel identical agents; ``gpu_ids`` are split
          into that many contiguous groups (one agent → all IDs visible; two agents
          on four IDs → two pairs, etc.).

        ``deadline`` / ``soft_stop`` / ``registry`` are forwarded to the
        chosen helper so spawned subprocesses are tracked and the dispatch
        loop can short-circuit on SoftStop.
        """
        # Pool mode: M tasks on N GPU slots (preferred)
        if tasks:
            return run_pool(
                tasks=tasks,
                gpu_ids=gpu_ids or [0],
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
                deadline=deadline,
                soft_stop=soft_stop,
                registry=registry,
            )

        # Heterogeneous mode: use agent_specs if provided (legacy)
        if agent_specs:
            return run_parallel_heterogeneous(
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
                deadline=deadline,
                soft_stop=soft_stop,
                registry=registry,
            )

        # Homogeneous mode (original behavior)
        logger.debug("Running %d parallel patch agents...", num_parallel)

        base_patch_dir = base_patch_dir.resolve()
        results_dir = base_patch_dir / "results" / "round_1"
        results_dir.mkdir(parents=True, exist_ok=True)
        worktree_base = results_dir / "worktrees"
        worktree_base.mkdir(parents=True, exist_ok=True)
        repo_path_resolved = repo_path.resolve()
        repo_path_str = str(repo_path_resolved)

        # Write task files (aligned with heterogeneous tasks/ structure)
        tasks_dir = base_patch_dir / "tasks" / "round_1"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_parallel):
            task_path = tasks_dir / f"parallel_{i}.md"
            task_path.write_text(f"---\nlabel: parallel_{i}\n---\n\n{task_content}\n")
        logger.debug("Wrote %d task files to %s", num_parallel, tasks_dir)

        # Initialize non-git repos as git repos for unified worktree management
        if not is_git_repo:
            logger.info("Initializing non-git repo as git for worktree management...")
            cls._init_as_git_repo(repo_path_resolved)
            is_git_repo = True  # Now it's a git repo

        if gpu_ids and len(gpu_ids) < num_parallel:
            logger.warning(
                "Fewer GPU IDs (%d) than parallel agents (%d); some agents get an empty GPU group "
                "(no HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES override).",
                len(gpu_ids),
                num_parallel,
            )

        agent_gpu_groups: list[list[int]] | None = (
            cls._gpu_groups_for_homogeneous_parallel(gpu_ids, num_parallel) if gpu_ids else None
        )

        def run_single_agent(agent_id: int):
            """Run a single parallel agent instance."""
            # Defense in depth: SoftStop may have fired between submit and start.
            if soft_stop is not None and soft_stop.is_set():
                logger.info("run_single_agent[%d]: SoftStop set before start; skipping", agent_id)
                return agent_id, None, "SoftStop", "skipped before start"

            # All repos use git worktree (non-git repos are initialized as git above)
            worktree_path = create_worktree(repo_path, worktree_base / f"slot_{agent_id}")
            worktree_path_str = str(worktree_path.resolve())

            logger.debug("Created worktree for agent %d: %s", agent_id, worktree_path)

            parallel_patch_dir = (results_dir / f"parallel_{agent_id}").resolve()
            parallel_patch_dir.mkdir(parents=True, exist_ok=True)
            parallel_agent_config = agent_config.copy()
            parallel_agent_config["patch_output_dir"] = str(parallel_patch_dir)
            # Force yolo mode for parallel agents (no interactive confirmation prompts)
            parallel_agent_config["mode"] = "yolo"
            parallel_agent_config["confirm_exit"] = False

            log_file = parallel_patch_dir / f"task_{agent_id}.log"

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
            if agent_gpu_groups is not None and agent_id < len(agent_gpu_groups):
                grp = agent_gpu_groups[agent_id]
                if grp:
                    devs = ",".join(str(g) for g in grp)
                    new_env["HIP_VISIBLE_DEVICES"] = devs
                    new_env["CUDA_VISIBLE_DEVICES"] = devs
                    new_env["GEAK_GPU_DEVICE"] = devs
                    logger.debug("Parallel agent %d assigned GPU(s) %s", agent_id, devs)
                    if console:
                        with _stdout_lock:
                            console.print(f"[bold green]Parallel agent {agent_id} using GPU(s) {devs}[/bold green]")
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
                if hasattr(agent, "_setup_save_and_test_context"):
                    agent._setup_save_and_test_context()
            if hasattr(agent, "log_file"):
                agent.log_file = log_file
            # Wall-clock soft-stop -> sub-agent step loop.
            if soft_stop is not None:
                agent._soft_stop = soft_stop
            # Run-level ProcessRegistry -> save_and_test inner subprocess.run
            # so the budget watchdog can SIGTERM/SIGKILL stuck benchmarks.
            if registry is not None:
                agent._registry = registry
                if hasattr(agent, "_setup_save_and_test_context"):
                    agent._setup_save_and_test_context()  # rebuild ctx with registry attached

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

            _task_label = tasks[agent_id].label if tasks and agent_id < len(tasks) else f"task_{agent_id}"
            logger.info(
                "[dim]Sub-agent %d (%s) started on GPU %s[/dim]",
                agent_id,
                _task_label,
                new_env.get("GEAK_GPU_DEVICE", "?"),
            )
            _agent_t0 = time.monotonic()
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
            _agent_elapsed = time.monotonic() - _agent_t0
            logger.info(
                "Sub-agent %d (%s) finished in %.0fs (exit=%s)",
                agent_id,
                _task_label,
                _agent_elapsed,
                exit_status,
            )

            return agent_id, agent, exit_status, result

        # Run parallel agents with periodic progress reporting
        results = []
        _progress_stop = threading.Event()
        _dispatch_t0 = time.monotonic()

        def _report_progress():
            """Periodically scan patch dirs and report sub-agent progress."""
            _interval = float(os.environ.get("GEAK_PROGRESS_INTERVAL", "30"))
            _prev_patches: dict[str, set[str]] = {}  # label -> set of patch filenames
            while not _progress_stop.wait(_interval):
                elapsed = time.monotonic() - _dispatch_t0
                patches_by_agent = []
                new_patch_paths: list[str] = []
                for i in range(num_parallel):
                    _label = tasks[i].label if tasks and i < len(tasks) else f"task_{i}"
                    pdir = results_dir / (f"parallel_{i}" if not tasks else _label)
                    cur_patches = {p.name for p in pdir.glob("*.patch")} if pdir.is_dir() else set()
                    count = len(cur_patches)
                    patches_by_agent.append((_label, count))
                    prev = _prev_patches.get(_label, set())
                    for pname in sorted(cur_patches - prev):
                        new_patch_paths.append(str(pdir / pname))
                    _prev_patches[_label] = cur_patches
                total_patches = sum(c for _, c in patches_by_agent)
                summary = ", ".join(f"{l}: {c}" for l, c in patches_by_agent if c > 0)
                logger.info(
                    "[dim]\\[running %.1fmin] Sub-agents working: %d total patches%s[/dim]",
                    elapsed / 60,
                    total_patches,
                    f" ({summary})" if summary else "",
                    extra={"progress_tick": True},
                )
                for pp in new_patch_paths:
                    logger.debug("[dim]  New patch: %s[/dim]", pp)

        _progress_thread = threading.Thread(target=_report_progress, daemon=True)
        _progress_thread.start()

        # Use poll loop so SoftStop is observed mid-dispatch. Manual executor
        # lifecycle (no ``with`` block) so we can detach via shutdown(wait=False,
        # cancel_futures=True) on SoftStop instead of blocking on stuck workers.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel)
        soft_stop_observed = False
        try:
            futures: dict[concurrent.futures.Future, int] = {}
            for i in range(num_parallel):
                if registry is not None:
                    with registry.lock:
                        if soft_stop is not None and soft_stop.is_set():
                            soft_stop_observed = True
                            break
                        fut = executor.submit(run_single_agent, i)
                        registry.register_future(fut)
                else:
                    if soft_stop is not None and soft_stop.is_set():
                        soft_stop_observed = True
                        break
                    fut = executor.submit(run_single_agent, i)
                futures[fut] = i

            pending = set(futures.keys())
            while pending:
                if soft_stop is not None and soft_stop.is_set():
                    logger.warning(
                        "ParallelAgent.run_parallel (homogeneous): SoftStop set; cancelling %d in-flight",
                        len(pending),
                    )
                    if registry is not None:
                        registry.terminate_all()
                    for f in pending:
                        f.cancel()
                    soft_stop_observed = True
                    break
                done, pending = concurrent.futures.wait(pending, timeout=2.0)
                for f in done:
                    agent_id = futures[f]
                    try:
                        result = f.result()
                        results.append(result)
                    except concurrent.futures.CancelledError:
                        logger.info("Homogeneous parallel agent %d cancelled", agent_id)
                    except Exception as e:
                        logger.error("Error in parallel agent %d: %s", agent_id, e, exc_info=True)
        finally:
            if soft_stop_observed:
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        _progress_stop.set()
        _progress_thread.join(timeout=2)
        return results
