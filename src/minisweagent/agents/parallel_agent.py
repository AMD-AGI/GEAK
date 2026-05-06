"""Agent with git patch saving and test execution capability."""

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.select_patch_agent import run_select_patch
from minisweagent.run.task_file import _neutralize_nested_git_repos
from minisweagent.run.utils.parallel_helpers import (
    redirect_output_to_file,
    run_pool,
)

logger = logging.getLogger(__name__)


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

        # Unified logic: always route through run_pool via the task-based
        # run_parallel entry point.  Config cleanup drops the pool-wiring keys
        # so only agent-relevant settings flow into the per-task ``agent_config``.
        self.run_parallel(
            num_parallel=num_parallel,
            repo_path=repo_path,
            is_git_repo=is_git_repo,
            task_content=task,
            agent_class=self.config.agent_class if self.config.agent_class else type(self),
            agent_config={
                k: v
                for k, v in self.config.__dict__.items()
                if k not in ("num_parallel", "repo", "gpu_ids", "agent_class", "tasks")
            },
            model_factory=model_factory,
            env_factory=env_factory,
            base_patch_dir=base_patch_dir,
            output=output,
            gpu_ids=self.config.gpu_ids,
            save_traj_fn=save_traj_fn,
            console=console,
            tasks=self.config.tasks,
        )

        metric = (
            self.config.metric or "Extract the performance metrics from the test output and calculate the best speedup."
        )
        if console:
            console.print(f"\n[bold green]Selecting best patch from {num_parallel} parallel runs...[/bold green]")
        logger.info("Selecting best patch from %d parallel runs...", num_parallel)

        # Cross-N rollup: the per-worker artefacts (parallel_0/, parallel_1/, ...
        # each containing patch_*.patch + best_results.json) sit DIRECTLY under
        # ``base_patch_dir``.  The legacy hardcoded ``results/round_1`` subdir
        # was a planned-mode artifact that doesn't exist in the fixed-mode
        # layout, which caused ``SelectPatchAgent`` to come up empty even when
        # individual workers had produced verified speedups.  Prefer
        # ``base_patch_dir`` directly; fall back to legacy ``results/round_1``
        # only when that layout actually exists (rare, planned-mode inheritance).
        legacy_round_dir = base_patch_dir / "results" / "round_1"
        results_dir = legacy_round_dir if legacy_round_dir.is_dir() else base_patch_dir
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
        tasks: list | None = None,
    ) -> list[tuple[int, Any, Any, Any]]:
        """Run multiple parallel agents and return their results.

        Callers must supply ``tasks`` (a ``list[AgentTask]``).  All
        execution modes — fixed (identical copies), planned
        (planner-generated per-task bodies), translate — flow through
        this task-based entry point.  Identical-copies workloads use
        ``pool_runner.build_fixed_tasks`` to materialise their task list.
        """
        if not tasks:
            raise ValueError(
                "ParallelAgent.run_parallel requires a non-empty `tasks` list; "
                "use pool_runner.build_fixed_tasks to materialise one for "
                "identical-copies (fixed-mode) workloads."
            )
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
        )
